"""VeriSpoof network: audio encoder -> audio tokenizer -> language model, plus a per-token head."""
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from verispoof.audio_tokenizer import AudioTokenizer
from verispoof.encoder import AudioEncoder

AUDIO_TOKEN = "<unused_token_0>"
SYSTEM = ("You are VeriSpoof, an assistant that analyses speech recordings for deepfake detection and for "
          "understanding how synthetic speech was generated.")
AUDIO = "<audio>"
END = "<|im_end|>"


class Network(nn.Module):
    def __init__(self, path, cfg, device="cuda", load_llm=True):
        """load_llm=False: only the audio front end and the input embedding table (for an external LLM engine)."""
        super().__init__()
        self.time_marks = cfg.get("time_marks")
        self.system = cfg.get("system_prompt", SYSTEM)
        self.window = cfg.get("window", 128000)
        dtype = getattr(torch, cfg.get("llm_dtype", "bfloat16"))
        self.amp = dtype != torch.float32  # float32 weights run without bf16 autocast
        self.text_tok = AutoTokenizer.from_pretrained(f"{path}/llm")
        if load_llm:
            self.llm = AutoModelForCausalLM.from_pretrained(f"{path}/llm", dtype=dtype, attn_implementation="sdpa")
            self.llm.requires_grad_(False)
            self.embed_tokens = self.llm.get_input_embeddings()
            h = self.llm.config.hidden_size
        else:
            from transformers import AutoConfig
            self.llm = None
            from safetensors import safe_open
            with safe_open(f"{path}/llm/model.safetensors", "pt") as f:
                w = f.get_tensor("model.embed_tokens.weight")
            self.embed_tokens = nn.Embedding.from_pretrained(w.to(dtype), freeze=True)
            h = AutoConfig.from_pretrained(f"{path}/llm").hidden_size
        self.audio_str = AUDIO_TOKEN
        self.audio_id = self.text_tok.convert_tokens_to_ids(AUDIO_TOKEN)
        self.audio_tok = AudioTokenizer(h, **cfg["tokenizer"])
        self.audio_tok.load_state_dict(load_file(f"{path}/audio_tokenizer.safetensors"))
        self.tok_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.tok_head.load_state_dict(load_file(f"{path}/token_head.safetensors"))
        self.encoder = AudioEncoder(f"{path}/audio_encoder.safetensors", device=device)
        self.pad_id = self.text_tok.pad_token_id if self.text_tok.pad_token_id is not None else self.text_tok.eos_token_id
        self.end_id = self.text_tok.convert_tokens_to_ids(END)

    def causal_lm(self):
        return self.llm

    def audio_layout(self, frame_s=0.08):
        a = self.audio_tok
        if not self.time_marks:
            return [a.n_tokens]
        step = frame_s * a.frame_stack
        k = max(1, round(self.time_marks / step))
        out = [a.n_global]
        for i in range(0, a.n_local, k):
            out += [f"[{i * step:.1f}s]", min(k, a.n_local - i)]
        return out

    def prompt_ids(self, prompt):
        block = "".join(self.audio_str * x if isinstance(x, int) else x for x in self.audio_layout())
        text = self.text_tok.apply_chat_template(
            [{"role": "system", "content": self.system}, {"role": "user", "content": prompt.replace(AUDIO, block)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.text_tok(text, add_special_tokens=False)["input_ids"]

    def text_ids(self, text):
        return self.text_tok(text, add_special_tokens=False)["input_ids"]

    def embed(self, input_ids, wav):
        emb = self.embed_tokens(input_ids)
        mask = input_ids == self.audio_id
        hs = self.encoder(wav)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.audio_tok(hs)
        assert mask.sum() == a.shape[0] * a.shape[1], "audio placeholders do not match audio inputs"
        return emb.masked_scatter(mask.unsqueeze(-1), a.to(emb.dtype).reshape(-1, emb.size(-1)))

    def token_logits(self, h, input_ids):
        n = self.audio_tok.n_tokens
        a = h[input_ids == self.audio_id].view(-1, n, h.size(-1))[:, self.audio_tok.n_global:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.tok_head(a).squeeze(-1).float()

    def seconds_per_token(self, frame_s=0.08):
        return frame_s * self.audio_tok.frame_stack
