"""Reasoning with vLLM: the audio front end (encoder, tokenizer, input embeddings) runs here, the language model runs
in vLLM from prompt embeddings. Generation only (explain, compare); detection scores use VeriSpoof.

    from verispoof.vllm_backend import VeriSpoofVLLM
    m = VeriSpoofVLLM.from_pretrained("AustinXiao/VeriSpoof-Base")
    m.explain(["a.wav", "b.wav"])       # list in, list out; batched by vLLM
"""
import json
import os

import torch

from verispoof.inference import COMPARE_QUESTION, EXPLAIN_QUESTION, fit_window, load_audio, parse_reasoning
from verispoof.model import AUDIO, Network


class VeriSpoofVLLM:
    def __init__(self, net, engine, device="cuda"):
        self.net, self.engine, self.device = net, engine, device

    @classmethod
    def from_pretrained(cls, path, device="cuda", gpu_memory_utilization=0.5, max_model_len=2048, revision=None,
                        **engine_kwargs):
        from vllm import LLM
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=revision)
        cfg = json.load(open(f"{path}/config.json"))
        net = Network(path, cfg, device=device, load_llm=False).to(device).eval()
        engine = LLM(model=f"{path}/llm", dtype=cfg.get("llm_dtype", "bfloat16"), enable_prompt_embeds=True,
                     gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, **engine_kwargs)
        return cls(net, engine, device)

    @torch.no_grad()
    def _embeds(self, prompt, groups, sr, batch_size):
        """groups: per request, the recordings in prompt order -> list of (L, H) prompt embeddings on the CPU."""
        ids = torch.tensor(self.net.prompt_ids(prompt), device=self.device)
        out = []
        for i in range(0, len(groups), batch_size):
            chunk = groups[i:i + batch_size]
            wav = torch.stack([torch.from_numpy(fit_window(load_audio(a, sr), self.net.window))
                               for g in chunk for a in g])
            e = self.net.embed(ids.expand(len(chunk), -1), wav.to(self.device))
            out += [x.cpu() for x in e]
        return out

    def _generate(self, embeds, max_new_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=[self.net.end_id])
        res = self.engine.generate([{"prompt_embeds": e} for e in embeds], sp, use_tqdm=False)
        return [r.outputs[0].text for r in res]

    def explain(self, audios, sr=None, max_new_tokens=320, batch_size=32):
        """Grounded detection reasoning for one recording or a list (greedy)."""
        single = not isinstance(audios, (list, tuple))
        audios = [audios] if single else list(audios)
        texts = self._generate(self._embeds(f"{AUDIO}\n{EXPLAIN_QUESTION}", [[a] for a in audios], sr, batch_size),
                               max_new_tokens)
        out = [dict(parse_reasoning(t), text=t) for t in texts]
        return out[0] if single else out

    def compare(self, pairs, sr=None, max_new_tokens=320, batch_size=32):
        """Source-verification reasoning for one (a, b) pair or a list of pairs (greedy)."""
        single = isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], tuple)
        pairs = [pairs] if single else list(pairs)
        texts = self._generate(self._embeds(f"Audio A: {AUDIO}\nAudio B: {AUDIO}\n{COMPARE_QUESTION}",
                                            [list(p) for p in pairs], sr, batch_size), max_new_tokens)
        out = [dict(parse_reasoning(t), text=t) for t in texts]
        return out[0] if single else out
