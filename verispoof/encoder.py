"""Audio encoder: Qwen3-ASR-1.7B audio tower architecture with VeriSpoof weights.

wav (B, 128000) float32 16 kHz -> (B, 104, 25, 1024) bf16 hidden states (input embedding and every layer output).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

AUT_REPO = "Qwen/Qwen3-ASR-1.7B-hf"  # architecture config and mel filters only


class LogMel(nn.Module):
    def __init__(self, mel_filters, n_fft=400, hop=160):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("mel", torch.as_tensor(mel_filters, dtype=torch.float32).T.contiguous())
        self.register_buffer("window", torch.hann_window(n_fft))

    @torch.autocast("cuda", enabled=False)
    def forward(self, wav):
        stft = torch.stft(wav.float(), self.n_fft, self.hop, window=self.window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self.mel @ mag, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0


class AudioEncoder:
    def __init__(self, weights, device="cuda"):
        from transformers import AutoConfig, AutoFeatureExtractor
        from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder
        self.device = device
        acfg = AutoConfig.from_pretrained(AUT_REPO).audio_config
        acfg._attn_implementation = "sdpa"
        self.encoder = Qwen3ASREncoder(acfg)
        self.encoder.load_state_dict(load_file(weights), strict=True)
        self.encoder.float()
        fe = AutoFeatureExtractor.from_pretrained(AUT_REPO)
        self.chunk = 2 * acfg.n_window
        self.logmel = LogMel(fe.mel_filters, fe.n_fft, fe.hop_length)
        for m in (self.encoder, self.logmel):
            m.eval().requires_grad_(False).to(device)
        self._hs = []
        self.encoder.layers[0].register_forward_pre_hook(
            lambda m, a, k: self._hs.append(a[0] if a else k["hidden_states"]), with_kwargs=True)
        for layer in self.encoder.layers:
            layer.register_forward_hook(lambda m, a, o: self._hs.append(o[0] if isinstance(o, tuple) else o))

    @torch.no_grad()
    def __call__(self, wav):
        wav = wav.to(self.device, torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mel = self.logmel(wav)
            T = mel.shape[-1]
            pad = (-T) % self.chunk
            mask = torch.ones(mel.shape[0], T + pad, dtype=torch.long, device=mel.device)
            mask[:, T:] = 0
            self._hs = []
            out = self.encoder(input_features=F.pad(mel, (0, pad)), input_features_mask=mask)
            hs, self._hs = self._hs, []
            hs[-1] = out.last_hidden_state
            B = wav.shape[0]
            hs = [h.reshape(B, -1, h.shape[-1]) for h in hs]
        return torch.stack(hs, 2).to(torch.bfloat16)
