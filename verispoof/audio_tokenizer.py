"""Audio tokenizer: encoder hidden states (B, T, n_layers, D) -> (B, n_global + T / frame_stack, d_llm) LLM inputs."""
import math

import torch
import torch.nn as nn


class AudioTokenizer(nn.Module):
    def __init__(self, d_llm, layer_mix, d_global=2048, d_local=2048, n_global=16, d_q=512, frame_stack=2,
                 n_frames=104):
        super().__init__()
        self.mix_logits = nn.Parameter(torch.zeros(layer_mix["n_mix"], layer_mix["n_layers"]))
        self.n_global, self.d_q = n_global, d_q
        self.frame_stack, self.n_frames = frame_stack, n_frames
        self.n_local = math.ceil(n_frames / frame_stack)
        self.g_mlp = nn.Sequential(nn.LayerNorm(d_global), nn.Linear(d_global, 1024), nn.GELU(),
                                   nn.Linear(1024, n_global * d_q))
        self.g_proj = nn.Sequential(nn.LayerNorm(d_q), nn.Linear(d_q, d_llm))
        self.f_proj = nn.Sequential(nn.LayerNorm(d_local * frame_stack), nn.Linear(d_local * frame_stack, 1024),
                                    nn.GELU(), nn.Linear(1024, d_llm))

    @property
    def n_tokens(self):
        return self.n_global + self.n_local

    def forward(self, hs):
        w = torch.softmax(self.mix_logits.float(), -1).to(hs.dtype)
        loc = torch.einsum("btld,ml->btmd", hs, w).flatten(2)
        g = loc.mean(1)
        glob = self.g_proj(self.g_mlp(g).view(g.size(0), self.n_global, self.d_q))
        assert loc.size(1) == self.n_frames, (loc.shape, self.n_frames)
        k = self.frame_stack
        pad = self.n_local * k - loc.size(1)
        x = nn.functional.pad(loc, (0, 0, 0, pad)) if pad else loc
        return torch.cat([glob, self.f_proj(x.reshape(loc.size(0), self.n_local, k * loc.size(2)))], 1)
