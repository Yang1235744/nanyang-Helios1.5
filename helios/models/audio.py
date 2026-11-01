"""音频模块：AudioVAE（mel → latent）+ AudioDiT（文本/视频条件下生成音轨）。

视频帧率 12fps，音频采样率 24000，hop 480 → 50 mel-fps。15 秒 = 750 mel 帧。
为控制显存，AudioDiT 把音频 latent 沿时间维做 1D 流动匹配。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def t_emb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class AudioResBlock1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, c), c)
        self.c1 = nn.Conv1d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(min(32, c), c)
        self.c2 = nn.Conv1d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class AudioVAE(nn.Module):
    """mel 谱面 [B, n_mels, T] ↔  latent [B, latent, T']."""

    def __init__(self, n_mels=128, latent=8, base=96):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_mels, base, 3, padding=1),
            AudioResBlock1d(base),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1),
            AudioResBlock1d(base * 2),
            nn.Conv1d(base * 2, base * 4, 3, stride=2, padding=1),
            AudioResBlock1d(base * 4),
            nn.Conv1d(base * 4, latent * 2, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(latent, base * 4, 1),
            AudioResBlock1d(base * 4),
            nn.ConvTranspose1d(base * 4, base * 2, 3, stride=2, padding=1, output_padding=1),
            AudioResBlock1d(base * 2),
            nn.ConvTranspose1d(base * 2, base, 3, stride=2, padding=1, output_padding=1),
            AudioResBlock1d(base),
            nn.Conv1d(base, n_mels, 3, padding=1),
        )

    def encode(self, mel):
        h = self.enc(mel)
        mu, logvar = h.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30, 20))
        return mu + std * torch.randn_like(std), mu, logvar

    def decode(self, z):
        return self.dec(z)


class AudioBlock(nn.Module):
    def __init__(self, dim, heads, text_dim):
        super().__init__()
        self.ada1 = nn.Sequential(nn.SiLU(), nn.Linear(text_dim, dim * 2))
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.h = heads
        self.proj = nn.Linear(dim, dim)
        self.ada2 = nn.Sequential(nn.SiLU(), nn.Linear(text_dim, dim * 2))
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.t_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.t_norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, text, t):
        B, N, C = x.shape
        cond = text[:, 0]
        s, b = self.ada1(cond).chunk(2, -1)
        ts, tb = self.t_ada(t).chunk(2, -1)
        s, b, ts, tb = s.unsqueeze(1), b.unsqueeze(1), ts.unsqueeze(1), tb.unsqueeze(1)
        h = self.n1(x) * (1 + s) + b
        h = self.t_norm(h) * (1 + ts) + tb
        q, k, v = self.qkv(h).chunk(3, -1)
        q = q.view(B, N, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, N, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, N, self.h, C // self.h).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(h)
        s2, b2 = self.ada2(cond).chunk(2, -1)
        s2, b2 = s2.unsqueeze(1), b2.unsqueeze(1)
        x = x + self.mlp(self.n2(x) * (1 + s2) + b2)
        return x


class AudioDiT(nn.Module):
    """1D 流动匹配生成音频 latent，条件为文本 + 视频内容摘要。"""

    def __init__(self, text_dim, latent=8, dim=768, depth=12, heads=12):
        super().__init__()
        self.in_proj = nn.Linear(latent, dim)
        self.blocks = nn.ModuleList([AudioBlock(dim, heads, text_dim) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent)

    def forward(self, z, t, text):
        x = self.in_proj(z.transpose(1, 2))   # (B, T, dim)
        te = t_emb(t, x.shape[-1])
        for blk in self.blocks:
            x = blk(x, text, te)
        return self.out(self.ln(x)).transpose(1, 2)   # (B, latent, T)
