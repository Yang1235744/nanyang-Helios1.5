"""HeliosDiT 1.5 中等：时空解耦注意力 + 文本交叉注意力，支持 15 秒长视频。

长视频关键：use_temporal 下，空间注意力逐帧、时间注意力跨帧；避免 180 帧全注意力的二次方代价。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))

    def forward(self, x, cond):
        s, b = self.proj(cond).chunk(2, dim=-1)
        s = s.unsqueeze(-2) if s.ndim == 2 else s
        b = b.unsqueeze(-2) if b.ndim == 2 else b
        return self.norm(x) * (1 + s) + b


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, N, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, N, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, N, self.h, C // self.h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx, mask=None):
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.h, C // self.h).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, -1)
        k = k.view(B, -1, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, -1, self.h, C // self.h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class SpatioTemporalBlock(nn.Module):
    """空间注意力（逐帧）+ 时间注意力（跨帧）+ 文本交叉注意力解耦。"""

    def __init__(self, dim, heads, text_dim, mlp_ratio):
        super().__init__()
        self.ada1 = AdaLN(dim, text_dim)
        self.attn_s = Attention(dim, heads)
        self.ada2 = AdaLN(dim, text_dim)
        self.attn_t = Attention(dim, heads)
        self.ada3 = AdaLN(dim, text_dim)
        self.cross = CrossAttention(dim, heads)
        self.ada4 = AdaLN(dim, text_dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, text, text_mask, F, S):
        B, N, C = x.shape
        pool = text.mean(dim=1)              # (B, C)
        # 空间注意力：时间维并入 batch
        xs = x.view(B, F, S, C).reshape(B * F, S, C)
        xs = xs + self.attn_s(self.ada1(xs, pool.repeat_interleave(F, dim=0)))
        # 时间注意力：空间维并入 batch
        xt = xs.view(B, F, S, C).permute(0, 2, 1, 3).reshape(B * S, F, C)
        cond_t = pool.unsqueeze(1).expand(-1, S, -1).reshape(B * S, C)
        xt = xt + self.attn_t(self.ada2(xt, cond_t))
        x = xt.view(B, S, F, C).permute(0, 2, 1, 3).reshape(B, N, C)
        # 交叉注意力到文本
        cmask = (text_mask == False).view(B, 1, 1, -1).expand(B, self.cross.h, N, -1).reshape(B * self.cross.h, N, -1)
        x = x + self.cross(self.ada3(x, pool), text, cmask)
        return x + self.mlp(self.ada4(x, pool))


class HeliosDiT(nn.Module):
    def __init__(self, text_dim, dim=1024, depth=24, heads=16, mlp_ratio=4.0,
                 patch_size=2, latent_channels=16, in_frames=None, in_size=None,
                 use_temporal=True, use_cross_attn=True, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.use_temporal = use_temporal
        self.t_proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.in_proj = nn.Conv3d(latent_channels, dim, (1, patch_size, patch_size),
                                 stride=(1, patch_size, patch_size))
        # 简化：不做真正的 3D patchify，用卷积把 latent 投到 dim
        self.in_proj = nn.Linear(latent_channels, dim)
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(dim, heads, text_dim, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_channels)

    def forward(self, z, t, text, text_mask):
        B, C, F, H, W = z.shape
        x = z.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
        x = self.in_proj(x)
        cond = self.t_proj(timestep_embedding(t, x.shape[-1]))
        for blk in self.blocks:
            x = blk(x, text, text_mask, F, H * W)
        x = self.out(self.ln(x))
        return x.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)
