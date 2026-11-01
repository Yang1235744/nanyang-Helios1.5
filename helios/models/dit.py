"""HeliosDiT 1.5 中等：空间 patch 化 + 时空可分离位置编码 + 时空解耦注意力。

相对 1.2 的改动：
  - 保留 1.2 的 patchify（patch_size）与可学习位置表（pos_t / pos_h / pos_w），
    位置信息按"第几帧 + 第几行 + 第几列"可分离相加，长视频与变分辨率都成立；
  - 注意力从 1.2 的全时空自注意力改为时空解耦：空间注意力逐帧、时间注意力跨帧，
    180 帧长视频的显存从 O((T·S)²) 降到 O(T·S² + S·T²)；
  - 条件调制（adaLN）用时间步嵌入 + 文本池化向量，所有 block 共享，避免每块重复池化。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def patchify(z, p):
    """[B,C,T,H,W] -> [B, T*h*w, C*p*p]"""
    B, C, T, H, W = z.shape
    h, w = H // p, W // p
    z = z.reshape(B, C, T, h, p, w, p).permute(0, 2, 3, 5, 1, 4, 6)
    return z.reshape(B, T * h * w, C * p * p)


def unpatchify(tokens, p, C, T, H, W):
    """[B, T*h*w, C*p*p] -> [B,C,T,H,W]"""
    B = tokens.shape[0]
    h, w = H // p, W // p
    x = tokens.reshape(B, T, h, w, C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
    return x.reshape(B, C, T, H, W)


class AdaLN(nn.Module):
    """adaLN-Zero 风格的调制：从条件向量产生 shift / scale / gate（并持有归一化层）。"""

    def __init__(self, dim, cond_dim, out_mult=2, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * out_mult))
        if zero_init:
            nn.init.zeros_(self.proj[1].weight)
            nn.init.zeros_(self.proj[1].bias)

    def forward(self, x, cond):
        """只依赖条件，返回 dim*out_mult 的调制参数（x 仅用于形状对齐）。"""
        return self.proj(cond)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, N, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, N, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, N, self.h, C // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, ctx_dim=None):
        super().__init__()
        self.h = heads
        ctx_dim = ctx_dim or dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim, dim * 2)      # 文本维度与 DiT 维度可以不同
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx, mask=None):
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.h, C // self.h).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, -1)
        k = k.view(B, -1, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, -1, self.h, C // self.h).transpose(1, 2)
        # SDPA 的 attn_mask 是 True=参与计算，而 ctx_mask 约定 True=padding，需要取反
        attn_mask = None if mask is None else (~mask)[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class SpatioTemporalBlock(nn.Module):
    """空间注意力（逐帧）→ 时间注意力（跨帧）→ 文本交叉注意力 → MLP。"""

    def __init__(self, dim, heads, mlp_ratio, ctx_dim=None):
        super().__init__()
        self.ada_s = AdaLN(dim, dim, out_mult=3)
        self.attn_s = SelfAttention(dim, heads)
        self.ada_t = AdaLN(dim, dim, out_mult=3)
        self.attn_t = SelfAttention(dim, heads)
        self.ada_x = AdaLN(dim, dim, out_mult=3)
        self.cross = CrossAttention(dim, heads, ctx_dim)
        self.ada_m = AdaLN(dim, dim, out_mult=3)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    @staticmethod
    def _modulate(norm, ada, x, cond):
        """返回 (调制后的输入, gate)；shift/scale/gate 三路输出。"""
        shift, scale, gate = ada(x, cond).chunk(3, dim=-1)
        shift = shift.unsqueeze(1) if shift.ndim == 2 else shift
        scale = scale.unsqueeze(1) if scale.ndim == 2 else scale
        gate = gate.unsqueeze(1) if gate.ndim == 2 else gate
        return norm(x) * (1 + scale) + shift, gate

    def forward(self, x, cond, ctx, ctx_mask, F, S):
        B, N, C = x.shape
        # ---- 空间注意力：时间维并入 batch，只在 H*W 内注意
        xs = x.view(B, F, S, C).reshape(B * F, S, C)
        hs, gs = self._modulate(self.ada_s.norm, self.ada_s, xs, cond.repeat_interleave(F, dim=0))
        xs = xs + gs * self.attn_s(hs)
        # ---- 时间注意力：空间维并入 batch，只在 T 内注意
        xt = xs.view(B, F, S, C).permute(0, 2, 1, 3).reshape(B * S, F, C)
        ht, gt = self._modulate(self.ada_t.norm, self.ada_t, xt,
                                cond.unsqueeze(1).expand(-1, S, -1).reshape(B * S, C))
        xt = xt + gt * self.attn_t(ht)
        x = xt.view(B, S, F, C).permute(0, 2, 1, 3).reshape(B, N, C)
        # ---- 文本交叉注意力
        hx, gx = self._modulate(self.ada_x.norm, self.ada_x, x, cond)
        x = x + gx * self.cross(hx, ctx, ctx_mask)
        # ---- MLP
        hm, gm = self._modulate(self.ada_m.norm, self.ada_m, x, cond)
        return x + gm * self.mlp(hm)


class HeliosDiT(nn.Module):
    def __init__(self, text_dim, dim=1024, depth=24, heads=16, mlp_ratio=4.0,
                 patch_size=2, latent_channels=16, max_frames=180, max_h=64, max_w=64,
                 **kwargs):
        super().__init__()
        self.p = patch_size
        self.patch_in = nn.Linear(latent_channels * patch_size * patch_size, dim)
        # 与 1.2 相同的可分离位置表：帧 / 行 / 列
        self.pos_t = nn.Parameter(torch.zeros(max_frames, dim))
        self.pos_h = nn.Parameter(torch.zeros(max_h, dim))
        self.pos_w = nn.Parameter(torch.zeros(max_w, dim))
        for pos in (self.pos_t, self.pos_h, self.pos_w):
            nn.init.trunc_normal_(pos, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ctx_pool = nn.Sequential(nn.Linear(text_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(dim, heads, mlp_ratio, text_dim) for _ in range(depth)
        ])
        self.final_ada = AdaLN(dim, dim, out_mult=2)
        self.final_proj = nn.Linear(dim, latent_channels * patch_size * patch_size)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, z, t, ctx, ctx_mask=None):
        """z:[B,C,T,H,W]，t:[B]∈[0,1]，ctx:[B,L,text_dim] -> 速度场 [B,C,T,H,W]"""
        B, C, F, H, W = z.shape
        h, w = H // self.p, W // self.p
        S = h * w
        assert F <= self.pos_t.shape[0], f"帧数 {F} 超出位置表上限 {self.pos_t.shape[0]}"
        assert h <= self.pos_h.shape[0] and w <= self.pos_w.shape[0], \
            f"patch 网格 {h}x{w} 超出位置表上限"

        x = self.patch_in(patchify(z, self.p))
        pos = (self.pos_t[:F, None, None, :] + self.pos_h[None, :h, None, :] + self.pos_w[None, None, :w, :])
        x = x + pos.reshape(1, F * S, -1)

        # 条件：时间步 + 文本池化（池化时屏蔽 padding）
        if ctx_mask is not None:
            keep = (~ctx_mask).float().unsqueeze(-1)
            pooled = (ctx * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            pooled = ctx.mean(1)
        cond = self.t_mlp(timestep_embedding(t, self.t_mlp[0].in_features)) + self.ctx_pool(pooled)

        for blk in self.blocks:
            x = blk(x, cond, ctx, ctx_mask, F, S)

        f_shift, f_scale = self.final_ada(x, cond).chunk(2, dim=-1)
        x = self.final_ada.norm(x) * (1 + f_scale.unsqueeze(1)) + f_shift.unsqueeze(1)
        return unpatchify(self.final_proj(x), self.p, C, F, H, W)
