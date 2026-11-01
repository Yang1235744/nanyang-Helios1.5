"""Rectified Flow：视频/图片 latent 流动匹配 + 音频 latent 流动匹配，Euler 采样 + CFG。"""

import torch


def flow_loss(model, z0, cond, mask, cfg_scale=1.0):
    z1 = torch.randn_like(z0)
    t = torch.rand(z0.shape[0], device=z0.device)
    shape = [1] * z0.ndim
    shape[0] = -1
    zt = (1 - t.view(*shape)) * z0 + t.view(*shape) * z1
    target = z1 - z0
    pred = model(zt, t, cond, mask)
    return ((pred - target) ** 2).mean()


def flow_loss_audio(model, z0, text):
    z1 = torch.randn_like(z0)
    t = torch.rand(z0.shape[0], device=z0.device)
    shape = [1] * z0.ndim
    shape[0] = -1
    zt = (1 - t.view(*shape)) * z0 + t.view(*shape) * z1
    pred = model(zt, t, text)
    return ((pred - (z1 - z0)) ** 2).mean()


@torch.no_grad()
def euler_sample(model, ctx_c, mask_c, ctx_u, mask_u, shape, steps, cfg_scale, device):
    z = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.zeros(shape[0], device=device)
    for _ in range(steps):
        vc = model(z, t, ctx_c, mask_c)
        vu = model(z, t, ctx_u, mask_u)
        v = vu + cfg_scale * (vc - vu)
        z = z + dt * v
        t = t + dt
    return z


@torch.no_grad()
def euler_sample_audio(model, text, shape, steps, device):
    z = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.zeros(shape[0], device=device)
    for _ in range(steps):
        z = z + dt * model(z, t, text)
        t = t + dt
    return z
