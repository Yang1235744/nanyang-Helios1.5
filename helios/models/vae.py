"""中等 VAE：逐帧 2D，8x 下采样到 16 通道 latent，支持高清（最高 1024px）。

相比 1.2：base_channels 128、num_res_blocks 3、latent_channels 16，
承载高清细节与长视频信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(c): return nn.GroupNorm(min(32, c), c)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = group_norm(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = group_norm(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.silu(self.n2(self.conv1(F.silu(self.n1(x))))))


class DownBlock(nn.Module):
    def __init__(self, cin, cout, num_res):
        super().__init__()
        self.res = nn.Sequential(*[ResBlock(cout) for _ in range(num_res)])
        self.conv_in = nn.Conv2d(cin, cout, 3, padding=1)
        self.down = nn.Conv2d(cout, cout, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res(x)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, cin, cout, num_res):
        super().__init__()
        self.res = nn.Sequential(*[ResBlock(cout) for _ in range(num_res)])
        self.conv_out = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv_in = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_in(x)
        if skip is not None:
            x = x + skip[:, :, : x.shape[2], : x.shape[3]]
        x = self.res(x)
        return self.conv_out(x)


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=16, base_channels=128,
                 channel_mults=(1, 2, 4, 4), num_res_blocks=3, downsample_factor=None):
        super().__init__()
        # 实际下采样只在相邻层之间发生：4 级通道 → 3 次 stride2 → 8x
        self.downsample_factor = 2 ** (len(channel_mults) - 1)
        self.latent_channels = latent_channels
        chs = [base_channels * m for m in channel_mults]
        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.downs = nn.ModuleList([
            DownBlock(chs[i], chs[i + 1], num_res_blocks) for i in range(len(chs) - 1)
        ])
        mid = chs[-1]
        self.mid = nn.Sequential(ResBlock(mid), ResBlock(mid))
        self.to_latent = nn.Conv2d(mid, latent_channels * 2, 1)
        self.from_latent = nn.Conv2d(latent_channels, mid, 1)
        # 解码通道：mid -> chs[-1] -> chs[-2] -> ... -> chs[0]
        outputs = list(reversed(chs))       # [chs[-1], chs[-2], ..., chs[0]]
        dec_cins = outputs[:-1]             # [mid, chs[-1], ..., chs[1]]
        dec_couts = outputs[1:]             # [chs[-1], chs[-2], ..., chs[0]]
        self.ups = nn.ModuleList([
            UpBlock(dec_cins[i], dec_couts[i], num_res_blocks) for i in range(len(chs) - 1)
        ])
        self.conv_out = nn.Conv2d(chs[0], in_channels, 3, padding=1)

    def encode(self, x):
        x = self.conv_in(x)
        skips = []
        for d in self.downs:
            x, s = d(x)
            skips.append(s)
        x = self.mid(x)
        mu, logvar = self.to_latent(x).chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30, 20))
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z):
        x = self.mid(self.from_latent(z))
        for u in self.ups:
            x = u(x)
        return self.conv_out(x)
