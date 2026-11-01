"""南阳 GPT 视觉编码器 —— 从零实现的看图模块（与 nanyang-GPT 独立仓库同源）。

图片(64x64)经过 3 级 stride-2 卷积下采样到 8x8 特征图，展平成 64 个视觉 token，
逐 token 线性投影到 GPT 维度，作为前缀喂给语言模型。
与文本 token 共用因果注意力：模型在后续文本位置可以"看到"全部视觉 token。
"""

import torch
import torch.nn as nn


class TinyVisionEncoder(nn.Module):
    """64px -> 8x8=64 个视觉 token。"""

    def __init__(self, dim, base=32, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.grid = image_size // 8                       # 8x8
        self.n_tokens = self.grid * self.grid             # 64
        ch = [3, base, base * 2, base * 4]
        self.conv = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 3, 2, 1), nn.GroupNorm(8, ch[1]), nn.SiLU(),
            nn.Conv2d(ch[1], ch[2], 3, 2, 1), nn.GroupNorm(8, ch[2]), nn.SiLU(),
            nn.Conv2d(ch[2], ch[3], 3, 2, 1), nn.GroupNorm(8, ch[3]), nn.SiLU(),
        )
        self.proj = nn.Sequential(nn.Linear(ch[3], dim), nn.LayerNorm(dim))

    def forward(self, img):
        """img: [B,3,S,S] ∈ [-1,1] -> [B, n_tokens, dim]"""
        h = self.conv(img)                                # [B, C, g, g]
        h = h.flatten(2).transpose(1, 2)                  # [B, g*g, C]
        return self.proj(h)


def load_image(path, size):
    """读图 -> [1,3,size,size] ∈ [-1,1]（中心裁剪成正方形后缩放）。"""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize(
        (size, size), Image.LANCZOS)
    t = torch.tensor(list(im.tobytes()), dtype=torch.float32).reshape(size, size, 3)
    return (t.permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0)
