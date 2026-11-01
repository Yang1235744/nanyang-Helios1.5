"""数据集：图片（同名 .txt 为 caption）+ 视频（同名 .txt 为 caption）。

视频帧数不足时循环补帧、超出时均匀抽帧，保证每个样本恰好 num_frames 帧，不丢帧。
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _caption(path):
    txt = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(txt):
        with open(txt, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _to_tensor(img):
    x = np.array(img.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1) / 127.5 - 1.0


class ImageDataset(Dataset):
    def __init__(self, root, size):
        self.files = []
        for dp, _, ns in os.walk(root):
            self.files += [os.path.join(dp, n) for n in ns if n.lower().endswith(IMG_EXT)]
        self.size = size
        if not self.files:
            raise RuntimeError(f"图片目录为空: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return _to_tensor(img), _caption(path) or "一张图片"


class VideoDataset(Dataset):
    def __init__(self, root, size, num_frames):
        self.files = []
        for dp, _, ns in os.walk(root):
            self.files += [os.path.join(dp, n) for n in ns if n.lower().endswith((".mp4", ".gif", ".avi", ".mov"))]
        self.size = size
        self.num_frames = num_frames
        if not self.files:
            raise RuntimeError(f"视频目录为空: {root}")

    def __len__(self):
        return len(self.files)

    def _read_frames(self, path):
        import imageio.v3 as iio
        # imageio v3 的插件名大小写敏感：视频走 FFMPEG 后端，GIF 走 pillow
        plugin = "FFMPEG" if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) else "pillow"
        frames = list(iio.imiter(path, plugin=plugin))
        imgs = [Image.fromarray(np.asarray(f)).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
                for f in frames]
        n = len(imgs)
        if n == 0:
            raise RuntimeError(f"读不到帧: {path}")
        if n >= self.num_frames:  # 均匀抽帧，保证首尾都在
            idx = np.linspace(0, n - 1, self.num_frames).round().astype(int)
            imgs = [imgs[i] for i in idx]
        else:  # 循环补帧，绝不留空帧
            imgs = [imgs[i % n] for i in range(self.num_frames)]
        return torch.stack([_to_tensor(im) for im in imgs], dim=1)  # [3,T,H,W]

    def __getitem__(self, i):
        path = self.files[i]
        return self._read_frames(path), _caption(path) or "一段视频"
