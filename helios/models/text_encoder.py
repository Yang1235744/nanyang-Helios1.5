"""字节级文本编码器（中等，dim 1024），中英文原生支持。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
VOCAB_SIZE = 259


def encode_text(text: str, max_len: int) -> list[int]:
    ids = [BOS_ID] + list(text.encode("utf-8"))[: max_len - 1] + [EOS_ID]
    return ids + [PAD_ID] * (max_len - len(ids))


class TextEncoder(nn.Module):
    def __init__(self, dim=1024, depth=12, heads=16, max_len=256, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, activation="gelu") for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)

    def forward(self, ids):
        T = ids.shape[1]
        mask = (ids != PAD_ID).unsqueeze(1).unsqueeze(1)
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for l in self.layers:
            x = l(x, src_key_padding_mask=(ids == PAD_ID))
        return self.ln(x)
