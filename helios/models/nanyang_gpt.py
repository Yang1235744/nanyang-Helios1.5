"""南阳 GPT —— 中等对话大模型，带工具调用（生成图片 / 生成视频）。

与 Helios 1.5 捆绑为统一中等模型：先对话，需要时触发图像/视频生成。
工具调用通过文本内的标记实现：
  <|gen_image|>描述
  <|gen_video|>描述
模型输出里出现这些标记即被解析为工具调用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE, encode_text

USER, BOT = "用户：", "助手："
TOOL_IMAGE = "<|gen_image|>"
TOOL_VIDEO = "<|gen_video|>"


class GPTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.n2(x))


class NanyangGPT(nn.Module):
    def __init__(self, dim=1024, depth=24, heads=16, max_len=2048, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([GPTBlock(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new=512, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(ids[:, -self.max_len:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            ids = torch.cat([ids, nxt], 1)
            if (nxt == EOS_ID).all():
                break
        return ids


def parse_tool(text: str):
    """解析模型输出，返回 (回复文本, 工具类型或 None, 工具参数)。"""
    if TOOL_IMAGE in text:
        prompt = text.split(TOOL_IMAGE, 1)[1].strip()
        return text.split(TOOL_IMAGE)[0].strip(), "image", prompt
    if TOOL_VIDEO in text:
        prompt = text.split(TOOL_VIDEO, 1)[1].strip()
        return text.split(TOOL_VIDEO)[0].strip(), "video", prompt
    return text.strip(), None, None
