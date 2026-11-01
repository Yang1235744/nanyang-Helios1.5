# nanyang-Helios 1.5 —— 中等视频/图片生成模型（含音频 + 南阳 GPT）

从零自研的中等规模多模态生成模型，对标豆包/千问的生成能力。在 [1.2 低等版](https://github.com/Yang1235744/nanyang-Helios1.2) 基础上全面升级。

## 能力

| 能力 | 规格 |
| --- | --- |
| 图片生成 | 最高 1024px 高清 |
| 视频生成 | **15 秒**（默认 12fps = 180 帧） |
| 音频生成 | 与视频同步的 15 秒音轨（24kHz） |
| 对话 | 捆绑南阳 GPT，先聊天再生成 |
| 权重规模 | 1.38B 参数 / fp32 约 **5.5 GB** |

## 架构

- `helios/models/vae.py` —— 中等 2D VAE（8x 下采样，16 通道 latent，base 128）
- `helios/models/text_encoder.py` —— 字节级文本编码器（dim 1024，中英文通吃）
- `helios/models/dit.py` —— HeliosDiT：**时空解耦注意力**（长视频关键）+ 文本交叉注意力 + adaLN-Zero
- `helios/models/audio.py` —— AudioVAE（mel↔latent）+ AudioDiT（1D 流动匹配生成音轨）
- `helios/models/nanyang_gpt.py` —— 南阳 GPT：对话大模型 + 工具调用（`<|gen_image|>` / `<|gen_video|>`）
- `helios/diffusion/flow.py` —— Rectified Flow + Euler 采样 + CFG

## 训练（四阶段）

```bash
# 1. VAE
python -m helios.train --config configs/helios_medium.yaml --stage vae
# 2. 视频/图片 DiT
python -m helios.train --config configs/helios_medium.yaml --stage dit
# 3. 音频（VAE + AudioDiT）
python -m helios.train --config configs/helios_medium.yaml --stage audio
# 4. 南阳 GPT
python -m helios.train --config configs/helios_medium.yaml --stage gpt
# 或一次性全跑
python -m helios.train --config configs/helios_medium.yaml --stage all
```

数据目录：`data/images/`（图片）、`data/videos/`（视频）、`data/audio/`（音频）、`data/corpus/`（对话语料）。

## 采样

```bash
# 图片
python -m helios.sample --ckpt runs/helios15_medium/latest_dit.pt \
    --prompt "雪地里的红狐" --type image --out fox

# 15 秒视频 + 音轨
python -m helios.sample --ckpt runs/helios15_medium/latest_dit.pt \
    --audio-vae runs/helios15_medium/audio_vae.pt \
    --audio-ckpt runs/helios15_medium/audio_dit.pt \
    --prompt "海浪拍打沙滩" --type video --out waves
```

## 验证

```bash
python scripts/smoke_test.py     # 7 项冒烟全过
python scripts/param_count.py    # 参数量 / 权重体积
```

## 路线图

- [x] 1.2：低等版，基础图片/短 GIF
- [x] 1.5：中等版，15s 视频 + 音频 + HD 图片 + 南阳 GPT 捆绑
- [ ] 2.0：大规模版，因果 3D VAE、时空分解注意力、EMA、更高分辨率

## 许可

MIT
