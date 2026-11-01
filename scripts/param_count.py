"""用正式中等配置估算参数量与权重体积（meta 设备，不占显存）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

cfg = yaml.safe_load(open("configs/helios_medium.yaml", encoding="utf-8"))
device = torch.device("meta")

from helios.models import AudioDiT, AudioVAE, HeliosDiT, NanyangGPT, TextEncoder, VAE

with torch.device(device):
    vae = VAE(**cfg["vae"])
    text_enc = TextEncoder(**cfg["text"])
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"])
    av = AudioVAE(cfg["audio"]["n_mels"], cfg["audio"]["audio_vae_latent"], cfg["audio"]["audio_vae_base"])
    ad = AudioDiT(text_dim=cfg["text"]["dim"], latent=cfg["audio"]["audio_vae_latent"],
                  dim=cfg["audio"]["audio_dit_dim"], depth=cfg["audio"]["audio_dit_depth"],
                  heads=cfg["audio"]["audio_dit_heads"])
    gpt = NanyangGPT(**cfg["gpt"])

parts = [("VAE", vae), ("TextEncoder", text_enc), ("HeliosDiT", dit),
         ("AudioVAE", av), ("AudioDiT", ad), ("南阳GPT", gpt)]
total = 0
for name, m in parts:
    n = sum(p.numel() for p in m.parameters())
    total += n
    print(f"{name:15s} {n/1e6:10.1f} M")
print(f"{'TOTAL':15s} {total/1e6:10.1f} M  ({total/1e9:.2f} B)")
print(f"fp16 权重: {total*2/1e9:.2f} GB")
print(f"fp32 权重: {total*4/1e9:.2f} GB")
