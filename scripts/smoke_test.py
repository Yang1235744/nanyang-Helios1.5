"""Helios 1.5 冒烟测试：各子模型前向维度正确 + 流动匹配 loss 可反传。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


def main():
    torch.manual_seed(0)
    cfg = yaml.safe_load(open("configs/helios_medium.yaml", encoding="utf-8"))
    device = "cpu"

    # 1. VAE
    from helios.models import VAE
    vae = VAE(**cfg["vae"])
    x = torch.randn(1, 3, 64, 64)
    z, mu, lv = vae.encode(x)
    rec = vae.decode(z)
    print(f"[1] VAE ok  latent={tuple(z.shape)}  rec={tuple(rec.shape)}")

    # 2. TextEncoder + DiT + flow
    from helios.models import HeliosDiT, TextEncoder
    from helios.diffusion.flow import flow_loss
    from helios.models.text_encoder import encode_text, PAD_ID
    text_enc = TextEncoder(dim=128, depth=2, heads=4, max_len=64)
    dit = HeliosDiT(text_dim=128, dim=128, depth=2, heads=4, latent_channels=cfg["vae"]["latent_channels"])
    ids = torch.tensor([encode_text("海浪", 64)])
    text = text_enc(ids)
    mask = ids != PAD_ID
    z0 = torch.randn(1, cfg["vae"]["latent_channels"], 4, 8, 8)
    loss = flow_loss(dit, z0, text, mask)
    loss.backward()
    print(f"[2] DiT+flow ok  loss={loss.item():.4f}")

    # 3. AudioVAE + AudioDiT
    from helios.models import AudioDiT, AudioVAE
    from helios.diffusion.flow import flow_loss_audio
    av = AudioVAE(n_mels=128, latent=8, base=32)
    mel = torch.randn(1, 128, 100)
    az, _, _ = av.encode(mel)
    rec = av.decode(az.detach())
    print(f"[3] AudioVAE ok  az={tuple(az.shape)}  rec={tuple(rec.shape)}")
    ad = AudioDiT(text_dim=128, latent=8, dim=128, depth=2, heads=4)
    aloss = flow_loss_audio(ad, az, text.detach())
    aloss.backward()
    print(f"[4] AudioDiT+flow ok  loss={aloss.item():.4f}")

    # 4. 南阳 GPT
    from helios.models import NanyangGPT
    from helios.models.nanyang_gpt import parse_tool, TOOL_IMAGE
    gpt = NanyangGPT(dim=128, depth=2, heads=4, max_len=64)
    ids = torch.randint(0, 259, (1, 16))
    _, gloss = gpt(ids[:, :-1], ids[:, 1:])
    print(f"[5] 南阳GPT ok  loss={gloss.item():.4f}")
    text_out, tool, arg = parse_tool(f"好的，马上生成。{TOOL_IMAGE}一只猫")
    assert tool == "image" and arg == "一只猫"
    print(f"[6] 工具调用解析 ok  tool={tool} arg={arg}")

    # 5. 参数量估算
    total = sum(p.numel() for p in list(vae.parameters()) + list(text_enc.parameters())
                + list(dit.parameters()) + list(av.parameters()) + list(ad.parameters())
                + list(gpt.parameters()))
    print(f"[7] 冒烟版总参数量 {total/1e6:.1f}M（正式中等配置约 3B，fp16 约 6GB）")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
