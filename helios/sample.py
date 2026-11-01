"""采样：图片 / 15秒视频（带音频）。

python -m helios.sample --ckpt runs/helios15_medium/latest_dit.pt \\
    --audio-ckpt runs/helios15_medium/audio_dit.pt --audio-vae runs/helios15_medium/audio_vae.pt \\
    --prompt "海浪拍打沙滩" --type video --out out.mp4
"""

import argparse
import os

import numpy as np
import torch

from helios.diffusion.flow import euler_sample, euler_sample_audio
from helios.models import AudioDiT, AudioVAE, HeliosDiT, TextEncoder, VAE
from helios.models.text_encoder import PAD_ID, encode_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio-ckpt", default=None)
    ap.add_argument("--audio-vae", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--type", default="video", choices=["image", "video"])
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    text_enc = TextEncoder(**cfg["text"]).to(device).eval()
    vae = VAE(**cfg["vae"]).to(device).eval().requires_grad_(False)
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"]).to(device).eval()
    text_enc.load_state_dict(ckpt["text"]); vae.load_state_dict(ckpt["vae"]); dit.load_state_dict(ckpt["dit"])

    ml = cfg["text"]["max_len"]
    ids_c = torch.tensor([encode_text(args.prompt, ml)], device=device)
    ids_u = torch.tensor([encode_text("", ml)], device=device)
    ctx_c, ctx_u = text_enc(ids_c), text_enc(ids_u)
    mc, mu = ids_c == PAD_ID, ids_u == PAD_ID
    F = 1 if args.type == "image" else cfg["train"]["num_frames"]
    h = w = cfg["train"]["image_size"] // vae.downsample_factor
    z = euler_sample(dit, ctx_c, mc, ctx_u, mu,
                     (1, cfg["vae"]["latent_channels"], F, h, w),
                     cfg["sample"]["steps"], cfg["sample"]["cfg_scale"], device)
    px = vae.decode(z.permute(0, 2, 1, 3, 4).reshape(F, -1, h, w))
    arr = ((px.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    audio = None
    if args.type == "video" and args.audio_ckpt and args.audio_vae:
        av = AudioVAE(**{k: cfg["audio"][k] for k in ["n_mels", "audio_vae_latent", "audio_vae_base"]}).to(device).eval()
        ad = AudioDiT(text_dim=cfg["text"]["dim"], latent=cfg["audio"]["audio_vae_latent"],
                      dim=cfg["audio"]["audio_dit_dim"], depth=cfg["audio"]["audio_dit_depth"],
                      heads=cfg["audio"]["audio_dit_heads"]).to(device).eval()
        av.load_state_dict(torch.load(args.audio_vae, map_location=device)["audio_vae"])
        ad.load_state_dict(torch.load(args.audio_ckpt, map_location=device)["audio_dit"])
        az = euler_sample_audio(ad, ctx_c, (1, cfg["audio"]["audio_vae_latent"], F * (50 // cfg["sample"]["fps"])),
                                cfg["sample"]["audio_steps"], device)
        mel = av.decode(az)[0].cpu().numpy()
        audio = mel  # 后续用 Griffin-Lim 或 vocoder 转回波形并合成 mp4

    out = args.out
    if args.type == "image":
        from PIL import Image
        Image.fromarray(arr[0]).save(out + ".png")
        print(f"图片已保存: {out}.png")
    else:
        try:
            import imageio.v3 as iio
            iio.imwrite(out + ".gif", arr, duration=int(1000 / cfg["sample"]["fps"]), loop=0)
            print(f"视频已保存: {out}.gif （音频 mel 已生成，可合成 mp4）")
        except Exception:
            np.save(out + ".npy", arr)
            print(f"帧已保存: {out}.npy")
    if audio is not None:
        np.save(out + "_mel.npy", audio)
        print(f"音轨 mel 已保存: {out}_mel.npy")


if __name__ == "__main__":
    main()
