"""采样：图片 / 15 秒视频（带同步音轨）。

音画对齐是硬保证：
  - 视频 = F 帧 @ fps → 时长 F/fps 秒，逐帧写入 mp4，不丢帧；
  - 音频 = round(F/fps * sr/hop) 个 mel 帧 → 时长完全相同；
  - 最后用 ffmpeg 封装并按 -shortest 收尾。

示例：
python -m helios.sample --ckpt runs/helios15_medium/latest_dit.pt \
    --audio-vae runs/helios15_medium/audio_vae.pt \
    --audio-ckpt runs/helios15_medium/audio_dit.pt \
    --prompt "海浪拍打沙滩" --type video --out waves.mp4

若音频权重不存在，只出无声视频（不会静默失败，会明确打印原因）。
"""

import argparse
import os

import torch

from helios.audio_vocoder import (count_frames, mel_to_wav, mux_audio, probe_duration,
                                 write_video_stream, write_wav)
from helios.diffusion.flow import euler_sample, euler_sample_audio
from helios.models import AudioDiT, AudioVAE, HeliosDiT, TextEncoder, VAE
from helios.models.text_encoder import PAD_ID, encode_text


def latents_to_frames(vae, z, chunk=8):
    """[B,C,T,h,w] → list[H,W,3] uint8，分块解码避免长视频爆显存，帧数一分不少。"""
    B, C, T, h, w = z.shape
    frames = []
    with torch.no_grad():
        for i in range(0, T, chunk):
            part = z[:, :, i:i + chunk]
            px = vae.decode(part.permute(0, 2, 1, 3, 4).reshape(-1, C, h, w))
            arr = ((px.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            frames.extend(arr)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--type", default="video", choices=["image", "video"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--size", type=int, default=None, help="像素边长，默认取配置")
    ap.add_argument("--frames", type=int, default=None, help="视频帧数，默认取配置")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--audio-ckpt", default=None)
    ap.add_argument("--audio-vae", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sample_cfg = cfg["sample"]
    fps = args.fps or sample_cfg["fps"]
    steps = args.steps or sample_cfg["steps"]
    cfg_scale = args.cfg if args.cfg is not None else sample_cfg["cfg_scale"]

    text_enc = TextEncoder(**cfg["text"]).to(device).eval()
    vae = VAE(**cfg["vae"]).to(device).eval().requires_grad_(False)
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"]).to(device).eval()
    text_enc.load_state_dict(ckpt["text"])
    vae.load_state_dict(ckpt["vae"])
    dit.load_state_dict(ckpt["dit"])

    F = 1 if args.type == "image" else (args.frames or cfg["train"]["num_frames"])
    size = args.size or (cfg["train"]["image_size"] if args.type == "image" else cfg["train"]["video_size"])
    ml = cfg["text"]["max_len"]
    ids_c = torch.tensor([encode_text(args.prompt, ml)], device=device)
    ids_u = torch.tensor([encode_text(args.negative, ml)], device=device)
    ctx_c, ctx_u = text_enc(ids_c), text_enc(ids_u)
    mask_c, mask_u = ids_c == PAD_ID, ids_u == PAD_ID

    h = w = size // vae.downsample_factor
    z = euler_sample(dit, ctx_c, mask_c, ctx_u, mask_u,
                     (1, cfg["vae"]["latent_channels"], F, h, w), steps, cfg_scale, device)
    frames = latents_to_frames(vae, z)
    assert len(frames) == F, f"解码帧数 {len(frames)} != 预期 {F}"

    out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out else "."
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(args.out))[0] or "out")

    if args.type == "image":
        from PIL import Image
        Image.fromarray(frames[0]).save(stem + ".png")
        print(f"图片已保存: {stem}.png ({size}x{size})")
        return

    # ---------- 音频：帧数与视频严格对齐 ----------
    a = cfg["audio"]
    sr, hop = a["sample_rate"], a["hop_length"]
    mel_frames = int(round(F / fps * (sr / hop)))
    wav_path = None
    if args.audio_ckpt and args.audio_vae:
        av = AudioVAE(a["n_mels"], a["audio_vae_latent"], a["audio_vae_base"]).to(device).eval()
        ad = AudioDiT(text_dim=cfg["text"]["dim"], latent=a["audio_vae_latent"],
                      dim=a["audio_dit_dim"], depth=a["audio_dit_depth"],
                      heads=a["audio_dit_heads"]).to(device).eval()
        av.load_state_dict(torch.load(args.audio_vae, map_location=device, weights_only=False)["audio_vae"])
        ad.load_state_dict(torch.load(args.audio_ckpt, map_location=device, weights_only=False)["audio_dit"])
        latent_len = mel_frames // 4 + 1          # AudioVAE 两级 stride2 下采样
        az = euler_sample_audio(ad, ctx_c, (1, a["audio_vae_latent"], latent_len),
                                sample_cfg["audio_steps"], device)
        mel = av.decode(az)[0][:, :mel_frames].cpu()
        wav = mel_to_wav(mel, sr=sr, hop=hop, n_fft=a.get("n_fft", 1024), n_iter=a.get("vocoder_iter", 64))
        wav_path = stem + ".wav"
        write_wav(wav_path, wav, sr)
        print(f"音轨已保存: {wav_path} ({wav.shape[-1] / sr:.2f}s, mel={mel_frames})")
    else:
        print("未提供 --audio-ckpt/--audio-vae，仅生成无声视频")

    # ---------- 视频：逐帧管道写入，不丢帧 ----------
    fh, fw = frames[0].shape[0], frames[0].shape[1]
    silent = stem + ".nosound.mp4"          # 中间文件，避免和最终 mp4 同名
    n = write_video_stream(silent, frames, fw, fh, fps)
    print(f"视频帧写入: {n} 帧 @ {fps}fps = {n / fps:.2f}s")

    if wav_path:
        final = args.out if args.out.lower().endswith(".mp4") else stem + ".mp4"
        mux_audio(silent, wav_path, final)
        os.remove(silent)
        print(f"成品（音画封装）: {final}")
    else:
        final = args.out if args.out.lower().endswith(".mp4") else stem + ".mp4"
        if os.path.abspath(silent) != os.path.abspath(final):
            os.replace(silent, final)
        print(f"成品（无声）: {final}")

    # ---------- 成片自检：帧数一帧不少、时长与音频严格对齐 ----------
    got = count_frames(final)
    dur = probe_duration(final)
    assert got == F, f"成片帧数 {got} != 生成帧数 {F}，出现丢帧"
    assert abs(dur - F / fps) < 0.2, f"成片时长 {dur:.2f}s 与目标 {F / fps:.2f}s 不符"
    print(f"自检通过: {got} 帧 @ {fps}fps = {got / fps:.2f}s，音频 {mel_frames * hop / sr:.2f}s")


if __name__ == "__main__":
    main()
