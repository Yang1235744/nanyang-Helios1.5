"""Helios 1.5 冒烟测试：模型通路 + 音画对齐 + 端到端出片（不依赖训练权重）。"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


def main():
    torch.manual_seed(0)
    cfg = yaml.safe_load(open("configs/helios_medium.yaml", encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="helios15-smoke-")

    # ---- 1. VAE 重建，通道映射正确
    from helios.models import VAE
    vae = VAE(**cfg["vae"])
    x = torch.randn(1, 3, 64, 64)
    z, mu, lv = vae.encode(x)
    rec = vae.decode(z)
    assert z.shape[1] == cfg["vae"]["latent_channels"] and rec.shape == x.shape
    print(f"[1] VAE ok  latent={tuple(z.shape)} rec={tuple(rec.shape)}")

    # ---- 2. DiT：patchify + 时空可分离位置编码 + 时空解耦注意力 + 文本维度≠DiT 维度
    from helios.diffusion.flow import flow_loss
    from helios.models import HeliosDiT, TextEncoder
    from helios.models.text_encoder import encode_text, PAD_ID
    text_enc = TextEncoder(dim=64, depth=2, heads=4, max_len=64)
    dit = HeliosDiT(text_dim=64, dim=128, depth=2, heads=4, patch_size=2,
                    latent_channels=cfg["vae"]["latent_channels"], max_frames=16, max_h=8, max_w=8)
    ids = torch.tensor([encode_text("海浪", 64)])
    ctx, mask = text_enc(ids), ids == PAD_ID
    z0 = torch.randn(1, cfg["vae"]["latent_channels"], 4, 16, 16)   # 4 帧 × 16x16 latent
    loss = flow_loss(dit, z0, ctx, mask)
    loss.backward()
    assert dit.pos_t.shape[0] == 16 and dit.pos_h.shape[0] == 8
    print(f"[2] DiT ok  tokens={4 * 64} 时空解耦 loss={loss.item():.4f}")

    # ---- 3. 音频 VAE + AudioDiT
    from helios.diffusion.flow import flow_loss_audio
    from helios.models import AudioDiT, AudioVAE
    av = AudioVAE(n_mels=128, latent=8, base=32)
    mel = torch.randn(1, 128, 100)
    az, _, _ = av.encode(mel)
    rec_mel = av.decode(az.detach())
    assert rec_mel.shape[-1] >= 100
    ad = AudioDiT(text_dim=64, latent=8, dim=128, depth=2, heads=4)
    aloss = flow_loss_audio(ad, az, ctx.detach())
    aloss.backward()
    print(f"[3] 音频 ok  latent={tuple(az.shape)} loss={aloss.item():.4f}")

    # ---- 4. 南阳 GPT + 工具调用解析
    from helios.models import NanyangGPT
    from helios.models.nanyang_gpt import TOOL_IMAGE, TOOL_VIDEO, parse_tool
    gpt = NanyangGPT(dim=128, depth=2, heads=4, max_len=64)
    ids = torch.randint(0, 259, (1, 16))
    _, gloss = gpt(ids[:, :-1], ids[:, 1:])
    reply, tool, arg = parse_tool(f"好的，马上生成。{TOOL_IMAGE}一只猫")
    assert tool == "image" and arg == "一只猫"
    reply2, tool2, arg2 = parse_tool(f"来一段视频。{TOOL_VIDEO}海边日落")
    assert tool2 == "video" and arg2 == "海边日落"
    print(f"[4] 南阳GPT ok  loss={gloss.item():.4f}  工具解析 ok")

    # ---- 5. 音画对齐（硬性数学关系）
    from helios.train import mel_frames_for
    sr, hop, fps, F = cfg["audio"]["sample_rate"], cfg["audio"]["hop_length"], cfg["sample"]["fps"], cfg["train"]["num_frames"]
    n_mel = mel_frames_for(F, fps, sr, hop)
    video_sec = F / fps
    audio_sec = n_mel * hop / sr
    assert n_mel == 750, n_mel
    assert abs(video_sec - audio_sec) < 1e-9, (video_sec, audio_sec)
    print(f"[5] 对齐 ok  {F} 帧@{fps}fps = {video_sec:.2f}s ↔ {n_mel} mel = {audio_sec:.2f}s")

    # ---- 6. 视频数据集不丢帧（短素材循环补帧、长素材均匀抽帧；gif + mp4 两种后端）
    import numpy as np
    from PIL import Image
    import imageio.v3 as iio
    from helios.audio_vocoder import write_video_stream
    from helios.data.dataset import VideoDataset
    vdir = os.path.join(tmp, "videos")
    os.makedirs(vdir, exist_ok=True)
    short = os.path.join(vdir, "short.gif")
    iio.imwrite(short, np.stack([np.full((16, 16, 3), i * 40, dtype="uint8") for i in range(5)]), loop=0)
    long_mp4 = os.path.join(vdir, "long.mp4")
    write_video_stream(long_mp4, [np.full((16, 16, 3), i * 8, dtype="uint8") for i in range(20)], 16, 16, 12)
    ds = VideoDataset(vdir, 16, 12)
    names = [os.path.basename(f) for f in ds.files]
    for name in ("short.gif", "long.mp4"):
        frames, cap = ds[names.index(name)]
        assert frames.shape == (3, 12, 16, 16), (name, frames.shape)
        first, last = frames[:, 0].mean().item(), frames[:, -1].mean().item()
        assert first != last, f"{name} 首尾帧相同，抽帧没覆盖整段"
    print(f"[6] 视频集 ok  5 帧 gif → 补帧 12；20 帧 mp4 → 抽帧 12，首尾帧不同")

    # ---- 7. 端到端出片：视频帧数 + 音轨时长一致
    from helios.audio_vocoder import mel_to_wav, mux_audio, write_wav
    from helios.diffusion.flow import euler_sample, euler_sample_audio
    from helios.sample import latents_to_frames
    Fs, size = 12, 32
    h = w = size // vae.downsample_factor
    zs = euler_sample(dit, ctx, mask, ctx, mask,
                      (1, cfg["vae"]["latent_channels"], Fs, h, w), 2, 1.0, "cpu")
    frames = latents_to_frames(vae, zs)
    assert len(frames) == Fs
    mel_n = mel_frames_for(Fs, fps, sr, hop)
    az = euler_sample_audio(ad, ctx, (1, 8, mel_n // 4 + 1), 2, "cpu")
    mel_out = av.decode(az)[0][:, :mel_n]
    wav = mel_to_wav(mel_out, sr=sr, hop=hop, n_fft=cfg["audio"]["n_fft"], n_iter=16)
    assert wav.shape[-1] == mel_n * hop
    silent = os.path.join(tmp, "v.mp4")
    n = write_video_stream(silent, frames, size, size, fps)
    wav_path = write_wav(os.path.join(tmp, "a.wav"), wav, sr)
    final = mux_audio(silent, wav_path, os.path.join(tmp, "out.mp4"))
    from helios.audio_vocoder import count_frames, probe_duration
    got = count_frames(final)
    dur = probe_duration(final)
    assert n == Fs and got == Fs, (n, got)
    assert abs(dur - Fs / fps) < 0.2, (n, dur)
    print(f"[7] 出片 ok  写入 {n} 帧 / 成片 {got} 帧 → mp4 {dur:.2f}s（目标 {Fs / fps:.2f}s），"
          f"音轨 {wav.shape[-1] / sr:.2f}s")
    print(f"     产物目录: {tmp}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
