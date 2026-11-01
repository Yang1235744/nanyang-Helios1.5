"""纯 Torch 音频后端：mel 滤波器组、Griffin-Lim 声码器、WAV 写出、音视频封装。

环境里没有 librosa / torchaudio / soundfile，所以这里全部用 torch + 标准库实现，
保证 mel → 波形 → 15 秒 WAV 这条链路不依赖任何额外安装。
"""

import math
import re
import struct
import subprocess
import wave

import torch


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=24000, n_fft=1024, n_mels=128, fmin=0.0, fmax=None):
    """返回 [n_mels, n_fft//2+1] 三角滤波器组。"""
    fmax = fmax or sr / 2
    bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, bins)
    mel_pts = torch.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = torch.tensor([_mel_to_hz(float(m)) for m in mel_pts])
    fb = torch.zeros(n_mels, bins)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (freqs - left) / (center - left).clamp(min=1e-5)
        down = (right - freqs) / (right - center).clamp(min=1e-5)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


def griffin_lim(mag, n_fft=1024, hop=480, n_iter=64, length=None, device="cpu"):
    """幅度谱 [bins, T] → 波形，迭代重建相位。"""
    window = torch.hann_window(n_fft, device=device)
    inv = torch.linalg.pinv(mel_filterbank(n_fft=n_fft).to(device))
    lin = (inv @ mag).clamp(min=1e-8)          # mel → 线性幅度
    n_frames = lin.shape[-1]
    angle = torch.zeros_like(lin)
    for _ in range(n_iter):
        spec = torch.polar(lin, angle)
        wav = torch.istft(spec, n_fft=n_fft, hop_length=hop, window=window, center=True)
        s2 = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
        # istft→stft 往返可能多出一帧边界，按 lin 的帧数对齐后再取相位
        angle = torch.angle(s2[..., :n_frames]).clone()
    return torch.istft(torch.polar(lin, angle), n_fft=n_fft, hop_length=hop, window=window,
                       center=True, length=length)


def mel_to_wav(mel, sr=24000, hop=480, n_fft=1024, n_iter=64):
    """mel [n_mels, T] → 波形 [samples]，时长严格 = T*hop/sr 秒。"""
    mel = mel.detach().to("cpu")
    target = int(round(mel.shape[-1] * hop))
    wav = griffin_lim(mel, n_fft=n_fft, hop=hop, n_iter=n_iter, length=target)
    peak = wav.abs().max().clamp(min=1e-6)
    return (wav / peak * 0.95).clamp(-1, 1)


def write_wav(path, wav, sr=24000):
    """写 16-bit PCM 单声道 WAV（标准库实现）。"""
    pcm = (wav.numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video_stream(path, frame_iter, width, height, fps, crf=18):
    """把 RGB uint8 帧逐帧管道给 ffmpeg，保证帧数与输入一致、不丢帧。

    每帧都校验尺寸：尺寸不符时 ffmpeg 会静默丢帧，这里直接报错，绝不产出缺帧成片。
    """
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    n = 0
    for f in frame_iter:
        if tuple(f.shape) != (height, width, 3):
            proc.kill()
            raise ValueError(f"第 {n + 1} 帧尺寸 {tuple(f.shape)} 与声明的 {height}x{width}x3 不一致，"
                             f"继续写会导致丢帧")
        proc.stdin.write(f.astype("uint8").tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 写视频失败: " + proc.stderr.read().decode("utf-8", "ignore")[-400:])
    return n


def mux_audio(video_path, wav_path, out_path):
    """把音轨封装进视频，收尾时刻严格取视频时长。

    不用 -shortest（配合 copy 时可能提前收尾或卡住），而是显式 -t = 视频时长：
    视频流是 copy，帧数不变；音频按同一时刻截断，音画在终点处严格对齐。
    """
    dur = probe_duration(video_path)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video_path, "-i", wav_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", f"{dur:.6f}", out_path]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 封装失败: " + r.stderr.decode("utf-8", "ignore")[-400:])
    return out_path


def count_frames(path):
    """统计容器里真实的视频帧数（解码后计数），用于校验成片没丢帧。"""
    r = subprocess.run([ffmpeg_exe(), "-i", path, "-map", "0:v:0", "-f", "null", "-"],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 读帧失败: " + r.stderr.decode("utf-8", "ignore")[-400:])
    text = r.stderr.decode("utf-8", "ignore")
    # ffmpeg 进度行用 \r 反复刷新，累计计数只会递增，取最大值最稳
    vals = [int(m) for m in re.findall(r"frame=\s*(\d+)", text)]
    return max(vals) if vals else 0


def probe_duration(path):
    """读容器元信息里的视频时长（秒），用于校验帧数与音频是否对齐。"""
    import imageio_ffmpeg
    meta = imageio_ffmpeg.read_frames(path).__next__()
    return float(meta["duration"])
