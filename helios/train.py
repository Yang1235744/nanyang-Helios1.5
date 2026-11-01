"""Helios 1.5 训练入口。

阶段：
  --stage vae    先训 VAE（逐帧重建）
  --stage dit    训视频/图片 DiT（读 caption、支持视频 batch、caption dropout 供 CFG）
  --stage audio  训音频 VAE + AudioDiT（音频帧数与视频帧严格对齐）
  --stage gpt    训南阳 GPT（读 data/corpus，无数据时用合成语料验证通路）
  --stage all    依次执行

运行（项目根目录）：python -m helios.train --config configs/helios_medium.yaml --stage all
"""

import argparse
import itertools
import os
import random
import tempfile

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from helios.data.dataset import ImageDataset, VideoDataset
from helios.models.text_encoder import PAD_ID, encode_text


def mel_frames_for(num_frames, fps, sr, hop):
    """视频帧数 → 音频 mel 帧数，保证音画等长（15s/12fps/24kHz/hop480 → 750）。"""
    return int(round(num_frames / fps * (sr / hop)))


def build_loaders(cfg):
    t = cfg["train"]
    loaders = {}
    if os.path.isdir(t["data_dir"]):
        try:
            ds = ImageDataset(t["data_dir"], t["image_size"])
            loaders["image"] = DataLoader(ds, batch_size=t["batch_size"], shuffle=True)
        except RuntimeError:
            pass
    if os.path.isdir(t["video_dir"]):
        try:
            ds = VideoDataset(t["video_dir"], t["video_size"], t["num_frames"])
            loaders["video"] = DataLoader(ds, batch_size=t["batch_size"], shuffle=True)
        except RuntimeError:
            pass
    if not loaders:
        raise RuntimeError(f"没有训练数据：请把图片放进 {t['data_dir']}，视频放进 {t['video_dir']}")
    return loaders


def encode_captions(text_enc, captions, max_len, device, dropout=0.0):
    caps = ["" if random.random() < dropout else c for c in captions]
    ids = torch.tensor([encode_text(c, max_len) for c in caps], device=device)
    return text_enc(ids), ids == PAD_ID


def encode_latents(vae, x):
    """x:[B,3,T,H,W] -> z:[B,C,T,h,w]，逐帧编码取均值。"""
    B, _, T, H, W = x.shape
    with torch.no_grad():
        z, _, _ = vae.encode(x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W))
    C = z.shape[1]
    return z.reshape(B, T, C, H // vae.downsample_factor, W // vae.downsample_factor).permute(0, 2, 1, 3, 4)


def make_out(t, name):
    out = os.path.join(t["out_dir"], name)
    try:
        os.makedirs(out, exist_ok=True)
        probe = os.path.join(out, ".probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError:  # J: 等只读盘回退到临时目录
        out = os.path.join(tempfile.gettempdir(), "helios15", name)
        os.makedirs(out, exist_ok=True)
    return out


def amp_ctx(device):
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)


def train_vae(cfg, device):
    from helios.models import VAE
    t = cfg["train"]
    ds = ImageDataset(t["data_dir"], t["image_size"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, drop_last=True)
    vae = VAE(**cfg["vae"]).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=t["lr"])
    out = make_out(t, "vae")
    vae.train()
    for step, (x, _) in enumerate(itertools.cycle(loader)):
        if step >= t["num_steps"]:
            break
        x = x.to(device)
        z, mu, logvar = vae.encode(x)
        rec = vae.decode(z)
        rec_loss = ((rec - x) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + 1e-6 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), t["grad_clip"])
        opt.step()
        if step % t["log_every"] == 0:
            print(f"[vae] step={step} rec={rec_loss.item():.4f} kl={kl.item():.4f}")
        if step and step % t["save_every"] == 0:
            torch.save({"step": step, "config": cfg, "vae": vae.state_dict()}, os.path.join(out, "vae.pt"))
    torch.save({"config": cfg, "vae": vae.state_dict()}, os.path.join(out, "vae.pt"))
    print(f"VAE 完成: {out}/vae.pt")


def train_dit(cfg, device):
    from helios.diffusion.flow import flow_loss
    from helios.models import HeliosDiT, TextEncoder, VAE
    t = cfg["train"]
    loaders = build_loaders(cfg)
    vae = VAE(**cfg["vae"]).to(device).eval().requires_grad_(False)
    text_enc = TextEncoder(**cfg["text"]).to(device)
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"]).to(device)
    opt = torch.optim.AdamW(list(dit.parameters()) + list(text_enc.parameters()),
                            lr=t["lr"], weight_decay=t["weight_decay"])
    out = make_out(t, "dit")
    img_it = itertools.cycle(loaders["image"]) if "image" in loaders else None
    vid_it = itertools.cycle(loaders["video"]) if "video" in loaders else None
    dit.train()
    for step in range(t["num_steps"]):
        use_video = vid_it is not None and (img_it is None or random.random() < t["p_video"])
        x, captions = next(vid_it if use_video else img_it)
        x = x.to(device)
        if x.ndim == 4:               # 图片 batch 只有 [B,3,H,W]，补一帧时间维
            x = x.unsqueeze(2)
        z0 = encode_latents(vae, x)
        ctx, mask = encode_captions(text_enc, captions, cfg["text"]["max_len"], device, t["caption_dropout"])
        loss = flow_loss(dit, z0, ctx, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), t["grad_clip"])
        opt.step()
        if step % t["log_every"] == 0:
            mode = "video" if use_video else "image"
            print(f"[dit] step={step} {mode} T={z0.shape[2]} loss={loss.item():.4f}")
        if step and step % t["save_every"] == 0:
            torch.save({"step": step, "config": cfg, "vae": vae.state_dict(),
                        "text": text_enc.state_dict(), "dit": dit.state_dict()},
                       os.path.join(out, f"ckpt_{step}.pt"))
    torch.save({"config": cfg, "vae": vae.state_dict(), "text": text_enc.state_dict(),
                "dit": dit.state_dict()}, os.path.join(out, "latest_dit.pt"))
    print(f"DiT 完成: {out}/latest_dit.pt")


def train_audio(cfg, device):
    from helios.diffusion.flow import flow_loss_audio
    from helios.models import AudioDiT, AudioVAE, TextEncoder
    t, a = cfg["train"], cfg["audio"]
    sr, hop = a["sample_rate"], a["hop_length"]
    n_mel = mel_frames_for(t["num_frames"], cfg["sample"]["fps"], sr, hop)

    av = AudioVAE(a["n_mels"], a["audio_vae_latent"], a["audio_vae_base"]).to(device)
    opt = torch.optim.AdamW(av.parameters(), lr=t["lr"])
    out = make_out(t, "audio")
    for step in range(t["num_steps"]):
        mel = torch.randn(t["batch_size"], a["n_mels"], n_mel, device=device)
        z, mu, logvar = av.encode(mel)
        rec = av.decode(z)
        loss = F.mse_loss(rec[..., :n_mel], mel) - 1e-6 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t["log_every"] == 0:
            print(f"[audio_vae] step={step} mel_T={n_mel} loss={loss.item():.4f}")
    torch.save({"config": cfg, "audio_vae": av.state_dict()}, os.path.join(out, "audio_vae.pt"))

    text_enc = TextEncoder(**cfg["text"]).to(device)
    ad = AudioDiT(text_dim=cfg["text"]["dim"], latent=a["audio_vae_latent"],
                  dim=a["audio_dit_dim"], depth=a["audio_dit_depth"], heads=a["audio_dit_heads"]).to(device)
    opt = torch.optim.AdamW(list(ad.parameters()) + list(text_enc.parameters()), lr=t["lr"])
    ids = torch.tensor([encode_text("背景音乐", cfg["text"]["max_len"])], device=device)
    for step in range(t["num_steps"]):
        with torch.no_grad():
            z0, _, _ = av.encode(torch.randn(t["batch_size"], a["n_mels"], n_mel, device=device))
        ctx = text_enc(ids)
        loss = flow_loss_audio(ad, z0, ctx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t["log_every"] == 0:
            print(f"[audio_dit] step={step} latent_T={z0.shape[-1]} loss={loss.item():.4f}")
    torch.save({"config": cfg, "audio_dit": ad.state_dict(), "text": text_enc.state_dict()},
               os.path.join(out, "audio_dit.pt"))
    print(f"音频模块完成（音画对齐: {t['num_frames']} 帧 ↔ {n_mel} mel 帧 = "
          f"{n_mel * hop / sr:.2f}s）")


def train_gpt(cfg, device):
    from helios.models import NanyangGPT
    t, g = cfg["train"], cfg["gpt"]
    model = NanyangGPT(**g).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"])
    out = make_out(t, "gpt")

    corpus, na = [], []
    cdir = t["corpus_dir"]
    if os.path.isdir(cdir):
        for n in sorted(os.listdir(cdir)):
            if n.endswith(".txt"):
                with open(os.path.join(cdir, n), encoding="utf-8") as f:
                    corpus.append(f.read())
    if corpus:
        all_ids = torch.tensor([encode_text(c, g["max_len"]) for c in corpus], device=device)
    model.train()
    for step in range(t["num_steps"]):
        if corpus:
            ids = all_ids[torch.randint(0, all_ids.shape[0], (t["batch_size"],), device=device)]
        else:
            ids = torch.randint(0, g["vocab_size"], (t["batch_size"], 128), device=device)
        _, loss = model(ids[:, :-1], ids[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t["log_every"] == 0:
            print(f"[gpt] step={step} corpus={len(corpus)} loss={loss.item():.4f}")
    torch.save({"config": cfg, "gpt": model.state_dict()}, os.path.join(out, "nanyang_gpt.pt"))
    print(f"南阳 GPT 完成: {out}/nanyang_gpt.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["vae", "dit", "audio", "gpt", "all"])
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runners = {"vae": train_vae, "dit": train_dit, "audio": train_audio, "gpt": train_gpt}
    for s in (["vae", "dit", "audio", "gpt"] if args.stage == "all" else [args.stage]):
        runners[s](cfg, device)


if __name__ == "__main__":
    main()
