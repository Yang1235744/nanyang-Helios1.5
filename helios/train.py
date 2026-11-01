"""Helios 1.5 训练入口。

阶段：
  --stage vae      先训 VAE（逐帧重建）
  --stage dit      训视频/图片 DiT
  --stage audio    训音频 VAE + AudioDiT
  --stage gpt      训南阳 GPT 语言模型
  --stage all      依次执行

运行（项目根目录）：python -m helios.train --config configs/helios_medium.yaml --stage all
"""

import argparse
import os
import tempfile

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class FrameDataset(Dataset):
    def __init__(self, root, size):
        self.files = []
        for dp, _, ns in os.walk(root):
            self.files += [os.path.join(dp, n) for n in ns
                           if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        self.size = size
        if not self.files:
            raise RuntimeError(f"图片目录为空: {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB").resize((self.size, self.size))
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1
        return x


def make_out(cfg_stage):
    out = cfg_stage["out_dir"]
    try:
        os.makedirs(out, exist_ok=True)
    except OSError:
        out = os.path.join(tempfile.gettempdir(), "helios15", cfg_stage.get("name", "out"))
        os.makedirs(out, exist_ok=True)
    return out


def train_vae(cfg, device):
    from helios.models import VAE
    t = cfg["train"]
    ds = FrameDataset(t["data_dir"], t["image_size"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True)
    vae = VAE(**cfg["vae"]).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=t["lr"])
    out = make_out({**t, "name": "vae"})
    vae.train()
    for step, x in enumerate(loader):
        x = x.to(device)
        z, mu, logvar = vae.encode(x)
        rec = vae.decode(z)
        rec_loss = ((rec - x) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + 1e-6 * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % t["log_every"] == 0:
            print(f"[vae] step={step} rec={rec_loss.item():.4f} kl={kl.item():.4f}")
        if step and step % t["save_every"] == 0:
            torch.save({"step": step, "config": cfg, "vae": vae.state_dict()},
                       os.path.join(out, "vae.pt"))
        if step >= t.get("num_steps_per_frame", 1000):
            break
    torch.save({"config": cfg, "vae": vae.state_dict()}, os.path.join(out, "vae.pt"))
    print(f"VAE 完成: {out}/vae.pt")


def train_dit(cfg, device):
    from helios.models import HeliosDiT, TextEncoder, VAE
    from helios.diffusion.flow import flow_loss
    from helios.models.text_encoder import encode_text, PAD_ID
    t = cfg["train"]
    ds = FrameDataset(t["data_dir"], t["image_size"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True)
    vae = VAE(**cfg["vae"]).to(device).eval().requires_grad_(False)
    text_enc = TextEncoder(**cfg["text"]).to(device)
    dit = HeliosDiT(text_dim=cfg["text"]["dim"], **cfg["dit"]).to(device)
    opt = torch.optim.AdamW(dit.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    out = make_out({**t, "name": "dit"})
    dit.train()
    for step, x in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            z0, _, _ = vae.encode(x.unsqueeze(2))  # B,C,1,H,W
        prompt = "一张图片"
        ids = torch.tensor([encode_text(prompt, cfg["text"]["max_len"])], device=device).repeat(x.shape[0], 1)
        mask = (ids != PAD_ID)
        text = text_enc(ids)
        loss = flow_loss(dit, z0, text, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), t["grad_clip"])
        opt.step()
        if step % t["log_every"] == 0:
            print(f"[dit] step={step} loss={loss.item():.4f}")
        if step and step % t["save_every"] == 0:
            torch.save({"step": step, "config": cfg, "vae": vae.state_dict(),
                        "text": text_enc.state_dict(), "dit": dit.state_dict()},
                       os.path.join(out, f"ckpt_{step}.pt"))
        if step >= t.get("num_steps_per_frame", 1000):
            break
    torch.save({"config": cfg, "vae": vae.state_dict(), "text": text_enc.state_dict(),
                "dit": dit.state_dict()}, os.path.join(out, "latest_dit.pt"))
    print(f"DiT 完成: {out}/latest_dit.pt")


def train_audio(cfg, device):
    from helios.models import AudioVAE, AudioDiT, TextEncoder
    from helios.diffusion.flow import flow_loss_audio
    t = cfg["train"]
    audio = cfg["audio"]
    audio_vae = AudioVAE(audio["n_mels"], audio["audio_vae_latent"], audio["audio_vae_base"]).to(device)
    opt_v = torch.optim.AdamW(audio_vae.parameters(), lr=t["lr"])
    out = make_out({**t, "name": "audio_vae"})
    # 用合成 mel 验证通路（真实数据放入 data/audio 后替换）
    for step in range(t.get("num_steps_per_frame", 1000)):
        mel = torch.randn(t["batch_size"], audio["n_mels"], 750, device=device)
        z, mu, logvar = audio_vae.encode(mel)
        rec = audio_vae.decode(z)
        loss = ((rec - mel) ** 2).mean() - 1e-6 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        opt_v.zero_grad(); loss.backward(); opt_v.step()
        if step % t["log_every"] == 0:
            print(f"[audio_vae] step={step} loss={loss.item():.4f}")
    torch.save({"config": cfg, "audio_vae": audio_vae.state_dict()}, os.path.join(out, "audio_vae.pt"))

    text_enc = TextEncoder(**cfg["text"]).to(device)
    dit = AudioDiT(text_dim=cfg["text"]["dim"], latent=audio["audio_vae_latent"],
                   dim=audio["audio_dit_dim"], depth=audio["audio_dit_depth"],
                   heads=audio["audio_dit_heads"]).to(device)
    opt = torch.optim.AdamW(dit.parameters(), lr=t["lr"])
    out2 = make_out({**t, "name": "audio_dit"})
    ids = torch.tensor([encode_text("背景音乐", cfg["text"]["max_len"])], device=device)
    text = text_enc(ids)
    for step in range(t.get("num_steps_per_frame", 1000)):
        z0 = torch.randn(t["batch_size"], audio["audio_vae_latent"], 188, device=device)  # 750/4
        loss = flow_loss_audio(dit, z0, text)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % t["log_every"] == 0:
            print(f"[audio_dit] step={step} loss={loss.item():.4f}")
    torch.save({"config": cfg, "audio_dit": dit.state_dict()}, os.path.join(out2, "audio_dit.pt"))
    print("音频模块完成")


def train_gpt(cfg, device):
    from helios.models import NanyangGPT
    t = cfg["train"]
    gpt_cfg = cfg["gpt"]
    model = NanyangGPT(**gpt_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"])
    out = make_out({**t, "name": "gpt"})
    model.train()
    # 合成语料验证通路；真实对话放入 data/corpus/*.txt
    for step in range(t.get("num_steps_per_frame", 1000)):
        ids = torch.randint(0, gpt_cfg["vocab_size"], (t["batch_size"], 128), device=device)
        _, loss = model(ids[:, :-1], ids[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % t["log_every"] == 0:
            print(f"[gpt] step={step} loss={loss.item():.4f}")
    torch.save({"config": cfg, "gpt": model.state_dict()}, os.path.join(out, "nanyang_gpt.pt"))
    print("南阳 GPT 完成")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["vae", "dit", "audio", "gpt", "all"])
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stages = ["vae", "dit", "audio", "gpt"] if args.stage == "all" else [args.stage]
    runners = {"vae": train_vae, "dit": train_dit, "audio": train_audio, "gpt": train_gpt}
    for s in stages:
        runners[s](cfg, device)


if __name__ == "__main__":
    main()
