import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lightning.pytorch as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


# -----------------------------
# Data
# -----------------------------

def load_mnist(root: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    xtr = tr.data.float().div(255).unsqueeze(1)
    xte = te.data.float().div(255).unsqueeze(1)
    return xtr, tr.targets, xte, te.targets


def add_noise(x: Tensor, sigma: float, gen: torch.Generator) -> Tensor:
    noise = torch.randn(x.shape, generator=gen) * sigma
    return (x + noise).clamp(0.0, 1.0)


# -----------------------------
# Models
# -----------------------------

class VAE(pl.LightningModule):
    def __init__(self, latent_dim: int = 16, lr: float = 1e-3, beta: float = 1.0,
                 train_noise: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),   # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 7x7
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),              # 28x28 (logits)
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode_logits(self, z: Tensor) -> Tensor:
        return self.dec(self.fc_dec(z))

    def _loss(self, x: Tensor):
        x_in = x
        if self.training and self.hparams.train_noise > 0:
            sigma = torch.rand(x.size(0), 1, 1, 1, device=x.device) * self.hparams.train_noise
            x_in = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
        mu, logvar = self.encode(x_in)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits = self.decode_logits(z)
        bs = x.size(0)
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / bs
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / bs
        return rec + self.hparams.beta * kl, rec, kl

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, rec, kl = self._loss(x)
        self.log_dict({"train/loss": loss, "train/rec": rec, "train/kl": kl},
                      prog_bar=True, on_epoch=True, on_step=False, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss, _, _ = self._loss(x)
        self.log("val/loss", loss, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class Classifier(pl.LightningModule):

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        acc = (self(x).argmax(1) == y).float().mean()
        self.log("val/acc", acc, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -----------------------------
# Helpers
# -----------------------------

def make_trainer(epochs: int) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


@torch.no_grad()
def reconstruct(vae: VAE, x: Tensor, device: torch.device,
                sample: bool = False, bs: int = 1000) -> Tensor:
    vae.eval().to(device)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        mu, logvar = vae.encode(xb)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        out.append(torch.sigmoid(vae.decode_logits(z)).cpu())
    return torch.cat(out)


@torch.no_grad()
def accuracy(clf: Classifier, x: Tensor, y: Tensor, device: torch.device,
             bs: int = 1000) -> float:
    clf.eval().to(device)
    correct = 0
    for i in range(0, len(x), bs):
        pred = clf(x[i:i + bs].to(device)).argmax(1).cpu()
        correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(x)


def save_grid(rows: List[Tuple[str, Tensor]], path: Path) -> None:
    n = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.1, len(rows) * 1.25))
    for r, (title, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def run_variant(args, train_noise, data, clf_base, device, out: Path) -> Dict:
    xtr, ytr, xte, yte = data
    out.mkdir(parents=True, exist_ok=True)

    def loader(x, y, shuffle):
        return DataLoader(TensorDataset(x, y), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=args.num_workers)

    # ---------- 1. VAE ----------
    print(f"\n=== 1. Training VAE (train_noise={train_noise}) ===")
    vae = VAE(latent_dim=args.latent_dim, beta=args.beta, train_noise=train_noise)
    make_trainer(args.epochs_vae).fit(vae, loader(xtr, ytr, True), loader(xte, yte, False))

    x_show = xte[:10]
    x_show_rec = reconstruct(vae, x_show, device)
    save_grid([("original", x_show),
               ("reconstruction", x_show_rec),
               ("|diff|", (x_show - x_show_rec).abs())], out / "reconstruction.png")

    # ---------- 2. Reconstructed classifier ----------
    print("\n-- reconstructed classifier (Decoder(Encoder(I)) train)")
    xtr_rec = reconstruct(vae, xtr, device, sample=args.sample_z)
    clf_rec = Classifier()
    make_trainer(args.epochs_clf).fit(clf_rec, loader(xtr_rec, ytr, True), loader(xte, yte, False))

    # ---------- 3. Test conditions ----------
    print(f"\n=== 3. Evaluation (train_noise={train_noise}) ===")
    gen = torch.Generator().manual_seed(args.seed)
    conditions: Dict[str, Tensor] = {
        "clean (original test)": xte,
        "vae(clean)": reconstruct(vae, xte, device),
    }
    for s in args.sigmas:
        noisy = add_noise(xte, s, gen)
        conditions[f"noisy s={s}"] = noisy
        conditions[f"vae(noisy s={s})"] = reconstruct(vae, noisy, device)

    results = {}
    header = f"{'condition':<26}{'acc baseline':>14}{'acc recon-clf':>15}{'MSE vs clean':>14}"
    print(header)
    print("-" * len(header))
    for name, x in conditions.items():
        mse = F.mse_loss(x, xte).item()
        a_b = accuracy(clf_base, x, yte, device)
        a_r = accuracy(clf_rec, x, yte, device)
        results[name] = {"acc_baseline": a_b, "acc_reconstructed": a_r, "mse_vs_clean": mse}
        print(f"{name:<26}{a_b:>14.4f}{a_r:>15.4f}{mse:>14.5f}")

    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    rows = [("clean", xte[:10])]
    rows += [(f"noisy s={s}", conditions[f"noisy s={s}"][:10]) for s in args.sigmas]
    rows += [(f"vae(noisy s={s})", conditions[f"vae(noisy s={s})"][:10]) for s in args.sigmas]
    save_grid(rows, out / "denoising.png")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs_vae", type=int, default=20)
    ap.add_argument("--epochs_clf", type=int, default=5)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--train_noise", type=float, nargs="+", default=[0.0, 0.3],
                    help="макс. sigma шуму на вході VAE під час навчання; "
                         "0.0 = класичний VAE на чистих даних. Можна кілька значень.")
    ap.add_argument("--sample_z", action="store_true",
                    help="семплювати z при побудові reconstructed train (інакше z = mu)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pl.seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_mnist(args.data_dir)
    xtr, ytr, xte, yte = data

    print("\n=== Baseline classifier (original train) ===")
    clf_base = Classifier()
    make_trainer(args.epochs_clf).fit(
        clf_base,
        DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers),
        DataLoader(TensorDataset(xte, yte), batch_size=args.batch_size,
                   num_workers=args.num_workers))

    summary = {}
    for tn in args.train_noise:
        pl.seed_everything(args.seed)
        summary[tn] = run_variant(args, tn, data, clf_base, device, out / f"vae_train_noise_{tn}")

    print("\n=== SUMMARY: MSE vs clean та accuracy (baseline / recon-clf) ===")
    for s in args.sigmas:
        print(f"\nsigma = {s}")
        r0 = next(iter(summary.values()))[f"noisy s={s}"]
        print(f"  {'noisy (без VAE)':<28} MSE {r0['mse_vs_clean']:.5f} | "
              f"acc {r0['acc_baseline']:.4f} / {r0['acc_reconstructed']:.4f}")
        for tn, res in summary.items():
            r = res[f"vae(noisy s={s})"]
            print(f"  {'VAE train_noise=' + str(tn):<28} MSE {r['mse_vs_clean']:.5f} | "
                  f"acc {r['acc_baseline']:.4f} / {r['acc_reconstructed']:.4f}")

    (out / "summary.json").write_text(
        json.dumps({str(k): v for k, v in summary.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nГотово. Результати у {out}")


if __name__ == "__main__":
    main()
