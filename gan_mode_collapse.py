import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--latent_dim", type=int, default=100)
p.add_argument("--major_class", type=int, default=1)
p.add_argument("--minor_frac", type=float, default=0.01,
               help="частка зразків інших класів, що залишається (1.0 = збалансовано)")
p.add_argument("--lr_g", type=float, default=2e-3)
p.add_argument("--lr_d", type=float, default=2e-5)
p.add_argument("--g_steps", type=int, default=2, help="кроків генератора на 1 крок дискримінатора")
p.add_argument("--baseline", action="store_true",
               help="збалансований датасет + однакові lr = 2e-4")
p.add_argument("--out", type=str, default="samples")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

if args.baseline:
    args.minor_frac, args.lr_g, args.lr_d, args.g_steps = 1.0, 2e-4, 2e-4, 1

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)


def make_imbalanced_mnist(major_class, minor_frac):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    g = torch.Generator().manual_seed(args.seed)
    targets = ds.targets
    keep = []
    for c in range(10):
        idx = (targets == c).nonzero(as_tuple=True)[0]
        if c != major_class:
            n = max(1, int(len(idx) * minor_frac))
            idx = idx[torch.randperm(len(idx), generator=g)[:n]]
        keep.append(idx)
    keep = torch.cat(keep)
    counts = torch.bincount(targets[keep], minlength=10).tolist()
    print(f"Розподіл класів у датасеті: {counts}")
    return Subset(ds, keep.tolist())


loader = DataLoader(make_imbalanced_mnist(args.major_class, args.minor_frac),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)


class Generator(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


G, D = Generator(args.latent_dim).to(device), Discriminator().to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
bce = nn.BCELoss()
fixed_z = torch.randn(16, args.latent_dim, device=device)


@torch.no_grad()
def diversity(n=256):
    x = G(torch.randn(n, args.latent_dim, device=device)).flatten(1)
    return torch.cdist(x, x).mean().item()


for epoch in range(1, args.epochs + 1):
    for real, _ in loader:
        real = real.to(device)
        bs = real.size(0)
        ones, zeros = torch.ones(bs, 1, device=device), torch.zeros(bs, 1, device=device)

        z = torch.randn(bs, args.latent_dim, device=device)
        fake = G(z).detach()
        loss_d = bce(D(real), ones) + bce(D(fake), zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        for _ in range(args.g_steps):
            z = torch.randn(bs, args.latent_dim, device=device)
            loss_g = bce(D(G(z)), ones)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    print(f"epoch {epoch:3d} | loss_d {loss_d.item():.3f} | loss_g {loss_g.item():.3f} "
          f"| diversity {diversity():.2f}")
    with torch.no_grad():
        save_image(G(fixed_z) * 0.5 + 0.5, f"{args.out}/epoch_{epoch:03d}.png", nrow=4)
