import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128
MODEL_NAMES = ["MLP", "CNN", "Attention"]

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders():
    """Fashion-MNIST: 55k train / 5k validation (from official train), official 10k test."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    full_train = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    split_gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train, [55000, 5000], generator=split_gen)
    shuffle_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_gen
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


class PatchAttention(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> linear to 64 -> 2 encoder layers -> mean pool -> linear."""

    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, layers=2, classes=10):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * patch, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, classes)

    def forward(self, x):
        b = x.shape[0]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, 1, 4, 4, 7, 7)
        x = x.reshape(b, -1, p * p)  # (B, 16, 49)
        x = self.proj(x)  # (B, 16, 64)
        x = self.encoder(x)  # (B, 16, 64)
        x = x.mean(dim=1)  # (B, 64)
        return self.head(x)


def build_model(name):
    set_seeds()
    if name == "MLP":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    if name == "CNN":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )
    if name == "Attention":
        return PatchAttention()
    raise ValueError(f"Unknown model name: {name}")
