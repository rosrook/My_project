"""
Training utilities for the decision model.

This module provides:
- training loop
- evaluation loop
- checkpoint saving
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = None

from .decision_model import DecisionModel


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 5
    device: str = "cuda"
    save_dir: str = "./checkpoints"


class ImageEmbedder(Protocol):
    """Callable that converts a batch of images into embeddings."""

    def __call__(self, images: "torch.Tensor") -> "torch.Tensor":
        ...


def train_one_epoch(
    model: "DecisionModel",
    dataloader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    device: str,
    embedder: ImageEmbedder,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        images = batch["image"]
        image_embeddings = embedder(images).to(device)
        failure_labels = batch["failure_labels"].to(device)
        factor_labels = batch["factor_labels"].to(device)

        outputs = model(image_embeddings)
        loss_fr = criterion(outputs["failure_logits"], failure_labels)
        loss_ff = criterion(outputs["factor_logits"], factor_labels)
        loss = loss_fr + loss_ff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {"train_loss": total_loss / max(len(dataloader), 1)}


def evaluate(
    model: "DecisionModel",
    dataloader: "DataLoader",
    device: str,
    embedder: ImageEmbedder,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            image_embeddings = embedder(images).to(device)
            failure_labels = batch["failure_labels"].to(device)
            factor_labels = batch["factor_labels"].to(device)

            outputs = model(image_embeddings)
            loss_fr = criterion(outputs["failure_logits"], failure_labels)
            loss_ff = criterion(outputs["factor_logits"], factor_labels)
            total_loss += (loss_fr + loss_ff).item()

    return {"eval_loss": total_loss / max(len(dataloader), 1)}


def save_checkpoint(model: "DecisionModel", save_dir: str, tag: str = "latest") -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / f"decision_model_{tag}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train(
    model: "DecisionModel",
    train_loader: "DataLoader",
    eval_loader: Optional["DataLoader"],
    config: TrainConfig,
    embedder: ImageEmbedder,
) -> None:
    if not HAS_TORCH:
        raise ImportError("torch is required for training")

    device = config.device if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, embedder)
        if eval_loader:
            eval_metrics = evaluate(model, eval_loader, device, embedder)
        else:
            eval_metrics = {}

        save_checkpoint(model, config.save_dir, tag=f"epoch_{epoch}")
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"eval_loss={eval_metrics.get('eval_loss', 0.0):.4f}"
        )
