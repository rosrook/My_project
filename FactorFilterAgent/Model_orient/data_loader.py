"""
Data loading utilities for decision-model training/inference.

This module provides:
- Config loading (failure reasons and factors metadata)
- Dataset object for training (image + multi-label targets)
- Batch collation helpers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class FailureReasonMeta:
    failure_reason_id: str
    description: str


@dataclass(frozen=True)
class FilteringFactorMeta:
    factor_id: str
    description: str


@dataclass(frozen=True)
class TrainingSample:
    image_path: str
    failure_reason_ids: List[str]
    factor_ids: List[str]
    image_id: Optional[str] = None


def load_config(config_path: str) -> Tuple[List[FailureReasonMeta], List[FilteringFactorMeta], Dict[str, List[str]]]:
    """Load failure reasons, filtering factors, and trigger map from JSON."""
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    failure_reasons = [
        FailureReasonMeta(**item) for item in data.get("failure_reasons", [])
    ]
    factors = [
        FilteringFactorMeta(**item) for item in data.get("filtering_factors", [])
    ]
    trigger_map = data.get("trigger_map", {})
    return failure_reasons, factors, trigger_map


class DecisionDataset:
    """
    Simple dataset for decision-model training.

    Each sample contains:
    - image_path
    - failure_reason_ids (multi-label)
    - factor_ids (multi-label)
    """

    def __init__(
        self,
        samples: Sequence[TrainingSample],
        failure_reason_ids: Sequence[str],
        factor_ids: Sequence[str],
        image_size: Optional[Tuple[int, int]] = (224, 224),
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        self.samples = list(samples)
        self.failure_reason_to_idx = {fid: i for i, fid in enumerate(failure_reason_ids)}
        self.factor_to_idx = {fid: i for i, fid in enumerate(factor_ids)}
        self.image_size = image_size
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size)
        if self.image_transform:
            image = self.image_transform(image)

        failure_labels = [0] * len(self.failure_reason_to_idx)
        for fr in sample.failure_reason_ids:
            if fr in self.failure_reason_to_idx:
                failure_labels[self.failure_reason_to_idx[fr]] = 1

        factor_labels = [0] * len(self.factor_to_idx)
        for fac in sample.factor_ids:
            if fac in self.factor_to_idx:
                factor_labels[self.factor_to_idx[fac]] = 1

        return {
            "image": image,
            "image_id": sample.image_id,
            "failure_labels": failure_labels,
            "factor_labels": factor_labels,
        }


def load_training_samples(jsonl_path: str) -> List[TrainingSample]:
    """
    Load training samples from a JSONL file.

    Expected JSONL schema per line:
    {
      "image_path": "...",
      "image_id": "...",
      "failure_reason_ids": ["FR_..."],
      "factor_ids": ["factor_..."]
    }
    """
    samples: List[TrainingSample] = []
    path = Path(jsonl_path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append(
                TrainingSample(
                    image_path=item["image_path"],
                    image_id=item.get("image_id"),
                    failure_reason_ids=item.get("failure_reason_ids", []),
                    factor_ids=item.get("factor_ids", []),
                )
            )
    return samples
