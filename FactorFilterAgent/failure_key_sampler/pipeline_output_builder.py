"""
Build error output JSON with pipeline mapping and base64 images.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ErrorOutputItem:
    sample_index: int
    id: int
    original_id: str
    pipeline_type: str
    pipeline_name: str
    jpg: str


def load_pipeline_config(
    config_path: str,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Load failure_id -> pipeline mapping from config.

    Expected JSON structure (example):
    {
      "pipelines": [
        {
          "failure_id": "FR_XXX",
          "pipeline_type": "type_a",
          "pipeline_name": "name_a"
        }
      ],
      "default_pipeline_type": "unknown",
      "default_pipeline_name": "unknown"
    }
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, Dict[str, str]] = {}
    for item in data.get("pipelines", []):
        failure_id = item.get("failure_id") or item.get("failure_key")
        if not failure_id:
            continue
        mapping[str(failure_id)] = {
            "pipeline_type": str(item.get("pipeline_type", "")),
            "pipeline_name": str(item.get("pipeline_name", "")),
        }

    defaults = {
        "pipeline_type": str(data.get("default_pipeline_type", "")),
        "pipeline_name": str(data.get("default_pipeline_name", "")),
    }
    return mapping, defaults


def resolve_image_path(
    image_id: str,
    image_path: Optional[str],
    image_dir: Optional[str],
) -> Optional[str]:
    if image_path:
        return image_path
    if not image_dir:
        return None

    base_dir = Path(image_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_id_path = Path(image_id)
    if image_id_path.suffix:
        candidate = base_dir / image_id
        if candidate.exists():
            return str(candidate)
    else:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = base_dir / f"{image_id}{ext}"
            if candidate.exists():
                return str(candidate)

    return None


def encode_image_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_error_output(
    samples: List[object],
    pipeline_config_path: str,
    image_dir: Optional[str],
) -> List[Dict[str, object]]:
    pipeline_map, defaults = load_pipeline_config(pipeline_config_path)
    error_records: List[Dict[str, object]] = []

    for sample_index, sample in enumerate(samples):
        sampled_failure = getattr(sample, "sampled_failure", None)
        if not sampled_failure:
            continue
        image_id = getattr(sample, "image_id", "")
        image_path = getattr(sample, "image_path", None)
        pipeline_info = pipeline_map.get(sampled_failure, defaults)

        resolved_path = resolve_image_path(image_id, image_path, image_dir)
        if not resolved_path:
            raise ValueError(
                f"Image path not found for image_id={image_id}. "
                "Provide --image_dir or include image_path in probing_results."
            )

        record_id = getattr(sample, "id", None)
        if record_id is None:
            try:
                record_id = int(str(image_id))
            except (TypeError, ValueError):
                record_id = sample_index

        error_records.append(
            {
                "sample_index": sample_index,
                "id": record_id,
                "source_a": {
                    "original_id": image_id,
                    "jpg": encode_image_base64(resolved_path),
                },
                "pipeline_type": pipeline_info.get("pipeline_type", ""),
                "pipeline_name": pipeline_info.get("pipeline_name", ""),
            }
        )

    return error_records


def write_error_output(
    samples: List[object],
    pipeline_config_path: str,
    error_output_path: str,
    image_dir: Optional[str],
) -> None:
    error_records = build_error_output(samples, pipeline_config_path, image_dir)
    error_path = Path(error_output_path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_records, f, ensure_ascii=False, indent=2)
