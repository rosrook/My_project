"""
Build error output JSON with pipeline mapping and base64 images.
"""

from __future__ import annotations

import base64
import json
import random
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
    start_index: int = 0,
) -> List[Dict[str, object]]:
    pipeline_map, defaults = load_pipeline_config(pipeline_config_path)
    error_records: List[Dict[str, object]] = []

    for offset, sample in enumerate(samples):
        sample_index = start_index + offset
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

        prefill_claim = getattr(sample, "prefill_claim", None)
        error_records.append(
            {
                "sample_index": sample_index,
                "id": record_id,
                "source_a": {
                    "original_id": image_id,
                    "jpg": encode_image_base64(resolved_path),
                },
                "prefill": {
                    "claim": prefill_claim,
                },
                "pipeline_type": pipeline_info.get("pipeline_type", ""),
                "pipeline_name": pipeline_info.get("pipeline_name", ""),
            }
        )

    return error_records


def _sample_failure_key(
    data: Dict[str, object],
    rng: random.Random,
) -> Optional[str]:
    breakdown = data.get("aggregated_failures", {}).get("failure_breakdown", {})
    keys: List[str] = []
    if isinstance(breakdown, dict):
        keys = [
            str(k)
            for k in breakdown.keys()
            if k not in {"null", "model_limitation"}
        ]

    if not keys:
        verifications = data.get("verifications", [])
        if isinstance(verifications, list):
            for verif in verifications:
                if not isinstance(verif, dict):
                    continue
                if verif.get("is_correct", True):
                    continue
                failure_id = verif.get("failure_id") or verif.get("failure_reason")
                if failure_id and failure_id not in {"null", "model_limitation"}:
                    keys.append(str(failure_id))

    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    return rng.choice(keys)


def _extract_prefill_claim(
    data: Dict[str, object],
    failure_id: str,
) -> Optional[str]:
    verifications = data.get("verifications", [])
    if not isinstance(verifications, list):
        return None
    for verif in verifications:
        if not isinstance(verif, dict):
            continue
        if verif.get("failure_id") != failure_id:
            continue
        metadata = verif.get("metadata", {})
        if isinstance(metadata, dict):
            original_template = metadata.get("original_template")
            if isinstance(original_template, str) and original_template.strip():
                return original_template
    return None


def _resolve_failure_image_path(json_path: Path) -> Optional[str]:
    stem = json_path.stem
    base_stems = [stem]
    if stem.endswith("_result"):
        base_stems.append(stem[: -len("_result")])

    for base_stem in base_stems:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = json_path.with_name(f"{base_stem}{ext}")
            if candidate.exists():
                return str(candidate)
    return None


def build_error_output_from_failure_root(
    failure_root: str,
    pipeline_config_path: str,
    random_seed: Optional[int] = None,
    start_index: int = 0,
) -> List[Dict[str, object]]:
    base_dir = Path(failure_root)
    if not base_dir.exists():
        raise FileNotFoundError(f"Failure root not found: {failure_root}")

    rank_dirs = sorted(
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("rank")
    )
    candidate_dirs = rank_dirs if rank_dirs else [base_dir]
    failure_dirs = [d / "failures" for d in candidate_dirs if (d / "failures").exists()]
    if not failure_dirs and (base_dir / "failures").exists():
        failure_dirs = [base_dir / "failures"]

    json_paths: List[Path] = []
    for failure_dir in failure_dirs:
        json_paths.extend(sorted(failure_dir.glob("*.json")))

    pipeline_map, defaults = load_pipeline_config(pipeline_config_path)
    rng = random.Random(random_seed)
    error_records: List[Dict[str, object]] = []

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue

        sampled_failure = _sample_failure_key(data, rng)
        if not sampled_failure:
            continue

        image_id = (
            data.get("image_id")
            or data.get("aggregated_failures", {}).get("image_id")
            or json_path.stem
        )
        image_path = _resolve_failure_image_path(json_path)
        if not image_path:
            raise FileNotFoundError(
                f"Image not found for failure json: {json_path}"
            )

        pipeline_info = pipeline_map.get(str(sampled_failure), defaults)
        prefill_claim = _extract_prefill_claim(data, str(sampled_failure))

        record_id = data.get("id", None)
        if record_id is None:
            try:
                record_id = int(str(image_id))
            except (TypeError, ValueError):
                record_id = len(error_records)

        sample_index = start_index + len(error_records)
        error_records.append(
            {
                "sample_index": sample_index,
                "id": record_id,
                "source_a": {
                    "original_id": str(image_id),
                    "jpg": encode_image_base64(image_path),
                },
                "prefill": {
                    "claim": prefill_claim,
                },
                "pipeline_type": pipeline_info.get("pipeline_type", ""),
                "pipeline_name": pipeline_info.get("pipeline_name", ""),
            }
        )

    return error_records


def write_error_output_from_failure_root(
    failure_root: str,
    pipeline_config_path: str,
    error_output_path: str,
    random_seed: Optional[int] = None,
    start_index: int = 0,
) -> None:
    error_records = build_error_output_from_failure_root(
        failure_root=failure_root,
        pipeline_config_path=pipeline_config_path,
        random_seed=random_seed,
        start_index=start_index,
    )
    error_path = Path(error_output_path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_records, f, ensure_ascii=False, indent=2)


def write_error_output(
    samples: List[object],
    pipeline_config_path: str,
    error_output_path: str,
    image_dir: Optional[str],
    start_index: int = 0,
) -> None:
    error_records = build_error_output(
        samples,
        pipeline_config_path,
        image_dir,
        start_index=start_index,
    )
    error_path = Path(error_output_path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_records, f, ensure_ascii=False, indent=2)
