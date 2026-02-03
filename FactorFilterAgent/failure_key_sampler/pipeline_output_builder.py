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

# 错误原因优先级（从低到高，索引越大优先级越高）
FAILURE_PRIORITY_ORDER = [
    "FR_SCALE_PROPORTION_ERROR",
    "FR_OBJECT_ORIENTATION_ERROR",
    "FR_OBJECT_COUNTING_CONFUSION",
    "FR_ABSOLUTE_POSITION_ERROR",
    "FR_RELATIVE_POSITION_ERROR",
    "FR_ABSENCE_REASONING_FAILURE",
    "FR_INTRA_CLASS_INSTANCE_POSITIONING_FAILURE",
    "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE",
]


def _get_failure_priority(failure_key: str) -> int:
    """
    获取错误原因的优先级（索引），不在列表中的返回 -1（最低优先级）。
    
    Returns:
        优先级索引（0=最低，越大越高），不在列表中的返回 -1
    """
    try:
        return FAILURE_PRIORITY_ORDER.index(failure_key)
    except ValueError:
        return -1


def _weighted_choice_by_priority(keys: List[str], rng: random.Random) -> str:
    """
    根据优先级进行加权随机选择。
    
    优先级越高（在 FAILURE_PRIORITY_ORDER 中索引越大），被选中的概率越大。
    使用平方权重放大优先级差距：权重 = (优先级索引 + 2)^2
    不在优先级列表中的错误原因权重为 1（最低）。
    
    Args:
        keys: 候选错误原因列表
        rng: 随机数生成器
    
    Returns:
        选中的错误原因
    """
    # 计算每个 key 的权重：使用平方权重放大优先级差距
    weights = []
    for key in keys:
        priority = _get_failure_priority(key)
        if priority >= 0:
            weight = (priority + 2) ** 2  # 平方权重：优先级0→4, 1→9, 2→16, ..., 7→81
        else:
            weight = 1  # 不在列表中的错误原因权重最低
        weights.append(weight)
    
    # 加权随机选择
    return rng.choices(keys, weights=weights, k=1)[0]


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
        prefill_prefilled_values = getattr(sample, "prefill_prefilled_values", None)
        prefill_payload: Dict[str, object] = {}
        if isinstance(prefill_claim, str) and prefill_claim.strip():
            prefill_payload["claim"] = prefill_claim
        if isinstance(prefill_prefilled_values, dict) and len(prefill_prefilled_values) > 0:
            prefill_payload["prefilled_values"] = prefill_prefilled_values
        error_records.append(
            {
                "sample_index": sample_index,
                "id": record_id,
                "source_a": {
                    "original_id": image_id,
                    "jpg": encode_image_base64(resolved_path),
                },
                "prefill": prefill_payload,
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
    return _weighted_choice_by_priority(keys, rng)


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


def _extract_prefilled_values_from_data(
    data: Dict[str, object],
    failure_id: str,
) -> Dict[str, str]:
    """
    Extract prefilled_values from claim template data.
    
    Returns:
        Dictionary mapping placeholder names to their values (e.g., {"OBJECT_A": "dog", "OBJECT_B": "hurdle"})
    """
    claim_templates = data.get("claim_templates", [])
    if not isinstance(claim_templates, list):
        return {}

    verif = None
    verifications = data.get("verifications", [])
    if isinstance(verifications, list):
        for item in verifications:
            if not isinstance(item, dict):
                continue
            candidate = item.get("failure_id") or item.get("failure_reason")
            if candidate is None:
                continue
            if str(candidate) == str(failure_id):
                verif = item
                break

    claim_id = None
    original_template = None
    if verif:
        claim_id = verif.get("claim_id") or verif.get("metadata", {}).get("claim_id")
        metadata = verif.get("metadata", {})
        if isinstance(metadata, dict):
            original_template = metadata.get("original_template")

    matched_template = None
    if claim_id:
        for template in claim_templates:
            if not isinstance(template, dict):
                continue
            if template.get("claim_id") == claim_id:
                matched_template = template
                break

    if matched_template is None and isinstance(original_template, str):
        for template in claim_templates:
            if not isinstance(template, dict):
                continue
            tmpl_meta = template.get("metadata", {})
            candidate = (
                (tmpl_meta.get("original_template_unfilled") if isinstance(tmpl_meta, dict) else None)
                or template.get("claim_template")
                or template.get("claim_text")
            )
            if isinstance(candidate, str) and candidate == original_template:
                matched_template = template
                break

    if not matched_template:
        return {}

    # Extract prefilled_values from metadata
    metadata = matched_template.get("metadata", {})
    if isinstance(metadata, dict):
        prefilled_values = metadata.get("prefilled_values", {})
        if isinstance(prefilled_values, dict) and len(prefilled_values) > 0:
            # Filter to only string values and normalize keys to uppercase
            result: Dict[str, str] = {}
            for k, v in prefilled_values.items():
                if isinstance(v, str) and v.strip():
                    result[str(k).upper()] = v.strip()
            return result

    return {}


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
        prefilled_values = _extract_prefilled_values_from_data(
            data,
            str(sampled_failure),
        )

        record_id = data.get("id", None)
        if record_id is None:
            try:
                record_id = int(str(image_id))
            except (TypeError, ValueError):
                record_id = len(error_records)

        sample_index = start_index + len(error_records)
        prefill_payload: Dict[str, object] = {}
        if isinstance(prefill_claim, str) and prefill_claim.strip():
            prefill_payload["claim"] = prefill_claim
        if isinstance(prefilled_values, dict) and len(prefilled_values) > 0:
            prefill_payload["prefilled_values"] = prefilled_values

        error_records.append(
            {
                "sample_index": sample_index,
                "id": record_id,
                "source_a": {
                    "original_id": str(image_id),
                    "jpg": encode_image_base64(image_path),
                },
                "prefill": prefill_payload,
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
