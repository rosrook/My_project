"""
Rule-based filtering module for factor-based image screening.

Design goals:
1) No baseline model usage (only cheap heuristics + VLM placeholder).
2) Two-stage coarse-to-fine filtering (cheap prefilter -> VLM verification).
3) Failure reasons act as routing nodes, not error judges.
4) Output structured results for downstream question/claim generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PARQUET = False
    pq = None


# -----------------------------
# Core data structures
# -----------------------------

@dataclass(frozen=True)
class FilteringFactor:
    """A concrete requirement about the image, not a model error description."""

    factor_id: str
    description: str
    # Question/claim types that can be triggered by this factor
    trigger_types: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FailureReason:
    """
    A routing node representing a specific failure reason.

    core_filtering_factors: the set of factors to verify in stage 2.
    """

    failure_reason_id: str
    description: str
    core_filtering_factors: List[FilteringFactor]
    trigger_types: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageRecord:
    """Lightweight image record with metadata."""

    image_path: str
    image_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FailureReasonMatch:
    """Matched failure reason with passed filtering factors."""

    failure_reason_id: str
    passed_factors: List[FilteringFactor] = field(default_factory=list)


@dataclass
class ImageFilterResult:
    """Structured output for a single image."""

    image_id: str
    image_path: str
    matched_failure_reasons: List[FailureReasonMatch] = field(default_factory=list)
    # Optional structured payload for downstream use
    structured_payload: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Dataset loader (standalone)
# -----------------------------

class LocalImageLoader:
    """
    Standalone loader with the same parquet semantics as ImageLoader.

    - Supports parquet with columns: jpg (binary) and conversations (struct list).
    - Generates <bytes:image_id> paths when image bytes exist.
    - Optional sampling by image count or parquet file count.
    """

    def __init__(
        self,
        parquet_dir: str,
        sample_size: Optional[int] = None,
        parquet_sample_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.sample_size = sample_size
        self.parquet_sample_size = parquet_sample_size
        self.random_seed = random_seed

        self._parquet_files: Optional[List[Path]] = None
        self._image_paths: Optional[List[str]] = None
        self._image_metadata: Optional[List[Dict[str, Any]]] = None

        if random_seed is not None:
            import random
            random.seed(random_seed)

    def _discover_parquet_files(self) -> List[Path]:
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")
        files = list(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")
        return files

    def _get_parquet_files(self) -> List[Path]:
        if self._parquet_files is None:
            all_files = self._discover_parquet_files()
            if self.parquet_sample_size is not None and self.parquet_sample_size < len(all_files):
                import random
                self._parquet_files = random.sample(all_files, self.parquet_sample_size)
            else:
                self._parquet_files = all_files
        return self._parquet_files

    def _load_single_parquet_file(self, parquet_file: Path) -> List[Dict[str, Any]]:
        if not HAS_PARQUET:
            raise ImportError("pyarrow is required for parquet loading")

        table = pq.read_table(parquet_file, use_threads=False)
        if table.num_rows == 0:
            return []

        records: List[Dict[str, Any]] = []
        file_stem = parquet_file.stem

        for row_idx in range(table.num_rows):
            record: Dict[str, Any] = {}
            for col_idx, col_name in enumerate(table.column_names):
                col = table.column(col_idx)
                value = col[row_idx]
                # jpg binary -> image_bytes
                if col_name == "jpg":
                    if hasattr(value, "as_py"):
                        record["image_bytes"] = value.as_py()
                    else:
                        record["image_bytes"] = value
                    continue
                # conversations list/struct -> python native
                col_type = col.type
                is_list_type = str(col_type).startswith("list")
                is_struct_type = str(col_type).startswith("struct")
                if is_list_type or is_struct_type:
                    record[col_name] = value.as_py() if hasattr(value, "as_py") else value
                else:
                    record[col_name] = value.as_py() if hasattr(value, "as_py") else value

            if "image_bytes" not in record:
                continue

            if "image_id" not in record or not record["image_id"]:
                record["image_id"] = f"{file_stem}_{row_idx}"

            records.append(record)

        return records

    def get_all_image_paths(self) -> List[str]:
        if self._image_paths is not None:
            return self._image_paths

        all_records: List[Dict[str, Any]] = []
        for parquet_file in self._get_parquet_files():
            all_records.extend(self._load_single_parquet_file(parquet_file))

        if not all_records:
            self._image_paths = []
            self._image_metadata = []
            return []

        image_paths = []
        for record in all_records:
            image_id = record.get("image_id", f"image_{len(image_paths)}")
            image_paths.append(f"<bytes:{image_id}>")

        if self.sample_size is not None and self.sample_size < len(image_paths):
            import random
            sampled_idx = random.sample(range(len(image_paths)), self.sample_size)
            image_paths = [image_paths[i] for i in sampled_idx]
            all_records = [all_records[i] for i in sampled_idx]

        self._image_paths = image_paths
        self._image_metadata = all_records
        return image_paths

    def get_image_id(self, image_path: str) -> str:
        if image_path.startswith("<bytes:"):
            return image_path[7:-1]
        return Path(image_path).stem

    def get_image_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        if not self._image_metadata:
            return None
        image_id = self.get_image_id(image_path)
        for record in self._image_metadata:
            if record.get("image_id") == image_id:
                return {k: v for k, v in record.items() if k not in {"image_bytes"}}
        return None


# -----------------------------
# Stage 1: Cheap prefilter
# -----------------------------

class CheapPrefilter(Protocol):
    """
    Cheap heuristic prefilter for filtering factors.

    - Should be fast and low-cost (no heavy VLM).
    - Allow false positives, avoid false negatives when possible.
    """

    def matches(self, image: ImageRecord, factor: FilteringFactor) -> bool:
        """Return True if the image possibly satisfies the factor."""


class DefaultCheapPrefilter:
    """Placeholder cheap prefilter (always True)."""

    def matches(self, image: ImageRecord, factor: FilteringFactor) -> bool:
        # Replace with rule-based heuristics as needed.
        return True


# -----------------------------
# Stage 2: VLM verification
# -----------------------------

class VLMVerifier(Protocol):
    """
    VLM verification interface (placeholder).

    Implementations should verify whether the image truly satisfies a factor.
    """

    def verify(self, image: ImageRecord, factor: FilteringFactor) -> bool:
        """Return True if the factor is satisfied."""


class DefaultVLMVerifier:
    """Placeholder verifier (always True)."""

    def verify(self, image: ImageRecord, factor: FilteringFactor) -> bool:
        # Replace with actual VLM inference.
        return True


# -----------------------------
# Routing + main pipeline
# -----------------------------

def _prefilter_image(
    image: ImageRecord,
    failure_reasons: Sequence[FailureReason],
    prefilter: CheapPrefilter,
) -> Dict[str, List[FilteringFactor]]:
    """
    Run cheap prefilter on all factors, grouped by failure reason.
    Returns a mapping: failure_reason_id -> matched factors (coarse).
    """

    matches: Dict[str, List[FilteringFactor]] = {}
    for reason in failure_reasons:
        matched = []
        for factor in reason.core_filtering_factors:
            if prefilter.matches(image, factor):
                matched.append(factor)
        if matched:
            matches[reason.failure_reason_id] = matched
    return matches


def _route_failure_reasons(
    prefilter_matches: Dict[str, List[FilteringFactor]],
    failure_reasons: Sequence[FailureReason],
    top_k: int = 1,
) -> List[FailureReason]:
    """
    Route an image to the most likely failure reasons.

    Strategy (simple default):
    - Score each failure reason by number of matched factors.
    - Return top_k reasons (ties preserved by score order).
    """

    if not prefilter_matches:
        return []

    reason_by_id = {r.failure_reason_id: r for r in failure_reasons}
    scored: List[Tuple[int, FailureReason]] = []
    for reason_id, factors in prefilter_matches.items():
        reason = reason_by_id.get(reason_id)
        if reason is None:
            continue
        scored.append((len(factors), reason))

    scored.sort(key=lambda x: x[0], reverse=True)
    routed = [r for _, r in scored[: max(top_k, 1)]]
    return routed


def _verify_core_factors(
    image: ImageRecord,
    reason: FailureReason,
    verifier: VLMVerifier,
) -> List[FilteringFactor]:
    """
    Verify core filtering factors using VLM (stage 2).
    """

    passed: List[FilteringFactor] = []
    for factor in reason.core_filtering_factors:
        if verifier.verify(image, factor):
            passed.append(factor)
    return passed


def run_rule_based_filtering(
    parquet_dir: str,
    failure_reasons: Sequence[FailureReason],
    prefilter: Optional[CheapPrefilter] = None,
    verifier: Optional[VLMVerifier] = None,
    sample_size: Optional[int] = None,
    parquet_sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    include_metadata: bool = False,
    top_k: int = 1,
) -> List[ImageFilterResult]:
    """
    Main pipeline: dataset -> structured filtering results.

    Steps:
    1) Load image paths using ImageLoader (consistent with existing loader).
    2) Stage 1: cheap prefilter to find candidate failure reasons.
    3) Routing: choose most likely failure reasons (top_k).
    4) Stage 2: verify core factors using VLM (placeholder).
    5) Return structured results for downstream use.
    """

    prefilter = prefilter or DefaultCheapPrefilter()
    verifier = verifier or DefaultVLMVerifier()

    image_loader = LocalImageLoader(
        parquet_dir=parquet_dir,
        sample_size=sample_size,
        parquet_sample_size=parquet_sample_size,
        random_seed=random_seed,
    )

    image_paths = image_loader.get_all_image_paths()
    results: List[ImageFilterResult] = []

    for image_path in image_paths:
        image_id = image_loader.get_image_id(image_path)
        metadata = image_loader.get_image_metadata(image_path) if include_metadata else None
        image = ImageRecord(image_path=image_path, image_id=image_id, metadata=metadata)

        prefilter_matches = _prefilter_image(image, failure_reasons, prefilter)
        routed_reasons = _route_failure_reasons(prefilter_matches, failure_reasons, top_k=top_k)

        matched_reasons: List[FailureReasonMatch] = []
        for reason in routed_reasons:
            passed_factors = _verify_core_factors(image, reason, verifier)
            if passed_factors:
                matched_reasons.append(
                    FailureReasonMatch(
                        failure_reason_id=reason.failure_reason_id,
                        passed_factors=passed_factors,
                    )
                )

        results.append(
            ImageFilterResult(
                image_id=image_id,
                image_path=image_path,
                matched_failure_reasons=matched_reasons,
                structured_payload={
                    "trigger_types": list(
                        {t for r in matched_reasons for t in _safe_reason_triggers(failure_reasons, r.failure_reason_id)}
                    ),
                },
            )
        )

    return results


def _safe_reason_triggers(
    failure_reasons: Sequence[FailureReason],
    reason_id: str,
) -> List[str]:
    for reason in failure_reasons:
        if reason.failure_reason_id == reason_id:
            return reason.trigger_types
    return []
