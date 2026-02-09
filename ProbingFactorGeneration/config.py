"""
Configuration and constants for the probing factor generation framework.
"""

from typing import Any, Dict, List, Optional
from enum import Enum


class FailureTaxonomy(Enum):
    """
    Failure reason taxonomy - can be extended with more categories.
    """
    # Vision-related failures
    VISUAL_ERROR = "visual_error"  # e.g., misrecognized object, color, position
    VISUAL_AMBIGUITY = "visual_ambiguity"  # e.g., unclear image, occlusion
    
    # Language-related failures
    LANGUAGE_MISUNDERSTANDING = "language_misunderstanding"  # e.g., ambiguous claim
    LANGUAGE_COMPLEXITY = "language_complexity"  # e.g., complex reasoning required
    
    # Reasoning failures
    REASONING_ERROR = "reasoning_error"  # e.g., logical inconsistency
    COMMONSENSE_ERROR = "commonsense_error"  # e.g., violates common sense
    
    # Model-specific failures
    MODEL_LIMITATION = "model_limitation"  # e.g., out of distribution
    UNCERTAIN = "uncertain"  # e.g., insufficient information


class ContentType(Enum):
    """
    Target content types for claim generation.
    """
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    ACTION = "action"
    SPATIAL = "spatial"
    COUNT = "count"
    TEXT = "text"


# Mapping from failure reasons to filtering factors
FAILURE_TO_FILTERING_FACTOR: Dict[str, str] = {
    FailureTaxonomy.VISUAL_ERROR.value: "visual_accuracy",
    FailureTaxonomy.VISUAL_AMBIGUITY.value: "visual_clarity",
    FailureTaxonomy.LANGUAGE_MISUNDERSTANDING.value: "language_clarity",
    FailureTaxonomy.LANGUAGE_COMPLEXITY.value: "reasoning_complexity",
    FailureTaxonomy.REASONING_ERROR.value: "reasoning_ability",
    FailureTaxonomy.COMMONSENSE_ERROR.value: "commonsense_knowledge",
    FailureTaxonomy.MODEL_LIMITATION.value: "distribution_coverage",
    FailureTaxonomy.UNCERTAIN.value: "information_sufficiency",
}


def get_filtering_factor(failure_reason: str) -> str:
    """
    Map failure reason to filtering factor.
    
    Args:
        failure_reason: Failure reason string
        
    Returns:
        Filtering factor string
    """
    return FAILURE_TO_FILTERING_FACTOR.get(failure_reason, "unknown")


def get_all_failure_reasons() -> List[str]:
    """Get all available failure reasons."""
    return [reason.value for reason in FailureTaxonomy]


def get_all_filtering_factors() -> List[str]:
    """Get all available filtering factors."""
    return list(set(FAILURE_TO_FILTERING_FACTOR.values()))


def calculate_optimal_concurrency(
    data_size: int,
    is_local_model: bool = False,
    avg_claims_per_image: int = 10,
    min_concurrent: int = 1,
    max_concurrent: int = 100
) -> int:
    """
    Automatically calculate optimal concurrency based on data size and model type.
    
    Args:
        data_size: Number of images to process
        is_local_model: Whether using local model (LLaVA) - typically slower and limited concurrency
        avg_claims_per_image: Average number of claims per image (affects total API calls)
        min_concurrent: Minimum allowed concurrency
        max_concurrent: Maximum allowed concurrency
    
    Returns:
        Optimal concurrency value
    """
    if is_local_model:
        # Local models (LLaVA) are GPU-bound, typically limited to 1-2 concurrent requests
        return min(2, max(min_concurrent, 1))
    
    # For API models, calculate based on data size
    total_requests = data_size * avg_claims_per_image * 2  # baseline + judge
    
    if data_size < 10:
        # Very small batches: use all data
        optimal = min(max(data_size, 1), 5)  # Ensure at least 1
    elif data_size < 100:
        # Small batches: moderate concurrency
        optimal = min(20, max(data_size // 5, 5))
    elif data_size < 1000:
        # Medium batches: balanced concurrency
        optimal = min(50, max(data_size // 20, 10))
    else:
        # Large batches (1000+): high concurrency
        # For 10000 images: 10000/100 = 100
        optimal = min(100, max(data_size // 100, 20))
    
    # Ensure within bounds and at least 1
    return max(1, max(min_concurrent, min(max_concurrent, optimal)))


def estimate_processing_time(
    num_images: int,
    avg_claims_per_image: int = 10,
    baseline_max_concurrent: int = 10,
    judge_max_concurrent: int = 10,
    baseline_time_per_request: float = 2.0,  # seconds per API call
    judge_time_per_request: float = 2.0,  # seconds per API call
    is_local_baseline: bool = False,
    is_local_judge: bool = False
) -> Dict[str, Any]:
    """
    Estimate processing time for a batch of images.
    
    Args:
        num_images: Number of images to process
        avg_claims_per_image: Average number of claims per image
        baseline_max_concurrent: Baseline model max concurrency
        judge_max_concurrent: Judge model max concurrency
        baseline_time_per_request: Average time per baseline API call (seconds)
        judge_time_per_request: Average time per judge API call (seconds)
        is_local_baseline: Whether baseline is local model (typically slower)
        is_local_judge: Whether judge is local model (typically slower)
    
    Returns:
        Dictionary with time estimates:
        {
            "total_images": int,
            "total_requests": int,  # baseline + judge requests
            "baseline_requests": int,
            "judge_requests": int,
            "baseline_time_seconds": float,
            "baseline_time_minutes": float,
            "baseline_time_hours": float,
            "judge_time_seconds": float,
            "judge_time_minutes": float,
            "judge_time_hours": float,
            "total_time_seconds": float,
            "total_time_minutes": float,
            "total_time_hours": float,
            "estimated_throughput": float  # images per hour
        }
    """
    # Adjust time per request for local models (typically slower)
    if is_local_baseline:
        baseline_time_per_request = 5.0  # Local models are typically slower
    if is_local_judge:
        judge_time_per_request = 5.0
    
    # Calculate total requests
    baseline_requests = num_images * avg_claims_per_image
    judge_requests = num_images * avg_claims_per_image
    total_requests = baseline_requests + judge_requests
    
    # Calculate time per stage (with concurrency)
    # Time = (total_requests / concurrency) * time_per_request
    # Protect against division by zero
    baseline_max_concurrent = max(1, baseline_max_concurrent)  # Ensure at least 1
    judge_max_concurrent = max(1, judge_max_concurrent)  # Ensure at least 1
    baseline_time = (baseline_requests / baseline_max_concurrent) * baseline_time_per_request
    judge_time = (judge_requests / judge_max_concurrent) * judge_time_per_request
    
    # Total time (stages run sequentially: baseline then judge)
    # Note: For pipeline, baseline and judge can partially overlap, but conservative estimate
    total_time = baseline_time + judge_time
    
    # Calculate throughput
    estimated_throughput = (num_images / total_time) * 3600 if total_time > 0 else 0
    
    return {
        "total_images": num_images,
        "total_requests": total_requests,
        "baseline_requests": baseline_requests,
        "judge_requests": judge_requests,
        "baseline_time_seconds": baseline_time,
        "baseline_time_minutes": baseline_time / 60,
        "baseline_time_hours": baseline_time / 3600,
        "judge_time_seconds": judge_time,
        "judge_time_minutes": judge_time / 60,
        "judge_time_hours": judge_time / 3600,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600,
        "estimated_throughput": estimated_throughput
    }


# Model configuration (can be overridden by environment variables or config files)
# These values can be overridden by environment variables or external config modules
import os

MODEL_CONFIG = {
    "MODEL_NAME": os.getenv("MODEL_NAME", "gemini-pro-vision"),  # Default model name
    "SERVICE_NAME": os.getenv("SERVICE_NAME", None),  # Service name for LBOpenAIAsyncClient (optional)
    "ENV": os.getenv("ENV", "prod"),  # Environment: prod/staging
    "API_KEY": os.getenv("API_KEY", None),  # API key (if needed)
    "BASE_URL": os.getenv("BASE_URL", None),  # API base URL (required if not using LBOpenAIAsyncClient)
    "MAX_CONCURRENT": int(os.getenv("MAX_CONCURRENT", "10")),  # Maximum concurrent requests
    "REQUEST_DELAY": float(os.getenv("REQUEST_DELAY", "0.0")),  # Delay between requests (seconds)
    "MAX_TOKENS": int(os.getenv("MAX_TOKENS", "1000")),  # Maximum tokens in response
    "TEMPERATURE": float(os.getenv("TEMPERATURE", "0.3")),  # Temperature for generation
    "USE_LB_CLIENT": os.getenv("USE_LB_CLIENT", "true").lower() == "true",  # Whether to use LBOpenAIAsyncClient
}
