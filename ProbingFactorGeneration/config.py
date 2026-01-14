"""
Configuration and constants for the probing factor generation framework.
"""

from typing import List, Dict, Optional
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
