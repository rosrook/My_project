"""
Core processing modules for probing factor generation.
"""

from .loaders.image_loader import ImageLoader
from .generators.claim_generator import ClaimGenerator
from .generators.predefined_claim_generator import PredefinedClaimGenerator
from .generators.template_claim_generator import TemplateClaimGenerator
from .aggregators.failure_aggregator import FailureAggregator
from .mappers.filtering_factor_mapper import FilteringFactorMapper

try:
    from .mappers.failure_reason_matcher import FailureReasonMatcher
    __all__ = [
        "ImageLoader",
        "ClaimGenerator",
        "PredefinedClaimGenerator",
        "TemplateClaimGenerator",
        "FailureAggregator",
        "FilteringFactorMapper",
        "FailureReasonMatcher",
    ]
except ImportError:
    __all__ = [
        "ImageLoader",
        "ClaimGenerator",
        "PredefinedClaimGenerator",
        "TemplateClaimGenerator",
        "FailureAggregator",
        "FilteringFactorMapper",
    ]
