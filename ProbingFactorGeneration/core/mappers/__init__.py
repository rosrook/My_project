"""Filtering factor mapping modules."""

from .filtering_factor_mapper import FilteringFactorMapper

try:
    from .failure_reason_matcher import FailureReasonMatcher
    __all__ = ["FilteringFactorMapper", "FailureReasonMatcher"]
except ImportError:
    __all__ = ["FilteringFactorMapper"]
