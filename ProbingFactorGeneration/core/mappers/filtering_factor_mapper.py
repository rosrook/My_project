"""
FilteringFactorMapper module: Map failure reasons to filtering factors.
"""

from typing import List, Dict, Any, Set

try:
    from ProbingFactorGeneration.config import get_filtering_factor, get_all_filtering_factors
except ImportError:
    # Fallback functions if config doesn't have these
    def get_filtering_factor(failure_reason: str) -> str:
        return failure_reason
    
    def get_all_filtering_factors() -> List[str]:
        return []


class FilteringFactorMapper:
    """
    Map failure reasons to reusable filtering factors for data construction.
    
    This mapper collects suggested_filtering_factors from failure reasons
    and provides methods to aggregate them.
    """
    
    def __init__(self, custom_mapping: Dict[str, str] = None):
        """
        Initialize FilteringFactorMapper.
        
        Args:
            custom_mapping: Optional custom mapping from failure reasons to filtering factors.
                           Currently not used, kept for backward compatibility.
        """
        self.custom_mapping = custom_mapping or {}
    
    def map(self, filtering_factors: List[str]) -> List[str]:
        """
        Map/process a list of filtering factors (identity function for now).
        
        Args:
            filtering_factors: List of filtering factor strings
            
        Returns:
            List of filtering factor strings (same list)
        """
        # For now, just return the list as-is
        # Could add deduplication or normalization in the future
        return list(set(filtering_factors))  # Deduplicate
    
    def map_batch(self, filtering_factors_list: List[List[str]]) -> List[str]:
        """
        Map a batch of filtering factor lists and merge them.
        
        Args:
            filtering_factors_list: List of filtering factor lists
            
        Returns:
            Merged and deduplicated list of all filtering factors
        """
        all_factors = set()
        for factors in filtering_factors_list:
            all_factors.update(factors)
        return sorted(list(all_factors))  # Return sorted for consistency
    
    def map_aggregated_failures(self, aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance aggregated result with filtering factor information.
        
        Args:
            aggregated_result: Aggregated failure result from FailureAggregator
            
        Returns:
            Enhanced aggregated result (same structure, no changes for now)
        """
        # For now, just return as-is
        # Filtering factors are collected per-image, not per-failure-breakdown
        return aggregated_result
    
    def get_all_filtering_factors(self) -> List[str]:
        """
        Get all available filtering factors.
        
        Returns:
            List of filtering factor strings
        """
        return get_all_filtering_factors()
