"""
FailureReasonMatcher module: Match failure reasons from failure_config.json.

This module loads failure reason configurations and matches them to claims
based on claim_id, target_failure_id, failure_category, and descriptions.
"""

from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path
import json


class FailureReasonMatcher:
    """
    Match failure reasons to claims and extract suggested filtering factors.
    
    This class loads failure_config.json and provides methods to:
    1. Match failure reasons by target_failure_id (from claim schema)
    2. Match failure reasons by failure_category
    3. Extract suggested_filtering_factors for matched failures
    """
    
    def __init__(self, failure_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize FailureReasonMatcher.
        
        Args:
            failure_config_path: Path to failure_config.json file.
                                Defaults to "configs/failure_config.example.json"
        """
        if failure_config_path is None:
            # Try to find config file relative to project root
            config_path = Path(__file__).parent.parent.parent / "configs" / "failure_config.example.json"
            self.config_path = config_path
        else:
            self.config_path = Path(failure_config_path)
        
        self.config: Dict[str, Any] = {}
        self.failure_reasons: List[Dict[str, Any]] = []
        self.failure_by_id: Dict[str, Dict[str, Any]] = {}  # failure_id -> failure_reason
        self.failure_by_category: Dict[str, List[Dict[str, Any]]] = {}  # category -> [failure_reasons]
        
        self.load_config()
    
    def load_config(self):
        """
        Load failure configuration from JSON file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
            ValueError: If config file format is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Failure config file not found: {self.config_path}\n"
                f"Please create a failure_config.json file.\n"
                f"See configs/failure_config.example.json for reference."
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Validate config structure
        if "failure_reasons" not in self.config:
            raise ValueError(
                f"Invalid failure config format: missing 'failure_reasons' field.\n"
                f"Config keys: {self.config.keys()}"
            )
        
        self.failure_reasons = self.config["failure_reasons"]
        
        # Build indices for fast lookup
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for fast failure reason lookup."""
        self.failure_by_id = {}
        self.failure_by_category = {}
        
        for failure_reason in self.failure_reasons:
            failure_id = failure_reason.get("failure_id")
            failure_category = failure_reason.get("failure_category")
            
            if failure_id:
                self.failure_by_id[failure_id] = failure_reason
            
            if failure_category:
                if failure_category not in self.failure_by_category:
                    self.failure_by_category[failure_category] = []
                self.failure_by_category[failure_category].append(failure_reason)
    
    def get_failure_by_id(self, failure_id: str) -> Optional[Dict[str, Any]]:
        """
        Get failure reason by failure_id.
        
        Args:
            failure_id: Failure ID (e.g., "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE")
            
        Returns:
            Failure reason dictionary or None if not found
        """
        return self.failure_by_id.get(failure_id)
    
    def get_failures_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all failure reasons in a category.
        
        Args:
            category: Failure category (e.g., "spatial_reasoning")
            
        Returns:
            List of failure reason dictionaries
        """
        return self.failure_by_category.get(category, [])
    
    def match_failure_for_claim(
        self,
        claim_template: Dict[str, Any],
        judge_failure_reason: Optional[str] = None,
        is_related: bool = True,
        not_related_judgment_correct: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Match failure reason for a claim template.
        
        This method:
        1. First tries to match by target_failure_id from claim schema
        2. Falls back to matching by failure_category if target_failure_id not found
        3. Considers not_related judgment correctness
        
        Args:
            claim_template: Claim template dictionary (should contain metadata with target_failure_id)
            judge_failure_reason: Failure reason string from judge model (optional)
            is_related: Whether baseline marked claim as related
            not_related_judgment_correct: Whether not_related judgment was correct (None if not applicable)
            
        Returns:
            Matched failure reason dictionary with suggested_filtering_factors, or None
        """
        # If claim is marked as not_related but judge says it should be related,
        # this is a failure - need to match appropriate failure reason
        if not is_related:
            if not_related_judgment_correct is False:
                # Baseline incorrectly marked as not_related
                # Try to match based on claim's target_failure_id
                metadata = claim_template.get("metadata", {})
                target_failure_id = metadata.get("target_failure_id")
                
                if target_failure_id:
                    failure_reason = self.get_failure_by_id(target_failure_id)
                    if failure_reason:
                        return failure_reason
                
                # Try to find a task applicability failure by category
                task_failures = self.get_failures_by_category("task_applicability")
                if task_failures:
                    return task_failures[0]
                
                # If no match found, return None (will be handled by caller)
                return None
            else:
                # Correctly marked as not_related, no failure
                return None
        
        # If judge provided a failure_reason, try to match it
        if judge_failure_reason:
            # Try direct match by failure_id
            matched = self.get_failure_by_id(judge_failure_reason)
            if matched:
                return matched
        
        # Try to match by target_failure_id from claim schema
        metadata = claim_template.get("metadata", {})
        target_failure_id = metadata.get("target_failure_id")
        
        if target_failure_id:
            matched = self.get_failure_by_id(target_failure_id)
            if matched:
                return matched
        
        # Fall back to matching by capability/failure_category
        capability = metadata.get("capability", "")
        if capability:
            # Map capability to failure_category (heuristic)
            category = self._capability_to_category(capability)
            if category:
                failures = self.get_failures_by_category(category)
                if failures:
                    # Return the first one (could be improved with better matching)
                    return failures[0]
        
        # No match found
        return None
    
    def _capability_to_category(self, capability: str) -> Optional[str]:
        """
        Map capability string to failure_category (heuristic mapping).
        
        Args:
            capability: Capability string from claim schema
            
        Returns:
            Failure category string or None
        """
        capability_lower = capability.lower()
        
        # Mapping based on common patterns
        if "spatial" in capability_lower or "localization" in capability_lower or "position" in capability_lower:
            return "spatial_reasoning"
        elif "concept" in capability_lower or "abstraction" in capability_lower:
            return "conceptual_grounding" if "grounding" in capability_lower else "conceptual_abstraction"
        elif "count" in capability_lower or "counting" in capability_lower:
            return "relational_reasoning"
        elif "absence" in capability_lower or "negative" in capability_lower:
            return "negative_reasoning"
        elif "occlusion" in capability_lower or "partial" in capability_lower or "robust" in capability_lower:
            return "robustness"
        elif "scale" in capability_lower or "proportion" in capability_lower or "size" in capability_lower:
            return "scale_reasoning"
        elif "orientation" in capability_lower or "direction" in capability_lower:
            return "spatial_reasoning"
        elif "place" in capability_lower or "geographic" in capability_lower or "map" in capability_lower:
            return "geographic_reasoning"
        elif "caption" in capability_lower or "text" in capability_lower or "alignment" in capability_lower:
            return "text_image_alignment"
        elif "entity" in capability_lower or "visual" in capability_lower or "recognition" in capability_lower:
            return "visual_grounding"
        
        return None
    
    def get_filtering_factors(
        self,
        failure_reason: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract suggested_filtering_factors from a failure reason.
        
        Args:
            failure_reason: Failure reason dictionary
            
        Returns:
            List of suggested filtering factors (empty list if no failure_reason)
        """
        if not failure_reason:
            return []
        
        return failure_reason.get("suggested_filtering_factors", [])
    
    def get_all_filtering_factors(self) -> Set[str]:
        """
        Get all unique filtering factors from all failure reasons.
        
        Returns:
            Set of all unique filtering factor strings
        """
        all_factors = set()
        for failure_reason in self.failure_reasons:
            factors = failure_reason.get("suggested_filtering_factors", [])
            all_factors.update(factors)
        return all_factors
