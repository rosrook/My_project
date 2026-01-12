"""
PredefinedClaimGenerator module: Load claims from predefined JSON configuration.

This module allows loading claims from a predefined claim_config.json file,
where claims are manually designed and associated with image IDs.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import json
from PIL import Image
from ...config import ContentType


class PredefinedClaimGenerator:
    """
    Generate claims from predefined JSON configuration file.
    
    Supports loading manually designed claims from claim_config.json,
    where claims can be associated with specific image IDs or used as templates.
    """
    
    def __init__(self, config_path: Union[str, Path] = "claim_config.json"):
        """
        Initialize PredefinedClaimGenerator.
        
        Args:
            config_path: Path to the claim configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self):
        """
        Load claim configuration from JSON file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
            ValueError: If config file format is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Claim config file not found: {self.config_path}\n"
                f"Please create a claim_config.json file with the following structure:\n"
                f"See configs/claim_config.example.json for reference."
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Validate config structure
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate the configuration structure.
        
        Raises:
            ValueError: If config structure is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Config should have either:
        # 1. "claims_by_image" - claims mapped by image_id
        # 2. "global_claims" - claims to apply to all images
        # 3. "claims" - list of claims (deprecated, for backward compatibility)
        
        valid_keys = ["claims_by_image", "global_claims", "claims", "metadata"]
        for key in self.config.keys():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid config key: {key}. "
                    f"Valid keys are: {valid_keys}"
                )
    
    def generate(self, image: Image.Image, image_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate claims for a given image from predefined config.
        
        Args:
            image: PIL Image object (not used for predefined claims, but kept for API compatibility)
            image_id: Image identifier (used to look up specific claims)
            
        Returns:
            List of structured claim dictionaries, each containing:
            {
                "claim_id": str,
                "claim_text": str,
                "content_type": str,
                "metadata": dict  # Additional claim metadata
            }
        """
        claims = []
        
        # Try to get image-specific claims first
        if image_id and "claims_by_image" in self.config:
            image_claims = self.config["claims_by_image"].get(image_id, [])
            if image_claims:
                claims.extend(self._normalize_claims(image_claims, image_id))
        
        # Add global claims (applied to all images)
        if "global_claims" in self.config:
            global_claims = self.config["global_claims"]
            claims.extend(self._normalize_claims(global_claims, image_id))
        
        # Backward compatibility: support "claims" key (treated as global)
        if "claims" in self.config and not claims:
            claims = self._normalize_claims(self.config["claims"], image_id)
        
        return claims
    
    def _normalize_claims(self, claims: Union[List[Dict], List[str], Dict], 
                         image_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Normalize claims to standard format.
        
        Args:
            claims: Claims in various formats:
                   - List of claim dicts with full structure
                   - List of claim strings (will be converted to dicts)
                   - Single dict (will be converted to list)
            image_id: Image ID for generating claim IDs
            
        Returns:
            List of normalized claim dictionaries
        """
        normalized = []
        
        # Handle different input formats
        if isinstance(claims, dict):
            # Single claim dict
            claims = [claims]
        elif isinstance(claims, str):
            # Single claim string
            claims = [claims]
        
        # Process each claim
        for idx, claim in enumerate(claims):
            if isinstance(claim, str):
                # Simple string format: convert to dict
                normalized_claim = {
                    "claim_id": self._generate_claim_id(image_id, idx),
                    "claim_text": claim,
                    "content_type": self._infer_content_type(claim),
                    "metadata": {
                        "source": "predefined_config",
                        "original_format": "string"
                    }
                }
            elif isinstance(claim, dict):
                # Dict format: ensure required fields exist
                normalized_claim = {
                    "claim_id": claim.get("claim_id") or self._generate_claim_id(image_id, idx),
                    "claim_text": claim.get("claim_text") or claim.get("text") or claim.get("claim"),
                    "content_type": claim.get("content_type") or self._infer_content_type(claim.get("claim_text", "")),
                    "metadata": {
                        **claim.get("metadata", {}),
                        "source": "predefined_config",
                        "original_format": "dict"
                    }
                }
                
                # Validate required fields
                if not normalized_claim["claim_text"]:
                    raise ValueError(
                        f"Claim dict must contain 'claim_text', 'text', or 'claim' field. "
                        f"Got: {claim.keys()}"
                    )
            else:
                raise ValueError(
                    f"Invalid claim format. Expected str or dict, got {type(claim)}"
                )
            
            normalized.append(normalized_claim)
        
        return normalized
    
    def _generate_claim_id(self, image_id: Optional[str], index: int) -> str:
        """
        Generate a unique claim ID.
        
        Args:
            image_id: Image identifier (optional)
            index: Claim index
            
        Returns:
            Generated claim ID string
        """
        if image_id:
            return f"{image_id}_claim_{index}"
        else:
            return f"claim_{index}"
    
    def _infer_content_type(self, claim_text: str) -> str:
        """
        Infer content type from claim text (simple heuristic).
        
        Args:
            claim_text: Claim text
            
        Returns:
            Inferred content type string
        """
        claim_lower = claim_text.lower()
        
        # Simple heuristics (can be enhanced)
        if any(word in claim_lower for word in ["how many", "count", "number of"]):
            return ContentType.COUNT.value
        elif any(word in claim_lower for word in ["left", "right", "above", "below", "beside", "near"]):
            return ContentType.SPATIAL.value
        elif any(word in claim_lower for word in ["doing", "action", "performing", "is"]):
            return ContentType.ACTION.value
        elif any(word in claim_lower for word in ["color", "size", "shape", "attribute"]):
            return ContentType.ATTRIBUTE.value
        elif any(word in claim_lower for word in ["contains", "has", "there is", "there are"]):
            return ContentType.OBJECT.value
        elif any(word in claim_lower for word in ["text", "says", "reads", "words"]):
            return ContentType.TEXT.value
        else:
            # Default to relation if unsure
            return ContentType.RELATION.value
    
    def generate_batch(self, images: List[Image.Image], 
                      image_ids: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate claims for a batch of images from predefined config.
        
        Args:
            images: List of PIL Image objects
            image_ids: Optional list of image identifiers
            
        Returns:
            Dictionary mapping image_id to list of claims:
            {
                "image_id_1": [claim1, claim2, ...],
                "image_id_2": [claim3, claim4, ...],
                ...
            }
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]
        
        if len(images) != len(image_ids):
            raise ValueError("Images and image_ids must have same length")
        
        results = {}
        for image, img_id in zip(images, image_ids):
            results[img_id] = self.generate(image, img_id)
        
        return results
    
    def get_all_image_ids(self) -> List[str]:
        """
        Get all image IDs defined in the config.
        
        Returns:
            List of image ID strings
        """
        if "claims_by_image" in self.config:
            return list(self.config["claims_by_image"].keys())
        return []
    
    def has_image_claims(self, image_id: str) -> bool:
        """
        Check if specific image has defined claims.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if image has specific claims, False otherwise
        """
        if "claims_by_image" in self.config:
            return image_id in self.config["claims_by_image"]
        return False
    
    def get_global_claims_count(self) -> int:
        """
        Get the number of global claims.
        
        Returns:
            Number of global claims
        """
        if "global_claims" in self.config:
            claims = self.config["global_claims"]
            if isinstance(claims, list):
                return len(claims)
            elif isinstance(claims, dict):
                return 1
            elif isinstance(claims, str):
                return 1
        return 0
