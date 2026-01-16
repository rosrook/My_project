"""
TemplateClaimGenerator module: Generate claim templates with placeholders.

This module loads claim templates (templates with replaceable elements) from JSON config,
which will be completed by the baseline model based on the image content.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import json
from PIL import Image
from ProbingFactorGeneration.config import ContentType


class TemplateClaimGenerator:
    """
    Generate claim templates with placeholders from predefined JSON configuration.
    
    Templates contain replaceable elements (placeholders) that will be filled
    by the baseline model based on image content.
    
    All templates are global and will be applied to all images.
    """
    
    def __init__(self, config_path: Union[str, Path] = "claim_config.json"):
        """
        Initialize TemplateClaimGenerator.
        
        Args:
            config_path: Path to the claim template configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self):
        """
        Load claim template configuration from JSON file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
            ValueError: If config file format is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Claim config file not found: {self.config_path}\n"
                f"Please create a claim_config.json file with claim templates.\n"
                f"See configs/claim_template.example.json for reference."
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
        
        # Config should have:
        # 1. "claim_schemas" - new format with rich metadata (preferred)
        # 2. "global_templates" - legacy format
        # 3. "templates" - backward compatibility (treated as global_templates)
        # 4. "metadata" - optional metadata (legacy format)
        # 5. "domain", "version", "description" - optional metadata (new format)
        # 6. "slot_type_definitions" - definitions of slot types (v1.1+)
        # 7. "global_guidelines" - global guidelines for template completion (v1.1+)
        
        valid_keys = [
            "claim_schemas",
            "global_templates",
            "templates",
            "metadata",
            "domain",
            "version",
            "description",
            "slot_type_definitions",
            "global_guidelines",
            "enabled_claim_ids",
            "enabled_claim_names",
        ]
        for key in self.config.keys():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid config key: {key}. "
                    f"Valid keys are: {valid_keys}"
                )
    
    def generate(self, image: Image.Image, image_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate claim templates for a given image.
        
        Args:
            image: PIL Image object (not used for templates, but kept for API compatibility)
            image_id: Image identifier (optional, used for generating template IDs)
            
        Returns:
            List of claim template dictionaries, each containing:
            {
                "claim_id": str,
                "claim_template": str,  # Template with placeholders (e.g., "The [OBJECT] is in the [REGION]")
                "content_type": str,
                "placeholders": List[str],  # List of placeholder/slot names
                "slots": Dict[str, Dict],  # Optional slot information (type, description, values, selection_criteria)
                "metadata": dict  # Additional template metadata (name, capability, baseline_instructions, not_related_conditions, etc.)
            }
            
            Supported formats:
            - New format: claim_schemas with rich metadata (name, capability, slots, baseline_instructions)
            - Legacy format: global_templates (simple templates)
        """
        templates = []
        
        # New format: claim_schemas (preferred)
        if "claim_schemas" in self.config:
            schemas = self.config["claim_schemas"]
            templates.extend(self._normalize_schemas(schemas))
        
        # Legacy format: global_templates
        if "global_templates" in self.config and not templates:
            global_templates = self.config["global_templates"]
            templates.extend(self._normalize_templates(global_templates, image_id))
        
        # Backward compatibility: support "templates" key (treated as global)
        if "templates" in self.config and not templates:
            templates = self._normalize_templates(self.config["templates"], image_id)
        
        # Also support legacy "global_claims" / "claims" if they exist
        # (treat them as templates if templates not found)
        if not templates:
            if "global_claims" in self.config:
                templates.extend(self._normalize_templates(self.config["global_claims"], image_id))
            if "claims" in self.config:
                templates.extend(self._normalize_templates(self.config["claims"], image_id))
        
        return self._filter_templates(templates)

    def _filter_templates(self, templates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter templates by enabled claim ids/names from config.
        """
        enabled_ids = self.config.get("enabled_claim_ids", [])
        enabled_names = self.config.get("enabled_claim_names", [])

        if isinstance(enabled_ids, str):
            enabled_ids = [enabled_ids]
        if isinstance(enabled_names, str):
            enabled_names = [enabled_names]

        enabled_ids = [v for v in enabled_ids if isinstance(v, str) and v.strip()]
        enabled_names = [v for v in enabled_names if isinstance(v, str) and v.strip()]

        if not enabled_ids and not enabled_names:
            return templates

        filtered = []
        for template in templates:
            claim_id = template.get("claim_id", "")
            claim_name = template.get("metadata", {}).get("name", "")
            if (enabled_ids and claim_id in enabled_ids) or (enabled_names and claim_name in enabled_names):
                filtered.append(template)
        return filtered
    
    def _normalize_templates(self, templates: Union[List[Dict], List[str], Dict], 
                            image_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Normalize templates to standard format.
        
        Args:
            templates: Templates in various formats
            image_id: Image ID for generating template IDs
            
        Returns:
            List of normalized template dictionaries
        """
        normalized = []
        
        # Handle different input formats
        if isinstance(templates, dict):
            # Single template dict
            templates = [templates]
        elif isinstance(templates, str):
            # Single template string
            templates = [templates]
        
        # Process each template
        for idx, template in enumerate(templates):
            if isinstance(template, str):
                # Simple string format: treat as template
                normalized_template = {
                    "claim_id": self._generate_template_id(image_id, idx),
                    "claim_template": template,
                    "content_type": self._infer_content_type(template),
                    "placeholders": self._extract_placeholders(template),
                    "metadata": {
                        "source": "predefined_config",
                        "original_format": "string"
                    }
                }
            elif isinstance(template, dict):
                # Dict format: ensure required fields exist
                template_text = (template.get("claim_template") or 
                               template.get("template") or 
                               template.get("claim_text") or 
                               template.get("text") or 
                               template.get("claim"))
                
                if not template_text:
                    raise ValueError(
                        f"Template dict must contain 'claim_template', 'template', 'claim_text', 'text', or 'claim' field. "
                        f"Got: {template.keys()}"
                    )
                
                prefill_slots = template.get("prefill_slots", [])
                if isinstance(prefill_slots, str):
                    prefill_slots = [prefill_slots]
                elif not isinstance(prefill_slots, list):
                    prefill_slots = []

                normalized_template = {
                    "claim_id": (template.get("claim_id") or 
                               self._generate_template_id(image_id, idx)),
                    "claim_template": template_text,
                    "content_type": (template.get("content_type") or 
                                   self._infer_content_type(template_text)),
                    "placeholders": (template.get("placeholders") or 
                                   self._extract_placeholders(template_text)),
                    "prefill_slots": prefill_slots,
                    "metadata": {
                        **template.get("metadata", {}),
                        "source": "predefined_config",
                        "original_format": "dict",
                        "prefill_slots": prefill_slots
                    }
                }
            else:
                raise ValueError(
                    f"Invalid template format. Expected str or dict, got {type(template)}"
                )
            
            normalized.append(normalized_template)
        
        return normalized
    
    def _normalize_schemas(self, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize claim schemas from new format to standard template format.
        
        Args:
            schemas: List of claim schema dictionaries
            
        Returns:
            List of normalized template dictionaries
        """
        normalized = []
        
        for schema in schemas:
            if not isinstance(schema, dict):
                raise ValueError(f"Each claim_schema must be a dictionary. Got: {type(schema)}")
            
            claim_template = schema.get("claim_template") or schema.get("template", "")
            if not claim_template:
                raise ValueError(
                    f"Claim schema must contain 'claim_template' field. "
                    f"Schema keys: {schema.keys()}"
                )
            
            # Extract slot/placeholder names from template and schema
            slot_names = list(schema.get("slots", {}).keys())
            placeholders = self._extract_placeholders(claim_template)
            prefill_slots = schema.get("prefill_slots", [])
            if isinstance(prefill_slots, str):
                prefill_slots = [prefill_slots]
            elif not isinstance(prefill_slots, list):
                prefill_slots = []
            
            # Ensure all placeholders have corresponding slots or vice versa
            if slot_names and placeholders:
                # Use intersection - slots defined in schema take precedence
                used_placeholders = [p for p in placeholders if p in slot_names] or placeholders
            elif slot_names:
                used_placeholders = slot_names
            else:
                used_placeholders = placeholders
            
            # Build slots information (including selection_criteria from v1.1+)
            slots_info = {}
            for slot_name in used_placeholders:
                if slot_name in schema.get("slots", {}):
                    slot_def = schema["slots"][slot_name]
                    slots_info[slot_name] = {
                        "type": slot_def.get("type", "text"),
                        "description": slot_def.get("description", ""),
                        "values": slot_def.get("values", []),  # For categorical types
                        "selection_criteria": slot_def.get("selection_criteria", "")  # v1.1+: how to select values
                    }
                else:
                    # Default slot info if not in schema
                    slots_info[slot_name] = {
                        "type": "text",
                        "description": "",
                        "values": [],
                        "selection_criteria": ""
                    }
            
            # Infer content_type from capability if available
            capability = schema.get("capability", "")
            content_type = schema.get("content_type") or self._infer_content_type_from_capability(capability)
            
            normalized_template = {
                "claim_id": schema.get("claim_id", ""),
                "claim_template": claim_template,
                "content_type": content_type or self._infer_content_type(claim_template),
                "placeholders": used_placeholders,
                "slots": slots_info,  # Rich slot information (including selection_criteria)
                "prefill_slots": prefill_slots,
                "metadata": {
                    "name": schema.get("name", ""),
                    "capability": capability,
                    "baseline_instructions": schema.get("baseline_instructions", []),
                    "expected_outputs": schema.get("expected_outputs", []),
                    "common_failure_modes": schema.get("common_failure_modes", []),
                    "not_related_conditions": schema.get("not_related_conditions", []),  # v1.1+: when to return NOT_RELATED
                    "target_failure_id": schema.get("target_failure_id", ""),  # v1.1+: target failure ID
                    "prefill_slots": prefill_slots,  # v1.1+: slots to prefill with judge
                    "source": "claim_schema",
                    "original_format": "schema"
                }
            }
            
            normalized.append(normalized_template)
        
        return normalized
    
    def _infer_content_type_from_capability(self, capability: str) -> Optional[str]:
        """
        Infer content type from capability string.
        
        Args:
            capability: Capability string (e.g., "coarse_object_localization")
            
        Returns:
            Inferred content type string or None
        """
        if not capability:
            return None
        
        capability_lower = capability.lower()
        
        if "spatial" in capability_lower or "localization" in capability_lower:
            return ContentType.SPATIAL.value
        elif "count" in capability_lower or "number" in capability_lower:
            return ContentType.COUNT.value
        elif "attribute" in capability_lower:
            return ContentType.ATTRIBUTE.value
        elif "action" in capability_lower:
            return ContentType.ACTION.value
        elif "text" in capability_lower:
            return ContentType.TEXT.value
        elif "object" in capability_lower:
            return ContentType.OBJECT.value
        else:
            return None
    
    def _extract_placeholders(self, template_text: str) -> List[str]:
        """
        Extract placeholder names from template text.
        
        Placeholders can be in the format:
        - {placeholder_name}
        - {{placeholder_name}}  (for escaping)
        - [placeholder_name]
        
        Args:
            template_text: Template text with placeholders
            
        Returns:
            List of placeholder names found in the template
        """
        import re
        placeholders = []
        
        # Match {placeholder_name} or [PLACEHOLDER_NAME] or [placeholder_name]
        # Note: New format uses [SLOT_NAME] (uppercase), but we support both
        patterns = [
            r'\{([^}]+)\}',      # {placeholder}
            r'\[([^\]]+)\]',     # [PLACEHOLDER] or [placeholder]
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, template_text)
            # Filter out escaped braces {{placeholder}} -> {placeholder} (only count once)
            for match in matches:
                if match not in placeholders:
                    placeholders.append(match)
        
        return placeholders
    
    def _generate_template_id(self, image_id: Optional[str], index: int) -> str:
        """
        Generate a unique template ID.
        
        Args:
            image_id: Image identifier (optional)
            index: Template index
            
        Returns:
            Generated template ID string
        """
        if image_id:
            return f"{image_id}_template_{index}"
        else:
            return f"template_{index}"
    
    def _infer_content_type(self, template_text: str) -> str:
        """
        Infer content type from template text (simple heuristic).
        
        Args:
            template_text: Template text
            
        Returns:
            Inferred content type string
        """
        template_lower = template_text.lower()
        
        # Simple heuristics (can be enhanced)
        if any(word in template_lower for word in ["how many", "count", "number of"]):
            return ContentType.COUNT.value
        elif any(word in template_lower for word in ["left", "right", "above", "below", "beside", "near"]):
            return ContentType.SPATIAL.value
        elif any(word in template_lower for word in ["doing", "action", "performing", "is"]):
            return ContentType.ACTION.value
        elif any(word in template_lower for word in ["color", "size", "shape", "attribute"]):
            return ContentType.ATTRIBUTE.value
        elif any(word in template_lower for word in ["contains", "has", "there is", "there are", "show"]):
            return ContentType.OBJECT.value
        elif any(word in template_lower for word in ["text", "says", "reads", "words"]):
            return ContentType.TEXT.value
        else:
            # Default to relation if unsure
            return ContentType.RELATION.value
    
    def generate_batch(self, images: List[Image.Image], 
                      image_ids: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate templates for a batch of images.
        
        Args:
            images: List of PIL Image objects
            image_ids: Optional list of image identifiers
            
        Returns:
            Dictionary mapping image_id to list of templates
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]
        
        if len(images) != len(image_ids):
            raise ValueError("Images and image_ids must have same length")
        
        results = {}
        for image, img_id in zip(images, image_ids):
            results[img_id] = self.generate(image, img_id)
        
        return results
    
    
    def get_global_templates_count(self) -> int:
        """
        Get the number of templates/schemas.
        
        Returns:
            Number of templates/schemas
        """
        # New format: claim_schemas
        if "claim_schemas" in self.config:
            schemas = self.config["claim_schemas"]
            if isinstance(schemas, list):
                return len(schemas)
        
        # Legacy format: global_templates
        if "global_templates" in self.config:
            templates = self.config["global_templates"]
            if isinstance(templates, list):
                return len(templates)
            elif isinstance(templates, dict):
                return 1
            elif isinstance(templates, str):
                return 1
        
        # Check legacy key
        if "global_claims" in self.config:
            claims = self.config["global_claims"]
            if isinstance(claims, list):
                return len(claims)
        return 0
