"""
JudgeModel module: Verify claims and generate failure reasons.

Supports verification of:
1. Completed claim templates (new design): Verify completed claims and explanations,
   and check if "not related" judgments are correct
2. Complete claims (legacy): Verify claim correctness
"""

from typing import Union, Dict, Any, List, Optional
from PIL import Image
import asyncio
import json
import re
import os
from pathlib import Path

try:
    from ProbingFactorGeneration.config import FailureTaxonomy, MODEL_CONFIG
except ImportError:
    # Fallback if config doesn't have FailureTaxonomy
    FailureTaxonomy = None
    from ProbingFactorGeneration.config import MODEL_CONFIG

try:
    from ProbingFactorGeneration.utils.async_client import AsyncGeminiClient
except ImportError:
    AsyncGeminiClient = None
    print("Warning: utils.async_client.AsyncGeminiClient not found. "
          "Please implement AsyncGeminiClient or install the required package.")


class JudgeModel:
    """
    Judge model interface for verifying claim correctness and identifying failures.
    
    Supports two verification modes:
    1. Template completion verification: Verify completed claims from templates
       and check if "not related" judgments are appropriate
    2. Complete claim verification: Verify correctness of complete claims
    """
    
    def __init__(self, model_name: str = None, model_config: Dict[str, Any] = None,
                 gpu_id: int = 0, max_concurrent: int = None, request_delay: float = None,
                 use_lb_client: bool = None, api_key: str = None, base_url: str = None):
        """
        Initialize JudgeModel.
        
        Args:
            model_name: Name or identifier of the judge model
            model_config: Configuration dictionary for model initialization
            gpu_id: GPU ID for process isolation
            max_concurrent: Maximum concurrent requests
            request_delay: Delay between requests in seconds
            use_lb_client: Whether to use LBOpenAIAsyncClient
            api_key: API key (overrides env/config if provided)
            base_url: API base URL (overrides env/config if provided)
        """
        self.model_name = model_name or MODEL_CONFIG.get("MODEL_NAME", "gemini-pro-vision")
        self.model_config = model_config or {}
        
        # Async client configuration (similar to BaselineModel)
        self.gpu_id = gpu_id
        
        # Auto-optimize concurrency if not explicitly set
        if max_concurrent is None:
            from ProbingFactorGeneration.config import calculate_optimal_concurrency
            self.max_concurrent = MODEL_CONFIG.get("MAX_CONCURRENT", 10)
            self._auto_optimize_concurrency = True
        else:
            self.max_concurrent = max_concurrent
            self._auto_optimize_concurrency = False
        
        self.request_delay = request_delay or MODEL_CONFIG.get("REQUEST_DELAY", 0.0)
        self.use_lb_client = use_lb_client if use_lb_client is not None else MODEL_CONFIG.get("USE_LB_CLIENT", True)
        
        # Store service_name, env, api_key, base_url for AsyncGeminiClient
        # Explicit params override env/config (for test scripts and direct invocation)
        self.service_name = MODEL_CONFIG.get("SERVICE_NAME") or os.getenv("SERVICE_NAME")
        self.env = MODEL_CONFIG.get("ENV", "prod") or os.getenv("ENV", "prod")
        self.api_key = api_key or MODEL_CONFIG.get("API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "1")
        self.base_url = base_url or MODEL_CONFIG.get("BASE_URL") or os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        # Model parameters
        self.max_tokens = self.model_config.get("max_tokens", MODEL_CONFIG.get("MAX_TOKENS", 1000))
        self.temperature = self.model_config.get("temperature", MODEL_CONFIG.get("TEMPERATURE", 0.3))
        
        self.failure_taxonomy = FailureTaxonomy
        self.model = None  # Will hold the actual model instance
        
        # Async client instance (similar to BaselineModel)
        self._async_client: Optional[Any] = None

        # Optional prompt logging (JSONL). Enabled by environment variable.
        # Intended for debugging/testing to inspect exactly what the judge receives.
        self._prompt_log_path: Optional[str] = (
            os.getenv("JUDGE_PROMPT_LOG_PATH")
            or os.getenv("JUDGE_PROMPT_LOG")
            or None
        )
        self._log_response: bool = os.getenv("JUDGE_LOG_RESPONSE", "").strip().lower() in ("1", "true", "yes")
        self._prompt_log_lock: asyncio.Lock = asyncio.Lock()

    async def _log_judge_prompt_async(self, record: Dict[str, Any]) -> None:
        """
        Append a single JSONL record to the prompt log file (if enabled).
        Never raises (logging must not break the pipeline).
        """
        if not self._prompt_log_path:
            return
        try:
            path = Path(self._prompt_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            async with self._prompt_log_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Best-effort logging only
            return
    
    def optimize_concurrency_for_data_size(self, data_size: int, avg_claims_per_image: int = 10):
        """
        Optimize concurrency based on data size (call before processing batch).
        
        Args:
            data_size: Number of images to process
            avg_claims_per_image: Average number of claims per image
        """
        if self._auto_optimize_concurrency:
            from ProbingFactorGeneration.config import calculate_optimal_concurrency
            optimal = calculate_optimal_concurrency(
                data_size=data_size,
                is_local_model=False,  # Judge models are typically API-based
                avg_claims_per_image=avg_claims_per_image
            )
            if optimal != self.max_concurrent:
                self.max_concurrent = optimal
                # Update semaphore if client already exists
                if hasattr(self, '_async_client') and self._async_client is not None:
                    if hasattr(self._async_client, 'semaphore'):
                        import asyncio
                        self._async_client.semaphore = asyncio.Semaphore(optimal)
                        self._async_client.max_concurrent = optimal
    
    async def _get_client(self) -> Any:
        """
        Get or create async client instance.
        
        Returns:
            AsyncGeminiClient instance
        """
        if AsyncGeminiClient is None:
            raise NotImplementedError(
                "AsyncGeminiClient is not available. "
                "Please install or implement utils.async_client.AsyncGeminiClient"
            )
        
        if self._async_client is None:
            # Pass service_name, env, api_key, base_url for AsyncGeminiClient
            self._async_client = AsyncGeminiClient(
                model_name=self.model_name,
                gpu_id=self.gpu_id,
                max_concurrent=self.max_concurrent,
                request_delay=self.request_delay,
                use_lb_client=self.use_lb_client,
                service_name=self.service_name,
                env=self.env,
                api_key=self.api_key,
                base_url=self.base_url
            )
            await self._async_client.__aenter__()
        
        return self._async_client
    
    async def _close_client(self):
        """Close async client if it exists."""
        if self._async_client is not None:
            try:
                await self._async_client.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._async_client = None
    
    def _image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert PIL Image to base64-encoded string.
        
        Args:
            image: PIL Image object
            format: Image format (JPEG, PNG, etc.)
            
        Returns:
            Base64-encoded image string (without data URI prefix)
        """
        import base64
        import io
        
        buffered = io.BytesIO()
        # Convert RGBA to RGB if needed for JPEG
        if format == "JPEG" and image.mode in ("RGBA", "LA", "P"):
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = rgb_image
        
        image.save(buffered, format=format, quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    
    def _build_precheck_prompt(
        self,
        original_template: str,
        not_related_conditions: Optional[List[str]] = None,
    ) -> str:
        """
        Build prompt for Step 1: lightweight precheck to determine if template is answerable.
        
        Args:
            original_template: Original template with placeholders
            not_related_conditions: Optional NOT_RELATED conditions from claim schema
            
        Returns:
            Formatted prompt for precheck
        """
        conditions_section = ""
        if not_related_conditions:
            # Provide claim-specific NOT_RELATED conditions as additional guidance
            joined = "; ".join(not_related_conditions)
            conditions_section = f"""

Additional guidance (NOT_RELATED conditions from schema, for your reference only):
- Treat the template as NOT answerable if one or more of the following clearly apply:
  {joined}
"""

        prompt = f"""You are evaluating whether a claim template can be answered based on the image.

Your task is to make a coarse-grained judgment: Can this template be meaningfully answered from the image?

Template: {original_template}{conditions_section}

Consider only:
- Whether the image contains the necessary visual properties to instantiate this template
- Whether the template is fundamentally answerable given the image content

Do NOT consider:
- Any baseline model outputs
- Any completed claims
- Any explanations

Please respond in JSON format:
{{
    "template_is_answerable": true/false,
    "confidence": "high" | "medium" | "low",
    "reason": "Brief explanation for your judgment"
}}"""
        
        return prompt
    
    async def _precheck_template_answerability_async(
        self,
        image: Image.Image,
        original_template: str,
        not_related_conditions: Optional[List[str]] = None,
        log_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Step 1: Precheck if template is answerable (lightweight judgment).
        
        Args:
            image: PIL Image object
            original_template: Original template with placeholders
            not_related_conditions: Optional NOT_RELATED conditions from claim schema
            
        Returns:
            Precheck result dictionary:
            {
                "template_is_answerable": bool,
                "confidence": str,  # "high" | "medium" | "low"
                "reason": str
            }
        """
        try:
            # Build precheck prompt (with optional NOT_RELATED conditions)
            prompt = self._build_precheck_prompt(
                original_template,
                not_related_conditions=not_related_conditions,
            )

            # Optional: log the exact prompt the judge receives (precheck)
            if log_context is None:
                log_context = {}
            await self._log_judge_prompt_async(
                {
                    **log_context,
                    "judge_call": "precheck",
                    "prompt": prompt,
                }
            )
            
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Request JSON response format
            response = await self._call_model_async(
                messages,
                response_format={"type": "json_object"}
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            if self._log_response and self._prompt_log_path:
                await self._log_judge_prompt_async({
                    **log_context,
                    "judge_call": "precheck",
                    "prompt": prompt,
                    "response": response_text,
                })
            # Parse JSON response
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from precheck response: {response_text[:200]}")
            
            return {
                "template_is_answerable": parsed.get("template_is_answerable", True),
                "confidence": parsed.get("confidence", "medium"),
                "reason": parsed.get("reason", "")
            }
            
        except Exception as e:
            # Check if this is a connection/network error (model disconnected)
            error_str = str(e).lower()
            is_connection_error = any(keyword in error_str for keyword in [
                "connection", "connect", "network", "timeout", "unreachable",
                "refused", "disconnected", "broken pipe", "reset", "failed to establish"
            ])
            
            if is_connection_error:
                # Model disconnected - return answerable=False to skip this claim
                return {
                    "template_is_answerable": False,
                    "confidence": "low",
                    "reason": f"Model connection error during precheck: {str(e)}",
                    "is_connection_error": True
                }
            else:
                # Other errors - return default answerable=True on error (fail open)
                return {
                    "template_is_answerable": True,
                    "confidence": "low",
                    "reason": f"Error during precheck: {str(e)}",
                    "is_connection_error": False
                }
    
    def _build_verification_prompt(
        self, 
        original_template: str, 
        completed_claim: str, 
        explanation: str, 
        is_related: bool,
        placeholders: List[str] = None,
        not_related_judgment_correct_rule: Optional[bool] = None,
        slots_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for verifying template completion.
        
        Note: This method is only called when template_is_answerable=true.
        The not_related judgment correctness is already determined by rule, not by judge.
        
        Args:
            original_template: Original template with placeholders
            completed_claim: Completed claim from baseline model
            explanation: Explanation from baseline model
            is_related: Whether baseline model marked it as related
            placeholders: List of placeholder names (optional)
            not_related_judgment_correct_rule: Pre-determined not_related judgment correctness (None if not applicable)
            slots_info: Slot information including selection_criteria (optional)
            
        Returns:
            Formatted prompt for verification
        """
        # Add rule-based not_related judgment fact (compact)
        not_related_section = ""
        if not_related_judgment_correct_rule is False:
            not_related_section = "**FACT**: Baseline incorrectly marked as not_related (template is answerable). not_related_judgment_correct=false, is_correct=false.\n\n"
        
        # Build baseline standards section (compact, multi-line for readability)
        # Note: baseline_instructions, NOT_RELATED conditions, and expected_outputs are NOT included
        baseline_standards_lines = []
        if slots_info:
            slot_items = []
            for slot_name, slot_data in slots_info.items():
                parts = [slot_name]
                if slot_data.get("type"):
                    parts.append(f"type:{slot_data['type']}")
                if slot_data.get("selection_criteria"):
                    parts.append(f"criteria:{slot_data['selection_criteria']}")
                if slot_data.get("values"):
                    parts.append(f"values:{','.join(slot_data['values'])}")
                slot_items.append(" | ".join(parts))
            if slot_items:
                baseline_standards_lines.append("Slots: " + "; ".join(slot_items))
        
        baseline_standards_section = ""
        if baseline_standards_lines:
            baseline_standards_section = "\n**Baseline standards**:\n" + "\n".join(baseline_standards_lines) + "\n"
        
        prompt = f"""{not_related_section}**Task**: Verify completed claim correctness.

**Template**: {original_template} | Placeholders: {', '.join(placeholders) if placeholders else 'None'}{baseline_standards_section}**CRITICAL - Baseline outputs to VERIFY** (These are HYPOTHESES, NOT facts. Do NOT treat as true or use as prior knowledge. Verify independently against image):
Claim: {completed_claim}
Explanation: {explanation}
Baseline marked as related: {is_related}

**Standards**: STRICT equivalences ("left"≠"bottom left"); ACCEPTABLE ranges/approx/synonyms. Verify claim follows baseline standards above. Judge SOLELY by image content, NOT by baseline outputs.

**Required output format (respond in JSON)**:
{{
    "is_correct": true/false,
    "claim_is_valid": true/false,
    "explanation_is_reasonable": true/false,
    "failure_reason": "failure_type" or null,
    "judge_explanation": "your explanation",
    "suggested_correction": "correction" or null
}}

**Failure types** (for failure_reason field): visual_error, visual_ambiguity, language_misunderstanding, language_complexity, reasoning_error, commonsense_error, model_limitation, uncertain"""
        
        return prompt

    def _build_prefill_prompt(
        self,
        claim_template: str,
        prefill_slots: List[str],
        slots_info: Optional[Dict[str, Any]] = None,
        claim_name: Optional[str] = None
    ) -> str:
        """
        Build prompt for pre-filling selected template slots.
        """
        slot_lines = []
        for slot_name in prefill_slots:
            slot_data = (slots_info or {}).get(slot_name, {})
            line = f"- {slot_name}"
            if slot_data.get("type"):
                line += f" (type: {slot_data['type']})"
            if slot_data.get("selection_criteria"):
                line += f"\n  Selection criteria: {slot_data['selection_criteria']}"
            if slot_data.get("values"):
                line += f"\n  Valid values: {', '.join(slot_data['values'])}"
            slot_lines.append(line)

        slots_section = "\n".join(slot_lines) if slot_lines else "- None"
        claim_name_line = f"Claim Name: {claim_name}\n" if claim_name else ""

        # Check if any prefill slot is noun-type (needs constraint disambiguation)
        NOUN_SLOT_TYPES = {
            "object_instance", "object_category", "abstract_concept",
            "categorical_value", "relative_spatial_relation"
        }
        has_noun_slots = any(
            (slots_info or {}).get(s, {}).get("type", "").lower() in NOUN_SLOT_TYPES
            for s in prefill_slots
        )

        noun_constraint_instruction = ""
        if has_noun_slots:
            noun_constraint_instruction = """
**For noun-type slots** (object_instance, object_category, abstract_concept, etc.): When the filled term could be ambiguous, add a brief parenthetical constraint after the noun to disambiguate. Format: `noun (brief constraint)`. Examples:
- car → car (four-wheeled passenger vehicles only, not trains or tracks)
- dog → dog (domestic canine, not wolf or fox)
- tree → tree (woody plant with trunk, not bush or shrub)
Keep the constraint brief (under 15 words). Only add when ambiguity exists; omit parentheses if the term is unambiguous.
"""

        prompt = f"""You are selecting core objects or values for specific placeholders in a claim template.
Choose suitable and moderately challenging objects when possible.
If the image is not relevant or you cannot find any suitable object, mark it as not relevant.
Fill ONLY the requested slots based on the image. Keep values concise.
If a slot cannot be determined, return an empty string for that slot.
{noun_constraint_instruction}
{claim_name_line}Claim Template: {claim_template}
Slots to prefill:
{slots_section}

Please respond in JSON format:
{{
  "is_relevant": true/false,
  "filled_values": {{"SLOT_NAME": "value", "...": "..."}}
}}
"""
        return prompt

    async def prefill_template_slots_async(
        self,
        image: Image.Image,
        claim_template: Dict[str, Any],
        prefill_slots: List[str]
    ) -> Dict[str, Any]:
        """
        Prefill selected slots in a claim template based on the image.
        """
        if not prefill_slots:
            return {"filled_values": {}, "metadata": {"source": "prefill_slots", "skipped": True}}

        try:
            template_text = claim_template.get("claim_template") or claim_template.get("claim_text", "")
            slots_info = claim_template.get("slots", {})
            claim_name = claim_template.get("metadata", {}).get("name")

            if not template_text:
                raise ValueError("claim_template must contain 'claim_template' or 'claim_text' field")

            # Validate and convert image
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")

            prompt = self._build_prefill_prompt(
                template_text,
                prefill_slots=prefill_slots,
                slots_info=slots_info if slots_info else None,
                claim_name=claim_name
            )

            image_base64 = self._image_to_base64(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]

            response = await self._call_model_async(
                messages,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            if self._log_response and self._prompt_log_path:
                await self._log_judge_prompt_async({
                    "judge_call": "prefill",
                    "claim_template": template_text,
                    "prefill_slots": prefill_slots,
                    "prompt": prompt,
                    "response": response_text,
                })
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

            filled_values = parsed.get("filled_values", {})
            if not isinstance(filled_values, dict):
                filled_values = {}
            is_relevant = parsed.get("is_relevant", None)

            # Keep only requested slots, coerce to strings
            normalized = {}
            for slot in prefill_slots:
                value = filled_values.get(slot, "")
                if value is None:
                    value = ""
                normalized[slot] = str(value)

            has_any_value = any(str(v).strip() for v in normalized.values())
            if is_relevant is None:
                is_relevant = bool(has_any_value)

            result = {
                "filled_values": normalized,
                "is_relevant": bool(is_relevant),
                "metadata": {
                    "response_text": response_text,
                    "source": "prefill_slots",
                    "prefill_slots": prefill_slots
                }
            }

            if hasattr(response, "usage"):
                result["metadata"]["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }

            return result
        except Exception as e:
            # Check if this is a connection/network error (model disconnected)
            error_str = str(e).lower()
            is_connection_error = any(keyword in error_str for keyword in [
                "connection", "connect", "network", "timeout", "unreachable",
                "refused", "disconnected", "broken pipe", "reset", "failed to establish"
            ])
            
            return {
                "filled_values": {},
                "is_relevant": False,
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "source": "prefill_slots",
                    "is_connection_error": is_connection_error,
                    "skip_reason": "model_connection_error" if is_connection_error else None
                }
            }

    async def prefill_template_slots_batch_async(
        self,
        images: List[Image.Image],
        claim_templates: List[Dict[str, Any]],
        prefill_slots_list: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Prefill selected slots for a batch of templates (async version).
        """
        if not (len(images) == len(claim_templates) == len(prefill_slots_list)):
            raise ValueError("Images, claim_templates, and prefill_slots_list must have same length")

        tasks = [
            self.prefill_template_slots_async(image, template, slots)
            for image, template, slots in zip(images, claim_templates, prefill_slots_list)
        ]
        return await asyncio.gather(*tasks)
    
    async def _call_model_async(
        self, 
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Call model API asynchronously.
        
        Args:
            messages: Message list for API call
            response_format: Optional response format
            
        Returns:
            API response object
        """
        client = await self._get_client()
        
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if response_format:
            request_params["response_format"] = response_format
        
        response = await client.chat.completions.create(**request_params)
        return response
    
    async def verify_completion_async(
        self,
        image: Image.Image,
        claim_template: Dict[str, Any],
        completion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify template completion result (async version).
        
        This method performs a two-step verification process:
        
        Step 1 (Precheck): Lightweight judgment to determine if template is answerable
        - Only uses original_template and image
        - Independent of baseline outputs
        - Returns template_is_answerable, confidence, reason
        
        Step 2 (Verification): Full verification with strict precision rules
        - Uses precheck result as premise fact
        - Evaluates completed_claim correctness with strict spatial relationship matching
        - Checks if baseline incorrectly marked as NOT_RELATED
        
        It checks:
        1. If the completed claim is correct based on the image content
        2. If the explanation is reasonable
        3. If "not related" judgment is correct (if baseline marked as not related)
        4. Strict precision for spatial/positional relationships
        
        Args:
            image: PIL Image object
            claim_template: Original template dictionary
            completion: Completion dictionary from BaselineModel.complete_template_async()
            
        Returns:
            Verification dictionary:
            {
                "is_correct": bool,  # Overall correctness
                "claim_is_valid": bool,  # If completed claim is valid
                "explanation_is_reasonable": bool,  # If explanation makes sense
                "not_related_judgment_correct": bool or None,  # If "not related" was correctly judged
                "failure_reason": str or None,  # Failure reason from taxonomy
                "judge_explanation": str,  # Judge's explanation
                "suggested_correction": str or None,  # Suggested correction if wrong
                "claim_id": str,  # Original claim_id
                "content_type": str,  # Content type
                "metadata": dict  # Additional metadata including precheck_result
            }
        """
        try:
            original_template = completion.get("metadata", {}).get("original_template") or claim_template.get("claim_template", "")
            completed_claim = completion.get("completed_claim", "")
            explanation = completion.get("explanation", "")
            is_related = completion.get("is_related", True)
            placeholders = completion.get("metadata", {}).get("placeholders", []) or claim_template.get("placeholders", [])
            
            # Extract baseline / schema info from claim_template
            slots_info = claim_template.get("slots", {})
            # not_related_conditions is stored in metadata by TemplateClaimGenerator
            not_related_conditions = (
                claim_template.get("not_related_conditions")
                or claim_template.get("metadata", {}).get("not_related_conditions")
            )
            
            # Step 1: Precheck template answerability (lightweight judgment)
            precheck_result = await self._precheck_template_answerability_async(
                image,
                original_template,
                not_related_conditions=not_related_conditions,
                log_context={
                    "claim_id": completion.get("claim_id", ""),
                    "content_type": completion.get("content_type", "relation"),
                    "image_id": (
                        completion.get("image_id")
                        or completion.get("metadata", {}).get("image_id")
                        or claim_template.get("image_id")
                        or claim_template.get("metadata", {}).get("image_id")
                        or ""
                    ),
                },
            )
            
            template_is_answerable = precheck_result.get("template_is_answerable", True)
            
            # Step 2: Early return if template is not answerable (skip second judge call)
            if not template_is_answerable:
                # Template cannot be answered - determine not_related judgment correctness based on baseline
                if not is_related:
                    # Baseline correctly marked as not_related
                    not_related_judgment_correct = True
                    failure_reason = None
                    judge_explanation = f"Template is not answerable from this image. Baseline correctly marked as not_related. {precheck_result.get('reason', '')}"
                    
                    return {
                        "is_correct": False,
                        "claim_is_valid": False,
                        "explanation_is_reasonable": False,  # Cannot be reasonable if template is unanswerable
                        "not_related_judgment_correct": not_related_judgment_correct,
                        "failure_reason": failure_reason,
                        "judge_explanation": judge_explanation,
                        "suggested_correction": None,
                        "claim_id": completion.get("claim_id", ""),
                        "content_type": completion.get("content_type", "relation"),
                        "metadata": {
                            "original_template": original_template,
                            "completed_claim": completed_claim,
                            "baseline_explanation": explanation,
                            "baseline_is_related": is_related,
                            "response_text": None,  # No second judge call
                            "source": "template_completion_verification",
                            "precheck_result": precheck_result,
                            "early_return": True,
                            "early_return_reason": "template_not_answerable"
                        }
                    }
                else:
                    # Baseline thought it was related, but template is not answerable
                    # This claim should be ignored/skipped, not marked as error
                    # Because if claim is not related, no subsequent questions can be constructed
                    return {
                        "is_correct": None,  # Not an error, just skipped
                        "claim_is_valid": False,
                        "explanation_is_reasonable": False,
                        "not_related_judgment_correct": None,  # Not applicable since baseline didn't mark as not_related
                        "failure_reason": None,  # Not a failure, just skipped
                        "judge_explanation": f"Template is not answerable from this image. This claim should be ignored. {precheck_result.get('reason', '')}",
                        "suggested_correction": None,
                        "claim_id": completion.get("claim_id", ""),
                        "content_type": completion.get("content_type", "relation"),
                        "skipped": True,  # Mark as skipped
                        "skip_reason": "template_not_answerable_but_baseline_thought_related",
                        "metadata": {
                            "original_template": original_template,
                            "completed_claim": completed_claim,
                            "baseline_explanation": explanation,
                            "baseline_is_related": is_related,
                            "response_text": None,  # No second judge call
                            "source": "template_completion_verification",
                            "precheck_result": precheck_result,
                            "early_return": True,
                            "early_return_reason": "template_not_answerable_skip"
                        }
                    }
            
            # Step 3: Template is answerable - call second judge for claim correctness verification
            # Determine not_related judgment correctness based on rule (not asking judge to judge)
            if not is_related:
                # Baseline incorrectly marked as not_related (since template_is_answerable=true)
                not_related_judgment_correct_rule = False
            else:
                # Baseline correctly marked as related (or didn't mark as not_related)
                not_related_judgment_correct_rule = None  # Not applicable
            
            # Build verification prompt (only for claim correctness, not_related is already determined)
            prompt = self._build_verification_prompt(
                original_template, 
                completed_claim, 
                explanation, 
                is_related,
                placeholders,
                not_related_judgment_correct_rule=not_related_judgment_correct_rule,
                slots_info=slots_info
            )

            # Optional: log the exact prompt the judge receives (verification)
            await self._log_judge_prompt_async(
                {
                    "judge_call": "verification",
                    "claim_id": completion.get("claim_id", ""),
                    "content_type": completion.get("content_type", "relation"),
                    "image_id": (
                        completion.get("image_id")
                        or completion.get("metadata", {}).get("image_id")
                        or claim_template.get("image_id")
                        or claim_template.get("metadata", {}).get("image_id")
                        or ""
                    ),
                    "prompt": prompt,
                }
            )
            
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Request JSON response format
            response = await self._call_model_async(
                messages,
                response_format={"type": "json_object"}
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            if self._log_response and self._prompt_log_path:
                await self._log_judge_prompt_async({
                    "judge_call": "verification",
                    "claim_id": completion.get("claim_id", ""),
                    "content_type": completion.get("content_type", "relation"),
                    "image_id": (
                        completion.get("image_id")
                        or completion.get("metadata", {}).get("image_id")
                        or claim_template.get("image_id")
                        or claim_template.get("metadata", {}).get("image_id")
                        or ""
                    ),
                    "prompt": prompt,
                    "response": response_text,
                })
            
            # Parse JSON response
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
            
            # Build verification result (Step 0: keep all existing fields)
            # Override not_related_judgment_correct with rule-based value
            verification = {
                "is_correct": parsed.get("is_correct", True),
                "claim_is_valid": parsed.get("claim_is_valid", True),
                "explanation_is_reasonable": parsed.get("explanation_is_reasonable", True),
                "not_related_judgment_correct": not_related_judgment_correct_rule,  # Use rule-based value
                "failure_reason": parsed.get("failure_reason"),
                "judge_explanation": parsed.get("judge_explanation", ""),
                "suggested_correction": parsed.get("suggested_correction"),
                "claim_id": completion.get("claim_id", ""),
                "content_type": completion.get("content_type", "relation"),
                "metadata": {
                    "original_template": original_template,
                    "completed_claim": completed_claim,
                    "baseline_explanation": explanation,
                    "baseline_is_related": is_related,
                    "response_text": response_text,
                    "source": "template_completion_verification",
                    "precheck_result": precheck_result
                }
            }
            
            # If baseline incorrectly marked as not_related, ensure is_correct is False
            if not_related_judgment_correct_rule is False:
                verification["is_correct"] = False
                if not verification.get("failure_reason"):
                    verification["failure_reason"] = "reasoning_error"
            
            # Add usage metadata if available
            if hasattr(response, "usage"):
                verification["metadata"]["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            
            return verification
            
        except Exception as e:
            # Check if this is a connection/network error (model disconnected)
            error_str = str(e).lower()
            is_connection_error = any(keyword in error_str for keyword in [
                "connection", "connect", "network", "timeout", "unreachable",
                "refused", "disconnected", "broken pipe", "reset", "failed to establish"
            ])
            
            if is_connection_error:
                # Model disconnected - skip this claim instead of marking as error
                return {
                    "is_correct": None,  # Not an error, just skipped
                    "claim_is_valid": None,
                    "explanation_is_reasonable": None,
                    "not_related_judgment_correct": None,
                    "failure_reason": None,  # Not a failure, just skipped
                    "judge_explanation": f"Model connection error, skipping claim: {str(e)}",
                    "suggested_correction": None,
                    "claim_id": completion.get("claim_id", ""),
                    "content_type": completion.get("content_type", "relation"),
                    "skipped": True,  # Mark as skipped
                    "skip_reason": "model_connection_error",
                    "metadata": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "is_connection_error": True
                    }
                }
            else:
                # Other errors - return error in metadata
                return {
                    "is_correct": False,
                    "claim_is_valid": False,
                    "explanation_is_reasonable": False,
                    "not_related_judgment_correct": None,
                    "failure_reason": "model_limitation",
                    "judge_explanation": f"Error during verification: {str(e)}",
                    "suggested_correction": None,
                    "claim_id": completion.get("claim_id", ""),
                    "content_type": completion.get("content_type", "relation"),
                    "metadata": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "is_connection_error": False
                    }
                }
    
    def verify_completion(
        self,
        image: Image.Image,
        claim_template: Dict[str, Any],
        completion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify template completion result (sync wrapper).
        
        Args:
            image: PIL Image object
            claim_template: Original template dictionary
            completion: Completion dictionary from BaselineModel
            
        Returns:
            Verification dictionary (same format as verify_completion_async())
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync verify_completion() from within an async context. "
                "Use verify_completion_async() instead."
            )
        except RuntimeError as e:
            if "Cannot use sync" in str(e):
                raise
            pass
        
        return asyncio.run(self._verify_completion_with_context(image, claim_template, completion))
    
    async def _verify_completion_with_context(
        self,
        image: Image.Image,
        claim_template: Dict[str, Any],
        completion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify completion with proper async context management."""
        try:
            await self._get_client()
            result = await self.verify_completion_async(image, claim_template, completion)
            return result
        finally:
            pass
    
    async def verify_completion_batch_async(
        self,
        images: List[Image.Image],
        claim_templates: List[Dict[str, Any]],
        completions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verify a batch of template completions (async version).
        
        Args:
            images: List of PIL Image objects
            claim_templates: List of template dictionaries
            completions: List of completion dictionaries
            
        Returns:
            List of verification dictionaries
        """
        if not (len(images) == len(claim_templates) == len(completions)):
            raise ValueError("Images, claim_templates, and completions must have same length")
        
        try:
            await self._get_client()
            
            # Create tasks for concurrent execution
            tasks = [
                self.verify_completion_async(image, template, completion)
                for image, template, completion in zip(images, claim_templates, completions)
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            verifications = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Check if this is a connection/network error (model disconnected)
                    error_str = str(result).lower()
                    is_connection_error = any(keyword in error_str for keyword in [
                        "connection", "connect", "network", "timeout", "unreachable",
                        "refused", "disconnected", "broken pipe", "reset", "failed to establish"
                    ])
                    
                    if is_connection_error:
                        # Model disconnected - skip this claim instead of marking as error
                        verifications.append({
                            "is_correct": None,  # Not an error, just skipped
                            "claim_is_valid": None,
                            "explanation_is_reasonable": None,
                            "not_related_judgment_correct": None,
                            "failure_reason": None,  # Not a failure, just skipped
                            "judge_explanation": f"Model connection error, skipping claim: {str(result)}",
                            "suggested_correction": None,
                            "claim_id": completions[i].get("claim_id", ""),
                            "content_type": completions[i].get("content_type", "relation"),
                            "skipped": True,  # Mark as skipped
                            "skip_reason": "model_connection_error",
                            "metadata": {
                                "error": str(result),
                                "error_type": type(result).__name__,
                                "index": i,
                                "is_connection_error": True
                            }
                        })
                    else:
                        # Other errors - mark as failure
                        verifications.append({
                            "is_correct": False,
                            "claim_is_valid": False,
                            "explanation_is_reasonable": False,
                            "not_related_judgment_correct": None,
                            "failure_reason": "model_limitation",
                            "judge_explanation": f"Error: {str(result)}",
                            "suggested_correction": None,
                            "claim_id": completions[i].get("claim_id", ""),
                            "content_type": completions[i].get("content_type", "relation"),
                            "metadata": {
                                "error": str(result),
                                "error_type": type(result).__name__,
                                "index": i,
                                "is_connection_error": False
                            }
                        })
                else:
                    verifications.append(result)
            
            return verifications
            
        finally:
            pass
    
    def verify_completion_batch(
        self,
        images: List[Image.Image],
        claim_templates: List[Dict[str, Any]],
        completions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verify a batch of template completions (sync wrapper).
        
        Args:
            images: List of PIL Image objects
            claim_templates: List of template dictionaries
            completions: List of completion dictionaries
            
        Returns:
            List of verification dictionaries
        """
        return asyncio.run(
            self._verify_completion_batch_with_context(images, claim_templates, completions)
        )
    
    async def _verify_completion_batch_with_context(
        self,
        images: List[Image.Image],
        claim_templates: List[Dict[str, Any]],
        completions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Verify completion batch with proper async context management."""
        return await self.verify_completion_batch_async(images, claim_templates, completions)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()
    
    def close(self):
        """
        Close async client (sync wrapper).
        Should be called when done with the model to clean up resources.
        """
        if self._async_client is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._close_client())
                else:
                    loop.run_until_complete(self._close_client())
            except RuntimeError:
                asyncio.run(self._close_client())
    
    def verify(self, image: Image.Image, claim: Dict[str, Any], 
              baseline_answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a claim's correctness and identify failure reason if incorrect.
        
        Args:
            image: PIL Image object
            claim: Claim dictionary containing at least "claim_text"
            baseline_answer: Baseline prediction dictionary from BaselineModel
            
        Returns:
            Verification dictionary:
            {
                "is_correct": bool,
                "failure_reason": str or None,  # Failure reason from taxonomy if incorrect
                "confidence": float,  # Optional confidence score
                "metadata": dict  # Additional verification metadata
            }
            
        TODO:
            - Implement claim verification logic
            - Compare baseline answer with ground truth or judge's assessment
            - Classify failure reason using failure taxonomy
            - Return appropriate failure category from FailureTaxonomy
        """
        # TODO: Implement verification logic
        # verification_result = self.model.verify(image, claim, baseline_answer)
        # failure_reason = self._classify_failure(verification_result)
        # return {
        #     "is_correct": verification_result["is_correct"],
        #     "failure_reason": failure_reason,
        #     "confidence": verification_result.get("confidence", None),
        #     "metadata": verification_result.get("metadata", {})
        # }
        raise NotImplementedError("JudgeModel.verify() not implemented")
    
    def verify_batch(self, images: List[Image.Image], 
                    claims: List[Dict[str, Any]], 
                    baseline_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify a batch of claims.
        
        Args:
            images: List of PIL Image objects
            claims: List of claim dictionaries
            baseline_answers: List of baseline prediction dictionaries
            
        Returns:
            List of verification dictionaries (same format as verify())
            
        TODO:
            - Implement batch verification
            - Optimize for batch processing
        """
        # TODO: Implement batch verification
        # if not (len(images) == len(claims) == len(baseline_answers)):
        #     raise ValueError("All inputs must have same length")
        # verifications = []
        # for image, claim, answer in zip(images, claims, baseline_answers):
        #     verif = self.verify(image, claim, answer)
        #     verifications.append(verif)
        # return verifications
        raise NotImplementedError("JudgeModel.verify_batch() not implemented")
    
    def _classify_failure(self, verification_result: Dict[str, Any]) -> Union[str, None]:
        """
        Classify failure reason from verification result.
        
        Args:
            verification_result: Raw verification result from model
            
        Returns:
            Failure reason string from FailureTaxonomy, or None if correct
            
        TODO:
            - Map verification result to failure taxonomy
            - Use model output to determine specific failure category
            - Handle edge cases (uncertain, ambiguous, etc.)
        """
        # TODO: Implement failure classification
        # if verification_result["is_correct"]:
        #     return None
        # 
        # # Classify based on verification result features
        # failure_features = verification_result.get("failure_features", {})
        # failure_reason = self._map_to_taxonomy(failure_features)
        # return failure_reason
        raise NotImplementedError("JudgeModel._classify_failure() not implemented")
    
    def get_available_failure_reasons(self) -> List[str]:
        """
        Get list of available failure reasons from taxonomy.
        
        Returns:
            List of failure reason strings
        """
        return [reason.value for reason in self.failure_taxonomy]
