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
                 use_lb_client: bool = None):
        """
        Initialize JudgeModel.
        
        Args:
            model_name: Name or identifier of the judge model
            model_config: Configuration dictionary for model initialization
            gpu_id: GPU ID for process isolation
            max_concurrent: Maximum concurrent requests
            request_delay: Delay between requests in seconds
            use_lb_client: Whether to use LBOpenAIAsyncClient
        """
        self.model_name = model_name or MODEL_CONFIG.get("MODEL_NAME", "gemini-pro-vision")
        self.model_config = model_config or {}
        
        # Async client configuration (similar to BaselineModel)
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent or MODEL_CONFIG.get("MAX_CONCURRENT", 10)
        self.request_delay = request_delay or MODEL_CONFIG.get("REQUEST_DELAY", 0.1)
        self.use_lb_client = use_lb_client if use_lb_client is not None else MODEL_CONFIG.get("USE_LB_CLIENT", True)
        
        # Store service_name, env, api_key for LBOpenAIAsyncClient
        self.service_name = MODEL_CONFIG.get("SERVICE_NAME") or os.getenv("SERVICE_NAME")
        self.env = MODEL_CONFIG.get("ENV", "prod") or os.getenv("ENV", "prod")
        self.api_key = MODEL_CONFIG.get("API_KEY") or os.getenv("API_KEY", "1")
        
        # Model parameters
        self.max_tokens = self.model_config.get("max_tokens", MODEL_CONFIG.get("MAX_TOKENS", 1000))
        self.temperature = self.model_config.get("temperature", MODEL_CONFIG.get("TEMPERATURE", 0.3))
        
        self.failure_taxonomy = FailureTaxonomy
        self.model = None  # Will hold the actual model instance
        
        # Async client instance (similar to BaselineModel)
        self._async_client: Optional[Any] = None
    
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
            # Pass service_name, env, api_key for LBOpenAIAsyncClient
            self._async_client = AsyncGeminiClient(
                model_name=self.model_name,
                gpu_id=self.gpu_id,
                max_concurrent=self.max_concurrent,
                request_delay=self.request_delay,
                use_lb_client=self.use_lb_client,
                service_name=self.service_name,
                env=self.env,
                api_key=self.api_key
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
    
    def _build_verification_prompt(
        self, 
        original_template: str, 
        completed_claim: str, 
        explanation: str, 
        is_related: bool,
        placeholders: List[str] = None
    ) -> str:
        """
        Build prompt for verifying template completion.
        
        Args:
            original_template: Original template with placeholders
            completed_claim: Completed claim from baseline model
            explanation: Explanation from baseline model
            is_related: Whether baseline model marked it as related
            placeholders: List of placeholder names (optional)
            
        Returns:
            Formatted prompt for verification
        """
        prompt = f"""You are a judge evaluating a claim completion task. Your task is to:
1. Verify if the completed claim is correct based on the image content
2. Verify if the explanation is reasonable
3. **Important**: If the baseline model marked it as "not related" (is_related=false), 
   check if the template could actually be answered based on the image.
   If you believe the template CAN be answered, this is a FAILURE - the baseline model 
   incorrectly rejected a valid template.

Original Template: {original_template}
Completed Claim: {completed_claim}
Baseline Explanation: {explanation}
Baseline marked as related: {is_related}
Placeholders in template: {', '.join(placeholders) if placeholders else 'None'}

Please respond in JSON format:
{{
    "is_correct": true/false,  // Overall correctness
    "claim_is_valid": true/false,  // If completed claim is valid (ignoring "not related")
    "explanation_is_reasonable": true/false,  // If explanation makes sense
    "not_related_judgment_correct": true/false/null,  // null if not marked as "not related", 
                                                      // true if correctly marked "not related",
                                                      // false if incorrectly marked "not related"
    "failure_reason": "failure_type" or null,  // From taxonomy if incorrect
    "judge_explanation": "Your detailed explanation for the judgment",
    "suggested_correction": "What the claim should be if incorrect, or null"
}}

Failure reasons (if is_correct is false):
- "visual_error": Visual recognition mistake
- "visual_ambiguity": Unclear image content
- "language_misunderstanding": Ambiguous claim
- "language_complexity": Complex reasoning required
- "reasoning_error": Logical inconsistency
- "commonsense_error": Violates common sense
- "model_limitation": Out of distribution
- "uncertain": Insufficient information

**Key rule**: If baseline marked as "not related" but you can see the template CAN be 
answered from the image, set "is_correct" to false and "not_related_judgment_correct" to false."""
        
        return prompt
    
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
        
        This is the new method for verifying completed claims from templates.
        It checks:
        1. If the completed claim is correct
        2. If the explanation is reasonable
        3. If "not related" judgment is correct (if baseline marked as not related)
        
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
                "metadata": dict  # Additional metadata
            }
        """
        try:
            original_template = completion.get("metadata", {}).get("original_template") or claim_template.get("claim_template", "")
            completed_claim = completion.get("completed_claim", "")
            explanation = completion.get("explanation", "")
            is_related = completion.get("is_related", True)
            placeholders = completion.get("metadata", {}).get("placeholders", []) or claim_template.get("placeholders", [])
            
            # Build verification prompt
            prompt = self._build_verification_prompt(
                original_template, 
                completed_claim, 
                explanation, 
                is_related,
                placeholders
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
            
            # Build verification result
            verification = {
                "is_correct": parsed.get("is_correct", True),
                "claim_is_valid": parsed.get("claim_is_valid", True),
                "explanation_is_reasonable": parsed.get("explanation_is_reasonable", True),
                "not_related_judgment_correct": parsed.get("not_related_judgment_correct"),
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
                    "source": "template_completion_verification"
                }
            }
            
            # Add usage metadata if available
            if hasattr(response, "usage"):
                verification["metadata"]["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            
            return verification
            
        except Exception as e:
            # Return error in metadata
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
                    "error_type": type(e).__name__
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
                            "index": i
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
