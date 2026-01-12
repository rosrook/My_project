"""
BaselineModel module: Call and run inference with baseline model using async API.

This module implements async model calling using AsyncGeminiClient for concurrent processing.

Usage Examples:

1. Sync usage (single prediction):
    model = BaselineModel(model_name="gemini-pro-vision")
    prediction = model.predict(image, claim)
    model.close()  # Clean up resources

2. Async usage (single prediction):
    async with BaselineModel() as model:
        prediction = await model.predict_async(image, claim)

3. Batch prediction (async - recommended):
    async with BaselineModel(max_concurrent=10) as model:
        predictions = await model.predict_batch_async(images, claims)

4. Batch prediction (sync wrapper):
    model = BaselineModel(max_concurrent=10)
    predictions = model.predict_batch(images, claims)
    model.close()

5. Multi-GPU processing:
    # Process batch with 8 GPUs, 10 concurrent requests per GPU
    async def process_multi_gpu(images, claims, num_gpus=8):
        tasks_per_gpu = len(images) // num_gpus
        results = []
        
        async def process_gpu(gpu_id, start_idx, end_idx):
            async with BaselineModel(
                gpu_id=gpu_id,
                max_concurrent=10
            ) as model:
                return await model.predict_batch_async(
                    images[start_idx:end_idx],
                    claims[start_idx:end_idx]
                )
        
        gpu_tasks = [
            process_gpu(i, i * tasks_per_gpu, (i + 1) * tasks_per_gpu)
            for i in range(num_gpus)
        ]
        
        all_results = await asyncio.gather(*gpu_tasks)
        for gpu_results in all_results:
            results.extend(gpu_results)
        return results

Note: Requires utils.async_client.AsyncGeminiClient to be available.
"""

from typing import Union, Dict, Any, List, Optional
from PIL import Image
import asyncio
import base64
import io
import json

try:
    from ...utils.async_client import AsyncGeminiClient
except ImportError:
    # Fallback: Define a placeholder if utils.async_client is not available
    AsyncGeminiClient = None
    print("Warning: utils.async_client.AsyncGeminiClient not found. "
          "Please implement AsyncGeminiClient or install the required package.")

# Try to import AsyncLLaVAClient for local models
try:
    from ...utils.async_llava_client import AsyncLLaVAClient
except ImportError:
    AsyncLLaVAClient = None

from ...config import MODEL_CONFIG


class BaselineModel:
    """
    Baseline model interface for predicting claim validity using async API calls.
    Returns True/False/Answer predictions for each claim.
    Supports both async and sync interfaces.
    """
    
    def __init__(
        self, 
        model_name: str = None, 
        model_config: Dict[str, Any] = None,
        gpu_id: int = 0,
        max_concurrent: int = None,
        request_delay: float = None,
        use_lb_client: bool = None,
        model_path: str = None,
        use_local_model: bool = None
    ):
        """
        Initialize BaselineModel.
        
        Args:
            model_name: Name or identifier of the baseline model (for API-based models)
            model_config: Configuration dictionary for model initialization
            gpu_id: GPU ID for process isolation (default: 0)
            max_concurrent: Maximum concurrent requests (default: from MODEL_CONFIG)
            request_delay: Delay between requests in seconds (default: from MODEL_CONFIG)
            use_lb_client: Whether to use LBOpenAIAsyncClient (default: from MODEL_CONFIG)
            model_path: Path to local model directory (for local models, e.g., LLaVA)
            use_local_model: Whether to use local model instead of API (auto-detected if model_path is provided)
        """
        self.model_name = model_name or MODEL_CONFIG.get("MODEL_NAME", "gemini-pro-vision")
        self.model_config = model_config or {}
        self.model_path = model_path or MODEL_CONFIG.get("MODEL_PATH")
        
        # Determine if using local model
        if use_local_model is None:
            self.use_local_model = self.model_path is not None
        else:
            self.use_local_model = use_local_model
        
        # Async client configuration
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent or MODEL_CONFIG.get("MAX_CONCURRENT", 10)
        self.request_delay = request_delay or MODEL_CONFIG.get("REQUEST_DELAY", 0.1)
        self.use_lb_client = use_lb_client if use_lb_client is not None else MODEL_CONFIG.get("USE_LB_CLIENT", True)
        
        # Model parameters
        self.max_tokens = self.model_config.get("max_tokens", MODEL_CONFIG.get("MAX_TOKENS", 1000))
        self.temperature = self.model_config.get("temperature", MODEL_CONFIG.get("TEMPERATURE", 0.3))
        
        # Async client instance (initialized when needed)
        self._async_client: Optional[Any] = None
        self._client_context = None
    
    async def _get_client(self) -> Any:
        """
        Get or create async client instance.
        Should be used within async context manager.
        
        Returns:
            AsyncGeminiClient or AsyncLLaVAClient instance
        """
        if self._async_client is None:
            if self.use_local_model:
                # Use local LLaVA model
                if AsyncLLaVAClient is None:
                    raise NotImplementedError(
                        "AsyncLLaVAClient is not available. "
                        "Please install transformers: pip install transformers"
                    )
                if not self.model_path:
                    raise ValueError(
                        "model_path is required when using local model. "
                        "Please provide model_path parameter or set MODEL_PATH in config."
                    )
                
                self._async_client = AsyncLLaVAClient(
                    model_path=self.model_path,
                    gpu_id=self.gpu_id,
                    max_concurrent=self.max_concurrent,
                    request_delay=self.request_delay,
                    device=self.model_config.get("device"),
                    torch_dtype=self.model_config.get("torch_dtype"),
                    load_in_8bit=self.model_config.get("load_in_8bit", False),
                    load_in_4bit=self.model_config.get("load_in_4bit", False),
                )
            else:
                # Use API-based model (AsyncGeminiClient)
                if AsyncGeminiClient is None:
                    raise NotImplementedError(
                        "AsyncGeminiClient is not available. "
                        "Please install or implement utils.async_client.AsyncGeminiClient"
                    )
                
                self._async_client = AsyncGeminiClient(
                    model_name=self.model_name,
                    gpu_id=self.gpu_id,
                    max_concurrent=self.max_concurrent,
                    request_delay=self.request_delay,
                    use_lb_client=self.use_lb_client
                )
            
            # Enter async context
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
        buffered = io.BytesIO()
        # Convert RGBA to RGB if needed for JPEG
        if format == "JPEG" and image.mode in ("RGBA", "LA", "P"):
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = rgb_image
        
        image.save(buffered, format=format, quality=85)  # Compress for efficiency
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    
    def _build_messages(self, image: Image.Image, claim: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build messages for API call.
        
        Args:
            image: PIL Image object
            claim: Claim dictionary containing at least "claim_text" or "claim_template"
            
        Returns:
            List of message dictionaries for OpenAI-compatible API
        """
        # Support both claim_text (complete claim) and claim_template (template)
        claim_text = claim.get("claim_text") or claim.get("claim_template", "")
        if not claim_text:
            raise ValueError("claim must contain 'claim_text' or 'claim_template' field")
        
        # Convert image to base64
        image_base64 = self._image_to_base64(image)
        
        # Build message content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": claim_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        return messages
    
    def _build_template_completion_prompt(
        self, 
        claim_template: str, 
        placeholders: List[str],
        slots_info: Optional[Dict[str, Any]] = None,
        baseline_instructions: Optional[List[str]] = None,
        expected_outputs: Optional[List[str]] = None,
        not_related_conditions: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for template completion (v1.1+ format with enhanced slot information).
        
        Args:
            claim_template: Template with placeholders (e.g., "The [OBJECT] is in the [REGION]")
            placeholders: List of placeholder names
            slots_info: Optional slot information (type, description, values, selection_criteria)
            baseline_instructions: Optional specific instructions for baseline model
            expected_outputs: Optional expected output values
            not_related_conditions: Optional conditions when to return NOT_RELATED (v1.1+)
            
        Returns:
            Formatted prompt for template completion
        """
        # Use custom instructions if available
        if baseline_instructions:
            instructions_section = "\n".join([
                f"{i+1}. {instruction}" for i, instruction in enumerate(baseline_instructions)
            ])
            task_description = f"""You are given a claim template about an image. Follow these instructions:
{instructions_section}"""
        else:
            task_description = """You are given a claim template about an image. Your task is to:
1. Complete the template by filling in the placeholders based on the image content
2. If the template is not relevant to the image, respond with "not related"
3. Provide an explanation for your completion"""
        
        # Build slots information section (v1.1+ format with selection_criteria)
        slots_section = ""
        if slots_info:
            slots_details = []
            for slot_name, slot_data in slots_info.items():
                slot_desc = f"  - {slot_name}"
                if slot_data.get("type"):
                    slot_desc += f" (type: {slot_data['type']})"
                if slot_data.get("description"):
                    slot_desc += f": {slot_data['description']}"
                # v1.1+: Include selection_criteria if available
                if slot_data.get("selection_criteria"):
                    slot_desc += f"\n    Selection criteria: {slot_data['selection_criteria']}"
                if slot_data.get("values"):
                    slot_desc += f"\n    Valid values: {', '.join(slot_data['values'])}"
                slots_details.append(slot_desc)
            
            if slots_details:
                slots_section = "\n\nSlot Information:\n" + "\n".join(slots_details)
        else:
            slots_section = f"\n\nPlaceholders to fill: {', '.join(placeholders) if placeholders else 'None'}"
        
        # Build expected outputs section
        outputs_section = ""
        if expected_outputs:
            outputs_section = f"\n\nExpected Outputs: {', '.join(expected_outputs)}"
            # Map to our format - use NOT_RELATED if in expected outputs, otherwise use lowercase
            if "NOT_RELATED" in expected_outputs:
                not_related_value = "NOT_RELATED"
            else:
                not_related_value = "not related"
        else:
            not_related_value = "not related"
        
        # Build NOT_RELATED conditions section (v1.1+)
        not_related_section = ""
        if not_related_conditions:
            conditions_list = "\n".join([
                f"  - {condition}" for condition in not_related_conditions
            ])
            not_related_section = f"\n\nReturn '{not_related_value}' (set is_related to false) if ANY of the following conditions apply:\n{conditions_list}"
        
        prompt = f"""{task_description}

Claim Template: {claim_template}
{slots_section}{outputs_section}{not_related_section}

Please respond in JSON format:
{{
    "completed_claim": "The completed claim with placeholders filled, or '{not_related_value}' if not applicable",
    "is_related": true/false,
    "explanation": "Your explanation for the completion or why it's not related",
    "filled_values": {{"slot1": "value1", ...}}  // Only if is_related is true
}}

Important guidelines:
- For each slot, independently select the most appropriate and salient element from the image based on the selection criteria provided
- Fill in all placeholders with specific values from the image when the template is relevant
- Return "{not_related_value}" (set is_related to false) only when the image fundamentally lacks the necessary visual properties to instantiate a meaningful claim
- All COUNT or numeric slots must return exact integer values, not ranges or approximations
- Claims should be evaluable as TRUE or FALSE based solely on visual content"""
        
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
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            API response object
        """
        client = await self._get_client()
        
        # Prepare request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if response_format:
            request_params["response_format"] = response_format
        
        # Call API
        response = await client.chat.completions.create(**request_params)
        return response
    
    def _parse_output(self, response_text: str, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse model output to prediction format.
        
        Args:
            response_text: Raw text response from model
            claim: Original claim dictionary (for context)
            
        Returns:
            Prediction dictionary:
            {
                "prediction": Union[bool, str],  # True/False or answer string
                "confidence": Optional[float],  # Confidence score if available
                "metadata": dict  # Additional metadata
            }
        """
        # Clean response text
        response_text = response_text.strip().lower()
        
        # Try to parse as JSON first
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                # If JSON, extract prediction and confidence
                prediction = parsed.get("prediction", parsed.get("answer", None))
                confidence = parsed.get("confidence", None)
                metadata = {k: v for k, v in parsed.items() 
                          if k not in ("prediction", "answer", "confidence")}
                if prediction is not None:
                    # Convert string booleans to actual booleans
                    if isinstance(prediction, str):
                        if prediction.lower() in ("true", "yes", "correct"):
                            prediction = True
                        elif prediction.lower() in ("false", "no", "incorrect"):
                            prediction = False
                    return {
                        "prediction": prediction,
                        "confidence": confidence,
                        "metadata": metadata or {}
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If not JSON or no prediction field, try to infer boolean/answer
        if response_text in ("true", "yes", "correct"):
            return {"prediction": True, "confidence": None, "metadata": {}}
        elif response_text in ("false", "no", "incorrect"):
            return {"prediction": False, "confidence": None, "metadata": {}}
        else:
            # Treat as answer string
            return {"prediction": response_text, "confidence": None, "metadata": {}}
    
    async def complete_template_async(
        self, 
        image: Image.Image, 
        claim_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete a claim template based on image content (async version).
        
        This is the new method for template-based claim generation.
        The model will fill placeholders in the template or return "not related".
        
        Args:
            image: PIL Image object
            claim_template: Template dictionary containing:
                - "claim_template": str (template with placeholders)
                - "placeholders": List[str] (list of placeholder names)
                - "content_type": str (optional)
                - "claim_id": str (optional)
                
        Returns:
            Completion dictionary:
            {
                "completed_claim": str,  # Completed claim or "not related"
                "is_related": bool,  # Whether template is relevant to image
                "explanation": str,  # Explanation for completion
                "filled_values": Dict[str, str],  # Filled placeholder values (if related)
                "claim_id": str,  # Original template claim_id
                "content_type": str,  # Content type from template
                "metadata": dict  # Additional metadata
            }
        """
        try:
            template_text = claim_template.get("claim_template") or claim_template.get("claim_text", "")
            placeholders = claim_template.get("placeholders", [])
            slots_info = claim_template.get("slots", {})
            metadata = claim_template.get("metadata", {})
            baseline_instructions = metadata.get("baseline_instructions", [])
            expected_outputs = metadata.get("expected_outputs", [])
            not_related_conditions = metadata.get("not_related_conditions", [])
            
            if not template_text:
                raise ValueError("claim_template must contain 'claim_template' or 'claim_text' field")
            
            # Build completion prompt with rich metadata if available (v1.1+ format)
            prompt = self._build_template_completion_prompt(
                template_text, 
                placeholders,
                slots_info=slots_info if slots_info else None,
                baseline_instructions=baseline_instructions if baseline_instructions else None,
                expected_outputs=expected_outputs if expected_outputs else None,
                not_related_conditions=not_related_conditions if not_related_conditions else None
            )
            
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Build messages for API call
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
                # Try to extract JSON from text if it's wrapped
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
            
            # Build completion result
            completion = {
                "completed_claim": parsed.get("completed_claim", ""),
                "is_related": parsed.get("is_related", True),
                "explanation": parsed.get("explanation", ""),
                "filled_values": parsed.get("filled_values", {}),
                "claim_id": claim_template.get("claim_id", ""),
                "content_type": claim_template.get("content_type", "relation"),
                "metadata": {
                    "original_template": template_text,
                    "placeholders": placeholders,
                    "response_text": response_text,
                    "source": "template_completion"
                }
            }
            
            # If marked as not related, ensure completed_claim is consistent with expected format
            if not completion["is_related"]:
                # Check if expected_outputs uses "NOT_RELATED" (uppercase) or "not related" (lowercase)
                metadata = claim_template.get("metadata", {})
                expected_outputs = metadata.get("expected_outputs", [])
                if "NOT_RELATED" in expected_outputs:
                    completion["completed_claim"] = "NOT_RELATED"
                else:
                    completion["completed_claim"] = "not related"
            
            # Add usage metadata if available
            if hasattr(response, "usage"):
                completion["metadata"]["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            
            return completion
            
        except Exception as e:
            # Return error in metadata
            return {
                "completed_claim": None,
                "is_related": False,
                "explanation": f"Error during template completion: {str(e)}",
                "filled_values": {},
                "claim_id": claim_template.get("claim_id", ""),
                "content_type": claim_template.get("content_type", "relation"),
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "original_template": claim_template.get("claim_template", "")
                }
            }
    
    def complete_template(
        self, 
        image: Image.Image, 
        claim_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete a claim template based on image content (sync wrapper).
        
        Args:
            image: PIL Image object
            claim_template: Template dictionary
            
        Returns:
            Completion dictionary (same format as complete_template_async())
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync complete_template() from within an async context. "
                "Use complete_template_async() instead."
            )
        except RuntimeError as e:
            if "Cannot use sync" in str(e):
                raise
            pass
        
        return asyncio.run(self._complete_template_with_context(image, claim_template))
    
    async def _complete_template_with_context(
        self, 
        image: Image.Image, 
        claim_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete template with proper async context management."""
        try:
            await self._get_client()
            result = await self.complete_template_async(image, claim_template)
            return result
        finally:
            pass
    
    async def complete_template_batch_async(
        self, 
        images: List[Image.Image], 
        claim_templates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Complete a batch of claim templates (async version).
        
        Args:
            images: List of PIL Image objects
            claim_templates: List of template dictionaries
            
        Returns:
            List of completion dictionaries (same format as complete_template_async())
        """
        if len(images) != len(claim_templates):
            raise ValueError("Images and claim_templates must have same length")
        
        try:
            await self._get_client()
            
            # Create tasks for concurrent execution
            tasks = [
                self.complete_template_async(image, template)
                for image, template in zip(images, claim_templates)
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            completions = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    completions.append({
                        "completed_claim": None,
                        "is_related": False,
                        "explanation": f"Error: {str(result)}",
                        "filled_values": {},
                        "claim_id": claim_templates[i].get("claim_id", ""),
                        "content_type": claim_templates[i].get("content_type", "relation"),
                        "metadata": {
                            "error": str(result),
                            "error_type": type(result).__name__,
                            "index": i
                        }
                    })
                else:
                    completions.append(result)
            
            return completions
            
        finally:
            pass
    
    def complete_template_batch(
        self, 
        images: List[Image.Image], 
        claim_templates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Complete a batch of claim templates (sync wrapper).
        
        Args:
            images: List of PIL Image objects
            claim_templates: List of template dictionaries
            
        Returns:
            List of completion dictionaries
        """
        return asyncio.run(self._complete_template_batch_with_context(images, claim_templates))
    
    async def _complete_template_batch_with_context(
        self, 
        images: List[Image.Image], 
        claim_templates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Complete template batch with proper async context management."""
        return await self.complete_template_batch_async(images, claim_templates)
    
    async def predict_async(self, image: Image.Image, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict baseline answer for a claim given an image (async version).
        
        Args:
            image: PIL Image object
            claim: Claim dictionary containing at least "claim_text"
            
        Returns:
            Prediction dictionary:
            {
                "prediction": Union[bool, str],  # True/False or answer string
                "confidence": Optional[float],  # Optional confidence score
                "metadata": dict  # Additional prediction metadata
            }
        """
        try:
            # Build messages
            messages = self._build_messages(image, claim)
            
            # Call model
            response = await self._call_model_async(messages)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Parse output
            prediction = self._parse_output(response_text, claim)
            
            # Add response metadata
            prediction["metadata"]["response_text"] = response_text
            if hasattr(response, "usage"):
                prediction["metadata"]["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            
            return prediction
            
        except Exception as e:
            # Return error in metadata
            return {
                "prediction": None,
                "confidence": None,
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }
    
    def predict(self, image: Image.Image, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict baseline answer for a claim given an image (sync wrapper).
        
        This is a synchronous wrapper around predict_async() using asyncio.run().
        For better performance with batch processing, use predict_batch() or predict_batch_async().
        
        Note: If called from within an async context, consider using predict_async() directly
        or installing nest_asyncio for better compatibility.
        
        Args:
            image: PIL Image object
            claim: Claim dictionary containing at least "claim_text"
            
        Returns:
            Prediction dictionary (same format as predict_async())
        """
        # Try to get existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a running loop, we need nest_asyncio or a different approach
            # For simplicity, raise an error and suggest using async version
            raise RuntimeError(
                "Cannot use sync predict() from within an async context. "
                "Use predict_async() instead, or install nest_asyncio and apply it first."
            )
        except RuntimeError as e:
            # Check if this is our custom error
            if "Cannot use sync" in str(e):
                raise
            # No running loop, safe to use asyncio.run()
            pass
        
        return asyncio.run(self._predict_with_context(image, claim))
    
    async def _predict_with_context(self, image: Image.Image, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict with proper async context management.
        
        Args:
            image: PIL Image object
            claim: Claim dictionary
            
        Returns:
            Prediction dictionary
        """
        try:
            # Ensure client is available
            await self._get_client()
            # Make prediction
            result = await self.predict_async(image, claim)
            return result
        finally:
            # Note: We don't close client here to allow reuse across multiple calls
            # Client should be closed explicitly via close() method or context manager
            pass
    
    async def predict_batch_async(
        self, 
        images: List[Image.Image], 
        claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predict baseline answers for a batch of image-claim pairs (async version).
        Uses asyncio.gather() for concurrent processing.
        
        Args:
            images: List of PIL Image objects
            claims: List of claim dictionaries
            
        Returns:
            List of prediction dictionaries (same format as predict_async())
        """
        if len(images) != len(claims):
            raise ValueError("Images and claims must have same length")
        
        try:
            # Ensure client is available
            await self._get_client()
            
            # Create tasks for concurrent execution
            tasks = [
                self.predict_async(image, claim)
                for image, claim in zip(images, claims)
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            predictions = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    predictions.append({
                        "prediction": None,
                        "confidence": None,
                        "metadata": {
                            "error": str(result),
                            "error_type": type(result).__name__,
                            "index": i
                        }
                    })
                else:
                    predictions.append(result)
            
            return predictions
            
        finally:
            # Note: Client remains open for potential reuse
            pass
    
    def predict_batch(
        self, 
        images: List[Image.Image], 
        claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predict baseline answers for a batch of image-claim pairs (sync wrapper).
        
        Args:
            images: List of PIL Image objects
            claims: List of claim dictionaries
            
        Returns:
            List of prediction dictionaries (same format as predict())
        """
        return asyncio.run(self._predict_batch_with_context(images, claims))
    
    async def _predict_batch_with_context(
        self, 
        images: List[Image.Image], 
        claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict batch with proper async context management."""
        return await self.predict_batch_async(images, claims)
    
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
                    # Schedule cleanup
                    asyncio.create_task(self._close_client())
                else:
                    loop.run_until_complete(self._close_client())
            except RuntimeError:
                # Create new loop for cleanup
                asyncio.run(self._close_client())
