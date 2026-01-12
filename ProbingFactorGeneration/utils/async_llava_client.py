"""
Async LLaVA Client - Local model inference with async interface.

This module provides AsyncLLaVAClient for running local LLaVA models with:
- OpenAI-compatible interface (chat.completions.create)
- Async inference using asyncio.to_thread
- GPU binding for process isolation
- Concurrent request control via Semaphore
- Support for JSON response format
"""

import asyncio
import base64
import io
import json
import os
import re
from typing import Union, Optional, List, Dict, Any
from PIL import Image
import torch

# Try to import transformers
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    # Also try LLaVA-specific classes if available
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    except ImportError:
        LlavaNextProcessor = None
        LlavaNextForConditionalGeneration = None
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoProcessor = None
    AutoModelForCausalLM = None
    LlavaNextProcessor = None
    LlavaNextForConditionalGeneration = None

# Import framework config
try:
    from ..config import MODEL_CONFIG
except ImportError:
    try:
        from ProbingFactorGeneration.config import MODEL_CONFIG
    except ImportError:
        MODEL_CONFIG = {
            "MODEL_NAME": None,
            "MAX_CONCURRENT": 1,  # Local models typically need lower concurrency
            "REQUEST_DELAY": 0.0,
        }


class AsyncLLaVAClient:
    """
    Async client for local LLaVA model inference.
    
    This client provides an OpenAI-compatible interface for calling local LLaVA models
    asynchronously with controlled concurrency and GPU isolation.
    """
    
    def __init__(
        self,
        model_path: str,
        gpu_id: Optional[int] = None,
        max_concurrent: int = None,
        request_delay: float = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize async LLaVA client.
        
        Args:
            model_path: Path to the local model directory (Hugging Face format)
            gpu_id: GPU ID for process isolation (sets CUDA_VISIBLE_DEVICES)
            max_concurrent: Maximum concurrent requests (default: 1 for local models)
            request_delay: Delay between requests in seconds (default: 0.0)
            device: Device to use ("cuda" or "cpu", default: "cuda" if available)
            torch_dtype: Data type for model (default: torch.float16 for CUDA, torch.float32 for CPU)
            load_in_8bit: Whether to load model in 8-bit (requires bitsandbytes)
            load_in_4bit: Whether to load model in 4-bit (requires bitsandbytes)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent or MODEL_CONFIG.get("MAX_CONCURRENT", 1)
        self.request_delay = request_delay or MODEL_CONFIG.get("REQUEST_DELAY", 0.0)
        
        # Set GPU visibility (for process isolation)
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if os.getenv("VERBOSE", "false").lower() == "true":
                print(f"[INFO] GPU binding: GPU {gpu_id}")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Determine dtype
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Model and processor will be loaded lazily
        self.processor = None
        self.model = None
        self._model_loaded = False
        
        # Semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    def _load_model(self):
        """Load model and processor (synchronous, called from async context)."""
        if self._model_loaded:
            return
        
        if os.getenv("VERBOSE", "false").lower() == "true":
            print(f"[INFO] Loading model from {self.model_path}")
        
        # Load processor - use AutoProcessor for compatibility
        try:
            # Try LLaVA Next processor first (if available)
            if LlavaNextProcessor is not None:
                self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
            else:
                # Fall back to AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as e:
            # Fall back to AutoProcessor if specific processor fails
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        # Add quantization options if requested
        if self.load_in_8bit or self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install with: pip install bitsandbytes"
                )
        
        # Load model - use AutoModelForCausalLM for compatibility
        try:
            # Try LLaVA Next model first (if available)
            if LlavaNextForConditionalGeneration is not None:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            else:
                # Fall back to AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
        except Exception as e:
            # Fall back to AutoModelForCausalLM if specific model class fails
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self._model_loaded = True
        
        if os.getenv("VERBOSE", "false").lower() == "true":
            print(f"[INFO] Model loaded on {self.device}")
    
    async def __aenter__(self):
        """Async context manager entry - load model in background thread."""
        # Load model in background thread to avoid blocking event loop
        await asyncio.to_thread(self._load_model)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup."""
        # Clear model from memory
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._model_loaded = False
        
        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def _decode_base64_image(self, image_data: str) -> Image.Image:
        """Decode base64 image string to PIL Image."""
        # Handle data URL format: data:image/jpeg;base64,xxxxx
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def _generate_response(
        self,
        text: str,
        image: Image.Image,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Generate response from model (synchronous, runs in thread).
        
        Args:
            text: Input text prompt
            image: PIL Image object
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format (e.g., {"type": "json_object"})
        
        Returns:
            Generated text response
        """
        if not self._model_loaded:
            self._load_model()
        
        # Prepare prompt
        # LLaVA models typically use a chat template
        # Format: "USER: <image>\n{text}\nASSISTANT:"
        prompt = f"USER: <image>\n{text}\nASSISTANT:"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            # Adjust generation parameters based on response format
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else 1.0,
                "do_sample": temperature > 0,
            }
            
            # If JSON format requested, add instruction to prompt
            if response_format and response_format.get("type") == "json_object":
                # Modify prompt to request JSON output
                json_prompt = f"{prompt}\nPlease respond in valid JSON format only."
                inputs = self.processor(
                    text=json_prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode output
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (everything after "ASSISTANT:")
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text.strip()
        
        # If JSON format requested, try to extract JSON from response
        if response_format and response_format.get("type") == "json_object":
            # Try to extract JSON object from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            # Validate JSON
            try:
                json.loads(response)  # Validate
            except json.JSONDecodeError:
                # If not valid JSON, wrap the response in a JSON object
                response = json.dumps({"response": response})
        
        return response
    
    async def analyze_image_async(
        self,
        image_input: Union[str, Image.Image],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Analyze image with text prompt (async).
        
        Args:
            image_input: Base64 image string or PIL Image
            prompt: Text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format (e.g., {"type": "json_object"})
        
        Returns:
            Generated text response
        """
        async with self.semaphore:
            if self.request_delay > 0:
                await asyncio.sleep(self.request_delay)
            
            # Convert image input to PIL Image if needed
            if isinstance(image_input, str):
                image = self._decode_base64_image(image_input)
            else:
                image = image_input
            
            # Run generation in thread to avoid blocking event loop
            response = await asyncio.to_thread(
                self._generate_response,
                prompt,
                image,
                temperature,
                max_tokens,
                response_format
            )
            
            return response
    
    # ==================== OpenAI Compatible Interface ====================
    
    class _Completions:
        """OpenAI-compatible completions interface"""
        def __init__(self, parent_client):
            self._parent = parent_client
        
        async def create(
            self,
            model: str,
            messages: list,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            stream: bool = False,
            response_format: Optional[Dict] = None,
            **kwargs
        ):
            """
            OpenAI-compatible chat.completions.create interface.
            
            Args:
                model: Model name (ignored, uses loaded model)
                messages: Message list, format: [{"role": "user", "content": [...]}]
                temperature: Temperature parameter
                max_tokens: Maximum token count
                stream: Whether to stream output (not supported)
                response_format: Response format (e.g., {"type": "json_object"})
                **kwargs: Other parameters (ignored)
            
            Returns:
                OpenAI-like response object
            """
            if stream:
                raise NotImplementedError("Streaming output not supported yet")
            
            if not self._parent._model_loaded:
                await self._parent.__aenter__()
            
            # Extract text and image from messages
            text_content = None
            image_input = None
            
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        # Simple string format (no image)
                        text_content = content
                    elif isinstance(content, list):
                        # List format with text and image
                        for item in content:
                            if item.get("type") == "text":
                                text_content = item.get("text", "")
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                # Extract base64 part or use full URL
                                if image_url.startswith("data:image"):
                                    # Format: data:image/jpeg;base64,xxxxx
                                    image_input = image_url.split(",", 1)[1]
                                else:
                                    # May be pure base64 string
                                    image_input = image_url
            
            if not text_content:
                raise ValueError("Messages must contain text content")
            
            if not image_input:
                raise ValueError("LLaVA model requires image input")
            
            # Call analyze_image_async
            response_text = await self._parent.analyze_image_async(
                image_input=image_input,
                prompt=text_content,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            # Build OpenAI-like response object
            class _Response:
                def __init__(self, content):
                    class _Choice:
                        def __init__(self, content):
                            class _Message:
                                def __init__(self, content):
                                    self.content = content
                            self.message = _Message(content)
                            self.finish_reason = "stop"
                            self.index = 0
                    self.choices = [_Choice(content)]
                    self.model = model
                    self.object = "chat.completion"
                    # Rough token estimates (4 chars per token)
                    self.usage = type('Usage', (), {
                        "prompt_tokens": len(text_content) // 4,
                        "completion_tokens": len(content) // 4,
                        "total_tokens": (len(text_content) + len(content)) // 4
                    })()
            
            return _Response(response_text)
    
    class _Chat:
        """OpenAI-compatible chat interface"""
        def __init__(self, parent_client):
            self._parent = parent_client
            self.completions = AsyncLLaVAClient._Completions(parent_client)
    
    @property
    def chat(self):
        """Return chat.completions interface (compatible with OpenAI API)"""
        return self._Chat(self)
