"""
Async API Client - Supports GPU binding and async concurrency.

This module provides AsyncGeminiClient for asynchronous model API calls with:
- GPU binding for process isolation
- Concurrent request control via Semaphore
- Support for LBOpenAIAsyncClient and AsyncOpenAI implementation
- Automatic image compression and base64 encoding
- OpenAI-compatible interface (chat.completions.create)
"""

import asyncio
import base64
import io
import json
import re
import os
import warnings
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from PIL import Image

# Try to import AsyncOpenAI
try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "openai package is required. Please install it with `pip install openai`."
    ) from e

# Try to import LBOpenAIAsyncClient (if available)
try:
    from redeuler.client.openai import LBOpenAIAsyncClient
    HAS_LB_CLIENT = True
except ImportError:
    HAS_LB_CLIENT = False
    LBOpenAIAsyncClient = None

# Import framework config
try:
    from ProbingFactorGeneration.config import MODEL_CONFIG
except ImportError:
        # Minimal fallback config
        MODEL_CONFIG = {
            "MODEL_NAME": None,
            "SERVICE_NAME": None,
            "ENV": "prod",
            "API_KEY": None,
            "BASE_URL": None,
            "MAX_CONCURRENT": 10,
            "REQUEST_DELAY": 0.1,
            "USE_LB_CLIENT": True,
        }


class AsyncGeminiClient:
    """
    Async API client supporting high concurrency and GPU binding.
    
    This client provides an OpenAI-compatible interface for calling vision-language models
    asynchronously with controlled concurrency and GPU isolation.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None, 
        base_url: Optional[str] = None, 
        gpu_id: Optional[int] = None,
        max_concurrent: int = None, 
        request_delay: float = None,
        service_name: Optional[str] = None, 
        env: Optional[str] = None,
        use_lb_client: Optional[bool] = None
    ):
        """
        Initialize async client.
        
        Args:
            api_key: API key (default: from MODEL_CONFIG or environment)
            model_name: Model name (default: from MODEL_CONFIG)
            base_url: API base URL (required if not using LBOpenAIAsyncClient, should include /v1, e.g., http://10.158.144.81:8000/v1)
            gpu_id: GPU ID for process isolation (doesn't affect API calls)
            max_concurrent: Maximum concurrent requests (default: from MODEL_CONFIG)
            request_delay: Delay between requests in seconds (default: from MODEL_CONFIG)
            service_name: Service name for LBOpenAIAsyncClient (default: from MODEL_CONFIG)
            env: Environment for LBOpenAIAsyncClient (default: from MODEL_CONFIG)
            use_lb_client: Whether to use LBOpenAIAsyncClient (default: from MODEL_CONFIG)
        """
        # Get config values with fallback to MODEL_CONFIG
        self.api_key = api_key or MODEL_CONFIG.get("API_KEY") or os.getenv("API_KEY")
        self.model_name = model_name or MODEL_CONFIG.get("MODEL_NAME")
        self.base_url = base_url or MODEL_CONFIG.get("BASE_URL")
        self.service_name = service_name or MODEL_CONFIG.get("SERVICE_NAME")
        self.env = env or MODEL_CONFIG.get("ENV", "prod")
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent or MODEL_CONFIG.get("MAX_CONCURRENT", 10)
        self.request_delay = request_delay or MODEL_CONFIG.get("REQUEST_DELAY", 0.1)
        self.use_lb_client = (use_lb_client if use_lb_client is not None 
                            else MODEL_CONFIG.get("USE_LB_CLIENT", True)) and HAS_LB_CLIENT
        
        # Initialize LBOpenAIAsyncClient if enabled
        if self.use_lb_client:
            if not self.service_name:
                raise ValueError(
                    "Service Name not set. Please set SERVICE_NAME in MODEL_CONFIG "
                    "or pass service_name parameter."
                )
            self.lb_client = LBOpenAIAsyncClient(
                service_name=self.service_name,
                env=self.env,
                api_key=self.api_key or "1"
            )
            self.client = None
            self._lb_client_needs_close = True
        else:
            # Use AsyncOpenAI (same as QA_Generator)
            if not self.api_key or self.api_key.strip() == "":
                raise ValueError(
                    "API Key not set or invalid!\n"
                    "Please set API_KEY in MODEL_CONFIG or pass api_key parameter, "
                    "or set environment variable: export API_KEY='your-api-key'"
                )
            
            if not self.base_url:
                raise ValueError(
                    "Base URL not set. Please set BASE_URL in MODEL_CONFIG "
                    "or pass base_url parameter."
                )
            
            self.client = AsyncOpenAI(
                api_key=self.api_key or "EMPTY",
                base_url=self.base_url,
            )
        
        # Set GPU visibility (for process isolation)
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if os.getenv("VERBOSE", "false").lower() == "true":
                print(f"[INFO] GPU binding: GPU {gpu_id}")
        
        # Semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.use_lb_client:
            # LBOpenAIAsyncClient manages its own session
            return self
        else:
            # AsyncOpenAI client is already initialized in __init__
            return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - clean up resources."""
        if self.use_lb_client:
            # Close LBOpenAIAsyncClient
            try:
                # Try close() method first (if available)
                if hasattr(self.lb_client, 'close'):
                    close_method = getattr(self.lb_client, 'close')
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                
                # Also check if lb_client itself is a context manager
                if hasattr(self.lb_client, '__aexit__'):
                    try:
                        await self.lb_client.__aexit__(None, None, None)
                    except Exception:
                        pass  # Ignore errors in context manager exit
            except Exception as e:
                warnings.warn(f"Warning when closing LBOpenAIAsyncClient: {e}", RuntimeWarning)
        else:
            # Close AsyncOpenAI client
            if self.client is not None:
                try:
                    if hasattr(self.client, "close"):
                        close_method = getattr(self.client, "close")
                        if asyncio.iscoroutinefunction(close_method):
                            await close_method()
                        else:
                            close_method()
                except Exception as e:
                    warnings.warn(f"Warning when closing AsyncOpenAI client: {e}", RuntimeWarning)
    
    def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """
        Encode image to base64, automatically compressing if too large.
        
        Args:
            image_input: Image input (PIL Image, file path, bytes, or base64 string)
            
        Returns:
            Base64-encoded image string (without data URI prefix)
        """
        try:
            image = self._load_image(image_input)
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image.close()
                image = rgb_image
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Check image size and compress if too large
            max_size = 2048  # Maximum dimension
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Try different quality levels to ensure base64 isn't too large
            quality = 85
            for attempt in range(3):
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality, optimize=True)
                buffer.seek(0)
                image_data = buffer.read()
                buffer.close()
                
                # Check size (base64 is ~33% larger than original)
                # Limit to 7MB raw data ≈ 9.3MB base64
                if len(image_data) <= 7 * 1024 * 1024:
                    break
                
                # Reduce quality if too large
                quality = max(50, quality - 15)
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Final check: if base64 is still too large, raise error
            if len(image_base64) > 10 * 1024 * 1024:  # 10MB base64
                raise ValueError(
                    f"Image base64 encoding still too large: {len(image_base64) / 1024 / 1024:.2f}MB "
                    f"(original size: {image.size}, compressed quality: {quality})"
                )
            
            return image_base64
        finally:
            if 'image' in locals() and image is not None:
                try:
                    image.close()
                except:
                    pass
    
    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """
        Load image from various input types.
        
        Args:
            image_input: Image input (PIL Image, file path, bytes, or base64 string)
            
        Returns:
            PIL Image object
        """
        if isinstance(image_input, Image.Image):
            return image_input
        
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        
        if isinstance(image_input, (str, Path)):
            image_str = str(image_input)
            
            if image_str.startswith(('http://', 'https://')):
                raise ValueError("URL images require async download. Use load_image_async.")
            
            # Check if it's a base64 string
            if image_str.startswith('data:image'):
                base64_data = image_str.split(',', 1)[1]
            elif len(image_str) > 100:
                try:
                    clean_str = re.sub(r'\s', '', image_str)
                    base64.b64decode(clean_str)
                    base64_data = clean_str
                except:
                    base64_data = None
            else:
                base64_data = None
            
            if base64_data:
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes))
            else:
                return Image.open(image_input)
        
        raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name to API-expected format.
        
        Args:
            model_name: Original model name (may contain path prefix, mixed case)
            
        Returns:
            Normalized model name (no path prefix; optional lowercase)
        """
        if not model_name:
            return model_name
        
        # Remove path prefix (e.g., /workspace/)
        normalized = model_name.strip()
        if normalized.startswith('/workspace/'):
            normalized = normalized[len('/workspace/'):]
        elif normalized.startswith('/'):
            normalized = normalized[1:]
        
        # Remove trailing slash
        normalized = normalized.rstrip('/')
        
        # Convert to lowercase only if explicitly enabled
        if os.getenv("LOWERCASE_MODEL_NAME", "false").lower() == "true":
            normalized = normalized.lower()
        
        return normalized
    
    async def analyze_image_async(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        prompt: str,
        temperature: float = 0.7,
        retry_on_401: bool = True,
        max_retries: int = 2,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Async image analysis.
        
        Args:
            image_input: Image input
            prompt: Prompt text
            temperature: Temperature parameter
            retry_on_401: Whether to retry on 401 errors (some APIs may false positive on concurrency)
            max_retries: Maximum retry attempts
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            Response text
        """
        async with self.semaphore:  # Control concurrency
            if not self.use_lb_client and not self.client:
                raise RuntimeError("Client not initialized. Use async with statement.")
            
            # Add request delay to avoid triggering API concurrency limits
            if self.request_delay > 0:
                await asyncio.sleep(self.request_delay)
            
            # Encode image
            image_base64 = self._encode_image(image_input)
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ]
            
            # Send request with retry mechanism
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    if self.use_lb_client:
                        # Use LBOpenAIAsyncClient
                        response = await self.lb_client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=4096,
                        )
                    else:
                        # Use AsyncOpenAI (same as QA_Generator)
                        response = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_completion_tokens=4096,
                        )
                    
                    return response.choices[0].message.content
                except Exception as e:
                    last_exception = e
                    # Check if it's a 401 error and retry is enabled
                    error_str = str(e)
                    is_401 = "401" in error_str or "Unauthorized" in error_str
                    
                    if is_401 and retry_on_401 and attempt < max_retries:
                        wait_time = (attempt + 1) * 0.5
                        print(f"[WARNING] 401 error, possibly concurrency limit, retrying after {wait_time:.1f}s ({attempt + 1}/{max_retries})... {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    elif attempt < max_retries:
                        wait_time = (attempt + 1) * 0.5
                        print(f"[WARNING] 请求失败，等待 {wait_time:.1f}s 后重试 ({attempt + 1}/{max_retries})... {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        break
            
            # If all retries failed, raise with detailed error message
            if last_exception:
                error_msg = f"Request failed: {str(last_exception)}"
                error_msg += f"\nModel: {self.model_name}"
                error_msg += f"\nBase URL: {self.base_url}"
                error_msg += f"\nMax concurrent: {self.max_concurrent}, Request delay: {self.request_delay}s"
                raise Exception(error_msg) from last_exception
            else:
                raise Exception("Request failed: all retries exhausted but no exception caught")
    
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
            max_tokens: int = 4096,
            stream: bool = False,
            response_format: Optional[Dict] = None,
            **kwargs
        ):
            """
            OpenAI-compatible chat.completions.create interface.
            
            Args:
                model: Model name
                messages: Message list, format: [{"role": "user", "content": [...]}]
                temperature: Temperature parameter
                max_tokens: Maximum token count
                stream: Whether to stream output (not supported yet)
                response_format: Response format (e.g., {"type": "json_object"})
                **kwargs: Other parameters
            
            Returns:
                OpenAI-like response object
            """
            if stream:
                raise NotImplementedError("Streaming output not supported yet")
            
            async with self._parent.semaphore:  # Control concurrency
                if self._parent.request_delay > 0:
                    await asyncio.sleep(self._parent.request_delay)
                
                # Use max_completion_tokens if provided, otherwise use max_tokens
                max_completion_tokens = kwargs.pop("max_completion_tokens", None)
                if max_completion_tokens is None:
                    max_completion_tokens = max_tokens
                
                # If using LBOpenAIAsyncClient, delegate to it
                if self._parent.use_lb_client:
                    return await self._parent.lb_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_completion_tokens,
                        response_format=response_format,
                        **kwargs
                    )
                
                # Use AsyncOpenAI (same as QA_Generator)
                if not self._parent.client:
                    raise RuntimeError("Client not initialized. Use async with statement.")
                
                return await self._parent.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    response_format=response_format,
                    **kwargs
                )
    
    class _Chat:
        """OpenAI-compatible chat interface"""
        def __init__(self, parent_client):
            self._parent = parent_client
            self.completions = AsyncGeminiClient._Completions(parent_client)
    
    @property
    def chat(self):
        """Return chat.completions interface (compatible with LBOpenAIAsyncClient)"""
        return self._Chat(self)
