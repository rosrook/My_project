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
from tempfile import NamedTemporaryFile
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
except ImportError as e:
    HAS_TRANSFORMERS = False
    AutoProcessor = None
    AutoModelForCausalLM = None
    LlavaNextProcessor = None
    LlavaNextForConditionalGeneration = None
    _TRANSFORMERS_IMPORT_ERROR = e
except Exception as e:
    # Catch any other import errors (e.g., missing dependencies)
    HAS_TRANSFORMERS = False
    AutoProcessor = None
    AutoModelForCausalLM = None
    LlavaNextProcessor = None
    LlavaNextForConditionalGeneration = None
    _TRANSFORMERS_IMPORT_ERROR = e
else:
    _TRANSFORMERS_IMPORT_ERROR = None

# Import framework config
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
            error_msg = "transformers library is required. Install with: pip install transformers"
            if '_TRANSFORMERS_IMPORT_ERROR' in globals() and _TRANSFORMERS_IMPORT_ERROR is not None:
                error_msg += f"\nImport error: {_TRANSFORMERS_IMPORT_ERROR}"
            raise ImportError(error_msg)
        
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
        
        # Check if qwen_vl_utils is installed (required for LLaVA OneVision models)
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen_vl_utils 未安装！请运行: pip install qwen-vl-utils\n"
                "这是 LLaVA-OneVision / Qwen2-VL 模型的必需依赖。"
            )
        
        # Load model - use AutoModelForCausalLM + trust_remote_code=True (按照 eval_vqa_hardness 的方式)
        use_device_map = "auto" if self.device == "cuda" else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",  # 使用 "auto" 让 transformers 自动选择
            device_map=use_device_map,
            trust_remote_code=True  # 必须添加
        )
        
        # 如果未使用device_map，手动移动到指定设备
        if use_device_map is None and self.device != "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Load processor - use AutoProcessor + trust_remote_code=True + max_pixels/min_pixels
        max_pixels = 3240000
        min_pixels = 200704
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,  # 必须添加
            max_pixels=max_pixels,   # 必须添加
            min_pixels=min_pixels    # 必须添加
        )
        
        # 保存 qwen_vl_utils 的引用供后续使用
        self.process_vision_info = process_vision_info
        
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
        
        使用 qwen_vl_utils 的方式（按照 eval_vqa_hardness/vqa_evaluator.py 的实现）。
        
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
        
        # 使用 qwen_vl_utils 的方式（按照 eval_vqa_hardness）
        temp_image_path = None
        try:
            # 步骤1: 保存图片为临时文件（qwen_vl_utils 需要文件路径或 URL）
            temp_file = NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_image_path = temp_file.name
            image.save(temp_image_path, 'JPEG')
            temp_file.close()
            
            # 确保路径格式正确（添加 file:// 前缀，按照 vlmevalkit 的 ensure_image_url）
            if not temp_image_path.startswith(('http://', 'https://', 'file://', 'data:image')):
                image_url = f"file://{temp_image_path}"
            else:
                image_url = temp_image_path
            
            # 步骤2: 构建消息（完全按照 vlmevalkit LLaVA_OneVision_1_5.generate_inner 的格式）
            content_list = [
                {"type": "image", "image": image_url},
                {"type": "text", "text": text}
            ]
            messages = [
                {"role": "user", "content": content_list}
            ]
            
            # 步骤3: 使用 processor.apply_chat_template（完全按照 vlmevalkit）
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 如果 JSON 格式请求，修改 prompt
            if response_format and response_format.get("type") == "json_object":
                # 在消息中添加 JSON 格式要求
                content_list = [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": f"{text}\nPlease respond in valid JSON format only."}
                ]
                messages = [{"role": "user", "content": content_list}]
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            
            # 步骤4: 使用 qwen_vl_utils.process_vision_info 处理视觉信息（完全按照 vlmevalkit）
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            # 步骤5: 调用 processor（完全按照 vlmevalkit）
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # 步骤6: 生成答案（完全按照 vlmevalkit）
            generate_kwargs = {
                "max_new_tokens": max_tokens,
            }
            # 添加 temperature 支持（如果 > 0）
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["do_sample"] = True
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generate_kwargs)
            
            # 步骤7: Trim input_ids（只保留新生成的部分，完全按照 vlmevalkit）
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # 步骤8: 解码（完全按照 vlmevalkit）
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
            
            # 如果 JSON 格式请求，验证并提取 JSON
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
        
        finally:
            # 清理临时文件
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
        
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
