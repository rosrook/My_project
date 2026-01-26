"""
异步API客户端 - 支持GPU绑定和异步并发
"""
import asyncio
import aiohttp
import base64
import io
import json
import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from PIL import Image
from QA_Generator.config import config
import os

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "openai package is required. Please install it with `pip install openai`."
    ) from e


class AsyncGeminiClient:
    """异步API客户端，支持高并发和GPU绑定"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 base_url: Optional[str] = None, gpu_id: Optional[int] = None,
                 max_concurrent: int = 10, request_delay: float = 0.1,
                 service_name: Optional[str] = None, env: Optional[str] = None,
                 use_lb_client: bool = False):
        """
        初始化异步客户端

        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础URL（如果使用LBOpenAIAsyncClient则不需要）
            gpu_id: 绑定的GPU ID（用于进程隔离，不影响API调用）
            max_concurrent: 最大并发请求数（建议1-5，某些API不支持高并发）
            request_delay: 每个请求之间的延迟（秒），用于避免触发API并发限制
            service_name: 服务名称（用于LBOpenAIAsyncClient，如果为None则从config读取）
            env: 环境（用于LBOpenAIAsyncClient，如果为None则从config读取）
            use_lb_client: 是否使用LBOpenAIAsyncClient（推荐，与vlmtool/generate_vqa一致）
        """
        self.api_key = api_key or getattr(config, "OPENAI_API_KEY", None) or config.API_KEY
        raw_model_name = model_name or config.MODEL_NAME
        # Normalize model name: remove /workspace/ prefix but keep original case
        self.model_name = self._normalize_model_name(raw_model_name) if raw_model_name else None
        self.base_url = base_url or getattr(config, "OPENAI_BASE_URL", None) or config.BASE_URL
        self.service_name = service_name or getattr(config, 'SERVICE_NAME', None)
        self.env = env or getattr(config, 'ENV', 'prod')
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent
        self.use_lb_client = False

        if not self.base_url:
            raise ValueError("OpenAI Base URL未设置")

        self.client = AsyncOpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
        )
        self.session: Optional[aiohttp.ClientSession] = None

        # 设置GPU可见性（用于进程隔离）
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[INFO] GPU绑定: GPU {gpu_id}")

        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 请求间隔控制（避免触发API并发限制）
        # 某些API可能不支持高并发，需要添加小延迟
        self.request_delay = request_delay

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if hasattr(self.client, "close"):
            close_method = getattr(self.client, "close")
            if asyncio.iscoroutinefunction(close_method):
                await close_method()
            else:
                close_method()

    def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """编码图片为base64，自动压缩过大图片"""
        try:
            image = self._load_image(image_input)

            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image.close()
                image = rgb_image
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # 检查图片尺寸，如果太大则压缩（避免 base64 过大导致 400 错误）
            max_size = 2048  # 最大边长
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # 尝试不同的质量级别，确保 base64 不会太大
            quality = 85
            for attempt in range(3):
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality, optimize=True)
                buffer.seek(0)
                image_data = buffer.read()
                buffer.close()

                # 检查 base64 大小（通常 base64 比原始数据大约 33%）
                # 某些 API 可能限制请求体大小，这里限制在 10MB 以内
                if len(image_data) <= 7 * 1024 * 1024:  # 7MB 原始数据 ≈ 9.3MB base64
                    break

                # 如果太大，降低质量
                quality = max(50, quality - 15)

            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # 最终检查：如果 base64 仍然太大，抛出错误
            if len(image_base64) > 10 * 1024 * 1024:  # 10MB base64
                raise ValueError(
                    f"图片 base64 编码后仍然过大: {len(image_base64) / 1024 / 1024:.2f}MB "
                    f"(原始尺寸: {image.size}, 压缩后质量: {quality})"
                )

            return image_base64
        finally:
            if 'image' in locals() and image is not None:
                try:
                    image.close()
                except:
                    pass

    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """加载图片"""
        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))

        if isinstance(image_input, (str, Path)):
            image_str = str(image_input)

            if image_str.startswith(('http://', 'https://')):
                # URL需要异步下载，这里先不支持
                raise ValueError("URL图片需要异步下载，请使用load_image_async")

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

        raise ValueError(f"不支持的图片类型: {type(image_input)}")

    def _normalize_model_name(self, model_name: str) -> str:
        """
        规范化模型名称，转换为 API 期望的格式

        Args:
            model_name: 原始模型名称（可能包含路径前缀、大小写混合等）

        Returns:
            规范化后的模型名称（无路径前缀，保持原始大小写）
        """
        if not model_name:
            return model_name

        # 移除路径前缀（如 /workspace/）
        normalized = model_name.strip()
        if normalized.startswith('/workspace/'):
            normalized = normalized[len('/workspace/'):]
        elif normalized.startswith('/'):
            normalized = normalized[1:]

        # 移除末尾的斜杠
        normalized = normalized.rstrip('/')

        # 保持原始大小写（与新调用方式一致：model="Qwen3-VL-235B-A22B-Instruct"）
        # 不再转换为全小写

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
        异步分析图片

        Args:
            image_input: 图片输入
            prompt: 提示词
            temperature: 温度参数
            retry_on_401: 是否在401错误时重试（某些API在并发时可能误报401）
            max_retries: 最大重试次数

        Returns:
            响应文本
        """
        async with self.semaphore:  # 限制并发数
            if self.request_delay > 0:
                await asyncio.sleep(self.request_delay)

            image_base64 = self._encode_image(image_input)
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

            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=4096,  # Use max_tokens (standard OpenAI parameter)
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 0.5
                        print(f"[WARNING] 请求失败，等待 {wait_time:.1f}s 后重试 ({attempt + 1}/{max_retries})... {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        break
            raise last_exception or Exception("请求失败：所有重试都失败，但未捕获到异常")

    # ==================== OpenAI 兼容接口 ====================

    class _Completions:
        """OpenAI兼容的completions接口"""
        def __init__(self, parent_client):
            self._parent = parent_client

        async def create(
            self,
            model: str,
            messages: list,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            max_completion_tokens: Optional[int] = None,
            stream: bool = False,
            response_format: Optional[Dict] = None,
            **kwargs
        ):
            """
            OpenAI兼容的chat.completions.create接口（参考vlmtool/generate_vqa的成功实践）

            Args:
                model: 模型名称
                messages: 消息列表，格式: [{"role": "user", "content": [...]}]
                temperature: 温度参数
                max_tokens: 最大token数
                stream: 是否流式输出（暂不支持）
                response_format: 响应格式（如 {"type": "json_object"}）
                **kwargs: 其他参数

            Returns:
                类似OpenAI API的响应对象
            """
            if stream:
                raise NotImplementedError("流式输出暂不支持")
            async with self._parent.semaphore:
                if self._parent.request_delay > 0:
                    await asyncio.sleep(self._parent.request_delay)
                
                # Support both max_tokens and max_completion_tokens for backward compatibility
                # Prefer max_completion_tokens if provided, otherwise use max_tokens
                final_max_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
                
                # Normalize model name (remove /workspace/ prefix)
                normalized_model = self._parent._normalize_model_name(model)
                
                return await self._parent.client.chat.completions.create(
                    model=normalized_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=final_max_tokens,  # Use max_tokens (standard OpenAI parameter)
                    **kwargs
                )

    class _Chat:
        """OpenAI兼容的chat接口"""
        def __init__(self, parent_client):
            self._parent = parent_client
            self.completions = AsyncGeminiClient._Completions(parent_client)

    @property
    def chat(self):
        """返回chat.completions接口（与LBOpenAIAsyncClient兼容）"""
        return self._Chat(self)

    async def filter_image_async(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        criteria_description: str,
        question: str,
        temperature: float = 0.3
    ) -> dict:
        """
        异步筛选图片

        Args:
            image_input: 图片输入
            criteria_description: 筛选标准描述
            question: 问题描述

        Returns:
            筛选结果字典
        """
        prompt = """
You are a professional image filtering and quality evaluation expert.

Your task is to evaluate whether an image meets the required criteria, and to assign a quality score based on how well it satisfies both required and optional standards.

Question Type:
{question}

Evaluation Criteria:
{criteria_description}

Please follow the rules below strictly.


1. Required (non-optional) Criteria:
- These are mandatory criteria.
- ALL required criteria must be satisfied for the image to pass.
- If ANY required criterion is not satisfied:
  - "passed" must be false
  - "score" must be 0.0
  - Optional criteria must NOT be considered
- If ALL required criteria are satisfied:
  - "passed" is true
  - The score starts from a base score of 0.1
  - The quality score for required criteria ranges from 0.1 to 0.6
  - The closer the image matches the required criteria (clarity, correctness, completeness, alignment),
    the closer the score should be to 0.6

2. Optional Criteria:
- Optional criteria are marked explicitly as "optional" in the criteria.
- Optional criteria are considered ONLY IF all required criteria are satisfied.
- Optional criteria represent higher difficulty and higher value.

Scoring with Optional Criteria:
- If there are NO optional criteria:
  - Automatically add 0.4 to the score (i.e., total score = required score + 0.4)
- If there ARE optional criteria:
  - A bonus score up to 0.4 is available
  - The more optional criteria are satisfied, and the better they are satisfied,
    the higher the bonus score (from 0.0 to 0.4)

3. Final Score:
- Total score = required score (max 0.6) + optional bonus (max 0.4)
- Final score must be in the range [0.0, 1.0]
- Images satisfying optional criteria well should score higher than those that only satisfy required criteria.

4. Confidence:
- "confidence" represents how confident you are in your judgment
- It must be a float between 0.0 and 1.0
- Confidence should be lower if the image is ambiguous or borderline


Return the result in JSON format ONLY, with no extra text:

{{
  "passed": true/false,
  "basic_score": float,
  "bonus_score": float (0.0-0.4),
  "total_score": float (0.0–1.0),
  "reason": "Detailed explanation of which required and optional criteria are satisfied or violated, and how the score is determined",
  "confidence": float (0.0–1.0)
}}""".format(question=question, criteria_description=criteria_description)

        try:
            response_text = await self.analyze_image_async(image_input, prompt, temperature)

            # 解析JSON响应
            json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_block_match:
                response_text = json_block_match.group(1)
            else:
                start_idx = response_text.find('{')
                if start_idx != -1:
                    brace_count = 0
                    for i in range(start_idx, len(response_text)):
                        if response_text[i] == '{':
                            brace_count += 1
                        elif response_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0 and '"passed"' in response_text[start_idx:i+1]:
                                response_text = response_text[start_idx:i+1]
                                break

            result = json.loads(response_text)

            # 验证结果格式
            if "passed" not in result:
                result["passed"] = False
            if "reason" not in result:
                result["reason"] = "无法解析筛选结果"
            if "confidence" not in result:
                result["confidence"] = 0.5

            return result

        except Exception as e:
            return {
                "passed": False,
                "reason": f"筛选过程出错: {str(e)}",
                "confidence": 0.0
            }


async def process_batch_async(
    items: List[Dict[str, Any]],
    num_gpus: int = 8,
    max_concurrent_per_gpu: int = 10
) -> List[Dict[str, Any]]:
    """
    使用多GPU异步处理批量数据

    Args:
        items: 待处理的数据项列表
        num_gpus: GPU数量
        max_concurrent_per_gpu: 每个GPU的最大并发数

    Returns:
        处理结果列表
    """
    # 将任务分配到不同的GPU
    tasks_per_gpu = len(items) // num_gpus
    gpu_tasks = []

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * tasks_per_gpu
        if gpu_id == num_gpus - 1:
            end_idx = len(items)  # 最后一个GPU处理剩余所有任务
        else:
            end_idx = (gpu_id + 1) * tasks_per_gpu

        gpu_tasks.append((gpu_id, items[start_idx:end_idx]))

    # 为每个GPU创建处理任务
    async def process_gpu_tasks(gpu_id: int, tasks: List[Dict]):
        """处理单个GPU的任务"""
        results = []
        async with AsyncGeminiClient(
            gpu_id=gpu_id,
            max_concurrent=max_concurrent_per_gpu
        ) as client:
            # 创建所有异步任务
            async_tasks = []
            for item in tasks:
                # 这里需要根据实际需求调用相应的异步方法
                # 示例：假设item包含image_input, criteria_description, question
                task = client.filter_image_async(
                    image_input=item.get("image_input"),
                    criteria_description=item.get("criteria_description", ""),
                    question=item.get("question", ""),
                    temperature=0.3
                )
                async_tasks.append(task)

            # 等待所有任务完成
            results = await asyncio.gather(*async_tasks, return_exceptions=True)

            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "error": str(result),
                        "item": tasks[i]
                    })
                else:
                    processed_results.append(result)

        return processed_results

    # 并发处理所有GPU的任务
    all_results = await asyncio.gather(*[
        process_gpu_tasks(gpu_id, tasks)
        for gpu_id, tasks in gpu_tasks
    ])

    # 合并结果
    final_results = []
    for gpu_results in all_results:
        final_results.extend(gpu_results)

    return final_results
