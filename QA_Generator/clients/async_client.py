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

# 尝试导入 LBOpenAIAsyncClient（如果可用）
try:
    from redeuler.client.openai import LBOpenAIAsyncClient
    HAS_LB_CLIENT = True
except ImportError:
    HAS_LB_CLIENT = False
    LBOpenAIAsyncClient = None


class AsyncGeminiClient:
    """异步API客户端，支持高并发和GPU绑定"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None,
                 base_url: Optional[str] = None, gpu_id: Optional[int] = None,
                 max_concurrent: int = 10, request_delay: float = 0.1,
                 service_name: Optional[str] = None, env: Optional[str] = None,
                 use_lb_client: bool = True):
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
        self.api_key = api_key or config.API_KEY
        self.model_name = model_name or config.MODEL_NAME
        self.base_url = base_url or config.BASE_URL
        self.service_name = service_name or getattr(config, 'SERVICE_NAME', None)
        self.env = env or getattr(config, 'ENV', 'prod')
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent
        self.use_lb_client = use_lb_client and HAS_LB_CLIENT

        # 如果使用 LBOpenAIAsyncClient，检查必需参数
        if self.use_lb_client:
            if not self.service_name:
                raise ValueError("Service Name未设置，请在config.py中设置SERVICE_NAME或在初始化时传入")
            # 初始化 LBOpenAIAsyncClient（与vlmtool/generate_vqa一致）
            # 注意：LBOpenAIAsyncClient 可能内部创建了 aiohttp.ClientSession
            self.lb_client = LBOpenAIAsyncClient(
                service_name=self.service_name,
                env=self.env,
                api_key=self.api_key or "1"
            )
            # LBOpenAIAsyncClient 内部处理会话，不需要自己创建
            self.session = None
            # 标记是否需要关闭 lb_client
            self._lb_client_needs_close = True
        else:
            # 使用自定义实现（向后兼容）
            # 检查 API Key 是否有效（只检查 None 和空字符串，允许 "1" 作为有效值）
            if not self.api_key or self.api_key.strip() == "":
                raise ValueError(
                    "API Key未设置或无效！\n"
                    "请设置环境变量 API_KEY，例如：\n"
                    "  export API_KEY='your-actual-api-key'\n"
                    "或在 .env 文件中设置：\n"
                    "  API_KEY=your-actual-api-key"
                )

            if not self.base_url:
                raise ValueError("Base URL未设置")

            # 创建会话
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
        if self.use_lb_client:
            # LBOpenAIAsyncClient 自己管理会话
            return self
        else:
            # 自定义实现：创建 aiohttp 会话
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.use_lb_client:
            # LBOpenAIAsyncClient 自己管理会话，需要正确关闭
            # 根据 vlmtool/generate_vqa 的实现，LBOpenAIAsyncClient 有 close() 方法
            try:
                # 方法1: 直接调用 close() 方法（与vlmtool一致）
                if hasattr(self.lb_client, 'close'):
                    close_method = getattr(self.lb_client, 'close')
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()

                # 方法2: 尝试关闭内部的 aiohttp session（如果存在）
                # LBOpenAIAsyncClient 可能内部创建了 aiohttp.ClientSession
                for attr_name in ['_client', 'client', '_session', 'session', '_http_client']:
                    if hasattr(self.lb_client, attr_name):
                        inner_obj = getattr(self.lb_client, attr_name)
                        if inner_obj is not None:
                            # 如果是 aiohttp.ClientSession，需要关闭
                            if isinstance(inner_obj, aiohttp.ClientSession):
                                if not inner_obj.closed:
                                    await inner_obj.close()
                                    await asyncio.sleep(0.1)  # 等待连接完全关闭
                                    break
                            # 如果有 close 方法
                            elif hasattr(inner_obj, 'close'):
                                close_method = getattr(inner_obj, 'close')
                                if asyncio.iscoroutinefunction(close_method):
                                    await close_method()
                                else:
                                    close_method()
                                break
            except Exception as e:
                # 记录警告但不抛出异常，避免影响正常流程
                import warnings
                warnings.warn(f"关闭 LBOpenAIAsyncClient 时出现警告: {e}", RuntimeWarning)
        else:
            # 自定义实现：关闭 aiohttp 会话
            if self.session and not self.session.closed:
                try:
                    await self.session.close()
                    # 等待一小段时间确保连接完全关闭
                    await asyncio.sleep(0.1)
                except Exception as e:
                    import warnings
                    warnings.warn(f"关闭 aiohttp session 时出现警告: {e}", RuntimeWarning)
                finally:
                    self.session = None

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
            规范化后的模型名称（全小写，无路径前缀）
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

        # 转换为全小写（API 要求）
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
            if not self.session:
                raise RuntimeError("Session not initialized. Use async with statement.")

            # 添加请求间隔，避免触发API并发限制
            if self.request_delay > 0:
                await asyncio.sleep(self.request_delay)

            # 编码图片
            image_base64 = self._encode_image(image_input)

            # 规范化模型名称（API 需要全小写，无路径前缀）
            normalized_model_name = self._normalize_model_name(self.model_name)

            # 构建请求
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": normalized_model_name,
                "messages": [
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
                ],
                "stream": False,
                "max_tokens": 4096,
                "temperature": temperature
            }

            # 如果指定了response_format，添加到payload中
            if response_format:
                payload["response_format"] = response_format

            # 发送请求（带重试机制）
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    async with self.session.post(url, json=payload) as response:
                        # 检查状态码
                        if response.status != 200:
                            error_text = await response.text()
                            error_msg = f"API请求失败: status={response.status}, error={error_text[:500]}"

                            # 如果是 400 错误，尝试获取更详细的错误信息
                            if response.status == 400:
                                try:
                                    error_json = await response.json()
                                    error_msg = f"API请求失败 (400 Bad Request): {error_json}"
                                except:
                                    pass

                            # 如果是401错误且允许重试，可能是并发导致的误报
                            if response.status == 401 and retry_on_401 and attempt < max_retries:
                                # 等待后重试（可能是并发限制导致的临时认证失败）
                                wait_time = (attempt + 1) * 0.5  # 递增等待时间：0.5s, 1.0s
                                print(f"[WARNING] 401错误，可能是并发限制，等待 {wait_time:.1f}s 后重试 ({attempt + 1}/{max_retries})...")
                                await asyncio.sleep(wait_time)
                                last_exception = aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=response.status,
                                    message=error_msg
                                )
                                continue  # 重试

                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=error_msg
                            )

                        result = await response.json()

                        # 检查响应格式
                        if "choices" not in result or len(result["choices"]) == 0:
                            raise ValueError(f"API响应格式错误: {result}")

                        return result["choices"][0]["message"]["content"]
                except aiohttp.ClientResponseError as e:
                    # 如果是401且允许重试，且还有重试机会
                    if e.status == 401 and retry_on_401 and attempt < max_retries:
                        wait_time = (attempt + 1) * 0.5
                        print(f"[WARNING] 401错误，可能是并发限制，等待 {wait_time:.1f}s 后重试 ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        last_exception = e
                        continue
                    # 其他错误或重试次数用完，直接抛出
                    raise
                except Exception as e:
                    # 非HTTP错误，直接抛出
                    raise

            # 如果所有重试都失败，抛出最后一个异常（带详细错误信息）
            if last_exception:
                error_msg = f"HTTP错误 {last_exception.status}: {last_exception.message}"
                if last_exception.status == 400:
                    error_msg += "\n可能的原因："
                    error_msg += "\n  1. 请求参数格式不正确"
                    error_msg += "\n  2. 图片 base64 编码有问题"
                    error_msg += "\n  3. 请求体大小超过限制"
                    error_msg += f"\n  4. 请求URL: {url}"
                    error_msg += f"\n  5. 原始模型名称: {self.model_name}"
                    error_msg += f"\n  6. 规范化模型名称: {self._normalize_model_name(self.model_name)}"
                elif last_exception.status == 401:
                    error_msg += "\n认证失败！可能的原因："
                    error_msg += "\n  1. API Key 未设置或无效"
                    error_msg += f"\n  2. 当前 API Key: {self.api_key[:10]}..." if len(self.api_key) > 10 else f"\n  2. 当前 API Key: {self.api_key}"
                    error_msg += "\n  3. 请检查环境变量 API_KEY 是否正确设置"
                    error_msg += "\n  4. 如果使用默认值 '1'，请设置正确的 API Key"
                    error_msg += "\n  5. 检查 API Key 是否已过期或被撤销"
                    error_msg += "\n  6. ⚠️ 可能是API并发限制：某些API不支持高并发，建议降低并发数（--concurrency 1-3）"
                    error_msg += f"\n  7. 当前并发设置: max_concurrent={self.max_concurrent}, request_delay={self.request_delay}s"
                raise Exception(error_msg) from last_exception
            else:
                # 理论上不应该到达这里，但为了安全
                raise Exception("请求失败：所有重试都失败，但未捕获到异常")

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

            # 如果使用 LBOpenAIAsyncClient，直接委托给它
            if self._parent.use_lb_client:
                async with self._parent.semaphore:  # 控制并发
                    if self._parent.request_delay > 0:
                        await asyncio.sleep(self._parent.request_delay)
                    return await self._parent.lb_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        **kwargs
                    )

            # 自定义实现（向后兼容）
            if not self._parent.session:
                raise RuntimeError("Session not initialized. Use async with statement.")

            # 从messages中提取文本和图像
            text_content = None
            image_input = None

            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, str):
                        # 简单字符串格式（没有图像）
                        text_content = content
                    elif isinstance(content, list):
                        # 列表格式，包含文本和图像
                        for item in content:
                            if item.get("type") == "text":
                                text_content = item.get("text", "")
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                # 提取base64部分或使用完整URL
                                if image_url.startswith("data:image"):
                                    # 格式: data:image/jpeg;base64,xxxxx
                                    image_input = image_url.split(",", 1)[1]
                                else:
                                    # 可能是纯base64字符串
                                    image_input = image_url

            if not text_content:
                raise ValueError("消息中必须包含文本内容")

            # 如果有图像，使用analyze_image_async；否则只发送文本
            if image_input:
                # 调用analyze_image_async（内部会处理base64编码）
                response_text = await self._parent.analyze_image_async(
                    image_input=image_input,
                    prompt=text_content,
                    temperature=temperature,
                    response_format=response_format
                )
            else:
                # 纯文本请求（暂不支持，因为当前API主要面向图像）
                raise ValueError("当前实现需要图像输入")

            # 构建类似OpenAI的响应对象
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
                    self.usage = {
                        "prompt_tokens": len(text_content) // 4,  # 粗略估算
                        "completion_tokens": len(content) // 4,
                        "total_tokens": (len(text_content) + len(content)) // 4
                    }

            return _Response(response_text)

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
