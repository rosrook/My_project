"""
VLM-based factor scorer.

This mirrors the async call pattern in ProbingFactorGeneration/models/judge.py
but targets filtering-factor satisfaction instead of claim verification.
"""

from __future__ import annotations

import base64
import io
import json
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from ProbingFactorGeneration.utils.async_client import AsyncGeminiClient
except ImportError:  # pragma: no cover - optional dependency
    AsyncGeminiClient = None


def split_required_optional(factors: List[str]) -> Tuple[List[str], List[str]]:
    """Split factors into required and optional lists."""
    required: List[str] = []
    optional: List[str] = []
    for factor in factors:
        if "[optional]" in factor.lower():
            cleaned = factor.replace("[optional]", "").strip()
            optional.append(cleaned)
        else:
            required.append(factor.strip())
    return required, optional


def compute_score(required_met: bool, optional_met_count: int, optional_total: int) -> float:
    if not required_met:
        return 0.0
    if optional_total <= 0:
        return 0.6
    bonus = 0.4 * (optional_met_count / optional_total)
    return 0.6 + bonus


class VLMFactorScorer:
    """
    Async scorer that verifies filtering factors using a VLM.

    Scoring logic:
    - All required factors must be satisfied. If any fail -> score 0.0.
    - If required satisfied, base score = 0.6
    - Optional factors add up to 0.4 total, proportional to how many are met.
    """

    def __init__(
        self,
        model_name: str = "gemini-pro-vision",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        gpu_id: int = 0,
        max_concurrent: Optional[int] = None,
        request_delay: float = 0.0,
        use_lb_client: Optional[bool] = None,
        service_name: Optional[str] = None,
        env: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gpu_id = gpu_id
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.use_lb_client = use_lb_client
        self.service_name = service_name
        self.env = env
        self.api_key = api_key
        self._async_client = None

    async def _get_client(self) -> Any:
        if AsyncGeminiClient is None:
            raise NotImplementedError(
                "AsyncGeminiClient is not available. "
                "Please install or implement ProbingFactorGeneration.utils.async_client.AsyncGeminiClient"
            )
        if self._async_client is None:
            self._async_client = AsyncGeminiClient(
                model_name=self.model_name,
                gpu_id=self.gpu_id,
                max_concurrent=self.max_concurrent,
                request_delay=self.request_delay,
                use_lb_client=self.use_lb_client,
                service_name=self.service_name,
                env=self.env,
                api_key=self.api_key,
            )
            await self._async_client.__aenter__()
        return self._async_client

    async def _close_client(self) -> None:
        if self._async_client is not None:
            try:
                await self._async_client.__aexit__(None, None, None)
            except Exception:
                pass

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_prompt(self, required: List[str], optional: List[str]) -> str:
        required_block = "\n".join([f"- {r}" for r in required]) or "- None"
        optional_block = "\n".join([f"- {o}" for o in optional]) or "- None"
        return (
            "You are evaluating whether an image satisfies filtering factors.\n"
            "Required factors MUST all be satisfied. Optional factors are bonus.\n\n"
            f"Required factors:\n{required_block}\n\n"
            f"Optional factors:\n{optional_block}\n\n"
            "Return JSON with fields:\n"
            "{\n"
            '  "required_met": true/false,\n'
            '  "optional_met": ["optional factor text", ...],\n'
            '  "explanation": "short explanation"\n'
            "}\n"
        )

    async def _call_model_async(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        client = await self._get_client()
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if response_format:
            request_params["response_format"] = response_format
        return await client.chat.completions.create(**request_params)

    async def score_async(self, image: Image.Image, suggested_factors: List[str]) -> Dict[str, Any]:
        required, optional = split_required_optional(suggested_factors)
        prompt = self._build_prompt(required, optional)
        image_base64 = self._image_to_base64(image)
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

        response = await self._call_model_async(messages, response_format={"type": "json_object"})
        response_text = response.choices[0].message.content

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

        required_met = bool(parsed.get("required_met", False))
        optional_met = parsed.get("optional_met", []) or []
        optional_met_count = len(optional_met)
        score = compute_score(required_met, optional_met_count, len(optional))

        return {
            "score": score,
            "required_met": required_met,
            "optional_met": optional_met,
            "optional_total": len(optional),
            "explanation": parsed.get("explanation", ""),
            "metadata": {
                "required_factors": required,
                "optional_factors": optional,
                "response_text": response_text,
            },
        }

    async def score_batch_async(
        self,
        images: List[Image.Image],
        suggested_factors_list: List[List[str]],
    ) -> List[Dict[str, Any]]:
        tasks = [
            self.score_async(image, factors)
            for image, factors in zip(images, suggested_factors_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                output.append({"score": 0.0, "explanation": str(result), "error": True})
            else:
                output.append(result)
        return output

    async def __aenter__(self) -> "VLMFactorScorer":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._close_client()
