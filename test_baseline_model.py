#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试本地 Baseline 模型（LLaVA 等）是否可用。

使用方式：
    修改 IMAGE_PATH 和 MODEL_PATH，然后：
        python test_baseline_local.py
"""

import asyncio
from pathlib import Path
from typing import Optional

from PIL import Image
from ProbingFactorGeneration.models import BaselineModel


async def test_baseline_local_once(
    image_path: str,
    question: str,
    model_path: str,
    model_name: Optional[str] = None,
):
    """
    使用本地 Baseline 模型（use_local_model=True）跑一次 (image, question)。
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"图片不存在: {img_path}")

    image = Image.open(str(img_path)).convert("RGB")

    # 注意：这里显式指定 use_local_model=True 和 model_path
    baseline = BaselineModel(
        model_name=model_name,          # 一般可以 None
        model_path=model_path,          # 本地 LLaVA 模型目录
        use_local_model=True,           # 强制使用本地模型而不是 API
        max_concurrent=1,               # 单次测试就够了
    )

    async with baseline:
        claim = {"claim_text": question}
        result = await baseline.predict_async(image, claim)
        # result 是一个 dict，通常包含 prediction / explanation 等字段
        return result


def main():
    # ====== 根据你的环境修改这三个参数 ======
    IMAGE_PATH = "/path/to/your/test_image.jpg"
    QUESTION = "这张图片里有几个人？"
    # 建议直接复用 run_full_pipeline.sh 里用的本地 baseline 路径，例如：
    # /mnt/tidal-alsh01/.../hf_baseline_model
    MODEL_PATH = "/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model"
    MODEL_NAME = None  # 一般本地模型这里不用填，留 None 即可
    # ==================================

    result = asyncio.run(
        test_baseline_local_once(
            image_path=IMAGE_PATH,
            question=QUESTION,
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
        )
    )

    print("=== Baseline 本地模型测试结果 ===")
    print(result)


if __name__ == "__main__":
    main()