"""
Simple tester for Baseline model backend.

Run directly (no arguments):

    python -m ProbingFactorGeneration.baseline_single_image_qa

Configure the image path / question / model name below in this file.
"""

import asyncio
from pathlib import Path
from typing import Optional

from PIL import Image

from ProbingFactorGeneration.models import BaselineModel


async def ask_baseline_once(
    image_path: str,
    question: str,
    model_name: Optional[str] = None,
    use_local_model: Optional[bool] = None,
    model_path: Optional[str] = None,
) -> str:
    """
    Call the Baseline backend once with (image, question) and return answer text.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(str(img_path)).convert("RGB")

    # Initialize BaselineModel; it will pick up MODEL_CONFIG by default
    baseline = BaselineModel(
        model_name=model_name,
        model_path=model_path,
        use_local_model=use_local_model,
    )

    async with baseline:
        # Build messages in the same way as normal baseline claims:
        # we treat the question as a simple "claim_text".
        claim = {"claim_text": question}
        messages = baseline._build_messages(image, claim)  # type: ignore[attr-defined]

        # Directly use BaselineModel's internal async call wrapper
        response = await baseline._call_model_async(messages)  # type: ignore[attr-defined]
        # For OpenAI-compatible clients, content is in choices[0].message.content
        answer = response.choices[0].message.content
        return answer


def main() -> None:
    # TODO: 修改为你自己的测试参数
    IMAGE_PATH = "/absolute/path/to/your/image.jpg"
    QUESTION = "这里有几个人？"
    MODEL_NAME = None  # 或者例如 "gemini-pro-vision" 等，留空则使用 MODEL_CONFIG 配置

    # 如果你想测试本地 LLaVA 模型，可以设置：
    USE_LOCAL_MODEL = None  # True / False / None
    MODEL_PATH = None  # 例如 "/path/to/llava/model"

    answer = asyncio.run(
        ask_baseline_once(
            image_path=IMAGE_PATH,
            question=QUESTION,
            model_name=MODEL_NAME,
            use_local_model=USE_LOCAL_MODEL,
            model_path=MODEL_PATH,
        )
    )

    print("=== Baseline Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

