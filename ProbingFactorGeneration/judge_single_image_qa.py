"""
Simple tester for Judge model backend.

Run directly (no arguments):

    python -m ProbingFactorGeneration.judge_single_image_qa

Configure the image path / question / model name below in this file.
"""

import asyncio
from pathlib import Path
from typing import Optional

from PIL import Image

from ProbingFactorGeneration.models import JudgeModel


async def ask_judge_once(
    image_path: str,
    question: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Call the Judge backend once with (image, question) and return answer text.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(str(img_path)).convert("RGB")

    # Initialize JudgeModel; it will pick up MODEL_CONFIG by default
    judge = JudgeModel(model_name=model_name)

    async with judge:
        # Reuse JudgeModel's image → base64 helper and low-level call method
        image_base64 = judge._image_to_base64(image)  # type: ignore[attr-defined]

        prompt = (
            "You are a helpful visual question answering model.\n"
            "Given the image and the question, answer concisely in the same language as the question.\n\n"
            f"Question: {question}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]

        # Directly use JudgeModel's internal async call wrapper
        response = await judge._call_model_async(messages)  # type: ignore[attr-defined]
        # For OpenAI-compatible clients, content is in choices[0].message.content
        answer = response.choices[0].message.content
        return answer


def main() -> None:
    # TODO: 修改为你自己的测试参数
    IMAGE_PATH = "/absolute/path/to/your/image.jpg"
    QUESTION = "这里有几个人？"
    MODEL_NAME = None  # 或者例如 "qwen-vl" 等，留空则使用 MODEL_CONFIG 里的配置

    answer = asyncio.run(
        ask_judge_once(
            image_path=IMAGE_PATH,
            question=QUESTION,
            model_name=MODEL_NAME,
        )
    )

    print("=== Judge Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

