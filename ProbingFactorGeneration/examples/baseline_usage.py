"""
Example usage of BaselineModel with async API calls.

This file demonstrates how to use the BaselineModel class for both sync and async operations.
"""

import asyncio
from pathlib import Path
from PIL import Image
from ..models import BaselineModel


async def example_async_single():
    """Example: Async single prediction using context manager."""
    print("Example 1: Async single prediction")
    
    # Prepare data
    image = Image.new("RGB", (100, 100), color="red")  # Dummy image
    claim = {"claim_text": "This image contains a red rectangle."}
    
    # Use async context manager
    async with BaselineModel(
        model_name="gemini-pro-vision",
        max_concurrent=5
    ) as model:
        prediction = await model.predict_async(image, claim)
        print(f"Prediction: {prediction}")
        print()


async def example_async_batch():
    """Example: Async batch prediction with concurrent processing."""
    print("Example 2: Async batch prediction")
    
    # Prepare batch data
    images = [Image.new("RGB", (100, 100), color="red") for _ in range(5)]
    claims = [
        {"claim_text": f"Image {i} contains a red rectangle."}
        for i in range(5)
    ]
    
    # Process batch concurrently
    async with BaselineModel(max_concurrent=10) as model:
        predictions = await model.predict_batch_async(images, claims)
        print(f"Processed {len(predictions)} predictions")
        for i, pred in enumerate(predictions):
            print(f"  Prediction {i}: {pred.get('prediction')}")
        print()


async def example_multi_gpu():
    """Example: Multi-GPU concurrent processing."""
    print("Example 3: Multi-GPU processing")
    
    # Prepare data
    num_items = 32
    num_gpus = 4
    images = [Image.new("RGB", (100, 100), color="red") for _ in range(num_items)]
    claims = [
        {"claim_text": f"Image {i} contains a red rectangle."}
        for i in range(num_items)
    ]
    
    # Split tasks across GPUs
    tasks_per_gpu = num_items // num_gpus
    
    async def process_gpu(gpu_id: int, start_idx: int, end_idx: int):
        """Process tasks for a specific GPU."""
        async with BaselineModel(
            gpu_id=gpu_id,
            max_concurrent=10
        ) as model:
            return await model.predict_batch_async(
                images[start_idx:end_idx],
                claims[start_idx:end_idx]
            )
    
    # Create tasks for each GPU
    gpu_tasks = [
        process_gpu(
            i,
            i * tasks_per_gpu,
            (i + 1) * tasks_per_gpu if i < num_gpus - 1 else num_items
        )
        for i in range(num_gpus)
    ]
    
    # Execute all GPU tasks concurrently
    all_results = await asyncio.gather(*gpu_tasks, return_exceptions=True)
    
    # Merge results
    results = []
    for gpu_id, gpu_results in enumerate(all_results):
        if isinstance(gpu_results, Exception):
            print(f"GPU {gpu_id} error: {gpu_results}")
        else:
            results.extend(gpu_results)
            print(f"GPU {gpu_id} processed {len(gpu_results)} items")
    
    print(f"Total processed: {len(results)}")
    print()


def example_sync_single():
    """Example: Sync single prediction (wrapper around async)."""
    print("Example 4: Sync single prediction")
    
    image = Image.new("RGB", (100, 100), color="red")
    claim = {"claim_text": "This image contains a red rectangle."}
    
    model = BaselineModel(model_name="gemini-pro-vision")
    try:
        prediction = model.predict(image, claim)
        print(f"Prediction: {prediction}")
    finally:
        model.close()
    print()


def example_sync_batch():
    """Example: Sync batch prediction (wrapper around async)."""
    print("Example 5: Sync batch prediction")
    
    images = [Image.new("RGB", (100, 100), color="red") for _ in range(3)]
    claims = [
        {"claim_text": f"Image {i} contains a red rectangle."}
        for i in range(3)
    ]
    
    model = BaselineModel(max_concurrent=10)
    try:
        predictions = model.predict_batch(images, claims)
        print(f"Processed {len(predictions)} predictions")
        for i, pred in enumerate(predictions):
            print(f"  Prediction {i}: {pred.get('prediction')}")
    finally:
        model.close()
    print()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("BaselineModel Usage Examples")
    print("=" * 60)
    print()
    
    # Note: These examples will fail if AsyncGeminiClient is not available
    # They are provided as templates for actual usage
    
    try:
        # Async examples
        await example_async_single()
        await example_async_batch()
        await example_multi_gpu()
        
        # Sync examples
        example_sync_single()
        example_sync_batch()
        
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("\nNote: AsyncGeminiClient must be implemented or imported.")
        print("Please ensure utils.async_client.AsyncGeminiClient is available.")
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    # Run async examples
    asyncio.run(main())
