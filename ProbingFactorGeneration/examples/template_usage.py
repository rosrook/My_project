"""
Example usage of template-based claim generation workflow.

This demonstrates the new template-based workflow:
1. Load claim templates with placeholders
2. Baseline model completes templates (fills placeholders or returns "not related")
3. Judge model verifies completions and explanations
"""

import asyncio
from pathlib import Path
from PIL import Image
from ..core import ImageLoader, TemplateClaimGenerator
from ..models import BaselineModel, JudgeModel
from ..core import FailureAggregator, FilteringFactorMapper
from ..io import DataSaver
from ..pipeline import ProbingFactorPipeline


async def example_template_completion():
    """Example: Template completion workflow."""
    print("Example: Template-based claim generation workflow")
    print("=" * 60)
    
    # Initialize components
    image_loader = ImageLoader()
    template_generator = TemplateClaimGenerator(config_path="configs/claim_template.example.json")
    baseline_model = BaselineModel(model_name="gemini-pro-vision")
    judge_model = JudgeModel(model_name="gemini-pro-vision")
    failure_aggregator = FailureAggregator()
    filtering_factor_mapper = FilteringFactorMapper()
    data_saver = DataSaver(output_dir="./output")
    
    # Create pipeline
    pipeline = ProbingFactorPipeline(
        image_loader=image_loader,
        claim_generator=template_generator,
        baseline_model=baseline_model,
        judge_model=judge_model,
        failure_aggregator=failure_aggregator,
        filtering_factor_mapper=filtering_factor_mapper,
        data_saver=data_saver
    )
    
    # Process single image with templates (async)
    async with baseline_model, judge_model:
        result = await pipeline.process_single_image_with_templates_async("image.jpg")
        
        print(f"\nImage ID: {result['image_id']}")
        print(f"\nClaim Templates ({len(result['claim_templates'])}):")
        for template in result['claim_templates']:
            print(f"  - {template['claim_id']}: {template['claim_template']}")
            print(f"    Placeholders: {template.get('placeholders', [])}")
        
        print(f"\nCompletions ({len(result['completions'])}):")
        for completion in result['completions']:
            print(f"  - Claim ID: {completion['claim_id']}")
            print(f"    Completed: {completion['completed_claim']}")
            print(f"    Is Related: {completion['is_related']}")
            print(f"    Explanation: {completion['explanation']}")
            if completion.get('filled_values'):
                print(f"    Filled Values: {completion['filled_values']}")
        
        print(f"\nVerifications ({len(result['verifications'])}):")
        for verif in result['verifications']:
            print(f"  - Claim ID: {verif['claim_id']}")
            print(f"    Is Correct: {verif['is_correct']}")
            if not verif['is_correct']:
                print(f"    Failure Reason: {verif.get('failure_reason')}")
            if verif.get('not_related_judgment_correct') is not None:
                print(f"    'Not Related' Judgment Correct: {verif['not_related_judgment_correct']}")
            print(f"    Judge Explanation: {verif.get('judge_explanation', '')[:100]}...")
        
        print("\n" + "=" * 60)


async def example_batch_processing():
    """Example: Batch processing with templates."""
    print("\nExample: Batch processing with templates")
    print("=" * 60)
    
    # Initialize components
    image_loader = ImageLoader(image_dir="./data/images")
    template_generator = TemplateClaimGenerator(config_path="configs/claim_template.example.json")
    baseline_model = BaselineModel(model_name="gemini-pro-vision", max_concurrent=10)
    judge_model = JudgeModel(model_name="gemini-pro-vision", max_concurrent=10)
    failure_aggregator = FailureAggregator()
    filtering_factor_mapper = FilteringFactorMapper()
    data_saver = DataSaver(output_dir="./output")
    
    # Create pipeline
    pipeline = ProbingFactorPipeline(
        image_loader=image_loader,
        claim_generator=template_generator,
        baseline_model=baseline_model,
        judge_model=judge_model,
        failure_aggregator=failure_aggregator,
        filtering_factor_mapper=filtering_factor_mapper,
        data_saver=data_saver
    )
    
    # Process batch
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    async with baseline_model, judge_model:
        results = await pipeline.process_batch_with_templates_async(image_paths)
        
        print(f"\nProcessed {len(results)} images")
        for result in results:
            print(f"\nImage: {result['image_id']}")
            print(f"  Templates: {len(result['claim_templates'])}")
            print(f"  Completions: {len(result['completions'])}")
            print(f"  Verifications: {len(result['verifications'])}")
            print(f"  Failures: {result['aggregated_failures'].get('failed_claims', 0)}")
        
        print("\n" + "=" * 60)


async def example_direct_api():
    """Example: Direct API usage without pipeline."""
    print("\nExample: Direct API usage")
    print("=" * 60)
    
    # Initialize
    template_generator = TemplateClaimGenerator(config_path="configs/claim_template.example.json")
    baseline_model = BaselineModel(model_name="gemini-pro-vision")
    judge_model = JudgeModel(model_name="gemini-pro-vision")
    
    # Prepare data
    image = Image.new("RGB", (100, 100), color="red")
    templates = template_generator.generate(image, image_id="test_image")
    
    print(f"\nGenerated {len(templates)} templates")
    for template in templates:
        print(f"  - {template['claim_template']}")
    
    # Complete templates
    async with baseline_model, judge_model:
        completions = await baseline_model.complete_template_batch_async(
            [image] * len(templates),
            templates
        )
        
        print(f"\nCompleted {len(completions)} templates")
        for completion in completions:
            print(f"  - Template: {completion['metadata']['original_template']}")
            print(f"    Completed: {completion['completed_claim']}")
            print(f"    Related: {completion['is_related']}")
            print(f"    Explanation: {completion['explanation'][:80]}...")
        
        # Verify completions
        verifications = await judge_model.verify_completion_batch_async(
            [image] * len(templates),
            templates,
            completions
        )
        
        print(f"\nVerifications:")
        for verif in verifications:
            print(f"  - Correct: {verif['is_correct']}")
            if verif.get('not_related_judgment_correct') is not None:
                print(f"    Not Related Judgment: {verif['not_related_judgment_correct']}")
            if not verif['is_correct']:
                print(f"    Failure: {verif.get('failure_reason')}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(example_template_completion())
        asyncio.run(example_batch_processing())
        asyncio.run(example_direct_api())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease create a claim_template.example.json file first.")
        print("See configs/claim_template.example.json for reference.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
