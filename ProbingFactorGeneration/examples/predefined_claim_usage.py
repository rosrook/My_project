"""
Example usage of PredefinedClaimGenerator with claim_config.json.

This file demonstrates how to use PredefinedClaimGenerator to load
manually designed claims from a JSON configuration file.
"""

from pathlib import Path
from PIL import Image
from ..core.generators import PredefinedClaimGenerator


def example_load_from_config():
    """Example: Load claims from claim_config.json."""
    print("Example: Load claims from predefined config")
    
    # Initialize generator with config file path
    generator = PredefinedClaimGenerator(config_path="claim_config.json")
    
    # Generate claims for an image
    image = Image.new("RGB", (100, 100), color="red")
    image_id = "image_001"
    
    claims = generator.generate(image, image_id)
    print(f"\nGenerated {len(claims)} claims for {image_id}:")
    for claim in claims:
        print(f"  - {claim['claim_id']}: {claim['claim_text']}")
        print(f"    Type: {claim['content_type']}")
        print(f"    Metadata: {claim.get('metadata', {})}")
    
    print()


def example_batch_processing():
    """Example: Batch processing with predefined claims."""
    print("Example: Batch processing with predefined claims")
    
    generator = PredefinedClaimGenerator(config_path="claim_config.json")
    
    # Prepare batch of images
    images = [
        Image.new("RGB", (100, 100), color="red"),
        Image.new("RGB", (100, 100), color="blue"),
        Image.new("RGB", (100, 100), color="green"),
    ]
    image_ids = ["image_001", "image_002", "image_003"]
    
    # Generate claims for all images
    results = generator.generate_batch(images, image_ids)
    
    for img_id, claims in results.items():
        print(f"\n{img_id}: {len(claims)} claims")
        for claim in claims:
            print(f"  - {claim['claim_text']}")
    
    print()


def example_check_claims():
    """Example: Check available claims."""
    print("Example: Check available claims")
    
    generator = PredefinedClaimGenerator(config_path="claim_config.json")
    
    # Get all image IDs with specific claims
    image_ids = generator.get_all_image_ids()
    print(f"\nImage IDs with specific claims: {image_ids}")
    
    # Check if an image has specific claims
    if generator.has_image_claims("image_001"):
        print("image_001 has specific claims")
    else:
        print("image_001 uses global claims only")
    
    # Get global claims count
    global_count = generator.get_global_claims_count()
    print(f"Number of global claims: {global_count}")
    
    print()


def example_in_pipeline():
    """Example: Using PredefinedClaimGenerator in pipeline."""
    print("Example: Using in pipeline")
    
    from ..core import ImageLoader, PredefinedClaimGenerator
    from ..models import BaselineModel, JudgeModel
    from ..core import FailureAggregator, FilteringFactorMapper
    from ..io import DataSaver
    from ..pipeline import ProbingFactorPipeline
    
    # Create pipeline with PredefinedClaimGenerator
    image_loader = ImageLoader(image_dir="./data/images")
    claim_generator = PredefinedClaimGenerator(config_path="claim_config.json")
    baseline_model = BaselineModel(model_name="gemini-pro-vision")
    judge_model = JudgeModel()
    failure_aggregator = FailureAggregator()
    filtering_factor_mapper = FilteringFactorMapper()
    data_saver = DataSaver(output_dir="./output")
    
    pipeline = ProbingFactorPipeline(
        image_loader=image_loader,
        claim_generator=claim_generator,
        baseline_model=baseline_model,
        judge_model=judge_model,
        failure_aggregator=failure_aggregator,
        filtering_factor_mapper=filtering_factor_mapper,
        data_saver=data_saver
    )
    
    print("Pipeline created with PredefinedClaimGenerator")
    print("You can now use pipeline.process_single_image() or pipeline.process_batch()")
    print()


if __name__ == "__main__":
    try:
        example_load_from_config()
        example_batch_processing()
        example_check_claims()
        example_in_pipeline()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease create a claim_config.json file first.")
        print("See configs/claim_config.example.json for reference.")
    except Exception as e:
        print(f"Error: {e}")
