"""
Main pipeline for probing factor generation.
Demonstrates the data flow through all modules.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image

from ProbingFactorGeneration.core import ImageLoader, ClaimGenerator, FailureAggregator, FilteringFactorMapper
from ProbingFactorGeneration.core.generators.template_claim_generator import TemplateClaimGenerator
from ProbingFactorGeneration.models import BaselineModel, JudgeModel
from ProbingFactorGeneration.io import DataSaver
import asyncio

try:
    from tqdm.asyncio import tqdm as atqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    atqdm = None

try:
    from ProbingFactorGeneration.core.mappers.failure_reason_matcher import FailureReasonMatcher
    HAS_FAILURE_MATCHER = True
except ImportError:
    HAS_FAILURE_MATCHER = False
    FailureReasonMatcher = None


class ProbingFactorPipeline:
    """
    Main pipeline orchestrating the probing factor generation process.
    """
    
    def __init__(
        self,
        image_loader: ImageLoader,
        claim_generator: ClaimGenerator,
        baseline_model: BaselineModel,
        judge_model: JudgeModel,
        failure_aggregator: FailureAggregator,
        filtering_factor_mapper: FilteringFactorMapper,
        data_saver: DataSaver,
        failure_reason_matcher: Optional[Any] = None,
        include_source_metadata: bool = False
    ):
        """
        Initialize pipeline with all required modules.
        
        Args:
            image_loader: ImageLoader instance
            claim_generator: ClaimGenerator instance
            baseline_model: BaselineModel instance
            judge_model: JudgeModel instance
            failure_aggregator: FailureAggregator instance
            filtering_factor_mapper: FilteringFactorMapper instance
            data_saver: DataSaver instance
            failure_reason_matcher: FailureReasonMatcher instance (optional, auto-created if not provided)
            include_source_metadata: If True, include source metadata (e.g., conversations) in results (for reference only)
        """
        self.image_loader = image_loader
        self.claim_generator = claim_generator
        self.baseline_model = baseline_model
        self.judge_model = judge_model
        self.failure_aggregator = failure_aggregator
        self.filtering_factor_mapper = filtering_factor_mapper
        self.data_saver = data_saver
        self.include_source_metadata = include_source_metadata
        
        # Initialize FailureReasonMatcher if not provided
        if failure_reason_matcher is None and HAS_FAILURE_MATCHER:
            self.failure_reason_matcher = FailureReasonMatcher()
        else:
            self.failure_reason_matcher = failure_reason_matcher
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete result dictionary:
            {
                "image_id": str,
                "claims": List[Dict],
                "baseline_answers": List[Dict],
                "verifications": List[Dict],
                "aggregated_failures": Dict,
                "filtering_factors": List[str] or Dict
            }
        """
        # Step 1: Load image
        image = self.image_loader.load(image_path)
        image_id = self.image_loader.get_image_id(image_path)
        
        # Step 2: Generate probing claims
        claims = self.claim_generator.generate(image, image_id)
        
        # Step 3: Get baseline predictions for each claim
        baseline_answers = []
        for claim in claims:
            baseline_answer = self.baseline_model.predict(image, claim)
            baseline_answers.append(baseline_answer)
        
        # Step 4: Judge verifies each claim and generates failure reasons
        verifications = []
        for claim, baseline_answer in zip(claims, baseline_answers):
            verification = self.judge_model.verify(image, claim, baseline_answer)
            verifications.append(verification)
        
        # Step 5: Aggregate failures for this image
        aggregated_failures = self.failure_aggregator.aggregate(
            image_id, claims, verifications
        )
        
        # Step 6: Map failure reasons to filtering factors
        # Extract failure reasons from verifications
        failure_reasons = [
            v.get("failure_reason") for v in verifications 
            if not v.get("is_correct", True) and v.get("failure_reason")
        ]
        filtering_factors = self.filtering_factor_mapper.map_batch(failure_reasons)
        
        # Also map aggregated failures to get distribution
        aggregated_with_factors = self.filtering_factor_mapper.map_aggregated_failures(
            aggregated_failures
        )
        
        # Construct result
        result = {
            "image_id": image_id,
            "claims": claims,
            "baseline_answers": baseline_answers,
            "verifications": verifications,
            "aggregated_failures": aggregated_with_factors,
            "filtering_factors": filtering_factors
        }
        
        return result
    
    async def process_single_image_with_templates_async(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the template-based pipeline (async version).
        
        Complete workflow:
        1. Load image
        2. Generate claim templates from claim_template.example_v1_1.json (with placeholders)
        3. Baseline model completes templates (fills placeholders or returns "not related")
        4. Judge model verifies completions and explanations (checks correctness and not_related judgments)
        5. Match failures to failure_config.example.json to get failure_id and suggested_filtering_factors
        6. Aggregate failures for the image
        7. Collect all suggested_filtering_factors from failed claims as the image's filtering factors
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete result dictionary:
            {
                "image_id": str,
                "claim_templates": List[Dict],  # Original templates from claim_template config
                "completions": List[Dict],  # Baseline completions (completed_claim, explanation, is_related)
                "verifications": List[Dict],  # Enhanced verifications with failure_id, failure_category, suggested_filtering_factors
                "aggregated_failures": Dict,  # Aggregated failure statistics
                "suggested_filtering_factors": List[str],  # All unique filtering factors for this image (from failed claims)
                "source_metadata": Dict (optional)  # Source metadata (e.g., conversations) if include_source_metadata=True
            }
        """
        # Step 1: Load image
        image = self.image_loader.load(image_path)
        image_id = self.image_loader.get_image_id(image_path)
        
        # Step 2: Generate claim templates (if using TemplateClaimGenerator)
        if isinstance(self.claim_generator, TemplateClaimGenerator):
            claim_templates = self.claim_generator.generate(image, image_id)
        else:
            # Fallback: use existing generator (for backward compatibility)
            claims = self.claim_generator.generate(image, image_id)
            # Convert to templates format
            claim_templates = [
                {
                    "claim_id": c.get("claim_id", ""),
                    "claim_template": c.get("claim_text", ""),
                    "content_type": c.get("content_type", "relation"),
                    "placeholders": [],
                    "metadata": c.get("metadata", {})
                }
                for c in claims
            ]
        
        # Step 3: Baseline model completes templates
        completions = await self.baseline_model.complete_template_batch_async(
            [image] * len(claim_templates),
            claim_templates
        )
        
        # Step 4: Judge model verifies completions
        verifications = await self.judge_model.verify_completion_batch_async(
            [image] * len(claim_templates),
            claim_templates,
            completions
        )
        
        # Step 5: Match failure reasons and extract filtering factors
        # For each verification, match to failure_config and get suggested_filtering_factors
        enhanced_verifications = []
        all_filtering_factors = []  # Collect all filtering factors for failed claims
        
        for claim_template, completion, verification in zip(claim_templates, completions, verifications):
            enhanced_verification = verification.copy()
            
            # Determine if this claim has a failure
            # A failure occurs if:
            # 1. is_correct is False (judge says the claim/completion is incorrect)
            # 2. is_related is False but not_related_judgment_correct is False (incorrectly marked as not_related)
            is_related = completion.get("is_related", True)
            not_related_judgment_correct = verification.get("not_related_judgment_correct")
            is_correct = verification.get("is_correct", True)
            
            # Check if this is a failure case
            is_failure = False
            if not is_correct:
                is_failure = True
            elif not is_related and not_related_judgment_correct is False:
                # Baseline incorrectly marked as not_related (judge says it should be related)
                is_failure = True
            
            # If it's a failure, match to failure_config
            if is_failure and self.failure_reason_matcher:
                matched_failure = self.failure_reason_matcher.match_failure_for_claim(
                    claim_template=claim_template,
                    judge_failure_reason=verification.get("failure_reason"),
                    is_related=is_related,
                    not_related_judgment_correct=not_related_judgment_correct
                )
                
                if matched_failure:
                    # Add failure_id and suggested_filtering_factors to verification
                    enhanced_verification["failure_id"] = matched_failure.get("failure_id")
                    enhanced_verification["failure_category"] = matched_failure.get("failure_category")
                    enhanced_verification["suggested_filtering_factors"] = matched_failure.get("suggested_filtering_factors", [])
                    
                    # Collect filtering factors for this failed claim
                    filtering_factors = matched_failure.get("suggested_filtering_factors", [])
                    if filtering_factors:
                        all_filtering_factors.append(filtering_factors)
                else:
                    # No match found, but still a failure
                    enhanced_verification["failure_id"] = None
                    enhanced_verification["suggested_filtering_factors"] = []
            else:
                # Not a failure
                enhanced_verification["failure_id"] = None
                enhanced_verification["suggested_filtering_factors"] = []
            
            enhanced_verifications.append(enhanced_verification)
        
        # Step 6: Aggregate failures for this image
        aggregated_failures = self.failure_aggregator.aggregate(
            image_id, 
            claim_templates, 
            enhanced_verifications
        )
        
        # Step 7: Merge all filtering factors from failed claims
        # Collect all unique filtering factors as the image's suggested filtering factors
        if all_filtering_factors:
            image_filtering_factors = self.filtering_factor_mapper.map_batch(all_filtering_factors)
        else:
            image_filtering_factors = []
        
        # Enhance aggregated failures
        aggregated_with_factors = self.filtering_factor_mapper.map_aggregated_failures(
            aggregated_failures
        )
        
        # Construct result
        result = {
            "image_id": image_id,
            # "claim_templates": claim_templates,
            "completions": completions,
            "verifications": enhanced_verifications,  # Enhanced with failure_id, failure_category, and suggested_filtering_factors
            "aggregated_failures": aggregated_with_factors,
            "suggested_filtering_factors": image_filtering_factors  # All unique filtering factors for this image (from failed claims)
        }
        
        # Optionally include source metadata (e.g., conversations) for reference
        if self.include_source_metadata:
            source_metadata = self.image_loader.get_image_metadata(image_path)
            if source_metadata:
                result["source_metadata"] = source_metadata  # For reference only, not used in processing
        
        return result
    
    def process_single_image_with_templates(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the template-based pipeline (sync wrapper).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Complete result dictionary (same format as process_single_image_with_templates_async())
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync process_single_image_with_templates() from within an async context. "
                "Use process_single_image_with_templates_async() instead."
            )
        except RuntimeError as e:
            if "Cannot use sync" in str(e):
                raise
            pass
        
        return asyncio.run(self.process_single_image_with_templates_async(image_path))
    
    async def process_batch_with_templates_async(self, image_paths: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images through the template-based pipeline (async version).
        Uses concurrent processing for better performance.
        
        Args:
            image_paths: List of paths to image files
            show_progress: Whether to show progress bar (requires tqdm)
            
        Returns:
            List of result dictionaries (one per image)
        """
        # Process all images concurrently
        tasks = [
            self.process_single_image_with_templates_async(image_path)
            for image_path in image_paths
        ]
        
        # Use tqdm for progress bar if available (with exception handling)
        if show_progress and HAS_TQDM and atqdm:
            # tqdm.gather doesn't support return_exceptions, so we use as_completed instead
            async def process_with_index(idx, task):
                try:
                    result = await task
                    return idx, result
                except Exception as e:
                    return idx, e
            
            indexed_tasks = [process_with_index(i, task) for i, task in enumerate(tasks)]
            
            results_list = []
            async for future in atqdm.as_completed(indexed_tasks, total=len(indexed_tasks), desc="Processing images", unit="img"):
                idx, result = await future
                results_list.append((idx, result))
            
            # Sort by index to maintain order
            results_list.sort(key=lambda x: x[0])
            results = [result for _, result in results_list]
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                image_id = self.image_loader.get_image_id(image_paths[i]) if i < len(image_paths) else f"image_{i}"
                processed_results.append({
                    "image_id": image_id,
                    "completions": [],
                    "verifications": [],
                    "aggregated_failures": {
                        "image_id": image_id,
                        "total_claims": 0,
                        "failed_claims": 0,
                        "success_rate": 0.0,
                        "failure_breakdown": {}
                    },
                    "suggested_filtering_factors": [],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_batch_with_templates(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of images through the template-based pipeline (sync wrapper).
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of result dictionaries (one per image)
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync process_batch_with_templates() from within an async context. "
                "Use process_batch_with_templates_async() instead."
            )
        except RuntimeError as e:
            if "Cannot use sync" in str(e):
                raise
            pass
        
        return asyncio.run(self.process_batch_with_templates_async(image_paths))
    
    async def run_and_save_with_templates_async(
        self, 
        image_paths: List[str], 
        output_filename: str = "probing_results",
        output_format: str = "json"
    ) -> Path:
        """
        Run template-based pipeline and save results (async version).
        
        Args:
            image_paths: List of paths to image files
            output_filename: Base filename for output (without extension)
            output_format: Output format ("json", "csv", or "yaml")
            
        Returns:
            Path to saved output file
        """
        # Process all images
        results = await self.process_batch_with_templates_async(image_paths)
        
        # Save results
        output_path = self.data_saver.save_results(
            results, output_filename, output_format
        )
        
        # Print statistics
        self._print_statistics_with_templates(results)
        
        return output_path
    
    def run_and_save_with_templates(
        self, 
        image_paths: List[str], 
        output_filename: str = "probing_results",
        output_format: str = "json"
    ) -> Path:
        """
        Run template-based pipeline and save results (sync wrapper).
        
        Args:
            image_paths: List of paths to image files
            output_filename: Base filename for output (without extension)
            output_format: Output format ("json", "csv", or "yaml")
            
        Returns:
            Path to saved output file
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync run_and_save_with_templates() from within an async context. "
                "Use run_and_save_with_templates_async() instead."
            )
        except RuntimeError as e:
            if "Cannot use sync" in str(e):
                raise
            pass
        
        return asyncio.run(self.run_and_save_with_templates_async(image_paths, output_filename, output_format))
    
    def _print_statistics_with_templates(self, results: List[Dict[str, Any]]):
        """Print summary statistics for template-based results."""
        # Compute error rates by content type
        content_type_stats = {}
        not_related_correction_count = 0
        total_not_related = 0
        
        for result in results:
            verifications = result.get("verifications", [])
            for verif in verifications:
                content_type = verif.get("content_type", "unknown")
                if content_type not in content_type_stats:
                    content_type_stats[content_type] = {"total": 0, "errors": 0}
                
                content_type_stats[content_type]["total"] += 1
                if not verif.get("is_correct", True):
                    content_type_stats[content_type]["errors"] += 1
                
                # Count not_related corrections
                not_related_judgment = verif.get("not_related_judgment_correct")
                if not_related_judgment is False:  # Judge corrected "not related"
                    not_related_correction_count += 1
                if not_related_judgment is not None:
                    total_not_related += 1
        
        # Compute filtering factor distribution
        factor_dist = self.data_saver.compute_filtering_factor_distribution(results)
        
        print("\n" + "="*50)
        print("Template-Based Pipeline Statistics")
        print("="*50)
        print(f"\nTotal images processed: {len(results)}")
        print(f"\nError rates by content type:")
        for content_type, stats in content_type_stats.items():
            rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            print(f"  {content_type}: {rate:.2%} ({stats['errors']}/{stats['total']})")
        
        if total_not_related > 0:
            print(f"\n'Not Related' Corrections:")
            print(f"  Judge corrected incorrect 'not related': {not_related_correction_count}/{total_not_related} ({not_related_correction_count/total_not_related:.2%})")
        
        print(f"\nFiltering factor distribution:")
        for factor, count in factor_dist.items():
            print(f"  {factor}: {count}")
        print("="*50 + "\n")
    
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of images through the pipeline.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of result dictionaries (one per image)
        """
        results = []
        for image_path in image_paths:
            result = self.process_single_image(image_path)
            results.append(result)
        return results
    
    def run_and_save(self, image_paths: List[str], 
                    output_filename: str = "probing_results",
                    output_format: str = "json") -> Path:
        """
        Run pipeline and save results.
        
        Args:
            image_paths: List of paths to image files
            output_filename: Base filename for output (without extension)
            output_format: Output format ("json", "csv", or "yaml")
            
        Returns:
            Path to saved output file
        """
        # Process all images
        results = self.process_batch(image_paths)
        
        # Save results
        output_path = self.data_saver.save_results(
            results, output_filename, output_format
        )
        
        # Print statistics
        self._print_statistics(results)
        
        return output_path
    
    def _print_statistics(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        # Compute error rates by claim type
        error_rates = self.data_saver.compute_error_rate_by_claim_type(results)
        
        # Compute filtering factor distribution
        factor_dist = self.data_saver.compute_filtering_factor_distribution(results)
        
        print("\n" + "="*50)
        print("Pipeline Statistics")
        print("="*50)
        print(f"\nTotal images processed: {len(results)}")
        print(f"\nError rates by claim type:")
        for claim_type, rate in error_rates.items():
            print(f"  {claim_type}: {rate:.2%}")
        print(f"\nFiltering factor distribution:")
        for factor, count in factor_dist.items():
            print(f"  {factor}: {count}")
        print("="*50 + "\n")


def create_pipeline(
    image_dir: Optional[str] = None,
    output_dir: str = "./output",
    baseline_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None
) -> ProbingFactorPipeline:
    """
    Factory function to create a pipeline with default configurations.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory for output files
        baseline_model_name: Name of baseline model
        judge_model_name: Name of judge model
        
    Returns:
        Configured ProbingFactorPipeline instance
    """
    # Initialize all modules
    image_loader = ImageLoader(image_dir=image_dir)
    claim_generator = ClaimGenerator()
    baseline_model = BaselineModel(model_name=baseline_model_name)
    judge_model = JudgeModel(model_name=judge_model_name)
    failure_aggregator = FailureAggregator()
    filtering_factor_mapper = FilteringFactorMapper()
    data_saver = DataSaver(output_dir=output_dir)
    
    # Create pipeline
    pipeline = ProbingFactorPipeline(
        image_loader=image_loader,
        claim_generator=claim_generator,
        baseline_model=baseline_model,
        judge_model=judge_model,
        failure_aggregator=failure_aggregator,
        filtering_factor_mapper=filtering_factor_mapper,
        data_saver=data_saver
    )
    
    return pipeline


def main():
    """
    Example main function demonstrating pipeline usage.
    """
    # Example: Process a small batch of images for testing
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]
    
    # Create pipeline
    pipeline = create_pipeline(
        image_dir="./data/images",
        output_dir="./output",
        baseline_model_name="my_baseline_model",
        judge_model_name="my_judge_model"
    )
    
    # Load models (must be implemented)
    # pipeline.baseline_model.load_model()
    # pipeline.judge_model.load_model()
    
    # Run pipeline on small batch
    # output_path = pipeline.run_and_save(
    #     image_paths[:2],  # Process first 2 images for testing
    #     output_filename="test_results",
    #     output_format="json"
    # )
    # 
    # print(f"Results saved to: {output_path}")
    
    print("Pipeline created. Implement model loading and run pipeline.run_and_save() to process images.")


if __name__ == "__main__":
    main()
