# Probing Factor Generation Framework

A modular, extensible Python framework for VQA data construction stage 1: **small-scale probing + failure → filtering factor**.

📖 **For detailed Claims design documentation, see [docs/CLAIMS_DESIGN.md](docs/CLAIMS_DESIGN.md)**

## Overview

This framework implements a research-iterable pipeline for:
1. Loading images
2. Generating probing claims
3. Running baseline model predictions
4. Verifying claims with judge model and identifying failures
5. Aggregating failures and mapping to filtering factors
6. Saving structured results for downstream QA generation

## Framework Structure

```
ProbingFactorGeneration/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration, failure taxonomy, constants
├── pipeline.py              # Main pipeline orchestrator
├── core/                    # Core processing modules
│   ├── __init__.py          # Core package exports
│   ├── loaders/             # Data loading modules
│   │   ├── __init__.py
│   │   └── image_loader.py  # ImageLoader: Load images/batches
│   ├── generators/          # Claim generation modules
│   │   ├── __init__.py
│   │   └── claim_generator.py  # ClaimGenerator: Generate probing claims
│   ├── aggregators/         # Failure aggregation modules
│   │   ├── __init__.py
│   │   └── failure_aggregator.py  # FailureAggregator: Aggregate failures per image
│   └── mappers/             # Filtering factor mapping modules
│       ├── __init__.py
│       └── filtering_factor_mapper.py  # Map failure reasons to filtering factors
├── models/                  # Model interfaces
│   ├── __init__.py          # Models package exports
│   ├── baseline.py          # BaselineModel: Predict claim validity (async API)
│   └── judge.py             # JudgeModel: Verify claims, identify failures
├── io/                      # Input/Output modules
│   ├── __init__.py          # IO package exports
│   └── saver.py             # DataSaver: Save results + statistics
├── utils/                   # Utilities package
│   ├── __init__.py          # Utils package exports
│   └── async_client.py      # AsyncGeminiClient: Async API client with GPU support
├── examples/                # Example scripts
│   ├── __init__.py
│   └── baseline_usage.py    # Usage examples for BaselineModel
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Module Interfaces

### ImageLoader
- `load(image_path)` → PIL Image
- `load_batch(image_paths)` → List[Image]
- `get_image_id(image_path)` → str

### ClaimGenerator
- `generate(image, image_id)` → List[Dict[claim]]
- `generate_batch(images, image_ids)` → Dict[image_id: List[claims]]

### PredefinedClaimGenerator
- `generate(image, image_id)` → List[Dict[claim]] (from JSON config)
- `generate_batch(images, image_ids)` → Dict[image_id: List[claims]]
- `get_all_image_ids()` → List[str] (get all image IDs in config)
- `has_image_claims(image_id)` → bool (check if image has specific claims)
- `get_global_claims_count()` → int (get number of global claims)

### BaselineModel
- `load_model()` → None
- `predict(image, claim)` → Dict[prediction, confidence, metadata]
- `predict_batch(images, claims)` → List[Dict]

### JudgeModel
- `load_model()` → None
- `verify(image, claim, baseline_answer)` → Dict[is_correct, failure_reason, ...]
- `verify_batch(...)` → List[Dict]

### FailureAggregator
- `aggregate(image_id, claims, verifications)` → Dict[aggregated_summary]
- `aggregate_batch(results)` → Dict[global_summary]

### FilteringFactorMapper
- `map(failure_reason)` → str (filtering_factor)
- `map_batch(failure_reasons)` → List[str]
- `map_aggregated_failures(aggregated_result)` → Dict[enhanced_result]

### DataSaver
- `save(data, output_path, format)` → Path
- `save_results(results, filename, format)` → Path
- `compute_error_rate_by_claim_type(results)` → Dict[claim_type: error_rate]
- `compute_filtering_factor_distribution(results)` → Dict[factor: count]

## Usage Example

### Using Default ClaimGenerator

```python
from ProbingFactorGeneration import create_pipeline

# Create pipeline with default config
pipeline = create_pipeline(
    image_dir="./data/images",
    output_dir="./output",
    baseline_model_name="my_baseline",
    judge_model_name="my_judge"
)

# Process images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
output_path = pipeline.run_and_save(
    image_paths,
    output_filename="results",
    output_format="json"
)
```

### Using PredefinedClaimGenerator with JSON Config

```python
from ProbingFactorGeneration import (
    ImageLoader, PredefinedClaimGenerator, BaselineModel,
    JudgeModel, FailureAggregator, FilteringFactorMapper, DataSaver
)
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline

# Create PredefinedClaimGenerator with config file
claim_generator = PredefinedClaimGenerator(config_path="claim_config.json")

# Create pipeline with predefined claim generator
image_loader = ImageLoader(image_dir="./data/images")
baseline_model = BaselineModel(model_name="gemini-pro-vision")
judge_model = JudgeModel()
failure_aggregator = FailureAggregator()
filtering_factor_mapper = FilteringFactorMapper()
data_saver = DataSaver(output_dir="./output")

pipeline = ProbingFactorPipeline(
    image_loader=image_loader,
    claim_generator=claim_generator,  # Use PredefinedClaimGenerator
    baseline_model=baseline_model,
    judge_model=judge_model,
    failure_aggregator=failure_aggregator,
    filtering_factor_mapper=filtering_factor_mapper,
    data_saver=data_saver
)

# Process images (claims will be loaded from claim_config.json)
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = pipeline.process_batch(image_paths)
```

### Claim Config File Format (claim_config.json)

Create a `claim_config.json` file with the following structure:

```json
{
  "metadata": {
    "description": "Predefined claims configuration",
    "version": "1.0"
  },
  "global_claims": [
    {
      "claim_id": "global_claim_1",
      "claim_text": "Does this image contain any text?",
      "content_type": "text",
      "metadata": {
        "difficulty": "easy"
      }
    },
    "Simple string claim that will be auto-converted"
  ],
  "claims_by_image": {
    "image_001": [
      {
        "claim_text": "Does this image show a person?",
        "content_type": "object"
      }
    ],
    "image_002": [
      "What is the dominant color in this image?"
    ]
  }
}
```

See `configs/claim_config.example.json` for a complete example.

## Data Flow

For each image:
1. **Load** → Image object
2. **Generate** → List of probing claims (JSON)
3. **Baseline Predict** → Predictions for each claim
4. **Judge Verify** → Verification + failure reasons
5. **Aggregate** → Failure summary per image
6. **Map** → Filtering factors from failure reasons
7. **Save** → Structured JSON/CSV/YAML output

## Output Format

Results are saved as structured JSON:

```json
{
  "results": [
    {
      "image_id": "img_001",
      "claims": [...],
      "baseline_answers": [...],
      "verifications": [...],
      "aggregated_failures": {
        "image_id": "img_001",
        "total_claims": 10,
        "failed_claims": 3,
        "success_rate": 0.7,
        "failure_breakdown": {...},
        "filtering_factor_distribution": {...}
      },
      "filtering_factors": [...]
    }
  ],
  "summary": {...}
}
```

## Failure Taxonomy

Defined in `config.py`, extensible:
- `VISUAL_ERROR`: Visual recognition mistakes
- `VISUAL_AMBIGUITY`: Unclear image content
- `LANGUAGE_MISUNDERSTANDING`: Ambiguous claim text
- `LANGUAGE_COMPLEXITY`: Complex reasoning required
- `REASONING_ERROR`: Logical inconsistencies
- `COMMONSENSE_ERROR`: Common sense violations
- `MODEL_LIMITATION`: Out-of-distribution issues
- `UNCERTAIN`: Insufficient information

## Extension Points

1. **Replace Models**: Implement `BaselineModel` and `JudgeModel` interfaces
2. **Extend Taxonomy**: Add new categories to `FailureTaxonomy` in `config.py`
3. **Custom Claim Generation**: Extend `ClaimGenerator.generate()`
4. **Custom Mapping**: Provide `custom_mapping` to `FilteringFactorMapper`

## Next Steps

1. Implement model loading and inference in `BaselineModel` and `JudgeModel`
2. Implement claim generation logic in `ClaimGenerator`
3. Add image preprocessing in `ImageLoader`
4. Test pipeline on small batch of images
5. Iteratively refine based on output analysis

## Async Client Integration

The framework includes `AsyncGeminiClient` in `utils/async_client.py`, which provides:
- **Async API calls** with controlled concurrency via Semaphore
- **GPU binding** for process isolation
- **LBOpenAIAsyncClient support** (when available) or custom aiohttp implementation
- **OpenAI-compatible interface** (`chat.completions.create`)
- **Automatic image compression** and base64 encoding
- **401 error retry** mechanism for concurrency-related false positives

The `BaselineModel` class uses `AsyncGeminiClient` internally and provides both:
- **Async interface**: `predict_async()`, `predict_batch_async()`
- **Sync wrapper**: `predict()`, `predict_batch()` (using `asyncio.run()`)

## Configuration

Configure the async client via `MODEL_CONFIG` in `config.py` or environment variables:

```python
# In config.py or via environment variables
MODEL_CONFIG = {
    "MODEL_NAME": "gemini-pro-vision",
    "SERVICE_NAME": "your-service-name",  # For LBOpenAIAsyncClient
    "ENV": "prod",
    "API_KEY": "your-api-key",
    "BASE_URL": "https://api.example.com",  # If not using LBOpenAIAsyncClient
    "MAX_CONCURRENT": 10,
    "REQUEST_DELAY": 0.1,
    "USE_LB_CLIENT": True,  # Use LBOpenAIAsyncClient if available
}
```

Or via environment variables:
```bash
export MODEL_NAME="gemini-pro-vision"
export SERVICE_NAME="your-service"
export API_KEY="your-api-key"
export MAX_CONCURRENT=10
```

## Installation

```bash
pip install -r requirements.txt
```

**Note**: If using LBOpenAIAsyncClient mode, you may need to install the `redeuler` package separately:
```bash
pip install redeuler  # Or install from your package source
```

## Notes

- `BaselineModel` is fully implemented with async API calls
- Other modules are skeleton classes with TODO comments
- Designed for research iteration and debugging
- Easy to extend and replace individual components
- Async client supports both LBOpenAIAsyncClient and custom aiohttp implementations