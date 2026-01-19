# ConvertDataFrame

This folder provides an end-to-end pipeline to convert a VQA JSON dataset (with
base64 images) into WebDataset (WDS) shards, and generate a quick check report.

## What it does
- `preprocessor.py`: VQA JSON (base64 image) -> JSON+image files
- `code2_fixed.py`: JSON+image files -> WDS `.tar` shards
- `check_wbe.py`: Inspect a few samples in the WDS output
- `run_convert_pipeline.py`: One-stop CLI that runs all three steps

## Quick start
```bash
python ConvertDataFrame/run_convert_pipeline.py \
  --input /path/to/vqa.json \
  --output_dir /path/to/wds_output
```

If the input is a directory, it will look for `vqa_dataset_successful_*.json`,
otherwise it will use all `*.json` in that directory.

## Outputs
- Intermediate files: `<output_dir>_preprocessed/`
- WDS shards: `<output_dir>/subtaskdata-*.tar`
- Check report: `<output_dir>/wds_check.txt`

## Common options
```bash
python ConvertDataFrame/run_convert_pipeline.py \
  --input /path/to/vqa_dir \
  --output_dir /path/to/wds_output \
  --work_dir /path/to/preprocessed \
  --maxcount 128 \
  --num_workers 16 \
  --check_samples 5 \
  --check_output /path/to/wds_check.txt
```

## Notes
- The preprocessor expects each item to include `image_base64`, `question` or
  `full_question`, and `answer`.
- The conversion step reads the JSON+image pairs produced by the preprocessor.
