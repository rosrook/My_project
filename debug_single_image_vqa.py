#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug helper: run VQA pipeline from a single local image.

Edit the CONFIG section below, then run:
  python debug_single_image_vqa.py
"""
import base64
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =========================
# CONFIG (edit here)
# =========================
PROJECT_ROOT = Path("/home/zhuxuzhou/My_project")

IMAGE_PATH = Path("/path/to/your.jpg")
PIPELINE_NAME = "question"  # e.g. question, object_position, caption, etc.

# Choose ONE of the following:
TARGET_OBJECT = "cat"
CLAIM_TEXT = ""  # if set, claim mode is used instead of target_object

# QA_Generator pipeline options
CONCURRENCY = 3
ENABLE_VALIDATION_EXEMPTIONS = True
REQUEST_DELAY = 0.1
NO_ASYNC = False
NO_INTERMEDIATE = False
BATCH_SIZE = 1000
MAX_SAMPLES = None
QUESTION_CONFIG = ""
ANSWER_CONFIG = ""

# Output
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "debug_single_image"
INPUT_JSON_NAME = "debug_input.json"
LOG_FILE = DEBUG_OUTPUT_DIR / f"vqa_debug_{datetime.now().strftime('%m%d_%H%M%S')}.log"


def die(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def build_record(image_path: Path, image_b64: str) -> dict:
    record = {
        "id": "debug_0001",
        "sample_index": 0,
        "pipeline_name": PIPELINE_NAME,
        "source_a": {
            "id": "debug_image",
            "image_path": str(image_path),
            "image_base64": image_b64,
        },
    }
    if CLAIM_TEXT:
        record["prefill"] = {"claim": CLAIM_TEXT}
    else:
        record["prefill"] = {"target_object": TARGET_OBJECT}
    return record


def main() -> None:
    if not PROJECT_ROOT.exists():
        die(f"PROJECT_ROOT not found: {PROJECT_ROOT}")
    if not IMAGE_PATH.exists():
        die(f"IMAGE_PATH not found: {IMAGE_PATH}")
    if not CLAIM_TEXT and not TARGET_OBJECT:
        die("Set either CLAIM_TEXT or TARGET_OBJECT in CONFIG.")

    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(IMAGE_PATH, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    input_path = DEBUG_OUTPUT_DIR / INPUT_JSON_NAME
    payload = [build_record(IMAGE_PATH, image_b64)]
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "QA_Generator" / "pipeline" / "pipeline.py"),
        str(input_path),
        "--concurrency",
        str(CONCURRENCY),
        "--request-delay",
        str(REQUEST_DELAY),
        "--batch-size",
        str(BATCH_SIZE),
    ]
    if ENABLE_VALIDATION_EXEMPTIONS:
        cmd.append("--enable-validation-exemptions")
    if NO_ASYNC:
        cmd.append("--no-async")
    if NO_INTERMEDIATE:
        cmd.append("--no-intermediate")
    if MAX_SAMPLES is not None:
        cmd.extend(["--max-samples", str(MAX_SAMPLES)])
    if QUESTION_CONFIG:
        cmd.extend(["--question-config", QUESTION_CONFIG])
    if ANSWER_CONFIG:
        cmd.extend(["--answer-config", ANSWER_CONFIG])

    print("== Debug input written ==")
    print(f"Input JSON: {input_path}")
    print("== Running QA_Generator pipeline ==")
    print("Command:", " ".join(cmd))

    with open(LOG_FILE, "w", encoding="utf-8") as log_f:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        die(f"Pipeline failed. See log: {LOG_FILE}")

    print("✅ Done.")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
