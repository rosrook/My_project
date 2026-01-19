"""
End-to-end conversion pipeline:
1) preprocessor: VQA JSON -> code2_fixed JSON+images
2) code2_fixed: JSON+images -> WebDataset
3) check_wbe: generate a check report
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ConvertDataFrame import preprocessor, code2_fixed, check_wbe


def _collect_vqa_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        vqa_files = list(input_path.glob("vqa_dataset_successful_*.json"))
        if not vqa_files:
            vqa_files = list(input_path.glob("*.json"))
        return vqa_files
    raise FileNotFoundError(f"Input not found: {input_path}")


def run_preprocess(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_file():
        preprocessor.convert_vqa_to_standard_format(
            str(input_path),
            str(output_dir),
        )
        return

    vqa_files = _collect_vqa_files(input_path)
    if not vqa_files:
        raise FileNotFoundError(f"No JSON files found in {input_path}")
    preprocessor.batch_convert([str(p) for p in vqa_files], str(output_dir))


def run_wds_convert(preprocessed_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    code2_args = argparse.Namespace(
        output_dir=str(output_dir),
        data_root=[str(preprocessed_dir)],
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        media=args.media,
        columns_messages=args.columns_messages,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_workers=args.num_workers,
        debug_file=args.debug_file,
    )

    manager = code2_fixed.Manager()
    code2_fixed.batch_size = code2_args.batch_size
    code2_fixed.output_queue = manager.Queue(maxsize=code2_args.num_workers * 100)

    code2_fixed.convert_parquet_to_wds(code2_args)


def _get_tar_paths(output_dir: Path) -> List[str]:
    tar_paths = sorted(output_dir.glob("subtaskdata-*.tar"))
    if tar_paths:
        return [str(p) for p in tar_paths]
    tar_paths = sorted(output_dir.glob("subtaskdata-*.tgz"))
    return [str(p) for p in tar_paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="VQA -> WDS conversion pipeline")
    parser.add_argument("--input", required=True, help="VQA JSON file or directory")
    parser.add_argument("--output_dir", required=True, help="WDS output directory")
    parser.add_argument(
        "--work_dir",
        default=None,
        help="Intermediate output dir for preprocessor",
    )
    parser.add_argument("--maxcount", type=int, default=128, help="Samples per shard")
    parser.add_argument("--maxsize", type=int, default=3000000000, help="Shard size")
    parser.add_argument(
        "--media",
        type=str,
        choices=["mix", "image", "video"],
        default="image",
        help="Media type",
    )
    parser.add_argument(
        "--columns_messages",
        type=str,
        default="conversations",
        help="Messages column name",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=100000,
        help="Shuffle buffer size",
    )
    parser.add_argument("--num_workers", type=int, default=16, help="Num workers")
    parser.add_argument("--debug_file", type=int, default=-1, help="Debug files")
    parser.add_argument(
        "--check_samples",
        type=int,
        default=5,
        help="Number of WDS samples to inspect",
    )
    parser.add_argument(
        "--check_output",
        default=None,
        help="Optional output path for check report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir.parent / f"{output_dir.name}_preprocessed"

    run_preprocess(input_path, work_dir)
    run_wds_convert(work_dir, output_dir, args)

    tar_paths = _get_tar_paths(output_dir)
    if not tar_paths:
        raise FileNotFoundError(f"No WDS tar files found in {output_dir}")

    check_output = args.check_output or str(output_dir / "wds_check.txt")
    check_wbe.write_check_report(
        tar_paths,
        output_path=check_output,
        num_samples=args.check_samples,
    )


if __name__ == "__main__":
    main()
