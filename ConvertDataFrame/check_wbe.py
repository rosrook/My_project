import argparse
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import webdataset as wds


def _normalize_dataset_paths(
    dataset_paths: Union[str, Sequence[str]],
) -> Union[str, List[str]]:
    if isinstance(dataset_paths, str):
        return dataset_paths
    return [str(p) for p in dataset_paths]


def check_wds(
    dataset_paths: Union[str, Sequence[str]],
    num_samples: int = 5,
) -> List[str]:
    dataset_paths = _normalize_dataset_paths(dataset_paths)
    ds = wds.WebDataset(dataset_paths)
    lines: List[str] = []
    for i, sample in enumerate(islice(ds, num_samples)):
        lines.append(f"\n===== Sample {i} =====")
        lines.append(f"Keys: {list(sample.keys())}")
        for k, v in sample.items():
            if isinstance(v, bytes):
                lines.append(f"  {k}: bytes ({len(v)} bytes)")
            else:
                lines.append(f"  {k}: {type(v)}")
    return lines


def write_check_report(
    dataset_paths: Union[str, Sequence[str]],
    output_path: Optional[str],
    num_samples: int = 5,
) -> None:
    lines = check_wds(dataset_paths, num_samples=num_samples)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    else:
        print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check WebDataset samples")
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="WebDataset tar pattern or file paths",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to inspect",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the check report",
    )
    args = parser.parse_args()
    write_check_report(args.dataset, args.output, num_samples=args.num_samples)


if __name__ == "__main__":
    main()

