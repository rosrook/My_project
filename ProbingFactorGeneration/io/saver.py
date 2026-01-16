"""
DataSaver module: Save structured results in multiple formats.
"""

from typing import Dict, List, Any, Union
from pathlib import Path
import json
import csv
import yaml
from collections import Counter


class DataSaver:
    """
    Save structured results in JSON, CSV, or YAML format.
    Provides statistical analysis interfaces.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "./output"):
        """
        Initialize DataSaver.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: Dict[str, Any], output_path: Union[str, Path], 
            format: str = "json") -> Path:
        """
        Save data to file in specified format.
        
        Args:
            data: Data dictionary to save
            output_path: Path to output file (relative to output_dir or absolute)
            format: Output format - "json", "csv", or "yaml"
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            self._save_json(data, output_path)
        elif format.lower() == "csv":
            self._save_csv(data, output_path)
        elif format.lower() == "yaml":
            self._save_yaml(data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path
    
    def save_results(self, results: List[Dict[str, Any]], 
                    filename: str = "probing_results", 
                    format: str = "json") -> Path:
        """
        Save complete probing results for all images.
        
        Args:
            results: List of result dictionaries, each containing:
                    {
                        "image_id": str,
                        "claim_templates": List[Dict],
                        "completions": List[Dict],
                        "verifications": List[Dict],
                        "aggregated_failures": Dict,
                        "suggested_filtering_factors": List[str]
                    }
            filename: Base filename (without extension)
            format: Output format
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.{format}"
        structured_data = {
            "results": results,
            "summary": self._generate_summary(results)
        }
        return self.save(structured_data, output_path, format)
    
    def compute_error_rate_by_claim_type(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute error rate for each claim type.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary mapping claim type to error rate:
            {
                "content_type_1": float,  # Error rate (0.0 to 1.0)
                "content_type_2": float,
                ...
            }
        """
        claim_type_stats = {}
        
        for result in results:
            verifications = result.get("verifications", [])
            
            for verif in verifications:
                content_type = verif.get("content_type", "unknown")
                if content_type not in claim_type_stats:
                    claim_type_stats[content_type] = {"total": 0, "errors": 0}
                
                claim_type_stats[content_type]["total"] += 1
                if not verif.get("is_correct", True):
                    claim_type_stats[content_type]["errors"] += 1
        
        error_rates = {
            ctype: stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            for ctype, stats in claim_type_stats.items()
        }
        
        return error_rates
    
    def compute_filtering_factor_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Compute distribution of filtering factors.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary mapping filtering factor to count:
            {
                "factor_1": int,
                "factor_2": int,
                ...
            }
        """
        factor_distribution = {}
        
        for result in results:
            factors = result.get("suggested_filtering_factors", [])
            if isinstance(factors, dict):
                # If it's a distribution dict, merge it
                for factor, count in factors.items():
                    factor_distribution[factor] = factor_distribution.get(factor, 0) + count
            elif isinstance(factors, list):
                # If it's a list, count occurrences
                for factor in factors:
                    factor_distribution[factor] = factor_distribution.get(factor, 0) + 1
        
        return factor_distribution
    
    def _save_json(self, data: Dict[str, Any], path: Path):
        """Save data as JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, data: Dict[str, Any], path: Path):
        """
        Save data as CSV file.
        
        TODO:
            - Flatten nested structures
            - Handle list fields appropriately
        """
        # TODO: Implement CSV flattening and saving
        raise NotImplementedError("DataSaver._save_csv() not implemented")
    
    def _save_yaml(self, data: Dict[str, Any], path: Path):
        """Save data as YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Summary dictionary with statistics
        """
        if not results:
            return {
                "total_images": 0,
                "total_claims": 0,
                "total_failures": 0,
                "overall_success_rate": 0.0
            }
        
        total_images = len(results)
        total_claims = sum(
            r.get("aggregated_failures", {}).get("total_claims", 0) for r in results
        )
        total_failures = sum(
            r.get("aggregated_failures", {}).get("failed_claims", 0) for r in results
        )
        
        error_rates = self.compute_error_rate_by_claim_type(results)
        factor_distribution = self.compute_filtering_factor_distribution(results)
        
        return {
            "total_images": total_images,
            "total_claims": total_claims,
            "total_failures": total_failures,
            "overall_success_rate": (total_claims - total_failures) / total_claims if total_claims > 0 else 0.0,
            "error_rates_by_claim_type": error_rates,
            "filtering_factor_distribution": factor_distribution
        }


def analyze_failure_collection(output_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze failure collection output from rank*/failures folders.

    Counts total collected cases (JSON files) and failure type distribution.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    rank_dirs = sorted(
        d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("rank")
    )
    candidate_dirs = rank_dirs if rank_dirs else [output_dir]

    failure_json_paths: List[Path] = []
    failure_image_paths: List[Path] = []
    for base_dir in candidate_dirs:
        failure_dir = base_dir / "failures"
        if not failure_dir.exists():
            continue
        failure_json_paths.extend(sorted(failure_dir.glob("*.json")))
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            failure_image_paths.extend(failure_dir.glob(ext))

    failure_type_counts: Counter = Counter()
    bad_json_files: List[str] = []
    for json_path in failure_json_paths:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            bad_json_files.append(str(json_path))
            continue

        breakdown = (
            data.get("aggregated_failures", {}).get("failure_breakdown")
            if isinstance(data, dict)
            else None
        )
        if isinstance(breakdown, dict) and breakdown:
            for key, value in breakdown.items():
                failure_key = key or "unknown"
                try:
                    count = int(value)
                except Exception:
                    count = 1
                failure_type_counts[failure_key] += count
            continue

        verifications = data.get("verifications", []) if isinstance(data, dict) else []
        used_verifications = False
        for verif in verifications:
            if verif.get("is_correct", True):
                continue
            failure_key = (
                verif.get("failure_id")
                or verif.get("failure_reason")
                or "unknown"
            )
            failure_type_counts[failure_key] += 1
            used_verifications = True

        if not used_verifications:
            failed_ids = (
                data.get("aggregated_failures", {}).get("failed_claim_ids", [])
                if isinstance(data, dict)
                else []
            )
            if isinstance(failed_ids, list):
                for failure_id in failed_ids:
                    failure_type_counts[str(failure_id) if failure_id else "unknown"] += 1

    failure_type_distribution = dict(
        sorted(failure_type_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    ranks_detected = [
        d.name for d in candidate_dirs if (d / "failures").exists()
    ]

    return {
        "total_collected": len(failure_json_paths),
        "total_images": len(failure_image_paths),
        "ranks_detected": ranks_detected,
        "failure_type_distribution": failure_type_distribution,
        "bad_json_files": bad_json_files,
    }
