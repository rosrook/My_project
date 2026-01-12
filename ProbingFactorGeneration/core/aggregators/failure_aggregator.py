"""
FailureAggregator module: Aggregate claim failures per image.
"""

from typing import List, Dict, Any, Optional, Set


class FailureAggregator:
    """
    Aggregate claim failures for each image to provide summary statistics.
    """
    
    def __init__(self):
        """Initialize FailureAggregator."""
        pass
    
    def aggregate(self, image_id: str, claims: List[Dict[str, Any]], 
                 verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate failures for a single image.
        
        Args:
            image_id: Image identifier
            claims: List of claim dictionaries (claim_templates)
            verifications: List of verification dictionaries from JudgeModel
            
        Returns:
            Aggregated failure summary:
            {
                "image_id": str,
                "total_claims": int,
                "failed_claims": int,
                "success_rate": float,
                "failure_breakdown": {
                    "failure_id_1": int,
                    "failure_id_2": int,
                    ...
                },
                "failed_claim_ids": List[str],
                "metadata": dict
            }
        """
        if len(claims) != len(verifications):
            raise ValueError("Claims and verifications must have same length")
        
        total = len(claims)
        
        failure_breakdown = {}
        failed_claim_ids = []
        failed_count = 0
        
        for claim, verif in zip(claims, verifications):
            # Check if this is a failure
            # A failure occurs if is_correct is False OR if not_related was incorrectly judged
            is_correct = verif.get("is_correct", True)
            not_related_judgment_correct = verif.get("not_related_judgment_correct")
            is_failure = False
            
            if not is_correct:
                is_failure = True
            elif not_related_judgment_correct is False:
                # Incorrectly marked as not_related (judge says baseline should have answered)
                is_failure = True
            
            if is_failure:
                failed_count += 1
                failed_claim_ids.append(claim.get("claim_id", ""))
                # Use failure_id from verification if available, otherwise use failure_reason
                failure_id = verif.get("failure_id") or verif.get("failure_reason", "unknown")
                failure_breakdown[failure_id] = failure_breakdown.get(failure_id, 0) + 1
        
        return {
            "image_id": image_id,
            "total_claims": total,
            "failed_claims": failed_count,
            "success_rate": (total - failed_count) / total if total > 0 else 0.0,
            "failure_breakdown": failure_breakdown,
            "failed_claim_ids": failed_claim_ids,
            "metadata": {}
        }
    
    def aggregate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate failures across multiple images.
        
        Args:
            results: List of individual image aggregation results
            
        Returns:
            Global aggregation summary:
            {
                "total_images": int,
                "total_claims": int,
                "total_failures": int,
                "overall_success_rate": float,
                "failure_breakdown": {
                    "failure_id_1": int,
                    ...
                },
                "per_image_summaries": List[Dict]  # Original per-image results
            }
        """
        if not results:
            return {
                "total_images": 0,
                "total_claims": 0,
                "total_failures": 0,
                "overall_success_rate": 0.0,
                "failure_breakdown": {},
                "per_image_summaries": []
            }
        
        total_images = len(results)
        total_claims = sum(r["total_claims"] for r in results)
        total_failures = sum(r["failed_claims"] for r in results)
        
        failure_breakdown = {}
        for r in results:
            for failure_id, count in r["failure_breakdown"].items():
                failure_breakdown[failure_id] = failure_breakdown.get(failure_id, 0) + count
        
        return {
            "total_images": total_images,
            "total_claims": total_claims,
            "total_failures": total_failures,
            "overall_success_rate": (total_claims - total_failures) / total_claims if total_claims > 0 else 0.0,
            "failure_breakdown": failure_breakdown,
            "per_image_summaries": results
        }
