#!/usr/bin/env python3
"""
测试所有导入是否正常工作
"""

import sys
from pathlib import Path

# Add My_project to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键模块的导入"""
    print("Testing imports...")
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.config...")
        from ProbingFactorGeneration.config import MODEL_CONFIG, ContentType, FailureTaxonomy
        print("    - MODEL_CONFIG:", type(MODEL_CONFIG))
        print("    - ContentType:", ContentType)
        print("    - FailureTaxonomy:", FailureTaxonomy)
    except Exception as e:
        print(f"  ✗ Failed to import config: {e}")
        return False
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.core...")
        from ProbingFactorGeneration.core import ImageLoader, TemplateClaimGenerator, FailureAggregator, FilteringFactorMapper
        print("    - ImageLoader:", ImageLoader)
        print("    - TemplateClaimGenerator:", TemplateClaimGenerator)
        print("    - FailureAggregator:", FailureAggregator)
        print("    - FilteringFactorMapper:", FilteringFactorMapper)
    except Exception as e:
        print(f"  ✗ Failed to import core: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.models...")
        from ProbingFactorGeneration.models import BaselineModel, JudgeModel
        print("    - BaselineModel:", BaselineModel)
        print("    - JudgeModel:", JudgeModel)
    except Exception as e:
        print(f"  ✗ Failed to import models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.io...")
        from ProbingFactorGeneration.io import DataSaver
        print("    - DataSaver:", DataSaver)
    except Exception as e:
        print(f"  ✗ Failed to import io: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.pipeline...")
        from ProbingFactorGeneration.pipeline import ProbingFactorPipeline
        print("    - ProbingFactorPipeline:", ProbingFactorPipeline)
    except Exception as e:
        print(f"  ✗ Failed to import pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.utils...")
        from ProbingFactorGeneration.utils import AsyncGeminiClient
        print("    - AsyncGeminiClient:", AsyncGeminiClient)
    except Exception as e:
        print(f"  ⚠ Warning importing utils (may be optional): {e}")
    
    try:
        print("  ✓ Importing ProbingFactorGeneration.core.mappers.failure_reason_matcher...")
        from ProbingFactorGeneration.core.mappers.failure_reason_matcher import FailureReasonMatcher
        print("    - FailureReasonMatcher:", FailureReasonMatcher)
    except Exception as e:
        print(f"  ⚠ Warning importing FailureReasonMatcher (may be optional): {e}")
    
    print("\n✓ All critical imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
