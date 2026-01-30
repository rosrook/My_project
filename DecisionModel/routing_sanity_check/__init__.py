"""
管线路由与验证：根据预测的 factor_id 组合调用/模拟筛选原语执行逻辑，验证可解释性与合理性。
"""

from .router import Router, PrimitiveExecutor
from .sanity_check import run_sanity_check, SanityCheckReport

__all__ = [
    "Router",
    "PrimitiveExecutor",
    "run_sanity_check",
    "SanityCheckReport",
]
