"""
根据 factor_id 组合调用或模拟对应的筛选原语执行逻辑（如 VLM 校验或规则检查）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..data.schemas import AbstractionResult, PrimitiveSpec


@dataclass
class PrimitiveExecutor:
    """
    单个筛选原语的执行逻辑：demo 中可模拟（返回通过/不通过或分数），
    实际可对接 VLM 校验或 Rule_based 规则。
    """
    factor_id: str
    description: str
    original_texts: List[str]
    # 可选：真实执行函数 (image_path, **kwargs) -> bool | float
    execute_fn: Optional[Callable[..., bool | float]] = None

    def run(self, image_path: str, **kwargs: object) -> bool | float:
        if self.execute_fn is not None:
            return self.execute_fn(image_path, **kwargs)
        # 模拟：无真实执行时返回 True（通过）
        return True


class Router:
    """
    根据 factor_id 组合路由到对应的筛选原语并执行（或模拟执行）。
    """

    def __init__(
        self,
        abstraction: AbstractionResult,
        executors: Optional[Dict[str, PrimitiveExecutor]] = None,
    ) -> None:
        self.abstraction = abstraction
        self.factor_id_to_texts = abstraction.factor_id_to_texts
        self.executors: Dict[str, PrimitiveExecutor] = executors or {}
        # 若未提供 executor，则为每个 factor_id 创建模拟 executor
        for p in abstraction.primitives:
            if p.factor_id not in self.executors:
                self.executors[p.factor_id] = PrimitiveExecutor(
                    factor_id=p.factor_id,
                    description=p.original_texts[0] if p.original_texts else "",
                    original_texts=p.original_texts,
                    execute_fn=None,
                )

    def get_primitive_specs(self, factor_ids: List[str]) -> List[PrimitiveSpec]:
        """将 factor_id 列表转为 PrimitiveSpec 列表（可解释）。"""
        specs: List[PrimitiveSpec] = []
        for fid in factor_ids:
            texts = self.factor_id_to_texts.get(fid, [])
            desc = texts[0] if texts else fid
            specs.append(PrimitiveSpec(factor_id=fid, description=desc, original_texts=texts))
        return specs

    def run_primitives(
        self,
        factor_ids: List[str],
        image_path: str,
        **kwargs: object,
    ) -> Dict[str, bool | float]:
        """
        对给定 factor_id 列表，依次执行对应原语（或模拟），返回 factor_id -> 结果。
        """
        results: Dict[str, bool | float] = {}
        for fid in factor_ids:
            ex = self.executors.get(fid)
            if ex is not None:
                results[fid] = ex.run(image_path, **kwargs)
        return results
