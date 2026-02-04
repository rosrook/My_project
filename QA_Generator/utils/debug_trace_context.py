"""
调试追踪上下文

用于在 debug 模式下为每条记录设置统一的 trace_id，便于跨阶段（Step 1/2/3）锁定同一条记录。
trace_id 格式: R{record_index}_S{sample_index}_I{image_id}
"""
import os
from contextvars import ContextVar
from typing import Any, Dict, Optional

# 当前记录的追踪信息（线程/协程安全）
_trace_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar("debug_trace_context", default=None)


def set_trace_context(
    record_index: int,
    sample_index: Optional[int] = None,
    image_id: Optional[str] = None,
    record_id: Optional[Any] = None,
    pipeline_name: Optional[str] = None,
    batch_num: Optional[int] = None,
) -> None:
    """
    设置当前记录的追踪上下文（由 pipeline 在处理每条记录前调用）
    trace_id 格式: [run_id_]B{batch_num}_R{record_index}_S{sample_index}_I{image_id}
    """
    run_id = os.environ.get("DEBUG_RUN_ID", "")
    run_prefix = f"{run_id}_" if run_id else ""
    # 标准化 image_id 为短字符串（避免特殊字符）
    safe_img = str(image_id or "").replace("/", "_")[:80] if image_id else ""
    trace_id = f"{run_prefix}"
    if batch_num is not None:
        trace_id += f"B{batch_num}_"
    trace_id += f"R{record_index}"
    if sample_index is not None:
        trace_id += f"_S{sample_index}"
    if safe_img:
        trace_id += f"_I{safe_img}"
    _trace_ctx.set({
        "trace_id": trace_id,
        "record_index": record_index,
        "sample_index": sample_index,
        "image_id": image_id,
        "record_id": record_id,
        "pipeline_name": pipeline_name,
        "batch_num": batch_num,
    })


def clear_trace_context() -> None:
    """清除追踪上下文"""
    _trace_ctx.set(None)


def get_trace_context() -> Optional[Dict[str, Any]]:
    """获取当前追踪上下文"""
    return _trace_ctx.get()


def get_trace_id() -> Optional[str]:
    """获取当前 trace_id，若无则返回 None"""
    ctx = _trace_ctx.get()
    return ctx.get("trace_id") if ctx else None
