"""
模型返回值记录工具

当环境变量 MODEL_RESPONSE_LOG_DIR 被设置时，将各阶段模型调用的 prompt 与 response 记录到 JSONL 文件。
用于排查槽位填充失败、问题生成失败、答案生成失败等问题。

每条记录包含 trace_id（若已设置），便于跨 Step 1/2/3 锁定同一条记录。
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

try:
    from QA_Generator.utils.debug_trace_context import get_trace_id
except ImportError:
    def get_trace_id() -> Optional[str]:
        return None


def _get_log_dir() -> Optional[Path]:
    """获取模型返回值记录目录（从环境变量读取）"""
    log_dir = os.environ.get("MODEL_RESPONSE_LOG_DIR")
    if not log_dir or not str(log_dir).strip():
        return None
    return Path(log_dir)


def log_model_response(
    stage: str,
    prompt: str,
    response: str,
    context: Optional[Dict[str, Any]] = None,
    record_index: Optional[int] = None,
    pipeline_name: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    记录一次模型调用的 prompt 与 response。

    Args:
        stage: 阶段名称，如 slot_filling, question_generation, question_validation, answer_generation, answer_validation
        prompt: 发送给模型的 prompt（可截断，避免过大）
        response: 模型返回的原始文本
        context: 额外上下文（如 claim, prefilled_values, slots 等）
        record_index: 记录索引
        pipeline_name: pipeline 名称
        error: 若调用失败，错误信息
    """
    log_dir = _get_log_dir()
    if log_dir is None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # 每个 stage 一个 JSONL 文件
        log_file = log_dir / f"{stage}.jsonl"
        # 截断过长的 prompt/response，避免单条记录过大
        max_prompt = 8000
        max_response = 4000
        record = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "prompt": prompt[:max_prompt] + ("..." if len(prompt) > max_prompt else ""),
            "response": response[:max_response] + ("..." if len(response) > max_response else ""),
            "prompt_length": len(prompt),
            "response_length": len(response),
        }
        trace_id = get_trace_id()
        if trace_id is not None:
            record["trace_id"] = trace_id
        if context is not None:
            record["context"] = context
        if record_index is not None:
            record["record_index"] = record_index
        if pipeline_name is not None:
            record["pipeline_name"] = pipeline_name
        if error is not None:
            record["error"] = error
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 记录失败不应影响主流程
