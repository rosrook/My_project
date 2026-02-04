"""
JSON 解析工具

统一处理 LLM 返回的 JSON 解析，支持：
- 从混合文本中提取 JSON 对象
- 布尔/数值/列表的类型安全解析
"""
import json
import re
from typing import Any, Dict, List, Optional


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    从响应中提取并解析 JSON 对象。
    使用贪婪匹配 { ... }，解析失败返回 None。
    """
    if not response or not isinstance(response, str):
        return None
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError):
        return None


def parse_bool(value: Any, default: bool = False) -> bool:
    """解析布尔值，支持字符串 "true"/"false" """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return default


def parse_float(value: Any, default: float = 0.5) -> float:
    """解析浮点数，支持字符串 "0.95" """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_issues(value: Any) -> List[str]:
    """解析 issues 字段，确保返回列表 """
    if isinstance(value, list):
        return [str(i) for i in value if i is not None]
    if value is not None and str(value).strip():
        return [str(value).strip()]
    return []
