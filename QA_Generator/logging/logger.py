"""
日志工具模块
支持同时输出到控制台和文件
"""
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """日志工具类，支持同时输出到控制台和文件"""

    def __init__(self, log_file: Optional[Path] = None):
        """
        初始化日志器

        Args:
            log_file: 日志文件路径（可选）
        """
        self.log_file = log_file
        self.log_file_handle = None

        if log_file:
            # 创建日志文件目录
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # 打开日志文件（追加模式）
            self.log_file_handle = open(log_file, 'a', encoding='utf-8')
            # 写入分隔符，表示新的运行开始
            self._write_to_file(f"\n{'='*80}\n")
            self._write_to_file(f"新的运行开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._write_to_file(f"{'='*80}\n")

    def _write_to_file(self, message: str):
        """写入日志文件"""
        if self.log_file_handle:
            self.log_file_handle.write(message)
            self.log_file_handle.flush()  # 立即刷新，确保实时写入

    def log(self, message: str, level: str = "INFO"):
        """
        记录日志（同时输出到控制台和文件）

        Args:
            message: 日志消息
            level: 日志级别（INFO, WARNING, ERROR）
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}\n"

        # 输出到控制台
        print(log_message, end='')

        # 写入文件
        self._write_to_file(log_message)

    def info(self, message: str):
        """记录信息日志"""
        self.log(message, "INFO")

    def warning(self, message: str):
        """记录警告日志"""
        self.log(message, "WARNING")

    def error(self, message: str):
        """记录错误日志"""
        self.log(message, "ERROR")

    def debug(self, message: str):
        """记录调试日志（详细的多行信息）"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 调试日志使用特殊格式，便于识别
        separator = "-" * 80
        log_message = f"[{timestamp}] [DEBUG] {separator}\n"
        log_message += f"[{timestamp}] [DEBUG] {message}\n"
        log_message += f"[{timestamp}] [DEBUG] {separator}\n"

        # 输出到控制台
        print(log_message, end='')

        # 写入文件
        self._write_to_file(log_message)

    def debug_dict(self, title: str, data: dict, max_length: int = 1000):
        """记录字典类型的调试信息"""
        import json
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        separator = "-" * 80

        # 格式化字典
        try:
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            # 如果太长，截断
            if len(formatted) > max_length:
                formatted = formatted[:max_length] + f"\n... (截断，总长度: {len(json.dumps(data, ensure_ascii=False))})"
        except:
            formatted = str(data)
            if len(formatted) > max_length:
                formatted = formatted[:max_length] + f"\n... (截断)"

        log_message = f"[{timestamp}] [DEBUG] {separator}\n"
        log_message += f"[{timestamp}] [DEBUG] {title}\n"
        log_message += f"[{timestamp}] [DEBUG] {separator}\n"
        log_message += f"{formatted}\n"
        log_message += f"[{timestamp}] [DEBUG] {separator}\n"

        # 输出到控制台
        print(log_message, end='')

        # 写入文件
        self._write_to_file(log_message)

    def close(self):
        """关闭日志文件"""
        if self.log_file_handle:
            self._write_to_file(f"\n{'='*80}\n")
            self._write_to_file(f"运行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._write_to_file(f"{'='*80}\n\n")
            self.log_file_handle.close()
            self.log_file_handle = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False


# 全局日志器实例（默认不使用文件）
_global_logger: Optional[Logger] = None


def set_global_logger(logger: Optional[Logger]):
    """设置全局日志器"""
    global _global_logger
    _global_logger = logger


def get_logger() -> Optional[Logger]:
    """获取全局日志器"""
    return _global_logger


def log_info(message: str):
    """记录信息日志（使用全局日志器）"""
    if _global_logger:
        _global_logger.info(message)
    else:
        print(f"[INFO] {message}")


def log_warning(message: str):
    """记录警告日志（使用全局日志器）"""
    if _global_logger:
        _global_logger.warning(message)
    else:
        print(f"[WARNING] {message}")


def log_error(message: str):
    """记录错误日志（使用全局日志器）"""
    if _global_logger:
        _global_logger.error(message)
    else:
        print(f"[ERROR] {message}")


def log_debug(message: str):
    """记录调试日志（使用全局日志器）"""
    if _global_logger:
        _global_logger.debug(message)
    else:
        print(f"[DEBUG] {message}")


def log_debug_dict(title: str, data: dict, max_length: int = 1000):
    """记录字典类型的调试信息（使用全局日志器）"""
    if _global_logger:
        _global_logger.debug_dict(title, data, max_length)
    else:
        import json
        print(f"[DEBUG] {title}")
        try:
            print(json.dumps(data, ensure_ascii=False, indent=2)[:max_length])
        except:
            print(str(data)[:max_length])
