"""
日志配置模块
所有日志记录到同一个按日期分类的文件中
"""

import logging
from pathlib import Path
from datetime import datetime

# 设置 httpx 日志级别为 WARNING，避免过多的调试信息
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)


def get_logger(name: str = __name__) -> logging.Logger:
    """获取 logger 实例

    Args:
        name: logger 名称（仅用于标识日志来源）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 如果已经有 handler，直接返回
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 创建日志目录
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # 生成日志文件名（按日期）
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"backend_{date_str}.log"

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # 添加处理器
    logger.addHandler(file_handler)

    return logger
