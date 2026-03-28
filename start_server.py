#!/usr/bin/env python3
"""
启动脚本
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
root_dir = Path(__file__).parent
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded .env from {env_file}")
else:
    print(f"Warning: .env file not found at {env_file}")

# 添加 backend 和 src 到路径
backend_dir = Path(__file__).parent / "backend"
src_dir = Path(__file__).parent / "src"

sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(src_dir))

import uvicorn
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
