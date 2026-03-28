"""
配置管理模块
集中管理所有配置项
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from langchain_community.embeddings import TextEmbedEmbeddings

# 启动时加载环境变量
from dotenv import load_dotenv
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


class Settings(BaseModel):
    """应用配置"""

    # 应用配置
    APP_NAME: str = "Travel Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API 密钥 - 从环境变量读取
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DASHSCOPE_API_KEY: Optional[str] = os.getenv("DASHSCOPE_API_KEY")
    AMAP_MAPS_API_KEY: Optional[str] = os.getenv("AMAP_MAPS_API_KEY")

    # 豆包 Seedream 图像生成配置
    SEEDREAM_API_KEY: Optional[str] = os.getenv("SEEDREAM_API_KEY")
    SEEDREAM_BASE_URL: str = os.getenv("SEEDREAM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    # Agent 配置
    RECURSION_LIMIT: int = 25
    MAX_CONVERSATION_HISTORY: int = 20
    MAX_INPUT_LENGTH: int = 200
    MAX_OUTPUT_LENGTH: int = 300

    # RAG 配置
    RAG_CHROMA_PERSIST_DIR: Path = Field(default_factory=lambda: Path("./data/chroma_db"))
    RAG_EMBEDDING_MODEL: str = "text-embedding-v4"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 3
    RAG_COLLECTION_NAME: str = "travel_knowledge"

    # 知识库配置
    KNOWLEDGE_BASE_DIR: Path = Field(default_factory=lambda: Path("./data/knowledge_base"))
    KNOWLEDGE_RAW_DIR: Path = Field(default_factory=lambda: Path("./data/knowledge_base/raw"))
    KNOWLEDGE_PROCESSED_DIR: Path = Field(default_factory=lambda: Path("./data/knowledge_base/processed"))

    # CORS 配置
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: list[str] = Field(default_factory=lambda: ["*"])

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
