"""
类型定义模块
提供所有类型的集中定义
"""

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """对话消息"""

    role: str
    content: str


class AgentConfig(BaseModel):
    """Agent 配置"""

    recursion_limit: int = 25
    max_history: int = 20
