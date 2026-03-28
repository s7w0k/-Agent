"""
Agent 管理器模块
单例模式管理 Agent 实例
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
backend_dir = Path(__file__).parent
root_dir = backend_dir.parent
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


class AgentManager:
    """Agent 管理器（单例模式）"""

    _instance: Optional["AgentManager"] = None
    _initialized: bool = False

    def __init__(self):
        if not self._initialized:
            self.agent_instance = None
            self.tools_manager = None
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "AgentManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> bool:
        """初始化 Agent（启动时调用）

        Returns:
            是否初始化成功
        """
        if self.agent_instance is not None:
            logger.info("Agent 已经初始化")
            return True

        try:
            await self._initialize_agent()
            logger.info("Agent 启动初始化成功")
            return True
        except Exception as e:
            logger.error(f"Agent 启动初始化失败：{e}")
            return False

    async def get_agent(self):
        """获取或创建 Agent 实例"""
        if self.agent_instance is None:
            await self._initialize_agent()
        return self.agent_instance

    async def _initialize_agent(self) -> None:
        """初始化 Agent（使用 LangGraph 多 Agent 架构）"""
        from mcp_tools import MCPToolsManager

        logger.info("初始化 LangGraph Multi-Agent...")

        # 初始化工具（带错误处理）
        try:
            self.tools_manager = MCPToolsManager()
            tools = await self.tools_manager.initialize()
            logger.info(f"初始化工具完成：{len(tools)} 个工具")
        except Exception as e:
            logger.warning(f"MCP 工具初始化失败：{e}，将使用无工具模式")
            tools = []

        # 创建 Agent
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY 未配置")
            raise ValueError("DEEPSEEK_API_KEY 未配置")

        # 使用 LangGraph 多 Agent 架构
        from backend.agents.coordinator import MultiAgentGraph
        self.agent_instance = MultiAgentGraph(
            api_key=api_key,
            tools=tools or [],
        )
        logger.info("MultiAgentGraph 实例创建完成")

    def is_ready(self) -> bool:
        """检查 Agent 是否就绪"""
        return self.agent_instance is not None

    def reset(self) -> None:
        """重置 Agent 实例"""
        self.agent_instance = None
        self.tools_manager = None
        logger.info("Agent 实例已重置")
