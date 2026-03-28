"""
MCP 工具管理器
使用 LangChain MCP Adapters 自动发现和加载工具
"""

import os
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from __init__ import get_logger


class MCPToolsManager:
    """MCP 工具管理器"""

    DEFAULT_CONFIGS = {
        "12306": {
            "command": "npx",
            "args": ["-y", "12306-mcp"],
            "transport": "stdio",
        },
        "amap": {
            "command": "npx",
            "args": ["-y", "@amap/amap-maps-mcp-server"],
            "transport": "stdio",
            "env": {"AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY", "")},
        },
    }

    def __init__(self):
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List = []
        self.logger = get_logger("mcp_tools")

    async def initialize(self, mcp_configs: Optional[Dict[str, Any]] = None) -> List:
        """初始化 MCP 客户端并获取工具"""
        self.logger.info("=" * 50)
        self.logger.info("开始初始化 MCP 工具管理器")

        configs = mcp_configs or self.DEFAULT_CONFIGS

        self.logger.info(f"已加载 {len(configs)} 个 MCP 服务器配置:")
        for name, config in configs.items():
            cmd = f"{config.get('command', '')} {' '.join(config.get('args', []))}"
            self.logger.info(f"  - {name}: {cmd}")

        try:
            self.client = MultiServerMCPClient(configs)
            self.tools = await self.client.get_tools()

            self.logger.info(f"已加载 {len(self.tools)} 个工具:")
            for tool in self.tools:
                self.logger.info(f"  - {tool.name}")

            self.logger.info("MCP 工具管理器初始化成功")
            self.logger.info("=" * 50)

            return self.tools

        except Exception as e:
            self.logger.error(f"MCP 工具管理器初始化失败：{e}", exc_info=True)
            raise

    def get_tools(self) -> List:
        """获取所有工具"""
        return self.tools

    def get_tools_by_server(self, server_name: str) -> List:
        """获取指定服务器的工具"""
        prefix = f"{server_name}__"
        return [t for t in self.tools if t.name.startswith(prefix)]

    async def close(self) -> None:
        """关闭 MCP 连接"""
        self.logger.info("关闭 MCP 连接")
        self.client = None
