"""
Travel Agent - 旅行规划助手
基于 LangGraph 的异步 Agent 实现
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加路径
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from monitor import AgentRealtimeMonitor
from prompts import SYSTEM_PROMPT


class TravelAgent:
    """旅行规划 Agent"""

    def __init__(
        self,
        api_key: str,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        enable_monitor: bool = True,
    ):
        self.api_key = api_key
        self.tools = tools
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.enable_monitor = enable_monitor

        # 初始化组件
        self.llm = self._create_llm()
        self.agent = self._create_agent()
        self.monitor: Optional[AgentRealtimeMonitor] = None

        if self.enable_monitor:
            self.monitor = AgentRealtimeMonitor()

    def _create_llm(self) -> ChatDeepSeek:
        """创建 LLM 实例"""
        return ChatDeepSeek(
            model="deepseek-chat",
            api_key=self.api_key,
            temperature=0.7,
            max_tokens=4096,
        )

    def _create_agent(self):
        """创建 Agent 实例"""
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    async def ainvoke(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        recursion_limit: int = 15,
    ) -> str:
        """异步调用 Agent

        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            recursion_limit: 最大递归次数

        Returns:
            Agent 回复内容
        """
        # 构建消息
        input_messages = self._build_messages(user_input, conversation_history)

        # 配置
        config: RunnableConfig = {"recursion_limit": recursion_limit}

        # 执行
        if self.monitor:
            result = await self.monitor.stream(
                agent=self.agent,
                input_data={"messages": input_messages},
                config=dict(config),
            )
        else:
            result = await self.agent.ainvoke(
                {"messages": input_messages}, config=config
            )

        # 获取回复
        messages = result.get("messages", [])
        return messages[-1].content if messages else ""

    def _build_messages(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []

        # 系统提示
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 历史对话
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ("user", "assistant"):
                        messages.append({"role": role, "content": content})

        # 当前输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def invoke(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        recursion_limit: int = 15,
    ) -> str:
        """同步调用 Agent"""
        return asyncio.run(
            self.ainvoke(user_input, conversation_history, recursion_limit)
        )

    def get_monitor_summary(self) -> Optional[Dict[str, Any]]:
        """获取监控摘要"""
        return self.monitor.get_summary() if self.monitor else None


def create_travel_agent(
    api_key: str,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    enable_monitor: bool = True,
) -> TravelAgent:
    """创建 TravelAgent 实例

    Args:
        api_key: DeepSeek API Key
        tools: 工具列表
        system_prompt: 系统提示词
        enable_monitor: 是否启用监控

    Returns:
        TravelAgent 实例
    """
    return TravelAgent(
        api_key=api_key,
        tools=tools,
        system_prompt=system_prompt,
        enable_monitor=enable_monitor,
    )
