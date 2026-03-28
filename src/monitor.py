"""
Agent 实时监控器
使用 LangGraph 的 stream_events API 实时追踪执行过程
"""

import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from __init__ import get_logger


class AgentRealtimeMonitor:
    """Agent 实时监控器

    实时监控 LangGraph 的执行过程：
    - 显示每个节点的执行
    - 显示工具调用的开始和结束
    - 显示 LLM 调用的开始和结束
    - 实时输出到日志和控制台
    """

    def __init__(self, logger=None, show_console: bool = True):
        self.logger = logger or get_logger("agent_monitor")
        self.show_console = show_console

        # 统计信息
        self.start_time: Optional[float] = None
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.current_step = 0

        # 事件记录
        self.events = []

    def _print(self, message: str, level: str = "info"):
        """同时输出到日志和控制台"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"

        if level == "info":
            self.logger.info(formatted)
        elif level == "error":
            self.logger.error(formatted)

        if self.show_console:
            print(formatted)

    async def stream(
        self, agent, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """流式执行 Agent 并实时监控

        Args:
            agent: 编译后的 Agent 图
            input_data: 输入数据
            config: 运行配置

        Returns:
            最终结果
        """
        self.start_time = time.time()
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.current_step = 0

        self._print("=" * 60, "info")
        self._print("🚀 Agent 执行开始", "info")
        self._print("=" * 60, "info")

        try:
            # 使用 astream 执行并收集结果，同时使用 astream_events 监控事件
            # astream_events 会自动执行图，不需要再调用 ainvoke
            final_result = None

            async for event in agent.astream_events(
                input=input_data,
                config=config or {},
                version="v2",
            ):
                await self._handle_event(event)

                # 从 on_chain_end 事件中提取最终结果
                if event.get("event") == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if output:
                        final_result = output

            if not final_result:
                # 如果 astream_events 没有返回结果，使用 ainvoke 获取（备用方案）
                # 但这不应该发生，如果发生了说明有问题
                self._print("⚠️ 警告：astream_events 未返回结果，使用备用方案", "error")
                final_result = await agent.ainvoke(input_data, config=config)

            self._print("=" * 60, "info")
            self._print(
                f"✅ Agent 执行完成 | 总耗时：{time.time() - self.start_time:.2f}s",
                "info",
            )
            self._print(
                f"📊 统计：LLM 调用={self.llm_call_count} 次 | 工具调用={self.tool_call_count} 次",
                "info",
            )
            self._print("=" * 60, "info")

            return final_result

        except Exception as e:
            self._print(f"❌ 执行失败：{e}", "error")
            raise

    async def _handle_event(self, event: Dict[str, Any]):
        """处理单个事件"""
        kind = event.get("event")
        metadata = event.get("metadata", {})
        name = metadata.get("langgraph_node", "unknown")

        # 忽略某些内部事件
        if name.startswith("__"):
            return

        self.events.append(event)

        # 根据事件类型处理
        if kind == "on_chain_start":
            await self._on_chain_start(metadata)

        elif kind == "on_chain_end":
            await self._on_chain_end(metadata, event.get("data", {}))

        elif kind == "on_chat_model_start":
            await self._on_llm_start(metadata)

        elif kind == "on_chat_model_end":
            await self._on_llm_end(metadata, event.get("data", {}))

        elif kind == "on_tool_start":
            await self._on_tool_start(metadata, event.get("data", {}))

        elif kind == "on_tool_end":
            await self._on_tool_end(metadata, event.get("data", {}))

    async def _on_chain_start(self, metadata: Dict[str, Any]):
        """节点开始执行"""
        node_name = metadata.get("langgraph_node", "unknown")
        if node_name and not node_name.startswith("__"):
            self.current_step += 1
            self._print(
                f"\n📍 步骤 {self.current_step}: 执行节点 [{node_name}]", "info"
            )

    async def _on_chain_end(self, metadata: Dict[str, Any], data: Dict[str, Any]):
        """节点执行结束"""
        node_name = metadata.get("langgraph_node", "unknown")
        if node_name and not node_name.startswith("__"):
            output = data.get("output", {})
            if isinstance(output, dict):
                messages = output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    msg_type = type(last_msg).__name__
                    self._print(
                        f"   ✅ 节点 [{node_name}] 完成 | 输出：{msg_type}", "info"
                    )

    async def _on_llm_start(self, metadata: Dict[str, Any]):
        """LLM 调用开始"""
        self.llm_call_count += 1
        self._print(f"   🤖 第 {self.llm_call_count} 次 LLM 调用开始...", "info")

    async def _on_llm_end(self, metadata: Dict[str, Any], data: Dict[str, Any]):
        """LLM 调用结束"""
        output = data.get("output", {})

        # 提取 token 使用
        token_info = ""
        if hasattr(output, "response_metadata"):
            usage = output.response_metadata.get("usage", {})
            if usage:
                token_info = f" | Token: {usage.get('prompt_tokens', 0)}→{usage.get('completion_tokens', 0)}"

        # 检查是否有工具调用
        if hasattr(output, "tool_calls") and output.tool_calls:
            self._print(
                f"   ✅ LLM 完成 | 决定调用 {len(output.tool_calls)} 个工具{token_info}",
                "info",
            )
            for tc in output.tool_calls:
                tool_name = tc.get("name", "unknown")
                self._print(f"      → 工具：{tool_name}", "info")
        else:
            self._print(f"   ✅ LLM 完成 | 生成最终回答{token_info}", "info")

    async def _on_tool_start(self, metadata: Dict[str, Any], data: Dict[str, Any]):
        """工具调用开始"""
        self.tool_call_count += 1
        tool_name = metadata.get("langgraph_node", "unknown")
        input_data = data.get("input", {})

        # 格式化输入
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data, ensure_ascii=False, default=str)
        else:
            input_str = str(input_data)

        # 截断长输入
        if len(input_str) > 150:
            input_str = input_str[:150] + "..."

        self._print(f"   🔧 第 {self.tool_call_count} 次工具调用：{tool_name}", "info")
        self._print(f"      参数：{input_str}", "info")

    async def _on_tool_end(self, metadata: Dict[str, Any], data: Dict[str, Any]):
        """工具调用结束"""
        tool_name = metadata.get("langgraph_node", "unknown")
        output = data.get("output", "")

        # 格式化输出
        if isinstance(output, str) and len(output) > 200:
            output = output[:200] + "..."
        elif isinstance(output, dict):
            output = json.dumps(output, ensure_ascii=False, default=str)[:200] + "..."

        self._print(f"   ✅ 工具完成：{tool_name}", "info")
        self._print(f"      返回：{output}", "info")

    def get_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat()
            if self.start_time
            else None,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "llm_calls": self.llm_call_count,
            "tool_calls": self.tool_call_count,
            "total_steps": self.current_step,
        }
