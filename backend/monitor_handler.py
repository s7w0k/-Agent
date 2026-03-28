"""
Monitor 事件处理器 - 将监控事件转换为前端可用的格式
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

# 添加 src 到路径
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from monitor import AgentRealtimeMonitor
from logger import get_logger
from config import settings

logger = get_logger(__name__)


class EventConverter:
    """事件转换器 - 将 LangGraph 事件转换为前端格式"""

    def __init__(self):
        """初始化计数器"""
        self.step_count = 0
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def convert(self, event: dict[str, Any]) -> Optional[dict[str, Any]]:
        """转换单个事件

        Args:
            event: LangGraph 原始事件

        Returns:
            转换后的事件字典，如果应该忽略则返回 None
        """
        kind = event.get("event")
        metadata = event.get("metadata", {})
        data = event.get("data", {})

        # 忽略内部节点
        if self._is_internal_node(metadata):
            return None

        # 事件分发
        handlers = {
            "on_chain_start": self._handle_chain_start,
            "on_chain_end": self._handle_chain_end,
            "on_chat_model_start": self._handle_chat_model_start,
            "on_chat_model_end": self._handle_chat_model_end,
            "on_tool_start": self._handle_tool_start,
            "on_tool_end": self._handle_tool_end,
        }

        handler = handlers.get(kind)
        if handler:
            return handler(metadata, data)

        return None

    def _is_internal_node(self, metadata: dict[str, Any]) -> bool:
        """检查是否为内部节点"""
        node_name = metadata.get("langgraph_node", "")
        return node_name.startswith("__")

    def _handle_chain_start(
        self, metadata: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理链开始事件"""
        self.step_count += 1
        node_name = metadata.get("langgraph_node", "unknown")

        return {
            "type": "step_start",
            "step": self.step_count,
            "node": node_name,
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_chain_end(
        self, metadata: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理链结束事件"""
        output = data.get("output", {})
        msg_type = self._get_message_type(output)
        node_name = metadata.get("langgraph_node", "unknown")

        return {
            "type": "step_end",
            "step": self.step_count,
            "node": node_name,
            "output_type": msg_type,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_message_type(self, output: Any) -> str:
        """获取消息类型"""
        if isinstance(output, dict):
            messages = output.get("messages", [])
            if messages:
                last_msg = messages[-1]
                return type(last_msg).__name__
        return "unknown"

    def _handle_chat_model_start(
        self, metadata: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理聊天模型开始事件"""
        self.llm_call_count += 1

        return {
            "type": "llm_start",
            "call_number": self.llm_call_count,
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_chat_model_end(
        self, metadata: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理聊天模型结束事件"""
        token_info = self._extract_token_info(metadata)
        self._update_token_stats(token_info)

        return {
            "type": "llm_end",
            "call_number": self.llm_call_count,
            "tokens": token_info,
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_token_info(self, metadata: dict[str, Any]) -> dict[str, int]:
        """从元数据中提取 token 信息"""
        usage = metadata.get("usage_metadata", {})
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _update_token_stats(self, token_info: dict[str, int]) -> None:
        """更新 token 统计"""
        self.input_tokens += token_info.get("input_tokens", 0)
        self.output_tokens += token_info.get("output_tokens", 0)
        self.total_tokens = self.input_tokens + self.output_tokens

    def _handle_tool_start(
        self, metadata: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理工具开始事件"""
        self.tool_call_count += 1
        input_data = data.get("input", {})
        input_str = self._format_input(input_data)
        node_name = metadata.get("langgraph_node", "unknown")

        return {
            "type": "tool_start",
            "call_number": self.tool_call_count,
            "tool_name": node_name,
            "input": input_str,
            "timestamp": datetime.now().isoformat(),
        }

    def _format_input(self, input_data: Any) -> str:
        """格式化输入数据"""
        if isinstance(input_data, dict):
            result = json.dumps(input_data, ensure_ascii=False, default=str)
        else:
            result = str(input_data)

        # 截断过长的输入
        max_length = settings.MAX_INPUT_LENGTH
        if len(result) > max_length:
            result = result[:max_length] + "..."

        return result

    def _handle_tool_end(
        self, metadata: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理工具结束事件"""
        output = self._format_output(data.get("output", ""))
        node_name = metadata.get("langgraph_node", "unknown")

        return {
            "type": "tool_end",
            "call_number": self.tool_call_count,
            "tool_name": node_name,
            "output": output,
            "timestamp": datetime.now().isoformat(),
        }

    def _format_output(self, output: Any) -> str:
        """格式化输出数据"""
        # 处理 ToolMessage 对象
        if self._is_tool_message(output):
            output = str(output.content) if hasattr(output, "content") else str(output)

        # 字符串截断
        if isinstance(output, str) and len(output) > settings.MAX_OUTPUT_LENGTH:
            output = output[: settings.MAX_OUTPUT_LENGTH] + "..."

        # 字典转 JSON
        if isinstance(output, dict):
            output = (
                json.dumps(output, ensure_ascii=False, default=str)[
                    : settings.MAX_OUTPUT_LENGTH
                ]
                + "..."
            )

        return str(output)

    def _is_tool_message(self, output: Any) -> bool:
        """检查是否为 ToolMessage 对象"""
        return (
            hasattr(output, "__class__") and output.__class__.__name__ == "ToolMessage"
        )

    def get_stats(self) -> dict[str, int]:
        """获取统计信息"""
        return {
            "llm_calls": self.llm_call_count,
            "tool_calls": self.tool_call_count,
            "steps": self.step_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class WSMonitor(AgentRealtimeMonitor):
    """WebSocket 监控器 - 生成事件流"""

    def __init__(self, logger=None, show_console: bool = False):
        super().__init__(logger, show_console)
        self.converter = EventConverter()
        self.final_content: Optional[str] = None
        self.start_time: float = 0

    async def stream(
        self,
        agent: Any,
        input_data: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式执行并生成事件

        Yields:
            监控事件字典
        """
        self._reset_stats()
        logger.debug("开始 stream 执行")

        # 发送开始事件
        yield self._create_start_event()

        try:
            async for event in agent.astream_events(
                input=input_data,
                config=config or {},
                version="v2",
            ):
                # 转换并发送事件
                ui_event = self.converter.convert(event)
                if ui_event:
                    yield ui_event

                # 提取最终回复
                self._extract_final_content(event)

            # 发送完成事件
            yield self._create_complete_event()

        except Exception as e:
            logger.error(f"stream 出错：{e}", exc_info=True)
            yield self._create_error_event(e)
            raise

    def _reset_stats(self) -> None:
        """重置统计信息"""
        self.start_time = time.time()
        self.final_content = None
        self.converter = EventConverter()

    def _create_start_event(self) -> dict[str, Any]:
        """创建开始事件"""
        return {
            "type": "start",
            "timestamp": datetime.now().isoformat(),
            "message": "Agent 执行开始",
        }

    def _extract_final_content(self, event: dict[str, Any]) -> None:
        """从事件中提取最终回复"""
        if event.get("event") != "on_chat_model_end":
            return

        output = event.get("data", {}).get("output", {})
        if not output:
            return

        # 检查是否为最终回复（没有工具调用）
        has_tool_calls = hasattr(output, "tool_calls") and output.tool_calls
        if has_tool_calls:
            return

        content = getattr(output, "content", None)
        if content:
            self.final_content = content
            logger.debug(f"获取到最终回复，长度：{len(content)}")

    def _create_complete_event(self) -> dict[str, Any]:
        """创建完成事件"""
        stats = self.converter.get_stats()
        logger.info(
            f"执行完成 | LLM 调用：{stats['llm_calls']} | "
            f"工具调用：{stats['tool_calls']} | "
            f"步骤：{stats['steps']} | "
            f"Token: {stats['total_tokens']} (输入:{stats['input_tokens']} + 输出:{stats['output_tokens']})"
        )

        return {
            "type": "complete",
            "timestamp": datetime.now().isoformat(),
            "llm_calls": stats["llm_calls"],
            "tool_calls": stats["tool_calls"],
            "total_steps": stats["steps"],
            "duration": time.time() - self.start_time,
            "final_content": self.final_content,
            "total_tokens": stats["total_tokens"],
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
        }

    def _create_error_event(self, error: Exception) -> dict[str, Any]:
        """创建错误事件"""
        return {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "message": str(error),
        }

    async def execute_and_stream(
        self,
        agent: Any,
        input_data: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """执行并返回最终结果（用于不通过 WebSocket 的场景）

        Returns:
            最终回复内容
        """
        async for _ in self.stream(agent, input_data, config):
            pass
        return self.final_content
