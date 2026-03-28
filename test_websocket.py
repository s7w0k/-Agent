#!/usr/bin/env python3
"""
后端 WebSocket 调试脚本
用于测试 Agent 执行和 token 统计功能
"""

import asyncio
import websockets
import json
import sys


async def test_websocket():
    """测试 WebSocket 接口"""
    uri = "ws://localhost:8000/ws"

    print(f"连接 WebSocket: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ 连接成功\n")

            # 发送测试消息
            test_message = "你好，请简单介绍一下你自己"
            print(f"发送消息：{test_message}\n")
            await websocket.send(test_message)

            # 接收并显示所有事件
            event_count = 0
            llm_count = 0
            token_total = 0

            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=60)
                    data = json.loads(response)
                    event_count += 1

                    event_type = data.get("type", "unknown")
                    print(f"返回 #{data}")
                    print(f"[{event_count}] 事件类型：{event_type}")

                    # 显示关键信息
                    if event_type == "received":
                        print(f"    消息：{data.get('message', '')}")

                    elif event_type == "llm_start":
                        llm_count += 1
                        print(f"    LLM 调用 #{llm_count}")

                    elif event_type == "llm_end":
                        tokens = data.get("tokens", {})
                        input_t = tokens.get("input_tokens", 0)
                        output_t = tokens.get("output_tokens", 0)
                        total_t = tokens.get("total_tokens", 0)
                        token_total += total_t
                        print(
                            f"    Token: {input_t} (输入) + {output_t} (输出) = {total_t}"
                        )
                        if data.get("has_tool_calls"):
                            print(f"    决定调用工具：{data.get('tool_names', [])}")

                    elif event_type == "tool_start":
                        print(f"    工具：{data.get('tool_name', 'unknown')}")
                        print(f"    输入：{data.get('input', '')[:100]}...")

                    elif event_type == "tool_end":
                        print(f"    工具完成：{data.get('tool_name', 'unknown')}")

                    elif event_type == "step_start":
                        print(
                            f"    步骤 {data.get('step', '?')}: 节点 [{data.get('node', 'unknown')}]"
                        )

                    elif event_type == "step_end":
                        print(f"    节点 [{data.get('node', 'unknown')}] 完成")

                    elif event_type == "complete":
                        print(f"    ✓ 执行完成")
                        print(f"    LLM 调用：{data.get('llm_calls', 0)} 次")
                        print(f"    工具调用：{data.get('tool_calls', 0)} 次")
                        print(f"    步骤：{data.get('total_steps', 0)}")
                        print(f"    耗时：{data.get('duration', 0):.2f}s")

                    elif event_type == "final":
                        print(f"    ✓ 最终结果")
                        print(f"    内容长度：{len(data.get('content', ''))} 字符")
                        print("\n" + "=" * 60)
                        print("📊 统计汇总:")
                        print(f"    总事件数：{event_count}")
                        print(f"    LLM 调用：{llm_count} 次")
                        print(f"    Token 总计：{token_total}")
                        print("=" * 60)
                        return

                    elif event_type == "error":
                        print(f"    ❌ 错误：{data.get('message', '')}")
                        return

                    print()

                except asyncio.TimeoutError:
                    print("❌ 等待响应超时")
                    return

    except websockets.exceptions.ConnectionClosed:
        print("❌ 连接已关闭")
    except Exception as e:
        print(f"❌ 错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 后端 WebSocket 调试工具")
    print("=" * 60)
    print()
    asyncio.run(test_websocket())
