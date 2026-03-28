#!/usr/bin/env python3
"""
测试上下文记忆功能
验证 AI 是否能记住之前的对话内容
"""

import asyncio
import websockets
import json


async def test_context():
    """测试多轮对话的上下文记忆"""
    uri = "ws://localhost:8000/ws"

    print("=" * 60)
    print("🧠 测试上下文记忆功能")
    print("=" * 60)
    print()

    async with websockets.connect(uri) as websocket:
        print("✅ 连接成功\n")

        # 第一轮对话
        print("-" * 60)
        print("第一轮：询问 AI 的名字")
        print("-" * 60)
        await websocket.send("你好，请介绍一下你自己")

        response = await websocket.recv()
        data = json.loads(response)
        print(f"收到：{data.get('type')}")

        # 等待最终回复
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("type") == "final":
                content = data.get("content", "")
                print(f"\nAI 回复：{content[:100]}...\n")
                break

        # 第二轮对话 - 测试上下文
        print("-" * 60)
        print("第二轮：询问刚才提到的信息（测试上下文）")
        print("-" * 60)
        await websocket.send("你刚才说你擅长什么？")

        response = await websocket.recv()
        data = json.loads(response)
        print(f"收到：{data.get('type')}")

        # 等待最终回复
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("type") == "final":
                content = data.get("content", "")
                print(f"\nAI 回复：{content[:200]}...\n")

                # 检查是否包含上下文信息
                if "擅长" in content or "专长" in content or "旅行" in content:
                    print("✅ 上下文记忆正常！AI 记住了之前的对话内容")
                else:
                    print("⚠️  AI 可能没有记住上下文")
                break

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_context())
