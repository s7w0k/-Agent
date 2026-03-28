"""
FastAPI 后端服务
提供 WebSocket 接口用于实时 Agent 执行
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_settings, settings
from backend.schemas import ConversationMessage
from agent_manager import AgentManager
from monitor_handler import WSMonitor
from logger import get_logger

# 初始化日志
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("应用启动，开始初始化 Agent...")
    agent_manager = AgentManager.get_instance()
    success = await agent_manager.initialize()
    if success:
        logger.info("Agent 初始化成功，服务就绪")
    else:
        logger.error("Agent 初始化失败，服务可能不可用")

    yield

    # 关闭时清理
    logger.info("应用关闭，清理资源...")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # 注册路由
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """注册应用路由"""
    app.get("/")(root)
    app.get("/api/health")(health_check)
    app.websocket("/ws")(websocket_endpoint)

    # 注册 RAG 路由
    from backend.rag.api_routes import router as rag_router
    app.include_router(rag_router)

    # 注册多 Agent 路由
    from backend.agents.api_routes import router as agent_router
    app.include_router(agent_router)

    # 挂载静态文件
    mount_static_files(app)


def mount_static_files(app: FastAPI) -> None:
    """挂载前端静态文件"""
    backend_dir = Path(__file__).parent
    frontend_dist = backend_dir.parent / "frontend" / "dist"

    if frontend_dist.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_dist), html=True),
            name="static",
        )
        logger.info(f"已挂载前端静态文件：{frontend_dist}")


async def root() -> dict:
    """健康检查 - 根路径"""
    logger.info("健康检查请求")
    return {"status": "ok", "message": f"{settings.APP_NAME} is running"}


async def health_check() -> dict:
    """健康检查 - API 路径"""
    agent_manager = AgentManager.get_instance()
    if agent_manager.is_ready():
        return {
            "status": "healthy",
            "agent_ready": True,
            "message": "服务运行正常",
        }
    else:
        return {
            "status": "initializing",
            "agent_ready": False,
            "message": "Agent 正在初始化中",
        }


async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket 聊天接口"""
    await websocket.accept()
    logger.info(f"WebSocket 连接建立：{websocket.client}")

    # 初始化对话历史
    conversation_history: list[ConversationMessage] = []
    agent_manager = AgentManager.get_instance()

    try:
        while True:
            await handle_client_message(
                websocket=websocket,
                conversation_history=conversation_history,
                agent_manager=agent_manager,
            )

    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket 错误：{e}", exc_info=True)
        await safe_send_error(websocket, e)


async def handle_client_message(
    websocket: WebSocket,
    conversation_history: list[ConversationMessage],
    agent_manager: AgentManager,
) -> None:
    """处理客户端消息"""
    # 接收用户消息（支持 JSON 格式或纯文本）
    raw_message = await websocket.receive_text()
    logger.info(f"收到原始消息：{raw_message[:100]}...")

    # 尝试解析 JSON
    user_input = raw_message
    try:
        import json
        data = json.loads(raw_message)
        if isinstance(data, dict):
            user_input = data.get("message", raw_message)
    except:
        pass

    logger.info(f"解析后用户消息：{user_input[:50]}...")

    # 发送确认
    await websocket.send_json(
        {
            "type": "received",
            "message": "开始处理...",
        }
    )

    # 获取 Agent
    agent = await agent_manager.get_agent()

    # RAG 检索（添加知识库上下文）
    context = await retrieve_context(user_input)
    if context:
        logger.info(f"RAG 检索到 {len(context)} 字符的上下文")
        await websocket.send_json({
            "type": "rag_context",
            "content": context[:500] + "..." if len(context) > 500 else context
        })

    # 构建输入消息（包含 RAG 上下文）
    input_messages = build_input_messages(
        user_input=user_input,
        conversation_history=conversation_history,
        context=context,
    )

    # 执行并流式传输
    await execute_and_stream(
        websocket=websocket,
        agent=agent,
        input_messages=input_messages,
        conversation_history=conversation_history,
        user_input=user_input,
    )


def build_input_messages(
    user_input: str,
    conversation_history: list[ConversationMessage],
    context: str = "",
) -> list[dict]:
    """构建输入消息列表

    Args:
        user_input: 用户输入
        conversation_history: 对话历史
        context: RAG 检索到的上下文
    """
    from skill_loader import SYSTEM_PROMPT

    # 构建系统提示词
    system_content = SYSTEM_PROMPT

    # 添加 RAG 上下文
    if context:
        system_content += f"\n\n## 相关知识库内容\n{context}\n"

    messages = [{"role": "system", "content": system_content}]
    messages.extend([msg.model_dump() for msg in conversation_history])
    messages.append({"role": "user", "content": user_input})

    return messages


async def retrieve_context(user_input: str) -> str:
    """检索知识库上下文

    Args:
        user_input: 用户输入

    Returns:
        格式化的检索结果
    """
    try:
        from backend.rag.rag_tool import get_rag_tool
        rag_tool = get_rag_tool()

        # 搜索知识库（同时搜索小红书和全网）
        results = await rag_tool.search(
            query=user_input,
            top_k=3,
            sources=None  # 搜索所有来源
        )

        return results
    except Exception as e:
        logger.warning(f"RAG 检索失败: {e}")
        return ""


async def execute_and_stream(
    websocket: WebSocket,
    agent,
    input_messages: list[dict],
    conversation_history: list[ConversationMessage],
    user_input: str,
) -> None:
    """执行 Agent 并流式传输结果"""
    logger.info(f"开始流式执行 MultiAgentGraph... (历史消息数：{len(conversation_history)})")

    try:
        # 提取用户请求（从消息历史中获取最后一条用户消息）
        actual_user_input = user_input
        if input_messages:
            for msg in reversed(input_messages):
                if msg.get("role") == "user":
                    actual_user_input = msg.get("content", user_input)
                    break

        final_content = None
        event_count = 0

        # 使用 MultiAgentGraph 的流式事件
        async for event in agent.stream_events(
            user_request=actual_user_input,
            enable_search=True,
            enable_visualization=True,
            style="friendly",
        ):
            await websocket.send_json(event)
            event_count += 1

            if event.get("type") == "complete":
                # 从事件中获取最终内容
                final_content = event.get("final_content", "")

        logger.info(f"Agent 执行完成，共 {event_count} 个事件")

        # 处理最终结果
        if final_content:
            await websocket.send_json(
                {
                    "type": "final",
                    "content": final_content,
                }
            )
            logger.info(f"发送最终结果，长度：{len(final_content)}")

            # 更新对话历史
            update_conversation_history(
                conversation_history=conversation_history,
                user_input=user_input,
                ai_response=final_content,
            )
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "未获取到响应",
                }
            )
            logger.warning("未获取到 Agent 响应")

    except asyncio.CancelledError:
        logger.warning("Agent 执行被取消（客户端断开连接）")
        raise
    except Exception as e:
        logger.error(f"Agent 执行出错：{e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": str(e)})


def update_conversation_history(
    conversation_history: list[ConversationMessage],
    user_input: str,
    ai_response: str,
) -> None:
    """更新对话历史"""
    conversation_history.append(ConversationMessage(role="user", content=user_input))
    conversation_history.append(
        ConversationMessage(role="assistant", content=ai_response)
    )

    # 限制历史长度
    max_history = settings.MAX_CONVERSATION_HISTORY
    if len(conversation_history) > max_history:
        conversation_history = conversation_history[-max_history:]

    logger.info(f"当前对话历史：{len(conversation_history)} 条消息")


async def safe_send_error(websocket: WebSocket, error: Exception) -> None:
    """安全地发送错误消息"""
    try:
        await websocket.send_json({"type": "error", "message": str(error)})
    except Exception:
        pass


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
