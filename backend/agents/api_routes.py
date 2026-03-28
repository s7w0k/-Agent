"""
多 Agent 系统 API 路由
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量 - 从项目根目录
root_dir = Path(__file__).parent.parent.parent
env_file = root_dir / ".env"
print(f"加载环境变量 from: {env_file}, exists: {env_file.exists()}")
if env_file.exists():
    load_dotenv(env_file, override=True)

# 验证环境变量
print(f"DEEPSEEK_API_KEY loaded: {bool(os.getenv('DEEPSEEK_API_KEY'))}")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multi-agent", tags=["多Agent"])


class MultiAgentRequest(BaseModel):
    """多 Agent 请求"""
    message: str
    enable_search: bool = True
    enable_visualization: bool = True
    style: str = "friendly"  # friendly / professional / fun


class MultiAgentResponse(BaseModel):
    """多 Agent 响应"""
    guide_content: str
    route_map: Optional[dict] = None
    travel_plan: Optional[dict] = None
    search_context: str = ""


@router.post("/chat", response_model=MultiAgentResponse)
async def multi_agent_chat(request: MultiAgentRequest):
    """多 Agent 聊天接口

    处理流程：
    1. Search Agent: 搜索 + RAG 检索
    2. Planner Agent: 生成旅行计划
    3. Writer Agent: 生成攻略文本
    4. Visualization Agent: 生成路线图（可选）
    """
    try:
        # 导入并获取协调器（单例）
        from backend.agents.coordinator import get_coordinator

        # 获取协调器实例
        coordinator = get_coordinator()

        # 处理请求
        result = await coordinator.process(
            user_request=request.message,
            enable_search=request.enable_search,
            enable_visualization=request.enable_visualization,
            style=request.style,
        )

        return MultiAgentResponse(
            guide_content=result.guide_content,
            route_map=result.route_map,
            travel_plan=result.travel_plan,
            search_context=result.search_context,
        )

    except Exception as e:
        logger.error(f"多 Agent 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def agent_status():
    """检查 Agent 状态"""
    try:
        from backend.agents.coordinator import get_coordinator
        coordinator = get_coordinator()
        
        # 获取内部 graph 的 agents
        graph = coordinator._graph

        return {
            "status": "ready",
            "agents": {
                "search": graph.search_agent is not None,
                "planner": graph.planner_agent is not None,
                "writer": graph.writer_agent is not None,
                "visualization": graph.visualization_agent is not None,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
