"""
MultiAgent Coordinator - LangGraph 版本
基于 LangGraph 状态图协调 Search、Planner、Writer、Visualization 四个 Agent
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.logger import get_logger
from backend.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


# ==================== 状态定义 ====================

class AgentState(BaseModel):
    """LangGraph 状态定义"""
    # 输入
    user_request: str = ""
    enable_search: bool = True
    enable_visualization: bool = True
    style: str = "friendly"
    
    # 中间状态
    search_context: str = ""
    travel_plan: Optional[Dict] = None
    guide_content: str = ""
    route_map: Optional[Dict] = None
    
    # 执行追踪
    current_node: str = ""
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent 响应"""
    guide_content: str
    route_map: Optional[Dict] = None
    search_context: str = ""
    travel_plan: Optional[Dict] = None


# ==================== LangGraph 节点 ====================

class MultiAgentGraph:
    """LangGraph 多 Agent 编排图"""
    
    def __init__(self, api_key: str, tools: List[Any] = None):
        self.api_key = api_key
        self.tools = tools or []
        self.search_agent = None
        self.planner_agent = None
        self.writer_agent = None
        self.visualization_agent = None
        self._graph = None
        self._init_agents()
    
    def _init_agents(self):
        """初始化所有 Agent"""
        from .search_agent import get_search_agent
        from .planner_agent import get_planner_agent
        from .writer_agent import get_writer_agent
        
        logger.info("初始化 LangGraph Multi-Agent...")
        
        self.search_agent = get_search_agent(self.api_key)
        self.planner_agent = get_planner_agent(self.api_key, self.tools)
        self.writer_agent = get_writer_agent(self.api_key)
        
        from .visualization_agent import get_visualization_agent
        seedream_key = settings.SEEDREAM_API_KEY or self.api_key
        self.visualization_agent = get_visualization_agent(seedream_key)
        
        logger.info("所有 Agent 初始化完成")
    
    async def search_node(self, state: AgentState) -> Dict:
        """搜索节点 - 搜索和 RAG"""
        logger.info("[LangGraph] Node: search_node")
        
        try:
            search_result = await self.search_agent.process(
                user_query=state.user_request,
                enable_search=state.enable_search,
            )
            return {
                "search_context": search_result.knowledge_context,
                "current_node": "search_node"
            }
        except Exception as e:
            logger.error(f"Search node error: {e}")
            return {
                "search_context": "",
                "error": str(e),
                "current_node": "search_node"
            }
    
    async def planner_node(self, state: AgentState) -> Dict:
        """规划节点 - 生成旅行计划"""
        logger.info("[LangGraph] Node: planner_node")
        
        try:
            travel_plan = await self.planner_agent.process(
                user_request=state.user_request,
                search_context=state.search_context,
                user_budget=None,
                user_days=None,
            )
            return {
                "travel_plan": travel_plan.model_dump() if hasattr(travel_plan, 'model_dump') else {},
                "current_node": "planner_node"
            }
        except Exception as e:
            logger.error(f"Planner node error: {e}")
            return {
                "travel_plan": None,
                "error": str(e),
                "current_node": "planner_node"
            }
    
    async def writer_node(self, state: AgentState) -> Dict:
        """写作节点 - 生成攻略文本"""
        logger.info("[LangGraph] Node: writer_node")
        
        # 处理 travel_plan 类型：可能是 dict 或 Pydantic 模型
        travel_plan = state.travel_plan
        if isinstance(travel_plan, dict):
            # 转换为 Pydantic 模型
            from .planner_agent import TravelPlan
            try:
                travel_plan = TravelPlan(**travel_plan)
            except Exception as e:
                logger.warning(f"转换 travel_plan 失败: {e}")
                travel_plan = None
        
        try:
            guide_content = await self.writer_agent.process(
                user_request=state.user_request,
                travel_plan=travel_plan,
                search_context=state.search_context,
                style=state.style,
            )
            # 使用 raw_output 获取原始 markdown 内容
            return {
                "guide_content": guide_content.raw_output if hasattr(guide_content, 'raw_output') else str(guide_content),
                "current_node": "writer_node"
            }
        except Exception as e:
            logger.error(f"Writer node error: {e}")
            return {
                "guide_content": "",
                "error": str(e),
                "current_node": "writer_node"
            }
    
    async def visualize_node(self, state: AgentState) -> Dict:
        """可视化节点 - 生成路线图"""
        logger.info("[LangGraph] Node: visualize_node")
        
        if not state.enable_visualization:
            return {"route_map": None, "current_node": "visualize_node"}
        
        # 处理 travel_plan 类型：可能是 dict 或 Pydantic 模型
        travel_plan = state.travel_plan
        if isinstance(travel_plan, dict):
            from .planner_agent import TravelPlan
            try:
                travel_plan = TravelPlan(**travel_plan)
            except Exception as e:
                logger.warning(f"转换 travel_plan 失败: {e}")
                travel_plan = None
        
        try:
            route_map_obj = await self.visualization_agent.process(
                travel_plan=travel_plan,
                guide_content=state.guide_content,
                style="travel poster",
            )
            return {
                "route_map": {
                    "description": route_map_obj.description,
                    "image_url": route_map_obj.image_url,
                    "spots_marked": route_map_obj.spots_marked,
                },
                "current_node": "visualize_node"
            }
        except Exception as e:
            logger.error(f"Visualize node error: {e}")
            return {
                "route_map": None,
                "error": str(e),
                "current_node": "visualize_node"
            }
    
    def should_search(self, state: AgentState) -> Literal["planner_node", "planner_node_no_search"]:
        """条件路由 - 是否启用搜索"""
        if state.enable_search:
            return "planner_node"
        return "planner_node_no_search"
    
    def should_visualize(self, state: AgentState) -> Literal["visualize_node", "END"]:
        """条件路由 - 是否启用可视化"""
        if state.enable_visualization:
            return "visualize_node"
        return END
    
    def build_graph(self) -> StateGraph:
        """构建 LangGraph"""
        if self._graph is not None:
            return self._graph
        
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("search_node", self.search_node)
        workflow.add_node("planner_node", self.planner_node)
        workflow.add_node("planner_node_no_search", self.planner_node)
        workflow.add_node("writer_node", self.writer_node)
        workflow.add_node("visualize_node", self.visualize_node)
        
        # 设置入口
        workflow.set_entry_point("search_node")
        
        # 定义边
        workflow.add_edge("search_node", "planner_node")
        workflow.add_edge("planner_node", "writer_node")
        workflow.add_edge("planner_node_no_search", "writer_node")
        workflow.add_edge("writer_node", "visualize_node")
        workflow.add_edge("visualize_node", END)
        
        # 编译图
        self._graph = workflow.compile()
        return self._graph

    async def stream_events(
        self,
        user_request: str,
        enable_search: bool = True,
        enable_visualization: bool = True,
        style: str = "friendly",
        thread_id: str = "default",
    ):
        """流式执行并产生事件（用于前端监控）
        
        Args:
            user_request: 用户请求
            enable_search: 是否启用搜索
            enable_visualization: 是否生成可视化
            style: 写作风格
            thread_id: 线程 ID
            
        Yields:
            监控事件字典
        """
        from datetime import datetime
        
        initial_state = AgentState(
            user_request=user_request,
            enable_search=enable_search,
            enable_visualization=enable_visualization,
            style=style,
        )
        
        graph = self.build_graph()
        config = {"configurable": {"thread_id": thread_id}}
        
        # 发送开始事件
        yield {
            "type": "start",
            "timestamp": datetime.now().isoformat(),
            "message": "Agent 执行开始",
        }
        
        step_count = 0
        final_content = ""
        
        try:
            async for event in graph.astream_events(
                initial_state.model_dump(),
                config=config,
                version="v2",
            ):
                kind = event.get("event")
                
                # 转换 LangGraph 事件为前端格式
                if kind == "on_chain_start":
                    step_count += 1
                    yield {
                        "type": "step_start",
                        "step": step_count,
                        "node": event.get("metadata", {}).get("langgraph_node", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                    }
                elif kind == "on_chain_end":
                    yield {
                        "type": "step_end",
                        "step": step_count,
                        "node": event.get("metadata", {}).get("langgraph_node", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                    }
                elif kind == "on_chat_model_start":
                    yield {
                        "type": "llm_start",
                        "timestamp": datetime.now().isoformat(),
                    }
                elif kind == "on_chat_model_end":
                    yield {
                        "type": "llm_end",
                        "timestamp": datetime.now().isoformat(),
                    }
                    
            # 执行完成后，获取最终结果
            final_state = await graph.ainvoke(
                initial_state.model_dump(),
                config=config,
            )
            final_content = final_state.get("guide_content", "")
            
            # 发送完成事件（包含最终内容）
            yield {
                "type": "complete",
                "timestamp": datetime.now().isoformat(),
                "message": "执行完成",
                "final_content": final_content,
            }
            
        except Exception as e:
            logger.error(f"Graph stream error: {e}")
            yield {
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "message": str(e),
            }
    
    async def process(
        self,
        user_request: str,
        enable_search: bool = True,
        enable_visualization: bool = True,
        style: str = "friendly",
        thread_id: str = "default",
    ) -> AgentResponse:
        """处理用户请求
        
        Args:
            user_request: 用户请求
            enable_search: 是否启用搜索
            enable_visualization: 是否生成可视化
            style: 写作风格
            thread_id: 线程 ID（用于状态持久化）
            
        Returns:
            AgentResponse: 包含攻略内容和路线图
        """
        logger.info(f"[LangGraph] 开始处理请求: {user_request[:50]}...")
        
        # 构建初始状态
        initial_state = AgentState(
            user_request=user_request,
            enable_search=enable_search,
            enable_visualization=enable_visualization,
            style=style,
        )
        
        # 获取编译后的图
        graph = self.build_graph()
        
        # 执行图
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await graph.ainvoke(initial_state.model_dump(), config)
        
        # 转换结果
        return AgentResponse(
            guide_content=final_state.get("guide_content", ""),
            route_map=final_state.get("route_map"),
            search_context=final_state.get("search_context", ""),
            travel_plan=final_state.get("travel_plan"),
        )


# ==================== 兼容旧接口 ====================

# 单例实例
_coordinator_instance = None


def get_coordinator(api_key: str = None, tools: List[Any] = None):
    """获取 MultiAgentCoordinator 单例实例"""
    global _coordinator_instance
    
    if _coordinator_instance is None:
        if api_key is None:
            from backend.config import get_settings
            settings = get_settings()
            api_key = settings.DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY")
        
        _coordinator_instance = MultiAgentCoordinator(api_key=api_key, tools=tools or [])
    
    return _coordinator_instance


class MultiAgentCoordinator:
    """兼容旧接口的协调器（内部使用 LangGraph）"""
    
    def __init__(self, api_key: str, tools: List[Any] = None):
        self._graph = MultiAgentGraph(api_key, tools)
    
    async def process(
        self,
        user_request: str,
        enable_search: bool = True,
        enable_visualization: bool = True,
        style: str = "friendly",
    ) -> AgentResponse:
        return await self._graph.process(
            user_request=user_request,
            enable_search=enable_search,
            enable_visualization=enable_visualization,
            style=style,
        )
