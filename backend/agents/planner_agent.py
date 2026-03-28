"""
Planner Agent - 日程规划 Agent
功能：
1. 根据用户需求生成日程草案
2. 安排景点顺序，控制地理路径合理性
3. 评估预算可行性
4. 调用地图和天气工具
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime, timedelta

from backend.logger import get_logger
from backend.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class Spot(BaseModel):
    """景点"""
    name: str
    address: Optional[str] = ""
    ticket_price: Optional[float] = 0.0  # 门票
    opening_hours: Optional[str] = ""  # 开放时间
    recommended_duration: Optional[int] = 120  # 建议游玩时长（分钟）
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = ""


class DayPlan(BaseModel):
    """单日计划"""
    date: str
    spots: List[Spot]
    transportation: str = ""  # 交通方式
    meals: List[str] = []  # 餐饮安排
    total_cost: float = 0.0  # 当日总花费


class TravelPlan(BaseModel):
    """旅行计划"""
    destination: str
    start_date: str
    end_date: str
    days: List[DayPlan]
    total_budget: float  # 总预算
    estimated_cost: float  # 预估花费
    budget_feasible: bool  # 预算是否可行
    notes: List[str] = []  # 注意事项
    weather_tips: List[str] = []  # 天气提示


class PlannerAgent:
    """规划 Agent - 负责行程安排和预算评估"""

    def __init__(self, api_key: str, tools: List[Any] = None):
        self.api_key = api_key
        self.tools = tools or []
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        from langchain_deepseek import ChatDeepSeek
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=self.api_key,
            temperature=0.5,
        )

    async def process(
        self,
        user_request: str,
        search_context: str,
        user_budget: Optional[float] = None,
        user_days: Optional[int] = None,
    ) -> TravelPlan:
        """处理用户请求，生成旅行计划

        Args:
            user_request: 用户需求描述
            search_context: 搜索和 RAG 获取的上下文信息
            user_budget: 用户预算
            user_days: 用户计划天数

        Returns:
            TravelPlan: 完整的旅行计划
        """
        logger.info(f"PlannerAgent 处理请求: {user_request[:50]}...")

        # 构建规划提示
        planning_prompt = self._build_planning_prompt(
            user_request, search_context, user_budget, user_days
        )

        # 调用 LLM 生成计划
        response = await self.llm.ainvoke(planning_prompt)
        content = response.content

        # 解析生成的计划
        travel_plan = self._parse_travel_plan(content, user_request, user_budget, user_days)

        # 评估预算可行性
        travel_plan = self._evaluate_budget(travel_plan, user_budget)

        # 获取天气信息（如果有工具）
        if self.tools:
            travel_plan = await self._enrich_with_weather(travel_plan)

        logger.info(f"PlannerAgent 完成，计划天数: {len(travel_plan.days)}")

        return travel_plan

    def _build_planning_prompt(
        self,
        user_request: str,
        search_context: str,
        user_budget: Optional[float],
        user_days: Optional[int],
    ) -> str:
        """构建规划提示"""
        prompt = f"""你是一个专业的旅行规划师。请根据以下信息生成详细的旅行计划。

## 用户需求
{user_request}

## 搜索和知识库信息
{search_context}

## 用户约束
{"预算: " + str(user_budget) + "元" if user_budget else "未指定预算"}
{"计划天数: " + str(user_days) + "天" if user_days else "未指定天数"}

## 输出要求
请生成一个完整的旅行计划，包含：
1. 目的地和行程概览
2. 每日具体安排（景点、交通、餐饮）
3. 预算估算
4. 注意事项

请以 JSON 格式输出，格式如下：
```json
{{
  "destination": "目的地",
  "start_date": "开始日期",
  "end_date": "结束日期",
  "days": [
    {{
      "date": "日期",
      "spots": [
        {{
          "name": "景点名称",
          "address": "地址",
          "ticket_price": 门票价格,
          "opening_hours": "开放时间",
          "recommended_duration": 建议时长(分钟)
        }}
      ],
      "transportation": "交通方式",
      "meals": ["早餐", "午餐", "晚餐"],
      "total_cost": 当日花费
    }}
  ],
  "estimated_cost": 总预估花费,
  "notes": ["注意事项"]
}}
```"""
        return prompt

    def _parse_travel_plan(
        self,
        content: str,
        user_request: str,
        user_budget: Optional[float],
        user_days: Optional[int],
    ) -> TravelPlan:
        """解析旅行计划"""
        import json
        import re

        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}

            # 构建 TravelPlan
            return TravelPlan(
                destination=data.get("destination", user_request),
                start_date=data.get("start_date", datetime.now().strftime("%Y-%m-%d")),
                end_date=data.get("end_date", (datetime.now() + timedelta(days=user_days or 3)).strftime("%Y-%m-%d")),
                days=[DayPlan(**day) for day in data.get("days", [])],
                total_budget=user_budget or 0,
                estimated_cost=data.get("estimated_cost", 0),
                budget_feasible=True,
                notes=data.get("notes", []),
            )

        except Exception as e:
            logger.warning(f"解析旅行计划失败: {e}")

            # 返回默认计划
            days_count = user_days or 3
            default_days = [
                DayPlan(
                    date=(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                    spots=[],
                    transportation="待定",
                    meals=[],
                    total_cost=0.0
                )
                for i in range(days_count)
            ]

            return TravelPlan(
                destination=user_request,
                start_date=datetime.now().strftime("%Y-%m-%d"),
                end_date=(datetime.now() + timedelta(days=days_count)).strftime("%Y-%m-%d"),
                days=default_days,
                total_budget=user_budget or 0,
                estimated_cost=0,
                budget_feasible=True,
                notes=["请查看详细行程"],
            )

    def _evaluate_budget(self, plan: TravelPlan, user_budget: Optional[float]) -> TravelPlan:
        """评估预算可行性"""
        if not user_budget or user_budget <= 0:
            plan.budget_feasible = True
            return plan

        plan.budget_feasible = plan.estimated_cost <= user_budget

        if not plan.budget_feasible:
            plan.notes.append(
                f"警告：预估花费 {plan.estimated_cost} 元超出预算 {user_budget} 元"
            )

        return plan

    async def _enrich_with_weather(self, plan: TravelPlan) -> TravelPlan:
        """使用工具获取天气信息"""
        # 实际项目中，这里会调用天气 API 工具
        # 例如：amap_weather 工具
        plan.weather_tips = [
            "建议提前查看目的地天气预报",
            "出行注意带好雨具",
        ]
        return plan

    def optimize_route(self, spots: List[Spot]) -> List[Spot]:
        """优化景点顺序（基于地理位置）

        Args:
            spots: 景点列表

        Returns:
            优化后的景点列表
        """
        # 简化实现：按纬度排序（实际项目中需要使用地图 API）
        if not spots:
            return spots

        valid_spots = [s for s in spots if s.latitude and s.longitude]
        if not valid_spots:
            return spots

        # 按纬度排序（从北到南）
        sorted_spots = sorted(valid_spots, key=lambda x: x.latitude, reverse=True)

        # 合并未定位的景点
        invalid_spots = [s for s in spots if not s.latitude or not s.longitude]

        return sorted_spots + invalid_spots


# 全局实例
_planner_agent: Optional["PlannerAgent"] = None


def get_planner_agent(api_key: str = None, tools: List[Any] = None) -> "PlannerAgent":
    """获取 PlannerAgent 全局实例"""
    global _planner_agent
    if _planner_agent is None:
        from backend.config import get_settings
        settings = get_settings()
        api_key = api_key or settings.DEEPSEEK_API_KEY
        _planner_agent = PlannerAgent(api_key, tools)
    return _planner_agent
