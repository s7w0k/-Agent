"""
Writer Agent - 攻略输出 Agent
功能：
按照攻略风格输出最终回答结果
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from backend.logger import get_logger
from backend.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class GuideContent(BaseModel):
    """攻略内容"""
    title: str  # 标题
    summary: str  # 摘要
    days: List[Dict]  # 每日详情
    tips: List[str]  # 小贴士
    budget: Dict  # 预算明细
    raw_output: str  # 原始输出


class WriterAgent:
    """写作 Agent - 负责将旅行计划转换为友好的攻略文本"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        from langchain_deepseek import ChatDeepSeek
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=self.api_key,
            temperature=0.8,  # 较高温度，产生更生动的输出
        )

    async def process(
        self,
        user_request: str,
        travel_plan: Any,
        search_context: str,
        style: str = "friendly",
    ) -> GuideContent:
        """生成旅行攻略

        Args:
            user_request: 用户原始需求
            travel_plan: Planner Agent 生成的计划
            search_context: 搜索和 RAG 获取的上下文
            style: 输出风格 (friendly/professional/fun)

        Returns:
            GuideContent: 格式化的攻略内容
        """
        logger.info(f"WriterAgent 生成攻略，风格: {style}")

        # 构建写作提示
        writing_prompt = self._build_writing_prompt(
            user_request, travel_plan, search_context, style
        )

        # 调用 LLM 生成内容
        response = await self.llm.ainvoke(writing_prompt)
        content = response.content

        # 解析内容
        guide = self._parse_guide_content(content, travel_plan)

        logger.info(f"WriterAgent 完成，生成 {len(guide.days)} 天攻略")

        return guide

    def _build_writing_prompt(
        self,
        user_request: str,
        travel_plan: Any,
        search_context: str,
        style: str,
    ) -> str:
        """构建写作提示 - 使用 travel_plan.md 模板格式"""

        # 转换旅行计划为字符串
        plan_str = self._format_travel_plan(travel_plan)

        # 风格指导
        style_guide = {
            "friendly": "使用亲切友好的语气，像朋友分享旅行经验一样，适当使用 emoji",
            "professional": "使用专业正式的语气，信息全面准确，适合规划参考",
            "fun": "使用活泼有趣的语气，增加旅行的趣味性和期待感",
        }

        prompt = f"""你是一个旅行计划规划专家。请根据以下信息生成一份完整的旅行计划，严格按照以下模板格式输出：

## 📋 旅行计划

### 🚄 交通信息
- 车次：[车次号]
- 出发时间：[时间]
- 到达时间：[时间]
- 余票情况：[票种及数量]

### 🌤️ 天气预报
- 日期：[日期]
- 天气：[晴/雨/多云]
- 温度：[最低~最高℃]

### 🗺️ 目的地导航
- 从车站到目的地：[路线描述]
- 预计耗时：[时间]

### ✅ 行程验证
- 到站时间：[合理/需注意]
- 天气状况：[适宜/需注意]

### 📅 推荐行程
[详细行程安排，包含每天的景点、美食、交通安排]

### 💰 预算估算
- 交通费：[金额]
- 住宿费：[金额]
- 餐饮费：[金额]
- 门票费：[金额]
- 其他：[金额]
- 总计：¥[总金额]

## 用户需求
{user_request}

## 旅行计划（由 Planner Agent 生成）
{plan_str}

## 参考信息
{search_context}

## 写作风格
{style_guide.get(style, style_guide['friendly'])}

请严格按照上述模板格式输出旅行计划，确保信息完整、准确！"""
        return prompt

    def _format_travel_plan(self, plan: Any) -> str:
        """格式化旅行计划"""
        lines = []

        lines.append(f"目的地: {plan.destination}")
        lines.append(f"行程: {plan.start_date} 至 {plan.end_date}")
        lines.append(f"预估费用: {plan.estimated_cost} 元")

        for i, day in enumerate(plan.days, 1):
            lines.append(f"\n--- 第 {i} 天 ({day.date}) ---")
            lines.append(f"交通: {day.transportation}")
            lines.append(f"餐饮: {', '.join(day.meals) if day.meals else '自理'}")
            lines.append(f"预计花费: {day.total_cost} 元")

            for spot in day.spots:
                lines.append(f"  - {spot.name}")
                if spot.ticket_price:
                    lines.append(f"    门票: {spot.ticket_price} 元")
                if spot.opening_hours:
                    lines.append(f"    开放时间: {spot.opening_hours}")

        if plan.notes:
            lines.append(f"\n注意事项: {', '.join(plan.notes)}")

        return "\n".join(lines)

    def _parse_guide_content(
        self,
        content: str,
        travel_plan: Any,
    ) -> GuideContent:
        """解析攻略内容"""

        # 提取标题（简单处理）
        lines = content.split("\n")
        title = lines[0] if lines else "旅行攻略"

        # 提取小贴士
        tips = []
        if "小贴士" in content or "tips" in content.lower():
            tip_section = content.split("小贴士")[-1]
            tips = [line.strip() for line in tip_section.split("\n") if line.strip()]

        # 简化处理：直接返回原始内容
        # 实际项目中可以进一步结构化解析
        return GuideContent(
            title=title,
            summary=self._extract_summary(content),
            days=self._extract_days(content, travel_plan),
            tips=tips[:5],  # 最多5条
            budget={
                "estimated": travel_plan.estimated_cost,
                "feasible": travel_plan.budget_feasible,
            },
            raw_output=content,
        )

    def _extract_summary(self, content: str) -> str:
        """提取摘要"""
        lines = content.split("\n")
        # 取前几行作为摘要
        summary_lines = []
        for line in lines[1:6]:
            if line.strip() and len(line.strip()) > 10:
                summary_lines.append(line.strip())
                if len(summary_lines) >= 2:
                    break
        return " ".join(summary_lines)

    def _extract_days(self, content: str, travel_plan: Any) -> List[Dict]:
        """提取每日详情"""
        days = []

        for day in travel_plan.days:
            day_info = {
                "date": day.date,
                "spots": [s.name for s in day.spots],
                "transportation": day.transportation,
                "meals": day.meals,
                "cost": day.total_cost,
            }
            days.append(day_info)

        return days


# 全局实例
_writer_agent: Optional["WriterAgent"] = None


def get_writer_agent(api_key: str = None) -> "WriterAgent":
    """获取 WriterAgent 全局实例"""
    global _writer_agent
    if _writer_agent is None:
        from backend.config import get_settings
        settings = get_settings()
        api_key = api_key or settings.DEEPSEEK_API_KEY
        _writer_agent = WriterAgent(api_key)
    return _writer_agent
