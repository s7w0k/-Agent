"""
Search Agent - 搜索与知识库管理 Agent
功能：
1. 根据用户提问搜索小红书和全网内容
2. 将搜索结果添加到知识库
3. 进行 RAG 检索，返回相关上下文
"""

import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from backend.logger import get_logger
from backend.config import get_settings
from backend.rag.retriever import get_retriever

logger = get_logger(__name__)
settings = get_settings()


class SearchResult(BaseModel):
    """搜索结果"""
    title: str
    content: str
    source: str  # 小红书/全网
    url: Optional[str] = ""
    relevance_score: Optional[float] = None


class SearchContext(BaseModel):
    """搜索上下文（返回给 Planner Agent）"""
    query: str
    results: List[SearchResult]
    knowledge_context: str  # RAG 检索结果
    total_found: int


class SearchAgent:
    """搜索 Agent - 负责信息检索和知识库管理"""

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
            temperature=0.7,
        )

    async def process(self, user_query: str, enable_search: bool = True) -> SearchContext:
        """处理用户查询

        Args:
            user_query: 用户问题
            enable_search: 是否启用外部搜索

        Returns:
            SearchContext: 包含搜索结果和 RAG 上下文的上下文对象
        """
        logger.info(f"SearchAgent 处理查询: {user_query[:50]}...")

        results = []

        # 1. 外部搜索（可选）
        if enable_search:
            search_results = await self._search_external(user_query)
            results.extend(search_results)

            # 将搜索结果添加到知识库
            if search_results:
                await self._add_to_knowledge_base(search_results)
                logger.info(f"已将 {len(search_results)} 条结果添加到知识库")

        # 2. RAG 检索
        knowledge_context = await self._rag_retrieve(user_query)

        return SearchContext(
            query=user_query,
            results=results,
            knowledge_context=knowledge_context,
            total_found=len(results)
        )

    async def _search_external(self, query: str) -> List[SearchResult]:
        """外部搜索（模拟实现）

        实际项目中可以接入：
        - 小红书搜索 API
        - 百度/Google 搜索 API
        - 携程、马蜂窝等旅行平台 API

        Args:
            query: 搜索关键词

        Returns:
            搜索结果列表
        """
        # 构建搜索提示
        search_prompt = f"""请搜索以下旅行相关问题的相关信息：
{query}

请返回搜索结果，包括：
1. 小红书上的相关攻略和用户分享
2. 全网相关的旅行信息

格式要求：
- 返回 JSON 数组格式
- 每个结果包含：title（标题）、content（主要内容）、source（来源：小红书/全网）
"""

        try:
            # 调用 LLM 生成搜索结果（实际项目中替换为真实搜索 API）
            response = await self.llm.ainvoke(search_prompt)
            content = response.content

            # 解析结果（简化处理，实际项目中需要解析真实 API 返回）
            results = self._parse_search_results(content, query)

            logger.info(f"外部搜索完成，找到 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.error(f"外部搜索失败: {e}")
            return []

    def _parse_search_results(self, content: str, query: str) -> List[SearchResult]:
        """解析搜索结果

        实际项目中这里需要解析真实搜索 API 的返回
        """
        # 这里返回模拟数据，实际项目中替换为真实搜索结果解析
        # 基于 query 生成相关的模拟结果
        results = []

        # 模拟小红书结果
        if "苏州" in query or "上海" in query:
            results.append(SearchResult(
                title="上海到苏州一日游攻略",
                content="上海到苏州高铁25分钟，票价约45元。拙政园门票70元，平江路免费，苏州博物馆需预约。",
                source="小红书",
                url=""
            ))

        # 模拟全网结果
        results.append(SearchResult(
            title="12306火车票购买指南",
            content="火车票提前15天预售，每天6:00-23:00开放。高铁二等座票价=里程×0.46元。",
            source="全网",
            url=""
        ))

        return results

    async def _add_to_knowledge_base(self, results: List[SearchResult]) -> None:
        """将搜索结果添加到知识库"""
        retriever = get_retriever()

        for result in results:
            try:
                await retriever.add_document(
                    content=result.content,
                    title=result.title,
                    source=result.source
                )
            except Exception as e:
                logger.warning(f"添加文档失败: {e}")

    async def _rag_retrieve(self, query: str, top_k: int = 5) -> str:
        """RAG 检索

        Args:
            query: 查询内容
            top_k: 返回结果数量

        Returns:
            格式化的检索结果
        """
        try:
            retriever = get_retriever()
            results = await retriever.retrieve(query, top_k=top_k)

            if not results:
                return ""

            # 格式化结果
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(
                    f"【{i}】{result.title}\n"
                    f"来源: {result.source}\n"
                    f"内容: {result.content}"
                )

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"RAG 检索失败: {e}")
            return ""

    async def search_only(self, query: str, sources: Optional[List[str]] = None) -> List[SearchResult]:
        """仅搜索，不添加到知识库

        Args:
            query: 查询内容
            sources: 数据源筛选 ["小红书", "全网"]

        Returns:
            搜索结果列表
        """
        results = await self._search_external(query)

        if sources:
            results = [r for r in results if r.source in sources]

        return results


# 全局实例
_search_agent: Optional[SearchAgent] = None


def get_search_agent(api_key: str = None) -> SearchAgent:
    """获取 SearchAgent 全局实例"""
    global _search_agent
    if _search_agent is None:
        from backend.config import get_settings
        settings = get_settings()
        api_key = api_key or settings.DEEPSEEK_API_KEY
        _search_agent = SearchAgent(api_key)
    return _search_agent
