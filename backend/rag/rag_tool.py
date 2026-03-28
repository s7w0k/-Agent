"""
RAG 集成工具
为 Agent 提供 RAG 能力
"""

from typing import List, Optional

from backend.logger import get_logger
from backend.rag.retriever import get_retriever, SearchResult

logger = get_logger(__name__)


class RAGTool:
    """RAG 工具类 - 供 Agent 调用"""

    def __init__(self):
        self.retriever = get_retriever()

    async def search(
        self,
        query: str,
        top_k: int = 3,
        sources: Optional[List[str]] = None
    ) -> str:
        """
        搜索知识库

        Args:
            query: 查询文本
            top_k: 返回结果数量
            sources: 数据源过滤

        Returns:
            格式化的搜索结果
        """
        results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            sources=sources
        )

        if not results:
            return "未找到相关内容"

        # 格式化输出
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. [{result.source}] {result.title}\n"
                f"   内容：{result.content[:200]}...\n"
                f"   相似度：{1 - result.distance:.2%}"
            )

        return "\n\n".join(formatted)

    async def search_xiaohongshu(self, query: str, top_k: int = 3) -> str:
        """搜索小红书"""
        results = await self.retriever.retrieve_xiaohongshu(query, top_k)

        if not results:
            return "未找到小红书相关内容"

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.title}\n"
                f"   内容：{result.content[:200]}...\n"
                f"   相似度：{1 - result.distance:.2%}"
            )

        return "\n\n".join(formatted)

    async def search_web(self, query: str, top_k: int = 3) -> str:
        """搜索全网内容"""
        results = await self.retriever.retrieve_web(query, top_k)

        if not results:
            return "未找到相关内容"

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. [{result.source}] {result.title}\n"
                f"   内容：{result.content[:200]}...\n"
                f"   相似度：{1 - result.distance:.2%}"
            )

        return "\n\n".join(formatted)

    async def add_knowledge(
        self,
        content: str,
        title: str,
        source: str = "全网"
    ) -> str:
        """添加知识到知识库"""
        success = await self.retriever.add_document(
            content=content,
            title=title,
            source=source
        )
        return "知识添加成功" if success else "知识添加失败"

    def get_stats(self) -> str:
        """获取知识库状态"""
        stats = self.retriever.get_knowledge_stats()

        if not stats:
            return "无法获取知识库状态"

        return (
            f"知识库状态：\n"
            f"- 集合名称：{stats.get('collection_name')}\n"
            f"- 文档数量：{stats.get('document_count')}\n"
            f"- 存储路径：{stats.get('persist_dir')}\n"
            f"- 块大小：{stats.get('chunk_size')}\n"
            f"- 块重叠：{stats.get('chunk_overlap')}"
        )


# 全局实例
_rag_tool: Optional[RAGTool] = None


def get_rag_tool() -> RAGTool:
    """获取 RAG 工具实例"""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGTool()
    return _rag_tool
