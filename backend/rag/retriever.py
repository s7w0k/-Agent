"""
检索器模块
整合文档处理、Embedding 和向量存储，提供统一的检索接口
支持多数据源检索（小红书、全网）
"""

from typing import List, Optional, Dict, Any

from backend.config import get_settings
from backend.logger import get_logger

from backend.rag.document_processor import get_document_processor
from backend.rag.vector_store import get_vector_store

logger = get_logger(__name__)

settings = get_settings()


class SearchResult:
    """搜索结果模型"""

    def __init__(self, content: str, source: str, title: str, distance: float = 0.0, metadata: dict = None):
        self.content = content
        self.source = source
        self.title = title
        self.distance = distance
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<SearchResult source={self.source} title={self.title} distance={self.distance:.4f}>"


class Retriever:
    """检索器 - 统一检索接口"""

    def __init__(self):
        self.document_processor = get_document_processor()
        self.vector_store = get_vector_store()

    async def add_document(
        self,
        content: str,
        title: str,
        source: str = "全网",
        metadata: Optional[dict] = None
    ) -> bool:
        """
        添加文档到知识库

        流程：处理文档 -> 分块 -> 存入向量库

        Args:
            content: 文档内容
            title: 文档标题
            source: 来源（小红书/全网）
            metadata: 额外元数据

        Returns:
            是否成功
        """
        try:
            # 1. 处理文档（清洗、分块）
            doc = self.document_processor.process_document(
                content=content,
                title=title,
                source=source,
                metadata=metadata
            )

            # 2. 提取块内容和元数据
            texts = [chunk.content for chunk in doc.chunks]
            metadatas = [chunk.metadata for chunk in doc.chunks]

            # 3. 添加到向量库
            success = await self.vector_store.add_documents(
                texts=texts,
                metadatas=metadatas,
                ids=[chunk.chunk_id for chunk in doc.chunks]
            )

            if success:
                logger.info(f"文档添加成功: {doc.doc_id}, {len(doc.chunks)} 个块")

            return success

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        sources: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量
            sources: 数据源过滤（小红书/全网），None 表示不限制

        Returns:
            搜索结果列表
        """
        top_k = top_k or settings.RAG_TOP_K

        # 构建过滤条件
        filter = None
        if sources:
            filter = {"source": {"$in": sources}}

        # 向量检索
        results = await self.vector_store.search(
            query=query,
            top_k=top_k,
            filter=filter
        )

        # 转换为 SearchResult 对象
        search_results = []
        for item in results:
            search_results.append(SearchResult(
                content=item["content"],
                source=item["metadata"].get("source", "未知"),
                title=item["metadata"].get("title", ""),
                distance=item.get("distance", 0.0),
                metadata=item["metadata"]
            ))

        logger.info(f"检索完成: query={query[:20]}..., 结果数={len(search_results)}")
        return search_results

    async def retrieve_xiaohongshu(self, query: str, top_k: int = None) -> List[SearchResult]:
        """检索小红书内容"""
        return await self.retrieve(query, top_k, sources=["小红书"])

    async def retrieve_web(self, query: str, top_k: int = None) -> List[SearchResult]:
        """检索全网内容"""
        return await self.retrieve(query, top_k, sources=["全网"])

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            collection_info = self.vector_store.get_collection_info()
            return {
                "collection_name": collection_info.get("name"),
                "document_count": collection_info.get("count", 0),
                "persist_dir": collection_info.get("persist_dir"),
                "chunk_size": settings.RAG_CHUNK_SIZE,
                "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 全局实例
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """获取检索器全局实例"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
