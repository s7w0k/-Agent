"""
Embedding 服务模块
使用阿里云 DashScope 进行文本嵌入 (LangChain 框架)
"""

from typing import List, Optional

from backend.config import get_settings
from backend.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()


class EmbeddingService:
    """Embedding 服务 - 使用阿里云 DashScope (LangChain)"""

    def __init__(self, model: str = None):
        self.model = model or settings.RAG_EMBEDDING_MODEL
        self._embeddings = None
        self._embedding_cache = {}

    def _get_embeddings(self):
        """获取 DashScope Embeddings 实例"""
        if self._embeddings is None:
            try:
                from langchain_community.embeddings import DashScopeEmbeddings
                self._embeddings = DashScopeEmbeddings(
                    model=self.model,
                    dashscope_api_key=settings.DASHSCOPE_API_KEY
                )
            except Exception as e:
                logger.error(f"初始化 DashScope Embeddings 失败: {e}")
                raise
        return self._embeddings

    async def embed_text(self, text: str) -> List[float]:
        """
        对单个文本进行嵌入

        Args:
            text: 待嵌入文本

        Returns:
            嵌入向量
        """
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embeddings = self._get_embeddings()
            embedding = embeddings.embed_query(text)
            embedding_dim = len(embedding)
            logger.info(f"Embedding 成功，维度: {embedding_dim}")

            if len(self._embedding_cache) < 1000:
                self._embedding_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Embedding 失败: {e}")
            return [0.0] * 1536

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入多个文本

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        try:
            embeddings = self._get_embeddings()
            embeddings_result = embeddings.embed_documents(texts)
            return embeddings_result

        except Exception as e:
            logger.error(f"批量 Embedding 失败: {e}")
            return [[0.0] * 1536 for _ in texts]

    def clear_cache(self):
        """清空缓存"""
        self._embedding_cache.clear()
        logger.info("Embedding 缓存已清空")


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务全局实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
