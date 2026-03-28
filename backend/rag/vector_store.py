"""
向量存储模块
使用 Chroma 进行向量存储和检索
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.logger import get_logger
from backend.rag.embedding_service import get_embedding_service

logger = get_logger(__name__)

settings = get_settings()


class VectorStore:
    """Chroma 向量存储"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.RAG_COLLECTION_NAME
        self.persist_dir = settings.RAG_CHROMA_PERSIST_DIR
        self._client = None
        self._collection = None

        # 确保目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """获取 Chroma 客户端"""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    )
                )
            except Exception as e:
                logger.error(f"初始化 Chroma 客户端失败: {e}")
                raise
        return self._client

    def _get_collection(self):
        """获取或创建集合"""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
                )
            except Exception as e:
                logger.error(f"获取集合失败: {e}")
                # 尝试删除重建
                try:
                    client.delete_collection(self.collection_name)
                    self._collection = client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                except Exception as e2:
                    logger.error(f"重建集合失败: {e2}")
                    raise
        return self._collection

    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        添加文档到向量库

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表（可选）

        Returns:
            是否成功
        """
        if not texts:
            return True

        try:
            embedding_service = get_embedding_service()
            embeddings = await embedding_service.embed_texts(texts)

            # 生成 ID
            if ids is None:
                ids = [f"doc_{i}_{hash(text)[:8]}" for i, text in enumerate(texts)]

            collection = self._get_collection()
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"成功添加 {len(texts)} 个文档到向量库")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def search(
        self,
        query: str,
        top_k: int = None,
        filter: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            相似文档列表
        """
        top_k = top_k or settings.RAG_TOP_K

        try:
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.embed_text(query)

            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter
            )

            # 格式化返回结果
            documents = []
            if results and results.get("documents"):
                for i in range(len(results["documents"][0])):
                    documents.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "id": results["ids"][0][i]
                    })

            return documents

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    async def delete_by_id(self, ids: List[str]) -> bool:
        """删除指定 ID 的文档"""
        try:
            collection = self._get_collection()
            collection.delete(ids=ids)
            logger.info(f"成功删除 {len(ids)} 个文档")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    async def delete_by_metadata(self, filter: dict) -> bool:
        """根据元数据删除文档"""
        try:
            collection = self._get_collection()
            # 获取匹配的文档
            results = collection.get(where=filter)
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
                logger.info(f"根据元数据删除 {len(results['ids'])} 个文档")
            return True
        except Exception as e:
            logger.error(f"根据元数据删除失败: {e}")
            return False

    def get_collection_info(self) -> dict:
        """获取集合信息"""
        try:
            collection = self._get_collection()
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_dir": str(self.persist_dir)
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}

    def reset(self) -> bool:
        """重置向量库"""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"向量库已重置: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"重置向量库失败: {e}")
            return False


# 全局实例
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取向量存储全局实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
