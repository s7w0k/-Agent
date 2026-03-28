"""
文档处理器模块
负责文档的原始保存、清洗、分块处理
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from backend.config import get_settings
from backend.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()


class DocumentChunk(BaseModel):
    """文档块模型"""
    chunk_id: str
    content: str
    metadata: dict
    index: int


class Document(BaseModel):
    """文档模型"""
    doc_id: str
    title: str
    source: str  # 来源：小红书、全网等
    content: str  # 清洗后的内容
    raw_content: str  # 原始内容
    metadata: dict
    created_at: datetime
    chunks: List[DocumentChunk] = []


class DocumentProcessor:
    """文档处理器 - 支持原始保存、清洗、分块"""

    def __init__(self):
        self.raw_dir = settings.KNOWLEDGE_RAW_DIR
        self.processed_dir = settings.KNOWLEDGE_PROCESSED_DIR
        self.chunk_size = settings.RAG_CHUNK_SIZE
        self.chunk_overlap = settings.RAG_CHUNK_OVERLAP

        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _generate_doc_id(self, content: str) -> str:
        """生成文档 ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_chunk_id(self, doc_id: str, index: int) -> str:
        """生成块 ID"""
        return f"{doc_id}_chunk_{index}"

    def save_raw_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        保存原始文档

        Args:
            content: 文档内容
            title: 文档标题
            source: 来源（小红书/全网）
            metadata: 额外元数据

        Returns:
            Document 对象
        """
        doc_id = self._generate_doc_id(content)

        # 构建元数据
        doc_metadata = {
            "source": source,
            "saved_at": datetime.now().isoformat(),
            **(metadata or {})
        }

        # 创建文档对象
        doc = Document(
            doc_id=doc_id,
            title=title,
            source=source,
            raw_content=content,
            content="",  # 待清洗
            metadata=doc_metadata,
            created_at=datetime.now(),
        )

        # 保存原始文档
        raw_file = self.raw_dir / f"{doc_id}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"原始文档已保存: {doc_id}")
        return doc

    def clean_document(self, content: str) -> str:
        """
        清洗文档内容

        Args:
            content: 原始内容

        Returns:
            清洗后的内容
        """
        # 去除多余空白
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        # 去除 URL（可选）
        # content = re.sub(r'http[s]?://\S+', '', content)

        # 去除特殊字符但保留中文、英文、数字和常用标点
        content = re.sub(r'''[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）【】《》.,!?;:'"()\[\]%]''', '', content)

        # 去除首尾空白
        content = content.strip()

        return content

    def chunk_document(self, content: str, doc_id: str, metadata: dict) -> List[DocumentChunk]:
        """
        将文档分块

        Args:
            content: 清洗后的内容
            doc_id: 文档 ID
            metadata: 文档元数据

        Returns:
            文档块列表
        """
        chunks = []
        start = 0
        index = 0

        while start < len(content):
            end = start + self.chunk_size

            # 尝试在句子边界切分
            if end < len(content):
                # 查找最后一个句号、逗号或句问号
                for sep in ['。', '！', '？', '，', '、', '.', '!', '?', ',']:
                    last_sep = content.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk = DocumentChunk(
                    chunk_id=self._generate_chunk_id(doc_id, index),
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_index": index,
                        "total_chars": len(content),
                    },
                    index=index,
                )
                chunks.append(chunk)
                index += 1

            # 移动起始位置（考虑重叠）
            start = end - self.chunk_overlap if end < len(content) else end

        logger.info(f"文档 {doc_id} 分为 {len(chunks)} 个块")
        return chunks

    def process_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        完整处理流程：保存原始 -> 清洗 -> 分块

        Args:
            content: 文档内容
            title: 文档标题
            source: 来源
            metadata: 额外元数据

        Returns:
            处理后的 Document 对象
        """
        # 1. 保存原始文档
        doc = self.save_raw_document(content, title, source, metadata)

        # 2. 清洗内容
        cleaned_content = self.clean_document(content)
        doc.content = cleaned_content

        # 3. 分块
        doc.chunks = self.chunk_document(
            cleaned_content,
            doc.doc_id,
            {
                "title": title,
                "source": source,
                **doc.metadata
            }
        )

        # 4. 保存处理后的文档
        processed_file = self.processed_dir / f"{doc.doc_id}.json"
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"文档处理完成: {doc.doc_id}, {len(doc.chunks)} 个块")
        return doc

    def load_processed_document(self, doc_id: str) -> Optional[Document]:
        """加载已处理的文档"""
        processed_file = self.processed_dir / f"{doc_id}.json"
        if not processed_file.exists():
            return None

        with open(processed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Document(**data)


# 全局实例
_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """获取文档处理器全局实例"""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
