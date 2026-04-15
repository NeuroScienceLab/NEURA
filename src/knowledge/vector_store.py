"""
Vector storage module - Using FAISS for local vector retrieval
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from src.config import OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL


@dataclass
class TextChunk:
    """文本块"""
    chunk_id: str
    content: str
    source: str  # 来源文件
    section_type: str  # 章节类型
    metadata: Dict


class VectorStore:
    """向量存储"""

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path) if store_path else OUTPUT_DIR / "vector_store"
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

        self._faiss = None
        self._embedding_client = None

    def _get_faiss(self):
        """延迟加载faiss"""
        if self._faiss is None:
            import faiss
            self._faiss = faiss
        return self._faiss

    def _get_embedding_client(self):
        """获取embedding客户端"""
        if self._embedding_client is None:
            from src.utils.embedding import get_embedding_client
            self._embedding_client = get_embedding_client()
        return self._embedding_client

    def add_document(self, doc_id: str, text: str, source: str, section_type: str = "other", metadata: Dict = None):
        """
        添加文档到存储

        Args:
            doc_id: 文档ID
            text: 文档文本
            source: 来源
            section_type: 章节类型
            metadata: 元数据
        """
        # 分块
        chunks = self._chunk_text(text, doc_id, source, section_type, metadata or {})
        self.chunks.extend(chunks)

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        source: str,
        section_type: str,
        metadata: Dict
    ) -> List[TextChunk]:
        """文本分块"""
        chunks = []
        words = text.split()

        if len(words) <= CHUNK_SIZE:
            chunks.append(TextChunk(
                chunk_id=f"{doc_id}_0",
                content=text,
                source=source,
                section_type=section_type,
                metadata=metadata
            ))
        else:
            start = 0
            chunk_idx = 0
            while start < len(words):
                end = min(start + CHUNK_SIZE, len(words))
                chunk_text = " ".join(words[start:end])

                chunks.append(TextChunk(
                    chunk_id=f"{doc_id}_{chunk_idx}",
                    content=chunk_text,
                    source=source,
                    section_type=section_type,
                    metadata=metadata
                ))

                start = end - CHUNK_OVERLAP
                chunk_idx += 1

        return chunks

    def build_index(self):
        """构建向量索引"""
        if not self.chunks:
            raise ValueError("没有文档可索引")

        faiss = self._get_faiss()
        client = self._get_embedding_client()

        # 批量生成嵌入
        texts = [chunk.content for chunk in self.chunks]
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = client.embed(batch)
            all_embeddings.append(embeddings)

        self.embeddings = np.vstack(all_embeddings).astype('float32')

        # 构建FAISS索引
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # 内积相似度
        # 归一化用于余弦相似度
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[TextChunk, float]]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            (文本块, 相似度)列表
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index()")

        faiss = self._get_faiss()
        client = self._get_embedding_client()

        # 生成查询嵌入
        query_embedding = client.embed(query).astype('float32')
        faiss.normalize_L2(query_embedding)

        # 搜索
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self):
        """保存存储"""
        faiss = self._get_faiss()

        # 保存chunks
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "source": c.source,
                "section_type": c.section_type,
                "metadata": c.metadata
            }
            for c in self.chunks
        ]
        with open(self.store_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # 保存embeddings和index
        if self.embeddings is not None:
            np.save(self.store_path / "embeddings.npy", self.embeddings)

        if self.index is not None:
            faiss.write_index(self.index, str(self.store_path / "index.faiss"))

    def load(self):
        """加载存储"""
        faiss = self._get_faiss()

        # 加载chunks
        chunks_path = self.store_path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [
                TextChunk(
                    chunk_id=c["chunk_id"],
                    content=c["content"],
                    source=c["source"],
                    section_type=c["section_type"],
                    metadata=c["metadata"]
                )
                for c in chunks_data
            ]

        # 加载embeddings
        embeddings_path = self.store_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

        # 加载index
        index_path = self.store_path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))


# 全局实例
_vector_store = None

def get_vector_store() -> VectorStore:
    """获取全局向量存储"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
