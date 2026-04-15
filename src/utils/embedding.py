"""
Embedding模块
"""
import httpx
import numpy as np
from typing import List, Union
from src.config import SILICONFLOW_API_BASE, SILICONFLOW_API_KEY, EMBEDDING_MODEL


class EmbeddingClient:

    def __init__(
        self,
        api_base: str = SILICONFLOW_API_BASE,
        api_key: str = SILICONFLOW_API_KEY,
        model: str = EMBEDDING_MODEL
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        生成文本嵌入向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        response = self.client.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]

        return np.array(embeddings)

    def close(self):
        """关闭客户端"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 全局客户端实例
_embedding_client = None

def get_embedding_client() -> EmbeddingClient:
    """获取全局Embedding客户端"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
