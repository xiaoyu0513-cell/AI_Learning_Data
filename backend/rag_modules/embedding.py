"""
Embedding 模块
使用本地 BAAI/bge-small-zh-v1.5 生成向量（免费离线，无需 GPU）
BGE 模型规范：文档(段落)不加前缀，查询(query)加检索指令前缀
"""
import logging
import os
from typing import List

# 国内环境走 HuggingFace 镜像下载模型（可通过环境变量覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 默认优先离线加载（模型已本地缓存），避免无网络时反复重试阻塞启动；
# 需在线重新下载时，显式设置 HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0 覆盖即可。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BGE 中文模型的查询指令前缀（提升检索效果）
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """BGE 向量编码器"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        logger.info("加载 embedding 模型: %s", model_name)
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """编码文档（不加前缀）"""
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]

    def encode_query(self, query: str) -> List[float]:
        """编码查询（加指令前缀）"""
        embedding = self.model.encode(
            [_QUERY_INSTRUCTION + query], normalize_embeddings=True
        )
        return embedding[0].tolist()