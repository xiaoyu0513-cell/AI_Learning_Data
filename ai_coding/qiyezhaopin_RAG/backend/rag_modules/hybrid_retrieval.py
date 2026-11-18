"""
传统混合检索模块（招聘领域）
向量检索（Milvus）+ BM25 稀疏检索，用 RRF(倒数排名融合) 合并。
"""
import re
import logging
from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """轻量中文分词：英文/数字整词 + 中文 bigram（无需额外依赖）"""
    text = text.lower()
    from .query import ALIASES
    tokens = [ALIASES.get(t, t).casefold() for t in re.findall(r"[a-z][a-z0-9]*(?:\+\+|#)?|[0-9]+", text)]
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(chunk[i:i+2] for i in range(len(chunk)-1))
        if len(chunk) == 1:
            tokens.append(chunk)
    return tokens


def _doc_key(d: Document) -> str:
    return d.metadata.get("node_id") or d.metadata.get("job_name") or d.page_content[:40]


class HybridRetrievalModule:
    """向量 + BM25 的混合检索器"""

    def __init__(self, config, milvus_module, llm_client=None):
        self.config = config
        self.milvus_module = milvus_module
        self.documents: List[Document] = []
        self.bm25 = None

    def initialize(self, documents: List[Document]):
        """传入语料构建 BM25 索引"""
        self.documents = documents
        tokenized_corpus = [_tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus) if documents else None
        logger.info("BM25 索引构建完成，共 %d 篇文档", len(documents))

    def bm25_search(self, query: str, top_k: int, filters=None) -> List[Document]:
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])
                  if scores[i] > 0 and (filters is None or filters.matches(self.documents[i]))][:top_k]
        docs = []
        for i in ranked:
            d = self.documents[i]
            docs.append(
                Document(
                    page_content=d.page_content,
                    metadata={
                        **d.metadata,
                        "score": float(scores[i]),
                        "search_type": "bm25",
                    },
                )
            )
        return docs

    def hybrid_search(self, query: str, top_k: int = None, rrf_k: int = _RRF_K, filters=None) -> List[Document]:
        """混合检索：向量 + BM25 → RRF 融合"""
        top_k = self.config.top_k if top_k is None else top_k
        if top_k <= 0:
            return []
        degraded = None
        try:
            # Filter skill/ID intersections before ANN using the authoritative corpus.
            if filters:
                filters = filters.model_copy(update={"restrict_ids": True, "job_ids": [
                    d.metadata["node_id"] for d in self.documents if filters.matches(d)]})
            vector_docs = self.milvus_module.vector_search(query, top_k, filters=filters)
        except Exception:
            logger.exception("向量检索失败，降级 BM25")
            vector_docs, degraded = [], "vector_unavailable"
        bm25_docs = self.bm25_search(query, top_k, filters=filters)

        fused: dict = {}
        for rank, d in enumerate(vector_docs):
            key = _doc_key(d)
            fused.setdefault(key, {"doc": d, "score": 0.0})
            fused[key]["score"] += 1.0 / (rrf_k + rank + 1)

        for rank, d in enumerate(bm25_docs):
            key = _doc_key(d)
            fused.setdefault(key, {"doc": d, "score": 0.0})
            fused[key]["score"] += 1.0 / (rrf_k + rank + 1)

        ranked = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
        docs = []
        for item in ranked:
            original = item["doc"]
            d = Document(page_content=original.page_content, metadata=dict(original.metadata))
            d.metadata["final_score"] = round(item["score"], 4)
            d.metadata["search_type"] = "hybrid"
            if degraded:
                d.metadata["fallback_reason"] = degraded
            docs.append(d)
        return docs

    def close(self):
        pass
