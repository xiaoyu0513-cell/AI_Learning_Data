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
    tokens = re.findall(r"[a-z0-9]+", text)
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(cn_chars) - 1):
        tokens.append(cn_chars[i] + cn_chars[i + 1])
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
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 索引构建完成，共 %d 篇文档", len(documents))

    def bm25_search(self, query: str, top_k: int) -> List[Document]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
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

    def hybrid_search(self, query: str, top_k: int = None, rrf_k: int = _RRF_K) -> List[Document]:
        """混合检索：向量 + BM25 → RRF 融合"""
        top_k = top_k or self.config.top_k
        vector_docs = self.milvus_module.vector_search(query, top_k)
        bm25_docs = self.bm25_search(query, top_k)

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
            d = item["doc"]
            d.metadata["final_score"] = round(item["score"], 4)
            d.metadata["search_type"] = "hybrid"
            docs.append(d)
        return docs

    def close(self):
        pass