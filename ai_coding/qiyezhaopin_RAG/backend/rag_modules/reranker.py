"""Lazy cross-encoder. Disabled/unavailable models are explicitly observable."""
import logging
from threading import Lock
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.lock = Lock()
        self.load_failed = False

    def rerank(self, query, documents, top_k):
        status = "disabled"
        scores = None
        if self.config.rerank_enabled and documents:
            try:
                with self.lock:
                    if self.model is None:
                        if self.load_failed:
                            raise RuntimeError("重排序模型尚未就绪，请安装模型后重启")
                        from sentence_transformers import CrossEncoder
                        try:
                            self.model = CrossEncoder(self.config.rerank_model, local_files_only=True)
                        except Exception:
                            self.load_failed = True
                            raise
                    scores = self.model.predict([(query, d.page_content) for d in documents],
                                                batch_size=self.config.rerank_batch_size)
                if len(scores) != len(documents):
                    raise ValueError("重排序分数数量不匹配")
                import math
                if not all(math.isfinite(float(s)) for s in scores):
                    raise ValueError("重排序返回非有限分数")
                status = "applied"
            except Exception:
                logger.exception("重排序不可用，保留融合顺序")
                scores = None
                status = "unavailable"
        result = []
        for i, doc in enumerate(documents):
            meta = {**doc.metadata, "rerank_status": status}
            if scores is not None:
                meta["rerank_score"] = float(scores[i])
            result.append(Document(page_content=doc.page_content, metadata=meta))
        if scores is not None:
            result.sort(key=lambda d: -d.metadata["rerank_score"])
        return result[:top_k]
