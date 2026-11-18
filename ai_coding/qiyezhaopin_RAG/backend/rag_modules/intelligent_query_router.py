"""Explainable low-cost routing and graph -> hybrid -> BM25 fallback."""
from dataclasses import dataclass
from enum import Enum
from collections import Counter
from langchain_core.documents import Document


class SearchStrategy(Enum):
    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"


@dataclass
class QueryAnalysis:
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    recommended_strategy: SearchStrategy
    confidence: float
    reasoning: str
    effective_strategy: str = ""
    fallback_reason: str = ""


class IntelligentQueryRouter:
    def __init__(self, traditional_retrieval, graph_rag_retrieval, llm_client, config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.config = config
        self.route_stats = Counter()

    def analyze_query(self, query, filters=None):
        relation = any(k in query for k in ("关系", "关联", "同时", "匹配", "适合", "学习路径", "前置", "依赖", "先学"))
        count = len(filters.skills) if filters else 0
        strategy = SearchStrategy.COMBINED if relation or count > 1 else SearchStrategy.HYBRID_TRADITIONAL
        return QueryAnalysis(.7 if relation else .3, .8 if relation else .2, relation,
                             count, strategy, .8, "结构化技能交集与语义证据结合" if strategy == SearchStrategy.COMBINED else "事实查询使用混合检索")

    def route_query(self, query, top_k=None, max_rounds=None, filters=None, enable_agentic=True):
        top_k = self.config.top_k if top_k is None else top_k
        analysis = self.analyze_query(query, filters)
        if not enable_agentic:
            analysis.recommended_strategy = SearchStrategy.HYBRID_TRADITIONAL
            analysis.reasoning = "已关闭图谱路由"
        strategy = analysis.recommended_strategy
        analysis.effective_strategy = strategy.value
        self.route_stats["total_queries"] += 1
        self.route_stats[strategy.value] += 1
        if top_k <= 0:
            return [], analysis
        lists = []
        if strategy in (SearchStrategy.GRAPH_RAG, SearchStrategy.COMBINED):
            try:
                graph = self.graph_rag_retrieval.graph_rag_search(query, top_k, filters=filters)
                if graph:
                    lists.append(graph)
                else:
                    analysis.fallback_reason = "graph_empty"
            except Exception:
                analysis.fallback_reason = "graph_unavailable"
            if analysis.fallback_reason:
                analysis.effective_strategy = SearchStrategy.HYBRID_TRADITIONAL.value
                self.route_stats["graph_fallbacks"] += 1
        if strategy != SearchStrategy.GRAPH_RAG or not lists:
            lists.append(self.traditional_retrieval.hybrid_search(query, top_k, filters=filters))
        fused = {}
        for docs in lists:
            for rank, d in enumerate(docs):
                key = d.metadata["node_id"]
                entry = fused.setdefault(key, [d, 0.0])
                entry[1] += 1 / (61 + rank)
        result = []
        for doc, score in sorted(fused.values(), key=lambda item: -item[1])[:top_k]:
            result.append(Document(page_content=doc.page_content, metadata={**doc.metadata,
                "final_score": score, "route_strategy": analysis.effective_strategy,
                "fallback_reason": analysis.fallback_reason or doc.metadata.get("fallback_reason", "")}))
        return result, analysis

    def get_route_statistics(self):
        return dict(self.route_stats)
