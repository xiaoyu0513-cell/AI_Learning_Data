"""
智能查询路由器
根据查询的「复杂度」和「关系密集度」，自动把问题路由到：
- hybrid_traditional: 简单信息查找
- graph_rag: 复杂关系推理 / 多跳 / 知识发现
- combined: 两者结合
"""
import hashlib
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


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


class IntelligentQueryRouter:
    def __init__(self, traditional_retrieval, graph_rag_retrieval, llm_client, config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0,
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        prompt = f"""
你是招聘知识图谱问答系统的查询分析专家。请分析下面的查询，判断它适合哪种检索策略。

查询：{query}

请从以下维度分析并返回 JSON：
1. query_complexity (0-1)：0-0.3 简单事实查找（如"Python 后端岗位有哪些"）；
   0.4-0.7 中等（如"北京哪些公司招 RAG 岗位"）；0.8-1.0 复杂推理（如"会 Go 和 K8s 的人适合哪些 AI 基础设施岗位"）
2. relationship_intensity (0-1)：0-0.3 单一实体信息；0.4-0.7 实体间关系；
   0.8-1.0 复杂关系网络/多跳
3. reasoning_required (bool)：是否需要多跳/因果/对比推理
4. entity_count (int)：查询中明确的实体数量
5. recommended_strategy：
   - hybrid_traditional：简单直接的信息查找
   - graph_rag：复杂关系推理与知识发现（涉及"谁和谁什么关系/多跳/技能-岗位关联"等）
   - combined：两者结合
6. confidence (0-1)、reasoning (一句话理由)

只返回一个合法 JSON 对象，不要多余文字。
示例：{{"query_complexity":0.7,"relationship_intensity":0.8,"reasoning_required":true,"entity_count":3,"recommended_strategy":"graph_rag","confidence":0.85,"reasoning":"涉及技能到岗位到分类的多跳关系"}}
"""
        try:
            r = self.llm_client.chat_json(prompt)
            return QueryAnalysis(
                query_complexity=float(r.get("query_complexity", 0.5)),
                relationship_intensity=float(r.get("relationship_intensity", 0.5)),
                reasoning_required=bool(r.get("reasoning_required", False)),
                entity_count=int(r.get("entity_count", 1)),
                recommended_strategy=SearchStrategy(
                    r.get("recommended_strategy", "hybrid_traditional")
                ),
                confidence=float(r.get("confidence", 0.5)),
                reasoning=r.get("reasoning", "默认"),
            )
        except Exception as e:
            logger.error("查询分析失败: %s", e)
            return self._rule_based_analysis(query)

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """降级：基于关键词的规则判断"""
        relation_kw = ["关系", "哪些", "谁", "适合", "匹配", "多跳", "关联", "路径", "相关"]
        intensity = sum(1 for kw in relation_kw if kw in query) / len(relation_kw)
        strategy = SearchStrategy.GRAPH_RAG if intensity > 0.1 else SearchStrategy.HYBRID_TRADITIONAL
        return QueryAnalysis(
            query_complexity=intensity,
            relationship_intensity=intensity,
            reasoning_required=intensity > 0.2,
            entity_count=len(query),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的降级分析",
        )

    def route_query(self, query: str, top_k: int = None, max_rounds: int = None) -> Tuple[List, QueryAnalysis]:
        top_k = top_k or self.config.top_k
        analysis = self.analyze_query(query)
        self._update_stats(analysis.recommended_strategy)

        if analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
            docs = self.graph_rag_retrieval.graph_rag_search(query, top_k)
        elif analysis.recommended_strategy == SearchStrategy.COMBINED:
            docs = self._combined_search(query, top_k)
        else:
            docs = self.traditional_retrieval.hybrid_search(query, top_k)

        for d in docs:
            d.metadata["route_strategy"] = analysis.recommended_strategy.value
        return docs, analysis

    def _combined_search(self, query: str, top_k: int) -> List:
        gk = max(1, top_k // 2)
        tk = top_k - gk
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, gk)
        trad_docs = self.traditional_retrieval.hybrid_search(query, tk)

        combined, seen = [], set()
        for d in graph_docs + trad_docs:
            key = hashlib.md5(d.page_content[:100].encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                d.metadata["search_source"] = d.metadata.get("search_type", "traditional")
                combined.append(d)
        return combined[:top_k]

    def _update_stats(self, strategy: SearchStrategy):
        self.route_stats["total_queries"] += 1
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        else:
            self.route_stats["combined_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        return {
            **self.route_stats,
            "traditional_ratio": round(self.route_stats["traditional_count"] / total, 4),
            "graph_rag_ratio": round(self.route_stats["graph_rag_count"] / total, 4),
            "combined_ratio": round(self.route_stats["combined_count"] / total, 4),
        }

    def explain_routing_decision(self, query: str) -> str:
        a = self.analyze_query(query)
        return (
            f"复杂度: {a.query_complexity:.2f} | 关系密集度: {a.relationship_intensity:.2f} | "
            f"推理需求: {'是' if a.reasoning_required else '否'} | 实体数: {a.entity_count}\n"
            f"推荐策略: {a.recommended_strategy.value} (置信度 {a.confidence:.2f})\n"
            f"理由: {a.reasoning}"
        )