"""
Graph RAG 检索模块（招聘领域）
基于 Neo4j 招聘知识图谱的关系推理检索：
- 自然语言 → 图查询意图（5 类）
- 多跳图遍历（Cypher 动态路径 + 评分）
- 知识子图提取（邻居遍历 + 图密度指标）
- 实体直接关系查询
"""
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class QueryType(Enum):
    ENTITY_RELATION = "entity_relation"   # A 和 B 什么关系
    MULTI_HOP = "multi_hop"               # 多跳推理
    SUBGRAPH = "subgraph"                 # 子图提取
    PATH_FINDING = "path_finding"         # 路径查找
    CLUSTERING = "clustering"             # 相似聚类


@dataclass
class GraphQuery:
    query_type: QueryType
    source_entities: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    max_depth: int = 2
    max_nodes: int = 50


class GraphRAGRetrieval:
    """招聘知识图谱的图检索系统"""

    SCHEMA_HINT = """
已知图中有以下节点与关系：
- 节点：Job(岗位, name/salaryRange/location/seniority/description)、
        Skill(技能, name/category)、Responsibility(岗位职责, description)、
        JobCategory(岗位分类, name)、SeniorityLevel(职级, name)、
        Company(公司, name/industry/location)、Candidate(候选人, name/yearsOfExperience)
- 关系：REQUIRES_SKILL(岗位→技能)、HAS_RESPONSIBILITY(岗位→职责)、
        BELONGS_TO_CATEGORY(岗位→分类)、HAS_SENIORITY_LEVEL(岗位→职级)、
        POSTED_BY(岗位→公司)、HAS_SKILL(候选人→技能)、APPLIED_TO(候选人→岗位)、
        PREREQUISITE_OF(技能→技能，前置学习关系)
"""

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        self.entity_cache = {}
        self.relation_cache = {}

    # ========== 连接与索引 ==========
    def initialize(self):
        if self.driver is not None:
            return
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
            with self.driver.session(database=self.config.neo4j_database) as s:
                s.run("RETURN 1")
            logger.info("Neo4j 连接成功")
        except Exception as e:
            logger.error("Neo4j 连接失败: %s", e)
            return
        self._build_graph_index()

    def _build_graph_index(self):
        try:
            with self.driver.session(database=self.config.neo4j_database) as s:
                for record in s.run(
                    "MATCH (n) WHERE n.name IS NOT NULL "
                    "RETURN labels(n) AS labels, n.name AS name, id(n) AS nid"
                ):
                    name = record["name"]
                    self.entity_cache[name] = {
                        "labels": record["labels"], "id": record["nid"]
                    }
                for record in s.run(
                    "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c"
                ):
                    self.relation_cache[record["t"]] = record["c"]
            logger.info(
                "图索引完成: %d 实体, %d 关系类型",
                len(self.entity_cache), len(self.relation_cache),
            )
        except Exception as e:
            logger.error("构建图索引失败: %s", e)

    # ========== NL → 图查询意图 ==========
    def understand_graph_query(self, query: str) -> GraphQuery:
        prompt = f"""
{self.SCHEMA_HINT}
请分析下面这个关于招聘的自然语言问题，把它映射到图查询结构上。

问题：{query}

请识别并返回 JSON：
1. query_type：
   - entity_relation: 询问两个实体间的直接关系（如"张三应聘了哪些岗位"）
   - multi_hop: 需要多跳推理（如"会 Go 和 K8s 的人适合哪些 AI 基础设施岗位" → 技能→岗位→分类）
   - subgraph: 需要围绕某个核心实体的完整知识网络（如"Python 岗位都要求哪些技能"）
   - path_finding: 找两个实体之间的路径
2. source_entities：图中大概率存在对应节点的具体实体名（岗位名/技能名/公司名/候选人名/分类名）
3. target_entities：仅在需要限定"路径终点"时填写
4. relation_types：希望优先考虑的关系类型（如["REQUIRES_SKILL","BELONGS_TO_CATEGORY"]）
5. max_depth：建议遍历深度（1-3 的整数）

只返回一个合法 JSON 对象，不要多余文字。
示例：{{"query_type":"multi_hop","source_entities":["Go","Kubernetes","AI基础设施"],"target_entities":[],"relation_types":["REQUIRES_SKILL","BELONGS_TO_CATEGORY"],"max_depth":2}}
"""
        try:
            result = self.llm_client.chat_json(prompt)
            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=int(result.get("max_depth", 2)),
            )
        except Exception as e:
            logger.error("意图理解失败: %s", e)
            return GraphQuery(QueryType.SUBGRAPH, source_entities=[query], max_depth=2)

    # ========== 图检索主入口 ==========
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.driver:
            return []

        gq = self.understand_graph_query(query)
        logger.info("图查询意图: %s -> %s", gq.query_type.value, gq.source_entities)

        results: List[Document] = []
        try:
            if gq.query_type == QueryType.ENTITY_RELATION:
                results = self._entity_relation_search(gq)
            elif gq.query_type in (QueryType.MULTI_HOP, QueryType.PATH_FINDING):
                results = self._multi_hop_search(gq)
            else:  # SUBGRAPH / CLUSTERING
                results = self._subgraph_search(gq, query)

            results = sorted(
                results,
                key=lambda x: x.metadata.get("relevance_score", 0.0),
                reverse=True,
            )
            return results[:top_k]
        except Exception as e:
            logger.error("图检索失败: %s", e)
            return []

    # ========== 三种检索策略 ==========
    def _multi_hop_search(self, gq: GraphQuery) -> List[Document]:
        cypher = """
        UNWIND $sources AS source_name
        MATCH (source)
        WHERE source.name CONTAINS source_name OR source.name = source_name
        MATCH path = (source)-[*1..$max_depth]-(target)
        WHERE source <> target
        WITH path, source, target, length(path) AS path_len,
             relationships(path) AS rels, nodes(path) AS pnodes
        WITH path, source, target, path_len, rels, pnodes,
             (1.0 / path_len) AS relevance
        ORDER BY relevance DESC, path_len ASC
        LIMIT 20
        RETURN path, source, target, path_len, rels, pnodes, relevance
        """
        docs = []
        with self.driver.session(database=self.config.neo4j_database) as s:
            for rec in s.run(
                cypher,
                sources=gq.source_entities or [""],
                max_depth=gq.max_depth,
            ):
                nodes = [
                    {"name": n.get("name", ""), "labels": list(n.labels)}
                    for n in rec["pnodes"]
                ]
                rels = [{"type": r.type} for r in rec["rels"]]
                desc = " -> ".join([n["name"] for n in nodes])
                docs.append(
                    Document(
                        page_content=f"关系路径: {desc}",
                        metadata={
                            "search_type": "graph_path",
                            "relevance_score": rec["relevance"],
                            "path_length": rec["path_len"],
                            "nodes": nodes,
                            "relationships": rels,
                        },
                    )
                )
        logger.info("多跳遍历完成: %d 条", len(docs))
        return docs

    def _subgraph_search(self, gq: GraphQuery, query: str) -> List[Document]:
        cypher = """
        UNWIND $sources AS entity_name
        MATCH (source)
        WHERE source.name CONTAINS entity_name OR source.name = entity_name
        MATCH (source)-[r*1..$max_depth]-(neighbor)
        WITH source,
             collect(DISTINCT neighbor) AS neighbors,
             [rel in collect(DISTINCT r) | rel] AS rels
        RETURN source,
               neighbors[0..$max_nodes] AS nodes,
               rels[0..$max_nodes] AS relationships,
               size(neighbors) AS node_count,
               size(rels) AS rel_count
        """
        docs = []
        with self.driver.session(database=self.config.neo4j_database) as s:
            record = s.run(
                cypher,
                sources=gq.source_entities or [""],
                max_depth=gq.max_depth,
                max_nodes=gq.max_nodes,
            ).single()
            if not record:
                return docs
            central = dict(record["source"])
            nodes = [dict(n) for n in record["nodes"]]
            rels = [dict(r) for r in record["relationships"]]
            node_names = [n.get("name", "?") for n in nodes]
            density = (
                record["rel_count"] / (record["node_count"] * (record["node_count"] - 1) / 2)
                if record["node_count"] > 1 else 0.0
            )
            docs.append(
                Document(
                    page_content=(
                        f"围绕 {central.get('name','?')} 的知识子图："
                        f"关联 {', '.join(node_names)}"
                    ),
                    metadata={
                        "search_type": "knowledge_subgraph",
                        "central_name": central.get("name", ""),
                        "relevance_score": 0.5,
                        "node_count": record["node_count"],
                        "relationship_count": record["rel_count"],
                        "graph_density": round(density, 4),
                        "nodes": nodes,
                        "relationships": rels,
                    },
                )
            )
        return docs

    def _entity_relation_search(self, gq: GraphQuery) -> List[Document]:
        if not gq.source_entities:
            return []
        cypher = """
        UNWIND $sources AS entity_name
        MATCH (source)
        WHERE source.name CONTAINS entity_name OR source.name = entity_name
        MATCH (source)-[r]-(neighbor)
        RETURN source, r, neighbor,
               type(r) AS rel_type
        LIMIT 30
        """
        docs = []
        with self.driver.session(database=self.config.neo4j_database) as s:
            for rec in s.run(cypher, sources=gq.source_entities):
                source = rec["source"]
                neighbor = rec["neighbor"]
                rel_type = rec["rel_type"]
                docs.append(
                    Document(
                        page_content=(
                            f"{source.get('name','?')} --[{rel_type}]--> "
                            f"{neighbor.get('name','?')}"
                        ),
                        metadata={
                            "search_type": "entity_relation",
                            "relevance_score": 1.0,
                            "relation_type": rel_type,
                        },
                    )
                )
        seen, uniq = set(), []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                uniq.append(d)
        return uniq

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None