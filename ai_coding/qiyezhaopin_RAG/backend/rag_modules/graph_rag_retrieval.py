"""Bounded graph retrieval with stable job evidence; no model-authored Cypher."""
from langchain_core.documents import Document
from .query import JobFilters, canonical_skill, mentions, ALIASES


class GraphRAGRetrieval:
    def __init__(self, config, llm_client=None, snapshot=None, documents=None):
        self.config = config
        self.snapshot = snapshot
        self.documents = {d.metadata["node_id"]: d for d in (documents or [])}

    def initialize(self):
        if self.snapshot is None:
            raise RuntimeError("图索引版本未加载")
        self.snapshot.connect()

    def graph_rag_search(self, query, top_k=5, filters=None):
        if top_k <= 0:
            return []
        filters = filters or JobFilters()
        eligible = [d.metadata["node_id"] for d in self.documents.values() if filters.matches(d)]
        if not eligible:
            return []
        self.initialize()
        prerequisite_query = any(k in query for k in ("学习路径", "前置", "依赖", "先学"))
        prerequisite_paths = []
        targets = set()
        if prerequisite_query:
            available = {s for d in self.documents.values() for s in d.metadata.get("skills", [])}
            targets = {s for s in available if mentions(query, s)}
            for alias, name in ALIASES.items():
                if mentions(query, alias):
                    targets.add(name)
            if not targets:
                return []
            prerequisite_paths = self.snapshot.run("""
                MATCH path=(p:RAGEntity {version:$version,kind:'Skill'})
                      -[:PREREQUISITE_OF*1..3]->(s:RAGEntity {version:$version,kind:'Skill'})
                WHERE s.name IN $targets AND all(n IN nodes(path) WHERE n.version=$version)
                RETURN [n IN nodes(path)|n.name] AS names
                ORDER BY length(path) DESC LIMIT 20
            """, targets=sorted(targets))
            eligible = [node_id for node_id in eligible if targets & set(self.documents[node_id].metadata.get("skills", []))]
        rows = self.snapshot.run("""
            MATCH (j:RAGEntity {version:$version,kind:'Job'})-[:POSTED_BY]->(c:RAGEntity)
            WHERE j.nodeId IN $ids
            OPTIONAL MATCH (j)-[r:REQUIRES_SKILL]->(s:RAGEntity)
            RETURN j.nodeId AS id,c.name AS company,
                   collect({skill:s.name,required:r.isRequired}) AS skills
            ORDER BY j.nodeId LIMIT $limit
        """, ids=eligible, limit=max(top_k, min(100, len(eligible))))
        result = []
        for row in rows:
            doc = self.documents.get(row["id"])
            if doc is None:
                continue
            skills = [x["skill"] for x in row["skills"] if x["skill"]]
            if not set(filters.skills) <= {canonical_skill(s) for s in skills}:
                continue
            matched = [s for s in skills if canonical_skill(s) in filters.skills]
            paths = [f"{doc.metadata['job_name']} -[POSTED_BY]-> {row['company']}"]
            paths += [f"{doc.metadata['job_name']} -[REQUIRES_SKILL]-> {s}" for s in (matched or skills)]
            paths += [" -[PREREQUISITE_OF]-> ".join(p["names"]) for p in prerequisite_paths
                      if p["names"][-1] in skills]
            result.append(Document(page_content=doc.page_content + "\n图谱证据：\n" + "\n".join(paths),
                metadata={**doc.metadata, "search_type": "graph", "score": len(matched) + 1,
                          "graph_paths": paths}))
        return sorted(result, key=lambda d: (-d.metadata["score"], d.metadata["node_id"]))[:top_k]

    def close(self):
        pass
