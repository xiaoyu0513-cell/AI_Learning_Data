"""Graph exploration pinned to the active RAG snapshot."""
from rag_modules.graph_snapshot import RELATIONS


class GraphService:
    def __init__(self, config, snapshot_provider=None):
        self.config = config
        self.snapshot_provider = snapshot_provider

    def close(self):
        pass

    def _graph(self):
        if not self.snapshot_provider:
            raise RuntimeError("活动图谱未初始化")
        return self.snapshot_provider().graph

    def search_nodes(self, query, limit=10):
        return self._graph().run(
            "MATCH (n:RAGEntity {version:$version}) WHERE n.name CONTAINS $q "
            "RETURN n.name AS name,n.kind AS label,n.nodeId AS id ORDER BY name LIMIT $limit",
            q=query, limit=max(1, min(50, limit)))

    def get_subgraph(self, node_id, max_depth=2, max_nodes=100):
        if not 1 <= max_depth <= 3 or not 1 <= max_nodes <= 150:
            raise ValueError("图谱深度应为1至3，节点上限应为1至150")
        graph = self._graph()
        # Limit neighbour expansion in Cypher, not after collecting an unbounded graph.
        nodes = graph.run(f"""
            MATCH (center:RAGEntity {{version:$version}})
            WHERE center.nodeId=$id OR center.name=$id
            OPTIONAL MATCH path=(center)-[*1..{int(max_depth)}]-(n:RAGEntity)
            WHERE all(v IN nodes(path) WHERE v.version=$version)
            WITH center,n ORDER BY n.key LIMIT $limit
            RETURN center,collect(DISTINCT n) AS neighbours
        """, id=node_id, limit=max(1, max_nodes-1))
        if not nodes:
            return {"nodes": [], "edges": []}
        found = {n["key"]: n for row in nodes for n in [row["center"]] + row["neighbours"]}
        found = dict(list(found.items())[:max_nodes])
        edges = graph.run("""
            MATCH (a:RAGEntity {version:$version})-[r]->(b:RAGEntity {version:$version})
            WHERE a.key IN $keys AND b.key IN $keys
            RETURN a.nodeId AS source,b.nodeId AS target,type(r) AS type
        """, keys=list(found))
        return {"nodes": [{"id": n["nodeId"], "label": n["name"], "type": n["kind"],
                           "properties": dict(n)} for n in found.values()], "edges": edges}

    def find_path(self, from_node, to_node):
        result = self._graph().run("""
            MATCH (a:RAGEntity {version:$version}),(b:RAGEntity {version:$version})
            WHERE (a.nodeId=$source OR a.name=$source) AND (b.nodeId=$target OR b.name=$target)
            MATCH path=shortestPath((a)-[*..6]-(b))
            WHERE all(n IN nodes(path) WHERE n.version=$version)
            RETURN [n IN nodes(path)|n.name] AS nodes,
                   [r IN relationships(path)|type(r)] AS relations,length(path) AS hops
            LIMIT 1
        """, source=from_node, target=to_node)
        return result[0] if result else {"nodes": [], "relations": [], "hops": 0}
