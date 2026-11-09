"""
图谱服务：子图提取、节点搜索、路径查找、人岗 Gap 分析
"""
import logging
from typing import List, Dict, Any

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(self, config):
        self.config = config
        self.driver = None

    def _connect(self):
        if self.driver is None:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
        return self.driver

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    @staticmethod
    def _node_to_dict(n) -> Dict[str, Any]:
        return {
            "name": n.get("name") or n.get("description"),
            "label": list(n.labels)[0] if n.labels else "",
            "properties": dict(n),
        }

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        self._connect()
        with self.driver.session(database=self.config.neo4j_database) as s:
            recs = s.run(
                "MATCH (n) WHERE n.name CONTAINS $q "
                "RETURN n.name AS name, labels(n)[0] AS label, id(n) AS id "
                "LIMIT $limit",
                q=query, limit=limit,
            )
            return [r.data() for r in recs]

    def get_subgraph(self, node_id: str, max_depth: int = 2, max_nodes: int = 100) -> Dict[str, Any]:
        self._connect()
        with self.driver.session(database=self.config.neo4j_database) as s:
            center_rec = s.run(
                "MATCH (n) WHERE n.name = $name OR n.nodeId = $name RETURN n",
                name=node_id,
            ).single()
            if not center_rec:
                return {"nodes": [], "edges": []}

            center = center_rec["n"]
            rec = s.run(
                f"""
                MATCH (center) WHERE center.name = $name OR center.nodeId = $name
                MATCH (center)-[r*1..{max_depth}]-(n)
                RETURN collect(DISTINCT n) AS nodes
                """,
                name=node_id,
            ).single()

            all_nodes = [center]
            if rec:
                all_nodes += list(rec["nodes"])
            all_nodes = all_nodes[:max_nodes]

            names = [n.get("name") for n in all_nodes if n.get("name") is not None]
            edges = []
            if names:
                edge_recs = s.run(
                    """
                    MATCH (a)-[r]->(b)
                    WHERE a.name IN $names AND b.name IN $names
                    RETURN a.name AS source, b.name AS target, type(r) AS type
                    """,
                    names=names,
                )
                edges = [e.data() for e in edge_recs]

        vis_nodes = []
        seen = set()
        for n in all_nodes:
            d = self._node_to_dict(n)
            if d["name"] in seen:
                continue
            seen.add(d["name"])
            vis_nodes.append({
                "id": d["name"],
                "label": d["name"],
                "type": d["label"],
                "properties": d["properties"],
            })
        return {"nodes": vis_nodes, "edges": edges}

    def find_path(self, from_node: str, to_node: str) -> Dict[str, Any]:
        self._connect()
        with self.driver.session(database=self.config.neo4j_database) as s:
            rec = s.run(
                """
                MATCH (a), (b)
                WHERE (a.name = $from OR a.nodeId = $from) AND (b.name = $to OR b.nodeId = $to)
                MATCH path = shortestPath((a)-[*..6]-(b))
                RETURN [n IN nodes(path) | n.name] AS nodes,
                       [r IN relationships(path) | type(r)] AS relations,
                       length(path) AS hops
                """,
                **{"from": from_node, "to": to_node},
            ).single()
            if not rec:
                return {"nodes": [], "relations": [], "hops": 0}
            return {
                "nodes": rec["nodes"],
                "relations": rec["relations"],
                "hops": rec["hops"],
            }