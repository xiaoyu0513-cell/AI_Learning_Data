"""Versioned Neo4j projection. Legacy nodes/constraints are left untouched."""
from neo4j import GraphDatabase, Query
from .index_store import digest

RELATIONS = {"BELONGS_TO_CATEGORY", "HAS_SENIORITY_LEVEL", "POSTED_BY",
             "REQUIRES_SKILL", "HAS_RESPONSIBILITY", "HAS_SKILL", "APPLIED_TO", "PREREQUISITE_OF"}


class GraphSnapshot:
    def __init__(self, config, version):
        self.config, self.version, self.driver = config, version, None

    def connect(self):
        if self.driver is None:
            driver = GraphDatabase.driver(self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                connection_timeout=5, connection_acquisition_timeout=5,
                max_transaction_retry_time=3)
            try:
                driver.verify_connectivity()
            except Exception:
                driver.close()
                raise
            self.driver = driver
        return self.driver

    def run(self, cypher, **params):
        self.connect()
        with self.driver.session(database=self.config.neo4j_database) as s:
            return [r.data() for r in s.run(Query(cypher, timeout=8), version=self.version, **params)]

    def build(self, data):
        self.connect()
        nodes, edges, lookup = [], [], {}

        def node(kind, name, properties, identity=None):
            identity = identity or digest([kind, name])
            key = f"{self.version}:{kind}:{identity}"
            nodes.append({"key": key, "kind": kind, "name": name,
                          "nodeId": identity, "properties": properties})
            lookup[kind, name] = key
            return key

        def edge(source, relation, kind, target, properties=None):
            if (kind, target) not in lookup:
                raise ValueError(f"未知图实体: {kind}/{target}")
            edges.append({"source": source, "type": relation,
                          "target": lookup[kind, target], "properties": properties or {}})

        for field, kind in (("categories", "JobCategory"), ("seniority_levels", "SeniorityLevel"),
                            ("companies", "Company"), ("skills", "Skill")):
            for item in data.get(field, []):
                node(kind, item["name"], item)
        for job in data["jobs"]:
            key = node("Job", job["name"], {k: v for k, v in job.items()
                        if k not in ("skills", "responsibilities")}, job["nodeId"])
            for field, rel, kind in (("category", "BELONGS_TO_CATEGORY", "JobCategory"),
                                      ("seniority", "HAS_SENIORITY_LEVEL", "SeniorityLevel"),
                                      ("company", "POSTED_BY", "Company")):
                edge(key, rel, kind, job[field])
            for skill in job.get("skills", []):
                edge(key, "REQUIRES_SKILL", "Skill", skill["name"],
                     {"proficiency": skill.get("proficiency", "熟悉"), "isRequired": skill.get("isRequired", True)})
            for index, resp in enumerate(job.get("responsibilities", [])):
                rid = digest([job["nodeId"], resp["description"]])
                node("Responsibility", resp["description"], resp, rid)
                edge(key, "HAS_RESPONSIBILITY", "Responsibility", resp["description"], {"order": resp.get("order", index)})
        for cand in data.get("candidates", []):
            key = node("Candidate", cand["name"], {k: v for k, v in cand.items()
                       if k not in ("skills", "applications")}, cand["nodeId"])
            for skill in cand.get("skills", []):
                edge(key, "HAS_SKILL", "Skill", skill["name"],
                     {k: v for k, v in skill.items() if k != "name"})
            for app in cand.get("applications", []):
                edge(key, "APPLIED_TO", "Job", app["job"],
                     {k: v for k, v in app.items() if k != "job"})
        for item in data.get("skill_prerequisites", []):
            for name in item.get("prerequisites", []):
                edge(lookup["Skill", name], "PREREQUISITE_OF", "Skill", item["skill"])

        with self.driver.session(database=self.config.neo4j_database) as s:
            s.run("CREATE CONSTRAINT rag_entity_key IF NOT EXISTS FOR (n:RAGEntity) REQUIRE n.key IS UNIQUE").consume()
            s.run("CREATE INDEX rag_entity_version IF NOT EXISTS FOR (n:RAGEntity) ON (n.version)").consume()
            def write(tx):
                tx.run("UNWIND $nodes AS row CREATE (n:RAGEntity {key:row.key,version:$version}) "
                       "SET n += row.properties, n.kind=row.kind, n.name=row.name, n.nodeId=row.nodeId",
                       nodes=nodes, version=self.version).consume()
                for relation in sorted(RELATIONS):
                    rows = [e for e in edges if e["type"] == relation]
                    tx.run(f"UNWIND $edges AS row MATCH (a:RAGEntity {{key:row.source}}), "
                           f"(b:RAGEntity {{key:row.target}}) CREATE (a)-[r:{relation}]->(b) SET r += row.properties",
                           edges=rows).consume()
            s.execute_write(write)
        return self.statistics()

    def statistics(self):
        return {r["kind"]: r["count"] for r in self.run(
            "MATCH (n:RAGEntity {version:$version}) RETURN n.kind AS kind,count(n) AS count")}

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
