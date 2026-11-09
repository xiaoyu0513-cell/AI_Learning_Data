"""
人岗匹配 Gap 分析与 What-if 推演（招聘领域）
- analyze_gap：分析候选人与岗位的技能差距 + 匹配率 + 学习路径 + 建议
- what_if：假设候选人获得某些技能后，能匹配哪些新岗位
"""
import logging
from typing import Dict, List

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GapAnalysisModule:
    def __init__(self, config, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

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
            logger.info("Gap 分析 Neo4j 连接成功")
        except Exception as e:
            logger.error("Gap 分析 Neo4j 连接失败: %s", e)

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def get_candidate_skills(self, candidate: str) -> List[str]:
        if not self.driver:
            return []
        with self.driver.session(database=self.config.neo4j_database) as s:
            rec = s.run(
                "MATCH (c:Candidate {name: $name})-[:HAS_SKILL]->(s:Skill) "
                "RETURN collect(s.name) AS skills",
                name=candidate,
            ).single()
        return list(rec["skills"]) if rec else []

    def get_job_required_skills(self, job: str) -> List[str]:
        if not self.driver:
            return []
        with self.driver.session(database=self.config.neo4j_database) as s:
            rec = s.run(
                "MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill) "
                "WHERE j.name = $name OR j.nodeId = $name "
                "RETURN collect(s.name) AS skills",
                name=job,
            ).single()
        return list(rec["skills"]) if rec else []

    def _learning_paths(self, skill: str, max_depth: int = 3) -> List[List[str]]:
        if not self.driver:
            return []
        with self.driver.session(database=self.config.neo4j_database) as s:
            results = s.run(
                f"MATCH path = (s:Skill {{name: $skill}})<-[:PREREQUISITE_OF*1..{max_depth}]-(p:Skill) "
                "RETURN [n IN nodes(path) | n.name] AS path, length(path) AS hops "
                "ORDER BY hops DESC LIMIT 5",
                skill=skill,
            )
            return [r["path"] for r in results]

    def analyze_gap(self, candidate: str, job: str) -> Dict:
        """分析候选人与岗位的技能 Gap（按名字/ nodeId 匹配）"""
        self.initialize()
        if not self.driver:
            return {"error": "Neo4j 不可用"}

        candidate_skills = self.get_candidate_skills(candidate)
        required_skills = self.get_job_required_skills(job)
        if not required_skills:
            return {"error": "未找到该岗位或其技能要求"}

        matched = [s for s in required_skills if s in candidate_skills]
        missing = [s for s in required_skills if s not in candidate_skills]
        total = len(required_skills)
        match_rate = len(matched) / total if total else 0.0

        learning_paths = []
        for skill in missing:
            paths = self._learning_paths(skill)
            learning_paths.append({
                "skill": skill,
                "prerequisites": paths[0] if paths else [],
            })

        return {
            "candidate": candidate,
            "job": job,
            "match_rate": round(match_rate, 3),
            "matched_skills": matched,
            "missing_skills": missing,
            "learning_paths": learning_paths,
            "recommendation": self._generate_recommendation(match_rate, matched, missing),
        }

    def what_if(self, candidate: str, acquired_skills: List[str]) -> Dict:
        """What-if 推演：假设候选人获得某些技能，能匹配哪些新岗位"""
        self.initialize()
        if not self.driver:
            return {"error": "Neo4j 不可用"}

        current_skills = set(self.get_candidate_skills(candidate))
        new_skills = current_skills | set(acquired_skills)

        with self.driver.session(database=self.config.neo4j_database) as s:
            jobs = [
                r.data()
                for r in s.run(
                    "MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill) "
                    "WITH j, collect(s.name) AS required "
                    "RETURN j.nodeId AS job_id, j.name AS job_name, required"
                )
            ]

        opportunities = []
        for job in jobs:
            required = set(job["required"])
            if not required:
                continue
            matched = new_skills & required
            match_rate = len(matched) / len(required)
            if match_rate >= 0.5:
                opportunities.append({
                    "job_id": job["job_id"],
                    "job_name": job["job_name"],
                    "match_rate": round(match_rate, 3),
                    "newly_matched": list(matched - current_skills),
                })

        opportunities.sort(key=lambda x: x["match_rate"], reverse=True)
        return {
            "candidate": candidate,
            "acquired_skills": acquired_skills,
            "new_opportunities": opportunities[:10],
        }

    def _generate_recommendation(self, match_rate: float, matched: List[str], missing: List[str]) -> str:
        if match_rate >= 0.8:
            return f"高度匹配。建议投递，重点突出：{', '.join(matched[:3])}"
        elif match_rate >= 0.5:
            return f"中等匹配。建议补充：{', '.join(missing[:3])}"
        else:
            return f"匹配度较低。需要系统学习：{', '.join(missing[:5])}"