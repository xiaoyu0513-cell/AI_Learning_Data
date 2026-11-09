"""
图数据准备模块（招聘领域）
职责：
1. 读取种子数据 data/seed/jobs.json（岗位/技能/职责/分类/职级/公司/候选人 + 关系）
2. 连接 Neo4j 并写入招聘知识图谱（节点 + 关系）
3. 把每个岗位组织成一份文本「岗位档案」，供 Milvus 向量索引与 BM25 使用
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GraphDataPreparationModule:
    """招聘知识图谱的数据准备与建图模块"""

    def __init__(
        self,
        seed_path: str,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        self.seed_path = Path(seed_path)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[GraphDatabase.driver] = None
        self.data: Dict[str, Any] = {}

    # ========== 数据加载 ==========
    def load_seed_data(self) -> Dict[str, Any]:
        if not self.seed_path.exists():
            raise FileNotFoundError(f"种子数据不存在: {self.seed_path}")
        with open(self.seed_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info(
            "种子数据加载完成: %d 分类, %d 职级, %d 公司, %d 技能, %d 岗位, %d 候选人",
            len(self.data.get("categories", [])),
            len(self.data.get("seniority_levels", [])),
            len(self.data.get("companies", [])),
            len(self.data.get("skills", [])),
            len(self.data.get("jobs", [])),
            len(self.data.get("candidates", [])),
        )
        return self.data

    # ========== Neo4j 连接 ==========
    def _connect(self):
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self.driver

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    # ========== 图构建 ==========
    def create_constraints(self):
        self._connect()
        constraints = [
            "CREATE CONSTRAINT job_name IF NOT EXISTS FOR (n:Job) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (n:Skill) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (n:Company) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (n:JobCategory) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT level_name IF NOT EXISTS FOR (n:SeniorityLevel) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT candidate_name IF NOT EXISTS FOR (n:Candidate) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT responsibility_desc IF NOT EXISTS FOR (n:Responsibility) REQUIRE n.description IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for cypher in constraints:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.warning("约束创建失败(可忽略): %s -> %s", cypher, e)
        logger.info("唯一性约束创建完成")

    def build_graph(self):
        """整体建图：清空旧的 → 写入节点 → 写入关系"""
        self.load_seed_data()
        self.create_constraints()

        self._write_categories()
        self._write_seniority_levels()
        self._write_companies()
        self._write_skills()
        self._write_jobs()
        self._write_candidates()
        self._write_skill_prerequisites()
        logger.info("招聘知识图谱构建完成")

    def _write_categories(self):
        with self.driver.session(database=self.database) as session:
            for c in self.data.get("categories", []):
                session.run(
                    "MERGE (n:JobCategory {name: $name}) SET n.description = $description",
                    name=c["name"], description=c.get("description", ""),
                )

    def _write_seniority_levels(self):
        with self.driver.session(database=self.database) as session:
            for lvl in self.data.get("seniority_levels", []):
                session.run("MERGE (n:SeniorityLevel {name: $name})", name=lvl["name"])

    def _write_companies(self):
        with self.driver.session(database=self.database) as session:
            for c in self.data.get("companies", []):
                session.run(
                    """
                    MERGE (n:Company {name: $name})
                    SET n.industry = $industry, n.size = $size, n.location = $location
                    """,
                    name=c["name"],
                    industry=c.get("industry", ""),
                    size=c.get("size", ""),
                    location=c.get("location", ""),
                )

    def _write_skills(self):
        with self.driver.session(database=self.database) as session:
            for s in self.data.get("skills", []):
                session.run(
                    "MERGE (n:Skill {name: $name}) SET n.category = $category",
                    name=s["name"], category=s.get("category", ""),
                )

    def _write_jobs(self):
        with self.driver.session(database=self.database) as session:
            for job in self.data.get("jobs", []):
                session.run(
                    """
                    MERGE (j:Job {name: $name})
                    SET j.nodeId = $nodeId,
                        j.salaryRange = $salaryRange,
                        j.location = $location,
                        j.seniority = $seniority,
                        j.description = $description
                    """,
                    name=job["name"],
                    nodeId=job["nodeId"],
                    salaryRange=job.get("salaryRange", "面议"),
                    location=job.get("location", "不限"),
                    seniority=job.get("seniority", ""),
                    description=job.get("description", ""),
                )
                # 岗位 → 分类
                session.run(
                    """
                    MATCH (j:Job {name: $job}), (c:JobCategory {name: $category})
                    MERGE (j)-[:BELONGS_TO_CATEGORY]->(c)
                    """,
                    job=job["name"], category=job.get("category", ""),
                )
                # 岗位 → 职级
                session.run(
                    """
                    MATCH (j:Job {name: $job}), (l:SeniorityLevel {name: $level})
                    MERGE (j)-[:HAS_SENIORITY_LEVEL]->(l)
                    """,
                    job=job["name"], level=job.get("seniority", ""),
                )
                # 岗位 → 公司
                session.run(
                    """
                    MATCH (j:Job {name: $job}), (c:Company {name: $company})
                    MERGE (j)-[:POSTED_BY]->(c)
                    """,
                    job=job["name"], company=job.get("company", ""),
                )
                # 岗位 → 技能
                for sk in job.get("skills", []):
                    session.run(
                        """
                        MATCH (j:Job {name: $job}), (s:Skill {name: $skill})
                        MERGE (j)-[r:REQUIRES_SKILL]->(s)
                        SET r.proficiency = $proficiency, r.isRequired = $isRequired
                        """,
                        job=job["name"], skill=sk["name"],
                        proficiency=sk.get("proficiency", "熟悉"),
                        isRequired=bool(sk.get("isRequired", True)),
                    )
                # 岗位 → 职责
                for resp in job.get("responsibilities", []):
                    session.run(
                        """
                        MERGE (r:Responsibility {description: $desc})
                        WITH r
                        MATCH (j:Job {name: $job})
                        MERGE (j)-[hr:HAS_RESPONSIBILITY]->(r)
                        SET hr.order = $order
                        """,
                        desc=resp["description"], job=job["name"],
                        order=int(resp.get("order", 1)),
                    )

    def _write_candidates(self):
        with self.driver.session(database=self.database) as session:
            for cand in self.data.get("candidates", []):
                session.run(
                    """
                    MERGE (c:Candidate {name: $name})
                    SET c.nodeId = $nodeId,
                        c.yearsOfExperience = $years,
                        c.education = $education
                    """,
                    name=cand["name"],
                    nodeId=cand["nodeId"],
                    years=int(cand.get("yearsOfExperience", 0)),
                    education=cand.get("education", ""),
                )
                for sk in cand.get("skills", []):
                    session.run(
                        """
                        MATCH (c:Candidate {name: $name}), (s:Skill {name: $skill})
                        MERGE (c)-[r:HAS_SKILL]->(s)
                        SET r.proficiency = $proficiency, r.yearsOfExperience = $years
                        """,
                        name=cand["name"], skill=sk["name"],
                        proficiency=sk.get("proficiency", "熟悉"),
                        years=float(sk.get("yearsOfExperience", 0)),
                    )
                for app in cand.get("applications", []):
                    session.run(
                        """
                        MATCH (c:Candidate {name: $name}), (j:Job {name: $job})
                        MERGE (c)-[r:APPLIED_TO]->(j)
                        SET r.status = $status, r.appliedDate = $date
                        """,
                        name=cand["name"], job=app.get("job", ""),
                        status=app.get("status", ""),
                        date=app.get("appliedDate", ""),
                    )

    def _write_skill_prerequisites(self):
        with self.driver.session(database=self.database) as session:
            for item in self.data.get("skill_prerequisites", []):
                for prereq in item.get("prerequisites", []):
                    session.run(
                        """
                        MATCH (p:Skill {name: $prereq}), (s:Skill {name: $skill})
                        MERGE (p)-[:PREREQUISITE_OF]->(s)
                        """,
                        prereq=prereq, skill=item["skill"],
                    )

    # ========== 生成岗位档案文档（供向量检索/BM25） ==========
    def build_job_documents(self) -> List[Document]:
        if not self.data:
            self.load_seed_data()

        documents = []
        for job in self.data.get("jobs", []):
            skills = job.get("skills", [])
            skill_lines = []
            for i, s in enumerate(skills, 1):
                req = "必需" if s.get("isRequired", True) else "加分"
                skill_lines.append(f"{i}. {s['name']}（{s.get('proficiency', '熟悉')}，{req}）")

            resp_lines = []
            for r in sorted(job.get("responsibilities", []), key=lambda x: x.get("order", 0)):
                resp_lines.append(f"- {r['description']}")

            text = (
                f"岗位：{job['name']}\n"
                f"公司：{job.get('company', '')}\n"
                f"分类：{job.get('category', '')}\n"
                f"职级：{job.get('seniority', '')}\n"
                f"薪资范围：{job.get('salaryRange', '面议')}\n"
                f"工作地点：{job.get('location', '不限')}\n"
                f"岗位描述：{job.get('description', '')}\n"
                f"所需技能：\n" + "\n".join(skill_lines) + "\n"
                f"岗位职责：\n" + "\n".join(resp_lines)
            )
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "job_name": job["name"],
                        "node_id": job["nodeId"],
                        "company": job.get("company", ""),
                        "category": job.get("category", ""),
                        "location": job.get("location", ""),
                        "salary_range": job.get("salaryRange", ""),
                        "seniority": job.get("seniority", ""),
                    },
                )
            )

        logger.info("生成岗位档案文档 %d 篇", len(documents))
        return documents

    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        if self.driver:
            with self.driver.session(database=self.database) as session:
                for label in ["Job", "Skill", "Company", "JobCategory", "SeniorityLevel", "Candidate", "Responsibility"]:
                    r = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
                    stats[label] = r["c"] if r else 0
        return stats