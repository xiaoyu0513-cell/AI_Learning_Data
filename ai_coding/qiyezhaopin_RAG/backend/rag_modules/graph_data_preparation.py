"""
图数据准备模块（招聘领域）
职责：
1. 读取种子数据 data/seed/jobs.json（岗位/技能/职责/分类/职级/公司/候选人 + 关系）
2. 校验岗位稳定 ID；图谱写入由 GraphSnapshot 负责
3. 把每个岗位组织成一份文本「岗位档案」，供 Milvus 向量索引与 BM25 使用
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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
        self.driver = None
        self.data: Dict[str, Any] = {}

    # ========== 数据加载 ==========
    def load_seed_data(self) -> Dict[str, Any]:
        if not self.seed_path.exists():
            raise FileNotFoundError(f"种子数据不存在: {self.seed_path}")
        with open(self.seed_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        ids = [j.get("nodeId") for j in self.data.get("jobs", [])]
        if not ids or any(not isinstance(i, str) or not i.strip() or len(i.encode()) > 128 for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("岗位必须具有非空且唯一的 nodeId，空知识库请勿发布")
        for job in self.data["jobs"]:
            for field in ("name", "company", "location", "seniority", "category"):
                if not isinstance(job.get(field), str) or not job[field].strip():
                    raise ValueError(f"岗位 {job['nodeId']} 的 {field} 必须是非空字符串")
            for skill in job.get("skills", []):
                if not isinstance(skill.get("name"), str) or not isinstance(skill.get("isRequired", True), bool):
                    raise ValueError("技能 name 必须是字符串，isRequired 必须是布尔值")
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

    def close(self):
        pass

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
                        "skills": [s["name"] for s in skills],
                        "required_skills": [s["name"] for s in skills if s.get("isRequired", True)],
                    },
                )
            )

        logger.info("生成岗位档案文档 %d 篇", len(documents))
        return documents
