"""Deterministic recruitment constraints shared by all retrieval branches."""
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

ALIASES = {"k8s": "Kubernetes", "golang": "Go", "springboot": "Spring Boot",
           "pg": "PostgreSQL", "es": "Elasticsearch", "ts": "TypeScript",
           "js": "JavaScript", "cpp": "C++"}
CITY_NAMES = {"北京", "上海", "广州", "深圳", "杭州", "合肥", "南京", "成都", "武汉", "西安", "苏州", "天津", "重庆"}


def canonical_skill(value: str) -> str:
    return ALIASES.get(value.strip().lower(), value.strip()).casefold()


def mentions(query: str, term: str) -> bool:
    # Do not match Java inside JavaScript, or Go inside Django.
    pattern = re.escape(term.casefold())
    if re.search(r"[a-z0-9]", term, re.I):
        pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9+#])"
    return bool(re.search(pattern, query.casefold()))


class JobFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: Optional[str] = Field(None, min_length=1, max_length=80)
    company: Optional[str] = Field(None, min_length=1, max_length=128)
    seniority: Optional[str] = Field(None, min_length=1, max_length=40)
    skills: List[str] = Field(default_factory=list, max_length=20)
    job_ids: List[str] = Field(default_factory=list, max_length=100)
    restrict_ids: bool = False

    @field_validator("skills")
    @classmethod
    def normalize(cls, values):
        if any(not x.strip() or len(x) > 100 for x in values):
            raise ValueError("技能名称不能为空或超过100字符")
        return sorted({canonical_skill(x) for x in values})

    def matches(self, document) -> bool:
        m = document.metadata
        for field in ("location", "company", "seniority"):
            value = getattr(self, field)
            if value and value.casefold() != str(m.get(field, "")).casefold():
                return False
        if self.restrict_ids and m.get("node_id") not in self.job_ids:
            return False
        available = {canonical_skill(x) for x in m.get("skills", [])}
        return set(self.skills) <= available


def parse_filters(query: str, data: dict, explicit=None) -> JobFilters:
    result = {}
    for field, values in (
        ("location", CITY_NAMES | {j.get("location", "") for j in data.get("jobs", [])}),
        ("company", {j.get("company", "") for j in data.get("jobs", [])}),
        ("seniority", {j.get("seniority", "") for j in data.get("jobs", [])}),
    ):
        found = [v for v in values if v and mentions(query, v)]
        # Ambiguous comparisons across two cities must not silently choose one.
        if len(found) == 1:
            result[field] = found[0]
    if "应届" in query and "seniority" not in result:
        result["seniority"] = "应届生"
    skills = {s["name"] for s in data.get("skills", [])}
    found_skills = {canonical_skill(s) for s in skills if mentions(query, s)}
    for alias, skill in ALIASES.items():
        if mentions(query, alias):
            found_skills.add(canonical_skill(skill))
    # Capability filters only when asking to require/use skills, not asking
    # "这个岗位需要学什么 Python" or "不会 Python".
    if not re.search(r"不会|不要求|无需|不需要|区别|对比|学习路径|先学|前置|依赖", query):
        result["skills"] = sorted(found_skills)
    if explicit:
        result.update(explicit.model_dump(exclude_unset=True))
    return JobFilters(**result)
