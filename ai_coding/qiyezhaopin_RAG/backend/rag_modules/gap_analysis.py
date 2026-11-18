"""Reproducible skill coverage tools; no LLM or hiring-probability claims."""
import json
from pathlib import Path
from .query import canonical_skill


class GapAnalysisModule:
    def __init__(self, config, llm_client=None, data=None):
        self.config = config
        self.data = data
        self.driver = None

    def initialize(self):
        if self.data is None:
            self.data = json.loads(Path(self.config.seed_path).read_text(encoding="utf-8"))

    def close(self):
        pass

    def _candidate(self, name):
        self.initialize()
        return next((c for c in self.data.get("candidates", []) if name in (c["name"], c["nodeId"])), None)

    def _job(self, name):
        self.initialize()
        return next((j for j in self.data.get("jobs", []) if name in (j["name"], j["nodeId"])), None)

    def get_candidate_skills(self, candidate):
        c = self._candidate(candidate)
        return [s["name"] for s in c.get("skills", [])] if c else []

    def get_job_required_skills(self, job):
        j = self._job(job)
        return [s["name"] for s in j.get("skills", []) if s.get("isRequired", True)] if j else []

    def _learning_paths(self, skill, max_depth=3):
        self.initialize()
        prerequisites = {i["skill"]: i.get("prerequisites", []) for i in self.data.get("skill_prerequisites", [])}
        result = []
        def visit(current, path):
            parents = [p for p in prerequisites.get(current, []) if p not in path]
            if not parents or len(path) > max_depth:
                result.append(list(reversed(path)))
                return
            for parent in parents:
                visit(parent, path + [parent])
        visit(skill, [skill])
        return sorted(result, key=lambda p: (-len(p), p))[:5]

    def analyze_gap(self, candidate, job):
        c, j = self._candidate(candidate), self._job(job)
        if c is None or j is None:
            return {"error": "候选人或岗位不存在"}
        owned = {canonical_skill(s["name"]) for s in c.get("skills", [])}
        required = self.get_job_required_skills(job)
        matched = [s for s in required if canonical_skill(s) in owned]
        missing = [s for s in required if canonical_skill(s) not in owned]
        bonus = [s["name"] for s in j.get("skills", []) if not s.get("isRequired", True)]
        return {"candidate": c["name"], "job": j["name"],
                "match_rate": round(len(matched) / len(required), 3) if required else 1.0,
                "matched_skills": matched, "missing_skills": missing, "bonus_skills": bonus,
                "learning_paths": [{"skill": s, "prerequisites": self._learning_paths(s)[0],
                                    "all_paths": self._learning_paths(s)} for s in missing],
                "recommendation": "优先补充必需技能：" + "、".join(missing) if missing else "已覆盖列出的必需技能；仍需核实熟练度和经验。",
                "method": "required_skill_coverage", "limitation": "仅按技能名称计算覆盖率，不评价熟练度，也不代表录用概率"}

    def what_if(self, candidate, acquired_skills):
        c = self._candidate(candidate)
        if c is None:
            return {"error": "候选人不存在"}
        current = {canonical_skill(s) for s in self.get_candidate_skills(candidate)}
        expanded = current | {canonical_skill(s) for s in acquired_skills}
        opportunities = []
        for job in self.data["jobs"]:
            names = self.get_job_required_skills(job["nodeId"])
            required = {canonical_skill(s) for s in names}
            if not required:
                continue
            before, after = len(current & required) / len(required), len(expanded & required) / len(required)
            if after > before:
                opportunities.append({"job_id": job["nodeId"], "job_name": job["name"],
                    "before_rate": round(before, 3), "match_rate": round(after, 3),
                    "delta": round(after-before, 3), "newly_eligible": before < .5 <= after,
                    "newly_matched": sorted(s for s in names if canonical_skill(s) in expanded - current)})
        opportunities.sort(key=lambda x: (-x["delta"], -x["match_rate"], x["job_id"]))
        return {"candidate": candidate, "acquired_skills": acquired_skills,
                "new_opportunities": [o for o in opportunities if o["newly_eligible"]][:10],
                "improved_matches": opportunities[:20]}
