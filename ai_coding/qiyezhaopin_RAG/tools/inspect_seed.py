"""阅读练习：直接检查真实种子数据，不连接数据库，也不调用模型。

这里用 JSON 演示生产代码中的技能覆盖率公式。
正式接口从 Neo4j 读取技能，入口见 backend/rag_modules/gap_analysis.py。
"""
import json
from pathlib import Path


def main():
    # 第一个断点：观察项目的数据来源与顶层字段。
    seed_path = Path(__file__).resolve().parents[1] / "backend/data/seed/jobs.json"
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    counts = {key: len(value) for key, value in data.items()}
    print("数据规模：", counts)

    # 第二个断点：展开 job，观察 skills 内的 isRequired 与 proficiency。
    job = next(item for item in data["jobs"] if item["nodeId"] == "job_020")
    candidate = next(item for item in data["candidates"] if item["nodeId"] == "cand_001")
    print("目标岗位：", job["name"])
    print("候选人：", candidate["name"])

    # 第三个断点：逐行观察交集、差集和匹配率。
    # 与现有后端相同，当前将必需和加分技能都纳入计算。
    candidate_skills = {item["name"] for item in candidate["skills"]}
    required_skills = [item["name"] for item in job["skills"]]
    matched = [skill for skill in required_skills if skill in candidate_skills]
    missing = [skill for skill in required_skills if skill not in candidate_skills]
    match_rate = len(matched) / len(required_skills) if required_skills else 0.0
    print("已匹配：", matched)
    print("缺少：", missing)
    print(f"当前技能覆盖率：{len(matched)}/{len(required_skills)} = {match_rate:.0%}")

    # 第四个断点：模拟学会 RAG 与向量检索，只演示同一岗位的变化。
    # 正式 what_if() 会遍历所有岗位，并筛选覆盖率 >= 0.5 的前 10 个。
    acquired_skills = {"RAG", "向量检索"}
    future_skills = candidate_skills | acquired_skills
    future_matched = [skill for skill in required_skills if skill in future_skills]
    future_rate = len(future_matched) / len(required_skills) if required_skills else 0.0
    print("新增技能：", sorted(acquired_skills))
    print(f"学习后的技能覆盖率：{future_rate:.0%}")


if __name__ == "__main__":
    main()
