"""Run from repository root: python backend/evaluation/run.py --mode offline.

offline = real BM25 only; full = existing version's vector/BM25/hybrid/graph/
rerank. No LLM calls. Retrieval quality is not answer accuracy or hallucination.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.query import JobFilters, parse_filters
from rag_modules.index_store import digest
from evaluation.metrics import score_ids, average


def run(mode="offline", k=5, output=None):
    gold_path = Path(__file__).with_name("gold.json")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    system = None
    if mode == "full":
        from system import RecruitmentGraphRAGSystem
        system = RecruitmentGraphRAGSystem(config)
        if not system.restore():
            raise RuntimeError("请先构建知识库后再运行 full 评测")
        data, docs, hybrid = system.snapshot.data, system.snapshot.documents, system.hybrid
    else:
        prep = GraphDataPreparationModule(config.seed_path, "", "", "")
        data = prep.load_seed_data()
        docs = prep.build_job_documents()
        hybrid = HybridRetrievalModule(config, None)
        hybrid.initialize(docs)
    by_id = {d.metadata["node_id"]: d for d in docs}
    variants = ["bm25"] if mode == "offline" else ["bm25", "vector", "hybrid", "graph", "rerank"]
    results = {}
    try:
        for variant in variants:
            if variant == "rerank" and not config.rerank_enabled:
                results[variant] = {"status": "skipped", "reason": "RERANK_ENABLED=false"}
                continue
            rows = []
            for case in gold["cases"]:
                query = case["query"]
                # Evaluate extraction separately: oracle filters never enter retrieval.
                parsed = parse_filters(query, data)
                expected = JobFilters(**case["filters"])
                started = time.perf_counter()
                error, retrieved = None, []
                try:
                    if variant == "bm25":
                        retrieved = hybrid.bm25_search(query, k, filters=parsed)
                    elif variant == "vector":
                        ids = [node_id for node_id, doc in by_id.items() if parsed.matches(doc)]
                        retrieved = system.milvus.vector_search(query, k, parsed.model_copy(update={"restrict_ids": True, "job_ids": ids}))
                    elif variant == "hybrid":
                        retrieved = hybrid.hybrid_search(query, k, filters=parsed)
                    elif variant == "graph":
                        retrieved = system.snapshot.router.graph_rag_retrieval.graph_rag_search(query, k, parsed)
                    else:
                        candidates = hybrid.hybrid_search(query, max(k, config.retrieval_candidates), filters=parsed)
                        retrieved = system.reranker.rerank(query, candidates, k)
                        if retrieved and retrieved[0].metadata.get("rerank_status") != "applied":
                            raise RuntimeError("reranker_unavailable")
                except Exception as exc:
                    error = type(exc).__name__
                ids = [d.metadata["node_id"] for d in retrieved]
                row = {"id": case["id"], "retrieved_ids": ids, **score_ids(ids, case["relevant_ids"], k),
                    "latency_ms": (time.perf_counter()-started)*1000,
                    "constraint_satisfaction": sum(expected.matches(d) for d in retrieved)/len(retrieved) if retrieved else None,
                    "filter_exact_match": parsed.model_dump() == expected.model_dump(), "error": error,
                    "fallback": any(d.metadata.get("fallback_reason") for d in retrieved)}
                rows.append(row)
            valid = [r for r in rows if not r["error"]]
            results[variant] = {"status": "ok" if len(valid) == len(rows) else "partial_failure", "cases": rows,
                "summary": {**{field: average(valid, field) for field in ("recall", "mrr", "hit", "abstention", "constraint_satisfaction", "filter_exact_match")},
                    "error_count": len(rows)-len(valid), "fallback_count": sum(r["fallback"] for r in rows),
                    "p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)),
                    "p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95))}}
        report = {"status": "ok", "created_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "k": k,
                  "case_count": len(gold["cases"]), "dataset_hash": digest(data), "gold_hash": digest(gold),
                  "index_version": system.snapshot.version if system else None, "llm_calls": 0,
                  "token_cost": None, "note": gold["description"], "results": results}
        path = Path(output or Path(config.state_dir)/"evaluation.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return report
    finally:
        if system:
            system.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "full"], default="offline")
    parser.add_argument("--k", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(args.mode, args.k, args.output)
    print(json.dumps({v: r.get("summary", r) for v, r in report["results"].items()}, indent=2))
    if any(r.get("status") == "partial_failure" for r in report["results"].values()):
        raise SystemExit(1)
