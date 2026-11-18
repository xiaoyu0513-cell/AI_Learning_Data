"""Opt in: pytest --run-integration -m integration. Uses isolated namespaces."""
import copy
import json
from pathlib import Path
from uuid import uuid4
import pytest
from config import Settings
from system import RecruitmentGraphRAGSystem
from services.graph_service import GraphService
from rag_modules.query import JobFilters


@pytest.mark.integration
def test_real_versioned_build_search_delete_restore_rollback(tmp_path):
    namespace = "rag_it_" + uuid4().hex[:12]
    config = Settings(state_dir=str(tmp_path), milvus_collection_name=namespace)
    source = json.loads(Path(config.seed_path).read_text(encoding="utf8"))
    data = copy.deepcopy(source)
    data["jobs"], data["candidates"] = data["jobs"][:2], []
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
    config.seed_path = str(path)
    system = RecruitmentGraphRAGSystem(config)
    restored = None
    try:
        first = system.build()
        assert first["milvus"]["row_count"] == first["graph"]["Job"] == 2
        version = system.snapshot.version
        docs, _ = system.ask_docs("腾讯 Python", filters=JobFilters(company="腾讯"))
        assert [d.metadata["node_id"] for d in docs] == ["job_001"]
        graph = GraphService(config, snapshot_provider=lambda: system.snapshot)
        subgraph = graph.get_subgraph("job_001", 1, 30)
        assert any(n["type"] == "Responsibility" for n in subgraph["nodes"])
        assert any(e["type"] == "HAS_RESPONSIBILITY" for e in subgraph["edges"])
        assert len(graph.get_subgraph("job_001", 1, 1)["nodes"]) == 1
        assert graph.find_path("job_001", "Python")["hops"] == 1
        # Existing Skill nodes include the prerequisite chain even when no
        # selected job requires Kubernetes, so validate the directed graph path.
        assert graph.find_path("Linux", "Kubernetes")["hops"] >= 1
        data["jobs"].pop()
        data["jobs"][0]["description"] += " integration updated"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
        second = system.build()
        assert second["changes"]["deleted"] == ["job_002"]
        assert second["changes"]["updated"] == ["job_001"]
        assert second["milvus"]["row_count"] == second["graph"]["Job"] == 1
        assert not system.ask_docs("杭州 Django", filters=JobFilters(location="杭州"))[0]
        restored = RecruitmentGraphRAGSystem(config)
        assert restored.restore() and len(restored.snapshot.documents) == 1
        restored.rollback(version)
        assert len(restored.snapshot.documents) == 2
        assert restored.ask_docs("杭州 Django", filters=JobFilters(location="杭州"))[0][0].metadata["node_id"] == "job_002"
    finally:
        # Only remove resources created by this test's manifest and unique prefix.
        from pymilvus import utility
        from rag_modules.graph_snapshot import GraphSnapshot
        for row in system.store.versions():
            assert row["collection"].startswith(namespace + "_v_")
            if utility.has_collection(row["collection"]):
                utility.drop_collection(row["collection"])
            graph = GraphSnapshot(config, row["id"])
            graph.run("MATCH (n:RAGEntity {version:$version}) DETACH DELETE n")
            graph.close()
        system.cleanup()
        if restored:
            restored.cleanup()
