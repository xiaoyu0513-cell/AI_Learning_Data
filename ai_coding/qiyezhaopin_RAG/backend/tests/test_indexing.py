import copy
from pathlib import Path
import json
import pytest
from rag_modules.index_store import IndexStore
from rag_modules.milvus_index import MilvusIndexConstructionModule
from system import RecruitmentGraphRAGSystem


def test_manifest_publish_failure_and_rollback(tmp_path):
    store = IndexStore(tmp_path)
    first, _ = store.prepare("one", "jobs", {}, [])
    store.publish(first)
    second, _ = store.prepare("two", "jobs", {}, [])
    store.fail(second, RuntimeError("private message"))
    assert store.active()["id"] == first
    with pytest.raises(ValueError):
        store.publish(second)
    assert "private message" not in json.dumps(store.versions())
    third, _ = store.prepare("three", "jobs", {}, [])
    store.publish(third)
    store.publish(first)
    assert IndexStore(tmp_path).active()["id"] == first


def test_build_lock_excludes_second_writer(tmp_path):
    store = IndexStore(tmp_path)
    with store.build_lock():
        with pytest.raises(RuntimeError):
            with store.build_lock():
                pass
    with store.build_lock():
        pass


def test_model_import_is_lazy(settings):
    system = RecruitmentGraphRAGSystem(settings)
    assert not system.ready
    assert system._store is None
    milvus = system._new_milvus("test")
    assert milvus._embedder is None and not milvus._connected


def test_incremental_vector_cache_reencodes_only_changed(settings, tmp_path):
    from langchain_core.documents import Document
    class Encoder:
        def __init__(self): self.calls = []
        def encode_documents(self, texts):
            self.calls += texts
            return [[1., 0.] for _ in texts]
    class Collection:
        def upsert(self, rows): self.rows = rows
        def flush(self): pass
        def load(self, **kwargs): pass
    store = IndexStore(tmp_path)
    module = MilvusIndexConstructionModule("", 0, "test", 2, "test-model")
    encoder = Encoder()
    module._embedder, module.collection = encoder, Collection()
    module.has_collection = lambda: False
    module._create_collection = lambda: None
    docs = [Document(page_content="one", metadata={"node_id":"job1", "content_hash":"1"}),
            Document(page_content="two", metadata={"node_id":"job2", "content_hash":"2"})]
    module.build_vector_index(docs, store)
    assert len(encoder.calls) == 2
    module.build_vector_index(docs, store)
    assert len(encoder.calls) == 2
    docs[0].page_content = "changed"
    module.build_vector_index(docs, store)
    assert encoder.calls == ["one", "two", "changed"]
    assert {row["id"] for row in module.collection.rows} == {"job1", "job2"}


def test_update_delete_failure_restore_and_rollback(settings, tmp_path, monkeypatch):
    import system as system_module
    original = json.loads(Path(settings.seed_path).read_text(encoding="utf-8"))
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    settings.seed_path = str(path)
    vectors, graphs = {}, {}
    class Vector:
        def __init__(self, collection): self.collection = collection
        def build_vector_index(self, docs, store): vectors[self.collection] = docs
        def load_collection(self): return self.collection in vectors
        def get_collection_stats(self): return {"row_count":len(vectors[self.collection])}
    class Graph:
        fail = False
        def __init__(self, config, version): self.version = version
        def build(self, data):
            if Graph.fail: raise RuntimeError("graph write failed")
            graphs[self.version] = len(data["jobs"])
        def statistics(self): return {"Job": graphs[self.version]}
        def close(self): pass
    monkeypatch.setattr(system_module, "GraphSnapshot", Graph)
    monkeypatch.setattr(RecruitmentGraphRAGSystem, "_new_milvus", lambda self, collection: Vector(collection))
    system = RecruitmentGraphRAGSystem(settings)
    initial = system.build()
    version1 = system.snapshot.version
    assert len(initial["changes"]["added"]) == 20
    assert system.build()["unchanged"]
    changed = copy.deepcopy(original)
    changed["jobs"][0]["description"] += " updated"
    removed = changed["jobs"].pop()
    # Graph fake does not validate application targets; real build does.
    path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    Graph.fail = True
    with pytest.raises(RuntimeError): system.build()
    assert system.snapshot.version == system.store.active()["id"] == version1
    Graph.fail = False
    result = system.build()
    assert result["changes"]["updated"] == ["job_001"]
    assert result["changes"]["deleted"] == [removed["nodeId"]]
    assert removed["nodeId"] not in {d.metadata["node_id"] for d in system.snapshot.documents}
    restored = RecruitmentGraphRAGSystem(settings)
    assert restored.restore() and len(restored.snapshot.documents) == 19
    restored.rollback(version1)
    assert len(restored.snapshot.documents) == 20
