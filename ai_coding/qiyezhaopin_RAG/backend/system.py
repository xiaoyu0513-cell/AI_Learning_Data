"""Versioned recruitment RAG coordinator. Importing this file performs no I/O."""
import logging
from dataclasses import dataclass
from threading import Lock
from typing import List
from langchain_core.documents import Document
from config import config as default_config
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.milvus_index import MilvusIndexConstructionModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_snapshot import GraphSnapshot
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter
from rag_modules.index_store import IndexStore, digest
from rag_modules.query import parse_filters
from rag_modules.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalSnapshot:
    version: str
    data: dict
    documents: list
    milvus: object
    graph: object
    hybrid: object
    router: object


class RecruitmentGraphRAGSystem:
    def __init__(self, config=None):
        self.config = config or default_config
        self.seed_path = self.config.seed_path
        self.data_module = GraphDataPreparationModule(self.seed_path, self.config.neo4j_uri,
            self.config.neo4j_user, self.config.neo4j_password, self.config.neo4j_database)
        self._snapshot = None
        self._retired = []
        self._store = None
        self._lock = Lock()
        self._llm_lock = Lock()
        self.llm = self.generation = None
        self.reranker = Reranker(self.config)

    @property
    def store(self):
        if self._store is None:
            self._store = IndexStore(self.config.state_dir)
        return self._store

    @property
    def ready(self):
        return self._snapshot is not None

    @property
    def snapshot(self):
        if self._snapshot is None:
            raise RuntimeError("知识库尚未构建，请先构建索引")
        return self._snapshot

    @property
    def milvus(self):
        return self.snapshot.milvus

    @property
    def hybrid(self):
        return self.snapshot.hybrid

    @property
    def router(self):
        return self._snapshot.router if self.ready else None

    def _new_milvus(self, collection):
        return MilvusIndexConstructionModule(self.config.milvus_host, self.config.milvus_port,
            collection, self.config.milvus_dimension, self.config.embedding_model)

    def _assemble(self, version, data, documents, milvus, graph):
        hybrid = HybridRetrievalModule(self.config, milvus)
        hybrid.initialize(documents)
        retrieval = GraphRAGRetrieval(self.config, snapshot=graph, documents=documents)
        router = IntelligentQueryRouter(hybrid, retrieval, None, self.config)
        return RetrievalSnapshot(version, data, documents, milvus, graph, hybrid, router)

    def restore(self):
        with self._lock:
            row = self.store.active()
            if row is None:
                return False
            documents = [Document(**d) for d in row["documents"]]
            snapshot = self._restore_row(row, documents)
            self._activate(snapshot)
            return True

    def _restore_row(self, row, documents):
        expected = digest([row["payload"], self.config.embedding_model, self.config.milvus_dimension, "schema-v2"])
        if row["fingerprint"] != expected:
            raise RuntimeError("索引与当前 Embedding/schema 配置不一致，请重新构建")
        milvus = self._new_milvus(row["collection"])
        if not milvus.load_collection():
            raise RuntimeError("活动版本的向量集合不存在，请使用 force=true 重新构建")
        graph = GraphSnapshot(self.config, row["id"])
        try:
            if milvus.get_collection_stats()["row_count"] != len(documents):
                raise RuntimeError("活动版本的向量数量不匹配")
            if graph.statistics().get("Job", 0) != len(documents):
                raise RuntimeError("活动版本的图谱数量不匹配")
            return self._assemble(row["id"], row["payload"], documents, milvus, graph)
        except Exception:
            graph.close()
            raise

    def _activate(self, snapshot):
        # Requests hold one snapshot for their entire turn; existing readers survive.
        if self._snapshot:
            self._retired.append(self._snapshot)
        self._snapshot = snapshot

    def build(self, force=False):
        with self._lock, self.store.build_lock():
            data = self.data_module.load_seed_data()
            documents = self.data_module.build_job_documents()
            for doc in documents:
                doc.metadata["content_hash"] = digest([doc.page_content, doc.metadata])
            fingerprint = digest([data, self.config.embedding_model, self.config.milvus_dimension, "schema-v2"])
            previous = self.store.active()
            if previous and previous["fingerprint"] == fingerprint and not force:
                if not self.ready or self.snapshot.version != previous["id"]:
                    self._activate(self._restore_row(previous, documents))
                return self.statistics() | {"unchanged": True}
            before = {d["metadata"]["node_id"]: d["metadata"]["content_hash"] for d in previous["documents"]} if previous else {}
            after = {d.metadata["node_id"]: d.metadata["content_hash"] for d in documents}
            changes = {"added": sorted(after.keys() - before.keys()), "deleted": sorted(before.keys() - after.keys()),
                       "updated": sorted(k for k in before.keys() & after.keys() if before[k] != after[k])}
            version, collection = self.store.prepare(fingerprint, self.config.milvus_collection_name,
                data, [{"page_content": d.page_content, "metadata": d.metadata} for d in documents])
            milvus, graph = self._new_milvus(collection), GraphSnapshot(self.config, version)
            try:
                milvus.build_vector_index(documents, store=self.store)
                graph.build(data)
                if milvus.get_collection_stats()["row_count"] != len(documents) or graph.statistics().get("Job", 0) != len(documents):
                    raise RuntimeError("新版本的图谱/向量数量校验失败")
                snapshot = self._assemble(version, data, documents, milvus, graph)
                stats = {"ready": True, "version": version, "graph": graph.statistics(),
                         "milvus": milvus.get_collection_stats(), "routing": {}}
                self.store.publish(version)
                self._activate(snapshot)
                return stats | {"changes": changes}
            except Exception as error:
                self.store.fail(version, error)
                graph.close()
                raise

    def statistics(self):
        snapshot = self._snapshot
        if snapshot is None:
            return {"ready": False, "milvus": {"row_count": 0}, "graph": {}, "routing": {}}
        return {"ready": True, "version": snapshot.version, "graph": snapshot.graph.statistics(),
                "milvus": snapshot.milvus.get_collection_stats(), "routing": snapshot.router.get_route_statistics()}

    def rollback(self, version):
        with self._lock, self.store.build_lock():
            row = self.store.get_version(version)
            if row is None:
                raise ValueError("没有找到可回滚的已就绪版本")
            documents = [Document(**d) for d in row["documents"]]
            snapshot = self._restore_row(row, documents)
            self.store.publish(version)
            self._activate(snapshot)
            return {"status": "ok", "version": version}

    def _ensure_llm(self):
        with self._llm_lock:
            if self.llm is None:
                from rag_modules.llm import LLMClient
                from rag_modules.generation_integration import GenerationIntegrationModule
                llm = LLMClient(self.config)
                self.generation = GenerationIntegrationModule(llm)
                self.llm = llm
        return self.llm

    def ask_docs(self, question, top_k=None, filters=None, enable_agentic=True, snapshot=None):
        snapshot = snapshot or self.snapshot
        parsed = parse_filters(question, snapshot.data, filters)
        k = self.config.top_k if top_k is None else top_k
        if k <= 0:
            return [], snapshot.router.analyze_query(question, parsed)
        docs, analysis = snapshot.router.route_query(question, max(k, self.config.retrieval_candidates),
            filters=parsed, enable_agentic=enable_agentic)
        return self.reranker.rerank(question, docs, k), analysis

    def ask(self, question):
        docs, analysis = self.ask_docs(question)
        self._ensure_llm()
        return self.generation.generate_answer(question, docs), docs, analysis

    def stream_answer(self, question, documents: List):
        self._ensure_llm()
        yield from self.generation.generate_answer_stream(question, documents)

    def cleanup(self):
        self.data_module.close()
        for snapshot in self._retired + ([self._snapshot] if self._snapshot else []):
            snapshot.graph.close()
        if self._snapshot:
            self._snapshot.milvus.close()
        if self.llm:
            self.llm.client.close()


system = RecruitmentGraphRAGSystem(default_config)
