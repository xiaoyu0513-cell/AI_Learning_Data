from types import SimpleNamespace
import pytest
from langchain_core.documents import Document
from rag_modules.query import JobFilters, parse_filters, mentions
from rag_modules.hybrid_retrieval import HybridRetrievalModule, _tokenize
from rag_modules.reranker import Reranker
from rag_modules.intelligent_query_router import IntelligentQueryRouter
from conftest import FakeVectors


@pytest.mark.parametrize("term,text", [("Go", "Django"), ("Java", "JavaScript"), ("C", "C++")])
def test_skill_boundaries(term, text):
    assert not mentions(text, term)


def test_tokenizer_preserves_languages_and_chinese_boundaries():
    tokens = _tokenize("C++ C# Go 上海，杭州 Python 北京")
    assert "c++" in tokens and "c#" in tokens
    assert "海杭" not in tokens and "州北" not in tokens


def test_skill_intersection_and_alias(documents):
    f = JobFilters(skills=["golang", "k8s"], location="杭州")
    assert not f.matches(documents[0]) and f.matches(documents[1])


def test_parse_unknown_city_and_explicit_override():
    data = {"skills": [{"name": "Python"}], "jobs": [{"location": "北京", "company": "腾讯", "seniority": "中级"}]}
    assert parse_filters("上海 Python 岗位", data).location == "上海"
    assert parse_filters("北京 Python", data, JobFilters(location="深圳")).location == "深圳"
    assert parse_filters("应届 Python 岗位", data).seniority == "应届生"


def test_empty_bm25_and_zero_k(settings):
    module = HybridRetrievalModule(settings, FakeVectors())
    module.initialize([])
    assert module.bm25_search("q", 5) == []
    assert module.hybrid_search("q", 0) == []


def test_rrf_deduplicates_without_mutating_corpus(settings, documents):
    module = HybridRetrievalModule(settings, FakeVectors(documents))
    module.initialize(documents)
    result = module.hybrid_search("Python FastAPI", 5)
    assert len({d.metadata["node_id"] for d in result}) == len(result)
    assert "final_score" not in documents[0].metadata


def test_no_zero_score_keyword_matches(settings, documents):
    module = HybridRetrievalModule(settings, FakeVectors())
    module.initialize(documents)
    assert module.bm25_search("nonexistentword", 5) == []


def test_vector_failure_keeps_keyword_retrieval(settings):
    docs = [Document(page_content=f"Python FastAPI {i}", metadata={"node_id": str(i), "skills": []}) for i in range(3)]
    docs += [Document(page_content="golang redis", metadata={"node_id": "x", "skills": []}) for _ in range(4)]
    module = HybridRetrievalModule(settings, FakeVectors(fail=True))
    module.initialize(docs)
    result = module.hybrid_search("FastAPI", 3)
    assert result and result[0].metadata["fallback_reason"] == "vector_unavailable"


def test_filters_apply_to_both_branches(settings, documents):
    module = HybridRetrievalModule(settings, FakeVectors(documents))
    module.initialize(documents)
    assert module.hybrid_search("Python", 5, filters=JobFilters(location="上海")) == []


def test_reranker_preserves_raw_scores(settings, documents):
    class Scorer:
        def predict(self, pairs, batch_size):
            return [-2, 8]
    settings.rerank_enabled = True
    result = Reranker(settings, Scorer()).rerank("q", documents, 1)
    assert result[0].metadata["node_id"] == "b"
    assert result[0].metadata["rerank_score"] == 8
    assert result[0].metadata["rerank_status"] == "applied"
    assert "rerank_score" not in documents[1].metadata


def test_reranker_failure_is_visible(settings, documents):
    class Broken:
        def predict(self, *args, **kwargs):
            raise RuntimeError("failed")
    settings.rerank_enabled = True
    result = Reranker(settings, Broken()).rerank("q", documents, 2)
    assert result[0].metadata["node_id"] == "a"
    assert result[0].metadata["rerank_status"] == "unavailable"


def test_graph_failure_routes_to_hybrid(settings, documents):
    class Graph:
        def graph_rag_search(self, *args, **kwargs):
            raise TimeoutError
    module = HybridRetrievalModule(settings, FakeVectors(documents))
    module.initialize(documents)
    router = IntelligentQueryRouter(module, Graph(), None, settings)
    result, analysis = router.route_query("同时匹配 Go Kubernetes", 1, filters=JobFilters())
    assert len(result) == 1
    assert analysis.fallback_reason == "graph_unavailable"
    assert analysis.effective_strategy == "hybrid_traditional"
    assert router.get_route_statistics()["graph_fallbacks"] == 1


def test_enable_agentic_switch_skips_graph(settings, documents):
    class Graph:
        def graph_rag_search(self, *args, **kwargs):
            pytest.fail("graph must not be called")
    module = HybridRetrievalModule(settings, FakeVectors(documents))
    module.initialize(documents)
    result, analysis = IntelligentQueryRouter(module, Graph(), None, settings).route_query("同时匹配", 1, enable_agentic=False)
    assert analysis.effective_strategy == "hybrid_traditional"


def test_empty_id_filter_skips_embedding_and_milvus(settings):
    from rag_modules.milvus_index import MilvusIndexConstructionModule
    module = MilvusIndexConstructionModule("localhost", 19530, "unused", 512, "unused-model")
    assert module.vector_search("q", filters=JobFilters(restrict_ids=True, job_ids=[])) == []
    assert module._embedder is None and not module._connected


def test_empty_graph_constraints_skip_database(settings, documents):
    from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
    class Snapshot:
        def connect(self):
            pytest.fail("unsatisfiable constraints must not connect")
    module = GraphRAGRetrieval(settings, snapshot=Snapshot(), documents=documents)
    assert module.graph_rag_search("q", filters=JobFilters(location="上海")) == []
