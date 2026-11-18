import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Use local Neo4j/Milvus and cached BGE model")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="requires --run-integration and running databases"))


@pytest.fixture
def settings(tmp_path):
    from config import Settings
    return Settings(_env_file=None, state_dir=str(tmp_path), request_timeout_seconds=3,
                    rerank_enabled=False, langfuse_enabled=False)


@pytest.fixture
def documents():
    return [Document(page_content="Python FastAPI 深圳 腾讯", metadata={"node_id": "a", "job_name": "Python后端", "location": "深圳", "company": "腾讯", "seniority": "中级", "skills": ["Python", "FastAPI"]}),
            Document(page_content="Go Kubernetes 杭州 阿里巴巴", metadata={"node_id": "b", "job_name": "云平台", "location": "杭州", "company": "阿里巴巴", "seniority": "高级", "skills": ["Go", "Kubernetes"]})]


class FakeVectors:
    def __init__(self, docs=None, fail=False):
        self.docs = docs or []
        self.fail = fail

    def vector_search(self, query, top_k, filters=None):
        if self.fail:
            raise TimeoutError("offline")
        return [d for d in self.docs if filters is None or filters.matches(d)][:top_k]
