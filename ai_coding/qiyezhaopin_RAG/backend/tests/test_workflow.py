import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from services.chat_service import ChatService
from rag_modules.reranker import Reranker
from rag_modules.intelligent_query_router import QueryAnalysis, SearchStrategy


class Router:
    def __init__(self, docs):
        self.docs, self.filters = docs, []

    def route_query(self, query, top_k, filters, enable_agentic):
        self.filters.append(filters)
        return [d for d in self.docs if filters.matches(d)], QueryAnalysis(
            .3, .2, False, 0, SearchStrategy.HYBRID_TRADITIONAL, .8, "test", "hybrid_traditional")


class Generator:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.block = False
        self.fail = False

    async def generate_async(self, question, docs, history, usage):
        self.calls.append((question, docs, history))
        self.started.set()
        try:
            if self.block:
                yield "partial"
                await asyncio.Event().wait()
            if self.fail:
                raise RuntimeError("upstream_error")
            usage.update(prompt_tokens=10, completion_tokens=5, finish_reason="stop")
            yield "证据结论 [1]"
        finally:
            self.closed.set()


def fake_system(settings, docs):
    router = Router(docs)
    snapshot = SimpleNamespace(version="v1", data={"jobs": [], "skills": [], "candidates": []}, router=router)
    generator = Generator()
    return SimpleNamespace(config=settings, ready=True, snapshot=snapshot, reranker=Reranker(settings),
                           generation=generator, _ensure_llm=lambda: None), router, generator


async def collect(service, query, session):
    return [event async for event in service.stream_chat(query, session)]


async def saved(settings, session):
    async with AsyncSqliteSaver.from_conn_string(str(Path(settings.state_dir)/"sessions.sqlite3")) as saver:
        checkpoint = await saver.aget_tuple({"configurable": {"thread_id":session}})
        return checkpoint.checkpoint["channel_values"]


@pytest.mark.asyncio
async def test_sqlite_memory_survives_new_service_and_ordinal(settings, documents):
    system, router, generator = fake_system(settings, documents)
    session = uuid4().hex
    events = await collect(ChatService(system), "推荐岗位", session)
    assert events[-1]["type"] == "done"
    assert [c["node_id"] for e in events if e["type"] == "citations" for c in e["content"]] == ["a", "b"]
    await collect(ChatService(system), "第二个岗位需要什么技能", session)
    assert router.filters[-1].job_ids == ["b"]
    assert router.filters[-1].restrict_ids
    assert generator.calls[-1][1][0].metadata["node_id"] == "b"
    assert len(generator.calls[-1][2]) == 2
    state = await saved(settings, session)
    assert state["previous_ids"] == ["b"]
    assert len(state["history"]) == 4


@pytest.mark.asyncio
async def test_new_session_does_not_inherit_other_session(settings, documents):
    system, router, generator = fake_system(settings, documents)
    service = ChatService(system)
    await collect(service, "推荐岗位", uuid4().hex)
    events = await collect(service, "第二个岗位", uuid4().hex)
    assert router.filters[-1].restrict_ids and router.filters[-1].job_ids == []
    assert not next(e["content"] for e in events if e["type"] == "citations")
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_index_change_invalidates_ordinal(settings, documents):
    system, router, generator = fake_system(settings, documents)
    service, session = ChatService(system), uuid4().hex
    await collect(service, "推荐岗位", session)
    system.snapshot.version = "v2"
    await collect(service, "第二个岗位", session)
    assert router.filters[-1].job_ids == []
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_failed_turn_keeps_completed_history(settings, documents):
    system, router, generator = fake_system(settings, documents)
    service, session = ChatService(system), uuid4().hex
    await collect(service, "推荐岗位", session)
    generator.fail = True
    events = await collect(service, "第二个岗位", session)
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "done" for e in events)
    state = await saved(settings, session)
    assert len(state["history"]) == 2 and state["previous_ids"] == ["a", "b"]


@pytest.mark.asyncio
async def test_deadline_closes_generation_and_releases_lock(settings, documents):
    system, router, generator = fake_system(settings, documents)
    settings.request_timeout_seconds = .3
    generator.block = True
    service, session = ChatService(system), uuid4().hex
    events = await collect(service, "推荐岗位", session)
    assert any(e["type"] == "error" and "超时" in e["content"] for e in events)
    assert not any(e["type"] == "done" for e in events)
    assert generator.closed.is_set()
    assert service._sessions == {}
    state = await saved(settings, session)
    assert not state.get("history")


@pytest.mark.asyncio
async def test_client_disconnect_cancels_upstream(settings, documents):
    system, router, generator = fake_system(settings, documents)
    generator.block = True
    service = ChatService(system)
    stream = service.stream_chat("推荐岗位", uuid4().hex)
    async for event in stream:
        if event["type"] == "token":
            break
    await stream.aclose()
    assert generator.closed.is_set()
    assert not service._sessions


@pytest.mark.asyncio
async def test_traces_contain_metadata_not_query_or_answer(settings, documents):
    system, _, _ = fake_system(settings, documents)
    await collect(ChatService(system), "private-content-123", uuid4().hex)
    content = (Path(settings.state_dir)/"traces.jsonl").read_text(encoding="utf8")
    assert "private-content-123" not in content
    assert "证据结论" not in content
    spans = [json.loads(line) for line in content.splitlines()]
    assert {s["stage"] for s in spans} >= {"parse", "retrieve", "rerank", "generate"}
    assert next(s for s in spans if s["stage"] == "generate")["usage"]["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_same_session_requests_serialize_completed_history(settings, documents):
    system, _, generator = fake_system(settings, documents)
    service, session = ChatService(system), uuid4().hex
    results = await asyncio.gather(collect(service, "推荐岗位", session), collect(service, "第二个岗位", session))
    assert all(events[-1]["type"] == "done" for events in results)
    assert len(generator.calls[1][2]) == 2
    assert generator.calls[1][1][0].metadata["node_id"] == "b"
    assert not service._sessions
