from types import SimpleNamespace
from contextlib import contextmanager
import pytest
from rag_modules.llm import LLMClient
from services.observability import Observability


@pytest.mark.asyncio
async def test_async_llm_stream_is_closed_on_client_stop(settings):
    class Stream:
        def __init__(self): self.closed = False
        def __aiter__(self): return self
        async def __anext__(self):
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                finish_reason=None, delta=SimpleNamespace(content="token"))])
        async def close(self): self.closed = True
    stream = Stream()
    class Completions:
        async def create(self, **kwargs): return stream
    client = LLMClient.__new__(LLMClient)
    client.config = settings
    client.async_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = client.stream([{"role":"user", "content":"test"}], {})
    assert await result.__anext__() == "token"
    await result.aclose()
    assert stream.closed


def test_langfuse_adapter_updates_final_stage_metadata(settings):
    class Span:
        def update(self, **kwargs): self.metadata = kwargs["metadata"]
    span = Span()
    class Client:
        @contextmanager
        def start_as_current_span(self, **kwargs):
            assert kwargs["trace_context"]["trace_id"] == "a"*32
            yield span
    observer = Observability(settings)
    observer.client = Client()
    with observer.span("generate", "a"*32) as metadata:
        metadata["usage"] = {"prompt_tokens": 10}
    assert span.metadata["usage"]["prompt_tokens"] == 10
    assert "latency_ms" in span.metadata
