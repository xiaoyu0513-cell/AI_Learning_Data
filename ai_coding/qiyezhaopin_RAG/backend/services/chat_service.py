"""Nonblocking SSE facade with request deadlines and durable LangGraph state."""
import asyncio
import logging
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from services.workflow import make_workflow
from services.observability import Observability

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, system):
        self.system = system
        self._observer = None
        self._sessions = {}
        self._capacity = asyncio.Semaphore(4)

    @property
    def observer(self):
        if self._observer is None:
            self._observer = Observability(self.system.config)
        return self._observer

    @asynccontextmanager
    async def _session_lock(self, session):
        # Entry count includes waiters so a completed request cannot drop their lock.
        entry = self._sessions.setdefault(session, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._sessions.pop(session, None)

    async def stream_chat(self, query, session_id=None, top_k=5, enable_agentic=True, filters=None):
        session_id = session_id or uuid4().hex
        request_id = uuid4().hex
        yield {"type": "session", "content": {"session_id": session_id, "request_id": request_id}}
        if not self.system.ready:
            yield {"type": "error", "content": "知识库尚未构建，请先构建知识库"}
            return
        snapshot = self.system.snapshot
        yield {"type": "status", "content": "正在检索同一版本的岗位证据..."}
        path = Path(self.system.config.state_dir) / "sessions.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Python 3.10 compatible deadline, including queueing and generation.
        queue = asyncio.Queue(maxsize=32)

        async def produce():
            try:
                async with self._capacity, self._session_lock(session_id):
                    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
                        workflow = make_workflow(self.system, snapshot, saver, self.observer, queue.put)
                        state = {"query": query, "top_k": top_k, "request_id": request_id,
                                 "explicit_filters": filters.model_dump(exclude_unset=True) if filters else {},
                                 "enable_agentic": enable_agentic}
                        with self.observer.span("chat", request_id, version=snapshot.version):
                            await workflow.ainvoke(state, config={"configurable": {"thread_id": session_id}})
                await queue.put({"type": "done"})
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("对话处理失败 request_id=%s", request_id)
                await queue.put({"type": "error", "content": f"处理失败，请检查服务日志。请求编号：{request_id}"})
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.system.config.request_timeout_seconds
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                event = await asyncio.wait_for(queue.get(), remaining)
                if event is None:
                    break
                yield event
        except asyncio.TimeoutError:
            yield {"type": "error", "content": f"请求超时，已停止生成。请求编号：{request_id}"}
        finally:
            task.cancel()
            # Cancelled producers must not deadlock on a full queue in their finally.
            while not queue.empty():
                queue.get_nowait()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def close(self):
        if self.system.llm:
            await self.system.llm.async_client.close()
        if self._observer:
            await asyncio.to_thread(self._observer.close)
