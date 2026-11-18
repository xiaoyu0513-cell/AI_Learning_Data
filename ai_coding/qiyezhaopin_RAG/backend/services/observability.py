"""Metadata-only stage spans. Content export is deliberately opt-in absent."""
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class Observability:
    def __init__(self, config):
        self.config = config
        self.lock = Lock()
        self.client = None
        if config.langfuse_enabled:
            if not config.langfuse_public_key or not config.langfuse_secret_key:
                raise ValueError("开启 Langfuse 必须配置 public/secret key")
            from langfuse import Langfuse
            self.client = Langfuse(public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key, host=config.langfuse_host)

    @contextmanager
    def span(self, name, request_id, **metadata):
        from contextlib import nullcontext
        started = time.perf_counter()
        record = {"request_id": request_id, "stage": name, **metadata}
        observation = self.client.start_as_current_span(name=name, metadata=record,
            trace_context={"trace_id": request_id}) if self.client else nullcontext()
        remote = None
        try:
            with observation as remote:
                try:
                    yield record
                except BaseException as exc:
                    record["error_type"] = type(exc).__name__
                    raise
                finally:
                    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    if remote is not None:
                        try:
                            remote.update(metadata=dict(record))
                        except Exception:
                            logger.exception("Langfuse span 更新失败")
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            path = Path(self.config.state_dir) / "traces.jsonl"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with self.lock, path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                logger.exception("本地 trace 写入失败")

    def close(self):
        if self.client:
            self.client.flush()
            self.client.shutdown()
