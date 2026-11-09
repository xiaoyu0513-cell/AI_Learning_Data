"""
对话服务：路由 → 检索 → 生成（SSE 流式）
事件类型：status / route_decision / citations / token / done
"""
import json
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, system):
        self.system = system

    async def stream_chat(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 5,
        enable_agentic: bool = True,
    ) -> AsyncGenerator[dict, None]:
        try:
            yield {"type": "status", "content": "正在分析查询..."}

            # 路由检索
            if not self.system.ready:
                yield {"type": "error", "content": "知识库尚未构建，请先调用 /api/admin/build"}
                return

            documents, analysis = self.system.ask_docs(query, top_k=top_k)
            _ = (session_id, enable_agentic)  # 预留：会话管理 / Agentic 开关

            yield {
                "type": "route_decision",
                "content": {
                    "strategy": analysis.recommended_strategy.value,
                    "confidence": analysis.confidence,
                    "complexity": analysis.query_complexity,
                    "relationship": analysis.relationship_intensity,
                    "reasoning_required": analysis.reasoning_required,
                    "entity_count": analysis.entity_count,
                    "reasoning": analysis.reasoning,
                },
            }

            yield {
                "type": "citations",
                "content": [
                    {
                        "index": i + 1,
                        "source": d.metadata.get("search_type", ""),
                        "node_id": d.metadata.get("node_id", ""),
                        "job_name": d.metadata.get("job_name", ""),
                        "strategy": d.metadata.get("route_strategy", ""),
                        "score": d.metadata.get("score", d.metadata.get("relevance_score", 0)),
                    }
                    for i, d in enumerate(documents)
                ],
            }

            # 流式生成
            async for token in self._sync_stream_to_async(query, documents):
                yield {"type": "token", "content": token}

            yield {"type": "done"}
        except Exception as e:
            logger.exception("对话处理失败")
            yield {"type": "error", "content": str(e)}

    async def _sync_stream_to_async(self, query: str, documents):
        """把同步生成器桥接成异步生成器"""
        for token in self.system.stream_answer(query, documents):
            yield token