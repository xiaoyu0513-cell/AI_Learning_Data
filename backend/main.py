"""
FastAPI 入口：招聘知识图谱 RAG API（SSE 流式对话 + 图谱 + Gap 分析 + 知识库管理）
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import config
from system import system
from services.chat_service import ChatService
from services.graph_service import GraphService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    system.cleanup()
    graph_service.close()


app = FastAPI(title="招聘知识图谱 RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_service = ChatService(system)
graph_service = GraphService(config)


# ========== 数据模型 ==========
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5
    enable_agentic: bool = True


class GapRequest(BaseModel):
    candidate: str
    job: str


class WhatIfRequest(BaseModel):
    candidate: str
    acquired_skills: List[str] = []


class BuildRequest(BaseModel):
    force: bool = False


# ========== 基础 ==========
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ready": system.ready,
        "milvus_rows": system.milvus.get_collection_stats().get("row_count", 0),
    }


@app.get("/api/stats")
async def stats():
    graph_stats = system.data_module.get_statistics() if system.data_module.driver else {}
    return {
        "graph": graph_stats,
        "milvus": system.milvus.get_collection_stats(),
        "routing": {} if system.router is None else system.router.get_route_statistics(),
    }


@app.post("/api/admin/build")
async def build(req: BuildRequest):
    try:
        stats = system.build(force=req.force)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        logger.exception("构建知识库失败")
        return {"status": "error", "message": str(e)}


# ========== 对话（SSE 流式） ==========
@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        async for event in chat_service.stream_chat(
            query=req.query,
            session_id=req.session_id,
            top_k=req.top_k,
            enable_agentic=req.enable_agentic,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ========== 图谱 ==========
@app.get("/api/graph/subgraph")
async def get_subgraph(node_id: str, max_depth: int = 2, max_nodes: int = 100):
    return graph_service.get_subgraph(node_id, max_depth, max_nodes)


@app.get("/api/graph/search")
async def search_nodes(query: str, limit: int = 10):
    return graph_service.search_nodes(query, limit)


@app.get("/api/graph/path")
async def find_path(from_node: str, to_node: str):
    return graph_service.find_path(from_node, to_node)


# ========== 人岗 Gap 分析 ==========
@app.post("/api/gap")
async def gap_analysis(req: GapRequest):
    system._ensure_llm()
    return system.gap.analyze_gap(req.candidate, req.job)


@app.post("/api/gap/whatif")
async def gap_whatif(req: WhatIfRequest):
    system._ensure_llm()
    return system.gap.what_if(req.candidate, req.acquired_skills)