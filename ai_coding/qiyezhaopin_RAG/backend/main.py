"""Recruitment Graph RAG API: bounded inputs, async I/O and versioned state."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from config import config
from system import system
from services.chat_service import ChatService
from services.graph_service import GraphService
from rag_modules.query import JobFilters
from rag_modules.gap_analysis import GapAnalysisModule

logger = logging.getLogger(__name__)
chat_service = ChatService(system)
graph_service = GraphService(config, snapshot_provider=lambda: system.snapshot)


@asynccontextmanager
async def lifespan(app):
    try:
        await asyncio.to_thread(system.restore)
    except Exception:
        logger.exception("活动索引恢复失败；服务仍可用，需重新构建")
    yield
    await chat_service.close()
    await asyncio.to_thread(system.cleanup)
    graph_service.close()


app = FastAPI(title="招聘知识图谱 RAG API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9_-]{16,64}$")
    top_k: int = Field(5, ge=1, le=20)
    enable_agentic: bool = True
    filters: Optional[JobFilters] = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class GapRequest(BaseModel):
    candidate: str = Field(min_length=1, max_length=128)
    job: str = Field(min_length=1, max_length=256)


class WhatIfRequest(BaseModel):
    candidate: str = Field(min_length=1, max_length=128)
    acquired_skills: List[str] = Field(default_factory=list, max_length=20)

    @field_validator("acquired_skills")
    @classmethod
    def validate_skills(cls, values):
        if any(not v.strip() or len(v) > 100 for v in values):
            raise ValueError("技能名称不能为空或过长")
        return values


class BuildRequest(BaseModel):
    force: bool = False


class RollbackRequest(BaseModel):
    version: str = Field(pattern=r"^[a-f0-9]{32}$")


def require_ready():
    if not system.ready:
        raise HTTPException(503, "知识库尚未就绪，请先构建索引")


@app.get("/api/health")
async def health():
    # Health must not trigger model loads or database connections.
    return {"status": "ok", "ready": system.ready,
            "milvus_rows": len(system.snapshot.documents) if system.ready else 0,
            "version": system.snapshot.version if system.ready else None}


@app.get("/api/stats")
async def stats():
    try:
        return await asyncio.to_thread(system.statistics)
    except Exception:
        logger.exception("索引状态读取失败")
        raise HTTPException(503, "索引数据库不可用")


@app.get("/api/catalog")
async def catalog():
    require_ready()
    snapshot = system.snapshot
    return {"version": snapshot.version,
            "candidates": [{"id": c["nodeId"], "name": c["name"], "years": c.get("yearsOfExperience", 0),
                            "education": c.get("education", ""), "skills": [s["name"] for s in c.get("skills", [])]}
                           for c in snapshot.data.get("candidates", [])],
            "jobs": [{"id": j["nodeId"], "name": j["name"], "company": j["company"]}
                     for j in snapshot.data.get("jobs", [])]}


@app.post("/api/admin/build")
async def build(req: BuildRequest):
    try:
        return {"status": "ok", "stats": await asyncio.to_thread(system.build, force=req.force)}
    except Exception:
        logger.exception("索引构建失败")
        raise HTTPException(503, "构建失败，上一活动版本继续服务；请检查后端日志")


@app.get("/api/admin/versions")
async def versions():
    return {"active": system.snapshot.version if system.ready else None,
            "versions": await asyncio.to_thread(system.store.versions)}


@app.post("/api/admin/rollback")
async def rollback(req: RollbackRequest):
    try:
        return await asyncio.to_thread(system.rollback, req.version)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception:
        logger.exception("版本回滚失败")
        raise HTTPException(503, "版本回滚失败，当前活动版本未切换")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    require_ready()
    async def event_stream():
        async for event in chat_service.stream_chat(req.query, req.session_id, req.top_k,
                                                    req.enable_agentic, req.filters):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/graph/subgraph")
async def get_subgraph(node_id: str = Query(min_length=1, max_length=256),
                       max_depth: int = Query(2, ge=1, le=3), max_nodes: int = Query(100, ge=1, le=150)):
    require_ready()
    return await asyncio.to_thread(graph_service.get_subgraph, node_id, max_depth, max_nodes)


@app.get("/api/graph/search")
async def search_nodes(query: str = Query(min_length=1, max_length=128), limit: int = Query(10, ge=1, le=50)):
    require_ready()
    return await asyncio.to_thread(graph_service.search_nodes, query, limit)


@app.get("/api/graph/path")
async def find_path(from_node: str = Query(min_length=1, max_length=256), to_node: str = Query(min_length=1, max_length=256)):
    require_ready()
    return await asyncio.to_thread(graph_service.find_path, from_node, to_node)


@app.post("/api/gap")
async def gap_analysis(req: GapRequest):
    require_ready()
    tool = GapAnalysisModule(config, data=system.snapshot.data)
    result = await asyncio.to_thread(tool.analyze_gap, req.candidate, req.job)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.post("/api/gap/whatif")
async def gap_whatif(req: WhatIfRequest):
    require_ready()
    result = await asyncio.to_thread(GapAnalysisModule(config, data=system.snapshot.data).what_if,
                                     req.candidate, req.acquired_skills)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/api/evaluation/latest")
async def latest_evaluation():
    from pathlib import Path
    path = Path(config.state_dir) / "evaluation.json"
    if not path.exists():
        return {"status": "not_run", "message": "尚未运行评测，请执行 backend/evaluation/run.py"}
    return json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
