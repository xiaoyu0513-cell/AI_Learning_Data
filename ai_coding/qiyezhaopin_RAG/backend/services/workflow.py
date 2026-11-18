"""Typed LangGraph workflow with SQLite checkpoints and completed-turn memory."""
import asyncio
import re
from dataclasses import asdict
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from rag_modules.query import JobFilters, parse_filters


class ChatState(TypedDict, total=False):
    query: str
    top_k: int
    explicit_filters: dict
    enable_agentic: bool
    request_id: str
    resolved_query: str
    filters: dict
    documents: list
    analysis: dict
    answer: str
    history: list
    previous_ids: list
    previous_query: str
    previous_version: str
    usage: dict
    gap: dict


def make_workflow(system, snapshot, saver, observer, emit):
    # Explicit async event sink also works on Python 3.10, whose task context
    # propagation differs from 3.11's LangGraph custom stream implementation.
    async def event(kind, content=None):
        payload = {"type": kind}
        if content is not None:
            payload["content"] = content
        await emit(payload)

    async def parse(state):
        with observer.span("parse", state["request_id"], version=snapshot.version):
            query = state["query"]
            filters = parse_filters(query, snapshot.data, JobFilters(**state["explicit_filters"]))
            same_version = state.get("previous_version") == snapshot.version
            followup = any(x in query for x in ("这些", "它们", "其中", "这个岗位", "那个岗位", "第一个", "第二个", "第三个"))
            resolved = query
            if followup and same_version and state.get("previous_ids"):
                ids = state["previous_ids"]
                for ordinal, index in (("第一个", 0), ("第二个", 1), ("第三个", 2)):
                    if ordinal in query:
                        ids = ids[index:index+1]
                        break
                filters = filters.model_copy(update={"restrict_ids": True, "job_ids": ids})
                resolved = state.get("previous_query", "")[:500] + "；追问：" + query
            elif followup:
                # Never silently interpret a stale ordinal against a new index.
                filters = filters.model_copy(update={"restrict_ids": True, "job_ids": []})
                await event("status", "没有可引用的同版本岗位，请先重新检索。")
            return {"resolved_query": resolved, "filters": filters.model_dump(), "answer": "", "usage": {}, "gap": {}}

    async def retrieve(state):
        with observer.span("retrieve", state["request_id"], version=snapshot.version) as trace:
            filters = JobFilters(**state["filters"])
            docs, analysis = await asyncio.wait_for(asyncio.to_thread(snapshot.router.route_query,
                state["resolved_query"], max(state["top_k"], system.config.retrieval_candidates),
                filters=filters, enable_agentic=state["enable_agentic"]), system.config.retrieval_timeout_seconds)
            trace.update(count=len(docs), route=analysis.effective_strategy, fallback_reason=analysis.fallback_reason)
            encoded = asdict(analysis)
            encoded["recommended_strategy"] = analysis.recommended_strategy.value
            await event("route_decision", {"strategy": analysis.effective_strategy,
                "confidence": analysis.confidence, "complexity": analysis.query_complexity,
                "relationship": analysis.relationship_intensity, "reasoning_required": analysis.reasoning_required,
                "entity_count": analysis.entity_count, "reasoning": analysis.reasoning,
                "fallback_reason": analysis.fallback_reason})
            return {"documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs], "analysis": encoded}

    async def rerank(state):
        with observer.span("rerank", state["request_id"]) as trace:
            docs = await asyncio.wait_for(asyncio.to_thread(system.reranker.rerank,
                state["resolved_query"], [Document(**d) for d in state["documents"]], state["top_k"]),
                system.config.retrieval_timeout_seconds)
            # Truncate the evidence list by whole document before assigning citations.
            budget, selected = system.config.context_max_chars, []
            for doc in docs:
                needed = len(doc.page_content) + 200
                if needed > budget:
                    break
                selected.append(doc)
                budget -= needed
            trace.update(count=len(selected), status=docs[0].metadata.get("rerank_status") if docs else "empty")
            return {"documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in selected]}

    async def gap(state):
        # Matching is a rule tool, not an LLM hiring probability.
        from rag_modules.gap_analysis import GapAnalysisModule
        candidates = [c for c in snapshot.data.get("candidates", []) if c["name"] in state["query"]]
        if len(candidates) == 1 and state["documents"] and any(k in state["query"] for k in ("差距", "缺少", "匹配")):
            result = GapAnalysisModule(system.config, data=snapshot.data).analyze_gap(
                candidates[0]["name"], state["documents"][0]["metadata"]["node_id"])
            return {"gap": result}
        return {"gap": {}}

    async def generate(state):
        with observer.span("generate", state["request_id"], version=snapshot.version) as trace:
            docs = [Document(**d) for d in state["documents"]]
            if docs and state.get("gap"):
                import json
                docs[0].page_content += "\n技能覆盖率工具结果（不是录用概率）：" + json.dumps(state["gap"], ensure_ascii=False)
            budget, selected = system.config.context_max_chars, []
            for doc in docs:
                size = len(doc.page_content) + 200
                if size > budget:
                    break
                selected.append(doc)
                budget -= size
            docs = selected
            await event("citations", [{"index": i+1, "source": d.metadata.get("search_type", ""),
                "node_id": d.metadata["node_id"], "job_name": d.metadata["job_name"],
                "strategy": d.metadata.get("route_strategy", ""),
                "score": d.metadata.get("rerank_score", d.metadata.get("final_score", 0)),
                "rerank_status": d.metadata.get("rerank_status"), "version": snapshot.version} for i, d in enumerate(docs)])
            answer, usage = "", {}
            if not docs:
                answer = "当前知识库没有找到满足条件的岗位，无法据此给出推荐。"
                await event("token", answer)
            else:
                system._ensure_llm()
                question = state["resolved_query"]
                if state.get("gap"):
                    await event("status", "已计算技能差距；覆盖率不代表录用概率。")
                async for token in system.generation.generate_async(question, docs, state.get("history", []), usage):
                    answer += token
                    await event("token", token)
                # Warn before commit rather than call a truncated completion successful.
                if usage.get("finish_reason") == "length":
                    raise RuntimeError("模型输出达到长度上限，请缩短问题后重试")
                if not answer.strip():
                    raise RuntimeError("模型没有返回有效回答")
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
            invalid = sorted(n for n in cited if n < 1 or n > len(docs))
            if invalid:
                await event("status", "回答含无效引用编号，请核对证据。")
            trace.update(usage=usage, invalid_citations=len(invalid), answer_chars=len(answer))
            history = (state.get("history", []) + [{"role": "user", "content": state["query"][:2000]},
                       {"role": "assistant", "content": answer[:4000]}])[-6:]
            return {"answer": answer, "usage": usage, "history": history,
                    "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                    "previous_ids": [d.metadata["node_id"] for d in docs],
                    "previous_query": state["resolved_query"][:1000], "previous_version": snapshot.version}

    builder = StateGraph(ChatState)
    for name, node in (("parse", parse), ("retrieve", retrieve), ("rerank", rerank), ("gap", gap), ("generate", generate)):
        builder.add_node(name, node)
    builder.add_edge(START, "parse")
    for a, b in (("parse", "retrieve"), ("retrieve", "rerank"), ("rerank", "gap"), ("gap", "generate")):
        builder.add_edge(a, b)
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=saver)
