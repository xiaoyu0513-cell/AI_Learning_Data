"""
生成集成模块
把检索到的文档/路径/子图作为上下文，交给 DeepSeek 生成最终答案（含引用溯源）。
"""
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _build_context(self, documents: List[Document]) -> str:
        parts = []
        for i, d in enumerate(documents, 1):
            source = d.metadata.get("search_type", d.metadata.get("search_source", "unknown"))
            name = d.metadata.get("job_name") or d.metadata.get("central_name") or ""
            header = f"[{i}] (来源:{source}"
            if name:
                header += f" | {name}"
            header += ")"
            parts.append(f"{header}\n{d.page_content}")
        return "\n\n".join(parts)

    def generate_answer(self, question: str, documents: List[Document]) -> str:
        if not documents:
            return "没有找到相关信息，请换个问法。"

        context = self._build_context(documents)
        system = (
            "你是招聘知识图谱智能助手。只能基于提供的检索上下文回答，"
            "不要编造上下文外的信息；涉及岗位-技能-公司多跳关系时，先说明推理链条。"
            "回答需简洁、结构化，并在关键结论后用 [n] 标注引用的证据来源。"
        )
        prompt = f"用户问题：{question}\n\n检索上下文：\n{context}\n\n请回答："
        return self.llm_client.chat(prompt, system=system)

    def generate_answer_stream(self, question: str, documents: List[Document]):
        if not documents:
            yield "没有找到相关信息，请换个问法。"
            return
        context = self._build_context(documents)
        system = (
            "你是招聘知识图谱智能助手。只能基于提供的检索上下文回答，"
            "关键结论后用 [n] 标注引用来源。"
        )
        prompt = f"用户问题：{question}\n\n检索上下文：\n{context}\n\n请回答："
        stream = self.llm_client.client.chat.completions.create(
            model=self.llm_client.config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.llm_client.config.temperature,
            max_tokens=self.llm_client.config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content