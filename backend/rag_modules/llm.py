"""
LLM 客户端封装（DeepSeek，走 OpenAI 兼容接口）
也供查询路由、图检索意图理解、最终生成共用
"""
import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """DeepSeek 对话客户端"""

    def __init__(self, config):
        self.config = config
        if not config.llm_api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置")
        self.client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    def chat(self, prompt: str, system: str = None, temperature: float = None,
             max_tokens: int = None) -> str:
        """普通对话，返回文本"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content

    def chat_json(self, prompt: str, system: str = None) -> dict:
        """请求并解析 JSON 输出（用于结构化分析）"""
        text = self.chat(prompt, system=system, temperature=0.1, max_tokens=1000)
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 输出中稳健抽取 JSON 对象"""
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("JSON 解析失败: %s | 原文: %s", e, text[:200])
            return {}