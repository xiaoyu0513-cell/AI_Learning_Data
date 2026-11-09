"""
招聘知识图谱 RAG 系统 —— 配置管理（Pydantic Settings）
参考 ai-saas-template 的分层配置思路，字段保持扁平以复用旧引擎模块接口。
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ===== Neo4j 图数据库 =====
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "recruitment-rag"
    neo4j_database: str = "neo4j"

    # ===== Milvus 向量库 =====
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "recruitment_knowledge"
    milvus_dimension: int = 512  # bge-small-zh-v1.5 向量维度
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ===== LLM（DeepSeek 走 OpenAI 兼容接口） =====
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.1
    max_tokens: int = 2048

    # ===== 检索配置 =====
    top_k: int = 5
    max_graph_depth: int = 2

    # ===== 数据 =====
    seed_path: str = str(BASE_DIR / "data" / "seed" / "jobs.json")

    @property
    def llm_api_key(self) -> str:
        """DeepSeek API Key，从环境变量 DEEPSEEK_API_KEY 读取"""
        return os.getenv("DEEPSEEK_API_KEY", "")


config = Settings()