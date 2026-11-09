"""
招聘知识图谱 RAG 系统主控
编排：数据建图 → 向量索引 → 双引擎检索 → 智能路由 → 生成
"""
import logging
from typing import List, Tuple, Optional

from config import config as default_config
from rag_modules.llm import LLMClient
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.milvus_index import MilvusIndexConstructionModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.gap_analysis import GapAnalysisModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecruitmentGraphRAGSystem:
    """招聘知识图谱智能问答系统"""

    def __init__(self, config=None):
        self.config = config or default_config
        self.seed_path = config.seed_path if hasattr(config, "seed_path") else "data/seed/jobs.json"

        # 基础模块（建库不需要 LLM）
        self.data_module = GraphDataPreparationModule(
            seed_path=self.seed_path,
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
        )
        self.milvus = MilvusIndexConstructionModule(
            host=self.config.milvus_host,
            port=self.config.milvus_port,
            collection_name=self.config.milvus_collection_name,
            dimension=self.config.milvus_dimension,
            embedding_model=self.config.embedding_model,
        )
        self.hybrid = HybridRetrievalModule(self.config, self.milvus)

        # 需要 LLM 的模块（惰性初始化）
        self.llm: Optional[LLMClient] = None
        self.graph: Optional[GraphRAGRetrieval] = None
        self.router: Optional[IntelligentQueryRouter] = None
        self.generation: Optional[GenerationIntegrationModule] = None
        self.gap: Optional[GapAnalysisModule] = None

        self.ready = False

    # ========== LLM 惰性初始化 ==========
    def _ensure_llm(self):
        if self.llm is None:
            self.llm = LLMClient(self.config)
            self.graph = GraphRAGRetrieval(self.config, self.llm)
            self.router = IntelligentQueryRouter(self.hybrid, self.graph, self.llm, self.config)
            self.generation = GenerationIntegrationModule(self.llm)
            self.gap = GapAnalysisModule(self.config, self.llm)
        return self.llm

    # ========== 知识库构建 ==========
    def build(self, force: bool = False):
        """建图 + 建向量索引 + 初始化检索器"""
        logger.info("开始构建知识库...")

        # 1. 图构建（写入 Neo4j）
        self.data_module.build_graph()

        # 2. 生成岗位档案文档
        documents = self.data_module.build_job_documents()

        # 3. 向量索引（幂等：已有则复用；force 则重建）
        if force and self.milvus.has_collection():
            self.milvus.delete_collection()
        if not self.milvus.has_collection():
            self.milvus.build_vector_index(documents)
        else:
            self.milvus.load_collection()

        # 4. 初始化检索器
        self.hybrid.initialize(documents)

        self.ready = True
        stats = self.data_module.get_statistics()
        stats["milvus"] = self.milvus.get_collection_stats()
        logger.info("知识库构建完成: %s", stats)
        return stats

    # ========== 问答 ==========
    def ask(self, question: str) -> Tuple[str, List, QueryAnalysis]:
        """统一问答入口（非流式）"""
        if not self.ready:
            raise RuntimeError("系统未初始化，请先调用 build()")
        self._ensure_llm()
        self.graph.initialize()

        docs, analysis = self.router.route_query(question)
        answer = self.generation.generate_answer(question, docs)
        return answer, docs, analysis

    def ask_docs(self, question: str, top_k: int = None) -> Tuple[List, QueryAnalysis]:
        """只做路由检索，返回文档与分析（供 SSE 服务拆分阶段输出）"""
        if not self.ready:
            raise RuntimeError("系统未初始化，请先调用 build()")
        self._ensure_llm()
        self.graph.initialize()
        return self.router.route_query(question, top_k)

    def stream_answer(self, question: str, documents: List):
        """基于已检索文档做流式生成"""
        self._ensure_llm()
        yield from self.generation.generate_answer_stream(question, documents)

    def cleanup(self):
        self.data_module.close()
        if self.graph:
            self.graph.close()
        if self.gap:
            self.gap.close()
        self.milvus.close()


# 模块级单例，供 FastAPI 复用
system = RecruitmentGraphRAGSystem(default_config)