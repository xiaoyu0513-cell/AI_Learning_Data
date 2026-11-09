"""
Milvus 向量索引构建模块（招聘领域）
把岗位档案文档向量化后写入 Milvus，并提供向量检索接口。
字段语义：company_name → job_name + node_id + category + location + seniority
"""
import logging
from typing import List, Optional, Dict, Any

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from langchain_core.documents import Document

from .embedding import Embedder

logger = logging.getLogger(__name__)


class MilvusIndexConstructionModule:
    """Milvus 向量库封装"""

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        dimension: int,
        embedding_model: str,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.embedder = Embedder(embedding_model)
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info("Milvus 连接成功: %s:%s", self.host, self.port)
        except Exception as e:
            logger.error("Milvus 连接失败: %s", e)

    # ========== 集合管理 ==========
    def has_collection(self) -> bool:
        try:
            return utility.has_collection(self.collection_name)
        except Exception:
            return False

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="job_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="salary_range", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="seniority", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields, description="招聘知识库")
        self.collection = Collection(self.collection_name, schema)

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        self.collection.create_index("embedding", index_params)
        logger.info("Milvus 集合创建完成: %s (dim=%d)", self.collection_name, self.dimension)

    def load_collection(self) -> bool:
        if not self.has_collection():
            return False
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info("Milvus 集合加载完成: %s", self.collection_name)
        return True

    # ========== 写入向量 ==========
    def build_vector_index(self, documents: List[Document]) -> bool:
        if not documents:
            logger.warning("无可写入的文档")
            return False
        if not self.has_collection():
            self._create_collection()
        else:
            self.collection = Collection(self.collection_name)

        texts = [d.page_content for d in documents]
        embeddings = self.embedder.encode_documents(texts)

        rows = [
            {
                "text": d.page_content,
                "job_name": d.metadata.get("job_name", ""),
                "node_id": d.metadata.get("node_id", ""),
                "company": d.metadata.get("company", ""),
                "category": d.metadata.get("category", ""),
                "location": d.metadata.get("location", ""),
                "salary_range": d.metadata.get("salary_range", ""),
                "seniority": d.metadata.get("seniority", ""),
                "embedding": emb,
            }
            for d, emb in zip(documents, embeddings)
        ]

        self.collection.insert(rows)
        self.collection.flush()
        self.collection.load()
        logger.info("写入向量 %d 条", len(rows))
        return True

    # ========== 向量检索 ==========
    def vector_search(self, query: str, top_k: int = 5) -> List[Document]:
        if self.collection is None:
            if not self.load_collection():
                return []

        query_vec = self.embedder.encode_query(query)
        self.collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        output_fields = ["text", "job_name", "node_id", "company", "category", "location", "salary_range", "seniority"]
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )

        documents = []
        for hit in results[0]:
            documents.append(
                Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "job_name": hit.entity.get("job_name", ""),
                        "node_id": hit.entity.get("node_id", ""),
                        "company": hit.entity.get("company", ""),
                        "category": hit.entity.get("category", ""),
                        "location": hit.entity.get("location", ""),
                        "salary_range": hit.entity.get("salary_range", ""),
                        "seniority": hit.entity.get("seniority", ""),
                        "score": float(hit.score),
                        "search_type": "vector",
                    },
                )
            )
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        if self.collection is None:
            if not self.load_collection():
                return {"row_count": 0}
        try:
            return {"row_count": self.collection.num_entities}
        except Exception:
            return {"row_count": 0}

    def delete_collection(self) -> bool:
        if self.has_collection():
            utility.drop_collection(self.collection_name)
            self.collection = None
            return True
        return False

    def close(self):
        try:
            connections.disconnect("default")
        except Exception:
            pass