"""
Milvus 向量索引构建模块（招聘领域）
把岗位档案文档向量化后写入 Milvus,并提供向量检索接口。
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

from .index_store import digest

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
        self.embedding_model = embedding_model
        self._embedder = None
        self._connected = False
        self.collection: Optional[Collection] = None

    @property
    def embedder(self):
        if self._embedder is None:
            from .embedding import Embedder
            self._embedder = Embedder(self.embedding_model)
            if self._embedder.dimension != self.dimension:
                raise ValueError("Embedding 维度与 Milvus schema 不匹配")
        return self._embedder

    def _connect(self):
        if not self._connected:
            connections.connect(alias="default", host=self.host, port=self.port, timeout=5)
            self._connected = True

    # ========== 集合管理 ==========
    def has_collection(self) -> bool:
        self._connect()
        return utility.has_collection(self.collection_name, timeout=5)

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16000),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="skills", dtype=DataType.JSON),
            FieldSchema(name="required_skills", dtype=DataType.JSON),
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
        self.collection.load(timeout=15)
        logger.info("Milvus 集合加载完成: %s", self.collection_name)
        return True

    # ========== 写入向量 ==========
    def build_vector_index(self, documents: List[Document], store=None) -> bool:
        if not documents:
            logger.warning("无可写入的文档")
            return False
        if not self.has_collection():
            self._create_collection()
        else:
            self.collection = Collection(self.collection_name)

        texts = [d.page_content for d in documents]
        keys = [digest([self.embedding_model, self.dimension, "bge-doc-v1", t]) for t in texts]
        embeddings = [store.get_vector(k) if store else None for k in keys]
        missing = [i for i, v in enumerate(embeddings) if v is None]
        if missing:
            encoded = self.embedder.encode_documents([texts[i] for i in missing])
            if len(encoded) != len(missing):
                raise ValueError("Embedding 返回数量不匹配")
            for i, vector in zip(missing, encoded):
                if len(vector) != self.dimension:
                    raise ValueError("Embedding 返回维度不匹配")
                embeddings[i] = vector
                if store:
                    store.put_vector(keys[i], vector)

        rows = [
            {
                "id": d.metadata["node_id"],
                "text": d.page_content,
                "content_hash": d.metadata["content_hash"],
                "skills": d.metadata.get("skills", []),
                "required_skills": d.metadata.get("required_skills", []),
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

        if any(len(row["text"].encode("utf-8")) > 16000 for row in rows):
            raise ValueError("岗位档案超过 Milvus 文本字段上限，请拆分后导入")
        self.collection.upsert(rows)
        self.collection.flush()
        self.collection.load(timeout=15)
        logger.info("写入向量 %d 条", len(rows))
        return True

    # ========== 向量检索 ==========
    def vector_search(self, query: str, top_k: int = 5, filters=None) -> List[Document]:
        if top_k <= 0:
            return []
        if filters and filters.restrict_ids and not filters.job_ids:
            return []
        if self.collection is None:
            if not self.load_collection():
                return []

        query_vec = self.embedder.encode_query(query)
        search_params = {"metric_type": "COSINE", "params": {"ef": max(64, top_k)}}
        output_fields = ["text", "job_name", "node_id", "company", "category", "location", "salary_range", "seniority", "skills", "required_skills", "content_hash"]
        import json
        clauses = []
        if filters:
            for field in ("location", "company", "seniority"):
                value = getattr(filters, field)
                if value:
                    clauses.append(f"{field} == {json.dumps(value, ensure_ascii=False)}")
            if filters.restrict_ids:
                if not filters.job_ids:
                    return []
                clauses.append("node_id in " + json.dumps(filters.job_ids, ensure_ascii=False))
        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
            expr=" and ".join(clauses) or None,
            timeout=10,
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
                        "skills": hit.entity.get("skills", []),
                        "required_skills": hit.entity.get("required_skills", []),
                        "content_hash": hit.entity.get("content_hash", ""),
                    },
                )
            )
        return [d for d in documents if filters is None or filters.matches(d)]

    def get_collection_stats(self) -> Dict[str, Any]:
        if self.collection is None:
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
