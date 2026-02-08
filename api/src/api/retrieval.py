from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from . import config


@lru_cache(maxsize=1)
def get_embedding_model() -> BedrockEmbedding:
    return BedrockEmbedding(
        model_name=config.BEDROCK_EMBED_MODEL_ID,
        region_name=config.AWS_REGION,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        table_name=config.PGVECTOR_TABLE,
        schema_name=config.PGVECTOR_SCHEMA,
        embed_dim=config.EMBED_DIM,
    )


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model

    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=config.TOP_K)


def _metadata_value(metadata: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            return str(value)
    return default


def _build_retrieval_query(question: str) -> str:
    lower = question.lower()
    if "minesweeper" in lower:
        return (
            f"{question}\n\n"
            "Focus on Minesweeper gameplay rules, adjacent mine counts, revealing cells, "
            "flagging cells, empty-area recursive reveal, and win/loss conditions."
        )
    return question


def retrieve(question: str) -> list[dict]:
    retriever = get_retriever()
    nodes = retriever.retrieve(_build_retrieval_query(question))

    evidence: list[dict] = []
    for node_with_score in nodes:
        node = node_with_score.node
        metadata = dict(node.metadata or {})

        evidence.append({
            "page": _metadata_value(
                metadata,
                "page_title",
                "page",
                "title",
                default="Unknown page",
            ),
            "section": _metadata_value(
                metadata,
                "section_title",
                "section",
                default="",
            ),
            "url": _metadata_value(metadata, "url", default=""),
            "revision_id": metadata.get("revision_id"),
            "excerpt": node.get_content(),
        })

    return evidence
