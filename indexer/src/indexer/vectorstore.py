from __future__ import annotations

from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore

from indexer.settings import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    PGVECTOR_TABLE,
    PGVECTOR_SCHEMA,
    EMBED_DIM,
)


def get_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=PGVECTOR_TABLE,
        schema_name=PGVECTOR_SCHEMA,
        embed_dim=EMBED_DIM,
    )


def get_storage_context() -> StorageContext:
    vector_store = get_vector_store()
    return StorageContext.from_defaults(vector_store=vector_store)
