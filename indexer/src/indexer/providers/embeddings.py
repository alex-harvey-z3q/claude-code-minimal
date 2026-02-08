from __future__ import annotations

from llama_index.embeddings.bedrock import BedrockEmbedding

from indexer import settings


def get_embedding_model() -> BedrockEmbedding:
    return BedrockEmbedding(
        model_name=settings.BEDROCK_EMBED_MODEL_ID,
        region_name=settings.AWS_REGION,
    )
