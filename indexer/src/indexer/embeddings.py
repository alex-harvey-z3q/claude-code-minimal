from __future__ import annotations

from llama_index.core import Settings
from llama_index.embeddings.bedrock import BedrockEmbedding

from indexer.providers import get_embedding_model


def configure_embeddings() -> BedrockEmbedding:
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    return embed_model


def embed(text: str) -> list[float]:
    model = configure_embeddings()
    return model.get_text_embedding(text)
