from llama_index.core import VectorStoreIndex

from indexer.embeddings import configure_embeddings
from indexer.loader import load_documents
from indexer.nodes import get_splitter
from indexer.vectorstore import get_storage_context


def main() -> None:
    configure_embeddings()
    docs = load_documents()
    splitter = get_splitter()
    storage_context = get_storage_context()

    VectorStoreIndex.from_documents(
        docs,
        transformations=[splitter],
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Indexed {len(docs)} documents into Postgres pgvector")


if __name__ == "__main__":
    main()
