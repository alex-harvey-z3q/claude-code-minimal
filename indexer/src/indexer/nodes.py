from llama_index.core.node_parser import SentenceSplitter


def get_splitter() -> SentenceSplitter:
    return SentenceSplitter(chunk_size=800, chunk_overlap=150)
