import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").lower()

# AWS / Bedrock
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")

BEDROCK_CHAT_MODEL_ID = os.getenv(
    "BEDROCK_CHAT_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)
BEDROCK_EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

BEDROCK_CONNECT_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_CONNECT_TIMEOUT_SECONDS", "10"))
BEDROCK_READ_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_READ_TIMEOUT_SECONDS", "300"))

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# pgvector
PGVECTOR_SCHEMA = os.getenv("PGVECTOR_SCHEMA", "public")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "data_wiki_rag_nodes")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# Retrieval / generation tuning
TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
DEFAULT_USE_RETRIEVAL = os.getenv("DEFAULT_USE_RETRIEVAL", "true").lower() in {
    "1",
    "true",
    "yes",
}

# Iterative workflow
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/workspace")
MAX_WORKFLOW_ITERS = int(os.getenv("MAX_WORKFLOW_ITERS", "10"))
TEST_TIMEOUT_SECONDS = int(os.getenv("TEST_TIMEOUT_SECONDS", "30"))
