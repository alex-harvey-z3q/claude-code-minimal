import os

# AWS / Bedrock
AWS_REGION = "ap-southeast-2"

BEDROCK_CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Database (required from environment)
DB_HOST = os.environ["DB_HOST"]
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# pgvector
PGVECTOR_SCHEMA = "public"
PGVECTOR_TABLE = "data_wiki_rag_nodes"
EMBED_DIM = int(os.environ["EMBED_DIM"])

# Retrieval / generation tuning
TOP_K = 5
TEMPERATURE = 0.2
MAX_TOKENS = 512
