import os

# AWS / Bedrock
AWS_REGION = "ap-southeast-2"
BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Storage
PARSED_BUCKET = os.environ["PARSED_BUCKET"]
PARSED_PREFIX = "docs/"

# Database (env-driven)
DB_HOST = os.environ["DB_HOST"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

DB_PORT = 5432
DB_NAME = "postgres"

# pgvector
PGVECTOR_TABLE = "data_wiki_rag_nodes"
PGVECTOR_SCHEMA = "public"

EMBED_DIM = int(os.environ["EMBED_DIM"])
