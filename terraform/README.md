# wiki-rag-terraform

Terraform for the Bedrock + Claude version of the Wikipedia RAG demo on AWS:
- S3 (raw + parsed)
- ECS Fargate (API + ingestion worker + indexer)
- RDS Postgres (for pgvector)
- Secrets Manager
- CloudWatch Logs
- EventBridge scheduled ingestion and indexing
- ALB for API

## What changed for Part VII

This Terraform now supports the new model-layer split:
- `LLM_PROVIDER=bedrock` by default
- `BEDROCK_CHAT_MODEL_ID` injected into the API task
- optional `EMBEDDING_PROVIDER=bedrock` support for API + indexer
- conditional OpenAI secret injection so OpenAI is only required when embeddings or chat still use OpenAI
- ECS task IAM includes Amazon Bedrock inference permissions

The default stack is:
- Claude on Bedrock for generation
- OpenAI embeddings for retrieval/indexing

That means the default secret payload still includes `OPENAI_API_KEY`.

## Secrets Manager payloads

### Default path: Bedrock generation + OpenAI embeddings

```bash
cat > /tmp/wiki-rag-app.json <<EOF
{
  "DB_PASSWORD": "your-postgres-password",
  "OPENAI_API_KEY": "sk-..."
}
EOF
```

### Fully Bedrock-native model layer

If you switch `embedding_provider` to `bedrock` in `locals.tf`, the ECS task definitions stop injecting `OPENAI_API_KEY`, so the secret can be reduced to:

```bash
cat > /tmp/wiki-rag-app.json <<EOF
{
  "DB_PASSWORD": "your-postgres-password"
}
EOF
```

## Apply

```bash
cd terraform
terraform init
terraform apply
```

After apply, build and push three images to the printed ECR repos:
- api image -> `...-api:latest`
- ingest image -> `...-ingest:latest`
- indexer image -> `...-indexer:latest`

## Notes

- Bedrock model access is region- and account-dependent. Make sure the Claude model you configure is enabled in your AWS account and available in `local.aws_region`.
- The default `bedrock_chat_model_id` in `locals.tf` is just a starter value. Swap it to the Claude model version you want to feature in the blog post.
- If you change the embedding model, keep `embed_dim` aligned with the vectors stored in pgvector.
