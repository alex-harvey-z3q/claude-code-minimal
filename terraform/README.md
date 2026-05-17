# claude-code-minimal Terraform

Terraform for the AWS deployment of the Claude-Code-like Wikipedia RAG demo:

- S3 buckets for raw and parsed content
- ECS Fargate for the API, ingestion worker, and indexer
- RDS Postgres with pgvector
- Secrets Manager for database credentials
- CloudWatch Logs
- EventBridge scheduled ingestion and indexing
- ALB for the API
- GitHub Actions OIDC role for deployment workflows

## Model Layer

The current stack is Bedrock-only:

- API chat model: `local.bedrock_chat_model_id`
- API/indexer embedding model: `local.bedrock_embed_model_id`
- Default region: `local.aws_region`

The default model IDs live in `locals.tf`. Bedrock model access is account and
region dependent, so confirm the configured models are enabled before deploying
or running local Bedrock calls.

## Secrets Manager

Create a Secrets Manager secret named `claude-code-minimal/app` before applying:

```json
{
  "DB_PASSWORD": "your-postgres-password"
}
```

The ECS task definitions inject `DB_PASSWORD` from that secret. No OpenAI secret
is required.

## Apply

```bash
cd terraform
terraform init
terraform apply
```

After apply, set the `github_actions_role_arn` output as the GitHub Actions
repository secret `AWS_ROLE_TO_ASSUME`.

Build and push the three service images to the printed ECR repos:

- API image -> `...-api:latest`
- ingest image -> `...-ingest:latest`
- indexer image -> `...-indexer:latest`

The root `.github/deploy.yml` file controls whether GitHub deploy workflows
actually run AWS deployment steps.

## Notes

- If you change the embedding model, keep `local.embed_dim` aligned with the
  vectors stored in pgvector.
- The API load balancer URL is available as the `api_url` Terraform output.
