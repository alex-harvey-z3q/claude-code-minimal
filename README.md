# claude-code-minimal

Minimal Claude-Code-like agent and Wikipedia RAG demo. It can run locally with
a filesystem sandbox, or on AWS using Terraform + ECS Fargate:

1. Ingest Wikipedia content into S3
2. Index parsed content into Postgres (pgvector)
3. Serve a FastAPI RAG/coding-agent API behind an ALB

---

## Architecture

```
ALB → ECS (FastAPI API)
        ↓
    pgvector (RDS)
        ↑
   Indexer (ECS task)
        ↑
   Parsed S3
        ↑
   Ingest (ECS task)
```

---

## Prereqs

- Terraform >= 1.6
- AWS CLI configured
- Python >= 3.12 for local development
- jq
- psql (optional)

---

## 1) Secrets

Create:

claude-code-minimal/app

```json
{
  "DB_PASSWORD": "your-db-password"
}
```

---

## 2) Deploy infra

```bash
cd terraform
terraform init
terraform apply
```

Set the printed `github_actions_role_arn` output as the GitHub Actions
`AWS_ROLE_TO_ASSUME` secret before enabling deployment workflows.

---

## 3) Push containers

ECS uses :latest, so you MUST deploy images:

- deploy-api
- deploy-ingest
- deploy-indexer

To push code to `main` without running deployment jobs, set
`deploy: false` in `.github/deploy.yml`. Deployment workflows may still start,
but the shared deploy action skips AWS credentials, ECR login, and deploy
scripts. Set `deploy: true` when the AWS infrastructure is ready for
deployments.

---

## 4) Bootstrap data

```bash
bash scripts/run_ingest.sh
bash scripts/run_indexer.sh
```

---

## 5) Test API

```bash
alb="$(terraform output -raw api_url)"
curl "$alb/query?q=What%20is%20Kubernetes?"
```

---

## Local agent development

The API and sandbox can run locally on a MacBook, but real agent responses still
need an LLM provider unless you use the smoke-test `fake` provider.

```bash
cd api
make venv
make install
cd ..
cp api/.env.example api/.env.local
make dev-api
```

Local provider options:

- `LLM_PROVIDER=fake` runs without Bedrock and creates a tiny smoke
  implementation when the coding workflow is called.
- `LLM_PROVIDER=bedrock` uses AWS credentials and the configured Bedrock model.
  The local example uses the AU Claude Sonnet 4.6 inference profile for
  `ap-southeast-2`.

`api/.env.example` sets `DEFAULT_USE_RETRIEVAL=false` so local workflow calls do
not require Postgres/pgvector. Enable retrieval only after the database and
embedding provider are available.

With the local API running, try:

```bash
curl -sG 'http://127.0.0.1:8000/query' \
  --data-urlencode 'q=Build a tiny Python hello module with unittest tests.' \
  --data-urlencode 'use_retrieval=false'
```

Run component linting with:

```bash
(cd api && make lint)
(cd ingest && make lint)
(cd indexer && make lint)
```

Each component lint runs Ruff plus Vulture dead-code detection.

Run API unit tests with coverage checking with:

```bash
(cd api && make test)
```

The API coverage floor is enforced at the current baseline in
`api/pyproject.toml`; raise it when adding tests.

---

## Tear Down

```bash
terraform destroy
```

You must empty:
- S3 buckets
- ECR repos

---

## License

MIT.
