# wiki-rag-bedrock

Terraform + ECS Fargate pipeline that:
1) ingests Wikipedia content into S3
2) indexes into Postgres (pgvector)
3) serves a FastAPI RAG API behind an ALB

---

## 🧱 Architecture

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
- jq
- psql (optional)

---

## 1) Secrets

Create:

wiki-rag-bedrock/app

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

---

## 3) Push containers

ECS uses :latest, so you MUST deploy images:

- deploy-api
- deploy-ingest
- deploy-indexer

To push code to `main` without running deployment jobs, set
`deploy: false` in `.github/deploy.yml`. Set it to `true` when the AWS
infrastructure is ready for deployments.

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
  The local example uses the APAC Claude Sonnet 4 inference profile for
  `ap-southeast-2`.

`api/.env.example` sets `DEFAULT_USE_RETRIEVAL=false` so local workflow calls do
not require Postgres/pgvector. Enable retrieval only after the database and
embedding provider are available.

---

## 🧹 Tear down

```bash
terraform destroy
```

You must empty:
- S3 buckets
- ECR repos

---

## License

MIT.
