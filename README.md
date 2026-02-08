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
