output "api_url" {
  value = "http://${aws_lb.api.dns_name}"
}

output "raw_bucket" { value = aws_s3_bucket.raw.bucket }
output "parsed_bucket" { value = aws_s3_bucket.parsed.bucket }
output "db_endpoint" { value = aws_db_instance.postgres.address }
output "db_username" { value = local.db_username }
output "ecr_api" { value = aws_ecr_repository.api.repository_url }
output "ecr_ingest" { value = aws_ecr_repository.ingest.repository_url }
output "ecr_indexer" { value = aws_ecr_repository.indexer.repository_url }

output "github_actions_role_arn" {
  value       = aws_iam_role.github_actions.arn
  description = "IAM role ARN for GitHub Actions OIDC (set as AWS_ROLE_TO_ASSUME secret)."
}

output "rds_endpoint" {
  description = "RDS endpoint hostname"
  value       = aws_db_instance.postgres.address
}

output "rds_public_ips" {
  description = "Resolved IPs for the RDS endpoint (can change)"
  value       = data.dns_a_record_set.rds_endpoint_a.addrs
}
