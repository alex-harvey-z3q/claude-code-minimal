resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.project}-api"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/ecs/${local.project}-ingest"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "indexer" {
  name              = "/ecs/${local.project}-indexer"
  retention_in_days = 14
}
