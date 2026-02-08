data "aws_secretsmanager_secret" "app" {
  name = "${local.project}/app"
}

data "aws_secretsmanager_secret_version" "app_current" {
  secret_id = data.aws_secretsmanager_secret.app.id
}

locals {
  # Expecting the secret string to be JSON like:
  # {"DB_PASSWORD":"..."}
  app_secrets = jsondecode(data.aws_secretsmanager_secret_version.app_current.secret_string)
  db_password = local.app_secrets.DB_PASSWORD
}
