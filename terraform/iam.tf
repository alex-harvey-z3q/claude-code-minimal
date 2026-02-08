############################################
# Assume-role policy documents
############################################

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "events_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

############################################
# ECS task execution role (pull images, write logs, read secrets, etc.)
############################################

resource "aws_iam_role" "task_execution" {
  name               = "${local.project}-task-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy_attachment" "task_exec" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "task_exec_secrets_doc" {
  statement {
    sid    = "AllowReadAppSecrets"
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
    ]
    resources = [data.aws_secretsmanager_secret.app.arn]
  }
}

resource "aws_iam_role_policy" "task_exec_secrets" {
  name   = "${local.project}-task-exec-secrets"
  role   = aws_iam_role.task_execution.id
  policy = data.aws_iam_policy_document.task_exec_secrets_doc.json
}

############################################
# ECS task role (application permissions)
############################################

resource "aws_iam_role" "task_role" {
  name               = "${local.project}-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

data "aws_iam_policy_document" "task_policy_doc" {
  statement {
    sid     = "AllowS3Access"
    effect  = "Allow"
    actions = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]

    resources = [
      aws_s3_bucket.raw.arn,
      "${aws_s3_bucket.raw.arn}/*",
      aws_s3_bucket.parsed.arn,
      "${aws_s3_bucket.parsed.arn}/*",
    ]
  }

  statement {
    sid     = "AllowBedrockInference"
    effect  = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = ["*"]
  }

  statement {
    sid     = "AllowAwsMarketplaceModelSubscription"
    effect  = "Allow"
    actions = [
      "aws-marketplace:Subscribe",
      "aws-marketplace:Unsubscribe",
      "aws-marketplace:ViewSubscriptions",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "task_policy" {
  name   = "${local.project}-task-policy"
  policy = data.aws_iam_policy_document.task_policy_doc.json
}

resource "aws_iam_role_policy_attachment" "task_policy_attach" {
  role       = aws_iam_role.task_role.name
  policy_arn = aws_iam_policy.task_policy.arn
}

############################################
# EventBridge role to run ECS tasks
############################################

resource "aws_iam_role" "events_run_task" {
  name               = "${local.project}-events-run-task"
  assume_role_policy = data.aws_iam_policy_document.events_assume.json
}

data "aws_iam_policy_document" "events_run_task_policy_doc" {
  statement {
    sid     = "AllowRunTasks"
    effect  = "Allow"
    actions = ["ecs:RunTask"]

    resources = [
      aws_ecs_task_definition.ingest.arn,
      aws_ecs_task_definition.indexer.arn,
    ]

    condition {
      test     = "ArnLike"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.this.arn]
    }
  }

  statement {
    sid     = "AllowPassTaskRoles"
    effect  = "Allow"
    actions = ["iam:PassRole"]

    resources = [
      aws_iam_role.task_execution.arn,
      aws_iam_role.task_role.arn,
    ]
  }
}

resource "aws_iam_role_policy" "events_run_task" {
  name   = "${local.project}-events-run-task"
  role   = aws_iam_role.events_run_task.id
  policy = data.aws_iam_policy_document.events_run_task_policy_doc.json
}
