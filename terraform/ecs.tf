############################################
# ECR Repositories
############################################

resource "aws_ecr_repository" "api" { name = "${local.project}-api" }
resource "aws_ecr_repository" "ingest" { name = "${local.project}-ingest" }
resource "aws_ecr_repository" "indexer" { name = "${local.project}-indexer" }

############################################
# ECS Cluster
############################################

resource "aws_ecs_cluster" "this" {
  name = local.project
}

############################################
# Task Definitions
############################################

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.project}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([{
    name         = "api"
    image        = "${aws_ecr_repository.api.repository_url}:latest"
    essential    = true
    portMappings = [{ containerPort = 8000, hostPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "AWS_REGION", value = local.aws_region },
      { name = "BEDROCK_CHAT_MODEL_ID", value = local.bedrock_chat_model_id },
      { name = "BEDROCK_EMBED_MODEL_ID", value = local.bedrock_embed_model_id },
      { name = "RAW_BUCKET", value = aws_s3_bucket.raw.bucket },
      { name = "PARSED_BUCKET", value = aws_s3_bucket.parsed.bucket },
      { name = "DB_HOST", value = aws_db_instance.postgres.address },
      { name = "DB_PORT", value = "5432" },
      { name = "DB_NAME", value = "postgres" },
      { name = "DB_USER", value = local.db_username },
      { name = "PGVECTOR_SCHEMA", value = "public" },
      { name = "PGVECTOR_TABLE", value = "data_wiki_rag_nodes" },
      { name = "EMBED_DIM", value = local.embed_dim }
    ]
    secrets = [
      { name = "DB_PASSWORD", valueFrom = "${data.aws_secretsmanager_secret.app.arn}:DB_PASSWORD::" }
    ]
    logConfiguration = {
      logDriver = "awslogs",
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name,
        awslogs-region        = local.aws_region,
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "ingest" {
  family                   = "${local.project}-ingest"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([{
    name      = "ingest"
    image     = "${aws_ecr_repository.ingest.repository_url}:latest"
    essential = true
    environment = [
      { name = "AWS_REGION", value = local.aws_region },
      { name = "RAW_BUCKET", value = aws_s3_bucket.raw.bucket },
      { name = "PARSED_BUCKET", value = aws_s3_bucket.parsed.bucket },
      { name = "DB_HOST", value = aws_db_instance.postgres.address },
      { name = "DB_PORT", value = "5432" },
      { name = "DB_NAME", value = "postgres" },
      { name = "DB_USER", value = local.db_username }
    ]
    secrets = [
      { name = "DB_PASSWORD", valueFrom = "${data.aws_secretsmanager_secret.app.arn}:DB_PASSWORD::" }
    ]
    logConfiguration = {
      logDriver = "awslogs",
      options = {
        awslogs-group         = aws_cloudwatch_log_group.ingest.name,
        awslogs-region        = local.aws_region,
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "indexer" {
  family                   = "${local.project}-indexer"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([{
    name      = "indexer"
    image     = "${aws_ecr_repository.indexer.repository_url}:latest"
    essential = true
    environment = [
      { name = "AWS_REGION", value = local.aws_region },
      { name = "BEDROCK_EMBED_MODEL_ID", value = local.bedrock_embed_model_id },
      { name = "EMBED_DIM", value = local.embed_dim },
      { name = "PARSED_BUCKET", value = aws_s3_bucket.parsed.bucket },
      { name = "DB_HOST", value = aws_db_instance.postgres.address },
      { name = "DB_PORT", value = "5432" },
      { name = "DB_NAME", value = "postgres" },
      { name = "DB_USER", value = local.db_username },
      { name = "PGVECTOR_SCHEMA", value = "public" },
      { name = "PGVECTOR_TABLE", value = "data_wiki_rag_nodes" }
    ]
    secrets = [
      { name = "DB_PASSWORD", valueFrom = "${data.aws_secretsmanager_secret.app.arn}:DB_PASSWORD::" }
    ]
    logConfiguration = {
      logDriver = "awslogs",
      options = {
        awslogs-group         = aws_cloudwatch_log_group.indexer.name,
        awslogs-region        = local.aws_region,
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

############################################
# ECS Services (Long-Running)
############################################

resource "aws_security_group" "alb" {
  name   = "${local.project}-alb-sg"
  vpc_id = aws_vpc.this.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "api" {
  name               = "${local.project}-alb"
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]

  idle_timeout = 300
}

resource "aws_lb_target_group" "api" {
  name        = "${local.project}-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.this.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 15
    timeout             = 5
    matcher             = "200"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_ecs_service" "api" {
  name            = "${local.project}-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}

############################################
# EventBridge Scheduling
############################################

resource "aws_cloudwatch_event_rule" "ingest_schedule" {
  name                = "${local.project}-ingest"
  description         = "Scheduled ingest"
  schedule_expression = "rate(6 hours)"
}

resource "aws_cloudwatch_event_target" "ingest" {
  rule      = aws_cloudwatch_event_rule.ingest_schedule.name
  target_id = "ecs-ingest"
  arn       = aws_ecs_cluster.this.arn
  role_arn  = aws_iam_role.events_run_task.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.ingest.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs_tasks.id]
      assign_public_ip = false
    }
  }

  lifecycle {
    ignore_changes = [ecs_target[0].task_definition_arn]
  }
}

resource "aws_cloudwatch_event_rule" "indexer_schedule" {
  name                = "${local.project}-indexer"
  description         = "Scheduled index rebuild"
  schedule_expression = "rate(12 hours)"
}

resource "aws_cloudwatch_event_target" "indexer" {
  rule      = aws_cloudwatch_event_rule.indexer_schedule.name
  target_id = "ecs-indexer"
  arn       = aws_ecs_cluster.this.arn
  role_arn  = aws_iam_role.events_run_task.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.indexer.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.ecs_tasks.id]
      assign_public_ip = false
    }
  }

  lifecycle {
    ignore_changes = [ecs_target[0].task_definition_arn]
  }
}
