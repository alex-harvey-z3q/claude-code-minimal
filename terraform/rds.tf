############################################
# DB Subnet Group (public subnets)
############################################
resource "aws_db_subnet_group" "db" {
  name       = "${local.project}-dbsubnets"
  subnet_ids = aws_subnet.public[*].id
}

############################################
# ECS Tasks Security Group (unchanged)
############################################
resource "aws_security_group" "ecs_tasks" {
  name   = "${local.project}-ecs-sg"
  vpc_id = aws_vpc.this.id

  # Allow ALB -> API tasks on 8000
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

############################################
# RDS Security Group
# - Allows ECS tasks
# - Allows your laptop IP (local.laptop_ip)
############################################
resource "aws_security_group" "rds" {
  name        = "${local.project}-rds-sg"
  description = "RDS access from ECS and laptop"
  vpc_id      = aws_vpc.this.id

  # ECS tasks -> RDS
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  # Laptop -> RDS
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [local.laptop_ip]
    description = "Laptop psql access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

############################################
# RDS Instance (public)
############################################
resource "aws_db_instance" "postgres" {
  identifier              = "${local.project}-pg"
  engine                  = "postgres"
  engine_version          = "15"
  instance_class          = "db.t4g.micro"
  allocated_storage       = 20
  db_subnet_group_name    = aws_db_subnet_group.db.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  username                = local.db_username
  password                = local.db_password
  publicly_accessible     = true
  skip_final_snapshot     = true
  deletion_protection     = false
  backup_retention_period = 0
}

data "dns_a_record_set" "rds_endpoint_a" {
  host = aws_db_instance.postgres.address
}
