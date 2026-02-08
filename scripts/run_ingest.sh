#!/usr/bin/env bash

set -euo pipefail

readonly REGION="ap-southeast-2"
readonly CLUSTER="wiki-rag-agentic"
readonly TASK_FAMILY="wiki-rag-agentic-ingest"
readonly SUBNET_NAME_PREFIX="wiki-rag-agentic-private-"

log() {
  echo "[INFO] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
}

get_latest_task_def_arn() {
  aws ecs describe-task-definition \
    --task-definition "$TASK_FAMILY" \
    --region "$REGION" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text
}

get_first_private_subnet_id() {
  # shellcheck disable=SC2016
  aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=$SUBNET_NAME_PREFIX*" \
    --query 'Subnets | sort_by(@, &Tags[?Key==`Name`].Value | [0])[0].SubnetId' \
    --output text
}

get_vpc_id_from_subnet() {
  local subnet_id="$1"

  aws ec2 describe-subnets \
    --region "$REGION" \
    --subnet-ids "$subnet_id" \
    --query 'Subnets[0].VpcId' \
    --output text
}

get_ecs_tasks_sg_id() {
  aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$CLUSTER-ecs-sg" \
    --query 'SecurityGroups[0].GroupId' \
    --output text
}

get_private_subnet_ids_csv() {
  local vpc_id="$1"

  # shellcheck disable=SC2016
  aws ec2 describe-subnets \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$vpc_id" "Name=tag:Name,Values=$SUBNET_NAME_PREFIX*" \
    --query 'Subnets | sort_by(@, &Tags[?Key==`Name`].Value | [0])[*].SubnetId' \
    --output text \
  | awk 'BEGIN{ORS="";} {for (i=1; i<=NF; i++) {printf "%s%s", $i, (i==NF ? "" : ",")}}'
}

run_task() {
  local task_def_arn="$1"
  local subnets_csv="$2"
  local sg_id="$3"

  aws ecs run-task \
    --cluster "$CLUSTER" \
    --launch-type FARGATE \
    --task-definition "$task_def_arn" \
    --network-configuration "awsvpcConfiguration={subnets=[$subnets_csv],securityGroups=[$sg_id],assignPublicIp=DISABLED}" \
    --region "$REGION" \
    --query 'tasks[0].taskArn' \
    --output text
}

wait_for_task() {
  local task_arn="$1"

  aws ecs wait tasks-stopped \
    --cluster "$CLUSTER" \
    --tasks "$task_arn" \
    --region "$REGION"
}

get_exit_code() {
  local task_arn="$1"

  aws ecs describe-tasks \
    --cluster "$CLUSTER" \
    --tasks "$task_arn" \
    --region "$REGION" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text
}

main() {
  require_cmd aws
  require_cmd awk

  log "Resolving latest task definition for $TASK_FAMILY..."

  local task_def_arn
  task_def_arn="$(get_latest_task_def_arn)"
  [[ -n "$task_def_arn" && "$task_def_arn" != "None" ]] || die "Could not resolve task definition for $TASK_FAMILY"

  log "Using task definition: $task_def_arn"
  log "Detecting VPC from subnet tags Name=$SUBNET_NAME_PREFIX* ..."

  local first_subnet_id
  first_subnet_id="$(get_first_private_subnet_id)"
  [[ -n "$first_subnet_id" && "$first_subnet_id" != "None" ]] \
    || die "No subnets found with tag Name=$SUBNET_NAME_PREFIX* in region $REGION"

  local vpc_id
  vpc_id="$(get_vpc_id_from_subnet "$first_subnet_id")"
  [[ -n "$vpc_id" && "$vpc_id" != "None" ]] || die "Could not determine VPC ID from subnet $first_subnet_id"

  log "VPC ID: $vpc_id"
  log "Finding private subnets in VPC $vpc_id with tag Name=$SUBNET_NAME_PREFIX* ..."

  local subnets_csv
  subnets_csv="$(get_private_subnet_ids_csv "$vpc_id")"
  [[ -n "$subnets_csv" ]] || die "No subnets found in VPC $vpc_id with tag Name=$SUBNET_NAME_PREFIX*"

  log "Subnets: $subnets_csv"
  log "Finding ECS tasks security group named $CLUSTER-ecs-sg ..."

  local sg_id
  sg_id="$(get_ecs_tasks_sg_id)"
  [[ -n "$sg_id" && "$sg_id" != "None" ]] || die "Could not find security group with group-name $CLUSTER-ecs-sg"

  log "Security group: $sg_id"
  log "Running ingest task..."

  local task_arn
  task_arn="$(run_task "$task_def_arn" "$subnets_csv" "$sg_id")"
  [[ -n "$task_arn" && "$task_arn" != "None" ]] || die "Failed to start ECS task."

  log "Task ARN: $task_arn"
  log "Waiting for task to finish..."

  wait_for_task "$task_arn"

  local exit_code
  exit_code="$(get_exit_code "$task_arn")"
  [[ "$exit_code" == "0" ]] || die "Ingest failed with exit code $exit_code"

  log "Ingest completed successfully."
}

main "$@"
