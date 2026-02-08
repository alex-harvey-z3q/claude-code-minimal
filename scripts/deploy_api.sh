#!/usr/bin/env bash
#
# Deploy the api ECS service task definition from GitHub Actions.
#
# This matches the ingest/indexer deploy pattern:
# - Build and push Docker image to ECR (tagged with GITHUB_SHA)
# - Register a new ECS task definition revision with the new image
# - Update the ECS service to use the new task definition
#
# Requires:
#   - Running inside GitHub Actions
#   - AWS credentials already configured (OIDC)
#   - GITHUB_SHA set
#   - aws, docker, jq installed

set -euo pipefail

readonly AWS_REGION="ap-southeast-2"
readonly ECR_REPO_NAME="wiki-rag-agentic-api"
readonly DOCKERFILE="api/Dockerfile"
readonly CONTEXT_DIR="api"
readonly CONTAINER_NAME="api"

# Terraform names (must match your ecs.tf)
readonly ECS_CLUSTER_NAME="wiki-rag-agentic"
readonly ECS_SERVICE_NAME="wiki-rag-agentic-api"

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

require_env() {
  [[ -n "${GITHUB_SHA:-}" ]] || die "GITHUB_SHA is not set (must run in GitHub Actions)"
}

get_account_id() {
  aws sts get-caller-identity --query Account --output text
}

build_and_push_image() {
  local tag="$1"

  local account_id registry image_uri latest_uri

  account_id="$(get_account_id)"
  registry="$account_id".dkr.ecr."$AWS_REGION".amazonaws.com
  image_uri="$registry"/"$ECR_REPO_NAME":"$tag"
  latest_uri="$registry"/"$ECR_REPO_NAME":latest

  log "Building image: $image_uri"
  docker build -f "$DOCKERFILE" -t "$image_uri" "$CONTEXT_DIR" >&2 || die "docker build failed"

  log "Pushing image: $image_uri"
  docker push "$image_uri" >&2 || die "docker push failed"

  log "Tagging image: $latest_uri"
  docker tag "$image_uri" "$latest_uri" || die "docker tag failed"

  log "Pushing image: $latest_uri"
  docker push "$latest_uri" >&2 || die "docker push failed"

  echo "$image_uri"
}

get_current_service_task_definition() {
  aws ecs describe-services \
    --cluster "$ECS_CLUSTER_NAME" \
    --services "$ECS_SERVICE_NAME" \
    --query 'services[0].taskDefinition' \
    --output text
}

register_new_task_definition_with_image() {
  local current_td_arn="$1"
  local new_image="$2"

  local new_td_json
  new_td_json="$(
    aws ecs describe-task-definition \
      --task-definition "$current_td_arn" \
      --query 'taskDefinition' \
      --output json \
      | jq --arg img "$new_image" --arg name "$CONTAINER_NAME" '
          .containerDefinitions |=
            map(if .name == $name then .image = $img else . end)
          | del(
              .taskDefinitionArn,
              .revision,
              .status,
              .requiresAttributes,
              .compatibilities,
              .registeredAt,
              .registeredBy
            )
        '
  )"

  aws ecs register-task-definition \
    --cli-input-json "$new_td_json" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text
}

update_service_task_definition() {
  local new_td_arn="$1"

  aws ecs update-service \
    --cluster "$ECS_CLUSTER_NAME" \
    --service "$ECS_SERVICE_NAME" \
    --task-definition "$new_td_arn" \
    --query 'service.taskDefinition' \
    --output text >/dev/null
}

wait_for_stable() {
  aws ecs wait services-stable \
    --cluster "$ECS_CLUSTER_NAME" \
    --services "$ECS_SERVICE_NAME"
}

main() {
  require_cmd aws
  require_cmd docker
  require_cmd jq

  require_env

  log "Deploying commit $GITHUB_SHA"

  local current_td_arn image_uri new_td_arn

  current_td_arn="$(get_current_service_task_definition)"
  [[ -n "$current_td_arn" && "$current_td_arn" != "None" ]] || die "Could not read current service task definition"
  log "Current task definition: $current_td_arn"

  image_uri="$(build_and_push_image "$GITHUB_SHA")"
  log "New image: $image_uri"

  new_td_arn="$(register_new_task_definition_with_image "$current_td_arn" "$image_uri")"
  log "New task definition: $new_td_arn"

  update_service_task_definition "$new_td_arn"
  log "Service updated"

  wait_for_stable
  log "Deploy complete"
}

main "$@"
