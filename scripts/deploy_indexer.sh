#!/usr/bin/env bash
#
# Deploy the indexer ECS scheduled task from GitHub Actions.
#
# Requires:
#   - Running inside GitHub Actions
#   - AWS credentials already configured (OIDC)
#   - GITHUB_SHA set
#   - aws, docker, jq installed

set -euo pipefail

readonly AWS_REGION="ap-southeast-2"
readonly ECR_REPO_NAME="wiki-rag-agentic-indexer"
readonly DOCKERFILE="indexer/Dockerfile"
readonly CONTEXT_DIR="indexer"
readonly CONTAINER_NAME="indexer"
readonly INDEXER_RULE_NAME="wiki-rag-agentic-indexer"
readonly TARGET_ID="ecs-indexer"

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

get_eventbridge_target_json() {
  aws events list-targets-by-rule \
    --rule "$INDEXER_RULE_NAME" \
    --query "Targets[?Id=='$TARGET_ID'] | [0]" \
    --output json
}

get_current_task_definition_arn_from_target() {
  local target_json="$1"
  echo "$target_json" | jq -r '.EcsParameters.TaskDefinitionArn'
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

update_eventbridge_target_task_definition() {
  local target_json="$1"
  local new_td_arn="$2"

  local updated_target
  updated_target="$(
    echo "$target_json" | jq --arg TD "$new_td_arn" '
      .EcsParameters.TaskDefinitionArn = $TD
      | {Id, Arn, RoleArn, EcsParameters, Input, InputPath, InputTransformer}
      | with_entries(select(.value != null))
    '
  )"

  aws events put-targets \
    --rule "$INDEXER_RULE_NAME" \
    --targets "[$updated_target]"
}

main() {
  require_cmd aws
  require_cmd docker
  require_cmd jq

  require_env

  log "Deploying commit $GITHUB_SHA"

  local target_json current_td_arn image_uri new_td_arn

  target_json="$(get_eventbridge_target_json)"
  [[ "$target_json" != "null" ]] || die "EventBridge target not found"

  current_td_arn="$(get_current_task_definition_arn_from_target "$target_json")"
  [[ -n "$current_td_arn" && "$current_td_arn" != "null" ]] \
    || die "Target missing TaskDefinitionArn"

  log "Current task definition: $current_td_arn"

  image_uri="$(build_and_push_image "$GITHUB_SHA")"
  log "New image: $image_uri"

  new_td_arn="$(register_new_task_definition_with_image "$current_td_arn" "$image_uri")"
  log "New task definition: $new_td_arn"

  update_eventbridge_target_task_definition "$target_json" "$new_td_arn"

  log "Deploy complete"
}

main "$@"
