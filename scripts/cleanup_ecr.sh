#!/usr/bin/env bash

set -euo pipefail

readonly REGION="ap-southeast-2"
readonly REPOS=("wiki-rag-agentic-api" "wiki-rag-agentic-ingest" "wiki-rag-agentic-indexer")

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

delete_all_images_in_repo() {
  local repo="$1"
  local digest digests

  log "Listing image digests in ECR repo $repo..."

  digests="$(aws ecr list-images \
    --region "$REGION" \
    --repository-name "$repo" \
    --query 'imageIds[?imageDigest!=null].imageDigest' \
    --output text)"

  if [[ -z "$digests" ]]; then
    log "Repo $repo already empty."
    return 0
  fi

  for digest in $digests; do
    log "Deleting image digest $digest from $repo..."
    aws ecr batch-delete-image \
      --region "$REGION" \
      --repository-name "$repo" \
      --image-ids "imageDigest=$digest" >/dev/null
  done

  log "Deleted images from $repo."
}

delete_repo() {
  local repo="$1"
  log "Deleting ECR repo $repo..."
  aws ecr delete-repository \
    --region "$REGION" \
    --repository-name "$repo" >/dev/null
  log "Deleted repo $repo."
}

main() {
  require_cmd aws

  local delete_repos="false"

  if [[ "${1:-}" == "--delete-repos" ]]; then
    delete_repos="true"
  fi

  for repo in "${REPOS[@]}"; do
    delete_all_images_in_repo "$repo"
    if [[ "$delete_repos" == "true" ]]; then
      delete_repo "$repo"
    fi
  done

  log "ECR cleanup complete."
  if [[ "$delete_repos" == "true" ]]; then
    log "Repos deleted."
  else
    log "Repos left in place. Re-run with --delete-repos to delete repositories too."
  fi
}

main "$@"
