#!/usr/bin/env bash
set -euo pipefail

readonly REGION="ap-southeast-2"

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

############################################
# Bucket cleanup helpers
############################################

_list_object_versions() {
  local bucket="$1"

  aws s3api list-object-versions \
    --region "$REGION" \
    --bucket "$bucket" \
    --max-items 1000 \
    --output json
}

_extract_counts() {
  # Output: "<versions>\t<delete_markers>"
  jq -r '[ (.Versions // [] | length),
           (.DeleteMarkers // [] | length) ] | @tsv'
}

_build_delete_payload() {
  jq -c '{
    Objects: (
      ((.Versions // []) | map({Key:.Key, VersionId:.VersionId})) +
      ((.DeleteMarkers // []) | map({Key:.Key, VersionId:.VersionId}))
    ),
    Quiet: true
  }'
}

############################################
# Bucket cleanup
############################################

empty_bucket_versions() {
  local bucket="$1"

  log "Emptying versioned bucket s3://$bucket..."

  while :; do
    local payload
    payload="$(_list_object_versions "$bucket")"

    local versions markers
    read -r versions markers <<< "$(echo "$payload" | _extract_counts)"

    if [[ "$versions" == "0" && "$markers" == "0" ]]; then
      log "Bucket s3://$bucket is empty."
      break
    fi

    local delete_json
    delete_json="$(echo "$payload" | _build_delete_payload)"

    log "Deleting $versions versions and $markers delete markers..."
    aws s3api delete-objects \
      --region "$REGION" \
      --bucket "$bucket" \
      --delete "$delete_json" >/dev/null
  done
}

delete_bucket() {
  local bucket="$1"

  log "Deleting bucket s3://$bucket..."
  aws s3api delete-bucket \
    --region "$REGION" \
    --bucket "$bucket"
}

main() {
  require_cmd aws
  require_cmd jq

  local buckets=(
    "wiki-rag-agentic-raw-885164491973"
    "wiki-rag-agentic-parsed-885164491973"
  )

  local bucket
  for bucket in "${buckets[@]}"; do
    empty_bucket_versions "$bucket"
    delete_bucket "$bucket"
  done

  log "S3 cleanup complete."
}

main "$@"
