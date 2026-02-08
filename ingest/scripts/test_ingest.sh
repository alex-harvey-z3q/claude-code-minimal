#!/usr/bin/env bash

# Smoke test for the Wikipedia ingestion task.
# Verifies that raw and parsed objects are written to S3.

set -euo pipefail

export AWS_REGION="ap-southeast-2"

readonly PAGES=(
  "Artificial intelligence"
  "Machine learning"
  "Large language model"
)

usage() {
  if [[ -n "$1" ]]; then
    echo "$1"
  fi

  cat <<EOF
Usage: $0 -r RAW_BUCKET -p PARSED_BUCKET -g AWS_REGION

Options:
  -r RAW_BUCKET     S3 bucket for raw Wikipedia pages
  -p PARSED_BUCKET  S3 bucket for parsed section documents
  -h                Show this help message
EOF
  exit 1
}

log() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

parse_args() {
  local opt OPTARG OPTIND

  while getopts ":r:p:h" opt; do
    case "$opt" in
      r) export RAW_BUCKET="$OPTARG" ;;
      p) export PARSED_BUCKET="$OPTARG" ;;
      h) usage
         exit 0
         ;;
      *) usage
         exit 1
         ;;
    esac
  done

  [[ -n "${RAW_BUCKET:-}" ]]    || usage "RAW_BUCKET is required"
  [[ -n "${PARSED_BUCKET:-}" ]] || usage "PARSED_BUCKET is required"
}

get_page_id() {
  local title="$1"
  PYTHONPATH=src python - <<EOF
from ingest.wikipedia import fetch_page
print(fetch_page("${title}")["pageid"])
EOF
}

count_s3_objects() {
  local bucket="$1"
  local prefix="$2"

  aws s3api list-objects-v2 \
    --bucket "${bucket}" \
    --prefix "${prefix}" \
    --query 'length(Contents[])' \
    --output text 2>/dev/null || echo 0
}

main() {
  parse_args "$@"

  log "Running ingestion task"
  python -m ingest.main

  local title page_id raw_count parsed_count

  for title in "${PAGES[@]}"; do
    page_id="$(get_page_id "${title}")"

    raw_count="$(count_s3_objects "${RAW_BUCKET}" "pages/${page_id}.json")"
    parsed_count="$(count_s3_objects "${PARSED_BUCKET}" "docs/${page_id}/")"

    [[ "${raw_count}" -ge 1 ]] || die "Missing raw page for '${title}'"
    [[ "${parsed_count}" -ge 1 ]] || die "Missing parsed docs for '${title}'"

    log "OK: ${title} (raw=${raw_count}, parsed=${parsed_count})"
  done

  log "Ingestion smoke test passed"
}

main "$@"
