#!/usr/bin/env bash

set -euo pipefail

env_file="${1:-api/.env.local}"

if [[ ! -f "$env_file" ]]; then
  (echo "Local env file not found: $env_file"
   echo "Create one with: cp api/.env.example api/.env.local") >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

export DEFAULT_USE_RETRIEVAL="${DEFAULT_USE_RETRIEVAL:-false}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-/tmp/claude-code-minimal-workspaces}"

cd api
export PYTHONPATH="${PYTHONPATH:-src}"

uvicorn_bin="${UVICORN_BIN:-.venv/bin/uvicorn}"
if [[ ! -x "$uvicorn_bin" ]]; then
  uvicorn_bin="${UVICORN_BIN:-uvicorn}"
fi

args=(api.main:app --host 127.0.0.1 --port "${PORT:-8000}")

if [[ "${DEV_RELOAD:-false}" == "true" ]]; then
  args+=(--reload)
fi

exec "$uvicorn_bin" "${args[@]}"
