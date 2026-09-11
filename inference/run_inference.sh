#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/inference${PYTHONPATH:+:${PYTHONPATH}}"

exec python -u "${REPO_ROOT}/inference/infer_task3_medgemma15.py" "$@"
