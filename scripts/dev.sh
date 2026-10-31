#!/usr/bin/env bash
# Dev helper: ./scripts/dev.sh {setup|test|lint|evals|run|check}
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
PY="${PYTHON:-python3}"

activate() {
  [[ -d "$VENV" ]] || { echo "venv não encontrado; rode: $0 setup" >&2; exit 1; }
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

cmd="${1:-help}"
case "$cmd" in
  setup)
    [[ -d "$VENV" ]] || "$PY" -m venv "$VENV"
    activate
    pip install -q --upgrade pip
    pip install -q -e ".[dev,api]"
    echo "ok: ambiente pronto em $VENV"
    ;;
  test)  activate; shift; pytest -q "$@" ;;
  lint)  activate; ruff check . && ruff format --check . ;;
  evals) activate; python -m analisador_vagas.evals --markdown ;;
  run)   activate; exec uvicorn analisador_vagas.api:app --app-dir src --reload --port "${PORT:-8000}" ;;
  check) "$0" lint && "$0" test && "$0" evals ;;
  *)
    echo "uso: $0 {setup|test [args pytest]|lint|evals|run|check}"
    [[ "$cmd" == help ]] || exit 2
    ;;
esac
