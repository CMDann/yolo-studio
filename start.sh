#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/yolo_studio"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${APP_DIR}" ]; then
  echo "Could not find application directory: ${APP_DIR}" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}. Running install.sh..."
  exec "${SCRIPT_DIR}/install.sh" "$@"
fi

if [ "${VIRTUAL_ENV:-}" != "${VENV_DIR}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

cd "${APP_DIR}"
exec python main.py "$@"
