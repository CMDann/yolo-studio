#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/yolo_studio"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${APP_DIR}/requirements.txt"

choose_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(choose_python)"; then
  echo "Python 3.10+ is required but no suitable interpreter was found in PATH." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION##*.}"

if [ "${PYTHON_MAJOR}" -ne 3 ] || [ "${PYTHON_MINOR}" -lt 10 ]; then
  echo "Detected ${PYTHON_BIN} ${PYTHON_VERSION}, but Python 3.10+ is required." >&2
  exit 1
fi

if [ ! -d "${APP_DIR}" ]; then
  echo "Could not find application directory: ${APP_DIR}" >&2
  exit 1
fi

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
  echo "Could not find requirements file: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [ -d "${VENV_DIR}" ] && [ -x "${VENV_DIR}/bin/python" ]; then
  VENV_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "${VENV_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]; then
    cat >&2 <<EOF
Existing virtual environment uses Python ${VENV_PYTHON_VERSION}, but installer selected ${PYTHON_VERSION}.
Please remove ${VENV_DIR} and rerun ./install.sh so dependencies install against one Python version.
EOF
    exit 1
  fi
fi

# Python 3.14 may require building pi-heif from source; ensure libheif is installed first.
if [ "${PYTHON_MINOR}" -ge 14 ] && [ "$(uname -s)" = "Darwin" ]; then
  LIBHEIF_HEADER=""
  for path in /opt/homebrew/include/libheif/heif.h /usr/local/include/libheif/heif.h; do
    if [ -f "${path}" ]; then
      LIBHEIF_HEADER="${path}"
      break
    fi
  done

  if [ -z "${LIBHEIF_HEADER}" ]; then
    cat >&2 <<'EOF'
Install failed risk: Python 3.14 may build pi-heif from source, which needs system libheif headers.

Fix options:
  1) Recommended: install Python 3.12 or 3.13 and rerun ./install.sh
  2) Keep Python 3.14: install libheif first (Homebrew):
       brew install libheif
EOF
    exit 1
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip

# Help source builds find Homebrew libraries on macOS.
if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix)"
  export CPPFLAGS="-I${BREW_PREFIX}/include ${CPPFLAGS:-}"
  export LDFLAGS="-L${BREW_PREFIX}/lib ${LDFLAGS:-}"
  export PKG_CONFIG_PATH="${BREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

python3 -m pip install -r "${REQUIREMENTS_FILE}"

cd "${APP_DIR}"
exec python3 main.py "$@"
