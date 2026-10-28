#!/bin/zsh
set -eu
cd "$(dirname "$0")/platform"
export SUPPORT_PYTHON="$PWD/.venv-support/bin/python"
exec .venv-api/bin/python run_local.py
