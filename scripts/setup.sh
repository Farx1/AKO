#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV="${UV:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV" ]]; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

"$UV" venv .venv --clear
# shellcheck disable=SC1091
source .venv/bin/activate
"$UV" pip install -e ".[dev]"

python scripts/check_environment.py
python -m pytest -q -m "not gpu"
echo "Setup complete. Activate: source .venv/bin/activate"
