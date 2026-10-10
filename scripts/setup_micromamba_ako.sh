#!/usr/bin/env bash
# Create a user-space CUDA/Triton env with micromamba (no sudo/apt required).
# Installs into ~/micromamba/envs/ako and editable-installs this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.local/bin" "$HOME/tmp"
cd "$HOME/tmp"

if [ ! -x "$HOME/.local/bin/micromamba" ]; then
  curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o micromamba.tar.bz2
  python3 - <<'PY'
import bz2, tarfile
from pathlib import Path
raw = bz2.decompress(Path("micromamba.tar.bz2").read_bytes())
Path("micromamba.tar").write_bytes(raw)
with tarfile.open("micromamba.tar") as tf:
    member = next(m for m in tf.getmembers() if m.name.endswith("bin/micromamba"))
    tf.extract(member, path=".")
src = Path(member.name)
dest = Path.home() / ".local/bin/micromamba"
dest.write_bytes(src.read_bytes())
dest.chmod(0o755)
print("installed", dest)
PY
fi

eval "$(micromamba shell hook -s bash)"
micromamba create -y -n ako -c conda-forge python=3.12 pip cxx-compiler c-compiler
micromamba activate ako

export CC="$(which gcc)"
export CXX="$(which g++)"
export CPATH="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))'):${CPATH:-}"

cd "$ROOT"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[dev,hf,gpu]"

python scripts/check_environment.py
echo "Activate later with: source $ROOT/activate_gpu.sh"
