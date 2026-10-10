#!/usr/bin/env bash
# Activate the micromamba GPU env used for AKO on WSL.
# Install once with: bash scripts/setup_micromamba_ako.sh
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate ako
export CC="$(which gcc)"
export CXX="$(which g++)"
export CPATH="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))'):${CPATH:-}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
