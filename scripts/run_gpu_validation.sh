#!/usr/bin/env bash
# Run GPU validation suite and publish artefacts under reports/rtx3070/.
# Usage:
#   source activate_gpu.sh
#   bash scripts/run_gpu_validation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${REPORT_DIR:-reports/rtx3070}"
mkdir -p "$OUT"

# Include the torch.compile baseline: a speedup measured only against the
# unfused reference overstates the win, since PyTorch fuses RMSNorm itself.
export AKO_ENABLE_COMPILE="${AKO_ENABLE_COMPILE:-1}"

echo "=== environment ==="
python scripts/check_environment.py | tee "$OUT/environment.txt"

echo "=== GPU tests ==="
pytest -q -m gpu --tb=line | tee "$OUT/pytest_gpu.txt"

echo "=== benchmark rmsnorm ==="
python scripts/benchmark_rmsnorm.py --device cuda --output-dir "$OUT"

echo "=== benchmark rope ==="
python scripts/benchmark_rope.py --device cuda --style neox --output-dir "$OUT"

echo "=== optimize_model fp16 ==="
python scripts/optimize_model.py --device cuda --dtype fp16 --fresh \
  --batch-size 4 --seq-len 1024 --hidden-dim 2048 --num-heads 16 --head-dim 128 \
  --output-dir "$OUT"
mv -f "$OUT/selected_config.json" "$OUT/selected_config_fp16.json"

echo "=== optimize_model bf16 (correctness proof for dtype budgets) ==="
python scripts/optimize_model.py --device cuda --dtype bf16 --fresh \
  --batch-size 4 --seq-len 1024 --hidden-dim 2048 --num-heads 16 --head-dim 128 \
  --output-dir "$OUT"
mv -f "$OUT/selected_config.json" "$OUT/selected_config_bf16.json"

echo "=== serving simulation (policy comparison on a request trace) ==="
python scripts/simulate_serving.py --device cuda --dtype fp16 \
  --preset small --requests 128 --max-batch 8 \
  --median-prompt 256 --median-decode 64 --max-prompt 1024 \
  --output-dir "$OUT"

echo "=== profile_model ==="
python scripts/profile_model.py --device cuda --dtype fp16 \
  --seq-len 512 --hidden-dim 2048 --n-layers 4 --n-heads 32 --n-kv-heads 4 \
  --output-dir "$OUT"

echo "=== summarize ==="
python scripts/summarize_reports.py --report-dir "$OUT" --output docs/RESULTS.md

echo "=== DONE ==="
ls -la "$OUT"
