#!/bin/bash
# TransitionSense Isambard-AI status probe — run on a login node (no GPU needed).
# Usage: bash scripts/isambard_status.sh
set -uo pipefail

ROOT=/projects/u6xn/transitionsense

echo "=== whoami / host ==="
hostname
whoami

echo "=== my jobs ==="
squeue -u "$(whoami)" -o "%.10i %.12P %.24j %.8T %.10M %.6D %R" || true

echo "=== recent finished jobs ==="
sacct -u "$(whoami)" --starttime=now-2days --format=JobID,JobName%24,State,Elapsed,ExitCode -n | tail -20 || true

echo "=== transitionsense tree ==="
for d in "$ROOT" "$ROOT/TransitionSense" /projects/u6xn/OpenTSLM; do
  if [ -d "$d" ]; then
    echo "--- $d"
    ls "$d" | head -30
  else
    echo "--- $d MISSING"
  fi
done

echo "=== window data ==="
for base in "$ROOT/data/windows" "$ROOT/TransitionSense/data/windows" "$ROOT/results"; do
  if [ -d "$base" ]; then
    echo "--- $base"; ls -la "$base" | grep -E "windows_|splits|timef" || true
  fi
done

echo "=== python envs ==="
for py in "$ROOT/venv/bin/python3" /projects/u6xn/transitionsense/venv/bin/python3; do
  if [ -x "$py" ]; then
    echo "--- $py"
    "$py" -c "import sys; print(sys.version.split()[0])" 2>&1
    "$py" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1 | head -2
    "$py" -c "import peft; print('peft', peft.__version__)" 2>&1 | head -2
  fi
done

echo "=== HF cache (gemma) ==="
ls /projects/u6xn/hf-cache/hub 2>/dev/null | grep -i gemma || echo "no gemma in cache"
du -sh /projects/u6xn/hf-cache 2>/dev/null || true

echo "=== staged results so far ==="
find "$ROOT" -name "test_metrics.json" -o -name "history.json" 2>/dev/null | head -20
find "$ROOT" -name "*.out" -path "*logs*" -newermt "-1 day" 2>/dev/null | head -10
