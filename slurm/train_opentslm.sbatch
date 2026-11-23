#!/bin/bash
#SBATCH --job-name=ts-opentslm
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# TransitionSense OpenTSLM-SP fine-tune on Isambard-AI (aarch64 GH200).
# Requires: TransitionSense sources + window NPZs staged, OpenTSLM sources staged,
#           HF weights pre-cached under /projects/u6xn/hf-cache (no downloads in-job).
#
# Layout is env-overridable; defaults assume the flat staging used by
# train_baseline.sbatch (project files directly under /projects/u6xn/transitionsense).
#   TS_ROOT   = TransitionSense project root (contains src/, data/, results/)
#   OTSLM_SRC = OpenTSLM src directory (contains opentslm/)
#   VENV_PY   = python with torch+transformers+peft (system-site-packages venv);
#               empty = use the SIF image's own python3
#
# Submit examples:
#   sbatch --export=ALL,LLM_ID=google/gemma-3-270m,LORA=1,EPOCHS=5 slurm/train_opentslm.sbatch
#   sbatch --export=ALL,LORA=1,EPOCHS=1,EXTRA_ARGS="--max-train-samples 512" \
#       slurm/train_opentslm.sbatch   # smoke run
set -euo pipefail

ROOT=/projects/u6xn/transitionsense
SIF=/projects/u6xn/sif-images/pytorch_26.09-py3.sif
TS_ROOT=${TS_ROOT:-$ROOT/TransitionSense}
OTSLM_SRC=${OTSLM_SRC:-$ROOT/OpenTSLM/src}
VENV_PY=${VENV_PY:-$ROOT/venv/bin/python3}
LLM_ID=${LLM_ID:-google/gemma-3-270m}
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-8}
EVAL_BATCH=${EVAL_BATCH:-8}
LORA_FLAG=""
if [ "${LORA:-0}" = "1" ]; then LORA_FLAG="--lora"; fi
# EXTRA_ARGS: extra CLI args, e.g. EXTRA_ARGS="--max-train-samples 512 --epochs 1"
EXTRA_ARGS=${EXTRA_ARGS:-}
DATA_DIR=${DATA_DIR:-$TS_ROOT/data/windows}
OUT_NAME=${OUT_NAME:-opentslm-$(basename "$LLM_ID" | tr './' '__')$([ "$LORA" = "1" ] && echo "-lora")}

mkdir -p "$TS_ROOT/logs"
cd "$TS_ROOT"

export HF_HOME=/projects/u6xn/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TS_ROOT/src:$OTSLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="$VENV_PY"
# -e/-x follow the symlink chain and fail on the host (venv python -> /usr/bin/python
# only resolves inside the container), so accept the path when it is a symlink too.
if [ ! -e "$PYTHON_BIN" ] && [ ! -L "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi

apptainer exec --nv --bind /projects/u6xn:/projects/u6xn "$SIF" "$PYTHON_BIN" \
  -m transitionsense.train_opentslm \
    --llm-id "$LLM_ID" \
    --data-dir "$DATA_DIR" \
    --output-dir "$TS_ROOT/results/$OUT_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --eval-batch-size "$EVAL_BATCH" \
    $LORA_FLAG \
    $EXTRA_ARGS \
    --device cuda
