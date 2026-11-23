#!/bin/bash
#SBATCH --job-name=ts-baseline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# TransitionSense supervised baseline on Isambard-AI (aarch64 GH200).
# Remote layout: /projects/u6xn/transitionsense/TransitionSense/{src,data,results,slurm}
# Submit from the TransitionSense dir:
#   sbatch --export=ALL,MODEL=resnet,EPOCHS=40 slurm/train_baseline.sbatch
#   sbatch --export=ALL,MODEL=transformer,EPOCHS=40 slurm/train_baseline.sbatch
set -euo pipefail

ROOT=/projects/u6xn/transitionsense/TransitionSense
SIF=/projects/u6xn/sif-images/pytorch_26.09-py3.sif
MODEL=${MODEL:-resnet}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-256}
LR=${LR:-1e-3}
SEED=${SEED:-0}
TIME_LIMIT=${TIME_LIMIT:-00:55:00}

mkdir -p "$ROOT/logs"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

apptainer exec --nv --bind /projects/u6xn:/projects/u6xn "$SIF" \
  python -m transitionsense.train \
    --data-dir "$ROOT/data/windows" \
    --model "$MODEL" \
    --output-dir "$ROOT/results/$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --seed "$SEED" \
    --device cuda
