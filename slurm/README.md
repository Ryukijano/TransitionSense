# Slurm jobs (Isambard-AI Phase 2, aarch64 GH200)

All jobs run inside the Apptainer PyTorch image and must not download anything. Stage code and
data on the login node first, then `sbatch`.

| Job | Purpose | Time |
| --- | --- | --- |
| `train_baseline.sbatch` | 1-D ResNet or patch-Transformer baseline on precomputed windows | 1 h |
| `train_opentslm.sbatch` | OpenTSLM-SP fine-tune (Gemma backbone, optional LoRA) + test generation | 2.5 h |

## Staging layout

`train_baseline.sbatch` assumes the TransitionSense project files sit directly under
`/projects/u6xn/transitionsense/` (`src/`, `data/windows/`, `results/`). `train_opentslm.sbatch`
overrides its roots via env vars if the layout differs:

- `TS_ROOT` — TransitionSense project root (default `/projects/u6xn/transitionsense`)
- `OTSLM_SRC` — OpenTSLM `src/` dir containing `opentslm/` (default `/projects/u6xn/OpenTSLM/src`)
- `VENV_PY` — python with torch/transformers/peft (default `$TS_ROOT/venv/bin/python3`,
  falls back to the SIF image's `python3` when absent)
- `DATA_DIR` — directory holding `windows_{train,validation,test}.npz`
  (default `$TS_ROOT/data/windows`)
- `LLM_ID`, `EPOCHS`, `BATCH`, `EVAL_BATCH`, `LORA` (0/1), `EXTRA_ARGS`, `OUT_NAME`

## Examples

```bash
# baselines
sbatch --export=ALL,MODEL=resnet,EPOCHS=40 slurm/train_baseline.sbatch
sbatch --export=ALL,MODEL=transformer,EPOCHS=40 slurm/train_baseline.sbatch

# OpenTSLM smoke (512 train windows, 1 epoch) then full runs
sbatch --export=ALL,LORA=1,EPOCHS=1,EXTRA_ARGS="--max-train-samples 512" slurm/train_opentslm.sbatch
sbatch --export=ALL,LORA=0,EPOCHS=5 slurm/train_opentslm.sbatch
sbatch --export=ALL,LORA=1,EPOCHS=5 slurm/train_opentslm.sbatch
```

The OpenTSLM job needs `google/gemma-3-270m` pre-cached under `/projects/u6xn/hf-cache`
(`HF_HUB_OFFLINE=1` is set; jobs cannot download). Check with
`scripts/isambard_status.sh` on a login node.
