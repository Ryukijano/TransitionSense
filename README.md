# TransitionSense

**Does temporal-language-model fine-tuning improve postural-transition recognition over conventional
depth models, on subjects the model has never seen?**

TransitionSense studies human activity and postural-transition recognition from raw inertial
signals on the UCI HAPT dataset (30 subjects, 50 Hz accelerometer + gyroscope, 12 classes:
6 activities + 6 postural transitions). All evaluation is **subject-disjoint**: the official UCI
test subjects form the test split, four deterministic train subjects form validation, and the
remaining seventeen train subjects form the training split.

Built for the [TimeNet Online Hackathon 2026](https://challenge.timenet.ai/).

## TimeNet integration

The dataset ships as a TimeNet connector, `uci/hapt` (PR: *pending*), which converts the pinned
UCI release into TimeF: 61 records (one per continuous trial), six signals per record
(accelerometer x/y/z in g, gyroscope x/y/z in rad/s at 50 Hz), and 1,214 scoped
`ClassificationTask`s carrying the official subject split plus a deterministic validation
carve-out. This repository consumes the **same pinned release** (SHA-256 verified) for window
extraction; `python -m transitionsense.prepare --timef-registry <registry>` cross-checks every
window's centre label and split against the built TimeF tasks.

## Pipeline

```
data/hapt (raw UCI release)
   └─ transitionsense.prepare ──► data/windows/windows_{train,validation,test}.npz
        128-sample windows, stride 64, centre-sample labelling, subject-disjoint
   ├─ transitionsense.train        ──► 1-D ResNet / patch Transformer baselines
   ├─ transitionsense.train_opentslm ──► OpenTSLM-SP (Gemma 3 270M + LoRA) fine-tune
   └─ transitionsense.visualize    ──► figures from saved JSON metrics
```

## Results

All numbers below are read from `results/**/test_metrics.json` / `opentslm_test_metrics.json`;
nothing is estimated.

| Model | macro-F1 | transition macro-F1 | accuracy | status |
| --- | --- | --- | --- | --- |
| ResNet1D (242.7k params, GH200) | 0.7695 | 0.6257 | 0.8846 | done |
| PatchTransformer1D (540.0k params, GH200) | 0.7632 | 0.6328 | 0.8655 | done |
| OpenTSLM-SP gemma-3-270m (frozen LLM) | _pending_ | _pending_ | _pending_ | training |
| OpenTSLM-SP gemma-3-270m + LoRA | _pending_ | _pending_ | _pending_ | training |

Baselines trained on Isambard-AI GH200 (`slurm/train_baseline.sbatch`, jobs 8744269/8744270);
local CPU replication in `results/local/` gives macro-F1 0.7768 / 0.7528 respectively.
OpenTSLM runs: jobs 8744578 (LoRA) / 8744579 (frozen).

Figures: `results/figures/` (confusion matrices, per-class F1, model comparison, example
transition windows).

## Reproduce

```bash
uv sync
python -m transitionsense.prepare --hapt-root ../data/hapt --output-dir data/windows
python -m transitionsense.train --data-dir data/windows --model resnet --output-dir results/resnet
python -m transitionsense.train --data-dir data/windows --model transformer --output-dir results/transformer
python -m transitionsense.visualize --results-dir results --hapt-root ../data/hapt
```

The OpenTSLM fine-tune runs on Isambard-AI GH200 via `slurm/train_opentslm.sbatch`
(see `slurm/README.md`).

## Layout

- `src/transitionsense/` — data loading, windowing, metrics, models, trainers, figures
- `slurm/` — Isambard-AI job scripts
- `results/` — saved metrics JSON, predictions, figures (numbers quoted above come from here)
- `docs/` — submission drafts

## License / data

UCI HAPT v2.1 is distributed for non-commercial research use; see the dataset card in the
`uci/hapt` TimeNet connector.
