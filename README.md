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

| Model | macro-F1 | transition macro-F1 | accuracy | parse rate | status |
| --- | --- | --- | --- | --- | --- |
| ResNet1D (242.7k params) | 0.7695 | 0.6257 | 0.8846 | n/a | done (GH200) |
| PatchTransformer1D (540.0k params) | 0.7632 | 0.6328 | 0.8655 | n/a | done (GH200) |
| OpenTSLM-SP gemma-3-270m (frozen LLM) | 0.0642 | 0.0293 | 0.0680 | 0.705 | done (GH200) |
| OpenTSLM-SP gemma-3-270m + LoRA r=16 | 0.1602 | 0.0814 | 0.1872 | 0.998 | done (GH200) |
| OpenTSLM-SP gemma-3-270m + LoRA (gentler LR) | 0.1898 | 0.0806 | 0.3521 | 0.985 | done (GH200) |

Baselines: Isambard-AI GH200 jobs 8744269/8744270; OpenTSLM: jobs 8744578 (LoRA) /
8744579 (frozen) / 8744973 (LoRA, lr 3e-5, early stop epoch 6). Generations parsed by
first label mention (`parse_rule` in each metrics JSON). Local CPU replication of the
baselines in `results/local/` gives macro-F1 0.7768 / 0.7528 — same ordering.

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
