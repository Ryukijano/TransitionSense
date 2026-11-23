# TransitionSense — submission drafts

All bracketed numbers are filled from `results/**/test_metrics.json` and
`results/opentslm-*/opentslm_test_metrics.json` only.

## One-line pitch

A `uci/hapt` TimeNet connector plus subject-disjoint benchmarks asking whether a
temporal-language-model fine-tune (OpenTSLM-SP, Gemma 3 270M + LoRA) recognises brief
postural transitions better than 1-D CNN and patch-Transformer baselines on unseen subjects.

## Dataset PR description (TimeNet)

**`uci/hapt` — Smartphone-Based Recognition of Human Activities and Postural Transitions (UCI HAPT v2.1)**

- 61 continuous trials, one `Record` each; two `Source`s (Accelerometer, Gyroscope) with three
  signals apiece at 50 Hz, float64, units g / rad/s.
- 1,214 scoped `ClassificationTask`s from `RawData/labels.txt` (1-indexed inclusive sample bounds
  converted to half-open `TimeInterval`s).
- Subject-disjoint splits: official UCI test subjects → `Split.TEST`; official train subjects with
  id divisible by 5 → `Split.VALIDATION`; remaining 17 train subjects → `Split.TRAIN`.
- Pinned archive + SHA-256; `license: OTHER` with UCI license URL (non-commercial research use).
- Fixture tests cover conversion, interval math, metadata, label validation, and a storage
  round-trip; validated end-to-end with `timenet-build` and `TimeNet.open("uci/hapt")`.

## Results table (fill from JSON)

| Model | macro-F1 | transition macro-F1 | parse rate | notes |
| --- | --- | --- | --- | --- |
| ResNet1D (242.7k params) | 0.7695 | 0.6257 | n/a | GH200 job 8744269 |
| PatchTransformer1D (540.0k params) | 0.7632 | 0.6328 | n/a | GH200 job 8744270 |
| OpenTSLM-SP gemma-3-270m | 0.0642 | 0.0293 | 0.705 | frozen LLM, job 8744579 |
| OpenTSLM-SP gemma-3-270m + LoRA r=16 | 0.1602 | 0.0814 | 0.998 | LoRA, job 8744578 |
| OpenTSLM-SP gemma-3-270m + LoRA (lr 3e-5) | 0.1898 | 0.0806 | 0.985 | gentler LoRA, job 8744973 |

## Key findings

- **Transitions are the bottleneck for every model.** Best transition macro-F1 is 0.6328
  (patch Transformer) vs ~0.90 for steady-state activities — brief postural transitions are
  genuinely under-recognized on unseen subjects.
- **The compact 1-D ResNet beats the larger patch Transformer** on subject-disjoint test
  (macro-F1 0.7695 vs 0.7632; 242.7k vs 540.0k params) — short IMU windows favor simple
  temporal convolutions.
- **OpenTSLM-SP fine-tuning does not beat supervised baselines at this scale.** Best TSLM
  variant (Gemma 3 270M + LoRA, gentler LR) reaches macro-F1 0.1898 / transition-F1 0.0806
  vs 0.7695 / 0.6257 for ResNet1D — roughly a 4x gap. Generations are degenerate after
  the first label token (label-salad), indicating the soft-prompt path needs far more
  data or a discriminative readout head to be competitive.
- **LoRA clearly beats the frozen-adapter variant** (0.1898 vs 0.0642 macro-F1 at best),
  so backbone adaptation matters — a gentler LoRA LR (3e-5) further improves over 1e-4 —
  but is not sufficient at this budget.

## Links

- Repository: {repo_url}
- TimeNet connector PR: {pr_url}
- Demo/showcase: {demo_url}
