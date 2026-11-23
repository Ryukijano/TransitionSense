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
| OpenTSLM-SP gemma-3-270m | {opentslm_macro_f1} | {opentslm_transition_f1} | {opentslm_parse} | frozen LLM, job 8744579 |
| OpenTSLM-SP gemma-3-270m + LoRA | {opentslm_lora_macro_f1} | {opentslm_lora_transition_f1} | {opentslm_lora_parse} | LoRA r=16, job 8744578 |

## Key findings (fill after results)

- {finding_1}
- {finding_2}
- {finding_3}

## Links

- Repository: {repo_url}
- TimeNet connector PR: {pr_url}
- Demo/showcase: {demo_url}
