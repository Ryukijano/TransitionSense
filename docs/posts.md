# Social posts — drafts (metrics verified against saved JSON)

Repo: https://github.com/Ryukijano/TransitionSense
Showcase: https://ryukijano.github.io/TransitionSense/
TimeNet PR: https://github.com/OpenTSLM/TimeNet/pull/850

## X post

```
TransitionSense for @TimeNetAI hackathon: does a temporal language model
actually help with postural TRANSITIONS, or do CNNs already win on unseen
subjects?

We added uci/hapt to TimeNet (61 trials, 6 IMU channels, 1,214 scoped tasks,
subject-disjoint splits) and benchmarked ResNet1D, a patch Transformer, and
OpenTSLM-SP fine-tuned on Gemma 3 270M (+LoRA) — all on Isambard GH200s.

Test macro-F1 / transition macro-F1 (9 unseen subjects, 3,718 windows):
- ResNet1D 242k: 0.770 / 0.626
- Patch Transformer 540k: 0.763 / 0.633
- OpenTSLM-SP frozen: 0.064 / 0.029
- OpenTSLM-SP + LoRA (best): 0.190 / 0.081

Honest result: the TSLM loses by ~5x. Code + connector PR + figures below.
```

## LinkedIn post

```
I built TransitionSense for the TimeNet Online Hackathon 2026: a controlled
study of whether temporal-language-model fine-tuning improves recognition of
brief postural transitions — the events that matter for fall detection and
rehabilitation — compared with conventional deep models, evaluated strictly
on subjects the models have never seen.

What shipped:
• A new uci/hapt dataset connector for TimeNet (PR #850): 61 continuous
  trials as TimeF records, six IMU channels at 50 Hz, 1,214 scoped
  classification tasks, subject-disjoint train/validation/test splits.
• A reproducible windowing pipeline (128-sample windows, centre-sample
  labelling) cross-checked against the TimeF tasks — 12,757 windows, zero
  subject overlap, zero label mismatches.
• Supervised baselines (1-D ResNet, patch Transformer) and an OpenTSLM-SP
  fine-tune of Gemma 3 270M with LoRA on Isambard-AI GH200 hardware.

Headline numbers (test split, 9 unseen subjects, 3,718 windows):
• macro-F1: ResNet 0.770 / Transformer 0.763 / OpenTSLM-SP 0.064 / +LoRA 0.190
• transition-only macro-F1: 0.626 / 0.633 / 0.029 / 0.081

The honest finding: at this scale and budget, the temporal-language-model
path does not beat small supervised CNNs — transitions remain the hardest
class family for every model (best transition F1 0.63 vs ~0.90 for
steady-state activities). LoRA improves the TSLM ~2.5x over the frozen
adapter but the gap stays large; generations also degenerate after the
first label token, pointing at the need for a discriminative readout or
more adaptation data.

Repo, connector PR, and live figures: https://github.com/Ryukijano/TransitionSense
```

## Discord post (project channel)

```
TransitionSense — UCI HAPT postural-transition recognition, subject-disjoint eval.

• TimeNet connector: uci/hapt — PR https://github.com/OpenTSLM/TimeNet/pull/850
• Baselines (GH200): ResNet1D macro-F1 0.770 / transition-F1 0.626; patch Transformer 0.763 / 0.633
• OpenTSLM-SP Gemma-3-270M: frozen 0.064 / 0.029; +LoRA (gentle LR) 0.190 / 0.081 (parse-rate ~1.0)
• Takeaway: TSLM fine-tune loses to small CNNs at this budget; transitions are the hard classes for everyone
• Repo + figures: https://github.com/Ryukijano/TransitionSense — demo: https://ryukijano.github.io/TransitionSense/
```
