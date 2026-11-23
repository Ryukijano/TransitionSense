# Social posts — drafts (fill metrics from saved JSON only)

## X post

```
TransitionSense for @TimeNetAI hackathon: does a temporal language model
actually help with postural TRANSITIONS, or do CNNs already win on unseen
subjects?

We added uci/hapt to TimeNet (61 trials, 6 IMU channels, 1,214 scoped tasks,
subject-disjoint splits) and benchmarked ResNet1D, a patch Transformer, and
OpenTSLM-SP fine-tuned on Gemma 3 270M (+LoRA).

Transition macro-F1 on 9 unseen subjects:
- ResNet1D: {x_resnet_tf1}
- Transformer: {x_transformer_tf1}
- OpenTSLM-SP: {x_opentslm_tf1}
- OpenTSLM-SP + LoRA: {x_opentslm_lora_tf1}

Code + connector PR + figures: {repo_url}
```

## LinkedIn post

```
I built TransitionSense for the TimeNet Online Hackathon 2026: a study of
whether temporal-language-model fine-tuning improves recognition of brief
postural transitions — the events that matter for fall detection and
rehabilitation — compared with conventional deep models, evaluated strictly
on subjects the models have never seen.

What shipped:
• A new uci/hapt dataset connector for TimeNet: 61 continuous trials as
  TimeF records, six IMU channels at 50 Hz, 1,214 scoped classification
  tasks, and subject-disjoint train/validation/test splits.
• A reproducible windowing pipeline (128-sample windows, centre-sample
  labelling) cross-checked against the TimeF tasks — 12,757 windows, zero
  subject overlap, zero label mismatches.
• Supervised baselines (1-D ResNet, patch Transformer) and an OpenTSLM-SP
  fine-tune of Gemma 3 270M with LoRA on Isambard-AI GH200 hardware.

Headline numbers (test split, 9 unseen subjects):
• Overall macro-F1: ResNet {li_resnet_f1} / Transformer {li_transformer_f1} /
  OpenTSLM-SP {li_opentslm_f1} / +LoRA {li_opentslm_lora_f1}
• Transition-only macro-F1: {li_resnet_tf1} / {li_transformer_tf1} /
  {li_opentslm_tf1} / {li_opentslm_lora_tf1}

Repo, connector PR, and figures: {repo_url}
```

## Discord post (project channel)

```
TransitionSense — UCI HAPT postural-transition recognition with subject-disjoint eval.

• TimeNet connector: uci/hapt (PR {pr_status})
• Baselines: ResNet1D + patch Transformer; TSLM: OpenTSLM-SP on Gemma 3 270M (+LoRA)
• Test macro-F1 / transition macro-F1: {discord_numbers}
• Repo: {repo_url}
```
