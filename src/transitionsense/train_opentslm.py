"""Fine-tune OpenTSLM-SP on HAPT windows: frozen Gemma backbone + trainable encoder/projector,
optionally with LoRA on the LLM.

Example (Isambard-AI, inside the PyTorch container):
    python -m transitionsense.train_opentslm \
        --llm-id google/gemma-3-270m --data-dir data/windows \
        --output-dir results/opentslm-gemma270m --epochs 5 --batch-size 8 --lora
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is only guaranteed on the cluster env; degrade gracefully

    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable

from transitionsense.hapt_qa import load_prompt_datasets, normalize_label
from transitionsense.labels import ACTIVITY_NAMES
from transitionsense.metrics import REJECT_INDEX, evaluate_with_reject, write_metrics

PATCH_SIZE = 4
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.03
WEIGHT_DECAY = 1e-2
ANSWER_RE = re.compile(r"Answer:\s*([A-Za-z_]+)")


def _collate(batch, *, patch_size: int = PATCH_SIZE):
    """Lazy import of the OpenTSLM collate to keep this module importable without opentslm."""
    from opentslm.time_series_datasets.util import (  # noqa: PLC0415
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=patch_size)


def _build_optimizer(model, args):
    """Parameter groups matching the OpenTSLM curriculum defaults."""
    from torch.optim import AdamW  # noqa: PLC0415

    groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": args.lr_encoder,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": list(model.projector.parameters()),
            "lr": args.lr_projector,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    if getattr(model, "lora_enabled", False):
        lora_params = model.get_lora_parameters()
        if not lora_params:
            raise RuntimeError("LoRA enabled but no trainable LoRA parameters found")
        groups.append({"params": lora_params, "lr": args.lr_lora, "weight_decay": WEIGHT_DECAY})
    return AdamW(groups)


def _parameter_counts(model) -> dict[str, int]:
    """Count LLM, adapter, and trainable parameters for the metrics payload."""
    llm_params = sum(p.numel() for p in model.llm.parameters())
    adapter_params = sum(p.numel() for p in model.encoder.parameters()) + sum(
        p.numel() for p in model.projector.parameters()
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "llm_params": llm_params,
        "adapter_params": adapter_params,
        "trainable_params": trainable,
    }


def _parse_answer(text: str) -> str:
    """Extract the label from generated text.

    Trained answers are bare labels, so generations usually contain no ``Answer:`` span;
    fall back to normalizing the whole decoded text (substring match against the vocabulary).
    """
    matches = ANSWER_RE.findall(text or "")
    if matches:
        return normalize_label(matches[-1])
    return normalize_label(text or "")


@torch.no_grad()
def _generate_predictions(model, loader, device, max_new_tokens: int) -> tuple[list[str], list[dict]]:
    """Generate answers for every sample; return ``(generated_texts, row_meta)``."""
    model.eval()
    generated: list[str] = []
    meta: list[dict] = []
    for batch in tqdm(loader, desc="generating", leave=False):
        texts = model.generate(batch, max_new_tokens=max_new_tokens)
        generated.extend(texts)
        for sample in batch:
            meta.append(
                {
                    "label": sample["label"],
                    "label_index": int(sample["label_index"]),
                    "subject_id": int(sample["subject_id"]),
                    "experiment_id": int(sample["experiment_id"]),
                    "window_start": int(sample["window_start"]),
                }
            )
    return generated, meta


def _write_predictions_jsonl(path: Path, generated: list[str], meta: list[dict]) -> None:
    """Write per-sample generation records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text, row in zip(generated, meta):
            handle.write(
                json.dumps({"generated": text, "gold": row["label"], **{
                    k: row[k] for k in ("label_index", "subject_id", "experiment_id", "window_start")
                }}, ensure_ascii=False)
                + "\n"
            )


def _evaluate_test(model, test_loader, device, output_dir: Path, args, extra: dict) -> dict:
    """Generate on the test split and write metrics + prediction artifacts."""
    generated, meta = _generate_predictions(model, test_loader, device, args.max_new_tokens)
    label_to_index = {name: index for index, name in enumerate(ACTIVITY_NAMES)}
    y_true = np.asarray([row["label_index"] for row in meta], dtype=np.int64)
    parsed = [_parse_answer(text) for text in generated]
    y_pred = np.asarray(
        [label_to_index[name] if name else REJECT_INDEX for name in parsed],
        dtype=np.int64,
    )
    payload = evaluate_with_reject(y_true, y_pred, extra=extra)
    write_metrics(output_dir / "opentslm_test_metrics.json", payload)
    _write_predictions_jsonl(output_dir / "predictions.jsonl", generated, meta)
    np.savez(
        output_dir / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        subject_id=np.asarray([row["subject_id"] for row in meta], dtype=np.int64),
        experiment_id=np.asarray([row["experiment_id"] for row in meta], dtype=np.int64),
        window_start=np.asarray([row["window_start"] for row in meta], dtype=np.int64),
    )
    print(
        f"test macro-F1 {payload['macro_f1']:.4f} transition-F1 {payload['transition_macro_f1']:.4f} "
        f"parse-rate {payload['parse_rate']:.4f}",
        flush=True,
    )
    return payload


def train(args: argparse.Namespace) -> dict:
    """Run the OpenTSLM-SP fine-tuning loop and test evaluation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The test metrics payload.
    """
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = OpenTSLMSP(llm_id=args.llm_id, device=str(device))
    if args.lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    datasets = load_prompt_datasets(args.data_dir, eos_token=model.get_eos_token())
    train_dataset = datasets["train"]
    if args.max_train_samples:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    val_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )

    if args.eval_only:
        checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"eval-only mode needs a checkpoint; missing {checkpoint}")
        model.load_from_file(str(checkpoint))
        extra = {
            "model": f"opentslm-sp/{args.llm_id}",
            "lora": bool(args.lora),
            "lora_r": args.lora_r if args.lora else None,
            **_parameter_counts(model),
            "eval_only": True,
            "checkpoint": str(checkpoint),
        }
        return _evaluate_test(model, test_loader, device, output_dir, args, extra)

    optimizer = _build_optimizer(model, args)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        train_loss = running / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += float(model.compute_loss(batch).detach())
        val_loss /= max(1, len(val_loader))

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}
        )
        print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}", flush=True)

        if val_loss + 1e-4 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            model.store_to_file(str(output_dir / "best.pt"))
            print(f"  saved best (epoch {epoch}, val {val_loss:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    checkpoint = output_dir / "best.pt"
    if not checkpoint.is_file():
        model.store_to_file(str(checkpoint))
    else:
        model.load_from_file(str(checkpoint))

    extra = {
        "model": f"opentslm-sp/{args.llm_id}",
        "lora": bool(args.lora),
        "lora_r": args.lora_r if args.lora else None,
        **_parameter_counts(model),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "batch_size": args.batch_size,
        "seed_note": "single run; seed fixed by torch default",
        "train_seconds": time.time() - t0,
    }
    return _evaluate_test(model, test_loader, device, output_dir, args, extra)


def main() -> None:
    """Parse arguments and run fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-id", default="google/gemma-3-270m")
    parser.add_argument("--data-dir", default="data/windows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr-encoder", type=float, default=2e-4)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate a checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for --eval-only (default: <output-dir>/best.pt).")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
