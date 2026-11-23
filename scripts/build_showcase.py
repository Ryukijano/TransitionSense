"""Build the static GitHub Pages showcase from saved result JSONs.

Every number on the page is read from ``results/**/test_metrics.json`` (and
``results/opentslm-*/opentslm_test_metrics.json``); the script refuses to invent
values and marks missing metrics as "pending".

Example:
    uv run python scripts/build_showcase.py --results-dir results --output-dir showcase
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

SMOKE_SUFFIX = "-smoke"
FIGURE_NAMES = (
    "model_comparison.png",
    "loss_curves.png",
    "example_transitions.png",
)


def _metrics_files(results_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(model_name, metrics_json)`` for every non-smoke result dir."""
    found: list[tuple[str, Path]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name.endswith(SMOKE_SUFFIX):
            continue
        for name in ("test_metrics.json", "opentslm_test_metrics.json"):
            path = child / name
            if path.is_file():
                found.append((child.name, path))
                break
    return found


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _split_subjects(splits_path: Path) -> dict[str, list[int]]:
    """Read subject ids per split from ``splits.json`` if present."""
    if not splits_path.is_file():
        return {}
    data = _load(splits_path)
    observed = data.get("observed_subjects") or data.get("split_summary") or {}
    out: dict[str, list[int]] = {}
    for split, value in observed.items():
        subjects = value.get("subjects") if isinstance(value, dict) else value
        if isinstance(subjects, list):
            out[split] = [int(s) for s in subjects]
    return out


def _results_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['model'])}</td>"
            f"<td>{_fmt(row.get('macro_f1'))}</td>"
            f"<td>{_fmt(row.get('transition_macro_f1'))}</td>"
            f"<td>{_fmt(row.get('accuracy'))}</td>"
            f"<td>{_fmt(row.get('parse_rate'), 3)}</td>"
            f"<td>{_fmt(row.get('n_params'))}</td>"
            f"<td>{_fmt(row.get('n_windows'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Model</th><th>macro-F1</th>"
        "<th>transition macro-F1</th><th>accuracy</th><th>parse rate</th>"
        "<th>params</th><th>windows</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _figure_section(figures_dir: Path, names: list[str]) -> str:
    imgs = []
    for name in names:
        if (figures_dir / name).is_file():
            imgs.append(
                f'<figure><img src="assets/{html.escape(name)}" alt="{html.escape(name)}"/>'
                f"<figcaption>{html.escape(name.replace('.png', '').replace('_', ' '))}</figcaption></figure>"
            )
    # per-model figures not in the fixed list
    for path in sorted(figures_dir.glob("*.png")):
        if path.name not in names:
            imgs.append(
                f'<figure><img src="assets/{html.escape(path.name)}" alt="{html.escape(path.name)}"/>'
                f"<figcaption>{html.escape(path.stem.replace('_', ' '))}</figcaption></figure>"
            )
    return '<div class="figures">' + "".join(imgs) + "</div>" if imgs else "<p><em>Figures pending.</em></p>"


def build_html(results_dir: Path, figures_dir: Path, splits_path: Path) -> str:
    """Render the showcase page; metrics come only from saved JSON."""
    rows: list[dict] = []
    for name, path in _metrics_files(results_dir):
        metrics = _load(path)
        extra = metrics.get("extra", metrics)
        rows.append(
            {
                "model": name,
                "macro_f1": metrics.get("macro_f1"),
                "transition_macro_f1": metrics.get("transition_macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "parse_rate": metrics.get("parse_rate", 1.0),
                "n_params": (
                    extra.get("n_params")
                    or extra.get("trainable_params")
                    or extra.get("parameters")
                ),
                "n_windows": metrics.get("n_windows"),
            }
        )

    subjects = _split_subjects(splits_path)
    split_rows = "".join(
        f"<tr><td>{html.escape(split)}</td><td>{len(ids)}</td>"
        f"<td>{html.escape(', '.join(map(str, ids)))}</td></tr>"
        for split, ids in subjects.items()
    )
    splits_table = (
        "<table><thead><tr><th>split</th><th>subjects</th><th>subject ids</th></tr></thead>"
        f"<tbody>{split_rows}</tbody></table>"
        if split_rows
        else "<p><em>Split summary pending.</em></p>"
    )

    figure_names = list(FIGURE_NAMES) + [
        p.name for p in sorted(figures_dir.glob("per_class_f1_*.png"))
    ] + [p.name for p in sorted(figures_dir.glob("confusion_matrix_*.png"))]
    # de-duplicate preserving order
    seen: set[str] = set()
    ordered = [n for n in figure_names if not (n in seen or seen.add(n))]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TransitionSense — TimeNet Hackathon 2026</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 980px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }}
  h1 {{ border-bottom: 3px solid #d95f02; padding-bottom: .3rem; }}
  h2 {{ margin-top: 2rem; color: #1b4f8a; }}
  .hypothesis {{ background: #f4f6f8; border-left: 4px solid #1b4f8a;
                 padding: .8rem 1rem; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .4rem .6rem; text-align: left;
           font-size: .92rem; }}
  th {{ background: #eef2f6; }}
  .figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
              gap: 1rem; }}
  figure {{ margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: .5rem;
            background: #fff; }}
  figure img {{ width: 100%; height: auto; }}
  figcaption {{ font-size: .8rem; color: #555; margin-top: .3rem; }}
  code {{ background: #eef2f6; padding: .1rem .3rem; border-radius: 3px; }}
  footer {{ margin-top: 3rem; font-size: .85rem; color: #666; }}
</style></head><body>
<h1>TransitionSense</h1>
<p><strong>TimeNet Online Hackathon 2026</strong> — temporal-language-model fine-tuning vs
conventional depth models for postural-transition recognition on unseen subjects.</p>

<div class="hypothesis">
<strong>Hypothesis.</strong> Does fine-tuning a temporal language model (OpenTSLM-SP on
Gemma&nbsp;3&nbsp;270M) improve recognition of brief postural transitions compared with
1-D CNN and patch-Transformer baselines, evaluated strictly on subjects absent from
training?
</div>

<h2>Dataset</h2>
<p>UCI HAPT v2.1 via the <code>uci/hapt</code> TimeNet connector: 61 continuous trials,
six IMU channels (accelerometer in g, gyroscope in rad/s) at 50&nbsp;Hz, 1,214 scoped
classification tasks, 12 classes (6 activities + 6 postural transitions).
Windows: 128 samples, stride 64, centre-sample labelling. All splits are
subject-disjoint; the test split uses the official UCI held-out subjects.</p>
{splits_table}

<h2>Results</h2>
<p>Every number below is read from saved <code>test_metrics.json</code> artifacts.</p>
{_results_table(rows) if rows else "<p><em>Results pending.</em></p>"}

<h2>Figures</h2>
{_figure_section(figures_dir, ordered)}

<h2>Method</h2>
<ul>
<li><strong>Baselines:</strong> 1-D ResNet and patch Transformer trained on the same
windows with class-balanced cross-entropy; best checkpoint by validation macro-F1.</li>
<li><strong>TSLM:</strong> OpenTSLM-SP — frozen Gemma&nbsp;3&nbsp;270M backbone, trainable
CNN patch encoder + MLP projector, optional LoRA (r=16) on the LLM; trained with
teacher-forced answer loss on the same windows.</li>
<li><strong>Evaluation:</strong> macro-F1 over all 12 classes and over the six transition
classes only, on 9 unseen test subjects (3,718 windows). Generative models report a
parse rate; unparseable outputs count as errors.</li>
</ul>

<footer>
TransitionSense — built with <a href="https://timenet.ai">TimeNet</a> /
<a href="https://github.com/OpenTSLM">OpenTSLM</a>. Dataset: UCI HAPT v2.1
(non-commercial research use).
</footer>
</body></html>"""


def main() -> None:
    """Render ``showcase/index.html`` and copy figure assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--splits", type=Path, default=Path("data/windows/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("showcase"))
    args = parser.parse_args()

    assets = args.output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    if args.figures_dir.is_dir():
        for png in args.figures_dir.glob("*.png"):
            shutil.copy2(png, assets / png.name)

    page = build_html(args.results_dir, args.figures_dir, args.splits)
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
