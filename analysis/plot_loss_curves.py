"""Parse BrainDyn training stdout logs and plot train/val loss curves."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Captures fold, epoch, train metrics, and (optionally) val metrics on one line.
LINE_RE = re.compile(
    r"Fold (?P<fold>\d+)/\d+ Epoch (?P<epoch>\d+) \| "
    r"train total=(?P<train_total>[\d.eE+-]+) mse=(?P<train_mse>[\d.eE+-]+) "
    r"mae=(?P<train_mae>[\d.eE+-]+) pcc=(?P<train_pcc>[\d.eE+-]+) scc=(?P<train_scc>[\d.eE+-]+)"
    r"(?: \| val total=(?P<val_total>[\d.eE+-]+) mse=(?P<val_mse>[\d.eE+-]+) "
    r"mae=(?P<val_mae>[\d.eE+-]+) pcc=(?P<val_pcc>[\d.eE+-]+) scc=(?P<val_scc>[\d.eE+-]+))?"
)


def parse_log(path: Path) -> dict[int, dict[str, list]]:
    """Returns {fold_idx: {"epoch": [...], "train_total": [...], "val_total": [...], ...}}"""
    per_fold: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            fold = int(d["fold"])
            per_fold[fold]["epoch"].append(int(d["epoch"]))
            for key in (
                "train_total",
                "train_mse",
                "train_mae",
                "train_pcc",
                "train_scc",
                "val_total",
                "val_mse",
                "val_mae",
                "val_pcc",
                "val_scc",
            ):
                val = d[key]
                per_fold[fold][key].append(float(val) if val is not None else np.nan)
    if not per_fold:
        raise ValueError(
            f"No matching log lines found in {path}. Check the file actually contains "
            "main.py's 'Fold X/Y Epoch ZZZ | train total=...' lines."
        )
    return per_fold


def plot_runs(runs: list[tuple[Path, str]], metric: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for (path, label), color in zip(runs, colors):
        per_fold = parse_log(path)
        for split, ax in zip(("train", "val"), axes):
            key = f"{split}_{metric}"
            fold_curves = []
            for fold, data in sorted(per_fold.items()):
                vals = data[key]
                if all(np.isnan(v) for v in vals):
                    continue  # e.g. val skipped via --val_every
                fold_curves.append(vals)
            if fold_curves:
                # stack folds onto a shared epoch axis (assumes aligned epochs)
                max_len = max(len(v) for v in fold_curves)
                stacked = np.full((len(fold_curves), max_len), np.nan)
                for i, v in enumerate(fold_curves):
                    stacked[i, : len(v)] = v
                mean_curve = np.nanmean(stacked, axis=0)
                std_curve = np.nanstd(stacked, axis=0)
                epochs = np.arange(1, max_len + 1)
                # dominant line: mean across folds
                ax.plot(epochs, mean_curve, color=color, linewidth=2.5, label=label)
                # error band: same color, more transparent (mean +/- 1 std across folds)
                ax.fill_between(
                    epochs,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            ax.set_title(f"{split} {metric}")
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)

    axes[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


def parse_run_arg(arg: str) -> tuple[Path, str]:
    if ":" in arg and not arg.startswith(("/", "./", "../")):
        path_str, label = arg.rsplit(":", 1)
    elif arg.count(":") >= 1 and Path(arg.split(":")[0]).suffix:
        path_str, label = arg.rsplit(":", 1)
    else:
        path_str, label = arg, Path(arg).stem
    return Path(path_str), label


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", help="PATH[:LABEL] for each run's stdout log")
    ap.add_argument(
        "--metric", default="total", choices=["total", "mse", "mae", "pcc", "scc"]
    )
    ap.add_argument("--out", default="loss_curves.png")
    args = ap.parse_args()

    runs = [parse_run_arg(a) for a in args.logs]
    plot_runs(runs, args.metric, Path(args.out))


if __name__ == "__main__":
    main()
