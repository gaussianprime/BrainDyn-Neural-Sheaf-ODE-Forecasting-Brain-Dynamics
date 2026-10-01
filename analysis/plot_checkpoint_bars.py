"""Bar chart comparing one metric across training runs / checkpoints."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ["total", "mse", "mae", "pcc", "scc", "dtw"]

# Header that opens a summary block, capturing the split name (Val / Test).
_HEADER_RE = re.compile(r"CV\s+(?P<split>Val|Test)\s+Summary", re.IGNORECASE)

# One metric line, e.g. "  MSE   : 0.001234 ± 0.000123" (std optional).
_SEP = r"(?:±|\+/-|\+-)"
_NUM = r"[-+]?[\d.]+(?:[eE][-+]?\d+)?"


def _metric_line_re(metric: str) -> re.Pattern:
    return re.compile(
        rf"^\s*{metric}\s*:\s*(?P<mean>{_NUM})(?:\s*{_SEP}\s*(?P<std>{_NUM}))?",
        re.IGNORECASE,
    )


def parse_summary(path: Path, metric: str, split: str) -> tuple[float, float]:
    """Return (mean, std) for `metric` under the `split` summary block.

    std is NaN when the log reports a single value with no ± term.
    """
    want_split = split.lower()
    line_re = _metric_line_re(metric)
    current_split: str | None = None
    lines = Path(path).read_text(errors="replace").splitlines()

    for line in lines:
        h = _HEADER_RE.search(line)
        if h:
            current_split = h.group("split").lower()
            continue
        if current_split != want_split:
            continue
        m = line_re.match(line)
        if m:
            mean = float(m.group("mean"))
            std = float(m.group("std")) if m.group("std") is not None else float("nan")
            return mean, std

    raise ValueError(
        f"Could not find '{metric}' under a 'CV {split.capitalize()} Summary' block "
        f"in {path}. Confirm the run finished and printed its summary."
    )


def parse_run_arg(arg: str) -> tuple[Path, str]:
    # Split on the last ':' only when it isn't part of a Windows drive / path.
    if ":" in arg:
        head, tail = arg.rsplit(":", 1)
        # A drive letter like C:\... leaves head length 1 -> treat as no label.
        if (
            head
            and len(head) > 1
            and not tail.strip().endswith((".out", ".log", ".txt"))
        ):
            return Path(head), tail
    return Path(arg), Path(arg).stem


def plot_bars(
    runs: list[tuple[Path, str]], metric: str, split: str, out_path: Path
) -> None:
    labels, means, stds = [], [], []
    for path, label in runs:
        mean, std = parse_summary(path, metric, split)
        labels.append(label)
        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)
    x = np.arange(len(runs))
    # Only draw error bars where a std was actually reported.
    yerr = np.where(np.isnan(stds), 0.0, stds)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(runs)), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    # Value labels above each bar.
    for bar, mean, std in zip(bars, means, stds):
        txt = f"{mean:.4f}" if abs(mean) < 100 else f"{mean:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0 if np.isnan(std) else std),
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{split.capitalize()} {metric.upper()} across checkpoints")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("logs", nargs="+", help="PATH[:LABEL] for each run's .out log")
    ap.add_argument("--metric", default="mse", choices=METRICS)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="checkpoint_bars.png")
    args = ap.parse_args()

    runs = [parse_run_arg(a) for a in args.logs]
    plot_bars(runs, args.metric, args.split, Path(args.out))


if __name__ == "__main__":
    main()
