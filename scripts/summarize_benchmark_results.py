#!/usr/bin/env python3
"""Aggregate per-fold train_nest_braindyn.py logs into a mean+-std table per dataset."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

METRIC_KEYS = ["total", "mse", "mae", "pcc", "scc", "dtw"]
N_FOLDS = 5

# (script prefix used in logs/benchmarks/<prefix>_<model>_fold<n>.log, model list)
DATASETS = {
    "nest_f01_structural_short": ["full", "nosheaf", "nolstm", "notime"],
    "nest_f01_structural_arlong": ["full", "nosheaf", "nolstm", "notime"],
}

SUMMARY_RE = re.compile(
    r"=== Benchmark Summary(?P<variant>[^=]*)===\n"
    r"(?P<body>(?:.+\n?){1,%d})" % len(METRIC_KEYS)
)
METRIC_LINE_RE = re.compile(r"^(\w+): mean=([-\d.eE]+) std=([-\d.eE]+)", re.MULTILINE)

VARIANTS = {
    "": "",
    "(perturbed horizon)": "_perturbed",
    "(target excluded)": "_excl_target",
    "(perturbed horizon, target excluded)": "_perturbed_excl_target",
}


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    """Map variant suffix -> metrics, omitting variants absent from the log."""
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out: dict[str, dict[str, float]] = {}
    for m in SUMMARY_RE.finditer(text):
        variant = m.group("variant").strip()
        if variant not in VARIANTS:
            print(f"  WARNING: unrecognized summary block {variant!r} in {path.name}")
            continue
        out[VARIANTS[variant]] = {
            k: float(v)
            for k, v, _ in (
                (mm.group(1), mm.group(2), mm.group(3))
                for mm in METRIC_LINE_RE.finditer(m.group("body"))
            )
        }
    return out


def mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var**0.5


def aggregate(log_dir: Path, prefix: str, models: list[str]):
    """Map variant suffix -> rows"""
    suffixes = list(VARIANTS.values())
    out: dict[str, list[dict]] = {s: [] for s in suffixes}
    for model in models:
        per_fold = {s: {k: [] for k in METRIC_KEYS} for s in suffixes}
        missing = []
        for fold in range(N_FOLDS):
            found = parse_log(log_dir / f"{prefix}_{model}_fold{fold}.log")
            if "" not in found:
                missing.append(fold)
                continue
            for suffix, metrics in found.items():
                for k in METRIC_KEYS:
                    if k in metrics:
                        per_fold[suffix][k].append(metrics[k])
        present = N_FOLDS - len(missing)
        base = {
            "model": model,
            "folds": f"{present}/{N_FOLDS}",
            "missing": ",".join(str(f) for f in missing) if missing else "",
        }
        for suffix in suffixes:
            if suffix and not any(per_fold[suffix].values()):
                continue
            row = dict(base)
            for k in METRIC_KEYS:
                if per_fold[suffix][k]:
                    m, s = mean_std(per_fold[suffix][k])
                    row[k] = f"{m:.4f}±{s:.4f}"
                else:
                    row[k] = "-"
            out[suffix].append(row)
    return out


def print_table(title: str, rows: list[dict]):
    if not rows:
        return
    cols = ["model", "folds"] + METRIC_KEYS + ["missing"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print(f"\n=== {title} ===")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = ["model", "folds"] + METRIC_KEYS + ["missing"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log_dir", default="logs/benchmarks")
    ap.add_argument("--out_dir", default="logs/benchmarks/summary")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {v: k for k, v in VARIANTS.items()}
    for prefix in args.datasets:
        by_variant = aggregate(log_dir, prefix, DATASETS[prefix])
        for suffix, rows in by_variant.items():
            if not rows:
                continue
            title = f"{prefix} {labels[suffix]}".strip()
            print_table(title, rows)
            write_csv(out_dir / f"{prefix}{suffix}.csv", rows)

    print(f"\nCSV files written to {out_dir}/")


if __name__ == "__main__":
    main()
