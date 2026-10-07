#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_evidence.artifacts import load_csv


def show_table(title: str, rows: list[dict[str, str]], columns: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print("\t".join(columns))
    for row in rows:
        print("\t".join(row.get(col, "") for col in columns))


def main() -> None:
    show_table(
        "Support-fact recovery",
        load_csv("artifacts/figures/fig1_selection.csv"),
        ["cell", "pca_auc", "pca_top2_recovery", "norm_auc"],
    )
    show_table(
        "Span ablation",
        load_csv("artifacts/figures/fig2_causal.csv"),
        ["cell", "gold_drop", "pca_drop", "random_drop"],
    )
    show_table(
        "Transfer",
        load_csv("artifacts/figures/fig4_transfer.csv"),
        ["cell", "discovery_auc", "transfer_auc", "transfer_recovery"],
    )
    show_table(
        "Bridge",
        load_csv("artifacts/figures/fig7_bridge.csv"),
        ["cell", "gold_drop", "pca_drop", "distractor_drop", "random_drop"],
    )
    show_table(
        "Headline baselines",
        load_csv("artifacts/figures/headline_baselines.csv"),
        ["cell", "value_write_pca_auc", "attention_mass_auc", "source_residual_pca_auc", "value_write_norm_auc"],
    )


if __name__ == "__main__":
    main()
