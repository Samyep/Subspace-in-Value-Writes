#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_evidence.artifacts import load_csv


def main() -> None:
    rows = load_csv("artifacts/case_studies/main_text_case_summary.csv")
    for row in rows:
        print(
            f"{row['case_name']}\t"
            f"cell={row['cell']}\t"
            f"answer={row['answer']}\t"
            f"gold={row['gold_labels']}\t"
            f"pca_selected={row['pca_selected']}"
        )


if __name__ == "__main__":
    main()
