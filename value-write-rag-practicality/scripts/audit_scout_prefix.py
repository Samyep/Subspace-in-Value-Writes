#!/usr/bin/env python3
"""Compare truncated and full Llama selector passes from the same runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import load_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument(
        "--truncated",
        type=Path,
        default=ROOT / "results/scout/llama_3b_l17/selections.jsonl",
    )
    parser.add_argument(
        "--full-scoring-dir",
        type=Path,
        default=ROOT / "results/cost/llama_3b/scoring",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/scout/llama_3b_l17/prefix_audit.json",
    )
    return parser.parse_args()


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs in {label}")
    return result


def load_result_directory(path: Path) -> list[dict[str, Any]]:
    merged = path / "merged.jsonl"
    if merged.exists():
        return load_jsonl(merged)
    parts = sorted(path.glob("part-*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"No full-model scoring JSONL in {path}")
    combined: dict[str, dict[str, Any]] = {}
    for part in parts:
        for row in load_jsonl(part):
            example_id = str(row["example_id"])
            if example_id in combined and combined[example_id] != row:
                raise ValueError(f"Conflicting duplicate {example_id} in {path}")
            combined[example_id] = row
    return list(combined.values())


def main() -> None:
    args = parse_args()
    expected = index(load_jsonl(args.data), str(args.data))
    truncated = index(load_jsonl(args.truncated), str(args.truncated))
    full = index(load_result_directory(args.full_scoring_dir), str(args.full_scoring_dir))
    expected_ids = set(expected)
    if len(expected_ids) != 400:
        raise SystemExit("The frozen data must contain 400 unique IDs")
    if set(truncated) != expected_ids or set(full) != expected_ids:
        raise SystemExit("Both selector passes must contain exactly the frozen 400 IDs")

    value_matches: list[bool] = []
    attention_matches: list[bool] = []
    value_differences: list[float] = []
    attention_differences: list[float] = []
    value_mismatch_ids: list[str] = []
    attention_mismatch_ids: list[str] = []
    for example_id in expected:
        short = truncated[example_id]
        complete = full[example_id]
        if short["document_ids"] != complete["document_ids"]:
            raise ValueError(f"Document order mismatch for {example_id}")
        value_match = short["value_selected_documents"] == complete[
            "value_selected_documents"
        ]
        attention_match = short["attention_selected_documents"] == complete[
            "attention_selected_documents"
        ]
        value_matches.append(value_match)
        attention_matches.append(attention_match)
        if not value_match:
            value_mismatch_ids.append(example_id)
        if not attention_match:
            attention_mismatch_ids.append(example_id)
        value_differences.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(short["value_scores"], dtype=np.float64)
                        - np.asarray(complete["value_scores"], dtype=np.float64)
                    )
                )
            )
        )
        attention_differences.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(short["attention_scores"], dtype=np.float64)
                        - np.asarray(complete["attention_scores"], dtype=np.float64)
                    )
                )
            )
        )

    result = {
        "n": len(expected_ids),
        "comparison": "same-runtime full 28-block versus truncated 18-block Llama",
        "value_selection_agreement": float(np.mean(value_matches)),
        "attention_selection_agreement": float(np.mean(attention_matches)),
        "maximum_value_score_difference": max(value_differences),
        "maximum_attention_score_difference": max(attention_differences),
        "value_mismatch_ids": value_mismatch_ids,
        "attention_mismatch_ids": attention_mismatch_ids,
    }
    write_json(args.output, result)
    print(
        "Same-runtime prefix audit: "
        f"value={result['value_selection_agreement']:.6f}, "
        f"attention={result['attention_selection_agreement']:.6f}, "
        f"value_max_diff={result['maximum_value_score_difference']:.8g}, "
        f"attention_max_diff={result['maximum_attention_score_difference']:.8g}"
    )


if __name__ == "__main__":
    main()
