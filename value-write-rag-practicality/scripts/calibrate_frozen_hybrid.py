#!/usr/bin/env python3
"""Apply the preregistered HotpotQA gate and freeze MuSiQue hybrid selections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import load_jsonl, write_json, write_jsonl
from vw_rag.prompting import select_top


READERS = ("qwen3_8b", "qwen_14b")
INTERIOR_WEIGHTS = (0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/hotpot_hybrid_calibration.jsonl"
    )
    parser.add_argument(
        "--bge", type=Path, default=ROOT / "results/hybrid_calibration/bge.jsonl"
    )
    parser.add_argument(
        "--value-qwen3",
        type=Path,
        default=ROOT / "results/hybrid_calibration/value_qwen3_8b.jsonl",
    )
    parser.add_argument(
        "--value-qwen14",
        type=Path,
        default=ROOT / "results/hybrid_calibration/value_qwen_14b.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/hybrid_calibration"
    )
    parser.add_argument("--minimum-improvement", type=float, default=0.01)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def zscore(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std())
    if std < 1e-12:
        return np.zeros_like(array)
    return (array - array.mean()) / std


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(left.std()) < 1e-12 or float(right.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    data = index(args.data)
    bge = index(args.bge)
    values = {
        "qwen3_8b": index(args.value_qwen3),
        "qwen_14b": index(args.value_qwen14),
    }
    if len(data) != 200:
        raise RuntimeError(f"Expected 200 held-out calibration rows, found {len(data)}")
    if any(set(item) != set(data) for item in (bge, *values.values())):
        raise RuntimeError("Hybrid calibration IDs do not align")

    weights = (0.0, *INTERIOR_WEIGHTS, 1.0)
    metric_values = {
        reader: {weight: [] for weight in weights} for reader in READERS
    }
    exact_values = {
        reader: {weight: [] for weight in weights} for reader in READERS
    }
    predictions = []
    correlations = {reader: [] for reader in READERS}
    disagreements = {reader: [] for reader in READERS}
    for example_id, row in sorted(data.items(), key=lambda item: item[1]["split_rank"]):
        document_ids = [str(item["document_id"]) for item in row["documents"]]
        gold = set(str(item) for item in row["retrieved_support_document_ids"])
        bge_item = bge[example_id]
        bge_depth = bge_item["by_depth"].get("10")
        if bge_depth is None or bge_depth["document_ids"] != document_ids:
            raise RuntimeError(f"BGE document order mismatch for {example_id}")
        bge_z = zscore([float(item) for item in bge_depth["scores"]])
        for reader in READERS:
            value_item = values[reader][example_id]
            if value_item["document_ids"] != document_ids:
                raise RuntimeError(f"Value document order mismatch for {reader} {example_id}")
            value_z = zscore([float(item) for item in value_item["value_scores"]])
            correlations[reader].append(correlation(value_z, bge_z))
            value_top = set(select_top(document_ids, value_z.tolist(), 2))
            bge_top = set(select_top(document_ids, bge_z.tolist(), 2))
            disagreements[reader].append(float(value_top != bge_top))
            selected_by_weight = {}
            for weight in weights:
                scores = weight * value_z + (1.0 - weight) * bge_z
                selected = select_top(document_ids, scores.tolist(), 2)
                hits = len(set(selected) & gold)
                metric_values[reader][weight].append(hits / 2)
                exact_values[reader][weight].append(float(hits == 2))
                selected_by_weight[f"{weight:.2f}"] = selected
            predictions.append(
                {
                    "example_id": example_id,
                    "split_rank": int(row["split_rank"]),
                    "reader": reader,
                    "selected_by_weight": selected_by_weight,
                }
            )

    per_reader = {}
    macro = {}
    for reader in READERS:
        per_reader[reader] = {
            "mean_score_correlation": float(np.mean(correlations[reader])),
            "top2_disagreement_rate": float(np.mean(disagreements[reader])),
            "support_recall_by_weight": {
                f"{weight:.2f}": float(np.mean(metric_values[reader][weight]))
                for weight in weights
            },
            "exact_support_pair_by_weight": {
                f"{weight:.2f}": float(np.mean(exact_values[reader][weight]))
                for weight in weights
            },
        }
    for weight in weights:
        macro[f"{weight:.2f}"] = float(
            np.mean(
                [
                    np.mean(metric_values[reader][weight])
                    for reader in READERS
                ]
            )
        )
    selected_weight = sorted(
        INTERIOR_WEIGHTS,
        key=lambda weight: (-macro[f"{weight:.2f}"], abs(weight - 0.5), weight),
    )[0]
    best_endpoint = max(macro["0.00"], macro["1.00"])
    selected_score = macro[f"{selected_weight:.2f}"]
    improvement = selected_score - best_endpoint
    passed = improvement >= args.minimum_improvement
    summary = {
        "protocol": {
            "n_examples": 200,
            "readers": list(READERS),
            "interior_weights": list(INTERIOR_WEIGHTS),
            "normalization": "within-question z-score",
            "selection_metric": "reader-macro support recall@2",
            "minimum_improvement": args.minimum_improvement,
        },
        "per_reader": per_reader,
        "macro_support_recall_by_weight": macro,
        "selected_weight": selected_weight,
        "best_endpoint_support_recall": best_endpoint,
        "selected_support_recall": selected_score,
        "improvement": improvement,
        "gate_passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "calibration_predictions.jsonl", predictions)
    write_json(args.output_dir / "hybrid_gate.json", summary)

    lines = [
        "# Frozen hybrid calibration",
        "",
        "The calibration set contains 200 HotpotQA retrieval examples disjoint "
        "from the 50 direction-fitting examples.",
        "",
        "| Value weight | Macro support recall@2 |",
        "|---:|---:|",
    ]
    for weight in weights:
        lines.append(f"| {weight:.2f} | {100*macro[f'{weight:.2f}']:.2f} |")
    lines.extend(
        [
            "",
            f"Selected weight: **{selected_weight:.2f}**.",
            f"Improvement over the better endpoint: **{100*improvement:.2f} points**.",
            f"Preregistered 1.00-point gate: **{'passed' if passed else 'failed'}**.",
            "",
        ]
    )
    (args.output_dir / "hybrid_gate.md").write_text("\n".join(lines), encoding="utf-8")

    if passed:
        main_bge = index(ROOT / "results/reranker/bge_v2_m3/selections.jsonl")
        for reader in READERS:
            reference = index(ROOT / "reference_results" / reader / "per_example.jsonl")
            output_rows = []
            for example_id, value_item in sorted(
                reference.items(), key=lambda item: item[1]["split_rank"]
            ):
                bge_item = main_bge[example_id]
                document_ids = [str(item) for item in value_item["document_ids"]]
                if bge_item["document_ids"] != document_ids:
                    raise RuntimeError(f"MuSiQue document mismatch for {example_id}")
                value_z = zscore([float(item) for item in value_item["value_write_scores"]])
                bge_z = zscore([float(item) for item in bge_item["scores"]])
                scores = selected_weight * value_z + (1.0 - selected_weight) * bge_z
                output_rows.append(
                    {
                        "example_id": example_id,
                        "split_rank": int(value_item["split_rank"]),
                        "reader": reader,
                        "lambda_value": selected_weight,
                        "document_ids": document_ids,
                        "scores": scores.tolist(),
                        "selected_documents": select_top(document_ids, scores.tolist(), 2),
                    }
                )
            write_jsonl(args.output_dir / f"musique_{reader}_selections.jsonl", output_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
