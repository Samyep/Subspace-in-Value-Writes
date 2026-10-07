#!/usr/bin/env python3
"""Aggregate the strong-reranker and end-to-end cost experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.config import MODEL_CONFIGS
from vw_rag.io_utils import load_jsonl, write_json
from vw_rag.stats import bootstrap_mean


CELL_LABELS = {
    "qwen_7b": "Qwen2.5-7B",
    "qwen_14b": "Qwen2.5-14B",
    "qwen3_8b": "Qwen3-8B",
    "llama_3b": "Llama-3.2-3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument(
        "--reranker-selections",
        type=Path,
        default=ROOT / "results/reranker/bge_v2_m3/selections.jsonl",
    )
    parser.add_argument(
        "--reader-root", type=Path, default=ROOT / "results/readers"
    )
    parser.add_argument("--cost-root", type=Path, default=ROOT / "results/cost")
    parser.add_argument(
        "--reference-root", type=Path, default=ROOT / "reference_results"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/aggregate"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    parser.add_argument("--allow-missing-cost", action="store_true")
    return parser.parse_args()


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
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
        raise FileNotFoundError(f"No result JSONL in {path}")
    combined: dict[str, dict[str, Any]] = {}
    for part in parts:
        for row in load_jsonl(part):
            example_id = str(row["example_id"])
            if example_id in combined and combined[example_id] != row:
                raise ValueError(f"Conflicting duplicate {example_id} across {path}")
            combined[example_id] = row
    return sorted(combined.values(), key=lambda row: int(row["split_rank"]))


def metric_summary(
    rows: list[dict[str, Any]], keys: list[str], draws: int, seed: int
) -> dict[str, Any]:
    return {
        key: bootstrap_mean(
            [float(row[key]) for row in rows], draws, seed + index
        )
        for index, key in enumerate(keys)
    }


def shared_question_macro(
    values_by_cell: dict[str, list[float]], draws: int, seed: int
) -> dict[str, Any]:
    arrays = [np.asarray(values_by_cell[cell], dtype=np.float64) for cell in CELL_LABELS]
    sizes = {len(array) for array in arrays}
    if len(sizes) != 1:
        raise ValueError("Cells do not share the same number of questions")
    matrix = np.stack(arrays, axis=0)
    per_question = matrix.mean(axis=0)
    return bootstrap_mean(per_question.tolist(), draws, seed)


def primary_results(args: argparse.Namespace, expected_ids: list[str]) -> dict[str, Any]:
    reranker_selection = index_rows(
        load_jsonl(args.reranker_selections), str(args.reranker_selections)
    )
    if set(reranker_selection) != set(expected_ids):
        raise ValueError("Reranker selection file must contain the frozen 400 IDs")
    result: dict[str, Any] = {"cells": {}}
    vectors: dict[str, dict[str, list[float]]] = {}
    for cell_index, cell in enumerate(CELL_LABELS):
        reference = index_rows(
            load_jsonl(args.reference_root / cell / "per_example.jsonl"),
            f"reference:{cell}",
        )
        reranker_reader = index_rows(
            load_result_directory(args.reader_root / cell / "reranker"),
            f"reranker-reader:{cell}",
        )
        if set(reference) != set(expected_ids) or set(reranker_reader) != set(expected_ids):
            raise ValueError(f"{cell} does not contain exactly the frozen 400 IDs")
        rows = []
        for example_id in expected_ids:
            ref = reference[example_id]
            reader = reranker_reader[example_id]
            selection = reranker_selection[example_id]
            if reader["selected_documents"] != selection["selected_documents"]:
                raise ValueError(f"Reranker selection mismatch for {cell}:{example_id}")
            rows.append(
                {
                    "full20_em": float(ref["baseline_em"]),
                    "full20_f1": float(ref["baseline_f1"]),
                    "value2_em": float(ref["value_write_em"]),
                    "value2_f1": float(ref["value_write_f1"]),
                    "attention2_em": float(ref["attention_em"]),
                    "attention2_f1": float(ref["attention_f1"]),
                    "retrieval2_em": float(ref["retriever_em"]),
                    "reranker2_em": float(reader["em"]),
                    "reranker2_f1": float(reader["f1"]),
                    "reranker2_support_recall": float(selection["support_recall"]),
                    "reranker2_exact_top2": float(selection["exact_top2"]),
                    "reranker2_prompt_token_ratio": float(reader["prompt_token_ratio"]),
                    "value_minus_reranker_em": float(ref["value_write_em"])
                    - float(reader["em"]),
                    "reranker_minus_full_em": float(reader["em"])
                    - float(ref["baseline_em"]),
                }
            )
        keys = list(rows[0])
        result["cells"][cell] = metric_summary(
            rows, keys, args.bootstrap_draws, args.seed + 1000 * cell_index
        )
        vectors[cell] = {key: [row[key] for row in rows] for key in keys}

    macro: dict[str, Any] = {}
    for metric_index, key in enumerate(next(iter(vectors.values()))):
        macro[key] = shared_question_macro(
            {cell: vectors[cell][key] for cell in CELL_LABELS},
            args.bootstrap_draws,
            args.seed + 10000 + metric_index,
        )
    result["macro_shared_question_bootstrap"] = macro
    return result


def cost_results(args: argparse.Namespace) -> dict[str, Any]:
    reranker_scores = index_rows(
        load_jsonl(args.reranker_selections), str(args.reranker_selections)
    )
    result: dict[str, Any] = {"cells": {}}
    for cell_index, cell in enumerate(CELL_LABELS):
        try:
            scoring = index_rows(
                load_result_directory(args.cost_root / cell / "scoring"),
                f"cost-scoring:{cell}",
            )
            generation = index_rows(
                load_result_directory(args.cost_root / cell / "generation"),
                f"cost-generation:{cell}",
            )
        except FileNotFoundError:
            if args.allow_missing_cost:
                result["cells"][cell] = {"status": "missing"}
                continue
            raise
        ids = sorted(
            set(scoring) & set(generation),
            key=lambda example_id: int(scoring[example_id]["split_rank"]),
        )
        rows = []
        for example_id in ids:
            score = scoring[example_id]
            answer = generation[example_id]
            reranker = reranker_scores[example_id]
            if score["timing_warmup"] or answer["timing_warmup"] or reranker["timing_warmup"]:
                continue
            row = {
                "full20_seconds": float(answer["full20_seconds"]),
                "value2_seconds": float(score["score_seconds"])
                + float(answer["value2_seconds"]),
                "attention2_seconds": float(score["score_seconds"])
                + float(answer["attention2_seconds"]),
                "retrieval2_seconds": float(answer["retrieval2_seconds"]),
                "reranker2_component_sum_seconds": float(reranker["score_seconds"])
                + float(answer["reranker2_seconds"]),
                "full20_input_tokens": float(answer["full20_prompt_tokens"]),
                "value2_total_input_tokens": float(score["prompt_tokens"])
                + float(answer["value2_prompt_tokens"]),
                "attention2_total_input_tokens": float(score["prompt_tokens"])
                + float(answer["attention2_prompt_tokens"]),
                "retrieval2_input_tokens": float(answer["retrieval2_prompt_tokens"]),
                "reranker2_total_input_tokens": float(reranker["pair_input_tokens"])
                + float(answer["reranker2_prompt_tokens"]),
            }
            row.update(
                {
                    "value2_time_ratio": row["value2_seconds"] / row["full20_seconds"],
                    "attention2_time_ratio": row["attention2_seconds"]
                    / row["full20_seconds"],
                    "retrieval2_time_ratio": row["retrieval2_seconds"]
                    / row["full20_seconds"],
                    "reranker2_component_time_ratio": row[
                        "reranker2_component_sum_seconds"
                    ]
                    / row["full20_seconds"],
                    "value2_total_token_ratio": row["value2_total_input_tokens"]
                    / row["full20_input_tokens"],
                    "reranker2_total_token_ratio": row[
                        "reranker2_total_input_tokens"
                    ]
                    / row["full20_input_tokens"],
                }
            )
            rows.append(row)
        if not rows:
            raise ValueError(f"No non-warmup paired cost rows for {cell}")
        result["cells"][cell] = {
            "n_paired_after_warmup": len(rows),
            "metrics": metric_summary(
                rows,
                list(rows[0]),
                args.bootstrap_draws,
                args.seed + 20000 + 1000 * cell_index,
            ),
        }
    result["interpretation"] = (
        "Value/attention totals sum the reader scoring pass and selected-context "
        "generation. Reranker totals are a component sum across the reranker and "
        "reader runs; compare wall time directly only when hardware metadata match."
    )
    return result


def percent(item: dict[str, Any]) -> str:
    return f"{100 * float(item['mean']):.2f}"


def write_report(path: Path, primary: dict[str, Any], cost: dict[str, Any]) -> None:
    lines = [
        "# Practical RAG follow-up results",
        "",
        "## Strong reranker comparison",
        "",
        "| Reader | Full-20 EM | Value-2 EM | Attention-2 EM | Reranker-2 EM | Value - reranker |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell, label in CELL_LABELS.items():
        metrics = primary["cells"][cell]
        lines.append(
            f"| {label} | {percent(metrics['full20_em'])} | "
            f"{percent(metrics['value2_em'])} | {percent(metrics['attention2_em'])} | "
            f"{percent(metrics['reranker2_em'])} | "
            f"{100 * float(metrics['value_minus_reranker_em']['mean']):+.2f} |"
        )
    macro = primary["macro_shared_question_bootstrap"]
    lines.append(
        f"| Macro | {percent(macro['full20_em'])} | {percent(macro['value2_em'])} | "
        f"{percent(macro['attention2_em'])} | {percent(macro['reranker2_em'])} | "
        f"{100 * float(macro['value_minus_reranker_em']['mean']):+.2f} |"
    )
    lines.extend(["", "## End-to-end cost audit", ""])
    for cell, label in CELL_LABELS.items():
        item = cost["cells"].get(cell, {"status": "missing"})
        if item.get("status") == "missing":
            lines.append(f"- {label}: missing")
            continue
        metrics = item["metrics"]
        lines.append(
            f"- {label}: Value-2/Full-20 time ratio "
            f"{float(metrics['value2_time_ratio']['mean']):.2f}; "
            f"Reranker-2 component/Full-20 ratio "
            f"{float(metrics['reranker2_component_time_ratio']['mean']):.2f}; "
            f"n={item['n_paired_after_warmup']}."
        )
    lines.extend(["", cost["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    expected_rows = sorted(load_jsonl(args.data), key=lambda row: int(row["split_rank"]))
    expected_ids = [str(row["example_id"]) for row in expected_rows]
    if len(expected_ids) != 400 or len(set(expected_ids)) != 400:
        raise SystemExit("The frozen evaluation data must contain 400 unique IDs")
    primary = primary_results(args, expected_ids)
    cost = cost_results(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "practicality_summary.json", {"primary": primary, "cost": cost})
    write_report(args.output_dir / "practicality_report.md", primary, cost)
    print(f"Wrote {args.output_dir / 'practicality_summary.json'}")
    print(f"Wrote {args.output_dir / 'practicality_report.md'}")


if __name__ == "__main__":
    main()
