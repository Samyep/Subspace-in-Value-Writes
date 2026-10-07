#!/usr/bin/env python3
"""Aggregate the truncated-scout cross-reader experiment.

The primary reader is Qwen2.5-14B and Qwen3-8B is the predeclared
confirmation reader. Confidence intervals resample the same frozen questions
so every reported contrast remains paired.
"""

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

from vw_rag.io_utils import load_jsonl, write_json
from vw_rag.stats import bootstrap_mean


READER_LABELS = {
    "qwen_14b": "Qwen2.5-14B",
    "qwen3_8b": "Qwen3-8B",
}
CONDITION_LABELS = {
    "full20": "Full-20",
    "bm25_2": "BM25-2",
    "reranker2": "BGE-2",
    "scout_attention2": "3B Attention-2",
    "scout_value2": "3B Value-2",
    "oracle2": "Oracle-2",
}
CONDITIONS = tuple(CONDITION_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument(
        "--scout-selections",
        type=Path,
        default=ROOT / "results/scout/llama_3b_l17/selections.jsonl",
    )
    parser.add_argument(
        "--reranker-selections",
        type=Path,
        default=ROOT / "results/reranker/bge_v2_m3/selections.jsonl",
    )
    parser.add_argument(
        "--reader-root", type=Path, default=ROOT / "results/scout_readers"
    )
    parser.add_argument(
        "--readers",
        nargs="+",
        choices=sorted(READER_LABELS),
        default=list(READER_LABELS),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/scout_aggregate"
    )
    parser.add_argument(
        "--prefix-audit",
        type=Path,
        default=ROOT / "results/scout/llama_3b_l17/prefix_audit.json",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
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


def summarize(
    rows: list[dict[str, float]], draws: int, seed: int
) -> dict[str, dict[str, Any]]:
    return {
        key: bootstrap_mean([row[key] for row in rows], draws, seed + index)
        for index, key in enumerate(rows[0])
    }


def shared_question_macro(
    vectors: dict[str, list[float]], draws: int, seed: int
) -> dict[str, Any]:
    arrays = [np.asarray(vector, dtype=np.float64) for vector in vectors.values()]
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Readers do not share the same number of questions")
    return bootstrap_mean(np.stack(arrays).mean(axis=0).tolist(), draws, seed)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_signature(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    runtime = config.get("runtime", {})
    return {
        "gpu": runtime.get("gpu"),
        "torch": runtime.get("torch"),
        "transformers": runtime.get("transformers"),
    }


def percent(item: dict[str, Any]) -> str:
    return f"{100.0 * float(item['mean']):.2f}"


def interval(item: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{scale * float(item['mean']):.2f} "
        f"[{scale * float(item['lo']):.2f}, {scale * float(item['hi']):.2f}]"
    )


def write_report(path: Path, result: dict[str, Any], readers: list[str]) -> None:
    lines = [
        "# Truncated 3B scout results",
        "",
        "The scout retains Llama-3.2-3B blocks 0--17. Value-2 and Attention-2 "
        "come from the same forward pass.",
        "",
        "## Selection checks",
        "",
        f"- Value top-2 agreement with the released 3B run: "
        f"{100.0 * result['selection']['value_full_model_agreement']['mean']:.2f}%.",
        f"- Attention top-2 agreement with the released 3B run: "
        f"{100.0 * result['selection']['attention_full_model_agreement']['mean']:.2f}%.",
        f"- Maximum absolute Value score difference from the released run: "
        f"{result['selection']['maximum_value_score_difference']:.8g}.",
        f"- Maximum absolute Attention score difference from the released run: "
        f"{result['selection']['maximum_attention_score_difference']:.8g}.",
        *(
            [
                f"- Same-runtime full/truncated Value top-2 agreement: "
                f"{100.0 * result['same_runtime_prefix_audit']['value_selection_agreement']:.2f}%.",
                f"- Same-runtime full/truncated Attention top-2 agreement: "
                f"{100.0 * result['same_runtime_prefix_audit']['attention_selection_agreement']:.2f}%.",
                f"- Same-runtime maximum absolute Value score difference: "
                f"{result['same_runtime_prefix_audit']['maximum_value_score_difference']:.8g}.",
                f"- Same-runtime maximum absolute Attention score difference: "
                f"{result['same_runtime_prefix_audit']['maximum_attention_score_difference']:.8g}.",
            ]
            if result.get("same_runtime_prefix_audit") is not None
            else []
        ),
        "",
        "## Answer quality",
        "",
        "| Reader | Method | EM | F1 | Support recall@2 | Prompt/full tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for reader in readers:
        metrics = result["readers"][reader]["quality"]
        for condition in CONDITIONS:
            support = "--"
            if condition != "full20":
                support = percent(metrics[f"{condition}_support_recall"])
            token_ratio = float(metrics[f"{condition}_prompt_token_ratio"]["mean"])
            lines.append(
                f"| {READER_LABELS[reader]} | {CONDITION_LABELS[condition]} | "
                f"{percent(metrics[f'{condition}_em'])} | "
                f"{percent(metrics[f'{condition}_f1'])} | {support} | "
                f"{token_ratio:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "| Reader | Contrast | EM points (95% CI) |",
            "|---|---|---:|",
        ]
    )
    for reader in readers:
        contrasts = result["readers"][reader]["contrasts"]
        for key, label in (
            ("scout_value2_em_minus_full20", "3B Value-2 minus Full-20"),
            (
                "scout_value2_em_minus_scout_attention2",
                "3B Value-2 minus 3B Attention-2",
            ),
            ("scout_value2_em_minus_reranker2", "3B Value-2 minus BGE-2"),
        ):
            lines.append(
                f"| {READER_LABELS[reader]} | {label} | "
                f"{interval(contrasts[key], 100.0)} |"
            )
    lines.extend(
        [
            "",
            "## End-to-end time",
            "",
            "Ratios include selection plus answer generation and exclude the first five "
            "warm-up questions.",
            "",
            "| Reader | Method | Seconds | Time/full ratio | Paired n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for reader in readers:
        cost = result["readers"][reader]["cost"]
        for condition in (
            "full20",
            "bm25_2",
            "reranker2",
            "scout_attention2",
            "scout_value2",
        ):
            lines.append(
                f"| {READER_LABELS[reader]} | {CONDITION_LABELS[condition]} | "
                f"{float(cost[f'{condition}_seconds']['mean']):.3f} | "
                f"{float(cost[f'{condition}_time_ratio']['mean']):.3f} | "
                f"{cost[f'{condition}_seconds']['n']} |"
            )
    lines.extend(["", result["cost_interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_rows = sorted(load_jsonl(args.data), key=lambda row: int(row["split_rank"]))
    expected_ids = [str(row["example_id"]) for row in data_rows]
    if len(expected_ids) != 400 or len(set(expected_ids)) != 400:
        raise SystemExit("The frozen evaluation data must contain 400 unique IDs")
    scout = index_rows(load_jsonl(args.scout_selections), str(args.scout_selections))
    reranker = index_rows(
        load_jsonl(args.reranker_selections), str(args.reranker_selections)
    )
    if set(scout) != set(expected_ids) or set(reranker) != set(expected_ids):
        raise SystemExit("Both selectors must contain exactly the frozen 400 IDs")

    selection_rows = [
        {
            "value_full_model_agreement": float(
                scout[example_id]["value_full_model_selection_match"]
            ),
            "attention_full_model_agreement": float(
                scout[example_id]["attention_full_model_selection_match"]
            ),
            "value_support_recall": float(scout[example_id]["value_support_recall"]),
            "attention_support_recall": float(
                scout[example_id]["attention_support_recall"]
            ),
            "reranker_support_recall": float(reranker[example_id]["support_recall"]),
        }
        for example_id in expected_ids
    ]
    selection = summarize(
        selection_rows, args.bootstrap_draws, args.seed
    )
    selection.update(
        {
            "maximum_value_score_difference": max(
                float(row["value_full_model_max_abs_score_difference"])
                for row in scout.values()
            ),
            "maximum_attention_score_difference": max(
                float(row["attention_full_model_max_abs_score_difference"])
                for row in scout.values()
            ),
        }
    )

    result: dict[str, Any] = {
        "protocol": {
            "n_questions": len(expected_ids),
            "primary_reader": "qwen_14b",
            "confirmation_reader": "qwen3_8b",
            "readers": args.readers,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "selection": selection,
        "same_runtime_prefix_audit": load_json(args.prefix_audit),
        "readers": {},
    }
    vectors: dict[str, dict[str, list[float]]] = {}
    scout_config = load_json(args.scout_selections.with_suffix(".run_config.json"))
    reranker_config = load_json(
        args.reranker_selections.with_suffix(".run_config.json")
    )

    for reader_index, reader in enumerate(args.readers):
        reader_rows = index_rows(
            load_result_directory(args.reader_root / reader), f"reader:{reader}"
        )
        if set(reader_rows) != set(expected_ids):
            raise ValueError(f"{reader} does not contain exactly the frozen 400 IDs")
        quality_rows: list[dict[str, float]] = []
        cost_rows: list[dict[str, float]] = []
        for example_id in expected_ids:
            answer = reader_rows[example_id]
            scout_row = scout[example_id]
            reranker_row = reranker[example_id]
            if answer["scout_value2_selected_documents"] != scout_row[
                "value_selected_documents"
            ]:
                raise ValueError(f"Value selection mismatch for {reader}:{example_id}")
            if answer["scout_attention2_selected_documents"] != scout_row[
                "attention_selected_documents"
            ]:
                raise ValueError(
                    f"Attention selection mismatch for {reader}:{example_id}"
                )
            if answer["reranker2_selected_documents"] != reranker_row[
                "selected_documents"
            ]:
                raise ValueError(f"Reranker selection mismatch for {reader}:{example_id}")

            full_tokens = float(answer["full20_prompt_tokens"])
            quality: dict[str, float] = {}
            for condition in CONDITIONS:
                quality[f"{condition}_em"] = float(answer[f"{condition}_em"])
                quality[f"{condition}_f1"] = float(answer[f"{condition}_f1"])
                quality[f"{condition}_prompt_token_ratio"] = (
                    float(answer[f"{condition}_prompt_tokens"]) / full_tokens
                )
                if condition != "full20":
                    quality[f"{condition}_support_recall"] = float(
                        answer[f"{condition}_support_recall"]
                    )
            quality.update(
                {
                    "scout_value2_em_minus_full20": float(
                        answer["scout_value2_em"] - answer["full20_em"]
                    ),
                    "scout_value2_em_minus_scout_attention2": float(
                        answer["scout_value2_em"]
                        - answer["scout_attention2_em"]
                    ),
                    "scout_value2_em_minus_reranker2": float(
                        answer["scout_value2_em"] - answer["reranker2_em"]
                    ),
                }
            )
            quality_rows.append(quality)

            if not (
                answer["timing_warmup"]
                or scout_row["timing_warmup"]
                or reranker_row["timing_warmup"]
            ):
                full_seconds = float(answer["full20_seconds"])
                cost = {
                    "full20_seconds": full_seconds,
                    "bm25_2_seconds": float(answer["bm25_2_seconds"]),
                    "reranker2_seconds": float(reranker_row["score_seconds"])
                    + float(answer["reranker2_seconds"]),
                    "scout_attention2_seconds": float(scout_row["scout_seconds"])
                    + float(answer["scout_attention2_seconds"]),
                    "scout_value2_seconds": float(scout_row["scout_seconds"])
                    + float(answer["scout_value2_seconds"]),
                }
                for condition in (
                    "full20",
                    "bm25_2",
                    "reranker2",
                    "scout_attention2",
                    "scout_value2",
                ):
                    cost[f"{condition}_time_ratio"] = (
                        cost[f"{condition}_seconds"] / full_seconds
                    )
                cost_rows.append(cost)

        quality_summary = summarize(
            quality_rows, args.bootstrap_draws, args.seed + 1000 * reader_index
        )
        contrast_names = (
            "scout_value2_em_minus_full20",
            "scout_value2_em_minus_scout_attention2",
            "scout_value2_em_minus_reranker2",
        )
        vectors[reader] = {
            key: [row[key] for row in quality_rows] for key in quality_rows[0]
        }
        reader_configs = sorted((args.reader_root / reader).glob("run_config-*.json"))
        reader_config = load_json(reader_configs[0]) if reader_configs else None
        result["readers"][reader] = {
            "quality": {
                key: value
                for key, value in quality_summary.items()
                if key not in contrast_names
            },
            "contrasts": {
                key: quality_summary[key] for key in contrast_names
            },
            "cost": summarize(
                cost_rows,
                args.bootstrap_draws,
                args.seed + 10000 + 1000 * reader_index,
            ),
            "runtime_signatures": {
                "scout": runtime_signature(scout_config),
                "reranker": runtime_signature(reranker_config),
                "reader": runtime_signature(reader_config),
            },
        }

    macro: dict[str, Any] = {}
    first_reader = args.readers[0]
    for metric_index, key in enumerate(vectors[first_reader]):
        macro[key] = shared_question_macro(
            {reader: vectors[reader][key] for reader in args.readers},
            args.bootstrap_draws,
            args.seed + 20000 + metric_index,
        )
    result["macro_shared_question_bootstrap"] = macro
    result["cost_interpretation"] = (
        "Scout totals add one shared truncated-Llama scoring pass to selected-context "
        "generation. BGE totals add cross-encoder scoring to generation. Direct wall-time "
        "comparisons require matching GPU and software signatures, which are recorded in "
        "the JSON report."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "scout_summary.json"
    report_path = args.output_dir / "scout_report.md"
    write_json(summary_path, result)
    write_report(report_path, result, args.readers)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
