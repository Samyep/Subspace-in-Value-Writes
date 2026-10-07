#!/usr/bin/env python3
"""Benchmark answer generation with optimized reader attention.

This script measures only answer-generation passes. The full-context
value-write scoring pass is measured by ``benchmark_value_scoring.py`` and is
combined with these rows by ``aggregate_practicality.py``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.config import MODEL_CONFIGS, get_model_config
from vw_rag.io_utils import (
    portable_path,
    append_jsonl,
    load_jsonl,
    runtime_metadata,
    select_shard,
    write_json,
)
from vw_rag.model_loading import generate_answer, load_model_and_tokenizer
from vw_rag.prompting import evaluate_prediction, reference_answers, render_prompt
from vw_rag.stats import bootstrap_mean, summarize_metrics
from vw_rag.timing import timed_call



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument(
        "--reranker-selections",
        type=Path,
        default=ROOT / "results/reranker/bge_v2_m3/selections.jsonl",
    )
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def rotated(items: list[str], offset: int) -> list[str]:
    if not items:
        return []
    shift = offset % len(items)
    return items[shift:] + items[:shift]


def main() -> None:
    args = parse_args()
    config = get_model_config(args.cell, args.model_id)
    args.reference_results = args.reference_results or (
        ROOT / "reference_results" / args.cell / "per_example.jsonl"
    )
    args.output_dir = args.output_dir or ROOT / "results/cost" / args.cell / "generation"
    required = [args.data, args.reference_results]
    if not args.skip_reranker:
        required.append(args.reranker_selections)
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    rows = select_shard(load_jsonl(args.data), args.shard_index, args.num_shards)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    references = index(args.reference_results)
    reranker = {} if args.skip_reranker else index(args.reranker_selections)
    requested_ids = {str(row["example_id"]) for row in rows}
    if requested_ids - set(references):
        raise SystemExit("Reference results do not cover the requested shard")
    if not args.skip_reranker and requested_ids - set(reranker):
        raise SystemExit("Reranker selections do not cover the requested shard")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"part-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    completed = index(result_path) if result_path.exists() else {}
    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attention_implementation,
    )
    run_config = {
        "cell": args.cell,
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "data": portable_path(args.data),
        "reference_results": portable_path(args.reference_results),
        "reranker_selections": (
            None if args.skip_reranker else portable_path(args.reranker_selections)
        ),
        "max_new_tokens": args.max_new_tokens,
        "attention_implementation": args.attention_implementation or "transformers_default",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows_in_shard": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "runtime": runtime_metadata(),
    }
    write_json(
        args.output_dir
        / f"run_config-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        run_config,
    )

    for position, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        reference = references[example_id]
        document_ids = [str(item["document_id"]) for item in row["documents"]]
        selections: dict[str, list[str] | None] = {
            "full20": None,
            "value2": [str(item) for item in reference["value_write_documents"]],
            "attention2": [str(item) for item in reference["attention_documents"]],
            "retrieval2": document_ids[:2],
        }
        if not args.skip_reranker:
            selections["reranker2"] = [
                str(item) for item in reranker[example_id]["selected_documents"]
            ]
        for condition, selected in selections.items():
            if selected is not None and (
                len(selected) != 2 or not set(selected).issubset(document_ids)
            ):
                raise RuntimeError(f"Invalid {condition} selection for {example_id}")

        condition_order = rotated(
            list(selections), int(row["split_rank"]) - 1
        )
        condition_results: dict[str, dict[str, Any]] = {}
        answers = reference_answers(row)
        for condition in condition_order:
            prompt, _ = render_prompt(loaded.tokenizer, row, selections[condition])
            gc.collect()
            if loaded.device.type == "cuda":
                torch.cuda.empty_cache()
            timed = timed_call(
                loaded.device,
                lambda prompt=prompt: generate_answer(
                    loaded, prompt, args.max_new_tokens
                ),
            )
            raw, answer, prompt_tokens, output_tokens = timed.value
            metrics = evaluate_prediction(answer, answers)
            condition_results[condition] = {
                "selected_documents": selections[condition],
                "raw": raw,
                "answer": answer,
                "em": metrics["em"],
                "f1": metrics["f1"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "seconds": timed.seconds,
                "peak_allocated_bytes": timed.peak_allocated_bytes,
                "peak_reserved_bytes": timed.peak_reserved_bytes,
                "baseline_allocated_bytes": timed.baseline_allocated_bytes,
            }

        result: dict[str, Any] = {
            "example_id": example_id,
            "split_rank": int(row["split_rank"]),
            "cell": args.cell,
            "condition_order": condition_order,
            "timing_warmup": position < args.timing_warmup_examples,
        }
        for condition, values in condition_results.items():
            for key, value in values.items():
                result[f"{condition}_{key}"] = value
        result["value2_em_minus_full20"] = result["value2_em"] - result["full20_em"]
        result["attention2_em_minus_full20"] = (
            result["attention2_em"] - result["full20_em"]
        )
        if "reranker2" in condition_results:
            result["reranker2_em_minus_full20"] = (
                result["reranker2_em"] - result["full20_em"]
            )
            result["reranker2_em_minus_value2"] = (
                result["reranker2_em"] - result["value2_em"]
            )
        append_jsonl(result_path, result)
        completed[example_id] = result
        print(
            f"{args.cell} generation {len(completed)}/{len(rows)} "
            f"rank={row['split_rank']} full={result['full20_em']:.0f} "
            f"value={result['value2_em']:.0f}",
            flush=True,
        )

    ordered = [completed[str(row["example_id"])] for row in rows]
    timed_rows = [row for row in ordered if not row["timing_warmup"]]
    conditions = ["full20", "value2", "attention2", "retrieval2"]
    if not args.skip_reranker:
        conditions.append("reranker2")
    answer_metrics = {}
    timing_metrics = {}
    for condition_index, condition in enumerate(conditions):
        answer_metrics[condition] = summarize_metrics(
            ordered,
            [f"{condition}_em", f"{condition}_f1", f"{condition}_prompt_tokens"],
            args.bootstrap_draws,
            args.seed + 100 * condition_index,
        )
        timing_metrics[condition] = summarize_metrics(
            timed_rows,
            [f"{condition}_seconds", f"{condition}_peak_allocated_bytes"],
            args.bootstrap_draws,
            args.seed + 1000 + 100 * condition_index,
        )
    contrasts = {
        "value2_minus_full20_em": bootstrap_mean(
            [row["value2_em_minus_full20"] for row in ordered],
            args.bootstrap_draws,
            args.seed + 2000,
        ),
        "attention2_minus_full20_em": bootstrap_mean(
            [row["attention2_em_minus_full20"] for row in ordered],
            args.bootstrap_draws,
            args.seed + 2001,
        ),
    }
    if not args.skip_reranker:
        contrasts.update(
            {
                "reranker2_minus_full20_em": bootstrap_mean(
                    [row["reranker2_em_minus_full20"] for row in ordered],
                    args.bootstrap_draws,
                    args.seed + 2002,
                ),
                "reranker2_minus_value2_em": bootstrap_mean(
                    [row["reranker2_em_minus_value2"] for row in ordered],
                    args.bootstrap_draws,
                    args.seed + 2003,
                ),
            }
        )
    write_json(
        args.output_dir
        / f"summary-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        {
            "run_config": run_config,
            "answer_metrics": answer_metrics,
            "timing_after_warmup": timing_metrics,
            "paired_contrasts": contrasts,
        },
    )
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
