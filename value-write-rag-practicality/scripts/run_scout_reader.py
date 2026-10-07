#!/usr/bin/env python3
"""Evaluate a frozen small-model scout with a different answer reader.

The default table compares Full-20, BM25-2, BGE-2, Llama-3B
Attention-scout-2, Llama-3B Value-scout-2, and Oracle-2. All generation
conditions are rerun with the same reader and rotated across questions.
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
    parser.add_argument("--reader-cell", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-id", default=None)
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
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-reranker", action="store_true")
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


def selections_for_row(
    row: dict[str, Any],
    scout: dict[str, Any],
    reference: dict[str, Any],
    reranker: dict[str, Any] | None,
) -> dict[str, list[str] | None]:
    document_ids = [str(item["document_id"]) for item in row["documents"]]
    selections: dict[str, list[str] | None] = {
        "full20": None,
        "bm25_2": document_ids[:2],
        "scout_attention2": [
            str(item) for item in scout["attention_selected_documents"]
        ],
        "scout_value2": [str(item) for item in scout["value_selected_documents"]],
        "oracle2": [str(item) for item in reference["oracle_documents"]],
    }
    if reranker is not None:
        selections["reranker2"] = [
            str(item) for item in reranker["selected_documents"]
        ]
    for condition, selected in selections.items():
        if selected is not None and (
            len(selected) != 2 or not set(selected).issubset(document_ids)
        ):
            raise ValueError(f"Invalid {condition} selection for {row['example_id']}")
    return selections


def main() -> None:
    args = parse_args()
    config = get_model_config(args.reader_cell, args.model_id)
    args.reference_results = args.reference_results or (
        ROOT / "reference_results" / args.reader_cell / "per_example.jsonl"
    )
    args.output_dir = args.output_dir or (
        ROOT / "results/scout_readers" / args.reader_cell
    )
    required = [args.data, args.scout_selections, args.reference_results]
    if not args.skip_reranker:
        required.append(args.reranker_selections)
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    rows = select_shard(load_jsonl(args.data), args.shard_index, args.num_shards)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    scout = index(args.scout_selections)
    references = index(args.reference_results)
    reranker = None if args.skip_reranker else index(args.reranker_selections)
    conditions = [
        "full20",
        "bm25_2",
        "scout_attention2",
        "scout_value2",
        "oracle2",
        *([] if args.skip_reranker else ["reranker2"]),
    ]
    requested_ids = {str(row["example_id"]) for row in rows}
    for label, values in (("scout", scout), ("reference", references)):
        missing = requested_ids - set(values)
        if missing:
            raise SystemExit(f"{label} input is missing IDs: {sorted(missing)[:3]}")
    if reranker is not None and requested_ids - set(reranker):
        raise SystemExit("Reranker selections do not cover the requested shard")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / (
        f"part-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    )
    completed = index(result_path) if result_path.exists() else {}
    unexpected = set(completed) - requested_ids
    if unexpected:
        raise SystemExit(f"Output contains unexpected IDs: {sorted(unexpected)[:3]}")

    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attention_implementation,
    )
    run_config = {
        "reader_cell": args.reader_cell,
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "data": portable_path(args.data),
        "scout_selections": portable_path(args.scout_selections),
        "reranker_selections": (
            None if args.skip_reranker else portable_path(args.reranker_selections)
        ),
        "reference_results": portable_path(args.reference_results),
        "conditions": conditions,
        "max_new_tokens": args.max_new_tokens,
        "attention_implementation": args.attention_implementation
        or "transformers_default",
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
        selection = selections_for_row(
            row,
            scout[example_id],
            references[example_id],
            None if reranker is None else reranker[example_id],
        )
        if list(selection) != conditions:
            raise RuntimeError("Internal condition order changed")
        if example_id in completed:
            missing_fields = [
                f"{condition}_em"
                for condition in selection
                if f"{condition}_em" not in completed[example_id]
            ]
            if missing_fields:
                raise RuntimeError(
                    f"Incomplete resumed row for {example_id}: {missing_fields}"
                )
            continue

        condition_order = rotated(list(selection), int(row["split_rank"]) - 1)
        answers = reference_answers(row)
        gold = {str(item) for item in row["retrieved_support_document_ids"]}
        result: dict[str, Any] = {
            "example_id": example_id,
            "split_rank": int(row["split_rank"]),
            "reader_cell": args.reader_cell,
            "condition_order": condition_order,
            "timing_warmup": position < args.timing_warmup_examples,
        }
        for condition in condition_order:
            selected = selection[condition]
            prompt, _ = render_prompt(loaded.tokenizer, row, selected)
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
            hits = 0 if selected is None else len(set(selected) & gold)
            values = {
                "selected_documents": selected,
                "support_recall": None if selected is None else hits / 2,
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
            for key, value in values.items():
                result[f"{condition}_{key}"] = value

        for condition in selection:
            if condition == "full20":
                continue
            result[f"{condition}_em_minus_full20"] = (
                result[f"{condition}_em"] - result["full20_em"]
            )
            result[f"{condition}_f1_minus_full20"] = (
                result[f"{condition}_f1"] - result["full20_f1"]
            )
        result["scout_value2_em_minus_scout_attention2"] = (
            result["scout_value2_em"] - result["scout_attention2_em"]
        )
        if reranker is not None:
            result["scout_value2_em_minus_reranker2"] = (
                result["scout_value2_em"] - result["reranker2_em"]
            )
        append_jsonl(result_path, result)
        completed[example_id] = result
        print(
            f"{args.reader_cell} scout-reader {len(completed)}/{len(rows)} "
            f"rank={row['split_rank']} full={result['full20_em']:.0f} "
            f"value={result['scout_value2_em']:.0f}",
            flush=True,
        )

    ordered = [completed[str(row["example_id"])] for row in rows]
    timed_rows = [row for row in ordered if not row["timing_warmup"]]
    answer_metrics = {}
    timing_metrics = {}
    for condition_index, condition in enumerate(conditions):
        answer_keys = [f"{condition}_em", f"{condition}_f1", f"{condition}_prompt_tokens"]
        if condition != "full20":
            answer_keys.append(f"{condition}_support_recall")
        answer_metrics[condition] = summarize_metrics(
            ordered,
            answer_keys,
            args.bootstrap_draws,
            args.seed + 100 * condition_index,
        )
        timing_metrics[condition] = summarize_metrics(
            timed_rows,
            [f"{condition}_seconds", f"{condition}_peak_allocated_bytes"],
            args.bootstrap_draws,
            args.seed + 1000 + 100 * condition_index,
        )
    contrast_keys = [
        f"{condition}_em_minus_full20"
        for condition in conditions
        if condition != "full20"
    ]
    contrast_keys.append("scout_value2_em_minus_scout_attention2")
    if reranker is not None:
        contrast_keys.append("scout_value2_em_minus_reranker2")
    contrasts = {
        key: bootstrap_mean(
            [float(row[key]) for row in ordered],
            args.bootstrap_draws,
            args.seed + 2000 + index_value,
        )
        for index_value, key in enumerate(contrast_keys)
    }
    write_json(
        args.output_dir
        / f"summary-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        {
            "run_config": run_config,
            "answer_metrics": answer_metrics,
            "generation_timing_after_warmup": timing_metrics,
            "paired_contrasts": contrasts,
        },
    )
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
