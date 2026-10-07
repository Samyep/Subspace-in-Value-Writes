#!/usr/bin/env python3
"""Run frozen Full-K, BGE-2, and 3B Value-scout-2 scaling cells."""

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
from vw_rag.model_loading import generate_answer_with_timing, load_model_and_tokenizer
from vw_rag.prompting import evaluate_prediction, reference_answers, render_prompt
from vw_rag.timing import timed_call


CONDITIONS = ("full", "bge2", "scout_value2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-cell", choices=("qwen3_8b", "qwen_14b"), required=True)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_long_context_pool.jsonl"
    )
    parser.add_argument(
        "--bge-selections",
        type=Path,
        default=ROOT / "results/context_scaling/bge/selections.jsonl",
    )
    parser.add_argument(
        "--scout-selections",
        type=Path,
        default=ROOT / "results/context_scaling/scout/selections.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--depths", default="20,40,80,160")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def rotated(items: list[tuple[int, str]], offset: int) -> list[tuple[int, str]]:
    shift = offset % len(items)
    return items[shift:] + items[:shift]


def main() -> None:
    args = parse_args()
    if args.reader_cell not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown reader: {args.reader_cell}")
    depths = [int(item) for item in args.depths.split(",") if item]
    args.output_dir = args.output_dir or (
        ROOT / "results/context_scaling/readers" / args.reader_cell
    )
    for path in (args.data, args.bge_selections, args.scout_selections):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
    rows = select_shard(load_jsonl(args.data), args.shard_index, args.num_shards)
    if args.max_examples:
        rows = rows[: args.max_examples]
    bge = index(args.bge_selections)
    scout = index(args.scout_selections)
    requested = {str(row["example_id"]) for row in rows}
    if requested - set(bge) or requested - set(scout):
        raise RuntimeError("Selector outputs do not cover the reader shard")

    config = get_model_config(args.reader_cell)
    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attention_implementation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / (
        f"part-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    )
    completed = index(result_path) if result_path.exists() else {}
    run_config = {
        "reader_cell": args.reader_cell,
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "data": portable_path(args.data),
        "bge_selections": portable_path(args.bge_selections),
        "scout_selections": portable_path(args.scout_selections),
        "depths": depths,
        "conditions": list(CONDITIONS),
        "max_new_tokens": args.max_new_tokens,
        "attention_implementation": args.attention_implementation,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "runtime": runtime_metadata(),
    }
    write_json(
        args.output_dir
        / f"run_config-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        run_config,
    )

    cells = [(depth, condition) for depth in depths for condition in CONDITIONS]
    for position, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        # Reclaim Python objects once per question. Clearing the CUDA allocator
        # before every one of the 12 cells adds substantial untimed overhead
        # while the model and all generation settings remain unchanged.
        gc.collect()
        if loaded.device.type == "cuda":
            torch.cuda.empty_cache()
        answers = reference_answers(row)
        cell_results: dict[str, dict[str, Any]] = {}
        cell_order = rotated(cells, int(row["split_rank"]) - 1)
        for depth, condition in cell_order:
            documents = row["documents"][:depth]
            document_ids = [str(item["document_id"]) for item in documents]
            if condition == "full":
                selected = document_ids
                selector_seconds = 0.0
                selector_peak = 0
            elif condition == "bge2":
                selected = [
                    str(item)
                    for item in bge[example_id]["by_depth"][str(depth)][
                        "selected_documents"
                    ]
                ]
                selector_seconds = float(
                    bge[example_id]["by_depth"][str(depth)]["score_seconds"]
                )
                selector_peak = bge[example_id]["by_depth"][str(depth)][
                    "peak_allocated_bytes"
                ]
            else:
                selected = [
                    str(item)
                    for item in scout[example_id]["by_depth"][str(depth)][
                        "selected_documents"
                    ]
                ]
                selector_seconds = float(
                    scout[example_id]["by_depth"][str(depth)]["selector_seconds"]
                )
                selector_peak = scout[example_id]["by_depth"][str(depth)][
                    "peak_allocated_bytes"
                ]
            if not set(selected).issubset(document_ids):
                raise RuntimeError(f"Invalid selection for {example_id} K={depth}")
            prompt, _ = render_prompt(loaded.tokenizer, row, selected)
            timed = timed_call(
                loaded.device,
                lambda prompt=prompt: generate_answer_with_timing(
                    loaded, prompt, args.max_new_tokens
                ),
            )
            (
                raw,
                answer,
                prompt_tokens,
                output_tokens,
                ttft_seconds,
                decode_seconds,
                internal_total_seconds,
                tokenization_seconds,
            ) = timed.value
            metrics = evaluate_prediction(answer, answers)
            selected_set = set(selected)
            selected_docs = [item for item in documents if item["document_id"] in selected_set]
            cell_results[f"k{depth}_{condition}"] = {
                "depth": depth,
                "condition": condition,
                "selected_documents": None if condition == "full" else selected,
                "support_recall": sum(bool(item["is_support"]) for item in selected_docs) / 2,
                "first_hop_hit": float(
                    any("first" in item["support_roles"] for item in selected_docs)
                ),
                "terminal_hop_hit": float(
                    any("terminal" in item["support_roles"] for item in selected_docs)
                ),
                "raw": raw,
                "answer": answer,
                "em": metrics["em"],
                "f1": metrics["f1"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "tokenization_seconds": tokenization_seconds,
                "ttft_seconds": ttft_seconds,
                "decode_seconds": decode_seconds,
                "reader_internal_seconds": internal_total_seconds,
                "reader_seconds": timed.seconds,
                "selector_seconds": selector_seconds,
                "end_to_end_seconds": selector_seconds + timed.seconds,
                "reader_peak_allocated_bytes": timed.peak_allocated_bytes,
                "reader_peak_reserved_bytes": timed.peak_reserved_bytes,
                "selector_peak_allocated_bytes": selector_peak,
            }
        result = {
            "example_id": example_id,
            "split_rank": int(row["split_rank"]),
            "reader_cell": args.reader_cell,
            "timing_warmup": position < args.timing_warmup_examples,
            "condition_order": [f"k{depth}_{condition}" for depth, condition in cell_order],
            "cells": {
                f"k{depth}_{condition}": cell_results[f"k{depth}_{condition}"]
                for depth in depths
                for condition in CONDITIONS
            },
        }
        append_jsonl(result_path, result)
        completed[example_id] = result
        print(
            f"{args.reader_cell} context scaling {len(completed)}/{len(rows)}",
            flush=True,
        )

    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
