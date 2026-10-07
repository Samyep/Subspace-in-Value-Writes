#!/usr/bin/env python3
"""Measure the full-context value-write/attention scoring pass.

This job loads the reader with eager attention, extracts value-write features,
and validates its selections against the released reference run. It does not
generate answers; answer latency is measured separately with the reader's
normal optimized attention implementation.
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
from vw_rag.model_loading import load_model_and_tokenizer
from vw_rag.prompting import select_top
from vw_rag.stats import summarize_metrics
from vw_rag.timing import timed_call
from vw_rag.value_write import (
    DocumentFeatureExtractor,
    extract_document_features,
    load_frozen_direction,
    value_write_scores,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument("--direction", type=Path, default=None)
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    parser.add_argument("--allow-selection-mismatch", action="store_true")
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def main() -> None:
    args = parse_args()
    config = get_model_config(args.cell, args.model_id)
    args.direction = args.direction or ROOT / "calibration" / args.cell / "frozen_direction.npz"
    args.reference_results = args.reference_results or (
        ROOT / "reference_results" / args.cell / "per_example.jsonl"
    )
    args.output_dir = args.output_dir or ROOT / "results/cost" / args.cell / "scoring"
    for path in (args.data, args.direction, args.reference_results):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    rows = select_shard(load_jsonl(args.data), args.shard_index, args.num_shards)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    references = index(args.reference_results)
    missing = {str(row["example_id"]) for row in rows} - set(references)
    if missing:
        raise SystemExit(f"Reference results are missing IDs: {sorted(missing)[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"part-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    completed = index(result_path) if result_path.exists() else {}
    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    direction = load_frozen_direction(args.direction, config.hidden_size)
    extractor = DocumentFeatureExtractor(loaded.model, config.writer_window)
    run_config = {
        "cell": args.cell,
        "model_id": config.model_id,
        "writer_window": list(config.writer_window),
        "data": portable_path(args.data),
        "direction": portable_path(args.direction),
        "reference_results": portable_path(args.reference_results),
        "max_prompt_tokens": args.max_prompt_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows_in_shard": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "attention_implementation": "eager",
        "runtime": runtime_metadata(),
    }
    write_json(
        args.output_dir
        / f"run_config-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        run_config,
    )

    try:
        for position, row in enumerate(rows):
            example_id = str(row["example_id"])
            if example_id in completed:
                continue
            timed = timed_call(
                loaded.device,
                lambda: extract_document_features(
                    extractor, loaded, row, args.max_prompt_tokens
                ),
            )
            features = timed.value
            if features is None:
                raise RuntimeError(f"Prompt exceeds frozen token limit: {example_id}")
            value_scores = value_write_scores(features, direction)
            value_selected = select_top(
                features.document_ids, value_scores.tolist(), top_k=2
            )
            attention_selected = select_top(
                features.document_ids, features.attention_mass.tolist(), top_k=2
            )
            reference = references[example_id]
            value_match = value_selected == [
                str(item) for item in reference["value_write_documents"]
            ]
            attention_match = attention_selected == [
                str(item) for item in reference["attention_documents"]
            ]
            if not args.allow_selection_mismatch and not (value_match and attention_match):
                raise RuntimeError(
                    f"Selection mismatch for {example_id}: "
                    f"value={value_match}, attention={attention_match}"
                )
            result = {
                "example_id": example_id,
                "split_rank": int(row["split_rank"]),
                "cell": args.cell,
                "prompt_tokens": features.prompt_tokens,
                "document_ids": features.document_ids,
                "document_token_lengths": features.token_lengths.tolist(),
                "value_scores": value_scores.tolist(),
                "attention_scores": features.attention_mass.tolist(),
                "value_selected_documents": value_selected,
                "attention_selected_documents": attention_selected,
                "value_reference_match": value_match,
                "attention_reference_match": attention_match,
                "score_seconds": timed.seconds,
                "peak_allocated_bytes": timed.peak_allocated_bytes,
                "peak_reserved_bytes": timed.peak_reserved_bytes,
                "baseline_allocated_bytes": timed.baseline_allocated_bytes,
                "timing_warmup": position < args.timing_warmup_examples,
            }
            append_jsonl(result_path, result)
            completed[example_id] = result
            print(
                f"{args.cell} scoring {len(completed)}/{len(rows)} "
                f"rank={row['split_rank']} seconds={timed.seconds:.3f}",
                flush=True,
            )
            del features
            gc.collect()
            if loaded.device.type == "cuda":
                torch.cuda.empty_cache()

        ordered = [completed[str(row["example_id"])] for row in rows]
        timed_rows = [row for row in ordered if not row["timing_warmup"]]
        summary = {
            "run_config": run_config,
            "selection_reproduction": {
                "value_all_match": all(row["value_reference_match"] for row in ordered),
                "attention_all_match": all(
                    row["attention_reference_match"] for row in ordered
                ),
            },
            "timing_after_warmup": summarize_metrics(
                timed_rows,
                ["score_seconds", "prompt_tokens", "peak_allocated_bytes"],
                args.bootstrap_draws,
                args.seed,
            ),
        }
        write_json(
            args.output_dir
            / f"summary-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
            summary,
        )
    finally:
        extractor.close()
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
