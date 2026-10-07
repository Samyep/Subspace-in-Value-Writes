#!/usr/bin/env python3
"""Score passages with an exact decoder-prefix value-write scout.

The default scout is Llama-3.2-3B through layer index 17 (18 decoder
blocks). The script extracts Value-2 and Attention-2 from one shared forward
pass and checks both selections against the released full-model run.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    parser.add_argument("--scout-cell", choices=sorted(MODEL_CONFIGS), default="llama_3b")
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument("--direction", type=Path, default=None)
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/scout/llama_3b_l17/selections.jsonl",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--truncate-to-layers",
        type=int,
        default=0,
        help="Number of decoder blocks to retain; 0 keeps through the writer window.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    parser.add_argument(
        "--require-exact-selection-agreement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require exact agreement with the released run. Keep this disabled "
            "when hardware or software versions differ; use audit_scout_prefix.py "
            "for a same-runtime full-model comparison."
        ),
    )
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Score shape mismatch: {left.shape} versus {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    args = parse_args()
    config = get_model_config(args.scout_cell, args.model_id)
    args.direction = args.direction or (
        ROOT / "calibration" / args.scout_cell / "frozen_direction.npz"
    )
    args.reference_results = args.reference_results or (
        ROOT / "reference_results" / args.scout_cell / "per_example.jsonl"
    )
    for path in (args.data, args.direction, args.reference_results):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    truncate_to_layers = args.truncate_to_layers or max(config.writer_window) + 1
    if truncate_to_layers <= max(config.writer_window):
        raise SystemExit(
            "The truncated model must retain every declared writer-window layer"
        )

    all_rows = load_jsonl(args.data)
    rows = select_shard(all_rows, args.shard_index, args.num_shards)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    references = index(args.reference_results)
    requested_ids = {str(row["example_id"]) for row in rows}
    if requested_ids - set(references):
        raise SystemExit("Full-model references do not cover the requested shard")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = index(args.output) if args.output.exists() else {}
    unexpected = set(completed) - requested_ids
    if unexpected:
        raise SystemExit(f"Output contains unexpected IDs: {sorted(unexpected)[:3]}")

    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
        truncate_to_layers=truncate_to_layers,
    )
    direction = load_frozen_direction(args.direction, config.hidden_size)
    extractor = DocumentFeatureExtractor(loaded.model, config.writer_window)
    run_config = {
        "scout_cell": args.scout_cell,
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "writer_window": list(config.writer_window),
        "original_num_hidden_layers": loaded.original_num_hidden_layers,
        "active_num_hidden_layers": loaded.active_num_hidden_layers,
        "exact_prefix_truncation": True,
        "data": portable_path(args.data),
        "direction": portable_path(args.direction),
        "full_model_reference": portable_path(args.reference_results),
        "max_prompt_tokens": args.max_prompt_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows_in_shard": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "require_released_reference_selection_agreement": (
            args.require_exact_selection_agreement
        ),
        "runtime": runtime_metadata(),
    }
    write_json(args.output.with_suffix(".run_config.json"), run_config)

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
            attention_scores = features.attention_mass.astype(np.float64)
            value_selected = select_top(
                features.document_ids, value_scores.tolist(), top_k=2
            )
            attention_selected = select_top(
                features.document_ids, attention_scores.tolist(), top_k=2
            )
            reference = references[example_id]
            reference_ids = [str(item) for item in reference["document_ids"]]
            if features.document_ids != reference_ids:
                raise RuntimeError(f"Document order changed for {example_id}")
            value_reference = [str(item) for item in reference["value_write_documents"]]
            attention_reference = [str(item) for item in reference["attention_documents"]]
            value_match = value_selected == value_reference
            attention_match = attention_selected == attention_reference
            if args.require_exact_selection_agreement and not (
                value_match and attention_match
            ):
                raise RuntimeError(
                    f"Truncated/full selection mismatch for {example_id}: "
                    f"value={value_match}, attention={attention_match}"
                )

            gold = {str(item) for item in row["retrieved_support_document_ids"]}
            value_hits = len(set(value_selected) & gold)
            attention_hits = len(set(attention_selected) & gold)
            result = {
                "example_id": example_id,
                "split_rank": int(row["split_rank"]),
                "scout_cell": args.scout_cell,
                "document_ids": features.document_ids,
                "value_scores": value_scores.tolist(),
                "attention_scores": attention_scores.tolist(),
                "value_selected_documents": value_selected,
                "attention_selected_documents": attention_selected,
                "value_support_recall": value_hits / 2,
                "attention_support_recall": attention_hits / 2,
                "value_exact_top2": float(value_hits == 2),
                "attention_exact_top2": float(attention_hits == 2),
                "value_full_model_selection_match": value_match,
                "attention_full_model_selection_match": attention_match,
                "value_full_model_max_abs_score_difference": max_abs_difference(
                    value_scores, np.asarray(reference["value_write_scores"])
                ),
                "attention_full_model_max_abs_score_difference": max_abs_difference(
                    attention_scores, np.asarray(reference["attention_mass_scores"])
                ),
                "prompt_tokens": features.prompt_tokens,
                "scout_seconds": timed.seconds,
                "peak_allocated_bytes": timed.peak_allocated_bytes,
                "peak_reserved_bytes": timed.peak_reserved_bytes,
                "baseline_allocated_bytes": timed.baseline_allocated_bytes,
                "timing_warmup": position < args.timing_warmup_examples,
            }
            append_jsonl(args.output, result)
            completed[example_id] = result
            print(
                f"{args.scout_cell} truncated scout {len(completed)}/{len(rows)} "
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
            "selection_agreement": {
                "value": float(
                    np.mean(
                        [row["value_full_model_selection_match"] for row in ordered]
                    )
                ),
                "attention": float(
                    np.mean(
                        [
                            row["attention_full_model_selection_match"]
                            for row in ordered
                        ]
                    )
                ),
            },
            "maximum_score_difference": {
                "value": max(
                    row["value_full_model_max_abs_score_difference"]
                    for row in ordered
                ),
                "attention": max(
                    row["attention_full_model_max_abs_score_difference"]
                    for row in ordered
                ),
            },
            "selection_metrics": summarize_metrics(
                ordered,
                [
                    "value_support_recall",
                    "attention_support_recall",
                    "value_exact_top2",
                    "attention_exact_top2",
                ],
                args.bootstrap_draws,
                args.seed,
            ),
            "shared_pass_timing_after_warmup": summarize_metrics(
                timed_rows,
                ["scout_seconds", "prompt_tokens", "peak_allocated_bytes"],
                args.bootstrap_draws,
                args.seed + 100,
            ),
        }
        write_json(args.output.with_suffix(".summary.json"), summary)
    finally:
        extractor.close()
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
