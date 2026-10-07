#!/usr/bin/env python3
"""Score every frozen retrieval depth with the exact 18-block 3B scout."""

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

from vw_rag.config import get_model_config
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
from vw_rag.timing import timed_call
from vw_rag.value_write import (
    FinalQueryDocumentFeatureExtractor,
    extract_document_features_final_query,
    load_frozen_direction,
    value_write_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_long_context_pool.jsonl"
    )
    parser.add_argument(
        "--direction", type=Path, default=ROOT / "calibration/llama_3b/frozen_direction.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/context_scaling/scout/selections.jsonl"
    )
    parser.add_argument("--depths", default="20,40,80,160")
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path) if path.exists() else []
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def rotated(items: list[int], offset: int) -> list[int]:
    shift = offset % len(items)
    return items[shift:] + items[:shift]


def main() -> None:
    args = parse_args()
    depths = [int(item) for item in args.depths.split(",") if item]
    config = get_model_config("llama_3b")
    rows = select_shard(load_jsonl(args.data), args.shard_index, args.num_shards)
    if args.max_examples:
        rows = rows[: args.max_examples]
    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="sdpa",
        truncate_to_layers=max(config.writer_window) + 1,
    )
    direction = load_frozen_direction(args.direction, config.hidden_size)
    extractor = FinalQueryDocumentFeatureExtractor(loaded.model, config.writer_window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = index(args.output)
    expected = {str(row["example_id"]) for row in rows}
    if set(completed) - expected:
        raise RuntimeError("Output contains IDs outside the requested shard")
    run_config = {
        "data": portable_path(args.data),
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "writer_window": list(config.writer_window),
        "active_num_hidden_layers": loaded.active_num_hidden_layers,
        "direction": portable_path(args.direction),
        "depths": depths,
        "max_prompt_tokens": args.max_prompt_tokens,
        "attention_implementation": "sdpa with exact final-query reconstruction",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "runtime": runtime_metadata(),
    }
    write_json(args.output.with_suffix(".run_config.json"), run_config)

    try:
        for position, row in enumerate(rows):
            example_id = str(row["example_id"])
            if example_id in completed:
                continue
            by_depth = {}
            for depth in rotated(depths, int(row["split_rank"]) - 1):
                prefix = dict(row)
                prefix["documents"] = row["documents"][:depth]
                gc.collect()
                if loaded.device.type == "cuda":
                    torch.cuda.empty_cache()
                timed = timed_call(
                    loaded.device,
                    lambda prefix=prefix: extract_document_features_final_query(
                        extractor, loaded, prefix, args.max_prompt_tokens
                    ),
                )
                features = timed.value
                if features is None:
                    raise RuntimeError(f"Prompt exceeds limit for {example_id} at K={depth}")
                scores = value_write_scores(features, direction)
                selected = select_top(features.document_ids, scores.tolist(), 2)
                document_by_id = {
                    str(item["document_id"]): item for item in prefix["documents"]
                }
                selected_docs = [document_by_id[item] for item in selected]
                by_depth[str(depth)] = {
                    "document_ids": features.document_ids,
                    "value_scores": scores.tolist(),
                    "selected_documents": selected,
                    "support_recall": sum(bool(item["is_support"]) for item in selected_docs) / 2,
                    "first_hop_hit": float(
                        any("first" in item["support_roles"] for item in selected_docs)
                    ),
                    "terminal_hop_hit": float(
                        any("terminal" in item["support_roles"] for item in selected_docs)
                    ),
                    "prompt_tokens": features.prompt_tokens,
                    "selector_seconds": timed.seconds,
                    "peak_allocated_bytes": timed.peak_allocated_bytes,
                    "peak_reserved_bytes": timed.peak_reserved_bytes,
                }
                del features
            result = {
                "example_id": example_id,
                "split_rank": int(row["split_rank"]),
                "timing_warmup": position < args.timing_warmup_examples,
                "by_depth": {str(depth): by_depth[str(depth)] for depth in depths},
            }
            append_jsonl(args.output, result)
            completed[example_id] = result
            print(f"3B Value scout long-context {len(completed)}/{len(rows)}", flush=True)
    finally:
        extractor.close()
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
