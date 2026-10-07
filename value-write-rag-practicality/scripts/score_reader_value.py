#!/usr/bin/env python3
"""Score a generic frozen retrieval set with a reader-specific Value direction."""

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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--direction", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path) if path.exists() else []
    result = {str(row["example_id"]): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate IDs in {path}")
    return result


def main() -> None:
    args = parse_args()
    config = get_model_config(args.cell)
    args.direction = args.direction or ROOT / "calibration" / args.cell / "frozen_direction.npz"
    rows = select_shard(load_jsonl(args.data), args.shard_index, args.num_shards)
    if args.max_examples:
        rows = rows[: args.max_examples]
    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    direction = load_frozen_direction(args.direction, config.hidden_size)
    extractor = DocumentFeatureExtractor(loaded.model, config.writer_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = index(args.output)
    run_config = {
        "cell": args.cell,
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "writer_window": list(config.writer_window),
        "data": portable_path(args.data),
        "direction": portable_path(args.direction),
        "max_prompt_tokens": args.max_prompt_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows": len(rows),
        "runtime": runtime_metadata(),
    }
    write_json(args.output.with_suffix(".run_config.json"), run_config)
    try:
        for row in rows:
            example_id = str(row["example_id"])
            if example_id in completed:
                continue
            timed = timed_call(
                loaded.device,
                lambda row=row: extract_document_features(
                    extractor, loaded, row, args.max_prompt_tokens
                ),
            )
            features = timed.value
            if features is None:
                raise RuntimeError(f"Prompt exceeds limit for {example_id}")
            scores = value_write_scores(features, direction)
            selected = select_top(features.document_ids, scores.tolist(), 2)
            gold = set(str(item) for item in row["retrieved_support_document_ids"])
            result = {
                "example_id": example_id,
                "split_rank": int(row["split_rank"]),
                "cell": args.cell,
                "document_ids": features.document_ids,
                "value_scores": scores.tolist(),
                "selected_documents": selected,
                "support_recall": len(set(selected) & gold) / 2,
                "prompt_tokens": features.prompt_tokens,
                "score_seconds": timed.seconds,
                "peak_allocated_bytes": timed.peak_allocated_bytes,
            }
            append_jsonl(args.output, result)
            completed[example_id] = result
            print(f"{args.cell} held-out Value {len(completed)}/{len(rows)}", flush=True)
            del features
            gc.collect()
            if loaded.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        extractor.close()
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
