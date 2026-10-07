#!/usr/bin/env python3
"""Score every frozen retrieval depth with BGE reranker v2 m3."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import (
    portable_path,
    append_jsonl,
    load_jsonl,
    runtime_metadata,
    select_shard,
    write_json,
)
from vw_rag.model_loading import resolve_device, resolve_dtype
from vw_rag.prompting import select_top
from vw_rag.timing import timed_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_long_context_pool.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/context_scaling/bge/selections.jsonl"
    )
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--depths", default="20,40,80,160")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
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


def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits[:, 0]
    return logits[:, -1]


@torch.inference_mode()
def score_documents(tokenizer, model, device, question, documents, batch_size, max_length):
    passages = [f"{item['title']}\n{item['text']}" for item in documents]
    scores: list[float] = []
    input_tokens = 0
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        encoded = tokenizer(
            [question] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_tokens += int(encoded["attention_mask"].sum().item())
        encoded = {key: value.to(device) for key, value in encoded.items()}
        scores.extend(logits_to_scores(model(**encoded).logits).float().cpu().tolist())
    return scores, input_tokens


def main() -> None:
    args = parse_args()
    depths = [int(item) for item in args.depths.split(",") if item]
    all_rows = load_jsonl(args.data)
    rows = select_shard(all_rows, args.shard_index, args.num_shards)
    if args.max_examples:
        rows = rows[: args.max_examples]
    for row in rows:
        if len(row["documents"]) < max(depths):
            raise RuntimeError(f"Insufficient documents for {row['example_id']}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, torch_dtype=dtype, **kwargs
        )
    model.to(device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = index(args.output)
    expected = {str(row["example_id"]) for row in rows}
    if set(completed) - expected:
        raise RuntimeError("Output contains IDs outside the requested shard")
    run_config = {
        "data": portable_path(args.data),
        "model_id": args.model_id,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "depths": depths,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "runtime": runtime_metadata(),
    }
    write_json(args.output.with_suffix(".run_config.json"), run_config)

    for position, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        by_depth = {}
        for depth in rotated(depths, int(row["split_rank"]) - 1):
            documents = row["documents"][:depth]
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            timed = timed_call(
                device,
                lambda documents=documents: score_documents(
                    tokenizer,
                    model,
                    device,
                    str(row["question"]),
                    documents,
                    args.batch_size,
                    args.max_length,
                ),
            )
            scores, pair_tokens = timed.value
            document_ids = [str(item["document_id"]) for item in documents]
            selected = select_top(document_ids, scores, 2)
            selected_docs = {document_ids.index(item) for item in selected}
            support_hits = sum(bool(documents[i]["is_support"]) for i in selected_docs)
            first_hit = any("first" in documents[i]["support_roles"] for i in selected_docs)
            terminal_hit = any("terminal" in documents[i]["support_roles"] for i in selected_docs)
            by_depth[str(depth)] = {
                "document_ids": document_ids,
                "scores": scores,
                "selected_documents": selected,
                "support_recall": support_hits / 2,
                "first_hop_hit": float(first_hit),
                "terminal_hop_hit": float(terminal_hit),
                "score_seconds": timed.seconds,
                "pair_input_tokens": pair_tokens,
                "peak_allocated_bytes": timed.peak_allocated_bytes,
                "peak_reserved_bytes": timed.peak_reserved_bytes,
            }
        result = {
            "example_id": example_id,
            "split_rank": int(row["split_rank"]),
            "timing_warmup": position < args.timing_warmup_examples,
            "by_depth": {str(depth): by_depth[str(depth)] for depth in depths},
        }
        append_jsonl(args.output, result)
        completed[example_id] = result
        print(f"BGE long-context {len(completed)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
