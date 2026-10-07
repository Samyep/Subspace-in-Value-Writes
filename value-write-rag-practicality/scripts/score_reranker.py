#!/usr/bin/env python3
"""Score the frozen MuSiQue top-20 passages with a strong cross-encoder.

The output contains one deterministic top-2 selection per question. It is
reader-independent and can be reused by all four reader jobs.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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
from vw_rag.stats import summarize_metrics
from vw_rag.timing import timed_call



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_pooled_rag_confirm.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/reranker/bge_v2_m3/selections.jsonl",
    )
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_reranker(args: argparse.Namespace):
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, torch_dtype=dtype, **kwargs
        )
    model.to(device)
    model.eval()
    return tokenizer, model, device, dtype


def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits[:, 0]
    return logits[:, -1]


@torch.inference_mode()
def score_question(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    question: str,
    documents: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> tuple[list[float], int]:
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
        outputs = model(**encoded, return_dict=True)
        scores.extend(logits_to_scores(outputs.logits).float().cpu().tolist())
    return scores, input_tokens


def validate_rows(rows: list[dict[str, Any]]) -> None:
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs in evaluation data")
    for row in rows:
        document_ids = [str(item["document_id"]) for item in row["documents"]]
        if len(document_ids) != 20 or len(set(document_ids)) != 20:
            raise ValueError(f"Expected 20 unique passages for {row['example_id']}")


def main() -> None:
    args = parse_args()
    if args.top_k != 2:
        raise SystemExit("The frozen two-hop protocol requires --top-k 2")
    all_rows = load_jsonl(args.data)
    validate_rows(all_rows)
    rows = select_shard(all_rows, args.shard_index, args.num_shards)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    completed = {
        str(row["example_id"]): row
        for row in (load_jsonl(args.output) if args.output.exists() else [])
    }
    expected_ids = {str(row["example_id"]) for row in rows}
    unexpected = set(completed) - expected_ids
    if unexpected:
        raise SystemExit(f"Output contains IDs outside this shard: {sorted(unexpected)[:3]}")

    tokenizer, model, device, dtype = load_reranker(args)
    run_config = {
        "data": portable_path(args.data),
        "model_id": args.model_id,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows_in_shard": len(rows),
        "timing_warmup_examples": args.timing_warmup_examples,
        "runtime": runtime_metadata(),
    }
    write_json(args.output.with_suffix(".run_config.json"), run_config)

    for position, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        documents = list(row["documents"])
        timed = timed_call(
            device,
            lambda: score_question(
                tokenizer,
                model,
                device,
                str(row["question"]),
                documents,
                args.batch_size,
                args.max_length,
            ),
        )
        scores, pair_input_tokens = timed.value
        document_ids = [str(item["document_id"]) for item in documents]
        selected = select_top(document_ids, scores, args.top_k)
        gold = set(str(item) for item in row["retrieved_support_document_ids"])
        hits = len(set(selected) & gold)
        binary_gold = np.asarray([int(item in gold) for item in document_ids])
        auc = (
            float(roc_auc_score(binary_gold, np.asarray(scores))) if gold else None
        )
        result = {
            "example_id": example_id,
            "split_rank": int(row["split_rank"]),
            "document_ids": document_ids,
            "scores": scores,
            "selected_documents": selected,
            "support_recall": hits / 2,
            "exact_top2": float(hits == 2),
            "auc": auc,
            "score_seconds": timed.seconds,
            "pair_input_tokens": pair_input_tokens,
            "peak_allocated_bytes": timed.peak_allocated_bytes,
            "peak_reserved_bytes": timed.peak_reserved_bytes,
            "timing_warmup": position < args.timing_warmup_examples,
        }
        append_jsonl(args.output, result)
        completed[example_id] = result
        print(
            f"reranker {len(completed)}/{len(rows)} rank={row['split_rank']} "
            f"support={hits}/2 seconds={timed.seconds:.3f}",
            flush=True,
        )
        if len(completed) % 25 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ordered = [completed[str(row["example_id"])] for row in rows]
    timed_rows = [row for row in ordered if not row["timing_warmup"]]
    summary = {
        "run_config": run_config,
        "selection_metrics": summarize_metrics(
            ordered,
            ["support_recall", "exact_top2", "auc"],
            args.bootstrap_draws,
            args.seed,
        ),
        "timing_metrics_after_warmup": summarize_metrics(
            timed_rows,
            ["score_seconds", "pair_input_tokens"],
            args.bootstrap_draws,
            args.seed + 100,
        ),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)


if __name__ == "__main__":
    main()
