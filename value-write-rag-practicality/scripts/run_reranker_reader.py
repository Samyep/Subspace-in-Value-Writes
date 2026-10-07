#!/usr/bin/env python3
"""Generate answers from cross-encoder-selected top-2 passages.

Run one reader cell per machine. The script is resumable and supports sharding.
It intentionally does not recompute value-write features; the released
reference rows provide the paired Full-20, Value-2, and Attention-2 outcomes.
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
    parser.add_argument(
        "--reranker-selections",
        type=Path,
        default=ROOT / "results/reranker/bge_v2_m3/selections.jsonl",
    )
    parser.add_argument("--reference-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing-warmup-examples", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    return parser.parse_args()


def indexed_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate example IDs in {path}")
    return result


def main() -> None:
    args = parse_args()
    config = get_model_config(args.cell, args.model_id)
    if args.output_dir is None:
        args.output_dir = ROOT / "results/readers" / args.cell / "reranker"
    if args.reference_results is None:
        args.reference_results = (
            ROOT / "reference_results" / args.cell / "per_example.jsonl"
        )
    for path in (args.data, args.reranker_selections, args.reference_results):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    all_rows = load_jsonl(args.data)
    rows = select_shard(all_rows, args.shard_index, args.num_shards)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    reranker = indexed_rows(args.reranker_selections)
    references = indexed_rows(args.reference_results)
    requested_ids = {str(row["example_id"]) for row in rows}
    for name, index in (("reranker", reranker), ("reference", references)):
        missing = requested_ids - set(index)
        if missing:
            raise SystemExit(f"{name} input is missing IDs: {sorted(missing)[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"part-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    completed = indexed_rows(result_path) if result_path.exists() else {}
    unexpected = set(completed) - requested_ids
    if unexpected:
        raise SystemExit(f"Output contains unexpected IDs: {sorted(unexpected)[:3]}")

    loaded = load_model_and_tokenizer(
        config.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=None,
    )
    run_config = {
        "cell": args.cell,
        "model_id": config.model_id,
        "model_commit": getattr(loaded.config, "_commit_hash", None),
        "data": portable_path(args.data),
        "reranker_selections": portable_path(args.reranker_selections),
        "reference_results": portable_path(args.reference_results),
        "max_new_tokens": args.max_new_tokens,
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
        document_ids = [str(item["document_id"]) for item in row["documents"]]
        selected = [str(item) for item in reranker[example_id]["selected_documents"]]
        if len(selected) != 2 or not set(selected).issubset(document_ids):
            raise RuntimeError(f"Invalid reranker selection for {example_id}")
        prompt, _ = render_prompt(loaded.tokenizer, row, selected)
        full_prompt, _ = render_prompt(loaded.tokenizer, row)
        timed = timed_call(
            loaded.device,
            lambda: generate_answer(loaded, prompt, args.max_new_tokens),
        )
        raw, answer, prompt_tokens, output_tokens = timed.value
        full_prompt_tokens = len(
            loaded.tokenizer.encode(full_prompt, add_special_tokens=False)
        )
        gold = set(str(item) for item in row["retrieved_support_document_ids"])
        hits = len(set(selected) & gold)
        metrics = evaluate_prediction(answer, reference_answers(row))
        reference = references[example_id]
        result = {
            "example_id": example_id,
            "split_rank": int(row["split_rank"]),
            "cell": args.cell,
            "selected_documents": selected,
            "support_recall": hits / 2,
            "exact_top2": float(hits == 2),
            "raw": raw,
            "answer": answer,
            "em": metrics["em"],
            "f1": metrics["f1"],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "full20_prompt_tokens": full_prompt_tokens,
            "prompt_token_ratio": prompt_tokens / full_prompt_tokens,
            "generation_seconds": timed.seconds,
            "peak_allocated_bytes": timed.peak_allocated_bytes,
            "peak_reserved_bytes": timed.peak_reserved_bytes,
            "baseline_allocated_bytes": timed.baseline_allocated_bytes,
            "timing_warmup": position < args.timing_warmup_examples,
            "reference_full20_em": float(reference["baseline_em"]),
            "reference_full20_f1": float(reference["baseline_f1"]),
            "reference_value2_em": float(reference["value_write_em"]),
            "reference_value2_f1": float(reference["value_write_f1"]),
            "reference_attention2_em": float(reference["attention_em"]),
            "reference_attention2_f1": float(reference["attention_f1"]),
        }
        result["em_minus_full20"] = result["em"] - result["reference_full20_em"]
        result["em_minus_value2"] = result["em"] - result["reference_value2_em"]
        result["em_minus_attention2"] = (
            result["em"] - result["reference_attention2_em"]
        )
        append_jsonl(result_path, result)
        completed[example_id] = result
        print(
            f"{args.cell} reranker-reader {len(completed)}/{len(rows)} "
            f"rank={row['split_rank']} em={metrics['em']:.0f} "
            f"seconds={timed.seconds:.3f}",
            flush=True,
        )
        if len(completed) % 20 == 0:
            gc.collect()
            if loaded.device.type == "cuda":
                torch.cuda.empty_cache()

    ordered = [completed[str(row["example_id"])] for row in rows]
    timed_rows = [row for row in ordered if not row["timing_warmup"]]
    summary = {
        "run_config": run_config,
        "answer_and_selection_metrics": summarize_metrics(
            ordered,
            [
                "em",
                "f1",
                "support_recall",
                "exact_top2",
                "prompt_token_ratio",
                "em_minus_full20",
                "em_minus_value2",
                "em_minus_attention2",
            ],
            args.bootstrap_draws,
            args.seed,
        ),
        "timing_after_warmup": summarize_metrics(
            timed_rows,
            ["generation_seconds", "prompt_tokens", "output_tokens"],
            args.bootstrap_draws,
            args.seed + 100,
        ),
        "paired_f1_minus_full20": bootstrap_mean(
            [row["f1"] - row["reference_full20_f1"] for row in ordered],
            args.bootstrap_draws,
            args.seed + 200,
        ),
    }
    write_json(
        args.output_dir
        / f"summary-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        summary,
    )
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
