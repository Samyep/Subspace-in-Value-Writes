#!/usr/bin/env python3
"""Validate frozen extension data and completed non-GPU analysis."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    base = rows(ROOT / "data/musique_pooled_rag_confirm.jsonl")
    long_rows = rows(ROOT / "data/musique_long_context_pool.jsonl")
    assert len(base) == len(long_rows) == 400
    assert [item["example_id"] for item in base] == [
        item["example_id"] for item in long_rows
    ]
    for original, extended in zip(base, long_rows):
        assert extended["depths"] == [20, 40, 80, 160]
        assert len(extended["documents"]) == 160
        assert len({item["document_id"] for item in extended["documents"]}) == 160
        assert len({item["corpus_document_id"] for item in extended["documents"]}) == 160
        for left, right in zip(original["documents"], extended["documents"][:20]):
            for key in ("document_id", "corpus_document_id", "title", "text"):
                assert left[key] == right[key]
        assert (
            extended["first_hop_corpus_document_id"]
            != extended["terminal_hop_corpus_document_id"]
        )
        for model in ("qwen3_8b", "qwen_14b", "llama_3b"):
            counts = extended["prompt_tokens_by_model_and_depth"][model]
            values = [counts[str(depth)] for depth in extended["depths"]]
            assert values == sorted(values)
            assert max(values) < 32768

    hotpot = rows(ROOT / "data/hotpot_hybrid_calibration.jsonl")
    assert len(hotpot) == 200
    assert len({item["example_id"] for item in hotpot}) == 200
    assert all(not item["direction_calibration_overlap"] for item in hotpot)
    assert all(len(item["documents"]) == 10 for item in hotpot)
    assert all(len(item["retrieved_support_document_ids"]) == 2 for item in hotpot)

    analysis = rows(ROOT / "results/evidence_analysis/per_example.jsonl")
    assert len(analysis) == 1600
    assert len({(item["reader"], item["example_id"]) for item in analysis}) == 1600
    print("extension validation passed: 400 long rows, 200 calibration rows, 1600 analysis rows")


if __name__ == "__main__":
    main()
