#!/usr/bin/env python3
"""Validate frozen inputs, directions, and released reference rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.config import MODEL_CONFIGS
from vw_rag.io_utils import load_jsonl



def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise AssertionError(f"Duplicate example IDs in {label}")
    return result


def main() -> None:
    data_path = ROOT / "data/musique_pooled_rag_confirm.jsonl"
    summary_path = data_path.with_suffix(".summary.json")
    rows = load_jsonl(data_path)
    assert len(rows) == 400
    by_id = index(rows, "evaluation data")
    assert [int(row["split_rank"]) for row in rows] == list(range(1, 401))
    for row in rows:
        documents = row["documents"]
        document_ids = [str(item["document_id"]) for item in documents]
        assert len(document_ids) == 20 and len(set(document_ids)) == 20
        assert len(row["support_corpus_document_ids"]) == 2
        labeled = [str(item["document_id"]) for item in documents if item["is_support"]]
        assert labeled == [str(item) for item in row["retrieved_support_document_ids"]]
        assert float(row["retrieval_support_recall"]) == len(labeled) / 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_count"] == 400

    direction_manifest = json.loads(
        (ROOT / "calibration/directions.json").read_text(encoding="utf-8")
    )["directions"]
    assert set(direction_manifest) == set(MODEL_CONFIGS)
    for cell, config in MODEL_CONFIGS.items():
        item = direction_manifest[cell]
        assert item["model_id"] == config.model_id
        assert item["writer_window"] == list(config.writer_window)
        path = ROOT / item["artifact"]
        assert path.is_file()
        with np.load(path) as frozen:
            assert frozen["mean"].shape == (config.hidden_size,)
            assert frozen["component"].shape == (config.hidden_size,)
            assert int(frozen["sign"]) in {-1, 1}

    expected_ids = set(by_id)
    aggregate = json.loads(
        (ROOT / "reference_results/aggregate_summary.json").read_text(encoding="utf-8")
    )
    for source in aggregate["sources"].values():
        assert (ROOT / source["summary"]).is_file()
        assert (ROOT / source["per_example"]).is_file()

    for cell in MODEL_CONFIGS:
        result_path = ROOT / "reference_results" / cell / "per_example.jsonl"
        reference = index(load_jsonl(result_path), f"reference:{cell}")
        assert set(reference) == expected_ids
        for example_id, result in reference.items():
            document_ids = {
                str(item["document_id"]) for item in by_id[example_id]["documents"]
            }
            assert len(result["value_write_documents"]) == 2
            assert len(result["attention_documents"]) == 2
            assert set(result["value_write_documents"]).issubset(document_ids)
            assert set(result["attention_documents"]).issubset(document_ids)
    print(
        "PASS: 400 frozen questions, four directions, four 400-row reference "
        "runs, and their structural constraints are consistent."
    )


if __name__ == "__main__":
    main()
