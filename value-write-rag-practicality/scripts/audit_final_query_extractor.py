#!/usr/bin/env python3
"""Audit long-context final-query reconstruction against dense eager attention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.config import get_model_config
from vw_rag.io_utils import load_jsonl, write_json
from vw_rag.model_loading import load_model_and_tokenizer
from vw_rag.prompting import select_top
from vw_rag.value_write import (
    DocumentFeatureExtractor,
    FinalQueryDocumentFeatureExtractor,
    extract_document_features,
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
        "--output", type=Path, default=ROOT / "results/context_scaling/final_query_audit.json"
    )
    parser.add_argument("--max-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_model_config("llama_3b")
    rows = load_jsonl(args.data)[: args.max_examples]
    loaded = load_model_and_tokenizer(
        config.model_id,
        attn_implementation="eager",
        truncate_to_layers=max(config.writer_window) + 1,
    )
    direction = load_frozen_direction(args.direction, config.hidden_size)
    result = []
    for row in rows:
        prefix = dict(row)
        prefix["documents"] = row["documents"][:20]
        dense_extractor = DocumentFeatureExtractor(loaded.model, config.writer_window)
        dense = extract_document_features(dense_extractor, loaded, prefix, 4096)
        dense_extractor.close()
        if dense is None:
            raise RuntimeError("Unexpected long prompt in K=20 audit")

        final_extractor = FinalQueryDocumentFeatureExtractor(
            loaded.model, config.writer_window
        )
        reconstructed = extract_document_features_final_query(
            final_extractor, loaded, prefix, 4096
        )
        final_extractor.close()
        if reconstructed is None:
            raise RuntimeError("Unexpected long prompt in final-query audit")
        dense_scores = value_write_scores(dense, direction)
        reconstructed_scores = value_write_scores(reconstructed, direction)
        configs = {
            id(loaded.model.config): loaded.model.config,
            id(loaded.model.model.config): loaded.model.model.config,
        }
        for model_config in configs.values():
            model_config._attn_implementation = "sdpa"
        sdpa_extractor = FinalQueryDocumentFeatureExtractor(
            loaded.model, config.writer_window
        )
        sdpa = extract_document_features_final_query(sdpa_extractor, loaded, prefix, 4096)
        sdpa_extractor.close()
        for model_config in configs.values():
            model_config._attn_implementation = "eager"
        if sdpa is None:
            raise RuntimeError("Unexpected long prompt in SDPA audit")
        sdpa_scores = value_write_scores(sdpa, direction)
        result.append(
            {
                "example_id": row["example_id"],
                "maximum_vector_difference": float(
                    np.max(np.abs(dense.vectors - reconstructed.vectors))
                ),
                "maximum_attention_difference": float(
                    np.max(np.abs(dense.attention_mass - reconstructed.attention_mass))
                ),
                "maximum_score_difference": float(
                    np.max(np.abs(dense_scores - reconstructed_scores))
                ),
                "dense_selection": select_top(
                    dense.document_ids, dense_scores.tolist(), 2
                ),
                "reconstructed_selection": select_top(
                    reconstructed.document_ids, reconstructed_scores.tolist(), 2
                ),
                "sdpa_maximum_score_difference": float(
                    np.max(np.abs(dense_scores - sdpa_scores))
                ),
                "sdpa_selection": select_top(sdpa.document_ids, sdpa_scores.tolist(), 2),
            }
        )
    summary = {
        "n": len(result),
        "all_selections_match": all(
            item["dense_selection"] == item["reconstructed_selection"]
            for item in result
        ),
        "all_sdpa_selections_match": all(
            item["dense_selection"] == item["sdpa_selection"] for item in result
        ),
        "maximum_vector_difference": max(
            item["maximum_vector_difference"] for item in result
        ),
        "maximum_attention_difference": max(
            item["maximum_attention_difference"] for item in result
        ),
        "maximum_score_difference": max(
            item["maximum_score_difference"] for item in result
        ),
        "sdpa_maximum_score_difference": max(
            item["sdpa_maximum_score_difference"] for item in result
        ),
        "per_example": result,
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
