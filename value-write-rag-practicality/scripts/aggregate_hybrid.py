#!/usr/bin/env python3
"""Aggregate the single preregistered Value+BGE hybrid, when its gate passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import load_jsonl, write_json
from vw_rag.stats import bootstrap_mean


READERS = {"qwen3_8b": "Qwen3-8B", "qwen_14b": "Qwen2.5-14B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        type=Path,
        default=ROOT / "results/hybrid_calibration/hybrid_gate.json",
    )
    parser.add_argument(
        "--hybrid-root", type=Path, default=ROOT / "results/hybrid/readers"
    )
    parser.add_argument(
        "--reference-root", type=Path, default=ROOT / "reference_results"
    )
    parser.add_argument(
        "--bge-root", type=Path, default=ROOT / "results/readers"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/hybrid/aggregate"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate IDs in {path}")
    return result


def result_rows(path: Path) -> dict[str, dict[str, Any]]:
    merged = path / "merged.jsonl"
    if merged.exists():
        return index(merged)
    files = sorted(path.glob("part-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No result parts in {path}")
    combined: dict[str, dict[str, Any]] = {}
    for file in files:
        for example_id, row in index(file).items():
            if example_id in combined and combined[example_id] != row:
                raise RuntimeError(f"Conflicting duplicate {example_id} in {path}")
            combined[example_id] = row
    return combined


def summary(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    return bootstrap_mean(values, draws, seed)


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate.read_text())
    if not gate["gate_passed"]:
        raise SystemExit("The frozen hybrid gate did not pass; no MuSiQue hybrid run is allowed")

    per_reader: dict[str, Any] = {}
    macro_vectors: dict[str, list[list[float]]] = {
        "hybrid_em": [],
        "value_em": [],
        "bge_em": [],
        "hybrid_minus_value_em": [],
        "hybrid_minus_bge_em": [],
        "hybrid_support_recall": [],
        "value_support_recall": [],
        "bge_support_recall": [],
    }
    for reader_index, reader in enumerate(READERS):
        hybrid = result_rows(args.hybrid_root / reader)
        value = index(args.reference_root / reader / "per_example.jsonl")
        bge = result_rows(args.bge_root / reader / "reranker")
        if len(hybrid) != 400 or set(hybrid) != set(value) or set(hybrid) != set(bge):
            raise RuntimeError(f"Inputs do not align for {reader}")
        ids = sorted(hybrid, key=lambda item: int(hybrid[item]["split_rank"]))
        vectors = {
            "hybrid_em": [float(hybrid[item]["em"]) for item in ids],
            "value_em": [float(value[item]["value_write_em"]) for item in ids],
            "bge_em": [float(bge[item]["em"]) for item in ids],
            "hybrid_minus_value_em": [
                float(hybrid[item]["em"]) - float(value[item]["value_write_em"])
                for item in ids
            ],
            "hybrid_minus_bge_em": [
                float(hybrid[item]["em"]) - float(bge[item]["em"]) for item in ids
            ],
            "hybrid_support_recall": [
                float(hybrid[item]["support_recall"]) for item in ids
            ],
            "value_support_recall": [
                float(value[item]["value_write_support_recall"]) for item in ids
            ],
            "bge_support_recall": [float(bge[item]["support_recall"]) for item in ids],
        }
        per_reader[reader] = {
            key: summary(items, args.bootstrap_draws, args.seed + 1000 * reader_index + i)
            for i, (key, items) in enumerate(vectors.items())
        }
        for key, items in vectors.items():
            macro_vectors[key].append(items)

    macro = {}
    for index_value, (key, reader_vectors) in enumerate(macro_vectors.items()):
        by_question = np.asarray(reader_vectors, dtype=np.float64).mean(axis=0).tolist()
        macro[key] = summary(
            by_question, args.bootstrap_draws, args.seed + 10000 + index_value
        )
    result = {
        "selected_value_weight": gate["selected_weight"],
        "calibration_improvement": gate["improvement"],
        "per_reader": per_reader,
        "macro_shared_question_bootstrap": macro,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "hybrid_summary.json", result)

    def percent(item: dict[str, Any]) -> str:
        return f"{100 * item['mean']:.2f} [{100 * item['lo']:.2f}, {100 * item['hi']:.2f}]"

    lines = [
        "# Frozen hybrid results",
        "",
        f"The held-out HotpotQA gate selected a Value weight of {gate['selected_weight']:.2f}.",
        "",
        "| Reader | Hybrid EM | Value EM | BGE EM | Hybrid - Value | Hybrid - BGE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for reader, label in READERS.items():
        item = per_reader[reader]
        lines.append(
            f"| {label} | {percent(item['hybrid_em'])} | {percent(item['value_em'])} | "
            f"{percent(item['bge_em'])} | {percent(item['hybrid_minus_value_em'])} | "
            f"{percent(item['hybrid_minus_bge_em'])} |"
        )
    lines.extend(
        [
            "",
            "The intervals use paired bootstrap resampling over the frozen 400 MuSiQue questions.",
            "",
        ]
    )
    (args.output_dir / "hybrid_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["macro_shared_question_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
