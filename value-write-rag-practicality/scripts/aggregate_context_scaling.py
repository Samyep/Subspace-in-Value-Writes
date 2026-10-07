#!/usr/bin/env python3
"""Aggregate the preregistered two-reader context-length scaling experiment."""

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
CONDITIONS = ("full", "bge2", "scout_value2")
LABELS = {"full": "Full-K", "bge2": "BGE-2", "scout_value2": "3B Value-scout-2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader-root", type=Path, default=ROOT / "results/context_scaling/readers"
    )
    parser.add_argument(
        "--data-summary",
        type=Path,
        default=ROOT / "data/musique_long_context_pool.summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/context_scaling/aggregate"
    )
    parser.add_argument("--depths", default="20,40,80,160")
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    return parser.parse_args()


def load_directory(path: Path) -> list[dict[str, Any]]:
    merged = path / "merged.jsonl"
    if merged.exists():
        rows = load_jsonl(merged)
    else:
        rows = []
        for part in sorted(path.glob("part-*.jsonl")):
            rows.extend(load_jsonl(part))
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate reader rows in {path}")
    return sorted(result.values(), key=lambda row: int(row["split_rank"]))


def ci(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    return bootstrap_mean(values, draws, seed)


def pct(item: dict[str, Any]) -> str:
    return f"{100*item['mean']:.2f}"


def interval(item: dict[str, Any], scale: float = 1.0) -> str:
    return f"{scale*item['mean']:.2f} [{scale*item['lo']:.2f}, {scale*item['hi']:.2f}]"


def main() -> None:
    args = parse_args()
    depths = [int(item) for item in args.depths.split(",") if item]
    data_summary = json.loads(args.data_summary.read_text())
    readers = {reader: load_directory(args.reader_root / reader) for reader in READERS}
    if any(len(rows) != 400 for rows in readers.values()):
        raise RuntimeError("Every reader must contain exactly 400 questions")
    id_lists = [[row["example_id"] for row in rows] for rows in readers.values()]
    if any(ids != id_lists[0] for ids in id_lists[1:]):
        raise RuntimeError("Reader question order differs")

    result: dict[str, Any] = {
        "data": data_summary,
        "readers": {},
        "macro_contrasts": {},
    }
    macro_vectors: dict[int, dict[str, list[list[float]]]] = {
        depth: {
            "scout_em_minus_full": [],
            "scout_time_minus_full": [],
            "bge_em_minus_full": [],
            "bge_time_minus_full": [],
        }
        for depth in depths
    }
    for reader_index, (reader, rows) in enumerate(readers.items()):
        reader_summary = {"depths": {}, "crossover_depth": None}
        for depth_index, depth in enumerate(depths):
            depth_summary: dict[str, Any] = {"conditions": {}, "contrasts": {}}
            for condition_index, condition in enumerate(CONDITIONS):
                cells = [row["cells"][f"k{depth}_{condition}"] for row in rows]
                timed = [cell for row, cell in zip(rows, cells) if not row["timing_warmup"]]
                quality_keys = (
                    "em",
                    "f1",
                    "support_recall",
                    "first_hop_hit",
                    "terminal_hop_hit",
                    "prompt_tokens",
                    "output_tokens",
                )
                timing_keys = (
                    "selector_seconds",
                    "ttft_seconds",
                    "decode_seconds",
                    "reader_seconds",
                    "end_to_end_seconds",
                )
                depth_summary["conditions"][condition] = {
                    **{
                        key: ci(
                            [float(cell[key]) for cell in cells],
                            args.bootstrap_draws,
                            args.seed + 10000*reader_index + 1000*depth_index + 20*condition_index + i,
                        )
                        for i, key in enumerate(quality_keys)
                    },
                    **{
                        key: ci(
                            [float(cell[key]) for cell in timed],
                            args.bootstrap_draws,
                            args.seed + 20000 + 10000*reader_index + 1000*depth_index + 20*condition_index + i,
                        )
                        for i, key in enumerate(timing_keys)
                    },
                }
            for selector_index, selector in enumerate(("bge2", "scout_value2")):
                contrasts = {}
                all_pairs = [
                    (
                        row["cells"][f"k{depth}_{selector}"],
                        row["cells"][f"k{depth}_full"],
                        bool(row["timing_warmup"]),
                    )
                    for row in rows
                ]
                for metric in ("em", "f1"):
                    contrasts[f"{metric}_minus_full"] = ci(
                        [float(left[metric]) - float(right[metric]) for left, right, _ in all_pairs],
                        args.bootstrap_draws,
                        args.seed + 30000 + 1000*reader_index + 100*depth_index + 10*selector_index,
                    )
                timing_pairs = [(left, right) for left, right, warm in all_pairs if not warm]
                contrasts["time_minus_full"] = ci(
                    [left["end_to_end_seconds"] - right["end_to_end_seconds"] for left, right in timing_pairs],
                    args.bootstrap_draws,
                    args.seed + 40000 + 1000*reader_index + 100*depth_index + selector_index,
                )
                contrasts["paired_time_ratio"] = ci(
                    [left["end_to_end_seconds"] / right["end_to_end_seconds"] for left, right in timing_pairs],
                    args.bootstrap_draws,
                    args.seed + 50000 + 1000*reader_index + 100*depth_index + selector_index,
                )
                depth_summary["contrasts"][selector] = contrasts
                prefix = "scout" if selector == "scout_value2" else "bge"
                macro_vectors[depth][f"{prefix}_em_minus_full"].append(
                    [float(left["em"]) - float(right["em"]) for left, right, _ in all_pairs]
                )
                macro_vectors[depth][f"{prefix}_time_minus_full"].append(
                    [
                        float(left["end_to_end_seconds"]) - float(right["end_to_end_seconds"])
                        for left, right, warm in all_pairs
                        if not warm
                    ]
                )
            if (
                reader_summary["crossover_depth"] is None
                and depth_summary["contrasts"]["scout_value2"]["time_minus_full"]["mean"] < 0
            ):
                reader_summary["crossover_depth"] = depth
            reader_summary["depths"][str(depth)] = depth_summary
        result["readers"][reader] = reader_summary

    for depth in depths:
        result["macro_contrasts"][str(depth)] = {}
        for index_value, (key, reader_vectors) in enumerate(macro_vectors[depth].items()):
            array = np.asarray(reader_vectors, dtype=np.float64)
            # Warm-up filtering is shared by shard position. Macro timing uses
            # the same paired non-warm-up questions; quality uses all 400.
            by_question = array.mean(axis=0).tolist()
            result["macro_contrasts"][str(depth)][key] = ci(
                by_question,
                args.bootstrap_draws,
                args.seed + 60000 + 100*depths.index(depth) + index_value,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "context_scaling_summary.json", result)
    lines = [
        "# Context-length crossover results",
        "",
        "## Quality and end-to-end latency",
        "",
        "End-to-end latency includes selector and reader time. Reader timing is "
        "split into TTFT and remaining decode time in the JSON summary.",
        "",
        "| Reader | K | Approx. median full tokens | Method | EM | F1 | End-to-end s | Time / Full |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for reader in READERS:
        token_label = reader
        for depth in depths:
            info = result["readers"][reader]["depths"][str(depth)]
            median_tokens = data_summary["prompt_token_distributions"][token_label][str(depth)]["median"]
            full_time = info["conditions"]["full"]["end_to_end_seconds"]["mean"]
            for condition in CONDITIONS:
                metrics = info["conditions"][condition]
                ratio = metrics["end_to_end_seconds"]["mean"] / full_time
                lines.append(
                    f"| {READERS[reader]} | {depth} | {median_tokens:.0f} | {LABELS[condition]} | "
                    f"{pct(metrics['em'])} | {pct(metrics['f1'])} | "
                    f"{metrics['end_to_end_seconds']['mean']:.3f} | {ratio:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Scout contrasts",
            "",
            "| Reader | K | EM points vs Full (95% CI) | End-to-end seconds vs Full (95% CI) | Paired time ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for reader in READERS:
        for depth in depths:
            contrast = result["readers"][reader]["depths"][str(depth)]["contrasts"]["scout_value2"]
            lines.append(
                f"| {READERS[reader]} | {depth} | {interval(contrast['em_minus_full'],100)} | "
                f"{interval(contrast['time_minus_full'])} | {interval(contrast['paired_time_ratio'])} |"
            )
    lines.extend(["", "## Crossover", ""])
    for reader in READERS:
        crossover = result["readers"][reader]["crossover_depth"]
        lines.append(
            f"- {READERS[reader]}: "
            + (f"first negative mean latency difference at K={crossover}." if crossover else "no crossover through K=160.")
        )
    (args.output_dir / "context_scaling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({r: result['readers'][r]['crossover_depth'] for r in READERS}, indent=2))


if __name__ == "__main__":
    main()
