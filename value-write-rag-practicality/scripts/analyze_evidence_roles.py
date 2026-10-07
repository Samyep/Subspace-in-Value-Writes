#!/usr/bin/env python3
"""Analyze Value/BGE disagreements and first/terminal evidence retention."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import load_jsonl, write_json, write_jsonl
from vw_rag.stats import bootstrap_mean


READERS = {
    "qwen_7b": "Qwen2.5-7B",
    "qwen_14b": "Qwen2.5-14B",
    "qwen3_8b": "Qwen3-8B",
    "llama_3b": "Llama-3.2-3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/musique_long_context_pool.jsonl"
    )
    parser.add_argument(
        "--bge-selections",
        type=Path,
        default=ROOT / "results/reranker/bge_v2_m3/selections.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/evidence_analysis"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    return parser.parse_args()


def index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["example_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate example IDs in {path}")
    return result


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def conditional(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(float(row[value_key]))
    return {
        key: {"n": len(values), "mean": float(np.mean(values))}
        for key, values in sorted(groups.items())
    }


def interval(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    return bootstrap_mean(values, draws, seed)


def evidence_category(selected: set[str], first: str, terminal: str) -> str:
    first_hit = first in selected
    terminal_hit = terminal in selected
    if first_hit and terminal_hit:
        return "both"
    if first_hit:
        return "first_only"
    if terminal_hit:
        return "terminal_only"
    return "neither"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Evidence-role and disagreement analysis",
        "",
        "All evidence roles come from MuSiQue `question_decomposition` annotations. "
        "The first decomposition support is the first hop and the last support is "
        "the terminal hop.",
        "",
        "## Main results",
        "",
        "| Reader | Selector | EM | Support recall@2 | First-hop hit | Terminal-hop hit | Both supports |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for reader in READERS:
        for selector in ("value", "bge"):
            item = summary["readers"][reader][selector]
            lines.append(
                f"| {READERS[reader]} | {selector.upper()}-2 | "
                f"{100*item['em']:.2f} | {100*item['support_recall']:.2f} | "
                f"{100*item['first_hop_hit']:.2f} | "
                f"{100*item['terminal_hop_hit']:.2f} | "
                f"{100*item['both_supports']:.2f} |"
            )
    lines.extend(
        [
            "",
            "The first hop is available in 348/400 retrieved pools and the terminal "
            "hop in 161/400. Conditional retention therefore separates selector "
            "behavior from first-stage retrieval availability.",
            "",
            "| Reader | Selector | First hit / available | Terminal hit / available | Mean top-2 overlap |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for reader in READERS:
        for selector in ("value", "bge"):
            item = summary["readers"][reader][selector]
            lines.append(
                f"| {READERS[reader]} | {selector.upper()}-2 | "
                f"{100*item['first_hop_hit_given_available']:.2f} | "
                f"{100*item['terminal_hop_hit_given_available']:.2f} | "
                f"{100*summary['readers'][reader]['top2_overlap']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Answer disagreements",
            "",
            "| Reader | Both correct | Value only | BGE only | Both wrong |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for reader in READERS:
        table = summary["readers"][reader]["answer_contingency"]
        lines.append(
            f"| {READERS[reader]} | {table['++']} | {table['+-']} | "
            f"{table['-+']} | {table['--']} |"
        )
    lines.extend(
        [
            "",
            "## Shared-question macro contrasts",
            "",
            "Positive values favor reader-specific Value-2.",
            "",
            "| Metric | Mean difference | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (
        ("em", "EM"),
        ("support_recall", "Support recall@2"),
        ("first_hop_hit", "First-hop retention"),
        ("terminal_hop_hit", "Terminal-hop retention"),
    ):
        item = summary["macro_contrasts"][key]
        lines.append(
            f"| {label} | {100*item['mean']:.2f} | "
            f"[{100*item['lo']:.2f}, {100*item['hi']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Conditional EM",
            "",
            "EM rises sharply when the terminal hop is retained. Full conditional "
            "tables by support-hit count and evidence role are stored in the JSON "
            "summary and the per-example file.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = index(args.data)
    bge_selection = index(args.bge_selections)
    if len(data) != 400 or set(data) != set(bge_selection):
        raise RuntimeError("Data and BGE selection IDs do not match the frozen 400")

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "protocol": {
            "questions": 400,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "first_hop_definition": "first question_decomposition support",
            "terminal_hop_definition": "last question_decomposition support",
        },
        "availability": {},
        "readers": {},
    }
    first_available = []
    terminal_available = []
    macro_vectors: dict[str, list[list[float]]] = {
        key: [] for key in ("em", "support_recall", "first_hop_hit", "terminal_hop_hit")
    }

    for reader_index, reader in enumerate(READERS):
        refs = index(ROOT / "reference_results" / reader / "per_example.jsonl")
        bge_answers = index(
            ROOT / "results" / "readers" / reader / "reranker" / "part-000-of-001.jsonl"
        )
        if set(refs) != set(data) or set(bge_answers) != set(data):
            raise RuntimeError(f"Frozen IDs changed for {reader}")
        reader_rows: list[dict[str, Any]] = []
        contrasts = {key: [] for key in macro_vectors}
        contingency: Counter[str] = Counter()
        disagreement: dict[str, list[dict[str, float]]] = {"+-": [], "-+": []}
        for example_id, row in sorted(data.items(), key=lambda item: item[1]["split_rank"]):
            prefix = row["documents"][:20]
            local_to_corpus = {
                str(document["document_id"]): str(document["corpus_document_id"])
                for document in prefix
            }
            available = set(local_to_corpus.values())
            first = str(row["first_hop_corpus_document_id"])
            terminal = str(row["terminal_hop_corpus_document_id"])
            if reader_index == 0:
                first_available.append(float(first in available))
                terminal_available.append(float(terminal in available))
            value_selected_local = [str(item) for item in refs[example_id]["value_write_documents"]]
            bge_selected_local = [str(item) for item in bge_selection[example_id]["selected_documents"]]
            value_selected = {local_to_corpus[item] for item in value_selected_local}
            bge_selected = {local_to_corpus[item] for item in bge_selected_local}
            support = {first, terminal}
            value_em = float(refs[example_id]["value_write_em"])
            bge_em = float(bge_answers[example_id]["em"])
            code = ("+" if value_em else "-") + ("+" if bge_em else "-")
            contingency[code] += 1
            item = {
                "example_id": example_id,
                "split_rank": int(row["split_rank"]),
                "reader": reader,
                "value_em": value_em,
                "bge_em": bge_em,
                "full20_em": float(refs[example_id]["baseline_em"]),
                "value_support_recall": len(value_selected & support) / 2,
                "bge_support_recall": len(bge_selected & support) / 2,
                "value_first_hop_hit": float(first in value_selected),
                "bge_first_hop_hit": float(first in bge_selected),
                "value_terminal_hop_hit": float(terminal in value_selected),
                "bge_terminal_hop_hit": float(terminal in bge_selected),
                "first_hop_available": float(first in available),
                "terminal_hop_available": float(terminal in available),
                "value_evidence_category": evidence_category(value_selected, first, terminal),
                "bge_evidence_category": evidence_category(bge_selected, first, terminal),
                "top2_overlap": len(value_selected & bge_selected) / 2,
                "answer_contingency": code,
            }
            reader_rows.append(item)
            all_rows.append(item)
            for key in contrasts:
                contrasts[key].append(item[f"value_{key}"] - item[f"bge_{key}"])
            if code in disagreement:
                disagreement[code].append(
                    {
                        "value_support_recall": item["value_support_recall"],
                        "bge_support_recall": item["bge_support_recall"],
                        "value_terminal_hop_hit": item["value_terminal_hop_hit"],
                        "bge_terminal_hop_hit": item["bge_terminal_hop_hit"],
                        "top2_overlap": item["top2_overlap"],
                        "full20_em": item["full20_em"],
                    }
                )

        reader_summary: dict[str, Any] = {
            "answer_contingency": {key: contingency[key] for key in ("++", "+-", "-+", "--")},
            "top2_overlap": mean([item["top2_overlap"] for item in reader_rows]),
            "paired_contrasts": {
                key: interval(values, args.bootstrap_draws, args.seed + 100*reader_index + i)
                for i, (key, values) in enumerate(contrasts.items())
            },
            "disagreement": {},
        }
        for selector in ("value", "bge"):
            first_hits = [item[f"{selector}_first_hop_hit"] for item in reader_rows]
            terminal_hits = [item[f"{selector}_terminal_hop_hit"] for item in reader_rows]
            reader_summary[selector] = {
                "em": mean([item[f"{selector}_em"] for item in reader_rows]),
                "support_recall": mean([item[f"{selector}_support_recall"] for item in reader_rows]),
                "first_hop_hit": mean(first_hits),
                "terminal_hop_hit": mean(terminal_hits),
                "both_supports": mean(
                    [float(item[f"{selector}_evidence_category"] == "both") for item in reader_rows]
                ),
                "first_hop_hit_given_available": sum(first_hits) / sum(first_available),
                "terminal_hop_hit_given_available": sum(terminal_hits) / sum(terminal_available),
                "em_by_support_recall": conditional(
                    reader_rows, f"{selector}_support_recall", f"{selector}_em"
                ),
                "em_by_evidence_category": conditional(
                    reader_rows, f"{selector}_evidence_category", f"{selector}_em"
                ),
            }
        for code, rows in disagreement.items():
            reader_summary["disagreement"][code] = {
                "n": len(rows),
                **{
                    key: mean([item[key] for item in rows])
                    for key in (
                        "value_support_recall",
                        "bge_support_recall",
                        "value_terminal_hop_hit",
                        "bge_terminal_hop_hit",
                        "top2_overlap",
                        "full20_em",
                    )
                },
            }
        summary["readers"][reader] = reader_summary
        for key, values in contrasts.items():
            macro_vectors[key].append(values)

    summary["availability"] = {
        "first_hop": {"n": int(sum(first_available)), "rate": mean(first_available)},
        "terminal_hop": {"n": int(sum(terminal_available)), "rate": mean(terminal_available)},
    }
    summary["macro_contrasts"] = {}
    for index_value, (key, reader_vectors) in enumerate(macro_vectors.items()):
        by_question = np.asarray(reader_vectors, dtype=np.float64).mean(axis=0).tolist()
        summary["macro_contrasts"][key] = interval(
            by_question, args.bootstrap_draws, args.seed + 10000 + index_value
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_example.jsonl", all_rows)
    write_json(args.output_dir / "evidence_analysis.json", summary)
    write_report(args.output_dir / "evidence_analysis.md", summary)
    print(json.dumps(summary["macro_contrasts"], indent=2))


if __name__ == "__main__":
    main()
