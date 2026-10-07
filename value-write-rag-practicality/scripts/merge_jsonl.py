#!/usr/bin/env python3
"""Merge resumable JSONL shards with duplicate and coverage checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vw_rag.io_utils import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-data", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("part-*.jsonl")))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("No input JSONL files found")
    return expanded


def main() -> None:
    args = parse_args()
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for path in expand_inputs(args.input):
        for row in load_jsonl(path):
            example_id = str(row["example_id"])
            if example_id in rows:
                if row != rows[example_id]:
                    raise SystemExit(
                        f"Conflicting duplicate {example_id} in {sources[example_id]} and {path}"
                    )
                continue
            rows[example_id] = row
            sources[example_id] = path

    if args.expected_data is not None:
        expected_rows = load_jsonl(args.expected_data)
        expected_ids = [str(row["example_id"]) for row in expected_rows]
        missing = set(expected_ids) - set(rows)
        extra = set(rows) - set(expected_ids)
        if extra:
            raise SystemExit(f"Unexpected IDs: {sorted(extra)[:5]}")
        if missing and not args.allow_partial:
            raise SystemExit(f"Missing {len(missing)} expected IDs: {sorted(missing)[:5]}")
        order = {example_id: index for index, example_id in enumerate(expected_ids)}
        merged = sorted(rows.values(), key=lambda row: order[str(row["example_id"])])
    else:
        merged = sorted(rows.values(), key=lambda row: int(row["split_rank"]))
    write_jsonl(args.output, merged)
    print(f"Wrote {len(merged)} unique rows to {args.output}")


if __name__ == "__main__":
    main()
