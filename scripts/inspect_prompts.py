#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_evidence.artifacts import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    rows = load_jsonl("artifacts/prompts/hotpot_400_eval_prompts.jsonl")
    for row in rows[: args.limit]:
        print(
            f"{row['prompt_id']}\t"
            f"rank={row['example_rank']}\t"
            f"family={row['family_name']}\t"
            f"pack={row['lexical_pack']}\t"
            f"gold={','.join(row['support_labels'])}\t"
            f"question={row['question']}"
        )


if __name__ == "__main__":
    main()
