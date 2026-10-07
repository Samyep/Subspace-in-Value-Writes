#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_evidence.artifacts import artifact_manifest, load_json, load_jsonl, repo_path


REQUIRED_FIGURES = [
    "fig1_selection.csv",
    "fig2_causal.csv",
    "fig3_dose_response.csv",
    "fig4_transfer.csv",
    "fig5_head_knockout.csv",
    "fig6_label_controls.csv",
    "fig7_bridge.csv",
    "headline_baselines.csv",
]


def main() -> None:
    manifest = artifact_manifest()
    missing = [item["path"] for item in manifest if not repo_path(item["path"]).exists()]
    if missing:
        raise SystemExit(f"missing={missing}")

    prompts = load_jsonl("artifacts/prompts/hotpot_400_eval_prompts.jsonl")
    summary = load_json("artifacts/prompts/hotpot_400_eval_prompt_summary.json")
    if len(prompts) != 800 or summary["prompt_count"] != 800:
        raise SystemExit("prompt count mismatch")
    if sorted({row["family_name"] for row in prompts}) != ["active_set", "working_set"]:
        raise SystemExit("unexpected prompt families")
    if sorted({row["lexical_pack"] for row in prompts}) != ["pack_a", "pack_b"]:
        raise SystemExit("unexpected lexical packs")

    for name in REQUIRED_FIGURES:
        if not repo_path("artifacts", "figures", name).exists():
            raise SystemExit(f"missing figure CSV: {name}")

    print(f"OK: {len(manifest)} artifact paths and their structure validated")


if __name__ == "__main__":
    main()
