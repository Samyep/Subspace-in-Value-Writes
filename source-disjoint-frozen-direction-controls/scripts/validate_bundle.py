#!/usr/bin/env python3
"""Validate split integrity, raw outputs, and released V4 estimates."""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/frozen_direction/main_v4"
AGGREGATE = ROOT / "results/frozen_direction/aggregate_v4/aggregate.json"

EXPECTED = {
    "llama_3b": (217, 0.027649769585253458, 0.013824884792626729, 0.013824884792626729, -0.008591805901173121, 0.03763497098123898),
    "llama_8b": (401, 0.3790523690773067, 0.06068162926018285, 0.3183707398171239, 0.24027016506462182, 0.3982222222222222),
    "qwen_7b": (521, 0.10556621880998081, 0.005758157389635317, 0.09980806142034548, 0.05838415394233683, 0.14645648547341822),
    "qwen_14b": (636, 0.07075471698113207, 0.017819706498951784, 0.052935010482180286, 0.022809534906104836, 0.08630537314515832),
    "mistral_7b": (287, 0.013937282229965157, 0.02206736353077816, -0.008130081300813004, -0.027668108484351306, 0.01282051282051282),
    "mistral_12b": (458, 0.09388646288209607, 0.029839883551673947, 0.06404657933042213, 0.026904671459808027, 0.10537646463185302),
}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    for relative in (
        "experiments/frozen_direction_intervention_v4.py",
        "experiments/aggregate_frozen_controls_v4.py",
        "experiments/PROTOCOL_v4.md",
        "data/hotpot_400_prompts.jsonl",
        "results/frozen_direction/aggregate_v4/aggregate.json",
        "results/frozen_direction/aggregate_v4/numerical_audit.json",
    ):
        require((ROOT / relative).is_file(), f"Missing required artifact: {relative}")

    data = load_jsonl(ROOT / "data/hotpot_400_prompts.jsonl")
    selected = [row for row in data if row["family_name"] in {"active_set", "working_set"}
                and row["lexical_pack"] in {"pack_a", "pack_b"}]
    require(len(selected) == 1600, "Expected 1,600 controlled prompts")
    require(len({row["prompt_id"] for row in selected}) == 1600, "Prompt IDs are not unique")
    require(len({row["example_id"] for row in selected}) == 400, "Expected 400 source questions")
    fit_ids = {row["example_id"] for row in selected if int(row["example_rank"]) <= 200}
    eval_ids = {row["example_id"] for row in selected if int(row["example_rank"]) > 200}
    require(len(fit_ids) == len(eval_ids) == 200 and fit_ids.isdisjoint(eval_ids),
            "Fit/evaluation source split is not 200 + 200 and disjoint")

    aggregate = load_json(AGGREGATE)
    require(aggregate["status"] == "complete" and not aggregate["smoke"], "Aggregate is incomplete")
    require(aggregate["draws"] == 20000 and aggregate["seed"] == 29911,
            "Bootstrap configuration changed")
    require([cell["cell"] for cell in aggregate["cells"]] == list(EXPECTED),
            "Aggregate model order changed")

    for cell in aggregate["cells"]:
        name = cell["cell"]
        directory = RESULTS / name
        manifest = load_json(directory / "manifest.json")
        summary = load_json(directory / "summary.json")
        rows = load_jsonl(directory / "per_prompt.jsonl")
        identity = manifest["identity"]
        require(summary["status"] == "complete", f"{name}: incomplete summary")
        require(len(rows) == len({row["prompt_id"] for row in rows}) == 800,
                f"{name}: expected 800 unique evaluation rows")
        require(summary["fit_prompts"] == summary["eval_prompts"] == 800,
                f"{name}: prompt counts changed")
        require(summary["fit_sources"] == summary["eval_sources"] == 200,
                f"{name}: source counts changed")
        require(set(identity["fit_example_ids"]).isdisjoint(identity["eval_example_ids"]),
                f"{name}: manifest source overlap")
        require(all(not row["smoke"] for row in rows), f"{name}: smoke rows in full results")

        estimate = cell["estimates"]["destroyed"]
        actual = (cell["baseline_correct"], estimate["pc1"], estimate["random_mean"],
                  estimate["difference"], *estimate["difference_ci95"])
        expected = EXPECTED[name]
        require(actual[0] == expected[0], f"{name}: eligible count changed")
        require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
                    for a, b in zip(actual[1:], expected[1:])),
                f"{name}: released estimates changed")

    audit = load_json(ROOT / "audits/v4_effect_independent_checks.json")
    require(audit["all_models_verified"], "Independent effect reconstruction did not pass")
    numerical = load_json(ROOT / "audits/v4_numerical_independent_checks.json")
    require(numerical["total_sites"] == 128610 and numerical["total_original_bound_failures"] == 2,
            "Independent numerical audit totals changed")
    print("Validated 400 disjoint-source questions, six model outputs, aggregate estimates, and audit totals.")


if __name__ == "__main__":
    main()
