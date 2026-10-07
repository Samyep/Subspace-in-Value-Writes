#!/usr/bin/env python3
"""Validate V4 numerical quality separately, then summarize all six controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

CELLS = ("llama_3b", "llama_8b", "qwen_7b", "qwen_14b", "mistral_7b", "mistral_12b")
NAMES = {"llama_3b": "Llama-3B", "llama_8b": "Llama-8B", "qwen_7b": "Qwen-7B",
         "qwen_14b": "Qwen-14B", "mistral_7b": "Mistral-7B", "mistral_12b": "Mistral-12B"}
KINDS = ("destroyed", "wrong_distinct_pair", "repeated_label", "distinct_one_gold", "distinct_zero_gold")


def category(prediction: list[str], gold: list[str]) -> str:
    assert len(prediction) == 2 and all(p in "ABCDEF" for p in prediction)
    if prediction[0] == prediction[1]:
        return "repeated_label"
    return "correct" if set(prediction) == set(gold) else "wrong_distinct_pair"


def outcomes(condition: dict, gold: list[str]) -> np.ndarray:
    cat = category(condition["prediction"], gold)
    assert cat == condition["category"]
    hits = len(set(condition["prediction"]) & set(gold))
    return np.asarray([cat != "correct", cat == "wrong_distinct_pair", cat == "repeated_label",
                       cat == "wrong_distinct_pair" and hits == 1,
                       cat == "wrong_distinct_pair" and hits == 0], dtype=float)


def clustered_intervals(totals: np.ndarray, draws: int = 20000, seed: int = 29911) -> dict:
    """Column 0 is clean-correct count; other columns are paired outcome sums."""
    assert totals.ndim == 2 and totals.shape[1] == 1 + 2 * len(KINDS)
    denom = totals[:, 0].sum()
    assert denom > 0
    point = totals[:, 1:].sum(axis=0) / denom
    rng = np.random.default_rng(seed)
    sampled = []
    remaining = draws
    while remaining:
        size = min(1000, remaining)
        indices = rng.integers(0, len(totals), size=(size, len(totals)))
        sums = totals[indices].sum(axis=1)
        if (sums[:, 0] == 0).any():
            raise RuntimeError("Zero baseline-correct denominator in a source bootstrap draw")
        sampled.append(sums[:, 1:] / sums[:, :1])
        remaining -= size
    distribution = np.concatenate(sampled)
    n = len(KINDS)
    result = {}
    for i, kind in enumerate(KINDS):
        delta = distribution[:, i] - distribution[:, n + i]
        result[kind] = {
            "pc1": float(point[i]),
            "random_mean": float(point[n + i]),
            "difference": float(point[i] - point[n + i]),
            "pc1_ci95": np.quantile(distribution[:, i], [0.025, 0.975]).tolist(),
            "random_mean_ci95": np.quantile(distribution[:, n + i], [0.025, 0.975]).tolist(),
            "difference_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    return result


def reassignment_intervals(totals: np.ndarray, draws: int = 20000, seed: int = 29911) -> dict:
    """Source columns: eligible n, lower numerator, upper numerator, flagged rollouts."""
    assert totals.ndim == 2 and totals.shape[1] == 4
    total = totals.sum(axis=0)
    assert total[0] > 0
    lower, upper = total[1:3] / total[0]
    width = total[3] / (3 * total[0])
    assert abs(upper - lower - width) < 1e-12
    rng = np.random.default_rng(seed)
    sampled = []
    for start in range(0, draws, 1000):
        indices = rng.integers(0, len(totals), size=(min(1000, draws - start), len(totals)))
        sums = totals[indices].sum(axis=1)
        assert (sums[:, 0] > 0).all()
        bounds = sums[:, 1:3] / sums[:, :1]
        assert np.allclose(bounds[:, 1] - bounds[:, 0], sums[:, 3] / (3 * sums[:, 0]), atol=1e-12)
        sampled.append(bounds)
    distribution = np.concatenate(sampled)
    return {
        "lower": float(lower), "upper": float(upper), "width": float(width),
        "flagged_random_rollouts": int(round(total[3])),
        "lower_ci95": np.quantile(distribution[:, 0], [0.025, 0.975]).tolist(),
        "upper_ci95": np.quantile(distribution[:, 1], [0.025, 0.975]).tolist(),
        "interpretation": "Worst-case reassignment of only random rollouts violating the original combined numerical bound; not bounds on an ideal exact-norm intervention",
    }


def site_matches_original_rule(site: dict, identity: dict) -> bool:
    target, actual = site["requested_norm"], site["achieved_norm"]
    bound = max(identity["norm_relative_tolerance"] * target,
                identity["norm_absolute_output_fraction"] * site["original_output_norm"])
    passes = abs(actual - target) <= bound
    assert passes == site["original_tolerance_passed"]
    assert bool(site["quantization_limited"]) == (not passes)
    if site["certified_fallback_used"]:
        bracket = site["certified_bracket"]
        lower, upper = bracket["lower"], bracket["upper"]
        assert upper["scalar_float32_bits"] - lower["scalar_float32_bits"] == 1
        assert bracket["adjacent_scalar_bits_difference"] == 1
        assert lower["achieved_norm"] < target <= upper["achieved_norm"]
        assert bracket["straddles_target"] and bracket["candidate_dtype"] == "torch.bfloat16"
        assert bracket["tie_rule"] == "lower"
        chosen = "lower" if abs(lower["achieved_norm"] - target) <= abs(upper["achieved_norm"] - target) else "upper"
        assert bracket["selected_endpoint"] == chosen
        assert actual == bracket[chosen]["achieved_norm"] and site["scale"] == bracket[chosen]["scale"]
        assert bracket["norm_gap"] == upper["achieved_norm"] - lower["achieved_norm"]
        assert bracket["original_combined_tolerance"] == bound
        assert bracket["original_tolerance_attainable"] == passes
        for endpoint in (lower, upper):
            scalar_bits = int(np.asarray(np.float32(endpoint["scale"])).view(np.uint32))
            assert scalar_bits == endpoint["scalar_float32_bits"]
            assert endpoint["absolute_norm_error"] == abs(endpoint["achieved_norm"] - target)
            assert endpoint["original_tolerance_passed"] == (endpoint["absolute_norm_error"] <= bound)
    else:
        assert passes
    return passes


def numerical_audit_cell(root: Path, cell: str, smoke: bool = False) -> dict:
    """Do not read or aggregate intervention correctness in this audit."""
    directory = root / cell
    identity = json.loads((directory / "manifest.json").read_text())["identity"]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["smoke"] == smoke
    records = [json.loads(line) for line in (directory / "per_prompt.jsonl").read_text().splitlines()]
    assert len(records) == summary["eval_prompts"]
    assert not (set(identity["fit_example_ids"]) & set(identity["eval_example_ids"]))
    sites = {(step, layer) for step in range(2) for layer in identity["writer_window"]}
    errors, output_errors, magnitudes, ratios = [], [], [], []
    flagged_sites = []
    flagged_rollouts = 0
    flagged_prompts, flagged_sources = set(), set()
    fallback_sites = 0
    floor_only_sites = 0
    eligible = 0
    for row in records:
        if row["pc1"] is None:
            assert row["random"] == []
            continue
        eligible += 1
        assert len(row["random"]) == 3
        treatment = {(s["step"], s["layer"]): s for s in row["pc1"]["perturbations"]}
        assert set(treatment) == sites and len(row["pc1"]["perturbations"]) == len(sites)
        for rep, control in enumerate(row["random"]):
            observed = {(s["step"], s["layer"]) for s in control["perturbations"]}
            assert observed == sites and len(control["perturbations"]) == len(sites)
            target_energy, actual_energy = 0.0, 0.0
            flagged = False
            for site in control["perturbations"]:
                target = treatment[(site["step"], site["layer"])]["achieved_norm"]
                assert target == site["requested_norm"]
                actual = site["achieved_norm"]
                error = abs(actual - target)
                passes = site_matches_original_rule(site, identity)
                fallback_sites += int(site["certified_fallback_used"])
                floor_only_sites += int(passes and error > identity["norm_relative_tolerance"] * target)
                errors.append(error / target if target else 0.0)
                output_errors.append(error / max(site["original_output_norm"], 1e-30))
                magnitudes.append(target)
                target_energy += target ** 2
                actual_energy += actual ** 2
                if not passes:
                    flagged = True
                    flagged_sites.append({"prompt_id": row["prompt_id"], "source_id": row["example_id"],
                                          "replicate": rep, **site})
            if target_energy:
                ratios.append(actual_energy / target_energy)
            else:
                assert actual_energy == 0
                ratios.append(1.0)
            if flagged:
                flagged_rollouts += 1
                flagged_prompts.add(row["prompt_id"])
                flagged_sources.add(row["example_id"])
    assert eligible == summary["baseline_correct"] and eligible > 0
    assert len(errors) == summary["random_intervention_sites"]
    assert len(flagged_sites) == summary["random_sites_original_tolerance_failed"]
    assert flagged_rollouts == summary["random_rollouts_with_original_tolerance_failure"]
    assert fallback_sites == summary["random_sites_certified_fallback"]
    return {
        "cell": cell, "fit_prompts": summary["fit_prompts"], "eval_prompts": len(records),
        "fit_sources": summary["fit_sources"], "eval_sources": summary["eval_sources"],
        "baseline_correct": eligible, "random_rollouts": 3 * eligible,
        "compared_sites": len(errors), "certified_fallback_sites": fallback_sites,
        "within_absolute_floor_only_sites": floor_only_sites,
        "original_rule_failed_sites": len(flagged_sites),
        "original_rule_failed_rollouts": flagged_rollouts,
        "original_rule_failed_prompts": len(flagged_prompts),
        "original_rule_failed_sources": len(flagged_sources),
        "maximum_reassignment_width": flagged_rollouts / (3 * eligible),
        "relative_error_quantiles": dict(zip(("median", "p95", "p99", "max"), np.quantile(errors, [0.5, 0.95, 0.99, 1]).tolist())),
        "output_relative_error_quantiles": dict(zip(("median", "p95", "p99", "max"), np.quantile(output_errors, [0.5, 0.95, 0.99, 1]).tolist())),
        "treatment_norm_quantiles": np.quantile(magnitudes, [0, 0.5, 0.95, 1]).tolist(),
        "rollout_energy_ratio_quantiles": dict(zip(("min", "median", "p95", "max"), np.quantile(ratios, [0, 0.5, 0.95, 1]).tolist())),
        "flagged_sites": flagged_sites,
    }


def summarize_cell(root: Path, cell: str, draws: int, seed: int, smoke: bool) -> dict:
    directory = root / cell
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = manifest["identity"]
    summary = json.loads((directory / "summary.json").read_text())
    retention = json.loads((directory / "prompt_retention.json").read_text())
    rendered = json.loads((directory / "rendered_prompts.json").read_text())
    assert summary["status"] == "complete" and summary["smoke"] == smoke
    rows = [json.loads(line) for line in (directory / "per_prompt.jsonl").read_text().splitlines()]
    assert len({r["prompt_id"] for r in rows}) == len(rows)
    planned_ids = set(identity["eval_prompt_ids"])
    retained_ids = {r["prompt_id"] for r in rendered if r["retained"] and r["prompt_id"] in planned_ids}
    excluded_ids = {r["prompt_id"] for r in retention["eval"]["excluded"]}
    assert retained_ids | excluded_ids == planned_ids and not (retained_ids & excluded_ids)
    assert {r["prompt_id"] for r in rows} == retained_ids
    assert len(rows) == retention["eval"]["after_prompts"]
    assert all(r["tokens"] > 768 for r in retention["eval"]["excluded"])
    assert all(r["tokens"] <= 768 for r in rendered if r["retained"])
    assert not (set(identity["fit_example_ids"]) & set(identity["eval_example_ids"]))
    assert all(r["smoke"] == smoke for r in rows)
    assert all(201 <= int(r["example_rank"]) <= 400 for r in rows)
    if not smoke:
        assert len(identity["fit_example_ids"]) == len(identity["eval_example_ids"]) == 200
    for audit_file in ("legacy_decoder_audit.json", "legacy_extractor_audit.json"):
        audit = json.loads((directory / audit_file).read_text())
        if audit_file.startswith("legacy_decoder"):
            assert audit["agreement"]
        else:
            assert audit["status"] == "passed", audit
    sites = {(step, layer) for step in range(2) for layer in identity["writer_window"]}
    groups = defaultdict(lambda: np.zeros(1 + 2 * len(KINDS)))
    sensitivity_groups = defaultdict(lambda: np.zeros(4))
    random_counts = np.zeros((3, len(KINDS)))
    mismatch_relative, mismatch_absolute, treatment_magnitudes = [], [], []
    mismatch_output_relative, energy_ratios = [], []
    floor_uses = 0
    baseline_counts = Counter()
    source_variant_counts = Counter()
    for row in rows:
        source_variant_counts[row["example_id"]] += 1
        group = groups[row["example_id"]]
        sensitivity = sensitivity_groups[row["example_id"]]
        base = category(row["baseline"]["prediction"], row["support_labels"])
        assert base == row["baseline"]["category"]
        baseline_counts[base] += 1
        if base != "correct":
            assert row["pc1"] is None and row["random"] == []
            continue
        assert row["pc1"] is not None and len(row["random"]) == 3
        assert [r["replicate"] for r in row["random"]] == [0, 1, 2]
        treatment_sites = {(p["step"], p["layer"]): p for p in row["pc1"]["perturbations"]}
        assert set(treatment_sites) == sites and len(treatment_sites) == len(row["pc1"]["perturbations"])
        control_vectors = []
        control_flags = []
        for rep, condition in enumerate(row["random"]):
            rollout_flag = False
            observed_sites = {(p["step"], p["layer"]) for p in condition["perturbations"]}
            assert observed_sites == sites and len(condition["perturbations"]) == len(sites)
            target_energy, actual_energy = 0.0, 0.0
            for perturbation in condition["perturbations"]:
                target = treatment_sites[(perturbation["step"], perturbation["layer"])]["achieved_norm"]
                assert perturbation["requested_norm"] == target
                achieved = perturbation["achieved_norm"]
                absolute = abs(achieved - target)
                relative = absolute / target if target else 0.0
                assert target or achieved == 0
                original_norm = perturbation.get("original_output_norm", 0.0)
                absolute_fraction = identity.get("norm_absolute_output_fraction", 0.0)
                bound = max(identity["norm_relative_tolerance"] * target, absolute_fraction * original_norm)
                passes = site_matches_original_rule(perturbation, identity)
                rollout_flag |= not passes
                used_floor = passes and absolute > identity["norm_relative_tolerance"] * target
                assert used_floor == perturbation.get("used_absolute_tolerance", False)
                floor_uses += int(used_floor)
                mismatch_relative.append(relative)
                mismatch_absolute.append(absolute)
                mismatch_output_relative.append(absolute / max(original_norm, 1e-30))
                treatment_magnitudes.append(target)
                target_energy += target ** 2
                actual_energy += achieved ** 2
            if target_energy:
                energy_ratios.append(actual_energy / target_energy)
            else:
                assert actual_energy == 0
                energy_ratios.append(1.0)
            values = outcomes(condition, row["support_labels"])
            random_counts[rep] += values
            control_vectors.append(values)
            control_flags.append(rollout_flag)
        group[0] += 1
        group[1:1 + len(KINDS)] += outcomes(row["pc1"], row["support_labels"])
        group[1 + len(KINDS):] += np.mean(control_vectors, axis=0)
        treatment_destroyed = float(outcomes(row["pc1"], row["support_labels"])[0])
        sensitivity[0] += 1
        sensitivity[1] += treatment_destroyed - np.mean([1.0 if flag else value[0] for value, flag in zip(control_vectors, control_flags)])
        sensitivity[2] += treatment_destroyed - np.mean([0.0 if flag else value[0] for value, flag in zip(control_vectors, control_flags)])
        sensitivity[3] += sum(control_flags)
    assert baseline_counts["correct"] == summary["baseline_correct"]
    group_matrix = np.stack([groups[source] for source in sorted(groups)])
    estimates = clustered_intervals(group_matrix, draws, seed)
    reassignment = reassignment_intervals(np.stack([sensitivity_groups[source] for source in sorted(groups)]), draws, seed)
    assert reassignment["lower"] - 1e-12 <= estimates["destroyed"]["difference"] <= reassignment["upper"] + 1e-12
    assert abs(estimates["destroyed"]["pc1"] - estimates["wrong_distinct_pair"]["pc1"] - estimates["repeated_label"]["pc1"]) < 1e-10
    assert abs(estimates["destroyed"]["random_mean"] - estimates["wrong_distinct_pair"]["random_mean"] - estimates["repeated_label"]["random_mean"]) < 1e-10
    norms = {
        "compared_sites": len(mismatch_relative),
        "relative_error_quantiles": dict(zip(("median", "p95", "p99", "max"), np.quantile(mismatch_relative, [0.5, 0.95, 0.99, 1.0]).tolist())),
        "maximum_absolute_error": max(mismatch_absolute),
        "absolute_floor_uses": floor_uses,
        "maximum_error_relative_to_output_norm": max(mismatch_output_relative),
        "rollout_energy_ratio_quantiles": dict(zip(("min", "median", "p95", "max"), np.quantile(energy_ratios, [0, 0.5, 0.95, 1]).tolist())),
        "treatment_norm_quantiles": np.quantile(treatment_magnitudes, [0, 0.5, 0.95, 1]).tolist(),
    }
    return {
        "cell": cell, "model_id": identity["model_id"], "model_snapshot": identity["resolved_snapshot"],
        "fit_prompts": summary["fit_prompts"], "fit_sources": summary["fit_sources"],
        "eval_prompts": len(rows), "eval_sources": len(groups),
        "prompt_retention": retention,
        "baseline_correct": baseline_counts["correct"], "baseline_categories": dict(baseline_counts),
        "sources_with_correct_prompt": int((group_matrix[:, 0] > 0).sum()),
        "source_variant_count_distribution": dict(Counter(source_variant_counts.values())),
        "estimates": estimates, "flagged_reassignment": reassignment,
        "random_rates_by_replicate": (random_counts / baseline_counts["correct"]).tolist(),
        "norm_matching": norms,
    }


def write_tables(output: Path, cells: list[dict]) -> None:
    rows = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2pt}",
            r"\caption{Frozen-direction intervention on disjoint source questions.}", r"\label{tab:frozen-controls}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}", r"\toprule",
            r"Model & $n$ & PC1 & Random & Difference [95\% CI] \\", r"\midrule"]
    errors = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2pt}",
              r"\caption{Distinct-pair and repeated-label error rates.}", r"\label{tab:frozen-errors}",
              r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}", r"\toprule",
              r"& \multicolumn{2}{c}{PC1} & \multicolumn{2}{c}{Random} \\",
              r"Model & Distinct & Repeated & Distinct & Repeated \\", r"\midrule"]
    sensitivity = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2pt}",
                   r"\caption{Sensitivity to numerical tolerance violations.}", r"\label{tab:frozen-sensitivity}",
                   r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}", r"\toprule",
                   r"Model & $F$ & Effect range & Lower-endpoint CI \\", r"\midrule"]
    for cell in cells:
        x = cell["estimates"]["destroyed"]
        lo, hi = x["difference_ci95"]
        rows.append(f"{NAMES[cell['cell']]} & {cell['baseline_correct']} & {x['pc1']:.3f} & {x['random_mean']:.3f} & {x['difference']:.3f} [{lo:.3f}, {hi:.3f}] " + r"\\")
        a, b = cell["estimates"]["wrong_distinct_pair"], cell["estimates"]["repeated_label"]
        errors.append(f"{NAMES[cell['cell']]} & {a['pc1']:.3f} & {b['pc1']:.3f} & {a['random_mean']:.3f} & {b['random_mean']:.3f} " + r"\\")
        r = cell["flagged_reassignment"]
        rlo, rhi = r["lower_ci95"]
        sensitivity.append(f"{NAMES[cell['cell']]} & {r['flagged_random_rollouts']} & [{r['lower']:.3f}, {r['upper']:.3f}] & [{rlo:.3f}, {rhi:.3f}] " + r"\\")
    for name, lines in (("frozen_controls_table.tex", rows), ("frozen_error_table.tex", errors),
                        ("frozen_sensitivity_table.tex", sensitivity)):
        lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
        (output / name).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=29911)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--norms-only", action="store_true", help="Audit all six models' numerical matching without computing comparative effects")
    args = parser.parse_args()
    if args.norms_only:
        cells = [numerical_audit_cell(args.input, cell, args.smoke) for cell in CELLS]
        result = {"status": "complete", "stage": "numerical_only", "scientific_effects_computed": False,
                  "smoke": args.smoke, "cells": cells}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "numerical_audit.json").write_text(json.dumps(result, indent=2) + "\n")
        for cell in cells:
            print(json.dumps({k: v for k, v in cell.items() if k != "flagged_sites"}))
        return
    cells = [summarize_cell(args.input, cell, args.draws, args.seed, args.smoke) for cell in CELLS]
    result = {"status": "complete", "smoke": args.smoke, "draws": args.draws, "seed": args.seed,
              "interval_unit": "source question; all retained variants and fixed random replicates together",
              "conditional_on": "fitted direction, three random seeds per prompt, fixed checkpoints and sources",
              "simultaneous_bounds": False, "cells": cells}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.smoke:
        write_tables(args.output, cells)
    for cell in cells:
        print(json.dumps({k: cell[k] for k in ("cell", "baseline_correct", "estimates", "norm_matching")}))


if __name__ == "__main__":
    main()
