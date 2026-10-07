#!/usr/bin/env python3
"""Render additional diagnostics from the complete, frozen V4 summaries."""
import argparse
import json
from pathlib import Path

from aggregate_frozen_controls_v4 import CELLS, NAMES


def write_table(directory, name, caption, label, spec, header, rows):
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{2pt}",
             "\\caption{" + caption + "}", "\\label{" + label + "}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + spec + "@{}}",
             r"\toprule", *header, r"\midrule", *rows, r"\bottomrule",
             r"\end{tabular*}", r"\end{table}"]
    (directory / name).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.directory / "numerical_audit.json").read_text())
    aggregate = json.loads((args.directory / "aggregate.json").read_text())
    assert audit["stage"] == "numerical_only" and not audit["scientific_effects_computed"]
    assert aggregate["status"] == "complete" and not aggregate["smoke"]
    assert tuple(c["cell"] for c in audit["cells"]) == CELLS
    assert tuple(c["cell"] for c in aggregate["cells"]) == CELLS
    norm_rows, replicate_rows = [], []
    for numerical, cell in zip(audit["cells"], aggregate["cells"]):
        assert numerical["baseline_correct"] == cell["baseline_correct"]
        errors = numerical["relative_error_quantiles"]
        norm_rows.append(
            f"{NAMES[cell['cell']]} & {100*errors['median']:.3f} & "
            f"{100*errors['p99']:.3f} & {100*errors['max']:.3f} & "
            f"{numerical['within_absolute_floor_only_sites']} & "
            f"{numerical['original_rule_failed_rollouts']} " + r"\\")
        rates = cell["random_rates_by_replicate"]
        assert len(rates) == 3
        replicate_rows.append(
            f"{NAMES[cell['cell']]} & {cell['sources_with_correct_prompt']} & "
            + " & ".join(f"{rate[0]:.3f}" for rate in rates) + " " + r"\\")
    write_table(args.directory, "frozen_norm_table.tex",
                "Realized edit-norm matching.", "tab:frozen-norms", "lrrrrr",
                [r"& \multicolumn{3}{c}{Relative error (\%)} & & \\",
                 r"Model & Median & 99th & Max. & Floor & $F$ \\"], norm_rows)
    write_table(args.directory, "frozen_replicates_table.tex",
                "Random-displacement destruction by replicate.", "tab:frozen-replicates", "lrrrr",
                [r"Model & Sources & Random 1 & Random 2 & Random 3 \\"], replicate_rows)


if __name__ == "__main__":
    main()
