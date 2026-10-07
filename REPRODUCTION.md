# Reproduction

The commands below validate the released files and rebuild the figure panels
from CSV inputs.

## Artifact Validation

```bash
python scripts/validate_artifacts.py
```

This checks required prompt rows, figure CSVs, and basic schema
expectations.

## Result Summary

```bash
python scripts/summarize_results.py
```

This prints the main support-fact recovery, causal ablation, transfer, bridge,
and baseline metrics.

## Figure Rebuild

```bash
python scripts/plot_figures.py
```

This command reads `artifacts/figures/*.csv` and writes regenerated plots to
`results/generated_figures/`.

## Prompt Inspection

```bash
python scripts/inspect_prompts.py --limit 3
```

This samples the frozen evaluation prompt file and prints row metadata.
