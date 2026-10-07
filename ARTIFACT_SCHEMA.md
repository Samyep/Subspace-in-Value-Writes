# Artifact Schema

## Prompt Rows

`artifacts/prompts/hotpot_400_eval_prompts.jsonl` contains one JSON object per
rendered prompt.

Important fields:

- `prompt_id`: stable prompt identifier.
- `example_id`: source HotpotQA example identifier.
- `example_rank`: frozen source-example rank.
- `family_name`: `active_set` or `working_set`.
- `lexical_pack`: prompt wording pack.
- `question`, `answer`: HotpotQA question and answer.
- `facts`: six candidate source facts with labels and support flags.
- `support_labels`: the two gold support labels.
- `prompt`: full model input text.

## Per-Example CSV Files

Per-example files use `prompt_id` or `example_id` to identify rows and contain
experiment-specific metric columns. Common fields include `cell`, selected
labels, gold labels, baseline correctness, intervention correctness, AUC, or
recovery indicators.

## Summary JSON Files

Summary files store aggregate metrics for one model cell. Common fields include
`cell`, `family`, `n_prompts` or `n_examples`, metric means, bootstrap
intervals, and short status labels.

## Figure CSV Files

`artifacts/figures/*.csv` are the source tables used by `scripts/plot_figures.py`.
Each row is aggregated to the panel level used in the paper figures.
