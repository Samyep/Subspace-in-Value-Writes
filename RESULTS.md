# Results

| Result | Files | Command |
|---|---|---|
| Prompt split and prompt families | `artifacts/prompts/` | `python scripts/validate_artifacts.py` |
| Support-fact recovery | `artifacts/discovery/`, `artifacts/figures/fig1_selection.csv` | `python scripts/summarize_results.py` |
| Span ablation | `artifacts/span_ablation/`, `artifacts/figures/fig2_causal.csv` | `python scripts/summarize_results.py` |
| Dose response | `artifacts/interventions/dose_response/`, `artifacts/figures/fig3_dose_response.csv` | `python scripts/plot_figures.py` |
| Prompt-disjoint transfer | `artifacts/transfer/`, `artifacts/figures/fig4_transfer.csv` | `python scripts/summarize_results.py` |
| Head-level concentration | `artifacts/controls/head_knockout/`, `artifacts/figures/fig5_head_knockout.csv` | `python scripts/plot_figures.py` |
| Label and sign controls | `artifacts/controls/`, `artifacts/figures/fig6_label_controls.csv` | `python scripts/plot_figures.py` |
| Free-form citation bridge | `artifacts/bridge/`, `artifacts/figures/fig7_bridge.csv` | `python scripts/summarize_results.py` |
| Ranking baselines | `artifacts/headline_baselines/`, `artifacts/figures/headline_baselines.csv` | `python scripts/summarize_results.py` |
| Case studies | `artifacts/case_studies/` | `python scripts/inspect_case_studies.py` |
