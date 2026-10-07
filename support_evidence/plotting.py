from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .artifacts import load_csv, repo_path


COLORS = {
    "pca": "#2F6B8F",
    "norm": "#8A8F98",
    "gold": "#4E4E4E",
    "random": "#C9CDD2",
    "qwen": "#59A14F",
    "mistral": "#E15759",
}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=220)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def _labels(rows):
    return [row.get("display", row.get("cell", "")) for row in rows]


def plot_fig1(out_dir: Path) -> None:
    rows = load_csv("artifacts/figures/fig1_selection.csv")
    x = range(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([i - 0.18 for i in x], [_float(r, "norm_auc") for r in rows], width=0.36, color=COLORS["norm"], label="Norm")
    axes[0].bar([i + 0.18 for i in x], [_float(r, "pca_auc") for r in rows], width=0.36, color=COLORS["pca"], label="PCA")
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("AUC")
    axes[0].set_xticks(list(x), _labels(rows), rotation=30, ha="right")
    axes[0].legend(frameon=False)
    axes[1].bar([i - 0.18 for i in x], [_float(r, "norm_top2_recovery") for r in rows], width=0.36, color=COLORS["norm"], label="Norm")
    axes[1].bar([i + 0.18 for i in x], [_float(r, "pca_top2_recovery") for r in rows], width=0.36, color=COLORS["pca"], label="PCA")
    axes[1].set_ylabel("Top-2 recovery")
    axes[1].set_xticks(list(x), _labels(rows), rotation=30, ha="right")
    axes[1].legend(frameon=False)
    _save(fig, out_dir, "fig1_selection")


def plot_fig2(out_dir: Path) -> None:
    rows = load_csv("artifacts/figures/fig2_causal.csv")
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - 0.25 for i in x], [_float(r, "gold_drop") for r in rows], width=0.25, color=COLORS["gold"], label="Gold")
    ax.bar(list(x), [_float(r, "pca_drop") for r in rows], width=0.25, color=COLORS["pca"], label="PCA")
    ax.bar([i + 0.25 for i in x], [_float(r, "random_drop") for r in rows], width=0.25, color=COLORS["random"], label="Random")
    ax.set_ylabel("Destruction rate")
    ax.set_xticks(list(x), _labels(rows), rotation=30, ha="right")
    ax.legend(frameon=False, ncol=3)
    _save(fig, out_dir, "fig2_causal")


def plot_fig3(out_dir: Path) -> None:
    rows = load_csv("artifacts/figures/fig3_dose_response.csv")
    by_cell = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cell, cell_rows in by_cell.items():
        cell_rows = sorted(cell_rows, key=lambda r: _float(r, "scale_factor"))
        ax.plot([_float(r, "scale_factor") for r in cell_rows], [_float(r, "accuracy") for r in cell_rows], marker="o", label=cell)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir, "fig3_dose_response")


def plot_simple_bar(csv_name: str, metric: str, out_dir: Path, name: str, ylabel: str) -> None:
    rows = load_csv(f"artifacts/figures/{csv_name}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(_labels(rows), [_float(r, metric) for r in rows], color=COLORS["pca"])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, out_dir, name)


def plot_all(out_dir: str | Path | None = None) -> None:
    out = Path(out_dir) if out_dir else repo_path("results", "generated_figures")
    plot_fig1(out)
    plot_fig2(out)
    plot_fig3(out)
    plot_simple_bar("fig4_transfer.csv", "transfer_auc", out, "fig4_transfer", "Transfer AUC")
    plot_simple_bar("fig5_head_knockout.csv", "mean_top_damage", out, "fig5_head_knockout", "Mean top-head damage")
    plot_simple_bar("fig6_label_controls.csv", "global_auc", out, "fig6_label_controls", "Global AUC")
    plot_simple_bar("fig7_bridge.csv", "pca_drop", out, "fig7_bridge", "PCA bridge drop")
