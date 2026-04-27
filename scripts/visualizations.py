"""
visualizations.py — Publication-quality comparison plots for all models.

Usage (in notebook):
    import importlib
    import scripts.visualizations as viz
    importlib.reload(viz)

    models = {
        "GPT-2 (Ours)": (results,        ranking_metrics,        group_metrics),
        "GPT-2+Gender": (results_gender,  ranking_metrics_gender, group_metrics_gender),
    }

    viz.plot_all(models, k_values=K_VALUES, save_dir="figures")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from metrics import evaluate_ranking, disparate_impact

# ── Style ──────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "GPT-2 (Ours)":  "#2ecc71",
    "GPT-2+Gender":  "#e74c3c",
}

GENDER_COLORS = {"M": "#3498db", "F": "#e74c3c"}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
})


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(models, k_values=(1, 3, 5, 10), save_dir="figures",
             ilap_scores=None, fairness_scores=None):
    """
    Generate all comparison plots and save to save_dir.

    Parameters
    ----------
    models : dict
        {model_label: (results_list, ranking_metrics_dict, group_metrics_dict)}
        ranking_metrics_dict  = evaluate_ranking(results, k_values)
        group_metrics_dict    = compute_group_metrics(results, user_gender_map, k_values)
    k_values     : tuple
    save_dir     : str   folder to save figures (created if missing)
    ilap_scores     : dict  optional {model_label: ilap_dict}
    fairness_scores : dict  optional {model_label: {"DI","EO","DP","CF"}}
    """
    os.makedirs(save_dir, exist_ok=True)

    _plot_metric_at_k(models, k_values, save_dir)
    _plot_comparison_at_k(models, k=10, save_dir=save_dir)
    _plot_gender_breakdown(models, k=10, save_dir=save_dir)
    _plot_fairness_gap(models, k_values, save_dir=save_dir)
    _plot_hr_k_lines(models, k_values, save_dir=save_dir)
    if ilap_scores:
        _plot_ilap_comparison(ilap_scores, save_dir=save_dir)
    if fairness_scores:
        _plot_fairness_metrics(fairness_scores, save_dir=save_dir)
        _plot_fairness_heatmap(fairness_scores, ilap_scores, save_dir=save_dir)

    print(f"All figures saved to '{save_dir}/'")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — HR / MRR / NDCG @ K for each model (3-panel bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_metric_at_k(models, k_values, save_dir):
    """3 subplots: HR@K, MRR@K, NDCG@K — one grouped bar cluster per model."""
    metric_names = ["HR", "MRR", "NDCG"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Recommendation Performance Across K Values", fontsize=14, fontweight="bold")

    x = np.arange(len(k_values))
    n_models = len(models)
    bar_w = 0.7 / n_models

    for ax, metric in zip(axes, metric_names):
        for idx, (label, (_, rm, _)) in enumerate(models.items()):
            vals = [rm[k][metric] for k in k_values]
            offset = (idx - n_models / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=label,
                          color=MODEL_COLORS.get(label, f"C{idx}"),
                          alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(f"{metric}@K")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in k_values])
        ax.set_ylabel(metric)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_metrics_at_k.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Side-by-side bar: HR@10, MRR@10, NDCG@10 per model
# ══════════════════════════════════════════════════════════════════════════════

def _plot_comparison_at_k(models, k, save_dir):
    """Clean bar chart: one bar per model per metric at a fixed K."""
    metric_names = ["HR", "MRR", "NDCG"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Model Comparison at K={k}", fontsize=14, fontweight="bold")

    labels = list(models.keys())
    colors = [MODEL_COLORS.get(l, f"C{i}") for i, l in enumerate(labels)]

    for ax, metric in zip(axes, metric_names):
        vals = [rm[k][metric] for _, (_, rm, _) in models.items()]
        bars = ax.bar(labels, vals, color=colors, alpha=0.88,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{metric}@{k}")
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(save_dir, f"fig2_comparison_at_{k}.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Gender breakdown: Male vs Female HR@10 per model
# ══════════════════════════════════════════════════════════════════════════════

def _plot_gender_breakdown(models, k, save_dir):
    """Grouped bars: M and F HR@K side-by-side for every model."""
    labels = list(models.keys())
    male_vals = [gm["M"][k]["HR"] for _, (_, _, gm) in models.items()]
    female_vals = [gm["F"][k]["HR"] for _, (_, _, gm) in models.items()]

    x = np.arange(len(labels))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_m = ax.bar(x - bar_w / 2, male_vals, bar_w,
                    label="Male", color=GENDER_COLORS["M"], alpha=0.88)
    bars_f = ax.bar(x + bar_w / 2, female_vals, bar_w,
                    label="Female", color=GENDER_COLORS["F"], alpha=0.88)

    for bar in list(bars_m) + list(bars_f):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_title(f"HR@{k} by Gender — All Models", fontweight="bold")
    ax.set_ylabel(f"HR@{k}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, max(male_vals + female_vals) * 1.22)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, f"fig3_gender_breakdown_hr{k}.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Fairness gap: |HR_M − HR_F| and Disparate Impact per model
# ══════════════════════════════════════════════════════════════════════════════

def _plot_fairness_gap(models, k_values, save_dir):
    """Two panels: absolute HR gap and Disparate Impact for K=10."""
    k = 10
    labels = list(models.keys())
    colors = [MODEL_COLORS.get(l, f"C{i}") for i, l in enumerate(labels)]

    gaps = [
        abs(gm["M"][k]["HR"] - gm["F"][k]["HR"])
        for _, (_, _, gm) in models.items()
    ]
    dis = [
        disparate_impact(gm, k)
        for _, (_, _, gm) in models.items()
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fairness Comparison at K={k}", fontsize=14, fontweight="bold")

    # Panel 1 — HR gap (lower is fairer)
    bars = ax1.bar(labels, gaps, color=colors, alpha=0.88)
    for bar, v in zip(bars, gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title(f"|HR_Male − HR_Female| @ {k}\n(lower = fairer)")
    ax1.set_ylabel("Absolute HR Gap")
    ax1.set_ylim(0, max(gaps) * 1.3 if max(gaps) > 0 else 0.1)
    ax1.tick_params(axis="x", rotation=15)

    # Panel 2 — Disparate Impact (≥ 0.8 = fair threshold)
    bars2 = ax2.bar(labels, dis, color=colors, alpha=0.88)
    ax2.axhline(0.8, color="#e67e22", linestyle="--", linewidth=1.5,
                label="Fair threshold (DI=0.8)")
    ax2.axhline(1.0, color="#27ae60", linestyle=":", linewidth=1.2,
                label="Perfect parity (DI=1.0)")
    for bar, v in zip(bars2, dis):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_title(f"Disparate Impact @ {k}\n(≥ 0.8 = fair)")
    ax2.set_ylabel("Disparate Impact")
    ax2.set_ylim(0, max(dis) * 1.3 if max(dis) > 0 else 1.5)
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, f"fig4_fairness_gap_k{k}.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — HR@K line chart: all models across K values
# ══════════════════════════════════════════════════════════════════════════════

def _plot_hr_k_lines(models, k_values, save_dir):
    """Line chart showing HR@K curve for each model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, (_, rm, _)) in enumerate(models.items()):
        hrs = [rm[k]["HR"] for k in k_values]
        color = MODEL_COLORS.get(label, f"C{idx}")
        ax.plot(k_values, hrs, marker="o", label=label,
                color=color, linewidth=2, markersize=6)
        for k, h in zip(k_values, hrs):
            ax.annotate(f"{h:.3f}", (k, h),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=color)

    ax.set_title("HR@K — All Models", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Hit Rate (HR@K)")
    ax.set_xticks(k_values)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "fig5_hr_k_lines.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — ILAP fairness metrics bar chart across all models
# ══════════════════════════════════════════════════════════════════════════════

def _plot_ilap_comparison(ilap_scores, save_dir):
    """Bar chart comparing key ILAP fairness metrics across all models."""
    ilap_keys = ["DF", "VU", "AU", "NU", "GCE"]
    labels = list(ilap_scores.keys())
    colors = [MODEL_COLORS.get(l, f"C{i}") for i, l in enumerate(labels)]

    n_metrics = len(ilap_keys)
    x = np.arange(n_metrics)
    n_models = len(labels)
    bar_w = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("ILAP Fairness Metrics — All Models (lower = fairer)",
                 fontsize=14, fontweight="bold")

    for idx, label in enumerate(labels):
        ilap = ilap_scores[label]
        vals = [abs(ilap.get(k, 0)) for k in ilap_keys]
        offset = (idx - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=label,
                      color=MODEL_COLORS.get(label, f"C{idx}"),
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(ilap_keys)
    ax.set_ylabel("Absolute value (lower = fairer)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig6_ilap_comparison.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"  Saved: {path}")

# =============================================================================
# Figure 7 — DI / EO / DP / CF bar chart across all models
# =============================================================================

def _plot_fairness_metrics(fairness_scores, save_dir):
    metrics_def = [
        ("Disparate Impact",        "DI",  0.8,  ">=0.8 = fair"),
        ("Equalized Opportunity",   "EO",  None, "lower = fairer"),
        ("Demographic Parity",      "DP",  None, "lower = fairer"),
        ("Counterfactual Fairness", "CF",  None, "higher = fairer"),
    ]
    labels = list(fairness_scores.keys())
    colors = [MODEL_COLORS.get(l, "C{}".format(i)) for i, l in enumerate(labels)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Fairness Metrics Comparison — All Models", fontsize=14, fontweight="bold")

    for ax, (title, key, threshold, note) in zip(axes, metrics_def):
        vals = [fairness_scores[l].get(key, 0) for l in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
        if threshold is not None:
            ax.axhline(threshold, color="#e67e22", linestyle="--", linewidth=1.5,
                       label="Threshold ({})".format(threshold))
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.02 if max(vals) > 0 else 0.02,
                    "{:.3f}".format(v), ha="center", va="bottom", fontsize=8.5)
        ax.set_title("{}\n({})".format(title, note), fontsize=10)
        ax.set_ylabel(key)
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig7_fairness_metrics.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print("  Saved: {}".format(path))


# =============================================================================
# Figure 8 — Fairness heatmap: all metrics x all models
# =============================================================================

def _plot_fairness_heatmap(fairness_scores, ilap_scores, save_dir):
    labels = list(fairness_scores.keys())

    row_defs = [
        ("Disparate Impact",        "DI",  "fair", True),
        ("Equalized Opportunity",   "EO",  "fair", False),
        ("Demographic Parity",      "DP",  "fair", False),
        ("Counterfactual Fairness", "CF",  "fair", True),
        ("Diff. Fairness (DF)",     "DF",  "ilap", False),
        ("Value Unfairness (VU)",   "VU",  "ilap", False),
        ("Abs. Unfairness (AU)",    "AU",  "ilap", False),
        ("Non-Parity (NU)",         "NU",  "ilap", False),
        ("KL Divergence (GCE)",     "GCE", "ilap", False),
    ]
    if ilap_scores is None:
        row_defs = [r for r in row_defs if r[2] == "fair"]

    matrix, row_labels, hib_flags = [], [], []
    for display, key, source, hib in row_defs:
        row = []
        for label in labels:
            if source == "fair":
                row.append(fairness_scores.get(label, {}).get(key, 0))
            else:
                row.append(abs(ilap_scores.get(label, {}).get(key, 0)))
        matrix.append(row)
        row_labels.append(display)
        hib_flags.append(hib)

    matrix = np.array(matrix, dtype=float)
    norm_matrix = np.zeros_like(matrix)
    for i, hib in enumerate(hib_flags):
        row = matrix[i]
        mn, mx = row.min(), row.max()
        if mx > mn:
            nr = (row - mn) / (mx - mn)
            norm_matrix[i] = nr if hib else 1 - nr
        else:
            norm_matrix[i] = 0.5

    fig, (ax_heat, ax_vals) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]}
    )
    fig.suptitle("Fairness Metrics Heatmap — All Models", fontsize=14, fontweight="bold")

    im = ax_heat.imshow(norm_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=15, ha="right")
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels)
    ax_heat.set_title("Relative fairness (green = better)", fontsize=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.03)

    ax_vals.axis("off")
    table_data = [["{:.4f}".format(matrix[r, c]) for c in range(len(labels))]
                  for r in range(len(row_labels))]
    tbl = ax_vals.table(cellText=table_data, rowLabels=row_labels,
                        colLabels=labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)
    ax_vals.set_title("Raw values", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig8_fairness_heatmap.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print("  Saved: {}".format(path))
