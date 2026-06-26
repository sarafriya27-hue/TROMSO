"""
run_analysis.py
Standalone script that reproduces the full analysis (cross-tabs, bias
diagnostics, ML modeling) WITHOUT Streamlit, saving all tables/plots to
the outputs/ folder. Useful for quick local runs, CI, or generating a
static report to share with stakeholders who don't have Streamlit running.

Usage:
    python run_analysis.py --input data/Insurance.csv --outdir outputs
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import (
    load_data, clean_data, crosstab_percent, run_bias_diagnostics,
    train_and_evaluate, metrics_table
)

sns.set_style("whitegrid")


def main(input_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    print(f"[1/5] Loading & cleaning data from {input_path} ...")
    raw_df = load_data(input_path)
    df = clean_data(raw_df)
    print(f"      -> {len(df)} usable rows after cleaning.")

    # ---------------------------------------------------------------- #
    # Objective 1: Cross-tabulation
    # ---------------------------------------------------------------- #
    print("[2/5] Running cross-tabulation analysis ...")
    cross_dims = ["AGE_BAND", "INCOME_BAND", "TEAM_GROUP", "PI_GENDER",
                  "OCCUPATION_GROUP", "MEDICAL_NONMED", "PAYMENT_MODE"]
    with pd.ExcelWriter(os.path.join(outdir, "crosstabs.xlsx")) as writer:
        for dim in cross_dims:
            ct_count, ct_pct = crosstab_percent(df, dim)
            ct_count.to_excel(writer, sheet_name=f"{dim[:20]}_count")
            ct_pct.to_excel(writer, sheet_name=f"{dim[:20]}_pct")
    print("      -> saved outputs/crosstabs.xlsx")

    # ---------------------------------------------------------------- #
    # Objective 2: Bias diagnostics
    # ---------------------------------------------------------------- #
    print("[3/5] Running bias diagnostics (chi-square + approval-rate gaps) ...")
    chi_df, approval_tables = run_bias_diagnostics(df)
    chi_df.to_csv(os.path.join(outdir, "chi_square_results.csv"), index=False)

    with pd.ExcelWriter(os.path.join(outdir, "approval_rate_by_dimension.xlsx")) as writer:
        for dim, tbl in approval_tables.items():
            tbl.to_excel(writer, sheet_name=dim[:30])

    for dim, tbl in approval_tables.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#C44E52" if v < df["TARGET"].mean()*100 else "#4C72B0"
                  for v in tbl["Approval_Rate"]]
        tbl["Approval_Rate"].plot(kind="bar", ax=ax, color=colors)
        ax.axhline(df["TARGET"].mean()*100, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Approval Rate (%)")
        ax.set_title(f"Approval Rate by {dim}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "plots", f"approval_rate_{dim}.png"), dpi=130)
        plt.close(fig)
    print("      -> saved chi_square_results.csv, approval_rate_by_dimension.xlsx, plots/")

    # ---------------------------------------------------------------- #
    # Objectives 3 & 4: Modeling
    # ---------------------------------------------------------------- #
    print("[4/5] Feature engineering + training KNN / DT / RF / GBM ...")
    results, X_test, y_test, preproc = train_and_evaluate(df)
    mt = metrics_table(results)
    mt.to_csv(os.path.join(outdir, "model_metrics.csv"))
    print(mt)

    # Train vs Test accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    import numpy as np
    x = np.arange(len(mt))
    width = 0.35
    ax.bar(x - width/2, mt["Train Accuracy"], width, label="Train Accuracy", color="#4C72B0")
    ax.bar(x + width/2, mt["Test Accuracy"], width, label="Test Accuracy", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(mt.index, rotation=15)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Train vs Test Accuracy per Model")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "plots", "train_vs_test_accuracy.png"), dpi=130)
    plt.close(fig)

    # Precision/Recall/F1
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    mt[["Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Precision / Recall / F1 by Model")
    plt.xticks(rotation=15); plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "plots", "precision_recall_f1.png"), dpi=130)
    plt.close(fig2)

    # ROC curves
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        ax3.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curves - All Models"); ax3.legend()
    plt.tight_layout()
    fig3.savefig(os.path.join(outdir, "plots", "roc_curves.png"), dpi=130)
    plt.close(fig3)

    # Confusion matrices
    fig4, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Repudiated", "Approved"],
                    yticklabels=["Repudiated", "Approved"], ax=ax, cbar=False)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    fig4.savefig(os.path.join(outdir, "plots", "confusion_matrices.png"), dpi=130)
    plt.close(fig4)

    # Feature importances
    tree_models = {n: r for n, r in results.items() if r["feature_importance"] is not None}
    fig5, axes5 = plt.subplots(1, len(tree_models), figsize=(6*len(tree_models), 5))
    if len(tree_models) == 1:
        axes5 = [axes5]
    for ax, (name, r) in zip(axes5, tree_models.items()):
        top_feat = r["feature_importance"].head(10)
        top_feat.sort_values().plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title(f"{name} - Top 10 Features")
    plt.tight_layout()
    fig5.savefig(os.path.join(outdir, "plots", "feature_importance.png"), dpi=130)
    plt.close(fig5)

    print("      -> saved model_metrics.csv and all plots/ images")

    # ---------------------------------------------------------------- #
    # Objective 5: Findings text file
    # ---------------------------------------------------------------- #
    print("[5/5] Writing findings.txt ...")
    sig_vars = chi_df[chi_df["significant_bias_(p<0.05)"] == "YES"]["variable"].tolist()
    lines = []
    lines.append("INSURANCE CLAIM SETTLEMENT - BIAS & PREDICTIVE ANALYSIS FINDINGS")
    lines.append("=" * 70)
    lines.append(f"Total claims analysed: {len(df)}")
    lines.append(f"Overall approval rate: {df['TARGET'].mean()*100:.1f}%")
    lines.append("")
    lines.append(f"Statistically significant bias dimensions (chi-square p<0.05): "
                 f"{', '.join(sig_vars) if sig_vars else 'None'}")
    lines.append("")
    lines.append("Approval-rate gap by dimension (max - min, percentage points):")
    for dim, tbl in approval_tables.items():
        gap = tbl["Approval_Rate"].max() - tbl["Approval_Rate"].min()
        flag = " <-- FLAGGED (>15pp)" if gap > 15 else ""
        lines.append(f"  - {dim}: {gap:.1f} pp{flag}")
    lines.append("")
    lines.append("Model performance summary:")
    lines.append(mt.to_string())
    lines.append("")
    best_model = mt["Test Accuracy"].idxmax()
    most_stable = mt["Overfit Gap (Train-Test Acc)"].abs().idxmin()
    lines.append(f"Best test accuracy: {best_model}")
    lines.append(f"Most stable (lowest overfit gap): {most_stable}")

    with open(os.path.join(outdir, "findings.txt"), "w") as f:
        f.write("\n".join(lines))
    print("      -> saved findings.txt")
    print("\nDone! All outputs saved in:", os.path.abspath(outdir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/Insurance.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    main(args.input, args.outdir)
