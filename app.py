"""
app.py
Streamlit Dashboard: Insurance Claim Settlement Bias & Predictive Analysis
---------------------------------------------------------------------------
Run locally:    streamlit run app.py
Deploy:         push this repo to GitHub, then deploy on share.streamlit.io
                (Streamlit Community Cloud) pointing at app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data, clean_data, crosstab_percent, run_bias_diagnostics,
    train_and_evaluate, metrics_table
)

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Claim Settlement Bias Analysis",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------------------------------- #
# SIDEBAR — DATA SOURCE
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Insurance Claims CSV", type=["csv"]
)
default_path = "data/Insurance.csv"

if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
    st.sidebar.success(f"Loaded uploaded file: {uploaded_file.name}")
else:
    try:
        raw_df = load_data(default_path)
        st.sidebar.info("Using bundled sample dataset (data/Insurance.csv)")
    except FileNotFoundError:
        st.error("No dataset found. Please upload a CSV file from the sidebar.")
        st.stop()

df = clean_data(raw_df)

st.sidebar.markdown("---")
st.sidebar.metric("Total Claims", len(df))
st.sidebar.metric("Approval Rate", f"{df['TARGET'].mean()*100:.1f}%")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📋 Cross-Tabulation", "🔍 Bias Diagnostics",
     "🤖 Predictive Modeling", "📝 Findings & Recommendations"]
)

# --------------------------------------------------------------------------- #
# PAGE: OVERVIEW
# --------------------------------------------------------------------------- #
if page == "🏠 Overview":
    st.title("📊 Insurance Claim Settlement — Bias & Predictive Analysis")
    st.markdown(
        "This dashboard investigates whether claim settlement outcomes "
        "(**Approved** vs **Repudiated**) show signs of bias across "
        "demographic and operational dimensions — age, income, gender, "
        "underwriting team/zone, occupation, and more — and builds "
        "classification models to predict settlement outcomes."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Approved", f"{(df['TARGET']==1).sum():,}",
                f"{df['TARGET'].mean()*100:.1f}%")
    col3.metric("Repudiated", f"{(df['TARGET']==0).sum():,}",
                f"{(1-df['TARGET'].mean())*100:.1f}%")
    col4.metric("Distinct Teams/Zones", df["TEAM"].nunique())

    st.subheader("Sample of Cleaned Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Numeric Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(df["PI_AGE"], bins=30, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Age Distribution")
    sns.histplot(df["PI_ANNUAL_INCOME"], bins=30, kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("Annual Income Distribution")
    axes[1].set_xlim(0, df["PI_ANNUAL_INCOME"].quantile(0.98))
    sns.histplot(df["SUM_ASSURED"], bins=30, kde=True, ax=axes[2], color="#C44E52")
    axes[2].set_title("Sum Assured Distribution")
    axes[2].set_xlim(0, df["SUM_ASSURED"].quantile(0.98))
    st.pyplot(fig)

    st.subheader("Overall Approval vs Repudiation")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    df["STATUS_LABEL"].value_counts().plot(
        kind="bar", color=["#4C72B0", "#C44E52"], ax=ax2
    )
    ax2.set_ylabel("Number of Claims")
    ax2.set_title("Claim Outcome Counts")
    st.pyplot(fig2)


# --------------------------------------------------------------------------- #
# PAGE: CROSS-TABULATION (Objective 1)
# --------------------------------------------------------------------------- #
elif page == "📋 Cross-Tabulation":
    st.title("📋 Descriptive Cross-Tabulation Analysis vs Policy Status")
    st.markdown(
        "Select a dimension below to see how claim approval/repudiation "
        "is distributed across its categories."
    )

    group_options = {
        "Age Band": "AGE_BAND",
        "Income Band": "INCOME_BAND",
        "Sum Assured Band": "SUM_ASSURED_BAND",
        "Team / Zone": "TEAM_GROUP",
        "Gender": "PI_GENDER",
        "Occupation Group": "OCCUPATION_GROUP",
        "Medical / Non-Medical": "MEDICAL_NONMED",
        "Payment Mode": "PAYMENT_MODE",
        "Early / Non-Early": "EARLY_NON",
        "Claim Reason Group": "REASON_GROUP",
        "State": "PI_STATE",
    }
    choice = st.selectbox("Choose dimension for cross-tabulation:",
                           list(group_options.keys()))
    col = group_options[choice]

    ct_count, ct_pct = crosstab_percent(df, col)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Counts")
        st.dataframe(ct_count, use_container_width=True)
    with c2:
        st.subheader("Row % (within each group)")
        st.dataframe(ct_pct, use_container_width=True)

    st.subheader(f"Approval Rate (%) by {choice}")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = ct_pct.sort_values("Approved", ascending=False) if "Approved" in ct_pct.columns else ct_pct
    plot_data["Approved"].plot(kind="bar", color="#4C72B0", ax=ax)
    ax.axhline(df["TARGET"].mean() * 100, color="red", linestyle="--",
               label="Overall Avg Approval Rate")
    ax.set_ylabel("Approval Rate (%)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    with st.expander("ℹ️ How to read this"):
        st.markdown(
            "- **Counts** table shows raw claim volume per category.\n"
            "- **Row %** table shows what % of each category's claims were "
            "approved vs repudiated.\n"
            "- The red dashed line marks the overall average approval rate "
            "— bars that deviate strongly from it across volume-significant "
            "groups are worth investigating for bias."
        )


# --------------------------------------------------------------------------- #
# PAGE: BIAS DIAGNOSTICS (Objective 2)
# --------------------------------------------------------------------------- #
elif page == "🔍 Bias Diagnostics":
    st.title("🔍 Diagnostic Analysis — Probing for Biased Settlement Behaviour")
    st.markdown(
        "We statistically test whether settlement outcome is **independent** "
        "of each dimension using a Chi-square test of independence. "
        "A statistically significant result (p < 0.05) suggests outcome is "
        "**not** random with respect to that dimension — a signal (not proof) "
        "of differential treatment that warrants deeper investigation."
    )

    chi_df, approval_tables = run_bias_diagnostics(df)

    st.subheader("Chi-Square Test Summary (sorted by strength of association)")
    st.dataframe(chi_df.style.format({"p_value": "{:.2e}"}), use_container_width=True)

    st.markdown(
        "**Interpretation:** Lower p-value ⇒ stronger evidence the variable "
        "is associated with claim outcome. Variables flagged `YES` should be "
        "examined closely, especially protected/sensitive attributes "
        "(age, income, gender) and operational ones (team/zone)."
    )

    st.markdown("---")
    st.subheader("Deep-Dive: Approval Rate by Dimension")

    tab_labels = list(approval_tables.keys())
    tabs = st.tabs(tab_labels)
    for tab, dim in zip(tabs, tab_labels):
        with tab:
            tbl = approval_tables[dim]
            col1, col2 = st.columns([1, 1.3])
            with col1:
                st.dataframe(tbl, use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = ["#C44E52" if v < df["TARGET"].mean()*100 else "#4C72B0"
                          for v in tbl["Approval_Rate"]]
                tbl["Approval_Rate"].plot(kind="bar", ax=ax, color=colors)
                ax.axhline(df["TARGET"].mean()*100, color="black", linestyle="--",
                           linewidth=1, label="Overall Avg")
                ax.set_ylabel("Approval Rate (%)")
                ax.set_title(f"Approval Rate by {dim}")
                ax.legend()
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            max_gap = tbl["Approval_Rate"].max() - tbl["Approval_Rate"].min()
            st.info(
                f"📌 Spread between highest and lowest approval-rate group "
                f"in **{dim}**: **{max_gap:.1f} percentage points** "
                f"(Highest: {tbl['Approval_Rate'].idxmax()} = "
                f"{tbl['Approval_Rate'].max():.1f}% | "
                f"Lowest: {tbl['Approval_Rate'].idxmin()} = "
                f"{tbl['Approval_Rate'].min():.1f}%)"
            )

    st.markdown("---")
    st.subheader("Multi-Dimensional View: Age × Income Approval Heatmap")
    pivot = df.pivot_table(
        index="AGE_BAND", columns="INCOME_BAND", values="TARGET",
        aggfunc="mean", observed=True
    ) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "Approval Rate (%)"})
    ax.set_title("Approval Rate (%): Age Band × Income Band")
    st.pyplot(fig)

    st.subheader("Multi-Dimensional View: Team × Gender Approval Heatmap")
    pivot2 = df.pivot_table(
        index="TEAM_GROUP", columns="PI_GENDER", values="TARGET",
        aggfunc="mean", observed=True
    ) * 100
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(pivot2, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax2,
                cbar_kws={"label": "Approval Rate (%)"})
    ax2.set_title("Approval Rate (%): Team × Gender")
    st.pyplot(fig2)


# --------------------------------------------------------------------------- #
# PAGE: PREDICTIVE MODELING (Objectives 3 & 4)
# --------------------------------------------------------------------------- #
elif page == "🤖 Predictive Modeling":
    st.title("🤖 Classification Modeling — KNN, Decision Tree, Random Forest, GBM")
    st.markdown(
        "**Feature engineering applied:** numeric scaling (age, income, sum "
        "assured), one-hot encoding of categoricals (gender, team, payment "
        "mode, early/non-early, medical status, occupation group, claim "
        "reason group), median/mode imputation for missing values, and "
        "stratified 70/30 train-test split."
    )

    test_size = st.slider("Test set size", 0.15, 0.4, 0.3, 0.05)

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training KNN, Decision Tree, Random Forest, Gradient Boosting..."):
            results, X_test, y_test, preproc = train_and_evaluate(df, test_size=test_size)
        st.session_state["results"] = results
        st.session_state["y_test"] = y_test
        st.success("Training complete!")

    if "results" in st.session_state:
        results = st.session_state["results"]
        y_test = st.session_state["y_test"]

        st.subheader("📊 Performance Metrics Summary")
        mt = metrics_table(results)
        st.dataframe(mt.style.background_gradient(cmap="Greens", axis=0),
                     use_container_width=True)

        st.subheader("Train vs Test Accuracy (Stability Check)")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(mt))
        width = 0.35
        ax.bar(x - width/2, mt["Train Accuracy"], width, label="Train Accuracy", color="#4C72B0")
        ax.bar(x + width/2, mt["Test Accuracy"], width, label="Test Accuracy", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(mt.index, rotation=15)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Train vs Test Accuracy per Model")
        st.pyplot(fig)

        st.subheader("Precision, Recall, F1-Score Comparison")
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        mt[["Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Precision / Recall / F1 by Model")
        plt.xticks(rotation=15)
        st.pyplot(fig2)

        st.subheader("ROC Curves (Model Stability / Discrimination Power)")
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        for name, r in results.items():
            ax3.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
        ax3.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curves — All Models")
        ax3.legend()
        st.pyplot(fig3)

        st.subheader("Confusion Matrices")
        cols = st.columns(4)
        for col, (name, r) in zip(cols, results.items()):
            with col:
                fig4, ax4 = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Repudiated", "Approved"],
                            yticklabels=["Repudiated", "Approved"], ax=ax4, cbar=False)
                ax4.set_title(name, fontsize=10)
                ax4.set_xlabel("Predicted")
                ax4.set_ylabel("Actual")
                st.pyplot(fig4)

        st.subheader("🌳 Feature Importance (Tree-Based Models)")
        st.markdown(
            "If demographic features (age, income, team, gender) rank highly "
            "here, the model is effectively learning — and reproducing — "
            "the same bias patterns seen in the diagnostics tab."
        )
        tree_models = {n: r for n, r in results.items() if r["feature_importance"] is not None}
        cols2 = st.columns(len(tree_models))
        for col, (name, r) in zip(cols2, tree_models.items()):
            with col:
                top_feat = r["feature_importance"].head(10)
                fig5, ax5 = plt.subplots(figsize=(5, 5))
                top_feat.sort_values().plot(kind="barh", ax=ax5, color="#55A868")
                ax5.set_title(f"{name} — Top 10 Features")
                st.pyplot(fig5)

        best_model = mt["Test Accuracy"].idxmax()
        most_stable = mt["Overfit Gap (Train-Test Acc)"].abs().idxmin()
        st.success(
            f"🏆 **Best Test Accuracy:** {best_model} "
            f"({mt.loc[best_model, 'Test Accuracy']:.3f}) | "
            f"⚖️ **Most Stable (lowest train-test gap):** {most_stable}"
        )
    else:
        st.info("Click **Train All Models** above to run the analysis.")


# --------------------------------------------------------------------------- #
# PAGE: FINDINGS
# --------------------------------------------------------------------------- #
elif page == "📝 Findings & Recommendations":
    st.title("📝 Findings & Recommendations")

    chi_df, approval_tables = run_bias_diagnostics(df)
    sig_vars = chi_df[chi_df["significant_bias_(p<0.05)"] == "YES"]["variable"].tolist()

    st.subheader("Key Statistical Findings")
    st.markdown(f"""
- Overall approval rate across the portfolio is **{df['TARGET'].mean()*100:.1f}%**
  ({(df['TARGET']==1).sum():,} approved vs {(df['TARGET']==0).sum():,} repudiated
  out of {len(df):,} claims).
- Chi-square testing found **{len(sig_vars)} of {len(chi_df)}** examined dimensions
  to be statistically associated with claim outcome (p < 0.05):
  **{', '.join(sig_vars) if sig_vars else 'None'}**.
- These associations do not on their own prove intentional bias, but they
  indicate the settlement outcome is *not independent* of these factors and
  merit a process/policy review — particularly for protected attributes
  like **age, gender, and income** and operational ones like **team/zone**.
""")

    for dim, tbl in approval_tables.items():
        gap = tbl["Approval_Rate"].max() - tbl["Approval_Rate"].min()
        if gap > 15:
            st.warning(
                f"⚠️ **{dim}**: a **{gap:.1f} pp** spread in approval rate "
                f"between **{tbl['Approval_Rate'].idxmax()}** "
                f"({tbl['Approval_Rate'].max():.1f}%) and "
                f"**{tbl['Approval_Rate'].idxmin()}** "
                f"({tbl['Approval_Rate'].min():.1f}%) — flagged for review."
            )

    st.subheader("Modeling Findings")
    st.markdown("""
- Train **All Models** in the *Predictive Modeling* tab to generate live
  metrics for KNN, Decision Tree, Random Forest, and Gradient Boosting.
- Tree-based feature-importance rankings reveal which factors most strongly
  drive the *predicted* settlement outcome — if demographic fields dominate
  the top of that list, it reinforces the diagnostic signal that the
  historical settlement process has been encoding bias the model has now learned.
- A model with high train accuracy but much lower test accuracy (large
  "Overfit Gap") is **unstable** and its feature-importance ranking should
  be treated cautiously; prefer the model with the best balance of test
  accuracy, F1-score, and ROC-AUC for any operational decisioning.
""")

    st.subheader("Recommendations")
    st.markdown("""
1. **Independent fairness audit**: Route flagged dimensions (team/zone,
   age band, income band, gender) through a formal fairness/actuarial audit
   before they are used in any rule-based or automated decisioning.
2. **Standardize settlement criteria**: Document objective, medically/
   actuarially justified criteria for approval/repudiation decisions to
   reduce discretion-driven variance across teams/zones.
3. **Monitor team-level approval rates** on an ongoing basis (control chart /
   dashboard alert) to catch drift or outlier teams early.
4. **Exclude or de-weight purely demographic features** (raw age, gender,
   income) from any automated underwriting/claims model unless there is a
   clear, defensible, actuarial justification — rely instead on claim-
   specific evidence (medical exam status, claim reason, documentation
   completeness).
5. **Re-run this analysis periodically** (e.g., quarterly) as new claims
   data accumulates, to verify whether corrective actions are closing the
   approval-rate gaps identified here.
""")

    st.caption(
        "This dashboard is a decision-support and diagnostic tool. "
        "Statistical association does not equal proof of intentional bias — "
        "findings should be validated with subject-matter experts, "
        "underwriting policy documentation, and legal/compliance review "
        "before any operational or HR action is taken."
    )
