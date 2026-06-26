"""
utils.py
Shared data-loading, cleaning, feature-engineering, bias-diagnostics and
modeling utilities for the Insurance Claim Settlement Bias Analysis project.

Used by both:
  - app.py            (Streamlit dashboard)
  - run_analysis.py    (standalone script that reproduces the same analysis
                         outside Streamlit, e.g. for local testing / CI)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. DATA LOADING & CLEANING
# --------------------------------------------------------------------------- #
def load_data(path_or_buffer):
    """Load the raw insurance claims CSV."""
    df = pd.read_csv(path_or_buffer)
    return df


def clean_data(df_raw):
    """
    Clean raw data:
      - Strip commas / cast SUM_ASSURED and PI_ANNUAL_INCOME to numeric
      - Standardise text columns (strip whitespace, consistent case for ZONE)
      - Fill missing PI_OCCUPATION / REASON_FOR_CLAIM
      - Create binary target: TARGET = 1 if Approved, 0 if Repudiated
      - Create human-readable AGE_BAND and INCOME_BAND bins
      - Collapse ZONE into a clean TEAM column (some duplicate casing, e.g.
        'South' vs 'SOUTH')
    """
    df = df_raw.copy()

    # Numeric cleanup -------------------------------------------------------
    for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PI_AGE"] = pd.to_numeric(df["PI_AGE"], errors="coerce")

    # Text cleanup ------------------------------------------------------------
    text_cols = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON",
                 "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE",
                 "REASON_FOR_CLAIM", "POLICY_STATUS"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "NaN", "None", ""]), col] = np.nan

    # Normalise ZONE/TEAM casing duplicates (e.g. 'South' / 'SOUTH')
    df["TEAM"] = df["ZONE"].str.upper().str.strip()

    # Fill missing categoricals
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Specified")

    # Drop rows with missing critical numeric/target fields
    df = df.dropna(subset=["PI_AGE", "PI_ANNUAL_INCOME", "SUM_ASSURED",
                            "POLICY_STATUS"]).reset_index(drop=True)

    # Target ------------------------------------------------------------------
    df["TARGET"] = df["POLICY_STATUS"].apply(
        lambda x: 1 if "approved" in x.lower() else 0
    )
    df["STATUS_LABEL"] = df["TARGET"].map({1: "Approved", 0: "Repudiated"})

    # Age bands -----------------------------------------------------------
    age_bins = [0, 25, 35, 45, 55, 65, 120]
    age_labels = ["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]
    df["AGE_BAND"] = pd.cut(df["PI_AGE"], bins=age_bins, labels=age_labels)

    # Income bands (quintile-based so groups are roughly balanced) -----------
    try:
        df["INCOME_BAND"] = pd.qcut(df["PI_ANNUAL_INCOME"], q=5,
                                     labels=["Q1 (Lowest)", "Q2", "Q3", "Q4",
                                             "Q5 (Highest)"])
    except ValueError:
        df["INCOME_BAND"] = pd.cut(df["PI_ANNUAL_INCOME"], bins=5)

    # Sum assured bands
    try:
        df["SUM_ASSURED_BAND"] = pd.qcut(df["SUM_ASSURED"], q=4,
                                          labels=["Low", "Medium", "High",
                                                  "Very High"])
    except ValueError:
        df["SUM_ASSURED_BAND"] = pd.cut(df["SUM_ASSURED"], bins=4)

    # Collapse rare occupations into "Other" for modeling readability
    top_occ = df["PI_OCCUPATION"].value_counts().nlargest(15).index
    df["OCCUPATION_GROUP"] = df["PI_OCCUPATION"].where(
        df["PI_OCCUPATION"].isin(top_occ), "Other"
    )

    # Collapse rare teams into "Other Team"
    top_team = df["TEAM"].value_counts().nlargest(12).index
    df["TEAM_GROUP"] = df["TEAM"].where(df["TEAM"].isin(top_team), "Other Team")

    # Collapse rare claim reasons into "Other"
    top_reason = df["REASON_FOR_CLAIM"].value_counts().nlargest(10).index
    df["REASON_GROUP"] = df["REASON_FOR_CLAIM"].where(
        df["REASON_FOR_CLAIM"].isin(top_reason), "Other"
    )

    return df


# --------------------------------------------------------------------------- #
# 2. CROSS-TABULATION (Objective 1)
# --------------------------------------------------------------------------- #
def crosstab_percent(df, group_col, status_col="STATUS_LABEL", normalize="index"):
    """Return count crosstab and row-percentage crosstab."""
    ct_count = pd.crosstab(df[group_col], df[status_col])
    ct_pct = pd.crosstab(df[group_col], df[status_col], normalize=normalize) * 100
    return ct_count, ct_pct.round(2)


# --------------------------------------------------------------------------- #
# 3. BIAS DIAGNOSTICS (Objective 2)
# --------------------------------------------------------------------------- #
def chi_square_test(df, group_col, status_col="STATUS_LABEL"):
    """Run a Chi-square test of independence between group_col and status_col."""
    contingency = pd.crosstab(df[group_col], df[status_col])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        "variable": group_col,
        "chi2_statistic": round(chi2, 3),
        "p_value": p,
        "degrees_of_freedom": dof,
        "significant_bias_(p<0.05)": "YES" if p < 0.05 else "No"
    }


def approval_rate_by_group(df, group_col, target_col="TARGET"):
    """Approval rate (%) and volume per group, sorted descending by rate."""
    summary = df.groupby(group_col, observed=True)[target_col].agg(
        ["mean", "count"]
    ).rename(columns={"mean": "Approval_Rate", "count": "N_Claims"})
    summary["Approval_Rate"] = (summary["Approval_Rate"] * 100).round(2)
    summary = summary.sort_values("Approval_Rate", ascending=False)
    return summary


def run_bias_diagnostics(df):
    """
    Run chi-square independence tests across the key bias-suspect dimensions
    (age, income, team/zone, gender, occupation, medical exam status,
    payment mode) and return a consolidated summary DataFrame, plus
    per-dimension approval-rate tables.
    """
    dims = ["AGE_BAND", "INCOME_BAND", "TEAM_GROUP", "PI_GENDER",
            "OCCUPATION_GROUP", "MEDICAL_NONMED", "PAYMENT_MODE",
            "EARLY_NON", "SUM_ASSURED_BAND"]

    chi_results = [chi_square_test(df, d) for d in dims]
    chi_df = pd.DataFrame(chi_results).sort_values("p_value")

    approval_tables = {d: approval_rate_by_group(df, d) for d in dims}
    return chi_df, approval_tables


# --------------------------------------------------------------------------- #
# 4. FEATURE ENGINEERING + MODELING (Objectives 3 & 4)
# --------------------------------------------------------------------------- #
FEATURE_COLS_NUMERIC = ["PI_AGE", "PI_ANNUAL_INCOME", "SUM_ASSURED"]
FEATURE_COLS_CATEGORICAL = ["PI_GENDER", "TEAM_GROUP", "PAYMENT_MODE",
                             "EARLY_NON", "MEDICAL_NONMED", "OCCUPATION_GROUP",
                             "REASON_GROUP"]


def build_preprocessor():
    """
    ColumnTransformer that:
      - scales numeric features (important for KNN, harmless for trees)
      - one-hot encodes categorical features
      - imputes any stray missing values
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, FEATURE_COLS_NUMERIC),
        ("cat", categorical_pipe, FEATURE_COLS_CATEGORICAL)
    ])
    return preprocessor


def get_model_zoo():
    """Return dict of {model_name: sklearn estimator} for the 4 required algos."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
    }


def train_and_evaluate(df, test_size=0.3, random_state=RANDOM_STATE):
    """
    Full ML pipeline:
      1. Train/test split (stratified on target)
      2. Fit preprocessing + each of the 4 models
      3. Compute train/test accuracy, precision, recall, f1, ROC-AUC
      4. Compute confusion matrices and ROC curve points

    Returns
    -------
    results : dict keyed by model name -> dict of metrics/artifacts
    X_test, y_test : held-out data (for any extra inspection)
    """
    X = df[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL]
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = get_model_zoo()
    results = {}

    for name, model in models.items():
        model.fit(X_train_t, y_train)

        train_pred = model.predict(X_train_t)
        test_pred = model.predict(X_test_t)

        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test_t)[:, 1]
        else:
            test_proba = test_pred.astype(float)

        fpr, tpr, _ = roc_curve(y_test, test_proba)
        roc_auc = auc(fpr, tpr)

        cm = confusion_matrix(y_test, test_pred)

        # Feature importance (tree-based models only)
        feat_importance = None
        if hasattr(model, "feature_importances_"):
            feat_names = preprocessor.get_feature_names_out()
            feat_importance = pd.Series(
                model.feature_importances_, index=feat_names
            ).sort_values(ascending=False)

        results[name] = {
            "model": model,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred, zero_division=0),
            "f1": f1_score(y_test, test_pred, zero_division=0),
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": cm,
            "feature_importance": feat_importance,
        }

    return results, X_test, y_test, preprocessor


def metrics_table(results):
    """Tidy summary table of all model metrics for display/plotting."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Train Accuracy": round(r["train_accuracy"], 4),
            "Test Accuracy": round(r["test_accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1"], 4),
            "ROC-AUC": round(r["roc_auc"], 4),
            "Overfit Gap (Train-Test Acc)": round(
                r["train_accuracy"] - r["test_accuracy"], 4
            ),
        })
    return pd.DataFrame(rows).set_index("Model")
