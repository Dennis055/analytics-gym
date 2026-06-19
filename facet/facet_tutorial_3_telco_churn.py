"""
FACET Tutorial 3 Equivalent: Water Drilling Incident Classification → Telco Churn
Mirrors Water_Drilling_Incident_Classification_with_Facet.ipynb.

Uses Telco customer churn dataset (analytics-gym/Telco) as industrial-context
binary classification equivalent.

Covers:
  - Full EDA → clean → feature engineer → train → explain → XAI flow
  - GradientBoostingClassifier (primary) + RandomForest (comparison)
  - plot_overview, plot_synergy(kind='both'), plot_redundancy(kind='both')
  - top_interactions  (domain insight: which service interactions drive churn)
  - plot_pdp for key business features (tenure, MonthlyCharges, Contract)
  - simulate to quantify impact of contract type on churn probability
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = "json"

print("=" * 60)
print("FACET Tutorial 3: Telco Churn Classification")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. Load Telco dataset
# ──────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(REPO_ROOT, "analytics-gym", "Telco", "datasets", "Telco-Customer-Churn.json")

with open(DATA_PATH, "r") as f:
    raw = json.load(f)

df_raw = pd.json_normalize(raw)
print(f"Raw shape after JSON normalize: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns[:10])}...")

# ──────────────────────────────────────────────────────────────
# 2. Preprocessing — mirror gamma_sol.ipynb logic
# ──────────────────────────────────────────────────────────────
df = df_raw.copy()

# Rename nested columns if present (pd.json_normalize may prefix them)
col_map = {}
for col in df.columns:
    new = col.replace("customer.", "").replace("phone.", "").replace("internet.", "").replace("account.", "")
    col_map[col] = new
df = df.rename(columns=col_map)
df.columns = [c.strip() for c in df.columns]

print(f"\nColumns after rename: {list(df.columns)}")

# TotalCharges: convert to numeric, blank → NaN → fillna(median)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# SeniorCitizen: 0/1 → 'No'/'Yes' for consistency
if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype in [int, float]:
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna("No")

# Drop identifier
drop_cols = [c for c in ["customerID", "CustomerID", "customer_id"] if c in df.columns]
df = df.drop(columns=drop_cols, errors="ignore")

# Churn target: normalize to 0/1
target_col = None
for c in ["Churn", "churn"]:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    raise ValueError(f"Cannot find Churn column. Columns: {list(df.columns)}")

df["churn_binary"] = (df[target_col].astype(str).str.upper().isin(["YES", "1", "TRUE"])).astype(int)
print(f"\nChurn rate: {df['churn_binary'].mean():.1%}")

# ──────────────────────────────────────────────────────────────
# 3. Feature engineering
# ──────────────────────────────────────────────────────────────
YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "SeniorCitizen",
]
for col in YES_NO_COLS:
    enc_col = f"{col}_enc"
    if col in df.columns:
        df[enc_col] = df[col].astype(str).str.upper().isin(["YES", "1"]).astype(int)

# Contract: ordinal
if "Contract" in df.columns:
    df["Contract_enc"] = df["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    ).fillna(0).astype(int)

# InternetService: ordinal
if "InternetService" in df.columns:
    df["Internet_enc"] = df["InternetService"].map(
        {"No": 0, "DSL": 1, "Fiber optic": 2}
    ).fillna(0).astype(int)

# PaymentMethod: binary (electronic check = 1)
if "PaymentMethod" in df.columns:
    df["is_auto_payment"] = df["PaymentMethod"].str.lower().str.contains("auto|bank|credit").astype(int)

# Derived
if "tenure" in df.columns and "MonthlyCharges" in df.columns:
    df["charges_per_month_ratio"] = df["MonthlyCharges"] / (df["tenure"].clip(lower=1))

service_enc_cols = [c for c in df.columns if c.endswith("_enc") and c not in
                    ["Contract_enc", "Internet_enc"]]
if service_enc_cols:
    df["num_services"] = df[service_enc_cols].sum(axis=1)

# ──────────────────────────────────────────────────────────────
# 4. Build model-ready frame
# ──────────────────────────────────────────────────────────────
NUM_COLS = []
for c in ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_month_ratio", "num_services"]:
    if c in df.columns:
        NUM_COLS.append(c)

ENC_COLS = [c for c in df.columns if c.endswith("_enc")]
FEATURES = NUM_COLS + ENC_COLS
FEATURES = [f for f in FEATURES if f in df.columns]

# Remove any features with all-NaN
FEATURES = [f for f in FEATURES if df[f].notna().sum() > 0]
MODEL_DF = df[["churn_binary"] + FEATURES].dropna().copy()

print(f"Model-ready shape: {MODEL_DF.shape}")
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Churn rate after dropna: {MODEL_DF['churn_binary'].mean():.1%}")

if len(FEATURES) < 3:
    raise ValueError(f"Too few features ({FEATURES}). Check column renaming above.")

# ──────────────────────────────────────────────────────────────
# 5. GAMMA_DNA session + EDA
# ──────────────────────────────────────────────────────────────
from gamma.pipeline import GAMMA_DNA

g = GAMMA_DNA(MODEL_DF, target="churn_binary", name="facet_telco_churn")
print(f"\n[1] Session created: {g}")

print("\n[2] Data quality:")
try:
    g.quality()
    print("  ✓ quality() OK")
except Exception as e:
    print(f"  ⚠ quality(): {e}")

# ──────────────────────────────────────────────────────────────
# 6. Train — GradientBoosting (primary) + RandomForest
# ──────────────────────────────────────────────────────────────
print("\n[3] Training GradientBoostingClassifier:")
result_gbc = g.train(
    model_type="gradient_boosting_classifier",
    features=FEATURES,
    test_size=0.2,
    random_state=42,
    run_cv=True,
    cv_folds=5,
    model_params={"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
)
print(f"  ✓ GBC trained  AUC={getattr(result_gbc, 'roc_auc', 'n/a')}")

print("\n[4] Training RandomForestClassifier:")
result_rf = g.train(
    model_type="random_forest_classifier",
    features=FEATURES,
    test_size=0.2,
    random_state=42,
    model_params={"n_estimators": 100, "max_depth": 8, "random_state": 42},
)
print(f"  ✓ RF trained   AUC={getattr(result_rf, 'roc_auc', 'n/a')}")

# ──────────────────────────────────────────────────────────────
# 7. Explain (using last trained = RF; pass GBC explicitly)
# ──────────────────────────────────────────────────────────────
print("\n[5] Explain GBC with SHAP:")
report_gbc = g.explain(compute_shap=True, compute_permutation=False, result=result_gbc)
print(f"  ✓ GBC explain OK  {report_gbc}")

print("\n[6] Explain RF with SHAP:")
report_rf = g.explain(compute_shap=True, compute_permutation=False, result=result_rf)
print(f"  ✓ RF explain OK   {report_rf}")

# ──────────────────────────────────────────────────────────────
# 8. XAI on GBC
# ──────────────────────────────────────────────────────────────
print("\n[7] GBC — plot_overview:")
fig_ov = report_gbc.plot_overview(max_display=10, show=False)
assert len(fig_ov.data) >= 2
print(f"  ✓ plot_overview OK  traces={len(fig_ov.data)}")

print("\n[8] GBC — plot_synergy(kind='both'):")
fig_syn = report_gbc.plot_synergy(kind="both", sample_size=300, show=False)
heatmaps = [t for t in fig_syn.data if t.type == "heatmap"]
scatters  = [t for t in fig_syn.data if t.type == "scatter"]
assert len(heatmaps) >= 1
assert len(scatters) >= 1
print(f"  ✓ plot_synergy(kind='both') OK  heatmap={len(heatmaps)}, dendrogram={len(scatters)}")

print("\n[9] GBC — plot_redundancy(kind='both'):")
fig_red = report_gbc.plot_redundancy(kind="both", show=False)
heatmaps = [t for t in fig_red.data if t.type == "heatmap"]
assert len(heatmaps) >= 1
print(f"  ✓ plot_redundancy(kind='both') OK  heatmap={len(heatmaps)}")

print("\n[10] GBC — top_interactions:")
top_ix = report_gbc.top_interactions(top_k=10, sample_size=300)
assert list(top_ix.columns) == ["feature_a", "feature_b", "mean_abs_interaction"]
assert top_ix["mean_abs_interaction"].is_monotonic_decreasing
print(f"  ✓ top_interactions OK  top pair: {top_ix.iloc[0]['feature_a']} ↔ {top_ix.iloc[0]['feature_b']}")
print(top_ix.head(5).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 9. PDP on key business features
# ──────────────────────────────────────────────────────────────
pdp_feats = []
for cand in ["tenure", "MonthlyCharges", "Contract_enc", "charges_per_month_ratio"]:
    if cand in FEATURES:
        pdp_feats.append(cand)
        if len(pdp_feats) == 3:
            break

print(f"\n[11] GBC — plot_pdp for {pdp_feats}:")
for feat in pdp_feats:
    fig_pdp = report_gbc.plot_pdp(feat, grid_points=20, ice_samples=30, show=False)
    assert len(fig_pdp.data) >= 2
    print(f"  ✓ plot_pdp('{feat}') OK  traces={len(fig_pdp.data)}")

# ──────────────────────────────────────────────────────────────
# 10. Simulate: contract impact on churn
# ──────────────────────────────────────────────────────────────
print("\n[12] GBC — simulate contract impact on churn:")
if "Contract_enc" in FEATURES:
    sim = report_gbc.simulate("Contract_enc", grid=[0, 1, 2], hold="median")
    assert sim.shape == (3, 2)
    labels = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    for _, row in sim.iterrows():
        label = labels.get(int(row["Contract_enc"]), str(row["Contract_enc"]))
        print(f"  {label}: P(churn)={row['prediction']:.3f}")
    print(f"  ✓ simulate('Contract_enc') OK")
elif FEATURES:
    feat = FEATURES[0]
    col_vals = MODEL_DF[feat]
    grid = list(np.linspace(col_vals.min(), col_vals.max(), 5))
    sim = report_gbc.simulate(feat, grid=grid, hold="median")
    assert sim.shape == (5, 2)
    print(f"  ✓ simulate('{feat}', 5-point grid) OK")

# ──────────────────────────────────────────────────────────────
# 11. Same XAI checks on RF
# ──────────────────────────────────────────────────────────────
print("\n[13] RF — synergy + redundancy:")
fig_rf_syn = report_rf.plot_synergy(kind="matrix", sample_size=300, show=False)
assert any(t.type == "heatmap" for t in fig_rf_syn.data)
print(f"  ✓ RF plot_synergy(kind='matrix') OK")

fig_rf_red = report_rf.plot_redundancy(kind="matrix", show=False)
assert any(t.type == "heatmap" for t in fig_rf_red.data)
print(f"  ✓ RF plot_redundancy(kind='matrix') OK")

print("\n" + "=" * 60)
print("✅ Tutorial 3 (Telco Churn): ALL CHECKS PASSED")
print("=" * 60)
