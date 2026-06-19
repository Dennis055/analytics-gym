"""
FACET Tutorial 1 Equivalent: Classification with GAMMA_DNA
Mirrors Classification_with_Facet.ipynb — Titanic survival prediction.

Covers:
  - EDA (g.skim, g.quality, g.eda)
  - Training RandomForestClassifier
  - SHAP feature importance (g.explain)
  - plot_overview, plot_synergy, plot_redundancy (kind="both")
  - top_interactions
  - plot_pdp
  - simulate
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = "json"  # headless

print("=" * 60)
print("FACET Tutorial 1: Titanic Classification")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. Load & preprocess Titanic
# ──────────────────────────────────────────────────────────────
import seaborn as sns
df_raw = sns.load_dataset("titanic")
print(f"Raw shape: {df_raw.shape}")

COLS = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "who"]
df = df_raw[COLS].copy()

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna("S")

df["sex_enc"]      = (df["sex"] == "female").astype(int)
df["embarked_enc"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)
df["who_enc"]      = df["who"].map({"man": 0, "woman": 1, "child": 2}).fillna(0).astype(int)

FEATURES = ["pclass", "sex_enc", "age", "sibsp", "parch", "fare", "embarked_enc", "who_enc"]
MODEL_DF = df[["survived"] + FEATURES].copy()
print(f"Model-ready shape: {MODEL_DF.shape}, nulls: {MODEL_DF.isnull().sum().sum()}")

# ──────────────────────────────────────────────────────────────
# 2. GAMMA_DNA session
# ──────────────────────────────────────────────────────────────
from gamma.pipeline import GAMMA_DNA

g = GAMMA_DNA(MODEL_DF, target="survived", name="facet_titanic")
print("\n[1] Session created")
print(g)

# ──────────────────────────────────────────────────────────────
# 3. EDA
# ──────────────────────────────────────────────────────────────
print("\n[2] Quick data overview (skim):")
try:
    g.skim()
    print("  ✓ skim() OK")
except Exception as e:
    print(f"  ✗ skim(): {e}")

print("\n[3] Data quality:")
try:
    g.quality()
    print("  ✓ quality() OK")
except Exception as e:
    print(f"  ✗ quality(): {e}")

# ──────────────────────────────────────────────────────────────
# 4. Train
# ──────────────────────────────────────────────────────────────
print("\n[4] Training RandomForestClassifier:")
result = g.train(
    model_type="random_forest_classifier",
    features=FEATURES,
    test_size=0.2,
    random_state=42,
    run_cv=True,
    cv_folds=5,
    model_params={"n_estimators": 100, "max_depth": 6, "random_state": 42},
)
print(f"  ✓ train() OK  |  test AUC = {getattr(result, 'roc_auc', 'n/a')}")

# ──────────────────────────────────────────────────────────────
# 5. Explain
# ──────────────────────────────────────────────────────────────
print("\n[5] Feature importance (SHAP):")
report = g.explain(compute_shap=True, compute_permutation=False)
print(f"  ✓ explain() OK  |  {report}")

# ──────────────────────────────────────────────────────────────
# 6. XAI: Overview
# ──────────────────────────────────────────────────────────────
print("\n[6] plot_overview:")
fig = report.plot_overview(max_display=8, show=False)
assert len(fig.data) >= 2, "overview should have >= 2 traces"
print(f"  ✓ plot_overview OK  |  traces={len(fig.data)}")

# ──────────────────────────────────────────────────────────────
# 7. XAI: Synergy
# ──────────────────────────────────────────────────────────────
print("\n[7] Synergy analysis (interaction SHAP):")
fig_syn = report.plot_synergy(kind="both", sample_size=300, show=False)
heatmaps = [t for t in fig_syn.data if t.type == "heatmap"]
scatters  = [t for t in fig_syn.data if t.type == "scatter"]
assert len(heatmaps) >= 1, "synergy 'both' should have heatmap"
assert len(scatters) >= 1, "synergy 'both' should have dendrogram scatter"
print(f"  ✓ plot_synergy(kind='both') OK  |  heatmap={len(heatmaps)}, dendrogram traces={len(scatters)}")

fig_syn_m = report.plot_synergy(kind="matrix", sample_size=300, show=False)
print(f"  ✓ plot_synergy(kind='matrix') OK")

fig_syn_d = report.plot_synergy(kind="dendrogram", sample_size=300, show=False)
print(f"  ✓ plot_synergy(kind='dendrogram') OK")

# ──────────────────────────────────────────────────────────────
# 8. XAI: Redundancy
# ──────────────────────────────────────────────────────────────
print("\n[8] Redundancy analysis (SHAP vector correlation):")
fig_red = report.plot_redundancy(kind="both", show=False)
heatmaps = [t for t in fig_red.data if t.type == "heatmap"]
assert len(heatmaps) >= 1, "redundancy 'both' should have heatmap"
print(f"  ✓ plot_redundancy(kind='both') OK  |  heatmap={len(heatmaps)}")

# ──────────────────────────────────────────────────────────────
# 9. XAI: Top interactions
# ──────────────────────────────────────────────────────────────
print("\n[9] Top feature interactions:")
top_ix = report.top_interactions(top_k=10, sample_size=300)
assert list(top_ix.columns) == ["feature_a", "feature_b", "mean_abs_interaction"]
assert len(top_ix) <= 10
assert top_ix["mean_abs_interaction"].is_monotonic_decreasing
print(f"  ✓ top_interactions() OK  |  top pair: {top_ix.iloc[0]['feature_a']} ↔ {top_ix.iloc[0]['feature_b']}")
print(top_ix.head(5).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 10. XAI: PDP
# ──────────────────────────────────────────────────────────────
print("\n[10] Partial Dependence Plot:")
fig_pdp = report.plot_pdp("age", grid_points=20, ice_samples=50, show=False)
assert len(fig_pdp.data) >= 2, "PDP should have >= 2 traces"
print(f"  ✓ plot_pdp('age', ice_samples=50) OK  |  traces={len(fig_pdp.data)}")

fig_pdp2 = report.plot_pdp("fare", grid_points=20, show=False)
print(f"  ✓ plot_pdp('fare') OK")

# ──────────────────────────────────────────────────────────────
# 11. XAI: Simulate
# ──────────────────────────────────────────────────────────────
print("\n[11] Feature simulation:")
sim_age = report.simulate("age", grid=list(range(5, 75, 5)), hold="median")
assert sim_age.shape == (14, 2)
assert list(sim_age.columns) == ["age", "prediction"]
print(f"  ✓ simulate('age', grid=range(5,75,5)) OK")
print(f"  Age 5: {sim_age.iloc[0]['prediction']:.3f} → Age 70: {sim_age.iloc[-1]['prediction']:.3f}")

sim_class = report.simulate("pclass", grid=[1, 2, 3], hold="mean")
assert sim_class.shape == (3, 2)
print(f"  ✓ simulate('pclass', grid=[1,2,3]) OK")
print(f"  pclass 1={sim_class.iloc[0]['prediction']:.3f}, 2={sim_class.iloc[1]['prediction']:.3f}, 3={sim_class.iloc[2]['prediction']:.3f}")

print("\n" + "=" * 60)
print("✅ Tutorial 1 (Titanic Classification): ALL CHECKS PASSED")
print("=" * 60)
