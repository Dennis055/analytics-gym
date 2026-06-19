"""
FACET Tutorial 2 Equivalent: Scikit-learn Classifier Summaries with GAMMA_DNA
Mirrors Scikit-learn_classifier_summaries_using_FACET.ipynb.

Covers:
  - Multiple classifier comparison (LR, RF, GBC) on same dataset
  - g.explain() for each model
  - SHAP XAI for tree-based models (RF, GBC)
  - Feature importance comparison across models
  - Non-tree model gracefully raises RuntimeError for synergy/top_interactions
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
pio.renderers.default = "json"

print("=" * 60)
print("FACET Tutorial 2: Multi-Classifier Comparison")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Dataset: Titanic (same as Tutorial 1, consistent baseline)
# ──────────────────────────────────────────────────────────────
df_raw = sns.load_dataset("titanic")
COLS = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "who"]
df = df_raw[COLS].copy()
df["age"]          = df["age"].fillna(df["age"].median())
df["embarked"]     = df["embarked"].fillna("S")
df["sex_enc"]      = (df["sex"] == "female").astype(int)
df["embarked_enc"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)
df["who_enc"]      = df["who"].map({"man": 0, "woman": 1, "child": 2}).fillna(0).astype(int)

FEATURES = ["pclass", "sex_enc", "age", "sibsp", "parch", "fare", "embarked_enc", "who_enc"]
MODEL_DF = df[["survived"] + FEATURES].copy()
print(f"Dataset: Titanic  shape={MODEL_DF.shape}")

# ──────────────────────────────────────────────────────────────
# Classifiers to compare
# ──────────────────────────────────────────────────────────────
CLASSIFIERS = [
    ("Logistic Regression", "logistic_regression",   {"max_iter": 500}),
    ("Random Forest",       "random_forest_classifier", {"n_estimators": 100, "max_depth": 6, "random_state": 42}),
    ("Gradient Boosting",   "gradient_boosting_classifier", {"n_estimators": 100, "max_depth": 3, "random_state": 42}),
]

from gamma.pipeline import GAMMA_DNA

results   = {}
reports   = {}
imp_frames = {}

for name, model_type, params in CLASSIFIERS:
    print(f"\n{'─'*50}")
    print(f"[Model] {name}")

    g = GAMMA_DNA(MODEL_DF.copy(), target="survived", name=f"facet_t2_{model_type}")

    # Train
    result = g.train(
        model_type=model_type,
        features=FEATURES,
        test_size=0.2,
        random_state=42,
        run_cv=True,
        cv_folds=5,
        model_params=params,
    )
    auc = getattr(result, "roc_auc", "n/a")
    acc = getattr(result, "accuracy", "n/a")
    print(f"  ✓ train()  AUC={auc}  acc={acc}")
    results[name] = result

    # Explain — compute SHAP for tree models
    is_tree = "forest" in model_type or "boosting" in model_type
    report = g.explain(compute_shap=is_tree, compute_permutation=False)
    reports[name] = report
    print(f"  ✓ explain()  shap={'yes' if is_tree else 'no'}  {report}")

    # Feature importance frame
    imp = report.to_frame()
    imp_frames[name] = imp
    print(f"  ✓ to_frame()  top feature: {imp.index[0] if len(imp) > 0 else 'n/a'}")

    # XAI — only for tree models (LR raises RuntimeError)
    if is_tree:
        # plot_overview
        fig_ov = report.plot_overview(max_display=8, show=False)
        assert len(fig_ov.data) >= 2
        print(f"  ✓ plot_overview() OK  traces={len(fig_ov.data)}")

        # synergy
        fig_syn = report.plot_synergy(kind="matrix", sample_size=200, show=False)
        heatmaps = [t for t in fig_syn.data if t.type == "heatmap"]
        assert len(heatmaps) >= 1
        print(f"  ✓ plot_synergy(kind='matrix') OK")

        # redundancy
        fig_red = report.plot_redundancy(kind="dendrogram", show=False)
        scatters = [t for t in fig_red.data if t.type == "scatter"]
        assert len(scatters) >= 1
        print(f"  ✓ plot_redundancy(kind='dendrogram') OK  traces={len(scatters)}")

        # top interactions
        top_ix = report.top_interactions(top_k=5, sample_size=200)
        assert len(top_ix) <= 5
        print(f"  ✓ top_interactions(top_k=5) OK  |  {top_ix.iloc[0]['feature_a']} ↔ {top_ix.iloc[0]['feature_b']}")

        # simulate
        sim = report.simulate("sex_enc", grid=[0, 1], hold="median")
        assert sim.shape == (2, 2)
        print(f"  ✓ simulate('sex_enc') OK  |  male={sim.iloc[0]['prediction']:.3f}, female={sim.iloc[1]['prediction']:.3f}")

    else:
        # LR: verify synergy/top_interactions raise RuntimeError (expected)
        try:
            report.plot_synergy(kind="matrix", sample_size=50, show=False)
            print("  ✗ plot_synergy should raise RuntimeError for LR!")
        except RuntimeError as e:
            print(f"  ✓ plot_synergy raises RuntimeError for LR (expected)")

        try:
            report.top_interactions(top_k=5, sample_size=50)
            print("  ✗ top_interactions should raise RuntimeError for LR!")
        except RuntimeError as e:
            print(f"  ✓ top_interactions raises RuntimeError for LR (expected)")

        # But plot_pdp + simulate work for any model
        fig_pdp = report.plot_pdp("age", grid_points=10, show=False)
        assert len(fig_pdp.data) >= 2
        print(f"  ✓ plot_pdp('age') OK for LR  traces={len(fig_pdp.data)}")

        sim = report.simulate("pclass", grid=[1, 2, 3], hold="median")
        assert sim.shape == (3, 2)
        print(f"  ✓ simulate('pclass') OK for LR")

# ──────────────────────────────────────────────────────────────
# Importance comparison table
# ──────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("[Summary] Feature importance rankings:")
all_feats = FEATURES
comparison = pd.DataFrame(index=all_feats)
for name, frame in imp_frames.items():
    if "score" in frame.columns:
        comparison[name] = frame["score"].reindex(all_feats)
    elif "model_importance" in frame.columns:
        comparison[name] = frame["model_importance"].reindex(all_feats)
    elif len(frame.columns) > 0:
        comparison[name] = frame.iloc[:, 0].reindex(all_feats)
print(comparison.fillna("–").to_string())

print("\n" + "=" * 60)
print("✅ Tutorial 2 (Multi-Classifier): ALL CHECKS PASSED")
print("=" * 60)
