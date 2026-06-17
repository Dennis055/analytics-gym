# analytics-gym

Practice datasets and Jupyter notebooks for data-science case studies, solved with the [GAMMA](https://github.com/Dennis055/DNALib) toolkit (`GAMMA_DNA`).

## Projects

| Folder | Description |
|--------|-------------|
| [`google/`](google/) | Google workforce diversity clustering (unsupervised learning) |
| [`sony-research/`](sony-research/) | Sony Research telecom churn prediction (classification + drift-aware deployment) |
| [`N2O/`](N2O/) | N26 user transaction regression — predict total income & total expenses |

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
# GAMMA lives in DAPS_Brix — add to PYTHONPATH in each notebook
```

## Notebooks

- `google/datasets/diversity_clustering.ipynb` — sklearn / PCA / clustering baseline
- `google/gamma_sol.ipynb` — same brief, **GAMMA_DNA only**
- `sony-research/gamma_sol.ipynb` — churn case study, **GAMMA_DNA v2 only**
- `N2O/gamma_sol.ipynb` — N26 income/expense regression, **GAMMA_DNA v2 only** (`datasets 2/`)
