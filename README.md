# GIRL: Geometric Invariant Risk Learning for Robust Traffic Safety Prediction

> **Geometric Invariant Risk Learning (GIRL)** combines environment-level invariant objectives with graph-based spatial regularization to produce robust traffic collision severity predictors under distribution shift — across cities, across years, and across policy regimes.

---

## Overview

Predictive models for traffic safety are increasingly deployed in environments that differ from their training data. Policy changes, infrastructure variation, and enforcement differences create distribution shifts that cause standard ERM-trained models to degrade — precisely in the cities or time periods where reliability matters most.

GIRL addresses this by jointly optimizing three objectives:

- **Empirical risk** — standard cross-entropy over training environments
- **Invariance regularization** — IRM-style penalty encouraging stable representations across environments
- **Geometric smoothness** — graph Laplacian penalty that constrains representations to vary smoothly over geographic space

An extension, **GIRL+DRO(G)**, adds coarse-group distributionally robust optimization on top of the GIRL objective to further improve worst-case performance.

---

## Repository Structure

.
├── models_v2.py                        # All model definitions and run_* training runners
├── one_city_out_v2_fixed.ipynb         # Protocol A: Cross-city OOD experiments
├── temporal_ood_v2_fixed.ipynb         # Protocol B: Temporal OOD experiments
├── policy_shift_v3_fixed.ipynb         # Protocol C: Policy-shift OOD experiments
├── policy_shift_orig.ipynb             # Original policy-shift analysis notebook
├── GIRL_Experimental_Guideline.docx    # Full experimental specification
└── results/                            # Auto-generated: per-seed JSON logs & checkpoints
    ├── cross_city_ood/
    ├── temporal_ood/
    └── policy_shift_*/

---

## Method

GIRL learns a GNN encoder `φ_θ` that maps collision events (with spatial coordinates) through a geographic graph, then passes representations to a linear classifier `h_w`. The training objective is:

L_GIRL(θ, w) = Σ_e R_e(θ, w)          ← ERM risk
             + λ Σ_e ‖∇_w R_e(θ, w)‖²  ← invariance penalty
             + μ tr(Z^T L Z)            ← geometric smoothness

where `L` is the graph Laplacian of a spatial kNN graph (k=20, Gaussian edge weights) constructed over collision events.

Special cases:
- `λ=0, μ=0` → standard GNN-ERM
- `μ=0` → IRM-style invariant learning
- `λ=0` → graph-regularized ERM (manifold smoothing only)

---

## Experimental Protocols

### Protocol A — Cross-City OOD
Leave-one-city-out evaluation. Train on all cities except one; test on the held-out city. Reports MeanEnvAcc, WorstEnvAcc, MeanEnvF1, WorstEnvF1.

### Protocol B — Temporal OOD
Train on 2013–2019, validate on 2020–2021, test on 2022–2024. Reports Accuracy, Macro-F1, ECE, NLL.

### Protocol C — Policy Shift
Train/validate on school-zone incidents, test on out-of-zone incidents. Reports Accuracy, Macro-F1, ECE, NLL, and reliability diagrams.

---

## Dataset

Experiments use a multi-year traffic collision dataset from **San Bernardino County, California**, sourced from California's [Traffic Injury Mapping System (TIMS)](https://tims.berkeley.edu/), spanning **2013–2024**. The dataset covers pedestrian and bicycle crashes, with particular focus on incidents occurring within or near school-zone boundaries.

> **Note:** The raw data is not included in this repository. Download it from TIMS and place the processed files in the expected path before running the notebooks.

---

## Baselines

| Category | Models |
|---|---|
| Non-graph | Logistic Regression, MLP (ERM), XGBoost |
| Graph | GNN-ERM, GNN-GeoReg |
| Invariant | IRM, VREx, GroupDRO, GNN-IRM |
| **Proposed** | **GIRL, GIRL+DRO(G), GIRL-VREx** |

---

## Evaluation Metrics

- **Performance:** Accuracy, Macro-F1, Balanced Accuracy
- **Robustness:** Worst-environment Accuracy, Worst-environment F1
- **Calibration:** ECE, NLL, Brier Score, Reliability Diagrams
- **Deployment utility:** Cost-sensitive utility under asymmetric FN/FP cost

---

## Installation

```bash
# Python 3.10+
pip install torch torch-geometric
pip install scikit-learn xgboost pandas numpy matplotlib
```

PyTorch Geometric installation depends on your CUDA version — see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Running Experiments

Open and run the notebooks in order for each protocol:

```bash
# Cross-city OOD
jupyter notebook one_city_out_v2_fixed.ipynb

# Temporal OOD
jupyter notebook temporal_ood_v2_fixed.ipynb

# Policy shift
jupyter notebook policy_shift_v3_fixed.ipynb
```

Each notebook runs all baselines and GIRL variants over 5 seeds and saves per-seed logs to `results/<protocol>/<model>/seed_<n>.json`.

---

## Reproducibility

- All experiments use **5 random seeds** (0–4); results are reported as mean ± std.
- Hyperparameters are tuned on the validation set only — the test set is never used for model selection.
- All hyperparameters and evaluation metrics are logged to JSON at the end of each run.
- Checkpoints are saved as `.pt` files (PyTorch models) or `.pkl` files (sklearn/XGBoost).

---

## Key Results

Across all three protocols, GIRL consistently improves worst-environment F1 and reduces calibration error (ECE) relative to ERM and graph-only baselines. Under asymmetric cost (FN/FP = 5), GIRL and GIRL+DRO(G) achieve higher decision utility than all non-invariant methods, with smaller performance degradation in the worst-performing city.


---

## License

This project is released for research purposes. See `LICENSE` for details.
