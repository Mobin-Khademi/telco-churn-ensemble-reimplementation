# Telco Customer Churn: Leak-Free Reimplementation of a Multi-Model Ensemble Paper

An independent reimplementation of the predictive pipeline described in El Attar & El-Hajj (2026), *"Explainable AI-driven customer churn prediction: a multi-model ensemble approach with SHAP-based feature analysis"*, *Frontiers in Artificial Intelligence* 9:1748799 ([doi:10.3389/frai.2026.1748799](https://doi.org/10.3389/frai.2026.1748799)), applied to the IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn).

**Headline result:** on a held-out test set the soft-voting ensemble reaches an **F1 of 0.633 for the churn class** (95% bootstrap CI 0.592–0.670) and a **ROC-AUC of 0.844**. This does **not** reproduce the ≈ 0.84 F1 reported in the paper; see [Why the numbers differ from the paper](#why-the-numbers-differ-from-the-paper).

## Repository contents

| File | Description |
|---|---|
| `main.ipynb` | The full pipeline, with outputs |
| `requirements.txt` | Pinned package versions used to produce the results |
| `.gitignore` | Excludes the dataset and the saved model |

The dataset (`telco_customer_churn.csv`) and the trained model (`churn_final_model.joblib`) are not stored in the repository. The model file is created by the last notebook cell.

## Pipeline

1. **Cleaning and feature engineering.** `customerID` is dropped, blank `TotalCharges` (customers with `tenure = 0`) is set to 0, and two features are added: `AvgMonthlyCharge` (`TotalCharges / tenure`, falling back to `MonthlyCharges`) and `HasMultipleServices`.
2. **Split.** Stratified 80/20 train/test split (5,634 / 1,409 customers).
3. **Encoding.** One-hot encoding fitted on the training set (47 features).
4. **Tuning.** `RandomizedSearchCV` (30 candidates, 5-fold stratified CV, scoring F1) for XGBoost, LightGBM and Gradient Boosting. **SMOTE is a step inside an `imblearn` pipeline**, so it is applied only to the training part of each fold and validation folds contain real data.
5. **Calibration.** Isotonic calibration (5-fold) fitted on the real, non-resampled class distribution.
6. **Ensemble.** Equal-weight soft voting of the three calibrated models.
7. **Threshold.** The F1-optimal decision threshold is chosen on **out-of-fold predictions of the training set** (result: 0.34), never on the test set.
8. **Evaluation.** One final evaluation on the untouched test set, with a bootstrap 95% confidence interval.
9. **Optional SHAP analysis** for the XGBoost model.
10. **Final model.** The last cell refits preprocessing and models on all data, saves them with the threshold to `churn_final_model.joblib`, and defines `predict_churn()` for new customers.

## Results

Test set (1,409 customers, 374 churners), single split. Thresholds were selected on training out-of-fold predictions.

| Model | Threshold | Precision | Recall | F1 (churn) | F1 (weighted) | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Soft Voting (calibrated)** | 0.34 | 0.557 | 0.733 | **0.633** | 0.783 | 0.844 | 0.137 |
| LightGBM (calibrated) | 0.32 | 0.538 | 0.765 | 0.631 | 0.774 | 0.843 | 0.137 |
| XGBoost (calibrated) | 0.34 | 0.556 | 0.727 | 0.630 | 0.782 | 0.844 | 0.136 |
| Gradient Boosting (calibrated) | 0.34 | 0.555 | 0.725 | 0.629 | 0.781 | 0.839 | 0.139 |
| Gradient Boosting (uncalibrated, 0.5) | 0.50 | 0.569 | 0.671 | 0.616 | 0.783 | 0.839 | 0.146 |
| XGBoost (uncalibrated, 0.5) | 0.50 | 0.586 | 0.639 | 0.611 | 0.787 | 0.845 | 0.139 |
| Soft Voting (calibrated, 0.5) | 0.50 | 0.650 | 0.521 | 0.579 | 0.791 | 0.844 | 0.137 |

(The notebook's final table lists all eleven variants.)

Soft-voting confusion matrix on the test set at threshold 0.34: 817 true negatives, 218 false positives, 100 false negatives, 274 true positives (derived from the reported precision/recall and support).

**How to read these numbers**

- **The models are statistically indistinguishable.** The F1 spread among the four calibrated models (0.629–0.633) is about 0.004, while the bootstrap CI of a single model is ± ~0.04 wide. The ensemble is a reasonable default for stability, not a proven winner. XGBoost alone performs almost identically and is simpler to explain with SHAP.
- **Calibration helped.** Brier score improved for all three models (XGBoost 0.1392 → 0.1364, LightGBM 0.1385 → 0.1365, Gradient Boosting 0.1459 → 0.1394).
- **Calibrated probabilities need a lower threshold.** With the real churn prior (≈ 26.5%), a threshold of 0.5 gives high precision but low recall. The selected threshold of 0.34 trades precision for recall (73% of churners caught, 56% of alerts correct).
- **Cross-validation and test now agree.** F1 on real validation folds during tuning was 0.62 (XGBoost), 0.60 (LightGBM) and 0.63 (Gradient Boosting), consistent with the test F1 of ≈ 0.63.

## Why the numbers differ from the paper

The paper (Table 7) reports accuracy, precision, recall and F1 of 0.84 for XGBoost, LightGBM, Gradient Boosting and the soft-voting ensemble, with ROC-AUC between 0.918 and 0.932. On a leak-free test set this repository obtains a churn-class F1 of ≈ 0.63, a weighted F1 of ≈ 0.78 and a ROC-AUC of ≈ 0.84.

What can and cannot be concluded:

- **The paper's 0.84 is probably not a churn-class F1.** Accuracy, precision, recall and F1 all being 0.84 is the pattern produced by weighted-average metrics (weighted recall always equals accuracy). The comparable number here is the weighted F1 (≈ 0.78). The paper does not state which averaging it uses, so this is an inference.
- **Averaging cannot explain the AUC gap.** ROC-AUC is threshold-free and not averaged over classes, yet the paper reports 0.93 (XGBoost) versus ≈ 0.84 here, and a Brier score of 0.106 versus ≈ 0.136 here. The evaluation therefore differs in some other way. The paper describes a leak-free protocol (SMOTE only on training folds), but it is not consistent about whether Table 7 comes from cross-validation or from the held-out test set (Sections 3.7.1 and 3.8.1 versus 4.3.1), so the cause cannot be identified from the text.
- **Calibration.** The paper states that calibration was fitted on the SMOTE-balanced training set (Section 3.4.4.2). The first version of this notebook did the same and calibration made the Brier score worse. In the paper's own Table 8, isotonic calibration also raised the Brier score of XGBoost, LightGBM and Gradient Boosting (e.g. 0.106 → 0.110 for XGBoost). Here, calibrating on the real distribution improved the Brier score of all three models.
- **How a protocol error can inflate scores.** The first version of this notebook applied SMOTE before cross-validation and obtained a CV F1 of ≈ 0.86 against a test F1 of ≈ 0.59. This shows how easily such a mistake yields numbers on the paper's scale; it is not a claim that the paper made it.

Problems found in the first version and fixed here:

- SMOTE applied before CV: moved inside an `imblearn` pipeline.
- Isotonic calibration fitted on SMOTE-balanced data: now fitted on the real distribution.
- Decision threshold tuned on the test set: now tuned on training out-of-fold predictions.
- LightGBM `bagging_fraction` had no effect without `bagging_freq`: `subsample_freq=1` is now fixed.

## Differences from the paper's description

This is an approximation of the paper's method, not an exact replica:

| Aspect | Paper | This repository |
|---|---|---|
| Missing `TotalCharges` (11 rows) | median imputation | set to 0 (these customers have `tenure = 0`) |
| `AvgMonthlyCharge` | `TotalCharges / (tenure + 1)` | `TotalCharges / tenure`, falling back to `MonthlyCharges` when `tenure = 0` |
| `HasMultipleServices` | sum over service categories (defined as a count) | binary: more than one of 8 service columns (`InternetService` not counted) |
| Contract | additionally encoded numerically (1 / 12 / 24) | one-hot only |
| Scaling | `StandardScaler` | none (tree models are scale-invariant) |
| Tuning | Optuna (objective stated as AUC in one place and F1 in another; best values not reported) | `RandomizedSearchCV`, 30 candidates, F1 |
| Models | seven models plus autoencoder segmentation | only XGBoost, LightGBM, Gradient Boosting and their soft vote |
| Library versions | scikit-learn 1.3.0, XGBoost 1.7.0, LightGBM 4.1.0 | see `requirements.txt` |

## Usage

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1. Download the IBM Telco Customer Churn dataset and save it as `telco_customer_churn.csv` next to the notebook.
2. Open `main.ipynb` and run all cells. The out-of-fold cell (cell 14) is the slowest; lower `n_iter` in the `tune` helper to speed up experiments.
3. The last cell writes `churn_final_model.joblib` and defines `predict_churn(df_raw)`, which takes raw rows with the original CSV columns and returns `churn_probability` and `churn_flag`.

Notes on the saved model:

- It is a pickle-based joblib file: load it only if you created it yourself, and only with the package versions in `requirements.txt`.
- The helper functions `prepare_raw` and `predict_churn` live in the notebook and are **not** stored inside the file. For standalone use, move them into a `predict.py`.
- The final model is refit on all 7,043 customers (train and test). The reported test metrics therefore describe the train-only models, and the deployed model has no held-out evaluation of its own.

## Limitations

- One train/test split; results carry sampling noise (see the bootstrap CI).
- The threshold maximizes F1. In practice it should be set from the business cost of a missed churner versus a wasted retention offer; the final notebook cell supports a cost-based option (`USE_COST_BASED_THRESHOLD`).
- Only one dataset; no external validation.
- SHAP explains the XGBoost model, not the whole ensemble.

## References

- El Attar, A., & El-Hajj, M. (2026). Explainable AI-driven customer churn prediction: a multi-model ensemble approach with SHAP-based feature analysis. *Frontiers in Artificial Intelligence*, 9, 1748799. https://doi.org/10.3389/frai.2026.1748799
- IBM Telco Customer Churn dataset, as distributed on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
