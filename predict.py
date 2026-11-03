#!/usr/bin/env python
"""
Predict customer churn with the final model saved by main.ipynb (churn_final_model.joblib).

Command line:
    python predict.py new_customers.csv
    python predict.py new_customers.csv --output scored.csv
    python predict.py new_customers.csv --model churn_final_model.joblib --threshold 0.30

As a module:
    from predict import load_bundle, predict_churn
    bundle = load_bundle("churn_final_model.joblib")
    result = predict_churn(df_raw, bundle)

The input must have the same raw columns as the original IBM Telco CSV
(customerID and Churn are optional and ignored for scoring).

SECURITY: joblib files are pickles. Only load a model file that you created yourself,
and use the same package versions as in requirements.txt.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ENGINEERED_COLUMNS = ["AvgMonthlyCharge", "HasMultipleServices"]


def load_bundle(path="churn_final_model.joblib"):
    """Load the saved bundle (preprocessor, calibrated models, threshold, column lists)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run the last cell of main.ipynb to create it."
        )
    bundle = joblib.load(path)
    expected = {"preprocessor", "models", "threshold", "feature_columns", "service_cols"}
    missing = expected - set(bundle)
    if missing:
        raise ValueError(f"Model file is missing keys: {sorted(missing)}")
    return bundle


def prepare_raw(df_raw, bundle):
    """Reproduce the notebook's cleaning and feature engineering on raw rows."""
    feature_columns = bundle["feature_columns"]
    service_cols = bundle["service_cols"]

    required = [c for c in feature_columns if c not in ENGINEERED_COLUMNS]
    absent = [c for c in required if c not in df_raw.columns]
    if absent:
        raise ValueError(f"Input is missing required columns: {absent}")

    d = df_raw.copy()
    d = d.drop(columns=[c for c in ["customerID", "Churn"] if c in d.columns])
    d = d.replace([" ", "?"], np.nan)

    # Same rule as training: blank TotalCharges (tenure == 0) becomes 0
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)

    d["AvgMonthlyCharge"] = (
        d["TotalCharges"] / d["tenure"].replace(0, np.nan)
    ).fillna(d["MonthlyCharges"])

    n_services = sum((d[c] == "Yes").astype(int) for c in service_cols)
    d["HasMultipleServices"] = (n_services > 1).astype(int)

    d = d[feature_columns]

    nan_cols = d.columns[d.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"Missing values found in columns: {nan_cols}")
    return d


def predict_churn(df_raw, bundle, threshold=None):
    """Return churn probability and a 0/1 flag for each row of df_raw."""
    thr = bundle["threshold"] if threshold is None else float(threshold)

    d = prepare_raw(df_raw, bundle)
    pre = bundle["preprocessor"]
    X_enc = pd.DataFrame(pre.transform(d), columns=pre.get_feature_names_out(), index=d.index)

    # Equal-weight soft voting over the calibrated models stored in the bundle
    prob = np.mean([m.predict_proba(X_enc)[:, 1] for m in bundle["models"].values()], axis=0)

    return pd.DataFrame(
        {
            "churn_probability": prob.round(4),
            "churn_flag": (prob >= thr).astype(int),
        },
        index=df_raw.index,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("input_csv", help="CSV with the raw Telco columns")
    parser.add_argument("--model", default="churn_final_model.joblib", help="path to the saved model")
    parser.add_argument("--output", default="predictions.csv", help="where to write the results")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the saved decision threshold (0-1)")
    args = parser.parse_args(argv)

    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")

    try:
        bundle = load_bundle(args.model)
        df_raw = pd.read_csv(args.input_csv)
        result = predict_churn(df_raw, bundle, args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if "customerID" in df_raw.columns:
        result.insert(0, "customerID", df_raw["customerID"])

    result.to_csv(args.output, index=False)
    thr = bundle["threshold"] if args.threshold is None else args.threshold
    print(f"Scored {len(result)} customers (threshold = {thr:.3f}).")
    print(f"Flagged as likely churners: {int(result['churn_flag'].sum())} "
          f"({result['churn_flag'].mean():.1%}).")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
