"""
Customer Churn Prediction - Inference / Prediction
====================================================
Once the model is trained, use this script to:
  1. Predict on a whole CSV of new customers, OR
  2. Predict for a single customer entered as a dict.

This is the kind of thing you'd expose as an API endpoint in production.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")


def load_artifacts():
    """Load the trained model, scaler, and feature list from disk."""
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    feats_path  = os.path.join(MODELS_DIR, "feature_names.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first."
        )

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(feats_path)
    return model, scaler, feature_names


def preprocess_single(customer: dict, scaler, feature_names: list) -> pd.DataFrame:
    """
    Convert a raw customer dict into the same feature space the model was trained on.
    Accepts the 'raw' fields (Geography, Gender as strings, etc.) and mirrors
    what preprocessing.py does.
    """
    df = pd.DataFrame([customer])

    # Feature engineering (mirrors preprocessing.py)
    df["balance_to_salary"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["zero_balance_flag"]   = (df["Balance"] == 0).astype(int)
    bins   = [0, 30, 40, 50, 60, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60+"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Encode Gender
    df["Gender"] = (df["Gender"] == "Male").astype(int)

    # One-hot Geography and age_group
    df = pd.get_dummies(df, columns=["Geography", "age_group"])

    # Align columns with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale continuous features
    scale_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "EstimatedSalary", "balance_to_salary", "products_per_tenure"
    ]
    scale_cols = [c for c in scale_cols if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df


def predict_single(customer: dict) -> dict:
    """
    Predict churn for a single customer.

    Returns dict with:
      - churn_probability : float  (0–1)
      - will_churn        : bool
      - risk_level        : str    (Low / Medium / High)
    """
    model, scaler, feature_names = load_artifacts()
    X = preprocess_single(customer, scaler, feature_names)

    prob = model.predict_proba(X)[0][1]
    will_churn = prob >= 0.5

    if prob < 0.30:
        risk = "Low"
    elif prob < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(float(prob), 4),
        "will_churn": bool(will_churn),
        "risk_level": risk,
    }


def predict_batch(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Predict churn for every customer in a CSV file.
    The CSV should have the same columns as the training data (minus 'Exited').
    """
    model, scaler, feature_names = load_artifacts()

    df_raw = pd.read_csv(csv_path)
    df_raw.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"],
                errors="ignore", inplace=True)

    # Feature engineering
    df = df_raw.copy()
    df["balance_to_salary"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["zero_balance_flag"]   = (df["Balance"] == 0).astype(int)
    bins   = [0, 30, 40, 50, 60, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60+"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df = pd.get_dummies(df, columns=["Geography", "age_group"])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    scale_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "EstimatedSalary", "balance_to_salary", "products_per_tenure"
    ]
    scale_cols = [c for c in scale_cols if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    probs = model.predict_proba(df)[:, 1]
    df_raw["churn_probability"] = probs.round(4)
    df_raw["predicted_churn"]   = (probs >= 0.5).astype(int)
    df_raw["risk_level"] = pd.cut(
        probs, bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"], right=True
    )

    if output_path:
        df_raw.to_csv(output_path, index=False)
        print(f"✓ Predictions saved → {output_path}")

    return df_raw


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # -- Single customer demo --
    sample_customer = {
        "CreditScore":     600,
        "Geography":       "Germany",   # Germany tends to have higher churn
        "Gender":          "Female",
        "Age":             42,
        "Tenure":          3,
        "Balance":         125000,
        "NumOfProducts":   1,
        "HasCrCard":       1,
        "IsActiveMember":  0,           # Inactive — red flag
        "EstimatedSalary": 101348,
    }

    print("\n" + "="*55)
    print("  SINGLE CUSTOMER PREDICTION")
    print("="*55)
    print("Customer profile:")
    for k, v in sample_customer.items():
        print(f"  {k:<22}: {v}")

    result = predict_single(sample_customer)
    print(f"\nPrediction:")
    print(f"  Churn probability : {result['churn_probability']:.2%}")
    print(f"  Will churn?       : {'Yes ⚠️' if result['will_churn'] else 'No ✅'}")
    print(f"  Risk level        : {result['risk_level']}")

    # -- Batch prediction demo --
    print("\n" + "="*55)
    print("  BATCH PREDICTION (first 10 rows of training data)")
    print("="*55)
    out_path = os.path.join(BASE_DIR, "reports", "batch_predictions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    preds = predict_batch(DATA_PATH, output_path=out_path)
    print(preds[["Geography", "Age", "Balance",
                  "churn_probability", "predicted_churn", "risk_level"]].head(10).to_string())
