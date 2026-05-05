"""
Customer Churn Prediction - Data Preprocessing
================================================
Handles all the gruntwork: cleaning, encoding, scaling, and train/test splitting.

I keep preprocessing in its own module so the modelling script stays clean,
and so you can swap in a different dataset with minimal changes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load CSV, drop irrelevant columns, handle types.
    The original Kaggle dataset has RowNumber, CustomerId, and Surname —
    none of which carry predictive value, so we ditch them.
    """
    df = pd.read_csv(path)
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Sanity checks
    assert df.isnull().sum().sum() == 0, "Unexpected nulls found — investigate!"
    assert "Exited" in df.columns, "'Exited' target column missing from dataset"

    print(f"✓ Data loaded and cleaned  →  {df.shape[0]:,} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A few hand-crafted features that intuitively make sense for churn:

      - balance_to_salary: Customers with very high balance relative to salary
        might be more sensitive to service changes.
      - zero_balance_flag: Binary flag — having £0 in an account is a strong
        signal (many churners have zero balance).
      - age_group: Bucket ages into meaningful life-stage groups.
      - products_per_tenure: How many products adopted per year with the bank.
    """
    df = df.copy()

    # Balance-to-salary ratio (avoid div-by-zero just in case)
    df["balance_to_salary"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Zero balance flag
    df["zero_balance_flag"] = (df["Balance"] == 0).astype(int)

    # Age group buckets
    bins   = [0, 30, 40, 50, 60, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60+"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    # Products per year of tenure
    df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    print("✓ Feature engineering done  →  new features added")
    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics, split into X/y.

    Returns:
        X_train, X_test, y_train, y_test, feature_names (list), scaler
    """
    df = df.copy()

    # --- Encode ---
    # Label encode binary: Gender
    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])

    # One-hot encode Geography and age_group (drop_first avoids dummy trap)
    df = pd.get_dummies(df, columns=["Geography", "age_group"], drop_first=True)

    # --- Separate target ---
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    feature_names = X.columns.tolist()

    # --- Train / test split (stratified to preserve churn ratio) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Scale ---
    # Only scale continuous columns; binary / dummies don't need it
    scale_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "EstimatedSalary", "balance_to_salary", "products_per_tenure"
    ]
    scale_cols = [c for c in scale_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    # --- Persist artefacts ---
    joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le_gender,     os.path.join(MODELS_DIR, "le_gender.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    print(f"✓ Encoding & scaling done")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Churn rate  —  train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_names, scaler


def get_preprocessed_data():
    """One-stop function for the modelling script to call."""
    df = load_and_clean()
    df = engineer_features(df)
    return encode_and_scale(df)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feat_names, _ = get_preprocessed_data()
    print(f"\nFeatures used ({len(feat_names)}):")
    for f in feat_names:
        print(f"  • {f}")
