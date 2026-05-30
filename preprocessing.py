"""
preprocessing.py
----------------
Data loading, cleaning, and preprocessing for the fraud detection project.
Uses the Kaggle Credit Card Fraud Detection dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Path configuration ────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")

# We use a reproducible subset for performance (20 000 rows keeps class ratio)
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── Main functions ─────────────────────────────────────────────────────────────

def load_raw_data(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load the Credit Card Fraud dataset and return a stratified sample.
    Performs basic data quality checks.
    """
    df = pd.read_csv(DATA_PATH)

    # ── Quality checks ────────────────────────────────────────────────────────
    n_missing = df.isnull().sum().sum()
    n_duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    # ── Stratified sample to preserve fraud ratio ─────────────────────────────
    if sample_size and len(df) > sample_size:
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["Class"],
            random_state=RANDOM_STATE,
        )
        df = df.reset_index(drop=True)

    df.attrs["n_missing"] = n_missing
    df.attrs["n_duplicates"] = n_duplicates

    return df


def preprocess(df: pd.DataFrame):
    """
    Feature engineering + scaling + train/test split.

    Steps:
    1. Scale 'Amount' and 'Time' with StandardScaler (V1-V28 already scaled).
    2. Drop original Amount/Time columns.
    3. 80/20 stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = df.copy()

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    # ── Features / target ─────────────────────────────────────────────────────
    X = df.drop(columns=["Class"])
    y = df["Class"]
    feature_names = list(X.columns)

    # ── Train / test split (80/20, stratified) ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test, scaler, feature_names


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return high-level dataset statistics for the dashboard."""
    total = len(df)
    fraud = int(df["Class"].sum())
    legit = total - fraud
    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legit,
        "fraud_pct": round(fraud / total * 100, 4),
        "n_features": df.shape[1] - 1,
    }
