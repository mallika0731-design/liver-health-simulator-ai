"""
preprocess.py
─────────────
Load, clean, and engineer features from ILPD.csv.

Steps
-----
1. Read raw CSV
2. Rename columns to consistent internal names
3. Impute / drop missing values
4. Encode gender (Male=1, Female=0)
5. Map Dataset: 1→1 (disease), 2→0 (healthy)
6. Normalize continuous features (MinMax to [0,1])
7. Return cleaned DataFrame + scaler
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Column alias map (original → internal) ──────────────────────────────────
COL_MAP = {
    "Age": "age",
    "Gender": "gender",
    "Total_Bilirubin": "total_bilirubin",
    "Direct_Bilirubin": "direct_bilirubin",
    "Alkaline_Phosphotase": "alkaline_phosphotase",
    "Alamine_Aminotransferase": "sgpt",
    "Aspartate_Aminotransferase": "sgot",
    "Total_Protiens": "total_proteins",
    "Albumin": "albumin",
    "Albumin_and_Globulin_Ratio": "ag_ratio",
    "Dataset": "liver_disease",
}

NUMERIC_FEATURES = [
    "age", "total_bilirubin", "direct_bilirubin",
    "alkaline_phosphotase", "sgpt", "sgot",
    "total_proteins", "albumin", "ag_ratio",
]


def load_and_clean(csv_path: str | Path) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Load ILPD CSV, clean, encode, and scale.

    Returns
    -------
    df     : cleaned DataFrame (unscaled – scaling columns appended with suffix _scaled)
    scaler : fitted MinMaxScaler (for inverse_transform in app)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # ── Rename columns ──────────────────────────────────────────────────────
    df = df.rename(columns=COL_MAP)

    # ── Encode gender ───────────────────────────────────────────────────────
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

    # ── Map target: 1=disease → 1, 2=healthy → 0 ────────────────────────────
    df["liver_disease"] = df["liver_disease"].map({1: 1, 2: 0})

    # ── Cast numerics ───────────────────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Impute missing with median ──────────────────────────────────────────
    missing_before = df.isnull().sum().sum()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(
        df[NUMERIC_FEATURES].median()
    )
    df.dropna(subset=["liver_disease"], inplace=True)
    logger.info("Imputed %d missing values", missing_before)

    # ── Clip extreme outliers (99th percentile cap) ─────────────────────────
    for col in NUMERIC_FEATURES:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)

    # ── Scale ───────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[NUMERIC_FEATURES])
    scaled_cols = [c + "_scaled" for c in NUMERIC_FEATURES]
    df[scaled_cols] = scaled

    df.reset_index(drop=True, inplace=True)
    logger.info("Cleaned dataset shape: %s", df.shape)
    return df, scaler


def save_cleaned(df: pd.DataFrame, out_path: str | Path) -> None:
    """Persist cleaned DataFrame to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved cleaned data → %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).parents[1]
    df, scaler = load_and_clean(root / "data" / "ILPD.csv")
    save_cleaned(df, root / "data" / "Cleaned_data.csv")
    print(df.head())
    print("\nTarget distribution:\n", df["liver_disease"].value_counts())
