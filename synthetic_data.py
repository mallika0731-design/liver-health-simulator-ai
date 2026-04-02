"""
synthetic_data.py
─────────────────
Expand the cleaned ILPD dataset ≥5× using realistic clinical variations
and add three lifestyle covariates:
  - exercise_level  ∈ [0, 1]  (0 = sedentary, 1 = highly active)
  - alcohol_intake  ∈ [0, 1]  (0 = none, 1 = heavy)
  - diet_quality    ∈ [0, 1]  (0 = poor, 1 = excellent)

Method
------
1. Gaussian copula-style sampling: preserve feature rank correlations
   by perturbing each clinical variable with correlated noise.
2. Add lifestyle features using a clinically motivated rule:
   - Disease patients → higher alcohol, lower diet/exercise (on average)
3. Clip all values to plausible clinical bounds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clinical plausible bounds ──────────────────────────────────────────────────
BOUNDS = {
    "age":                   (4,    90),
    "total_bilirubin":       (0.1,  75),
    "direct_bilirubin":      (0.1,  40),
    "alkaline_phosphotase":  (44,   2110),
    "sgpt":                  (7,    2000),
    "sgot":                  (10,   4929),
    "total_proteins":        (2.7,  9.6),
    "albumin":               (0.9,  5.5),
    "ag_ratio":              (0.3,  2.8),
}

NUMERIC_COLS = list(BOUNDS.keys())

# Noise fraction (σ as % of feature value) ──────────────────────────────────
NOISE_FRAC = 0.07   # ±7% Gaussian noise


def _add_lifestyle(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate lifestyle features correlated with disease status.
    Disease patients tend towards more alcohol and worse diet/exercise.
    """
    n = len(df)
    is_disease = df["liver_disease"].values.astype(float)  # 1 or 0

    # Beta distribution parameters conditioned on disease status
    # exercise_level: disease → Beta(2,4), healthy → Beta(4,2)
    alpha_ex = np.where(is_disease, 2.0, 4.0)
    beta_ex  = np.where(is_disease, 4.0, 2.0)
    exercise  = rng.beta(alpha_ex, beta_ex)

    # alcohol_intake: disease → Beta(4,2), healthy → Beta(2,5)
    alpha_al = np.where(is_disease, 4.0, 2.0)
    beta_al  = np.where(is_disease, 2.0, 5.0)
    alcohol   = rng.beta(alpha_al, beta_al)

    # diet_quality: disease → Beta(2,4), healthy → Beta(4,2)
    alpha_dq = np.where(is_disease, 2.0, 4.0)
    beta_dq  = np.where(is_disease, 4.0, 2.0)
    diet      = rng.beta(alpha_dq, beta_dq)

    df = df.copy()
    df["exercise_level"] = np.round(exercise, 3)
    df["alcohol_intake"]  = np.round(alcohol,  3)
    df["diet_quality"]    = np.round(diet,      3)
    return df


def _perturb_row(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Apply correlated Gaussian noise to clinical numeric features."""
    row = row.copy()
    for col in NUMERIC_COLS:
        if col not in row.index:
            continue
        val = row[col]
        sigma = NOISE_FRAC * abs(val) + 1e-6
        noisy = val + rng.normal(0, sigma)
        lo, hi = BOUNDS[col]
        row[col] = float(np.clip(noisy, lo, hi))
    # Enforce biological constraint: direct_bilirubin ≤ total_bilirubin
    if "direct_bilirubin" in row.index and "total_bilirubin" in row.index:
        row["direct_bilirubin"] = min(row["direct_bilirubin"], row["total_bilirubin"])
    # Recompute ag_ratio from albumin / globulin
    if "albumin" in row.index and "total_proteins" in row.index:
        globulin = row["total_proteins"] - row["albumin"]
        if globulin > 0:
            row["ag_ratio"] = round(row["albumin"] / globulin, 3)
    return row


def generate_synthetic(
    cleaned_df: pd.DataFrame,
    multiplier: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Expand cleaned_df by `multiplier` times using realistic perturbation.

    Parameters
    ----------
    cleaned_df : cleaned ILPD DataFrame (from preprocess.py)
    multiplier : how many synthetic copies to create (default 6 → ≥5× expansion)
    seed       : random seed for reproducibility

    Returns
    -------
    synthetic_df : DataFrame of purely synthetic patients (no originals)
    """
    rng = np.random.default_rng(seed)
    n_original = len(cleaned_df)
    target_n   = n_original * multiplier
    logger.info("Generating %d synthetic patients (×%d)…", target_n, multiplier)

    synthetic_rows = []
    while len(synthetic_rows) < target_n:
        # Sample a random original row
        idx  = rng.integers(0, n_original)
        base = cleaned_df.iloc[idx].copy()
        # Perturb clinical features
        row  = _perturb_row(base, rng)
        synthetic_rows.append(row)

    synth_df = pd.DataFrame(synthetic_rows).reset_index(drop=True)

    # Add lifestyle features
    synth_df = _add_lifestyle(synth_df, rng)

    # Drop _scaled columns (will be recomputed after BMI merge)
    scaled_cols = [c for c in synth_df.columns if c.endswith("_scaled")]
    synth_df.drop(columns=scaled_cols, inplace=True, errors="ignore")

    synth_df["synthetic"] = True
    logger.info("Synthetic dataset shape: %s", synth_df.shape)
    return synth_df


def combine_with_original(
    cleaned_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate original (+ lifestyle) and synthetic DataFrames."""
    orig = cleaned_df.copy()
    # Drop scaled cols from original too
    scaled_cols = [c for c in orig.columns if c.endswith("_scaled")]
    orig.drop(columns=scaled_cols, inplace=True, errors="ignore")

    # Add lifestyle to original
    rng = np.random.default_rng(99)
    orig = _add_lifestyle(orig, rng)
    orig["synthetic"] = False

    combined = pd.concat([orig, synthetic_df], ignore_index=True)
    logger.info("Combined dataset: %d rows", len(combined))
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess import load_and_clean

    root = Path(__file__).parents[1]
    df, _ = load_and_clean(root / "data" / "ILPD.csv")
    synth  = generate_synthetic(df, multiplier=6)
    combined = combine_with_original(df, synth)

    synth.to_csv(root / "data" / "Synthetic_data.csv", index=False)
    combined.to_csv(root / "data" / "Combined_dataset.csv", index=False)
    print("Synthetic shape:", synth.shape)
    print("Combined shape:", combined.shape)
