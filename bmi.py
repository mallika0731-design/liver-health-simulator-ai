"""
bmi.py
──────
Add realistic height, weight, and BMI to any patient DataFrame.

Since ILPD does not include anthropometric data, we synthesise them
from a population model conditioned on age, gender, and disease status.

BMI categories (WHO):
  < 18.5  → Underweight
  18.5–25 → Normal
  25–30   → Overweight
  30–35   → Obese I
  ≥ 35    → Obese II+
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Height & weight distribution parameters by gender (approximate Indian population)
HEIGHT_PARAMS = {
    1: (1.68, 0.065),   # Male:   μ=168 cm, σ=6.5 cm
    0: (1.55, 0.060),   # Female: μ=155 cm, σ=6.0 cm
}
WEIGHT_PARAMS = {
    1: (72,   14),      # Male:   μ=72 kg,  σ=14 kg
    0: (60,   12),      # Female: μ=60 kg,  σ=12 kg
}

BMI_CATEGORIES = [
    (18.5, "Underweight"),
    (25.0, "Normal"),
    (30.0, "Overweight"),
    (35.0, "Obese I"),
    (float("inf"), "Obese II+"),
]


def _bmi_category(bmi: float) -> str:
    for threshold, label in BMI_CATEGORIES:
        if bmi < threshold:
            return label
    return "Obese II+"


def add_bmi(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Generate height_m, weight_kg, bmi, and bmi_category for each patient.

    If a 'gender' column exists (1=Male, 0=Female), it is used to
    condition the distributions; otherwise Male parameters are used.

    Parameters
    ----------
    df   : patient DataFrame (may include 'gender' column)
    seed : random seed

    Returns
    -------
    df with additional columns: height_m, weight_kg, bmi, bmi_category
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    genders = df["gender"].values if "gender" in df.columns else np.ones(n, dtype=int)

    heights = np.zeros(n)
    weights = np.zeros(n)

    for g_code in (1, 0):
        mask = genders == g_code
        if mask.sum() == 0:
            continue
        h_mu, h_sigma = HEIGHT_PARAMS[g_code]
        w_mu, w_sigma = WEIGHT_PARAMS[g_code]

        # Disease patients (liver_disease==1) skew heavier (obesity link)
        if "liver_disease" in df.columns:
            disease_mask = mask & (df["liver_disease"].values == 1)
            healthy_mask = mask & (df["liver_disease"].values == 0)
        else:
            disease_mask = mask
            healthy_mask = np.zeros(n, dtype=bool)

        # Obese patients: mean weight +12 kg
        if disease_mask.sum() > 0:
            heights[disease_mask] = rng.normal(h_mu, h_sigma, disease_mask.sum())
            weights[disease_mask] = rng.normal(w_mu + 10, w_sigma, disease_mask.sum())

        if healthy_mask.sum() > 0:
            heights[healthy_mask] = rng.normal(h_mu, h_sigma, healthy_mask.sum())
            weights[healthy_mask] = rng.normal(w_mu, w_sigma, healthy_mask.sum())

    # Clip to plausible range
    heights = np.clip(heights, 1.40, 2.00)
    weights = np.clip(weights, 35.0, 130.0)

    bmi = weights / (heights ** 2)
    bmi_cat = [_bmi_category(b) for b in bmi]

    df = df.copy()
    df["height_m"]     = np.round(heights, 3)
    df["weight_kg"]    = np.round(weights, 1)
    df["bmi"]          = np.round(bmi, 2)
    df["bmi_category"] = bmi_cat

    logger.info(
        "BMI added. Distribution:\n%s",
        pd.Series(bmi_cat).value_counts().to_string()
    )
    return df


def compute_bmi_single(weight_kg: float, height_m: float) -> tuple[float, str]:
    """Compute BMI and category for a single patient (Streamlit use)."""
    if height_m <= 0:
        raise ValueError("Height must be > 0")
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2), _bmi_category(bmi)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess import load_and_clean
    from synthetic_data import generate_synthetic, combine_with_original

    root = Path(__file__).parents[1]
    df, _ = load_and_clean(root / "data" / "ILPD.csv")
    synth  = generate_synthetic(df)
    combined = combine_with_original(df, synth)
    combined = add_bmi(combined)

    print(combined[["bmi", "bmi_category"]].describe())
    print(combined["bmi_category"].value_counts())
