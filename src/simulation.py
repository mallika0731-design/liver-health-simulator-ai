"""
simulation.py
─────────────
Digital Twin Simulator — time-evolution of liver disease risk.

State Transition Model
──────────────────────
S_t  → current liver health state (dict of patient features)
S_{t+1} = f(S_t, intervention_t)

Biological rules encoded:
  • BMI decreases if exercise↑ and diet↑ (capped at -0.5 BMI/month)
  • Liver enzymes (SGPT, SGOT) gradually normalise if alcohol↓
  • Albumin improves with diet quality
  • Age increases monotonically
  • Bilirubin partially resolves with low alcohol + high diet

Each time step = 1 month.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Physiological update rates (per month) ───────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def evolve_state(
    state: dict,
    exercise: float,
    alcohol: float,
    diet: float,
    months: int = 12,
    model=None,
    boot_models=None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Simulate liver disease risk over `months` months given lifestyle interventions.

    Parameters
    ----------
    state        : current patient state (dict of feature values)
    exercise     : exercise level ∈ [0,1] for the simulation period
    alcohol      : alcohol intake ∈ [0,1] for the simulation period
    diet         : diet quality ∈ [0,1] for the simulation period
    months       : number of months to simulate
    model        : trained main model (optional – skips prediction if None)
    boot_models  : bootstrap ensemble for CI
    feature_cols : ordered feature list

    Returns
    -------
    DataFrame with columns: month, bmi, sgpt, sgot, albumin,
                             total_bilirubin, mean_prob, lower_ci, upper_ci
    """
    from model import predict_with_ci, _engineer_features  # lazy import

    s = deepcopy(state)
    s["exercise_level"] = exercise
    s["alcohol_intake"]  = alcohol
    s["diet_quality"]    = diet

    records = []

    for month in range(months + 1):
        rec = {"month": month}

        # ── Snapshot biomarkers ────────────────────────────────────────────
        rec["bmi"]             = round(s.get("bmi", 25.0), 2)
        rec["sgpt"]            = round(s.get("sgpt", 40.0), 1)
        rec["sgot"]            = round(s.get("sgot", 35.0), 1)
        rec["albumin"]         = round(s.get("albumin", 3.5), 2)
        rec["total_bilirubin"] = round(s.get("total_bilirubin", 1.0), 2)
        rec["ag_ratio"]        = round(s.get("ag_ratio", 1.2), 2)

        # ── Predict risk ───────────────────────────────────────────────────
        if model is not None and feature_cols is not None:
            try:
                result = predict_with_ci(s, model, boot_models, feature_cols)
                rec["mean_prob"] = result["mean_prob"]
                rec["lower_ci"]  = result["lower_ci"]
                rec["upper_ci"]  = result["upper_ci"]
                rec["risk_level"]= result["risk_level"]
            except Exception as e:
                logger.warning("Prediction failed at month %d: %s", month, e)
                rec["mean_prob"] = 0.5
                rec["lower_ci"]  = 0.4
                rec["upper_ci"]  = 0.6
                rec["risk_level"]= "Unknown"

        records.append(rec)

        if month == months:
            break

        # ── Physiological update for next month ────────────────────────────

        # BMI: weight loss driven by exercise & diet vs alcohol (promotes fat)
        bmi_delta = (
            - 0.15 * exercise          # exercise burns calories
            - 0.10 * diet              # better diet → lower BMI
            + 0.08 * alcohol           # alcohol calories / liver fat
        )
        s["bmi"] = _clamp(s.get("bmi", 25.0) + bmi_delta, 16.0, 55.0)

        # Recompute height/weight consistent BMI (keep height constant)
        h = s.get("height_m", 1.65)
        s["weight_kg"] = round(s["bmi"] * h ** 2, 1)

        # Liver enzymes: normalise if alcohol low; worsen if alcohol high
        enzyme_trend = (
            - 0.04 * (1 - alcohol)     # recovery effect
            + 0.06 * alcohol           # damage effect
            - 0.02 * diet              # anti-inflammatory diet
        )
        s["sgpt"] = _clamp(
            s.get("sgpt", 40.0) * (1 + enzyme_trend),
            7, 2000
        )
        s["sgot"] = _clamp(
            s.get("sgot", 35.0) * (1 + enzyme_trend * 0.8),
            10, 4929
        )

        # Albumin: improves with good diet
        alb_delta = 0.02 * diet - 0.015 * alcohol
        s["albumin"] = _clamp(s.get("albumin", 3.5) + alb_delta, 0.9, 5.5)

        # Total bilirubin: partially resolves with no alcohol + good diet
        bili_trend = (
            - 0.03 * (1 - alcohol)
            - 0.02 * diet
            + 0.04 * alcohol
        )
        s["total_bilirubin"] = _clamp(
            s.get("total_bilirubin", 1.0) * (1 + bili_trend),
            0.1, 75
        )
        s["direct_bilirubin"] = _clamp(
            s["total_bilirubin"] * 0.4,
            0.1, 40
        )

        # AG ratio improves with albumin
        tp = s.get("total_proteins", 6.5)
        glob = tp - s["albumin"]
        if glob > 0:
            s["ag_ratio"] = _clamp(s["albumin"] / glob, 0.3, 2.8)

        # Alkaline phosphatase
        s["alkaline_phosphotase"] = _clamp(
            s.get("alkaline_phosphotase", 200) * (1 + enzyme_trend * 0.5),
            44, 2110
        )

        # Age increments by 1/12 year per month (cosmetic but realistic)
        s["age"] = s.get("age", 40) + 1 / 12

    df = pd.DataFrame(records)
    return df


def scenario_comparison(
    base_state: dict,
    scenarios: dict[str, dict],
    months: int = 24,
    model=None,
    boot_models=None,
    feature_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run multiple lifestyle scenarios and return a dict of DataFrames.

    Parameters
    ----------
    base_state  : current patient state
    scenarios   : { label: {exercise, alcohol, diet} }
    months      : simulation horizon

    Returns
    -------
    { label: evolution_df }
    """
    results = {}
    for label, params in scenarios.items():
        logger.info("Running scenario: %s", label)
        df = evolve_state(
            base_state,
            exercise=params["exercise"],
            alcohol=params["alcohol"],
            diet=params["diet"],
            months=months,
            model=model,
            boot_models=boot_models,
            feature_cols=feature_cols,
        )
        df["scenario"] = label
        results[label] = df
    return results


DEFAULT_SCENARIOS = {
    "Current Lifestyle": {"exercise": 0.2, "alcohol": 0.6, "diet": 0.3},
    "Moderate Improvement": {"exercise": 0.5, "alcohol": 0.3, "diet": 0.6},
    "Optimal Lifestyle": {"exercise": 0.9, "alcohol": 0.05, "diet": 0.9},
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = {
        "age": 45, "gender": 1, "bmi": 30.0, "height_m": 1.70, "weight_kg": 86,
        "total_bilirubin": 2.5, "direct_bilirubin": 1.0,
        "alkaline_phosphotase": 300, "sgpt": 80, "sgot": 70,
        "total_proteins": 6.5, "albumin": 3.0, "ag_ratio": 0.85,
        "exercise_level": 0.2, "alcohol_intake": 0.6, "diet_quality": 0.3,
    }
    results = scenario_comparison(base, DEFAULT_SCENARIOS, months=12)
    for name, df in results.items():
        print(f"\n{name}:\n", df[["month", "bmi", "sgpt", "mean_prob"]].to_string(index=False))
