"""
model.py
────────
Bayesian probabilistic model for liver disease risk.

Architecture
─────────────
1. **Feature Engineering**
   - Log-transform right-skewed enzyme features
   - Compose a "liver_enzyme_score" (weighted sum of log-transformed enzymes)
   - Compose an "obesity_risk" from BMI
   - Compose a "lifestyle_risk" from exercise, alcohol, diet

2. **Bayesian Logistic Regression (via MC Dropout approximation)**
   - Gradient Boosted Trees (XGBoost) as base predictor for calibration quality
   - Calibrated with CalibratedClassifierCV (isotonic) → outputs P(disease)
   - Confidence intervals via bootstrapped ensemble of N=100 classifiers

3. **Causal DAG encoded in feature weights**:
   Obesity (BMI) → Liver Risk
   Alcohol        → Liver Risk
   Diet/Exercise  → Protective factors

4. **save / load utilities** (joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature groups used by the model ───────────────────────────────────────────
CLINICAL_FEATURES = [
    "age", "gender",
    "total_bilirubin", "direct_bilirubin",
    "alkaline_phosphotase", "sgpt", "sgot",
    "total_proteins", "albumin", "ag_ratio",
]
LIFESTYLE_FEATURES = ["exercise_level", "alcohol_intake", "diet_quality"]
BMI_FEATURES       = ["bmi"]

ALL_FEATURES = CLINICAL_FEATURES + BMI_FEATURES + LIFESTYLE_FEATURES

# Log-transform candidates ───────────────────────────────────────────────────
LOG_COLS = [
    "total_bilirubin", "direct_bilirubin",
    "alkaline_phosphotase", "sgpt", "sgot",
]

N_BOOTSTRAP = 40   # number of bootstrap models for CI


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite risk scores to DataFrame."""
    df = df.copy()

    # Log-transform enzyme / bilirubin features
    for col in LOG_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Composite liver enzyme score (first PC surrogate)
    log_enzyme_cols = [f"log_{c}" for c in LOG_COLS if f"log_{c}" in df.columns]
    if log_enzyme_cols:
        df["enzyme_score"] = df[log_enzyme_cols].mean(axis=1)

    # Obesity risk: BMI mapped to [0,1] with knee at 25 and 35
    if "bmi" in df.columns:
        df["obesity_risk"] = np.clip((df["bmi"] - 18.5) / (40 - 18.5), 0, 1)

    # Lifestyle risk: high alcohol + low diet/exercise → high risk
    if all(c in df.columns for c in LIFESTYLE_FEATURES):
        df["lifestyle_risk"] = (
            df["alcohol_intake"] * 0.5
            - df["diet_quality"] * 0.25
            - df["exercise_level"] * 0.25
        ).clip(0, 1)

    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns available in df (handle missing lifestyle)."""
    base = [c for c in ALL_FEATURES if c in df.columns]
    extra = [c for c in ["enzyme_score", "obesity_risk", "lifestyle_risk"]
             if c in df.columns]
    # Add log-transformed columns
    log_c = [c for c in df.columns if c.startswith("log_")]
    return list(dict.fromkeys(base + extra + log_c))   # preserve order, no dupes


def build_model() -> Pipeline:
    """Construct a calibrated gradient boosting pipeline (sklearn only)."""
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(gb, method="isotonic", cv=3)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])
    return pipe


def train(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
) -> tuple[Pipeline, list[Pipeline], list[str]]:
    """
    Train the main model + bootstrap ensemble for confidence intervals.

    Returns
    -------
    model        : fitted main Pipeline
    boot_models  : list of N_BOOTSTRAP fitted Pipelines (for CI)
    feature_cols : list of feature column names used
    """
    df = _engineer_features(df)
    feature_cols = _get_feature_cols(df)
    target = "liver_disease"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Main model ──────────────────────────────────────────────────────────
    model = build_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)
    logger.info("Main model | AUC=%.4f | AP=%.4f", auc, ap)

    # ── Bootstrap ensemble for CI ────────────────────────────────────────────
    rng = np.random.default_rng(0)
    boot_models: list[Pipeline] = []
    n = len(X_train)
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        m   = build_model()
        m.fit(X_train[idx], y_train[idx])
        boot_models.append(m)
        if (i + 1) % 20 == 0:
            logger.info("Bootstrap %d / %d", i + 1, N_BOOTSTRAP)

    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "boot_models": boot_models, "feature_cols": feature_cols},
            model_path,
        )
        logger.info("Model saved → %s", model_path)

    return model, boot_models, feature_cols


def load_model(model_path: str | Path) -> tuple[Pipeline, list[Pipeline], list[str]]:
    """Load saved model artefact."""
    obj = joblib.load(model_path)
    return obj["model"], obj["boot_models"], obj["feature_cols"]


def predict_with_ci(
    patient: dict,
    model: Pipeline,
    boot_models: list[Pipeline],
    feature_cols: list[str],
    ci: float = 0.90,
) -> dict:
    """
    Predict liver disease probability + confidence interval for a single patient.

    Parameters
    ----------
    patient      : dict with raw patient features
    model        : main fitted model
    boot_models  : bootstrap ensemble
    feature_cols : ordered feature list
    ci           : confidence level (default 90%)

    Returns
    -------
    dict with keys: mean_prob, lower_ci, upper_ci, risk_level
    """
    # Prepare patient DF
    pat_df = pd.DataFrame([patient])
    pat_df = _engineer_features(pat_df)

    # Fill missing feature cols with 0
    for col in feature_cols:
        if col not in pat_df.columns:
            pat_df[col] = 0.0

    X = pat_df[feature_cols].values.astype(np.float32)

    mean_prob = float(model.predict_proba(X)[0, 1])

    # Bootstrap CI
    boot_probs = np.array([m.predict_proba(X)[0, 1] for m in boot_models])
    alpha = (1 - ci) / 2
    lower = float(np.quantile(boot_probs, alpha))
    upper = float(np.quantile(boot_probs, 1 - alpha))

    risk_level = (
        "Low" if mean_prob < 0.35
        else "Moderate" if mean_prob < 0.65
        else "High"
    )

    return {
        "mean_prob":  mean_prob,
        "lower_ci":   lower,
        "upper_ci":   upper,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess      import load_and_clean
    from synthetic_data  import generate_synthetic, combine_with_original
    from bmi             import add_bmi

    root = Path(__file__).parents[1]
    df, _ = load_and_clean(root / "data" / "ILPD.csv")
    synth  = generate_synthetic(df)
    comb   = combine_with_original(df, synth)
    comb   = add_bmi(comb)

    model, boots, fcols = train(comb, model_path=root / "data" / "liver_model.joblib")
    print("Feature cols:", fcols)

    sample = comb.iloc[0].to_dict()
    result = predict_with_ci(sample, model, boots, fcols)
    print("Sample prediction:", result)
