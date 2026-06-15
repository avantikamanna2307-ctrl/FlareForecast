"""
FlareForecast API

Predicts 48-hour flare risk for POTS / dysautonomia patients using
lifestyle, environmental, and medication features.
"""

from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_DAYS = 730  # 2 years of synthetic daily data
FLARE_THRESHOLD = 7.0

FEATURE_COLS = [
    "sleep_hours",
    "water_intake",
    "sodium_mg",
    "standing_hours",
    "stress_level",
    "weather_pressure",
    "temperature",
    "meds_taken",
]

LABEL_COL = "flare_next_48h"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_patient_data(n_days: int = N_DAYS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic daily health records for a POTS/dysautonomia patient.

    The underlying risk score is driven by lifestyle/environmental factors.
    Flare labels are derived from that risk score, and symptom_score is a noisy
    reflection of the same risk (so it represents "how you feel today" without
    being used to predict the future).
    """
    rng = np.random.RandomState(seed)
    days = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")

    # Daily lifestyle / weather features
    sleep_hours = np.clip(rng.normal(7.0, 1.5, n_days), 3, 12).astype(float)
    water_intake = np.clip(rng.normal(2.2, 0.7, n_days), 0.5, 5.0).astype(float)
    sodium_mg = np.clip(rng.normal(3000, 800, n_days), 1000, 6000).astype(float)
    standing_hours = np.clip(rng.normal(4.0, 2.0, n_days), 0, 12).astype(float)
    stress_level = np.clip(rng.randint(1, 11, n_days) + rng.normal(0, 1.0, n_days), 1, 10).astype(float)
    weather_pressure = np.clip(rng.normal(1013, 12, n_days), 980, 1040).astype(float)
    temperature = np.clip(rng.normal(22, 6, n_days), 5, 40).astype(float)
    meds_taken = rng.binomial(1, 0.82, n_days).astype(int)

    # Underlying risk: higher = more likely to flare
    # Lifestyle factors that are protective have negative weights; risky ones positive.
    risk_score = (
        8.0
        - 0.8 * sleep_hours
        + 1.0 * stress_level
        + 0.003 * (1013 - weather_pressure) ** 2
        + 0.3 * standing_hours
        - 0.12 * water_intake * 10
        - 0.0002 * sodium_mg
        - 1.0 * meds_taken
        + rng.normal(0, 1.2, n_days)
    )
    risk_score = np.clip(risk_score, 0, 10)

    # Symptom score is a noisy reflection of today's risk (not used for prediction)
    symptom_score = np.clip(risk_score + rng.normal(0, 1.0, n_days), 0, 10).astype(float)

    # Flare: risk is high enough that symptoms are likely to worsen in the next 48h
    flare = (risk_score > FLARE_THRESHOLD).astype(int)

    # Add realistic label noise (~5% flipped labels)
    noise_mask = rng.random(n_days) < 0.05
    flare[noise_mask] = 1 - flare[noise_mask]

    return pd.DataFrame({
        "date": days,
        "sleep_hours": sleep_hours,
        "water_intake": water_intake,
        "sodium_mg": sodium_mg,
        "standing_hours": standing_hours,
        "stress_level": stress_level,
        "weather_pressure": weather_pressure,
        "temperature": temperature,
        "meds_taken": meds_taken,
        "symptom_score": symptom_score,
        LABEL_COL: flare,
    })


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame) -> tuple:
    """
    Train a GradientBoosting classifier on lifestyle + weather features.
    Returns (model, scaler, train_score, test_score, feature_importance).
    """
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED, solver="lbfgs")
    model.fit(X_train_scaled, y_train)

    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    feature_importance = dict(zip(FEATURE_COLS, np.abs(model.coef_[0]).tolist()))

    return model, scaler, train_score, test_score, feature_importance


# Train once at import time (re-trained on server restart)
_df = generate_patient_data()
_model, _scaler, _train_score, _test_score, _feature_importance = train_model(_df)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep")
    water_intake: float = Field(..., ge=0, le=10, description="Liters of water consumed")
    sodium_mg: float = Field(..., ge=0, le=10000, description="Milligrams of sodium consumed")
    standing_hours: float = Field(..., ge=0, le=24, description="Hours spent standing or active")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level 1-10")
    weather_pressure: float = Field(..., ge=900, le=1100, description="Barometric pressure in hPa")
    temperature: float = Field(..., ge=-20, le=50, description="Temperature in Celsius")
    meds_taken: int = Field(..., ge=0, le=1, description="Whether meds/electrolytes were taken")
    symptom_score: float = Field(..., ge=0, le=10, description="Current symptom score 0-10")


class PredictResponse(BaseModel):
    flare_probability: float
    risk_level: str
    top_factors: List[Dict]
    model_accuracy: float
    symptom_score: float  # echoed back for display


# ---------------------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------------------

def explain_prediction(input_df: pd.DataFrame) -> List[Dict]:
    """
    Return the top 3 lifestyle factors that contribute to risk for this prediction.

    We convert each feature value to a z-score (standard deviations from its
    dataset mean) and weight by the model's feature importance. This makes the
    impact comparable across features with different scales (e.g., mg vs hours).
    """
    means = _df[FEATURE_COLS].mean()
    stds = _df[FEATURE_COLS].std()
    importances = pd.Series(_feature_importance)

    contributions = []
    row = input_df.iloc[0].to_dict()
    for col in FEATURE_COLS:
        val = float(row[col])
        z_score = (val - means[col]) / stds[col] if stds[col] > 0 else 0

        # Direction: for protective factors, below-average is risky;
        # for risky factors, above-average is risky.
        if col in {"sleep_hours", "water_intake", "sodium_mg", "meds_taken"}:
            impact = -z_score * importances[col]  # below avg -> positive impact on risk
        else:
            impact = z_score * importances[col]  # above avg -> positive impact on risk

        if impact > 0:
            contributions.append({
                "factor": col,
                "direction": "above" if z_score > 0 else "below",
                "delta": round(z_score, 2),  # now in standard-deviation units
                "impact": round(impact, 4),
            })

    contributions.sort(key=lambda x: x["impact"], reverse=True)
    return contributions[:3]


def risk_level_from_prob(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Moderate"
    return "High"


def predict_probability(input_df: pd.DataFrame) -> float:
    """Return flare probability for a feature DataFrame."""
    scaled = app.state.scaler.transform(input_df)
    return float(app.state.model.predict_proba(scaled)[0][1])


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.df = _df
    app.state.model = _model
    app.state.scaler = _scaler
    app.state.train_score = _train_score
    app.state.test_score = _test_score
    app.state.feature_importance = _feature_importance
    yield
    # Shutdown


app = FastAPI(
    title="FlareForecast API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "train_accuracy": round(app.state.train_score, 3),
        "test_accuracy": round(app.state.test_score, 3),
    }


@app.get("/sample-data")
def sample_data(n_days: int = 30):
    """Return recent sample data for the dashboard timeline."""
    recent = app.state.df.tail(n_days).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    return recent.to_dict(orient="records")


@app.get("/feature-importance")
def get_feature_importance():
    return app.state.feature_importance


@app.post("/whatif")
def whatif(req: PredictRequest):
    """
    For each feature, show how the risk would change if that feature were at its
    dataset mean. This powers the 'what-if' simulator in the frontend.
    """
    baseline_dict = {
        "sleep_hours": req.sleep_hours,
        "water_intake": req.water_intake,
        "sodium_mg": req.sodium_mg,
        "standing_hours": req.standing_hours,
        "stress_level": req.stress_level,
        "weather_pressure": req.weather_pressure,
        "temperature": req.temperature,
        "meds_taken": req.meds_taken,
    }
    baseline = pd.DataFrame([baseline_dict])
    baseline_prob = predict_probability(baseline)

    means = app.state.df[FEATURE_COLS].mean()
    results = []

    for col in FEATURE_COLS:
        modified_dict = baseline_dict.copy()
        modified_dict[col] = means[col]
        modified = pd.DataFrame([modified_dict])
        fixed_prob = predict_probability(modified)
        results.append({
            "factor": col,
            "current_value": round(baseline_dict[col], 2),
            "fixed_value": round(means[col], 2),
            "baseline_probability": round(max(0.001, min(0.999, baseline_prob)), 3),
            "fixed_probability": round(max(0.001, min(0.999, fixed_prob)), 3),
            "delta": round(baseline_prob - fixed_prob, 3),
        })

    # Only return factors where moving to mean actually changes risk
    results.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return results


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # The model predicts from lifestyle + weather only; symptom_score is echoed back.
    features = pd.DataFrame([{
        "sleep_hours": req.sleep_hours,
        "water_intake": req.water_intake,
        "sodium_mg": req.sodium_mg,
        "standing_hours": req.standing_hours,
        "stress_level": req.stress_level,
        "weather_pressure": req.weather_pressure,
        "temperature": req.temperature,
        "meds_taken": req.meds_taken,
    }])

    features_scaled = app.state.scaler.transform(features)
    prob = float(app.state.model.predict_proba(features_scaled)[0][1])

    # Floor/ceiling slightly so the UI never shows exactly 0% or 100%
    prob_display = max(0.001, min(0.999, prob))

    top_factors = explain_prediction(features)

    return PredictResponse(
        flare_probability=round(prob_display, 3),
        risk_level=risk_level_from_prob(prob),
        top_factors=top_factors,
        model_accuracy=round(app.state.test_score, 3),
        symptom_score=req.symptom_score,
    )


@app.get("/")
def root():
    return {"message": "FlareForecast API is running", "docs": "/docs"}
