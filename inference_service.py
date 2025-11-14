# inference_service.py
from datetime import datetime
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# ---- Load model & scaler ----
MODEL_PATH = "lstm_migraine_model.h5"
SCALER_PATH = "scaler.pkl"
FEATURES = ["workload_0_10", "stress_0_10", "hrv_rmssd_ms"]
SEQ_LEN = 24
TOP_K = 3

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Migraine LSTM API")

# ---------- Request/Response schemas ----------
class FeatureVector(BaseModel):
    workload_0_10: float
    stress_0_10: float
    hrv_rmssd_ms: float

class PredictRequest(BaseModel):
    timestamp: datetime 
    window: List[FeatureVector]  # length must be 24

class Factor(BaseModel):
    feature: str
    score: float

class PredictResponse(BaseModel):
    p_next_hour: float
    top_factors: List[Factor]
    timestamp: datetime


# ---------- Integrated Gradients ----------
@tf.function
def _ig_path_integral(fn, x, baseline, m_steps: int = 64):
    alphas = tf.linspace(0.0, 1.0, m_steps)
    grads_sum = tf.zeros_like(x, dtype=tf.float32)
    for alpha in alphas:
        x_alpha = baseline + alpha * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x_alpha)
            y = fn(x_alpha)
        grads = tape.gradient(y, x_alpha)
        grads_sum += grads
    return (x - baseline) * grads_sum / tf.cast(m_steps, tf.float32)

def predict_with_attributions(model, X_window, feature_names, top_k):
    x = tf.convert_to_tensor(X_window[np.newaxis, ...], dtype=tf.float32)
    baseline_np = np.zeros_like(X_window, dtype=np.float32)
    baseline = tf.convert_to_tensor(baseline_np[np.newaxis, ...], dtype=tf.float32)

    def fn(x_in):
        y = model(x_in, training=False)
        return tf.squeeze(y, axis=-1)

    attrs = _ig_path_integral(fn, x, baseline, m_steps=64)[0].numpy()
    last_step = np.abs(attrs[-1, :])
    mean_time = np.mean(np.abs(attrs), axis=0)
    feat_scores = 0.7 * last_step + 0.3 * mean_time

    if feat_scores.sum() > 0:
        feat_scores_norm = feat_scores / feat_scores.sum()
    else:
        feat_scores_norm = feat_scores

    order = np.argsort(-feat_scores_norm)
    factors = [
        {"feature": feature_names[i], "score": float(feat_scores_norm[i])}
        for i in order[:top_k]
    ]

    prob = float(model.predict(x, verbose=0).ravel()[0])
    return prob, factors


# ---------- API endpoint ----------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.window) != SEQ_LEN:
        raise ValueError(f"window length must be {SEQ_LEN}, got {len(req.window)}")

    # build numpy array shape (seq_len, num_features)
    arr = np.array([[fv.workload_0_10, fv.stress_0_10, fv.hrv_rmssd_ms] for fv in req.window])
    arr_scaled = scaler.transform(arr)

    prob, factors = predict_with_attributions(model, arr_scaled, FEATURES, TOP_K)
    return PredictResponse(
        probability=prob,
        top_factors=[Factor(**f) for f in factors],
        timestamp=req.timestamp
    )
