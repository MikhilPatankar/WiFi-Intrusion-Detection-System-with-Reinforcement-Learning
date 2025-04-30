

# --- File: src/backend/app/services/prediction_service.py ---

import logging
import numpy as np
from stable_baselines3 import DQN # Or PPO, etc.
import tensorflow as tf
from tensorflow import keras
import joblib
import os
from typing import Tuple, Dict, Any

from ..core.config import settings
from ..models import NetworkStateFeatures

log = logging.getLogger(__name__)

# --- Global variables ---
ml_agent = None
feature_scaler = None
ae_model = None
ae_threshold = None

def load_dependencies():
    """Loads the RL model, feature scaler, AE model, and AE threshold."""
    global ml_agent, feature_scaler, ae_model, ae_threshold
    model_path = settings.MODEL_PATH
    scaler_path = settings.SCALER_PATH
    ae_model_path = settings.AE_MODEL_PATH
    ae_threshold_path = settings.AE_THRESHOLD_PATH
    models_loaded = True

    # --- Load RL Model ---
    if not os.path.exists(model_path): log.error(f"RL Model not found: {model_path}"); ml_agent = None; models_loaded = False
    else:
        try: ml_agent = DQN.load(model_path, device='cpu'); log.info(f"Loaded RL model: {model_path}")
        except Exception as e: log.error(f"Error loading RL model: {e}"); ml_agent = None; models_loaded = False

    # --- Load Scaler ---
    if not os.path.exists(scaler_path): log.error(f"Scaler not found: {scaler_path}"); feature_scaler = None; models_loaded = False
    else:
        try: feature_scaler = joblib.load(scaler_path); assert hasattr(feature_scaler, 'transform'); log.info(f"Loaded scaler: {scaler_path}")
        except Exception as e: log.error(f"Error loading scaler: {e}"); feature_scaler = None; models_loaded = False

    # --- Load AE Model ---
    if not os.path.exists(ae_model_path): log.error(f"AE Model not found: {ae_model_path}"); ae_model = None; models_loaded = False
    else:
        try:
            ae_model = keras.models.load_model(ae_model_path)
            # Optional: Warm-up prediction
            if feature_scaler and hasattr(feature_scaler, 'n_features_in_'):
                 dummy_input = np.zeros((1, feature_scaler.n_features_in_), dtype=np.float32)
                 ae_model.predict(dummy_input, verbose=0)
            log.info(f"Loaded AE model: {ae_model_path}")
        except Exception as e: log.error(f"Error loading AE model: {e}"); ae_model = None; models_loaded = False

    # --- Load AE Threshold ---
    if not os.path.exists(ae_threshold_path): log.error(f"AE Threshold not found: {ae_threshold_path}"); ae_threshold = None; models_loaded = False
    else:
        try: ae_threshold = joblib.load(ae_threshold_path); assert isinstance(ae_threshold, float); log.info(f"Loaded AE threshold: {ae_threshold:.6f}")
        except Exception as e: log.error(f"Error loading AE threshold: {e}"); ae_threshold = None; models_loaded = False

    if not models_loaded: log.warning("One or more models/scalers failed to load.")
    return models_loaded

async def run_hybrid_prediction(features: NetworkStateFeatures) -> Dict[str, Any]:
    """Performs scaling, runs RL and AE models, combines results."""
    if feature_scaler is None: raise RuntimeError("Feature scaler unavailable.")
    if ml_agent is None: raise RuntimeError("RL model unavailable.")
    if ae_model is None: raise RuntimeError("AE model unavailable.")
    if ae_threshold is None: raise RuntimeError("AE threshold unavailable.")

    results = { "rl_prediction": -1, "ae_anomaly_flag": None, "final_status": "Error", "reconstruction_error": None, "scaled_features": None }
    try:
        # 1. Preprocessing
        raw_vector = np.array(features.feature_vector).reshape(1, -1)
        if raw_vector.shape[1] != feature_scaler.n_features_in_: raise ValueError(f"Input features ({raw_vector.shape[1]}) != scaler features ({feature_scaler.n_features_in_})")
        scaled_vector = feature_scaler.transform(raw_vector).astype(np.float32)
        results["scaled_features"] = scaled_vector[0].tolist()

        # 2. Run RL Agent
        rl_action, _ = ml_agent.predict(scaled_vector, deterministic=True)
        rl_prediction = int(rl_action[0])
        results["rl_prediction"] = rl_prediction
        log.debug(f"RL Prediction: {rl_prediction}")

        # 3. Run AE & Check Threshold
        ae_reconstruction = ae_model.predict(scaled_vector, verbose=0)
        mse = np.mean(np.power(scaled_vector - ae_reconstruction, 2), axis=1)[0]
        ae_anomaly_flag = bool(mse > ae_threshold)
        results["ae_anomaly_flag"] = ae_anomaly_flag
        results["reconstruction_error"] = float(mse)
        log.debug(f"AE Error: {mse:.6f}, Threshold: {ae_threshold:.6f}, AE Flag: {ae_anomaly_flag}")

        # 4. Combine Results
        if ae_anomaly_flag: results["final_status"] = "Potential Novelty"
        else: results["final_status"] = "Known Attack" if rl_prediction == 1 else "Normal"

        log.info(f"Hybrid Prediction: RL={rl_prediction}, AE Flag={ae_anomaly_flag}, Final='{results['final_status']}'")
        return results
    except ValueError as e: log.error(f"Input/Shape error: {e}"); raise
    except Exception as e: log.error(f"Hybrid prediction error: {e}", exc_info=True); raise RuntimeError(f"Prediction failed: {e}")

