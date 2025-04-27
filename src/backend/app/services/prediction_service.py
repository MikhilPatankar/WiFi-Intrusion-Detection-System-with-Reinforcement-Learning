
# --- File: src/backend/app/services/prediction_service.py ---
# No changes needed here, assumes config provides correct paths

import logging
import numpy as np
from stable_baselines3 import DQN
from ..core.config import settings
from ..models import NetworkStateFeatures
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

ml_agent = None
feature_scaler = None

def load_dependencies():
    global ml_agent, feature_scaler
    model_path = settings.MODEL_PATH
    scaler_path = settings.SCALER_PATH
    # Load Model
    if not os.path.exists(model_path): logging.error(f"RL Model file not found at: {model_path}"); ml_agent = None
    else:
        try: ml_agent = DQN.load(model_path, device='cpu'); logging.info(f"Successfully loaded RL model from: {model_path}")
        except Exception as e: logging.error(f"Error loading RL model from {model_path}: {e}", exc_info=True); ml_agent = None
    # Load Scaler
    if not os.path.exists(scaler_path): logging.error(f"Feature Scaler file not found at: {scaler_path}"); feature_scaler = None
    else:
        try:
            feature_scaler = joblib.load(scaler_path); logging.info(f"Successfully loaded feature scaler from: {scaler_path}")
            if not hasattr(feature_scaler, 'transform'): logging.error(f"Loaded object from {scaler_path} is not a valid scaler."); feature_scaler = None
        except Exception as e: logging.error(f"Error loading feature scaler from {scaler_path}: {e}", exc_info=True); feature_scaler = None

async def predict_anomaly(features: NetworkStateFeatures) -> int:
    if ml_agent is None: raise RuntimeError("RL model is not available.")
    if feature_scaler is None: raise RuntimeError("Feature scaler is not available.")
    try:
        raw_vector = np.array(features.feature_vector).reshape(1, -1)
        if raw_vector.shape[1] != feature_scaler.n_features_in_: raise ValueError(f"Input length ({raw_vector.shape[1]}) != scaler expected ({feature_scaler.n_features_in_})")
        scaled_vector = feature_scaler.transform(raw_vector)
        observation = scaled_vector.astype(np.float32)
        action, _ = ml_agent.predict(observation, deterministic=True)
        prediction = int(action[0])
        logging.debug(f"Prediction performed. Scaled shape: {observation.shape}, Output action: {prediction}")
        return prediction
    except ValueError as e: logging.error(f"Preprocessing error: {e}", exc_info=True); raise RuntimeError(f"Preprocessing failed: {e}")
    except Exception as e: logging.error(f"Prediction/preprocessing error: {e}", exc_info=True); raise RuntimeError(f"Prediction failed: {e}")

