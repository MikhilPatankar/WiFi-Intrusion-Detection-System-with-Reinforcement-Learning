
# --- File: src/backend/app/services/prediction_service.py ---

import logging
import numpy as np
from stable_baselines3 import DQN # Or PPO, etc.
from ..core.config import settings
from ..models import NetworkStateFeatures
import os
import joblib # For loading the scaler
from sklearn.preprocessing import MinMaxScaler # Or the specific scaler used

# --- Global variables ---
ml_agent = None
feature_scaler = None # Variable to hold the loaded scaler

def load_dependencies():
    """Loads the RL model and the feature scaler."""
    global ml_agent, feature_scaler
    model_path = settings.MODEL_PATH
    scaler_path = settings.SCALER_PATH

    # Load Model
    if not os.path.exists(model_path):
        logging.error(f"RL Model file not found at: {model_path}")
        ml_agent = None
    else:
        try:
            ml_agent = DQN.load(model_path, device='cpu')
            logging.info(f"Successfully loaded RL model from: {model_path}")
        except Exception as e:
            logging.error(f"Error loading RL model from {model_path}: {e}", exc_info=True)
            ml_agent = None

    # Load Scaler
    if not os.path.exists(scaler_path):
        logging.error(f"Feature Scaler file not found at: {scaler_path}")
        feature_scaler = None
    else:
        try:
            feature_scaler = joblib.load(scaler_path)
            logging.info(f"Successfully loaded feature scaler from: {scaler_path}")
            # Basic check: Ensure it's a scaler object (adjust type check if needed)
            if not hasattr(feature_scaler, 'transform'):
                 logging.error(f"Loaded object from {scaler_path} does not appear to be a valid scaler.")
                 feature_scaler = None
        except Exception as e:
            logging.error(f"Error loading feature scaler from {scaler_path}: {e}", exc_info=True)
            feature_scaler = None

async def predict_anomaly(features: NetworkStateFeatures) -> int:
    """
    Performs preprocessing and prediction using the loaded RL agent and scaler.
    """
    if ml_agent is None:
        raise RuntimeError("RL model is not available for prediction.")
    if feature_scaler is None:
        raise RuntimeError("Feature scaler is not available for preprocessing.")

    try:
        # --- Preprocessing ---
        raw_vector = np.array(features.feature_vector).reshape(1, -1)

        # Apply the loaded scaler
        # Ensure the input vector shape matches what the scaler expects
        # This assumes the input feature_vector has the correct number of features
        # in the correct order as used during training/scaler fitting.
        if raw_vector.shape[1] != feature_scaler.n_features_in_:
             raise ValueError(f"Input feature vector length ({raw_vector.shape[1]}) does not match scaler expected features ({feature_scaler.n_features_in_})")

        scaled_vector = feature_scaler.transform(raw_vector)
        observation = scaled_vector.astype(np.float32) # Ensure correct dtype

        # --- Prediction ---
        action, _states = ml_agent.predict(observation, deterministic=True)
        prediction = int(action[0])

        logging.debug(f"Prediction performed. Scaled shape: {observation.shape}, Output action: {prediction}")
        return prediction

    except ValueError as e:
        logging.error(f"Preprocessing error (likely feature mismatch): {e}", exc_info=True)
        raise RuntimeError(f"Preprocessing failed: {e}")
    except Exception as e:
        logging.error(f"Error during prediction/preprocessing: {e}", exc_info=True)
        raise RuntimeError(f"Prediction failed: {e}")

