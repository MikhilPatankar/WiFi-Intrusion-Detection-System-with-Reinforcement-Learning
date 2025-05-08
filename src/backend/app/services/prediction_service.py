# --- File: src/backend/app/services/prediction_service.py ---

import logging
import numpy as np
from stable_baselines3 import DQN # Or PPO, etc.
from stable_baselines3.common.vec_env import DummyVecEnv # For action_space check
from gymnasium import spaces # For action_space type check
import tensorflow as tf
from tensorflow import keras
import joblib
import os
from typing import Tuple, Dict, Any, Optional
import torch
import requests # For fetching attack types
import time # For retries

from ..core.config import settings
from ..models import NetworkStateFeatures

log = logging.getLogger(__name__)

# --- Global variables ---
ml_agent = None
feature_scaler = None
ae_model = None
ae_threshold = None
# NEW: Map from RL action index to a dictionary containing label_name and type_id
action_details_map: Optional[Dict[int, Dict[str, Any]]] = None

def _fetch_and_build_action_details_map() -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Fetches active attack types from the API and builds an index-to-details map.
    Map: rl_action_index -> {'label_name': str, 'type_id': int}
    "Normal" is always index 0 with type_id 0. Other attack types are sorted alphabetically.
    """
    global action_details_map # To modify the global var
    if not settings.WIDS_API_BASE_URL:
        log.error("WIDS_API_BASE_URL is not configured. Cannot fetch attack types.")
        action_details_map = None
        return None

    api_endpoint = f"{settings.WIDS_API_BASE_URL}/attack_types/?include_inactive=false"
    log.info(f"Fetching current active attack types from: {api_endpoint} for prediction service.")
    
    current_map_build: Dict[int, Dict[str, Any]] = {
        0: {"label_name": "Normal", "type_id": 0} # "Normal" is always index 0, type_id 0
    }
    next_action_index = 1 # Start indexing for attacks from 1

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(api_endpoint, timeout=10)
            response.raise_for_status()
            # attack_types_data is List[Dict[str, Any]], e.g., [{'type_id': 1, 'type_name': 'DoS', ...}, ...]
            attack_types_data = response.json()

            # Filter out "Normal" if present in API response, then sort others by name
            # Ensure each item 't' is a dict and has 'type_name' and 'type_id'
            fetched_attack_objects = []
            for t in attack_types_data:
                if isinstance(t, dict) and t.get('type_name') and t['type_name'] != "Normal" and 'type_id' in t:
                    fetched_attack_objects.append(t)
                elif isinstance(t, dict) and t.get('type_name') == "Normal":
                    continue # Skip "Normal" from API if present, we handle it manually
                else:
                    log.warning(f"Skipping invalid attack type object from API: {t}")
            
            sorted_attack_objects = sorted(fetched_attack_objects, key=lambda x: x['type_name'].lower())

            for attack_obj in sorted_attack_objects:
                current_map_build[next_action_index] = {
                    "label_name": attack_obj['type_name'],
                    "type_id": attack_obj['type_id']
                }
                next_action_index += 1
            
            action_details_map = current_map_build # Assign to global
            log.info(f"Prediction service: Built action_details_map with {len(action_details_map)} classes: {action_details_map}")
            return action_details_map
        except requests.exceptions.RequestException as e:
            log.error(f"API request failed for attack types (attempt {attempt+1}/{retries}): {e}")
        except Exception as e: # Catch other errors like JSONDecodeError or unexpected structure
            log.error(f"Error processing attack types (attempt {attempt+1}/{retries}): {e}", exc_info=True)
        
        if attempt < retries - 1:
            log.info(f"Retrying in 5 seconds...")
            time.sleep(5)
            
    log.error("Failed to fetch attack types for prediction service after multiple retries. `action_details_map` will be None.")
    action_details_map = None # Ensure it's None on failure
    return None


def load_dependencies():
    """Loads the RL model, feature scaler, AE model, AE threshold, and attack details map."""
    global ml_agent, feature_scaler, ae_model, ae_threshold, action_details_map # Updated global name
    model_path = settings.MODEL_PATH
    scaler_path = settings.SCALER_PATH
    ae_model_path = settings.AE_MODEL_PATH
    ae_threshold_path = settings.AE_THRESHOLD_PATH
    models_loaded_successfully = True

    _fetch_and_build_action_details_map() # Updated function call
    if action_details_map is None:
        log.warning("`action_details_map` could not be initialized. RL predictions will use generic labels and default attack_ids.")
        # Depending on strictness, you might set models_loaded_successfully = False here

    if not os.path.exists(model_path):
        log.error(f"RL Model not found: {model_path}"); ml_agent = None; models_loaded_successfully = False
    else:
        try:
            ml_agent = DQN.load(model_path, device='cpu')
            log.info(f"Loaded RL model: {model_path}")

            if ml_agent and action_details_map: # Check with the new map
                if hasattr(ml_agent, 'action_space') and isinstance(ml_agent.action_space, spaces.Discrete):
                    if ml_agent.action_space.n != len(action_details_map):
                        log.error(
                            f"CRITICAL MISMATCH: RL model action space size ({ml_agent.action_space.n}) "
                            f"does not match fetched number of classes ({len(action_details_map)} from API: "
                            f"{ {k: v['label_name'] for k, v in action_details_map.items()} }). "
                            "Predictions may be misinterpreted."
                        )
                        models_loaded_successfully = False
                    else:
                        log.info(f"RL model action space size ({ml_agent.action_space.n}) matches "
                                 f"fetched number of classes ({len(action_details_map)}).")
                else:
                    log.warning("RL model action space not found or not Discrete. Cannot verify class count match.")
            elif ml_agent and not action_details_map:
                 log.warning("RL model loaded, but action_details_map is missing. Cannot verify class count match.")
        except Exception as e:
            log.error(f"Error loading RL model: {e}"); ml_agent = None; models_loaded_successfully = False

    if not os.path.exists(scaler_path):
        log.error(f"Scaler not found: {scaler_path}"); feature_scaler = None; models_loaded_successfully = False
    else:
        try:
            feature_scaler = joblib.load(scaler_path)
            assert hasattr(feature_scaler, 'transform'), "Scaler missing 'transform' method"
            assert hasattr(feature_scaler, 'n_features_in_'), "Scaler missing 'n_features_in_' attribute"
            log.info(f"Loaded scaler: {scaler_path}")
        except Exception as e:
            log.error(f"Error loading scaler: {e}"); feature_scaler = None; models_loaded_successfully = False
    
    if models_loaded_successfully and ml_agent and feature_scaler and \
       hasattr(ml_agent, 'observation_space') and ml_agent.observation_space is not None and \
       hasattr(feature_scaler, 'n_features_in_') and \
       ml_agent.observation_space.shape[0] == feature_scaler.n_features_in_:
        try:
            if not hasattr(ml_agent, '_warmed_up_successfully'):
                dummy_obs_rl = np.zeros(ml_agent.observation_space.shape, dtype=ml_agent.observation_space.dtype)
                ml_agent.predict(dummy_obs_rl, deterministic=True)
                ml_agent._warmed_up_successfully = True
                log.info("RL model warm-up successful.")
        except Exception as wu_e:
            log.warning(f"RL model warm-up failed: {wu_e}")
    elif models_loaded_successfully:
        log.warning("Skipping RL model warm-up due to missing components or previous errors.")


    if not os.path.exists(ae_model_path):
        log.error(f"AE Model not found: {ae_model_path}"); ae_model = None; models_loaded_successfully = False
    else:
        try:
            custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
            ae_model = keras.models.load_model(ae_model_path, custom_objects=custom_objects)
            if models_loaded_successfully and feature_scaler and hasattr(feature_scaler, 'n_features_in_'):
                 dummy_input_ae = np.zeros((1, feature_scaler.n_features_in_), dtype=np.float32)
                 ae_model.predict(dummy_input_ae, verbose=0)
                 log.info("AE model warm-up successful.")
            elif models_loaded_successfully:
                log.warning("Skipping AE model warm-up due to feature scaler unavailability or previous errors.")
        except Exception as e:
            log.error(f"Error loading AE model: {e}"); ae_model = None; models_loaded_successfully = False

    if not os.path.exists(ae_threshold_path):
        log.error(f"AE Threshold not found: {ae_threshold_path}"); ae_threshold = None; models_loaded_successfully = False
    else:
        try:
            ae_threshold = joblib.load(ae_threshold_path)
            assert isinstance(ae_threshold, float), "AE threshold is not a float"
            assert ae_threshold >= 0, "AE threshold cannot be negative for MSE"
            log.info(f"Loaded AE threshold: {ae_threshold:.6f}")
        except AssertionError as e_assert:
            log.error(f"Invalid AE threshold: {e_assert}"); ae_threshold = None; models_loaded_successfully = False
        except Exception as e:
            log.error(f"Error loading AE threshold: {e}"); ae_threshold = None; models_loaded_successfully = False

    if not models_loaded_successfully:
        log.warning("One or more models/scalers/thresholds/maps failed to load. Prediction service may be impaired.")
    
    return models_loaded_successfully


async def run_hybrid_prediction(features: NetworkStateFeatures) -> Dict[str, Any]:
    """Performs scaling, runs RL and AE models, combines results, and includes confidence scores."""
    if feature_scaler is None: raise RuntimeError("Feature scaler unavailable.")
    if ml_agent is None: raise RuntimeError("RL model unavailable.")
    if ae_model is None: raise RuntimeError("AE model unavailable.")
    if ae_threshold is None: raise RuntimeError("AE threshold unavailable.")

    results = {
        "rl_prediction": -1,            # RL action index
        "rl_predicted_label": "Error",  # String label for RL prediction
        "attack_id": -1,                # Corresponding type_id from DB, 0 for Normal, -1 for error/unmapped
        "rl_confidence": None,
        "ae_anomaly_flag": None,
        "ae_confidence": None,
        "final_status": "Error",
        "reconstruction_error": None,
        "scaled_features": None
    }
    try:
        raw_vector = np.array(features.feature_vector).reshape(1, -1)
        if raw_vector.shape[1] != feature_scaler.n_features_in_:
            raise ValueError(f"Input features ({raw_vector.shape[1]}) != scaler features ({feature_scaler.n_features_in_})")
        scaled_vector = feature_scaler.transform(raw_vector).astype(np.float32)
        results["scaled_features"] = scaled_vector[0].tolist()

        obs_tensor = torch.as_tensor(scaled_vector).to(ml_agent.device)
        with torch.no_grad():
            q_values = ml_agent.q_net(obs_tensor)
        
        rl_action_tensor = q_values.argmax(dim=1)
        rl_prediction_index = int(rl_action_tensor.item())
        results["rl_prediction"] = rl_prediction_index
        
        softmax_q = torch.softmax(q_values, dim=1)
        rl_confidence = float(softmax_q[0, rl_prediction_index].item())
        results["rl_confidence"] = rl_confidence

        # Determine predicted label and attack_id using action_details_map
        current_rl_predicted_label = "Error" # Default
        current_attack_id = -1 # Default

        if action_details_map and rl_prediction_index in action_details_map:
            details = action_details_map[rl_prediction_index]
            current_rl_predicted_label = details["label_name"]
            current_attack_id = details["type_id"]
        elif rl_prediction_index == 0: # Fallback if map is missing, but 0 is always Normal
            current_rl_predicted_label = "Normal"
            current_attack_id = 0
            if not action_details_map:
                 log.warning(f"action_details_map is missing. RL prediction index {rl_prediction_index} defaulted to 'Normal' (ID 0).")
            else: # Map exists, but index not found (should ideally not happen if index is 0)
                 log.warning(f"RL prediction index {rl_prediction_index} (expected Normal) not fully resolved by map. Defaulting to 'Normal' (ID 0).")
        else: # Fallback for other indices if map is missing or index is out of bounds
            current_rl_predicted_label = f"Unmapped Attack (Index {rl_prediction_index})"
            # current_attack_id remains -1
            log.warning(
                f"RL prediction index {rl_prediction_index} could not be mapped. "
                f"action_details_map available: {bool(action_details_map)}. "
                f"Label: '{current_rl_predicted_label}', Attack ID: {current_attack_id}."
            )
        
        results["rl_predicted_label"] = current_rl_predicted_label
        results["attack_id"] = current_attack_id
        log.debug(
            f"RL Prediction Index: {rl_prediction_index}, Label: '{current_rl_predicted_label}', "
            f"Attack ID: {current_attack_id}, Confidence: {rl_confidence:.4f}"
        )

        ae_reconstruction = ae_model.predict(scaled_vector, verbose=0)
        mse = np.mean(np.power(scaled_vector - ae_reconstruction, 2), axis=1)[0]
        ae_anomaly_flag = bool(mse > ae_threshold)
        
        results["ae_anomaly_flag"] = ae_anomaly_flag
        results["reconstruction_error"] = float(mse)

        ae_confidence_val = 0.0
        if ae_anomaly_flag:
            if mse > 0: ae_confidence_val = (mse - ae_threshold) / mse
            else: ae_confidence_val = 0.0 
        else:
            if ae_threshold > 0: ae_confidence_val = (ae_threshold - mse) / ae_threshold
            elif ae_threshold == 0: ae_confidence_val = 1.0 
        
        ae_confidence_val = max(0.0, min(1.0, float(ae_confidence_val)))
        results["ae_confidence"] = ae_confidence_val
        log.debug(f"AE Error: {mse:.6f}, Threshold: {ae_threshold:.6f}, AE Flag: {ae_anomaly_flag}, AE Confidence: {ae_confidence_val:.4f}")

        # 4. Combine Results (using current_rl_predicted_label)
        if ae_anomaly_flag:
            if current_rl_predicted_label == "Normal":
                results["final_status"] = "Potential Novelty / Unknown Anomaly"
            else: 
                results["final_status"] = f"Anomalous {current_rl_predicted_label}"
        else: 
            results["final_status"] = current_rl_predicted_label

        log.info(
            f"Hybrid Prediction: RL='{results['rl_predicted_label']}' (Idx:{results['rl_prediction']}, AttackID:{results['attack_id']}, Conf:{results['rl_confidence']:.2f}), "
            f"AE Flag={results['ae_anomaly_flag']} (Conf:{results['ae_confidence']:.2f}), "
            f"MSE={results['reconstruction_error']:.4f}, Final='{results['final_status']}'"
        )
        return results
    except ValueError as e:
        log.error(f"Input/Shape error during prediction: {e}")
        results["final_status"] = f"Error: Input/Shape - {str(e)}" 
        raise
    except RuntimeError as e:
        log.error(f"Runtime error during prediction: {e}", exc_info=True)
        results["final_status"] = f"Error: Runtime - {str(e)}"
        raise
    except Exception as e:
        log.error(f"Unexpected hybrid prediction error: {e}", exc_info=True)
        results["final_status"] = f"Error: Unexpected - {str(e)}"
        raise RuntimeError(f"Prediction failed due to an unexpected error: {e}")

