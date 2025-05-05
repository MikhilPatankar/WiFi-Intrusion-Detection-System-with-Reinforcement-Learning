import uvicorn
import logging
import numpy as np
from stable_baselines3 import DQN # Or PPO, etc.
import tensorflow as tf
from tensorflow import keras
import joblib
import os
from typing import Tuple, Dict, Any

from fastapi import FastAPI, Request, HTTPException # Import FastAPI, Request, and HTTPException
from fastapi.responses import JSONResponse # Import JSONResponse for custom responses

logging.basicConfig(level=logging.INFO,  # Set desired logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')

scaler_path = "../models/wids_scaler.joblib"
ae_model_path = "../models/anomaly_autoencoder.h5"
ae_threshold_path = "../models/ae_threshold.joblib"
models_loaded = True

app = FastAPI()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PORT = os.getenv("PORT", 8050)

try: feature_scaler = joblib.load(scaler_path); assert hasattr(feature_scaler, 'transform'); logging.info(f"Loaded scaler: {scaler_path}")
except Exception as e: logging.error(f"Error loading scaler: {e}"); feature_scaler = None; models_loaded = False

if not os.path.exists(ae_model_path): logging.error(f"AE Model not found: {ae_model_path}"); ae_model = None; models_loaded = False
else:
    try:
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError()
            # Or potentially: 'mse': tf.keras.metrics.MeanSquaredError()
            # Or even the functional version: 'mse': tf.keras.losses.mean_squared_error
        }
        ae_model = keras.models.load_model(ae_model_path, custom_objects=custom_objects)
        # Optional: Warm-up prediction
        if feature_scaler and hasattr(feature_scaler, 'n_features_in_'):
                dummy_input = np.zeros((1, feature_scaler.n_features_in_), dtype=np.float32)
                ae_model.predict(dummy_input, verbose=0)
        logging.info(f"Loaded AE model: {ae_model_path}")
    except Exception as e: logging.error(f"Error loading AE model: {e}"); ae_model = None; models_loaded = False

# --- Load AE Threshold ---
if not os.path.exists(ae_threshold_path): logging.error(f"AE Threshold not found: {ae_threshold_path}"); ae_threshold = None; models_loaded = False
else:
    try: ae_threshold = joblib.load(ae_threshold_path); assert isinstance(ae_threshold, float); logging.info(f"Loaded AE threshold: {ae_threshold:.6f}")
    except Exception as e: logging.error(f"Error loading AE threshold: {e}"); ae_threshold = None; models_loaded = False

if not models_loaded: logging.warning("One or more models/scalers failed to load.")

async def prediction(parameters):
    features = []
    for f in parameters:
        features.append(parameters[f])
    
    if feature_scaler is None: raise RuntimeError("Feature scaler unavailable.")
    if ae_model is None: raise RuntimeError("AE model unavailable.")
    if ae_threshold is None: raise RuntimeError("AE threshold unavailable.")

    results = { "rl_prediction": -1, "ae_anomaly_flag": None, "final_status": "Error", "reconstruction_error": None, "scaled_features": None }
    try:
        raw_vector = np.array(features.feature_vector).reshape(1, -1)
        if raw_vector.shape[1] != feature_scaler.n_features_in_: raise ValueError(f"Input features ({raw_vector.shape[1]}) != scaler features ({feature_scaler.n_features_in_})")
        scaled_vector = feature_scaler.transform(raw_vector).astype(np.float32)
        results["scaled_features"] = scaled_vector[0].tolist()

        ae_reconstruction = ae_model.predict(scaled_vector, verbose=0)
        mse = np.mean(np.power(scaled_vector - ae_reconstruction, 2), axis=1)[0]
        ae_anomaly_flag = bool(mse > ae_threshold)
        results["ae_anomaly_flag"] = ae_anomaly_flag
        results["reconstruction_error"] = float(mse)
        logging.debug(f"AE Error: {mse:.6f}, Threshold: {ae_threshold:.6f}, AE Flag: {ae_anomaly_flag}")

        if ae_anomaly_flag: results["final_status"] = "Potential Novelty"
        
        logging.info(f"AE Flag={ae_anomaly_flag} | Final={results['final_status']} | Reconstruction Error={results['reconstruction_error']}")
    
    except ValueError as e: logging.error(f"Input/Shape error: {e}"); raise
    except Exception as e: logging.error(f"Hybrid prediction error: {e}", exc_info=True); raise RuntimeError(f"Prediction failed: {e}")


@app.post("/events", status_code=200)
async def handle_event(request: Request):
    try:
        data: Any = await request.json()
        if data:
            logging.info("Received Packet Data!")
            prediction(parameters=data)
            return {'status': 'success', 'message': 'Event received'}
        else:
            logging.warning("Received empty JSON data in POST request to /events")
            raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Empty request body'})

    except ValueError:
        logging.error("Error parsing JSON data from /events POST request")
        raise HTTPException(status_code=400, detail={'status': 'error', 'message': 'Invalid JSON format'})
    except Exception as e:
        logging.error(f"Error handling /events POST: {e}")
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': 'Internal server error'}
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(PORT), debug=True)
    #uvicorn.run("ae_prediction_test:app", host='0.0.0.0', port=int(PORT), reload=True)
