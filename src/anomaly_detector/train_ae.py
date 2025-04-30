# src/anomaly_detector/train_ae.py
# Trains an Autoencoder for unsupervised anomaly detection on normal data.

import os
import logging
import argparse
import pandas as pd
import numpy as np
import joblib # To load the scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # Assuming this was used
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
import sys

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_PROCESSED_DATA_PATH = "../../data/processed/extracted_features.csv"
DEFAULT_SCALER_PATH = "../../models/wids_scaler.joblib"
DEFAULT_AE_MODEL_SAVE_PATH = "../../models/anomaly_autoencoder.h5"
DEFAULT_THRESHOLD_SAVE_PATH = "../../models/ae_threshold.joblib"
VAL_SPLIT = 0.2; EPOCHS = 50; BATCH_SIZE = 64; ENCODING_DIM = 32; LEARNING_RATE = 1e-3

# --- Helper Functions ---
def load_and_prepare_data(data_path: Path, scaler_path: Path) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, int]:
    """Loads data, selects normal samples, loads scaler, and scales data."""
    log.info(f"Loading data from {data_path}")
    try: df = pd.read_csv(data_path); log.info(f"Loaded data shape: {df.shape}")
    except FileNotFoundError: log.error(f"Data file not found: {data_path}"); raise
    except Exception as e: log.error(f"Error loading data: {e}"); raise

    if 'label' not in df.columns: raise ValueError("Data CSV must contain a 'label' column (0 for Normal).")
    normal_df = df[df['label'] == 0].copy()
    log.info(f"Filtered {len(normal_df)} normal samples for AE training.")
    if len(normal_df) == 0: raise ValueError("No normal data found (label=0).")

    exclude_cols = ['label', 'timestamp', 'event_uid', 'id']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols: raise ValueError("Could not identify feature columns.")
    normal_features = normal_df[feature_cols].values
    num_features = normal_features.shape[1]
    log.info(f"Identified {num_features} feature columns.")

    log.info(f"Loading scaler from: {scaler_path}")
    try:
        scaler = joblib.load(scaler_path); assert hasattr(scaler, 'transform');
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != num_features: raise ValueError(f"Scaler expected {scaler.n_features_in_}, data has {num_features}.")
        log.info("Scaler loaded successfully.")
    except FileNotFoundError: log.error(f"Scaler file not found: {scaler_path}"); raise
    except Exception as e: log.error(f"Error loading scaler: {e}"); raise

    log.info("Scaling normal data...")
    try: scaled_normal_data = scaler.transform(normal_features); log.info("Data scaling complete.")
    except Exception as e: log.error(f"Error scaling data: {e}"); raise

    X_train, X_val = train_test_split(scaled_normal_data, test_size=VAL_SPLIT, random_state=42)
    log.info(f"Split normal data: Train={X_train.shape[0]}, Validation={X_val.shape[0]}")
    return X_train, X_val, scaler, num_features

def build_autoencoder(input_dim: int, encoding_dim: int, scaler_type: str) -> keras.Model:
    """Builds a simple MLP Autoencoder model."""
    log.info(f"Building Autoencoder: Input Dim={input_dim}, Encoding Dim={encoding_dim}")
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dropout(0.1)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dropout(0.1)(decoded)
    # Choose output activation based on scaler
    output_activation = 'sigmoid' if scaler_type == 'MinMaxScaler' else 'linear'
    log.info(f"Using output activation: {output_activation} based on scaler type: {scaler_type}")
    decoded = layers.Dense(input_dim, activation=output_activation)(decoded)
    autoencoder = keras.Model(input_layer, decoded, name="Autoencoder")
    autoencoder.summary(print_fn=log.info)
    return autoencoder

def calculate_threshold(model: keras.Model, x_val: np.ndarray) -> float:
    """Calculates reconstruction error threshold on validation data."""
    log.info("Calculating reconstruction error threshold...")
    reconstructions = model.predict(x_val)
    mse = np.mean(np.power(x_val - reconstructions, 2), axis=1)
    mean_err = np.mean(mse); std_err = np.std(mse)
    threshold = mean_err + 3 * std_err # Mean + 3*StdDev
    log.info(f"Reconstruction Error Stats (Validation): Mean={mean_err:.6f}, StdDev={std_err:.6f}")
    log.info(f"Calculated Threshold (Mean + 3*StdDev): {threshold:.6f}")
    return float(threshold)

# --- Main Training Function ---
def train_ae(data_path: Path, scaler_path: Path, model_save_path: Path, threshold_save_path: Path, encoding_dim: int, epochs: int, batch_size: int, lr: float):
    """Loads data, builds, trains, and saves the Autoencoder and threshold."""
    try:
        X_train, X_val, scaler, num_features = load_and_prepare_data(data_path, scaler_path)
        scaler_type = type(scaler).__name__ # Get scaler type name for activation choice
        autoencoder = build_autoencoder(num_features, encoding_dim, scaler_type)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        autoencoder.compile(optimizer=optimizer, loss='mse')
        log.info(f"Starting Autoencoder training for {epochs} epochs...")
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_val, X_val), callbacks=[early_stopping, reduce_lr], verbose=1)
        log.info("Autoencoder training complete.")
        threshold = calculate_threshold(autoencoder, X_val)
        log.info(f"Saving AE model to: {model_save_path}"); model_save_path.parent.mkdir(parents=True, exist_ok=True); autoencoder.save(model_save_path)
        log.info(f"Saving threshold to: {threshold_save_path}"); threshold_save_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(threshold, threshold_save_path)
        log.info("--- Autoencoder Training Finished Successfully ---")
    except Exception as e: log.error(f"AE training error: {e}", exc_info=True); sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AE for WIDS anomaly detection.")
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, help="Path to processed CSV (must include 'label').")
    parser.add_argument("--scaler", default=DEFAULT_SCALER_PATH, help="Path to saved scaler (.joblib).")
    parser.add_argument("--save-model", default=DEFAULT_AE_MODEL_SAVE_PATH, help="Path to save AE model (.h5).")
    parser.add_argument("--save-threshold", default=DEFAULT_THRESHOLD_SAVE_PATH, help="Path to save threshold (.joblib).")
    parser.add_argument("--encoding-dim", type=int, default=ENCODING_DIM, help="AE bottleneck layer dimension.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    args = parser.parse_args()
    train_ae(Path(args.data), Path(args.scaler), Path(args.save_model), Path(args.save_threshold), args.encoding_dim, args.epochs, args.batch_size, args.lr)

