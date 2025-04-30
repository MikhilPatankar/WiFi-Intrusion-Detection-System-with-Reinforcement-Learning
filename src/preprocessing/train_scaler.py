# src/preprocessing/train_scaler.py
# Loads processed feature data, fits a scaler, and saves it.

import pandas as pd
import joblib
import argparse
import logging
import os
from sklearn.preprocessing import MinMaxScaler # Or StandardScaler, etc.
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_PROCESSED_DATA_PATH = "../../data/processed/extracted_features.csv"
DEFAULT_SCALER_SAVE_PATH = "../../models/wids_scaler.joblib"

def train_and_save_scaler(data_path: Path, save_path: Path):
    """Loads data, fits a MinMaxScaler, and saves it."""
    log.info(f"Loading data for scaler training from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        log.info(f"Loaded data shape: {df.shape}")
    except FileNotFoundError:
        log.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        log.error(f"Error loading data: {e}", exc_info=True)
        raise

    # Identify feature columns (exclude non-numeric or identifier/label columns)
    # Adapt this based on your actual CSV columns
    exclude_cols = ['label', 'timestamp', 'event_uid', 'id'] # Add any other non-feature cols
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    if not feature_cols:
        log.error("No numeric feature columns found in the data. Cannot train scaler.")
        raise ValueError("No numeric feature columns identified.")

    log.info(f"Using {len(feature_cols)} columns for scaling: {feature_cols}")
    features_data = df[feature_cols].values

    # Handle potential NaNs before scaling (important!)
    # Option 1: Impute (e.g., with median) - consistent with potential feature extractor handling
    # from sklearn.impute import SimpleImputer
    # imputer = SimpleImputer(strategy='median')
    # features_data = imputer.fit_transform(features_data)
    # log.info("Applied median imputation to handle NaNs before scaling.")
    # Option 2: Error if NaNs exist (forces handling upstream)
    if pd.DataFrame(features_data).isnull().values.any():
         log.error("NaN values found in feature data. Please handle NaNs before training the scaler.")
         raise ValueError("NaN values detected in features.")


    # Initialize and fit the scaler
    # MinMaxScaler scales features to [0, 1], often suitable for neural nets
    scaler = MinMaxScaler()
    log.info("Fitting MinMaxScaler...")
    try:
        scaler.fit(features_data)
        log.info("Scaler fitting complete.")
    except Exception as e:
        log.error(f"Error fitting scaler: {e}", exc_info=True)
        raise

    # Save the scaler
    log.info(f"Saving scaler to: {save_path}")
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        log.info("Scaler saved successfully.")
    except Exception as e:
        log.error(f"Error saving scaler: {e}", exc_info=True)
        raise

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a feature scaler.")
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH,
                        help="Path to the processed data CSV file containing features.")
    parser.add_argument("--save-path", default=DEFAULT_SCALER_SAVE_PATH,
                        help="Path to save the fitted scaler (.joblib).")
    args = parser.parse_args()

    data_file = Path(args.data)
    save_file = Path(args.save_path)

    try:
        train_and_save_scaler(data_file, save_file)
    except Exception as e:
        log.error("Scaler training failed.")
        # Error details are logged within the function
