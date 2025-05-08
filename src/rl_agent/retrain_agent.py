# src/rl_agent/retrain_agent.py
# Script to fine-tune the RL agent using multi-class human-labeled data.
# NOTE: This version assumes the loaded model's architecture (output layer size)
# matches the *current* number of classes known by the backend API.
# It does NOT implement dynamic layer resizing or catastrophic forgetting mitigation (e.g., EWC).

import os
import logging
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import DQN # Or PPO
from stable_baselines3.common.env_util import make_vec_env
import joblib
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, sessionmaker
from sqlalchemy.future import select
import asyncio
from pathlib import Path
import sys
import requests # To fetch attack types
import time
from typing import List, Dict, Optional
import gymnasium as gym
from gymnasium import spaces

# --- Path Setup & Imports ---
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    backend_app_path = project_root / 'src' / 'backend'
    rl_agent_path = project_root / 'src' / 'rl_agent'
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
    if str(backend_app_path) not in sys.path: sys.path.insert(0, str(backend_app_path))
    if str(rl_agent_path) not in sys.path: sys.path.insert(0, str(rl_agent_path))

    from backend.app.core.config import settings as api_settings
    from backend.app.schemas import EventLog
    # Import the MULTI-CLASS environment
    from wids_env import WidsEnvMultiClass
except ImportError as e: print(f"Import Error: {e}"); sys.exit(1)
except FileNotFoundError: print("Error: Structure mismatch."); sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
MODEL_LOAD_PATH = Path(api_settings.MODEL_PATH)
MODEL_SAVE_PATH = Path(api_settings.MODEL_PATH) # Overwrite
SCALER_PATH = Path(api_settings.SCALER_PATH)
DATABASE_URL = api_settings.DATABASE_URL
WIDS_API_URL = api_settings.WIDS_API_BASE_URL # Get API URL from backend settings

# Fine-tuning parameters
FINETUNE_TIMESTEPS = 2000 # More steps might be needed for multi-class
FINETUNE_LEARNING_RATE = 3e-5 # Generally smaller LR for fine-tuning
# Multi-class reward config (should align with WidsEnvMultiClass defaults or be customized)
FINETUNE_REWARD_CONFIG = {
    'Normal': 5.0,             # Higher reward for confirming normal
    'Default_Attack': 20.0,    # Higher reward for confirming attacks
    'Default_FP': -10.0,       # Penalty for labeling Normal as Attack
    'Default_FN': -100.0,      # High penalty for labeling Attack as Normal
    'Default_Mismatch': -15.0  # Penalty for confusing attacks
}

engine_args = {"echo": False} # Set echo=True for SQL logging
if DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

# --- Database Setup ---
# Create the asynchronous SQLAlchemy engine
engine = create_async_engine(
    DATABASE_URL,
    **engine_args,
    pool_size=50,
    max_overflow=50,
    pool_timeout=30)

# Create an asynchronous session factory
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
# --- Helper Functions ---

def fetch_current_attack_types(api_url: str) -> Dict[str, int]:
    """Fetches active attack types and returns a name -> index mapping."""
    api_endpoint = f"{api_url}/attack_types/?include_inactive=false"
    log.info(f"Fetching current active attack types from: {api_endpoint}")
    label_to_action: Dict[str, int] = {}
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(api_endpoint, timeout=10)
            response.raise_for_status()
            attack_types_data = response.json()
            # Build mapping: "Normal" is always 0
            class_labels = ["Normal"]
            fetched_types = sorted([t['type_name'] for t in attack_types_data if t.get('type_name') and t['type_name'] != "Normal"], key=str.lower)
            class_labels.extend(fetched_types)
            label_to_action = {label: i for i, label in enumerate(class_labels)}
            log.info(f"Fetched {len(label_to_action)} classes for mapping: {list(label_to_action.keys())}")
            return label_to_action
        except requests.exceptions.RequestException as e: log.error(f"API req failed (attempt {attempt+1}/{retries}): {e}")
        except Exception as e: log.error(f"Error processing types (attempt {attempt+1}/{retries}): {e}")
        if attempt < retries - 1: time.sleep(5)
    log.error("Failed to fetch attack types after multiple retries.")
    return {} # Return empty dict on failure


async def fetch_labeled_data_for_finetuning(label_to_action_map: Dict[str, int]) -> pd.DataFrame:
    """Fetches labeled events and maps labels to current action indices."""
    log.info("Fetching labeled data for fine-tuning...")
    if not label_to_action_map:
        log.error("Cannot fetch data without a valid label-to-action map.")
        return pd.DataFrame()

    async with AsyncSessionFactory() as session:
        try:
            stmt = select(EventLog).where(EventLog.human_label != None).where(EventLog.human_label != 'Uncertain').order_by(EventLog.labeling_timestamp.desc()) # Exclude uncertain
            result = await session.execute(stmt)
            labeled_events = result.scalars().all()
            log.info(f"Fetched {len(labeled_events)} non-uncertain labeled events.")
        except Exception as e: log.error(f"DB error fetching labeled data: {e}"); return pd.DataFrame()
    if not labeled_events: return pd.DataFrame()

    data_list = []
    skipped_unknown = 0
    for event in labeled_events:
        if not isinstance(event.features_data, list): continue # Skip if features format is wrong

        human_label = event.human_label # Get the string label
        action_index = label_to_action_map.get(human_label) # Map to current index

        if action_index is not None:
            # Only include events where the human label corresponds to a currently known class
            data_list.append({
                'feature_vector': event.features_data, # Store UNscaled features
                'label': human_label, # Store original string label for env init
                'action_label': action_index # Store the mapped action index
            })
        else:
            # This label might be for a type that was deleted or is new and not yet processed by backend fully
            log.warning(f"Skipping event {event.event_uid}: Label '{human_label}' not found in current action map.")
            skipped_unknown += 1

    log.info(f"Prepared {len(data_list)} samples for fine-tuning. Skipped {skipped_unknown} events with unknown labels.")
    return pd.DataFrame(data_list)


async def run_finetuning(args):
    """Main async function for multi-class fine-tuning."""
    log.info("--- Starting RL Agent Multi-Class Fine-tuning ---")

    # 1. Fetch Current Class Mapping from API
    label_to_action_map = fetch_current_attack_types(args.api_url)
    if not label_to_action_map: return
    num_classes = len(label_to_action_map)

    # 2. Load Scaler
    log.info(f"Loading scaler: {args.scaler_path}")
    if not args.scaler_path.exists(): log.error("Scaler file not found."); return
    try: scaler = joblib.load(args.scaler_path); num_features = scaler.n_features_in_; log.info(f"Scaler loaded. Features: {num_features}")
    except Exception as e: log.error(f"Error loading scaler: {e}"); return

    # 3. Fetch Labeled Data (Unscaled features)
    labeled_df_unscaled = await fetch_labeled_data_for_finetuning(label_to_action_map)
    if labeled_df_unscaled.empty: log.info("No suitable labeled data found for fine-tuning."); return

    # 4. Scale Fetched Features
    try:
        if 'feature_vector' not in labeled_df_unscaled.columns: raise ValueError("'feature_vector' missing.")
        features_unscaled = np.vstack(labeled_df_unscaled['feature_vector'].values)
        if features_unscaled.shape[1] != num_features: raise ValueError(f"Feature length mismatch DB ({features_unscaled.shape[1]}) vs scaler ({num_features}).")
        if np.isnan(features_unscaled).any(): log.warning("NaNs in fetched features. Filling with 0."); features_unscaled = np.nan_to_num(features_unscaled, nan=0.0)
        features_scaled = scaler.transform(features_unscaled)
        # Create DataFrame with scaled features and original string label (for env init)
        labeled_df_scaled = pd.DataFrame()
        labeled_df_scaled['feature_vector'] = [list(vec) for vec in features_scaled]
        labeled_df_scaled['label'] = labeled_df_unscaled['label'].values
        log.info(f"Scaled {len(labeled_df_scaled)} labeled samples.")
    except Exception as e: log.error(f"Error scaling fetched data: {e}"); return

    # 5. Load Existing RL Model
    log.info(f"Loading model: {args.model_load_path}")
    if not args.model_load_path.exists(): log.error("Model file not found."); return
    try:
        # Load the model structure. We will set the env later.
        # IMPORTANT: This assumes the loaded model's output layer size MATCHES the current num_classes.
        # If classes were added since last save, this load might fail or require architecture modification (not implemented here).
        model = DQN.load(args.model_load_path, custom_objects={'learning_rate': 0.0})
        log.info(f"Model loaded: {type(model)}")
        # --- Verification (Optional but Recommended) ---
        # Check if model's action space matches current classes
        if isinstance(model.action_space, spaces.Discrete) and model.action_space.n != num_classes:
             log.error(f"Model action space size ({model.action_space.n}) does not match current number of classes ({num_classes})!")
             log.error("Retraining requires model architecture adaptation (e.g., output layer resize, EWC) - Not implemented in this script.")
             # Decide how to handle: exit, try anyway (might fail), or implement adaptation.
             # For now, we exit.
             return
        log.info(f"Model action space size ({model.action_space.n}) matches current classes ({num_classes}).")
        # --- End Verification ---
    except Exception as e: log.error(f"Error loading model: {e}"); return

    # 6. Create Fine-tuning Environment (using WidsEnvMultiClass)
    try:
        env_kwargs = {
            'data_df': labeled_df_scaled, # Pass scaled data + string labels
            'wids_api_url': args.api_url, # Needed for env to init its own class map
            'reward_config': FINETUNE_REWARD_CONFIG, # Use fine-tuning rewards
            'max_steps': len(labeled_df_scaled) # One pass through labeled data
        }
        finetune_env = make_vec_env(lambda: WidsEnvMultiClass(**env_kwargs), n_envs=1)
        log.info("Created fine-tuning environment.")
    except Exception as e: log.error(f"Error creating fine-tuning env: {e}"); return

    # 7. Fine-tune the Model
    try:
        model.set_env(finetune_env)
        model.learning_rate = args.lr
        log.info(f"Starting fine-tuning: Timesteps={args.timesteps}, LR={args.lr}")
        # Use the 'action_label' column (integers) implicitly handled by WidsEnvMultiClass's step method
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=True, log_interval=1)
        log.info("Fine-tuning complete.")
    except Exception as e: log.error(f"Error during fine-tuning: {e}")
    finally: finetune_env.close(); log.info("Fine-tuning environment closed.")

    # 8. Save the Fine-tuned Model
    try:
        args.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_save_path)
        log.info(f"Fine-tuned model saved to: {args.model_save_path}")
    except Exception as e: log.error(f"Error saving fine-tuned model: {e}")

    log.info("--- Multi-Class Fine-tuning Script Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune WIDS RL agent with multi-class labeled data.")
    parser.add_argument("--model-load-path", type=Path, default=MODEL_LOAD_PATH)
    parser.add_argument("--model-save-path", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--scaler-path", type=Path, default=SCALER_PATH)
    parser.add_argument("--api-url", default=WIDS_API_URL, help="Base URL of the WIDS backend API.")
    parser.add_argument("--timesteps", type=int, default=FINETUNE_TIMESTEPS)
    parser.add_argument("--lr", type=float, default=FINETUNE_LEARNING_RATE)
    args = parser.parse_args()

    # Add reward config directly if needed (not via args for simplicity here)
    # args.high_reward = FINETUNE_REWARD_CONFIG['Default_Attack'] # Example if needed separately
    # args.high_penalty = FINETUNE_REWARD_CONFIG['Default_FN']

    asyncio.run(run_finetuning(args))

