# src/rl_agent/train_agent.py
# Trains the initial RL agent (e.g., DQN) using processed and scaled data.

import os
import logging
import argparse
import pandas as pd
import numpy as np
import joblib # To load scaler
from stable_baselines3 import DQN # Or PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import sys

# --- Import custom environment ---
# Assuming structure allows importing from rl_agent
try:
    from wids_env import WidsEnvMultiClass
except ImportError:
     # Add path if run from project root
     project_root = Path(__file__).resolve().parent.parent.parent
     src_path = project_root / 'src'
     sys.path.insert(0, str(src_path))
     from rl_agent.wids_env import WidsEnvMultiClass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_PROCESSED_DATA_PATH = "../../data/processed/extracted_features.csv"
DEFAULT_SCALER_PATH = "../../models/wids_scaler.joblib"
DEFAULT_MODEL_SAVE_PATH = "../../models/wids_dqn_agent.zip"
DEFAULT_LOG_DIR = "../../logs/rl_initial_training/"
DEFAULT_TOTAL_TIMESTEPS = 50000 # Adjust as needed
CHECKPOINT_FREQ = 10000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5

# Reward config (should match env defaults or be passed)
REWARD_CONFIG = {'tp': 10, 'tn': 1, 'fp': -2, 'fn': -20}

def train_rl_agent(data_path: Path, scaler_path: Path, model_save_path: Path, log_dir: Path, total_timesteps: int):
    """Loads data, scales it, creates env, trains, and saves the RL agent."""
    log.info("--- Starting Initial RL Agent Training ---")

    # 1. Load Data
    log.info(f"Loading data from: {data_path}")
    try: df = pd.read_csv(data_path); log.info(f"Loaded data shape: {df.shape}")
    except FileNotFoundError: log.error(f"Data file not found: {data_path}"); return
    except Exception as e: log.error(f"Error loading data: {e}"); return

    # 2. Load Scaler
    log.info(f"Loading scaler from: {scaler_path}")
    try: scaler = joblib.load(scaler_path); assert hasattr(scaler, 'transform'); log.info("Scaler loaded.")
    except FileNotFoundError: log.error(f"Scaler file not found: {scaler_path}"); return
    except Exception as e: log.error(f"Error loading scaler: {e}"); return

    # 3. Prepare Data for Env (Select features, Scale, Keep Labels)
    if 'label' not in df.columns: log.error("Data must include 'label' column."); return
    exclude_cols = ['label', 'timestamp', 'event_uid', 'id'] # Adapt as needed
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols: log.error("No numeric feature columns found."); return
    log.info(f"Using {len(feature_cols)} features for training.")

    try:
        features_unscaled = df[feature_cols].values
        # Handle NaNs before scaling (e.g., fill with 0 or use imputation fitted on training data)
        if np.isnan(features_unscaled).any():
             log.warning("NaNs detected in features before scaling. Filling with 0 for training.")
             features_unscaled = np.nan_to_num(features_unscaled, nan=0.0) # Example: fill with 0

        features_scaled = scaler.transform(features_unscaled)
        # Create DataFrame expected by WidsEnvMultiClass
        scaled_df = pd.DataFrame(features_scaled, columns=feature_cols)
        scaled_df['label'] = df['label'].values # Add labels back
        log.info("Data scaling complete.")
    except Exception as e: log.error(f"Error scaling data: {e}"); return

    # 4. Create and Vectorize Environment
    # Pass the scaled DataFrame to the environment constructor
    env_kwargs = {'data_df': scaled_df, 'reward_config': REWARD_CONFIG}
    # Use Monitor wrapper for logging episode rewards/lengths
    env = make_vec_env(lambda: Monitor(WidsEnvMultiClass(**env_kwargs)), n_envs=1)
    log.info("Created vectorized environment.")

    # 5. Configure Callbacks
    log_dir.mkdir(parents=True, exist_ok=True)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=str(log_dir), name_prefix="rl_model_checkpoint")
    # Eval env: Use a separate validation split of scaled data if available
    # For simplicity here, re-using the same env instance for eval (not best practice)
    eval_callback = EvalCallback(env, best_model_save_path=str(log_dir / 'best_model'), log_path=str(log_dir), eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
    combined_callback = [checkpoint_callback, eval_callback]

    # 6. Instantiate and Train the Agent (Example: DQN)
    model = DQN(
        "MlpPolicy", env, verbose=1, tensorboard_log=str(log_dir / "tensorboard/"),
        learning_rate=1e-4, buffer_size=100000, learning_starts=1000, batch_size=64,
        tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, seed=42
    )
    log.info(f"Instantiated {type(model).__name__} agent.")
    log.info(f"Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=combined_callback, log_interval=10, tb_log_name="DQN_WIDS_Initial")
        log.info("Training finished.")
    except Exception as e: log.error(f"Error during training: {e}", exc_info=True); env.close(); return
    finally: env.close(); log.info("Environment closed.")

    # 7. Save the Final Model
    try: model.save(model_save_path); log.info(f"Final model saved to {model_save_path}")
    except Exception as e: log.error(f"Error saving final model: {e}")

    log.info("--- Initial RL Training Script Completed ---")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train initial WIDS RL agent.")
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, help="Path to processed CSV data.")
    parser.add_argument("--scaler", default=DEFAULT_SCALER_PATH, help="Path to saved scaler (.joblib).")
    parser.add_argument("--save-model", default=DEFAULT_MODEL_SAVE_PATH, help="Path to save trained RL model (.zip).")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for logs and checkpoints.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="Total training timesteps.")
    args = parser.parse_args()

    train_rl_agent(Path(args.data), Path(args.scaler), Path(args.save_model), Path(args.log_dir), args.timesteps)

