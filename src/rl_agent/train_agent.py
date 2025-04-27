# src/rl_agent/train_agent.py

import os
import logging
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

# Import the custom environment
from wids_env import WidsEnv # Assuming wids_env.py is in the same directory or Python path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# These could be loaded from a config file (e.g., YAML) in a real project
DEFAULT_DATA_PATH = "data/processed/dummy_wids_data.csv" # Use dummy data created by wids_env.py if run directly
DEFAULT_LOG_DIR = "logs/rl_training/"
DEFAULT_MODEL_SAVE_PATH = "models/wids_dqn_agent"
DEFAULT_TOTAL_TIMESTEPS = 10000 # Reduce for quick testing; increase significantly for real training
CHECKPOINT_FREQ = 5000 # Save model every N steps
EVAL_FREQ = 2000 # Evaluate model every N steps

# Reward configuration (should match the one used in WidsEnv or be passed to it)
REWARD_CONFIG = {'tp': 10, 'tn': 1, 'fp': -2, 'fn': -20}
# If using EvalCallback, a reward threshold can stop training early
# Calculate a potential target based on average reward per step if desired
# Example: Aim for an average reward > 0 over an episode of 100 steps
# REWARD_THRESHOLD = 0 * 100 # Needs careful calculation based on data/rewards

def train_agent(data_path, total_timesteps, log_dir, model_save_path, use_dummy_data=False):
    """
    Trains a DQN agent using the custom WidsEnv.

    Args:
        data_path (str): Path to the training data CSV file.
        total_timesteps (int): Total number of training steps.
        log_dir (str): Directory to save training logs and models.
        model_save_path (str): Base path/filename to save the final model.
        use_dummy_data (bool): If True, creates dummy data for training.
    """
    logging.info("--- Starting RL Agent Training ---")

    # --- Create Dummy Data if needed ---
    if use_dummy_data:
        import numpy as np
        import pandas as pd
        logging.info(f"Creating dummy data at: {data_path}")
        dummy_data = {
            'feature1': np.random.rand(500), # Larger dummy dataset
            'feature2': np.random.rand(500) * 10,
            'feature3': np.random.randint(0, 5, 500),
            'label': np.random.randint(0, 2, 500)
        }
        dummy_df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        dummy_df.to_csv(data_path, index=False)

    # --- Create and Vectorize Environment ---
    # Using make_vec_env is good practice, even for n_envs=1
    # It automatically wraps the environment with Monitor for logging episode rewards/lengths
    # Pass environment keyword arguments using env_kwargs
    env_kwargs = {'data_path': data_path, 'reward_config': REWARD_CONFIG}
    env = make_vec_env(lambda: WidsEnv(**env_kwargs), n_envs=1)
    logging.info("Created vectorized environment.")

    # --- Configure Callbacks ---
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)

    # Save checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=log_dir,
        name_prefix="wids_dqn_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True, # Important if using VecNormalize wrapper
    )

    # Evaluate the agent periodically and save the best model
    # Requires a separate evaluation environment (can be the same class with different data split)
    # For simplicity here, we use the same environment instance for evaluation
    # In practice, use a dedicated validation dataset for eval_env
    eval_env = make_vec_env(lambda: WidsEnv(**env_kwargs), n_envs=1) # Re-create for eval
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True, # Use deterministic actions for evaluation
        render=False,
        n_eval_episodes=5 # Evaluate over 5 episodes
    )

    # Optional: Stop training if a reward threshold is met
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    # combined_callback = CallbackList([checkpoint_callback, eval_callback, stop_callback])
    combined_callback = [checkpoint_callback, eval_callback] # List of callbacks

    # --- Instantiate and Train the Agent ---
    # DQN Hyperparameters (example, requires tuning)
    # See Stable Baselines3 documentation for details
    model = DQN(
        "MlpPolicy",          # Use Multi-Layer Perceptron policy
        env,
        verbose=1,            # Print training progress
        tensorboard_log=os.path.join(log_dir, "tensorboard/"), # Log for TensorBoard
        learning_rate=1e-4,   # Example learning rate
        buffer_size=50000,    # Size of the replay buffer
        learning_starts=1000, # Steps before learning starts
        batch_size=32,        # Samples per gradient update
        tau=1.0,              # Soft update coefficient (1.0 = hard update)
        gamma=0.99,           # Discount factor
        train_freq=4,         # Update the model every 4 steps
        gradient_steps=1,     # How many gradient steps per update
        # exploration_fraction=0.1, # Fraction of entire training period over which exploration rate is reduced
        # exploration_final_eps=0.05, # Final value of random action probability
        # target_update_interval=1000, # Update target network every N steps (alternative to tau)
        seed=42               # For reproducibility
    )

    logging.info(f"Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            log_interval=10, # Log scalar values (like reward) every N episodes
            tb_log_name="DQN_WIDS"
        )
        logging.info("Training finished.")

        # --- Save the Final Model ---
        model.save(model_save_path)
        logging.info(f"Final model saved to {model_save_path}")

        # Optional: Save the replay buffer separately if needed later
        # model.save_replay_buffer(os.path.join(model_dir, "dqn_replay_buffer"))

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
    finally:
        # Close environments
        env.close()
        eval_env.close()
        logging.info("Environments closed.")
        # Clean up dummy data if created
        if use_dummy_data and os.path.exists(data_path):
            os.remove(data_path)
            logging.info(f"Removed dummy data file: {data_path}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for WIDS.")
    parser.add_argument("--data", default=None, help=f"Path to training data CSV. If omitted, uses dummy data at {DEFAULT_DATA_PATH}")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS, help="Total training timesteps.")
    parser.add_argument("--logdir", default=DEFAULT_LOG_DIR, help="Directory for logs and models.")
    parser.add_argument("--savepath", default=DEFAULT_MODEL_SAVE_PATH, help="Path to save the final trained model.")

    args = parser.parse_args()

    # Determine data path and whether to use dummy data
    use_dummy = args.data is None
    data_file_path = DEFAULT_DATA_PATH if use_dummy else args.data

    train_agent(
        data_path=data_file_path,
        total_timesteps=args.timesteps,
        log_dir=args.logdir,
        model_save_path=args.savepath,
        use_dummy_data=use_dummy
    )

    logging.info("--- Training Script Completed ---")
