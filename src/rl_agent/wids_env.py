# src/rl_agent/wids_env.py
# Custom Gymnasium environment for WIDS RL training.
# Assumes input data is ALREADY SCALED.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class WidsEnv(gym.Env):
    """
    Custom Gymnasium Environment for WiFi Intrusion Detection RL Agent.
    Assumes input data is a DataFrame with SCALED features and a 'label' column.
    Label convention: 0 for Normal, 1 for Known Attack.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self, data_df: pd.DataFrame, reward_config: dict = None, max_steps: int = None):
        """
        Initializes the environment using a pre-loaded DataFrame.

        Args:
            data_df (pd.DataFrame): DataFrame containing scaled features and 'label'.
            reward_config (dict, optional): Defines rewards {'tp', 'tn', 'fp', 'fn'}.
            max_steps (int, optional): Max steps per episode. Defaults to dataset length.
        """
        super().__init__()
        log.info(f"Initializing WidsEnv with DataFrame shape: {data_df.shape}")

        self.data = data_df.reset_index(drop=True) # Ensure clean index

        # --- Validate Input Data ---
        if self.data.empty:
            raise ValueError("Input DataFrame 'data_df' cannot be empty.")
        if 'label' not in self.data.columns:
            raise ValueError("Input DataFrame must contain a 'label' column (0=Normal, 1=Known Attack).")
        # Identify feature columns (assume all others are features)
        self.feature_columns = [col for col in self.data.columns if col != 'label']
        if not self.feature_columns:
            raise ValueError("Input DataFrame must contain feature columns besides 'label'.")
        log.info(f"Using {len(self.feature_columns)} feature columns.")

        # Check for NaNs in features
        if self.data[self.feature_columns].isnull().values.any():
             log.warning("NaN values detected in input features. Ensure data is cleaned/imputed before passing to env.")
             # Option: Fill here, but better to handle upstream before scaling
             # self.data[self.feature_columns] = self.data[self.feature_columns].fillna(0)

        self.labels = self.data['label'].values
        # Extract features as NumPy array, ensure float32
        self.scaled_features = self.data[self.feature_columns].values.astype(np.float32)

        # --- Define Action and Observation Space ---
        self.num_features = self.scaled_features.shape[1]
        # Action: 0 = Classify as Normal, 1 = Classify as Known Attack
        self.action_space = spaces.Discrete(2)
        # Observation: Scaled feature vector
        # Determine bounds from data (assuming scaled, e.g., [0, 1] or standardized)
        # Using [0, 1] as a common default for MinMax scaled data
        # If using StandardScaler, bounds might be wider (e.g., -5 to 5) or use -inf, +inf
        low_bounds = np.zeros(self.num_features, dtype=np.float32) # Adjust if not MinMax [0,1]
        high_bounds = np.ones(self.num_features, dtype=np.float32) # Adjust if not MinMax [0,1]
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # --- Reward Configuration ---
        default_rewards = {'tp': 10, 'tn': 1, 'fp': -2, 'fn': -20} # TP=Attack, TN=Normal
        self.reward_config = reward_config if reward_config is not None else default_rewards
        log.info(f"Using reward config: {self.reward_config}")

        # --- Episode State ---
        self.max_steps = max_steps if max_steps is not None else len(self.data)
        self.current_step = 0
        self.total_reward = 0

        log.info("WidsEnv initialized successfully.")

    def _get_obs(self):
        """Returns the observation for the current step."""
        # Handle potential index out of bounds if called after termination
        idx = min(self.current_step, len(self.scaled_features) - 1)
        return self.scaled_features[idx]

    def _get_info(self):
        """Returns auxiliary information."""
        idx = min(self.current_step, len(self.labels) - 1)
        return {"true_label": self.labels[idx]}

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of the data."""
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
        log.debug("Environment reset.")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Executes one step based on the agent's action."""
        if self.current_step >= len(self.data):
             # Should not happen if terminated is handled correctly, but as safeguard
             log.warning("Step called after end of data. Returning zero reward/obs.")
             return np.zeros(self.num_features, dtype=np.float32), 0, True, False, {}

        terminated = False
        truncated = False # Usually False unless using fixed episode length

        # --- Calculate Reward ---
        true_label = self.labels[self.current_step]
        predicted_label = int(action) # Agent's action (0 or 1)
        reward = 0

        if predicted_label == 1 and true_label == 1: # True Positive (Attack detected)
            reward = self.reward_config.get('tp', 10)
        elif predicted_label == 0 and true_label == 0: # True Negative (Normal identified)
            reward = self.reward_config.get('tn', 1)
        elif predicted_label == 1 and true_label == 0: # False Positive (Normal classified as Attack)
            reward = self.reward_config.get('fp', -2)
        elif predicted_label == 0 and true_label == 1: # False Negative (Attack classified as Normal)
            reward = self.reward_config.get('fn', -20)

        self.total_reward += reward

        # --- Move to next state ---
        self.current_step += 1

        # --- Check for termination ---
        if self.current_step >= len(self.data):
            terminated = True
            log.debug(f"Episode terminated (end of data) at step {self.current_step}. Total reward: {self.total_reward}")
        # Check for truncation (optional, if max_steps is set)
        if self.max_steps is not None and self.current_step >= self.max_steps:
             truncated = True
             terminated = True # Often terminate when truncated in this setup
             log.debug(f"Episode truncated (max steps) at step {self.current_step}. Total reward: {self.total_reward}")

        # Get next observation and info
        observation = self._get_obs() if not terminated else np.zeros(self.num_features, dtype=np.float32)
        info = self._get_info() if not terminated else {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        log.info("Closing WidsEnv.")
        pass

# --- Example Usage (for testing the environment directly) ---
if __name__ == '__main__':
    log.info("--- Testing WidsEnv ---")
    # Create dummy SCALED data
    num_samples = 200
    num_features = 10
    dummy_scaled_features = np.random.rand(num_samples, num_features).astype(np.float32) # Assume scaled [0,1]
    dummy_labels = np.random.randint(0, 2, num_samples) # 0=Normal, 1=Attack
    dummy_df = pd.DataFrame(dummy_scaled_features, columns=[f'f{i}' for i in range(num_features)])
    dummy_df['label'] = dummy_labels
    log.info(f"Created dummy scaled DataFrame with shape: {dummy_df.shape}")

    try:
        # Instantiate the environment with the DataFrame
        env = WidsEnv(data_df=dummy_df, max_steps=100)

        # Test with environment checker
        from stable_baselines3.common.env_checker import check_env
        log.info("Running environment checker...")
        check_env(env)
        log.info("Environment check passed.")

        # Test reset and step
        log.info("Testing reset()...")
        obs, info = env.reset()
        log.info(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
        log.info(f"Initial info: {info}")

        log.info("Testing step()...")
        total_ep_reward = 0
        for i in range(105): # Exceed max_steps to test truncation
            action = env.action_space.sample() # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_ep_reward += reward
            log.debug(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Term={terminated}, Trunc={truncated}")
            if terminated or truncated:
                log.info(f"Episode finished at step {i+1}. Total Reward: {total_ep_reward}")
                break
        env.close()
        log.info("--- WidsEnv Test Complete ---")

    except Exception as e:
        log.error(f"Error during environment testing: {e}", exc_info=True)
