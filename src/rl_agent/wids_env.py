# src/rl_agent/wids_env.py
# Custom Gymnasium environment for WIDS RL training.
# Dynamically fetches attack types from backend API for multi-class action space.
# Assumes input data is ALREADY SCALED.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import requests # To fetch attack types from backend API
import time
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Default API URL (can be overridden)
DEFAULT_WIDS_API_URL = "http://127.0.0.1:8000" # URL of your backend API

class WidsEnvMultiClass(gym.Env):
    """
    Custom Gymnasium Environment for Multi-Class WIDS RL Agent.
    Fetches attack types from the backend API to define the action space dynamically.
    Assumes input data_df contains SCALED features and a 'label' column
    where 'label' is the STRING NAME of the class (e.g., "Normal", "Deauth", "NewAttackX").
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 data_df: pd.DataFrame,
                 wids_api_url: str = DEFAULT_WIDS_API_URL,
                 reward_config: Optional[Dict[str, float]] = None,
                 max_steps: Optional[int] = None):
        """
        Initializes the environment.

        Args:
            data_df (pd.DataFrame): DataFrame containing scaled features and 'label' (string names).
            wids_api_url (str): Base URL of the WIDS backend API.
            reward_config (dict, optional): Defines rewards. Keys should match class names
                                            (e.g., 'Normal', 'Deauth', 'Default_Attack', 'Default_FN', 'Default_FP').
            max_steps (int, optional): Max steps per episode. Defaults to dataset length.
        """
        super().__init__()
        log.info(f"Initializing WidsEnvMultiClass with DataFrame shape: {data_df.shape}")
        self.wids_api_url = wids_api_url
        self.data = data_df.reset_index(drop=True)

        # --- Validate Input Data ---
        if self.data.empty: raise ValueError("Input DataFrame 'data_df' cannot be empty.")
        if 'label' not in self.data.columns: raise ValueError("Input DataFrame must contain a 'label' column with string class names.")
        self.feature_columns = [col for col in self.data.columns if col != 'label']
        if not self.feature_columns: raise ValueError("Input DataFrame must contain feature columns besides 'label'.")
        log.info(f"Using {len(self.feature_columns)} feature columns.")
        if self.data[self.feature_columns].isnull().values.any(): log.warning("NaN values detected in input features.")
        self.num_features = len(self.feature_columns)

        # --- Fetch Attack Types from Backend API ---
        self.class_labels: List[str] = []
        self.action_map: Dict[int, str] = {} # Map action index -> class name
        self.label_to_action: Dict[str, int] = {} # Map class name -> action index
        self._fetch_and_set_attack_types() # Call helper method

        if not self.class_labels:
             raise RuntimeError("Failed to fetch or process attack types from API. Cannot initialize environment.")

        self.num_classes = len(self.class_labels)
        log.info(f"Initialized with {self.num_classes} classes: {self.class_labels}")

        # --- Prepare Data (Map string labels to action indices) ---
        # Add an 'action_label' column based on the fetched mapping
        self.data['action_label'] = self.data['label'].map(self.label_to_action)
        # Handle any labels in the data that weren't in the fetched types (treat as 'Normal' or error out)
        unknown_labels = self.data[self.data['action_label'].isnull()]['label'].unique()
        if len(unknown_labels) > 0:
            log.warning(f"Data contains labels not found in fetched active types: {unknown_labels}. Mapping them to 'Normal' (action 0).")
            normal_action_index = self.label_to_action.get("Normal", 0) # Get index for Normal
            self.data['action_label'].fillna(normal_action_index, inplace=True)
        self.action_labels = self.data['action_label'].values.astype(int) # Store integer labels for step function

        # Extract features as NumPy array
        self.scaled_features = self.data[self.feature_columns].values.astype(np.float32)

        # --- Define Action and Observation Space ---
        self.action_space = spaces.Discrete(self.num_classes) # N classes
        # Assume features are scaled (e.g., MinMax to [0, 1])
        low_bounds = np.zeros(self.num_features, dtype=np.float32)
        high_bounds = np.ones(self.num_features, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # --- Reward Configuration (Multi-Class) ---
        # Example: Reward correct classifications, penalize incorrect ones.
        # Higher penalty for missing attacks (predicting Normal when it's an Attack).
        # Specific rewards/penalties can be defined per attack type if needed.
        default_rewards = {
            'Normal': 1.0,              # Reward for correctly identifying Normal
            'Default_Attack': 10.0,     # Default reward for correctly identifying any attack
            'Default_FP': -2.0,         # Default penalty for predicting Attack when Normal (False Positive)
            'Default_FN': -50.0,        # Default penalty for predicting Normal when Attack (False Negative)
            'Default_Mismatch': -5.0    # Default penalty for predicting Attack X when Attack Y
        }
        self.reward_config = reward_config if reward_config is not None else default_rewards
        log.info(f"Using reward config: {self.reward_config}")

        # --- Episode State ---
        self.max_steps = max_steps if max_steps is not None else len(self.data)
        self.current_step = 0
        self.total_reward = 0

        log.info("WidsEnvMultiClass initialized successfully.")

    def _fetch_and_set_attack_types(self):
        """Fetches active attack types from the backend API."""
        api_endpoint = f"{self.wids_api_url}/attack_types/?include_inactive=false"
        log.info(f"Fetching active attack types from: {api_endpoint}")
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(api_endpoint, timeout=10)
                response.raise_for_status()
                attack_types_data = response.json() # Expects list of dicts like AttackTypeRead model

                # Ensure "Normal" is always class 0
                self.class_labels = ["Normal"]
                # Add other active types fetched from API
                # Sort them consistently (e.g., by name or id) to ensure mapping stability
                fetched_types = sorted(
                    [t['type_name'] for t in attack_types_data if t.get('type_name') and t['type_name'] != "Normal"],
                    key=str.lower
                )
                self.class_labels.extend(fetched_types)

                # Create mappings
                self.action_map = {i: label for i, label in enumerate(self.class_labels)}
                self.label_to_action = {label: i for i, label in self.action_map.items()}
                log.info(f"Successfully fetched and processed {len(self.class_labels)} active classes.")
                return # Success
            except requests.exceptions.RequestException as e:
                log.error(f"API request failed (attempt {attempt+1}/{retries}): {e}")
            except Exception as e:
                 log.error(f"Error processing attack types (attempt {attempt+1}/{retries}): {e}", exc_info=True)

            if attempt < retries - 1:
                log.info("Retrying after 5 seconds...")
                time.sleep(5)
            else:
                 log.error("Failed to fetch attack types after multiple retries.")
                 # Optionally fallback to a default list or raise error
                 # self.class_labels = ["Normal", "Generic_Attack"] # Example fallback
                 # self.action_map = {0: "Normal", 1: "Generic_Attack"}
                 # self.label_to_action = {"Normal": 0, "Generic_Attack": 1}


    def _get_obs(self):
        idx = min(self.current_step, len(self.scaled_features) - 1)
        return self.scaled_features[idx]

    def _get_info(self):
        idx = min(self.current_step, len(self.action_labels) - 1)
        true_action_label = self.action_labels[idx]
        true_class_name = self.action_map.get(true_action_label, "Unknown")
        return {"true_label": true_class_name, "true_action": true_action_label}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
        # Optional: Re-fetch attack types on reset if they might change frequently?
        # self._fetch_and_set_attack_types() # Be careful, this changes action space size!
        log.debug("Environment reset.")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Executes one step, calculates multi-class reward."""
        if self.current_step >= len(self.data):
             return np.zeros(self.num_features, dtype=np.float32), 0, True, False, {}

        terminated = False; truncated = False
        predicted_action = int(action)
        true_action = self.action_labels[self.current_step]

        predicted_class = self.action_map.get(predicted_action, "InvalidAction")
        true_class = self.action_map.get(true_action, "UnknownTrueLabel")

        # --- Calculate Multi-Class Reward ---
        reward = 0
        if predicted_action == true_action:
            # Correct Classification
            if true_class == "Normal":
                reward = self.reward_config.get('Normal', self.reward_config.get('tn', 1.0)) # Use specific or default TN
            else: # Correctly identified an attack
                reward = self.reward_config.get(true_class, self.reward_config.get('Default_Attack', 10.0)) # Use specific or default TP
        else:
            # Incorrect Classification
            if true_class == "Normal": # False Positive (Normal classified as Attack)
                reward = self.reward_config.get(f'FP_{predicted_class}', self.reward_config.get('Default_FP', -2.0))
            elif predicted_class == "Normal": # False Negative (Attack classified as Normal)
                reward = self.reward_config.get(f'FN_{true_class}', self.reward_config.get('Default_FN', -50.0))
            else: # Attack Mismatch (Attack X classified as Attack Y)
                reward = self.reward_config.get(f'MM_{true_class}_{predicted_class}', self.reward_config.get('Default_Mismatch', -5.0))
        # --- End Reward Calculation ---

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= len(self.data): terminated = True
        if self.max_steps is not None and self.current_step >= self.max_steps: truncated = True; terminated = True

        observation = self._get_obs() if not terminated else np.zeros(self.num_features, dtype=np.float32)
        info = self._get_info() if not terminated else {}

        return observation, reward, terminated, truncated, info

    def close(self):
        log.info("Closing WidsEnvMultiClass.")
        pass

# --- Example Usage ---
if __name__ == '__main__':
    log.info("--- Testing WidsEnvMultiClass ---")
    # Requires backend API running at DEFAULT_WIDS_API_URL to fetch types
    # Create dummy SCALED data with STRING labels
    num_samples = 200; num_features = 10
    dummy_scaled_features = np.random.rand(num_samples, num_features).astype(np.float32)
    # Example labels - MUST match types known by the backend API
    possible_labels = ["Normal", "Deauth", "Injection"] # Example
    dummy_labels = np.random.choice(possible_labels, num_samples)
    dummy_df = pd.DataFrame(dummy_scaled_features, columns=[f'f{i}' for i in range(num_features)])
    dummy_df['label'] = dummy_labels
    log.info(f"Created dummy scaled DataFrame shape: {dummy_df.shape}")

    try:
        env = WidsEnvMultiClass(data_df=dummy_df, wids_api_url=DEFAULT_WIDS_API_URL)
        from stable_baselines3.common.env_checker import check_env
        log.info("Running environment checker...")
        check_env(env) # Will fail if API call fails
        log.info("Environment check passed.")
        obs, info = env.reset(); log.info(f"Reset successful. Initial obs shape: {obs.shape}")
        for i in range(5): action = env.action_space.sample(); obs, reward, term, trunc, info = env.step(action); log.debug(f"Step {i+1}: Action={action}, Reward={reward:.1f}, Info={info}")
        env.close()
    except Exception as e: log.error(f"Error during env testing: {e}", exc_info=True)
    log.info("--- WidsEnvMultiClass Test Complete ---")

