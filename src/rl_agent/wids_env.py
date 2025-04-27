# src/rl_agent/wids_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WidsEnv(gym.Env):
    """
    Custom Gymnasium Environment for WiFi Intrusion Detection.

    Assumes input data is a preprocessed CSV file where each row represents
    a network state (e.g., aggregated features over a time window or flow)
    and includes a ground truth label (0 for Normal, 1 for Anomaly/Attack).
    """
    metadata = {'render_modes': [], 'render_fps': 4} # No rendering needed

    def __init__(self, data_path, reward_config=None, max_steps=None):
        """
        Initializes the environment.

        Args:
            data_path (str): Path to the preprocessed CSV data file.
                             Must contain numeric features and a 'label' column.
            reward_config (dict, optional): Dictionary defining rewards.
                                            Defaults provided if None.
                                            Example: {'tp': 10, 'tn': 1, 'fp': -2, 'fn': -20}
            max_steps (int, optional): Maximum number of steps per episode.
                                       If None, episode runs through the entire dataset.
        """
        super().__init__()
        logging.info(f"Initializing WidsEnv with data from: {data_path}")

        # --- Load and Preprocess Data ---
        try:
            self.data = pd.read_csv(data_path)
            logging.info(f"Loaded data with shape: {self.data.shape}")
        except FileNotFoundError:
            logging.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading data from {data_path}: {e}")
            raise

        # Separate features (X) and labels (y)
        if 'label' not in self.data.columns:
            raise ValueError("Data must contain a 'label' column.")
        self.labels = self.data['label'].values
        # Assume all other columns are features for now
        # In a real scenario, explicitly select feature columns
        self.features = self.data.drop('label', axis=1)

        # --- Data Preprocessing (Example: Scaling) ---
        # IMPORTANT: In a real project, use scalers fitted on the *training* data only
        # and apply the *same* scaler to validation/test/live data.
        # For simplicity here, we scale the entire loaded dataset.
        # Also handle potential non-numeric columns if not already done
        numeric_feature_cols = self.features.select_dtypes(include=np.number).columns
        if len(numeric_feature_cols) != len(self.features.columns):
            logging.warning("Non-numeric columns found in features. Attempting to handle...")
            # Example: Label encode object columns (could be improved)
            for col in self.features.select_dtypes(include='object').columns:
                 # Check if the column looks like MAC/IP addresses - skip encoding these for now
                 # A better approach would be specific embedding or feature hashing
                 if not any(addr_part in col for addr_part in ['addr', 'ip']):
                     logging.info(f"Label Encoding column: {col}")
                     le = LabelEncoder()
                     self.features[col] = le.fit_transform(self.features[col].astype(str)) # Convert to string first
                 else:
                     logging.warning(f"Skipping encoding for potential address column: {col}. Consider specific handling.")
                     # Drop address columns if they can't be used directly
                     self.features = self.features.drop(col, axis=1)

            # Re-select numeric columns after potential encoding/dropping
            numeric_feature_cols = self.features.select_dtypes(include=np.number).columns


        # Handle potential NaN values (Example: fill with median)
        if self.features[numeric_feature_cols].isnull().values.any():
            logging.warning("NaN values found in features. Filling with median.")
            self.features[numeric_feature_cols] = self.features[numeric_feature_cols].fillna(
                self.features[numeric_feature_cols].median()
            )

        # Apply Min-Max Scaling to numeric features
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.features[numeric_feature_cols])
        logging.info(f"Applied Min-Max scaling to {len(numeric_feature_cols)} numeric features.")

        # --- Define Action and Observation Space ---
        self.num_features = self.scaled_features.shape[1]
        # Action: 0 = Classify as Normal, 1 = Classify as Anomaly
        self.action_space = spaces.Discrete(2)
        # Observation: Scaled feature vector
        # Define low/high bounds based on scaler (MinMaxScaler outputs [0, 1])
        low_bounds = np.zeros(self.num_features, dtype=np.float32)
        high_bounds = np.ones(self.num_features, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # --- Reward Configuration ---
        default_rewards = {'tp': 10, 'tn': 1, 'fp': -2, 'fn': -20}
        self.reward_config = reward_config if reward_config is not None else default_rewards
        logging.info(f"Using reward configuration: {self.reward_config}")

        # --- Episode State ---
        self.max_steps = max_steps if max_steps is not None else len(self.data)
        self.current_step = 0
        self.total_reward = 0

        logging.info("WidsEnv initialized successfully.")

    def _get_obs(self):
        """Returns the observation for the current step."""
        return self.scaled_features[self.current_step].astype(np.float32)

    def _get_info(self):
        """Returns auxiliary information (optional)."""
        # Could include things like the true label for debugging/analysis
        return {"true_label": self.labels[self.current_step]}

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of the data."""
        super().reset(seed=seed) # Important for reproducibility

        self.current_step = 0
        self.total_reward = 0
        logging.debug("Environment reset.")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Executes one step in the environment based on the agent's action.

        Args:
            action (int): The action chosen by the agent (0 or 1).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        terminated = False
        truncated = False

        # --- Calculate Reward ---
        true_label = self.labels[self.current_step]
        reward = 0

        if action == 1 and true_label == 1: # True Positive (Anomaly correctly identified)
            reward = self.reward_config.get('tp', 10)
        elif action == 0 and true_label == 0: # True Negative (Normal correctly identified)
            reward = self.reward_config.get('tn', 1)
        elif action == 1 and true_label == 0: # False Positive (Normal classified as Anomaly)
            reward = self.reward_config.get('fp', -2)
        elif action == 0 and true_label == 1: # False Negative (Anomaly classified as Normal)
            reward = self.reward_config.get('fn', -20)

        self.total_reward += reward

        # --- Move to next state ---
        self.current_step += 1

        # --- Check for termination/truncation ---
        if self.current_step >= len(self.data):
            terminated = True # End of dataset reached
            logging.debug(f"Episode terminated (end of data) at step {self.current_step}. Total reward: {self.total_reward}")
        elif self.current_step >= self.max_steps:
             truncated = True # Max steps reached
             logging.debug(f"Episode truncated (max steps) at step {self.current_step}. Total reward: {self.total_reward}")


        # Get next observation and info
        # Handle edge case where terminated/truncated on the last step
        if terminated or truncated:
             # Return the last valid observation or a zero vector if preferred
             observation = self._get_obs() if self.current_step < len(self.data) else np.zeros(self.num_features, dtype=np.float32)
             info = self._get_info() if self.current_step < len(self.data) else {}
        else:
            observation = self._get_obs()
            info = self._get_info()

        # Log step details (optional, can be verbose)
        # logging.debug(
        #     f"Step: {self.current_step}, Action: {action}, True Label: {true_label}, "
        #     f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
        # )

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up any resources (if needed)."""
        logging.info("Closing WidsEnv.")
        # No specific resources to close in this basic version
        pass

# --- Example Usage (for testing the environment) ---
if __name__ == '__main__':
    # Create a dummy CSV file for testing
    dummy_data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100) * 10,
        'feature3': np.random.randint(0, 5, 100),
        'some_address': ['192.168.1.'+str(i) for i in range(100)], # Example non-numeric
        'label': np.random.randint(0, 2, 100) # Random labels 0 or 1
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_path = "data/processed/dummy_wids_data.csv"
    os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
    dummy_df.to_csv(dummy_path, index=False)
    logging.info(f"Created dummy data file at: {dummy_path}")

    # Instantiate the environment
    try:
        env = WidsEnv(data_path=dummy_path, max_steps=50)

        # Test with environment checker
        from stable_baselines3.common.env_checker import check_env
        logging.info("Running environment checker...")
        check_env(env)
        logging.info("Environment check passed.")

        # Test reset and step
        logging.info("Testing reset()...")
        obs, info = env.reset()
        logging.info(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
        logging.info(f"Initial info: {info}")

        logging.info("Testing step()...")
        for i in range(5):
            action = env.action_space.sample() # Take random action
            obs, reward, terminated, truncated, info = env.step(action)
            logging.info(
                f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}, "
                f"Obs shape={obs.shape}, Info={info}"
            )
            if terminated or truncated:
                logging.info("Episode finished.")
                break

        env.close()

    except Exception as e:
        logging.error(f"Error during environment testing: {e}", exc_info=True)
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            logging.info(f"Removed dummy data file: {dummy_path}")

