# --- File: src/backend/app/core/config.py ---

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings.
    Reads environment variables or uses default values.
    """
    PROJECT_NAME: str = "WIDS FastAPI Backend"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "API for WiFi Intrusion Detection System with RL"

    # --- Database Configuration ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/wids_events.db")

    # --- Model and Scaler Configuration ---
    # Path to the saved RL agent model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../../models/wids_dqn_agent.zip") # Adjust path as needed
    # Path to the saved feature scaler (e.g., from scikit-learn)
    SCALER_PATH: str = os.getenv("SCALER_PATH", "../../models/wids_scaler.joblib") # Adjust path as needed

    class Config:
        # env_file = ".env"
        pass

settings = Settings()
