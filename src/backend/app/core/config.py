# --- File: src/backend/app/core/config.py ---

import os
from pydantic_settings import BaseSettings

# --- Helper to determine base directory more reliably ---
# This assumes config.py is in src/backend/app/core
# We want the path relative to the 'src/backend' directory typically
try:
    # __file__ gives the path to the current file (config.py)
    # Go up two levels to get to src/backend/
    BACKEND_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_DATA_DIR = os.path.join(BACKEND_ROOT_DIR, "data")
    DEFAULT_MODELS_DIR = os.path.join(BACKEND_ROOT_DIR, "..", "models") # Assumes models is sibling to src

    # Construct default paths relative to the determined backend root
    DEFAULT_DB_PATH = os.path.join(DEFAULT_DATA_DIR, "wids_events.db")
    DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "wids_dqn_agent.zip")
    DEFAULT_SCALER_PATH = os.path.join(DEFAULT_MODELS_DIR, "wids_scaler.joblib")

    # Format SQLite URL correctly for absolute paths (needs three slashes ///)
    DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}"

except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive environments)
    print("Warning: Could not reliably determine backend root directory via __file__. Using relative paths.")
    BACKEND_ROOT_DIR = "."
    DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./data/wids_events.db"
    DEFAULT_MODEL_PATH = "../../models/wids_dqn_agent.zip"
    DEFAULT_SCALER_PATH = "../../models/wids_scaler.joblib"


class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings.
    Reads environment variables or uses default values.
    """
    PROJECT_NAME: str = "WIDS FastAPI Backend"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "API for WiFi Intrusion Detection System with RL"

    # --- Database Configuration ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    # --- Model and Scaler Configuration ---
    MODEL_PATH: str = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    SCALER_PATH: str = os.getenv("SCALER_PATH", DEFAULT_SCALER_PATH)

    # Store the calculated root directory for potential use elsewhere
    BACKEND_DIR: str = BACKEND_ROOT_DIR

    # Use 'memory://' for simple in-memory, or 'redis://localhost:6379' for Redis
    BROADCAST_URL: str = os.getenv("BROADCAST_URL", "memory://")

    class Config:
        # env_file = ".env"
        pass

settings = Settings()
print(f"Debug: Determined Backend Root: {settings.BACKEND_DIR}")
print(f"Debug: Default DB URL: {settings.DATABASE_URL}")
print(f"Debug: Default Model Path: {settings.MODEL_PATH}")
print(f"Debug: Default Scaler Path: {settings.SCALER_PATH}")
print(f"Broadcast URL: {settings.BROADCAST_URL}")
