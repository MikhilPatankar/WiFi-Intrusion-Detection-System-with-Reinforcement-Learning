
import os
from pydantic_settings import BaseSettings
from pathlib import Path

# Determine base directory more reliably
try:
    BACKEND_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DEFAULT_DATA_DIR = BACKEND_ROOT_DIR / "data"
    # Assume models dir is sibling to src dir
    DEFAULT_MODELS_DIR = BACKEND_ROOT_DIR.parent / "models"
    DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "wids_events.db"
    DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "wids_dqn_agent.zip" # RL Model
    DEFAULT_SCALER_PATH = DEFAULT_MODELS_DIR / "wids_scaler.joblib"
    DEFAULT_AE_MODEL_PATH = DEFAULT_MODELS_DIR / "anomaly_autoencoder.h5"
    DEFAULT_AE_THRESHOLD_PATH = DEFAULT_MODELS_DIR / "ae_threshold.joblib"
    # Format SQLite URL correctly for absolute paths
    DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{DEFAULT_DB_PATH.resolve()}"
except NameError:
    # Fallback if __file__ is not defined
    print("Warning: Could not reliably determine backend root directory via __file__. Using relative paths.")
    BACKEND_ROOT_DIR = Path(".")
    DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./data/wids_events.db"
    DEFAULT_MODEL_PATH = Path("../../models/wids_dqn_agent.zip")
    DEFAULT_SCALER_PATH = Path("../../models/wids_scaler.joblib")
    DEFAULT_AE_MODEL_PATH = Path("../../models/anomaly_autoencoder.h5")
    DEFAULT_AE_THRESHOLD_PATH = Path("../../models/ae_threshold.joblib")

class Settings(BaseSettings):
    """Backend API Application Settings"""
    PROJECT_NAME: str = "WIDS FastAPI Backend API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "API for WIDS Hybrid Detection (RL+AE) & Logging"

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    # Model and Scaler Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    SCALER_PATH: str = os.getenv("SCALER_PATH", str(DEFAULT_SCALER_PATH))
    AE_MODEL_PATH: str = os.getenv("AE_MODEL_PATH", str(DEFAULT_AE_MODEL_PATH))
    AE_THRESHOLD_PATH: str = os.getenv("AE_THRESHOLD_PATH", str(DEFAULT_AE_THRESHOLD_PATH))

    # Other Config
    BACKEND_DIR: str = str(BACKEND_ROOT_DIR)
    # Use 'memory://' for simple in-memory, or 'redis://localhost:6379' for Redis
    BROADCAST_URL: str = os.getenv("BROADCAST_URL", "memory://")

    class Config:
        # env_file = ".env" # Optional: Load from .env file in backend root
        pass

settings = Settings()

# Log loaded settings for verification during startup
print(f"--- Backend API Settings ---")
print(f"Database URL: {settings.DATABASE_URL}")
print(f"RL Model Path: {settings.MODEL_PATH}")
print(f"Scaler Path: {settings.SCALER_PATH}")
print(f"AE Model Path: {settings.AE_MODEL_PATH}")
print(f"AE Threshold Path: {settings.AE_THRESHOLD_PATH}")
print(f"Broadcast URL: {settings.BROADCAST_URL}")
print(f"----------------------------")
