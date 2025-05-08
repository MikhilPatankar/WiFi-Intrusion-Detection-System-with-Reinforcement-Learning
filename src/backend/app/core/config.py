import os
from pydantic_settings import BaseSettings
from pathlib import Path

# Determine base directory structure
try:
    _CONFIG_FILE_PATH = Path(__file__).resolve()
    # Assuming config.py is in ProjectRoot/src/backend/app/core/
    CORE_DIR = _CONFIG_FILE_PATH.parent
    APP_DIR = CORE_DIR.parent           # ProjectRoot/src/backend/app/
    BACKEND_MODULE_DIR = APP_DIR.parent # ProjectRoot/src/backend/
    SRC_DIR = BACKEND_MODULE_DIR.parent # ProjectRoot/src/
    PROJECT_ROOT = SRC_DIR.parent       # ProjectRoot/

    # Database path: ProjectRoot/src/backend/app/db/wids_events.db
    DEFAULT_DATA_DIR = APP_DIR / "db"
    DEFAULT_DB_PATH = (DEFAULT_DATA_DIR / "wids_events.db").resolve()

    # Models path: ProjectRoot/models/
    # Comment: "Assume models dir is sibling to src dir"
    # src dir is PROJECT_ROOT / "src"
    # Sibling models dir is PROJECT_ROOT / "models"
    DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
    DEFAULT_MODEL_PATH = (DEFAULT_MODELS_DIR / "wids_dqn_agent.zip").resolve()
    DEFAULT_SCALER_PATH = (DEFAULT_MODELS_DIR / "wids_scaler.joblib").resolve()
    DEFAULT_AE_MODEL_PATH = (DEFAULT_MODELS_DIR / "anomaly_autoencoder.h5").resolve()
    DEFAULT_AE_THRESHOLD_PATH = (DEFAULT_MODELS_DIR / "ae_threshold.joblib").resolve()

    # This variable is used for settings.BACKEND_DIR.
    # The original code set BACKEND_ROOT_DIR = Path(__file__).resolve().parent.parent.parent,
    # which corresponds to BACKEND_MODULE_DIR (ProjectRoot/src/backend).
    EFFECTIVE_BACKEND_ROOT_FOR_SETTING = BACKEND_MODULE_DIR

except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive environments)
    print("Warning: __file__ is not defined. Using fallback paths relative to Current Working Directory (CWD).")
    
    # In fallback, EFFECTIVE_BACKEND_ROOT_FOR_SETTING becomes CWD.
    EFFECTIVE_BACKEND_ROOT_FOR_SETTING = Path(".").resolve()

    # Fallback for DB Path: CWD/src/backend/app/db/wids_events.db
    # This matches the structure implied by the original fallback's DEFAULT_DATABASE_URL:
    # "sqlite+aiosqlite:///./src/backend/app/db/wids_events.db"
    DEFAULT_DB_PATH = (Path(".") / "src" / "backend" / "app" / "db" / "wids_events.db").resolve()

    # Fallback for Model Paths:
    # Original fallback paths like Path("../../models/wids_dqn_agent.zip") were relative to config.py's location.
    # Without __file__, we can't reliably know config.py's location.
    # A common fallback assumption is that CWD is the project root.
    print("Warning: Fallback model paths assume CWD is the project root (e.g., 'WiFi-Intrusion-Detection-System-with-Reinforcement-Learning').")
    # If CWD is ProjectRoot, then models are in CWD/models/
    DEFAULT_MODEL_PATH = (Path(".") / "models" / "wids_dqn_agent.zip").resolve()
    DEFAULT_SCALER_PATH = (Path(".") / "models" / "wids_scaler.joblib").resolve()
    DEFAULT_AE_MODEL_PATH = (Path(".") / "models" / "anomaly_autoencoder.h5").resolve()
    DEFAULT_AE_THRESHOLD_PATH = (Path(".") / "models" / "ae_threshold.joblib").resolve()

# --- Create DB directory and file if they don't exist ---
# DEFAULT_DB_PATH is an absolute Path object from either try or except block.
db_dir = DEFAULT_DB_PATH.parent
try:
    print(f"Ensuring database directory exists: {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True) # Create parent directories if they don't exist

    if not DEFAULT_DB_PATH.exists():
        # Create an empty file. SQLite will use this file.
        print(f"Database file {DEFAULT_DB_PATH} not found, creating placeholder...")
        DEFAULT_DB_PATH.touch(exist_ok=True) 
        print(f"Database file placeholder created at: {DEFAULT_DB_PATH}")
    # else:
    #     print(f"Database file already exists at: {DEFAULT_DB_PATH}")
except OSError as e:
    # This is a potentially critical failure.
    print(f"ERROR: Could not create database directory or file at {DEFAULT_DB_PATH}. Error: {e}")
    print("The application may fail to start or operate correctly if this path is not usable.")
    # Depending on requirements, one might raise SystemExit here:
    # raise SystemExit(f"Failed to prepare database path: {e}")
except Exception as e:
    # Catch any other unexpected errors during path manipulation or file creation.
    print(f"UNEXPECTED ERROR during database path setup for {DEFAULT_DB_PATH}. Error: {e}")
    print("The application may fail to start or operate correctly.")
    # raise SystemExit(f"Unexpected error during database path setup: {e}")


# --- Define Database URL ---
# DEFAULT_DB_PATH is resolved and absolute.
DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}"




class Settings(BaseSettings):
    """Backend API Application Settings"""
    PROJECT_NAME: str = "WIDS FastAPI Backend API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "API for WIDS Hybrid Detection (RL+AE) & Logging"

    WIDS_API_BASE_URL = "http://127.0.0.1:80"

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    # Model and Scaler Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    SCALER_PATH: str = os.getenv("SCALER_PATH", str(DEFAULT_SCALER_PATH))
    AE_MODEL_PATH: str = os.getenv("AE_MODEL_PATH", str(DEFAULT_AE_MODEL_PATH))
    AE_THRESHOLD_PATH: str = os.getenv("AE_THRESHOLD_PATH", str(DEFAULT_AE_THRESHOLD_PATH))

    # Other Config
    # BACKEND_DIR is intended to be the root of the 'backend' module, e.g., ProjectRoot/src/backend
    BACKEND_DIR: str = str(EFFECTIVE_BACKEND_ROOT_FOR_SETTING)
    # Use 'memory://' for simple in-memory, or 'redis://localhost:6379' for Redis
    BROADCAST_URL: str = os.getenv("BROADCAST_URL", "memory://")

    class Config:
        # env_file = ".env" # Optional: Load from .env file in backend root
        pass

settings = Settings()

# Log loaded settings for verification during startup
print(f"--- Backend API Settings ---")
print(f"Effective Backend Dir (settings.BACKEND_DIR): {settings.BACKEND_DIR}")
print(f"Database URL: {settings.DATABASE_URL}")
print(f"RL Model Path: {settings.MODEL_PATH}")
print(f"Scaler Path: {settings.SCALER_PATH}")
print(f"AE Model Path: {settings.AE_MODEL_PATH}")
print(f"AE Threshold Path: {settings.AE_THRESHOLD_PATH}")
print(f"Broadcast URL: {settings.BROADCAST_URL}")
print(f"----------------------------")
