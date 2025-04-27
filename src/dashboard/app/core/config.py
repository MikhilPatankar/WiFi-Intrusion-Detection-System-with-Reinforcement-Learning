# src/dashboard/app/core/config.py

import os
from pydantic_settings import BaseSettings
from pathlib import Path

# Determine base directory for the dashboard app
# Assumes config.py is in src/dashboard/app/core
try:
    DASHBOARD_APP_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
     DASHBOARD_APP_DIR = Path(".").resolve() # Fallback

DEFAULT_TEMPLATES_DIR = DASHBOARD_APP_DIR / "templates"
DEFAULT_STATIC_DIR = DASHBOARD_APP_DIR / "static"

class DashboardSettings(BaseSettings):
    """Dashboard Application Settings"""
    APP_NAME: str = "WIDS Dashboard"
    # --- Authentication Settings ---
    # IMPORTANT: Use a strong, randomly generated secret key
    AUTH_SECRET_KEY: str = os.getenv("DASHBOARD_AUTH_SECRET_KEY", "change-this-super-secret-dashboard-key")
    AUTH_ALGORITHM: str = "HS256"
    AUTH_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 # Token validity duration

    # --- Backend API Connection ---
    # URL of the separate WIDS Backend API
    WIDS_API_BASE_URL: str = os.getenv("WIDS_API_BASE_URL", "http://127.0.0.1:8000")

    # --- Directory Settings ---
    TEMPLATES_DIR: str = str(DEFAULT_TEMPLATES_DIR)
    STATIC_DIR: str = str(DEFAULT_STATIC_DIR)

    class Config:
        # Load from .env file if present (e.g., place .env in src/dashboard/)
        # env_file = ".env"
        # env_file_encoding = 'utf-8'
        pass

settings = DashboardSettings()

print(f"Dashboard Tempates Dir: {settings.TEMPLATES_DIR}")
print(f"Dashboard Static Dir: {settings.STATIC_DIR}")
print(f"Target WIDS API URL: {settings.WIDS_API_BASE_URL}")

