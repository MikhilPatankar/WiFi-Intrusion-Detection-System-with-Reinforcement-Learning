
# --- File: src/backend/main.py ---

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import uvicorn
import os # Import os

# Import configuration, routers, db setup, services
from app.core.config import settings
from app.routers import prediction, logging as logging_router
from app.db import create_db_and_tables
from app.services import prediction_service # Import the service module

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logging.info("Application startup...")

    # --- Load ML Model and Scaler ---
    logging.info("Loading RL model and feature scaler...")
    prediction_service.load_dependencies() # Call the combined loading function
    if prediction_service.ml_agent is None:
         logging.warning("ML Model loading failed. Prediction endpoint might not work.")
    if prediction_service.feature_scaler is None:
         logging.warning("Feature Scaler loading failed. Prediction endpoint might not work.")
         # Decide if the app should fail if scaler doesn't load
         # raise RuntimeError("Failed to load critical feature scaler.")

    # --- Create Database Tables ---
    logging.info("Initializing database...")
    await create_db_and_tables()
    logging.info("Database initialized.")

    yield # Application runs

    # --- Cleanup on shutdown ---
    logging.info("Application shutdown...")
    prediction_service.ml_agent = None
    prediction_service.feature_scaler = None
    logging.info("Resources cleaned up.")


# Create FastAPI app instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)

# Include API routers
app.include_router(prediction.router)
app.include_router(logging_router.router)

# Root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}!"}


# --- Main execution block ---
if __name__ == "__main__":
    logging.info("Starting Uvicorn server...")
    # Ensure data directory exists for SQLite
    if settings.DATABASE_URL.startswith("sqlite"):
        db_path = settings.DATABASE_URL.split("///")[1]
        db_dir = os.path.dirname(db_path)
        if db_dir: # Check if directory part exists
             os.makedirs(db_dir, exist_ok=True)
             logging.info(f"Ensured database directory exists: {db_dir}")

    # Ensure models directory exists relative to main.py for default paths
    # This assumes main.py is in src/backend/
    default_models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(default_models_dir, exist_ok=True)
    logging.info(f"Ensured default models directory exists: {default_models_dir}")


    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Set reload=False for production
        log_level="info"
    )

