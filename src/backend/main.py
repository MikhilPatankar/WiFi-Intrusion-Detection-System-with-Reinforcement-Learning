
# --- File: src/backend/main.py ---

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import prediction, logging as logging_router
from app.db import create_db_and_tables
from app.services import prediction_service
from app.broadcast import broadcast # Import broadcaster


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PORT: int = int(os.getenv("PORT", 8000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup...")
    # --- Ensure directories exist ---
    # Use paths from settings which are now more robustly calculated
    db_url_path = settings.DATABASE_URL.split("///")[-1] # Get path part of sqlite url
    db_dir = os.path.dirname(db_url_path)
    model_dir = os.path.dirname(settings.MODEL_PATH)
    scaler_dir = os.path.dirname(settings.SCALER_PATH)

    if settings.DATABASE_URL.startswith("sqlite") and db_dir:
        os.makedirs(db_dir, exist_ok=True)
        logging.info(f"Ensured database directory exists: {db_dir}")
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        logging.info(f"Ensured model directory exists: {model_dir}")
    if scaler_dir:
         os.makedirs(scaler_dir, exist_ok=True)
         logging.info(f"Ensured scaler directory exists: {scaler_dir}")

    # --- Load Dependencies ---
    logging.info("Loading RL model and feature scaler...")
    prediction_service.load_dependencies()
    if prediction_service.ml_agent is None: logging.warning("ML Model loading failed.")
    if prediction_service.feature_scaler is None: logging.warning("Feature Scaler loading failed.")

    logging.info("Initializing database...")
    await create_db_and_tables()
    logging.info("Database initialized.")

    await broadcast.connect()
    logging.info("Broadcaster connected.")

    logging.info("Startup complete.")
    yield
    logging.info("Shutdown...");

    await broadcast.disconnect()
    logging.info("Broadcaster disconnected.")

    logging.info("Application shutdown...")
    prediction_service.ml_agent = None
    prediction_service.feature_scaler = None
    logging.info("Resources cleaned up.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)

origins = ["http://localhost", "http://localhost:8080", "http://127.0.0.1:8050", "null"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(prediction.router)
app.include_router(logging_router.router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"{settings.PROJECT_NAME} is running!"}

# --- Main execution block ---
if __name__ == "__main__":
    logging.info("Starting Uvicorn server...")
    # Directory checks are now handled in lifespan
    uvicorn.run( "main:app", host="0.0.0.0", port=PORT, reload=True, log_level="info")

