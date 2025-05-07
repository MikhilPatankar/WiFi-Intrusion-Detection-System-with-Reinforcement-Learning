from fastapi import FastAPI; from contextlib import asynccontextmanager; import logging; import uvicorn; import os
from fastapi.middleware.cors import CORSMiddleware; from app.core.config import settings
# Import all routers
from app.routers import prediction, logging as logging_router, attack_types as attack_types_router
from app.db import create_db_and_tables
from app.services import prediction_service # Import prediction service
from app.broadcast import broadcast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Startup...")

    # Ensure directories exist
    db_dir=os.path.dirname(settings.DATABASE_URL.split("///")[-1])
    model_dir=os.path.dirname(settings.MODEL_PATH)
    scaler_dir=os.path.dirname(settings.SCALER_PATH)
    ae_model_dir = os.path.dirname(settings.AE_MODEL_PATH)
    ae_thresh_dir = os.path.dirname(settings.AE_THRESHOLD_PATH)

    if settings.DATABASE_URL.startswith("sqlite") and db_dir: os.makedirs(db_dir, exist_ok=True)
    if model_dir: os.makedirs(model_dir, exist_ok=True)
    if scaler_dir: os.makedirs(scaler_dir, exist_ok=True)
    if ae_model_dir: os.makedirs(ae_model_dir, exist_ok=True)
    if ae_thresh_dir: os.makedirs(ae_thresh_dir, exist_ok=True)

    # Load Dependencies (RL, Scaler, AE, Threshold)
    all_loaded = prediction_service.load_dependencies()
    if not all_loaded: logging.critical("CRITICAL ERROR: Failed to load essential models/scaler.")

    await create_db_and_tables(); # Init DB
    await broadcast.connect(); logging.info("Broadcaster connected.") # Connect Broadcaster

    logging.info("Startup complete.")
    yield
    logging.info("Shutdown...")

    await broadcast.disconnect(); logging.info("Broadcaster disconnected.") # Disconnect Broadcaster
    prediction_service.ml_agent = None; prediction_service.feature_scaler = None; prediction_service.ae_model = None; prediction_service.ae_threshold = None; # Clear refs
    logging.info("Cleanup complete.")

# --- App Creation and Routing ---
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, description=settings.DESCRIPTION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(prediction.router)
app.include_router(logging_router.router)
app.include_router(attack_types_router.router)


@app.get("/", tags=["Root"])
async def read_root(): return {"message": f"{settings.PROJECT_NAME} is running!"}
if __name__ == "__main__": 
    logging.info("Starting Uvicorn server for WIDS API...")
    uvicorn.run( "main:app", host="0.0.0.0", port=80, reload=True, log_level="info")
