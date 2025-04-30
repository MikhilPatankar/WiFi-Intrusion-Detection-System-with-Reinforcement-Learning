
# --- File: src/backend/app/routers/prediction.py ---

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
from typing import Any, Dict

# Updated models needed
from ..models import NetworkStateFeatures, EventLogCreate, HybridPredictionResult # Use new response/create models

from ..services import prediction_service, log_service
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_async_session
import datetime
import numpy as np

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Updated Background Task Function (uses EventLogCreate which now has hybrid fields)
async def background_log_event(db: AsyncSession, event_data_dict: Dict[str, Any]):
    """Background task to log event details using the log_service."""
    try:
        # Create Pydantic model from dict for validation before passing to service
        event_log_create = EventLogCreate(**event_data_dict)
        await log_service.create_event_log(db, event_log_create)
    except Exception as e:
        log.error(f"Background logging/publishing failed: {e}", exc_info=True)

# Updated Predict Endpoint
@router.post("/", response_model=HybridPredictionResult)
async def predict_network_state_hybrid(
    features: NetworkStateFeatures,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session)
):
    """Receives features, performs hybrid prediction, logs, returns combined status."""
    log.info("Hybrid prediction request received.")
    try:
        prediction_results = await prediction_service.run_hybrid_prediction(features)
        event_data_to_log = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "features_data": features.feature_vector, # Original unscaled features
            "prediction": prediction_results["rl_prediction"],
            "ae_anomaly_flag": prediction_results["ae_anomaly_flag"],
            "reconstruction_error": prediction_results["reconstruction_error"],
            "final_status": prediction_results["final_status"],
            "initial_reward": None
        }
        background_tasks.add_task(background_log_event, db, event_data_to_log)
        return HybridPredictionResult(**prediction_results) # Pass dict directly

    except RuntimeError as e: log.error(f"Service error: {e}"); raise HTTPException(status_code=503, detail=f"{e}")
    except ValueError as e: log.error(f"Input error: {e}"); raise HTTPException(status_code=422, detail=f"{e}")
    except Exception as e: log.error(f"Prediction error: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Prediction failed.")
