
# --- File: src/backend/app/routers/prediction.py ---
# No changes needed here

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
from ..models import NetworkStateFeatures, PredictionResult, EventLogCreate
from ..services import prediction_service, log_service
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_async_session
import datetime
import numpy as np

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=PredictionResult)
async def predict_network_state(features: NetworkStateFeatures, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_async_session)):
    logging.info("Received prediction request.")
    try:
        prediction = await prediction_service.predict_anomaly(features)
        logging.info(f"Prediction result: {prediction}")
        log_features = features.feature_vector
        if isinstance(log_features, np.ndarray): log_features = log_features.tolist()
        event_data_to_log = EventLogCreate(timestamp=datetime.datetime.now(), features_data=log_features, prediction=prediction, initial_reward=None)
        background_tasks.add_task(log_service.create_event_log, db, event_data_to_log)
        return PredictionResult(prediction=prediction, status="prediction_queued_for_logging")
    except RuntimeError as e: logging.error(f"Prediction service error: {e}", exc_info=True); raise HTTPException(status_code=503, detail=f"Prediction service unavailable: {e}")
    except ValueError as e: logging.error(f"Prediction input error: {e}", exc_info=True); raise HTTPException(status_code=422, detail=f"Invalid input features: {e}")
    except Exception as e: logging.error(f"Unexpected error during prediction: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")


