
# --- File: src/backend/app/routers/prediction.py ---

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
from ..models import NetworkStateFeatures, PredictionResult, EventLogCreate
from ..services import prediction_service, log_service
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_async_session
import datetime
import numpy as np # Import numpy

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

@router.post("/", response_model=PredictionResult)
async def predict_network_state(
    features: NetworkStateFeatures,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session)
):
    """
    Receives network state features, predicts if it's an anomaly using the RL agent,
    and logs the event in the background.
    """
    logging.info("Received prediction request.")
    try:
        # 1. Get prediction (includes preprocessing now)
        prediction = await prediction_service.predict_anomaly(features)
        logging.info(f"Prediction result: {prediction}")

        # 2. Prepare data for logging
        # Ensure features_data is JSON serializable (list in this case)
        log_features = features.feature_vector
        if isinstance(log_features, np.ndarray):
            log_features = log_features.tolist() # Convert numpy array if needed

        event_data_to_log = EventLogCreate(
            timestamp=datetime.datetime.now(),
            features_data=log_features, # Log the original input vector
            prediction=prediction,
            initial_reward=None
        )

        # 3. Add logging to background tasks
        background_tasks.add_task(log_service.create_event_log, db, event_data_to_log)

        return PredictionResult(prediction=prediction, status="prediction_queued_for_logging")

    except RuntimeError as e:
        logging.error(f"Prediction service error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Prediction service unavailable: {e}")
    except ValueError as e:
         logging.error(f"Prediction input error: {e}", exc_info=True)
         raise HTTPException(status_code=422, detail=f"Invalid input features: {e}") # Unprocessable Entity
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")


