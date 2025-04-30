
# --- File: src/backend/app/models.py ---

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import datetime

# --- API Request Models ---
class NetworkStateFeatures(BaseModel):
    """Input features for prediction."""
    feature_vector: List[float] = Field(..., example=[0.5, 1.2, 0.0, 1.0, 55.0])

class EventLabel(BaseModel):
    """Input for labeling an event."""
    event_uid: str = Field(..., example="evt_abc123xyz")
    human_label: str = Field(..., example="Confirmed Attack")

# --- API Response Models ---
class HybridPredictionResult(BaseModel):
    """Response from the hybrid prediction endpoint."""
    rl_prediction: int = Field(..., example=1, description="Raw prediction from RL model (0: Normal, 1: Known Attack)")
    ae_anomaly_flag: Optional[bool] = Field(None, example=False, description="Flag indicating if AE detected anomaly/novelty (True/False)")
    reconstruction_error: Optional[float] = Field(None, example=0.00123, description="AE reconstruction error value")
    final_status: str = Field(..., example="Known Attack", description="Combined status ('Normal', 'Known Attack', 'Potential Novelty', 'Error')")

class LabelUpdateResult(BaseModel):
    """Response after submitting a label."""
    status: str = Field("success", example="success")
    event_uid: str
    assigned_label: str

class AttackTypeRead(BaseModel):
    """Response model for listing attack types."""
    type_id: int
    type_name: str
    description: Optional[str] = None
    is_active: bool
    created_by: Optional[str] = None
    creation_timestamp: datetime.datetime
    class Config: orm_mode = True

class EventStats(BaseModel):
    """Response model for event statistics."""
    total_events: int = 0; anomaly_count: int = 0; normal_count: int = 0; labeled_count: int = 0; unlabeled_count: int = 0; anomalies_last_hour: int = 0; anomalies_last_24h: int = 0; label_distribution: Dict[str, int] = {}

class TimeSeriesPoint(BaseModel):
    """Data point for time series."""
    time_bucket: datetime.datetime; anomaly_count: int = 0; normal_count: int = 0

class TimeSeriesData(BaseModel):
    """Response model for time series data."""
    interval: str; data_points: List[TimeSeriesPoint] = []


# --- Data Transfer/Validation Models (Internal/Service Layer/SSE) ---
class EventLogBase(BaseModel):
    """Base model for event log data, including hybrid fields."""
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    features_data: list # Original unscaled features
    prediction: int # RL Prediction
    initial_reward: Optional[float] = None
    # AE/Hybrid Fields
    ae_anomaly_flag: Optional[bool] = None
    reconstruction_error: Optional[float] = None
    final_status: Optional[str] = None
    # Labeling Fields
    type_id: Optional[int] = None
    human_label: Optional[str] = None
    labeling_user_id: Optional[str] = None
    labeling_timestamp: Optional[datetime.datetime] = None

class EventLogCreate(EventLogBase):
    """Model used when creating an event log via API/background task."""
    pass

class EventLogRead(EventLogBase):
    """Model used when reading event log data (e.g., for SSE, GET /events)."""
    id: int # Primary key from DB
    event_uid: str # Unique identifier string
    class Config: orm_mode = True

class EventLogReadList(BaseModel):
    """Response model for listing multiple event logs."""
    total_count: int
    events: List[EventLogRead] # Use the updated EventLogRead

