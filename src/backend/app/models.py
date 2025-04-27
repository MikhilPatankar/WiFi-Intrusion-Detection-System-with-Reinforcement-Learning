
# --- File: src/backend/app/models.py ---
# Change: Use List[Any] or list for features_data in API models

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union # Import Union if needed elsewhere
import datetime

# --- Request/Input Models ---
class NetworkStateFeatures(BaseModel):
    feature_vector: List[float] = Field(..., example=[0.5, 1.2, 0.0, 1.0, 55.0])

class EventLabel(BaseModel):
    event_uid: str = Field(..., example="evt_abc123xyz")
    human_label: str = Field(..., example="Confirmed Attack")

# --- Response/Output Models ---
class PredictionResult(BaseModel):
    prediction: int = Field(..., example=1)
    status: str = Field("success", example="success")

class LabelUpdateResult(BaseModel):
    status: str = Field("success", example="success")
    event_uid: str
    assigned_label: str

# --- Database Interaction Models (for API input/output) ---
class EventLogBase(BaseModel):
    """Base for event log data (used for creation and reading)."""
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    # Use a more specific type for API schema generation, e.g., list or List[Any]
    # This assumes the primary data structure passed via API is a list (like the feature vector)
    features_data: list = Field(..., example=[0.5, 1.2, 0.0, 1.0, 55.0])
    prediction: int
    initial_reward: Optional[float] = None
    human_label: Optional[str] = None
    label_timestamp: Optional[datetime.datetime] = None

class EventLogCreate(EventLogBase):
    """Model used only when creating an event log."""
    pass # Inherits fields, including the updated features_data type

class EventLogRead(EventLogBase):
    """Model used when reading event log data (includes DB fields)."""
    id: int # Primary key from DB
    event_uid: str # Unique identifier string
    # Inherits fields, including the updated features_data type

    class Config:
         orm_mode = True # Enable reading data from ORM objects

class EventLogReadList(BaseModel):
    """Response model for listing multiple event logs."""
    total_count: int
    events: List[EventLogRead]


class EventStats(BaseModel):
    """Model for returning event statistics."""
    total_events: int = 0
    anomaly_count: int = 0
    normal_count: int = 0
    labeled_count: int = 0
    unlabeled_count: int = 0
    # --- NEW Time-based Stats ---
    anomalies_last_hour: int = 0
    anomalies_last_24h: int = 0
    # --- End NEW ---
    label_distribution: Dict[str, int] = {}

# --- NEW Time Series Models ---
class TimeSeriesPoint(BaseModel):
    time_bucket: datetime.datetime # Start time of the bucket
    anomaly_count: int = 0
    normal_count: int = 0

class TimeSeriesData(BaseModel):
    interval: str # e.g., "hour", "day"
    data_points: List[TimeSeriesPoint] = []
# --- End NEW ---
