
# --- File: src/backend/app/models.py ---

from pydantic import BaseModel, Field
from typing import List, Optional, Any
import datetime

# --- Request/Input Models ---

class NetworkStateFeatures(BaseModel):
    """Input features for prediction."""
    # Assuming features are passed as a flat list/vector matching training order
    feature_vector: List[float] = Field(..., example=[0.5, 1.2, 0.0, 1.0, 55.0])

class EventLabel(BaseModel):
    """Input for labeling an event."""
    event_uid: str = Field(..., example="evt_abc123xyz", description="Unique identifier of the event to label")
    human_label: str = Field(..., example="Confirmed Attack", description="Label assigned by human")

# --- Response/Output Models ---

class PredictionResult(BaseModel):
    """Prediction output."""
    prediction: int = Field(..., example=1, description="0: Normal, 1: Anomaly")
    status: str = Field("success", example="success")
    # event_uid is not reliably available here due to background logging
    # event_uid: Optional[str] = Field(None, example="evt_12345", description="Unique ID assigned if logged")

class LabelUpdateResult(BaseModel):
    """Response after submitting a label."""
    status: str = Field("success", example="success")
    event_uid: str
    assigned_label: str

# --- Database Interaction Models ---

class EventLogBase(BaseModel):
    """Base for event log data (used for creation and reading)."""
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    features_data: dict | List[Any] # Can store raw features dict or processed vector
    prediction: int
    initial_reward: Optional[float] = None
    human_label: Optional[str] = None
    label_timestamp: Optional[datetime.datetime] = None

class EventLogCreate(EventLogBase):
    """Model used only when creating an event log."""
    # Inherits all fields from Base
    pass

class EventLogRead(EventLogBase):
    """Model used when reading event log data (includes DB fields)."""
    id: int # Primary key from DB
    event_uid: str # Unique identifier string

    class Config:
         orm_mode = True # Enable reading data from ORM objects

class EventLogReadList(BaseModel):
    """Response model for listing multiple event logs."""
    total_count: int
    events: List[EventLogRead]

