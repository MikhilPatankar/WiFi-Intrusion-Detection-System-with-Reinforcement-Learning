
# --- File: src/backend/app/schemas.py ---

from sqlmodel import SQLModel, Field, Column
from typing import Optional, List, Any, Union
import datetime
import uuid
import json
import logging
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSONType
from sqlalchemy.dialects.postgresql import JSONB as PGJSONBType
from sqlalchemy import TypeDecorator, TEXT, Index

# Custom JSON Type (same as before)
class JSONType(TypeDecorator): impl = TEXT; cache_ok = True; load_dialect_impl = lambda self, dialect: dialect.type_descriptor(PGJSONBType(astext_type=TEXT())) if dialect.name == 'postgresql' else dialect.type_descriptor(SQLiteJSONType()) if dialect.name == 'sqlite' else dialect.type_descriptor(TEXT()); process_bind_param = lambda self, value, dialect: json.dumps(value) if value is not None else value; process_result_value = lambda self, value, dialect: json.loads(value) if isinstance(value, str) else value if value is not None else value

# AttackTypes Table (ensure it exists)
class AttackTypes(SQLModel, table=True):
    type_id: Optional[int] = Field(default=None, primary_key=True)
    type_name: str = Field(index=True, unique=True, nullable=False, max_length=255)
    description: Optional[str] = Field(default=None)
    created_by: Optional[str] = Field(default=None, max_length=255)
    creation_timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    class Config: table_args = {"extend_existing": True}

# EventLog Table (Updated with hybrid fields)
class EventLog(SQLModel, table=True):
    # Core Fields
    id: Optional[int] = Field(default=None, primary_key=True)
    event_uid: str = Field(default_factory=lambda: f"evt_{uuid.uuid4()}", index=True, unique=True, nullable=False)
    timestamp: datetime.datetime = Field(index=True, nullable=False)
    features_data: Union[dict, list] = Field(sa_column=Column(JSONType), default=[]) # Original features

    # RL Prediction Field
    prediction: int = Field(nullable=False, description="Raw prediction from RL model (0=Normal, 1=Known Attack)")

    # --- NEW AE/Hybrid Fields ---
    ae_anomaly_flag: Optional[bool] = Field(default=None, index=True, description="Anomaly detected by AE (True/False)")
    reconstruction_error: Optional[float] = Field(default=None, description="AE reconstruction error value")
    final_status: Optional[str] = Field(default=None, index=True, max_length=50, description="Combined status (Normal, Known Attack, Potential Novelty)")
    # --- End NEW ---

    # Labeling Fields
    type_id: Optional[int] = Field(default=None, foreign_key="attacktypes.type_id")
    human_label: Optional[str] = Field(default=None, index=True, max_length=255)
    labeling_user_id: Optional[str] = Field(default=None, max_length=255)
    labeling_timestamp: Optional[datetime.datetime] = Field(default=None)

    # Other optional fields
    initial_reward: Optional[float] = Field(default=None)

    class Config:
        # Add indexes for potentially frequently queried columns
        table_args = (
            Index("ix_eventlog_timestamp", "timestamp"),
            Index("ix_eventlog_final_status", "final_status"),
            Index("ix_eventlog_ae_anomaly_flag", "ae_anomaly_flag"),
             {"extend_existing": True}
        )
