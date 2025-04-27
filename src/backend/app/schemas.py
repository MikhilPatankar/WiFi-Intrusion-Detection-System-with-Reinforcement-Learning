
# --- File: src/backend/app/schemas.py ---
# Change: Use Union[dict, list] for storage flexibility in DB model

from sqlmodel import SQLModel, Field, Column
from typing import Optional, List, Any, Union # Import Union
import datetime
import uuid
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSONType
from sqlalchemy.dialects.postgresql import JSONB as PGJSONBType
from sqlalchemy import TypeDecorator, TEXT
import json
import logging

class JSONType(TypeDecorator):
    impl = TEXT
    cache_ok = True
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql': return dialect.type_descriptor(PGJSONBType(astext_type=TEXT()))
        elif dialect.name == 'sqlite': return dialect.type_descriptor(SQLiteJSONType())
        else: return dialect.type_descriptor(TEXT())
    def process_bind_param(self, value, dialect):
        if value is not None: return json.dumps(value)
        return value
    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                if isinstance(value, str): return json.loads(value)
                return value
            except (json.JSONDecodeError, TypeError):
                 logging.warning(f"Failed to decode JSON from DB: {value}")
                 return value
        return value

class EventLog(SQLModel, table=True):
    """Database model for storing event logs."""
    id: Optional[int] = Field(default=None, primary_key=True)
    event_uid: str = Field(default_factory=lambda: f"evt_{uuid.uuid4()}", index=True, unique=True, nullable=False)
    timestamp: datetime.datetime = Field(index=True, nullable=False)
    # Use Union[dict, list] here for DB flexibility, allowing storage of dicts or lists
    # The JSONType decorator handles serialization/deserialization
    features_data: Union[dict, list] = Field(sa_column=Column(JSONType), default=[]) # Default to empty list
    prediction: int = Field(nullable=False)
    initial_reward: Optional[float] = Field(default=None)
    human_label: Optional[str] = Field(default=None, index=True)
    label_timestamp: Optional[datetime.datetime] = Field(default=None)
    class Config: table_args = {"extend_existing": True}
