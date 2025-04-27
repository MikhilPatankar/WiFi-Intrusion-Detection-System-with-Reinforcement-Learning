
# --- File: src/backend/app/schemas.py ---

from sqlmodel import SQLModel, Field, Column
from typing import Optional, List, Any
import datetime
import uuid
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSONType
from sqlalchemy.dialects.postgresql import JSONB as PGJSONBType
from sqlalchemy import TypeDecorator, TEXT
import json # Import json for TypeDecorator

# Custom JSON TypeDecorator for cross-database compatibility (stores as JSON string)
class JSONType(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect):
        # Use native JSON types if available, otherwise fallback to TEXT
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PGJSONBType(astext_type=TEXT())) # Ensure TEXT fallback
        elif dialect.name == 'sqlite':
            return dialect.type_descriptor(SQLiteJSONType())
        else:
            return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value, dialect):
        # Convert Python dict/list to JSON string before saving
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        # Convert JSON string from DB back to Python dict/list
        if value is not None:
            try:
                # Handle potential double-encoding if TEXT was used
                if isinstance(value, str):
                    return json.loads(value)
                return value # Assume already decoded if not string (e.g., native JSON)
            except (json.JSONDecodeError, TypeError):
                 # Log error or return raw value if decoding fails
                 logging.warning(f"Failed to decode JSON from DB: {value}")
                 return value # Return raw value on error
        return value


class EventLog(SQLModel, table=True):
    """Database model for storing event logs."""
    id: Optional[int] = Field(default=None, primary_key=True)
    event_uid: str = Field(default_factory=lambda: f"evt_{uuid.uuid4()}", index=True, unique=True, nullable=False)
    timestamp: datetime.datetime = Field(index=True, nullable=False)
    # Use the custom JSONType decorator
    features_data: Any = Field(sa_column=Column(JSONType), default={})
    prediction: int = Field(nullable=False)
    initial_reward: Optional[float] = Field(default=None)
    human_label: Optional[str] = Field(default=None, index=True)
    label_timestamp: Optional[datetime.datetime] = Field(default=None)

    class Config:
        table_args = {"extend_existing": True}

