
# --- File: src/backend/app/services/log_service.py ---
# No changes needed

import logging
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..schemas import EventLog
from ..models import EventLogCreate, EventLabel
from sqlalchemy import desc, func
from typing import List
import numpy as np
import datetime # Import datetime

async def create_event_log(db: AsyncSession, event_data: EventLogCreate) -> EventLog:
    try:
        log_features = event_data.features_data # Already typed as 'list' in EventLogCreate
        # Create EventLog instance using the input data
        # The features_data (list) will be serialized by JSONType
        db_event = EventLog(**event_data.dict())
        db.add(db_event); await db.commit(); await db.refresh(db_event)
        logging.info(f"Created event log with UID: {db_event.event_uid}")
        return db_event
    except Exception as e: await db.rollback(); logging.error(f"Error creating event log: {e}", exc_info=True); raise

async def get_event_log_by_uid(db: AsyncSession, event_uid: str) -> EventLog | None:
    try: result = await db.execute(select(EventLog).where(EventLog.event_uid == event_uid)); return result.scalars().first()
    except Exception as e: logging.error(f"Error retrieving event log {event_uid}: {e}", exc_info=True); return None

async def get_event_logs(db: AsyncSession, skip: int = 0, limit: int = 100, unlabeled_only: bool = False) -> List[EventLog]:
    try:
        query = select(EventLog)
        if unlabeled_only: query = query.where(EventLog.human_label == None)
        query = query.order_by(desc(EventLog.timestamp)).offset(skip).limit(limit)
        result = await db.execute(query); return result.scalars().all()
    except Exception as e: logging.error(f"Error retrieving event logs: {e}", exc_info=True); return []

async def get_event_logs_count(db: AsyncSession, unlabeled_only: bool = False) -> int:
    try:
        query = select(func.count(EventLog.id))
        if unlabeled_only: query = query.where(EventLog.human_label == None)
        result = await db.execute(query); count = result.scalar_one_or_none()
        return count if count is not None else 0
    except Exception as e: logging.error(f"Error getting event log count: {e}", exc_info=True); return 0

async def update_event_label(db: AsyncSession, label_data: EventLabel) -> EventLog | None:
    try:
        db_event = await get_event_log_by_uid(db, label_data.event_uid)
        if db_event is None: logging.warning(f"Event log UID {label_data.event_uid} not found for labeling."); return None
        db_event.human_label = label_data.human_label; db_event.label_timestamp = datetime.datetime.now()
        db.add(db_event); await db.commit(); await db.refresh(db_event)
        logging.info(f"Updated label for event UID {db_event.event_uid} to '{label_data.human_label}'")
        return db_event
    except Exception as e: await db.rollback(); logging.error(f"Error updating label for event {label_data.event_uid}: {e}", exc_info=True); raise
