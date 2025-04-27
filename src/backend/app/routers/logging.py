
# --- File: src/backend/app/routers/logging.py ---
# No changes needed here

from fastapi import APIRouter, HTTPException, Depends, Query
import logging
from typing import List
from ..models import EventLabel, LabelUpdateResult, EventLogRead, EventLogReadList
from ..services import log_service
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_async_session
from sqlalchemy import func

router = APIRouter(prefix="/events", tags=["Events & Labeling"])

@router.get("/", response_model=EventLogReadList)
async def read_events(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500), unlabeled_only: bool = Query(False), db: AsyncSession = Depends(get_async_session)):
    logging.info(f"Reading events: skip={skip}, limit={limit}, unlabeled={unlabeled_only}")
    try:
        events = await log_service.get_event_logs(db, skip=skip, limit=limit, unlabeled_only=unlabeled_only)
        total_count = await log_service.get_event_logs_count(db, unlabeled_only=unlabeled_only)
        return EventLogReadList(total_count=total_count, events=events)
    except Exception as e: logging.error(f"Error reading events: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Failed to retrieve events.")

@router.get("/{event_uid}", response_model=EventLogRead)
async def read_event(event_uid: str, db: AsyncSession = Depends(get_async_session)):
    logging.info(f"Reading event UID: {event_uid}")
    db_event = await log_service.get_event_log_by_uid(db, event_uid)
    if db_event is None: logging.warning(f"Event UID {event_uid} not found."); raise HTTPException(status_code=404, detail=f"Event UID {event_uid} not found")
    return db_event

@router.post("/label", response_model=LabelUpdateResult)
async def label_event(label_data: EventLabel, db: AsyncSession = Depends(get_async_session)):
    logging.info(f"Labeling event UID: {label_data.event_uid}")
    try:
        updated_event = await log_service.update_event_label(db, label_data)
        if updated_event is None: logging.warning(f"Failed to label event: UID {label_data.event_uid} not found."); raise HTTPException(status_code=404, detail=f"Event UID {label_data.event_uid} not found.")
        logging.info(f"Successfully labeled event UID: {updated_event.event_uid}")
        return LabelUpdateResult(status="success", event_uid=updated_event.event_uid, assigned_label=updated_event.human_label)
    except HTTPException: raise
    except Exception as e: logging.error(f"Error labeling event {label_data.event_uid}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Failed to label event.")
