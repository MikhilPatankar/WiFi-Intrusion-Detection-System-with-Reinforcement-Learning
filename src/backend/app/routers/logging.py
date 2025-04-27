
# --- File: src/backend/app/routers/logging.py ---

from fastapi import APIRouter, HTTPException, Depends, Query # Import Query for pagination
import logging
from typing import List # For response model typing
from ..models import EventLabel, LabelUpdateResult, EventLogRead, EventLogReadList
from ..services import log_service
from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_async_session
from sqlalchemy import func # Import func for count

router = APIRouter(
    prefix="/events",
    tags=["Events & Labeling"],
)

@router.get("/", response_model=EventLogReadList)
async def read_events(
    skip: int = Query(0, ge=0, description="Number of events to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of events to return"),
    unlabeled_only: bool = Query(False, description="Filter for events without a human label"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Retrieve a list of logged events, ordered by most recent timestamp.
    Supports pagination and filtering for unlabeled events.
    """
    logging.info(f"Received request to read events: skip={skip}, limit={limit}, unlabeled_only={unlabeled_only}")
    try:
        events = await log_service.get_event_logs(db, skip=skip, limit=limit, unlabeled_only=unlabeled_only)
        # For accurate total count with filtering, we need another query
        total_count = await log_service.get_event_logs_count(db, unlabeled_only=unlabeled_only)

        # Convert ORM objects to Pydantic models for the response
        # This happens automatically if response_model is set correctly and models use Config.orm_mode=True
        return EventLogReadList(total_count=total_count, events=events)
    except Exception as e:
        logging.error(f"Error reading events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while retrieving events.")


@router.get("/{event_uid}", response_model=EventLogRead)
async def read_event(
    event_uid: str,
    db: AsyncSession = Depends(get_async_session)
):
    """Retrieve details for a specific event by its unique ID."""
    logging.info(f"Received request for event UID: {event_uid}")
    db_event = await log_service.get_event_log_by_uid(db, event_uid)
    if db_event is None:
        logging.warning(f"Event UID {event_uid} not found.")
        raise HTTPException(status_code=404, detail=f"Event with UID {event_uid} not found")
    return db_event


@router.post("/label", response_model=LabelUpdateResult)
async def label_event(
    label_data: EventLabel,
    db: AsyncSession = Depends(get_async_session)
):
    """Receives a human label for a specific event and updates the database."""
    logging.info(f"Received labeling request for event UID: {label_data.event_uid}")
    try:
        updated_event = await log_service.update_event_label(db, label_data)
        if updated_event is None:
            logging.warning(f"Failed to label event: UID {label_data.event_uid} not found.")
            raise HTTPException(status_code=404, detail=f"Event with UID {label_data.event_uid} not found.")

        logging.info(f"Successfully labeled event UID: {updated_event.event_uid}")
        return LabelUpdateResult(
            status="success",
            event_uid=updated_event.event_uid,
            assigned_label=updated_event.human_label
        )
    except HTTPException:
         raise
    except Exception as e:
        logging.error(f"Unexpected error during labeling event {label_data.event_uid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during labeling: {e}")

