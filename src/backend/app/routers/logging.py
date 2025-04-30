
# --- File: src/backend/app/routers/logging.py ---

from fastapi import APIRouter, HTTPException, Depends, Query, Request
import logging; from typing import List, AsyncGenerator; import asyncio; import json
from sse_starlette.sse import EventSourceResponse
# Ensure EventLogRead includes the new fields for SSE
from ..models import EventLabel, LabelUpdateResult, EventLogRead, EventLogReadList, EventStats, TimeSeriesData, AttackTypeRead
from ..services import log_service; from sqlalchemy.ext.asyncio import AsyncSession
from ..db import get_async_session; from sqlalchemy import func
from ..broadcast import broadcast
from ..services.log_service import EVENT_CHANNEL

log = logging.getLogger(__name__)
router_attack_types = APIRouter(prefix="/attack_types", tags=["Attack Types"])
router = APIRouter(prefix="/events", tags=["Events & Labeling"])

@router_attack_types.get("/", response_model=List[AttackTypeRead])
async def read_attack_types(include_inactive: bool = Query(False), db: AsyncSession = Depends(get_async_session)):
    """Retrieve a list of all defined attack classifications."""
    log.info(f"Request for attack types (inactive={include_inactive}).")
    try: types = await log_service.get_attack_types(db, include_inactive=include_inactive); return types
    except Exception as e: log.error(f"Failed get attack types: {e}"); raise HTTPException(status_code=500, detail="Failed.")

@router.get("/stream")
async def event_stream(request: Request):
    """SSE endpoint to stream newly created events (now includes hybrid fields)."""
    log.info("Client connected to SSE event stream.")
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async with broadcast.subscribe(channel=EVENT_CHANNEL) as subscriber:
                async for event in subscriber:
                    if await request.is_disconnected(): log.info("SSE client disconnected."); break
                    # Message is already JSON string of EventLogRead (which has hybrid fields)
                    yield f"data: {event.message}\n\n"
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError: log.info("SSE generator cancelled.")
        except Exception as e: log.error(f"Error in SSE generator: {e}", exc_info=True)
        finally: log.info("SSE event generator finished.")
    return EventSourceResponse(event_generator())

@router.get("/stats", response_model=EventStats)
async def get_stats(db: AsyncSession = Depends(get_async_session)):
    """Retrieve aggregated statistics about logged events."""
    log.info("Request for event statistics.")
    try: stats = await log_service.get_event_statistics(db); return stats
    except Exception as e: log.error(f"Failed get stats: {e}"); raise HTTPException(status_code=500, detail="Failed.")

@router.get("/timeseries", response_model=TimeSeriesData)
async def get_timeseries_data(interval: str = Query("hour", enum=["hour", "day"]), hours_ago: int = Query(24, ge=1, le=168), db: AsyncSession = Depends(get_async_session)):
    """Retrieve time series data for event counts."""
    log.info(f"Request for time series: interval={interval}, hours={hours_ago}")
    try: timeseries = await log_service.get_event_timeseries(db, interval=interval, hours_ago=hours_ago); return timeseries
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: log.error(f"Failed get timeseries: {e}"); raise HTTPException(status_code=500, detail="Failed.")

@router.get("/", response_model=EventLogReadList)
async def read_events(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500), unlabeled_only: bool = Query(False), db: AsyncSession = Depends(get_async_session)):
    """Retrieve a paginated list of logged events."""
    log.info(f"Reading events: skip={skip}, limit={limit}, unlabeled={unlabeled_only}")
    try: events = await log_service.get_event_logs(db, skip=skip, limit=limit, unlabeled_only=unlabeled_only); total_count = await log_service.get_event_logs_count(db, unlabeled_only=unlabeled_only); return EventLogReadList(total_count=total_count, events=events)
    except Exception as e: log.error(f"Error reading events: {e}"); raise HTTPException(status_code=500, detail="Failed.")

@router.get("/{event_uid}", response_model=EventLogRead)
async def read_event(event_uid: str, db: AsyncSession = Depends(get_async_session)):
    """Retrieve details for a specific event by its unique ID."""
    log.info(f"Reading event: {event_uid}")
    db_event = await log_service.get_event_log_by_uid(db, event_uid)
    if db_event is None: raise HTTPException(status_code=404, detail=f"Event UID '{event_uid}' not found.")
    return db_event

@router.post("/label", response_model=LabelUpdateResult)
async def label_event(label_data: EventLabel, db: AsyncSession = Depends(get_async_session)):
     """Submit a human label for a specific event."""
     log.info(f"Labeling event: {label_data.event_uid}")
     try:
         updated_event = await log_service.update_event_label(db, label_data)
         if updated_event is None: raise HTTPException(status_code=404, detail=f"Event UID '{label_data.event_uid}' not found for labeling.")
         return LabelUpdateResult(status="success", event_uid=updated_event.event_uid, assigned_label=updated_event.human_label)
     except Exception as e: log.error(f"Error labeling {label_data.event_uid}: {e}"); raise HTTPException(status_code=500, detail="Failed.")
