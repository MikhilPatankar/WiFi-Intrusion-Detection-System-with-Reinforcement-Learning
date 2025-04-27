
# --- File: src/backend/app/services/log_service.py ---

import logging
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..schemas import EventLog
from ..models import EventLogCreate, EventLabel
from sqlalchemy import desc # For ordering results

async def create_event_log(db: AsyncSession, event_data: EventLogCreate) -> EventLog:
    """Creates a new event log entry in the database."""
    try:
        # Ensure features_data is serializable (convert NumPy arrays if necessary)
        if isinstance(event_data.features_data, np.ndarray):
            event_data.features_data = event_data.features_data.tolist()
        elif isinstance(event_data.features_data, list):
             # Check elements within the list
             event_data.features_data = [item.tolist() if isinstance(item, np.ndarray) else item for item in event_data.features_data]

        db_event = EventLog.from_orm(event_data)
        db.add(db_event)
        await db.commit()
        await db.refresh(db_event)
        logging.info(f"Created event log with UID: {db_event.event_uid}")
        return db_event
    except Exception as e:
        await db.rollback()
        logging.error(f"Error creating event log: {e}", exc_info=True)
        raise

async def get_event_log_by_uid(db: AsyncSession, event_uid: str) -> EventLog | None:
    """Retrieves a single event log by its unique ID."""
    try:
        result = await db.execute(select(EventLog).where(EventLog.event_uid == event_uid))
        event = result.scalars().first()
        return event
    except Exception as e:
        logging.error(f"Error retrieving event log {event_uid}: {e}", exc_info=True)
        return None

async def get_event_logs(db: AsyncSession, skip: int = 0, limit: int = 100, unlabeled_only: bool = False) -> List[EventLog]:
    """
    Retrieves a list of event logs, with optional filtering and pagination.
    """
    try:
        query = select(EventLog)
        if unlabeled_only:
            query = query.where(EventLog.human_label == None) # Filter for unlabeled

        query = query.order_by(desc(EventLog.timestamp)).offset(skip).limit(limit) # Order by most recent

        result = await db.execute(query)
        events = result.scalars().all()
        return events
    except Exception as e:
        logging.error(f"Error retrieving event logs: {e}", exc_info=True)
        return [] # Return empty list on error

async def get_event_logs_count(db: AsyncSession, unlabeled_only: bool = False) -> int:
    """Gets the total count of event logs, with optional filtering."""
    try:
        query = select(func.count(EventLog.id)) # Use func.count for efficiency
        if unlabeled_only:
            query = query.where(EventLog.human_label == None)

        result = await db.execute(query)
        count = result.scalar_one_or_none() # scalar_one_or_none handles potential None result
        return count if count is not None else 0
    except Exception as e:
        logging.error(f"Error getting event log count: {e}", exc_info=True)
        return 0


async def update_event_label(db: AsyncSession, label_data: EventLabel) -> EventLog | None:
    """Updates the human label for a specific event log."""
    try:
        db_event = await get_event_log_by_uid(db, label_data.event_uid)
        if db_event is None:
            logging.warning(f"Event log with UID {label_data.event_uid} not found for labeling.")
            return None

        db_event.human_label = label_data.human_label
        db_event.label_timestamp = datetime.datetime.now()
        db.add(db_event)
        await db.commit()
        await db.refresh(db_event)
        logging.info(f"Updated label for event UID {db_event.event_uid} to '{label_data.human_label}'")
        return db_event
    except Exception as e:
        await db.rollback()
        logging.error(f"Error updating label for event {label_data.event_uid}: {e}", exc_info=True)
        raise

