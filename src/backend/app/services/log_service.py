
import logging; from sqlalchemy.future import select; from sqlalchemy.ext.asyncio import AsyncSession
from ..schemas import EventLog, AttackTypes
from ..models import EventLogCreate, EventLabel, EventStats, TimeSeriesPoint, TimeSeriesData, EventLogRead, AttackTypeRead
from sqlalchemy import desc, func, case, text, and_
from typing import List, Dict, Any
import numpy as np; import datetime; from datetime import timedelta
from ..broadcast import broadcast; import json

log = logging.getLogger(__name__)
EVENT_CHANNEL = "wids-events"

async def create_event_log(db: AsyncSession, event_data: EventLogCreate) -> EventLog:
    """Creates log entry with hybrid fields and publishes it."""
    try:
        # Create DB model instance from the Pydantic model's dict
        db_event = EventLog(**event_data.dict(exclude_unset=True))
        db.add(db_event); await db.commit(); await db.refresh(db_event)
        log.info(f"Created event log: {db_event.event_uid} with status: {db_event.final_status}")

        # Publish the new event (with hybrid fields)
        try:
            # Convert ORM model back to the Pydantic Read model for consistent SSE format
            event_to_publish = EventLogRead.from_orm(db_event) # EventLogRead now includes hybrid fields
            await broadcast.publish(channel=EVENT_CHANNEL, message=event_to_publish.json())
            log.info(f"Published event {db_event.event_uid} to channel '{EVENT_CHANNEL}'")
        except Exception as pub_err:
            log.error(f"Failed to publish event {db_event.event_uid}: {pub_err}", exc_info=True)

        return db_event
    except Exception as e: await db.rollback(); log.error(f"Error creating event log: {e}", exc_info=True); raise

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
        db_event.human_label = label_data.human_label; db_event.labeling_timestamp = datetime.datetime.now()
        db.add(db_event); await db.commit(); await db.refresh(db_event)
        logging.info(f"Updated label for event UID {db_event.event_uid} to '{label_data.human_label}'")

        # --- ADDED: Update/Add Attack Type ---
        await update_attack_types(db, label_data)  # Call the new function

        return db_event
    except Exception as e: await db.rollback(); logging.error(f"Error updating label for event {label_data.event_uid}: {e}", exc_info=True); raise

async def update_attack_types(db: AsyncSession, label_data: EventLabel) -> bool:
    """
    Checks if a label exists as an attack type. If not, adds it as a new attack type.

    Args:
        db (AsyncSession): Database session.
        label_data (EventLabel): Contains the event UID and the human label.

    Returns:
        bool: True if a new attack type was added, False otherwise (including errors).
    """
    try:
        # 1. Get all attack types (including inactive for now, to check for duplicates)
        existing_types = await get_attack_types(db, include_inactive=True)

        # Check if the label (lowercased) already exists, ignoring case for comparison
        label_lower = label_data.human_label.lower()
        if any(t.type_name.lower() == label_lower for t in existing_types):
            logging.info(f"Label '{label_data.human_label}' (case-insensitive) already exists as an attack type. No update needed.")
            return False  # No new type added

        # 2. If it doesn't exist, prepare and add a new one
        logging.info(f"Label '{label_data.human_label}' is a new attack type. Adding to database.")

        # Generate a new type_id (find max and increment)
        max_id = max((t.type_id for t in existing_types), default=0)
        new_type_id = max_id + 1

        # Create a new AttackTypes object (using lowercase for consistency in DB)
        new_type = AttackTypes(
            type_id=new_type_id,
            type_name=label_data.human_label,  # Store the label as provided, case-sensitive
            description=f"Automatically added attack type based on human label: {label_data.human_label}",
            is_active=True,
            created_by="system", # Or get from current user if available
            creation_timestamp=datetime.datetime.now(datetime.timezone.utc)
        )

        # Add and commit to the database
        db.add(new_type)
        await db.commit()
        await db.refresh(new_type)

        logging.info(f"Added new attack type: '{new_type.type_name}' (ID: {new_type.type_id})")
        return True  # New type added

    except Exception as e:
        await db.rollback()  # Rollback in case of errors
        logging.error(f"Error while updating/adding attack types: {e}", exc_info=True)
        return False # Indicate failure


async def get_event_statistics(db: AsyncSession) -> EventStats:
    """Calculates various statistics about the logged events, including recent anomalies."""
    try:
        now = datetime.datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        twenty_four_hours_ago = now - timedelta(hours=24)

        # Use case for conditional aggregation within a single query
        stmt = select(
            func.count(EventLog.id).label("total_events"),
            func.sum(case((EventLog.ae_anomaly_flag == 1, 1), else_=0)).label("anomaly_count"),
            func.sum(case((EventLog.prediction == 0, 1), else_=0)).label("normal_count"),
            func.sum(case((and_(EventLog.ae_anomaly_flag == 1, EventLog.human_label != None), 1), else_=0)).label("labeled_count"),
            func.sum(case((and_(EventLog.ae_anomaly_flag == 1, EventLog.human_label == None), 1), else_=0)).label("unlabeled_count"),
            # --- NEW Time-based Counts ---
            func.sum(case((and_(EventLog.ae_anomaly_flag == 1, EventLog.timestamp >= one_hour_ago), 1), else_=0)).label("anomalies_last_hour"),
            func.sum(case((and_(EventLog.ae_anomaly_flag == 1, EventLog.timestamp >= twenty_four_hours_ago), 1),else_=0)).label("anomalies_last_24h")
            # --- End NEW ---
        )
        result = await db.execute(stmt)
        counts = result.mappings().first()

        # Query for label distribution (same as before)
        label_dist_stmt = select(EventLog.human_label, func.count(EventLog.id).label("count")).where(EventLog.human_label != None).group_by(EventLog.human_label)
        label_result = await db.execute(label_dist_stmt)
        label_distribution = {row.human_label: row.count for row in label_result.mappings().all()}

        stats = EventStats(
            total_events=int(counts.get("total_events", 0)) if counts else 0,
            anomaly_count=int(counts.get("anomaly_count", 0)) if counts else 0,
            normal_count=int(counts.get("normal_count", 0)) if counts else 0,
            labeled_count=int(counts.get("labeled_count", 0)) if counts else 0,
            unlabeled_count=int(counts.get("unlabeled_count", 0)) if counts else 0,
            anomalies_last_hour=int(counts.get("anomalies_last_hour", 0)) if counts else 0, # NEW
            anomalies_last_24h=int(counts.get("anomalies_last_24h", 0)) if counts else 0,   # NEW
            label_distribution=label_distribution
        )
        logging.info(f"Calculated event statistics: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Error calculating event statistics: {e}", exc_info=True)
        return EventStats() # Return default stats on error


# --- NEW Time Series Function ---
async def get_event_timeseries(db: AsyncSession, interval: str = "hour", hours_ago: int = 24) -> TimeSeriesData:
    """
    Aggregates event counts (normal vs anomaly) over time intervals.

    Args:
        db (AsyncSession): Database session.
        interval (str): Time interval ('hour' or 'day').
        hours_ago (int): How many hours back to fetch data for.

    Returns:
        TimeSeriesData: Object containing interval and data points.
    """
    if interval not in ["hour", "day"]:
        raise ValueError("Invalid interval specified. Use 'hour' or 'day'.")

    now = datetime.datetime.now()
    start_time = now - timedelta(hours=hours_ago)

    # --- Database specific time truncation ---
    # This part is database-dependent. Using common functions.
    # SQLite: strftime('%Y-%m-%d %H:00:00', timestamp) for hour, strftime('%Y-%m-%d 00:00:00', timestamp) for day
    # PostgreSQL: date_trunc('hour', timestamp), date_trunc('day', timestamp)

    # Use text() for potentially database-specific functions if needed, or adapt ORM query
    # Example using SQLAlchemy functions (might need adjustments per dialect)
    if interval == "hour":
        # Using func.strftime for SQLite compatibility, adapt for PostgreSQL if needed
        time_bucket_expr = func.strftime('%Y-%m-%d %H:00:00', EventLog.timestamp).label("time_bucket_str")
        # For PostgreSQL: time_bucket_expr = func.date_trunc('hour', EventLog.timestamp).label("time_bucket_dt")
    else: # day
        time_bucket_expr = func.strftime('%Y-%m-%d 00:00:00', EventLog.timestamp).label("time_bucket_str")
        # For PostgreSQL: time_bucket_expr = func.date_trunc('day', EventLog.timestamp).label("time_bucket_dt")

    stmt = select(
        time_bucket_expr,
        func.sum(case((EventLog.ae_anomaly_flag == 1, 1), else_=0)).label("anomaly_count"),
        func.sum(case((EventLog.prediction == 0, 1), else_=0)).label("normal_count")
    ).where(EventLog.timestamp >= start_time) \
     .group_by(time_bucket_expr) \
     .order_by(time_bucket_expr)

    data_points = []
    try:
        result = await db.execute(stmt)
        for row in result.mappings().all():
            # Parse the string timestamp back to datetime
            # Handle potential parsing errors
            try:
                 # Adjust format string based on what strftime produces
                 bucket_dt = datetime.datetime.strptime(row["time_bucket_str"], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                 logging.warning(f"Could not parse time bucket string: {row['time_bucket_str']}")
                 continue # Skip this data point if parsing fails

            data_points.append(TimeSeriesPoint(
                time_bucket=bucket_dt,
                anomaly_count=row.get("anomaly_count", 0),
                normal_count=row.get("normal_count", 0)
            ))
        logging.info(f"Fetched {len(data_points)} time series points for interval '{interval}'")
        return TimeSeriesData(interval=interval, data_points=data_points)

    except Exception as e:
        logging.error(f"Error fetching time series data: {e}", exc_info=True)
        return TimeSeriesData(interval=interval, data_points=[]) # Return empty on error


async def get_attack_types(db: AsyncSession, include_inactive: bool = False) -> List[AttackTypes]:
    """
    Fetches defined attack types from the database.

    Args:
        db (AsyncSession): The database session dependency.
        include_inactive (bool): Whether to include types marked as inactive.

    Returns:
        List[AttackTypes]: A list of AttackTypes ORM objects.
                           Returns an empty list if an error occurs.
    """
    log.debug(f"Fetching attack types from DB (include_inactive={include_inactive})...")
    try:
        # Start building the select statement
        stmt = select(AttackTypes)

        # Conditionally filter based on the is_active flag
        if not include_inactive:
            stmt = stmt.where(AttackTypes.is_active == True)

        # Order consistently (e.g., by ID or name)
        stmt = stmt.order_by(AttackTypes.type_id)

        # Execute the query
        result = await db.execute(stmt)
        types = result.scalars().all() # Get all results as ORM objects

        log.info(f"Fetched {len(types)} attack types from database.")
        return list(types) # Return as a standard list

    except Exception as e:
        # Log the error with traceback for debugging
        log.error(f"Database error fetching attack types: {e}", exc_info=True)
        # Return an empty list in case of error to prevent downstream issues
        return []
