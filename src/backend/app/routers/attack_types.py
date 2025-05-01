# src/backend/app/routers/attack_types.py
# Router for handling attack type related endpoints

from fastapi import APIRouter, HTTPException, Depends, Query
import logging
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

# Import necessary models and services
# Adjust relative paths if needed based on your structure
try:
    from ..models import AttackTypeRead
    from ..services import log_service # Attack type logic is in log_service
    from ..db import get_async_session
except ImportError:
    # Handle potential import errors if structure differs
    # This might happen if running this file directly or structure isn't standard
    logging.error("Failed to import necessary modules for attack_types router.")
    # Define dummy models/functions for linting if needed, but raise error ideally
    raise

log = logging.getLogger(__name__)

# Define the router for attack types
router = APIRouter(
    prefix="/attack_types",
    tags=["Attack Types"] # Tag for API docs grouping
)

@router.get("/", response_model=List[AttackTypeRead])
async def read_attack_types(
    include_inactive: bool = Query(False, description="Include inactive attack types in the list"),
    db: AsyncSession = Depends(get_async_session)
):
    """Retrieve a list of all defined attack classifications."""
    log.info(f"Request received for attack types (include_inactive={include_inactive}).")
    try:
        types = await log_service.get_attack_types(db, include_inactive=include_inactive)
        # Pydantic handles the conversion from ORM objects to the response model
        return types
    except Exception as e:
        log.error(f"Failed to get attack types: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve attack types.")

# Add other attack type related endpoints here later if needed (e.g., POST to create, PUT to update)