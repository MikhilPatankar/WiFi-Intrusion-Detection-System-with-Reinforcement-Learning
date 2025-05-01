
# --- File: src/backend/app/db.py ---

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from .core.config import settings
from sqlmodel import SQLModel # Required for table creation

# Configure engine arguments (especially for SQLite)
engine_args = {"echo": False} # Set echo=True for SQL logging
if settings.DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

# Create the asynchronous SQLAlchemy engine
engine = create_async_engine(settings.DATABASE_URL, **engine_args)

# Create an asynchronous session factory
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False, # Recommended for FastAPI background tasks
    autocommit=False,
    autoflush=False,
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency function that yields an async session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_db_and_tables():
    """Creates database tables based on SQLModel metadata if they don't exist."""
    async with engine.begin() as conn:
        logging.info("Checking/Creating database tables...")
        # SQLModel.metadata.create_all checks for table existence before creating
        await conn.run_sync(SQLModel.metadata.create_all)
        logging.info("Database tables checked/created.")
