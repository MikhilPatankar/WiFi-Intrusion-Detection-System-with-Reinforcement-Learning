
# --- File: src/backend/app/db.py ---
# No changes needed here assuming config.py provides the correct DATABASE_URL

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging
from .core.config import settings # Import updated settings
from sqlmodel import SQLModel

# Create the async engine
engine_args = {"echo": False}
if settings.DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(settings.DATABASE_URL, **engine_args)

# Create an async session factory
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_async_session() -> AsyncSession:
    """Dependency function that yields an async session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_db_and_tables():
    """Creates database tables based on SQLModel metadata."""
    async with engine.begin() as conn:
        logging.info("Attempting to create database tables if they don't exist...")
        await conn.run_sync(SQLModel.metadata.create_all)
        logging.info("Database tables checked/created.")

