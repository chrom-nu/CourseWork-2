import os
from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.config import settings

Base = declarative_base()

async_url = settings.DATABASE_URL
sync_url = settings.SYNC_DATABASE_URL

# Async connect args
is_async_sqlite = async_url.startswith("sqlite")
async_connect_args = {"check_same_thread": False} if is_async_sqlite else {"timeout": 3}

# Async engine for FastAPI endpoints
async_engine = create_async_engine(
    async_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=async_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Sync engine for SQLAdmin and migrations
is_sync_sqlite = sync_url.startswith("sqlite")
sync_connect_args = {"check_same_thread": False} if is_sync_sqlite else {"connect_timeout": 3}

sync_engine = create_engine(
    sync_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=sync_connect_args,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
