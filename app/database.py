"""
Database connection and session management for AI Inference Gateway.

Uses SQLAlchemy 2.0 with async support for PostgreSQL.
Provides connection pooling, session management, and dependency injection.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.config import settings
from app.exceptions import DatabaseError

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy declarative base for ORM models
Base = declarative_base()

# Global engine instance - initialized lazily
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """
    Get or create the async database engine.
    
    Uses connection pooling for efficient database access.
    Configures pool size and overflow based on application needs.
    
    Returns:
        AsyncEngine: SQLAlchemy async engine instance
    """
    global _engine
    
    if _engine is None:
        try:
            # Use asyncpg driver for true async PostgreSQL
            database_url = settings.database_async_url
            
            # Configure engine with connection pooling
            # For development/testing, we might use NullPool to avoid connection issues
            if settings.is_development:
                _engine = create_async_engine(
                    database_url,
                    echo=settings.DEBUG,
                    poolclass=NullPool,
                    future=True
                )
            else:
                _engine = create_async_engine(
                    database_url,
                    echo=settings.DEBUG,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,  # Verify connections before use
                    pool_recycle=300,    # Recycle connections after 5 minutes
                    future=True
                )
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise DatabaseError("engine_initialization", {"error": str(e)})
    
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session maker.
    
    Session maker is a factory for creating new database sessions.
    Each session is independent and should be used for a single request.
    
    Returns:
        async_sessionmaker[AsyncSession]: Session factory
    """
    global _async_session_maker
    
    if _async_session_maker is None:
        engine = get_engine()
        
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
            autocommit=False,
            autoflush=False
        )
    
    return _async_session_maker


async def create_tables() -> None:
    """
    Create all database tables based on ORM models.
    
    Should only be used in development. In production,
    use Alembic migrations for schema management.
    """
    engine = get_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")


async def drop_tables() -> None:
    """Drop all database tables - USE WITH CAUTION!"""
    engine = get_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.warning("Database tables dropped")


async def close_database() -> None:
    """Close database connections and cleanup resources."""
    global _engine, _async_session_maker
    
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    
    Automatically handles session lifecycle:
    - Creates new session
    - Commits on successful completion
    - Rollbacks on exception
    - Always closes session
    
    Example:
        async with get_db_session() as session:
            result = await session.execute(query)
    """
    session_maker = get_session_maker()
    session = session_maker()
    
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Use this in FastAPI route dependencies:
    
    @app.get("/items")
    async def get_items(db: AsyncSession = Depends(get_db)):
        ...
    
    Yields:
        AsyncSession: Database session for request handling
    """
    async with get_db_session() as session:
        yield session


# Convenience function for testing
async def init_db() -> None:
    """Initialize database - create tables if they don't exist."""
    if settings.is_development:
        await create_tables()


async def close_db_connection() -> None:
    """Close all database connections."""
    await close_database()
