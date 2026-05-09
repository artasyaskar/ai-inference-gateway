"""
SQLAlchemy ORM models for AI Inference Gateway.

Defines database tables for users, API usage tracking, and inference request history.
Uses SQLAlchemy 2.0 async features for non-blocking database operations.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean,
    ForeignKey, Index, Enum as SQLEnum, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database import Base


class UserTier(PyEnum):
    """User subscription tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class RequestStatus(PyEnum):
    """Status of an inference request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(Base):
    """
    User model for multi-tenant support.
    
    Stores user information, API keys, and subscription tier.
    """
    
    __tablename__ = "users"
    
    # Primary key using UUID for security and distributed systems
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )
    
    # API key for authentication (unique and indexed for fast lookup)
    api_key: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    
    # User tier for rate limiting and feature access
    tier: Mapped[UserTier] = mapped_column(
        SQLEnum(UserTier),
        default=UserTier.FREE,
        nullable=False,
        index=True
    )
    
    # Email for notifications and account management (optional)
    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True
    )
    
    # User name or organization name
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    api_usage: Mapped[List["APIUsage"]] = relationship(
        "APIUsage",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    inference_requests: Mapped[List["InferenceRequest"]] = relationship(
        "InferenceRequest",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, tier={self.tier.value}, active={self.is_active})>"
    
    @property
    def daily_request_limit(self) -> int:
        """Get daily request limit based on tier."""
        from app.config import settings
        limits = {
            UserTier.FREE: settings.RATE_LIMIT_FREE_TIER,
            UserTier.PRO: settings.RATE_LIMIT_PRO_TIER,
            UserTier.ENTERPRISE: settings.RATE_LIMIT_ENTERPRISE_TIER
        }
        return limits.get(self.tier, settings.RATE_LIMIT_FREE_TIER)


class APIUsage(Base):
    """
    API usage tracking model.
    
    Records daily API usage per user for rate limiting and analytics.
    One record per user per day.
    """
    
    __tablename__ = "api_usage"
    
    # Composite primary key: user_id + date
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Usage statistics
    requests_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    tokens_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Date for this usage record (normalized to midnight)
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
        index=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="api_usage")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('ix_api_usage_user_date', 'user_id', 'date', unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<APIUsage(user={self.user_id}, date={self.date.date()}, requests={self.requests_count})>"


class InferenceRequest(Base):
    """
    Inference request history model.
    
    Stores complete history of all inference requests for analytics,
    debugging, and compliance purposes.
    """
    
    __tablename__ = "inference_requests"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )
    
    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Request details
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    
    # Input data (stored as JSON for flexibility)
    input_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
    )
    
    # Output data (stored as JSON)
    output_data: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
    )
    
    # Performance metrics
    latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    tokens_input: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    tokens_output: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    tokens_total: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Request status
    status: Mapped[RequestStatus] = mapped_column(
        SQLEnum(RequestStatus),
        default=RequestStatus.PENDING,
        nullable=False,
        index=True
    )
    
    # Error information (if failed)
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Additional metadata (renamed to avoid SQLAlchemy reserved keyword)
    request_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        default=dict,
        nullable=True
    )
    
    # Celery task ID for async tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )
    
    # Whether result was served from cache
    cached: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # IP address (for security/analytics)
    client_ip: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True
    )
    
    # User agent string
    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Priority level
    priority: Mapped[str] = mapped_column(
        String(20),
        default="normal",
        nullable=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
        index=True
    )
    
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="inference_requests")
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_inference_requests_user_created', 'user_id', 'created_at'),
        Index('ix_inference_requests_status_created', 'status', 'created_at'),
        Index('ix_inference_requests_model_created', 'model_name', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<InferenceRequest(id={self.id}, model={self.model_name}, status={self.status.value})>"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate request duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ModelCache(Base):
    """
    Model cache tracking for cache management.
    
    Tracks cached inference results for cache invalidation and statistics.
    """
    
    __tablename__ = "model_cache"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    # Cache key (hash of model + input)
    cache_key: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hex
        unique=True,
        nullable=False,
        index=True
    )
    
    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )
    
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    
    # Cached output
    output_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )
    
    # Cache hit statistics
    hit_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )
    
    # TTL and timestamps
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False
    )
    
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Index for cache cleanup queries
    __table_args__ = (
        Index('ix_model_cache_model_expires', 'model_name', 'expires_at'),
    )


class SystemMetrics(Base):
    """
    System metrics for monitoring and analytics.
    
    Stores aggregated system performance data.
    """
    
    __tablename__ = "system_metrics"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    # Metric timestamp (bucketed)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
        index=True
    )
    
    # Request metrics
    total_requests: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    successful_requests: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    failed_requests: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Performance metrics
    avg_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    min_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    max_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    # Resource metrics
    cpu_usage_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    memory_usage_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    gpu_usage_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    
    # Model metrics (JSON for flexibility)
    model_metrics: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        default=dict,
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False
    )
