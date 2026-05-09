"""
Configuration management for AI Inference Gateway.

Uses Pydantic Settings for environment variable parsing and validation.
Provides strongly-typed configuration with defaults and validation.
"""

import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    
    # Application Settings
    APP_NAME: str = Field(default="AI Inference Gateway", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    ENVIRONMENT: str = Field(default="development", description="Environment (development/staging/production)")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Database Settings
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/inference_gateway",
        description="PostgreSQL connection URL"
    )
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="inference_gateway")
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: str = Field(default="postgres")
    
    # Redis Settings
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    
    # Celery Settings
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    CELERY_CONCURRENCY: int = Field(default=4, ge=1, le=32)
    
    # JWT Authentication
    SECRET_KEY: str = Field(
        default="your-super-secret-key-change-this-in-production",
        min_length=32,
        description="Secret key for JWT token generation"
    )
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=30)
    
    # Rate Limiting
    RATE_LIMIT_FREE_TIER: int = Field(default=100, ge=0)
    RATE_LIMIT_PRO_TIER: int = Field(default=10000, ge=0)
    RATE_LIMIT_ENTERPRISE_TIER: int = Field(default=100000, ge=0)
    
    # Model Configuration
    MODEL_CACHE_DIR: str = Field(default="./models_cache")
    DEVICE: str = Field(default="cpu", pattern="^(cpu|cuda|auto)$")
    BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    BATCH_WAIT_TIME_MS: int = Field(default=100, ge=10, le=1000)
    MAX_SEQUENCE_LENGTH: int = Field(default=512, ge=64, le=4096)
    
    # Inference Engine
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, ge=1, le=1000)
    REQUEST_TIMEOUT_SECONDS: int = Field(default=60, ge=5, le=300)
    ENABLE_MODEL_FALLBACK: bool = Field(default=True)
    CACHE_TTL_SECONDS: int = Field(default=86400, ge=300, le=604800)  # 24 hours default
    
    # Monitoring & Logging
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    ENABLE_PROMETHEUS: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(default=9090)
    GRAFANA_PORT: int = Field(default=3000)
    
    # Feature Flags
    ENABLE_BATCHING: bool = Field(default=True)
    ENABLE_CACHING: bool = Field(default=True)
    ENABLE_RATE_LIMITING: bool = Field(default=True)
    ENABLE_ASYNC_PROCESSING: bool = Field(default=True)
    
    # Supported Models - comma-separated list
    # Format: model_name:model_path_or_hf_id:task_type
    SUPPORTED_MODELS: str = Field(
        default="gpt2:gpt2:text-generation,all-MiniLM-L6-v2:sentence-transformers/all-MiniLM-L6-v2:embeddings"
    )
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v.lower()
    
    @validator("DEVICE")
    def validate_device(cls, v: str) -> str:
        """Validate device selection and auto-detect CUDA availability."""
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v
    
    @validator("MODEL_CACHE_DIR")
    def validate_model_cache_dir(cls, v: str) -> str:
        """Ensure model cache directory exists."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"
    
    @property
    def database_async_url(self) -> str:
        """Generate async database URL for asyncpg."""
        # Replace postgresql:// with postgresql+asyncpg://
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to avoid reloading settings on every call.
    Settings are loaded once at startup and cached.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()


# Re-export settings for direct import
__all__ = ["Settings", "get_settings", "settings"]
