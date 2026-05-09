"""
Models module for AI Inference Gateway.

Contains Pydantic schemas for API validation and SQLAlchemy ORM models
for database persistence.
"""

from app.models.schemas import (
    # Request/Response Models
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ModelInfo,
    ModelListResponse,
    HealthResponse,
    # Authentication Models
    LoginRequest,
    TokenResponse,
    UserInfo,
    # Common Models
    ErrorResponse,
    PaginatedResponse,
    # Enums
    TaskType,
    RequestStatus
)

from app.models.database_models import (
    User,
    APIUsage,
    InferenceRequest as InferenceRequestDB,
    Base
)

__all__ = [
    # Schemas
    "InferenceRequest",
    "InferenceResponse",
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "ModelInfo",
    "ModelListResponse",
    "HealthResponse",
    "LoginRequest",
    "TokenResponse",
    "UserInfo",
    "ErrorResponse",
    "PaginatedResponse",
    "TaskType",
    "RequestStatus",
    # Database Models
    "User",
    "APIUsage",
    "InferenceRequestDB",
    "Base"
]
