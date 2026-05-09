"""
Pydantic schemas for AI Inference Gateway API.

Provides request/response models with full validation, documentation,
and type safety for all API endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class TaskType(str, Enum):
    """Supported AI inference task types."""
    TEXT_GENERATION = "text-generation"
    EMBEDDINGS = "embeddings"
    CLASSIFICATION = "classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


class RequestStatus(str, Enum):
    """Status of an inference request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class InferenceRequest(BaseModel):
    """
    Single inference request model.
    
    Contains the input text, selected model, and optional parameters
    for controlling the inference behavior.
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model": "gpt2",
            "input": "Once upon a time in a land far away",
            "parameters": {
                "max_length": 100,
                "temperature": 0.8,
                "top_p": 0.95
            },
            "task_type": "text-generation",
            "use_cache": True,
            "priority": "normal"
        }
    })
    
    model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Model identifier (e.g., 'gpt2', 'bert-base-uncased')",
        examples=["gpt2", "all-MiniLM-L6-v2"]
    )
    
    input: Union[str, List[str], Dict[str, Any]] = Field(
        ...,
        description="Input data for inference. Can be text string, list of texts, or structured data",
        examples=[
            "Once upon a time...",
            ["Text 1", "Text 2"],
            {"question": "What is AI?", "context": "AI is..."}
        ]
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model-specific parameters (temperature, max_tokens, etc.)"
    )
    
    task_type: TaskType = Field(
        default=TaskType.TEXT_GENERATION,
        description="Type of inference task to perform"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results if available"
    )
    
    priority: str = Field(
        default="normal",
        pattern="^(low|normal|high)$",
        description="Request priority for queue processing"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Optional custom request ID (auto-generated if not provided)"
    )
    
    @field_validator('request_id', mode='before')
    @classmethod
    def set_request_id(cls, v):
        """Auto-generate request ID if not provided."""
        return v or str(uuid4())


class InferenceResult(BaseModel):
    """Individual inference result within a response."""
    
    output: Union[str, List[float], List[str], Dict[str, Any]] = Field(
        ...,
        description="Model output - varies by task type"
    )
    
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Number of tokens consumed"
    )
    
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Time taken to process this result in milliseconds"
    )


class InferenceResponse(BaseModel):
    """
    Single inference response model.
    
    Contains the generated output, metadata about the inference process,
    and request tracking information.
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "model": "gpt2",
            "status": "completed",
            "results": [{
                "output": "Once upon a time in a land far away, there lived...",
                "tokens_used": 50,
                "processing_time_ms": 125.5
            }],
            "metadata": {
                "latency_ms": 150.2,
                "model_version": "1.0.0",
                "device": "cpu",
                "cached": False
            },
            "created_at": "2024-01-15T10:30:00Z"
        }
    })
    
    request_id: str = Field(
        ...,
        description="Unique identifier for the request"
    )
    
    model: str = Field(
        ...,
        description="Model used for inference"
    )
    
    status: RequestStatus = Field(
        default=RequestStatus.COMPLETED,
        description="Status of the inference request"
    )
    
    results: List[InferenceResult] = Field(
        default_factory=list,
        description="List of inference results (one per input)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the inference"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the request was created"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when inference completed"
    )


class BatchInferenceRequest(BaseModel):
    """
    Batch inference request model for processing multiple inputs.
    
    All inputs are processed with the same model and parameters.
    Results are returned in the same order as the inputs.
    """
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "model": "all-MiniLM-L6-v2",
            "inputs": [
                "First text to process",
                "Second text to process",
                "Third text to process"
            ],
            "task_type": "embeddings",
            "batch_size": 16
        }
    })
    
    model: str = Field(
        ...,
        description="Model identifier for batch processing"
    )
    
    inputs: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of inputs to process in batch"
    )
    
    task_type: TaskType = Field(
        default=TaskType.EMBEDDINGS,
        description="Type of task for all inputs"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameters applied to all batch items"
    )
    
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=256,
        description="Batch processing size (uses system default if not specified)"
    )
    
    priority: str = Field(
        default="normal",
        pattern="^(low|normal|high)$",
        description="Priority for batch processing"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Whether to use caching for batch results"
    )


class BatchInferenceResponse(BaseModel):
    """Response for batch inference requests."""
    
    request_id: str = Field(..., description="Unique batch request ID")
    model: str = Field(..., description="Model used for batch processing")
    status: RequestStatus = Field(..., description="Overall batch status")
    total_items: int = Field(..., ge=0, description="Total number of items in batch")
    completed_items: int = Field(..., ge=0, description="Number of successfully processed items")
    failed_items: int = Field(..., ge=0, description="Number of failed items")
    results: List[InferenceResult] = Field(default_factory=list, description="Individual results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ModelInfo(BaseModel):
    """Information about an available AI model."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "gpt2",
            "name": "GPT-2",
            "version": "1.0.0",
            "description": "Generative Pre-trained Transformer 2",
            "task_types": ["text-generation"],
            "parameters": "124M",
            "device": "cpu",
            "loaded": True,
            "max_sequence_length": 1024,
            "tags": ["text-generation", "gpt", "transformer"]
        }
    })
    
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    version: str = Field(default="1.0.0", description="Model version")
    description: Optional[str] = Field(default=None, description="Model description")
    task_types: List[TaskType] = Field(default_factory=list, description="Supported task types")
    parameters: Optional[str] = Field(default=None, description="Model size (e.g., '124M', '1.5B')")
    device: str = Field(default="cpu", description="Device where model is loaded")
    loaded: bool = Field(default=False, description="Whether model is currently loaded")
    max_sequence_length: int = Field(default=512, description="Maximum input sequence length")
    tags: List[str] = Field(default_factory=list, description="Model tags/keywords")
    license: Optional[str] = Field(default=None, description="Model license information")
    
    @field_validator('parameters', mode='before')
    @classmethod
    def format_parameters(cls, v):
        """Format parameter count as human-readable string."""
        if isinstance(v, int):
            if v >= 1_000_000_000:
                return f"{v / 1_000_000_000:.1f}B"
            elif v >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"
            elif v >= 1_000:
                return f"{v / 1_000:.1f}K"
        return v


class ModelListResponse(BaseModel):
    """Response containing list of available models."""
    
    models: List[ModelInfo] = Field(default_factory=list, description="Available models")
    total: int = Field(..., ge=0, description="Total number of models available")
    loaded: int = Field(..., ge=0, description="Number of currently loaded models")


class HealthStatus(BaseModel):
    """Health status of a system component."""
    
    status: str = Field(..., pattern="^(healthy|unhealthy|degraded)$")
    response_time_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """System health check response."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2024-01-15T10:30:00Z",
            "components": {
                "database": {"status": "healthy", "response_time_ms": 12.5},
                "redis": {"status": "healthy", "response_time_ms": 3.2},
                "models": {"status": "healthy", "loaded": 2}
            },
            "system": {
                "cpu_usage_percent": 25.5,
                "memory_usage_percent": 45.2,
                "active_requests": 5
            }
        }
    })
    
    status: str = Field(..., pattern="^(healthy|unhealthy|degraded)$")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime in seconds")
    components: Dict[str, HealthStatus] = Field(default_factory=dict, description="Component health")
    system: Optional[Dict[str, Any]] = Field(default=None, description="System metrics")


class LoginRequest(BaseModel):
    """User login request."""
    
    api_key: str = Field(
        ...,
        min_length=10,
        description="User API key for authentication"
    )
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key format."""
        if not v.startswith(('ak_', 'pk_', 'test_')):
            raise ValueError("Invalid API key format. Must start with 'ak_', 'pk_', or 'test_'")
        return v


class TokenResponse(BaseModel):
    """Token response after successful authentication."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    scopes: List[str] = Field(default_factory=list, description="Granted scopes")


class UserInfo(BaseModel):
    """User information response."""
    
    user_id: str = Field(..., description="Unique user identifier")
    tier: UserTier = Field(..., description="Subscription tier")
    api_key: str = Field(..., description="Masked API key")
    requests_today: int = Field(default=0, description="Requests made today")
    requests_limit: int = Field(..., description="Daily request limit")
    created_at: Optional[datetime] = None
    
    @field_validator('api_key')
    @classmethod
    def mask_api_key(cls, v):
        """Mask API key for security in responses."""
        if len(v) > 8:
            return f"{v[:4]}****{v[-4:]}"
        return "****"


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: ErrorDetail = Field(..., description="Error information")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Base paginated response."""
    
    items: List[Any] = Field(default_factory=list, description="Response items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=1000, description="Items per page")
    pages: int = Field(..., ge=1, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class InferenceHistoryResponse(PaginatedResponse):
    """Paginated inference history response."""
    
    items: List[InferenceResponse] = Field(default_factory=list)
