"""
Custom exceptions for AI Inference Gateway.

Provides structured exception classes for different error scenarios
with appropriate HTTP status codes and error details.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class GatewayException(Exception):
    """Base exception for all gateway errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details
            }
        }


class ModelNotFoundError(GatewayException):
    """Raised when a requested model is not available."""
    
    def __init__(
        self,
        model_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Model '{model_name}' not found or not available",
            error_code="MODEL_NOT_FOUND",
            details={"model_name": model_name, **(details or {})},
            status_code=status.HTTP_404_NOT_FOUND
        )


class InferenceTimeoutError(GatewayException):
    """Raised when inference exceeds the timeout threshold."""
    
    def __init__(
        self,
        model_name: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Inference for model '{model_name}' timed out after {timeout_seconds}s",
            error_code="INFERENCE_TIMEOUT",
            details={
                "model_name": model_name,
                "timeout_seconds": timeout_seconds,
                **(details or {})
            },
            status_code=status.HTTP_504_GATEWAY_TIMEOUT
        )


class RateLimitExceededError(GatewayException):
    """Raised when API rate limit is exceeded."""
    
    def __init__(
        self,
        user_id: str,
        limit: int,
        current_count: int,
        tier: str = "free",
        reset_time: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Rate limit exceeded. Limit: {limit}, Current: {current_count}",
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "user_id": user_id,
                "tier": tier,
                "limit": limit,
                "current_count": current_count,
                "reset_time": reset_time,
                **(details or {})
            },
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )


class InvalidInputError(GatewayException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            details=error_details,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class AuthenticationError(GatewayException):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            details=details or {},
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationError(GatewayException):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if resource:
            error_details["resource"] = resource
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_FAILED",
            details=error_details,
            status_code=status.HTTP_403_FORBIDDEN
        )


class ModelLoadingError(GatewayException):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        model_name: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOADING_FAILED",
            details={
                "model_name": model_name,
                "reason": reason,
                **(details or {})
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class BatchProcessingError(GatewayException):
    """Raised when batch processing fails."""
    
    def __init__(
        self,
        batch_id: str,
        failed_count: int,
        total_count: int,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Batch {batch_id} processing failed for {failed_count}/{total_count} items",
            error_code="BATCH_PROCESSING_FAILED",
            details={
                "batch_id": batch_id,
                "failed_count": failed_count,
                "total_count": total_count,
                **(details or {})
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class DatabaseError(GatewayException):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Database error during {operation}",
            error_code="DATABASE_ERROR",
            details={"operation": operation, **(details or {})},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class CacheError(GatewayException):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Cache error during {operation}",
            error_code="CACHE_ERROR",
            details={"operation": operation, **(details or {})},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def raise_http_exception(exc: GatewayException) -> HTTPException:
    """Convert GatewayException to FastAPI HTTPException."""
    return HTTPException(
        status_code=exc.status_code,
        detail=exc.to_dict()
    )
