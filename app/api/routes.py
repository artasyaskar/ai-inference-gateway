"""
Main API routes for AI Inference Gateway.

Combines all API endpoints and provides authentication routes.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.auth.jwt_handler import create_token_pair, verify_token
from app.auth.middleware import get_current_user, TokenData
from app.database import get_db
from app.models.schemas import (
    LoginRequest,
    TokenResponse,
    UserInfo,
    ModelListResponse,
    ModelInfo,
    ErrorResponse
)
from app.models.database_models import User, UserTier
from app.exceptions import AuthenticationError
from app.monitoring.metrics import record_auth_attempt
from app.api.health_check import router as health_router
from app.api.inference import router as inference_router

# Configure logging
logger = logging.getLogger(__name__)

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health_router)
api_router.include_router(inference_router)


@api_router.post(
    "/auth/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate and get access token",
    description="Authenticate using API key and receive JWT tokens for accessing protected endpoints."
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Authenticate user and generate JWT tokens.
    
    Validates the provided API key and returns access and refresh tokens.
    The access token is used for all subsequent API requests.
    
    Args:
        request: Login credentials with API key
        db: Database session
    
    Returns:
        TokenResponse with access and refresh tokens
    
    Raises:
        HTTPException: 401 if authentication fails
    
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/login" \
          -H "Content-Type: application/json" \
          -d '{"api_key": "ak_test_1234567890abcdef"}'
        ```
    """
    try:
        # Query user by API key
        result = await db.execute(
            select(User).where(
                User.api_key == request.api_key,
                User.is_active == True
            )
        )
        user = result.scalar_one_or_none()
        
        if not user:
            logger.warning(f"Authentication failed: invalid API key")
            record_auth_attempt(success=False)
            raise AuthenticationError("Invalid API key")
        
        # Update last login
        user.last_login_at = datetime.utcnow()
        await db.commit()
        
        # Determine scopes based on tier
        from app.auth.middleware import Scopes
        scopes = Scopes.FREE_TIER
        if user.tier == UserTier.PRO:
            scopes = Scopes.PRO_TIER
        elif user.tier == UserTier.ENTERPRISE:
            scopes = Scopes.ENTERPRISE_TIER
        
        # Generate tokens
        tokens = create_token_pair(
            user_id=user.id,
            api_key=user.api_key,
            tier=user.tier.value,
            scopes=scopes
        )
        
        record_auth_attempt(success=True)
        
        logger.info(f"User {user.id} authenticated successfully")
        
        return TokenResponse(**tokens, scopes=scopes)
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": e.error_code,
                    "message": e.message
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


@api_router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
    description="Use refresh token to obtain a new access token."
)
async def refresh_token(request: Request) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    Extracts refresh token from Authorization header and issues
    a new access token pair.
    
    Args:
        request: HTTP request with refresh token in Authorization header
    
    Returns:
        TokenResponse with new tokens
    """
    # Extract refresh token from header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "INVALID_TOKEN", "message": "Refresh token required"}}
        )
    
    refresh_token = auth_header[7:]  # Remove "Bearer "
    
    try:
        # Verify refresh token and get user info
        token_data = verify_token(refresh_token, token_type="refresh")
        
        # Generate new tokens
        tokens = create_token_pair(
            user_id=token_data.user_id,
            api_key=token_data.api_key,
            tier=token_data.tier,
            scopes=token_data.scopes
        )
        
        return TokenResponse(**tokens, scopes=token_data.scopes)
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": e.error_code, "message": e.message}}
        )


@api_router.get(
    "/auth/me",
    response_model=UserInfo,
    status_code=status.HTTP_200_OK,
    summary="Get current user info",
    description="Returns information about the authenticated user."
)
async def get_current_user_info(
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
) -> UserInfo:
    """
    Get information about the authenticated user.
    
    Returns user profile including tier, usage statistics, and limits.
    
    Args:
        db: Database session
        current_user: Authenticated user from token
    
    Returns:
        UserInfo with user details
    """
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == current_user.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "USER_NOT_FOUND", "message": "User not found"}}
        )
    
    # Get today's usage
    from app.models.database_models import APIUsage
    from sqlalchemy import func
    
    today = datetime.utcnow().date()
    usage_result = await db.execute(
        select(APIUsage).where(
            APIUsage.user_id == user.id,
            func.date(APIUsage.date) == today
        )
    )
    today_usage = usage_result.scalar_one_or_none()
    
    return UserInfo(
        user_id=user.id,
        tier=user.tier.value,
        api_key=user.api_key,
        requests_today=today_usage.requests_count if today_usage else 0,
        requests_limit=user.daily_request_limit,
        created_at=user.created_at
    )


@api_router.get(
    "/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List available models",
    description="Returns a list of all available AI models with their capabilities and status."
)
async def list_models(
    current_user: Optional[TokenData] = Depends(get_current_user)
) -> ModelListResponse:
    """
    List all available AI models.
    
    Returns model information including supported tasks, version,
    and current loading status. No authentication required.
    
    Args:
        current_user: Optional authenticated user
    
    Returns:
        ModelListResponse with list of available models
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/models"
        ```
    """
    from app.inference.model_manager import model_manager
    from app.models.schemas import TaskType
    
    # Get available models from manager
    available_models = model_manager.list_available_models()
    
    models = []
    for model_info in available_models:
        # Map task types
        task_types = []
        for task in model_info.get("task_types", []):
            try:
                task_types.append(TaskType(task))
            except ValueError:
                pass
        
        model = ModelInfo(
            id=model_info.get("id", "unknown"),
            name=model_info.get("name", model_info.get("id", "Unknown")),
            version=model_info.get("version", "1.0.0"),
            description=model_info.get("description"),
            task_types=task_types or [TaskType.TEXT_GENERATION],
            parameters=model_info.get("parameters"),
            device=model_info.get("device", "cpu"),
            loaded=model_info.get("loaded", False),
            max_sequence_length=model_info.get("max_sequence_length", 512),
            tags=model_info.get("tags", []),
            license=model_info.get("license")
        )
        models.append(model)
    
    return ModelListResponse(
        models=models,
        total=len(models),
        loaded=sum(1 for m in models if m.loaded)
    )


@api_router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get model details",
    description="Returns detailed information about a specific model."
)
async def get_model_details(
    model_id: str,
    current_user: Optional[TokenData] = Depends(get_current_user)
) -> ModelInfo:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier
        current_user: Optional authenticated user
    
    Returns:
        ModelInfo with detailed model information
    """
    from app.inference.model_manager import model_manager
    from models.schemas import TaskType
    
    model_info = model_manager.get_model_info(model_id)
    
    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "MODEL_NOT_FOUND",
                    "message": f"Model '{model_id}' not found"
                }
            }
        )
    
    # Map task types
    task_types = []
    for task in model_info.get("task_types", []):
        try:
            task_types.append(TaskType(task))
        except ValueError:
            pass
    
    return ModelInfo(
        id=model_info.get("id", model_id),
        name=model_info.get("name", model_id),
        version=model_info.get("version", "1.0.0"),
        description=model_info.get("description"),
        task_types=task_types or [TaskType.TEXT_GENERATION],
        parameters=model_info.get("parameters"),
        device=model_info.get("device", "cpu"),
        loaded=model_info.get("loaded", False),
        max_sequence_length=model_info.get("max_sequence_length", 512),
        tags=model_info.get("tags", []),
        license=model_info.get("license")
    )


# Make TokenData and other imports available
from typing import Optional
