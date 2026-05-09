"""
Inference API endpoints for AI Inference Gateway.

Provides endpoints for submitting inference requests, checking request status,
and retrieving results. Supports both synchronous and asynchronous processing.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.middleware import get_current_user, TokenData
from app.database import get_db
from app.models.schemas import (
    InferenceRequest as InferenceRequestSchema,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    RequestStatus
)
from app.exceptions import (
    ModelNotFoundError,
    RateLimitExceededError,
    InvalidInputError,
    raise_http_exception
)
from app.inference.batch_processor import batch_processor
from app.inference.inference_engine import inference_engine
from app.monitoring.metrics import (
    record_inference_request,
    record_inference_latency,
    record_batch_size
)
from app.cache.redis_cache import redis_cache
from app.config import settings
from app.models.database_models import InferenceRequest as InferenceRequestDB, RequestStatus as DBRequestStatus

# Configure logging
logger = logging.getLogger(__name__)

# API Router
router = APIRouter(tags=["Inference"], prefix="/inference")


async def check_rate_limit(user_id: str, tier: str, db: AsyncSession) -> bool:
    """
    Check if user has exceeded their rate limit.
    
    Args:
        user_id: User identifier
        tier: User subscription tier
        db: Database session
    
    Returns:
        bool: True if within limit, False if exceeded
    
    Raises:
        RateLimitExceededError: If rate limit is exceeded
    """
    if not settings.ENABLE_RATE_LIMITING:
        return True
    
    from app.config import settings as app_settings
    
    # Get daily limits based on tier
    limits = {
        "free": app_settings.RATE_LIMIT_FREE_TIER,
        "pro": app_settings.RATE_LIMIT_PRO_TIER,
        "enterprise": app_settings.RATE_LIMIT_ENTERPRISE_TIER
    }
    daily_limit = limits.get(tier, app_settings.RATE_LIMIT_FREE_TIER)
    
    # Check Redis counter first (fast path)
    cache_key = f"rate_limit:{user_id}:{datetime.utcnow().strftime('%Y-%m-%d')}"
    current_count = await redis_cache.get(cache_key) or 0
    
    if int(current_count) >= daily_limit:
        raise RateLimitExceededError(
            user_id=user_id,
            limit=daily_limit,
            current_count=int(current_count),
            tier=tier
        )
    
    return True


async def get_cached_result(cache_key: str) -> Optional[InferenceResponse]:
    """
    Try to get cached inference result.
    
    Args:
        cache_key: Cache key for the request
    
    Returns:
        Cached response if available, None otherwise
    """
    if not settings.ENABLE_CACHING:
        return None
    
    cached = await redis_cache.get(cache_key)
    if cached:
        # Deserialize and return
        from app.models.schemas import InferenceResponse
        try:
            return InferenceResponse.model_validate(cached)
        except Exception:
            logger.warning("Failed to deserialize cached result")
    
    return None


def generate_cache_key(model: str, input_data: str, params: dict) -> str:
    """Generate cache key from request parameters."""
    import hashlib
    import json
    
    # Create deterministic string from inputs
    key_data = f"{model}:{input_data}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(key_data.encode()).hexdigest()


@router.post(
    "",
    response_model=InferenceResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit inference request",
    description="Submit a single inference request. Returns immediately with request ID for async processing."
)
async def submit_inference(
    request: InferenceRequestSchema,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
) -> InferenceResponse:
    """
    Submit a single inference request.
    
    The request is processed asynchronously. For immediate results,
    poll the GET /inference/{request_id} endpoint.
    
    Args:
        request: Inference request with model, input, and parameters
        background_tasks: FastAPI background tasks
        db: Database session
        current_user: Authenticated user
    
    Returns:
        InferenceResponse with request ID and pending status
    """
    # Check rate limit
    await check_rate_limit(current_user.user_id, current_user.tier, db)
    
    # Check cache if enabled
    if request.use_cache and settings.ENABLE_CACHING:
        cache_key = generate_cache_key(
            request.model,
            str(request.input),
            request.parameters or {}
        )
        cached_result = await get_cached_result(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for request {request.request_id}")
            return cached_result
    
    # Create database record
    db_request = InferenceRequestDB(
        id=request.request_id or str(datetime.utcnow().timestamp()),
        user_id=current_user.user_id,
        model_name=request.model,
        task_type=request.task_type.value,
        input_data={"input": request.input, "parameters": request.parameters},
        status=DBRequestStatus.PENDING,
        priority=request.priority,
        cached=False
    )
    
    db.add(db_request)
    await db.commit()
    
    # Submit to batch processor
    if settings.ENABLE_BATCHING:
        await batch_processor.submit_request(
            request_id=db_request.id,
            model_name=request.model,
            inputs=[request.input],
            parameters=request.parameters or {},
            task_type=request.task_type.value,
            priority=request.priority
        )
    else:
        # Process immediately using Celery
        from app.celery_tasks.inference_tasks import process_inference_task
        process_inference_task.delay(
            db_request.id,
            request.model,
            request.input,
            request.parameters or {},
            request.task_type.value,
            current_user.user_id
        )
    
    # Record metrics
    record_inference_request(request.model, "pending")
    
    logger.info(f"Submitted inference request {db_request.id} for model {request.model}")
    
    return InferenceResponse(
        request_id=db_request.id,
        model=request.model,
        status=RequestStatus.PENDING,
        results=[],
        metadata={
            "queued_at": datetime.utcnow().isoformat(),
            "priority": request.priority
        },
        created_at=datetime.utcnow()
    )


@router.post(
    "/batch",
    response_model=BatchInferenceResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch inference request",
    description="Submit multiple inference requests as a batch for efficient processing."
)
async def submit_batch_inference(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
) -> BatchInferenceResponse:
    """
    Submit a batch inference request.
    
    Processes multiple inputs together for better efficiency.
    Results are returned in the same order as inputs.
    
    Args:
        request: Batch request with multiple inputs
        background_tasks: FastAPI background tasks
        db: Database session
        current_user: Authenticated user
    
    Returns:
        BatchInferenceResponse with batch request ID
    """
    # Check rate limit (counts as one request but with multiplier for batch size)
    await check_rate_limit(current_user.user_id, current_user.tier, db)
    
    # Generate batch request ID
    import uuid
    batch_id = str(uuid.uuid4())
    
    # Create database records for each item
    request_ids = []
    for i, input_item in enumerate(request.inputs):
        req_id = f"{batch_id}_{i}"
        request_ids.append(req_id)
        
        db_request = InferenceRequestDB(
            id=req_id,
            user_id=current_user.user_id,
            model_name=request.model,
            task_type=request.task_type.value,
            input_data={"input": input_item, "parameters": request.parameters},
            status=DBRequestStatus.PENDING,
            priority=request.priority,
            cached=False
        )
        db.add(db_request)
    
    await db.commit()
    
    # Submit batch to processor
    await batch_processor.submit_batch(
        batch_id=batch_id,
        model_name=request.model,
        inputs=request.inputs,
        parameters=request.parameters or {},
        task_type=request.task_type.value,
        batch_size=request.batch_size or settings.BATCH_SIZE,
        priority=request.priority
    )
    
    # Record metrics
    record_batch_size(len(request.inputs))
    record_inference_request(request.model, "batch_pending")
    
    logger.info(f"Submitted batch request {batch_id} with {len(request.inputs)} items")
    
    return BatchInferenceResponse(
        request_id=batch_id,
        model=request.model,
        status=RequestStatus.PENDING,
        total_items=len(request.inputs),
        completed_items=0,
        failed_items=0,
        results=[],
        metadata={
            "queued_at": datetime.utcnow().isoformat(),
            "priority": request.priority,
            "individual_request_ids": request_ids
        },
        created_at=datetime.utcnow()
    )


@router.get(
    "/{request_id}",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get inference result",
    description="Retrieve the status and results of a submitted inference request."
)
async def get_inference_result(
    request_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
) -> InferenceResponse:
    """
    Get the result of an inference request.
    
    Poll this endpoint to check status and retrieve results
    for async inference requests.
    
    Args:
        request_id: Unique request identifier
        db: Database session
        current_user: Authenticated user
    
    Returns:
        InferenceResponse with current status and results if available
    """
    # Query database for request
    from sqlalchemy import select
    
    result = await db.execute(
        select(InferenceRequestDB).where(
            InferenceRequestDB.id == request_id,
            InferenceRequestDB.user_id == current_user.user_id
        )
    )
    db_request = result.scalar_one_or_none()
    
    if not db_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "REQUEST_NOT_FOUND",
                    "message": f"Inference request {request_id} not found"
                }
            }
        )
    
    # Map status
    status_mapping = {
        DBRequestStatus.PENDING: RequestStatus.PENDING,
        DBRequestStatus.PROCESSING: RequestStatus.PROCESSING,
        DBRequestStatus.COMPLETED: RequestStatus.COMPLETED,
        DBRequestStatus.FAILED: RequestStatus.FAILED,
        DBRequestStatus.CANCELLED: RequestStatus.CANCELLED
    }
    
    # Build response
    response = InferenceResponse(
        request_id=db_request.id,
        model=db_request.model_name,
        status=status_mapping.get(db_request.status, RequestStatus.PENDING),
        metadata={
            "latency_ms": db_request.latency_ms,
            "tokens_input": db_request.tokens_input,
            "tokens_output": db_request.tokens_output,
            "cached": db_request.cached
        },
        error=db_request.error_message,
        created_at=db_request.created_at,
        completed_at=db_request.completed_at
    )
    
    # Add results if completed
    if db_request.status == DBRequestStatus.COMPLETED and db_request.output_data:
        from app.models.schemas import InferenceResult
        response.results = [
            InferenceResult(
                output=db_request.output_data.get("output", ""),
                tokens_used=db_request.tokens_total,
                processing_time_ms=db_request.latency_ms or 0
            )
        ]
    
    return response


@router.get(
    "/batch/{batch_id}",
    response_model=BatchInferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get batch inference result",
    description="Retrieve the status and results of a batch inference request."
)
async def get_batch_result(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
) -> BatchInferenceResponse:
    """
    Get the results of a batch inference request.
    
    Returns aggregated status and all individual results.
    
    Args:
        batch_id: Batch request identifier
        db: Database session
        current_user: Authenticated user
    
    Returns:
        BatchInferenceResponse with batch status and results
    """
    from sqlalchemy import select, func
    
    # Query all requests in batch
    pattern = f"{batch_id}_%"
    result = await db.execute(
        select(InferenceRequestDB).where(
            InferenceRequestDB.id.like(pattern),
            InferenceRequestDB.user_id == current_user.user_id
        )
    )
    requests = result.scalars().all()
    
    if not requests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "BATCH_NOT_FOUND",
                    "message": f"Batch request {batch_id} not found"
                }
            }
        )
    
    # Aggregate status
    total = len(requests)
    completed = sum(1 for r in requests if r.status == DBRequestStatus.COMPLETED)
    failed = sum(1 for r in requests if r.status == DBRequestStatus.FAILED)
    pending = total - completed - failed
    
    # Determine overall status
    if failed == total:
        overall_status = RequestStatus.FAILED
    elif completed == total:
        overall_status = RequestStatus.COMPLETED
    elif pending == total:
        overall_status = RequestStatus.PENDING
    else:
        overall_status = RequestStatus.PROCESSING
    
    # Build results list
    from app.models.schemas import InferenceResult
    results = []
    for req in sorted(requests, key=lambda r: r.id):
        if req.status == DBRequestStatus.COMPLETED and req.output_data:
            results.append(InferenceResult(
                output=req.output_data.get("output", ""),
                tokens_used=req.tokens_total,
                processing_time_ms=req.latency_ms or 0
            ))
    
    return BatchInferenceResponse(
        request_id=batch_id,
        model=requests[0].model_name if requests else "unknown",
        status=overall_status,
        total_items=total,
        completed_items=completed,
        failed_items=failed,
        results=results,
        metadata={
            "pending_items": pending
        },
        created_at=min(r.created_at for r in requests) if requests else datetime.utcnow(),
        completed_at=max(r.completed_at for r in requests if r.completed_at) if any(r.completed_at for r in requests) else None
    )


@router.delete(
    "/{request_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel inference request",
    description="Cancel a pending inference request."
)
async def cancel_inference(
    request_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
) -> None:
    """
    Cancel a pending inference request.
    
    Can only cancel requests that are still pending or processing.
    
    Args:
        request_id: Request to cancel
        db: Database session
        current_user: Authenticated user
    """
    from sqlalchemy import select
    
    result = await db.execute(
        select(InferenceRequestDB).where(
            InferenceRequestDB.id == request_id,
            InferenceRequestDB.user_id == current_user.user_id,
            InferenceRequestDB.status.in_([DBRequestStatus.PENDING, DBRequestStatus.PROCESSING])
        )
    )
    db_request = result.scalar_one_or_none()
    
    if not db_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "REQUEST_NOT_FOUND",
                    "message": "Request not found or already completed"
                }
            }
        )
    
    db_request.status = DBRequestStatus.CANCELLED
    await db.commit()
    
    logger.info(f"Cancelled inference request {request_id}")
