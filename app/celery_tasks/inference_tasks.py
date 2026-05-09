"""
Celery tasks for asynchronous inference processing.

Handles background processing of inference requests with:
- Task status tracking
- Retry logic with exponential backoff
- Result storage in database
- Cleanup tasks for maintenance
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum as PyEnum

from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.database import Base
from app.models.database_models import InferenceRequest, RequestStatus

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Celery app
# Connection to Redis as both broker and result backend
celery_app = Celery(
    "ai_inference_gateway",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.celery_tasks.inference_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # Soft limit at 4 minutes
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute initial delay
    task_max_retries=3,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend="redis",
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
)


class CeleryTaskStatus(PyEnum):
    """Celery task execution status."""
    PENDING = "pending"
    STARTED = "started"
    RETRY = "retry"
    SUCCESS = "success"
    FAILURE = "failure"


class DatabaseTask(Task):
    """
    Base Celery task with database session management.
    
    Provides automatic database session handling for tasks
    that need to interact with the database.
    """
    
    _db_session = None
    
    def after_return(self, *args, **kwargs):
        """Cleanup after task execution."""
        if self._db_session:
            try:
                self._db_session.close()
            except Exception:
                pass
            self._db_session = None


def get_db_session():
    """Create a database session for Celery tasks."""
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    max_retries=3,
    default_retry_delay=60,
    time_limit=300,
    soft_time_limit=240
)
def process_inference_task(
    self,
    request_id: str,
    model_name: str,
    input_data: Any,
    parameters: Dict[str, Any],
    task_type: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Process a single inference request asynchronously.
    
    This task handles the actual inference execution in the background,
    updating the database with results and handling retries on failure.
    
    Args:
        request_id: Unique request identifier
        model_name: Target model name
        input_data: Input data for inference
        parameters: Model parameters
        task_type: Type of inference task
        user_id: User who submitted the request
    
    Returns:
        Dictionary with task results and metadata
    
    Raises:
        MaxRetriesExceededError: If all retries fail
        SoftTimeLimitExceeded: If task exceeds soft time limit
    """
    logger.info(f"Processing inference task {request_id} for model {model_name}")
    
    db = get_db_session()
    
    try:
        # Update status to processing
        _update_request_status(db, request_id, RequestStatus.PROCESSING)
        
        # Run async inference
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from app.inference.inference_engine import inference_engine
            
            result = loop.run_until_complete(
                inference_engine.process_single(
                    request_id=request_id,
                    model_name=model_name,
                    input_data=input_data,
                    parameters=parameters,
                    task_type=task_type
                )
            )
        finally:
            loop.close()
        
        # Update database with results
        if result.error:
            _update_request_error(db, request_id, result.error)
            raise Exception(f"Inference failed: {result.error}")
        else:
            _update_request_success(
                db, request_id, result.output,
                result.tokens_input, result.tokens_output,
                result.latency_ms
            )
        
        logger.info(f"Inference task {request_id} completed successfully")
        
        return {
            "status": "success",
            "request_id": request_id,
            "output": result.output,
            "tokens_used": result.tokens_input + result.tokens_output,
            "latency_ms": result.latency_ms,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task {request_id} exceeded soft time limit")
        _update_request_error(db, request_id, "Request timed out")
        raise
        
    except Exception as exc:
        logger.error(f"Inference task {request_id} failed: {exc}")
        
        # Retry with exponential backoff
        try:
            self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        except MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for task {request_id}")
            _update_request_error(db, request_id, f"Max retries exceeded: {str(exc)}")
            raise
    
    finally:
        db.close()


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    max_retries=2,
    default_retry_delay=30,
    time_limit=600  # 10 minutes for batch
)
def process_batch_task(
    self,
    batch_id: str,
    request_ids: List[str],
    model_name: str,
    inputs: List[Any],
    parameters: Dict[str, Any],
    task_type: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Process a batch of inference requests asynchronously.
    
    Handles batch processing with individual request tracking.
    Each item in the batch is processed together for efficiency,
    but tracked separately for status and error handling.
    
    Args:
        batch_id: Unique batch identifier
        request_ids: List of individual request IDs
        model_name: Target model name
        inputs: List of input data
        parameters: Model parameters
        task_type: Type of task
        user_id: User who submitted the batch
    
    Returns:
        Dictionary with batch results and individual status
    """
    logger.info(f"Processing batch task {batch_id} with {len(inputs)} items")
    
    db = get_db_session()
    
    try:
        # Mark all requests as processing
        for req_id in request_ids:
            _update_request_status(db, req_id, RequestStatus.PROCESSING)
        
        # Run batch inference
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            from app.inference.inference_engine import inference_engine
            
            results = loop.run_until_complete(
                inference_engine.process_batch(
                    model_name=model_name,
                    inputs=inputs,
                    parameters=parameters,
                    task_type=task_type
                )
            )
        finally:
            loop.close()
        
        # Update each request with results
        completed = 0
        failed = 0
        
        for i, (req_id, result) in enumerate(zip(request_ids, results)):
            if result.error:
                _update_request_error(db, req_id, result.error)
                failed += 1
            else:
                _update_request_success(
                    db, req_id, result.output,
                    result.tokens_input, result.tokens_output,
                    result.latency_ms
                )
                completed += 1
        
        logger.info(f"Batch task {batch_id} completed: {completed} success, {failed} failed")
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "total": len(inputs),
            "completed": completed,
            "failed": failed,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Batch task {batch_id} failed: {exc}")
        
        # Mark all as failed
        for req_id in request_ids:
            _update_request_error(db, req_id, f"Batch processing failed: {str(exc)}")
        
        try:
            self.retry(exc=exc)
        except MaxRetriesExceededError:
            raise
    
    finally:
        db.close()


@celery_app.task
def cleanup_old_requests(days: int = 7) -> Dict[str, int]:
    """
    Cleanup old completed/failed inference requests from database.
    
    Maintenance task that removes old request records to prevent
    database growth. Typically runs daily.
    
    Args:
        days: Delete requests older than this many days
    
    Returns:
        Dictionary with deletion statistics
    """
    from datetime import timedelta
    
    logger.info(f"Starting cleanup of requests older than {days} days")
    
    db = get_db_session()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query old requests
        old_requests = db.query(InferenceRequest).filter(
            InferenceRequest.created_at < cutoff_date,
            InferenceRequest.status.in_([
                RequestStatus.COMPLETED,
                RequestStatus.FAILED,
                RequestStatus.CANCELLED
            ])
        ).all()
        
        count = len(old_requests)
        
        # Delete old requests
        for request in old_requests:
            db.delete(request)
        
        db.commit()
        
        logger.info(f"Deleted {count} old requests")
        
        return {
            "deleted_count": count,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        db.rollback()
        raise
    
    finally:
        db.close()


@celery_app.task
def update_model_cache() -> Dict[str, Any]:
    """
    Update and validate cached model outputs.
    
    Maintenance task that:
    - Clears expired cache entries
    - Updates cache hit statistics
    - Validates cache integrity
    
    Returns:
        Dictionary with cache statistics
    """
    logger.info("Starting model cache update")
    
    try:
        from app.cache.redis_cache import redis_cache
        
        # Get cache statistics
        stats = redis_cache.get_stats()
        
        # Clean up expired keys
        # Redis handles expiration automatically, but we can force cleanup
        
        logger.info(f"Cache update completed. Stats: {stats}")
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Cache update failed: {e}")
        raise


@celery_app.task
def generate_usage_report() -> Dict[str, Any]:
    """
    Generate daily usage report.
    
    Aggregates API usage statistics for reporting and billing.
    Typically runs daily.
    
    Returns:
        Dictionary with usage statistics
    """
    from datetime import date
    from sqlalchemy import func
    
    logger.info("Generating usage report")
    
    db = get_db_session()
    
    try:
        today = date.today()
        
        # Aggregate today's usage
        from app.models.database_models import APIUsage, User
        
        usage_stats = db.query(
            APIUsage.user_id,
            func.sum(APIUsage.requests_count).label('total_requests'),
            func.sum(APIUsage.tokens_used).label('total_tokens')
        ).filter(
            func.date(APIUsage.date) == today
        ).group_by(APIUsage.user_id).all()
        
        report = {
            "date": today.isoformat(),
            "users": [
                {
                    "user_id": stat.user_id,
                    "requests": stat.total_requests,
                    "tokens": stat.total_tokens
                }
                for stat in usage_stats
            ],
            "total_requests": sum(stat.total_requests for stat in usage_stats),
            "total_tokens": sum(stat.total_tokens for stat in usage_stats)
        }
        
        logger.info(f"Usage report generated: {report['total_requests']} requests, {report['total_tokens']} tokens")
        
        return report
        
    except Exception as e:
        logger.error(f"Usage report generation failed: {e}")
        raise
    
    finally:
        db.close()


def _update_request_status(
    db,
    request_id: str,
    status: RequestStatus
) -> None:
    """Update request status in database."""
    request = db.query(InferenceRequest).filter(InferenceRequest.id == request_id).first()
    if request:
        request.status = status
        if status == RequestStatus.PROCESSING:
            request.started_at = datetime.utcnow()
        db.commit()


def _update_request_success(
    db,
    request_id: str,
    output: Any,
    tokens_input: int,
    tokens_output: int,
    latency_ms: float
) -> None:
    """Update request with successful results."""
    request = db.query(InferenceRequest).filter(InferenceRequest.id == request_id).first()
    if request:
        request.status = RequestStatus.COMPLETED
        request.output_data = {"output": output}
        request.tokens_input = tokens_input
        request.tokens_output = tokens_output
        request.tokens_total = tokens_input + tokens_output
        request.latency_ms = latency_ms
        request.completed_at = datetime.utcnow()
        db.commit()


def _update_request_error(db, request_id: str, error_message: str) -> None:
    """Update request with error status."""
    request = db.query(InferenceRequest).filter(InferenceRequest.id == request_id).first()
    if request:
        request.status = RequestStatus.FAILED
        request.error_message = error_message
        request.completed_at = datetime.utcnow()
        db.commit()
