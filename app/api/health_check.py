"""
Health check endpoints for AI Inference Gateway.

Provides system health monitoring, liveness and readiness probes,
and component status information for monitoring and orchestration.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.database import get_db
from app.models.schemas import HealthResponse, HealthStatus
from app.config import settings
from app.monitoring.metrics import record_health_check
from app.cache.redis_cache import redis_cache

# Configure logging
logger = logging.getLogger(__name__)

# API Router
router = APIRouter(tags=["Health"], prefix="/health")

# Store application start time for uptime calculation
_START_TIME = time.time()


async def check_database(db: AsyncSession) -> HealthStatus:
    """
    Check database connectivity and response time.
    
    Executes a simple query to verify database is responsive.
    
    Args:
        db: Database session
    
    Returns:
        HealthStatus with database health information
    """
    start_time = time.time()
    
    try:
        # Execute simple query to test connectivity
        result = await db.execute(text("SELECT 1"))
        await result.scalar()
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return HealthStatus(
            status="healthy",
            response_time_ms=round(response_time, 2),
            message="Database connection OK"
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Database health check failed: {e}")
        
        return HealthStatus(
            status="unhealthy",
            response_time_ms=round(response_time, 2),
            message=f"Database connection failed: {str(e)}"
        )


async def check_redis() -> HealthStatus:
    """
    Check Redis connectivity and response time.
    
    Pings Redis server to verify it's accessible.
    
    Returns:
        HealthStatus with Redis health information
    """
    start_time = time.time()
    
    try:
        # Ping Redis to check connectivity
        await redis_cache.ping()
        
        response_time = (time.time() - start_time) * 1000
        
        return HealthStatus(
            status="healthy",
            response_time_ms=round(response_time, 2),
            message="Redis connection OK"
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Redis health check failed: {e}")
        
        return HealthStatus(
            status="unhealthy",
            response_time_ms=round(response_time, 2),
            message=f"Redis connection failed: {str(e)}"
        )


async def check_models() -> HealthStatus:
    """
    Check model service availability.
    
    Returns basic status as model loading is handled
    by the inference engine module.
    
    Returns:
        HealthStatus with model service information
    """
    try:
        # Import here to avoid circular imports
        from app.inference.model_manager import model_manager
        
        loaded_count = len(model_manager.loaded_models)
        available_count = len(model_manager.available_models)
        
        # Determine status based on model availability
        if loaded_count > 0 or available_count > 0:
            status = "healthy"
            message = f"{loaded_count} models loaded, {available_count} available"
        else:
            status = "degraded"
            message = "No models currently loaded"
        
        return HealthStatus(
            status=status,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return HealthStatus(
            status="unhealthy",
            message=f"Model service error: {str(e)}"
        )


def get_system_metrics() -> Dict[str, Any]:
    """
    Gather system resource metrics.
    
    Returns CPU usage, memory usage, and active request count.
    
    Returns:
        Dictionary with system metrics
    """
    import psutil
    
    try:
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics = {
            "cpu_usage_percent": round(cpu_percent, 1),
            "memory_usage_percent": round(memory.percent, 1),
            "memory_available_mb": round(memory.available / (1024 * 1024), 1),
            "memory_total_mb": round(memory.total / (1024 * 1024), 1)
        }
        
        # Add GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_metrics = []
                for i in range(gpu_count):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)
                    gpu_metrics.append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated_mb": round(memory_allocated, 1),
                        "memory_reserved_mb": round(memory_reserved, 1)
                    })
                metrics["gpus"] = gpu_metrics
        except ImportError:
            pass
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {"error": "Unable to collect system metrics"}


def calculate_uptime() -> float:
    """Calculate service uptime in seconds."""
    return time.time() - _START_TIME


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns overall system health status including all component health checks."
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Checks all system components (database, Redis, models) and returns
    aggregated health status. Suitable for load balancer health checks
    and monitoring systems.
    
    Returns:
        HealthResponse with component statuses and system metrics
    
    Status Codes:
        - 200: System is healthy
        - 503: One or more components are unhealthy
    """
    # Check all components concurrently
    db_health, redis_health, model_health = await check_database(db), await check_redis(), await check_models()
    
    # Determine overall status
    statuses = [db_health.status, redis_health.status, model_health.status]
    
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Record health check metrics
    record_health_check(overall_status)
    
    response = HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        uptime_seconds=round(calculate_uptime(), 2),
        components={
            "database": db_health,
            "redis": redis_health,
            "models": model_health
        },
        system=get_system_metrics()
    )
    
    # Log unhealthy status
    if overall_status != "healthy":
        logger.warning(f"Health check returned {overall_status} status")
    
    return response


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes/Docker. Returns 200 if service is running."
)
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe for container orchestration.
    
    This endpoint should return 200 as long as the application
    process is running. It's used by Kubernetes to determine
    if the container should be restarted.
    
    Returns:
        Simple status message
    """
    return {"status": "alive"}


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Readiness check for Kubernetes. Returns 200 if service is ready to accept traffic."
)
async def readiness_probe(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Readiness probe for container orchestration.
    
    Checks if the service is ready to handle requests by verifying
    database connectivity. Used by Kubernetes to add/remove pod
    from service load balancing.
    
    Returns:
        Status and details about readiness
    
    Status Codes:
        - 200: Ready to accept traffic
        - 503: Not ready (dependencies unavailable)
    """
    # Check critical dependencies
    db_health = await check_database(db)
    redis_health = await check_redis()
    
    ready = db_health.status == "healthy" and redis_health.status == "healthy"
    
    status_code = 200 if ready else 503
    status_text = "ready" if ready else "not_ready"
    
    return {
        "status": status_text,
        "checks": {
            "database": db_health.status,
            "redis": redis_health.status
        }
    }


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="System metrics",
    description="Returns current system resource metrics."
)
async def system_metrics() -> Dict[str, Any]:
    """
    Get current system metrics.
    
    Returns CPU, memory, and other system resource usage.
    Useful for monitoring dashboards.
    
    Returns:
        Dictionary with system metrics
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": round(calculate_uptime(), 2),
        **get_system_metrics()
    }
