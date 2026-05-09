"""
Prometheus metrics for AI Inference Gateway.

Collects and exposes metrics for monitoring and alerting including:
- Request counters and latency histograms
- Batch processing metrics
- Authentication metrics
- Health check status
"""

import logging
from typing import Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create metrics registry
metrics_registry = CollectorRegistry()

# Application info
app_info = Info(
    "app",
    "Application information",
    registry=metrics_registry
)
app_info.info({
    "name": settings.APP_NAME,
    "version": settings.APP_VERSION,
    "environment": settings.ENVIRONMENT
})

# Inference request counter
inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["model", "status"],
    registry=metrics_registry
)

# Inference latency histogram
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Inference request latency in seconds",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=metrics_registry
)

# Tokens used counter
inference_tokens_total = Counter(
    "inference_tokens_total",
    "Total number of tokens processed",
    ["model", "token_type"],
    registry=metrics_registry
)

# Active requests gauge
active_requests = Gauge(
    "active_requests",
    "Number of currently active inference requests",
    ["model"],
    registry=metrics_registry
)

# Batch processing metrics
batch_size_histogram = Histogram(
    "batch_size",
    "Distribution of batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    registry=metrics_registry
)

batch_wait_time_seconds = Histogram(
    "batch_wait_time_seconds",
    "Time spent waiting for batch to fill",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=metrics_registry
)

# Authentication metrics
auth_attempts_total = Counter(
    "auth_attempts_total",
    "Total authentication attempts",
    ["success"],
    registry=metrics_registry
)

# Rate limiting metrics
rate_limit_hits_total = Counter(
    "rate_limit_hits_total",
    "Total rate limit exceeded events",
    ["user_tier"],
    registry=metrics_registry
)

# Health check metrics
health_check_status = Gauge(
    "health_check_status",
    "Health check status (1=healthy, 0=unhealthy, 0.5=degraded)",
    ["component"],
    registry=metrics_registry
)

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"],
    registry=metrics_registry
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=metrics_registry
)

# Model metrics
models_loaded = Gauge(
    "models_loaded",
    "Number of models currently loaded",
    registry=metrics_registry
)

model_memory_bytes = Gauge(
    "model_memory_bytes",
    "Memory used by loaded models",
    ["model"],
    registry=metrics_registry
)

# Queue metrics
queue_size = Gauge(
    "queue_size",
    "Number of pending requests in queue",
    ["model"],
    registry=metrics_registry
)

# Celery metrics
celery_tasks_total = Counter(
    "celery_tasks_total",
    "Total Celery tasks executed",
    ["task_name", "status"],
    registry=metrics_registry
)

celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds",
    "Celery task execution duration",
    ["task_name"],
    registry=metrics_registry
)

# Database metrics
db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
    registry=metrics_registry
)

db_connections_idle = Gauge(
    "db_connections_idle",
    "Idle database connections",
    registry=metrics_registry
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total errors by type",
    ["error_type"],
    registry=metrics_registry
)


def record_inference_request(model: str, status: str) -> None:
    """
    Record an inference request.
    
    Args:
        model: Model name
        status: Request status (pending, completed, failed, timeout)
    """
    inference_requests_total.labels(model=model, status=status).inc()
    
    # Track active requests
    if status == "processing":
        active_requests.labels(model=model).inc()
    elif status in ("completed", "failed", "timeout"):
        active_requests.labels(model=model).dec()


def record_inference_latency(model: str, duration_seconds: float) -> None:
    """
    Record inference latency.
    
    Args:
        model: Model name
        duration_seconds: Latency in seconds
    """
    inference_latency_seconds.labels(model=model).observe(duration_seconds)


def record_inference_tokens(model: str, count: int, token_type: str = "total") -> None:
    """
    Record tokens used in inference.
    
    Args:
        model: Model name
        count: Number of tokens
        token_type: Type of tokens (input, output, total)
    """
    inference_tokens_total.labels(model=model, token_type=token_type).inc(count)


def record_batch_size(size: int) -> None:
    """
    Record the size of a processed batch.
    
    Args:
        size: Number of requests in batch
    """
    batch_size_histogram.observe(size)


def record_batch_wait_time(wait_seconds: float) -> None:
    """
    Record time spent waiting for batch to fill.
    
    Args:
        wait_seconds: Time in seconds
    """
    batch_wait_time_seconds.observe(wait_seconds)


def record_auth_attempt(success: bool) -> None:
    """
    Record an authentication attempt.
    
    Args:
        success: Whether authentication succeeded
    """
    auth_attempts_total.labels(success="true" if success else "false").inc()


def record_rate_limit_hit(tier: str) -> None:
    """
    Record a rate limit hit.
    
    Args:
        tier: User tier that hit the limit
    """
    rate_limit_hits_total.labels(user_tier=tier).inc()


def record_health_check(status: str, component: str = "overall") -> None:
    """
    Record health check status.
    
    Args:
        status: Health status (healthy, unhealthy, degraded)
        component: Component being checked
    """
    # Map status to numeric value
    status_values = {
        "healthy": 1.0,
        "degraded": 0.5,
        "unhealthy": 0.0
    }
    
    value = status_values.get(status, 0.0)
    health_check_status.labels(component=component).set(value)


def record_cache_hit(cache_type: str = "model") -> None:
    """
    Record a cache hit.
    
    Args:
        cache_type: Type of cache
    """
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "model") -> None:
    """
    Record a cache miss.
    
    Args:
        cache_type: Type of cache
    """
    cache_misses_total.labels(cache_type=cache_type).inc()


def update_models_loaded(count: int) -> None:
    """
    Update the count of loaded models.
    
    Args:
        count: Number of loaded models
    """
    models_loaded.set(count)


def update_queue_size(model: str, size: int) -> None:
    """
    Update queue size for a model.
    
    Args:
        model: Model name
        size: Current queue size
    """
    queue_size.labels(model=model).set(size)


def record_celery_task(task_name: str, status: str, duration_seconds: Optional[float] = None) -> None:
    """
    Record Celery task execution.
    
    Args:
        task_name: Name of the task
        status: Task status (started, success, failure, retry)
        duration_seconds: Execution duration (for completed tasks)
    """
    celery_tasks_total.labels(task_name=task_name, status=status).inc()
    
    if duration_seconds is not None:
        celery_task_duration_seconds.labels(task_name=task_name).observe(duration_seconds)


def record_error(error_type: str) -> None:
    """
    Record an error occurrence.
    
    Args:
        error_type: Type of error
    """
    errors_total.labels(error_type=error_type).inc()


def get_prometheus_metrics() -> tuple:
    """
    Generate Prometheus metrics in exposition format.
    
    Returns:
        Tuple of (content_type, content) for HTTP response
    """
    content = generate_latest(metrics_registry)
    return CONTENT_TYPE_LATEST, content


# Convenience function for middleware
class MetricsMiddleware:
    """FastAPI middleware for automatic request metrics collection."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        from starlette.requests import Request
        import time
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Track request
        path = request.url.path
        method = request.method
        
        # Create response wrapper to capture status
        status_code = 200
        
        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, wrapped_send)
        except Exception:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Only record for inference endpoints
            if "/inference" in path:
                # Extract model from path if available
                model = "unknown"
                if "/inference/" in path and path != "/api/v1/inference":
                    parts = path.split("/")
                    if len(parts) > 3:
                        model = parts[-1]  # Last part might be model or request_id
                
                status_label = "success" if status_code < 400 else "error"
                record_inference_request(model, status_label)
                record_inference_latency(model, duration)
