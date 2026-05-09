"""
Monitoring module for AI Inference Gateway.

Provides Prometheus metrics, structured logging, and health monitoring.
"""

from app.monitoring.metrics import (
    record_inference_request,
    record_inference_latency,
    record_inference_tokens,
    record_batch_size,
    record_batch_wait_time,
    record_auth_attempt,
    record_health_check,
    metrics_registry,
    get_prometheus_metrics
)
from app.monitoring.logging_config import (
    setup_logging,
    get_logger,
    StructuredLogFormatter
)

__all__ = [
    # Metrics
    "record_inference_request",
    "record_inference_latency",
    "record_inference_tokens",
    "record_batch_size",
    "record_batch_wait_time",
    "record_auth_attempt",
    "record_health_check",
    "metrics_registry",
    "get_prometheus_metrics",
    # Logging
    "setup_logging",
    "get_logger",
    "StructuredLogFormatter"
]
