"""
AI Inference Gateway

A scalable FastAPI-based service for distributed AI model inference with
request batching, async processing, load balancing, and multi-tenant support.

Modules:
    - auth: JWT authentication and authorization
    - api: REST API endpoints
    - inference: Model loading and inference engine
    - celery_tasks: Async task processing
    - monitoring: Metrics and logging
    - cache: Redis caching utilities
    - models: Pydantic schemas and database models
"""

__version__ = "1.0.0"
__author__ = "AI Inference Gateway Team"

import logging
from app.config import settings

# Configure root logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"AI Inference Gateway v{__version__} initialized")
