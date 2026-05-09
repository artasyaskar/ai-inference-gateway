"""
Celery worker entry point for AI Inference Gateway.

This script starts the Celery worker for processing background tasks.
Run with: celery -A celery_worker worker --loglevel=info
"""

import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.celery_tasks.inference_tasks import celery_app
from app.config import settings
from app.monitoring.logging_config import setup_logging

# Setup logging
setup_logging()

# Configure Celery worker from settings
celery_app.conf.update(
    worker_concurrency=settings.CELERY_CONCURRENCY,
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
)

if __name__ == "__main__":
    # Start worker
    celery_app.start()
