"""
API module for AI Inference Gateway.

Contains REST API endpoints for authentication, inference, and health checks.
"""

from app.api.routes import api_router
from app.api.health_check import router as health_router
from app.api.inference import router as inference_router

__all__ = ["api_router", "health_router", "inference_router"]
