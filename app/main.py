"""
Main FastAPI application for AI Inference Gateway.

Entry point for the web service providing:
- FastAPI application initialization
- Middleware configuration
- Database setup
- API route registration
- CORS and error handling
- Prometheus metrics endpoint
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.config import settings
from app.database import init_db, close_db_connection
from app.api.routes import api_router
from app.auth.middleware import AuthMiddleware
from app.monitoring.logging_config import setup_logging, log_request, set_request_context, clear_request_context
from app.monitoring.metrics import MetricsMiddleware
from app.inference.batch_processor import batch_processor
from app.exceptions import GatewayException

# Setup logging
setup_logging()

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Initialize database connections
    - Start batch processor
    - Cleanup on shutdown
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database
    await init_db()
    
    # Start batch processor
    await batch_processor.start()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop batch processor
    await batch_processor.stop()
    
    # Close database connections
    await close_db_connection()
    
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description="Scalable AI inference service with request batching, async processing, and multi-tenant support",
        version=settings.APP_VERSION,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"]
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add authentication middleware
    app.add_middleware(AuthMiddleware)
    
    # Add metrics middleware for request tracking
    app.add_middleware(MetricsMiddleware)
    
    # Include API routes
    app.include_router(api_router)
    
    # Mount Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Add error handlers
    add_error_handlers(app)
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Log all HTTP requests with timing.
        
        Tracks request duration and logs structured data.
        """
        import time
        from uuid import uuid4
        
        # Generate request ID
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # Set logging context
        set_request_context(request_id=request_id)
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request
            log_request(
                logger=logger,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                extra={"request_id": request_id}
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={"request_id": request_id, "duration_ms": duration_ms}
            )
            raise
            
        finally:
            clear_request_context()
    
    return app


def add_error_handlers(app: FastAPI) -> None:
    """
    Add exception handlers to the application.
    
    Args:
        app: FastAPI application
    """
    
    @app.exception_handler(GatewayException)
    async def gateway_exception_handler(request: Request, exc: GatewayException):
        """
        Handle custom gateway exceptions.
        
        Returns structured error responses with proper HTTP status codes.
        """
        logger.error(
            f"Gateway exception: {exc.error_code} - {exc.message}",
            extra={
                "error_code": exc.error_code,
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle unexpected exceptions.
        
        Returns generic 500 error without exposing internal details.
        """
        request_id = getattr(request.state, 'request_id', None)
        
        logger.exception(
            f"Unhandled exception: {str(exc)}",
            extra={"request_id": request_id}
        )
        
        # In production, don't expose internal error details
        message = "Internal server error" if settings.is_production else str(exc)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": message,
                    "request_id": request_id
                }
            }
        )


# Create application instance
app = create_application()


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - returns API information.
    
    Useful for quick health checks and API discovery.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn when executed directly
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )
