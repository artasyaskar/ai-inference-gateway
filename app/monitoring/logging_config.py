"""
Structured logging configuration for AI Inference Gateway.

Provides JSON-formatted logging for production environments with
request context tracking and correlation IDs.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

from app.config import settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class StructuredLogFormatter(logging.Formatter):
    """
    JSON structured log formatter for production logging.
    
    Outputs log records as JSON objects with standardized fields:
    - timestamp: ISO format timestamp
    - level: Log level
    - logger: Logger name
    - message: Log message
    - request_id: Request correlation ID (if available)
    - user_id: User identifier (if available)
    - extra: Additional context fields
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "levelname", "levelno", "pathname",
                    "filename", "module", "exc_info", "exc_text", "stack_info",
                    "lineno", "funcName", "created", "msecs", "relativeCreated",
                    "thread", "threadName", "processName", "process", "message"
                ):
                    log_data[key] = value
        
        # Add source location for debug/trace levels
        if record.levelno <= logging.DEBUG:
            log_data["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }
        
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.
    
    Adds ANSI color codes to log output for better readability.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m"
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with color codes."""
        # Get color
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    use_json: Optional[bool] = None
) -> None:
    """
    Setup application logging configuration.
    
    Configures root logger with appropriate handlers and formatters
    based on environment (JSON for production, colored for dev).
    
    Args:
        log_level: Override log level (uses settings.LOG_LEVEL if not provided)
        use_json: Force JSON formatting (auto-detected from environment if not provided)
    """
    level = (log_level or settings.LOG_LEVEL).upper()
    
    # Determine if we should use JSON formatting
    if use_json is None:
        use_json = settings.is_production
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    if use_json:
        # JSON formatter for production
        console_handler.setFormatter(StructuredLogFormatter())
    else:
        # Colored formatter for development
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt))
    
    root_logger.addHandler(console_handler)
    
    # File handler for production (rotating log)
    if settings.is_production:
        file_handler = logging.handlers.RotatingFileHandler(
            "logs/app.log",
            maxBytes=10_000_000,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredLogFormatter())
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
    
    root_logger.info(f"Logging configured (level={level}, json={use_json})")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_request_context(request_id: Optional[str] = None, user_id: Optional[str] = None) -> None:
    """
    Set request context for logging.
    
    Sets context variables that will be included in all log messages
    within the current context (async task or thread).
    
    Args:
        request_id: Unique request identifier
        user_id: User identifier
    """
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context() -> None:
    """Clear request context variables."""
    request_id_var.set(None)
    user_id_var.set(None)


class RequestContextFilter(logging.Filter):
    """Filter that adds request context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to record."""
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an HTTP request with structured data.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        user_id: User identifier
        extra: Additional fields to log
    """
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2)
    }
    
    if user_id:
        log_data["user_id"] = user_id
    
    if extra:
        log_data.update(extra)
    
    # Log at appropriate level based on status
    if status_code >= 500:
        logger.error(f"Request failed: {log_data}")
    elif status_code >= 400:
        logger.warning(f"Request error: {log_data}")
    else:
        logger.info(f"Request completed: {log_data}")


def log_inference_request(
    logger: logging.Logger,
    request_id: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    latency_ms: float,
    cached: bool = False,
    user_id: Optional[str] = None
) -> None:
    """
    Log an inference request with structured data.
    
    Args:
        logger: Logger instance
        request_id: Request identifier
        model: Model used
        tokens_input: Input token count
        tokens_output: Output token count
        latency_ms: Processing latency
        cached: Whether result was cached
        user_id: User identifier
    """
    log_data = {
        "request_id": request_id,
        "model": model,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_total": tokens_input + tokens_output,
        "latency_ms": round(latency_ms, 2),
        "cached": cached
    }
    
    if user_id:
        log_data["user_id"] = user_id
    
    logger.info(f"Inference completed: {log_data}")
