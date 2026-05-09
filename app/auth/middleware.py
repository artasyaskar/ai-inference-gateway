"""
Authentication middleware for AI Inference Gateway.

Provides FastAPI dependencies and middleware for protecting routes
with JWT Bearer token authentication.
"""

import logging
from typing import Optional, Callable, List
from functools import wraps

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.auth.jwt_handler import verify_token, TokenData, create_access_token
from app.config import settings
from app.exceptions import AuthenticationError, AuthorizationError

# Configure logging
logger = logging.getLogger(__name__)

# HTTP Bearer security scheme for OpenAPI documentation
security = HTTPBearer(
    scheme_name="Bearer",
    description="Enter your JWT token",
    auto_error=False  # We'll handle errors manually for better control
)


class AuthMiddleware:
    """
    FastAPI middleware for JWT authentication.
    
    Can be applied globally to protect all routes, or selectively
    using the require_auth decorator.
    """
    
    def __init__(
        self,
        app=None,
        excluded_paths: Optional[List[str]] = None,
        required_scopes: Optional[List[str]] = None
    ):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI app instance (passed automatically by add_middleware)
            excluded_paths: URL paths to exclude from authentication (e.g., ['/health', '/docs'])
            required_scopes: Default scopes required for all protected routes
        """
        self.app = app
        self.excluded_paths = excluded_paths or [
            "/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/models"  # Allow listing models without auth
        ]
        self.required_scopes = required_scopes or []
    
    async def __call__(self, scope, receive, send):
        """
        Process each request through authentication check (ASGI middleware style).
        
        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        from starlette.requests import Request
        
        request = Request(scope, receive)
        
        # Skip authentication for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            await self.app(scope, receive, send)
            return
        
        # Extract and validate token
        try:
            token_data = await self._extract_and_verify_token(request)
            request.state.user = token_data  # Attach user to request state
            
        except AuthenticationError as e:
            logger.warning(f"Authentication failed for {path}: {e.message}")
            from starlette.responses import JSONResponse
            response = JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "AUTHENTICATION_REQUIRED",
                        "message": e.message
                    }
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
            await response(scope, receive, send)
            return
        
        except AuthorizationError as e:
            logger.warning(f"Authorization failed for {path}: {e.message}")
            from starlette.responses import JSONResponse
            response = JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": e.error_code,
                        "message": e.message,
                        "details": e.details
                    }
                }
            )
            await response(scope, receive, send)
            return
        
        # Continue to next middleware/route
        await self.app(scope, receive, send)
    
    async def _extract_and_verify_token(self, request: Request) -> TokenData:
        """
        Extract Bearer token from Authorization header and verify it.
        
        Args:
            request: FastAPI request
        
        Returns:
            TokenData: Verified token data
        
        Raises:
            AuthenticationError: If token is missing or invalid
        """
        # Get Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            raise AuthenticationError("Authorization header is required")
        
        # Check Bearer scheme
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError(
                "Invalid authorization header format. Expected: 'Bearer <token>'"
            )
        
        token = parts[1]
        
        # Verify the token
        return verify_token(token, token_type="access")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenData:
    """
    FastAPI dependency to extract and verify the current user from JWT token.
    
    Use this in route dependencies to require authentication:
    
    @app.get("/protected")
    async def protected_route(user: TokenData = Depends(get_current_user)):
        return {"message": f"Hello {user.user_id}"}
    
    Args:
        credentials: HTTP Authorization credentials from Bearer token
    
    Returns:
        TokenData: Verified user token data
    
    Raises:
        HTTPException: 401 if authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": "Authorization header with Bearer token is required"
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    try:
        token_data = verify_token(token, token_type="access")
        return token_data
        
    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": e.error_code,
                    "message": e.message,
                    "details": e.details
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """
    FastAPI dependency for optional authentication.
    
    Returns user data if authenticated, None otherwise.
    Useful for endpoints that work with or without authentication.
    
    Example:
        @app.get("/items")
        async def get_items(user: Optional[TokenData] = Depends(get_current_user_optional)):
            if user:
                return {"items": get_user_items(user.user_id)}
            return {"items": get_public_items()}
    """
    if credentials is None:
        return None
    
    try:
        return verify_token(credentials.credentials, token_type="access")
    except AuthenticationError:
        return None


def require_auth(scopes: Optional[List[str]] = None) -> Callable:
    """
    Decorator to require specific scopes for route access.
    
    Can be used as a decorator on FastAPI route functions:
    
    @app.post("/admin/users")
    @require_auth(scopes=["admin:users:write"])
    async def create_user(...):
        ...
    
    Args:
        scopes: List of required scopes. User must have ALL specified scopes.
    
    Returns:
        Callable: Decorator function
    """
    required_scopes = scopes or []
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs or dependencies
            user = kwargs.get('user') or kwargs.get('current_user')
            
            if user is None:
                # Try to find in args (less common)
                for arg in args:
                    if isinstance(arg, TokenData):
                        user = arg
                        break
            
            if user is None:
                raise AuthorizationError(
                    "Authentication required but no user context found"
                )
            
            # Check scopes if required
            if required_scopes:
                user_scopes = set(user.scopes or [])
                missing_scopes = set(required_scopes) - user_scopes
                
                if missing_scopes:
                    raise AuthorizationError(
                        f"Insufficient permissions. Missing scopes: {', '.join(missing_scopes)}",
                        details={"missing_scopes": list(missing_scopes)}
                    )
            
            # Call the actual function
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def require_tier(min_tier: str) -> Callable:
    """
    Decorator to require minimum subscription tier.
    
    Tiers in order: free < pro < enterprise
    
    Args:
        min_tier: Minimum required tier
    
    Returns:
        Callable: Decorator function
    """
    tier_levels = {"free": 0, "pro": 1, "enterprise": 2}
    min_level = tier_levels.get(min_tier.lower(), 0)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('user') or kwargs.get('current_user')
            
            if user is None:
                for arg in args:
                    if isinstance(arg, TokenData):
                        user = arg
                        break
            
            if user is None:
                raise AuthorizationError("Authentication required")
            
            user_level = tier_levels.get(user.tier.lower(), 0)
            
            if user_level < min_level:
                raise AuthorizationError(
                    f"This feature requires {min_tier} tier or higher. "
                    f"Current tier: {user.tier}",
                    details={
                        "required_tier": min_tier,
                        "current_tier": user.tier
                    }
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Common scope definitions
class Scopes:
    """Standard permission scopes for the gateway."""
    
    INFERENCE_READ = "inference:read"
    INFERENCE_WRITE = "inference:write"
    MODELS_READ = "models:read"
    MODELS_WRITE = "models:write"
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    
    # Tier-based scope groups
    FREE_TIER = [INFERENCE_READ, INFERENCE_WRITE, MODELS_READ]
    PRO_TIER = FREE_TIER + [MODELS_WRITE, USER_READ]
    ENTERPRISE_TIER = PRO_TIER + [USER_WRITE, ADMIN_READ]
    ADMIN_TIER = ENTERPRISE_TIER + [ADMIN_WRITE]
