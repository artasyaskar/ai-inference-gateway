"""
Authentication module for AI Inference Gateway.

Provides JWT token handling and authentication middleware.
"""

from app.auth.jwt_handler import (
    create_access_token,
    decode_token,
    verify_token,
    TokenData,
    create_token_pair
)
from app.auth.middleware import (
    require_auth,
    get_current_user,
    get_current_user_optional,
    AuthMiddleware
)

__all__ = [
    "create_access_token",
    "decode_token",
    "verify_token",
    "TokenData",
    "create_token_pair",
    "require_auth",
    "get_current_user",
    "get_current_user_optional",
    "AuthMiddleware"
]
