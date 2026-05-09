"""
JWT Token generation and validation for AI Inference Gateway.

Provides secure token creation, decoding, and validation using python-jose.
Supports access tokens with configurable expiration and refresh tokens.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from pydantic import BaseModel, Field

from app.config import settings
from app.exceptions import AuthenticationError

# Configure logging
logger = logging.getLogger(__name__)


class TokenData(BaseModel):
    """Data structure for decoded JWT token contents."""
    
    user_id: str = Field(..., description="Unique user identifier")
    api_key: Optional[str] = Field(None, description="User's API key")
    tier: str = Field(default="free", description="User subscription tier")
    scopes: list[str] = Field(default_factory=list, description="Token scopes/permissions")
    exp: Optional[datetime] = Field(None, description="Token expiration time")
    iat: Optional[datetime] = Field(None, description="Token issued at time")
    jti: Optional[str] = Field(None, description="JWT ID (unique token identifier)")
    
    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if self.exp is None:
            return False
        return datetime.now(timezone.utc) > self.exp


def create_access_token(
    user_id: str,
    api_key: Optional[str] = None,
    tier: str = "free",
    scopes: Optional[list[str]] = None,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT access token for authenticated users.
    
    The token includes claims for user identification, API key, subscription tier,
    and expiration time. All sensitive data is cryptographically signed.
    
    Args:
        user_id: Unique identifier for the user
        api_key: User's API key for additional verification
        tier: User's subscription tier (free, pro, enterprise)
        scopes: List of permission scopes granted to the token
        expires_delta: Custom expiration time (defaults to settings)
        additional_claims: Any additional claims to include
    
    Returns:
        str: Encoded JWT token string
    
    Example:
        >>> token = create_access_token(
        ...     user_id="user_123",
        ...     api_key="ak_live_xxx",
        ...     tier="pro",
        ...     scopes=["inference:read", "inference:write"]
        ... )
    """
    # Use configured expiration if not specified
    if expires_delta is None:
        expires_delta = timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)
    
    # Calculate expiration time with UTC timezone
    expire = datetime.now(timezone.utc) + expires_delta
    issued_at = datetime.now(timezone.utc)
    
    # Build JWT payload (claims)
    to_encode: Dict[str, Any] = {
        "sub": user_id,  # Subject - the user
        "iat": issued_at,  # Issued at time
        "exp": expire,  # Expiration time
        "iss": settings.APP_NAME,  # Issuer
        "aud": settings.APP_NAME,  # Audience
        "type": "access",  # Token type
        "tier": tier,  # Subscription tier
    }
    
    # Add optional claims
    if api_key:
        to_encode["api_key"] = api_key
    
    if scopes:
        to_encode["scopes"] = scopes
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    # Generate unique JWT ID for token revocation tracking
    import uuid
    to_encode["jti"] = str(uuid.uuid4())
    
    try:
        # Encode the claims with the secret key
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        logger.debug(f"Created access token for user {user_id}, expires at {expire}")
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise AuthenticationError("Failed to generate authentication token")


def create_refresh_token(user_id: str) -> str:
    """
    Create a long-lived refresh token for token renewal.
    
    Refresh tokens have a longer expiration than access tokens
    and are used to obtain new access tokens without re-authentication.
    
    Args:
        user_id: User identifier
    
    Returns:
        str: Encoded refresh token
    """
    expire = datetime.now(timezone.utc) + timedelta(days=30)  # 30 days
    issued_at = datetime.now(timezone.utc)
    
    to_encode: Dict[str, Any] = {
        "sub": user_id,
        "iat": issued_at,
        "exp": expire,
        "iss": settings.APP_NAME,
        "aud": settings.APP_NAME,
        "type": "refresh",
    }
    
    import uuid
    to_encode["jti"] = str(uuid.uuid4())
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )


def create_token_pair(
    user_id: str,
    api_key: Optional[str] = None,
    tier: str = "free",
    scopes: Optional[list[str]] = None
) -> Dict[str, str]:
    """
    Create both access and refresh tokens.
    
    Returns a dictionary containing both tokens for immediate use.
    
    Args:
        user_id: User identifier
        api_key: User's API key
        tier: Subscription tier
        scopes: Permission scopes
    
    Returns:
        Dict with 'access_token' and 'refresh_token' keys
    """
    access_token = create_access_token(
        user_id=user_id,
        api_key=api_key,
        tier=tier,
        scopes=scopes
    )
    refresh_token = create_refresh_token(user_id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60  # seconds
    }


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token without verifying signature.
    
    Use with caution - only for inspection purposes. Use verify_token
    for secure token validation.
    
    Args:
        token: JWT token string
    
    Returns:
        Dict containing decoded token claims
    
    Raises:
        AuthenticationError: If token format is invalid
    """
    try:
        # Decode without verification (for inspection only)
        payload = jwt.decode(
            token,
            key="",
            options={"verify_signature": False, "verify_exp": False}
        )
        return payload
    except JWTError as e:
        logger.warning(f"Failed to decode token: {e}")
        raise AuthenticationError(f"Invalid token format: {str(e)}")


def verify_token(token: str, token_type: str = "access") -> TokenData:
    """
    Verify and decode a JWT token with full security checks.
    
    Performs signature verification, expiration check, issuer validation,
    and audience validation. Returns structured token data.
    
    Args:
        token: JWT token string to verify
        token_type: Expected token type ('access' or 'refresh')
    
    Returns:
        TokenData: Structured token data with user information
    
    Raises:
        AuthenticationError: If token is invalid, expired, or verification fails
    
    Example:
        >>> try:
        ...     token_data = verify_token("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...")
        ...     print(f"User: {token_data.user_id}")
        ... except AuthenticationError:
        ...     print("Invalid token")
    """
    try:
        # Decode with full verification
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            issuer=settings.APP_NAME,
            audience=settings.APP_NAME
        )
        
        # Verify token type
        if payload.get("type") != token_type:
            raise AuthenticationError(
                f"Invalid token type. Expected {token_type}, got {payload.get('type')}"
            )
        
        # Extract and validate required claims
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Token missing required 'sub' claim")
        
        # Parse timestamps
        exp_timestamp = payload.get("exp")
        iat_timestamp = payload.get("iat")
        
        exp = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc) if exp_timestamp else None
        iat = datetime.fromtimestamp(iat_timestamp, tz=timezone.utc) if iat_timestamp else None
        
        # Build TokenData
        token_data = TokenData(
            user_id=user_id,
            api_key=payload.get("api_key"),
            tier=payload.get("tier", "free"),
            scopes=payload.get("scopes", []),
            exp=exp,
            iat=iat,
            jti=payload.get("jti")
        )
        
        logger.debug(f"Verified token for user {user_id}")
        return token_data
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise AuthenticationError("Token has expired. Please log in again.")
        
    except jwt.JWTClaimsError as e:
        logger.warning(f"Token claims error: {e}")
        raise AuthenticationError(f"Token claims validation failed: {str(e)}")
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise AuthenticationError("Invalid authentication token")
        
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}")
        raise AuthenticationError("Token verification failed")


def refresh_access_token(refresh_token: str) -> str:
    """
    Create a new access token using a valid refresh token.
    
    Verifies the refresh token and issues a new access token
    with the same user claims.
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        str: New access token
    
    Raises:
        AuthenticationError: If refresh token is invalid
    """
    # Verify the refresh token
    token_data = verify_token(refresh_token, token_type="refresh")
    
    # Create new access token with same user info
    return create_access_token(
        user_id=token_data.user_id,
        api_key=token_data.api_key,
        tier=token_data.tier,
        scopes=token_data.scopes
    )


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Get the expiration time of a token without full verification.
    
    Useful for client-side token management.
    
    Args:
        token: JWT token
    
    Returns:
        datetime: Expiration time, or None if not present
    """
    try:
        payload = decode_token(token)
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            return datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
    except Exception:
        pass
    return None
