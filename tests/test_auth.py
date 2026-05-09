"""
Authentication tests for AI Inference Gateway.

Tests JWT token generation, validation, and authentication endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_handler import (
    create_access_token,
    verify_token,
    create_token_pair,
    decode_token
)
from app.exceptions import AuthenticationError


class TestJWTHandler:
    """Test JWT token creation and validation."""
    
    def test_create_access_token(self, sample_jwt_payload):
        """Test creating an access token."""
        token = create_access_token(
            user_id=sample_jwt_payload["sub"],
            api_key=sample_jwt_payload["api_key"],
            tier=sample_jwt_payload["tier"],
            scopes=sample_jwt_payload["scopes"]
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_valid_token(self, sample_jwt_payload):
        """Test verifying a valid token."""
        token = create_access_token(
            user_id=sample_jwt_payload["sub"],
            api_key=sample_jwt_payload["api_key"],
            tier=sample_jwt_payload["tier"],
            scopes=sample_jwt_payload["scopes"]
        )
        
        token_data = verify_token(token, token_type="access")
        
        assert token_data.user_id == sample_jwt_payload["sub"]
        assert token_data.api_key == sample_jwt_payload["api_key"]
        assert token_data.tier == sample_jwt_payload["tier"]
        assert token_data.scopes == sample_jwt_payload["scopes"]
    
    def test_verify_invalid_token(self):
        """Test verifying an invalid token."""
        with pytest.raises(AuthenticationError):
            verify_token("invalid_token", token_type="access")
    
    def test_create_token_pair(self, sample_jwt_payload):
        """Test creating token pair (access + refresh)."""
        tokens = create_token_pair(
            user_id=sample_jwt_payload["sub"],
            api_key=sample_jwt_payload["api_key"],
            tier=sample_jwt_payload["tier"],
            scopes=sample_jwt_payload["scopes"]
        )
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "token_type" in tokens
        assert tokens["token_type"] == "bearer"
        assert "expires_in" in tokens
    
    def test_decode_token(self, sample_jwt_payload):
        """Test decoding a token without verification."""
        token = create_access_token(
            user_id=sample_jwt_payload["sub"],
            api_key=sample_jwt_payload["api_key"]
        )
        
        payload = decode_token(token)
        
        assert payload["sub"] == sample_jwt_payload["sub"]
        assert payload["api_key"] == sample_jwt_payload["api_key"]


class TestAuthEndpoints:
    """Test authentication API endpoints."""
    
    @pytest.mark.asyncio
    async def test_login_success(self, client, create_test_user):
        """Test successful login with valid API key."""
        response = client.post(
            "/api/v1/auth/login",
            json={"api_key": "ak_test_sampleapikey123456"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_api_key(self, client):
        """Test login with invalid API key."""
        response = client.post(
            "/api/v1/auth/login",
            json={"api_key": "invalid_key"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, client, auth_headers, create_test_user):
        """Test getting current user info."""
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_test_123"
        assert data["tier"] == "pro"
    
    def test_protected_endpoint_no_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401


class TestAuthMiddleware:
    """Test authentication middleware."""
    
    def test_valid_token_header(self, client, auth_headers):
        """Test request with valid Authorization header."""
        # This endpoint requires auth
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        # Should either succeed or fail for other reasons (like missing user)
        # but not for authentication
        assert response.status_code != 401
    
    def test_missing_auth_header(self, client):
        """Test request without Authorization header."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "AUTHENTICATION_REQUIRED"
    
    def test_malformed_auth_header(self, client):
        """Test request with malformed Authorization header."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "InvalidFormat token123"}
        )
        
        assert response.status_code == 401


class TestTokenScopes:
    """Test token scope validation."""
    
    def test_token_with_scopes(self, sample_jwt_payload):
        """Test token contains correct scopes."""
        token = create_access_token(
            user_id=sample_jwt_payload["sub"],
            scopes=["inference:read", "admin:write"]
        )
        
        token_data = verify_token(token)
        
        assert "inference:read" in token_data.scopes
        assert "admin:write" in token_data.scopes
    
    def test_token_without_scopes(self):
        """Test token without scopes."""
        token = create_access_token(user_id="user_123")
        
        token_data = verify_token(token)
        
        assert token_data.scopes == []
