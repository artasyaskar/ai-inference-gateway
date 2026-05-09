"""
Pytest configuration and fixtures.

Provides test fixtures for database, authentication, and mocked models.
"""

import os
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/inference_gateway_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Use DB 15 for tests
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-do-not-use-in-production"

from app.main import app
from app.database import Base, get_db
from app.config import settings

# Test database URL
TEST_DATABASE_URL = settings.database_async_url

# Create test engine
engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=NullPool,
    future=True,
    echo=False
)

# Create test session factory
TestingSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


@pytest_asyncio.fixture(scope="session")
async def setup_database():
    """
    Setup test database - create all tables.
    
    Runs once per test session.
    """
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Drop all tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(setup_database):
    """
    Create a fresh database session for each test.
    
    Automatically rolls back after each test.
    """
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
def client(db_session):
    """
    Create a test client with overridden database dependency.
    """
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clear overrides after test
    app.dependency_overrides.clear()


@pytest.fixture
def sample_api_key():
    """Sample API key for testing."""
    return "ak_test_sampleapikey123456"


@pytest.fixture
def sample_jwt_payload():
    """Sample JWT payload for testing."""
    return {
        "sub": "user_test_123",
        "api_key": "ak_test_sampleapikey123456",
        "tier": "pro",
        "scopes": ["inference:read", "inference:write"],
        "type": "access"
    }


@pytest.fixture
def mock_text_generation_model():
    """Mock text generation model for testing."""
    class MockModel:
        def generate(self, **kwargs):
            class Output:
                def __init__(self):
                    self.sequences = [[1, 2, 3, 4, 5]]  # Token IDs
            return Output()
        
        def __call__(self, **kwargs):
            return type('obj', (object,), {
                'logits': [[0.1, 0.2, 0.7]]
            })()
    
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = 0
        
        def __call__(self, text, **kwargs):
            class Output:
                def __init__(self):
                    self.input_ids = [[1, 2, 3]]
                    self.attention_mask = [[1, 1, 1]]
            return Output()
        
        def decode(self, token_ids, **kwargs):
            return "Generated text output"
        
        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]
    
    return MockTokenizer()


@pytest_asyncio.fixture
async def create_test_user(db_session):
    """
    Create a test user in the database.
    
    Returns the created user.
    """
    from app.models.database_models import User, UserTier
    
    user = User(
        id="user_test_123",
        api_key="ak_test_sampleapikey123456",
        tier=UserTier.PRO,
        is_active=True
    )
    
    db_session.add(user)
    await db_session.commit()
    
    return user


@pytest.fixture
def auth_headers(create_test_user):
    """
    Generate authentication headers with JWT token.
    
    Requires create_test_user fixture.
    """
    from app.auth.jwt_handler import create_access_token
    
    token = create_access_token(
        user_id="user_test_123",
        api_key="ak_test_sampleapikey123456",
        tier="pro",
        scopes=["inference:read", "inference:write"]
    )
    
    return {"Authorization": f"Bearer {token}"}
