"""
Inference endpoint tests for AI Inference Gateway.

Tests inference request submission, status checking, and results.
"""

import pytest
from unittest.mock import patch, AsyncMock

from app.models.schemas import InferenceRequest, TaskType


class TestInferenceEndpoints:
    """Test inference API endpoints."""
    
    @pytest.mark.asyncio
    async def test_submit_inference_request(self, client, auth_headers):
        """Test submitting a single inference request."""
        # Mock the batch processor
        with patch("app.api.inference.batch_processor") as mock_processor:
            mock_processor.submit_request = AsyncMock()
            
            request_data = {
                "model": "gpt2",
                "input": "Once upon a time",
                "task_type": "text-generation",
                "parameters": {
                    "max_length": 50
                }
            }
            
            response = client.post(
                "/api/v1/inference",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 202
            data = response.json()
            assert "request_id" in data
            assert data["model"] == "gpt2"
            assert data["status"] in ["pending", "processing"]
    
    @pytest.mark.asyncio
    async def test_submit_inference_missing_model(self, client, auth_headers):
        """Test submitting inference without model."""
        request_data = {
            "input": "Test input"
            # Missing model
        }
        
        response = client.post(
            "/api/v1/inference",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_submit_batch_inference(self, client, auth_headers):
        """Test submitting batch inference request."""
        with patch("app.api.inference.batch_processor") as mock_processor:
            mock_processor.submit_batch = AsyncMock()
            
            request_data = {
                "model": "all-MiniLM-L6-v2",
                "inputs": [
                    "First text",
                    "Second text",
                    "Third text"
                ],
                "task_type": "embeddings",
                "batch_size": 16
            }
            
            response = client.post(
                "/api/v1/inference/batch",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 202
            data = response.json()
            assert "request_id" in data  # batch_id
            assert data["total_items"] == 3
            assert data["model"] == "all-MiniLM-L6-v2"
    
    @pytest.mark.asyncio
    async def test_get_inference_result_not_found(self, client, auth_headers):
        """Test getting result for non-existent request."""
        response = client.get(
            "/api/v1/inference/non-existent-id",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    @pytest.mark.asyncio
    async def test_inference_unauthorized(self, client):
        """Test inference without authentication."""
        request_data = {
            "model": "gpt2",
            "input": "Test"
        }
        
        response = client.post("/api/v1/inference", json=request_data)
        
        assert response.status_code == 401


class TestInferenceRequestValidation:
    """Test inference request validation."""
    
    def test_valid_inference_request(self):
        """Test creating a valid inference request."""
        request = InferenceRequest(
            model="gpt2",
            input="Test input",
            task_type=TaskType.TEXT_GENERATION,
            parameters={"max_length": 100}
        )
        
        assert request.model == "gpt2"
        assert request.input == "Test input"
        assert request.task_type == TaskType.TEXT_GENERATION
    
    def test_inference_request_auto_request_id(self):
        """Test that request_id is auto-generated if not provided."""
        request = InferenceRequest(
            model="gpt2",
            input="Test"
        )
        
        assert request.request_id is not None
        assert len(request.request_id) > 0
    
    def test_inference_request_invalid_priority(self):
        """Test validation of priority field."""
        with pytest.raises(ValueError):
            InferenceRequest(
                model="gpt2",
                input="Test",
                priority="invalid_priority"  # Should be low, normal, or high
            )


class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_get_model_details(self, client):
        """Test getting details for a specific model."""
        # Note: This test assumes at least one model is available
        response = client.get("/api/v1/models/gpt2")
        
        # Should either return model info or 404 if model doesn't exist
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "task_types" in data
    
    def test_get_model_not_found(self, client):
        """Test getting details for non-existent model."""
        response = client.get("/api/v1/models/non-existent-model-12345")
        
        assert response.status_code == 404


class TestInferenceSchemas:
    """Test inference-related Pydantic schemas."""
    
    def test_inference_response_creation(self):
        """Test creating inference response."""
        from app.models.schemas import InferenceResponse, InferenceResult, RequestStatus
        
        result = InferenceResult(
            output="Generated text",
            tokens_used=50,
            processing_time_ms=100.5
        )
        
        response = InferenceResponse(
            request_id="test-123",
            model="gpt2",
            status=RequestStatus.COMPLETED,
            results=[result],
            metadata={"latency_ms": 120.0}
        )
        
        assert response.request_id == "test-123"
        assert len(response.results) == 1
        assert response.results[0].tokens_used == 50
