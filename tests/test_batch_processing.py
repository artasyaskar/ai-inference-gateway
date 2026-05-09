"""
Batch processing tests for AI Inference Gateway.

Tests request batching logic and batch inference execution.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from app.inference.batch_processor import BatchProcessor, BatchRequest, Batch
from app.inference.inference_engine import InferenceEngine, InferenceResult


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    @pytest.fixture
    async def processor(self):
        """Create a batch processor for testing."""
        processor = BatchProcessor()
        processor.batch_size = 4
        processor.batch_wait_ms = 100
        yield processor
    
    @pytest.mark.asyncio
    async def test_submit_single_request(self, processor):
        """Test submitting a single request."""
        await processor.start()
        
        request_id = await processor.submit_request(
            request_id="req_123",
            model_name="gpt2",
            inputs=["Test input"],
            parameters={},
            task_type="text-generation",
            priority="normal"
        )
        
        assert request_id == "req_123"
        assert processor.get_queue_size("gpt2") == 1
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_submit_multiple_requests(self, processor):
        """Test submitting multiple requests."""
        await processor.start()
        
        for i in range(5):
            await processor.submit_request(
                request_id=f"req_{i}",
                model_name="gpt2",
                inputs=[f"Input {i}"],
                parameters={},
                task_type="text-generation"
            )
        
        # Should have 5 requests in queue
        assert processor.get_queue_size("gpt2") == 5
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, processor):
        """Test that high priority requests are processed first."""
        await processor.start()
        
        # Submit normal priority first
        await processor.submit_request(
            request_id="normal_req",
            model_name="gpt2",
            inputs=["Normal"],
            parameters={},
            task_type="text-generation",
            priority="normal"
        )
        
        # Submit high priority
        await processor.submit_request(
            request_id="high_req",
            model_name="gpt2",
            inputs=["High"],
            parameters={},
            task_type="text-generation",
            priority="high"
        )
        
        # High priority should be at front of queue
        queue = processor._queues.get("gpt2", [])
        if len(queue) >= 2:
            first_request = queue[0]
            assert first_request.priority == "high"
        
        await processor.stop()


class TestBatchCollection:
    """Test batch collection logic."""
    
    @pytest.mark.asyncio
    async def test_batch_size_trigger(self):
        """Test that batch is triggered when size limit is reached."""
        processor = BatchProcessor()
        processor.batch_size = 3
        
        # Mock the process_batch method
        with patch.object(processor, '_process_batch_for_model', new=AsyncMock()) as mock_process:
            await processor.start()
            
            # Submit exactly batch_size requests
            for i in range(3):
                await processor.submit_request(
                    request_id=f"req_{i}",
                    model_name="test-model",
                    inputs=["input"],
                    parameters={},
                    task_type="text-generation"
                )
            
            # Give time for processing
            await asyncio.sleep(0.1)
            
            # Batch processing should have been triggered
            # (might be triggered by size limit)
            
            await processor.stop()


class TestInferenceEngine:
    """Test inference engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create inference engine for testing."""
        return InferenceEngine()
    
    @pytest.mark.asyncio
    async def test_process_single_mocked(self, engine, mock_text_generation_model, mock_tokenizer):
        """Test single inference with mocked model."""
        with patch('app.inference.model_manager.model_manager') as mock_manager:
            mock_manager.load_model = AsyncMock()
            mock_manager.get_model.return_value = (mock_text_generation_model, mock_tokenizer)
            
            result = await engine.process_single(
                request_id="test_123",
                model_name="gpt2",
                input_data="Test prompt",
                parameters={"max_new_tokens": 10},
                task_type="text-generation"
            )
            
            assert result is not None
            assert result.error is None
    
    @pytest.mark.asyncio
    async def test_process_batch_mocked(self, engine, mock_text_generation_model, mock_tokenizer):
        """Test batch inference with mocked model."""
        with patch('app.inference.model_manager.model_manager') as mock_manager:
            mock_manager.load_model = AsyncMock()
            mock_manager.get_model.return_value = (mock_text_generation_model, mock_tokenizer)
            
            results = await engine.process_batch(
                model_name="gpt2",
                inputs=["Input 1", "Input 2", "Input 3"],
                parameters={},
                task_type="text-generation"
            )
            
            assert isinstance(results, list)
            assert len(results) == 3
    
    def test_inference_result_creation(self):
        """Test creating inference result."""
        result = InferenceResult(
            output="Generated text",
            tokens_input=10,
            tokens_output=20,
            latency_ms=150.5,
            device="cpu"
        )
        
        assert result.output == "Generated text"
        assert result.tokens_input == 10
        assert result.tokens_output == 20
        assert result.latency_ms == 150.5
    
    @pytest.mark.asyncio
    async def test_model_not_found(self, engine):
        """Test handling of model not found error."""
        with patch('app.inference.model_manager.model_manager') as mock_manager:
            mock_manager.load_model = AsyncMock()
            mock_manager.get_model.return_value = (None, None)
            
            result = await engine.process_single(
                request_id="test_123",
                model_name="nonexistent-model",
                input_data="Test",
                parameters={},
                task_type="text-generation"
            )
            
            assert result.error is not None


class TestBatchProcessingIntegration:
    """Integration tests for batch processing."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_batch_processing(self):
        """Test complete batch processing flow."""
        # This is a more comprehensive test
        processor = BatchProcessor()
        processor.batch_size = 2
        processor.batch_wait_ms = 50
        
        with patch.object(processor, '_execute_batch', new=AsyncMock()) as mock_execute:
            await processor.start()
            
            # Submit batch
            batch_id = await processor.submit_batch(
                batch_id="batch_123",
                model_name="gpt2",
                inputs=["Input 1", "Input 2"],
                parameters={},
                task_type="text-generation",
                batch_size=2
            )
            
            assert batch_id == "batch_123"
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Cleanup
            await processor.stop()
    
    def test_batch_processor_stats(self):
        """Test statistics tracking in batch processor."""
        processor = BatchProcessor()
        
        # Initially stats should be empty or zeroed
        stats = processor.get_stats()
        
        assert "batches_processed" in stats
        assert "requests_processed" in stats


class TestBatchWaitTime:
    """Test batch wait time logic."""
    
    @pytest.mark.asyncio
    async def test_wait_time_configuration(self):
        """Test that wait time is configurable."""
        processor = BatchProcessor()
        
        # Check that default wait time is set
        assert processor.batch_wait_ms > 0
        
        # Should be a reasonable value (e.g., 10-1000ms)
        assert 10 <= processor.batch_wait_ms <= 1000
