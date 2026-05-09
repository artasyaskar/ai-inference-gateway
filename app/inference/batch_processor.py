"""
Batch processor for AI Inference Gateway.

Implements intelligent request batching to maximize throughput.
Collects requests over a time window and processes them together.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import time

from app.config import settings
from app.inference.inference_engine import inference_engine
from app.monitoring.metrics import record_batch_size, record_batch_wait_time

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Individual request waiting to be batched."""
    request_id: str
    model_name: str
    input_data: Any
    parameters: Dict[str, Any]
    task_type: str
    priority: str = "normal"
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    future: Optional[asyncio.Future] = None


@dataclass
class Batch:
    """A batch of requests ready for processing."""
    batch_id: str
    model_name: str
    task_type: str
    requests: List[BatchRequest]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def size(self) -> int:
        """Number of requests in the batch."""
        return len(self.requests)


class BatchProcessor:
    """
    Intelligent batch processor for inference requests.
    
    Implements request aggregation with:
    - Time-based batching (wait for configurable time window)
    - Size-based batching (process when batch reaches size limit)
    - Priority handling (high priority requests may trigger early processing)
    
    This maximizes GPU/CPU utilization by processing multiple
    requests together in a single model forward pass.
    """
    
    def __init__(self):
        # Configuration
        self.batch_size = settings.BATCH_SIZE
        self.batch_wait_ms = settings.BATCH_WAIT_TIME_MS
        self.max_concurrent_batches = 4
        
        # Request queues per model
        # Dict: model_name -> deque of BatchRequest
        self._queues: Dict[str, deque] = {}
        
        # Processing state
        self._processing: Dict[str, bool] = {}  # model_name -> is_processing
        self._batch_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self._stats = {
            "batches_processed": 0,
            "requests_processed": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0
        }
        
        # Lock for thread-safe queue operations
        self._lock = asyncio.Lock()
        
        # Background batch collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the batch processor background tasks."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._batch_collection_loop())
        logger.info(f"Batch processor started (batch_size={self.batch_size}, wait_ms={self.batch_wait_ms})")
    
    async def stop(self) -> None:
        """Stop the batch processor and process remaining requests."""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining requests
        await self._flush_all_queues()
        
        logger.info("Batch processor stopped")
    
    async def submit_request(
        self,
        request_id: str,
        model_name: str,
        inputs: List[Any],
        parameters: Dict[str, Any],
        task_type: str,
        priority: str = "normal"
    ) -> str:
        """
        Submit a request to the batch processor.
        
        Args:
            request_id: Unique request identifier
            model_name: Target model
            inputs: List of input items
            parameters: Inference parameters
            task_type: Type of task
            priority: Request priority (low, normal, high)
        
        Returns:
            str: Request ID
        """
        async with self._lock:
            # Initialize queue for this model if needed
            if model_name not in self._queues:
                self._queues[model_name] = deque()
                self._processing[model_name] = False
            
            # Create batch request for each input
            for i, input_data in enumerate(inputs):
                req_id = f"{request_id}_{i}" if len(inputs) > 1 else request_id
                
                batch_req = BatchRequest(
                    request_id=req_id,
                    model_name=model_name,
                    input_data=input_data,
                    parameters=parameters,
                    task_type=task_type,
                    priority=priority
                )
                
                # Add to queue (high priority goes to front)
                if priority == "high":
                    self._queues[model_name].appendleft(batch_req)
                else:
                    self._queues[model_name].append(batch_req)
        
        # Trigger immediate processing for high priority or full batch
        await self._check_and_trigger_processing(model_name)
        
        return request_id
    
    async def submit_batch(
        self,
        batch_id: str,
        model_name: str,
        inputs: List[Any],
        parameters: Dict[str, Any],
        task_type: str,
        batch_size: Optional[int] = None,
        priority: str = "normal"
    ) -> str:
        """
        Submit a pre-grouped batch of requests.
        
        Args:
            batch_id: Unique batch identifier
            model_name: Target model
            inputs: List of input items
            parameters: Inference parameters
            task_type: Type of task
            batch_size: Override batch size for this request
            priority: Request priority
        
        Returns:
            str: Batch ID
        """
        # Submit each input as a separate request with batch ID prefix
        await self.submit_request(
            request_id=batch_id,
            model_name=model_name,
            inputs=inputs,
            parameters=parameters,
            task_type=task_type,
            priority=priority
        )
        
        return batch_id
    
    async def _batch_collection_loop(self) -> None:
        """
        Background loop that collects requests into batches.
        
        Waits for the configured time window, then triggers
        batch processing for any non-empty queues.
        """
        while self._running:
            try:
                # Wait for batch collection window
                await asyncio.sleep(self.batch_wait_ms / 1000)
                
                # Process all non-empty queues
                for model_name in list(self._queues.keys()):
                    await self._process_batch_for_model(model_name)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch collection loop: {e}")
    
    async def _check_and_trigger_processing(self, model_name: str) -> None:
        """
        Check if batch should be processed immediately.
        
        Triggers processing if:
        - Queue has reached batch size
        - A high priority request is present
        """
        async with self._lock:
            queue = self._queues.get(model_name)
            if not queue:
                return
            
            # Check if we should process immediately
            should_process = len(queue) >= self.batch_size
            
            # Check for high priority requests
            if not should_process:
                for req in queue:
                    if req.priority == "high":
                        should_process = True
                        break
        
        if should_process and not self._processing.get(model_name, False):
            await self._process_batch_for_model(model_name)
    
    async def _process_batch_for_model(self, model_name: str) -> None:
        """
        Process a batch of requests for a specific model.
        
        Extracts requests from the queue, groups them into a batch,
        and sends to the inference engine.
        """
        async with self._lock:
            # Check if already processing
            if self._processing.get(model_name, False):
                return
            
            queue = self._queues.get(model_name)
            if not queue or len(queue) == 0:
                return
            
            # Mark as processing
            self._processing[model_name] = True
            
            # Extract batch from queue
            batch_size = min(len(queue), self.batch_size)
            batch_requests = []
            
            for _ in range(batch_size):
                if queue:
                    batch_requests.append(queue.popleft())
        
        if not batch_requests:
            self._processing[model_name] = False
            return
        
        # Create batch
        import uuid
        batch = Batch(
            batch_id=str(uuid.uuid4()),
            model_name=model_name,
            task_type=batch_requests[0].task_type,
            requests=batch_requests
        )
        
        # Record metrics
        record_batch_size(batch.size)
        
        # Process batch
        try:
            await self._execute_batch(batch)
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
        finally:
            self._processing[model_name] = False
    
    async def _execute_batch(self, batch: Batch) -> None:
        """
        Execute a batch of inference requests.
        
        Sends the batch to the inference engine and handles results.
        
        Args:
            batch: Batch to process
        """
        start_time = time.time()
        
        try:
            # Prepare inputs
            inputs = [req.input_data for req in batch.requests]
            parameters = batch.requests[0].parameters  # Use first request's params
            
            # Process through inference engine
            results = await inference_engine.process_batch(
                model_name=batch.model_name,
                inputs=inputs,
                parameters=parameters,
                task_type=batch.task_type
            )
            
            # Record metrics
            wait_time = (time.time() - start_time) * 1000
            record_batch_wait_time(wait_time)
            
            # Update statistics
            self._stats["batches_processed"] += 1
            self._stats["requests_processed"] += batch.size
            
            logger.info(
                f"Processed batch {batch.batch_id}: {batch.size} requests "
                f"for {batch.model_name} in {wait_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Batch {batch.batch_id} processing failed: {e}")
            raise
    
    async def _flush_all_queues(self) -> None:
        """Process all remaining requests in queues."""
        for model_name in list(self._queues.keys()):
            await self._process_batch_for_model(model_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            **self._stats,
            "queue_sizes": {
                model: len(queue)
                for model, queue in self._queues.items()
            },
            "processing_status": self._processing.copy()
        }
    
    def get_queue_size(self, model_name: str) -> int:
        """Get the current queue size for a model."""
        return len(self._queues.get(model_name, deque()))


# Global batch processor instance
batch_processor = BatchProcessor()
