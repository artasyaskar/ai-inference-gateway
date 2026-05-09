"""
Core inference engine for AI Inference Gateway.

Handles the actual execution of AI model inference with support for:
- Multiple model types and tasks
- Batch processing
- Error handling and fallbacks
- Performance optimization
"""

import asyncio
import logging
import time
from typing import List, Any, Dict, Optional, Union
from dataclasses import dataclass

from app.config import settings
from app.inference.model_manager import model_manager
from app.exceptions import (
    InferenceTimeoutError,
    ModelNotFoundError,
    ModelLoadingError
)
from app.monitoring.metrics import (
    record_inference_request,
    record_inference_latency,
    record_inference_tokens
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of a single inference."""
    output: Any
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    model_version: str = "1.0.0"
    device: str = "cpu"
    cached: bool = False
    error: Optional[str] = None


class InferenceEngine:
    """
    Core inference engine for executing AI model predictions.
    
    Features:
    - Automatic model loading and caching
    - Batch inference for efficiency
    - Timeout handling
    - Model fallback on errors
    - Token counting and metrics
    """
    
    def __init__(self):
        self.request_timeout = settings.REQUEST_TIMEOUT_SECONDS
        self.max_concurrent = settings.MAX_CONCURRENT_REQUESTS
        self.enable_fallback = settings.ENABLE_MODEL_FALLBACK
        
        # Semaphore for limiting concurrent requests
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Track active requests
        self._active_requests = 0
    
    async def process_single(
        self,
        request_id: str,
        model_name: str,
        input_data: Any,
        parameters: Dict[str, Any],
        task_type: str
    ) -> InferenceResult:
        """
        Process a single inference request.
        
        Args:
            request_id: Unique request identifier
            model_name: Target model
            input_data: Input text or data
            parameters: Model-specific parameters
            task_type: Type of inference task
        
        Returns:
            InferenceResult with output and metadata
        """
        async with self._semaphore:
            self._active_requests += 1
            start_time = time.time()
            
            try:
                # Load model if not already loaded
                await model_manager.load_model(model_name, task_type)
                model, tokenizer = model_manager.get_model(model_name)
                
                if not model:
                    raise ModelNotFoundError(model_name)
                
                # Execute inference with timeout
                result = await asyncio.wait_for(
                    self._execute_inference(
                        model=model,
                        tokenizer=tokenizer,
                        input_data=input_data,
                        parameters=parameters,
                        task_type=task_type
                    ),
                    timeout=self.request_timeout
                )
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                result.latency_ms = latency_ms
                
                # Record metrics
                record_inference_request(model_name, "completed")
                record_inference_latency(model_name, latency_ms / 1000)  # Convert to seconds for histogram
                record_inference_tokens(model_name, result.tokens_input + result.tokens_output)
                
                logger.info(
                    f"Inference completed for {request_id}: "
                    f"{result.tokens_input + result.tokens_output} tokens in {latency_ms:.2f}ms"
                )
                
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Inference timeout for {request_id}")
                record_inference_request(model_name, "timeout")
                raise InferenceTimeoutError(model_name, self.request_timeout)
                
            except Exception as e:
                logger.error(f"Inference failed for {request_id}: {e}")
                record_inference_request(model_name, "failed")
                
                # Try fallback if enabled
                if self.enable_fallback:
                    fallback_result = await self._try_fallback(
                        model_name, input_data, parameters, task_type
                    )
                    if fallback_result:
                        return fallback_result
                
                return InferenceResult(
                    output=None,
                    error=str(e),
                    latency_ms=(time.time() - start_time) * 1000
                )
                
            finally:
                self._active_requests -= 1
    
    async def process_batch(
        self,
        model_name: str,
        inputs: List[Any],
        parameters: Dict[str, Any],
        task_type: str
    ) -> List[InferenceResult]:
        """
        Process a batch of inference requests.
        
        Args:
            model_name: Target model
            inputs: List of input items
            parameters: Model parameters
            task_type: Task type
        
        Returns:
            List of InferenceResult objects
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Load model
                await model_manager.load_model(model_name, task_type)
                model, tokenizer = model_manager.get_model(model_name)
                
                if not model:
                    raise ModelNotFoundError(model_name)
                
                # Execute batch inference
                results = await self._execute_batch_inference(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    parameters=parameters,
                    task_type=task_type
                )
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                record_inference_request(model_name, "batch_completed")
                record_inference_latency(model_name, latency_ms / 1000)
                
                total_tokens = sum(r.tokens_input + r.tokens_output for r in results)
                record_inference_tokens(model_name, total_tokens)
                
                logger.info(
                    f"Batch inference completed for {model_name}: "
                    f"{len(inputs)} items in {latency_ms:.2f}ms"
                )
                
                return results
                
            except Exception as e:
                logger.error(f"Batch inference failed for {model_name}: {e}")
                # Return error results for all inputs
                return [
                    InferenceResult(output=None, error=str(e))
                    for _ in inputs
                ]
    
    async def _execute_inference(
        self,
        model: Any,
        tokenizer: Any,
        input_data: Any,
        parameters: Dict[str, Any],
        task_type: str
    ) -> InferenceResult:
        """
        Execute inference based on task type.
        
        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            input_data: Input data
            parameters: Inference parameters
            task_type: Type of task
        
        Returns:
            InferenceResult
        """
        if task_type == "text-generation":
            return await self._text_generation(model, tokenizer, input_data, parameters)
        elif task_type == "embeddings":
            return await self._generate_embeddings(model, input_data, parameters)
        elif task_type == "classification":
            return await self._classification(model, tokenizer, input_data, parameters)
        else:
            return InferenceResult(
                output=None,
                error=f"Unsupported task type: {task_type}"
            )
    
    async def _execute_batch_inference(
        self,
        model: Any,
        tokenizer: Any,
        inputs: List[Any],
        parameters: Dict[str, Any],
        task_type: str
    ) -> List[InferenceResult]:
        """
        Execute batch inference based on task type.
        
        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            inputs: List of input items
            parameters: Inference parameters
            task_type: Task type
        
        Returns:
            List of InferenceResult objects
        """
        if task_type == "text-generation":
            return await self._batch_text_generation(model, tokenizer, inputs, parameters)
        elif task_type == "embeddings":
            return await self._batch_generate_embeddings(model, inputs, parameters)
        else:
            # Fallback to individual processing
            results = []
            for input_data in inputs:
                result = await self._execute_inference(
                    model, tokenizer, input_data, parameters, task_type
                )
                results.append(result)
            return results
    
    async def _text_generation(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        parameters: Dict[str, Any]
    ) -> InferenceResult:
        """
        Generate text using a causal language model.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Prompt text
            parameters: Generation parameters
        
        Returns:
            InferenceResult with generated text
        """
        import torch
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_SEQUENCE_LENGTH
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Count input tokens
        tokens_input = inputs["input_ids"].shape[1]
        
        # Generation parameters
        max_length = parameters.get("max_length", 100)
        max_new_tokens = parameters.get("max_new_tokens", min(max_length, 100))
        temperature = parameters.get("temperature", 1.0)
        top_p = parameters.get("top_p", 0.95)
        do_sample = parameters.get("do_sample", True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(
            outputs[0][tokens_input:],
            skip_special_tokens=True
        )
        
        # Count output tokens
        tokens_output = outputs.shape[1] - tokens_input
        
        return InferenceResult(
            output=generated_text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            device=str(device)
        )
    
    async def _batch_text_generation(
        self,
        model: Any,
        tokenizer: Any,
        input_texts: List[str],
        parameters: Dict[str, Any]
    ) -> List[InferenceResult]:
        """
        Batch text generation for multiple inputs.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            input_texts: List of prompt texts
            parameters: Generation parameters
        
        Returns:
            List of InferenceResult objects
        """
        import torch
        
        # Tokenize all inputs
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
            padding=True
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Count input tokens per item
        attention_mask = inputs.get("attention_mask")
        tokens_inputs = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [0] * len(input_texts)
        
        # Generation parameters
        max_new_tokens = parameters.get("max_new_tokens", 100)
        temperature = parameters.get("temperature", 1.0)
        top_p = parameters.get("top_p", 0.95)
        do_sample = parameters.get("do_sample", True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            input_length = tokens_inputs[i] if i < len(tokens_inputs) else inputs["input_ids"].shape[1]
            generated_text = tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            )
            
            tokens_output = len(output) - input_length
            
            results.append(InferenceResult(
                output=generated_text,
                tokens_input=input_length,
                tokens_output=tokens_output,
                device=str(device)
            ))
        
        return results
    
    async def _generate_embeddings(
        self,
        model: Any,
        input_text: str,
        parameters: Dict[str, Any]
    ) -> InferenceResult:
        """
        Generate embeddings for a single text.
        
        Args:
            model: Sentence transformer model
            input_text: Input text
            parameters: Embedding parameters
        
        Returns:
            InferenceResult with embedding vector
        """
        # SentenceTransformer encodes in one call
        embedding = model.encode([input_text], convert_to_numpy=True)[0]
        
        # Get device info
        device = "cuda" if hasattr(model, 'device') and "cuda" in str(model.device) else "cpu"
        
        # Estimate token count (rough approximation)
        tokens_input = len(input_text.split())
        
        return InferenceResult(
            output=embedding.tolist(),
            tokens_input=tokens_input,
            tokens_output=len(embedding),
            device=device
        )
    
    async def _batch_generate_embeddings(
        self,
        model: Any,
        input_texts: List[str],
        parameters: Dict[str, Any]
    ) -> List[InferenceResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            model: Sentence transformer model
            input_texts: List of input texts
            parameters: Embedding parameters
        
        Returns:
            List of InferenceResult objects
        """
        # Batch encode
        embeddings = model.encode(input_texts, convert_to_numpy=True)
        
        device = "cuda" if hasattr(model, 'device') and "cuda" in str(model.device) else "cpu"
        
        results = []
        for i, embedding in enumerate(embeddings):
            tokens_input = len(input_texts[i].split())
            
            results.append(InferenceResult(
                output=embedding.tolist(),
                tokens_input=tokens_input,
                tokens_output=len(embedding),
                device=device
            ))
        
        return results
    
    async def _classification(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        parameters: Dict[str, Any]
    ) -> InferenceResult:
        """
        Classify a single text.
        
        Args:
            model: Classification model
            tokenizer: Tokenizer
            input_text: Input text
            parameters: Classification parameters
        
        Returns:
            InferenceResult with classification result
        """
        import torch
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_SEQUENCE_LENGTH
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        tokens_input = inputs["input_ids"].shape[1]
        
        # Classify
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # Get label if available
        id2label = model.config.id2label if hasattr(model, 'config') else {}
        label = id2label.get(predicted_class, f"class_{predicted_class}")
        
        return InferenceResult(
            output={
                "label": label,
                "confidence": confidence,
                "class_id": predicted_class
            },
            tokens_input=tokens_input,
            tokens_output=1,
            device=str(device)
        )
    
    async def _try_fallback(
        self,
        original_model: str,
        input_data: Any,
        parameters: Dict[str, Any],
        task_type: str
    ) -> Optional[InferenceResult]:
        """
        Try to use a fallback model when primary model fails.
        
        Args:
            original_model: Original model that failed
            input_data: Input data
            parameters: Parameters
            task_type: Task type
        
        Returns:
            InferenceResult if fallback succeeds, None otherwise
        """
        if not self.enable_fallback:
            return None
        
        # Find alternative models with same task type
        alternatives = model_manager.get_models_by_task(task_type)
        alternatives = [m for m in alternatives if m != original_model]
        
        if not alternatives:
            return None
        
        # Try first alternative
        fallback_model = alternatives[0]
        
        try:
            logger.info(f"Trying fallback model: {fallback_model}")
            
            await model_manager.load_model(fallback_model, task_type)
            model, tokenizer = model_manager.get_model(fallback_model)
            
            result = await self._execute_inference(
                model, tokenizer, input_data, parameters, task_type
            )
            
            # Mark as fallback
            result.cached = False  # Actually a fallback, but using this flag for tracking
            logger.info(f"Fallback to {fallback_model} successful")
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback model {fallback_model} also failed: {e}")
            return None
    
    @property
    def active_requests(self) -> int:
        """Current number of active inference requests."""
        return self._active_requests


# Global inference engine instance
inference_engine = InferenceEngine()
