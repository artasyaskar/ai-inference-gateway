"""
Model loader for AI Inference Gateway.

Handles loading AI models from local storage or Hugging Face Hub.
Supports multiple model types and automatic device selection.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from app.config import settings
from app.exceptions import ModelLoadingError

# Configure logging
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading of AI/ML models from various sources.
    
    Supports:
    - Local model files
    - Hugging Face Hub models
    - Different model types (text generation, embeddings, etc.)
    - Automatic device selection (CPU/CUDA)
    """
    
    def __init__(self):
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}
        
    def _get_device(self) -> str:
        """
        Determine the best available device for model inference.
        
        Returns:
            str: Device name ('cuda' or 'cpu')
        """
        try:
            import torch
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                return "cuda"
            elif settings.DEVICE == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Auto-selected device: {device}")
                return device
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
        
        return "cpu"
    
    async def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        task_type: str = "text-generation",
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a model and its tokenizer/processor.
        
        Args:
            model_name: Identifier for the model
            model_path: Local path or Hugging Face model ID (defaults to model_name)
            task_type: Type of task (text-generation, embeddings, etc.)
            **kwargs: Additional loading parameters
        
        Returns:
            Tuple of (model, tokenizer/processor)
        
        Raises:
            ModelLoadingError: If model fails to load
        """
        cache_key = f"{model_name}:{task_type}"
        
        # Return cached model if available
        if cache_key in self._loaded_models:
            logger.debug(f"Using cached model: {model_name}")
            return self._loaded_models[cache_key]
        
        model_path = model_path or model_name
        device = self._get_device()
        
        try:
            logger.info(f"Loading model '{model_name}' from '{model_path}' for task '{task_type}'")
            
            if task_type == "text-generation":
                model, tokenizer = await self._load_text_generation_model(
                    model_path, device, **kwargs
                )
            elif task_type == "embeddings":
                model, tokenizer = await self._load_embedding_model(
                    model_path, device, **kwargs
                )
            elif task_type == "classification":
                model, tokenizer = await self._load_classification_model(
                    model_path, device, **kwargs
                )
            else:
                raise ModelLoadingError(
                    model_name,
                    f"Unsupported task type: {task_type}"
                )
            
            # Cache loaded model
            self._loaded_models[cache_key] = (model, tokenizer)
            
            logger.info(f"Successfully loaded model '{model_name}' on {device}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise ModelLoadingError(model_name, str(e))
    
    async def _load_text_generation_model(
        self,
        model_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a text generation model (GPT, LLaMA, etc.).
        
        Args:
            model_path: Hugging Face model ID or local path
            device: Device to load model on
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Determine torch dtype based on device
        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype is None and device == "cuda":
            import torch
            # Use float16 for GPU to save memory
            torch_dtype = torch.float16
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=kwargs.get("trust_remote_code", False)
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=kwargs.get("trust_remote_code", False)
        )
        
        if device == "cpu":
            model = model.to(device)
        
        return model, tokenizer
    
    async def _load_embedding_model(
        self,
        model_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a sentence embedding model.
        
        Args:
            model_path: Hugging Face model ID or local path
            device: Device to load model on
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(
            model_path,
            cache_folder=str(self.cache_dir),
            device=device
        )
        
        # SentenceTransformer includes tokenizer internally
        return model, None
    
    async def _load_classification_model(
        self,
        model_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a classification model.
        
        Args:
            model_path: Hugging Face model ID or local path
            device: Device to load model on
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.cache_dir
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir=self.cache_dir
        )
        
        model = model.to(device)
        
        return model, tokenizer
    
    def unload_model(self, model_name: str, task_type: str = "text-generation") -> bool:
        """
        Unload a model from memory to free resources.
        
        Args:
            model_name: Model identifier
            task_type: Task type of the model
        
        Returns:
            bool: True if model was unloaded, False if not found
        """
        cache_key = f"{model_name}:{task_type}"
        
        if cache_key in self._loaded_models:
            model, _ = self._loaded_models[cache_key]
            
            # Clear model from memory
            del model
            del self._loaded_models[cache_key]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info(f"Unloaded model: {model_name}")
            return True
        
        return False
    
    def is_model_loaded(self, model_name: str, task_type: str = "text-generation") -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_name: Model identifier
            task_type: Task type
        
        Returns:
            bool: True if model is loaded
        """
        cache_key = f"{model_name}:{task_type}"
        return cache_key in self._loaded_models
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded models.
        
        Returns:
            Dictionary with loaded model information
        """
        info = {}
        
        for cache_key, (model, _) in self._loaded_models.items():
            model_name, task_type = cache_key.split(":", 1)
            
            # Get model size if available
            model_size = None
            try:
                import torch
                if hasattr(model, 'parameters'):
                    param_count = sum(p.numel() for p in model.parameters())
                    model_size = param_count
            except Exception:
                pass
            
            info[cache_key] = {
                "name": model_name,
                "task_type": task_type,
                "loaded": True,
                "parameters": model_size,
                "device": next(model.parameters()).device.type if hasattr(model, 'parameters') else "unknown"
            }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear all loaded models from memory."""
        for cache_key in list(self._loaded_models.keys()):
            model_name, task_type = cache_key.split(":", 1)
            self.unload_model(model_name, task_type)
        
        logger.info("Cleared all loaded models")


# Global model loader instance
model_loader = ModelLoader()
