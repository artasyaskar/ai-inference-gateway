"""
Model manager for AI Inference Gateway.

Manages the lifecycle of AI models including loading, caching,
version tracking, and resource management.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from app.config import settings
from app.inference.model_loader import model_loader
from app.exceptions import ModelNotFoundError, ModelLoadingError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a managed model."""
    id: str
    name: str
    source: str  # 'local' or 'huggingface'
    path: str  # Local path or Hugging Face ID
    task_types: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""
    parameters: Optional[int] = None
    max_sequence_length: int = 512
    tags: List[str] = field(default_factory=list)
    loaded: bool = False
    loaded_at: Optional[datetime] = None
    device: str = "cpu"
    last_used_at: Optional[datetime] = None
    use_count: int = 0


class ModelManager:
    """
    Central manager for all AI models in the gateway.
    
    Handles:
    - Model registration and discovery
    - Lazy loading on first use
    - Resource management and cleanup
    - Model versioning and metadata
    """
    
    def __init__(self):
        self._available_models: Dict[str, ModelMetadata] = {}
        self._loaded_models: Dict[str, Any] = {}  # model_id -> (model, tokenizer)
        self._load_errors: Dict[str, str] = {}  # model_id -> error message
        
        # Initialize with configured models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """
        Initialize model registry from configuration.
        
        Parses SUPPORTED_MODELS environment variable to register
        available models.
        """
        if not settings.SUPPORTED_MODELS:
            logger.warning("No models configured in SUPPORTED_MODELS")
            return
        
        # Parse SUPPORTED_MODELS format: "name:path:task_type,name2:path2:task_type2"
        models_config = settings.SUPPORTED_MODELS.split(",")
        
        for config in models_config:
            config = config.strip()
            if not config:
                continue
            
            parts = config.split(":")
            if len(parts) >= 2:
                model_id = parts[0].strip()
                model_path = parts[1].strip()
                task_type = parts[2].strip() if len(parts) > 2 else "text-generation"
                
                # Determine source
                source = "local" if "/" in model_path or "\\" in model_path or model_path.startswith("./") else "huggingface"
                
                metadata = ModelMetadata(
                    id=model_id,
                    name=model_id.replace("-", " ").title(),
                    source=source,
                    path=model_path,
                    task_types=[task_type],
                    description=f"{model_id} model for {task_type}"
                )
                
                self._available_models[model_id] = metadata
                logger.info(f"Registered model: {model_id} ({source}:{model_path})")
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        task_types: List[str],
        **metadata
    ) -> ModelMetadata:
        """
        Register a new model with the manager.
        
        Args:
            model_id: Unique model identifier
            model_path: Local path or Hugging Face ID
            task_types: List of supported task types
            **metadata: Additional metadata fields
        
        Returns:
            ModelMetadata for the registered model
        """
        source = "local" if "/" in model_path or "\\" in model_path else "huggingface"
        
        model_meta = ModelMetadata(
            id=model_id,
            name=metadata.get("name", model_id.replace("-", " ").title()),
            source=source,
            path=model_path,
            task_types=task_types,
            version=metadata.get("version", "1.0.0"),
            description=metadata.get("description", ""),
            parameters=metadata.get("parameters"),
            max_sequence_length=metadata.get("max_sequence_length", 512),
            tags=metadata.get("tags", [])
        )
        
        self._available_models[model_id] = model_meta
        logger.info(f"Registered model: {model_id}")
        
        return model_meta
    
    async def load_model(
        self,
        model_id: str,
        task_type: Optional[str] = None
    ) -> tuple:
        """
        Load a model if not already loaded.
        
        Args:
            model_id: Model identifier
            task_type: Specific task type to load for
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            ModelNotFoundError: If model is not registered
            ModelLoadingError: If model fails to load
        """
        if model_id not in self._available_models:
            raise ModelNotFoundError(model_id)
        
        metadata = self._available_models[model_id]
        task_type = task_type or metadata.task_types[0]
        
        # Check if already loaded
        if model_id in self._loaded_models:
            logger.debug(f"Model {model_id} already loaded")
            metadata.last_used_at = datetime.utcnow()
            metadata.use_count += 1
            return self._loaded_models[model_id]
        
        # Load the model
        try:
            logger.info(f"Loading model: {model_id} ({task_type})")
            
            model, tokenizer = await model_loader.load_model(
                model_name=model_id,
                model_path=metadata.path,
                task_type=task_type
            )
            
            # Store loaded model
            self._loaded_models[model_id] = (model, tokenizer)
            
            # Update metadata
            metadata.loaded = True
            metadata.loaded_at = datetime.utcnow()
            metadata.last_used_at = datetime.utcnow()
            metadata.use_count = 1
            
            # Try to determine device
            try:
                import torch
                if hasattr(model, 'device'):
                    metadata.device = str(model.device)
                elif hasattr(model, 'model'):
                    metadata.device = str(model.model.device)
                else:
                    metadata.device = "cpu"
            except:
                metadata.device = "cpu"
            
            logger.info(f"Successfully loaded model: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            self._load_errors[model_id] = str(e)
            raise ModelLoadingError(model_id, str(e))
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model to free resources.
        
        Args:
            model_id: Model to unload
        
        Returns:
            bool: True if unloaded, False if not loaded
        """
        if model_id not in self._loaded_models:
            return False
        
        # Get task type from metadata
        metadata = self._available_models.get(model_id)
        task_type = metadata.task_types[0] if metadata else "text-generation"
        
        # Unload from loader
        model_loader.unload_model(model_id, task_type)
        
        # Remove from loaded models
        del self._loaded_models[model_id]
        
        # Update metadata
        if metadata:
            metadata.loaded = False
            metadata.loaded_at = None
        
        logger.info(f"Unloaded model: {model_id}")
        return True
    
    def get_model(self, model_id: str) -> Optional[tuple]:
        """
        Get a loaded model if available.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Tuple of (model, tokenizer) or None if not loaded
        """
        return self._loaded_models.get(model_id)
    
    def is_model_available(self, model_id: str) -> bool:
        """
        Check if a model is registered and available.
        
        Args:
            model_id: Model identifier
        
        Returns:
            bool: True if model is registered
        """
        return model_id in self._available_models
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a model is currently loaded in memory.
        
        Args:
            model_id: Model identifier
        
        Returns:
            bool: True if loaded
        """
        return model_id in self._loaded_models
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dictionary with model information or None
        """
        metadata = self._available_models.get(model_id)
        if not metadata:
            return None
        
        return {
            "id": metadata.id,
            "name": metadata.name,
            "source": metadata.source,
            "path": metadata.path,
            "task_types": metadata.task_types,
            "version": metadata.version,
            "description": metadata.description,
            "parameters": metadata.parameters,
            "max_sequence_length": metadata.max_sequence_length,
            "tags": metadata.tags,
            "loaded": metadata.loaded,
            "device": metadata.device,
            "loaded_at": metadata.loaded_at.isoformat() if metadata.loaded_at else None,
            "last_used_at": metadata.last_used_at.isoformat() if metadata.last_used_at else None,
            "use_count": metadata.use_count,
            "error": self._load_errors.get(model_id)
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their status.
        
        Returns:
            List of model information dictionaries
        """
        return [
            self.get_model_info(model_id)
            for model_id in self._available_models.keys()
        ]
    
    @property
    def available_models(self) -> List[str]:
        """List of registered model IDs."""
        return list(self._available_models.keys())
    
    @property
    def loaded_models(self) -> List[str]:
        """List of currently loaded model IDs."""
        return list(self._loaded_models.keys())
    
    def get_models_by_task(self, task_type: str) -> List[str]:
        """
        Get models that support a specific task type.
        
        Args:
            task_type: Task type to filter by
        
        Returns:
            List of model IDs
        """
        return [
            model_id
            for model_id, metadata in self._available_models.items()
            if task_type in metadata.task_types
        ]
    
    def unload_all_models(self) -> int:
        """
        Unload all models to free resources.
        
        Returns:
            int: Number of models unloaded
        """
        count = 0
        for model_id in list(self._loaded_models.keys()):
            if self.unload_model(model_id):
                count += 1
        
        logger.info(f"Unloaded {count} models")
        return count
    
    def cleanup_idle_models(self, idle_minutes: int = 30) -> int:
        """
        Unload models that haven't been used recently.
        
        Args:
            idle_minutes: Minutes of inactivity before cleanup
        
        Returns:
            int: Number of models cleaned up
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(minutes=idle_minutes)
        to_unload = []
        
        for model_id, metadata in self._available_models.items():
            if metadata.loaded and metadata.last_used_at and metadata.last_used_at < cutoff:
                to_unload.append(model_id)
        
        for model_id in to_unload:
            self.unload_model(model_id)
        
        if to_unload:
            logger.info(f"Cleaned up {len(to_unload)} idle models")
        
        return len(to_unload)


# Global model manager instance
model_manager = ModelManager()
