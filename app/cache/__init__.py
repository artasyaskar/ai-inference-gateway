"""
Cache module for AI Inference Gateway.

Provides Redis-based caching for model outputs and metadata.
"""

from app.cache.redis_cache import redis_cache

__all__ = ["redis_cache"]
