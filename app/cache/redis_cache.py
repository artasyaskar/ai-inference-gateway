"""
Redis caching utilities for AI Inference Gateway.

Provides caching for:
- Model inference outputs
- Model metadata
- Rate limit counters
- User session data
"""

import json
import hashlib
import logging
from typing import Optional, Any, Dict, Union
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.config import settings
from app.monitoring.metrics import record_cache_hit, record_cache_miss

# Configure logging
logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache manager for AI Inference Gateway.
    
    Provides async Redis operations with:
    - Automatic connection pooling
    - JSON serialization
    - Key namespacing
    - TTL management
    - Cache statistics
    """
    
    def __init__(self):
        self._redis: Optional[Redis] = None
        self._namespace = "ai_gateway"
        self._default_ttl = settings.CACHE_TTL_SECONDS
    
    async def _get_redis(self) -> Redis:
        """
        Get or create Redis connection.
        
        Returns:
            Redis: Async Redis client
        """
        if self._redis is None:
            try:
                # Parse Redis URL
                redis_url = settings.REDIS_URL
                
                # Create connection with pooling
                self._redis = await aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=50,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30
                )
                
                logger.info("Redis connection established")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis
    
    def _make_key(self, key: str, namespace: Optional[str] = None) -> str:
        """
        Create namespaced Redis key.
        
        Args:
            key: Base key
            namespace: Optional sub-namespace
        
        Returns:
            str: Full namespaced key
        """
        if namespace:
            return f"{self._namespace}:{namespace}:{key}"
        return f"{self._namespace}:{key}"
    
    async def ping(self) -> bool:
        """
        Check Redis connectivity.
        
        Returns:
            bool: True if connected
        """
        try:
            redis = await self._get_redis()
            return await redis.ping()
        except Exception:
            return False
    
    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            namespace: Key namespace
        
        Returns:
            Cached value or None if not found
        """
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key, namespace)
            
            value = await redis.get(full_key)
            
            if value is not None:
                # Record hit and deserialize
                record_cache_hit(namespace or "default")
                return json.loads(value)
            
            # Record miss
            record_cache_miss(namespace or "default")
            return None
            
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses default if not specified)
            namespace: Key namespace
        
        Returns:
            bool: True if successfully cached
        """
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key, namespace)
            
            # Serialize value to JSON
            serialized = json.dumps(value, default=str)
            
            # Set with TTL
            ttl = ttl or self._default_ttl
            await redis.setex(full_key, ttl, serialized)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            namespace: Key namespace
        
        Returns:
            bool: True if deleted (or not found)
        """
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key, namespace)
            await redis.delete(full_key)
            return True
            
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            namespace: Key namespace
        
        Returns:
            bool: True if key exists
        """
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key, namespace)
            return await redis.exists(full_key) > 0
            
        except Exception:
            return False
    
    async def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> int:
        """
        Increment a counter in cache.
        
        Args:
            key: Counter key
            amount: Amount to increment
            ttl: TTL for new keys
            namespace: Key namespace
        
        Returns:
            int: New counter value
        """
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key, namespace)
            
            # Increment
            new_value = await redis.incrby(full_key, amount)
            
            # Set TTL if key is new
            if new_value == amount:
                ttl = ttl or self._default_ttl
                await redis.expire(full_key, ttl)
            
            return new_value
            
        except Exception as e:
            logger.warning(f"Cache increment failed for key {key}: {e}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        getter_func,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> Any:
        """
        Get from cache or compute and store.
        
        Args:
            key: Cache key
            getter_func: Async function to compute value if not cached
            ttl: TTL for cached value
            namespace: Key namespace
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key, namespace)
        if cached is not None:
            return cached
        
        # Compute value
        value = await getter_func()
        
        # Store in cache
        await self.set(key, value, ttl, namespace)
        
        return value
    
    async def cache_inference_result(
        self,
        model: str,
        input_hash: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache model inference result.
        
        Args:
            model: Model name
            input_hash: Hash of input data
            result: Inference result
            ttl: Cache TTL
        
        Returns:
            bool: True if cached
        """
        cache_key = f"inference:{model}:{input_hash}"
        return await self.set(cache_key, result, ttl, namespace="model")
    
    async def get_cached_inference(
        self,
        model: str,
        input_hash: str
    ) -> Optional[Any]:
        """
        Get cached inference result.
        
        Args:
            model: Model name
            input_hash: Hash of input data
        
        Returns:
            Cached result or None
        """
        cache_key = f"inference:{model}:{input_hash}"
        return await self.get(cache_key, namespace="model")
    
    async def cache_model_metadata(
        self,
        model: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache model metadata.
        
        Args:
            model: Model name
            metadata: Model metadata dictionary
            ttl: Cache TTL
        
        Returns:
            bool: True if cached
        """
        cache_key = f"metadata:{model}"
        return await self.set(cache_key, metadata, ttl, namespace="model")
    
    async def get_cached_model_metadata(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get cached model metadata.
        
        Args:
            model: Model name
        
        Returns:
            Metadata dictionary or None
        """
        cache_key = f"metadata:{model}"
        return await self.get(cache_key, namespace="model")
    
    async def get_rate_limit_count(
        self,
        user_id: str,
        date: Optional[str] = None
    ) -> int:
        """
        Get current rate limit count for user.
        
        Args:
            user_id: User identifier
            date: Date string (defaults to today)
        
        Returns:
            int: Current request count
        """
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        cache_key = f"rate_limit:{user_id}:{date}"
        
        count = await self.get(cache_key, namespace="ratelimit")
        return int(count) if count else 0
    
    async def increment_rate_limit(
        self,
        user_id: str,
        date: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> int:
        """
        Increment rate limit counter for user.
        
        Args:
            user_id: User identifier
            date: Date string (defaults to today)
            ttl: TTL for the counter (24 hours default)
        
        Returns:
            int: New count
        """
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        cache_key = f"rate_limit:{user_id}:{date}"
        
        # Default TTL: end of day
        if ttl is None:
            now = datetime.utcnow()
            end_of_day = datetime(now.year, now.month, now.day, 23, 59, 59)
            ttl = int((end_of_day - now).total_seconds()) + 1
        
        return await self.increment(cache_key, 1, ttl, namespace="ratelimit")
    
    async def clear_rate_limit(self, user_id: str, date: Optional[str] = None) -> bool:
        """
        Clear rate limit counter for user.
        
        Args:
            user_id: User identifier
            date: Date string
        
        Returns:
            bool: True if cleared
        """
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        cache_key = f"rate_limit:{user_id}:{date}"
        return await self.delete(cache_key, namespace="ratelimit")
    
    @staticmethod
    def hash_input(input_data: Any) -> str:
        """
        Create hash of input data for cache key.
        
        Args:
            input_data: Input data to hash
        
        Returns:
            str: SHA-256 hash hex digest
        """
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(input_str.encode()).hexdigest()
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            redis = await self._get_redis()
            
            info = await redis.info()
            
            return {
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")
    
    async def flush_namespace(self, namespace: str) -> int:
        """
        Clear all keys in a namespace.
        
        Args:
            namespace: Namespace to clear
        
        Returns:
            int: Number of keys deleted
        """
        try:
            redis = await self._get_redis()
            pattern = f"{self._namespace}:{namespace}:*"
            
            # Find and delete matching keys
            keys = []
            async for key in redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await redis.delete(*keys)
            
            return len(keys)
            
        except Exception as e:
            logger.warning(f"Failed to flush namespace {namespace}: {e}")
            return 0


# Global cache instance
redis_cache = RedisCache()
