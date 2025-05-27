"""
Embedding Service for Agentic RAG Agent
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import hashlib
import json

import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config import RAGConfig
from utils.logger import get_logger
from utils.metrics import MetricsCollector

logger = get_logger(__name__)


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    texts: List[str]
    model: str
    request_id: Optional[str] = None
    batch_size: Optional[int] = None
    priority: int = 0  # Higher number = higher priority


@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    embeddings: List[List[float]]
    model: str
    request_id: Optional[str] = None
    processing_time: float = 0.0
    cache_hits: int = 0
    api_calls: int = 0


class EmbeddingCache:
    """LRU Cache for embeddings with persistence"""
    
    def __init__(self, max_size: int = 10000, persist_path: str = None):
        self.max_size = max_size
        self.persist_path = persist_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        
        # Load cache from disk if available
        if persist_path:
            self._load_cache()
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        if key in self.cache:
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return self.cache[key]['embedding']
        return None
    
    def put(self, key: str, embedding: List[float], model: str):
        """Store embedding in cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Store embedding with metadata
        self.cache[key] = {
            'embedding': embedding,
            'model': model,
            'timestamp': time.time()
        }
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from persistent storage"""
        try:
            if self.persist_path and os.path.exists(self.persist_path):
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    self.access_order = data.get('access_order', [])
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
    
    def save_cache(self):
        """Save cache to persistent storage"""
        try:
            if self.persist_path:
                data = {
                    'cache': self.cache,
                    'access_order': self.access_order
                }
                with open(self.persist_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_request_count', 1), 1),
            'memory_usage_mb': len(json.dumps(self.cache)) / (1024 * 1024)
        }


class EmbeddingService:
    """
    Service for generating embeddings with caching, batching, and multiple providers
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = MetricsCollector()
        
        # Initialize cache
        cache_path = "./embedding_cache.json" if config.enable_embedding_cache else None
        self.cache = EmbeddingCache(
            max_size=config.embedding_cache_size,
            persist_path=cache_path
        )
        
        # Initialize clients
        self.openai_client = openai.OpenAI()
        self.local_models = {}
        
        # Request queue for batching
        self.request_queue = asyncio.Queue()
        self.batch_processor_task = None
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'total_processing_time': 0.0,
            'batch_requests': 0
        }
        
        logger.info(f"Embedding service initialized with model: {config.embedding_model}")
    
    async def start(self):
        """Start the embedding service background tasks"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("Embedding service batch processor started")
    
    async def stop(self):
        """Stop the embedding service and cleanup"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Save cache
        self.cache.save_cache()
        logger.info("Embedding service stopped")
    
    async def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model: str = None,
        use_cache: bool = True,
        batch_size: int = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for text(s) with caching and batching
        
        Args:
            texts: Single text or list of texts
            model: Model to use (defaults to config model)
            use_cache: Whether to use caching
            batch_size: Batch size for processing
            
        Returns:
            Single embedding or list of embeddings
        """
        start_time = time.time()
        
        # Normalize inputs
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        model = model or self.config.embedding_model
        batch_size = batch_size or self.config.embedding_batch_size
        
        # Check cache first
        cached_embeddings = []
        cache_misses = []
        cache_miss_indices = []
        
        if use_cache and self.config.enable_embedding_cache:
            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key(text, model)
                cached_embedding = self.cache.get(cache_key)
                
                if cached_embedding:
                    cached_embeddings.append(cached_embedding)
                    self.stats['cache_hits'] += 1
                else:
                    cached_embeddings.append(None)
                    cache_misses.append(text)
                    cache_miss_indices.append(i)
        else:
            cache_misses = texts
            cache_miss_indices = list(range(len(texts)))
            cached_embeddings = [None] * len(texts)
        
        # Generate embeddings for cache misses
        new_embeddings = []
        if cache_misses:
            if model.startswith('text-embedding'):
                # OpenAI model
                new_embeddings = await self._get_openai_embeddings(
                    cache_misses, model, batch_size
                )
            else:
                # Local model
                new_embeddings = await self._get_local_embeddings(
                    cache_misses, model, batch_size
                )
            
            # Cache new embeddings
            if use_cache and self.config.enable_embedding_cache:
                for text, embedding in zip(cache_misses, new_embeddings):
                    cache_key = self._generate_cache_key(text, model)
                    self.cache.put(cache_key, embedding, model)
        
        # Combine cached and new embeddings
        result_embeddings = cached_embeddings.copy()
        for i, new_embedding in enumerate(new_embeddings):
            result_embeddings[cache_miss_indices[i]] = new_embedding
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['total_requests'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['api_calls'] += len(cache_misses) if cache_misses else 0
        
        # Record metrics
        self.metrics.record_embedding_request(
            model=model,
            text_count=len(texts),
            cache_hits=len(texts) - len(cache_misses),
            processing_time=processing_time
        )
        
        # Return single embedding or list
        if single_text:
            return result_embeddings[0]
        else:
            return result_embeddings
    
    async def get_embeddings_batch(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Process embedding request with full response metadata
        """
        start_time = time.time()
        
        embeddings = await self.get_embeddings(
            texts=request.texts,
            model=request.model,
            batch_size=request.batch_size
        )
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model,
            request_id=request.request_id,
            processing_time=processing_time,
            cache_hits=0,  # Would be calculated during processing
            api_calls=0    # Would be calculated during processing
        )
    
    async def _get_openai_embeddings(
        self, 
        texts: List[str], 
        model: str, 
        batch_size: int
    ) -> List[List[float]]:
        """Get embeddings from OpenAI API"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await asyncio.to_thread(
                    self.openai_client.embeddings.create,
                    input=batch,
                    model=model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error getting OpenAI embeddings: {e}")
                # Return zero embeddings for failed batch
                embedding_dim = 3072 if 'large' in model else 1536
                all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
        
        return all_embeddings
    
    async def _get_local_embeddings(
        self, 
        texts: List[str], 
        model: str, 
        batch_size: int
    ) -> List[List[float]]:
        """Get embeddings from local model"""
        # Load model if not already loaded
        if model not in self.local_models:
            try:
                self.local_models[model] = SentenceTransformer(model)
                logger.info(f"Loaded local embedding model: {model}")
            except Exception as e:
                logger.error(f"Failed to load local model {model}: {e}")
                # Return zero embeddings
                return [[0.0] * 384] * len(texts)  # Default dimension
        
        local_model = self.local_models[model]
        
        try:
            # Generate embeddings
            embeddings = await asyncio.to_thread(
                local_model.encode,
                texts,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            return [[0.0] * 384] * len(texts)
    
    async def _batch_processor(self):
        """Background task to process embedding requests in batches"""
        while True:
            try:
                # Collect requests for batching
                requests = []
                deadline = time.time() + 0.1  # 100ms batch window
                
                while time.time() < deadline and len(requests) < self.config.embedding_batch_size:
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=max(0.01, deadline - time.time())
                        )
                        requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if requests:
                    await self._process_batch_requests(requests)
                    self.stats['batch_requests'] += 1
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch_requests(self, requests: List[EmbeddingRequest]):
        """Process a batch of embedding requests"""
        try:
            # Group by model
            model_groups = {}
            for request in requests:
                if request.model not in model_groups:
                    model_groups[request.model] = []
                model_groups[request.model].append(request)
            
            # Process each model group
            for model, model_requests in model_groups.items():
                all_texts = []
                request_mapping = []
                
                for request in model_requests:
                    start_idx = len(all_texts)
                    all_texts.extend(request.texts)
                    end_idx = len(all_texts)
                    request_mapping.append((request, start_idx, end_idx))
                
                # Get embeddings for all texts
                embeddings = await self.get_embeddings(
                    texts=all_texts,
                    model=model,
                    use_cache=True
                )
                
                # Distribute results back to requests
                for request, start_idx, end_idx in request_mapping:
                    request_embeddings = embeddings[start_idx:end_idx]
                    # Here you would send the response back to the requester
                    # This could be through a callback, future, or response queue
                    
        except Exception as e:
            logger.error(f"Error processing batch requests: {e}")
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        cache_stats = self.cache.stats()
        
        return {
            'requests': self.stats,
            'cache': cache_stats,
            'models_loaded': list(self.local_models.keys()),
            'queue_size': self.request_queue.qsize(),
            'avg_processing_time': (
                self.stats['total_processing_time'] / 
                max(self.stats['total_requests'], 1)
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                max(self.stats['total_requests'], 1)
            )
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    async def warmup_cache(self, texts: List[str], model: str = None):
        """Warm up cache with frequently used texts"""
        model = model or self.config.embedding_model
        
        logger.info(f"Warming up embedding cache with {len(texts)} texts")
        await self.get_embeddings(texts, model=model, use_cache=True)
        logger.info("Cache warmup completed")
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model.startswith('text-embedding'):
            # OpenAI model
            dimensions = {
                'text-embedding-ada-002': 1536,
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072
            }
            
            return {
                'provider': 'openai',
                'dimensions': dimensions.get(model, 1536),
                'max_tokens': 8191,
                'pricing_per_1k_tokens': 0.0001  # Approximate
            }
        else:
            # Local model
            if model in self.local_models:
                local_model = self.local_models[model]
                return {
                    'provider': 'local',
                    'dimensions': local_model.get_sentence_embedding_dimension(),
                    'max_tokens': local_model.max_seq_length,
                    'pricing_per_1k_tokens': 0.0  # Free for local models
                }
            else:
                return {
                    'provider': 'local',
                    'dimensions': 384,  # Default
                    'max_tokens': 512,   # Default
                    'pricing_per_1k_tokens': 0.0
                }


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(config: RAGConfig = None) -> EmbeddingService:
    """Get singleton embedding service instance"""
    global _embedding_service
    
    if _embedding_service is None:
        if config is None:
            raise ValueError("Config required for first initialization")
        _embedding_service = EmbeddingService(config)
    
    return _embedding_service


async def cleanup_embedding_service():
    """Cleanup embedding service on shutdown"""
    global _embedding_service
    
    if _embedding_service:
        await _embedding_service.stop()
        _embedding_service = None