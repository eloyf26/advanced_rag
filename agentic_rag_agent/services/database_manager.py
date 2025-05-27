"""
Enhanced Database Manager for RAG Operations with Comprehensive Error Handling
"""

import asyncio
import logging
import os  # FIXED: Added missing import
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from contextlib import asynccontextmanager
import time

from supabase import create_client, Client
import openai
from sentence_transformers import CrossEncoder
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import RAGConfig
from models.request_models import SearchFilters
from models.response_models import DocumentChunk, SearchResults
from utils.logger import get_logger
from utils.metrics import get_metrics_collector

logger = get_logger(__name__)


class DatabaseConnectionError(Exception):
    """Database connection related errors"""
    pass


class EmbeddingServiceError(Exception):
    """Embedding service related errors"""
    pass


class SearchError(Exception):
    """Search operation related errors"""
    pass


class RAGDatabaseManager:
    """
    Enhanced database manager with comprehensive error handling and resilience
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = get_metrics_collector()
        
        # Initialize clients with error handling
        self.supabase: Optional[Client] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.reranker: Optional[CrossEncoder] = None
        
        # Connection state
        self._db_healthy = False
        self._embedding_healthy = False
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        
        # Caching
        self.embedding_cache = {}
        self.query_cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Circuit breaker states
        self._db_circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'closed'  # closed, open, half-open
        }
        
        self._embedding_circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'closed'
        }
        
        # Initialize services
        asyncio.create_task(self._initialize_services())
        
        logger.info("RAG Database Manager initialized with enhanced error handling")
    
    async def _initialize_services(self):
        """Initialize database and AI services with error handling"""
        try:
            # Initialize Supabase client
            if not self.supabase:
                self.supabase = create_client(
                    self.config.supabase_url, 
                    self.config.supabase_key
                )
                logger.info("Supabase client initialized")
            
            # Initialize OpenAI client
            if not self.openai_client:
                self.openai_client = openai.AsyncOpenAI()
                logger.info("OpenAI client initialized")
            
            # Initialize reranker if enabled
            if self.config.enable_reranking and not self.reranker:
                try:
                    self.reranker = CrossEncoder(self.config.rerank_model)
                    logger.info(f"Loaded reranker: {self.config.rerank_model}")
                except Exception as e:
                    logger.warning(f"Failed to load reranker: {e}")
                    self.config.enable_reranking = False
            
            # Test connections
            await self._health_check()
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise
    
    async def _health_check(self, force: bool = False) -> Dict[str, bool]:
        """Comprehensive health check for all services"""
        current_time = time.time()
        
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return {
                'database': self._db_healthy,
                'embedding': self._embedding_healthy
            }
        
        health_status = {}
        
        # Test database connection
        try:
            if self.supabase:
                result = self.supabase.table(self.config.table_name).select("count").limit(1).execute()
                self._db_healthy = True
                health_status['database'] = True
                self._reset_circuit_breaker('db')
                logger.debug("Database health check passed")
            else:
                self._db_healthy = False
                health_status['database'] = False
        except Exception as e:
            self._db_healthy = False
            health_status['database'] = False
            self._record_circuit_breaker_failure('db')
            logger.error(f"Database health check failed: {e}")
        
        # Test embedding service
        try:
            if self.openai_client:
                # Test with a simple embedding request
                response = await self.openai_client.embeddings.create(
                    input="test",
                    model=self.config.embedding_model
                )
                self._embedding_healthy = True
                health_status['embedding'] = True
                self._reset_circuit_breaker('embedding')
                logger.debug("Embedding service health check passed")
            else:
                self._embedding_healthy = False
                health_status['embedding'] = False
        except Exception as e:
            self._embedding_healthy = False
            health_status['embedding'] = False
            self._record_circuit_breaker_failure('embedding')
            logger.error(f"Embedding service health check failed: {e}")
        
        self._last_health_check = current_time
        return health_status
    
    def _record_circuit_breaker_failure(self, service: str):
        """Record a circuit breaker failure"""
        if service == 'db':
            breaker = self._db_circuit_breaker
        elif service == 'embedding':
            breaker = self._embedding_circuit_breaker
        else:
            return
        
        breaker['failures'] += 1
        breaker['last_failure'] = time.time()
        
        # Open circuit if too many failures
        if breaker['failures'] >= 5:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {service} service")
    
    def _reset_circuit_breaker(self, service: str):
        """Reset circuit breaker on successful operation"""
        if service == 'db':
            breaker = self._db_circuit_breaker
        elif service == 'embedding':
            breaker = self._embedding_circuit_breaker
        else:
            return
        
        if breaker['state'] != 'closed':
            logger.info(f"Circuit breaker closed for {service} service")
        
        breaker['failures'] = 0
        breaker['state'] = 'closed'
    
    def _is_circuit_breaker_open(self, service: str) -> bool:
        """Check if circuit breaker is open"""
        if service == 'db':
            breaker = self._db_circuit_breaker
        elif service == 'embedding':
            breaker = self._embedding_circuit_breaker
        else:
            return False
        
        if breaker['state'] == 'open':
            # Try to half-open after 60 seconds
            if time.time() - breaker['last_failure'] > 60:
                breaker['state'] = 'half-open'
                return False
            return True
        
        return False
    
    async def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            health = await self._health_check(force=True)
            return health.get('database', False)
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def test_embedding_service(self) -> bool:
        """Test embedding service connectivity"""
        try:
            health = await self._health_check(force=True)
            return health.get('embedding', False)
        except Exception as e:
            logger.error(f"Embedding service test failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(EmbeddingServiceError)
    )
    async def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query with caching and error handling
        """
        if self._is_circuit_breaker_open('embedding'):
            raise EmbeddingServiceError("Embedding service circuit breaker is open")
        
        # Check cache first
        if self.config.enable_embedding_cache:
            async with self._cache_lock:
                if query in self.embedding_cache:
                    self.metrics.record_cache_event('embedding', 'hit')
                    return self.embedding_cache[query]
        
        try:
            if not self.openai_client:
                await self._initialize_services()
            
            response = await self.openai_client.embeddings.create(
                input=query,
                model=self.config.embedding_model
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            if self.config.enable_embedding_cache:
                async with self._cache_lock:
                    # Simple LRU cache implementation
                    if len(self.embedding_cache) >= self.config.embedding_cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self.embedding_cache))
                        del self.embedding_cache[oldest_key]
                    
                    self.embedding_cache[query] = embedding
                    self.metrics.record_cache_event('embedding', 'miss')
            
            self._reset_circuit_breaker('embedding')
            return embedding
            
        except Exception as e:
            self._record_circuit_breaker_failure('embedding')
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingServiceError(f"Failed to generate embedding: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(DatabaseConnectionError)
    )
    async def hybrid_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Perform hybrid search with comprehensive error handling
        """
        if self._is_circuit_breaker_open('db'):
            raise DatabaseConnectionError("Database circuit breaker is open")
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.get_query_embedding(query)
            
            # Prepare filters for SQL function
            file_types_filter = filters.file_types if filters.file_types else None
            date_filter = filters.date_after if filters.date_after else None
            
            # Call the hybrid search function with error handling
            if not self.supabase:
                raise DatabaseConnectionError("Supabase client not initialized")
            
            result = self.supabase.rpc(
                'hybrid_search',
                {
                    'query_embedding': query_embedding,
                    'query_text': query,
                    'similarity_threshold': filters.similarity_threshold,
                    'limit_count': min(filters.max_results * 2, self.config.rerank_top_k),
                    'file_types': file_types_filter,
                    'date_filter': date_filter.isoformat() if date_filter else None,
                    'vector_weight': self.config.vector_weight,
                    'bm25_weight': self.config.bm25_weight
                }
            ).execute()
            
            if not result.data:
                logger.warning(f"No results found for query: {query[:100]}")
                return []
            
            # Convert to DocumentChunk objects
            chunks = self._convert_db_results_to_chunks(result.data)
            
            # Apply reranking if enabled
            if self.config.enable_reranking and self.reranker and chunks:
                chunks = await self._rerank_chunks(query, chunks)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics.record_search(
                method='hybrid',
                duration=processing_time,
                results_count=len(chunks),
                status='success'
            )
            
            self._reset_circuit_breaker('db')
            return chunks[:filters.max_results]
            
        except EmbeddingServiceError:
            # Re-raise embedding errors
            raise
        except Exception as e:
            self._record_circuit_breaker_failure('db')
            processing_time = time.time() - start_time
            self.metrics.record_search(
                method='hybrid',
                duration=processing_time,
                results_count=0,
                status='error'
            )
            logger.error(f"Error in hybrid search: {e}")
            
            # Try fallback search if available
            if self.config.enable_graceful_degradation:
                return await self._fallback_search(query, filters)
            
            raise SearchError(f"Hybrid search failed: {str(e)}")
    
    async def _fallback_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Fallback search using keyword-only search when vector search fails
        """
        try:
            logger.info("Attempting fallback keyword search")
            return await self.keyword_search(query, filters)
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []
    
    async def semantic_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Perform semantic search with error handling
        """
        try:
            query_embedding = await self.get_query_embedding(query)
            
            result = self.supabase.rpc(
                'semantic_search',
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': filters.similarity_threshold,
                    'limit_count': filters.max_results,
                    'file_types': filters.file_types,
                    'date_filter': filters.date_after.isoformat() if filters.date_after else None
                }
            ).execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            if self.config.enable_graceful_degradation:
                return await self._fallback_search(query, filters)
            raise SearchError(f"Semantic search failed: {str(e)}")
    
    async def keyword_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Perform keyword search with error handling
        """
        try:
            if not self.supabase:
                raise DatabaseConnectionError("Supabase client not initialized")
            
            result = self.supabase.rpc(
                'keyword_search',
                {
                    'query_text': query,
                    'limit_count': filters.max_results,
                    'file_types': filters.file_types,
                    'date_filter': filters.date_after.isoformat() if filters.date_after else None
                }
            ).execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            raise SearchError(f"Keyword search failed: {str(e)}")
    
    async def get_document_context(
        self, 
        document_id: str, 
        chunk_index: int, 
        context_window: int = 3
    ) -> List[DocumentChunk]:
        """
        Get surrounding chunks for better context with error handling
        """
        try:
            if not self.supabase:
                raise DatabaseConnectionError("Supabase client not initialized")
            
            result = self.supabase.table(self.config.table_name).select("*").eq(
                'document_id', document_id
            ).gte(
                'chunk_index', max(0, chunk_index - context_window)
            ).lte(
                'chunk_index', chunk_index + context_window
            ).order('chunk_index').execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return []
    
    async def get_similar_chunks(
        self, 
        chunk_id: str, 
        similarity_threshold: float = 0.8,
        limit: int = 5
    ) -> List[DocumentChunk]:
        """
        Find chunks similar to a given chunk with error handling
        """
        try:
            if not self.supabase:
                raise DatabaseConnectionError("Supabase client not initialized")
            
            result = self.supabase.rpc(
                'find_similar_chunks',
                {
                    'target_chunk_id': chunk_id,
                    'similarity_threshold': similarity_threshold,
                    'limit_count': limit
                }
            ).execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics with error handling
        """
        try:
            if not self.supabase:
                raise DatabaseConnectionError("Supabase client not initialized")
            
            result = self.supabase.rpc('get_document_stats').execute()
            
            if result.data:
                stats = result.data[0]
                return {
                    'total_documents': stats.get('total_documents', 0),
                    'total_chunks': stats.get('total_chunks', 0),
                    'unique_files': stats.get('unique_files', 0),
                    'file_types': stats.get('file_types', []),
                    'avg_chunk_size': stats.get('avg_chunk_size', 0),
                    'total_size_mb': stats.get('total_size_mb', 0),
                    'oldest_document': stats.get('oldest_document'),
                    'newest_document': stats.get('newest_document'),
                    'vector_dimension': stats.get('vector_dimension', 0)
                }
            else:
                return self._get_empty_stats()
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return self._get_empty_stats()
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics when database is unavailable"""
        return {
            'total_documents': 0,
            'total_chunks': 0,
            'unique_files': 0,
            'file_types': [],
            'avg_chunk_size': 0,
            'total_size_mb': 0,
            'oldest_document': None,
            'newest_document': None,
            'vector_dimension': 0
        }
    
    async def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Rerank chunks using cross-encoder model with error handling
        """
        if not self.reranker or not chunks:
            return chunks
        
        try:
            # Prepare query-document pairs for reranking
            query_doc_pairs = []
            for chunk in chunks:
                # Use title + content for better reranking
                doc_text = f"{chunk.title or ''} {chunk.content}".strip()
                query_doc_pairs.append([query, doc_text])
            
            # Get reranking scores with timeout
            rerank_scores = await asyncio.wait_for(
                asyncio.to_thread(self.reranker.predict, query_doc_pairs),
                timeout=30.0  # 30 second timeout
            )
            
            # Update chunks with rerank scores
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(rerank_scores[i])
                # Update combined score
                chunk.combined_score = (
                    0.6 * chunk.rerank_score +
                    0.25 * chunk.similarity_score +
                    0.15 * chunk.bm25_score
                )
            
            # Sort by rerank score (descending)
            chunks.sort(key=lambda x: x.rerank_score, reverse=True)
            
            logger.info(f"Reranked {len(chunks)} chunks")
            return chunks
            
        except asyncio.TimeoutError:
            logger.warning("Reranking timed out, returning original order")
            return chunks
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return chunks
    
    def _convert_db_results_to_chunks(self, db_results: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Convert database results to DocumentChunk objects with error handling
        """
        chunks = []
        
        for row in db_results:
            try:
                chunk = DocumentChunk(
                    id=row.get('id', ''),
                    content=row.get('content', ''),
                    similarity_score=float(row.get('similarity_score', 0.0)),
                    bm25_score=float(row.get('bm25_score', 0.0)),
                    combined_score=float(row.get('combined_score', 0.0)),
                    file_name=row.get('file_name', ''),
                    file_type=row.get('file_type', ''),
                    chunk_index=int(row.get('chunk_index', 0)),
                    title=row.get('title'),
                    summary=row.get('summary'),
                    keywords=row.get('keywords', []),
                    entities=row.get('entities', []),
                    document_id=row.get('document_id'),
                    chunk_context=self._create_chunk_context(row)
                )
                chunks.append(chunk)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting database row to chunk: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error converting database row: {e}")
                continue
        
        return chunks
    
    def _create_chunk_context(self, row: Dict[str, Any]) -> Optional[str]:
        """
        Create context information for a chunk
        """
        try:
            context_parts = []
            
            if row.get('previous_chunk_preview'):
                context_parts.append(f"Previous: {row['previous_chunk_preview']}")
            
            if row.get('next_chunk_preview'):
                context_parts.append(f"Next: {row['next_chunk_preview']}")
            
            return " | ".join(context_parts) if context_parts else None
        except Exception as e:
            logger.debug(f"Error creating chunk context: {e}")
            return None
    
    async def cleanup_old_cache(self):
        """
        Clean up old cache entries with error handling
        """
        try:
            async with self._cache_lock:
                # Simple cache cleanup - remove half the entries if cache is full
                if len(self.embedding_cache) >= self.config.embedding_cache_size:
                    items_to_remove = len(self.embedding_cache) // 2
                    keys_to_remove = list(self.embedding_cache.keys())[:items_to_remove]
                    
                    for key in keys_to_remove:
                        del self.embedding_cache[key]
                    
                    logger.info(f"Cleaned up {items_to_remove} embedding cache entries")
                
                # Clean up query cache if it exists
                if hasattr(self, 'query_cache') and len(self.query_cache) > 1000:
                    items_to_remove = len(self.query_cache) // 2
                    keys_to_remove = list(self.query_cache.keys())[:items_to_remove]
                    
                    for key in keys_to_remove:
                        del self.query_cache[key]
                    
                    logger.info(f"Cleaned up {items_to_remove} query cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def clear_all_caches(self):
        """
        Clear all caches
        """
        try:
            self.embedding_cache.clear()
            self.query_cache.clear()
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status
        """
        try:
            health = await self._health_check(force=True)
            
            return {
                'database_connected': health.get('database', False),
                'embedding_service_healthy': health.get('embedding', False),
                'cache_size': len(self.embedding_cache),
                'db_circuit_breaker': self._db_circuit_breaker['state'],
                'embedding_circuit_breaker': self._embedding_circuit_breaker['state'],
                'last_health_check': datetime.fromtimestamp(self._last_health_check).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'database_connected': False,
                'embedding_service_healthy': False,
                'error': str(e)
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_services()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            # Cleanup resources
            if hasattr(self.openai_client, 'close'):
                await self.openai_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")