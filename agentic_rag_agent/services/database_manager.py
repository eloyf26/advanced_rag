"""
Database Manager for RAG Operations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from supabase import create_client, Client
import openai
from sentence_transformers import CrossEncoder

from config import RAGConfig
from models.request_models import SearchFilters
from models.response_models import DocumentChunk, SearchResults

logger = logging.getLogger(__name__)


class RAGDatabaseManager:
    """
    Manages all database operations for the RAG system
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize clients
        self.supabase: Client = create_client(config.supabase_url, config.supabase_key)
        self.openai_client = openai.OpenAI()
        
        # Initialize reranker if enabled
        self.reranker = None
        if config.enable_reranking:
            try:
                self.reranker = CrossEncoder(config.rerank_model)
                logger.info(f"Loaded reranker: {config.rerank_model}")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self.config.enable_reranking = False
        
        # Connection pool and caching
        self.embedding_cache = {}
        self.query_cache = {}
        
        logger.info("RAG Database Manager initialized")
    
    async def test_connection(self) -> bool:
        """
        Test database connectivity
        """
        try:
            result = self.supabase.table(self.config.table_name).select("count").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def test_embedding_service(self) -> bool:
        """
        Test embedding service connectivity
        """
        try:
            await self.get_query_embedding("test")
            return True
        except Exception as e:
            logger.error(f"Embedding service test failed: {e}")
            return False
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query with caching
        """
        # Check cache first
        if self.config.enable_embedding_cache and query in self.embedding_cache:
            return self.embedding_cache[query]
        
        try:
            response = self.openai_client.embeddings.create(
                input=query,
                model=self.config.embedding_model
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            if self.config.enable_embedding_cache:
                # Simple LRU cache implementation
                if len(self.embedding_cache) >= self.config.embedding_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
                
                self.embedding_cache[query] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def hybrid_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Perform hybrid search combining vector similarity and BM25
        """
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = await self.get_query_embedding(query)
            
            # Prepare filters for SQL function
            file_types_filter = filters.file_types if filters.file_types else None
            date_filter = filters.date_after if filters.date_after else None
            
            # Call the hybrid search function
            result = self.supabase.rpc(
                'hybrid_search',
                {
                    'query_embedding': query_embedding,
                    'query_text': query,
                    'similarity_threshold': filters.similarity_threshold,
                    'limit_count': min(filters.max_results * 2, self.config.rerank_top_k),
                    'file_types': file_types_filter,
                    'date_filter': date_filter.isoformat() if date_filter else None
                }
            ).execute()
            
            # Convert to DocumentChunk objects
            chunks = self._convert_db_results_to_chunks(result.data)
            
            # Apply reranking if enabled
            if self.config.enable_reranking and self.reranker and chunks:
                chunks = await self._rerank_chunks(query, chunks)
            
            # Return top results
            return chunks[:filters.max_results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def semantic_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Perform semantic search using vector similarity only
        """
        try:
            query_embedding = await self.get_query_embedding(query)
            
            result = self.supabase.rpc(
                'semantic_search',
                {
                    'query_embedding': query_embedding,
                    'similarity_threshold': filters.similarity_threshold,
                    'limit_count': filters.max_results,
                    'file_types': filters.file_types
                }
            ).execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def keyword_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
        """
        Perform keyword search using BM25 only
        """
        try:
            result = self.supabase.rpc(
                'keyword_search',
                {
                    'query_text': query,
                    'limit_count': filters.max_results,
                    'file_types': filters.file_types
                }
            ).execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def get_document_context(
        self, 
        document_id: str, 
        chunk_index: int, 
        context_window: int = 3
    ) -> List[DocumentChunk]:
        """
        Get surrounding chunks for better context
        """
        try:
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
        Find chunks similar to a given chunk
        """
        try:
            # First get the target chunk
            chunk_result = self.supabase.table(self.config.table_name).select(
                "embedding, content"
            ).eq('id', chunk_id).execute()
            
            if not chunk_result.data:
                return []
            
            chunk_embedding = chunk_result.data[0]['embedding']
            
            # Find similar chunks
            result = self.supabase.rpc(
                'semantic_search',
                {
                    'query_embedding': chunk_embedding,
                    'similarity_threshold': similarity_threshold,
                    'limit_count': limit + 1  # +1 because original chunk will be included
                }
            ).execute()
            
            # Filter out the original chunk
            similar_chunks = [
                chunk for chunk in self._convert_db_results_to_chunks(result.data)
                if chunk.id != chunk_id
            ]
            
            return similar_chunks[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []
    
    async def get_chunks_by_keywords(
        self, 
        keywords: List[str], 
        limit: int = 10
    ) -> List[DocumentChunk]:
        """
        Get chunks that contain specific keywords
        """
        try:
            # Create a query that matches any of the keywords
            keyword_query = " OR ".join(keywords)
            
            result = self.supabase.rpc(
                'keyword_search',
                {
                    'query_text': keyword_query,
                    'limit_count': limit
                }
            ).execute()
            
            return self._convert_db_results_to_chunks(result.data)
            
        except Exception as e:
            logger.error(f"Error searching by keywords: {e}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics
        """
        try:
            result = self.supabase.rpc('get_document_stats').execute()
            
            if result.data:
                stats = result.data[0]
                return {
                    'total_documents': stats.get('total_documents', 0),
                    'total_chunks': stats.get('total_chunks', 0),
                    'file_types': stats.get('file_types', []),
                    'processing_dates': stats.get('processing_dates', []),
                    'avg_chunk_size': stats.get('avg_chunk_size', 0)
                }
            else:
                return {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'file_types': [],
                    'processing_dates': [],
                    'avg_chunk_size': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    async def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Rerank chunks using cross-encoder model
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
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Update chunks with rerank scores
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(rerank_scores[i])
            
            # Sort by rerank score (descending)
            chunks.sort(key=lambda x: x.rerank_score, reverse=True)
            
            logger.info(f"Reranked {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return chunks
    
    def _convert_db_results_to_chunks(self, db_results: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Convert database results to DocumentChunk objects
        """
        chunks = []
        
        for row in db_results:
            try:
                chunk = DocumentChunk(
                    id=row.get('id', ''),
                    content=row.get('content', ''),
                    similarity_score=row.get('similarity_score', 0.0),
                    bm25_score=row.get('bm25_score', 0.0),
                    combined_score=row.get('combined_score', 0.0),
                    file_name=row.get('file_name', ''),
                    file_type=row.get('file_type', ''),
                    chunk_index=row.get('chunk_index', 0),
                    title=row.get('title'),
                    summary=row.get('summary'),
                    keywords=row.get('keywords', []),
                    entities=row.get('entities', []),
                    document_id=row.get('document_id'),
                    chunk_context=self._create_chunk_context(row)
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Error converting database row to chunk: {e}")
                continue
        
        return chunks
    
    def _create_chunk_context(self, row: Dict[str, Any]) -> Optional[str]:
        """
        Create context information for a chunk
        """
        context_parts = []
        
        if row.get('previous_chunk_preview'):
            context_parts.append(f"Previous: {row['previous_chunk_preview']}")
        
        if row.get('next_chunk_preview'):
            context_parts.append(f"Next: {row['next_chunk_preview']}")
        
        return " | ".join(context_parts) if context_parts else None
    
    async def cleanup_old_cache(self):
        """
        Clean up old cache entries
        """
        try:
            # Simple cache cleanup - remove half the entries if cache is full
            if len(self.embedding_cache) >= self.config.embedding_cache_size:
                items_to_remove = len(self.embedding_cache) // 2
                keys_to_remove = list(self.embedding_cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    del self.embedding_cache[key]
                
                logger.info(f"Cleaned up {items_to_remove} cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    async def batch_search(
        self, 
        queries: List[str], 
        search_method: str = "hybrid"
    ) -> List[SearchResults]:
        """
        Perform batch search for multiple queries
        """
        results = []
        
        # Process queries concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)
        
        async def search_single_query(query: str) -> SearchResults:
            async with semaphore:
                filters = SearchFilters(max_results=self.config.default_max_results)
                
                try:
                    start_time = datetime.now()
                    
                    if search_method == "hybrid":
                        chunks = await self.hybrid_search(query, filters)
                    elif search_method == "semantic":
                        chunks = await self.semantic_search(query, filters)
                    elif search_method == "keyword":
                        chunks = await self.keyword_search(query, filters)
                    else:
                        chunks = await self.hybrid_search(query, filters)
                    
                    search_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    return SearchResults(
                        query=query,
                        chunks=chunks,
                        total_found=len(chunks),
                        search_time_ms=search_time,
                        search_method=search_method,
                        filters_applied=filters.dict()
                    )
                    
                except Exception as e:
                    logger.error(f"Error in batch search for query '{query}': {e}")
                    return SearchResults(
                        query=query,
                        chunks=[],
                        total_found=0,
                        search_time_ms=0,
                        search_method=search_method,
                        filters_applied=filters.dict()
                    )
        
        # Execute all searches concurrently
        tasks = [search_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def update_search_statistics(self, search_stats: Dict[str, Any]):
        """
        Update BM25 and other search statistics
        """
        try:
            # Call the update function
            self.supabase.rpc('update_bm25_stats').execute()
            logger.info("Search statistics updated")
        except Exception as e:
            logger.error(f"Error updating search statistics: {e}")
    
    async def get_search_performance_metrics(self) -> Dict[str, Any]:
        """
        Get search performance metrics
        """
        try:
            # This would query actual performance metrics from the database
            # For now, return placeholder metrics
            return {
                'avg_search_time_ms': 0.0,
                'cache_hit_rate': 0.0,
                'total_searches': 0,
                'successful_searches': 0,
                'failed_searches': 0
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def clear_all_caches(self):
        """
        Clear all caches
        """
        self.embedding_cache.clear()
        self.query_cache.clear()
        logger.info("All caches cleared")