"""
Search Tools for Agentic RAG Agent
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.request_models import SearchFilters
from models.response_models import SearchResults, DocumentChunk
from services.database_manager import RAGDatabaseManager

logger = logging.getLogger(__name__)


class SearchToolkit:
    """
    Collection of search tools for the agentic RAG system
    """
    
    def __init__(self, db_manager: RAGDatabaseManager):
        self.db_manager = db_manager
        
        # Search strategy mappings
        self.search_strategies = {
            'hybrid': self.hybrid_search,
            'semantic': self.semantic_search,
            'keyword': self.keyword_search,
            'contextual': self.contextual_search,
            'similarity': self.similarity_search
        }
    
    async def hybrid_search(self, query: str, filters: SearchFilters) -> SearchResults:
        """
        Perform hybrid search combining vector and keyword search
        """
        start_time = datetime.now()
        
        try:
            chunks = await self.db_manager.hybrid_search(query, filters)
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResults(
                query=query,
                chunks=chunks,
                total_found=len(chunks),
                search_time_ms=search_time,
                search_method="hybrid",
                filters_applied=filters.dict(),
                index_used="hybrid_index",
                cache_hit=False  # Would be determined by actual cache implementation
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self._create_empty_search_results(query, "hybrid", filters)
    
    async def semantic_search(self, query: str, filters: SearchFilters) -> SearchResults:
        """
        Perform pure semantic vector search
        """
        start_time = datetime.now()
        
        try:
            chunks = await self.db_manager.semantic_search(query, filters)
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResults(
                query=query,
                chunks=chunks,
                total_found=len(chunks),
                search_time_ms=search_time,
                search_method="semantic",
                filters_applied=filters.dict(),
                index_used="vector_index",
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._create_empty_search_results(query, "semantic", filters)
    
    async def keyword_search(self, query: str, filters: SearchFilters) -> SearchResults:
        """
        Perform keyword-based BM25 search
        """
        start_time = datetime.now()
        
        try:
            chunks = await self.db_manager.keyword_search(query, filters)
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResults(
                query=query,
                chunks=chunks,
                total_found=len(chunks),
                search_time_ms=search_time,
                search_method="keyword",
                filters_applied=filters.dict(),
                index_used="text_index",
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return self._create_empty_search_results(query, "keyword", filters)
    
    async def contextual_search(
        self, 
        query: str, 
        filters: SearchFilters,
        context_chunks: List[DocumentChunk] = None
    ) -> SearchResults:
        """
        Perform contextual search considering existing context
        """
        start_time = datetime.now()
        
        try:
            # Enhance query with context information
            enhanced_query = self._enhance_query_with_context(query, context_chunks)
            
            # Perform hybrid search with enhanced query
            chunks = await self.db_manager.hybrid_search(enhanced_query, filters)
            
            # Filter out chunks that are too similar to context
            if context_chunks:
                chunks = self._filter_similar_to_context(chunks, context_chunks)
            
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResults(
                query=enhanced_query,
                chunks=chunks,
                total_found=len(chunks),
                search_time_ms=search_time,
                search_method="contextual",
                filters_applied=filters.dict(),
                index_used="hybrid_index",
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Error in contextual search: {e}")
            return self._create_empty_search_results(query, "contextual", filters)
    
    async def similarity_search(
        self, 
        reference_chunk: DocumentChunk, 
        filters: SearchFilters
    ) -> SearchResults:
        """
        Find chunks similar to a reference chunk
        """
        start_time = datetime.now()
        
        try:
            chunks = await self.db_manager.get_similar_chunks(
                chunk_id=reference_chunk.id,
                similarity_threshold=filters.similarity_threshold,
                limit=filters.max_results
            )
            
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResults(
                query=f"Similar to: {reference_chunk.content[:100]}...",
                chunks=chunks,
                total_found=len(chunks),
                search_time_ms=search_time,
                search_method="similarity",
                filters_applied=filters.dict(),
                index_used="vector_index",
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return self._create_empty_search_results(
                f"Similar to chunk {reference_chunk.id}", "similarity", filters
            )
    
    async def multi_strategy_search(
        self, 
        query: str, 
        strategies: List[str], 
        filters: SearchFilters
    ) -> Dict[str, SearchResults]:
        """
        Perform search using multiple strategies and return all results
        """
        results = {}
        
        # Execute searches concurrently
        tasks = []
        for strategy in strategies:
            if strategy in self.search_strategies:
                task = self.search_strategies[strategy](query, filters)
                tasks.append((strategy, task))
        
        # Wait for all searches to complete
        completed_results = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        # Collect results
        for i, (strategy, _) in enumerate(tasks):
            result = completed_results[i]
            if isinstance(result, Exception):
                logger.error(f"Error in {strategy} search: {result}")
                results[strategy] = self._create_empty_search_results(query, strategy, filters)
            else:
                results[strategy] = result
        
        return results
    
    async def adaptive_search(
        self, 
        query: str, 
        filters: SearchFilters,
        fallback_strategies: List[str] = None
    ) -> SearchResults:
        """
        Adaptive search that tries different strategies based on results quality
        """
        if fallback_strategies is None:
            fallback_strategies = ['hybrid', 'semantic', 'keyword']
        
        best_result = None
        best_score = 0.0
        
        for strategy in fallback_strategies:
            try:
                result = await self.search_strategies[strategy](query, filters)
                
                # Calculate quality score for this result
                quality_score = self._calculate_result_quality(result)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_result = result
                
                # If we get a good result, stop trying other strategies
                if quality_score > 0.8:
                    break
                    
            except Exception as e:
                logger.error(f"Error in adaptive search with {strategy}: {e}")
                continue
        
        return best_result or self._create_empty_search_results(query, "adaptive", filters)
    
    async def search_with_expansion(
        self, 
        query: str, 
        filters: SearchFilters,
        expansion_terms: List[str] = None
    ) -> SearchResults:
        """
        Search with query expansion using related terms
        """
        try:
            # Generate expansion terms if not provided
            if expansion_terms is None:
                expansion_terms = await self._generate_expansion_terms(query)
            
            # Create expanded query
            expanded_query = f"{query} {' '.join(expansion_terms)}"
            
            # Perform search with expanded query
            return await self.hybrid_search(expanded_query, filters)
            
        except Exception as e:
            logger.error(f"Error in search with expansion: {e}")
            return self._create_empty_search_results(query, "expanded", filters)
    
    async def federated_search(
        self, 
        query: str, 
        filters_list: List[SearchFilters]
    ) -> List[SearchResults]:
        """
        Perform federated search across different filter configurations
        """
        results = []
        
        # Execute searches with different filters concurrently
        tasks = [self.hybrid_search(query, filters) for filters in filters_list]
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Error in federated search {i}: {result}")
                results.append(self._create_empty_search_results(query, "federated", filters_list[i]))
            else:
                results.append(result)
        
        return results
    
    def _enhance_query_with_context(
        self, 
        query: str, 
        context_chunks: List[DocumentChunk]
    ) -> str:
        """
        Enhance query with context information
        """
        if not context_chunks:
            return query
        
        # Extract key terms from context
        context_terms = set()
        for chunk in context_chunks[:3]:  # Use top 3 context chunks
            if chunk.keywords:
                context_terms.update(chunk.keywords[:3])  # Top 3 keywords per chunk
        
        # Add relevant context terms to query
        enhanced_terms = list(context_terms)[:5]  # Limit to 5 additional terms
        
        if enhanced_terms:
            return f"{query} {' '.join(enhanced_terms)}"
        else:
            return query
    
    def _filter_similar_to_context(
        self, 
        chunks: List[DocumentChunk], 
        context_chunks: List[DocumentChunk],
        similarity_threshold: float = 0.9
    ) -> List[DocumentChunk]:
        """
        Filter out chunks that are too similar to existing context
        """
        if not context_chunks:
            return chunks
        
        context_content = set()
        for chunk in context_chunks:
            # Create a simplified representation for comparison
            words = set(chunk.content.lower().split())
            context_content.update(words)
        
        filtered_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            
            # Calculate overlap with context
            if context_content:
                overlap = len(chunk_words & context_content) / len(chunk_words | context_content)
                
                # Keep chunks that are not too similar to context
                if overlap < similarity_threshold:
                    filtered_chunks.append(chunk)
            else:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    async def _generate_expansion_terms(self, query: str) -> List[str]:
        """
        Generate query expansion terms (placeholder implementation)
        """
        # In a real implementation, this could use:
        # - WordNet for synonyms
        # - Word embeddings for related terms
        # - Domain-specific thesauri
        # - LLM-based expansion
        
        # Simple expansion based on common patterns
        expansion_terms = []
        
        query_lower = query.lower()
        
        # Add related terms based on common patterns
        if 'machine learning' in query_lower:
            expansion_terms.extend(['AI', 'artificial intelligence', 'ML', 'algorithms'])
        elif 'data science' in query_lower:
            expansion_terms.extend(['analytics', 'statistics', 'data analysis'])
        elif 'programming' in query_lower:
            expansion_terms.extend(['coding', 'development', 'software'])
        
        return expansion_terms[:3]  # Limit to 3 expansion terms
    
    def _calculate_result_quality(self, result: SearchResults) -> float:
        """
        Calculate quality score for search results
        """
        if not result.chunks:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Number of results
        result_count_score = min(len(result.chunks) / 5.0, 1.0)
        quality_factors.append(result_count_score)
        
        # Factor 2: Average similarity/relevance scores
        avg_similarity = sum(chunk.similarity_score for chunk in result.chunks) / len(result.chunks)
        quality_factors.append(avg_similarity)
        
        # Factor 3: Source diversity
        unique_sources = len(set(chunk.file_name for chunk in result.chunks))
        source_diversity = min(unique_sources / 3.0, 1.0)
        quality_factors.append(source_diversity)
        
        # Factor 4: Content length (longer content often means more detailed)
        avg_content_length = sum(len(chunk.content) for chunk in result.chunks) / len(result.chunks)
        length_score = min(avg_content_length / 500.0, 1.0)  # Normalize to 500 chars
        quality_factors.append(length_score)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _create_empty_search_results(
        self, 
        query: str, 
        method: str, 
        filters: SearchFilters
    ) -> SearchResults:
        """
        Create empty search results for error cases
        """
        return SearchResults(
            query=query,
            chunks=[],
            total_found=0,
            search_time_ms=0,
            search_method=method,
            filters_applied=filters.dict(),
            cache_hit=False
        )