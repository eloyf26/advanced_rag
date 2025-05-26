"""
Search Coordination Agent for Agentic RAG
"""

import asyncio
import logging
from typing import List, Tuple, Optional, Set
from datetime import datetime

from config import RAGConfig
from models.request_models import SearchFilters
from models.response_models import SearchResults, QueryPlan, IterativeSearchState, DocumentChunk
from services.database_manager import RAGDatabaseManager

logger = logging.getLogger(__name__)


class SearchCoordinationAgent:
    """
    Agent responsible for coordinating iterative search strategies
    """
    
    def __init__(self, config: RAGConfig, db_manager: RAGDatabaseManager):
        self.config = config
        self.db_manager = db_manager
        
        # Search state tracking
        self.search_history = []
        self.covered_topics = set()
        self.source_diversity_targets = {
            'file_types': 3,
            'unique_sources': 5,
            'content_domains': 2
        }
    
    async def execute_iterative_search(
        self,
        query_plan: QueryPlan,
        max_iterations: int = 3,
        min_sources: int = 3,
        file_types: Optional[List[str]] = None
    ) -> Tuple[List[SearchResults], IterativeSearchState]:
        """
        Execute iterative search strategy based on query plan
        """
        search_iterations = []
        all_chunks = []
        unique_sources = set()
        coverage_gaps = []
        
        logger.info(f"Starting iterative search with {len(query_plan.sub_queries)} sub-queries")
        
        for iteration in range(max_iterations):
            logger.info(f"Search iteration {iteration + 1}/{max_iterations}")
            
            # Determine search focus for this iteration
            search_focus = self._determine_search_focus(
                iteration, query_plan, all_chunks, coverage_gaps
            )
            
            # Select query for this iteration
            current_query = self._select_iteration_query(
                iteration, query_plan, search_focus
            )
            
            # Adjust search parameters based on iteration and findings
            search_filters = self._create_iteration_filters(
                iteration, file_types, all_chunks, unique_sources
            )
            
            # Execute search for this iteration
            iteration_results = await self._execute_single_search(
                current_query, query_plan.search_strategy, search_filters
            )
            
            if iteration_results:
                search_iterations.append(iteration_results)
                
                # Update tracking
                new_chunks = [
                    chunk for chunk in iteration_results.chunks 
                    if chunk.id not in {c.id for c in all_chunks}
                ]
                all_chunks.extend(new_chunks)
                unique_sources.update(chunk.file_name for chunk in new_chunks)
                
                # Analyze coverage
                coverage_analysis = self._analyze_coverage(
                    query_plan.original_query, all_chunks
                )
                coverage_gaps = coverage_analysis['gaps']
                
                logger.info(f"Iteration {iteration + 1}: Found {len(new_chunks)} new chunks, {len(unique_sources)} unique sources")
            
            # Decide whether to continue
            should_continue = self._should_continue_search(
                iteration, max_iterations, all_chunks, unique_sources, 
                min_sources, coverage_gaps
            )
            
            if not should_continue:
                logger.info(f"Stopping search after {iteration + 1} iterations")
                break
        
        # Create final state
        final_state = IterativeSearchState(
            iteration=len(search_iterations),
            total_chunks_found=len(all_chunks),
            unique_sources=len(unique_sources),
            coverage_gaps=coverage_gaps,
            should_continue=False,
            reasoning=self._create_completion_reasoning(
                len(search_iterations), len(all_chunks), len(unique_sources)
            ),
            search_focus=search_focus if search_iterations else None,
            strategy_adjustments=self._get_strategy_adjustments()
        )
        
        return search_iterations, final_state
    
    def _determine_search_focus(
        self, 
        iteration: int, 
        query_plan: QueryPlan, 
        existing_chunks: List[DocumentChunk],
        coverage_gaps: List[str]
    ) -> str:
        """
        Determine what to focus on for this iteration
        """
        if iteration == 0:
            return "broad_discovery"
        elif iteration == 1:
            if coverage_gaps:
                return f"gap_filling: {coverage_gaps[0]}"
            else:
                return "depth_enhancement"
        else:
            if len(existing_chunks) < 3:
                return "source_diversification"
            else:
                return "quality_refinement"
    
    def _select_iteration_query(
        self, 
        iteration: int, 
        query_plan: QueryPlan, 
        search_focus: str
    ) -> str:
        """
        Select the appropriate query for this iteration
        """
        if iteration < len(query_plan.sub_queries):
            return query_plan.sub_queries[iteration]
        elif "gap_filling" in search_focus:
            gap_topic = search_focus.split(": ")[1]
            return f"{query_plan.original_query} {gap_topic}"
        else:
            # Generate variation of original query
            return self._generate_query_variation(query_plan.original_query, iteration)
    
    def _generate_query_variation(self, original_query: str, iteration: int) -> str:
        """
        Generate variations of the original query for additional iterations
        """
        variations = [
            f"detailed information about {original_query}",
            f"examples and case studies of {original_query}",
            f"background and context for {original_query}",
            f"latest developments in {original_query}"
        ]
        
        variation_index = (iteration - len(variations)) % len(variations)
        return variations[variation_index]
    
    def _create_iteration_filters(
        self,
        iteration: int,
        file_types: Optional[List[str]],
        existing_chunks: List[DocumentChunk],
        unique_sources: Set[str]
    ) -> SearchFilters:
        """
        Create search filters adapted for this iteration
        """
        # Base filters
        filters = SearchFilters(
            file_types=file_types,
            similarity_threshold=self.config.default_similarity_threshold,
            max_results=self.config.default_max_results
        )
        
        # Adjust based on iteration
        if iteration == 0:
            # First iteration: broad search
            filters.similarity_threshold = 0.6
            filters.max_results = 15
        elif iteration == 1:
            # Second iteration: more focused
            filters.similarity_threshold = 0.7
            filters.max_results = 10
            
            # Try to diversify file types
            if existing_chunks:
                existing_types = set(chunk.file_type for chunk in existing_chunks)
                if file_types:
                    remaining_types = [ft for ft in file_types if ft not in existing_types]
                    if remaining_types:
                        filters.file_types = remaining_types[:3]
        else:
            # Later iterations: high precision
            filters.similarity_threshold = 0.75
            filters.max_results = 8
        
        return filters
    
    async def _execute_single_search(
        self, 
        query: str, 
        strategy: str, 
        filters: SearchFilters
    ) -> Optional[SearchResults]:
        """
        Execute a single search iteration
        """
        try:
            start_time = datetime.now()
            
            if strategy == "hybrid":
                chunks = await self.db_manager.hybrid_search(query, filters)
            elif strategy == "semantic":
                chunks = await self.db_manager.semantic_search(query, filters)
            elif strategy == "keyword":
                chunks = await self.db_manager.keyword_search(query, filters)
            else:
                # Default to hybrid
                chunks = await self.db_manager.hybrid_search(query, filters)
            
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResults(
                query=query,
                chunks=chunks,
                total_found=len(chunks),
                search_time_ms=search_time,
                search_method=strategy,
                filters_applied=filters.dict(),
                cache_hit=False  # Would be determined by actual cache implementation
            )
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return None
    
    def _analyze_coverage(
        self, 
        original_query: str, 
        chunks: List[DocumentChunk]
    ) -> dict:
        """
        Analyze how well the current chunks cover the query
        """
        if not chunks:
            return {'gaps': ['no_results'], 'coverage_score': 0.0}
        
        # Extract query keywords
        query_keywords = set(original_query.lower().split())
        query_keywords = {word for word in query_keywords if len(word) > 3}
        
        # Extract keywords from chunks
        found_keywords = set()
        for chunk in chunks:
            found_keywords.update(chunk.keywords)
            # Also extract from content (simplified)
            content_words = set(chunk.content.lower().split())
            found_keywords.update(word for word in content_words if len(word) > 3)
        
        # Find gaps
        gaps = list(query_keywords - found_keywords)
        
        # Calculate coverage score
        coverage_score = len(query_keywords & found_keywords) / max(len(query_keywords), 1)
        
        return {
            'gaps': gaps[:3],  # Top 3 gaps
            'coverage_score': coverage_score,
            'found_keywords': list(found_keywords)
        }
    
    def _should_continue_search(
        self,
        iteration: int,
        max_iterations: int,
        all_chunks: List[DocumentChunk],
        unique_sources: Set[str],
        min_sources: int,
        coverage_gaps: List[str]
    ) -> bool:
        """
        Decide whether to continue with another search iteration
        """
        # Don't exceed max iterations
        if iteration >= max_iterations - 1:
            return False
        
        # Continue if we don't have enough sources
        if len(unique_sources) < min_sources:
            return True
        
        # Continue if we have significant coverage gaps and few results
        if coverage_gaps and len(all_chunks) < 5:
            return True
        
        # Stop if we have good coverage
        if len(all_chunks) >= 8 and len(unique_sources) >= min_sources:
            return False
        
        # Continue if the last iteration found new information
        return len(all_chunks) > 0
    
    def _create_completion_reasoning(
        self, 
        iterations: int, 
        total_chunks: int, 
        unique_sources: int
    ) -> str:
        """
        Create reasoning for search completion
        """
        reasons = []
        
        if iterations == 1:
            reasons.append("Single iteration provided sufficient results")
        else:
            reasons.append(f"Completed {iterations} search iterations")
        
        if total_chunks > 8:
            reasons.append("Found comprehensive set of relevant documents")
        elif total_chunks > 3:
            reasons.append("Found adequate number of relevant sources")
        else:
            reasons.append("Limited relevant information available")
        
        if unique_sources >= 5:
            reasons.append("Good source diversity achieved")
        elif unique_sources >= 3:
            reasons.append("Adequate source diversity")
        else:
            reasons.append("Limited source diversity")
        
        return ". ".join(reasons) + "."
    
    def _get_strategy_adjustments(self) -> List[str]:
        """
        Get list of strategy adjustments made during search
        """
        # This would track actual adjustments made during search
        # For now, return placeholder
        return [
            "Adjusted similarity thresholds per iteration",
            "Diversified file type targeting",
            "Refined query focus based on gaps"
        ]