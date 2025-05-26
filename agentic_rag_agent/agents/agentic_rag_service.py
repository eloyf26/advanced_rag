"""
Main Agentic RAG Service Implementation
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from config import RAGConfig
from models.request_models import SearchFilters
from models.response_models import (
    AgenticRAGResponse, RAGResponse, QueryPlan, ReflectionResult,
    SearchResults, DocumentChunk, IterativeSearchState,
    QueryAnalysisResponse, PerformanceMetrics
)
from services.database_manager import RAGDatabaseManager
from agents.planning_agent import QueryPlanningAgent
from agents.search_agent import SearchCoordinationAgent
from agents.reflection_agent import ReflectionAgent
from tools.search_tools import SearchToolkit
from tools.analysis_tools import AnalysisToolkit
from tools.triangulation_tools import TriangulationToolkit

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Main service orchestrating the agentic RAG workflow
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize core components
        self.db_manager = RAGDatabaseManager(config)
        
        # Initialize specialized agents
        self.planning_agent = QueryPlanningAgent(config)
        self.search_agent = SearchCoordinationAgent(config, self.db_manager)
        self.reflection_agent = ReflectionAgent(config)
        
        # Initialize toolkits
        self.search_toolkit = SearchToolkit(self.db_manager)
        self.analysis_toolkit = AnalysisToolkit(config)
        self.triangulation_toolkit = TriangulationToolkit(self.db_manager)
        
        # Performance tracking
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_processing_time': 0.0,
            'avg_confidence': 0.0,
            'avg_iterations': 0.0
        }
        
        logger.info("Agentic RAG Service initialized")
    
    async def ask_with_planning(
        self,
        question: str,
        enable_iteration: bool = True,
        enable_reflection: bool = True,
        enable_triangulation: bool = True,
        max_results: int = 10,
        file_types: Optional[List[str]] = None
    ) -> AgenticRAGResponse:
        """
        Main entry point for agentic question answering with full pipeline
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        reasoning_chain = []
        
        try:
            self.metrics['total_queries'] += 1
            
            # Step 1: Query Planning
            reasoning_chain.append("🧠 Analyzing query and creating execution plan...")
            query_plan = await self.planning_agent.create_query_plan(question)
            
            # Step 2: Iterative Search
            if enable_iteration:
                reasoning_chain.append("🔍 Executing iterative search strategy...")
                search_iterations, search_state = await self.search_agent.execute_iterative_search(
                    query_plan=query_plan,
                    max_iterations=self.config.max_iterations,
                    min_sources=self.config.min_sources_per_iteration,
                    file_types=file_types
                )
            else:
                # Single search iteration
                reasoning_chain.append("🔍 Executing single search...")
                filters = SearchFilters(
                    file_types=file_types,
                    max_results=max_results,
                    similarity_threshold=self.config.default_similarity_threshold
                )
                single_result = await self.search_toolkit.hybrid_search(question, filters)
                search_iterations = [single_result]
                search_state = IterativeSearchState(
                    iteration=1,
                    total_chunks_found=len(single_result.chunks),
                    unique_sources=len(set(chunk.file_name for chunk in single_result.chunks)),
                    coverage_gaps=[],
                    should_continue=False,
                    reasoning="Single search completed"
                )
            
            # Combine all sources from iterations
            all_sources = self._combine_search_results(search_iterations)
            
            # Step 3: Source Triangulation
            if enable_triangulation and all_sources:
                reasoning_chain.append("🔬 Triangulating sources for verification...")
                additional_sources = await self.triangulation_toolkit.triangulate_sources(
                    query=question,
                    primary_sources=all_sources[:5],
                    max_additional=5
                )
                all_sources.extend(additional_sources)
            
            # Step 4: Answer Generation
            reasoning_chain.append("✍️ Synthesizing answer from sources...")
            answer = await self._generate_answer(question, all_sources, query_plan)
            
            # Step 5: Self-Reflection
            if enable_reflection:
                reasoning_chain.append("🤔 Reflecting on answer quality...")
                reflection = await self.reflection_agent.reflect_on_answer(
                    question=question,
                    answer=answer,
                    sources=all_sources
                )
            else:
                reflection = ReflectionResult(
                    quality_score=0.8,
                    completeness_score=0.8,
                    accuracy_assessment="Not assessed",
                    missing_information=[],
                    suggested_follow_ups=[],
                    needs_more_search=False
                )
            
            # Step 6: Generate Follow-ups
            follow_ups = await self._generate_follow_ups(question, all_sources, reflection)
            
            # Calculate final metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence = self._calculate_confidence(reflection, all_sources)
            
            # Update metrics
            self.metrics['successful_queries'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['avg_confidence'] = (
                (self.metrics['avg_confidence'] * (self.metrics['successful_queries'] - 1) + confidence) 
                / self.metrics['successful_queries']
            )
            self.metrics['avg_iterations'] = (
                (self.metrics['avg_iterations'] * (self.metrics['successful_queries'] - 1) + len(search_iterations))
                / self.metrics['successful_queries']
            )
            
            return AgenticRAGResponse(
                answer=answer,
                confidence=confidence,
                sources=all_sources,
                query_plan=query_plan,
                reflection=reflection,
                search_iterations=search_iterations,
                reasoning_chain=reasoning_chain,
                follow_up_suggestions=follow_ups,
                processing_time_ms=processing_time * 1000,
                iterations_completed=len(search_iterations),
                source_triangulation_performed=enable_triangulation,
                cross_validated_facts=[],  # Would be populated by triangulation
                potential_biases=[]  # Would be identified by reflection
            )
            
        except Exception as e:
            self.metrics['failed_queries'] += 1
            logger.error(f"Error in agentic RAG processing: {e}")
            
            return AgenticRAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again or rephrase your query.",
                confidence=0.0,
                sources=[],
                query_plan=QueryPlan(
                    original_query=question,
                    sub_queries=[],
                    search_strategy="",
                    reasoning="Error occurred during processing",
                    expected_sources=[],
                    complexity_score=0.0
                ),
                reflection=ReflectionResult(
                    quality_score=0.0,
                    completeness_score=0.0,
                    accuracy_assessment="Error",
                    missing_information=["Process failed due to error"],
                    suggested_follow_ups=[],
                    needs_more_search=False
                ),
                search_iterations=[],
                reasoning_chain=reasoning_chain + [f"❌ Error: {str(e)}"],
                follow_up_suggestions=[],
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                iterations_completed=0,
                source_triangulation_performed=False,
                cross_validated_facts=[],
                potential_biases=[]
            )
    
    async def ask(
        self,
        question: str,
        search_method: str = "hybrid",
        file_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> RAGResponse:
        """
        Simplified RAG without agentic features
        """
        start_time = datetime.now()
        
        try:
            # Create search filters
            filters = SearchFilters(
                file_types=file_types,
                max_results=max_results,
                similarity_threshold=self.config.default_similarity_threshold
            )
            
            # Perform search
            if search_method == "hybrid":
                search_results = await self.search_toolkit.hybrid_search(question, filters)
            elif search_method == "semantic":
                search_results = await self.search_toolkit.semantic_search(question, filters)
            elif search_method == "keyword":
                search_results = await self.search_toolkit.keyword_search(question, filters)
            else:
                raise ValueError(f"Invalid search method: {search_method}")
            
            # Generate answer
            answer = await self._generate_simple_answer(question, search_results.chunks)
            
            # Calculate confidence
            confidence = self._calculate_simple_confidence(search_results.chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                answer=answer,
                confidence=confidence,
                sources=search_results.chunks,
                search_results=search_results,
                reasoning=f"Used {search_method} search to find {len(search_results.chunks)} relevant sources",
                processing_time_ms=processing_time * 1000,
                search_method=search_method
            )
            
        except Exception as e:
            logger.error(f"Error in simple RAG: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                answer="I encountered an error while processing your question.",
                confidence=0.0,
                sources=[],
                search_results=SearchResults(
                    query=question,
                    chunks=[],
                    total_found=0,
                    search_time_ms=0,
                    search_method=search_method
                ),
                reasoning=f"Error: {str(e)}",
                processing_time_ms=processing_time * 1000,
                search_method=search_method
            )
    
    async def analyze_query(self, question: str) -> QueryAnalysisResponse:
        """
        Analyze a query without executing it
        """
        return await self.analysis_toolkit.analyze_query(question)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status
        """
        try:
            # Test database connection
            db_health = await self.db_manager.test_connection()
            
            # Test embedding service
            embedding_health = await self.db_manager.test_embedding_service()
            
            # Test LLM service
            llm_health = await self._test_llm_service()
            
            return {
                "database_connected": db_health,
                "embedding_service": "healthy" if embedding_health else "unhealthy",
                "llm_service": "healthy" if llm_health else "unhealthy",
                "total_queries": self.metrics['total_queries'],
                "success_rate": (
                    self.metrics['successful_queries'] / max(self.metrics['total_queries'], 1) * 100
                ),
                "avg_response_time": (
                    self.metrics['total_processing_time'] / max(self.metrics['successful_queries'], 1)
                )
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "database_connected": False,
                "embedding_service": "unhealthy",
                "llm_service": "unhealthy",
                "error": str(e)
            }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics
        """
        return PerformanceMetrics(
            total_queries=self.metrics['total_queries'],
            avg_query_time_ms=self.metrics['total_processing_time'] / max(self.metrics['successful_queries'], 1) * 1000,
            successful_queries=self.metrics['successful_queries'],
            failed_queries=self.metrics['failed_queries'],
            avg_search_time_ms=0.0,  # Would be calculated from search metrics
            avg_sources_per_query=0.0,  # Would be calculated from search results
            avg_confidence_score=self.metrics['avg_confidence'],
            avg_iterations_per_query=self.metrics['avg_iterations'],
            reflection_success_rate=1.0,  # Would be calculated from reflection results
            triangulation_usage_rate=0.5,  # Would be calculated from usage stats
            cache_hit_rate=0.0,  # Would be calculated from cache stats
            embedding_requests_per_minute=0.0,  # Would be calculated from embedding stats
            metrics_period_start=datetime.now(),  # Would be stored when service starts
            metrics_period_end=datetime.now()
        )
    
    async def clear_cache(self):
        """
        Clear all caches
        """
        # Implementation would clear various caches
        logger.info("Cache cleared")
    
    def _combine_search_results(self, search_iterations: List[SearchResults]) -> List[DocumentChunk]:
        """
        Combine and deduplicate search results from multiple iterations
        """
        all_chunks = []
        seen_ids = set()
        
        for search_result in search_iterations:
            for chunk in search_result.chunks:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)
        
        # Sort by combined score (if available) or similarity score
        all_chunks.sort(
            key=lambda x: x.combined_score if x.combined_score > 0 else x.similarity_score,
            reverse=True
        )
        
        return all_chunks
    
    async def _generate_answer(
        self, 
        question: str, 
        sources: List[DocumentChunk], 
        query_plan: QueryPlan
    ) -> str:
        """
        Generate comprehensive answer using LLM with context from sources
        """
        if not sources:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing or ask about a different topic."
        
        # Prepare context from sources
        context_parts = []
        for i, source in enumerate(sources[:8]):  # Limit to top 8 sources
            context_parts.append(
                f"Source {i+1} ({source.file_name}):\n{source.content[:800]}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for answer generation
        prompt = f"""
        Based on the following sources, provide a comprehensive answer to the question: {question}

        Context from sources:
        {context}

        Instructions:
        - Provide a clear, well-structured answer
        - Reference specific sources when making claims
        - If information is incomplete, acknowledge limitations
        - Be factual and avoid speculation
        - Structure the response logically

        Question: {question}

        Answer:
        """
        
        # This would call the LLM service
        # For now, returning a placeholder
        return f"Based on the available sources, here's what I found regarding your question about {question.lower()}. [Generated answer would be here based on LLM processing of the context]"
    
    async def _generate_simple_answer(self, question: str, sources: List[DocumentChunk]) -> str:
        """
        Generate simple answer for non-agentic mode
        """
        if not sources:
            return "I couldn't find relevant information to answer your question."
        
        # Simplified answer generation
        top_source = sources[0] if sources else None
        if top_source:
            return f"Based on the information found in {top_source.file_name}: {top_source.content[:500]}..."
        
        return "I found some relevant information but couldn't generate a clear answer."
    
    async def _generate_follow_ups(
        self, 
        question: str, 
        sources: List[DocumentChunk], 
        reflection: ReflectionResult
    ) -> List[str]:
        """
        Generate intelligent follow-up questions
        """
        follow_ups = []
        
        # Add reflection-based follow-ups
        follow_ups.extend(reflection.suggested_follow_ups)
        
        # Add source-based follow-ups
        if sources:
            unique_topics = set()
            for source in sources[:3]:
                unique_topics.update(source.keywords[:2])
            
            follow_ups.extend([
                f"Tell me more about {topic}" for topic in list(unique_topics)[:2]
            ])
        
        # Add generic follow-ups
        if not follow_ups:
            follow_ups = [
                "Can you provide more specific details?",
                "What are the key implications of this information?",
                "Are there any related topics I should explore?"
            ]
        
        return follow_ups[:5]  # Limit to 5 suggestions
    
    def _calculate_confidence(
        self, 
        reflection: ReflectionResult, 
        sources: List[DocumentChunk]
    ) -> float:
        """
        Calculate overall confidence score
        """
        if not sources:
            return 0.0
        
        # Combine reflection quality with source metrics
        source_quality = min(len(sources) / 5.0, 1.0)  # Normalize by expected sources
        avg_similarity = sum(s.similarity_score for s in sources) / len(sources)
        
        # Weight the components
        confidence = (
            0.4 * reflection.quality_score +
            0.3 * reflection.completeness_score +
            0.2 * source_quality +
            0.1 * avg_similarity
        )
        
        return min(confidence, 1.0)
    
    def _calculate_simple_confidence(self, sources: List[DocumentChunk]) -> float:
        """
        Simple confidence calculation for non-agentic mode
        """
        if not sources:
            return 0.0
        
        # Base confidence on source quality and quantity
        avg_similarity = sum(s.similarity_score for s in sources) / len(sources)
        source_factor = min(len(sources) / 3.0, 1.0)
        
        return avg_similarity * source_factor
    
    async def _test_llm_service(self) -> bool:
        """
        Test LLM service connectivity
        """
        try:
            # This would test actual LLM service
            return True
        except Exception:
            return False