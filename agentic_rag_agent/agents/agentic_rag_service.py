"""
Complete Agentic RAG Service Implementation with All Fixes and Enhancements
File: agentic_rag_agent/agents/agentic_rag_service.py
"""

import asyncio
import logging
import os  # FIXED: Added missing import
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from config import RAGConfig
from models.request_models import SearchFilters
from models.response_models import (
    AgenticRAGResponse, RAGResponse, QueryPlan, ReflectionResult,
    SearchResults, DocumentChunk, IterativeSearchState,
    QueryAnalysisResponse, PerformanceMetrics
)
from services.database_manager import RAGDatabaseManager
from services.llm_service import LLMService
from agents.planning_agent import QueryPlanningAgent
from agents.search_agent import SearchCoordinationAgent
from agents.reflection_agent import ReflectionAgent
from tools.search_tools import SearchToolkit  # FIXED: Corrected import name
from tools.analysis_tools import AnalysisToolkit
from tools.triangulation_tools import TriangulationToolkit
from utils.logger import get_logger
from utils.metrics import get_metrics_collector

logger = get_logger(__name__)


class AgenticRAGService:
    """
    Main service orchestrating the agentic RAG workflow with complete implementation
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize core components
        self.db_manager = RAGDatabaseManager(config)
        self.llm_service = LLMService(config)
        
        # Initialize specialized agents
        self.planning_agent = QueryPlanningAgent(config)
        self.search_agent = SearchCoordinationAgent(config, self.db_manager)
        self.reflection_agent = ReflectionAgent(config)
        
        # Initialize toolkits
        self.search_toolkit = SearchToolkit(self.db_manager)
        self.analysis_toolkit = AnalysisToolkit(config)
        self.triangulation_toolkit = TriangulationToolkit(self.db_manager)
        
        # Initialize metrics collector
        self.metrics = get_metrics_collector()
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_processing_time': 0.0,
            'avg_confidence': 0.0,
            'avg_iterations': 0.0,
            'llm_calls': 0,
            'cache_hits': 0
        }
        
        # Cache management
        self._query_cache = {}
        self._query_cache_max_size = 1000
        self._last_answer_context = {}
        
        # Service state
        self._startup_time = datetime.now()
        self._background_tasks = []
        
        logger.info("Agentic RAG Service initialized with complete implementation")
    
    async def start_background_services(self):
        """Start background services and tasks"""
        try:
            # Start embedding service background processor
            if hasattr(self, 'embedding_service'):
                await self.embedding_service.start()
            
            # Start cache cleanup task
            task1 = asyncio.create_task(self._periodic_cache_cleanup())
            self._background_tasks.append(task1)
            
            # Start metrics collection task
            task2 = asyncio.create_task(self._periodic_metrics_collection())
            self._background_tasks.append(task2)
            
            # Start health monitoring task
            task3 = asyncio.create_task(self._periodic_health_monitoring())
            self._background_tasks.append(task3)
            
            logger.info("Background services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup service resources"""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Stop embedding service
            if hasattr(self, 'embedding_service'):
                await self.embedding_service.stop()
            
            # Clear caches
            await self.clear_cache()
            
            # Close database connections
            if hasattr(self.db_manager, '__aexit__'):
                await self.db_manager.__aexit__(None, None, None)
            
            logger.info("Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during service cleanup: {e}")
    
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
            self.stats['total_queries'] += 1
            
            # Step 1: Query Planning
            reasoning_chain.append("🧠 Analyzing query and creating execution plan...")
            query_plan = await self.planning_agent.create_query_plan(question)
            
            # Enhanced query if needed
            if len(question.split()) < 5:  # Short queries benefit from enhancement
                reasoning_chain.append("🔍 Enhancing query for better search results...")
                enhanced_question = await self.llm_service.enhance_query(
                    question, 
                    f"Query type: {query_plan.search_strategy}, Complexity: {query_plan.complexity_score}"
                )
                # Use enhanced query for search but keep original for response
                search_question = enhanced_question
                self.stats['llm_calls'] += 1
            else:
                search_question = question
            
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
                single_result = await self.search_toolkit.hybrid_search(search_question, filters)
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
                triangulation_result = await self.triangulation_toolkit.verify_information_consistency(all_sources)
                
                # Get additional verification sources
                additional_sources = await self.triangulation_toolkit.triangulate_sources(
                    query=question,
                    primary_sources=all_sources[:5],
                    max_additional=5
                )
                all_sources.extend(additional_sources)
                
                reasoning_chain.append(f"✓ Source triangulation complete. Consistency score: {triangulation_result['consistency_score']:.2f}")
            
            # Step 4: Answer Generation with LLM
            reasoning_chain.append("✍️ Synthesizing comprehensive answer from sources...")
            answer = await self._generate_answer(question, all_sources, query_plan)
            self.stats['llm_calls'] += 1
            
            # Store answer context for follow-ups
            self._last_answer_context[query_id] = {
                'question': question,
                'answer': answer,
                'sources': all_sources[:5],  # Store top 5 sources
                'timestamp': datetime.now()
            }
            
            # Step 5: Self-Reflection
            if enable_reflection:
                reasoning_chain.append("🤔 Reflecting on answer quality...")
                reflection = await self.reflection_agent.reflect_on_answer(
                    question=question,
                    answer=answer,
                    sources=all_sources
                )
                
                # Check if additional search is recommended
                if reflection.needs_more_search and len(search_iterations) < self.config.max_iterations:
                    reasoning_chain.append("🔄 Reflection suggests additional search needed...")
                    # Perform one more search iteration based on reflection
                    additional_search = await self._perform_reflection_based_search(
                        question, reflection.missing_information, all_sources
                    )
                    if additional_search.chunks:
                        search_iterations.append(additional_search)
                        all_sources.extend(additional_search.chunks)
                        # Regenerate answer with new sources
                        answer = await self._generate_answer(question, all_sources, query_plan)
                        self.stats['llm_calls'] += 1
                        reasoning_chain.append("✨ Answer enhanced with additional sources")
            else:
                reflection = ReflectionResult(
                    quality_score=0.8,
                    completeness_score=0.8,
                    accuracy_assessment="Not assessed",
                    missing_information=[],
                    suggested_follow_ups=[],
                    needs_more_search=False
                )
            
            # Step 6: Generate Follow-ups with LLM
            reasoning_chain.append("💡 Generating intelligent follow-up suggestions...")
            follow_ups = await self._generate_follow_ups(question, answer, all_sources, reflection)
            self.stats['llm_calls'] += 1
            
            # Step 7: Extract cross-validated facts and potential biases
            cross_validated_facts = []
            potential_biases = []
            
            if enable_triangulation and len(all_sources) > 2:
                # Simple fact extraction from highly similar sources
                cross_validated_facts = self._extract_cross_validated_facts(all_sources)
                potential_biases = self._identify_potential_biases(all_sources)
            
            # Calculate final metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence = self._calculate_confidence(reflection, all_sources)
            
            # Update metrics
            self.stats['successful_queries'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_confidence'] = (
                (self.stats['avg_confidence'] * (self.stats['successful_queries'] - 1) + confidence) 
                / self.stats['successful_queries']
            )
            self.stats['avg_iterations'] = (
                (self.stats['avg_iterations'] * (self.stats['successful_queries'] - 1) + len(search_iterations))
                / self.stats['successful_queries']
            )
            
            # Record metrics
            self.metrics.record_query(
                query_type='agentic',
                duration=processing_time,
                status='success'
            )
            
            return AgenticRAGResponse(
                answer=answer,
                confidence=confidence,
                sources=all_sources[:15],  # Limit sources in response
                query_plan=query_plan,
                reflection=reflection,
                search_iterations=search_iterations,
                reasoning_chain=reasoning_chain,
                follow_up_suggestions=follow_ups,
                processing_time_ms=processing_time * 1000,
                tokens_used=self._estimate_tokens_used(question, answer, all_sources),
                iterations_completed=len(search_iterations),
                source_triangulation_performed=enable_triangulation,
                cross_validated_facts=cross_validated_facts,
                potential_biases=potential_biases
            )
            
        except Exception as e:
            self.stats['failed_queries'] += 1
            logger.error(f"Error in agentic RAG processing: {e}")
            
            # Generate error response with LLM if possible
            try:
                error_answer = await self.llm_service.generate_simple_answer(
                    question, 
                    DocumentChunk(
                        id="error",
                        content=f"An error occurred while processing your question: {str(e)}",
                        file_name="system_error",
                        file_type="error",
                        chunk_index=0,
                        similarity_score=0.0
                    )
                )
                self.stats['llm_calls'] += 1
            except:
                error_answer = "I apologize, but I encountered an error while processing your question. Please try again or rephrase your query."
            
            return AgenticRAGResponse(
                answer=error_answer,
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
        Simplified RAG without agentic features - now with LLM integration
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
            
            # Generate answer with LLM
            if search_results.chunks:
                answer = await self.llm_service.generate_simple_answer(question, search_results.chunks[0])
                self.stats['llm_calls'] += 1
            else:
                answer = "I couldn't find relevant information to answer your question."
            
            # Calculate confidence
            confidence = self._calculate_simple_confidence(search_results.chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics.record_query(
                query_type='simple',
                duration=processing_time,
                status='success'
            )
            
            return RAGResponse(
                answer=answer,
                confidence=confidence,
                sources=search_results.chunks,
                search_results=search_results,
                reasoning=f"Used {search_method} search to find {len(search_results.chunks)} relevant sources, then generated answer using {self.config.llm_model}",
                processing_time_ms=processing_time * 1000,
                search_method=search_method
            )
            
        except Exception as e:
            logger.error(f"Error in simple RAG: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.metrics.record_query(
                query_type='simple',
                duration=processing_time,
                status='error'
            )
            
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
        Analyze a query without executing it - enhanced with LLM insights
        """
        try:
            # Get basic analysis
            analysis = await self.analysis_toolkit.analyze_query(question)
            
            # Enhance with LLM content analysis
            content_analysis = await self.llm_service.analyze_content(question)
            self.stats['llm_calls'] += 1
            
            # Merge insights
            analysis.optimization_suggestions.extend([
                f"LLM suggests this is {content_analysis.get('content_type', 'general')} content",
                f"Complexity level: {content_analysis.get('complexity', 'unknown')}"
            ])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return await self.analysis_toolkit.analyze_query(question)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status including LLM service
        """
        try:
            # Test database connection
            db_health = await self.db_manager.test_connection()
            
            # Test embedding service
            embedding_health = await self.db_manager.test_embedding_service()
            
            # Test LLM service
            llm_health = await self.llm_service.test_llm_connection()
            
            return {
                "database_connected": db_health,
                "embedding_service": "healthy" if embedding_health else "unhealthy",
                "llm_service": "healthy" if llm_health else "unhealthy",
                "total_queries": self.stats['total_queries'],
                "success_rate": (
                    self.stats['successful_queries'] / max(self.stats['total_queries'], 1) * 100
                ),
                "avg_response_time": (
                    self.stats['total_processing_time'] / max(self.stats['successful_queries'], 1)
                ),
                "llm_calls_total": self.stats['llm_calls'],
                "cache_hits": self.stats['cache_hits'],
                "cache_hit_rate": self.stats['cache_hits'] / max(self.stats['total_queries'], 1)
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
            total_queries=self.stats['total_queries'],
            avg_query_time_ms=self.stats['total_processing_time'] / max(self.stats['successful_queries'], 1) * 1000,
            successful_queries=self.stats['successful_queries'],
            failed_queries=self.stats['failed_queries'],
            avg_search_time_ms=0.0,  # Would be calculated from search metrics
            avg_sources_per_query=0.0,  # Would be calculated from search results
            avg_confidence_score=self.stats['avg_confidence'],
            avg_iterations_per_query=self.stats['avg_iterations'],
            reflection_success_rate=1.0,  # Would be calculated from reflection results
            triangulation_usage_rate=0.5,  # Would be calculated from usage stats
            cache_hit_rate=self.stats['cache_hits'] / max(self.stats['total_queries'], 1),
            avg_tokens_per_query=self._estimate_avg_tokens(),
            embedding_requests_per_minute=0.0,  # Would be calculated from embedding stats
            metrics_period_start=self._startup_time,
            metrics_period_end=datetime.now()
        )
    
    async def clear_cache(self):
        """
        Clear all caches
        """
        # Clear answer context cache
        self._last_answer_context.clear()
        
        # Clear query cache
        self._query_cache.clear()
        
        # Clear database manager caches
        self.db_manager.clear_all_caches()
        
        logger.info("All caches cleared")
    
    async def clear_query_cache(self):
        """Clear query-specific caches"""
        try:
            self._query_cache.clear()
            self._last_answer_context.clear()
            logger.info("Query cache cleared")
        except Exception as e:
            logger.error(f"Error clearing query cache: {e}")
    
    async def clear_embedding_cache(self):
        """Clear embedding caches"""
        try:
            self.db_manager.clear_all_caches()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Error clearing embedding cache: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = {
                "query_cache": {
                    "size": len(self._query_cache),
                    "max_size": self._query_cache_max_size
                },
                "embedding_cache": {},
                "total_size_mb": 0,
                "hit_rate": 0,
                "time_saved_ms": 0,
                "requests_served": 0
            }
            
            # Get database manager cache stats
            if hasattr(self.db_manager, 'get_health_status'):
                db_health = await self.db_manager.get_health_status()
                stats["cache_size"] = db_health.get("cache_size", 0)
            
            stats["hit_rate"] = self.stats['cache_hits'] / max(self.stats['total_queries'], 1)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "query_cache": {},
                "embedding_cache": {},
                "total_size_mb": 0,
                "hit_rate": 0,
                "time_saved_ms": 0,
                "requests_served": 0,
                "error": str(e)
            }
    
    async def debug_query(
        self, 
        question: str, 
        debug_level: str = "full",
        include_intermediate_results: bool = True,
        include_timing: bool = True
    ) -> Dict[str, Any]:
        """
        Debug query processing with detailed information
        """
        debug_info = {
            "question": question,
            "debug_level": debug_level,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": [],
            "timing": {},
            "intermediate_results": {},
            "final_result": None
        }
        
        try:
            start_time = datetime.now()
            
            # Step 1: Query Analysis
            if include_timing:
                step_start = datetime.now()
            
            analysis = await self.analyze_query(question)
            debug_info["steps"].append("query_analysis_completed")
            
            if include_timing:
                debug_info["timing"]["query_analysis"] = (datetime.now() - step_start).total_seconds()
            
            if include_intermediate_results:
                debug_info["intermediate_results"]["query_analysis"] = analysis.dict()
            
            # Step 2: Query Planning
            if debug_level in ["detailed", "full"]:
                if include_timing:
                    step_start = datetime.now()
                
                query_plan = await self.planning_agent.create_query_plan(question)
                debug_info["steps"].append("query_planning_completed")
                
                if include_timing:
                    debug_info["timing"]["query_planning"] = (datetime.now() - step_start).total_seconds()
                
                if include_intermediate_results:
                    debug_info["intermediate_results"]["query_planning"] = query_plan.dict()
            
            # Step 3: Search Execution
            if include_timing:
                step_start = datetime.now()
            
            # Execute a simplified search for debugging
            filters = SearchFilters(max_results=5, similarity_threshold=0.7)
            search_results = await self.search_toolkit.hybrid_search(question, filters)
            debug_info["steps"].append("search_execution_completed")
            
            if include_timing:
                debug_info["timing"]["search_execution"] = (datetime.now() - step_start).total_seconds()
            
            if include_intermediate_results:
                debug_info["intermediate_results"]["search_results"] = {
                    "total_found": search_results.total_found,
                    "search_time_ms": search_results.search_time_ms,
                    "search_method": search_results.search_method,
                    "chunks_preview": [
                        {
                            "id": chunk.id,
                            "similarity_score": chunk.similarity_score,
                            "file_name": chunk.file_name,
                            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                        }
                        for chunk in search_results.chunks[:3]
                    ]
                }
            
            # Step 4: Answer Generation (if full debug)
            if debug_level == "full":
                if include_timing:
                    step_start = datetime.now()
                
                answer = await self.llm_service.generate_simple_answer(
                    question, 
                    search_results.chunks[0] if search_results.chunks else None
                )
                debug_info["steps"].append("answer_generation_completed")
                
                if include_timing:
                    debug_info["timing"]["answer_generation"] = (datetime.now() - step_start).total_seconds()
                
                debug_info["final_result"] = {
                    "answer": answer,
                    "sources_used": len(search_results.chunks),
                    "confidence": 0.8  # Placeholder confidence
                }
            
            # Calculate total time
            if include_timing:
                debug_info["timing"]["total_processing"] = (datetime.now() - start_time).total_seconds()
            
            debug_info["status"] = "success"
            debug_info["steps"].append("debug_completed")
            
            return debug_info
            
        except Exception as e:
            debug_info["status"] = "error"
            debug_info["error"] = str(e)
            debug_info["steps"].append(f"error_occurred: {str(e)}")
            logger.error(f"Debug query failed: {e}")
            return debug_info
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        try:
            return {
                "service_name": "Agentic RAG Agent",
                "version": "1.0.0",
                "startup_time": self._startup_time.isoformat(),
                "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
                "configuration": {
                    "llm_model": self.config.llm_model,
                    "embedding_model": self.config.embedding_model,
                    "enable_reranking": self.config.enable_reranking,
                    "max_iterations": self.config.max_iterations,
                    "enable_query_planning": self.config.enable_query_planning,
                    "enable_source_triangulation": self.config.enable_source_triangulation,
                    "enable_self_reflection": self.config.enable_self_reflection
                },
                "features": {
                    "agentic_processing": True,
                    "iterative_search": True,
                    "source_triangulation": self.config.enable_source_triangulation,
                    "self_reflection": self.config.enable_self_reflection,
                    "query_planning": self.config.enable_query_planning,
                    "hybrid_search": True,
                    "reranking": self.config.enable_reranking,
                    "caching": self.config.enable_query_cache,
                    "batch_processing": True,
                    "websocket_support": True,
                    "metrics_collection": True
                },
                "limits": {
                    "max_results": self.config.default_max_results,
                    "max_iterations": self.config.max_iterations,
                    "max_concurrent_searches": self.config.max_concurrent_searches,
                    "search_timeout_seconds": self.config.search_timeout_seconds,
                    "max_context_length": self.config.max_context_length
                }
            }
        except Exception as e:
            logger.error(f"Error getting service info: {e}")
            return {"error": str(e)}
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Test database connection
            try:
                db_healthy = await self.db_manager.test_connection()
                if not db_healthy:
                    validation_results["errors"].append("Database connection failed")
                    validation_results["valid"] = False
            except Exception as e:
                validation_results["errors"].append(f"Database test failed: {str(e)}")
                validation_results["valid"] = False
            
            # Test embedding service
            try:
                embedding_healthy = await self.db_manager.test_embedding_service()
                if not embedding_healthy:
                    validation_results["errors"].append("Embedding service unavailable")
                    validation_results["valid"] = False
            except Exception as e:
                validation_results["errors"].append(f"Embedding service test failed: {str(e)}")
                validation_results["valid"] = False
            
            # Test LLM service
            try:
                llm_healthy = await self.llm_service.test_llm_connection()
                if not llm_healthy:
                    validation_results["errors"].append("LLM service unavailable")
                    validation_results["valid"] = False
            except Exception as e:
                validation_results["errors"].append(f"LLM service test failed: {str(e)}")
                validation_results["valid"] = False
            
            # Configuration warnings
            if self.config.default_similarity_threshold < 0.5:
                validation_results["warnings"].append("Very low similarity threshold may return irrelevant results")
            
            if self.config.max_iterations > 5:
                validation_results["warnings"].append("High max_iterations may cause slow responses")
            
            if not self.config.enable_query_cache:
                validation_results["recommendations"].append("Enable query caching for better performance")
            
            if not self.config.enable_reranking:
                validation_results["recommendations"].append("Enable reranking for improved result quality")
            
            # Performance recommendations
            if self.config.max_concurrent_searches < 2:
                validation_results["recommendations"].append("Consider increasing max_concurrent_searches for better throughput")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            validation_results["valid"] = False
        
        return validation_results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get basic health
            health = await self.get_health_status()
            
            # Get performance metrics
            metrics = self.get_performance_metrics()
            
            # Get cache stats
            cache_stats = await self.get_cache_stats()
            
            # Get service info
            service_info = self.get_service_info()
            
            # Get configuration validation
            config_validation = await self.validate_configuration()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy" if health.get("database_connected") and health.get("embedding_service") == "healthy" else "unhealthy",
                "health": health,
                "performance": metrics.dict(),
                "cache": cache_stats,
                "service": service_info,
                "configuration": config_validation
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    # Background task methods
    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old answer contexts
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=24)
                
                keys_to_remove = []
                for key, context in self._last_answer_context.items():
                    if context.get('timestamp', current_time) < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self._last_answer_context[key]
                
                if keys_to_remove:
                    logger.info(f"Cleaned up {len(keys_to_remove)} old answer contexts")
                
                # Clean up database caches
                if hasattr(self.db_manager, 'cleanup_old_cache'):
                    await self.db_manager.cleanup_old_cache()
                
            except Exception as e:
                logger.error(f"Error in periodic cache cleanup: {e}")
    
    async def _periodic_metrics_collection(self):
        """Periodic metrics collection and export"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Collect and export metrics
                metrics_summary = self.metrics.get_metrics_summary(hours=1)
                
                # Log key metrics
                logger.info(
                    "Metrics summary",
                    extra={
                        'total_queries': self.stats['total_queries'],
                        'success_rate': self.stats['successful_queries'] / max(self.stats['total_queries'], 1),
                        'avg_processing_time': self.stats['total_processing_time'] / max(self.stats['successful_queries'], 1),
                        'cache_hit_rate': self.stats.get('cache_hits', 0) / max(self.stats['total_queries'], 1)
                    }
                )
                
                # Export metrics to file if configured
                if self.config.debug_mode:
                    metrics_file = f"./logs/metrics_{datetime.now().strftime('%Y%m%d_%H')}.json"
                    try:
                        os.makedirs("./logs", exist_ok=True)
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics_summary, f, indent=2, default=str)
                    except Exception as e:
                        logger.debug(f"Could not write metrics file: {e}")
                
            except Exception as e:
                logger.error(f"Error in periodic metrics collection: {e}")
    
    async def _periodic_health_monitoring(self):
        """Periodic health monitoring and alerting"""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Check service health
                health_status = await self.get_health_status()
                
                # Check for issues
                issues = []
                
                if not health_status.get("database_connected", False):
                    issues.append("Database connection lost")
                
                if health_status.get("embedding_service") != "healthy":
                    issues.append("Embedding service unhealthy")
                
                if health_status.get("llm_service") != "healthy":
                    issues.append("LLM service unhealthy")
                
                success_rate = health_status.get("success_rate", 0)
                if success_rate < 90:  # Less than 90% success rate
                    issues.append(f"Low success rate: {success_rate:.1f}%")
                
                avg_response_time = health_status.get("avg_response_time", 0)
                if avg_response_time > 30:  # More than 30 seconds average
                    issues.append(f"High response time: {avg_response_time:.1f}s")
                
                if issues:
                    logger.warning(
                        "Health monitoring detected issues",
                        extra={
                            'issues': issues,
                            'health_status': health_status
                        }
                    )
                else:
                    logger.debug("Health monitoring: All systems operational")
                
            except Exception as e:
                logger.error(f"Error in periodic health monitoring: {e}")
    
    # Private helper methods
    async def _generate_answer(
        self, 
        question: str, 
        sources: List[DocumentChunk], 
        query_plan: QueryPlan
    ) -> str:
        """
        Generate comprehensive answer using LLM with context from sources
        """
        return await self.llm_service.generate_answer(question, sources, query_plan)
    
    async def _generate_follow_ups(
        self, 
        question: str, 
        answer: str,
        sources: List[DocumentChunk], 
        reflection: ReflectionResult
    ) -> List[str]:
        """
        Generate intelligent follow-up questions using LLM + reflection
        """
        follow_ups = []
        
        # Start with reflection-based follow-ups
        follow_ups.extend(reflection.suggested_follow_ups)
        
        # Add LLM-generated follow-ups
        llm_follow_ups = await self.llm_service.generate_follow_ups(question, answer, sources)
        follow_ups.extend(llm_follow_ups)
        
        # Add source-based follow-ups
        if sources:
            unique_topics = set()
            for source in sources[:3]:
                if source.keywords:
                    unique_topics.update(source.keywords[:2])
            
            follow_ups.extend([
                f"Tell me more about {topic}" for topic in list(unique_topics)[:2]
            ])
        
        # Add generic helpful follow-ups if we don't have enough
        if len(follow_ups) < 3:
            follow_ups.extend([
                "What are the practical implications of this information?",
                "Can you provide more specific examples?",
                "How does this relate to current best practices?"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_follow_ups = []
        for follow_up in follow_ups:
            follow_up_lower = follow_up.lower()
            if follow_up_lower not in seen and len(follow_up) > 10:
                unique_follow_ups.append(follow_up)
                seen.add(follow_up_lower)
        
        return unique_follow_ups[:5]  # Limit to 5 suggestions
    
    async def _perform_reflection_based_search(
        self,
        question: str,
        missing_info: List[str],
        existing_sources: List[DocumentChunk]
    ) -> SearchResults:
        """
        Perform additional search based on reflection feedback
        """
        try:
            # Create search query from missing information
            if missing_info:
                additional_query = f"{question} {' '.join(missing_info[:2])}"
            else:
                additional_query = question
            
            # Search with relaxed threshold to find more sources
            filters = SearchFilters(
                max_results=5,
                similarity_threshold=max(self.config.default_similarity_threshold - 0.1, 0.5)
            )
            
            return await self.search_toolkit.hybrid_search(additional_query, filters)
            
        except Exception as e:
            logger.error(f"Error in reflection-based search: {e}")
            return SearchResults(
                query=question,
                chunks=[],
                total_found=0,
                search_time_ms=0,
                search_method="reflection_based"
            )
    
    def _extract_cross_validated_facts(self, sources: List[DocumentChunk]) -> List[str]:
        """
        Extract facts that appear in multiple sources
        """
        if len(sources) < 2:
            return []
        
        facts = []
        
        # Simple approach: find sentences that appear in multiple sources
        all_sentences = []
        for source in sources:
            sentences = [s.strip() for s in source.content.split('.') if len(s.strip()) > 20]
            all_sentences.extend([(s, source.file_name) for s in sentences])
        
        # Group similar sentences (basic string matching)
        sentence_groups = {}
        for sentence, source in all_sentences:
            key = sentence.lower()[:50]  # Use first 50 chars as key
            if key not in sentence_groups:
                sentence_groups[key] = []
            sentence_groups[key].append((sentence, source))
        
        # Find facts mentioned in multiple sources
        for key, group in sentence_groups.items():
            sources_mentioned = set(source for _, source in group)
            if len(sources_mentioned) >= 2:  # Fact appears in 2+ sources
                facts.append(group[0][0])  # Take the first occurrence
        
        return facts[:5]  # Limit to 5 cross-validated facts
    
    def _identify_potential_biases(self, sources: List[DocumentChunk]) -> List[str]:
        """
        Identify potential biases in the source collection
        """
        biases = []
        
        # Check for source diversity
        file_types = set(source.file_type for source in sources)
        if len(file_types) == 1:
            biases.append(f"All sources are {list(file_types)[0]} files - consider diverse source types")
        
        # Check for domain bias
        source_names = [source.file_name.lower() for source in sources]
        if all('academic' in name or 'research' in name for name in source_names):
            biases.append("Sources appear to be primarily academic - consider practical perspectives")
        elif all('blog' in name or 'opinion' in name for name in source_names):
            biases.append("Sources appear to be opinion-based - consider authoritative sources")
        
        return biases[:3]  # Limit to 3 bias warnings
    
    def _estimate_tokens_used(self, question: str, answer: str, sources: List[DocumentChunk]) -> int:
        """
        Estimate tokens used in LLM calls
        """
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = len(question) + len(answer)
        
        # Add source content used in prompts (estimated)
        source_chars = sum(len(source.content[:800]) for source in sources[:8])
        total_chars += source_chars
        
        # Add prompt overhead (estimated)
        prompt_overhead = 500  # Estimated prompt template characters
        total_chars += prompt_overhead * self.stats['llm_calls']
        
        return total_chars // 4  # Convert to approximate tokens
    
    def _estimate_avg_tokens(self) -> int:
        """
        Estimate average tokens per query
        """
        if self.stats['successful_queries'] == 0:
            return 0
        
        # Rough estimation based on typical usage
        avg_tokens = (self.stats['llm_calls'] * 1000) // max(self.stats['successful_queries'], 1)
        return avg_tokens
    
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