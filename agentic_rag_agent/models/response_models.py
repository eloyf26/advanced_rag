"""
Response models for the PydanticAI Agentic RAG Agent
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a document chunk with metadata"""
    
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Text content of the chunk")
    
    # Scores
    similarity_score: float = Field(0.0, description="Vector similarity score")
    bm25_score: float = Field(0.0, description="BM25 keyword score")
    combined_score: float = Field(0.0, description="Combined hybrid score")
    rerank_score: float = Field(0.0, description="Cross-encoder rerank score")
    
    # Metadata
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type/extension")
    chunk_index: int = Field(..., description="Index of chunk within document")
    
    # Enhanced metadata
    title: Optional[str] = Field(None, description="Extracted title")
    summary: Optional[str] = Field(None, description="Generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities")
    
    # Context
    document_id: Optional[str] = Field(None, description="Parent document ID")
    chunk_context: Optional[str] = Field(None, description="Surrounding context")


class SearchResults(BaseModel):
    """Search results with metadata"""
    
    query: str = Field(..., description="Original search query")
    chunks: List[DocumentChunk] = Field(..., description="Retrieved document chunks")
    total_found: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    search_method: str = Field(..., description="Search method used")
    
    # Search metadata
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Filters applied to search")
    index_used: Optional[str] = Field(None, description="Database index used")
    cache_hit: bool = Field(False, description="Whether result was cached")


class QueryPlan(BaseModel):
    """Represents a query execution plan"""
    
    original_query: str = Field(..., description="Original user query")
    sub_queries: List[str] = Field(..., description="Generated sub-queries")
    search_strategy: str = Field(..., description="Selected search strategy")
    reasoning: str = Field(..., description="Planning rationale")
    expected_sources: List[str] = Field(default_factory=list, description="Expected source types")
    complexity_score: float = Field(..., description="Query complexity score (0-1)")
    
    # Execution metadata
    estimated_time: Optional[float] = Field(None, description="Estimated processing time")
    recommended_iterations: Optional[int] = Field(None, description="Recommended search iterations")


class ReflectionResult(BaseModel):
    """Results from self-reflection on answer quality"""
    
    quality_score: float = Field(..., description="Overall quality score (0-1)")
    completeness_score: float = Field(..., description="Completeness score (0-1)")
    accuracy_assessment: str = Field(..., description="Accuracy assessment")
    missing_information: List[str] = Field(default_factory=list, description="Identified missing information")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    needs_more_search: bool = Field(False, description="Whether additional search is recommended")
    
    # Detailed assessment
    reasoning_quality: Optional[float] = Field(None, description="Quality of reasoning (0-1)")
    source_diversity: Optional[float] = Field(None, description="Diversity of sources (0-1)")
    factual_consistency: Optional[float] = Field(None, description="Factual consistency score (0-1)")


class IterativeSearchState(BaseModel):
    """State management for iterative search"""
    
    iteration: int = Field(..., description="Current iteration number")
    total_chunks_found: int = Field(..., description="Total chunks found across iterations")
    unique_sources: int = Field(..., description="Number of unique source files")
    coverage_gaps: List[str] = Field(default_factory=list, description="Identified coverage gaps")
    should_continue: bool = Field(..., description="Whether to continue searching")
    reasoning: str = Field(..., description="Reasoning for continuation decision")
    
    # Iteration metadata
    search_focus: Optional[str] = Field(None, description="Focus area for current iteration")
    strategy_adjustments: List[str] = Field(default_factory=list, description="Strategy adjustments made")


class AgenticRAGResponse(BaseModel):
    """Enhanced response with agentic features"""
    
    # Core response
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[DocumentChunk] = Field(..., description="Source documents used")
    
    # Agentic components
    query_plan: QueryPlan = Field(..., description="Query execution plan")
    reflection: ReflectionResult = Field(..., description="Self-reflection results")
    search_iterations: List[SearchResults] = Field(..., description="Search iteration history")
    reasoning_chain: List[str] = Field(..., description="Step-by-step reasoning")
    follow_up_suggestions: List[str] = Field(..., description="Suggested follow-up questions")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Total processing time")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    iterations_completed: int = Field(..., description="Number of search iterations")
    
    # Quality indicators
    source_triangulation_performed: bool = Field(False, description="Whether triangulation was performed")
    cross_validated_facts: List[str] = Field(default_factory=list, description="Cross-validated facts")
    potential_biases: List[str] = Field(default_factory=list, description="Identified potential biases")


class RAGResponse(BaseModel):
    """Standard RAG response (non-agentic)"""
    
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[DocumentChunk] = Field(..., description="Source documents")
    search_results: SearchResults = Field(..., description="Search results")
    reasoning: Optional[str] = Field(None, description="Basic reasoning")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Processing time")
    search_method: str = Field(..., description="Search method used")


class BatchQueryResponse(BaseModel):
    """Response for batch query requests"""
    
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    total_questions: int = Field(..., description="Total number of questions")
    message: str = Field(..., description="Status message")
    
    # Progress tracking
    completed: Optional[int] = Field(None, description="Number of completed questions")
    failed: Optional[int] = Field(None, description="Number of failed questions")
    
    # Results (populated when complete)
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Question-answer pairs")
    errors: Optional[List[Dict[str, str]]] = Field(None, description="Error details")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Overall service status")
    service: str = Field(..., description="Service name")
    database_connected: bool = Field(..., description="Database connection status")
    embedding_service: str = Field(..., description="Embedding service status")
    llm_service: str = Field(..., description="LLM service status")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")
    
    # Performance indicators
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")


class QueryAnalysisResponse(BaseModel):
    """Response for query analysis requests"""
    
    query: str = Field(..., description="Original query")
    complexity_score: float = Field(..., description="Query complexity (0-1)")
    estimated_processing_time: float = Field(..., description="Estimated processing time in seconds")
    recommended_strategy: str = Field(..., description="Recommended search strategy")
    
    # Analysis details
    query_type: str = Field(..., description="Classified query type")
    key_concepts: List[str] = Field(..., description="Identified key concepts")
    predicted_sources: List[str] = Field(..., description="Predicted relevant source types")
    suggested_filters: Dict[str, Any] = Field(default_factory=dict, description="Suggested search filters")
    
    # Optimization suggestions
    optimization_suggestions: List[str] = Field(default_factory=list, description="Query optimization suggestions")
    potential_challenges: List[str] = Field(default_factory=list, description="Potential processing challenges")


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    
    # Query metrics
    total_queries: int = Field(..., description="Total queries processed")
    avg_query_time_ms: float = Field(..., description="Average query processing time")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    
    # Search metrics
    avg_search_time_ms: float = Field(..., description="Average search time")
    avg_sources_per_query: float = Field(..., description="Average sources retrieved per query")
    avg_confidence_score: float = Field(..., description="Average confidence score")
    
    # Agentic metrics
    avg_iterations_per_query: float = Field(..., description="Average search iterations")
    reflection_success_rate: float = Field(..., description="Reflection success rate")
    triangulation_usage_rate: float = Field(..., description="Triangulation usage rate")
    
    # Resource metrics
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    avg_tokens_per_query: Optional[int] = Field(None, description="Average tokens per query")
    embedding_requests_per_minute: float = Field(..., description="Embedding requests per minute")
    
    # Time-based metrics
    metrics_period_start: datetime = Field(..., description="Start of metrics period")
    metrics_period_end: datetime = Field(..., description="End of metrics period")


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    # Recovery suggestions
    suggested_actions: Optional[List[str]] = Field(None, description="Suggested recovery actions")
    retry_recommended: bool = Field(False, description="Whether retry is recommended")


class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    
    query_cache: Dict[str, Any] = Field(..., description="Query cache statistics")
    embedding_cache: Dict[str, Any] = Field(..., description="Embedding cache statistics")
    
    # Overall stats
    total_cache_size_mb: float = Field(..., description="Total cache size in MB")
    cache_hit_rate: float = Field(..., description="Overall cache hit rate")
    cache_miss_rate: float = Field(..., description="Overall cache miss rate")
    
    # Performance impact
    time_saved_ms: float = Field(..., description="Total time saved by caching")
    requests_served_from_cache: int = Field(..., description="Requests served from cache")


class ConfigResponse(BaseModel):
    """Configuration response"""
    
    current_config: Dict[str, Any] = Field(..., description="Current configuration")
    modifiable_settings: List[str] = Field(..., description="Settings that can be modified")
    
    # Validation results
    config_valid: bool = Field(..., description="Whether configuration is valid")
    validation_errors: List[str] = Field(default_factory=list, description="Configuration validation errors")
    
    # Recommendations
    optimization_recommendations: List[str] = Field(default_factory=list, description="Configuration optimization recommendations")