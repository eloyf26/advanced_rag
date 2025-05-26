"""
Request models for the PydanticAI Agentic RAG Agent
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for single question queries"""
    
    question: str = Field(..., description="The user's question")
    
    # Agentic features
    enable_iteration: bool = Field(True, description="Enable iterative search")
    enable_reflection: bool = Field(True, description="Enable self-reflection and quality assessment")
    enable_triangulation: bool = Field(True, description="Enable source triangulation")
    
    # Search configuration
    search_method: Optional[str] = Field("hybrid", description="Search method: 'hybrid', 'semantic', or 'keyword'")
    max_results: int = Field(10, description="Maximum number of results to return")
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold")
    
    # Filters
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    date_after: Optional[datetime] = Field(None, description="Only include documents after this date")
    
    # Advanced options
    max_iterations: Optional[int] = Field(None, description="Override default max iterations")
    include_reasoning: bool = Field(True, description="Include reasoning chain in response")
    include_sources: bool = Field(True, description="Include source documents in response")


class BatchQueryRequest(BaseModel):
    """Request model for batch question processing"""
    
    questions: List[str] = Field(..., description="List of questions to process")
    
    # Processing options
    enable_agentic: bool = Field(True, description="Use full agentic processing")
    enable_iteration: bool = Field(True, description="Enable iterative search for each question")
    enable_reflection: bool = Field(True, description="Enable reflection for each question")
    enable_triangulation: bool = Field(False, description="Enable triangulation (slower for batch)")
    
    # Search configuration
    search_method: Optional[str] = Field("hybrid", description="Search method for all questions")
    max_results: int = Field(10, description="Maximum results per question")
    
    # Batch processing
    max_concurrency: int = Field(3, description="Maximum concurrent questions to process")
    timeout_per_question: int = Field(30, description="Timeout per question in seconds")


class SearchFilters(BaseModel):
    """Filters for search queries"""
    
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    date_after: Optional[datetime] = Field(None, description="Only include documents after this date")
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold")
    max_results: int = Field(10, description="Maximum number of results")
    
    # Advanced filters
    keywords_required: Optional[List[str]] = Field(None, description="Required keywords in results")
    keywords_excluded: Optional[List[str]] = Field(None, description="Excluded keywords from results")
    min_word_count: Optional[int] = Field(None, description="Minimum word count for chunks")
    max_word_count: Optional[int] = Field(None, description="Maximum word count for chunks")


class AnalyzeQueryRequest(BaseModel):
    """Request model for query analysis"""
    
    question: str = Field(..., description="The question to analyze")
    include_suggestions: bool = Field(True, description="Include optimization suggestions")
    include_complexity: bool = Field(True, description="Include complexity analysis")


class CacheRequest(BaseModel):
    """Request model for cache operations"""
    
    operation: str = Field(..., description="Cache operation: 'clear', 'stats', 'warmup'")
    cache_type: Optional[str] = Field(None, description="Type of cache: 'query', 'embedding', 'all'")
    warmup_queries: Optional[List[str]] = Field(None, description="Queries to warmup cache with")


class ConfigUpdateRequest(BaseModel):
    """Request model for runtime configuration updates"""
    
    # Search parameters
    similarity_threshold: Optional[float] = Field(None, ge=0, le=1)
    max_results: Optional[int] = Field(None, ge=1, le=100)
    vector_weight: Optional[float] = Field(None, ge=0, le=1)
    bm25_weight: Optional[float] = Field(None, ge=0, le=1)
    
    # Agentic parameters
    max_iterations: Optional[int] = Field(None, ge=1, le=10)
    enable_reflection: Optional[bool] = Field(None)
    enable_triangulation: Optional[bool] = Field(None)
    
    # Performance parameters
    max_concurrent_searches: Optional[int] = Field(None, ge=1, le=10)
    search_timeout_seconds: Optional[int] = Field(None, ge=5, le=120)


class DebugRequest(BaseModel):
    """Request model for debugging operations"""
    
    question: str = Field(..., description="Question for debugging")
    debug_level: str = Field("full", description="Debug level: 'basic', 'detailed', 'full'")
    include_intermediate_results: bool = Field(True, description="Include intermediate search results")
    include_timing: bool = Field(True, description="Include timing information")
    trace_execution: bool = Field(False, description="Enable execution tracing")


class MultiModalQueryRequest(BaseModel):
    """Request model for multi-modal queries"""
    
    text_query: str = Field(..., description="Text component of the query")
    
    # File inputs
    image_path: Optional[str] = Field(None, description="Path to image file")
    audio_path: Optional[str] = Field(None, description="Path to audio file")
    document_path: Optional[str] = Field(None, description="Path to document file")
    
    # Processing options
    enable_ocr: bool = Field(True, description="Enable OCR for images")
    enable_transcription: bool = Field(True, description="Enable audio transcription")
    
    # Standard query options
    enable_iteration: bool = Field(True, description="Enable iterative search")
    enable_reflection: bool = Field(True, description="Enable reflection")
    max_results: int = Field(10, description="Maximum results")


class FeedbackRequest(BaseModel):
    """Request model for user feedback on responses"""
    
    query_id: str = Field(..., description="ID of the original query")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    
    # Specific feedback categories
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5, description="Accuracy rating")
    completeness_rating: Optional[int] = Field(None, ge=1, le=5, description="Completeness rating")
    relevance_rating: Optional[int] = Field(None, ge=1, le=5, description="Relevance rating")
    
    # Issues
    issues: Optional[List[str]] = Field(None, description="List of issues encountered")
    suggestions: Optional[str] = Field(None, description="User suggestions for improvement")