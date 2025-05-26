"""
Configuration management for the PydanticAI Agentic RAG Agent
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RAGConfig:
    """Configuration for the agentic RAG agent"""
    
    # Database configuration
    supabase_url: str
    supabase_key: str
    table_name: str = "rag_documents"
    
    # Model configuration
    llm_model: str = "gpt-4-turbo"
    embedding_model: str = "text-embedding-3-large"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Search configuration
    default_similarity_threshold: float = 0.7
    default_max_results: int = 10
    enable_reranking: bool = True
    rerank_top_k: int = 20
    
    # Hybrid search weights
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # Response configuration
    max_context_length: int = 8000
    min_confidence_threshold: float = 0.3
    
    # Agentic features
    max_iterations: int = 3
    min_sources_per_iteration: int = 3
    enable_query_planning: bool = True
    enable_source_triangulation: bool = True
    enable_self_reflection: bool = True
    
    # Performance settings
    max_concurrent_searches: int = 3
    search_timeout_seconds: int = 30
    embedding_batch_size: int = 100
    
    # Caching
    enable_query_cache: bool = True
    cache_ttl_minutes: int = 60
    enable_embedding_cache: bool = True
    embedding_cache_size: int = 10000
    
    # Logging and debugging
    log_level: str = "INFO"
    debug_mode: bool = False
    log_reasoning_steps: bool = True
    save_intermediate_results: bool = False
    
    # Error handling
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    enable_graceful_degradation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL is required")
        if not self.supabase_key:
            raise ValueError("SUPABASE_KEY is required")
        
        # Validate weights
        if abs(self.vector_weight + self.bm25_weight - 1.0) > 0.01:
            raise ValueError("vector_weight + bm25_weight must equal 1.0")
        
        # Validate thresholds
        if not 0 <= self.default_similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")


def get_config() -> RAGConfig:
    """Get configuration from environment variables"""
    return RAGConfig(
        # Database configuration
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_SERVICE_KEY", ""),
        table_name=os.getenv("TABLE_NAME", "rag_documents"),
        
        # Model configuration
        llm_model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        
        # Search configuration
        default_similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
        default_max_results=int(os.getenv("MAX_RESULTS", "10")),
        enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "20")),
        
        # Hybrid search weights
        vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7")),
        bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
        
        # Response configuration
        max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "8000")),
        min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.3")),
        
        # Agentic features
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        min_sources_per_iteration=int(os.getenv("MIN_SOURCES_PER_ITERATION", "3")),
        enable_query_planning=os.getenv("ENABLE_QUERY_PLANNING", "true").lower() == "true",
        enable_source_triangulation=os.getenv("ENABLE_SOURCE_TRIANGULATION", "true").lower() == "true",
        enable_self_reflection=os.getenv("ENABLE_SELF_REFLECTION", "true").lower() == "true",
        
        # Performance settings
        max_concurrent_searches=int(os.getenv("MAX_CONCURRENT_SEARCHES", "3")),
        search_timeout_seconds=int(os.getenv("SEARCH_TIMEOUT_SECONDS", "30")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
        
        # Caching
        enable_query_cache=os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true",
        cache_ttl_minutes=int(os.getenv("CACHE_TTL_MINUTES", "60")),
        enable_embedding_cache=os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true",
        embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "10000")),
        
        # Logging and debugging
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_reasoning_steps=os.getenv("LOG_REASONING_STEPS", "true").lower() == "true",
        save_intermediate_results=os.getenv("SAVE_INTERMEDIATE_RESULTS", "false").lower() == "true",
        
        # Error handling
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        enable_graceful_degradation=os.getenv("ENABLE_GRACEFUL_DEGRADATION", "true").lower() == "true"
    )