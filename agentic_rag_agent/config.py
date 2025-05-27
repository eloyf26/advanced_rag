import os
from dataclasses import dataclass
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for the agentic RAG agent with validation"""
    
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
    
    # NEW: Security settings
    enable_authentication: bool = False
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    
    # NEW: File processing settings
    max_file_size_mb: int = 100
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        errors = []
        
        # Required fields
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        elif not self.supabase_url.startswith(('http://', 'https://')):
            errors.append("SUPABASE_URL must be a valid URL")
            
        if not self.supabase_key:
            errors.append("SUPABASE_KEY is required")
        elif len(self.supabase_key) < 20:
            errors.append("SUPABASE_KEY appears to be invalid (too short)")
        
        # Validate weights
        if abs(self.vector_weight + self.bm25_weight - 1.0) > 0.01:
            errors.append("vector_weight + bm25_weight must equal 1.0")
        
        # Validate thresholds
        if not 0 <= self.default_similarity_threshold <= 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        if not 0 <= self.min_confidence_threshold <= 1:
            errors.append("min_confidence_threshold must be between 0 and 1")
        
        # Validate ranges
        if self.max_iterations < 1 or self.max_iterations > 10:
            errors.append("max_iterations must be between 1 and 10")
        
        if self.min_sources_per_iteration < 1:
            errors.append("min_sources_per_iteration must be at least 1")
        
        if self.max_concurrent_searches < 1 or self.max_concurrent_searches > 20:
            errors.append("max_concurrent_searches must be between 1 and 20")
        
        if self.embedding_batch_size < 1 or self.embedding_batch_size > 1000:
            errors.append("embedding_batch_size must be between 1 and 1000")
        
        # Log warnings for suboptimal settings
        warnings = []
        
        if self.default_similarity_threshold < 0.5:
            warnings.append("Low similarity threshold may return irrelevant results")
        
        if self.max_iterations > 5:
            warnings.append("High max_iterations may cause slow responses")
        
        if self.embedding_cache_size < 1000:
            warnings.append("Small embedding cache may reduce performance")
        
        # Report errors and warnings
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)
        
        if warnings:
            warning_msg = "Configuration warnings:\n" + "\n".join(f"- {w}" for w in warnings)
            logger.warning(warning_msg)


def get_config() -> RAGConfig:
    """Get configuration from environment variables with validation"""
    try:
        config = RAGConfig(
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
            enable_graceful_degradation=os.getenv("ENABLE_GRACEFUL_DEGRADATION", "true").lower() == "true",
            
            # Security
            enable_authentication=os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60")),
            
            # File processing
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        logger.info("Configuration loaded successfully")
        return config
        
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise ValueError(f"Configuration loading failed: {e}")


def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set these variables before starting the service."
        )