"""
Configuration management for the LlamaIndex Ingestion Service
"""

import os
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline"""
    
    # Chunking parameters
    chunk_size: int = 1024
    chunk_overlap: int = 200
    enable_semantic_chunking: bool = True
    enable_hierarchical_chunking: bool = True
    
    # Processing options
    extract_metadata: bool = True
    enable_ocr: bool = True
    enable_speech_to_text: bool = True
    max_file_size_mb: int = 100
    
    # Database configuration
    supabase_url: str = ""
    supabase_key: str = ""
    table_name: str = "rag_documents"
    
    # Model configuration
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4-turbo"
    
    # Processing limits
    max_concurrent_files: int = 5
    batch_size: int = 100
    
    # Cache configuration
    cache_dir: str = "./ingestion_cache"
    enable_cache: bool = True
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance tuning
    memory_limit_gb: float = 8.0
    cpu_limit: int = 4
    
    # File type specific settings
    ocr_confidence_threshold: float = 0.5
    audio_chunk_duration: int = 30  # seconds
    code_chunk_lines: int = 40
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL is required")
        if not self.supabase_key:
            raise ValueError("SUPABASE_KEY is required")
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate chunk parameters
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Validate file size limit
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")


def get_config() -> IngestionConfig:
    """Get configuration from environment variables"""
    return IngestionConfig(
        # Chunking parameters
        chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        enable_semantic_chunking=os.getenv("ENABLE_SEMANTIC_CHUNKING", "true").lower() == "true",
        enable_hierarchical_chunking=os.getenv("ENABLE_HIERARCHICAL_CHUNKING", "true").lower() == "true",
        
        # Processing options
        extract_metadata=os.getenv("EXTRACT_METADATA", "true").lower() == "true",
        enable_ocr=os.getenv("ENABLE_OCR", "true").lower() == "true",
        enable_speech_to_text=os.getenv("ENABLE_SPEECH_TO_TEXT", "true").lower() == "true",
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
        
        # Database configuration
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_SERVICE_KEY", ""),
        table_name=os.getenv("TABLE_NAME", "rag_documents"),
        
        # Model configuration
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
        
        # Processing limits
        max_concurrent_files=int(os.getenv("MAX_CONCURRENT_FILES", "5")),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        
        # Cache configuration
        cache_dir=os.getenv("CACHE_DIR", "./ingestion_cache"),
        enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
        
        # Logging configuration
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
        
        # Performance tuning
        memory_limit_gb=float(os.getenv("MEMORY_LIMIT_GB", "8.0")),
        cpu_limit=int(os.getenv("CPU_LIMIT", "4")),
        
        # File type specific settings
        ocr_confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5")),
        audio_chunk_duration=int(os.getenv("AUDIO_CHUNK_DURATION", "30")),
        code_chunk_lines=int(os.getenv("CODE_CHUNK_LINES", "40"))
    )