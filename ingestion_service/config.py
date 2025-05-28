"""
Enhanced Configuration management for the LlamaIndex Ingestion Service
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline with comprehensive validation"""
    
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
    openai_api_key: str = ""
    
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
    
    # Additional settings
    supported_extensions: List[str] = field(default_factory=lambda: [
        'pdf', 'docx', 'doc', 'txt', 'md', 'rtf',  # Documents
        'csv', 'xlsx', 'xls',  # Spreadsheets
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff',  # Images
        'mp3', 'wav', 'm4a', 'flac',  # Audio
        'py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'sql',  # Code
        'json', 'xml', 'yaml', 'yml',  # Structured data
        'html', 'htm', 'css',  # Web
        'zip', 'tar', 'gz'  # Archives
    ])
    
    # Batch API configuration
    use_batch_api: bool = True  
    batch_api_threshold: int = 100  # Use batch API for >= 100 chunks
    batch_api_wait_timeout: int = 300  # Max seconds to wait for small batches
    batch_check_interval: int = 300  # Check pending batches every 5 minutes
    prefer_cost_savings: bool = True  # Prefer batch API when possible
    max_regular_api_batch: int = 20  # Max embeddings per regular API call
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_required_fields()
        self._validate_ranges()
        self._validate_paths()
        self._setup_logging()
        
        logger.info("Configuration validated successfully")
    
    def _validate_required_fields(self):
        """Validate required configuration fields"""
        errors = []
        
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        
        if not self.supabase_key:
            errors.append("SUPABASE_SERVICE_KEY is required")
        
        # OpenAI API key validation
        if not self.openai_api_key:
            # Try to get from environment
            self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
            if not self.openai_api_key:
                errors.append("OPENAI_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _validate_ranges(self):
        """Validate numeric ranges and constraints"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_concurrent_files <= 0:
            raise ValueError("max_concurrent_files must be positive")
        
        if not (0.0 <= self.ocr_confidence_threshold <= 1.0):
            raise ValueError("ocr_confidence_threshold must be between 0.0 and 1.0")
        
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        
        if self.cpu_limit <= 0:
            raise ValueError("cpu_limit must be positive")
    
    def _validate_paths(self):
        """Validate and create necessary paths"""
        # Create cache directory
        cache_path = Path(self.cache_dir)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory created/verified: {cache_path}")
        except Exception as e:
            raise ValueError(f"Failed to create cache directory {self.cache_dir}: {e}")
        
        # Validate log file path if specified
        if self.log_file:
            log_path = Path(self.log_file)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                # Test write access
                with open(log_path, 'a') as f:
                    pass
                logger.info(f"Log file path validated: {log_path}")
            except Exception as e:
                raise ValueError(f"Invalid log file path {self.log_file}: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_levels:
            logger.warning(f"Invalid log level {self.log_level}, defaulting to INFO")
            self.log_level = "INFO"
        else:
            self.log_level = self.log_level.upper()
    
    def get_summary(self) -> dict:
        """Get a summary of the configuration for logging/debugging"""
        return {
            'chunking': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'semantic_chunking': self.enable_semantic_chunking,
                'hierarchical_chunking': self.enable_hierarchical_chunking
            },
            'processing': {
                'extract_metadata': self.extract_metadata,
                'enable_ocr': self.enable_ocr,
                'enable_speech_to_text': self.enable_speech_to_text,
                'max_file_size_mb': self.max_file_size_mb,
                'max_concurrent_files': self.max_concurrent_files,
                'batch_size': self.batch_size
            },
            'models': {
                'embedding_model': self.embedding_model,
                'llm_model': self.llm_model
            },
            'database': {
                'table_name': self.table_name,
                'supabase_url_set': bool(self.supabase_url)
            },
            'cache': {
                'enable_cache': self.enable_cache,
                'cache_dir': self.cache_dir
            },
            'supported_extensions_count': len(self.supported_extensions)
        }
    
    def update_from_dict(self, updates: dict):
        """Update configuration from dictionary (for runtime updates)"""
        for key, value in updates.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after updates
        self._validate_ranges()


def get_config() -> IngestionConfig:
    """Get configuration from environment variables with comprehensive defaults"""
    
    # Helper function to convert string boolean
    def str_to_bool(value: str, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        return value.lower() in ("true", "1", "yes", "on") if value else default
    
    # Helper function to convert string list
    def str_to_list(value: str, default: List[str] = None) -> List[str]:
        if not value:
            return default or []
        return [item.strip() for item in value.split(',') if item.strip()]
    
    try:
        config = IngestionConfig(
            # Chunking parameters
            chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            enable_semantic_chunking=str_to_bool(os.getenv("ENABLE_SEMANTIC_CHUNKING", "true")),
            enable_hierarchical_chunking=str_to_bool(os.getenv("ENABLE_HIERARCHICAL_CHUNKING", "true")),
            
            # Processing options
            extract_metadata=str_to_bool(os.getenv("EXTRACT_METADATA", "true")),
            enable_ocr=str_to_bool(os.getenv("ENABLE_OCR", "true")),
            enable_speech_to_text=str_to_bool(os.getenv("ENABLE_SPEECH_TO_TEXT", "true")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            
            # Database configuration
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_SERVICE_KEY", ""),
            table_name=os.getenv("TABLE_NAME", "rag_documents"),
            
            # Model configuration
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4-turbo"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            
            # Processing limits
            max_concurrent_files=int(os.getenv("MAX_CONCURRENT_FILES", "5")),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            
            # Cache configuration
            cache_dir=os.getenv("CACHE_DIR", "./ingestion_cache"),
            enable_cache=str_to_bool(os.getenv("ENABLE_CACHE", "true")),
            
            # Logging configuration
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            
            # Performance tuning
            memory_limit_gb=float(os.getenv("MEMORY_LIMIT_GB", "8.0")),
            cpu_limit=int(os.getenv("CPU_LIMIT", "4")),
            
            # File type specific settings
            ocr_confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5")),
            audio_chunk_duration=int(os.getenv("AUDIO_CHUNK_DURATION", "30")),
            code_chunk_lines=int(os.getenv("CODE_CHUNK_LINES", "40")),
            
            # Supported extensions (can be overridden)
            supported_extensions=str_to_list(
                os.getenv("SUPPORTED_EXTENSIONS"), 
                ['pdf', 'docx', 'doc', 'txt', 'md', 'rtf', 'csv', 'xlsx', 'xls',
                 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'mp3', 'wav', 'm4a', 'flac',
                 'py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'sql',
                 'json', 'xml', 'yaml', 'yml', 'html', 'htm', 'css', 'zip', 'tar', 'gz']
            ),
            
            # Batch API settings
            use_batch_api=str_to_bool(os.getenv("USE_BATCH_API", "true")),
            batch_api_threshold=int(os.getenv("BATCH_API_THRESHOLD", "100")),
            batch_api_wait_timeout=int(os.getenv("BATCH_API_WAIT_TIMEOUT", "300")),
            batch_check_interval=int(os.getenv("BATCH_CHECK_INTERVAL", "300")),
            prefer_cost_savings=str_to_bool(os.getenv("PREFER_COST_SAVINGS", "true")),
            max_regular_api_batch=int(os.getenv("MAX_REGULAR_API_BATCH", "20"))        
        )
        
        logger.info("Configuration loaded successfully from environment")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_environment() -> List[str]:
    """Validate environment for missing or invalid settings"""
    issues = []
    
    # Check required environment variables
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY", 
        "OPENAI_API_KEY"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Missing required environment variable: {var}")
    
    # Check numeric values
    numeric_vars = {
        "CHUNK_SIZE": (100, 4000),
        "CHUNK_OVERLAP": (0, 1000), 
        "MAX_FILE_SIZE_MB": (1, 1000),
        "BATCH_SIZE": (1, 1000),
        "MAX_CONCURRENT_FILES": (1, 20)
    }
    
    for var, (min_val, max_val) in numeric_vars.items():
        value = os.getenv(var)
        if value:
            try:
                num_val = int(value)
                if not (min_val <= num_val <= max_val):
                    issues.append(f"{var}={num_val} is outside valid range [{min_val}, {max_val}]")
            except ValueError:
                issues.append(f"{var}={value} is not a valid integer")
    
    # Check boolean values
    boolean_vars = [
        "ENABLE_SEMANTIC_CHUNKING",
        "ENABLE_HIERARCHICAL_CHUNKING", 
        "EXTRACT_METADATA",
        "ENABLE_OCR",
        "ENABLE_SPEECH_TO_TEXT",
        "ENABLE_CACHE"
    ]
    
    for var in boolean_vars:
        value = os.getenv(var)
        if value and value.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            issues.append(f"{var}={value} is not a valid boolean value")
    
    # Check Supabase URL format
    supabase_url = os.getenv("SUPABASE_URL")
    if supabase_url and not supabase_url.startswith("https://"):
        issues.append("SUPABASE_URL should start with https://")
    
    # Check model names
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    if embedding_model not in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
        issues.append(f"EMBEDDING_MODEL={embedding_model} may not be supported")
    
    return issues


def create_example_env_file(file_path: str = ".env.example"):
    """Create an example environment file with all configuration options"""
    
    env_content = """# LlamaIndex Ingestion Service Configuration

# Required Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Chunking Configuration
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
ENABLE_SEMANTIC_CHUNKING=true
ENABLE_HIERARCHICAL_CHUNKING=true

# Processing Configuration
EXTRACT_METADATA=true
ENABLE_OCR=true
ENABLE_SPEECH_TO_TEXT=true
MAX_FILE_SIZE_MB=100

# Database Configuration
TABLE_NAME=rag_documents

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4-turbo

# Performance Configuration
MAX_CONCURRENT_FILES=5
BATCH_SIZE=100
MEMORY_LIMIT_GB=8.0
CPU_LIMIT=4

# Cache Configuration
CACHE_DIR=./ingestion_cache
ENABLE_CACHE=true

# Logging Configuration
LOG_LEVEL=INFO
# LOG_FILE=./logs/ingestion.log

# File Type Specific Settings
OCR_CONFIDENCE_THRESHOLD=0.5
AUDIO_CHUNK_DURATION=30
CODE_CHUNK_LINES=40

# Supported Extensions (comma-separated)
# SUPPORTED_EXTENSIONS=pdf,docx,txt,md,csv,xlsx,jpg,png,mp3,wav,py,js,json,html

# Service Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=false
"""
    
    try:
        with open(file_path, 'w') as f:
            f.write(env_content)
        print(f"Example environment file created: {file_path}")
    except Exception as e:
        print(f"Failed to create example environment file: {e}")


if __name__ == "__main__":
    # Validate current environment
    issues = validate_environment()
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nCreate a proper .env file to resolve these issues.")
        create_example_env_file()
    else:
        print("Environment validation passed!")
        
        try:
            config = get_config()
            print("Configuration loaded successfully!")
            print("Configuration summary:")
            import json
            print(json.dumps(config.get_summary(), indent=2))
        except Exception as e:
            print(f"Failed to load configuration: {e}")