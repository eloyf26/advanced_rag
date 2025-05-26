"""
Logging configuration for Agentic RAG Agent
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'query'):
            log_entry['query'] = record.query
        
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class RAGLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for RAG-specific context"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add context from extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        return msg, kwargs
    
    def log_query(self, level, query, processing_time=None, request_id=None, **kwargs):
        """Log query-specific information"""
        extra = {
            'query': query,
            'request_id': request_id,
            'processing_time': processing_time,
            **kwargs
        }
        self.log(level, f"Query processed: {query[:100]}...", extra=extra)
    
    def log_search(self, level, method, results_count, processing_time, **kwargs):
        """Log search-specific information"""
        extra = {
            'search_method': method,
            'results_count': results_count,
            'processing_time': processing_time,
            **kwargs
        }
        self.log(level, f"Search completed: {method} returned {results_count} results", extra=extra)
    
    def log_error(self, level, error, error_type=None, context=None, **kwargs):
        """Log error with additional context"""
        extra = {
            'error_type': error_type or type(error).__name__,
            'context': context,
            **kwargs
        }
        self.log(level, f"Error occurred: {str(error)}", extra=extra, exc_info=True)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_structured: bool = True,
    enable_file_rotation: bool = True,
    max_file_size: str = "10MB",
    backup_count: int = 5
) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_structured: Enable structured JSON logging
        enable_file_rotation: Enable log file rotation
        max_file_size: Maximum size per log file
        backup_count: Number of backup files to keep
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if enable_file_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=_parse_size(max_file_size),
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # RAG-specific loggers
    logging.getLogger("agentic_rag").setLevel(getattr(logging, log_level.upper()))


def get_logger(name: str, **context) -> RAGLoggerAdapter:
    """
    Get a configured logger with RAG-specific adapter
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all log messages
    
    Returns:
        Configured RAGLoggerAdapter instance
    """
    base_logger = logging.getLogger(name)
    return RAGLoggerAdapter(base_logger, context)


def get_structured_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger using structlog
    
    Args:
        name: Logger name
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggingMiddleware:
    """Middleware for logging HTTP requests and responses"""
    
    def __init__(self, app, logger: Optional[logging.Logger] = None):
        self.app = app
        self.logger = logger or get_logger(__name__)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = datetime.utcnow()
            
            # Log request
            self.logger.info(
                "HTTP request started",
                extra={
                    'method': scope['method'],
                    'path': scope['path'],
                    'query_string': scope.get('query_string', b'').decode(),
                    'client': scope.get('client', ['unknown', 0])[0]
                }
            )
            
            # Process request
            await self.app(scope, receive, send)
            
            # Log completion
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                "HTTP request completed",
                extra={
                    'method': scope['method'],
                    'path': scope['path'],
                    'processing_time': processing_time
                }
            )
        else:
            await self.app(scope, receive, send)


class QueryLoggingContext:
    """Context manager for query-specific logging"""
    
    def __init__(self, query: str, request_id: str = None):
        self.query = query
        self.request_id = request_id
        self.start_time = None
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            "Query processing started",
            extra={
                'query': self.query,
                'request_id': self.request_id
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        processing_time = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                "Query processing completed successfully",
                extra={
                    'query': self.query,
                    'request_id': self.request_id,
                    'processing_time': processing_time
                }
            )
        else:
            self.logger.error(
                "Query processing failed",
                extra={
                    'query': self.query,
                    'request_id': self.request_id,
                    'processing_time': processing_time,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                },
                exc_info=True
            )


def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes"""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


# Performance logging utilities
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, logger: logging.Logger = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        self.logger.info(
            f"Operation completed: {self.operation_name}",
            extra={
                'operation': self.operation_name,
                'duration': duration,
                'success': exc_type is None
            }
        )


def log_function_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        with PerformanceTimer(func.__name__, logger):
            return func(*args, **kwargs)
    
    return wrapper


async def log_async_function_performance(func):
    """Decorator to log async function performance"""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        with PerformanceTimer(func.__name__, logger):
            return await func(*args, **kwargs)
    
    return wrapper


# Error tracking utilities
class ErrorTracker:
    """Track and aggregate errors for monitoring"""
    
    def __init__(self):
        self.errors = {}
        self.logger = get_logger(__name__)
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track an error occurrence"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        key = f"{error_type}:{error_msg}"
        
        if key not in self.errors:
            self.errors[key] = {
                'count': 0,
                'first_seen': datetime.utcnow(),
                'last_seen': datetime.utcnow(),
                'contexts': []
            }
        
        self.errors[key]['count'] += 1
        self.errors[key]['last_seen'] = datetime.utcnow()
        
        if context:
            self.errors[key]['contexts'].append(context)
        
        # Log the error
        self.logger.error(
            f"Error tracked: {error_type}",
            extra={
                'error_type': error_type,
                'error_message': error_msg,
                'context': context,
                'occurrence_count': self.errors[key]['count']
            }
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors"""
        return {
            'total_unique_errors': len(self.errors),
            'total_error_count': sum(e['count'] for e in self.errors.values()),
            'most_frequent_errors': sorted(
                self.errors.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]
        }


# Global error tracker instance
error_tracker = ErrorTracker()