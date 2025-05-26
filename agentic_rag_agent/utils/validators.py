"""
Input validation utilities for Agentic RAG Agent
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import mimetypes

from pydantic import BaseModel, ValidationError, validator
from pydantic.validators import str_validator

from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


class QueryValidator:
    """Validator for search queries"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bOR\b.*=.*\bOR\b)",
        r"(\'\s*(OR|AND)\s*\'\s*=\s*\')",
        r"(\;\s*(DROP|DELETE|UPDATE|INSERT))"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>"
    ]
    
    def __init__(self, 
                 min_length: int = 1,
                 max_length: int = 1000,
                 allow_special_chars: bool = True,
                 check_injection: bool = True):
        self.min_length = min_length
        self.max_length = max_length
        self.allow_special_chars = allow_special_chars
        self.check_injection = check_injection
    
    def validate(self, query: str) -> str:
        """
        Validate and sanitize query string
        
        Args:
            query: Query string to validate
            
        Returns:
            Sanitized query string
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(query, str):
            raise ValidationError("Query must be a string", "query", query)
        
        # Strip whitespace
        query = query.strip()
        
        # Check length
        if len(query) < self.min_length:
            raise ValidationError(
                f"Query too short (minimum {self.min_length} characters)",
                "query", query
            )
        
        if len(query) > self.max_length:
            raise ValidationError(
                f"Query too long (maximum {self.max_length} characters)",
                "query", query
            )
        
        # Check for empty or whitespace-only
        if not query or query.isspace():
            raise ValidationError("Query cannot be empty or whitespace only", "query", query)
        
        # Check for injection attacks
        if self.check_injection:
            self._check_injection_patterns(query)
        
        # Sanitize if needed
        if not self.allow_special_chars:
            query = self._sanitize_special_chars(query)
        
        return query
    
    def _check_injection_patterns(self, query: str):
        """Check for SQL injection and XSS patterns"""
        query_upper = query.upper()
        
        # Check SQL injection patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {query[:100]}")
                raise ValidationError("Query contains potentially dangerous content", "query")
        
        # Check XSS patterns
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {query[:100]}")
                raise ValidationError("Query contains potentially dangerous content", "query")
    
    def _sanitize_special_chars(self, query: str) -> str:
        """Remove or escape special characters"""
        # Allow alphanumeric, spaces, and basic punctuation
        sanitized = re.sub(r'[^\w\s\-_.?!,;:]', '', query)
        return sanitized


class FileValidator:
    """Validator for file inputs"""
    
    # Allowed file extensions by category
    ALLOWED_EXTENSIONS = {
        'documents': ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf'],
        'spreadsheets': ['.csv', '.xlsx', '.xls'],
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
        'audio': ['.mp3', '.wav', '.m4a', '.flac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'code': ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go'],
        'data': ['.json', '.xml', '.yaml', '.yml'],
        'web': ['.html', '.htm', '.css'],
        'archives': ['.zip', '.tar', '.gz', '.rar']
    }
    
    # Dangerous extensions to block
    DANGEROUS_EXTENSIONS = [
        '.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.dll',
        '.msi', '.vbs', '.js', '.jar', '.app', '.deb', '.rpm'
    ]
    
    def __init__(self,
                 max_size_mb: int = 100,
                 allowed_categories: List[str] = None,
                 check_content: bool = True):
        self.max_size_mb = max_size_mb
        self.allowed_categories = allowed_categories or list(self.ALLOWED_EXTENSIONS.keys())
        self.check_content = check_content
        
        # Build allowed extensions list
        self.allowed_extensions = set()
        for category in self.allowed_categories:
            self.allowed_extensions.update(self.ALLOWED_EXTENSIONS.get(category, []))
    
    def validate_path(self, file_path: str) -> Path:
        """
        Validate file path
        
        Args:
            file_path: Path to file
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(file_path, (str, Path)):
            raise ValidationError("File path must be string or Path object", "file_path", file_path)
        
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}", "file_path", file_path)
        
        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}", "file_path", file_path)
        
        # Check file extension
        extension = path.suffix.lower()
        
        # Block dangerous extensions
        if extension in self.DANGEROUS_EXTENSIONS:
            raise ValidationError(f"File type not allowed: {extension}", "file_path", file_path)
        
        # Check if extension is allowed
        if extension not in self.allowed_extensions:
            raise ValidationError(
                f"File extension '{extension}' not allowed. Allowed: {sorted(self.allowed_extensions)}",
                "file_path", file_path
            )
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB (max: {self.max_size_mb}MB)",
                "file_path", file_path
            )
        
        # Content-based validation
        if self.check_content:
            self._validate_file_content(path)
        
        return path
    
    def validate_paths(self, file_paths: List[str]) -> List[Path]:
        """Validate multiple file paths"""
        if not isinstance(file_paths, list):
            raise ValidationError("File paths must be a list", "file_paths", file_paths)
        
        if len(file_paths) == 0:
            raise ValidationError("File paths list cannot be empty", "file_paths", file_paths)
        
        if len(file_paths) > 100:  # Reasonable limit
            raise ValidationError("Too many files (max: 100)", "file_paths", file_paths)
        
        validated_paths = []
        for file_path in file_paths:
            validated_paths.append(self.validate_path(file_path))
        
        return validated_paths
    
    def _validate_file_content(self, path: Path):
        """Validate file content (basic checks)"""
        try:
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            
            # Read first few bytes to check for file signatures
            with open(path, 'rb') as f:
                header = f.read(512)  # First 512 bytes
            
            # Check for executable signatures
            executable_signatures = [
                b'MZ',  # Windows executable
                b'\x7fELF',  # Linux executable
                b'\xca\xfe\xba\xbe',  # Java class file
                b'PK\x03\x04\x14\x00\x08\x00\x08\x00',  # Potential malicious ZIP
            ]
            
            for sig in executable_signatures:
                if header.startswith(sig):
                    raise ValidationError("File appears to be executable", "file_content")
            
        except PermissionError:
            raise ValidationError("Cannot read file (permission denied)", "file_access")
        except Exception as e:
            logger.warning(f"Content validation failed for {path}: {e}")
            # Don't fail validation for content check errors, just log


class ConfigValidator:
    """Validator for configuration values"""
    
    @staticmethod
    def validate_similarity_threshold(threshold: float) -> float:
        """Validate similarity threshold"""
        if not isinstance(threshold, (int, float)):
            raise ValidationError("Similarity threshold must be a number", "similarity_threshold", threshold)
        
        if not 0.0 <= threshold <= 1.0:
            raise ValidationError("Similarity threshold must be between 0.0 and 1.0", "similarity_threshold", threshold)
        
        return float(threshold)
    
    @staticmethod
    def validate_max_results(max_results: int) -> int:
        """Validate max results parameter"""
        if not isinstance(max_results, int):
            raise ValidationError("Max results must be an integer", "max_results", max_results)
        
        if max_results < 1:
            raise ValidationError("Max results must be at least 1", "max_results", max_results)
        
        if max_results > 1000:
            raise ValidationError("Max results cannot exceed 1000", "max_results", max_results)
        
        return max_results
    
    @staticmethod
    def validate_file_types(file_types: List[str]) -> List[str]:
        """Validate file types list"""
        if not isinstance(file_types, list):
            raise ValidationError("File types must be a list", "file_types", file_types)
        
        valid_extensions = set()
        for category_exts in FileValidator.ALLOWED_EXTENSIONS.values():
            valid_extensions.update(ext.lstrip('.') for ext in category_exts)
        
        validated_types = []
        for file_type in file_types:
            if not isinstance(file_type, str):
                raise ValidationError("File type must be a string", "file_types", file_type)
            
            # Normalize extension
            normalized = file_type.lower().lstrip('.')
            
            if normalized not in valid_extensions:
                raise ValidationError(f"Invalid file type: {file_type}", "file_types", file_type)
            
            validated_types.append(normalized)
        
        return list(set(validated_types))  # Remove duplicates
    
    @staticmethod
    def validate_date_filter(date_str: str) -> datetime:
        """Validate date filter"""
        if not isinstance(date_str, str):
            raise ValidationError("Date must be a string", "date_filter", date_str)
        
        # Try different date formats
        date_formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try relative dates
        if date_str.lower() in ['today', 'yesterday']:
            if date_str.lower() == 'today':
                return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            else:  # yesterday
                return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        
        raise ValidationError(f"Invalid date format: {date_str}", "date_filter", date_str)


class RequestValidator:
    """Comprehensive request validator"""
    
    def __init__(self):
        self.query_validator = QueryValidator()
        self.file_validator = FileValidator()
        self.config_validator = ConfigValidator()
    
    def validate_search_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search request"""
        validated = {}
        
        # Required fields
        if 'question' not in request_data:
            raise ValidationError("Question is required", "question")
        
        validated['question'] = self.query_validator.validate(request_data['question'])
        
        # Optional fields with defaults
        validated['enable_iteration'] = bool(request_data.get('enable_iteration', True))
        validated['enable_reflection'] = bool(request_data.get('enable_reflection', True))
        validated['enable_triangulation'] = bool(request_data.get('enable_triangulation', True))
        
        # Search configuration
        if 'search_method' in request_data:
            method = request_data['search_method']
            if method not in ['hybrid', 'semantic', 'keyword']:
                raise ValidationError("Invalid search method", "search_method", method)
            validated['search_method'] = method
        
        if 'max_results' in request_data:
            validated['max_results'] = self.config_validator.validate_max_results(
                request_data['max_results']
            )
        
        if 'similarity_threshold' in request_data:
            validated['similarity_threshold'] = self.config_validator.validate_similarity_threshold(
                request_data['similarity_threshold']
            )
        
        if 'file_types' in request_data:
            validated['file_types'] = self.config_validator.validate_file_types(
                request_data['file_types']
            )
        
        if 'date_after' in request_data:
            validated['date_after'] = self.config_validator.validate_date_filter(
                request_data['date_after']
            )
        
        return validated
    
    def validate_ingestion_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ingestion request"""
        validated = {}
        
        # File paths validation
        if 'file_paths' in request_data:
            validated['file_paths'] = [
                str(path) for path in self.file_validator.validate_paths(request_data['file_paths'])
            ]
        elif 'directory_path' in request_data:
            dir_path = request_data['directory_path']
            if not isinstance(dir_path, (str, Path)):
                raise ValidationError("Directory path must be string or Path", "directory_path")
            
            path = Path(dir_path)
            if not path.exists() or not path.is_dir():
                raise ValidationError("Directory does not exist", "directory_path", dir_path)
            
            validated['directory_path'] = str(path)
            validated['recursive'] = bool(request_data.get('recursive', True))
            
            if 'file_extensions' in request_data:
                validated['file_extensions'] = self.config_validator.validate_file_types(
                    request_data['file_extensions']
                )
        else:
            raise ValidationError("Either file_paths or directory_path is required")
        
        # Optional processing parameters
        if 'batch_size' in request_data:
            batch_size = request_data['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1000:
                raise ValidationError("Batch size must be between 1 and 1000", "batch_size", batch_size)
            validated['batch_size'] = batch_size
        
        return validated
    
    def validate_batch_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate batch processing request"""
        validated = {}
        
        # Questions list
        if 'questions' not in request_data:
            raise ValidationError("Questions list is required", "questions")
        
        questions = request_data['questions']
        if not isinstance(questions, list):
            raise ValidationError("Questions must be a list", "questions", questions)
        
        if len(questions) == 0:
            raise ValidationError("Questions list cannot be empty", "questions")
        
        if len(questions) > 50:  # Reasonable batch limit
            raise ValidationError("Too many questions in batch (max: 50)", "questions")
        
        validated['questions'] = [
            self.query_validator.validate(q) for q in questions
        ]
        
        # Batch processing options
        validated['enable_agentic'] = bool(request_data.get('enable_agentic', True))
        validated['enable_iteration'] = bool(request_data.get('enable_iteration', True))
        validated['enable_reflection'] = bool(request_data.get('enable_reflection', True))
        validated['enable_triangulation'] = bool(request_data.get('enable_triangulation', False))
        
        # Performance limits
        if 'max_concurrency' in request_data:
            concurrency = request_data['max_concurrency']
            if not isinstance(concurrency, int) or concurrency < 1 or concurrency > 10:
                raise ValidationError("Max concurrency must be between 1 and 10", "max_concurrency")
            validated['max_concurrency'] = concurrency
        
        if 'timeout_per_question' in request_data:
            timeout = request_data['timeout_per_question']
            if not isinstance(timeout, int) or timeout < 5 or timeout > 300:
                raise ValidationError("Timeout must be between 5 and 300 seconds", "timeout_per_question")
            validated['timeout_per_question'] = timeout
        
        return validated


class SanitizationError(Exception):
    """Error during sanitization"""
    pass


class TextSanitizer:
    """Text sanitization utilities"""
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize query text"""
        if not isinstance(query, str):
            return ""
        
        # Strip whitespace
        query = query.strip()
        
        # Remove null bytes
        query = query.replace('\x00', '')
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove control characters except newlines and tabs
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', query)
        
        return query
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not isinstance(filename, str):
            return "unnamed_file"
        
        # Remove directory separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure not empty
        if not filename:
            filename = "unnamed_file"
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_len = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_len] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def sanitize_user_input(text: str, max_length: int = 1000) -> str:
        """General user input sanitization"""
        if not isinstance(text, str):
            return ""
        
        # Basic sanitization
        text = text.strip()
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        return text


class RateLimitValidator:
    """Rate limiting validation"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_history = {}
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        if client_id in self.request_history:
            self.request_history[client_id] = [
                timestamp for timestamp in self.request_history[client_id]
                if timestamp > minute_ago
            ]
        else:
            self.request_history[client_id] = []
        
        # Check current count
        current_count = len(self.request_history[client_id])
        
        if current_count >= self.max_requests_per_minute:
            return False
        
        # Add current request
        self.request_history[client_id].append(now)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if client_id not in self.request_history:
            return self.max_requests_per_minute
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Count recent requests
        recent_requests = sum(
            1 for timestamp in self.request_history[client_id]
            if timestamp > minute_ago
        )
        
        return max(0, self.max_requests_per_minute - recent_requests)


# Global validators
query_validator = QueryValidator()
file_validator = FileValidator()
config_validator = ConfigValidator()
request_validator = RequestValidator()
text_sanitizer = TextSanitizer()
rate_limit_validator = RateLimitValidator()


# Convenience functions
def validate_query(query: str) -> str:
    """Validate and sanitize query"""
    return query_validator.validate(query)


def validate_file_path(file_path: str) -> Path:
    """Validate file path"""
    return file_validator.validate_path(file_path)


def validate_file_paths(file_paths: List[str]) -> List[Path]:
    """Validate multiple file paths"""
    return file_validator.validate_paths(file_paths)


def validate_search_params(**params) -> Dict[str, Any]:
    """Validate search parameters"""
    return request_validator.validate_search_request(params)


def sanitize_text(text: str) -> str:
    """Sanitize user input text"""
    return text_sanitizer.sanitize_user_input(text)


def check_rate_limit(client_id: str) -> bool:
    """Check rate limit for client"""
    return rate_limit_validator.check_rate_limit(client_id)