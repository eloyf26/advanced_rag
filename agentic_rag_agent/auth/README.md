# `/auth/` Folder Documentation

## Overview

The `/auth/` folder contains security components for authentication, authorization, and protection of the agentic RAG system. This module provides comprehensive security features including rate limiting, input sanitization, authentication mechanisms, and security headers to ensure safe operation in production environments.

## 🎯 Design Principles

- **Defense in Depth**: Multiple layers of security protection
- **Zero Trust**: Validate and sanitize all inputs
- **Performance**: Lightweight security that doesn't impact response times
- **Flexibility**: Support for multiple authentication methods
- **Standards Compliance**: Follow security best practices and standards

## 📁 File Structure

```
auth/
└── security.py          # Authentication, rate limiting, and security utilities
```

## 🔧 Component Details

### `security.py` - Core Security Module

**Purpose**: Provides comprehensive security features including authentication, rate limiting, input sanitization, and security middleware for production deployment.

#### Key Security Features

##### 1. Authentication Systems

**JWT Authentication**
```python
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token with configurable expiration"""
    
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user information"""
```

**API Key Authentication**
```python
def verify_api_key(request: Request):
    """Verify API key from X-API-Key header"""
    # Supports multiple API keys with different permissions
    valid_api_keys = {
        "demo-key-123": {"name": "demo-user", "permissions": ["read", "write"]},
        "readonly-key-456": {"name": "readonly-user", "permissions": ["read"]}
    }
```

**Features**:
- **JWT Tokens**: Stateless authentication with configurable expiration
- **API Keys**: Simple key-based authentication for service-to-service calls
- **Permission System**: Role-based access control with read/write permissions
- **Multiple Auth Methods**: Support for both JWT and API key authentication

##### 2. Rate Limiting

**In-Memory Rate Limiter**
```python
class RateLimiter:
    """Simple in-memory rate limiter with sliding window"""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = {}  # Track requests per client
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is within rate limits"""
```

**Client Identification**
```python
def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Uses X-Forwarded-For header or client IP
```

**Features**:
- **Sliding Window**: Time-based rate limiting with automatic cleanup
- **Per-Client Limits**: Individual rate limits per IP/client
- **Configurable Thresholds**: Adjustable request limits and time windows
- **Header Support**: Proper handling of proxy headers (X-Forwarded-For)

##### 3. Input Sanitization

**Query Sanitization**
```python
def sanitize_input(text: str, max_length: int = 5000) -> str:
    """Sanitize user input against injection attacks"""
    # Removes null bytes, control characters
    # Normalizes whitespace
    # Checks for dangerous patterns
```

**Injection Attack Prevention**
```python
# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
    r"(--|\#|\/\*|\*\/)",
    r"(\bOR\b.*=.*\bOR\b)"
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*="
]
```

**Filename Sanitization**
```python
def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Removes dangerous characters
    # Prevents directory traversal
    # Limits length appropriately
```

**Features**:
- **SQL Injection Protection**: Pattern-based detection and blocking
- **XSS Prevention**: Script tag and JavaScript event handler detection
- **Path Traversal Protection**: Safe filename handling
- **Input Validation**: Length limits and character filtering

##### 4. Security Headers Middleware

**Comprehensive Security Headers**
```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
```

**Security Headers Applied**:
- **X-Content-Type-Options**: Prevents MIME sniffing attacks
- **X-Frame-Options**: Prevents clickjacking attacks
- **X-XSS-Protection**: Enables browser XSS filtering
- **Strict-Transport-Security**: Enforces HTTPS connections
- **Content-Security-Policy**: Controls resource loading
- **Referrer-Policy**: Controls referrer information leakage

##### 5. Password Security

**Secure Password Handling**
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
```

**Features**:
- **bcrypt Hashing**: Industry-standard password hashing
- **Salt Generation**: Automatic salt generation for each password
- **Timing Attack Protection**: Constant-time comparison

## 🔗 Integration with Main Application

### FastAPI Integration

**Adding Security to Main Application**
```python
# In main.py
from auth.security import (
    rate_limit_check, verify_api_key, sanitize_input, 
    SecurityHeadersMiddleware
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)

# Secure endpoints
@app.post("/ask", dependencies=[Depends(rate_limit_check), Depends(verify_api_key)])
async def ask_question(request: QueryRequest):
    # Sanitize input
    request.question = sanitize_input(request.question)
    # ... rest of function
```

### Configuration-Based Security

**Environment-Controlled Security Features**
```python
# Enable/disable security features via environment variables
ENABLE_AUTHENTICATION = os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true"
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
```

### Security Dependencies

**Flexible Security Enforcement**
```python
def get_security_dependencies():
    """Get security dependencies based on configuration"""
    deps = []
    
    if os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true":
        deps.append(Depends(rate_limit_check))
    
    if os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true":
        deps.append(Depends(verify_api_key))
    
    return deps
```

## 🛡️ Security Best Practices Implemented

### 1. Input Validation
- **Comprehensive Sanitization**: All user inputs are sanitized
- **Length Limits**: Configurable maximum input lengths
- **Pattern Detection**: Recognition of common attack patterns
- **Character Filtering**: Removal of dangerous characters

### 2. Authentication Security
- **Token Expiration**: Configurable JWT token lifetimes
- **Secure Storage**: Tokens stored securely on client side
- **Multiple Methods**: Support for different authentication approaches
- **Permission Granularity**: Fine-grained access control

### 3. Rate Limiting
- **Abuse Prevention**: Protection against DoS attacks
- **Fair Usage**: Ensuring equitable resource access
- **Adaptive Limits**: Different limits for different user types
- **Monitoring**: Rate limit violation tracking

### 4. Transport Security
- **HTTPS Enforcement**: Strict transport security headers
- **Secure Headers**: Comprehensive security header implementation
- **CORS Protection**: Configurable cross-origin resource sharing
- **Content Security**: CSP headers to prevent XSS

## 🔧 Configuration Options

### Environment Variables

```bash
# Authentication
ENABLE_AUTHENTICATION=false
JWT_SECRET_KEY=your-very-long-and-secure-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Key Authentication
API_KEY_HEADER=X-API-Key
VALID_API_KEYS=demo-key-123,readonly-key-456

# Rate Limiting
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=60

# Security Features
ENABLE_CORS=true
CORS_ORIGINS=*
```

### Security Levels

**Development Mode**
```python
# Relaxed security for development
ENABLE_AUTHENTICATION=false
ENABLE_RATE_LIMITING=false
DEBUG_MODE=true
```

**Production Mode**
```python
# Full security for production
ENABLE_AUTHENTICATION=true
ENABLE_RATE_LIMITING=true
ENABLE_CORS=false
DEBUG_MODE=false
```

## 🚀 Usage Examples

### Basic Security Setup
```python
from fastapi import FastAPI, Depends
from auth.security import rate_limit_check, verify_api_key, SecurityHeadersMiddleware

app = FastAPI()

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Secure endpoint
@app.post("/api/secure-endpoint")
async def secure_endpoint(
    data: dict,
    client_id: str = Depends(rate_limit_check),
    user: dict = Depends(verify_api_key)
):
    return {"message": "Secure response", "user": user["name"]}
```

### Custom Authentication
```python
# Custom API key validation
async def custom_auth(request: Request):
    api_key = request.headers.get("Authorization")
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization")
    
    # Custom validation logic
    token = api_key.split(" ")[1]
    if not validate_custom_token(token):
        raise HTTPException(401, "Invalid token")
    
    return {"user_id": extract_user_from_token(token)}
```

### Input Sanitization
```python
from auth.security import sanitize_input

@app.post("/api/query")
async def process_query(request: QueryRequest):
    # Sanitize all user inputs
    safe_question = sanitize_input(request.question)
    safe_context = sanitize_input(request.context or "")
    
    # Process with sanitized inputs
    return await process_safe_query(safe_question, safe_context)
```

## 🔍 Monitoring and Logging

### Security Event Logging
- **Authentication Failures**: Failed login attempts
- **Rate Limit Violations**: Clients exceeding limits
- **Injection Attempts**: Detected attack patterns
- **Authorization Errors**: Permission violations

### Metrics Collection
- **Request Rates**: Per-client request frequencies
- **Authentication Success Rates**: Login success metrics
- **Security Violations**: Count of security events
- **Performance Impact**: Security overhead measurement

## 🛠️ Customization and Extension

### Adding New Authentication Methods
```python
def verify_custom_auth(request: Request):
    """Custom authentication method"""
    # Implement custom logic
    auth_header = request.headers.get("X-Custom-Auth")
    if not validate_custom_auth(auth_header):
        raise HTTPException(401, "Custom auth failed")
    return {"method": "custom", "validated": True}
```

### Custom Rate Limiting
```python
class CustomRateLimiter(RateLimiter):
    """Extended rate limiter with user-specific limits"""
    
    def __init__(self):
        super().__init__()
        self.user_limits = {
            "premium": 1000,
            "standard": 100,
            "free": 10
        }
    
    def is_allowed(self, client_id: str, user_tier: str = "standard") -> bool:
        limit = self.user_limits.get(user_tier, 100)
        # Custom logic with tier-based limits
```

### Security Middleware Extensions
```python
class EnhancedSecurityMiddleware(SecurityHeadersMiddleware):
    """Extended security middleware with custom features"""
    
    async def dispatch(self, request, call_next):
        # Add custom security checks
        if not self.validate_request_signature(request):
            return JSONResponse({"error": "Invalid signature"}, status_code=401)
        
        # Call parent middleware
        return await super().dispatch(request, call_next)
```

## 🎯 Production Considerations

### Performance
- **Minimal Overhead**: Security checks optimized for speed
- **Caching**: Rate limit data cached in memory
- **Async Operations**: Non-blocking security validations

### Scalability
- **Stateless Design**: JWT tokens for horizontal scaling
- **External Storage**: Option to use Redis for rate limiting
- **Load Balancer Support**: Proper header handling for proxies

### Compliance
- **Data Protection**: No sensitive data in logs
- **Audit Trail**: Comprehensive security event logging
- **Standards Compliance**: Following OWASP security guidelines

The auth folder provides robust security infrastructure that protects the agentic RAG system while maintaining high performance and flexibility for different deployment scenarios.