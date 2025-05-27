"""
Authentication and Security Module
File: agentic_rag_agent/auth/security.py
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = secrets.token_urlsafe(32)  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


rate_limiter = RateLimiter()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Use IP address or API key as client ID
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host


async def rate_limit_check(request: Request):
    """Rate limiting dependency"""
    client_id = get_client_id(request)
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Try again later."
        )
    
    return client_id


def hash_password(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


# API Key authentication for simpler use cases
def verify_api_key(request: Request):
    """Verify API key from header"""
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required")
    
    # In production, store API keys in database
    valid_api_keys = {
        "demo-key-123": {"name": "demo-user", "permissions": ["read", "write"]},
        "readonly-key-456": {"name": "readonly-user", "permissions": ["read"]}
    }
    
    if api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return valid_api_keys[api_key]


# Input sanitization
def sanitize_input(text: str, max_length: int = 5000) -> str:
    """Sanitize user input"""
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Input must be a string")
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length
    if len(sanitized) > max_length:
        raise HTTPException(status_code=400, detail=f"Input too long (max {max_length} characters)")
    
    # Check for potential injection attacks
    dangerous_patterns = [
        "javascript:", "<script", "onload=", "onerror=", 
        "eval(", "exec(", "__import__", "subprocess"
    ]
    
    lower_text = sanitized.lower()
    for pattern in dangerous_patterns:
        if pattern in lower_text:
            raise HTTPException(status_code=400, detail="Potentially dangerous input detected")
    
    return sanitized


# Security headers middleware
from fastapi import FastAPI
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
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


# Updated main.py with security
def add_security_to_main():
    """
    Add these to main.py:
    
    from auth.security import (
        rate_limit_check, verify_api_key, sanitize_input, 
        SecurityHeadersMiddleware
    )
    
    # Add middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Update endpoints with security
    @app.post("/ask", dependencies=[Depends(rate_limit_check), Depends(verify_api_key)])
    async def ask_question(request: QueryRequest):
        # Sanitize input
        request.question = sanitize_input(request.question)
        # ... rest of function
    """
    pass