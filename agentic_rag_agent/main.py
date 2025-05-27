"""
Main entry point for the PydanticAI Agentic RAG Agent
"""

import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.security import HTTPBearer
import uvicorn

# Import configuration and core services
from config import get_config, validate_environment
from agents.agentic_rag_service import AgenticRAGService

# Import data models
from models.request_models import (
    QueryRequest, BatchQueryRequest, AnalyzeQueryRequest,
    CacheRequest, ConfigUpdateRequest, DebugRequest
)
from models.response_models import (
    AgenticRAGResponse, BatchQueryResponse, HealthResponse,
    QueryAnalysisResponse, PerformanceMetrics, ErrorResponse,
    CacheStatsResponse
)

# Import security and utilities
from auth.security import (
    rate_limit_check, verify_api_key, sanitize_input, 
    SecurityHeadersMiddleware, get_client_id
)
from utils.logger import get_logger, QueryLoggingContext
from utils.validators import validate_query, validate_search_params
from utils.metrics import get_metrics_collector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

# Initialize metrics collector
metrics = get_metrics_collector()

# Global service instance
rag_service: Optional[AgenticRAGService] = None

# Task storage for background processing
task_storage: Dict[str, Dict[str, Any]] = {}

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global rag_service
    
    try:
        # Validate environment variables
        validate_environment()
        
        # Initialize configuration
        config = get_config()
        
        # Initialize the RAG service
        rag_service = AgenticRAGService(config)
        
        # Start background services
        await rag_service.start_background_services()
        
        logger.info("Agentic RAG service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise
    
    # Shutdown
    try:
        if rag_service:
            await rag_service.cleanup()
        logger.info("Agentic RAG service shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# FastAPI app with lifespan management
app = FastAPI(
    title="PydanticAI Agentic RAG Agent",
    description="Advanced RAG system with planning, reflection, and iterative search",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware with configurable origins
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_id = get_client_id(request)
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                'method': request.method,
                'path': request.url.path,
                'client_id': client_id,
                'user_agent': request.headers.get('user-agent', 'unknown')
            }
        )
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Record metrics
            metrics.record_query(
                query_type='api_request',
                duration=processing_time,
                status='success'
            )
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': response.status_code,
                    'processing_time': processing_time,
                    'client_id': client_id
                }
            )
            
            # Add response headers
            response.headers["X-Processing-Time"] = str(processing_time)
            response.headers["X-Request-ID"] = client_id
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            metrics.record_error('request_error', 'api', client_id=client_id)
            
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    'method': request.method,
                    'path': request.url.path,
                    'error': str(e),
                    'processing_time': processing_time,
                    'client_id': client_id
                }
            )
            
            raise


app.add_middleware(RequestLoggingMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=time.time()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=time.time(),
            suggested_actions=["Please try again later", "Contact support if issue persists"]
        ).dict()
    )


# Helper functions
def get_security_dependencies():
    """Get security dependencies based on configuration"""
    deps = []
    
    if os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true":
        deps.append(Depends(rate_limit_check))
    
    if os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true":
        deps.append(Depends(verify_api_key))
    
    return deps


def validate_service():
    """Validate that service is available"""
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )


# Core endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        validate_service()
        
        # Get health status from service
        health_status = await rag_service.get_health_status()
        
        # Determine overall status
        overall_status = "healthy"
        if not health_status.get("database_connected", False):
            overall_status = "unhealthy"
        elif not health_status.get("embedding_service") == "healthy":
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            service="agentic_rag",
            database_connected=health_status.get("database_connected", False),
            embedding_service=health_status.get("embedding_service", "unknown"),
            llm_service=health_status.get("llm_service", "unknown"),
            details=health_status,
            avg_response_time_ms=health_status.get("avg_response_time", 0) * 1000,
            cache_hit_rate=health_status.get("cache_hit_rate", 0),
            error_rate=health_status.get("error_rate", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/ask", response_model=AgenticRAGResponse, dependencies=get_security_dependencies())
async def ask_question(request: QueryRequest):
    """Ask a question with full agentic capabilities"""
    validate_service()
    
    try:
        # Validate and sanitize input
        validated_question = validate_query(request.question)
        validated_params = validate_search_params(
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            file_types=request.file_types
        )
        
        with QueryLoggingContext(validated_question):
            response = await rag_service.ask_with_planning(
                question=validated_question,
                enable_iteration=request.enable_iteration,
                enable_reflection=request.enable_reflection,
                enable_triangulation=request.enable_triangulation,
                max_results=validated_params.get('max_results', 10),
                file_types=validated_params.get('file_types')
            )
            
            return response
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ask/simple", dependencies=get_security_dependencies())
async def ask_simple(request: QueryRequest):
    """Ask a question with simplified RAG (no agentic features)"""
    validate_service()
    
    try:
        validated_question = validate_query(request.question)
        
        with QueryLoggingContext(validated_question):
            response = await rag_service.ask(
                question=validated_question,
                search_method=request.search_method or "hybrid",
                file_types=request.file_types,
                max_results=request.max_results
            )
            
            return response
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing simple question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ask/batch", response_model=BatchQueryResponse, dependencies=get_security_dependencies())
async def ask_batch(request: BatchQueryRequest, background_tasks: BackgroundTasks):
    """Process multiple questions in batch"""
    validate_service()
    
    try:
        # Validate questions
        validated_questions = [validate_query(q) for q in request.questions]
        
        # Generate task ID
        task_id = f"batch_{int(time.time())}_{len(task_storage)}"
        
        # Initialize task status
        task_storage[task_id] = {
            "status": "pending",
            "total_questions": len(validated_questions),
            "completed": 0,
            "results": [],
            "errors": [],
            "created_at": time.time()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_batch_background,
            task_id,
            request,
            validated_questions
        )
        
        return BatchQueryResponse(
            task_id=task_id,
            status="started",
            total_questions=len(validated_questions),
            message=f"Processing {len(validated_questions)} questions"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting batch processing: {str(e)}"
        )


@app.get("/ask/batch/{task_id}")
async def get_batch_status(task_id: str, client_id: str = Depends(get_client_id)):
    """Get the status of a batch processing task"""
    if task_id not in task_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_data = task_storage[task_id].copy()
    
    # Add progress information
    if task_data["total_questions"] > 0:
        task_data["progress_percent"] = (task_data["completed"] / task_data["total_questions"]) * 100
    else:
        task_data["progress_percent"] = 0
    
    # Clean up old completed tasks (older than 1 hour)
    current_time = time.time()
    if (task_data.get("status") == "completed" and 
        current_time - task_data.get("created_at", 0) > 3600):
        del task_storage[task_id]
    
    return task_data


@app.post("/analyze/query", response_model=QueryAnalysisResponse, dependencies=get_security_dependencies())
async def analyze_query(request: AnalyzeQueryRequest):
    """Analyze a query without executing it"""
    validate_service()
    
    try:
        validated_question = validate_query(request.question)
        
        analysis = await rag_service.analyze_query(validated_question)
        return analysis
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing query: {str(e)}"
        )


@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics():
    """Get performance metrics"""
    validate_service()
    
    try:
        metrics_data = rag_service.get_performance_metrics()
        return metrics_data
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}"
        )


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics"""
    try:
        prometheus_metrics = metrics.get_prometheus_metrics()
        return Response(content=prometheus_metrics, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}"
        )


@app.post("/cache/clear", dependencies=get_security_dependencies())
async def clear_cache(request: CacheRequest):
    """Clear caches"""
    validate_service()
    
    try:
        if request.cache_type in ["query", "all"]:
            await rag_service.clear_query_cache()
        
        if request.cache_type in ["embedding", "all"]:
            await rag_service.clear_embedding_cache()
        
        return {"message": f"Cache cleared successfully: {request.cache_type}"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics"""
    validate_service()
    
    try:
        stats = await rag_service.get_cache_stats()
        return CacheStatsResponse(
            query_cache=stats.get("query_cache", {}),
            embedding_cache=stats.get("embedding_cache", {}),
            total_cache_size_mb=stats.get("total_size_mb", 0),
            cache_hit_rate=stats.get("hit_rate", 0),
            cache_miss_rate=1 - stats.get("hit_rate", 0),
            time_saved_ms=stats.get("time_saved_ms", 0),
            requests_served_from_cache=stats.get("requests_served", 0)
        )
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting cache stats: {str(e)}"
        )


@app.get("/search/test")
async def test_search():
    """Test search functionality"""
    validate_service()
    
    try:
        # Test with a simple query
        test_response = await rag_service.ask(
            question="test query",
            search_method="semantic",
            max_results=1
        )
        
        return {
            "search_working": True,
            "test_results": len(test_response.sources),
            "confidence": test_response.confidence,
            "processing_time_ms": test_response.processing_time_ms,
            "message": "Search functionality is working"
        }
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        return {
            "search_working": False,
            "error": str(e),
            "message": "Search functionality test failed"
        }


@app.post("/debug", dependencies=get_security_dependencies())
async def debug_query(request: DebugRequest):
    """Debug query processing"""
    validate_service()
    
    if not os.getenv("DEBUG_MODE", "false").lower() == "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug mode not enabled"
        )
    
    try:
        debug_response = await rag_service.debug_query(
            question=request.question,
            debug_level=request.debug_level,
            include_intermediate_results=request.include_intermediate_results,
            include_timing=request.include_timing
        )
        
        return debug_response
        
    except Exception as e:
        logger.error(f"Debug query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug query failed: {str(e)}"
        )


# Background task processing
async def process_batch_background(
    task_id: str, 
    request: BatchQueryRequest, 
    validated_questions: List[str]
):
    """Background task for batch processing"""
    try:
        task_storage[task_id]["status"] = "processing"
        
        results = []
        errors = []
        
        # Process questions with concurrency control
        semaphore = asyncio.Semaphore(request.max_concurrency)
        
        async def process_single_question(question: str, index: int):
            async with semaphore:
                try:
                    if request.enable_agentic:
                        response = await rag_service.ask_with_planning(
                            question=question,
                            enable_iteration=request.enable_iteration,
                            enable_reflection=request.enable_reflection,
                            enable_triangulation=request.enable_triangulation,
                            max_results=request.max_results
                        )
                    else:
                        response = await rag_service.ask(
                            question=question,
                            search_method=request.search_method or "hybrid",
                            max_results=request.max_results
                        )
                    
                    return {
                        "index": index,
                        "question": question,
                        "response": response.dict(),
                        "status": "success"
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing question {index}: {e}")
                    return {
                        "index": index,
                        "question": question,
                        "error": str(e),
                        "status": "error"
                    }
        
        # Process all questions concurrently
        tasks = [
            process_single_question(question, i) 
            for i, question in enumerate(validated_questions)
        ]
        
        # Process with timeout
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=request.timeout_per_question * len(validated_questions)
            )
            
            # Sort results by index and separate successes from errors
            for result in sorted(completed_tasks, key=lambda x: x.get("index", 0)):
                if isinstance(result, Exception):
                    errors.append({
                        "question": "unknown",
                        "error": str(result)
                    })
                elif result.get("status") == "success":
                    results.append({
                        "question": result["question"],
                        "response": result["response"]
                    })
                else:
                    errors.append({
                        "question": result["question"],
                        "error": result["error"]
                    })
                
                # Update progress
                task_storage[task_id]["completed"] = len(results) + len(errors)
                
        except asyncio.TimeoutError:
            logger.error(f"Batch task {task_id} timed out")
            task_storage[task_id]["status"] = "timeout"
            return
        
        # Update final status
        task_storage[task_id].update({
            "status": "completed",
            "results": results,
            "errors": errors,
            "completed_at": time.time()
        })
        
        logger.info(f"Batch task {task_id} completed: {len(results)} successful, {len(errors)} errors")
        
    except Exception as e:
        logger.error(f"Batch task {task_id} failed: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": time.time()
        })


# WebSocket endpoint for streaming responses
from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    """WebSocket endpoint for streaming responses"""
    await websocket.accept()
    
    try:
        while True:
            # Receive question
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            question = request_data.get("question")
            if not question:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Question is required"
                }))
                continue
            
            # Validate question
            try:
                validated_question = validate_query(question)
            except ValueError as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": f"Invalid question: {str(e)}"
                }))
                continue
            
            # Send start message
            await websocket.send_text(json.dumps({
                "type": "start",
                "message": "Processing your question...",
                "question": validated_question
            }))
            
            try:
                validate_service()
                
                # Process with agentic features
                response = await rag_service.ask_with_planning(
                    question=validated_question,
                    enable_iteration=request_data.get("enable_iteration", True),
                    enable_reflection=request_data.get("enable_reflection", True),
                    enable_triangulation=request_data.get("enable_triangulation", True)
                )
                
                # Send reasoning steps
                for step in response.reasoning_chain:
                    await websocket.send_text(json.dumps({
                        "type": "reasoning",
                        "step": step
                    }))
                
                # Send final response
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "data": response.dict()
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "Connection error occurred"
            }))
        except:
            pass


# Startup validation
@app.on_event("startup")
async def startup_validation():
    """Additional startup validation"""
    try:
        # Check required environment variables
        required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise RuntimeError(f"Missing environment variables: {missing_vars}")
        
        logger.info("Startup validation completed successfully")
        
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        raise


if __name__ == "__main__":
    # Configuration for development
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Reload mode requires single worker
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )