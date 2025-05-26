"""
Main entry point for the PydanticAI Agentic RAG Agent
"""

import asyncio
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import get_config
from agents.agentic_rag_service import AgenticRAGService
from models.request_models import QueryRequest, BatchQueryRequest
from models.response_models import AgenticRAGResponse, BatchQueryResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="PydanticAI Agentic RAG Agent",
    description="Advanced RAG system with planning, reflection, and iterative search",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
rag_service: Optional[AgenticRAGService] = None

# Task storage for background processing
task_storage = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup"""
    global rag_service
    
    try:
        config = get_config()
        rag_service = AgenticRAGService(config)
        logger.info("Agentic RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down agentic RAG service")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Test database connection
        health_status = await rag_service.get_health_status()
        return HealthResponse(
            status="healthy",
            service="agentic_rag",
            database_connected=health_status["database_connected"],
            embedding_service=health_status["embedding_service"],
            llm_service=health_status["llm_service"],
            details=health_status
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/ask", response_model=AgenticRAGResponse)
async def ask_question(request: QueryRequest):
    """Ask a question with full agentic capabilities"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        response = await rag_service.ask_with_planning(
            question=request.question,
            enable_iteration=request.enable_iteration,
            enable_reflection=request.enable_reflection,
            enable_triangulation=request.enable_triangulation,
            max_results=request.max_results,
            file_types=request.file_types
        )
        return response
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask/simple")
async def ask_simple(request: QueryRequest):
    """Ask a question with simplified RAG (no agentic features)"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        response = await rag_service.ask(
            question=request.question,
            search_method=request.search_method or "hybrid",
            file_types=request.file_types,
            max_results=request.max_results
        )
        return response
    except Exception as e:
        logger.error(f"Error processing simple question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask/batch", response_model=BatchQueryResponse)
async def ask_batch(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks
):
    """Process multiple questions in batch"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Generate task ID
    task_id = f"batch_{len(task_storage)}"
    
    # Initialize task status
    task_storage[task_id] = {
        "status": "pending",
        "total_questions": len(request.questions),
        "completed": 0,
        "results": [],
        "errors": []
    }
    
    # Start background processing
    background_tasks.add_task(
        process_batch_background,
        task_id,
        request
    )
    
    return BatchQueryResponse(
        task_id=task_id,
        status="started",
        total_questions=len(request.questions),
        message=f"Processing {len(request.questions)} questions"
    )

@app.get("/ask/batch/{task_id}")
async def get_batch_status(task_id: str):
    """Get the status of a batch processing task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_storage[task_id]

@app.post("/analyze/query")
async def analyze_query(question: str):
    """Analyze a query without executing it"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        analysis = await rag_service.analyze_query(question)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing query: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        metrics = rag_service.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        await rag_service.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.get("/search/test")
async def test_search():
    """Test search functionality"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
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
            "message": "Search functionality is working"
        }
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        return {
            "search_working": False,
            "error": str(e),
            "message": "Search functionality test failed"
        }

async def process_batch_background(task_id: str, request: BatchQueryRequest):
    """Background task for batch processing"""
    try:
        task_storage[task_id]["status"] = "processing"
        
        results = []
        errors = []
        
        for i, question in enumerate(request.questions):
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
                
                results.append({
                    "question": question,
                    "response": response.dict()
                })
                
            except Exception as e:
                errors.append({
                    "question": question,
                    "error": str(e)
                })
            
            # Update progress
            task_storage[task_id]["completed"] = i + 1
        
        # Update final status
        task_storage[task_id].update({
            "status": "completed",
            "results": results,
            "errors": errors
        })
        
        logger.info(f"Batch task {task_id} completed: {len(results)} successful, {len(errors)} errors")
        
    except Exception as e:
        logger.error(f"Batch task {task_id} failed: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e)
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
                    "error": "Question is required"
                }))
                continue
            
            # Send start message
            await websocket.send_text(json.dumps({
                "type": "start",
                "message": "Processing your question..."
            }))
            
            try:
                # Process with agentic features
                response = await rag_service.ask_with_planning(
                    question=question,
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )