"""
Main entry point for the LlamaIndex Ingestion Service
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import get_config
from processors.ingestion_service import SupabaseRAGIngestionService, IngestionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LlamaIndex Ingestion Service",
    description="Comprehensive document ingestion pipeline for RAG systems",
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
ingestion_service: Optional[SupabaseRAGIngestionService] = None

# Request/Response models
class IngestFilesRequest(BaseModel):
    file_paths: List[str]
    batch_size: Optional[int] = None

class IngestDirectoryRequest(BaseModel):
    directory_path: str
    recursive: bool = True
    file_extensions: Optional[List[str]] = None

class IngestionResponse(BaseModel):
    task_id: str
    status: str
    message: str

class IngestionStatusResponse(BaseModel):
    task_id: str
    status: str
    processed: List[str]
    failed: List[str]
    total_documents: int
    total_chunks: int
    processing_time: float
    error: Optional[str] = None

# Task storage (in production, use Redis or database)
task_storage = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the ingestion service on startup"""
    global ingestion_service
    
    try:
        config = get_config()
        ingestion_service = SupabaseRAGIngestionService(config)
        logger.info("Ingestion service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ingestion service: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ingestion service")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Test database connection
        stats = ingestion_service.get_ingestion_stats()
        return {
            "status": "healthy",
            "service": "ingestion",
            "database": "connected",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/ingest/files", response_model=IngestionResponse)
async def ingest_files(
    request: IngestFilesRequest,
    background_tasks: BackgroundTasks
):
    """Ingest multiple files asynchronously"""
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file paths
    missing_files = []
    for file_path in request.file_paths:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise HTTPException(
            status_code=400, 
            detail=f"Files not found: {missing_files}"
        )
    
    # Generate task ID
    task_id = f"ingest_{len(task_storage)}"
    
    # Initialize task status
    task_storage[task_id] = {
        "status": "pending",
        "processed": [],
        "failed": [],
        "total_documents": 0,
        "total_chunks": 0,
        "processing_time": 0.0,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_files_background,
        task_id,
        request.file_paths,
        request.batch_size
    )
    
    return IngestionResponse(
        task_id=task_id,
        status="started",
        message=f"Processing {len(request.file_paths)} files"
    )

@app.post("/ingest/directory", response_model=IngestionResponse)
async def ingest_directory(
    request: IngestDirectoryRequest,
    background_tasks: BackgroundTasks
):
    """Ingest all files in a directory asynchronously"""
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate directory
    directory = Path(request.directory_path)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: {request.directory_path}"
        )
    
    # Generate task ID
    task_id = f"ingest_dir_{len(task_storage)}"
    
    # Initialize task status
    task_storage[task_id] = {
        "status": "pending",
        "processed": [],
        "failed": [],
        "total_documents": 0,
        "total_chunks": 0,
        "processing_time": 0.0,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_directory_background,
        task_id,
        request.directory_path,
        request.recursive,
        request.file_extensions
    )
    
    return IngestionResponse(
        task_id=task_id,
        status="started",
        message=f"Processing directory: {request.directory_path}"
    )

@app.post("/ingest/upload", response_model=IngestionResponse)
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and ingest files directly"""
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Create upload directory
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    file_paths = []
    for file in files:
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_paths.append(str(file_path))
    
    # Generate task ID
    task_id = f"upload_{len(task_storage)}"
    
    # Initialize task status
    task_storage[task_id] = {
        "status": "pending",
        "processed": [],
        "failed": [],
        "total_documents": 0,
        "total_chunks": 0,
        "processing_time": 0.0,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_files_background,
        task_id,
        file_paths,
        None
    )
    
    return IngestionResponse(
        task_id=task_id,
        status="started",
        message=f"Processing {len(files)} uploaded files"
    )

@app.get("/ingest/status/{task_id}", response_model=IngestionStatusResponse)
async def get_ingestion_status(task_id: str):
    """Get the status of an ingestion task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_storage[task_id]
    return IngestionStatusResponse(
        task_id=task_id,
        **task_info
    )

@app.get("/ingest/tasks")
async def list_tasks():
    """List all ingestion tasks"""
    return {
        "tasks": [
            {"task_id": task_id, "status": info["status"]}
            for task_id, info in task_storage.items()
        ]
    }

@app.delete("/ingest/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_storage[task_id]
    if task_info["status"] == "processing":
        raise HTTPException(status_code=400, detail="Cannot delete active task")
    
    del task_storage[task_id]
    return {"message": "Task deleted successfully"}

@app.get("/stats")
async def get_service_stats():
    """Get overall service statistics"""
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        db_stats = ingestion_service.get_ingestion_stats()
        
        # Calculate task statistics
        task_stats = {
            "total_tasks": len(task_storage),
            "completed_tasks": sum(1 for t in task_storage.values() if t["status"] == "completed"),
            "failed_tasks": sum(1 for t in task_storage.values() if t["status"] == "failed"),
            "processing_tasks": sum(1 for t in task_storage.values() if t["status"] == "processing")
        }
        
        return {
            "database_stats": db_stats,
            "task_stats": task_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

async def process_files_background(task_id: str, file_paths: List[str], batch_size: Optional[int]):
    """Background task for processing files"""
    try:
        # Update status
        task_storage[task_id]["status"] = "processing"
        
        # Override batch size if provided
        if batch_size:
            ingestion_service.config.batch_size = batch_size
        
        # Process files
        results = await ingestion_service.ingest_files(file_paths)
        
        # Update task status with results
        task_storage[task_id].update({
            "status": "completed",
            "processed": results["processed"],
            "failed": results["failed"],
            "total_documents": results["total_documents"],
            "total_chunks": results["total_chunks"],
            "processing_time": results["processing_time"]
        })
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e)
        })

async def process_directory_background(
    task_id: str, 
    directory_path: str, 
    recursive: bool, 
    file_extensions: Optional[List[str]]
):
    """Background task for processing directory"""
    try:
        # Update status
        task_storage[task_id]["status"] = "processing"
        
        # Process directory
        results = await ingestion_service.ingest_directory(
            directory_path=directory_path,
            recursive=recursive,
            file_extensions=file_extensions
        )
        
        # Update task status with results
        task_storage[task_id].update({
            "status": "completed",
            "processed": results["processed"],
            "failed": results["failed"], 
            "total_documents": results["total_documents"],
            "total_chunks": results["total_chunks"],
            "processing_time": results["processing_time"]
        })
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e)
        })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )