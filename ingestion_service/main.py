"""
Main entry point for the LlamaIndex Ingestion Service
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import get_config, IngestionConfig, validate_environment
from processors.ingestion_service import SupabaseRAGIngestionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingestion_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Global service instance
ingestion_service: Optional[SupabaseRAGIngestionService] = None

# Request/Response models
class IngestFilesRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    batch_size: Optional[int] = Field(None, description="Override batch size for processing")
    
    class Config:
        schema_extra = {
            "example": {
                "file_paths": ["/path/to/document1.pdf", "/path/to/document2.docx"],
                "batch_size": 50
            }
        }

class IngestDirectoryRequest(BaseModel):
    directory_path: str = Field(..., description="Path to directory to ingest")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    file_extensions: Optional[List[str]] = Field(None, description="Filter by file extensions")
    
    class Config:
        schema_extra = {
            "example": {
                "directory_path": "/path/to/documents",
                "recursive": True,
                "file_extensions": ["pdf", "docx", "txt", "md"]
            }
        }

class IngestionResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Human-readable message")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "ingest_123",
                "status": "started",
                "message": "Processing 5 files"
            }
        }

class IngestionStatusResponse(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current status")
    processed: List[str] = Field(default_factory=list, description="Successfully processed files")
    failed: List[str] = Field(default_factory=list, description="Failed file paths")
    total_documents: int = Field(0, description="Total documents created")
    total_chunks: int = Field(0, description="Total chunks generated")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "ingest_123",
                "status": "completed",
                "processed": ["/path/to/doc1.pdf", "/path/to/doc2.docx"],
                "failed": [],
                "total_documents": 2,
                "total_chunks": 45,
                "processing_time": 12.5,
                "error": None
            }
        }

class ServiceStatsResponse(BaseModel):
    database_stats: Dict[str, Any] = Field(default_factory=dict)
    task_stats: Dict[str, Any] = Field(default_factory=dict)
    service_info: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    service: str
    database: str
    stats: Dict[str, Any]
    timestamp: str

# Task storage (in production, use Redis or database)
task_storage: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ingestion_service
    
    try:
        # Startup
        logger.info("Starting Ingestion Service...")
        
        # Validate environment first
        env_issues = validate_environment()
        if env_issues:
            logger.error("Environment validation failed:")
            for issue in env_issues:
                logger.error(f"  - {issue}")
            raise ValueError("Invalid environment configuration. Check logs for details.")
        
        config = get_config()
        
        # Validate configuration
        if not config.supabase_url or not config.supabase_key:
            raise ValueError("Missing required Supabase configuration")
        
        # Initialize service
        ingestion_service = SupabaseRAGIngestionService(config)
        logger.info("Ingestion service initialized successfully")
        
        # Test database connection
        try:
            connected = await ingestion_service.test_connection()
            if connected:
                logger.info("Database connection verified")
            else:
                logger.warning("Database connection test failed")
        except Exception as e:
            logger.warning(f"Database connection test error: {e}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize ingestion service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ingestion service")
        if ingestion_service:
            await ingestion_service.cleanup_resources()

# Create FastAPI app with lifespan
app = FastAPI(
    title="LlamaIndex Ingestion Service",
    description="Comprehensive document ingestion pipeline for RAG systems with multi-modal support",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for service availability
async def get_service() -> SupabaseRAGIngestionService:
    """Get the ingestion service instance"""
    if ingestion_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Ingestion service not initialized. Check configuration and database connection."
        )
    return ingestion_service

@app.get("/health", response_model=HealthResponse)
async def health_check(service: SupabaseRAGIngestionService = Depends(get_service)):
    """Comprehensive health check endpoint"""
    try:
        # Test database connection and get stats
        stats = service.get_ingestion_stats()
        
        return HealthResponse(
            status="healthy",
            service="ingestion",
            database="connected",
            stats=stats,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/ingest/files", response_model=IngestionResponse)
async def ingest_files(
    request: IngestFilesRequest,
    background_tasks: BackgroundTasks,
    service: SupabaseRAGIngestionService = Depends(get_service)
):
    """Ingest multiple files asynchronously"""
    try:
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
        task_id = f"ingest_files_{len(task_storage)}"
        
        # Initialize task status
        task_storage[task_id] = {
            "status": "pending",
            "processed": [],
            "failed": [],
            "total_documents": 0,
            "total_chunks": 0,
            "processing_time": 0.0,
            "error": None,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_files_background,
            task_id,
            request.file_paths,
            request.batch_size,
            service
        )
        
        return IngestionResponse(
            task_id=task_id,
            status="started",
            message=f"Processing {len(request.file_paths)} files"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting file ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/directory", response_model=IngestionResponse)
async def ingest_directory(
    request: IngestDirectoryRequest,
    background_tasks: BackgroundTasks,
    service: SupabaseRAGIngestionService = Depends(get_service)
):
    """Ingest all files in a directory asynchronously"""
    try:
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
            "error": None,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_directory_background,
            task_id,
            request.directory_path,
            request.recursive,
            request.file_extensions,
            service
        )
        
        return IngestionResponse(
            task_id=task_id,
            status="started",
            message=f"Processing directory: {request.directory_path}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting directory ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload", response_model=IngestionResponse)
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    service: SupabaseRAGIngestionService = Depends(get_service)
):
    """Upload and ingest files directly"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create upload directory
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        file_paths = []
        for file in files:
            if not file.filename:
                continue
                
            # Validate file size
            file_size = 0
            content = await file.read()
            file_size = len(content)
            
            if file_size > service.config.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds maximum size limit"
                )
            
            # Save file
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(content)
            file_paths.append(str(file_path))
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
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
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "uploaded_files": [f.filename for f in files if f.filename]
        }
        
        # Start background processing
        background_tasks.add_task(
            process_files_background,
            task_id,
            file_paths,
            None,
            service
        )
        
        return IngestionResponse(
            task_id=task_id,
            status="started",
            message=f"Processing {len(files)} uploaded files"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload and ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest/status/{task_id}", response_model=IngestionStatusResponse)
async def get_ingestion_status(task_id: str):
    """Get the status of an ingestion task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = task_storage[task_id]
    return IngestionStatusResponse(
        task_id=task_id,
        **{k: v for k, v in task_info.items() if k != 'created_at'}
    )

@app.get("/ingest/tasks")
async def list_tasks():
    """List all ingestion tasks with their current status"""
    return {
        "total_tasks": len(task_storage),
        "tasks": [
            {
                "task_id": task_id, 
                "status": info["status"],
                "created_at": info.get("created_at", ""),
                "processing_time": info.get("processing_time", 0.0)
            }
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

@app.get("/stats", response_model=ServiceStatsResponse)
async def get_service_stats(service: SupabaseRAGIngestionService = Depends(get_service)):
    """Get comprehensive service statistics"""
    try:
        db_stats = service.get_ingestion_stats()
        
        # Calculate task statistics
        task_stats = {
            "total_tasks": len(task_storage),
            "completed_tasks": sum(1 for t in task_storage.values() if t["status"] == "completed"),
            "failed_tasks": sum(1 for t in task_storage.values() if t["status"] == "failed"),
            "processing_tasks": sum(1 for t in task_storage.values() if t["status"] == "processing"),
            "pending_tasks": sum(1 for t in task_storage.values() if t["status"] == "pending")
        }
        
        # Service information
        service_info = {
            "supported_file_types": service.file_processor.get_supported_types(),
            "processor_info": service.get_processor_info(),
            "configuration": {
                "chunk_size": service.config.chunk_size,
                "chunk_overlap": service.config.chunk_overlap,
                "max_file_size_mb": service.config.max_file_size_mb,
                "batch_size": service.config.batch_size,
                "enable_ocr": service.config.enable_ocr,
                "enable_speech_to_text": service.config.enable_speech_to_text
            }
        }
        
        return ServiceStatsResponse(
            database_stats=db_stats,
            task_stats=task_stats,
            service_info=service_info
        )
    except Exception as e:
        logger.error(f"Error getting service stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get current service configuration"""
    config = get_config()
    return config.get_summary()

# Background task functions
async def process_files_background(
    task_id: str, 
    file_paths: List[str], 
    batch_size: Optional[int],
    service: SupabaseRAGIngestionService
):
    """Background task for processing files"""
    import time
    start_time = time.time()
    
    try:
        # Update status
        task_storage[task_id]["status"] = "processing"
        
        # Override batch size if provided
        original_batch_size = None
        if batch_size and batch_size > 0:
            original_batch_size = service.config.batch_size
            service.config.batch_size = batch_size
        
        # Process files
        results = await service.ingest_files(file_paths)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update task status with results
        task_storage[task_id].update({
            "status": "completed",
            "processed": results["processed"],
            "failed": results["failed"],
            "total_documents": results["total_documents"],
            "total_chunks": results["total_chunks"],
            "processing_time": processing_time
        })
        
        logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
        
        # Restore original batch size if changed
        if original_batch_size is not None:
            service.config.batch_size = original_batch_size
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Task {task_id} failed after {processing_time:.2f}s: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e),
            "processing_time": processing_time
        })

async def process_directory_background(
    task_id: str, 
    directory_path: str, 
    recursive: bool, 
    file_extensions: Optional[List[str]],
    service: SupabaseRAGIngestionService
):
    """Background task for processing directory"""
    import time
    start_time = time.time()
    
    try:
        # Update status
        task_storage[task_id]["status"] = "processing"
        
        # Process directory
        results = await service.ingest_directory(
            directory_path=directory_path,
            recursive=recursive,
            file_extensions=file_extensions
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update task status with results
        task_storage[task_id].update({
            "status": "completed",
            "processed": results["processed"],
            "failed": results["failed"], 
            "total_documents": results["total_documents"],
            "total_chunks": results["total_chunks"],
            "processing_time": processing_time
        })
        
        logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Task {task_id} failed after {processing_time:.2f}s: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "error": str(e),
            "processing_time": processing_time
        })

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting Ingestion Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )