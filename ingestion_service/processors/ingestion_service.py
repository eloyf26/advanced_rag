"""
Enhanced Core Ingestion Service Implementation with Hybrid Embedding Support
File: ingestion_service/processors/ingestion_service.py
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import json
import os
import tempfile

import aiofiles
from supabase import create_client, Client
import openai
from llama_index.core import Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import IngestionConfig
from .context_aware_chunker import ContextAwareChunker
from .multimodal_processor import MultiModalFileProcessor
from .metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class SupabaseRAGIngestionService:
    """
    Main ingestion service for processing and storing documents in Supabase
    Now with hybrid embedding support (Regular API + Batch API)
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        # Validate configuration
        if not config.supabase_url or not config.supabase_key:
            raise ValueError("Missing required Supabase configuration (SUPABASE_URL and SUPABASE_SERVICE_KEY)")
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(config.supabase_url, config.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
        
        # Initialize OpenAI client
        try:
            openai_api_key = config.__dict__.get('openai_api_key') or openai.api_key
            if not openai_api_key:
                # Try to get from environment
                import os
                openai_api_key = os.getenv('OPENAI_API_KEY')
                if not openai_api_key:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Configure LlamaIndex settings
        try:
            Settings.llm = OpenAI(model=config.llm_model, api_key=openai_api_key)
            Settings.embed_model = OpenAIEmbedding(model=config.embedding_model, api_key=openai_api_key)
            logger.info("LlamaIndex settings configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure LlamaIndex settings: {e}")
            raise
        
        # Initialize processors
        try:
            self.chunker = ContextAwareChunker(config)
            self.file_processor = MultiModalFileProcessor(config)
            self.metadata_extractor = MetadataExtractor(config)
            logger.info("Processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise
        
        # Performance tracking
        self.stats = {
            'total_files_processed': 0,
            'total_documents_created': 0,
            'total_chunks_created': 0,
            'total_processing_time': 0.0,
            'files_failed': 0,
            'service_start_time': time.time(),
            'embeddings_via_regular_api': 0,
            'embeddings_via_batch_api': 0,
            'batch_api_savings': 0.0
        }
        
        logger.info("Supabase RAG Ingestion Service initialized successfully")
    
    async def ingest_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest multiple files and store in Supabase
        """
        if not file_paths:
            return {
                'processed': [],
                'failed': [],
                'total_documents': 0,
                'total_chunks': 0,
                'processing_time': 0.0,
                'embedding_method': 'none'
            }
        
        start_time = time.time()
        processed_files = []
        failed_files = []
        total_documents = 0
        total_chunks = 0
        embedding_method = 'none'
        
        # Process files with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        
        async def process_single_file(file_path: str):
            async with semaphore:
                try:
                    result = await self._process_file(file_path)
                    if result['success']:
                        processed_files.append(file_path)
                        return result['documents'], result['chunks'], result.get('embedding_method', 'regular')
                    else:
                        failed_files.append(file_path)
                        logger.warning(f"Failed to process {file_path}: {result.get('error', 'Unknown error')}")
                        return 0, 0, 'none'
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    failed_files.append(file_path)
                    return 0, 0, 'none'
        
        # Execute all file processing tasks
        try:
            tasks = [process_single_file(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            batch_api_used = False
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                    failed_files.append(file_paths[i])
                elif isinstance(result, tuple) and len(result) == 3:
                    docs, chunks, method = result
                    total_documents += docs
                    total_chunks += chunks
                    if method == 'batch':
                        batch_api_used = True
                else:
                    logger.warning(f"Unexpected result format for task {i}: {result}")
            
            # Determine overall embedding method
            if batch_api_used:
                embedding_method = 'batch'
            elif total_chunks > 0:
                embedding_method = 'regular'
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.stats['total_files_processed'] += len(processed_files)
        self.stats['total_documents_created'] += total_documents
        self.stats['total_chunks_created'] += total_chunks
        self.stats['total_processing_time'] += processing_time
        self.stats['files_failed'] += len(failed_files)
        
        logger.info(f"Batch processing completed: {len(processed_files)} processed, {len(failed_files)} failed, {processing_time:.2f}s")
        
        return {
            'processed': processed_files,
            'failed': failed_files,
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'processing_time': processing_time,
            'embedding_method': embedding_method
        }
    
    async def ingest_directory(
        self, 
        directory_path: str, 
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest all files in a directory
        """
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Collect files
        file_paths = []
        pattern = "**/*" if recursive else "*"
        
        try:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    # Check file extension filter
                    if file_extensions is None or file_path.suffix.lower().lstrip('.') in [ext.lower() for ext in file_extensions]:
                        # Check file size
                        if file_path.stat().st_size <= self.config.max_file_size_mb * 1024 * 1024:
                            file_paths.append(str(file_path))
                        else:
                            logger.warning(f"Skipping {file_path}: exceeds size limit")
        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {e}")
            raise
        
        logger.info(f"Found {len(file_paths)} files to process in {directory_path}")
        
        # Process collected files
        return await self.ingest_files(file_paths)
    
    async def _process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline
        """
        try:
            file_path_obj = Path(file_path)
            
            # Validate file exists and is accessible
            if not file_path_obj.exists():
                return {'success': False, 'error': 'File not found', 'documents': 0, 'chunks': 0}
            
            if not file_path_obj.is_file():
                return {'success': False, 'error': 'Path is not a file', 'documents': 0, 'chunks': 0}
            
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Load and process file content
            documents = await self.file_processor.process_file(file_path)
            
            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return {'success': False, 'error': 'No content extracted', 'documents': 0, 'chunks': 0}
            
            # Step 2: Create document metadata
            document_id = str(uuid.uuid4())
            file_metadata = {
                'document_id': document_id,
                'file_path': str(file_path),
                'file_name': file_path_obj.name,
                'file_type': file_path_obj.suffix.lower().lstrip('.'),
                'file_size': file_path_obj.stat().st_size,
                'file_modified': datetime.fromtimestamp(file_path_obj.stat().st_mtime),
                'processed_at': datetime.utcnow()
            }
            
            # Step 3: Process each document
            total_chunks = 0
            all_chunks = []  # Collect all chunks for potential batch processing
            
            for doc_index, document in enumerate(documents):
                try:
                    # Add file metadata to document
                    document.metadata.update(file_metadata)
                    document.metadata['doc_index'] = doc_index
                    
                    # Step 4: Chunk the document
                    chunks = await self.chunker.chunk_document(document)
                    
                    if not chunks:
                        logger.warning(f"No chunks created for document {doc_index} in {file_path}")
                        continue
                    
                    # Step 5: Extract metadata for each chunk
                    enhanced_chunks = await self.metadata_extractor.extract_metadata(chunks)
                    
                    all_chunks.extend(enhanced_chunks)
                    total_chunks += len(enhanced_chunks)
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_index} in {file_path}: {e}")
                    continue
            
            if total_chunks == 0:
                return {'success': False, 'error': 'No chunks created', 'documents': len(documents), 'chunks': 0}
            
            # Step 6: Store all chunks with hybrid embedding strategy
            embedding_method = await self._store_chunks(all_chunks, document_id)
            
            logger.info(f"Successfully processed {file_path}: {len(documents)} documents, {total_chunks} chunks")
            
            return {
                'success': True,
                'documents': len(documents),
                'chunks': total_chunks,
                'embedding_method': embedding_method
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {'success': False, 'error': str(e), 'documents': 0, 'chunks': 0}
    
    async def _store_chunks(self, chunks: List[Document], document_id: str) -> str:
        """
        Store document chunks with hybrid embedding strategy
        Returns the embedding method used: 'regular', 'batch', or 'none'
        """
        if not chunks:
            return 'none'
        
        total_chunks = len(chunks)
        logger.info(f"Storing {total_chunks} chunks for document {document_id}")
        
        # Decide strategy based on chunk count and configuration
        use_batch_api = (
            hasattr(self.config, 'use_batch_api') and 
            self.config.use_batch_api and 
            total_chunks >= getattr(self.config, 'batch_api_threshold', 100)
        )
        
        if use_batch_api:
            logger.info(f"Using Batch API for {total_chunks} chunks (50% cost savings)")
            await self._store_chunks_batch_api(chunks, document_id)
            return 'batch'
        else:
            logger.info(f"Using regular API for {total_chunks} chunks (immediate processing)")
            await self._store_chunks_regular_api(chunks, document_id)
            return 'regular'
    
    async def _store_chunks_regular_api(self, chunks: List[Document], document_id: str):
        """
        Store chunks using regular API (immediate embeddings)
        """
        batch_size = self.config.batch_size
        max_regular_batch = getattr(self.config, 'max_regular_api_batch', 20)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_data = []
            
            # Generate embeddings in parallel batches
            embeddings = await self._generate_embeddings_batch([chunk.text for chunk in batch])
            
            for chunk_index, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {i + chunk_index}")
                    continue
                
                chunk_data = self._prepare_chunk_data(chunk, embedding, document_id, i + chunk_index, len(chunks))
                batch_data.append(chunk_data)
            
            # Insert batch into Supabase
            if batch_data:
                try:
                    result = self.supabase.table(self.config.table_name).insert(batch_data).execute()
                    logger.debug(f"Inserted batch of {len(batch_data)} chunks")
                    self.stats['embeddings_via_regular_api'] += len(batch_data)
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    await self._insert_chunks_individually(batch_data)
    
    async def _store_chunks_batch_api(self, chunks: List[Document], document_id: str):
        """
        Store chunks using Batch API (delayed embeddings)
        """
        # First, store chunks without embeddings
        batch_size = self.config.batch_size
        chunk_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_data = []
            
            for chunk_index, chunk in enumerate(batch):
                # Prepare chunk without embedding
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                chunk_data = self._prepare_chunk_data(
                    chunk, 
                    None,  # No embedding yet
                    document_id, 
                    i + chunk_index, 
                    len(chunks)
                )
                chunk_data['id'] = chunk_id
                chunk_data['embedding_status'] = 'pending'
                # Store empty embedding vector
                chunk_data['embedding'] = [0.0] * 3072  # Placeholder for text-embedding-3-large
                
                batch_data.append(chunk_data)
            
            # Insert batch
            if batch_data:
                try:
                    result = self.supabase.table(self.config.table_name).insert(batch_data).execute()
                    logger.debug(f"Inserted {len(batch_data)} chunks without embeddings")
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    continue
        
        # Create batch embedding job
        try:
            batch_id = await self._create_batch_embedding_job(chunks, chunk_ids, document_id)
            logger.info(f"Created batch job {batch_id} for {len(chunks)} embeddings")
            self.stats['embeddings_via_batch_api'] += len(chunks)
            
            # Estimate cost savings
            tokens_estimate = sum(len(chunk.text.split()) * 1.3 for chunk in chunks)  # Rough token estimate
            regular_cost = (tokens_estimate / 1000) * 0.00013
            batch_cost = (tokens_estimate / 1000) * 0.000065
            savings = regular_cost - batch_cost
            self.stats['batch_api_savings'] += savings
            
        except Exception as e:
            logger.error(f"Failed to create batch job: {e}")
            # Fall back to regular API
            logger.info("Falling back to regular API for embeddings")
            await self._update_chunks_with_embeddings(chunks, chunk_ids)
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts using regular API with batching
        """
        embeddings = []
        batch_size = getattr(self.config, 'max_regular_api_batch', 20)  # OpenAI allows up to 2048
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Truncate texts if needed
                truncated_texts = [text[:8000] for text in batch_texts]
                
                response = await asyncio.to_thread(
                    self.openai_client.embeddings.create,
                    input=truncated_texts,
                    model=self.config.embedding_model
                )
                
                for data in response.data:
                    embeddings.append(data.embedding)
            
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch_texts))
        
        return embeddings
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using OpenAI (single text)
        """
        if not text or not text.strip():
            return None
        
        try:
            # Truncate text if too long (OpenAI has token limits)
            max_chars = 8000  # Conservative limit
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.debug(f"Truncated text to {max_chars} characters for embedding")
            
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=text,
                model=self.config.embedding_model
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                logger.error("No embedding data received from OpenAI")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return None instead of zero embedding
            return None
    
    async def _create_batch_embedding_job(self, chunks: List[Document], chunk_ids: List[str], document_id: str) -> str:
        """
        Create OpenAI Batch API job for embeddings
        """
        # Create JSONL file for batch
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for idx, chunk in enumerate(chunks):
                request = {
                    "custom_id": chunk_ids[idx],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.config.embedding_model,
                        "input": chunk.text[:8000]  # Truncate if needed
                    }
                }
                f.write(json.dumps(request) + '\n')
            
            batch_file_path = f.name
        
        try:
            # Upload file to OpenAI
            with open(batch_file_path, 'rb') as f:
                batch_file = await asyncio.to_thread(
                    self.openai_client.files.create,
                    file=f,
                    purpose="batch"
                )
            
            # Create batch
            batch = await asyncio.to_thread(
                self.openai_client.batches.create,
                input_file_id=batch_file.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "document_id": document_id,
                    "chunk_count": str(len(chunks))
                }
            )
            
            # Store batch job info in database
            job_data = {
                "batch_id": batch.id,
                "document_id": document_id,
                "status": "pending",
                "chunk_count": len(chunks),
                "metadata": {
                    "chunk_ids": chunk_ids,
                    "file_id": batch_file.id
                }
            }
            
            self.supabase.table("embedding_batch_jobs").insert(job_data).execute()
            
            return batch.id
            
        finally:
            # Clean up temp file
            os.unlink(batch_file_path)
    
    async def _update_chunks_with_embeddings(self, chunks: List[Document], chunk_ids: List[str]):
        """
        Fallback method to update chunks with embeddings using regular API
        """
        embeddings = await self._generate_embeddings_batch([chunk.text for chunk in chunks])
        
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            if embedding:
                try:
                    self.supabase.table(self.config.table_name).update({
                        "embedding": embedding,
                        "embedding_status": "complete"
                    }).eq("id", chunk_id).execute()
                except Exception as e:
                    logger.error(f"Failed to update chunk {chunk_id}: {e}")
    
    async def process_pending_batches(self) -> Dict[str, Any]:
        """
        Check and process completed batch embedding jobs
        """
        # Get pending jobs
        result = self.supabase.table("embedding_batch_jobs").select("*").eq("status", "pending").execute()
        pending_jobs = result.data
        
        processed = 0
        failed = 0
        
        for job in pending_jobs:
            try:
                # Check batch status with OpenAI
                batch = await asyncio.to_thread(
                    self.openai_client.batches.retrieve,
                    job['batch_id']
                )
                
                if batch.status == "completed":
                    logger.info(f"Processing completed batch {job['batch_id']}")
                    await self._process_completed_batch(job, batch)
                    processed += 1
                    
                elif batch.status == "failed":
                    logger.error(f"Batch {job['batch_id']} failed: {batch.errors}")
                    self.supabase.table("embedding_batch_jobs").update({
                        "status": "failed",
                        "error": str(batch.errors)
                    }).eq("batch_id", job['batch_id']).execute()
                    failed += 1
                    
                elif batch.status == "expired":
                    logger.warning(f"Batch {job['batch_id']} expired")
                    self.supabase.table("embedding_batch_jobs").update({
                        "status": "expired"
                    }).eq("batch_id", job['batch_id']).execute()
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing batch {job['batch_id']}: {e}")
                failed += 1
        
        return {
            "pending": len(pending_jobs),
            "processed": processed,
            "failed": failed
        }
    
    async def _process_completed_batch(self, job: Dict, batch: Any):
        """
        Process a completed batch and update embeddings
        """
        # Get result file
        result_file_response = await asyncio.to_thread(
            self.openai_client.files.content,
            batch.output_file_id
        )
        
        # Parse results
        embeddings_map = {}
        for line in result_file_response.text.strip().split('\n'):
            result = json.loads(line)
            chunk_id = result['custom_id']
            
            if result['response']['status_code'] == 200:
                embedding = result['response']['body']['data'][0]['embedding']
                embeddings_map[chunk_id] = embedding
            else:
                logger.error(f"Failed embedding for chunk {chunk_id}: {result['response']}")
        
        # Update chunks with embeddings
        chunk_ids = job['metadata']['chunk_ids']
        for chunk_id in chunk_ids:
            if chunk_id in embeddings_map:
                try:
                    self.supabase.table(self.config.table_name).update({
                        "embedding": embeddings_map[chunk_id],
                        "embedding_status": "complete"
                    }).eq("id", chunk_id).execute()
                except Exception as e:
                    logger.error(f"Failed to update chunk {chunk_id}: {e}")
        
        # Update batch job status
        self.supabase.table("embedding_batch_jobs").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }).eq("batch_id", job['batch_id']).execute()
        
        logger.info(f"Updated {len(embeddings_map)} embeddings from batch {job['batch_id']}")
    
    def _prepare_chunk_data(self, chunk: Document, embedding: Optional[List[float]], 
                           document_id: str, chunk_index: int, total_chunks: int) -> Dict:
        """
        Prepare chunk data for database insertion
        """
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
        
        return {
            'content': chunk.text,
            'embedding': embedding,
            'document_id': document_id,
            'chunk_id': str(uuid.uuid4()),
            'file_path': chunk.metadata.get('file_path'),
            'file_name': chunk.metadata.get('file_name'),
            'file_type': chunk.metadata.get('file_type'),
            'file_size': chunk.metadata.get('file_size'),
            'file_modified': chunk.metadata.get('file_modified'),
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'parent_node_id': chunk.metadata.get('parent_node_id'),
            'chunk_type': chunk.metadata.get('chunk_type', 'standard'),
            'word_count': chunk.metadata.get('word_count', len(chunk.text.split())),
            'char_count': chunk.metadata.get('char_count', len(chunk.text)),
            'processed_at': chunk.metadata.get('processed_at'),
            'extraction_method': chunk.metadata.get('extraction_method', 'standard'),
            'title': chunk.metadata.get('title'),
            'summary': chunk.metadata.get('summary'),
            'keywords': chunk.metadata.get('keywords', []),
            'entities': chunk.metadata.get('entities', []),
            'questions_answered': chunk.metadata.get('questions_answered', []),
            'previous_chunk_preview': chunk.metadata.get('previous_chunk_preview'),
            'next_chunk_preview': chunk.metadata.get('next_chunk_preview'),
            'content_hash': abs(hash(content_hash)) % (10 ** 10)  # Convert to bigint
        }
    
    async def _insert_chunks_individually(self, chunk_data_list: List[Dict]):
        """
        Insert chunks one by one when batch insert fails
        """
        success_count = 0
        for i, chunk_data in enumerate(chunk_data_list):
            try:
                self.supabase.table(self.config.table_name).insert(chunk_data).execute()
                success_count += 1
            except Exception as e:
                logger.error(f"Error inserting individual chunk {i}: {e}")
                continue
        
        logger.info(f"Individual insertion completed: {success_count}/{len(chunk_data_list)} chunks inserted")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics including batch API usage
        """
        try:
            # Get database stats
            db_result = self.supabase.rpc('get_document_stats').execute()
            db_stats = db_result.data[0] if db_result.data else {}
            
            # Calculate uptime
            uptime_seconds = time.time() - self.stats['service_start_time']
            
            # Get batch API stats
            batch_stats = {}
            if hasattr(self.config, 'use_batch_api') and self.config.use_batch_api:
                batch_result = self.supabase.table("embedding_batch_jobs").select("status").execute()
                status_counts = {}
                for job in batch_result.data:
                    status = job['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                batch_stats = {
                    'batch_jobs': status_counts,
                    'embeddings_via_regular_api': self.stats['embeddings_via_regular_api'],
                    'embeddings_via_batch_api': self.stats['embeddings_via_batch_api'],
                    'estimated_savings_usd': round(self.stats['batch_api_savings'], 2)
                }
            
            return {
                'processing_stats': {
                    **self.stats,
                    'uptime_seconds': uptime_seconds,
                    'avg_processing_time_per_file': (
                        self.stats['total_processing_time'] / 
                        max(self.stats['total_files_processed'], 1)
                    ),
                    'success_rate': (
                        self.stats['total_files_processed'] / 
                        max(self.stats['total_files_processed'] + self.stats['files_failed'], 1)
                    )
                },
                'database_stats': db_stats,
                'batch_api_stats': batch_stats,
                'service_status': 'healthy',
                'configuration': {
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'embedding_model': self.config.embedding_model,
                    'llm_model': self.config.llm_model,
                    'max_file_size_mb': self.config.max_file_size_mb,
                    'batch_size': self.config.batch_size,
                    'max_concurrent_files': self.config.max_concurrent_files,
                    'use_batch_api': getattr(self.config, 'use_batch_api', False),
                    'batch_api_threshold': getattr(self.config, 'batch_api_threshold', 100)
                }
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'processing_stats': self.stats,
                'database_stats': {},
                'batch_api_stats': {},
                'service_status': 'error',
                'error': str(e),
                'configuration': {}
            }
    
    async def test_connection(self) -> bool:
        """
        Test connections to external services
        """
        try:
            # Test Supabase connection
            result = self.supabase.table(self.config.table_name).select("count").limit(1).execute()
            logger.info("Supabase connection test successful")
            
            # Test OpenAI connection
            test_embedding = await self._generate_embedding("test")
            if test_embedding:
                logger.info("OpenAI connection test successful")
            else:
                logger.warning("OpenAI connection test returned no embedding")
            
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def cleanup_resources(self):
        """
        Cleanup resources on shutdown
        """
        try:
            # Close any open connections if needed
            logger.info("Cleaning up ingestion service resources")
            # Add any cleanup logic here
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about available processors
        """
        return {
            'file_processor': {
                'supported_types': self.file_processor.get_supported_types(),
                'stats': self.file_processor.get_processor_stats()
            },
            'chunker': {
                'stats': self.chunker.get_chunking_stats()
            },
            'metadata_extractor': {
                'stats': self.metadata_extractor.get_extraction_stats()
            }
        }