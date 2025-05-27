"""
Fixed Core Ingestion Service Implementation
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
            'service_start_time': time.time()
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
                'processing_time': 0.0
            }
        
        start_time = time.time()
        processed_files = []
        failed_files = []
        total_documents = 0
        total_chunks = 0
        
        # Process files with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        
        async def process_single_file(file_path: str):
            async with semaphore:
                try:
                    result = await self._process_file(file_path)
                    if result['success']:
                        processed_files.append(file_path)
                        return result['documents'], result['chunks']
                    else:
                        failed_files.append(file_path)
                        logger.warning(f"Failed to process {file_path}: {result.get('error', 'Unknown error')}")
                        return 0, 0
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    failed_files.append(file_path)
                    return 0, 0
        
        # Execute all file processing tasks
        try:
            tasks = [process_single_file(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                    failed_files.append(file_paths[i])
                elif isinstance(result, tuple) and len(result) == 2:
                    docs, chunks = result
                    total_documents += docs
                    total_chunks += chunks
                else:
                    logger.warning(f"Unexpected result format for task {i}: {result}")
                    
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
            'processing_time': processing_time
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
                    
                    # Step 6: Store chunks in Supabase
                    await self._store_chunks(enhanced_chunks, document_id)
                    
                    total_chunks += len(enhanced_chunks)
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_index} in {file_path}: {e}")
                    continue
            
            if total_chunks == 0:
                return {'success': False, 'error': 'No chunks created', 'documents': len(documents), 'chunks': 0}
            
            logger.info(f"Successfully processed {file_path}: {len(documents)} documents, {total_chunks} chunks")
            
            return {
                'success': True,
                'documents': len(documents),
                'chunks': total_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {'success': False, 'error': str(e), 'documents': 0, 'chunks': 0}
    
    async def _store_chunks(self, chunks: List[Document], document_id: str):
        """
        Store document chunks in Supabase with embeddings
        """
        if not chunks:
            return
        
        # Process in batches
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_data = []
            
            for chunk_index, chunk in enumerate(batch):
                try:
                    # Generate embedding
                    embedding = await self._generate_embedding(chunk.text)
                    
                    if not embedding:
                        logger.warning(f"Failed to generate embedding for chunk {i + chunk_index}")
                        continue
                    
                    # Calculate content hash for deduplication
                    content_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
                    
                    # Prepare chunk data
                    chunk_data = {
                        'id': str(uuid.uuid4()),
                        'content': chunk.text,
                        'embedding': embedding,
                        'document_id': document_id,
                        'chunk_id': str(uuid.uuid4()),
                        'file_path': chunk.metadata.get('file_path'),
                        'file_name': chunk.metadata.get('file_name'),
                        'file_type': chunk.metadata.get('file_type'),
                        'file_size': chunk.metadata.get('file_size'),
                        'file_modified': chunk.metadata.get('file_modified'),
                        'chunk_index': i + chunk_index,
                        'total_chunks': len(chunks),
                        'processed_at': chunk.metadata.get('processed_at'),
                        'extraction_method': chunk.metadata.get('extraction_method', 'standard'),
                        'title': chunk.metadata.get('title'),
                        'summary': chunk.metadata.get('summary'),
                        'keywords': chunk.metadata.get('keywords', []),
                        'entities': chunk.metadata.get('entities', []),
                        'word_count': len(chunk.text.split()),
                        'char_count': len(chunk.text),
                        'content_hash': abs(hash(content_hash)) % (10 ** 10)  # Convert to bigint
                    }
                    
                    batch_data.append(chunk_data)
                    
                except Exception as e:
                    logger.error(f"Error preparing chunk data for chunk {i + chunk_index}: {e}")
                    continue
            
            # Insert batch into Supabase
            if batch_data:
                try:
                    result = self.supabase.table(self.config.table_name).insert(batch_data).execute()
                    logger.debug(f"Inserted batch of {len(batch_data)} chunks")
                except Exception as e:
                    logger.error(f"Error inserting batch into Supabase: {e}")
                    # Try inserting one by one for better error handling
                    await self._insert_chunks_individually(batch_data)
    
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
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using OpenAI
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
            # Return zero embedding as fallback
            return [0.0] * 3072  # text-embedding-3-large dimension
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics
        """
        try:
            # Get database stats
            db_result = self.supabase.rpc('get_document_stats').execute()
            db_stats = db_result.data[0] if db_result.data else {}
            
            # Calculate uptime
            uptime_seconds = time.time() - self.stats['service_start_time']
            
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
                'service_status': 'healthy',
                'configuration': {
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'embedding_model': self.config.embedding_model,
                    'llm_model': self.config.llm_model,
                    'max_file_size_mb': self.config.max_file_size_mb,
                    'batch_size': self.config.batch_size,
                    'max_concurrent_files': self.config.max_concurrent_files
                }
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'processing_stats': self.stats,
                'database_stats': {},
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
            await self._generate_embedding("test")
            logger.info("OpenAI connection test successful")
            
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