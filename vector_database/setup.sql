-- Complete setup script for Supabase Vector Database
-- Run this script in your Supabase SQL editor or via psql

-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the main documents table for storing vector embeddings and metadata
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Content and embedding
    content TEXT NOT NULL,
    embedding VECTOR(3072), -- text-embedding-3-large dimension
    
    -- Document metadata
    document_id UUID,
    chunk_id UUID,
    file_path TEXT,
    file_name TEXT,
    file_type TEXT,
    file_size BIGINT,
    file_modified TIMESTAMPTZ,
    
    -- Chunk metadata
    chunk_index INTEGER,
    total_chunks INTEGER,
    parent_node_id TEXT,
    chunk_type TEXT,
    word_count INTEGER,
    char_count INTEGER,
    
    -- Processing metadata
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    extraction_method TEXT,
    
    -- Enhanced metadata for hybrid search
    title TEXT,
    summary TEXT,
    keywords TEXT[],
    entities TEXT[],
    questions_answered TEXT[],
    
    -- Contextual metadata
    previous_chunk_preview TEXT,
    next_chunk_preview TEXT,
    
    -- Search optimization
    content_hash BIGINT,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content || ' ' || COALESCE(title, '') || ' ' || COALESCE(summary, ''))) STORED,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance optimization

-- Vector similarity search index (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS rag_documents_embedding_idx 
ON rag_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX IF NOT EXISTS rag_documents_search_vector_idx 
ON rag_documents USING gin(search_vector);

-- Metadata indexes for filtering
CREATE INDEX IF NOT EXISTS rag_documents_file_type_idx ON rag_documents(file_type);
CREATE INDEX IF NOT EXISTS rag_documents_document_id_idx ON rag_documents(document_id);
CREATE INDEX IF NOT EXISTS rag_documents_chunk_id_idx ON rag_documents(chunk_id);
CREATE INDEX IF NOT EXISTS rag_documents_processed_at_idx ON rag_documents(processed_at);
CREATE INDEX IF NOT EXISTS rag_documents_file_path_idx ON rag_documents(file_path);
CREATE INDEX IF NOT EXISTS rag_documents_content_hash_idx ON rag_documents(content_hash);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS rag_documents_file_type_processed_idx 
ON rag_documents(file_type, processed_at);

CREATE INDEX IF NOT EXISTS rag_documents_doc_chunk_idx 
ON rag_documents(document_id, chunk_index);

-- Create a table for storing BM25 statistics for hybrid search
CREATE TABLE IF NOT EXISTS bm25_stats (
    id SERIAL PRIMARY KEY,
    total_documents INTEGER NOT NULL,
    average_document_length FLOAT NOT NULL,
    term_frequencies JSONB NOT NULL,
    document_frequencies JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Initialize with empty stats
INSERT INTO bm25_stats (total_documents, average_document_length, term_frequencies, document_frequencies)
VALUES (0, 0, '{}', '{}')
ON CONFLICT DO NOTHING;

-- Table to track batch embedding jobs
CREATE TABLE IF NOT EXISTS embedding_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id TEXT NOT NULL UNIQUE,
    document_id UUID,
    status TEXT NOT NULL DEFAULT 'pending',
    chunk_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    metadata JSONB
);

-- Index for finding pending jobs
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON embedding_batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_document ON embedding_batch_jobs(document_id);

-- Add a column to track chunks waiting for embeddings
ALTER TABLE rag_documents 
ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'complete';

-- Index for finding chunks without embeddings
CREATE INDEX IF NOT EXISTS idx_documents_embedding_status ON rag_documents(embedding_status)
WHERE embedding_status != 'complete';