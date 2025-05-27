-- Vector Database Search Functions
-- Implementation of hybrid, semantic, and keyword search functions
-- Run this after setup.sql to complete the database functionality

-- =============================================================================
-- HYBRID SEARCH FUNCTION
-- Combines vector similarity search with BM25 keyword scoring
-- =============================================================================

CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(3072),
    query_text TEXT,
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_count INTEGER DEFAULT 10,
    file_types TEXT[] DEFAULT NULL,
    date_filter TIMESTAMPTZ DEFAULT NULL,
    vector_weight FLOAT DEFAULT 0.7,
    bm25_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    embedding VECTOR(3072),
    similarity_score FLOAT,
    bm25_score FLOAT,
    combined_score FLOAT,
    file_name TEXT,
    file_type TEXT,
    file_path TEXT,
    chunk_index INTEGER,
    document_id UUID,
    title TEXT,
    summary TEXT,
    keywords TEXT[],
    entities TEXT[],
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
) AS $$
DECLARE
    avg_doc_length FLOAT;
    total_docs INTEGER;
    k1 FLOAT := 1.2;
    b FLOAT := 0.75;
BEGIN
    -- Get BM25 parameters
    SELECT average_document_length, total_documents 
    INTO avg_doc_length, total_docs
    FROM bm25_stats 
    ORDER BY updated_at DESC 
    LIMIT 1;
    
    -- Use defaults if stats not available
    IF avg_doc_length IS NULL THEN
        avg_doc_length := 500.0;
        total_docs := 1;
    END IF;

    RETURN QUERY
    WITH vector_results AS (
        -- Vector similarity search
        SELECT 
            r.id,
            r.content,
            r.embedding,
            (1 - (r.embedding <=> query_embedding)) AS similarity_score,
            r.file_name,
            r.file_type,
            r.file_path,
            r.chunk_index,
            r.document_id,
            r.title,
            r.summary,
            r.keywords,
            r.entities,
            r.created_at,
            r.updated_at,
            LENGTH(r.content) AS doc_length
        FROM rag_documents r
        WHERE (1 - (r.embedding <=> query_embedding)) >= similarity_threshold
            AND (file_types IS NULL OR r.file_type = ANY(file_types))
            AND (date_filter IS NULL OR r.created_at >= date_filter)
        ORDER BY r.embedding <=> query_embedding
        LIMIT limit_count * 3  -- Get more candidates for reranking
    ),
    bm25_scores AS (
        -- Calculate BM25 scores for vector results
        SELECT 
            vr.*,
            CASE 
                WHEN query_text IS NOT NULL AND LENGTH(query_text) > 0 THEN
                    ts_rank_cd(
                        to_tsvector('english', vr.content),
                        plainto_tsquery('english', query_text),
                        32 -- Use document length normalization
                    ) * (
                        -- BM25-like scoring adjustment
                        (k1 + 1) / (
                            k1 * (
                                (1 - b) + b * (vr.doc_length / avg_doc_length)
                            ) + 1
                        )
                    )
                ELSE 0.0
            END AS bm25_score
        FROM vector_results vr
    )
    SELECT 
        bs.id,
        bs.content,
        bs.embedding,
        bs.similarity_score,
        bs.bm25_score,
        -- Combined score with weights
        (vector_weight * bs.similarity_score + bm25_weight * bs.bm25_score) AS combined_score,
        bs.file_name,
        bs.file_type,
        bs.file_path,
        bs.chunk_index,
        bs.document_id,
        bs.title,
        bs.summary,
        bs.keywords,
        bs.entities,
        bs.created_at,
        bs.updated_at
    FROM bm25_scores bs
    ORDER BY (vector_weight * bs.similarity_score + bm25_weight * bs.bm25_score) DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SEMANTIC SEARCH FUNCTION
-- Pure vector similarity search
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding VECTOR(3072),
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_count INTEGER DEFAULT 10,
    file_types TEXT[] DEFAULT NULL,
    date_filter TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    embedding VECTOR(3072),
    similarity_score FLOAT,
    bm25_score FLOAT,
    combined_score FLOAT,
    file_name TEXT,
    file_type TEXT,
    file_path TEXT,
    chunk_index INTEGER,
    document_id UUID,
    title TEXT,
    summary TEXT,
    keywords TEXT[],
    entities TEXT[],
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id,
        r.content,
        r.embedding,
        (1 - (r.embedding <=> query_embedding)) AS similarity_score,
        0.0::FLOAT AS bm25_score,  -- No BM25 for pure semantic search
        (1 - (r.embedding <=> query_embedding)) AS combined_score,
        r.file_name,
        r.file_type,
        r.file_path,
        r.chunk_index,
        r.document_id,
        r.title,
        r.summary,
        r.keywords,
        r.entities,
        r.created_at,
        r.updated_at
    FROM rag_documents r
    WHERE (1 - (r.embedding <=> query_embedding)) >= similarity_threshold
        AND (file_types IS NULL OR r.file_type = ANY(file_types))
        AND (date_filter IS NULL OR r.created_at >= date_filter)
    ORDER BY r.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- KEYWORD SEARCH FUNCTION
-- Pure text-based search using PostgreSQL full-text search
-- =============================================================================

CREATE OR REPLACE FUNCTION keyword_search(
    query_text TEXT,
    limit_count INTEGER DEFAULT 10,
    file_types TEXT[] DEFAULT NULL,
    date_filter TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    embedding VECTOR(3072),
    similarity_score FLOAT,
    bm25_score FLOAT,
    combined_score FLOAT,
    file_name TEXT,
    file_type TEXT,
    file_path TEXT,
    chunk_index INTEGER,
    document_id UUID,
    title TEXT,
    summary TEXT,
    keywords TEXT[],
    entities TEXT[],
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id,
        r.content,
        r.embedding,
        0.0::FLOAT AS similarity_score,  -- No vector similarity for keyword search
        ts_rank_cd(r.search_vector, plainto_tsquery('english', query_text), 32) AS bm25_score,
        ts_rank_cd(r.search_vector, plainto_tsquery('english', query_text), 32) AS combined_score,
        r.file_name,
        r.file_type,
        r.file_path,
        r.chunk_index,
        r.document_id,
        r.title,
        r.summary,
        r.keywords,
        r.entities,
        r.created_at,
        r.updated_at
    FROM rag_documents r
    WHERE r.search_vector @@ plainto_tsquery('english', query_text)
        AND (file_types IS NULL OR r.file_type = ANY(file_types))
        AND (date_filter IS NULL OR r.created_at >= date_filter)
    ORDER BY ts_rank_cd(r.search_vector, plainto_tsquery('english', query_text), 32) DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- DOCUMENT STATISTICS FUNCTION
-- Get comprehensive database statistics
-- =============================================================================

CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE (
    total_documents BIGINT,
    total_chunks BIGINT,
    unique_files BIGINT,
    file_types TEXT[],
    avg_chunk_size FLOAT,
    total_size_mb FLOAT,
    oldest_document TIMESTAMPTZ,
    newest_document TIMESTAMPTZ,
    vector_dimension INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT document_id) AS total_documents,
        COUNT(*) AS total_chunks,
        COUNT(DISTINCT file_name) AS unique_files,
        ARRAY_AGG(DISTINCT file_type) AS file_types,
        AVG(LENGTH(content))::FLOAT AS avg_chunk_size,
        (pg_total_relation_size('rag_documents') / (1024.0 * 1024.0))::FLOAT AS total_size_mb,
        MIN(created_at) AS oldest_document,
        MAX(created_at) AS newest_document,
        COALESCE(vector_dims((SELECT embedding FROM rag_documents LIMIT 1)), 0) AS vector_dimension
    FROM rag_documents;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- BM25 STATISTICS UPDATE FUNCTION
-- Update BM25 statistics for better scoring
-- =============================================================================

CREATE OR REPLACE FUNCTION update_bm25_stats()
RETURNS VOID AS $$
DECLARE
    doc_count INTEGER;
    avg_length FLOAT;
    term_freqs JSONB;
    doc_freqs JSONB;
BEGIN
    -- Calculate basic statistics
    SELECT 
        COUNT(*),
        AVG(LENGTH(content))
    INTO doc_count, avg_length
    FROM rag_documents;
    
    -- For now, use simplified term/document frequencies
    -- In a full implementation, this would calculate actual TF-IDF statistics
    term_freqs := '{}';
    doc_freqs := '{}';
    
    -- Update or insert statistics
    INSERT INTO bm25_stats (total_documents, average_document_length, term_frequencies, document_frequencies)
    VALUES (doc_count, avg_length, term_freqs, doc_freqs)
    ON CONFLICT (id) DO UPDATE SET
        total_documents = EXCLUDED.total_documents,
        average_document_length = EXCLUDED.average_document_length,
        term_frequencies = EXCLUDED.term_frequencies,
        document_frequencies = EXCLUDED.document_frequencies,
        updated_at = NOW();
        
    -- If no existing stats, just insert
    IF NOT FOUND THEN
        INSERT INTO bm25_stats (total_documents, average_document_length, term_frequencies, document_frequencies)
        VALUES (doc_count, avg_length, term_freqs, doc_freqs);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SIMILARITY SEARCH BY CHUNK ID
-- Find chunks similar to a given chunk
-- =============================================================================

CREATE OR REPLACE FUNCTION find_similar_chunks(
    target_chunk_id UUID,
    similarity_threshold FLOAT DEFAULT 0.8,
    limit_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    similarity_score FLOAT,
    file_name TEXT,
    file_type TEXT,
    chunk_index INTEGER,
    document_id UUID
) AS $$
DECLARE
    target_embedding VECTOR(3072);
BEGIN
    -- Get the embedding of the target chunk
    SELECT embedding INTO target_embedding
    FROM rag_documents
    WHERE rag_documents.id = target_chunk_id;
    
    IF target_embedding IS NULL THEN
        RAISE EXCEPTION 'Chunk with id % not found', target_chunk_id;
    END IF;
    
    RETURN QUERY
    SELECT 
        r.id,
        r.content,
        (1 - (r.embedding <=> target_embedding)) AS similarity_score,
        r.file_name,
        r.file_type,
        r.chunk_index,
        r.document_id
    FROM rag_documents r
    WHERE r.id != target_chunk_id  -- Exclude the target chunk itself
        AND (1 - (r.embedding <=> target_embedding)) >= similarity_threshold
    ORDER BY r.embedding <=> target_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- CLEANUP FUNCTIONS
-- Utility functions for maintenance
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_old_documents(
    older_than_days INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM rag_documents
    WHERE created_at < (NOW() - INTERVAL '1 day' * older_than_days);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Update BM25 stats after cleanup
    PERFORM update_bm25_stats();
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SEARCH PERFORMANCE MONITORING
-- Function to monitor search performance
-- =============================================================================

CREATE OR REPLACE FUNCTION search_performance_test()
RETURNS TABLE (
    test_name TEXT,
    execution_time_ms FLOAT,
    results_count INTEGER,
    avg_similarity FLOAT
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    test_embedding VECTOR(3072);
    result_count INTEGER;
    avg_sim FLOAT;
BEGIN
    -- Generate a random test embedding
    SELECT ARRAY(SELECT random() FROM generate_series(1, 3072))::VECTOR(3072) INTO test_embedding;
    
    -- Test hybrid search
    start_time := clock_timestamp();
    SELECT COUNT(*), AVG(similarity_score)
    INTO result_count, avg_sim
    FROM hybrid_search(test_embedding, 'test query', 0.5, 10);
    end_time := clock_timestamp();
    
    RETURN QUERY SELECT 
        'hybrid_search'::TEXT,
        EXTRACT(MILLISECONDS FROM (end_time - start_time))::FLOAT,
        result_count,
        COALESCE(avg_sim, 0.0);
    
    -- Test semantic search
    start_time := clock_timestamp();
    SELECT COUNT(*), AVG(similarity_score)
    INTO result_count, avg_sim
    FROM semantic_search(test_embedding, 0.5, 10);
    end_time := clock_timestamp();
    
    RETURN QUERY SELECT 
        'semantic_search'::TEXT,
        EXTRACT(MILLISECONDS FROM (end_time - start_time))::FLOAT,
        result_count,
        COALESCE(avg_sim, 0.0);
    
    -- Test keyword search
    start_time := clock_timestamp();
    SELECT COUNT(*), AVG(bm25_score)
    INTO result_count, avg_sim
    FROM keyword_search('test query', 10);
    end_time := clock_timestamp();
    
    RETURN QUERY SELECT 
        'keyword_search'::TEXT,
        EXTRACT(MILLISECONDS FROM (end_time - start_time))::FLOAT,
        result_count,
        COALESCE(avg_sim, 0.0);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- GRANT PERMISSIONS
-- Grant necessary permissions for the functions
-- =============================================================================

-- Grant execute permissions to authenticated users (adjust as needed)
GRANT EXECUTE ON FUNCTION hybrid_search TO authenticated;
GRANT EXECUTE ON FUNCTION semantic_search TO authenticated;
GRANT EXECUTE ON FUNCTION keyword_search TO authenticated;
GRANT EXECUTE ON FUNCTION get_document_stats TO authenticated;
GRANT EXECUTE ON FUNCTION find_similar_chunks TO authenticated;
GRANT EXECUTE ON FUNCTION search_performance_test TO authenticated;

-- Grant update permissions for BM25 stats (restrict as needed)
GRANT EXECUTE ON FUNCTION update_bm25_stats TO service_role;
GRANT EXECUTE ON FUNCTION cleanup_old_documents TO service_role;

-- =============================================================================
-- INITIALIZE BM25 STATISTICS
-- Initialize the BM25 statistics table
-- =============================================================================

SELECT update_bm25_stats();

-- =============================================================================
-- CREATE INDEXES FOR BETTER PERFORMANCE
-- Additional indexes for optimized search performance
-- =============================================================================

-- Index for file type filtering
CREATE INDEX IF NOT EXISTS idx_rag_documents_file_type_created 
ON rag_documents(file_type, created_at);

-- Index for document ID lookups
CREATE INDEX IF NOT EXISTS idx_rag_documents_document_id_chunk_index 
ON rag_documents(document_id, chunk_index);

-- Partial index for recent documents
CREATE INDEX IF NOT EXISTS idx_rag_documents_recent 
ON rag_documents(created_at, file_type) 
WHERE created_at > (NOW() - INTERVAL '30 days');

-- Index to support similarity searches with filters
CREATE INDEX IF NOT EXISTS idx_rag_documents_composite 
ON rag_documents(file_type, created_at, similarity_score) 
WHERE embedding IS NOT NULL;

COMMENT ON FUNCTION hybrid_search IS 'Combines vector similarity search with BM25 keyword scoring for optimal retrieval';
COMMENT ON FUNCTION semantic_search IS 'Pure vector similarity search using cosine distance';
COMMENT ON FUNCTION keyword_search IS 'Text-based search using PostgreSQL full-text search with BM25-like scoring';
COMMENT ON FUNCTION get_document_stats IS 'Returns comprehensive statistics about the document collection';
COMMENT ON FUNCTION update_bm25_stats IS 'Updates BM25 statistics for improved search scoring';
COMMENT ON FUNCTION find_similar_chunks IS 'Finds chunks similar to a given chunk using vector similarity';
COMMENT ON FUNCTION cleanup_old_documents IS 'Removes documents older than specified days and updates statistics';
COMMENT ON FUNCTION search_performance_test IS 'Tests search function performance and returns timing statistics';