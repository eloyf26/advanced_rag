# Supabase Vector Database Documentation

## Overview

The Supabase Vector Database service provides a **complete, standalone** PostgreSQL-based vector storage solution optimized for RAG (Retrieval-Augmented Generation) applications. It combines the power of pgvector for semantic search with traditional full-text search capabilities, offering hybrid retrieval mechanisms with advanced query functions and performance optimizations.

This database can operate **independently** and serve multiple applications, not just the ingestion service. It provides a complete API through SQL functions and direct database access.

## 🎯 Goals

- **Hybrid Search Capabilities**: Combine vector similarity and BM25 keyword search for optimal retrieval
- **High Performance**: Optimized indexes and query functions for fast similarity search at scale
- **Scalable Architecture**: Support for millions of document chunks with efficient storage
- **Production Ready**: Row-level security, automated cleanup, and comprehensive monitoring
- **Advanced Features**: Semantic search, keyword search, triangulation, and statistical analysis
- **Independent Operation**: Complete standalone database that can serve multiple applications

## 🏗️ Database Architecture

### Complete File Structure

```
📁 vector_database/
├── 📄 setup.sql                       # Complete database schema
├── 📄 functions.sql                   # Advanced search functions
├── 📄 security.sql                    # Security policies (optional)
├── 📄 scripts/
│   ├── 📄 deploy.sh                   # Enhanced deployment script
│   ├── 📄 backup.sh                   # Backup utilities
│   ├── 📄 monitor.sh                  # Monitoring scripts
│   └── 📄 cleanup.sh                  # Maintenance utilities
├── 📄 migrations/
│   ├── 📄 001_initial_schema.sql      # Initial table creation
│   ├── 📄 002_indexes.sql             # Performance indexes
│   └── 📄 003_functions.sql           # Search functions
└── 📄 README.md                       # This documentation
```

### Core Schema Design

#### Main Storage Table: `rag_documents`

```sql
CREATE TABLE rag_documents (
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
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content || ' ' || 
        COALESCE(title, '') || ' ' || 
        COALESCE(summary, ''))
    ) STORED,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### Statistics Table: `bm25_stats`

```sql
CREATE TABLE bm25_stats (
    id SERIAL PRIMARY KEY,
    total_documents INTEGER NOT NULL,
    average_document_length FLOAT NOT NULL,
    term_frequencies JSONB NOT NULL,
    document_frequencies JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 🚀 Independent Deployment

### Prerequisites

- Supabase account (free tier available)
- PostgreSQL 14+ with pgvector extension
- `psql` client tools
- Basic understanding of SQL and PostgreSQL

### Quick Setup (Standalone)

1. **Supabase Project Setup**
```bash
# Create a new Supabase project at https://supabase.com
# Note your project URL and service key
```

2. **Enable pgvector Extension**
```sql
-- In Supabase SQL Editor or via psql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. **Clone and Setup**
```bash
git clone <repository-url>
cd vector_database/

# Set environment variables
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_KEY="your_service_key_here"
```

4. **Deploy Database Schema**
```bash
# Make deployment script executable
chmod +x scripts/deploy.sh

# Run deployment with verification
./scripts/deploy.sh

# Or with options
./scripts/deploy.sh --verbose --skip-tests
```

5. **Verify Installation**
```bash
# Test connection
PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.$(echo $SUPABASE_URL | sed 's|https://||' | sed 's|\.supabase\.co.*|.supabase.co|')" \
  -U postgres \
  -d postgres \
  -c "SELECT * FROM get_document_stats();"
```

### Advanced Deployment Options

#### 1. Dry Run Deployment
```bash
# See what would be deployed without making changes
./scripts/deploy.sh --dry-run
```

#### 2. Verbose Deployment
```bash
# Get detailed deployment information
./scripts/deploy.sh --verbose
```

#### 3. Skip Connection Tests
```bash
# Deploy without preliminary tests (for CI/CD)
./scripts/deploy.sh --skip-tests
```

#### 4. Manual Deployment
```bash
# Execute SQL files manually
PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.your-host.supabase.co" \
  -U postgres \
  -d postgres \
  -f setup.sql

PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.your-host.supabase.co" \
  -U postgres \
  -d postgres \
  -f functions.sql
```

## 📊 Advanced Search Functions

The database provides sophisticated search capabilities through custom SQL functions:

### 1. Hybrid Search (Recommended)

Combines vector similarity with BM25 keyword scoring:

```sql
SELECT * FROM hybrid_search(
    query_embedding => '[0.1, 0.2, ...]'::vector(3072),
    query_text => 'machine learning algorithms',
    similarity_threshold => 0.7,
    limit_count => 10,
    file_types => ARRAY['pdf', 'docx'],
    date_filter => NOW() - INTERVAL '30 days',
    vector_weight => 0.7,
    bm25_weight => 0.3
);
```

**Features:**
- Weighted combination of vector and keyword scores
- File type and date filtering
- Configurable similarity thresholds
- Optimized performance with HNSW indexing

### 2. Semantic Search

Pure vector similarity search:

```sql
SELECT * FROM semantic_search(
    query_embedding => '[0.1, 0.2, ...]'::vector(3072),
    similarity_threshold => 0.7,
    limit_count => 10,
    file_types => ARRAY['pdf'],
    date_filter => NOW() - INTERVAL '7 days'
);
```

**Use Cases:**
- Conceptual similarity queries
- Cross-lingual search (with multilingual embeddings)
- Semantic clustering and analysis

### 3. Keyword Search

Traditional BM25 text search:

```sql
SELECT * FROM keyword_search(
    query_text => 'natural language processing',
    limit_count => 10,
    file_types => ARRAY['txt', 'md'],
    date_filter => NOW() - INTERVAL '1 month'
);
```

**Use Cases:**
- Exact term matching
- Boolean queries with operators
- Legacy search compatibility

### 4. Similarity by Document

Find documents similar to a given document:

```sql
SELECT * FROM find_similar_chunks(
    target_chunk_id => 'uuid-here',
    similarity_threshold => 0.8,
    limit_count => 5
);
```

### 5. Database Statistics

Comprehensive database analytics:

```sql
SELECT * FROM get_document_stats();
```

Returns:
- Total documents and chunks
- File type distribution
- Average chunk size
- Database size metrics
- Vector dimensions
- Date ranges

## ⚙️ Performance Optimization

### Index Configuration

The database uses optimized indexes for maximum performance:

#### HNSW Vector Index
```sql
CREATE INDEX rag_documents_embedding_idx 
ON rag_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Tuning Parameters:**
- `m = 16`: Good balance of accuracy and memory usage
- `ef_construction = 64`: Build-time effort for accuracy

#### GIN Text Search Index
```sql
CREATE INDEX rag_documents_search_vector_idx 
ON rag_documents USING gin(search_vector);
```

#### Composite Indexes
```sql
-- Optimized for common filter patterns
CREATE INDEX rag_documents_file_type_processed_idx 
ON rag_documents(file_type, processed_at);

CREATE INDEX rag_documents_doc_chunk_idx 
ON rag_documents(document_id, chunk_index);
```

### Performance Monitoring

#### Query Performance
```sql
-- Monitor search function performance
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename = 'rag_documents';

-- Check index usage
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE schemaname = 'public';
```

#### Storage Analysis
```sql
-- Table and index sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Performance Tuning Scripts

The database includes performance testing and optimization utilities:

```sql
-- Run performance test
SELECT * FROM search_performance_test();
```

Returns timing for different search strategies with sample data.

## 🔧 Configuration and Scaling

### Memory Configuration

For optimal performance, configure PostgreSQL settings:

```sql
-- Recommended settings for vector workloads
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Vector-specific settings
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
SELECT pg_reload_conf();
```

### Scaling Recommendations

#### Small Scale (< 100K documents)
- Default Supabase configuration
- Standard indexes
- Basic monitoring

#### Medium Scale (100K-1M documents)
- Increase `shared_buffers` to 512MB
- Consider read replicas
- Monitor query performance

#### Large Scale (1M+ documents)
- Dedicated database instance
- Partitioning strategy by date or document type
- Connection pooling
- Automated maintenance scripts

### Horizontal Scaling

#### Read Replicas
```sql
-- Configure read replica for search workloads
-- Primary: Write operations (ingestion)
-- Replica: Read operations (search)
```

#### Partitioning Strategy
```sql
-- Partition by date for time-series data
CREATE TABLE rag_documents_2024_01 
PARTITION OF rag_documents
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Partition by file type for workload isolation
CREATE TABLE rag_documents_pdf 
PARTITION OF rag_documents
FOR VALUES IN ('pdf');
```

## 🔐 Security and Access Control

### Row Level Security (RLS)

```sql
-- Enable RLS on documents table
ALTER TABLE rag_documents ENABLE ROW LEVEL SECURITY;

-- Example policies
CREATE POLICY "user_documents" ON rag_documents
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "public_read" ON rag_documents
    FOR SELECT USING (visibility = 'public');
```

### Function Security

```sql
-- Grant appropriate permissions
GRANT EXECUTE ON FUNCTION hybrid_search TO authenticated;
GRANT EXECUTE ON FUNCTION semantic_search TO authenticated;
GRANT EXECUTE ON FUNCTION keyword_search TO authenticated;
GRANT EXECUTE ON FUNCTION get_document_stats TO authenticated;

-- Restrict administrative functions
GRANT EXECUTE ON FUNCTION update_bm25_stats TO service_role;
GRANT EXECUTE ON FUNCTION cleanup_old_documents TO service_role;
```

### API Security

For direct database access, implement proper authentication:
- Service-level authentication with service keys
- Row-level security for multi-tenant applications
- Function-level permissions for different user roles

## 🛠️ Maintenance and Administration

### Automated Maintenance

#### Update Statistics
```sql
-- Update BM25 statistics (run periodically)
SELECT update_bm25_stats();

-- Analyze table statistics
ANALYZE rag_documents;
ANALYZE bm25_stats;
```

#### Cleanup Operations
```sql
-- Remove old documents (older than 90 days)
SELECT cleanup_old_documents(90);

-- Remove duplicate content
DELETE FROM rag_documents a
USING rag_documents b
WHERE a.id < b.id 
AND a.content_hash = b.content_hash;
```

#### Index Maintenance
```sql
-- Rebuild indexes (maintenance window)
REINDEX INDEX CONCURRENTLY rag_documents_embedding_idx;
REINDEX INDEX CONCURRENTLY rag_documents_search_vector_idx;
```

### Backup and Recovery

#### Backup Strategy
```bash
# Full database backup
pg_dump -h db.your-host.supabase.co -U postgres -d postgres > backup.sql

# Table-specific backup
pg_dump -h db.your-host.supabase.co -U postgres -d postgres -t rag_documents > documents_backup.sql

# Data-only backup
pg_dump -h db.your-host.supabase.co -U postgres -d postgres --data-only -t rag_documents > data_backup.sql
```

#### Recovery Procedures
```bash
# Restore full database
psql -h db.your-host.supabase.co -U postgres -d postgres < backup.sql

# Restore specific table
psql -h db.your-host.supabase.co -U postgres -d postgres < documents_backup.sql
```

### Monitoring Scripts

#### Database Health Monitoring
```bash
# Create monitoring script
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
# Database health monitoring

# Check connection
PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.$DB_HOST" -U postgres -d postgres \
  -c "SELECT 1;" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Database connection: OK"
else
    echo "❌ Database connection: FAILED"
    exit 1
fi

# Check document count
DOC_COUNT=$(PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.$DB_HOST" -U postgres -d postgres \
  -t -c "SELECT total_chunks FROM get_document_stats();" | tr -d ' ')

echo "📊 Total documents: $DOC_COUNT"

# Check recent activity
RECENT=$(PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.$DB_HOST" -U postgres -d postgres \
  -t -c "SELECT COUNT(*) FROM rag_documents WHERE created_at > NOW() - INTERVAL '24 hours';" | tr -d ' ')

echo "📈 Documents added (24h): $RECENT"
EOF

chmod +x scripts/monitor.sh
```

## 🧪 Testing and Validation

### Connection Testing
```bash
# Test basic connection
./scripts/deploy.sh --dry-run

# Test search functions
PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.your-host.supabase.co" \
  -U postgres \
  -d postgres \
  -c "SELECT * FROM search_performance_test();"
```

### Data Validation
```sql
-- Validate vector dimensions
SELECT 
    id,
    vector_dims(embedding) as dimensions,
    CASE 
        WHEN vector_dims(embedding) = 3072 THEN 'OK'
        ELSE 'INVALID'
    END as status
FROM rag_documents 
WHERE embedding IS NOT NULL
LIMIT 10;

-- Check for null embeddings
SELECT COUNT(*) as null_embeddings
FROM rag_documents 
WHERE embedding IS NULL;

-- Validate content integrity
SELECT 
    COUNT(*) as total_docs,
    COUNT(CASE WHEN content IS NOT NULL AND LENGTH(content) > 0 THEN 1 END) as valid_content,
    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings
FROM rag_documents;
```

### Performance Testing
```sql
-- Test search performance with realistic queries
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM hybrid_search(
    query_embedding => (SELECT embedding FROM rag_documents LIMIT 1),
    query_text => 'machine learning',
    similarity_threshold => 0.7,
    limit_count => 10
);
```

## 🔗 Integration Examples

### Direct SQL Access
```python
import psycopg2
import os

# Connect to database
conn = psycopg2.connect(
    host=f"db.{os.getenv('SUPABASE_URL').replace('https://', '').split('.')[0]}.supabase.co",
    database="postgres",
    user="postgres",
    password=os.getenv('SUPABASE_SERVICE_KEY')
)

# Execute search
cur = conn.cursor()
cur.execute("""
    SELECT id, content, similarity_score 
    FROM hybrid_search(%s, %s, 0.7, 10)
""", (query_embedding, "machine learning"))

results = cur.fetchall()
```

### Supabase Client Integration
```python
from supabase import create_client
import os

# Initialize client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Search documents
result = supabase.rpc('hybrid_search', {
    'query_embedding': query_embedding,
    'query_text': 'machine learning',
    'similarity_threshold': 0.7,
    'limit_count': 10
}).execute()

documents = result.data
```

### REST API Access
```bash
# Direct RPC call via Supabase REST API
curl -X POST "https://your-project.supabase.co/rest/v1/rpc/hybrid_search" \
  -H "apikey: YOUR_ANON_KEY" \
  -H "Authorization: Bearer YOUR_SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.1, 0.2, ...],
    "query_text": "machine learning",
    "similarity_threshold": 0.7,
    "limit_count": 10
  }'
```

## 🚨 Troubleshooting

### Common Issues

#### 1. pgvector Extension Not Available
```bash
# Enable in Supabase dashboard
# Database > Extensions > Enable "vector"

# Or via SQL
CREATE EXTENSION IF NOT EXISTS vector;
```

#### 2. Connection Failures
```bash
# Verify connection details
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY

# Test basic connection
PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
  -h "db.your-host.supabase.co" \
  -U postgres \
  -d postgres \
  -c "SELECT version();"
```

#### 3. Performance Issues
```sql
-- Check index usage
SELECT schemaname, tablename, attname, n_distinct
FROM pg_stats 
WHERE tablename = 'rag_documents';

-- Analyze slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements 
WHERE query LIKE '%hybrid_search%'
ORDER BY mean_exec_time DESC;
```

#### 4. Storage Issues
```sql
-- Check database size
SELECT 
    pg_size_pretty(pg_database_size('postgres')) as database_size,
    pg_size_pretty(pg_total_relation_size('rag_documents')) as table_size;

-- Check for bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_stat_get_live_tuples(c.oid) as live_tuples,
    pg_stat_get_dead_tuples(c.oid) as dead_tuples
FROM pg_tables t
JOIN pg_class c ON c.relname = t.tablename
WHERE schemaname = 'public';
```

### Error Recovery

#### Corrupted Indexes
```sql
-- Rebuild corrupted indexes
DROP INDEX CONCURRENTLY rag_documents_embedding_idx;
CREATE INDEX CONCURRENTLY rag_documents_embedding_idx 
ON rag_documents USING hnsw (embedding vector_cosine_ops);
```

#### Missing Statistics
```sql
-- Reinitialize BM25 statistics
DELETE FROM bm25_stats;
SELECT update_bm25_stats();
```

## 📈 Production Considerations

### Performance Monitoring
- Set up automated monitoring for query performance
- Monitor index effectiveness and usage
- Track storage growth and plan capacity
- Set up alerts for connection failures and slow queries

### Backup Strategy
- Daily automated backups
- Point-in-time recovery capability
- Test backup restoration procedures
- Document recovery procedures

### Security
- Regular security updates
- Access audit logging
- Principle of least privilege
- Network security (VPC, firewall rules)

### Scalability Planning
- Monitor growth trends
- Plan for partitioning when approaching 1M+ documents
- Consider read replicas for query workloads
- Evaluate connection pooling needs

The Supabase Vector Database provides a complete, production-ready foundation for RAG applications with the flexibility to operate independently while serving multiple clients and use cases.