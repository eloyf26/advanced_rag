# Deploy via Supabase Dashboard

This guide shows how to deploy the vector database schema using the Supabase Dashboard SQL editor - no local tools required!

## Prerequisites

- A Supabase account and project
- The SQL files from this repository

## Step-by-Step Deployment

### 1. Enable pgvector Extension

1. Go to your [Supabase Dashboard](https://app.supabase.com)
2. Select your project
3. Navigate to **Database → Extensions**
4. Search for "vector"
5. Click the toggle to enable it
6. Wait for confirmation

### 2. Deploy Main Schema

1. Go to **SQL Editor** in the sidebar
2. Click **New query**
3. Copy the entire contents of `vector_database/setup.sql`
4. Paste into the SQL editor
5. Click **Run** (or press Ctrl/Cmd + Enter)
6. You should see "Success. No rows returned"

### 3. Deploy Search Functions

1. Create another new query
2. Copy the entire contents of `vector_database/functions.sql`
3. Paste into the SQL editor
4. Click **Run**
5. You should see "Success. No rows returned"

### 4. Verify Deployment

Run this verification query in a new SQL editor tab:

```sql
-- Check tables
SELECT tablename 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN ('rag_documents', 'bm25_stats');

-- Check functions  
SELECT proname 
FROM pg_proc 
JOIN pg_namespace ON pg_proc.pronamespace = pg_namespace.oid
WHERE pg_namespace.nspname = 'public'
AND proname IN ('hybrid_search', 'semantic_search', 'keyword_search', 'get_document_stats');

-- Test the stats function
SELECT * FROM get_document_stats();
```

You should see:
- 2 tables: `rag_documents` and `bm25_stats`
- 4+ functions including the search functions
- Stats showing 0 documents (empty database)

## 5. Get Connection Details

For the ingestion service and RAG agent, you'll need:

1. **Database URL**: 
   - Go to Settings → Database
   - Copy the "URI" connection string
   
2. **Service Key** (for API access):
   - Go to Settings → API
   - Copy the "service_role" key (has full access)
   
3. **Database Password** (for direct connections):
   - Go to Settings → Database
   - Find the password or reset it if needed

## Troubleshooting

### "Permission denied" errors
- Make sure you're using the correct project
- Try refreshing the page and running again

### "Extension not found" errors  
- Ensure pgvector is enabled (step 1)
- Wait a moment for it to fully activate

### "Syntax error" messages
- Make sure you copied the ENTIRE SQL file
- Check for any accidental modifications

## Next Steps

1. Configure your `.env` file with the connection details
2. Test the connection using the Python test script
3. Start ingesting documents!

## Quick Test

After deployment, test your setup:

```sql
-- Insert a test document
INSERT INTO rag_documents (
    content,
    embedding,
    file_name,
    file_type
) VALUES (
    'This is a test document for the RAG system.',
    array_fill(0.1, ARRAY[3072])::vector(3072),
    'test.txt',
    'txt'
);

-- Search for it
SELECT id, content, file_name 
FROM semantic_search(
    array_fill(0.1, ARRAY[3072])::vector(3072),
    0.5,
    10
);

-- Clean up
DELETE FROM rag_documents WHERE file_name = 'test.txt';
```