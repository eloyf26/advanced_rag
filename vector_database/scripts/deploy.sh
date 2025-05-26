#!/bin/bash

# Deployment script for Supabase Vector Database
# Usage: ./scripts/deploy.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}🚀 Starting Supabase Vector Database Deployment${NC}"

# Check required environment variables
if [ -z "$SUPABASE_URL" ]; then
    echo -e "${RED}❌ Error: SUPABASE_URL environment variable is not set${NC}"
    exit 1
fi

if [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo -e "${RED}❌ Error: SUPABASE_SERVICE_KEY environment variable is not set${NC}"
    exit 1
fi

# Extract database connection details from Supabase URL
DB_HOST=$(echo "$SUPABASE_URL" | sed 's|https://||' | sed 's|\.supabase\.co.*|.supabase.co|')
DB_NAME="postgres"
DB_USER="postgres"

# Function to execute SQL file
execute_sql_file() {
    local sql_file="$1"
    local description="$2"
    
    echo -e "${YELLOW}📄 Executing: $description${NC}"
    
    if [ ! -f "$sql_file" ]; then
        echo -e "${RED}❌ Error: SQL file $sql_file not found${NC}"
        exit 1
    fi
    
    # Use psql to execute the SQL file
    PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -f "$sql_file" \
        -v ON_ERROR_STOP=1 \
        --quiet
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully executed: $description${NC}"
    else
        echo -e "${RED}❌ Failed to execute: $description${NC}"
        exit 1
    fi
}

# Function to test database connection
test_connection() {
    echo -e "${YELLOW}🔍 Testing database connection...${NC}"
    
    PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT 1;" \
        --quiet > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Database connection successful${NC}"
    else
        echo -e "${RED}❌ Database connection failed${NC}"
        echo -e "${RED}   Please check your SUPABASE_URL and SUPABASE_SERVICE_KEY${NC}"
        exit 1
    fi
}

# Function to check if vector extension is available
check_vector_extension() {
    echo -e "${YELLOW}🔍 Checking pgvector extension availability...${NC}"
    
    PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT extname FROM pg_available_extensions WHERE extname = 'vector';" \
        --quiet --tuples-only | grep -q "vector"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ pgvector extension is available${NC}"
    else
        echo -e "${RED}❌ pgvector extension is not available${NC}"
        echo -e "${RED}   Please enable pgvector in your Supabase project${NC}"
        exit 1
    fi
}

# Function to verify deployment
verify_deployment() {
    echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
    
    # Check if tables exist
    tables=$(PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('rag_documents', 'bm25_stats');" \
        --quiet --tuples-only | wc -l)
    
    if [ "$tables" -eq 2 ]; then
        echo -e "${GREEN}✅ All tables created successfully${NC}"
    else
        echo -e "${RED}❌ Some tables are missing${NC}"
        exit 1
    fi
    
    # Check if functions exist
    functions=$(PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT proname FROM pg_proc WHERE proname IN ('hybrid_search', 'semantic_search', 'keyword_search');" \
        --quiet --tuples-only | wc -l)
    
    if [ "$functions" -eq 3 ]; then
        echo -e "${GREEN}✅ All functions created successfully${NC}"
    else
        echo -e "${RED}❌ Some functions are missing${NC}"
        exit 1
    fi
    
    # Test a simple query
    PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT * FROM get_document_stats();" \
        --quiet > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Database functions are working correctly${NC}"
    else
        echo -e "${RED}❌ Database functions test failed${NC}"
        exit 1
    fi
}

# Main deployment process
main() {
    echo -e "${GREEN}Starting deployment process...${NC}"
    
    # Step 1: Test connection
    test_connection
    
    # Step 2: Check vector extension
    check_vector_extension
    
    # Step 3: Deploy schema
    execute_sql_file "$PROJECT_DIR/setup.sql" "Main database schema"
    
    # Step 4: Deploy functions
    execute_sql_file "$PROJECT_DIR/functions.sql" "Search functions"
    
    # Step 5: Deploy security (if exists)
    if [ -f "$PROJECT_DIR/security.sql" ]; then
        execute_sql_file "$PROJECT_DIR/security.sql" "Security policies"
    fi
    
    # Step 6: Verify deployment
    verify_deployment
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${GREEN}Your vector database is ready for use.${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Update your application configuration with the database details"
    echo "2. Start ingesting documents using the ingestion service"
    echo "3. Test queries using the RAG agent"
    echo ""
    echo -e "${YELLOW}Database connection details:${NC}"
    echo "Host: db.$DB_HOST"
    echo "Database: $DB_NAME"
    echo "User: $DB_USER"
}

# Check if psql is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}❌ Error: psql is not installed${NC}"
    echo "Please install PostgreSQL client tools"
    exit 1
fi

# Run main deployment
main