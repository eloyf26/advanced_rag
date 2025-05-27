#!/bin/bash

# Enhanced Deployment script for Supabase Vector Database
# Usage: ./scripts/deploy.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DRY_RUN=false
VERBOSE=false
SKIP_TESTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dry-run     Show what would be done without executing"
            echo "  --verbose     Enable verbose output"
            echo "  --skip-tests  Skip connection and validation tests"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🚀 Starting Supabase Vector Database Deployment${NC}"

# Function to log messages
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${BLUE}[$timestamp] INFO: $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] WARNING: $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${NC}"
            ;;
    esac
    
    if [ "$VERBOSE" = true ]; then
        echo "[$timestamp] $level: $message" >> "${PROJECT_DIR}/deployment.log"
    fi
}

# Check required environment variables
check_environment() {
    log "INFO" "Checking environment variables..."
    
    local missing_vars=()
    
    if [ -z "$SUPABASE_URL" ]; then
        missing_vars+=("SUPABASE_URL")
    fi
    
    if [ -z "$SUPABASE_SERVICE_KEY" ]; then
        missing_vars+=("SUPABASE_SERVICE_KEY")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log "ERROR" "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Please set these variables before running the deployment:"
        echo "  export SUPABASE_URL=\"your_supabase_url\""
        echo "  export SUPABASE_SERVICE_KEY=\"your_service_key\""
        exit 1
    fi
    
    log "SUCCESS" "Environment variables are set"
}

# Extract database connection details from Supabase URL
extract_db_details() {
    log "INFO" "Extracting database connection details..."
    
    # Extract host from URL
    DB_HOST=$(echo "$SUPABASE_URL" | sed 's|https://||' | sed 's|\.supabase\.co.*|.supabase.co|')
    DB_NAME="postgres"
    DB_USER="postgres"
    
    if [ "$VERBOSE" = true ]; then
        log "INFO" "Database host: $DB_HOST"
        log "INFO" "Database name: $DB_NAME"
        log "INFO" "Database user: $DB_USER"
    fi
}

# Check if psql is installed
check_psql() {
    log "INFO" "Checking for PostgreSQL client..."
    
    if ! command -v psql &> /dev/null; then
        log "ERROR" "psql is not installed"
        echo ""
        echo "Please install PostgreSQL client tools:"
        echo "  Ubuntu/Debian: sudo apt-get install postgresql-client"
        echo "  CentOS/RHEL: sudo yum install postgresql"
        echo "  macOS: brew install postgresql"
        exit 1
    fi
    
    log "SUCCESS" "PostgreSQL client found"
}

# Function to execute SQL file
execute_sql_file() {
    local sql_file="$1"
    local description="$2"
    
    log "INFO" "Executing: $description"
    
    if [ ! -f "$sql_file" ]; then
        log "ERROR" "SQL file $sql_file not found"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY RUN] Would execute: $sql_file"
        return 0
    fi
    
    # Use psql to execute the SQL file
    if PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -f "$sql_file" \
        -v ON_ERROR_STOP=1 \
        --quiet > /dev/null 2>&1; then
        
        log "SUCCESS" "Successfully executed: $description"
    else
        log "ERROR" "Failed to execute: $description"
        
        # Show detailed error in verbose mode
        if [ "$VERBOSE" = true ]; then
            log "INFO" "Attempting to get detailed error information..."
            PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
                -h "db.$DB_HOST" \
                -U "$DB_USER" \
                -d "$DB_NAME" \
                -f "$sql_file" \
                -v ON_ERROR_STOP=1 2>&1 | head -20
        fi
        
        exit 1
    fi
}

# Function to test database connection
test_connection() {
    log "INFO" "Testing database connection..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY RUN] Would test database connection"
        return 0
    fi
    
    if PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT 1;" \
        --quiet > /dev/null 2>&1; then
        
        log "SUCCESS" "Database connection successful"
    else
        log "ERROR" "Database connection failed"
        echo ""
        echo "Please check your SUPABASE_URL and SUPABASE_SERVICE_KEY"
        echo "Current SUPABASE_URL: $SUPABASE_URL"
        echo "Make sure your service key has the necessary permissions"
        exit 1
    fi
}

# Function to check if vector extension is available
check_vector_extension() {
    log "INFO" "Checking pgvector extension availability..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY RUN] Would check pgvector extension"
        return 0
    fi
    
    if PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT extname FROM pg_available_extensions WHERE extname = 'vector';" \
        --quiet --tuples-only | grep -q "vector"; then
        
        log "SUCCESS" "pgvector extension is available"
    else
        log "ERROR" "pgvector extension is not available"
        echo ""
        echo "Please enable pgvector in your Supabase project:"
        echo "1. Go to your Supabase dashboard"
        echo "2. Navigate to Database > Extensions"
        echo "3. Enable the 'vector' extension"
        exit 1
    fi
}

# Function to verify deployment
verify_deployment() {
    log "INFO" "Verifying deployment..."
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "[DRY RUN] Would verify deployment"
        return 0
    fi
    
    # Check if tables exist
    local tables_result=$(PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('rag_documents', 'bm25_stats');" \
        --quiet --tuples-only 2>/dev/null)
    
    local table_count=$(echo "$tables_result" | wc -l)
    
    if [ "$table_count" -eq 2 ]; then
        log "SUCCESS" "All tables created successfully"
    else
        log "ERROR" "Some tables are missing"
        log "INFO" "Found tables: $tables_result"
        exit 1
    fi
    
    # Check if functions exist
    local functions_result=$(PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT proname FROM pg_proc WHERE proname IN ('hybrid_search', 'semantic_search', 'keyword_search', 'get_document_stats');" \
        --quiet --tuples-only 2>/dev/null)
    
    local function_count=$(echo "$functions_result" | wc -l)
    
    if [ "$function_count" -ge 3 ]; then
        log "SUCCESS" "All functions created successfully"
    else
        log "WARNING" "Some functions might be missing"
        if [ "$VERBOSE" = true ]; then
            log "INFO" "Found functions: $functions_result"
        fi
    fi
    
    # Test a simple query
    if PGPASSWORD="$SUPABASE_SERVICE_KEY" psql \
        -h "db.$DB_HOST" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "SELECT * FROM get_document_stats();" \
        --quiet > /dev/null 2>&1; then
        
        log "SUCCESS" "Database functions are working correctly"
    else
        log "WARNING" "Database functions test failed - this might be expected for empty database"
    fi
}

# Function to show deployment summary
show_summary() {
    echo ""
    echo -e "${GREEN}🎉 Deployment Summary${NC}"
    echo "===================="
    echo "Database Host: db.$DB_HOST"
    echo "Database Name: $DB_NAME"
    echo "Database User: $DB_USER"
    echo "Deployment Time: $(date)"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Update your application configuration with the database details"
    echo "2. Start ingesting documents using the ingestion service"
    echo "3. Test queries using the RAG agent"
    echo ""
    echo -e "${YELLOW}Useful commands:${NC}"
    echo "# Test the database connection:"
    echo "PGPASSWORD=\"\$SUPABASE_SERVICE_KEY\" psql -h \"db.$DB_HOST\" -U \"$DB_USER\" -d \"$DB_NAME\" -c \"SELECT * FROM get_document_stats();\""
    echo ""
    echo "# Check table structure:"
    echo "PGPASSWORD=\"\$SUPABASE_SERVICE_KEY\" psql -h \"db.$DB_HOST\" -U \"$DB_USER\" -d \"$DB_NAME\" -c \"\\dt\""
}

# Main deployment process
main() {
    log "INFO" "Starting deployment process..."
    
    # Pre-flight checks
    check_environment
    extract_db_details
    check_psql
    
    if [ "$SKIP_TESTS" = false ]; then
        # Connection and availability tests
        test_connection
        check_vector_extension
    fi
    
    # Deploy schema
    execute_sql_file "$PROJECT_DIR/setup.sql" "Main database schema"
    
    # Deploy functions
    execute_sql_file "$PROJECT_DIR/functions.sql" "Search functions"
    
    # Deploy security (if exists)
    if [ -f "$PROJECT_DIR/security.sql" ]; then
        execute_sql_file "$PROJECT_DIR/security.sql" "Security policies"
    else
        log "INFO" "No security.sql file found, skipping security policies"
    fi
    
    if [ "$SKIP_TESTS" = false ]; then
        # Verify deployment
        verify_deployment
    fi
    
    if [ "$DRY_RUN" = false ]; then
        log "SUCCESS" "Deployment completed successfully!"
        show_summary
    else
        log "INFO" "Dry run completed successfully - no changes were made"
    fi
}

# Trap errors and cleanup
trap 'log "ERROR" "Deployment failed at line $LINENO"' ERR

# Run main deployment
main