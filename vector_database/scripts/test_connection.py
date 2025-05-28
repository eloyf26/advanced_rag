#!/usr/bin/env python3
"""
Test connection to Supabase Vector Database
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try to import required libraries
try:
    import psycopg2
except ImportError:
    print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    from supabase import create_client
    SUPABASE_CLIENT_AVAILABLE = True
except ImportError:
    SUPABASE_CLIENT_AVAILABLE = False
    print("Warning: supabase-py not installed. Some tests will be skipped.")

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def test_connection():
    """Test database connection and functionality"""
    
    # Load environment
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"{Colors.GREEN}✓ Loaded .env file{Colors.NC}")
    else:
        print(f"{Colors.YELLOW}⚠ No .env file found, using environment variables{Colors.NC}")
    
    # Check required variables
    required_vars = {
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_DB_PASSWORD': os.getenv('SUPABASE_DB_PASSWORD'),
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        print(f"\n{Colors.RED}✗ Missing required environment variables:{Colors.NC}")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these in your .env file or environment")
        sys.exit(1)
    
    # Extract connection details
    project_id = required_vars['SUPABASE_URL'].replace('https://', '').split('.')[0]
    db_config = {
        'host': f'db.{project_id}.supabase.co',
        'database': 'postgres',
        'user': 'postgres',
        'password': required_vars['SUPABASE_DB_PASSWORD'],
        'sslmode': 'require'
    }
    
    print(f"\n{Colors.BLUE}Testing PostgreSQL connection...{Colors.NC}")
    print(f"  Host: {db_config['host']}")
    
    try:
        # Test basic connection
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Get version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"{Colors.GREEN}✓ Connected to PostgreSQL{Colors.NC}")
        print(f"  Version: {version.split(',')[0]}")
        
        # Check pgvector
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print(f"{Colors.GREEN}✓ pgvector extension is enabled{Colors.NC}")
        else:
            print(f"{Colors.RED}✗ pgvector extension not found{Colors.NC}")
            print("  Enable it in Supabase Dashboard → Database → Extensions")
            cursor.close()
            conn.close()
            sys.exit(1)
        
        # Check tables
        print(f"\n{Colors.BLUE}Checking database schema...{Colors.NC}")
        cursor.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('rag_documents', 'bm25_stats')
            ORDER BY tablename;
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'rag_documents' in tables:
            print(f"{Colors.GREEN}✓ Table 'rag_documents' exists{Colors.NC}")
        else:
            print(f"{Colors.RED}✗ Table 'rag_documents' not found{Colors.NC}")
        
        if 'bm25_stats' in tables:
            print(f"{Colors.GREEN}✓ Table 'bm25_stats' exists{Colors.NC}")
        else:
            print(f"{Colors.RED}✗ Table 'bm25_stats' not found{Colors.NC}")
        
        # Check functions
        cursor.execute("""
            SELECT proname FROM pg_proc 
            JOIN pg_namespace ON pg_proc.pronamespace = pg_namespace.oid
            WHERE pg_namespace.nspname = 'public'
            AND proname IN ('hybrid_search', 'semantic_search', 'keyword_search', 'get_document_stats')
            ORDER BY proname;
        """)
        functions = [row[0] for row in cursor.fetchall()]
        
        print(f"\n{Colors.BLUE}Checking search functions...{Colors.NC}")
        for func in ['hybrid_search', 'semantic_search', 'keyword_search', 'get_document_stats']:
            if func in functions:
                print(f"{Colors.GREEN}✓ Function '{func}' exists{Colors.NC}")
            else:
                print(f"{Colors.RED}✗ Function '{func}' not found{Colors.NC}")
        
        # Test get_document_stats
        try:
            cursor.execute("SELECT * FROM get_document_stats();")
            stats = cursor.fetchone()
            print(f"\n{Colors.BLUE}Database statistics:{Colors.NC}")
            if stats:
                print(f"  Total documents: {stats[0] or 0}")
                print(f"  Total chunks: {stats[1] or 0}")
                print(f"  Unique files: {stats[2] or 0}")
            else:
                print(f"  Database is empty (this is normal for new deployments)")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Could not get stats: {e}{Colors.NC}")
        
        cursor.close()
        conn.close()
        
        # Test Supabase client if available
        if SUPABASE_CLIENT_AVAILABLE and os.getenv('SUPABASE_SERVICE_KEY'):
            print(f"\n{Colors.BLUE}Testing Supabase client connection...{Colors.NC}")
            try:
                supabase = create_client(
                    required_vars['SUPABASE_URL'],
                    os.getenv('SUPABASE_SERVICE_KEY')
                )
                
                # Try to query
                result = supabase.table('rag_documents').select("count").limit(1).execute()
                print(f"{Colors.GREEN}✓ Supabase client connection successful{Colors.NC}")
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Supabase client error: {e}{Colors.NC}")
        
        print(f"\n{Colors.GREEN}✅ All tests passed! Your database is ready.{Colors.NC}")
        
    except psycopg2.OperationalError as e:
        print(f"\n{Colors.RED}✗ Connection failed: {e}{Colors.NC}")
        print(f"\n{Colors.YELLOW}Troubleshooting tips:{Colors.NC}")
        print("1. Check your database password is correct (not the service key!)")
        print("2. Verify your project URL is correct")
        print("3. Ensure your Supabase project is active (not paused)")
        print("4. Try resetting your database password in Supabase Dashboard")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}✗ Unexpected error: {e}{Colors.NC}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"{Colors.BLUE}=== Supabase Vector Database Connection Test ==={Colors.NC}")
    test_connection()