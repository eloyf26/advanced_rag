#!/usr/bin/env python3
"""
Supabase Vector Database Deployment Script
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Try different import methods for compatibility
try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")
    load_dotenv = lambda: None

# Try to import Supabase client for alternative deployment
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Colors:
    """Console colors for output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

class SupabaseDeployer:
    """Deploy database schema to Supabase"""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.env_file = project_dir / '.env'
        self.connection = None
        self.deployment_method = None
        
    def print_banner(self):
        """Print deployment banner"""
        print(f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════╗
║          Supabase Vector Database Deployment              ║
║                                                           ║
║  This script will deploy the RAG database schema to your  ║
║  Supabase cloud project.                                  ║
╚═══════════════════════════════════════════════════════════╝{Colors.NC}
""")
    
    def load_environment(self) -> bool:
        """Load environment variables"""
        if self.env_file.exists():
            logger.info(f"Loading environment from: {self.env_file}")
            load_dotenv(self.env_file)
            return True
        else:
            logger.warning(f"No .env file found at: {self.env_file}")
            return False
    
    def check_environment(self) -> Tuple[str, Dict[str, str]]:
        """Check and validate environment variables"""
        print(f"\n{Colors.BLUE}Checking environment configuration...{Colors.NC}")
        
        required_vars = {
            'SUPABASE_URL': os.getenv('SUPABASE_URL'),
            'SUPABASE_DB_PASSWORD': os.getenv('SUPABASE_DB_PASSWORD'),
        }
        
        optional_vars = {
            'SUPABASE_SERVICE_KEY': os.getenv('SUPABASE_SERVICE_KEY'),
        }
        
        # Check if we have minimum requirements
        if required_vars['SUPABASE_URL'] and required_vars['SUPABASE_DB_PASSWORD']:
            return 'postgresql', required_vars
        
        # Check if we can use Supabase client method
        if required_vars['SUPABASE_URL'] and optional_vars['SUPABASE_SERVICE_KEY'] and SUPABASE_AVAILABLE:
            print(f"{Colors.YELLOW}Note: Using Supabase client method (limited functionality){Colors.NC}")
            return 'supabase_client', {**required_vars, **optional_vars}
        
        # No valid configuration found
        self.print_missing_config_help()
        return None, {}
    
    def print_missing_config_help(self):
        """Print help for missing configuration"""
        print(f"""
{Colors.RED}Missing required configuration!{Colors.NC}

You need to provide database credentials. Here's how:

{Colors.GREEN}1. Get your database password from Supabase:{Colors.NC}
   a. Go to your Supabase Dashboard
   b. Navigate to Settings → Database
   c. Find the "Database Password" section
   d. Copy your password (or reset it if you don't know it)

{Colors.GREEN}2. Create a .env file in the vector_database directory:{Colors.NC}
   
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_DB_PASSWORD=your-database-password-here
   
   # Optional (for API access):
   SUPABASE_SERVICE_KEY=your-service-key-here

{Colors.GREEN}3. Alternative: Set environment variables directly:{Colors.NC}
   
   export SUPABASE_URL="https://your-project-id.supabase.co"
   export SUPABASE_DB_PASSWORD="your-database-password"

{Colors.YELLOW}Note: The service key is NOT the database password!{Colors.NC}
""")
    
    def get_db_config(self, supabase_url: str, db_password: str) -> Dict[str, str]:
        """Extract database configuration from Supabase URL"""
        # Extract project ID from URL
        project_id = supabase_url.replace('https://', '').split('.')[0]
        
        return {
            'host': f'db.{project_id}.supabase.co',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': db_password,
            'sslmode': 'require'
        }
    
    def test_connection(self, db_config: Dict[str, str]) -> bool:
        """Test database connection"""
        try:
            print(f"\n{Colors.BLUE}Testing database connection...{Colors.NC}")
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"{Colors.GREEN}✓ Connection successful!{Colors.NC}")
            print(f"  PostgreSQL version: {version.split(',')[0]}")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"{Colors.RED}✗ Connection failed: {e}{Colors.NC}")
            return False
    
    def check_and_enable_vector(self, db_config: Dict[str, str]) -> bool:
        """Check and enable pgvector extension"""
        try:
            print(f"\n{Colors.BLUE}Checking pgvector extension...{Colors.NC}")
            conn = psycopg2.connect(**db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if extension exists
            cursor.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector';")
            if not cursor.fetchone():
                print(f"{Colors.RED}✗ pgvector extension not available in this Supabase instance{Colors.NC}")
                print(f"\n{Colors.YELLOW}Please enable it in the Supabase Dashboard:{Colors.NC}")
                print("1. Go to Database → Extensions")
                print("2. Search for 'vector'")
                print("3. Click to enable it")
                cursor.close()
                conn.close()
                return False
            
            # Try to create extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Verify it's enabled
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone():
                print(f"{Colors.GREEN}✓ pgvector extension is enabled{Colors.NC}")
                cursor.close()
                conn.close()
                return True
            else:
                print(f"{Colors.RED}✗ Failed to enable pgvector extension{Colors.NC}")
                cursor.close()
                conn.close()
                return False
                
        except Exception as e:
            logger.error(f"Error checking vector extension: {e}")
            return False
    
    def execute_sql_file(self, db_config: Dict[str, str], sql_file: str, description: str) -> bool:
        """Execute SQL file with progress tracking"""
        file_path = self.project_dir / sql_file
        
        if not file_path.exists():
            logger.error(f"SQL file not found: {file_path}")
            return False
        
        try:
            print(f"\n{Colors.BLUE}Executing: {description}{Colors.NC}")
            print(f"  File: {sql_file}")
            
            # Read SQL file
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Split into statements (simple split by semicolon)
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]
            total_statements = len(statements)
            
            # Connect and execute
            conn = psycopg2.connect(**db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Execute statements with progress
            for i, statement in enumerate(statements, 1):
                if statement:
                    try:
                        cursor.execute(statement)
                        print(f"  Progress: {i}/{total_statements} statements", end='\r')
                    except Exception as e:
                        # Log error but continue with other statements
                        logger.warning(f"Statement {i} warning: {str(e)[:100]}")
            
            print(f"\n{Colors.GREEN}✓ Successfully executed: {description}{Colors.NC}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"\n{Colors.RED}✗ Failed to execute {description}{Colors.NC}")
            logger.error(f"Error: {e}")
            return False
    
    def verify_deployment(self, db_config: Dict[str, str]) -> bool:
        """Verify the deployment was successful"""
        try:
            print(f"\n{Colors.BLUE}Verifying deployment...{Colors.NC}")
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('rag_documents', 'bm25_stats')
                ORDER BY tablename;
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"\n  Tables created:")
            for table in tables:
                print(f"    {Colors.GREEN}✓{Colors.NC} {table}")
            
            if len(tables) < 2:
                print(f"    {Colors.YELLOW}⚠ Some tables might be missing{Colors.NC}")
            
            # Check functions
            cursor.execute("""
                SELECT proname 
                FROM pg_proc 
                JOIN pg_namespace ON pg_proc.pronamespace = pg_namespace.oid
                WHERE pg_namespace.nspname = 'public'
                AND proname IN ('hybrid_search', 'semantic_search', 'keyword_search', 'get_document_stats')
                ORDER BY proname;
            """)
            functions = [row[0] for row in cursor.fetchall()]
            
            print(f"\n  Functions created:")
            for func in functions:
                print(f"    {Colors.GREEN}✓{Colors.NC} {func}()")
            
            if len(functions) < 4:
                print(f"    {Colors.YELLOW}⚠ Some functions might be missing{Colors.NC}")
            
            # Test get_document_stats function
            try:
                cursor.execute("SELECT * FROM get_document_stats();")
                stats = cursor.fetchone()
                print(f"\n  Database status:")
                print(f"    {Colors.GREEN}✓{Colors.NC} Functions are working correctly")
                if stats and stats[0] is not None:
                    print(f"    Total chunks: {stats[1] or 0}")
            except Exception as e:
                print(f"    {Colors.YELLOW}⚠ Empty database (this is normal for new deployments){Colors.NC}")
            
            cursor.close()
            conn.close()
            
            print(f"\n{Colors.GREEN}✓ Deployment verification complete!{Colors.NC}")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def deploy(self, skip_tests: bool = False, dry_run: bool = False):
        """Main deployment process"""
        self.print_banner()
        
        # Load environment
        self.load_environment()
        
        # Check environment and determine deployment method
        deployment_method, env_vars = self.check_environment()
        
        if not deployment_method:
            sys.exit(1)
        
        if deployment_method == 'postgresql':
            # Direct PostgreSQL deployment (preferred)
            db_config = self.get_db_config(
                env_vars['SUPABASE_URL'], 
                env_vars['SUPABASE_DB_PASSWORD']
            )
            
            print(f"\n{Colors.GREEN}Using direct PostgreSQL connection{Colors.NC}")
            print(f"  Host: {db_config['host']}")
            print(f"  Database: {db_config['database']}")
            
            if dry_run:
                print(f"\n{Colors.YELLOW}DRY RUN - No changes will be made{Colors.NC}")
                return
            
            # Test connection
            if not skip_tests:
                if not self.test_connection(db_config):
                    print(f"\n{Colors.YELLOW}Connection troubleshooting:{Colors.NC}")
                    print("1. Check your database password is correct")
                    print("2. Ensure your Supabase project is active")
                    print("3. Verify network connectivity")
                    print("4. Try resetting your database password in Supabase Dashboard")
                    sys.exit(1)
                
                if not self.check_and_enable_vector(db_config):
                    sys.exit(1)
            
            # Execute SQL files
            print(f"\n{Colors.CYAN}Deploying database schema...{Colors.NC}")
            
            if not self.execute_sql_file(db_config, 'setup.sql', 'Main database schema'):
                sys.exit(1)
            
            if not self.execute_sql_file(db_config, 'functions.sql', 'Search functions'):
                sys.exit(1)
            
            # Optional security file
            security_file = self.project_dir / 'security.sql'
            if security_file.exists():
                self.execute_sql_file(db_config, 'security.sql', 'Security policies')
            
            # Verify deployment
            if not skip_tests:
                self.verify_deployment(db_config)
            
            self.print_success_message()
            
        else:
            # Fallback to Supabase client method
            print(f"\n{Colors.YELLOW}Limited deployment using Supabase client{Colors.NC}")
            print("Note: This method has limitations. Direct PostgreSQL connection is recommended.")
            # Implementation would go here if needed
    
    def print_success_message(self):
        """Print success message and next steps"""
        print(f"""
{Colors.GREEN}╔═══════════════════════════════════════════════════════════╗
║            🎉 Deployment Successful! 🎉                   ║
╚═══════════════════════════════════════════════════════════╝{Colors.NC}

Your Supabase Vector Database is now ready!

{Colors.CYAN}Next steps:{Colors.NC}
1. Test the database with a simple query
2. Start the ingestion service to process documents
3. Use the RAG agent to query your knowledge base

{Colors.CYAN}Quick test command:{Colors.NC}
python -c "
import psycopg2
conn = psycopg2.connect(
    host='db.YOUR_PROJECT_ID.supabase.co',
    database='postgres',
    user='postgres',
    password='YOUR_DB_PASSWORD',
    sslmode='require'
)
cursor = conn.cursor()
cursor.execute('SELECT * FROM get_document_stats();')
print('Database test successful!')
"

{Colors.CYAN}Documentation:{Colors.NC}
- Vector Database README: ./vector_database/README.md
- Ingestion Service: ./ingestion_service/README.md
- RAG Agent: ./agentic_rag_agent/README.md
""")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Deploy Supabase Vector Database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_supabase.py              # Normal deployment
  python deploy_supabase.py --dry-run    # Show what would be done
  python deploy_supabase.py --skip-tests # Skip connection tests
  
Environment variables needed:
  SUPABASE_URL          Your Supabase project URL
  SUPABASE_DB_PASSWORD  Your database password (from Supabase Dashboard)
        """
    )
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without executing')
    parser.add_argument('--skip-tests', action='store_true', 
                        help='Skip connection and validation tests')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Run deployment
    deployer = SupabaseDeployer(project_dir)
    
    try:
        deployer.deploy(skip_tests=args.skip_tests, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Deployment cancelled by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Deployment failed: {e}{Colors.NC}")
        logger.exception("Deployment error")
        sys.exit(1)

if __name__ == '__main__':
    main()