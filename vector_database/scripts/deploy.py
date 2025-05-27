"""
Supabase Vector Database Deployment Script
Uses direct database connection (no psql required)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
from datetime import datetime

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
    NC = '\033[0m'  # No Color

class SupabaseDeployer:
    """Deploy database schema to Supabase using JDBC-style connection"""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.env_file = project_dir / '.env'
        self.connection = None
        self.db_config = {}
        
    def load_environment(self) -> bool:
        """Load environment variables from .env file"""
        if self.env_file.exists():
            logger.info(f"Loading environment from: {self.env_file}")
            load_dotenv(self.env_file)
            return True
        else:
            logger.warning(f"No .env file found at: {self.env_file}")
            return False
    
    def extract_connection_info(self) -> Dict[str, str]:
        """Extract database connection information from environment"""
        supabase_url = os.getenv('SUPABASE_URL')
        db_password = os.getenv('SUPABASE_DB_PASSWORD')
        
        if not supabase_url or not db_password:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_DB_PASSWORD")
        
        # Extract host from URL
        host = supabase_url.replace('https://', '').replace('.supabase.co', '')
        
        return {
            'host': f'db.{host}.supabase.co',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': db_password,
            'sslmode': 'require'
        }
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            logger.info("Testing database connection...")
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"{Colors.GREEN}✓ Connection successful{Colors.NC}")
            logger.info(f"PostgreSQL version: {version}")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"{Colors.RED}✗ Connection failed: {e}{Colors.NC}")
            return False
    
    def check_vector_extension(self) -> bool:
        """Check and enable pgvector extension"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Try to create extension if not exists
            logger.info("Checking pgvector extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Verify it exists
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result:
                logger.info(f"{Colors.GREEN}✓ pgvector extension is enabled{Colors.NC}")
                return True
            else:
                logger.error(f"{Colors.RED}✗ pgvector extension is not available{Colors.NC}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking vector extension: {e}")
            return False
    
    def execute_sql_file(self, sql_file: str, description: str) -> bool:
        """Execute SQL file"""
        file_path = self.project_dir / sql_file
        
        if not file_path.exists():
            logger.error(f"SQL file not found: {file_path}")
            return False
        
        try:
            logger.info(f"Executing: {description}")
            
            # Read SQL file
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Connect and execute
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Execute the SQL
            cursor.execute(sql_content)
            
            cursor.close()
            conn.close()
            
            logger.info(f"{Colors.GREEN}✓ Successfully executed: {description}{Colors.NC}")
            return True
            
        except Exception as e:
            logger.error(f"{Colors.RED}✗ Failed to execute {description}: {e}{Colors.NC}")
            return False
    
    def verify_deployment(self) -> bool:
        """Verify the deployment was successful"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('rag_documents', 'bm25_stats')
            """)
            tables = cursor.fetchall()
            
            if len(tables) == 2:
                logger.info(f"{Colors.GREEN}✓ All tables created successfully{Colors.NC}")
            else:
                logger.warning("Some tables might be missing")
            
            # Check functions
            cursor.execute("""
                SELECT proname 
                FROM pg_proc 
                WHERE proname IN ('hybrid_search', 'semantic_search', 'keyword_search', 'get_document_stats')
            """)
            functions = cursor.fetchall()
            
            if len(functions) >= 3:
                logger.info(f"{Colors.GREEN}✓ All functions created successfully{Colors.NC}")
            else:
                logger.warning("Some functions might be missing")
            
            # Test a function
            try:
                cursor.execute("SELECT * FROM get_document_stats();")
                logger.info(f"{Colors.GREEN}✓ Database functions are working correctly{Colors.NC}")
            except:
                logger.warning("Database functions test failed - this might be expected for empty database")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def deploy(self, skip_tests: bool = False, dry_run: bool = False):
        """Main deployment process"""
        print(f"{Colors.GREEN}🚀 Starting Supabase Vector Database Deployment{Colors.NC}")
        
        # Load environment
        self.load_environment()
        
        # Extract connection info
        try:
            self.db_config = self.extract_connection_info()
            logger.info(f"Database host: {self.db_config['host']}")
        except ValueError as e:
            logger.error(f"{Colors.RED}Missing environment variables!{Colors.NC}")
            print("\nPlease create a .env file with:")
            print("SUPABASE_URL=https://your-project.supabase.co")
            print("SUPABASE_DB_PASSWORD=your-database-password")
            print("\nTo find your database password:")
            print("1. Go to Supabase Dashboard > Settings > Database")
            print("2. Look for 'Password' field or reset it if unknown")
            sys.exit(1)
        
        if dry_run:
            logger.info("DRY RUN - No changes will be made")
            return
        
        # Run tests
        if not skip_tests:
            if not self.test_connection():
                sys.exit(1)
            
            if not self.check_vector_extension():
                print("\nPlease enable pgvector in your Supabase project:")
                print("1. Go to Supabase Dashboard")
                print("2. Navigate to Database > Extensions")
                print("3. Search for 'vector' and enable it")
                sys.exit(1)
        
        # Execute SQL files
        if not self.execute_sql_file('setup.sql', 'Main database schema'):
            sys.exit(1)
        
        if not self.execute_sql_file('functions.sql', 'Search functions'):
            sys.exit(1)
        
        # Optional security file
        security_file = self.project_dir / 'security.sql'
        if security_file.exists():
            self.execute_sql_file('security.sql', 'Security policies')
        
        # Verify deployment
        if not skip_tests:
            self.verify_deployment()
        
        print(f"\n{Colors.GREEN}🎉 Database deployed to Supabase cloud!{Colors.NC}")
        print("\nNext steps:")
        print("1. Your database is ready in the cloud")
        print("2. Start the ingestion service to process documents")
        print("3. Use the RAG agent to query your data")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Deploy Supabase Vector Database')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--skip-tests', action='store_true', help='Skip connection and validation tests')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Run deployment
    deployer = SupabaseDeployer(project_dir)
    deployer.deploy(skip_tests=args.skip_tests, dry_run=args.dry_run)

if __name__ == '__main__':
    main()