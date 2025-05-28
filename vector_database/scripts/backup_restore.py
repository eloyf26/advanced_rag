#!/usr/bin/env python3
"""
Backup and restore utilities for Supabase Vector Database
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

class DatabaseBackup:
    """Handle backup and restore operations"""
    
    def __init__(self):
        # Load environment
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        # Get connection details
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.db_password = os.getenv('SUPABASE_DB_PASSWORD')
        
        if not self.supabase_url or not self.db_password:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_DB_PASSWORD")
        
        # Extract project ID
        project_id = self.supabase_url.replace('https://', '').split('.')[0]
        
        self.db_config = {
            'host': f'db.{project_id}.supabase.co',
            'database': 'postgres',
            'user': 'postgres',
            'password': self.db_password,
            'sslmode': 'require'
        }
    
    def backup_data(self, output_file: str = None):
        """Backup rag_documents table to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'backup_rag_documents_{timestamp}.json'
        
        print(f"Starting backup to {output_file}...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM rag_documents")
            total_count = cursor.fetchone()['count']
            print(f"Total documents to backup: {total_count}")
            
            if total_count == 0:
                print("No documents to backup")
                return
            
            # Fetch data in batches
            batch_size = 1000
            offset = 0
            documents = []
            
            while offset < total_count:
                cursor.execute("""
                    SELECT 
                        id::text, content, 
                        embedding::text as embedding_text,
                        document_id::text, chunk_id::text,
                        file_path, file_name, file_type, file_size,
                        file_modified, chunk_index, total_chunks,
                        parent_node_id, chunk_type, word_count, char_count,
                        processed_at, extraction_method, title, summary,
                        keywords, entities, questions_answered,
                        previous_chunk_preview, next_chunk_preview,
                        content_hash, created_at, updated_at
                    FROM rag_documents
                    ORDER BY created_at
                    LIMIT %s OFFSET %s
                """, (batch_size, offset))
                
                batch = cursor.fetchall()
                documents.extend(batch)
                offset += batch_size
                print(f"Progress: {min(offset, total_count)}/{total_count}")
            
            # Convert datetime objects to strings
            for doc in documents:
                for key, value in doc.items():
                    if hasattr(value, 'isoformat'):
                        doc[key] = value.isoformat()
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'backup_date': datetime.now().isoformat(),
                    'total_documents': len(documents),
                    'documents': documents
                }, f, indent=2, ensure_ascii=False)
            
            cursor.close()
            conn.close()
            
            print(f"✅ Backup completed: {len(documents)} documents saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            sys.exit(1)
    
    def restore_data(self, input_file: str, clear_existing: bool = False):
        """Restore rag_documents from JSON backup"""
        if not Path(input_file).exists():
            print(f"❌ Backup file not found: {input_file}")
            sys.exit(1)
        
        print(f"Loading backup from {input_file}...")
        
        try:
            # Load backup data
            with open(input_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            documents = backup_data['documents']
            print(f"Found {len(documents)} documents to restore")
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Clear existing data if requested
            if clear_existing:
                print("Clearing existing data...")
                cursor.execute("TRUNCATE TABLE rag_documents CASCADE")
                conn.commit()
            
            # Restore documents
            restored_count = 0
            for doc in documents:
                try:
                    # Convert embedding text back to vector
                    embedding_text = doc.get('embedding_text', '')
                    if embedding_text:
                        # Remove brackets and split
                        embedding_values = embedding_text.strip('[]').split(',')
                        embedding_array = [float(v) for v in embedding_values]
                    else:
                        embedding_array = [0.0] * 3072
                    
                    cursor.execute("""
                        INSERT INTO rag_documents (
                            id, content, embedding, document_id, chunk_id,
                            file_path, file_name, file_type, file_size,
                            file_modified, chunk_index, total_chunks,
                            parent_node_id, chunk_type, word_count, char_count,
                            processed_at, extraction_method, title, summary,
                            keywords, entities, questions_answered,
                            previous_chunk_preview, next_chunk_preview,
                            content_hash, created_at, updated_at
                        ) VALUES (
                            %s, %s, %s::vector(3072), %s, %s,
                            %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (id) DO NOTHING
                    """, (
                        doc.get('id'), doc.get('content'), embedding_array,
                        doc.get('document_id'), doc.get('chunk_id'),
                        doc.get('file_path'), doc.get('file_name'),
                        doc.get('file_type'), doc.get('file_size'),
                        doc.get('file_modified'), doc.get('chunk_index'),
                        doc.get('total_chunks'), doc.get('parent_node_id'),
                        doc.get('chunk_type'), doc.get('word_count'),
                        doc.get('char_count'), doc.get('processed_at'),
                        doc.get('extraction_method'), doc.get('title'),
                        doc.get('summary'), doc.get('keywords'),
                        doc.get('entities'), doc.get('questions_answered'),
                        doc.get('previous_chunk_preview'), doc.get('next_chunk_preview'),
                        doc.get('content_hash'), doc.get('created_at'),
                        doc.get('updated_at')
                    ))
                    
                    restored_count += 1
                    if restored_count % 100 == 0:
                        print(f"Progress: {restored_count}/{len(documents)}")
                        conn.commit()
                
                except Exception as e:
                    print(f"Warning: Failed to restore document {doc.get('id')}: {e}")
                    continue
            
            conn.commit()
            
            # Update BM25 stats
            print("Updating BM25 statistics...")
            cursor.execute("SELECT update_bm25_stats()")
            conn.commit()
            
            cursor.close()
            conn.close()
            
            print(f"✅ Restore completed: {restored_count} documents restored")
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            sys.exit(1)
    
    def backup_schema(self, output_dir: str = "schema_backup"):
        """Backup database schema and functions"""
        Path(output_dir).mkdir(exist_ok=True)
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get table DDL
            cursor.execute("""
                SELECT pg_get_ddl('TABLE', 'rag_documents'::regclass::oid)
            """)
            table_ddl = cursor.fetchone()[0]
            
            with open(f"{output_dir}/tables.sql", 'w') as f:
                f.write("-- Table definitions\n")
                f.write(table_ddl)
                f.write(";\n")
            
            # Get function definitions
            cursor.execute("""
                SELECT proname, pg_get_functiondef(oid)
                FROM pg_proc
                JOIN pg_namespace ON pg_proc.pronamespace = pg_namespace.oid
                WHERE pg_namespace.nspname = 'public'
                AND proname IN ('hybrid_search', 'semantic_search', 
                               'keyword_search', 'get_document_stats')
            """)
            
            with open(f"{output_dir}/functions.sql", 'w') as f:
                f.write("-- Function definitions\n")
                for name, definition in cursor.fetchall():
                    f.write(f"\n-- Function: {name}\n")
                    f.write(definition)
                    f.write(";\n")
            
            cursor.close()
            conn.close()
            
            print(f"✅ Schema backed up to {output_dir}/")
            
        except Exception as e:
            print(f"❌ Schema backup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Backup and restore Supabase Vector Database')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup database')
    backup_parser.add_argument('-o', '--output', help='Output file name')
    backup_parser.add_argument('--schema', action='store_true', help='Also backup schema')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database')
    restore_parser.add_argument('input', help='Input backup file')
    restore_parser.add_argument('--clear', action='store_true', 
                               help='Clear existing data before restore')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    backup = DatabaseBackup()
    
    if args.command == 'backup':
        backup.backup_data(args.output)
        if args.schema:
            backup.backup_schema()
    elif args.command == 'restore':
        backup.restore_data(args.input, args.clear)

if __name__ == '__main__':
    main()