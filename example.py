'''
Before running this, you must have completed the previous steps:
1. Supabase Setup: 
    - Install lightweight postgres: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads (only cli tools, last checkbox in the list)
    - Create a new project
    - Create vector_database/.env with the following:
        - SUPABASE_URL=
        - SUPABASE_SERVICE_KEY=
    - Add postgres to path, for example: C:\Program Files\PostgreSQL\16\bin
    - Run .\scripts\deploy.ps1 -Verbose
1. Ingestion Service Setup: 
    
3. Agentic RAG Service
'''


import asyncio
import httpx
import os
from pathlib import Path
import time

# Configuration
INGESTION_URL = "http://localhost:8000"
RAG_URL = "http://localhost:8001"
LOCAL_FOLDER = "/path/to/your/documents"

async def ingest_local_folder():
    """Ingest all documents from a local folder"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Start ingestion
        print(f"📁 Ingesting folder: {LOCAL_FOLDER}")
        response = await client.post(
            f"{INGESTION_URL}/ingest/directory",
            json={
                "directory_path": LOCAL_FOLDER,
                "recursive": True,
                "file_extensions": ["pdf", "docx", "txt", "md", "csv"]
            }
        )
        task_id = response.json()["task_id"]
        print(f"✅ Ingestion started: {task_id}")
        
        # 2. Monitor progress
        while True:
            status_response = await client.get(
                f"{INGESTION_URL}/ingest/status/{task_id}"
            )
            status = status_response.json()
            
            print(f"📊 Status: {status['status']} - "
                  f"Processed: {len(status['processed'])} files, "
                  f"Chunks: {status['total_chunks']}")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            await asyncio.sleep(5)
        
        if status['status'] == 'failed':
            print(f"❌ Ingestion failed: {status['error']}")
            return False
        
        print(f"✅ Ingestion completed in {status['processing_time']:.2f} seconds")
        print(f"📄 Total documents: {status['total_documents']}")
        print(f"📦 Total chunks: {status['total_chunks']}")
        return True

async def query_documents(question: str, use_agentic: bool = True):
    """Query the ingested documents"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        endpoint = "/ask" if use_agentic else "/ask/simple"
        
        print(f"\n🤔 Asking: {question}")
        print(f"🧠 Using {'agentic' if use_agentic else 'simple'} mode")
        
        response = await client.post(
            f"{RAG_URL}{endpoint}",
            json={
                "question": question,
                "enable_iteration": True,
                "enable_reflection": True,
                "enable_triangulation": True,
                "max_results": 10
            }
        )
        
        result = response.json()
        
        print(f"\n📝 Answer:")
        print(result['answer'])
        print(f"\n🎯 Confidence: {result['confidence']:.2%}")
        print(f"📚 Sources used: {len(result['sources'])}")
        
        if use_agentic and 'follow_up_suggestions' in result:
            print(f"\n💡 Follow-up suggestions:")
            for i, suggestion in enumerate(result['follow_up_suggestions'][:3]):
                print(f"  {i+1}. {suggestion}")
        
        return result

async def main():
    """Complete workflow example"""
    # Step 1: Ingest documents
    success = await ingest_local_folder()
    if not success:
        return
    
    # Wait a moment for indexing
    print("\n⏳ Waiting for indexing...")
    await asyncio.sleep(5)
    
    # Step 2: Query the documents
    questions = [
        "What are the main topics covered in these documents?",
        "Summarize the key findings across all documents",
        "What methodologies or approaches are discussed?",
        "Are there any contradictions or disagreements between documents?"
    ]
    
    for question in questions:
        await query_documents(question, use_agentic=True)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())