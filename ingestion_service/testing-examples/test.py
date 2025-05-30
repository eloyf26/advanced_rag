import os
import time
import requests
import json
from pathlib import Path
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your service runs on a different host/port
TEST_FOLDER = "./test_documents"  # Local folder with test documents

def setup_test_folder():
    """Create test folder with sample files if it doesn't exist."""
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
        
        # Create a sample text file
        with open(f"{TEST_FOLDER}/sample.txt", "w") as f:
            f.write("""This is a sample document for testing the ingestion service.
It contains multiple lines to test chunking functionality.
The service should process this file successfully.
Let's add some more content to make it interesting.
Machine learning is a fascinating field of study.
Natural language processing helps computers understand human language.""")
        
        # Create a markdown file
        with open(f"{TEST_FOLDER}/readme.md", "w") as f:
            f.write("""# Test Document

This is a test markdown file with various elements.

## Section 1: Introduction
This section introduces the document.

## Section 2: Content
- Bullet point 1
- Bullet point 2
- Bullet point 3

### Subsection 2.1
Some code example:
```python
def hello_world():
    print("Hello, World!")
```

## Section 3: Conclusion
This concludes our test document.""")
        
        # Create a Python file for code testing
        with open(f"{TEST_FOLDER}/example.py", "w") as f:
            f.write("""# Example Python file for testing code chunking

import os
import sys

def calculate_sum(a, b):
    '''Calculate the sum of two numbers'''
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        self.result += value
        return self.result
    
    def reset(self):
        self.result = 0

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5))
    print(calc.add(3))
""")
        
        print(f"✅ Created test folder at {TEST_FOLDER} with sample files")
    else:
        print(f"📁 Using existing test folder at {TEST_FOLDER}")
        # List existing files
        files = list(Path(TEST_FOLDER).glob("*"))
        print(f"   Found {len(files)} files:")
        for f in files[:5]:  # Show first 5 files
            print(f"   - {f.name}")

def check_health():
    """Check if the ingestion service is healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Service is healthy")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            if 'batch_api' in health_data:
                print(f"   Batch API: {'Enabled' if health_data['batch_api'].get('enabled') else 'Disabled'}")
            return True
        else:
            print(f"❌ Service health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Could not connect to the service: {e}")
        return False

def get_service_stats():
    """Get service statistics."""
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("\n📊 Service Statistics:")
            
            # Database stats
            if 'database_stats' in stats:
                db_stats = stats['database_stats']
                print(f"   Database:")
                print(f"   - Total chunks: {db_stats.get('total_chunks', 0)}")
                print(f"   - Total documents: {db_stats.get('total_documents', 0)}")
                print(f"   - Unique files: {db_stats.get('unique_files', 0)}")
            
            # Task stats
            if 'task_stats' in stats:
                task_stats = stats['task_stats']
                print(f"   Tasks:")
                print(f"   - Total: {task_stats.get('total_tasks', 0)}")
                print(f"   - Completed: {task_stats.get('completed_tasks', 0)}")
                print(f"   - Failed: {task_stats.get('failed_tasks', 0)}")
            
            # Batch API stats
            if 'batch_api_stats' in stats and stats['batch_api_stats']:
                batch_stats = stats['batch_api_stats']
                print(f"   Batch API:")
                print(f"   - Regular API embeddings: {batch_stats.get('embeddings_via_regular_api', 0)}")
                print(f"   - Batch API embeddings: {batch_stats.get('embeddings_via_batch_api', 0)}")
                if 'estimated_savings_usd' in batch_stats:
                    print(f"   - Estimated savings: ${batch_stats['estimated_savings_usd']:.2f}")
        else:
            print(f"❌ Failed to get service stats: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Error getting stats: {e}")

def ingest_files(file_paths=None):
    """Ingest specific files."""
    if file_paths is None:
        # Use default test files
        file_paths = []
        for ext in ['txt', 'md', 'py']:
            files = list(Path(TEST_FOLDER).glob(f"*.{ext}"))
            # Convert to absolute paths with forward slashes for cross-platform compatibility
            for f in files[:2]:  # Max 2 files per type
                # Use as_posix() to ensure forward slashes
                file_paths.append(str(f.absolute()).replace('\\', '/'))
    
    if not file_paths:
        print("❌ No files found to ingest")
        return None
    
    print(f"\n📄 Ingesting {len(file_paths)} files:")
    for fp in file_paths:
        print(f"   - {Path(fp).name}")
    
    # Ensure all paths use forward slashes
    file_paths = [fp.replace('\\', '/') for fp in file_paths]
    
    payload = {
        "file_paths": file_paths,
        "batch_size": 10  # Small batch size for testing
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ingest/files", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ingestion started")
            print(f"   Task ID: {result.get('task_id')}")
            print(f"   Embedding method: {result.get('embedding_method', 'unknown')}")
            return result.get("task_id")
        else:
            print(f"❌ Failed to ingest files: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def check_task_status(task_id, show_details=True):
    """Check the status of a task."""
    if not task_id:
        return None
    
    try:
        response = requests.get(f"{BASE_URL}/ingest/status/{task_id}", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if show_details:
                print(f"\n🔍 Task Status: {result.get('status', 'unknown')}")
                print(f"   Task ID: {task_id}")
                print(f"   Processed: {len(result.get('processed', []))}")
                print(f"   Failed: {len(result.get('failed', []))}")
                print(f"   Total chunks: {result.get('total_chunks', 0)}")
                print(f"   Embedding method: {result.get('embedding_method', 'unknown')}")
                if result.get('error'):
                    print(f"   Error: {result['error']}")
            return result
        else:
            print(f"❌ Failed to check task status: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"❌ Error checking status: {e}")
        return None

def wait_for_task_completion(task_id, max_attempts=30, delay=2):
    """Wait for a task to complete."""
    if not task_id:
        return None
    
    print(f"\n⏳ Waiting for task {task_id} to complete...")
    
    for attempt in range(max_attempts):
        task_status = check_task_status(task_id, show_details=False)
        if not task_status:
            return None
            
        status = task_status.get("status")
        if status == "completed":
            print(f"✅ Task completed successfully!")
            check_task_status(task_id, show_details=True)  # Show final details
            return task_status
        elif status == "failed":
            print(f"❌ Task failed!")
            check_task_status(task_id, show_details=True)  # Show error details
            return task_status
        else:
            # Show progress indicator
            print(f"   Status: {status} (attempt {attempt+1}/{max_attempts})", end='\r')
            time.sleep(delay)
    
    print(f"\n⏱️ Timeout: Task did not complete within {max_attempts * delay} seconds")
    return None

def test_batch_api():
    """Test batch API functionality."""
    print("\n🔄 Testing Batch API...")
    
    # Check batch status
    try:
        response = requests.get(f"{BASE_URL}/batch/status", timeout=5)
        if response.status_code == 200:
            batch_status = response.json()
            print(f"✅ Batch API Status:")
            print(f"   Enabled: {batch_status.get('batch_api_enabled', False)}")
            print(f"   Threshold: {batch_status.get('batch_threshold', 'N/A')} chunks")
            if batch_status.get('batch_jobs'):
                print(f"   Jobs: {batch_status['batch_jobs']}")
            if batch_status.get('pending_embeddings', 0) > 0:
                print(f"   Pending embeddings: {batch_status['pending_embeddings']}")
        else:
            print(f"❌ Failed to get batch status")
    except Exception as e:
        print(f"❌ Error checking batch status: {e}")

def test_single_file():
    """Test ingesting a single file for debugging."""
    print("\n🧪 Testing Single File Ingestion")
    
    # Create a test file
    test_file = Path(TEST_FOLDER) / "test_single.txt"
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a single test file for debugging purposes.\nIt should be processed without any issues.")
    
    print(f"✅ Created test file: {test_file.name}")
    
    # Get absolute path with forward slashes
    file_path = str(test_file.absolute()).replace('\\', '/')
    print(f"📍 Full path: {file_path}")
    
    # Check if file exists
    if not test_file.exists():
        print("❌ File doesn't exist!")
        return False
    
    print(f"✅ File exists: {test_file.stat().st_size} bytes")
    
    # Try to ingest
    payload = {
        "file_paths": [file_path],
        "batch_size": 10
    }
    
    print(f"📤 Sending request with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/ingest/files", json=payload, timeout=10)
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            
            task_id = result.get("task_id")
            if task_id:
                # Wait for completion
                final_status = wait_for_task_completion(task_id, max_attempts=10, delay=1)
                return final_status and final_status.get('status') == 'completed'
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def run_minimal_test():
    """Run a minimal test to verify basic functionality."""
    print("\n" + "="*60)
    print("🧪 Running Minimal Test")
    print("="*60)
    
    # Check health
    if not check_health():
        print("\n❌ Service is not healthy. Please check:")
        print("   1. Is the service running? (python main.py)")
        print("   2. Are all environment variables set correctly?")
        print("   3. Can the service connect to Supabase?")
        return False
    
    # Create a simple test file
    test_file = Path(TEST_FOLDER) / "minimal_test.txt"
    with open(test_file, "w") as f:
        f.write("This is a minimal test file. It should process successfully.")
    
    # Ingest the file - use forward slashes for path
    file_path = str(test_file.absolute()).replace('\\', '/')
    task_id = ingest_files([file_path])
    if not task_id:
        print("\n❌ Failed to start ingestion")
        return False
    
    # Wait for completion
    result = wait_for_task_completion(task_id, max_attempts=10, delay=1)
    if not result:
        print("\n❌ Task did not complete")
        return False
    
    if result.get('status') == 'completed' and result.get('total_chunks', 0) > 0:
        print("\n✅ Minimal test passed!")
        return True
    else:
        print("\n❌ Minimal test failed")
        return False

def run_comprehensive_test():
    """Run comprehensive tests."""
    print("\n" + "="*60)
    print("🚀 Starting Comprehensive Ingestion Service Test")
    print("="*60)
    print(f"   Service URL: {BASE_URL}")
    print(f"   Test folder: {TEST_FOLDER}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Setup
    setup_test_folder()
    
    # Health check
    if not check_health():
        print("\n⚠️ Running minimal test first...")
        if not run_minimal_test():
            print("\n❌ Service is not working properly. Stopping tests.")
            return
    
    # Get initial stats
    get_service_stats()
    
    # Test different file types
    print("\n📋 Test Suite:")
    
    # Test 1: Text and Markdown files
    print("\n1️⃣ Testing text and markdown files...")
    files = [
        Path(TEST_FOLDER) / "sample.txt",
        Path(TEST_FOLDER) / "readme.md",
        Path(TEST_FOLDER) / "cartaconsumo.pdf"
    ]
    # Convert to proper paths with forward slashes
    file_paths = [str(f.absolute()).replace('\\', '/') for f in files if f.exists()]
    if file_paths:
        task_id = ingest_files(file_paths)
        if task_id:
            wait_for_task_completion(task_id)
    else:
        print("   ⚠️ No test files found")
    
    # Test 2: Code files
    print("\n2️⃣ Testing code files...")
    code_file = Path(TEST_FOLDER) / "example.py"
    if code_file.exists():
        file_path = str(code_file.absolute()).replace('\\', '/')
        task_id = ingest_files([file_path])
        if task_id:
            wait_for_task_completion(task_id)
    else:
        print("   ⚠️ No code file found")
    
    # Test 3: Batch API
    test_batch_api()
    
    # Final stats
    print("\n📊 Final Statistics:")
    get_service_stats()
    
    print("\n" + "="*60)
    print("✅ Test Suite Completed")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Ingestion Service")
    parser.add_argument("--minimal", action="store_true", help="Run minimal test only")
    parser.add_argument("--single", action="store_true", help="Test single file ingestion")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    args = parser.parse_args()
    
    BASE_URL = args.url
    
    # Show current working directory for debugging
    if args.debug:
        print(f"🔍 Debug Information:")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Test folder path: {Path(TEST_FOLDER).absolute()}")
        print(f"   Platform: {os.name}")
        print(f"   Path separator: {os.sep}")
    
    if args.single:
        test_single_file()
    elif args.minimal:
        run_minimal_test()
    else:
        run_comprehensive_test()