import os
import time
import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your service runs on a different host/port
TEST_FOLDER = "./test_documents"  # Local folder with test documents

def setup_test_folder():
    """Create test folder with sample files if it doesn't exist."""
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
        
        # Create a sample text file
        with open(f"{TEST_FOLDER}/sample.txt", "w") as f:
            f.write("This is a sample document for testing the ingestion service.")
        
        # Create a markdown file
        with open(f"{TEST_FOLDER}/readme.md", "w") as f:
            f.write("# Test Document\n\nThis is a test markdown file.")
        
        print(f"Created test folder at {TEST_FOLDER} with sample files")
    else:
        print(f"Using existing test folder at {TEST_FOLDER}")

def check_health():
    """Check if the ingestion service is healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Service is healthy:", json.dumps(response.json(), indent=2))
            return True
        else:
            print("❌ Service health check failed:", response.status_code)
            return False
    except requests.RequestException as e:
        print(f"❌ Could not connect to the service: {e}")
        return False

def get_service_stats():
    """Get service statistics."""
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            print("📊 Service statistics:", json.dumps(response.json(), indent=2))
        else:
            print("❌ Failed to get service stats:", response.status_code)
    except requests.RequestException as e:
        print(f"❌ Error: {e}")

def ingest_files():
    """Ingest specific files."""
    file_paths = [
        os.path.abspath(f"{TEST_FOLDER}/sample.txt"),
        os.path.abspath(f"{TEST_FOLDER}/readme.md")
    ]
    
    payload = {
        "file_paths": file_paths,
        "batch_size": 10  # Small batch size for testing
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ingest/files", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"📄 Files ingestion started: {result}")
            return result.get("task_id")
        else:
            print(f"❌ Failed to ingest files: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def ingest_directory():
    """Ingest an entire directory."""
    payload = {
        "directory_path": os.path.abspath(TEST_FOLDER),
        "recursive": True,
        "file_extensions": ["txt", "md"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ingest/directory", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"📁 Directory ingestion started: {result}")
            return result.get("task_id")
        else:
            print(f"❌ Failed to ingest directory: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def upload_and_ingest():
    """Upload and ingest files."""
    files = {
        'file1': open(f"{TEST_FOLDER}/sample.txt", 'rb'),
        'file2': open(f"{TEST_FOLDER}/readme.md", 'rb')
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ingest/upload", files=files)
        if response.status_code == 200:
            result = response.json()
            print(f"📤 Upload and ingest started: {result}")
            return result.get("task_id")
        else:
            print(f"❌ Failed to upload and ingest: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        # Close file handles
        for f in files.values():
            f.close()

def check_task_status(task_id):
    """Check the status of a task."""
    if not task_id:
        return None
    
    try:
        response = requests.get(f"{BASE_URL}/ingest/status/{task_id}")
        if response.status_code == 200:
            result = response.json()
            print(f"🔍 Task status: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Failed to check task status: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def check_batch_status():
    """Check the status of batch processing."""
    try:
        response = requests.get(f"{BASE_URL}/batch/status")
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Batch status: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Failed to check batch status: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def list_batch_jobs():
    """List batch jobs."""
    try:
        response = requests.get(f"{BASE_URL}/batch/jobs?status=pending&limit=10")
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Batch jobs: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Failed to list batch jobs: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def process_pending_batches():
    """Process pending batches."""
    try:
        response = requests.post(f"{BASE_URL}/batch/process-pending")
        if response.status_code == 200:
            result = response.json()
            print(f"⚙️ Processing pending batches: {result}")
            return result.get("task_id")
        else:
            print(f"❌ Failed to process pending batches: {response.status_code}")
            print(response.text)
            return None
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def wait_for_task_completion(task_id, max_attempts=10, delay=2):
    """Wait for a task to complete, checking status periodically."""
    if not task_id:
        return None
    
    print(f"⏳ Waiting for task {task_id} to complete...")
    
    for attempt in range(max_attempts):
        task_status = check_task_status(task_id)
        if not task_status:
            return None
            
        status = task_status.get("status")
        if status == "completed":
            print(f"✅ Task completed successfully")
            return task_status
        elif status == "failed":
            print(f"❌ Task failed")
            return task_status
        elif status in ["started", "processing"]:
            print(f"⏳ Task still in progress (attempt {attempt+1}/{max_attempts}), waiting {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"❓ Unknown task status: {status}")
            return task_status
    
    print(f"⏱️ Max attempts reached, task did not complete in time")
    return None

def run_test_simulation():
    """Run a complete test simulation."""
    print("=" * 50)
    print("🚀 Starting Ingestion Service Test Simulation")
    print("=" * 50)
    
    # Setup test environment
    setup_test_folder()
    
    # Check if service is available
    if not check_health():
        print("Cannot continue with tests as service is not available")
        return
    
    # Get initial service stats
    print("\n📊 Initial Service Stats:")
    get_service_stats()
    
    # Test 1: Ingest specific files
    print("\n📄 Test 1: Ingesting specific files")
    task_id_1 = ingest_files()
    wait_for_task_completion(task_id_1)
    
    # Test 2: Ingest directory
    print("\n📁 Test 2: Ingesting directory")
    task_id_2 = ingest_directory()
    wait_for_task_completion(task_id_2)
    
    # Test 3: Upload and ingest
    print("\n📤 Test 3: Upload and ingest")
    task_id_3 = upload_and_ingest()
    wait_for_task_completion(task_id_3)
    
    # Test 4: Check batch status
    print("\n📊 Test 4: Checking batch status")
    check_batch_status()
    
    # Test 5: List batch jobs
    print("\n📋 Test 5: Listing batch jobs")
    list_batch_jobs()
    
    # Test 6: Process pending batches
    print("\n⚙️ Test 6: Processing pending batches")
    task_id_6 = process_pending_batches()
    
    # Final service stats
    print("\n📊 Final Service Stats:")
    get_service_stats()
    
    print("\n" + "=" * 50)
    print("🏁 Test Simulation Completed")
    print("=" * 50)

if __name__ == "__main__":
    run_test_simulation()