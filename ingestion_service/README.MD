# LlamaIndex Ingestion Service Documentation

## Overview

The LlamaIndex Ingestion Service is a comprehensive document processing pipeline designed to ingest, process, and prepare diverse file types for Retrieval-Augmented Generation (RAG) systems. It leverages LlamaIndex's powerful document processing capabilities with advanced context-aware chunking, multi-modal processing, and seamless integration with Supabase vector databases.

## 🎯 Goals

- **Universal File Support**: Process 15+ file types including documents, spreadsheets, images, audio, code, and archives
- **Context-Aware Processing**: Intelligent chunking that preserves document structure and semantic meaning
- **Multi-Modal Intelligence**: Extract text from images (OCR), transcribe audio, and handle complex document formats
- **Scalable Architecture**: Concurrent processing with configurable batch sizes and resource limits
- **Production Ready**: Comprehensive error handling, logging, and monitoring capabilities

## 🏗️ Service Architecture

### Core Components

```
📁 ingestion_service/
├── 📄 main.py                          # Service entry point
├── 📄 config.py                        # Configuration management
├── 📄 processors/
│   ├── 📄 context_aware_chunker.py     # Intelligent chunking logic
│   ├── 📄 multimodal_processor.py      # File type processors
│   └── 📄 metadata_extractor.py        # Metadata enhancement
├── 📄 storage/
│   ├── 📄 supabase_manager.py          # Database operations
│   └── 📄 vector_store.py              # Vector storage interface
├── 📄 utils/
│   ├── 📄 file_validators.py           # File validation utilities
│   ├── 📄 logger.py                    # Logging configuration
│   └── 📄 metrics.py                   # Performance monitoring
└── 📄 requirements.txt                 # Dependencies
```

### Processing Pipeline

```mermaid
graph LR
    A[File Input] --> B[File Validation]
    B --> C[Type Detection]
    C --> D[Multi-Modal Processing]
    D --> E[Context-Aware Chunking]
    E --> F[Metadata Extraction]
    F --> G[Embedding Generation]
    G --> H[Vector Storage]
```

## 🔧 Key Features

### 1. Context-Aware Chunking (CAG)

The `ContextAwareChunker` intelligently adapts chunking strategies based on content type:

- **Code Content**: Preserves function/class boundaries using `CodeSplitter`
- **Structured Content**: Uses `HierarchicalNodeParser` for documents with headers/sections
- **Tabular Data**: Maintains table structure integrity
- **Semantic Content**: Leverages `SemanticSplitterNodeParser` for natural language
- **Default Content**: Falls back to sentence-based chunking

### 2. Multi-Modal File Processing

Supports comprehensive file format coverage:

| Category | Formats | Processing Method |
|----------|---------|-------------------|
| Documents | PDF, DOCX, DOC, TXT, MD, RTF | Native readers + text extraction |
| Spreadsheets | CSV, XLSX, XLS | Structured data parsing |
| Images | JPG, PNG, GIF, BMP | OCR with EasyOCR |
| Audio | MP3, WAV, M4A | Speech-to-text with Whisper |
| Code | PY, JS, JAVA, CPP, SQL | Syntax-aware processing |
| Archives | ZIP, TAR, GZ | Recursive extraction |
| Structured | JSON, XML, YAML | Schema-aware parsing |

### 3. Enhanced Metadata Extraction

Each processed document is enriched with:

- **File Metadata**: Path, type, size, modification date
- **Content Metadata**: Word count, character count, content hash
- **Semantic Metadata**: Titles, summaries, keywords, entities
- **Contextual Metadata**: Chunk relationships, preview snippets
- **Processing Metadata**: Extraction method, confidence scores

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ingestion_service/

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for full functionality
pip install easyocr whisper-openai
```

### Environment Configuration

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Optional
LOG_LEVEL=INFO
MAX_WORKERS=5
CACHE_DIR=./ingestion_cache
```

### Basic Usage

```python
import asyncio
from ingestion_service import SupabaseRAGIngestionService, IngestionConfig

async def main():
    # Configure the service
    config = IngestionConfig(
        chunk_size=1024,
        chunk_overlap=200,
        enable_semantic_chunking=True,
        enable_ocr=True,
        enable_speech_to_text=True,
        supabase_url="your_supabase_url",
        supabase_key="your_supabase_key",
        table_name="rag_documents",
        max_concurrent_files=3,
        batch_size=50
    )
    
    # Initialize service
    service = SupabaseRAGIngestionService(config)
    
    # Ingest files
    file_paths = [
        "documents/report.pdf",
        "data/analysis.xlsx", 
        "images/diagram.png"
    ]
    
    results = await service.ingest_files(file_paths)
    print(f"Processed: {results['processed']}")
    print(f"Total chunks: {results['total_chunks']}")
    
    # Ingest entire directory
    directory_results = await service.ingest_directory(
        "documents/",
        recursive=True,
        file_extensions=['.pdf', '.docx', '.txt']
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚙️ Configuration Options

### IngestionConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1024 | Maximum characters per chunk |
| `chunk_overlap` | int | 200 | Overlap between consecutive chunks |
| `enable_semantic_chunking` | bool | True | Use semantic boundaries for chunking |
| `enable_hierarchical_chunking` | bool | True | Preserve document structure |
| `extract_metadata` | bool | True | Extract semantic metadata |
| `enable_ocr` | bool | True | Process images with OCR |
| `enable_speech_to_text` | bool | True | Transcribe audio files |
| `max_file_size_mb` | int | 100 | Maximum file size limit |
| `max_concurrent_files` | int | 5 | Concurrent processing limit |
| `batch_size` | int | 100 | Batch size for vector storage |
| `embedding_model` | str | "text-embedding-3-large" | OpenAI embedding model |
| `llm_model` | str | "gpt-4-turbo" | LLM for metadata extraction |

## 📊 Monitoring and Metrics

### Performance Tracking

```python
# Get ingestion statistics
stats = service.get_ingestion_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Vector dimension: {stats['vector_dimension']}")

# Monitor processing results
results = await service.ingest_files(file_paths)
print(f"Processing time: {results['processing_time']} seconds")
print(f"Success rate: {len(results['processed'])/len(file_paths)*100}%")
```

### Error Handling

The service provides comprehensive error handling:

- **File Validation**: Size limits, format verification
- **Processing Errors**: Graceful fallbacks and logging
- **Batch Processing**: Individual file failure isolation
- **Resource Management**: Memory and connection pooling

## 🔧 Advanced Usage

### Custom File Processors

```python
class CustomProcessor(MultiModalFileProcessor):
    def _process_custom_format(self, file_path: str) -> List[Document]:
        # Implement custom processing logic
        pass

# Register custom processor
config.custom_processors = {'custom_ext': CustomProcessor}
```

### Pipeline Customization

```python
# Custom transformation pipeline
custom_extractors = [
    TitleExtractor(nodes=3),
    CustomMetadataExtractor(),
    KeywordExtractor(keywords=15)
]

service.pipeline.transformations = [
    ContextAwareChunker(config),
    *custom_extractors,
    Settings.embed_model
]
```

## 🚨 Troubleshooting

### Common Issues

**1. OCR/Whisper Dependencies**
```bash
# Install system dependencies for OCR
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# For Whisper, ensure FFmpeg is installed
sudo apt-get install ffmpeg
```

**2. Memory Issues with Large Files**
```python
# Reduce batch size and enable streaming
config.batch_size = 20
config.max_concurrent_files = 2
config.max_file_size_mb = 50
```

**3. Supabase Connection Issues**
```python
# Verify credentials and network connectivity
try:
    service.supabase.table(config.table_name).select("count").execute()
except Exception as e:
    logger.error(f"Supabase connection failed: {e}")
```

## 📈 Performance Optimization

### Scaling Recommendations

- **Small Scale (< 1000 files)**: Default configuration
- **Medium Scale (1000-10000 files)**: Increase `max_concurrent_files` to 8-10
- **Large Scale (> 10000 files)**: Consider distributed processing with Celery/Redis

### Resource Usage

| Component | CPU Usage | Memory Usage | I/O Impact |
|-----------|-----------|--------------|------------|
| Text Processing | Low | Low | Low |
| OCR Processing | High | Medium | Medium |
| Audio Transcription | High | High | Medium |
| Vector Generation | Medium | Medium | High |

## 🔗 Integration Examples

### With Existing Pipelines

```python
# Integration with existing data pipelines
from your_pipeline import DataPipeline

class IntegratedProcessor:
    def __init__(self):
        self.rag_service = SupabaseRAGIngestionService(config)
        self.data_pipeline = DataPipeline()
    
    async def process_batch(self, data_batch):
        # Pre-process with existing pipeline
        processed_files = self.data_pipeline.transform(data_batch)
        
        # Ingest with RAG service
        results = await self.rag_service.ingest_files(processed_files)
        
        # Post-process results
        return self.data_pipeline.finalize(results)
```

## 📝 API Reference

### SupabaseRAGIngestionService

#### Methods

- `ingest_files(file_paths: List[str]) -> Dict[str, Any]`
- `ingest_directory(directory_path: str, recursive: bool = True) -> Dict[str, Any]`
- `get_ingestion_stats() -> Dict[str, Any]`

#### Returns

```python
{
    'processed': List[str],      # Successfully processed files
    'failed': List[str],         # Failed file paths
    'total_documents': int,      # Total documents created
    'total_chunks': int,         # Total chunks generated
    'processing_time': float     # Processing time in seconds
}
```