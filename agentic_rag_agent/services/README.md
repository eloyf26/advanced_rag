# `/services/` Folder Documentation

## Overview

The `/services/` folder contains the core infrastructure services that power the agentic RAG system. These services provide fundamental capabilities like database operations, embedding generation, result reranking, and language model integration with comprehensive error handling, caching, and performance optimization.

## 🎯 Design Principles

- **Resilience**: Circuit breakers, retries, and graceful degradation
- **Performance**: Caching, batching, and async processing
- **Observability**: Comprehensive logging and metrics collection
- **Scalability**: Connection pooling and resource management
- **Security**: Input validation and secure API usage

## 📁 File Structure

```
services/
├── database_manager.py    # Vector database operations with Supabase
├── embedding_service.py   # Embedding generation with caching
├── reranking_service.py   # Result reranking with multiple strategies
└── llm_service.py         # Language model integration and prompts
```

## 🔧 Component Details

### 1. `database_manager.py` - Vector Database Operations

**Purpose**: Manages all interactions with the Supabase vector database, providing hybrid search capabilities with comprehensive error handling.

#### Key Features

##### Circuit Breaker Pattern
```python
class RAGDatabaseManager:
    def __init__(self, config: RAGConfig):
        # Circuit breaker states
        self._db_circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'closed'  # closed, open, half-open
        }
```

**Circuit Breaker States**:
- **Closed**: Normal operation, requests pass through
- **Open**: Service unavailable, requests fail fast
- **Half-Open**: Testing recovery, limited requests allowed

##### Hybrid Search Implementation
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def hybrid_search(self, query: str, filters: SearchFilters) -> List[DocumentChunk]:
    """Hybrid search combining vector and keyword search"""
    
    # Generate query embedding
    query_embedding = await self.get_query_embedding(query)
    
    # Call database function
    result = self.supabase.rpc('hybrid_search', {
        'query_embedding': query_embedding,
        'query_text': query,
        'similarity_threshold': filters.similarity_threshold,
        'vector_weight': self.config.vector_weight,
        'bm25_weight': self.config.bm25_weight
    }).execute()
```

**Search Methods**:
- **Hybrid Search**: Combines vector similarity and keyword matching
- **Semantic Search**: Pure vector similarity search
- **Keyword Search**: BM25-based text search
- **Contextual Search**: Enhanced search considering existing context

##### Embedding Management
```python
async def get_query_embedding(self, query: str) -> List[float]:
    """Generate embedding with caching and error handling"""
    
    # Check cache first
    if self.config.enable_embedding_cache:
        cached = self.embedding_cache.get(query)
        if cached:
            return cached
    
    # Generate new embedding
    response = await self.openai_client.embeddings.create(
        input=query,
        model=self.config.embedding_model
    )
    
    # Cache result
    embedding = response.data[0].embedding
    self.embedding_cache[query] = embedding
    return embedding
```

##### Health Monitoring
```python
async def _health_check(self, force: bool = False) -> Dict[str, bool]:
    """Comprehensive health check for all services"""
    
    # Test database connection
    try:
        result = self.supabase.table(self.config.table_name).select("count").limit(1).execute()
        self._db_healthy = True
    except Exception as e:
        self._db_healthy = False
        self._record_circuit_breaker_failure('db')
    
    # Test embedding service
    try:
        response = await self.openai_client.embeddings.create(
            input="test", model=self.config.embedding_model
        )
        self._embedding_healthy = True
    except Exception as e:
        self._embedding_healthy = False
        self._record_circuit_breaker_failure('embedding')
```

#### Database Operations

##### Document Retrieval
```python
async def get_document_context(
    self, 
    document_id: str, 
    chunk_index: int, 
    context_window: int = 3
) -> List[DocumentChunk]:
    """Get surrounding chunks for better context"""
```

##### Similarity Search
```python
async def get_similar_chunks(
    self, 
    chunk_id: str, 
    similarity_threshold: float = 0.8,
    limit: int = 5
) -> List[DocumentChunk]:
    """Find chunks similar to a given chunk"""
```

##### Statistics and Analytics
```python
async def get_database_stats(self) -> Dict[str, Any]:
    """Get comprehensive database statistics"""
    
    result = self.supabase.rpc('get_document_stats').execute()
    return {
        'total_documents': stats.get('total_documents', 0),
        'total_chunks': stats.get('total_chunks', 0),
        'unique_files': stats.get('unique_files', 0),
        'file_types': stats.get('file_types', []),
        'vector_dimension': stats.get('vector_dimension', 0)
    }
```

### 2. `embedding_service.py` - Embedding Generation with Caching

**Purpose**: Provides efficient embedding generation with comprehensive caching, batching, and multiple provider support.

#### Key Features

##### Multi-Provider Support
```python
class EmbeddingService:
    def __init__(self, config: RAGConfig):
        # Initialize clients
        self.openai_client = openai.OpenAI()
        self.local_models = {}  # For local sentence-transformers
        
        # Request queue for batching
        self.request_queue = asyncio.Queue()
```

**Supported Providers**:
- **OpenAI**: text-embedding-3-large, text-embedding-3-small
- **Local Models**: sentence-transformers models
- **Custom Models**: Extensible for other providers

##### Advanced Caching
```python
class EmbeddingCache:
    """LRU Cache for embeddings with persistence"""
    
    def __init__(self, max_size: int = 10000, persist_path: str = None):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        
    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache with LRU update"""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]['embedding']
        return None
```

**Cache Features**:
- **LRU Eviction**: Removes least recently used entries
- **Persistence**: Optional disk persistence
- **Metadata Tracking**: Stores model, timestamp, access count
- **Statistics**: Hit rates and performance metrics

##### Batch Processing
```python
async def get_embeddings(
    self, 
    texts: Union[str, List[str]], 
    model: str = None,
    batch_size: int = None
) -> Union[List[float], List[List[float]]]:
    """Get embeddings with caching and batching"""
    
    # Check cache for each text
    cached_embeddings = []
    cache_misses = []
    
    for text in texts:
        cached = self.cache.get(cache_key)
        if cached:
            cached_embeddings.append(cached)
        else:
            cache_misses.append(text)
    
    # Generate embeddings for cache misses
    if cache_misses:
        new_embeddings = await self._get_openai_embeddings(
            cache_misses, model, batch_size
        )
```

##### Background Processing
```python
async def _batch_processor(self):
    """Background task to process embedding requests in batches"""
    while True:
        # Collect requests for batching
        requests = []
        deadline = time.time() + 0.1  # 100ms batch window
        
        while time.time() < deadline:
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=max(0.01, deadline - time.time())
                )
                requests.append(request)
            except asyncio.TimeoutError:
                break
        
        if requests:
            await self._process_batch_requests(requests)
```

#### Performance Optimization

##### Model Information
```python
def get_model_info(self, model: str) -> Dict[str, Any]:
    """Get information about a specific model"""
    if model.startswith('text-embedding'):
        dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        return {
            'provider': 'openai',
            'dimensions': dimensions.get(model, 1536),
            'max_tokens': 8191,
            'pricing_per_1k_tokens': 0.0001
        }
```

##### Service Statistics
```python
def get_service_stats(self) -> Dict[str, Any]:
    """Get comprehensive service statistics"""
    return {
        'requests': self.stats,
        'cache': self.cache.stats(),
        'models_loaded': list(self.local_models.keys()),
        'queue_size': self.request_queue.qsize(),
        'avg_processing_time': (
            self.stats['total_processing_time'] / 
            max(self.stats['total_requests'], 1)
        ),
        'cache_hit_rate': (
            self.stats['cache_hits'] / 
            max(self.stats['total_requests'], 1)
        )
    }
```

### 3. `reranking_service.py` - Result Reranking with Multiple Strategies

**Purpose**: Provides sophisticated result reranking using multiple strategies to improve relevance and ranking quality.

#### Reranking Strategies

##### Cross-Encoder Reranking
```python
class CrossEncoderReranker:
    """Cross-encoder based reranking"""
    
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name, device=self.device)
    
    async def rerank(
        self, 
        query: str, 
        documents: List[DocumentChunk], 
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Rerank documents using cross-encoder"""
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for doc in documents:
            doc_text = self._prepare_document_text(doc)
            query_doc_pairs.append([query, doc_text])
        
        # Get reranking scores
        scores = await asyncio.to_thread(
            self.model.predict, query_doc_pairs
        )
        
        # Update document scores
        for i, doc in enumerate(documents):
            doc.rerank_score = float(scores[i])
            doc.combined_score = self._combine_scores(doc)
        
        # Sort by rerank score
        return sorted(documents, key=lambda x: x.rerank_score, reverse=True)[:top_k]
```

**Cross-Encoder Features**:
- **Fine-grained Relevance**: Token-level attention mechanisms
- **Query-Document Interaction**: Direct comparison scoring
- **Multiple Models**: Support for various cross-encoder architectures
- **Batch Processing**: Efficient processing of multiple documents

##### ColBERT-Style Reranking
```python
class ColBERTReranker:
    """ColBERT-style token-level reranking"""
    
    async def rerank(
        self, 
        query: str, 
        documents: List[DocumentChunk], 
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """ColBERT-style reranking with token interactions"""
        
        for doc in documents:
            # Token-level interaction scoring
            query_tokens = set(query.lower().split())
            doc_tokens = set(doc.content.lower().split())
            
            overlap_score = len(query_tokens & doc_tokens) / max(len(query_tokens | doc_tokens), 1)
            doc.rerank_score = overlap_score
            doc.combined_score = self._combine_scores(doc)
```

##### Custom Reranking
```python
class CustomReranker:
    """Custom reranking logic with multiple factors"""
    
    def _calculate_custom_score(self, query: str, doc: DocumentChunk) -> float:
        """Calculate custom reranking score"""
        factors = []
        
        # Factor 1: Original similarity score (40%)
        factors.append(doc.similarity_score * 0.4)
        
        # Factor 2: BM25 score (30%)
        factors.append(doc.bm25_score * 0.3)
        
        # Factor 3: Document quality indicators (20%)
        quality_score = 0.0
        if doc.title: quality_score += 0.1
        if doc.summary: quality_score += 0.1
        if doc.keywords and len(doc.keywords) > 3: quality_score += 0.1
        if len(doc.content) > 200: quality_score += 0.1
        factors.append(quality_score * 0.2)
        
        # Factor 4: File type preference (10%)
        file_type_scores = {'pdf': 0.9, 'docx': 0.8, 'md': 0.7, 'txt': 0.6}
        file_score = file_type_scores.get(doc.file_type, 0.5)
        factors.append(file_score * 0.1)
        
        return sum(factors)
```

#### Advanced Reranking Features

##### Combined Strategy Reranking
```python
async def _combined_reranking(
    self, 
    query: str, 
    documents: List[DocumentChunk], 
    top_k: int
) -> List[DocumentChunk]:
    """Combine multiple reranking strategies"""
    
    # Get results from different rerankers
    cross_encoder_docs = await self.rerankers[RerankingStrategy.CROSS_ENCODER].rerank(
        query, documents.copy(), top_k * 2
    )
    
    custom_docs = await self.rerankers[RerankingStrategy.CUSTOM].rerank(
        query, documents.copy(), top_k * 2
    )
    
    # Ensemble combination
    combined_docs = self._ensemble_combine(
        query, documents, cross_encoder_docs, custom_docs
    )
    
    return combined_docs[:top_k]
```

##### Adaptive Reranking
```python
async def adaptive_reranking(
    self,
    query: str,
    documents: List[DocumentChunk],
    quality_threshold: float = 0.8
) -> RerankingResponse:
    """Adaptive reranking that chooses optimal strategy"""
    
    # Analyze query characteristics
    query_length = len(query.split())
    has_technical_terms = any(term in query.lower() for term in [
        'algorithm', 'implementation', 'technical'
    ])
    
    # Choose strategy based on characteristics
    if query_length <= 3 and not has_technical_terms:
        strategy = RerankingStrategy.CUSTOM  # Fast for simple queries
    elif has_technical_terms or query_length > 10:
        strategy = RerankingStrategy.CROSS_ENCODER  # Accuracy for complex
    else:
        strategy = RerankingStrategy.COMBINED  # Balanced approach
    
    # Perform reranking with quality validation
    response = await self.rerank_documents(query, documents, strategy)
    
    # Retry with different strategy if quality is poor
    if response.processing_time > 0 and len(response.reranked_documents) > 3:
        quality = await self.validate_reranking_quality(
            query, documents, response.reranked_documents
        )
        
        if quality.get('score_improvement', 0) < quality_threshold:
            # Retry with cross-encoder for better quality
            response = await self.rerank_documents(
                query, documents, RerankingStrategy.CROSS_ENCODER
            )
    
    return response
```

##### Quality Validation
```python
async def validate_reranking_quality(
    self,
    query: str,
    original_docs: List[DocumentChunk],
    reranked_docs: List[DocumentChunk]
) -> Dict[str, Any]:
    """Validate the quality of reranking results"""
    
    quality_metrics = {}
    
    # Metric 1: Score improvement
    original_avg_score = statistics.mean([
        doc.similarity_score for doc in original_docs[:len(reranked_docs)]
    ])
    reranked_avg_score = statistics.mean([
        doc.rerank_score for doc in reranked_docs
    ])
    quality_metrics['score_improvement'] = reranked_avg_score - original_avg_score
    
    # Metric 2: Rank correlation
    original_ranks = {doc.id: i for i, doc in enumerate(original_docs)}
    reranked_ranks = {doc.id: i for i, doc in enumerate(reranked_docs)}
    
    rank_changes = []
    for doc in reranked_docs:
        if doc.id in original_ranks:
            change = abs(original_ranks[doc.id] - reranked_ranks[doc.id])
            rank_changes.append(change)
    
    quality_metrics['avg_rank_change'] = statistics.mean(rank_changes)
    quality_metrics['max_rank_change'] = max(rank_changes) if rank_changes else 0
    
    # Metric 3: Top-k stability
    top_3_original = set(doc.id for doc in original_docs[:3])
    top_3_reranked = set(doc.id for doc in reranked_docs[:3])
    quality_metrics['top_3_stability'] = len(top_3_original & top_3_reranked) / 3
    
    return quality_metrics
```

### 4. `llm_service.py` - Language Model Integration

**Purpose**: Provides comprehensive language model integration with sophisticated prompt engineering and error handling.

#### Prompt Engineering

##### Structured Prompts
```python
class LLMService:
    def __init__(self, config: RAGConfig):
        self.prompts = {
            'answer_generation': """
Based on the provided sources, answer the user's question comprehensively and accurately.

User Question: {question}

Retrieved Sources:
{sources}

Instructions:
- Provide a clear, well-structured answer based on the sources
- Cite specific sources when making claims (use source numbers)
- If information is incomplete or conflicting, acknowledge this
- Be factual and avoid speculation beyond what the sources support
- Structure your response logically with clear sections if appropriate

Answer:""",
            
            'follow_up_generation': """
Based on the conversation context, generate 3-5 intelligent follow-up questions.

Original Question: {question}
Answer Provided: {answer}
Sources Used: {source_count} sources

Generate follow-up questions that are:
- Specific and actionable
- Build naturally on the information provided
- Help explore different aspects of the topic
- Suitable for further research

Follow-up Questions:""",
            
            'query_enhancement': """
Enhance the user's query to improve search results by adding relevant context and synonyms.

Original Query: {query}
Context: {context}

Enhanced Query:"""
        }
```

**Prompt Categories**:
- **Answer Generation**: Comprehensive response synthesis
- **Follow-up Generation**: Intelligent question suggestions
- **Query Enhancement**: Search query improvement
- **Content Analysis**: Document analysis and categorization
- **Summarization**: Content summarization

##### Answer Generation
```python
async def generate_answer(
    self, 
    question: str, 
    sources: List[DocumentChunk], 
    query_plan: QueryPlan
) -> str:
    """Generate comprehensive answer using retrieved sources"""
    
    if not sources:
        return "I couldn't find any relevant information to answer your question."
    
    # Prepare sources context
    sources_text = self._format_sources_for_prompt(sources[:8])
    
    # Create structured prompt
    prompt = self.prompts['answer_generation'].format(
        question=question,
        sources=sources_text
    )
    
    # Generate response with appropriate parameters
    response = await self.client.chat.completions.create(
        model=self.config.llm_model,
        messages=[
            {
                "role": "system", 
                "content": "You are a knowledgeable research assistant that provides accurate, well-sourced answers."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for factual responses
        max_tokens=2000
    )
    
    answer = response.choices[0].message.content.strip()
    return self._post_process_answer(answer, sources)
```

##### Follow-up Generation
```python
async def generate_follow_ups(
    self, 
    question: str, 
    answer: str, 
    sources: List[DocumentChunk]
) -> List[str]:
    """Generate intelligent follow-up questions"""
    
    prompt = self.prompts['follow_up_generation'].format(
        question=question,
        answer=answer[:800],  # Truncate long answers
        source_count=len(sources)
    )
    
    response = await self.client.chat.completions.create(
        model=self.config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at generating insightful follow-up questions."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,  # Slight creativity for variety
        max_tokens=300
    )
    
    follow_ups_text = response.choices[0].message.content.strip()
    return self._parse_follow_ups(follow_ups_text)[:5]
```

#### Advanced Features

##### Content Analysis
```python
async def analyze_content(self, content: str) -> Dict[str, Any]:
    """Analyze content for type, themes, and audience"""
    
    content_sample = content[:1500]
    prompt = self.prompts['content_analysis'].format(content=content_sample)
    
    response = await self.client.chat.completions.create(
        model=self.config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert content analyst."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    
    analysis_text = response.choices[0].message.content.strip()
    
    # Parse analysis into structured format
    analysis = {
        'full_analysis': analysis_text,
        'content_type': 'general',
        'complexity': 'medium',
        'domain': 'general'
    }
    
    # Extract specific categorizations
    if 'technical' in analysis_text.lower():
        analysis['content_type'] = 'technical'
    elif 'academic' in analysis_text.lower():
        analysis['content_type'] = 'academic'
    elif 'business' in analysis_text.lower():
        analysis['content_type'] = 'business'
    
    return analysis
```

##### Query Enhancement
```python
async def enhance_query(self, query: str, context: str = "") -> str:
    """Enhance query for better search results"""
    
    prompt = self.prompts['query_enhancement'].format(
        query=query,
        context=context or "No additional context provided"
    )
    
    response = await self.client.chat.completions.create(
        model=self.config.llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at optimizing search queries."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=150
    )
    
    enhanced_query = response.choices[0].message.content.strip()
    
    # Ensure enhancement isn't too different from original
    if len(enhanced_query) > len(query) * 3:
        return query  # Fallback to original
    
    return enhanced_query
```

#### Error Handling and Reliability

##### Connection Testing
```python
async def test_llm_connection(self) -> bool:
    """Test LLM service connectivity"""
    try:
        response = await self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "user", "content": "Respond with 'OK' if you can process this message."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        return "ok" in response.choices[0].message.content.lower()
        
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False
```

##### Response Post-Processing
```python
def _post_process_answer(self, answer: str, sources: List[DocumentChunk]) -> str:
    """Post-process generated answer for quality"""
    
    # Ensure source citations are properly formatted
    if not any(f"source {i}" in answer.lower() for i in range(1, 6)):
        if sources:
            answer += f"\n\nBased on information from {len(sources)} source(s) including {sources[0].file_name}"
            if len(sources) > 1:
                answer += " and others"
            answer += "."
    
    return answer
```

##### Follow-up Parsing
```python
def _parse_follow_ups(self, follow_ups_text: str) -> List[str]:
    """Parse follow-up questions from LLM response"""
    
    lines = follow_ups_text.split('\n')
    follow_ups = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering and bullet points
        line = re.sub(r'^\d+\.\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        
        # Ensure it ends with a question mark
        if not line.endswith('?'):
            line += '?'
        
        if len(line) > 10:  # Minimum length filter
            follow_ups.append(line)
    
    return follow_ups
```

## 🔗 Service Integration Patterns

### 1. Service Orchestration
```python
class AgenticRAGService:
    def __init__(self, config: RAGConfig):
        # Initialize all services
        self.db_manager = RAGDatabaseManager(config)
        self.llm_service = LLMService(config)
        self.embedding_service = EmbeddingService(config)
        self.reranking_service = RerankingService(config)
    
    async def ask_with_planning(self, question: str) -> AgenticRAGResponse:
        # Coordinate all services
        search_results = await self.db_manager.hybrid_search(query, filters)
        if self.config.enable_reranking:
            reranked = await self.reranking_service.rerank_documents(query, search_results.chunks)
            search_results.chunks = reranked.reranked_documents
        
        answer = await self.llm_service.generate_answer(question, search_results.chunks, query_plan)
        return response
```

### 2. Error Handling Chain
```python
# Database Manager -> Embedding Service -> LLM Service
try:
    search_results = await self.db_manager.hybrid_search(query, filters)
except DatabaseConnectionError:
    # Fallback to keyword search
    search_results = await self.db_manager.keyword_search(query, filters)
except EmbeddingServiceError:
    # Use cached embeddings or simplified search
    search_results = await self._fallback_search(query, filters)
```

### 3. Performance Monitoring
```python
# Each service records metrics
self.metrics.record_search('hybrid', duration, len(results), 'success')
self.metrics.record_embedding_request(model, count, cache_hits, processing_time)
self.metrics.record_reranking_request(strategy, doc_count, processing_time, top_k)
```

## 🎯 Configuration and Tuning

### Service-Specific Configuration
```python
# Database Manager
config.enable_embedding_cache = True
config.embedding_cache_size = 10000
config.max_retries = 3

# Reranking Service
config.enable_reranking = True
config.rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
config.rerank_top_k = 20

# LLM Service
config.llm_model = "gpt-4-turbo"
config.max_context_length = 8000
config.temperature = 0.3
```

### Performance Optimization
```python
# For Speed
config.enable_reranking = False
config.embedding_batch_size = 200
config.max_concurrent_searches = 5

# For Quality
config.enable_reranking = True
config.rerank_top_k = 20
config.enable_triangulation = True

# For Scale
config.enable_embedding_cache = True
config.embedding_cache_size = 50000
config.connection_pool_size = 20
```

## 🛠️ Best Practices

### 1. Error Handling
- **Implement circuit breakers** for external services
- **Provide graceful degradation** when services fail
- **Log comprehensive error context** for debugging
- **Use exponential backoff** for retries

### 2. Performance
- **Cache frequently accessed data** (embeddings, query results)
- **Batch requests** when possible
- **Monitor service health** continuously
- **Optimize database queries** and indexes

### 3. Security
- **Validate all inputs** before processing
- **Use secure API practices** (rate limiting, authentication)
- **Sanitize user content** to prevent injection attacks
- **Implement proper access controls** for sensitive operations

### 4. Monitoring
- **Track key performance metrics** (latency, throughput, error rates)
- **Set up alerting** for service degradation
- **Monitor resource usage** (memory, CPU, network)
- **Analyze usage patterns** for optimization opportunities

The services folder provides the robust infrastructure foundation that enables the agentic RAG system to operate reliably at scale, with comprehensive error handling, performance optimization, and observability features.