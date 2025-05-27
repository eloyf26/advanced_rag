# `/models/` Folder Documentation

## Overview

The `/models/` folder contains all data structure definitions using Pydantic models that ensure type safety, validation, and serialization throughout the agentic RAG system. These models serve as contracts between different components and provide comprehensive data validation.

## 🎯 Design Principles

- **Type Safety**: Strong typing with runtime validation
- **Data Validation**: Automatic input sanitization and constraint checking
- **Serialization**: JSON-compatible for API responses
- **Documentation**: Self-documenting with field descriptions
- **Extensibility**: Easy to extend for new features

## 📁 File Structure

```
models/
├── request_models.py     # Input validation and API requests
├── response_models.py    # Output formats and API responses
└── internal_models.py    # Internal processing state management
```

## 🔧 Component Details

### 1. `request_models.py` - Input Validation and API Requests

**Purpose**: Defines and validates all incoming data structures for API endpoints.

#### Core Request Models

##### QueryRequest - Single Question Processing
```python
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    
    # Agentic features
    enable_iteration: bool = Field(True, description="Enable iterative search")
    enable_reflection: bool = Field(True, description="Enable self-reflection")
    enable_triangulation: bool = Field(True, description="Enable source triangulation")
    
    # Search configuration
    search_method: Optional[str] = Field("hybrid", description="Search method")
    max_results: int = Field(10, description="Maximum results")
    similarity_threshold: float = Field(0.7, description="Similarity threshold")
    
    # Filters
    file_types: Optional[List[str]] = Field(None, description="File type filters")
    date_after: Optional[datetime] = Field(None, description="Date filter")
```

**Usage Example**:
```python
request = QueryRequest(
    question="Compare machine learning frameworks",
    enable_iteration=True,
    enable_reflection=True,
    max_results=15,
    file_types=["pdf", "md"]
)
```

##### BatchQueryRequest - Multiple Question Processing
```python
class BatchQueryRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions")
    
    # Processing options
    enable_agentic: bool = Field(True, description="Use full agentic processing")
    max_concurrency: int = Field(3, description="Concurrent question processing")
    timeout_per_question: int = Field(30, description="Timeout per question")
```

##### SearchFilters - Search Configuration
```python
class SearchFilters(BaseModel):
    file_types: Optional[List[str]] = Field(None, description="File type filters")
    date_after: Optional[datetime] = Field(None, description="Date filter")
    similarity_threshold: float = Field(0.7, description="Similarity threshold")
    max_results: int = Field(10, description="Maximum results")
    
    # Advanced filters
    keywords_required: Optional[List[str]] = Field(None, description="Required keywords")
    keywords_excluded: Optional[List[str]] = Field(None, description="Excluded keywords")
    min_word_count: Optional[int] = Field(None, description="Minimum word count")
    max_word_count: Optional[int] = Field(None, description="Maximum word count")
```

#### Specialized Request Models

##### AnalyzeQueryRequest - Query Analysis
```python
class AnalyzeQueryRequest(BaseModel):
    question: str = Field(..., description="Question to analyze")
    include_suggestions: bool = Field(True, description="Include optimization suggestions")
    include_complexity: bool = Field(True, description="Include complexity analysis")
```

##### ConfigUpdateRequest - Runtime Configuration
```python
class ConfigUpdateRequest(BaseModel):
    # Search parameters
    similarity_threshold: Optional[float] = Field(None, ge=0, le=1)
    max_results: Optional[int] = Field(None, ge=1, le=100)
    vector_weight: Optional[float] = Field(None, ge=0, le=1)
    bm25_weight: Optional[float] = Field(None, ge=0, le=1)
    
    # Agentic parameters
    max_iterations: Optional[int] = Field(None, ge=1, le=10)
    enable_reflection: Optional[bool] = Field(None)
    enable_triangulation: Optional[bool] = Field(None)
```

**Validation Features**:
- **Range Validation**: Ensures values are within acceptable bounds
- **Type Coercion**: Automatic type conversion where appropriate
- **Required Fields**: Prevents missing critical data
- **Custom Validators**: Domain-specific validation logic

### 2. `response_models.py` - Output Formats and API Responses

**Purpose**: Defines comprehensive response structures with rich metadata for API endpoints.

#### Core Response Models

##### DocumentChunk - Individual Document Fragment
```python
class DocumentChunk(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Text content of the chunk")
    
    # Scores
    similarity_score: float = Field(0.0, description="Vector similarity score")
    bm25_score: float = Field(0.0, description="BM25 keyword score")
    combined_score: float = Field(0.0, description="Combined hybrid score")
    rerank_score: float = Field(0.0, description="Cross-encoder rerank score")
    
    # Metadata
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type/extension")
    chunk_index: int = Field(..., description="Index of chunk within document")
    
    # Enhanced metadata
    title: Optional[str] = Field(None, description="Extracted title")
    summary: Optional[str] = Field(None, description="Generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities")
    
    # Context
    document_id: Optional[str] = Field(None, description="Parent document ID")
    chunk_context: Optional[str] = Field(None, description="Surrounding context")
```

**Score Types**:
- **similarity_score**: Vector similarity (0-1)
- **bm25_score**: Keyword relevance (0-1)
- **combined_score**: Weighted hybrid score
- **rerank_score**: Cross-encoder refinement

##### QueryPlan - Execution Strategy
```python
class QueryPlan(BaseModel):
    original_query: str = Field(..., description="Original user query")
    sub_queries: List[str] = Field(..., description="Generated sub-questions")
    search_strategy: str = Field(..., description="Selected search strategy")
    reasoning: str = Field(..., description="Planning rationale")
    expected_sources: List[str] = Field(default_factory=list, description="Expected source types")
    complexity_score: float = Field(..., description="Query complexity score (0-1)")
    
    # Execution metadata
    estimated_time: Optional[float] = Field(None, description="Estimated processing time")
    recommended_iterations: Optional[int] = Field(None, description="Recommended iterations")
```

##### ReflectionResult - Quality Assessment
```python
class ReflectionResult(BaseModel):
    quality_score: float = Field(..., description="Overall quality score (0-1)")
    completeness_score: float = Field(..., description="Completeness score (0-1)")
    accuracy_assessment: str = Field(..., description="Accuracy assessment")
    missing_information: List[str] = Field(default_factory=list, description="Missing info")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="Follow-ups")
    needs_more_search: bool = Field(False, description="Needs additional search")
    
    # Detailed assessment
    reasoning_quality: Optional[float] = Field(None, description="Reasoning quality (0-1)")
    source_diversity: Optional[float] = Field(None, description="Source diversity (0-1)")
    factual_consistency: Optional[float] = Field(None, description="Factual consistency (0-1)")
```

##### AgenticRAGResponse - Complete Agentic Response
```python
class AgenticRAGResponse(BaseModel):
    # Core response
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[DocumentChunk] = Field(..., description="Source documents used")
    
    # Agentic components
    query_plan: QueryPlan = Field(..., description="Query execution plan")
    reflection: ReflectionResult = Field(..., description="Self-reflection results")
    search_iterations: List[SearchResults] = Field(..., description="Search iteration history")
    reasoning_chain: List[str] = Field(..., description="Step-by-step reasoning")
    follow_up_suggestions: List[str] = Field(..., description="Suggested follow-ups")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Total processing time")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    iterations_completed: int = Field(..., description="Number of search iterations")
    
    # Quality indicators
    source_triangulation_performed: bool = Field(False, description="Triangulation performed")
    cross_validated_facts: List[str] = Field(default_factory=list, description="Cross-validated facts")
    potential_biases: List[str] = Field(default_factory=list, description="Identified biases")
```

##### SearchResults - Search Operation Results
```python
class SearchResults(BaseModel):
    query: str = Field(..., description="Original search query")
    chunks: List[DocumentChunk] = Field(..., description="Retrieved document chunks")
    total_found: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    search_method: str = Field(..., description="Search method used")
    
    # Search metadata
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    index_used: Optional[str] = Field(None, description="Database index used")
    cache_hit: bool = Field(False, description="Whether result was cached")
```

#### Specialized Response Models

##### QueryAnalysisResponse - Query Analysis Results
```python
class QueryAnalysisResponse(BaseModel):
    query: str = Field(..., description="Original query")
    complexity_score: float = Field(..., description="Query complexity (0-1)")
    estimated_processing_time: float = Field(..., description="Estimated time in seconds")
    recommended_strategy: str = Field(..., description="Recommended search strategy")
    
    # Analysis details
    query_type: str = Field(..., description="Classified query type")
    key_concepts: List[str] = Field(..., description="Identified key concepts")
    predicted_sources: List[str] = Field(..., description="Predicted relevant source types")
    suggested_filters: Dict[str, Any] = Field(default_factory=dict, description="Suggested filters")
    
    # Optimization suggestions
    optimization_suggestions: List[str] = Field(default_factory=list, description="Optimizations")
    potential_challenges: List[str] = Field(default_factory=list, description="Potential challenges")
```

##### PerformanceMetrics - System Performance Data
```python
class PerformanceMetrics(BaseModel):
    # Query metrics
    total_queries: int = Field(..., description="Total queries processed")
    avg_query_time_ms: float = Field(..., description="Average query processing time")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    
    # Search metrics
    avg_search_time_ms: float = Field(..., description="Average search time")
    avg_sources_per_query: float = Field(..., description="Average sources per query")
    avg_confidence_score: float = Field(..., description="Average confidence score")
    
    # Agentic metrics
    avg_iterations_per_query: float = Field(..., description="Average search iterations")
    reflection_success_rate: float = Field(..., description="Reflection success rate")
    triangulation_usage_rate: float = Field(..., description="Triangulation usage rate")
    
    # Resource metrics
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    avg_tokens_per_query: Optional[int] = Field(None, description="Average tokens per query")
    embedding_requests_per_minute: float = Field(..., description="Embedding requests per minute")
    
    # Time-based metrics
    metrics_period_start: datetime = Field(..., description="Start of metrics period")
    metrics_period_end: datetime = Field(..., description="End of metrics period")
```

##### ErrorResponse - Error Handling
```python
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    # Recovery suggestions
    suggested_actions: Optional[List[str]] = Field(None, description="Suggested recovery actions")
    retry_recommended: bool = Field(False, description="Whether retry is recommended")
```

### 3. `internal_models.py` - Internal Processing State Management

**Purpose**: Defines internal data structures for complex workflow state management and inter-component communication.

#### Core Internal Models

##### InternalDocumentChunk - Enhanced Processing Metadata
```python
class InternalDocumentChunk(BaseModel):
    # All fields from DocumentChunk plus:
    
    # Internal processing metadata
    retrieval_round: int = 0
    retrieval_strategy: str = "unknown"
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_reliability_score: float = 0.5
    content_quality_score: float = 0.5
    relevance_explanation: Optional[str] = None
    
    # Triangulation metadata
    cross_references: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    supporting_chunks: List[str] = Field(default_factory=list)
    
    # Performance metadata
    embedding_time: Optional[float] = None
    search_time: Optional[float] = None
    rerank_time: Optional[float] = None
```

##### QueryExecutionContext - Execution State Tracking
```python
class QueryExecutionContext(BaseModel):
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str
    processed_query: str
    
    # Execution metadata
    current_stage: ProcessingStage = ProcessingStage.INITIALIZED
    priority: Priority = Priority.MEDIUM
    start_time: datetime = Field(default_factory=datetime.utcnow)
    
    # Configuration
    enable_iteration: bool = True
    enable_reflection: bool = True
    enable_triangulation: bool = True
    max_iterations: int = 3
    
    # State tracking
    search_iterations: List[SearchIteration] = Field(default_factory=list)
    all_retrieved_chunks: List[InternalDocumentChunk] = Field(default_factory=list)
    triangulation_chunks: List[InternalDocumentChunk] = Field(default_factory=list)
    
    # Analysis results
    query_complexity: float = 0.5
    estimated_processing_time: float = 10.0
    confidence_threshold: float = 0.7
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Performance tracking
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
```

##### SearchIteration - Individual Search Round State
```python
class SearchIteration(BaseModel):
    iteration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration_number: int
    status: SearchIterationStatus = SearchIterationStatus.PENDING
    
    # Search parameters
    query: str
    search_strategy: str
    similarity_threshold: float
    max_results: int
    file_types: Optional[List[str]] = None
    
    # Results
    chunks_found: List[InternalDocumentChunk] = Field(default_factory=list)
    unique_sources: Set[str] = Field(default_factory=set)
    coverage_gaps: List[str] = Field(default_factory=list)
    
    # Performance metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: float = 0.0
    
    # Decision making
    should_continue: bool = True
    continuation_reason: Optional[str] = None
    strategy_adjustments: List[str] = Field(default_factory=list)
```

#### Specialized Internal Models

##### ConceptNode - Knowledge Graph Element
```python
class ConceptNode(BaseModel):
    concept: str
    importance: float = 0.5
    query_frequency: int = 1
    related_concepts: Set[str] = Field(default_factory=set)
    
    # Source information
    source_chunks: List[str] = Field(default_factory=list)
    definition: Optional[str] = None
    examples: List[str] = Field(default_factory=list)
    
    # Analysis metadata
    coverage_score: float = 0.0
    consensus_score: float = 0.0
    controversy_score: float = 0.0
```

##### FactualClaim - Verification Target
```python
class FactualClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str
    source_chunk_id: str
    confidence: float = 0.5
    
    # Verification status
    verified: bool = False
    supporting_chunks: List[str] = Field(default_factory=list)
    contradicting_chunks: List[str] = Field(default_factory=list)
    
    # Claim analysis
    claim_type: str = "factual"  # factual, opinion, prediction, etc.
    verifiability: float = 0.5
    importance: float = 0.5
    
    # Context
    topic: Optional[str] = None
    domain: Optional[str] = None
    temporal_context: Optional[str] = None
```

##### WorkflowStep - Process Tracking
```python
class WorkflowStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str
    step_type: str  # planning, search, analysis, synthesis, etc.
    
    # Execution
    status: str = "pending"  # pending, running, completed, failed, skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: float = 0.0
    
    # Input/Output
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision making
    decision_rationale: Optional[str] = None
    alternatives_considered: List[str] = Field(default_factory=list)
    confidence_in_decision: float = 0.5
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    fallback_used: bool = False
```

## 🎯 Key Features and Benefits

### 1. Type Safety and Validation

**Automatic Validation**:
```python
# This will raise ValidationError
request = QueryRequest(
    question="",  # Too short
    max_results=1000,  # Too large
    similarity_threshold=1.5  # Out of range
)
```

**Type Coercion**:
```python
# Automatic conversion
request = QueryRequest(
    question="What is ML?",
    max_results="10",  # String to int
    enable_iteration="true"  # String to bool
)
```

### 2. Rich Metadata and Context

**Comprehensive Tracking**:
- Processing timestamps and durations
- Error context and recovery suggestions
- Quality assessments and confidence scores
- Source provenance and reliability

**Example Usage**:
```python
response = await rag_service.ask_with_planning(question)
print(f"Confidence: {response.confidence:.2f}")
print(f"Quality Score: {response.reflection.quality_score:.2f}")
print(f"Processing Time: {response.processing_time_ms:.0f}ms")
print(f"Sources Used: {len(response.sources)}")
print(f"Iterations: {response.iterations_completed}")
```

### 3. Extensibility and Customization

**Adding Custom Fields**:
```python
class CustomDocumentChunk(DocumentChunk):
    domain_score: float = Field(0.0, description="Domain-specific relevance")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Custom Validation**:
```python
class CustomQueryRequest(QueryRequest):
    @validator('question')
    def validate_question_domain(cls, v):
        if 'sensitive_topic' in v.lower():
            raise ValueError('Sensitive topics not allowed')
        return v
```

### 4. Serialization and API Integration

**JSON Serialization**:
```python
# Automatic JSON conversion
response_dict = response.dict()
json_string = response.json()

# With exclusions
filtered_dict = response.dict(exclude={'internal_metadata'})
```

**API Response Formatting**:
```python
@app.post("/ask", response_model=AgenticRAGResponse)
async def ask_question(request: QueryRequest):
    # Automatic validation and serialization
    response = await rag_service.ask_with_planning(request.question)
    return response  # Automatically serialized to JSON
```

## 🔄 Model Relationships

### Inheritance Hierarchy
```
BaseModel (Pydantic)
├── DocumentChunk
│   └── InternalDocumentChunk
├── QueryRequest
│   └── BatchQueryRequest
├── AgenticRAGResponse
│   └── RAGResponse
└── ErrorResponse
```

### Composition Patterns
```
AgenticRAGResponse
├── QueryPlan
├── ReflectionResult
├── List[SearchResults]
│   └── List[DocumentChunk]
└── List[DocumentChunk] (sources)
```

## 🛠️ Usage Patterns

### 1. Request Validation
```python
from models.request_models import QueryRequest, validate_query

try:
    request = QueryRequest(**request_data)
    validated_question = validate_query(request.question)
except ValidationError as e:
    return ErrorResponse(
        error=str(e),
        error_code="VALIDATION_ERROR"
    )
```

### 2. Response Construction
```python
from models.response_models import AgenticRAGResponse

response = AgenticRAGResponse(
    answer=generated_answer,
    confidence=calculated_confidence,
    sources=retrieved_chunks,
    query_plan=execution_plan,
    reflection=quality_assessment,
    search_iterations=search_history,
    reasoning_chain=step_descriptions,
    follow_up_suggestions=follow_ups,
    processing_time_ms=processing_duration * 1000,
    iterations_completed=len(search_iterations)
)
```

### 3. Internal State Management
```python
from models.internal_models import QueryExecutionContext, SearchIteration

# Create execution context
context = QueryExecutionContext(
    user_query=question,
    processed_query=enhanced_question,
    enable_iteration=True,
    max_iterations=3
)

# Track search iterations
iteration = SearchIteration(
    iteration_number=1,
    query=search_query,
    search_strategy="hybrid",
    similarity_threshold=0.7
)
```

## 🔧 Configuration and Customization

### Environment-Based Configuration
```python
# Models adapt to configuration
class DocumentChunk(BaseModel):
    class Config:
        # Allow additional fields in development
        extra = "allow" if os.getenv("DEBUG_MODE") else "forbid"
        # Validate assignments
        validate_assignment = True
        # Use enum values
        use_enum_values = True
```

### Custom Validators
```python
class QueryRequest(BaseModel):
    question: str
    
    @validator('question')
    def validate_question_length(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Question too short')
        if len(v) > 1000:
            raise ValueError('Question too long')
        return v.strip()
    
    @validator('similarity_threshold')
    def validate_threshold_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v
```

### Factory Functions
```python
# In internal_models.py
def create_execution_context(
    user_query: str,
    enable_iteration: bool = True,
    priority: Priority = Priority.MEDIUM
) -> QueryExecutionContext:
    """Factory function to create query execution context"""
    return QueryExecutionContext(
        user_query=user_query,
        processed_query=user_query.strip(),
        priority=priority,
        enable_iteration=enable_iteration
    )
```

## 🎯 Best Practices

### 1. Model Design
- **Keep models focused** on single responsibilities
- **Use descriptive field names** and documentation
- **Provide sensible defaults** for optional fields
- **Validate constraints** at the model level

### 2. Error Handling
- **Use custom validators** for domain-specific rules
- **Provide helpful error messages** for validation failures
- **Include context** in error responses
- **Suggest corrections** when possible

### 3. Performance
- **Use Field defaults** to avoid unnecessary computation
- **Implement lazy loading** for expensive fields
- **Cache validated models** when appropriate
- **Optimize serialization** for large responses

### 4. Evolution and Versioning
- **Add optional fields** for backward compatibility
- **Use aliases** for field name changes
- **Version your models** for major changes
- **Document breaking changes** clearly

The models folder provides the foundation for all data handling in the agentic RAG system, ensuring type safety, validation, and clear contracts between components while supporting the complex workflows required for agentic processing.