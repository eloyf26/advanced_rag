# `/agents/` Folder Documentation

## Overview

The `/agents/` folder contains the core intelligent components that provide the "agentic" capabilities of the RAG system. These agents work together to plan, execute, and reflect on complex question-answering tasks, going beyond simple retrieval-generation patterns.

## 🧠 Core Philosophy

The agents implement a collaborative intelligence pattern where each agent has specialized responsibilities:

- **Planning Agent**: Strategic thinking and query decomposition
- **Search Coordination Agent**: Tactical search execution and iteration management
- **Reflection Agent**: Quality assessment and self-improvement
- **Main Service**: Orchestration and workflow management

## 📁 File Structure

```
agents/
├── agentic_rag_service.py    # Main orchestration service
├── planning_agent.py         # Query planning and strategy
├── reflection_agent.py       # Quality assessment and reflection
└── search_agent.py           # Search coordination and iteration
```

## 🔧 Component Details

### 1. `agentic_rag_service.py` - Main Orchestration Service

**Purpose**: Central coordinator that orchestrates the entire agentic RAG workflow.

**Key Responsibilities**:
- Service lifecycle management
- Agent coordination and communication
- Response synthesis and formatting
- Error handling and graceful degradation
- Performance monitoring and metrics collection

**Core Methods**:

```python
async def ask_with_planning(
    question: str,
    enable_iteration: bool = True,
    enable_reflection: bool = True,
    enable_triangulation: bool = True
) -> AgenticRAGResponse:
    """Full agentic processing with all capabilities"""
```

**Workflow Stages**:
1. **Query Enhancement**: Improve short or ambiguous queries
2. **Planning**: Create execution strategy with Planning Agent
3. **Iterative Search**: Execute multi-round search with Search Agent
4. **Triangulation**: Verify sources and seek alternative perspectives
5. **Answer Generation**: Synthesize response using LLM Service
6. **Reflection**: Assess quality and completeness
7. **Follow-up Generation**: Create intelligent suggestions

**Configuration Integration**:
- Respects all RAGConfig settings
- Adapts behavior based on enabled features
- Provides fallback modes for service degradation

### 2. `planning_agent.py` - Query Planning and Strategy

**Purpose**: Analyzes queries and creates optimal execution strategies.

**Key Capabilities**:

#### Query Classification
```python
def _classify_query_type(self, question: str) -> str:
    """Classify query into types: definitional, procedural, comparative, etc."""
```

- **Definitional**: "What is machine learning?"
- **Procedural**: "How to implement neural networks?"
- **Comparative**: "Compare TensorFlow vs PyTorch"
- **Causal**: "Why does overfitting occur?"
- **Analytical**: "Analyze the effectiveness of..."

#### Complexity Assessment
```python
def _calculate_complexity(self, question: str) -> float:
    """Calculate complexity score (0-1) based on multiple factors"""
```

**Complexity Factors**:
- Query length and structure
- Technical terminology density
- Multi-part question indicators
- Domain specificity
- Analytical depth requirements

#### Strategic Planning
```python
async def create_query_plan(self, question: str) -> QueryPlan:
    """Create comprehensive execution plan"""
```

**Planning Output**:
- Sub-query decomposition
- Search strategy selection
- Expected source types
- Processing time estimates
- Iteration recommendations

**Example Query Plan**:
```python
QueryPlan(
    original_query="Compare ML algorithms for NLP",
    sub_queries=[
        "machine learning algorithms natural language processing",
        "NLP algorithm performance comparison",
        "text classification algorithm evaluation"
    ],
    search_strategy="hybrid",
    complexity_score=0.7,
    estimated_time=15.0,
    recommended_iterations=2
)
```

### 3. `search_agent.py` - Search Coordination

**Purpose**: Orchestrates iterative search processes with adaptive strategies.

**Key Features**:

#### Iterative Search Management
```python
async def execute_iterative_search(
    query_plan: QueryPlan,
    max_iterations: int = 3,
    min_sources: int = 3
) -> Tuple[List[SearchResults], IterativeSearchState]:
    """Execute multi-round adaptive search"""
```

**Iteration Strategy**:
1. **Broad Discovery**: Wide-net search for initial coverage
2. **Gap Filling**: Target specific missing information
3. **Quality Refinement**: Focus on high-quality sources

#### Adaptive Parameter Tuning
```python
def _create_iteration_filters(
    iteration: int,
    existing_chunks: List[DocumentChunk],
    unique_sources: Set[str]
) -> SearchFilters:
    """Adapt search parameters per iteration"""
```

**Parameter Adaptation**:
- **Iteration 1**: Lower threshold (0.6), more results (15)
- **Iteration 2**: Standard threshold (0.7), diverse file types
- **Iteration 3+**: Higher threshold (0.75), precision focus

#### Coverage Analysis
```python
def _analyze_coverage(
    original_query: str,
    chunks: List[DocumentChunk]
) -> dict:
    """Analyze information coverage and identify gaps"""
```

**Coverage Metrics**:
- Keyword coverage from query
- Source diversity assessment
- Content depth evaluation
- Gap identification

### 4. `reflection_agent.py` - Quality Assessment

**Purpose**: Provides comprehensive quality assessment and self-reflection capabilities.

**Assessment Dimensions**:

#### Quality Score Calculation
```python
def _assess_answer_quality(
    question: str,
    answer: str,
    sources: List[DocumentChunk]
) -> float:
    """Multi-factor quality assessment"""
```

**Quality Factors** (weighted):
- **Length Appropriateness** (15%): Optimal length for question type
- **Question-Answer Relevance** (30%): Direct addressing of query
- **Source Utilization** (25%): Effective use of retrieved sources
- **Language Quality** (15%): Clarity and vocabulary diversity
- **Structure** (15%): Organization and flow

#### Completeness Analysis
```python
def _assess_completeness(self, question: str, answer: str) -> float:
    """Assess coverage of question components"""
```

**Completeness Factors**:
- Question component identification
- Coverage verification per component
- Question complexity adjustment
- Missing aspect identification

#### Missing Information Detection
```python
def _identify_missing_information(
    question: str,
    answer: str,
    sources: List[DocumentChunk]
) -> List[str]:
    """Identify gaps in answer coverage"""
```

**Gap Types**:
- **Question Type Gaps**: Missing definitions, processes, or explanations
- **Source Diversity Gaps**: Limited source variety
- **Depth Gaps**: Insufficient detail or examples
- **Perspective Gaps**: Missing alternative viewpoints

#### Confidence Assessment
```python
def _assess_accuracy(self, answer: str, sources: List[DocumentChunk]) -> str:
    """Determine confidence level based on source quality"""
```

**Confidence Levels**:
- **High Confidence**: Strong source quality + factual consistency
- **Medium Confidence**: Good sources with minor gaps
- **Low Confidence**: Limited or inconsistent sources

## 🔄 Agent Interaction Patterns

### 1. Sequential Processing Pattern
```python
# Main workflow in agentic_rag_service.py
query_plan = await self.planning_agent.create_query_plan(question)
search_results, state = await self.search_agent.execute_iterative_search(query_plan)
reflection = await self.reflection_agent.reflect_on_answer(question, answer, sources)
```

### 2. Feedback Loop Pattern
```python
# Reflection-based additional search
if reflection.needs_more_search:
    additional_search = await self._perform_reflection_based_search(
        question, reflection.missing_information, all_sources
    )
```

### 3. Adaptive Strategy Pattern
```python
# Planning influences search strategy
search_strategy = planning_agent.determine_search_strategy(query_type, complexity)
search_agent.execute_with_strategy(search_strategy)
```

## 📊 Performance Monitoring

### Metrics Collection
Each agent tracks performance metrics:

```python
# In agentic_rag_service.py
self.metrics.record_query(
    query_type='agentic',
    duration=processing_time,
    status='success'
)
```

### Health Monitoring
```python
async def get_health_status(self) -> Dict[str, Any]:
    """Comprehensive health assessment"""
    return {
        "success_rate": successful_queries / total_queries,
        "avg_processing_time": avg_time,
        "avg_iterations": avg_iterations,
        "confidence_scores": confidence_distribution
    }
```

## 🛠️ Configuration and Customization

### Agent Configuration
```python
# In config.py
@dataclass
class RAGConfig:
    # Agent behavior
    max_iterations: int = 3
    min_sources_per_iteration: int = 3
    enable_query_planning: bool = True
    enable_source_triangulation: bool = True
    enable_self_reflection: bool = True
    
    # Quality thresholds
    min_confidence_threshold: float = 0.3
    reflection_quality_threshold: float = 0.6
```

### Customization Points

#### 1. Custom Planning Strategies
```python
# Add new query classification patterns
self.analytical_patterns.append(r'\b(deep_dive|comprehensive_analysis)\b')

# Add domain-specific indicators
self.domain_indicators['finance'] = ['trading', 'investment', 'portfolio']
```

#### 2. Custom Search Strategies
```python
# Add new search focus areas
def _determine_search_focus(self, iteration: int, existing_data: List) -> str:
    if custom_condition:
        return "domain_specific_search"
    return default_focus
```

#### 3. Custom Reflection Criteria
```python
# Add domain-specific quality assessments
def _assess_domain_quality(self, answer: str, domain: str) -> float:
    if domain == "medical":
        return self._assess_medical_accuracy(answer)
    return default_quality_score
```

## 🚀 Usage Examples

### Basic Agentic Processing
```python
rag_service = AgenticRAGService(config)

response = await rag_service.ask_with_planning(
    question="Analyze the effectiveness of different machine learning algorithms for time series forecasting",
    enable_iteration=True,
    enable_reflection=True,
    enable_triangulation=True
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2f}")
print(f"Iterations: {response.iterations_completed}")
print(f"Quality Score: {response.reflection.quality_score:.2f}")
```

### Simplified Processing
```python
response = await rag_service.ask(
    question="What is TensorFlow?",
    search_method="hybrid",
    max_results=5
)
```

### Query Analysis Only
```python
analysis = await rag_service.analyze_query(
    "Compare the performance characteristics of Redis vs MongoDB for caching use cases"
)

print(f"Complexity: {analysis.complexity_score}")
print(f"Strategy: {analysis.recommended_strategy}")
print(f"Estimated Time: {analysis.estimated_processing_time}s")
```

## 🔧 Debugging and Troubleshooting

### Debug Mode
```python
# Enable detailed logging
config.debug_mode = True
config.log_reasoning_steps = True

# Use debug endpoint
response = await rag_service.debug_query(
    question="test question",
    debug_level="full",
    include_intermediate_results=True
)
```

### Common Issues and Solutions

#### 1. Low Quality Scores
- **Cause**: Poor source utilization or unclear answers
- **Solution**: Adjust similarity thresholds, enable reranking

#### 2. Slow Processing
- **Cause**: Too many iterations or complex triangulation
- **Solution**: Reduce max_iterations, disable triangulation for speed

#### 3. Inconsistent Results
- **Cause**: Varying source quality or search randomness
- **Solution**: Enable caching, increase source diversity requirements

## 🎯 Best Practices

### 1. Configuration Tuning
- **Start with defaults** and adjust based on use case
- **Monitor performance metrics** to guide optimization
- **Test with representative queries** from your domain

### 2. Error Handling
- **Always provide fallback responses** when agents fail
- **Log detailed error information** for debugging
- **Implement graceful degradation** for partial failures

### 3. Performance Optimization
- **Cache frequently used query plans** and analysis results
- **Batch similar queries** for efficiency
- **Monitor resource usage** and adjust concurrency limits

### 4. Quality Assurance
- **Regularly review reflection feedback** for improvement opportunities
- **Analyze low-confidence responses** for pattern identification
- **Validate agent decisions** against expected outcomes

The agents folder represents the cognitive core of the agentic RAG system, providing the intelligence needed to handle complex queries through planning, iteration, and reflection.