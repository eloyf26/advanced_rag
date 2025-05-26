"""
Internal Data Models for Agentic RAG Agent
These models are used for internal processing and communication between components
"""

from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


# Enums for internal state management
class ProcessingStage(Enum):
    """Processing stages for tracking query execution"""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    SEARCHING = "searching"
    TRIANGULATING = "triangulating"
    SYNTHESIZING = "synthesizing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchIterationStatus(Enum):
    """Status of search iterations"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConfidenceLevel(Enum):
    """Confidence levels for assessments"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Priority(Enum):
    """Priority levels for processing"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


# Core internal models
class InternalDocumentChunk(BaseModel):
    """Enhanced document chunk for internal processing"""
    
    # Basic fields from DocumentChunk
    id: str
    content: str
    similarity_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0
    file_name: str
    file_type: str
    chunk_index: int
    
    # Enhanced metadata
    title: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    document_id: Optional[str] = None
    
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
    
    class Config:
        use_enum_values = True


class SearchIteration(BaseModel):
    """Internal model for tracking search iterations"""
    
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
    
    class Config:
        use_enum_values = True


class QueryExecutionContext(BaseModel):
    """Internal context for query execution tracking"""
    
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
    
    class Config:
        use_enum_values = True


class ConceptNode(BaseModel):
    """Node representing a concept in query analysis"""
    
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


class QueryConceptGraph(BaseModel):
    """Graph of concepts extracted from query and sources"""
    
    primary_concepts: List[ConceptNode] = Field(default_factory=list)
    secondary_concepts: List[ConceptNode] = Field(default_factory=list)
    concept_relationships: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Coverage analysis
    covered_concepts: Set[str] = Field(default_factory=set)
    missing_concepts: Set[str] = Field(default_factory=set)
    over_represented_concepts: Set[str] = Field(default_factory=set)
    
    # Quality metrics
    graph_completeness: float = 0.0
    concept_clarity: float = 0.0
    relationship_strength: float = 0.0


class FactualClaim(BaseModel):
    """Represents a factual claim that needs verification"""
    
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


class TriangulationResult(BaseModel):
    """Results from source triangulation process"""
    
    primary_sources: List[InternalDocumentChunk]
    verification_sources: List[InternalDocumentChunk]
    
    # Verification results
    verified_claims: List[FactualClaim] = Field(default_factory=list)
    disputed_claims: List[FactualClaim] = Field(default_factory=list)
    unverified_claims: List[FactualClaim] = Field(default_factory=list)
    
    # Consistency analysis
    consistency_score: float = 0.0
    source_agreement: Dict[str, float] = Field(default_factory=dict)
    contradiction_points: List[str] = Field(default_factory=list)
    
    # Source quality assessment
    source_diversity_score: float = 0.0
    authority_scores: Dict[str, float] = Field(default_factory=dict)
    recency_scores: Dict[str, float] = Field(default_factory=dict)


class ReflectionAssessment(BaseModel):
    """Internal model for reflection assessment"""
    
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Quality dimensions
    content_quality: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    
    # Detailed analysis
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_elements: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    # Confidence assessment
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_factors: Dict[str, float] = Field(default_factory=dict)
    uncertainty_areas: List[str] = Field(default_factory=list)
    
    # Follow-up recommendations
    requires_additional_search: bool = False
    suggested_search_directions: List[str] = Field(default_factory=list)
    alternative_approaches: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class ProcessingMetrics(BaseModel):
    """Internal metrics for processing performance"""
    
    # Timing metrics
    total_processing_time: float = 0.0
    planning_time: float = 0.0
    search_time: float = 0.0
    triangulation_time: float = 0.0
    synthesis_time: float = 0.0
    reflection_time: float = 0.0
    
    # Resource metrics
    peak_memory_usage: float = 0.0
    cpu_utilization: float = 0.0
    api_calls_made: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Quality metrics
    sources_retrieved: int = 0
    unique_sources: int = 0
    iterations_completed: int = 0
    confidence_achieved: float = 0.0
    
    # Error metrics
    errors_encountered: int = 0
    warnings_generated: int = 0
    retries_attempted: int = 0


class CacheEntry(BaseModel):
    """Internal model for cache entries"""
    
    cache_key: str
    cache_type: str  # query, embedding, search_result, etc.
    content: Any
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    # Content metadata
    content_hash: Optional[str] = None
    content_size: int = 0
    compression_used: bool = False
    
    # Performance
    generation_time: float = 0.0
    hit_rate: float = 0.0


class WorkflowStep(BaseModel):
    """Individual step in the agentic workflow"""
    
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


class AgentState(BaseModel):
    """Internal state of the agentic system"""
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Current execution
    current_execution: Optional[QueryExecutionContext] = None
    workflow_steps: List[WorkflowStep] = Field(default_factory=list)
    
    # Knowledge state
    concept_graph: Optional[QueryConceptGraph] = None
    active_claims: List[FactualClaim] = Field(default_factory=list)
    
    # Performance state
    metrics: ProcessingMetrics = Field(default_factory=ProcessingMetrics)
    
    # System state
    available_resources: Dict[str, Any] = Field(default_factory=dict)
    active_connections: Dict[str, Any] = Field(default_factory=dict)
    cache_status: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration
    system_config: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class SearchStrategy(BaseModel):
    """Internal model for search strategy configuration"""
    
    strategy_name: str
    primary_method: str  # hybrid, semantic, keyword
    fallback_methods: List[str] = Field(default_factory=list)
    
    # Parameters
    similarity_threshold: float = 0.7
    max_results_per_iteration: int = 10
    enable_reranking: bool = True
    reranking_top_k: int = 20
    
    # Iteration control
    max_iterations: int = 3
    convergence_threshold: float = 0.1
    diversity_threshold: float = 0.3
    
    # Quality control
    min_confidence: float = 0.5
    source_diversity_requirement: float = 0.5
    temporal_preference: Optional[str] = None  # recent, any, historical
    
    # Performance optimization
    enable_caching: bool = True
    parallel_search: bool = False
    batch_size: int = 5


class AnswerSynthesis(BaseModel):
    """Internal model for answer synthesis process"""
    
    synthesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Source material
    primary_sources: List[InternalDocumentChunk]
    supporting_sources: List[InternalDocumentChunk]
    concept_graph: Optional[QueryConceptGraph] = None
    
    # Synthesis process
    synthesis_strategy: str = "comprehensive"  # brief, comprehensive, analytical
    target_length: Optional[int] = None
    include_citations: bool = True
    include_confidence: bool = True
    
    # Quality requirements
    minimum_sources: int = 2
    require_triangulation: bool = False
    fact_checking_level: str = "basic"  # none, basic, thorough
    
    # Output configuration
    structured_output: bool = False
    include_follow_ups: bool = True
    explanation_depth: str = "medium"  # brief, medium, detailed
    
    # Performance tracking
    tokens_used: int = 0
    synthesis_time: float = 0.0
    revision_count: int = 0


# Utility models for internal communication
@dataclass
class ProcessingEvent:
    """Event for internal communication between components"""
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_component: str = "unknown"
    target_component: Optional[str] = None
    priority: Priority = Priority.MEDIUM


@dataclass
class ComponentHealth:
    """Health status of system components"""
    component_name: str
    status: str  # healthy, degraded, unhealthy, offline
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_time: float = 0.0
    error_rate: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


class SystemDiagnostics(BaseModel):
    """Comprehensive system diagnostics"""
    
    # Component health
    component_health: Dict[str, ComponentHealth] = Field(default_factory=dict)
    
    # Performance metrics
    overall_performance: ProcessingMetrics = Field(default_factory=ProcessingMetrics)
    
    # Resource usage
    system_resources: Dict[str, float] = Field(default_factory=dict)
    
    # Error tracking
    recent_errors: List[str] = Field(default_factory=list)
    error_patterns: Dict[str, int] = Field(default_factory=dict)
    
    # Capacity planning
    current_load: float = 0.0
    capacity_utilization: float = 0.0
    bottlenecks: List[str] = Field(default_factory=list)
    
    # Recommendations
    optimization_suggestions: List[str] = Field(default_factory=list)
    scaling_recommendations: List[str] = Field(default_factory=list)


# Factory functions for creating internal models
def create_execution_context(
    user_query: str,
    enable_iteration: bool = True,
    enable_reflection: bool = True,
    enable_triangulation: bool = True,
    priority: Priority = Priority.MEDIUM
) -> QueryExecutionContext:
    """Factory function to create query execution context"""
    return QueryExecutionContext(
        user_query=user_query,
        processed_query=user_query.strip(),
        priority=priority,
        enable_iteration=enable_iteration,
        enable_reflection=enable_reflection,
        enable_triangulation=enable_triangulation
    )


def create_search_iteration(
    iteration_number: int,
    query: str,
    strategy: str = "hybrid",
    similarity_threshold: float = 0.7,
    max_results: int = 10
) -> SearchIteration:
    """Factory function to create search iteration"""
    return SearchIteration(
        iteration_number=iteration_number,
        query=query,
        search_strategy=strategy,
        similarity_threshold=similarity_threshold,
        max_results=max_results
    )


def create_workflow_step(
    step_name: str,
    step_type: str,
    inputs: Dict[str, Any] = None
) -> WorkflowStep:
    """Factory function to create workflow step"""
    return WorkflowStep(
        step_name=step_name,
        step_type=step_type,
        inputs=inputs or {}
    )


# Type aliases for convenience
InternalChunkList = List[InternalDocumentChunk]
SearchIterationList = List[SearchIteration]
ConceptNodeList = List[ConceptNode]
FactualClaimList = List[FactualClaim]
WorkflowStepList = List[WorkflowStep]