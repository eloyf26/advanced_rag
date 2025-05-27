# `/utils/` Folder Documentation

## Overview

The `/utils/` folder contains essential shared utilities that provide foundational capabilities across the entire agentic RAG system. These utilities handle logging, performance monitoring, input validation, and security - ensuring robust, secure, and observable operation of all system components.

## 🎯 Design Philosophy

- **Reliability**: Battle-tested utilities with comprehensive error handling
- **Observability**: Rich logging and metrics for system visibility
- **Security**: Input validation and sanitization to prevent attacks
- **Performance**: Efficient implementations with minimal overhead
- **Reusability**: Clean interfaces usable across all components

## 📁 File Structure

```
utils/
├── logger.py         # Structured logging with context tracking
├── metrics.py        # Performance monitoring and Prometheus integration
└── validators.py     # Input validation and security sanitization
```

## 🔧 Component Details

### 1. `logger.py` - Structured Logging with Context

**Purpose**: Provides comprehensive logging capabilities with structured JSON output, contextual information tracking, and specialized adapters for RAG operations.

#### Key Features

##### Structured JSON Logging
- Custom formatter that outputs structured JSON logs with timestamps, levels, and contextual metadata
- Automatic extraction of request IDs, query information, processing times, and error contexts
- Support for both human-readable console output and machine-parseable file logs

##### RAG-Specific Logger Adapter
- Specialized logging methods for query processing, search operations, and error handling
- Built-in correlation ID tracking for distributed request tracing
- Automatic performance timing and success/failure status logging

##### Configuration System
- Support for multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Configurable output destinations (console, file, both)
- Log rotation with configurable file size limits and backup counts
- Integration with structured logging libraries like `structlog`

##### Context Managers and Decorators
- `QueryLoggingContext`: Automatically logs query start, completion, and timing
- `PerformanceTimer`: Times operations and logs duration with success status
- Function decorators for automatic performance logging of sync and async functions

**Usage Patterns**:
```python
# Basic setup
logger = get_logger(__name__, component='search_agent')

# Query-specific logging
logger.log_query(logging.INFO, "What is ML?", processing_time=2.5, confidence=0.87)

# Contextual query processing
with QueryLoggingContext("machine learning query", "req_123"):
    result = process_complex_query()

# Performance timing
with PerformanceTimer("embedding_generation"):
    embeddings = generate_embeddings(texts)
```

#### Advanced Features

##### Error Tracking and Context
- Automatic exception logging with stack traces and contextual information
- Error categorization and tracking for pattern analysis
- Integration with error tracking services through structured output

##### HTTP Request Middleware
- Automatic logging of all HTTP requests with timing, status codes, and client information
- Correlation ID propagation across service boundaries
- Request/response payload logging (configurable for security)

### 2. `metrics.py` - Performance Monitoring and Prometheus Integration

**Purpose**: Comprehensive performance monitoring system with time-series data collection, Prometheus integration, and real-time dashboard generation.

#### Core Architecture

##### Multi-Backend Metrics Collection
- In-memory time-series storage with configurable retention periods
- Prometheus metrics integration for production monitoring
- Automatic metric aggregation and statistical analysis
- Thread-safe operations with proper locking mechanisms

##### Time Series Data Management
- Sliding window data retention with automatic cleanup
- Statistical analysis including mean, median, percentiles, and standard deviation
- Tag-based metric organization for multi-dimensional analysis
- Efficient memory usage with circular buffers

#### Metrics Categories

##### Query Metrics
- Total queries processed, success/failure rates, processing times
- Query type breakdown (agentic vs simple RAG)
- Confidence score distributions and trends
- Token usage tracking and cost analysis

##### Search Metrics
- Search method performance (hybrid, semantic, keyword)
- Results quality metrics and relevance scores
- Cache hit rates and performance improvements
- Source diversity and triangulation effectiveness

##### System Metrics
- Memory usage, CPU utilization, and resource consumption
- Active connection counts and queue depths
- Error rates by component and error type
- Response time percentiles and SLA compliance

##### Agent-Specific Metrics
- Planning agent decision accuracy and timing
- Search iteration effectiveness and convergence rates
- Reflection quality scores and improvement suggestions
- Triangulation success rates and bias detection

#### Prometheus Integration

##### Pre-configured Metric Types
- **Counters**: Total requests, errors, cache hits
- **Histograms**: Response times, processing durations, token usage
- **Gauges**: Active connections, queue sizes, memory usage
- **Summaries**: Complex multi-dimensional metrics with quantiles

##### Dashboard Generation
- Built-in HTML dashboard generator with real-time updates
- Grafana-compatible metric exposition
- Alert threshold configuration and notification integration
- Historical trend analysis and capacity planning metrics

**Usage Examples**:
```python
# Global metrics instance
metrics = get_metrics_collector()

# Record query processing
metrics.record_query('agentic', duration=2.5, status='success', confidence=0.87)

# Track search operations
metrics.record_search('hybrid', duration=0.8, results_count=10, similarity_threshold=0.7)

# Performance profiling
profiler = PerformanceProfiler(metrics)
profile_id = profiler.start_profile("complex_query", "agentic_processing")
profiler.add_checkpoint(profile_id, "planning_complete")
profiler.add_checkpoint(profile_id, "search_complete")
results = profiler.end_profile(profile_id)
```

#### Advanced Features

##### Performance Profiling
- Multi-checkpoint timing for complex operations
- Automatic bottleneck identification and reporting
- Performance regression detection and alerting
- Resource usage correlation with performance metrics

##### Dashboard and Visualization
- Real-time HTML dashboard with automatic refresh
- Export capabilities for external monitoring systems
- Custom metric visualization with charts and graphs
- Alert integration for threshold breaches

##### Background Monitoring
- Automatic periodic metrics collection and export
- System health monitoring with configurable thresholds
- Capacity planning metrics and trend analysis
- Integration with external monitoring platforms (Datadog, New Relic)

### 3. `validators.py` - Input Validation and Security

**Purpose**: Comprehensive input validation, sanitization, and security controls to protect against injection attacks, ensure data integrity, and maintain system stability.

#### Security-First Validation

##### Query Security Validation
- SQL injection pattern detection using regex and heuristics
- XSS (Cross-Site Scripting) prevention with HTML tag filtering
- Command injection protection for system-level operations
- Length limits and character encoding validation

##### File Security Validation
- Extension whitelist/blacklist with security-focused defaults
- File signature verification to prevent extension spoofing
- Size limits and resource consumption protection
- Content scanning for embedded executables and malicious payloads

#### Validation Categories

##### Input Sanitization
- **Query Sanitization**: Remove dangerous characters while preserving search intent
- **Filename Sanitization**: Safe filename generation for storage operations
- **User Input Cleaning**: General purpose text cleaning with configurable strictness
- **Path Traversal Prevention**: Protection against directory traversal attacks

##### Data Type Validation
- **Similarity Thresholds**: Range validation (0.0-1.0) with type coercion
- **Result Limits**: Integer validation with reasonable bounds (1-1000)
- **Date Filters**: Multiple format support with relative date parsing
- **File Types**: Extension validation with category-based grouping

##### Request Validation
- **Search Requests**: Complete validation of query parameters and filters
- **Batch Requests**: Array validation with concurrency and timeout limits
- **Configuration Updates**: Runtime parameter validation with safety checks
- **File Upload Requests**: Multi-stage validation for ingestion operations

#### Advanced Security Features

##### Rate Limiting
- In-memory rate limiting with sliding window algorithms
- Per-client tracking with configurable limits and time windows
- Automatic cooldown periods and progressive penalties
- Integration with reverse proxy rate limiting

##### Input Pattern Analysis
- Machine learning-based anomaly detection for unusual input patterns
- Behavioral analysis for potential attack identification
- Automatic blacklisting of suspicious IP addresses or API keys
- Integration with threat intelligence feeds

**Usage Patterns**:
```python
# Basic validation
validated_query = validate_query("What is machine learning?")
validated_files = validate_file_paths(["/path/to/doc1.pdf", "/path/to/doc2.txt"])

# Request validation
request_validator = RequestValidator()
validated_request = request_validator.validate_search_request({
    "question": "Compare ML frameworks",
    "max_results": 10,
    "file_types": ["pdf", "md"]
})

# Rate limiting
if not check_rate_limit(client_id):
    raise HTTPException(status_code=429, detail="Rate limit exceeded")

# File security
file_validator = FileValidator(max_size_mb=50, allowed_categories=['documents'])
safe_path = file_validator.validate_path("/uploads/document.pdf")
```

#### Configuration and Customization

##### Validator Configuration
- Configurable security levels (strict, moderate, permissive)
- Custom pattern definitions for domain-specific validation
- Integration with external security services and threat feeds
- Whitelist/blacklist management with automatic updates

##### Error Handling and Recovery
- Detailed error messages with security-safe information disclosure
- Automatic sanitization suggestions for rejected inputs
- Graceful degradation for partial validation failures
- Audit logging for all validation decisions and security events

## 🔄 Integration Patterns

### Cross-Component Usage

#### Logging Integration
Every component in the system uses the structured logging utilities:
- Agents log decision-making processes and performance metrics
- Services log API calls, database operations, and error conditions
- Tools log analysis results and processing statistics
- Main service logs request handling and response generation

#### Metrics Collection
Comprehensive metrics collection across all system components:
- Database operations (query times, connection pool usage)
- LLM service calls (token usage, response times, costs)
- Cache performance (hit rates, eviction statistics)
- Agent decision quality (confidence scores, iteration counts)

#### Validation Pipeline
Multi-layer validation ensures system security and stability:
- API gateway validation for all incoming requests
- Component-level validation for internal operations
- Data storage validation for persistence operations
- Real-time monitoring for validation failure patterns

### Configuration Management

#### Environment-Based Configuration
All utilities respect environment variables for configuration:
- `LOG_LEVEL`: Controls logging verbosity
- `ENABLE_PROMETHEUS_METRICS`: Toggles Prometheus integration
- `ENABLE_RATE_LIMITING`: Controls rate limiting enforcement
- `MAX_FILE_SIZE_MB`: Sets file upload limits

#### Runtime Configuration Updates
Support for runtime configuration changes without service restart:
- Log level adjustments for debugging
- Metrics collection interval modifications
- Validation rule updates for new threat patterns
- Rate limiting threshold adjustments

## 🚀 Production Deployment

### Monitoring and Observability

#### Log Aggregation
- Structured JSON logs compatible with ELK stack, Splunk, and other log aggregators
- Automatic correlation ID propagation for distributed tracing
- Log sampling and filtering to manage volume in high-traffic scenarios
- Integration with cloud logging services (CloudWatch, Google Cloud Logging)

#### Metrics and Alerting
- Prometheus metrics export for integration with monitoring infrastructure
- Pre-configured Grafana dashboards for system visualization
- Alert rules for critical system metrics (error rates, response times)
- Integration with PagerDuty, Slack, and other notification systems

### Security and Compliance

#### Security Monitoring
- Real-time attack detection and prevention
- Audit logging for all security-relevant events
- Compliance reporting for data protection regulations
- Integration with SIEM systems for security event correlation

#### Performance Optimization
- Automatic performance baseline establishment
- Anomaly detection for performance regressions
- Resource usage optimization recommendations
- Capacity planning support with growth projections

## 🛠️ Development and Testing

### Local Development
- Simplified logging configuration for development environments
- Mock metrics collectors for testing without external dependencies
- Relaxed validation rules for development and testing scenarios
- Debug modes with enhanced logging and error reporting

### Testing Support
- Test fixtures for validation scenarios
- Mock metrics collection for unit testing
- Logging capture utilities for test assertions
- Performance testing helpers with timing validation

### Customization and Extension
- Plugin architecture for custom validators
- Configurable logging formatters and processors
- Custom metrics collectors for domain-specific monitoring
- Extension points for additional security validation rules

The utils folder provides the essential infrastructure that makes the agentic RAG system production-ready, secure, and observable. These utilities ensure that every component operates reliably while providing the visibility and security controls necessary for enterprise deployment.