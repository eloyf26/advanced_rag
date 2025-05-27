# `/tools/` Folder Documentation

## Overview

The `/tools/` folder contains specialized utility functions that provide advanced capabilities for search execution, content analysis, and source verification. These tools implement sophisticated algorithms for pattern detection, quality assessment, and information validation that support the agentic decision-making process.

## 🎯 Design Philosophy

- **Modularity**: Self-contained tools with clear interfaces
- **Extensibility**: Easy to add new analysis and search capabilities
- **Intelligence**: Advanced algorithms for pattern recognition and assessment
- **Reliability**: Robust error handling and fallback mechanisms
- **Performance**: Optimized for efficiency and scalability

## 📁 File Structure

```
tools/
├── search_tools.py          # Advanced search implementations
├── analysis_tools.py        # Query and content analysis utilities
└── triangulation_tools.py   # Source verification and bias detection
```

## 🔧 Component Details

### 1. `search_tools.py` - Advanced Search Implementations

**Purpose**: Provides sophisticated search capabilities beyond basic database operations, including multi-strategy search, adaptive algorithms, and result optimization.

#### Core Search Strategies

##### Multi-Strategy Search
```python
class SearchToolkit:
    def __init__(self, db_manager: RAGDatabaseManager):
        self.search_strategies = {
            'hybrid': self.hybrid_search,
            'semantic': self.semantic_search,
            'keyword': self.keyword_search,
            'contextual': self.contextual_search,
            'similarity': self.similarity_search
        }
```

**Available Strategies**:
- **Hybrid**: Vector + keyword combination
- **Semantic**: Pure vector similarity
- **Keyword**: BM25 text matching
- **Contextual**: Context-aware search
- **Similarity**: Document-to-document similarity

##### Contextual Search
```python
async def contextual_search(
    self, 
    query: str, 
    filters: SearchFilters,
    context_chunks: List[DocumentChunk] = None
) -> SearchResults:
    """Perform contextual search considering existing context"""
    
    # Enhance query with context information
    enhanced_query = self._enhance_query_with_context(query, context_chunks)
    
    # Perform hybrid search with enhanced query
    chunks = await self.db_manager.hybrid_search(enhanced_query, filters)
    
    # Filter out chunks too similar to context
    if context_chunks:
        chunks = self._filter_similar_to_context(chunks, context_chunks)
    
    return SearchResults(
        query=enhanced_query,
        chunks=chunks,
        search_method="contextual"
    )
```

**Context Enhancement Features**:
- **Query Expansion**: Add relevant terms from existing context
- **Duplicate Filtering**: Remove overly similar results
- **Relevance Scoring**: Boost novel information
- **Context Bridging**: Find connecting information

##### Adaptive Search
```python
async def adaptive_search(
    self, 
    query: str, 
    filters: SearchFilters,
    fallback_strategies: List[str] = None
) -> SearchResults:
    """Adaptive search that tries different strategies based on results quality"""
    
    if fallback_strategies is None:
        fallback_strategies = ['hybrid', 'semantic', 'keyword']
    
    best_result = None
    best_score = 0.0
    
    for strategy in fallback_strategies:
        try:
            result = await self.search_strategies[strategy](query, filters)
            
            # Calculate quality score for this result
            quality_score = self._calculate_result_quality(result)
            
            if quality_score > best_score:
                best_score = quality_score
                best_result = result
            
            # Stop if we get good results
            if quality_score > 0.8:
                break
                
        except Exception as e:
            logger.error(f"Error in adaptive search with {strategy}: {e}")
            continue
    
    return best_result or self._create_empty_search_results(query, "adaptive", filters)
```

#### Advanced Search Features

##### Search with Expansion
```python
async def search_with_expansion(
    self, 
    query: str, 
    filters: SearchFilters,
    expansion_terms: List[str] = None
) -> SearchResults:
    """Search with query expansion using related terms"""
    
    # Generate expansion terms if not provided
    if expansion_terms is None:
        expansion_terms = await self._generate_expansion_terms(query)
    
    # Create expanded query
    expanded_query = f"{query} {' '.join(expansion_terms)}"
    
    return await self.hybrid_search(expanded_query, filters)

def _generate_expansion_terms(self, query: str) -> List[str]:
    """Generate query expansion terms"""
    expansion_terms = []
    query_lower = query.lower()
    
    # Domain-specific expansion
    if 'machine learning' in query_lower:
        expansion_terms.extend(['AI', 'artificial intelligence', 'ML', 'algorithms'])
    elif 'data science' in query_lower:
        expansion_terms.extend(['analytics', 'statistics', 'data analysis'])
    elif 'programming' in query_lower:
        expansion_terms.extend(['coding', 'development', 'software'])
    
    return expansion_terms[:3]  # Limit to 3 terms
```

##### Federated Search
```python
async def federated_search(
    self, 
    query: str, 
    filters_list: List[SearchFilters]
) -> List[SearchResults]:
    """Perform federated search across different filter configurations"""
    
    # Execute searches with different filters concurrently
    tasks = [self.hybrid_search(query, filters) for filters in filters_list]
    search_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for i, result in enumerate(search_results):
        if isinstance(result, Exception):
            logger.error(f"Error in federated search {i}: {result}")
            results.append(self._create_empty_search_results(query, "federated", filters_list[i]))
        else:
            results.append(result)
    
    return results
```

#### Quality Assessment

##### Result Quality Calculation
```python
def _calculate_result_quality(self, result: SearchResults) -> float:
    """Calculate quality score for search results"""
    
    if not result.chunks:
        return 0.0
    
    quality_factors = []
    
    # Factor 1: Number of results (normalized)
    result_count_score = min(len(result.chunks) / 5.0, 1.0)
    quality_factors.append(result_count_score)
    
    # Factor 2: Average similarity/relevance scores
    avg_similarity = sum(chunk.similarity_score for chunk in result.chunks) / len(result.chunks)
    quality_factors.append(avg_similarity)
    
    # Factor 3: Source diversity
    unique_sources = len(set(chunk.file_name for chunk in result.chunks))
    source_diversity = min(unique_sources / 3.0, 1.0)
    quality_factors.append(source_diversity)
    
    # Factor 4: Content length (indicates detail level)
    avg_content_length = sum(len(chunk.content) for chunk in result.chunks) / len(result.chunks)
    length_score = min(avg_content_length / 500.0, 1.0)
    quality_factors.append(length_score)
    
    return sum(quality_factors) / len(quality_factors)
```

### 2. `analysis_tools.py` - Query and Content Analysis

**Purpose**: Provides comprehensive analysis capabilities for queries, content patterns, and search effectiveness to support intelligent decision-making.

#### Query Analysis

##### Comprehensive Query Analysis
```python
class AnalysisToolkit:
    async def analyze_query(self, question: str) -> QueryAnalysisResponse:
        """Comprehensive query analysis"""
        
        # Basic analysis components
        complexity_score = self._calculate_complexity_score(question)
        query_type = self._classify_query_type(question)
        key_concepts = self._extract_key_concepts(question)
        
        # Domain and strategy analysis
        predicted_domains = self._predict_domains(question)
        predicted_sources = self._predict_source_types(question, predicted_domains)
        recommended_strategy = self._recommend_strategy(query_type, complexity_score)
        
        # Optimization and challenges
        optimization_suggestions = self._generate_optimization_suggestions(question)
        potential_challenges = self._identify_potential_challenges(question, complexity_score)
        
        return QueryAnalysisResponse(
            query=question,
            complexity_score=complexity_score,
            query_type=query_type,
            key_concepts=key_concepts,
            predicted_sources=predicted_sources,
            recommended_strategy=recommended_strategy,
            optimization_suggestions=optimization_suggestions,
            potential_challenges=potential_challenges
        )
```

##### Complexity Scoring
```python
def _calculate_complexity_score(self, question: str) -> float:
    """Calculate question complexity score (0-1)"""
    
    complexity_factors = []
    question_lower = question.lower()
    
    # Length factor
    word_count = len(question.split())
    length_factor = min(word_count / 25.0, 1.0)
    complexity_factors.append(length_factor)
    
    # Complexity indicators
    high_complexity_indicators = [
        'comprehensive', 'detailed', 'analyze', 'compare', 'evaluate'
    ]
    indicator_count = sum(1 for indicator in high_complexity_indicators if indicator in question_lower)
    indicator_factor = min(indicator_count / 3.0, 1.0)
    complexity_factors.append(indicator_factor)
    
    # Multi-part question factor
    conjunctions = ['and', 'or', 'also', 'additionally', 'furthermore']
    multi_part_factor = min(
        sum(1 for conj in conjunctions if conj in question_lower) / 3.0, 1.0
    )
    complexity_factors.append(multi_part_factor)
    
    # Technical terminology
    technical_terms = self._count_technical_terms(question)
    technical_factor = min(technical_terms / 5.0, 1.0)
    complexity_factors.append(technical_factor)
    
    # Calculate weighted average
    weights = [0.2, 0.3, 0.2, 0.3]
    return sum(w * f for w, f in zip(weights, complexity_factors))
```

##### Query Classification
```python
def _classify_query_type(self, question: str) -> str:
    """Classify the type of query"""
    
    question_lower = question.lower().strip()
    
    # Classification patterns
    if any(question_lower.startswith(pattern) for pattern in [
        'what is', 'what are', 'define', 'explain what'
    ]):
        return 'definitional'
    
    elif any(question_lower.startswith(pattern) for pattern in [
        'how to', 'how do', 'how can', 'what steps'
    ]):
        return 'procedural'
    
    elif any(pattern in question_lower for pattern in [
        'compare', 'contrast', 'difference', 'versus', 'vs'
    ]):
        return 'comparative'
    
    elif any(question_lower.startswith(pattern) for pattern in [
        'why', 'why do', 'why is', 'explain why'
    ]):
        return 'causal'
    
    elif any(pattern in question_lower for pattern in [
        'analyze', 'evaluate', 'assess', 'critique'
    ]):
        return 'analytical'
    
    else:
        return 'general'
```

##### Domain Prediction
```python
def _predict_domains(self, question: str) -> List[str]:
    """Predict relevant domains for the question"""
    
    question_lower = question.lower()
    domain_keywords = {
        'technical': [
            'algorithm', 'implementation', 'architecture', 'framework',
            'programming', 'coding', 'software', 'system', 'technical'
        ],
        'business': [
            'strategy', 'market', 'revenue', 'profit', 'customer',
            'business', 'commercial', 'financial', 'economic'
        ],
        'academic': [
            'research', 'study', 'analysis', 'theory', 'methodology',
            'academic', 'scientific', 'scholarly', 'peer-reviewed'
        ],
        'practical': [
            'how-to', 'guide', 'tutorial', 'instructions', 'steps',
            'practical', 'hands-on', 'example', 'case study'
        ]
    }
    
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        if score > 0:
            domain_scores[domain] = score
    
    # Return domains sorted by relevance
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    return [domain for domain, _ in sorted_domains]
```

#### Content Pattern Analysis

##### Content Distribution Analysis
```python
def analyze_content_patterns(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Analyze patterns in retrieved content"""
    
    if not chunks:
        return {'patterns': [], 'insights': []}
    
    analysis = {
        'content_distribution': self._analyze_content_distribution(chunks),
        'source_diversity': self._analyze_source_diversity(chunks),
        'topic_coverage': self._analyze_topic_coverage(chunks),
        'quality_indicators': self._analyze_quality_indicators(chunks),
        'content_gaps': self._identify_content_gaps(chunks),
        'redundancy_analysis': self._analyze_content_redundancy(chunks)
    }
    
    return analysis

def _analyze_source_diversity(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Analyze diversity of sources"""
    
    if not chunks:
        return {}
    
    unique_files = set(chunk.file_name for chunk in chunks)
    unique_types = set(chunk.file_type for chunk in chunks)
    
    # Calculate diversity metrics
    file_diversity = len(unique_files) / len(chunks)
    type_diversity = len(unique_types) / max(len(chunks), 1)
    
    return {
        'unique_files': len(unique_files),
        'unique_file_types': len(unique_types),
        'file_diversity_ratio': file_diversity,
        'type_diversity_ratio': type_diversity,
        'diversity_score': (file_diversity + type_diversity) / 2
    }
```

##### Topic Coverage Analysis
```python
def _analyze_topic_coverage(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Analyze topic coverage across chunks"""
    
    if not chunks:
        return {}
    
    # Collect all keywords
    all_keywords = []
    for chunk in chunks:
        if chunk.keywords:
            all_keywords.extend(chunk.keywords)
    
    # Count keyword frequencies
    keyword_counts = Counter(all_keywords)
    main_topics = [keyword for keyword, count in keyword_counts.most_common(10)]
    
    return {
        'total_keywords': len(all_keywords),
        'unique_keywords': len(set(all_keywords)),
        'main_topics': main_topics,
        'keyword_frequency': dict(keyword_counts.most_common(20))
    }
```

#### Search Effectiveness Analysis

##### Search Performance Assessment
```python
def analyze_search_effectiveness(
    self, 
    query: str, 
    results: List[DocumentChunk]
) -> Dict[str, Any]:
    """Analyze how effectively the search results match the query"""
    
    effectiveness = {
        'relevance_score': self._calculate_relevance_score(query, results),
        'coverage_score': self._calculate_coverage_score(query, results),
        'diversity_score': self._calculate_diversity_score(results),
        'completeness_assessment': self._assess_completeness(query, results),
        'improvement_suggestions': self._suggest_search_improvements(query, results)
    }
    
    return effectiveness

def _calculate_relevance_score(self, query: str, results: List[DocumentChunk]) -> float:
    """Calculate overall relevance score for search results"""
    
    if not results:
        return 0.0
    
    query_words = set(query.lower().split())
    relevance_scores = []
    
    for chunk in results:
        # Calculate word overlap
        content_words = set(chunk.content.lower().split())
        overlap = len(query_words & content_words) / max(len(query_words), 1)
        
        # Weight by similarity score if available
        weighted_relevance = overlap * (chunk.similarity_score if chunk.similarity_score > 0 else 0.5)
        relevance_scores.append(weighted_relevance)
    
    return statistics.mean(relevance_scores)
```

##### Improvement Suggestions
```python
def _suggest_search_improvements(self, query: str, results: List[DocumentChunk]) -> List[str]:
    """Suggest improvements for search effectiveness"""
    
    suggestions = []
    
    if not results:
        suggestions.append("Try broader search terms or reduce similarity threshold")
        return suggestions
    
    # Analyze current results
    coverage_score = self._calculate_coverage_score(query, results)
    diversity_score = self._calculate_diversity_score(results)
    
    if coverage_score < 0.6:
        suggestions.append("Consider expanding query with related terms")
    
    if diversity_score < 0.5:
        suggestions.append("Try searching across different file types")
    
    if len(results) < 3:
        suggestions.append("Reduce similarity threshold to find more results")
    
    if len(results) > 15:
        suggestions.append("Increase similarity threshold for more focused results")
    
    # Source-specific suggestions
    file_types = set(chunk.file_type for chunk in results)
    if len(file_types) == 1:
        suggestions.append("Search across additional file types for diverse perspectives")
    
    return suggestions
```

### 3. `triangulation_tools.py` - Source Verification and Bias Detection

**Purpose**: Implements sophisticated algorithms for source triangulation, information verification, and bias detection to ensure accuracy and reliability.

#### Source Triangulation

##### Triangulation Process
```python
class TriangulationToolkit:
    async def triangulate_sources(
        self, 
        query: str, 
        primary_sources: List[DocumentChunk], 
        max_additional: int = 5
    ) -> List[DocumentChunk]:
        """Find additional sources to triangulate and verify information"""
        
        if not primary_sources:
            return []
        
        # Extract key concepts and claims from primary sources
        key_concepts = self._extract_key_concepts_from_sources(primary_sources)
        potential_claims = self._extract_potential_claims(primary_sources)
        
        # Generate verification queries
        verification_queries = self._generate_verification_queries(
            query, key_concepts, potential_claims
        )
        
        # Search for triangulating sources
        additional_sources = await self._search_for_triangulating_sources(
            verification_queries, primary_sources, max_additional
        )
        
        # Score and rank by triangulation value
        scored_sources = self._score_triangulation_value(
            additional_sources, primary_sources, key_concepts
        )
        
        return scored_sources
```

##### Key Concept Extraction
```python
def _extract_key_concepts_from_sources(self, sources: List[DocumentChunk]) -> List[str]:
    """Extract key concepts that should be verified across sources"""
    
    all_keywords = []
    concept_frequency = Counter()
    
    # Collect keywords from all sources
    for source in sources:
        if source.keywords:
            all_keywords.extend(source.keywords)
    
    # Extract concepts from content using simple NLP
    for source in sources:
        content_concepts = self._extract_concepts_from_text(source.content)
        all_keywords.extend(content_concepts)
    
    # Count frequency and return most common
    concept_frequency.update(all_keywords)
    return [concept for concept, _ in concept_frequency.most_common(10)]

def _extract_concepts_from_text(self, text: str) -> List[str]:
    """Extract concepts from text using simple NLP techniques"""
    
    # Extract capitalized words (potential proper nouns)
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # Filter for meaningful concepts
    stop_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'In', 'On', 'At'}
    concepts = [word.lower() for word in words if word not in stop_words and len(word) > 3]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for concept in concepts:
        if concept not in seen:
            unique_concepts.append(concept)
            seen.add(concept)
    
    return unique_concepts[:10]
```

##### Verification Query Generation
```python
def _generate_verification_queries(
    self, 
    original_query: str, 
    key_concepts: List[str], 
    claims: List[str]
) -> List[str]:
    """Generate queries to find verification sources"""
    
    verification_queries = []
    
    # Concept-based verification queries
    for concept in key_concepts[:3]:
        verification_queries.extend([
            f"verify {concept} information",
            f"alternative view {concept}",
            f"criticism {concept}",
            f"evidence {concept}"
        ])
    
    # Claim-based verification queries
    for claim in claims[:2]:
        # Extract main subject from claim
        words = claim.split()[:5]  # First 5 words
        subject = ' '.join(words)
        verification_queries.append(f"verify {subject}")
    
    # General verification queries
    main_topic = original_query.split()[:3]  # First 3 words
    topic_str = ' '.join(main_topic)
    verification_queries.extend([
        f"independent research {topic_str}",
        f"peer review {topic_str}",
        f"fact check {topic_str}"
    ])
    
    return verification_queries[:8]  # Limit to 8 queries
```

#### Information Verification

##### Consistency Verification
```python
async def verify_information_consistency(
    self, 
    sources: List[DocumentChunk]
) -> Dict[str, Any]:
    """Verify consistency of information across sources"""
    
    verification_results = {
        'consistency_score': 0.0,
        'cross_references': [],
        'contradictions': [],
        'consensus_points': [],
        'reliability_assessment': {},
        'source_authority_scores': {}
    }
    
    if len(sources) < 2:
        verification_results['consistency_score'] = 0.5
        return verification_results
    
    # Analyze cross-references
    cross_refs = await self._analyze_cross_references(sources)
    verification_results['cross_references'] = cross_refs
    
    # Detect contradictions
    contradictions = self._detect_information_contradictions(sources)
    verification_results['contradictions'] = contradictions
    
    # Find consensus points
    consensus = self._find_consensus_points(sources)
    verification_results['consensus_points'] = consensus
    
    # Assess source reliability
    reliability = self._assess_source_reliability(sources)
    verification_results['reliability_assessment'] = reliability
    
    # Calculate overall consistency score
    consistency_score = self._calculate_consistency_score(
        cross_refs, contradictions, consensus, reliability
    )
    verification_results['consistency_score'] = consistency_score
    
    return verification_results
```

##### Contradiction Detection
```python
def _detect_information_contradictions(self, sources: List[DocumentChunk]) -> List[Dict[str, Any]]:
    """Detect potential contradictions between sources"""
    
    contradictions = []
    
    # Opposing word pairs that indicate contradictions
    opposing_pairs = [
        ('increase', 'decrease'), ('positive', 'negative'), ('effective', 'ineffective'),
        ('successful', 'failed'), ('good', 'bad'), ('high', 'low'),
        ('better', 'worse'), ('advantage', 'disadvantage')
    ]
    
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources[i+1:], i+1):
            content1_lower = source1.content.lower()
            content2_lower = source2.content.lower()
            
            for pos_term, neg_term in opposing_pairs:
                if pos_term in content1_lower and neg_term in content2_lower:
                    contradictions.append({
                        'source1_id': source1.id,
                        'source2_id': source2.id,
                        'source1_file': source1.file_name,
                        'source2_file': source2.file_name,
                        'contradiction_type': f"{pos_term} vs {neg_term}",
                        'confidence': 0.6
                    })
    
    return contradictions
```

##### Consensus Analysis
```python
def _find_consensus_points(self, sources: List[DocumentChunk]) -> List[Dict[str, Any]]:
    """Find points of consensus across sources"""
    
    consensus_points = []
    
    # Find concepts mentioned in multiple sources
    concept_sources = defaultdict(list)
    
    for source in sources:
        if source.keywords:
            for keyword in source.keywords:
                concept_sources[keyword].append(source.id)
    
    # Identify consensus (concepts in multiple sources)
    for concept, source_ids in concept_sources.items():
        if len(source_ids) >= max(2, len(sources) * 0.5):  # At least 2 sources or 50%
            consensus_points.append({
                'concept': concept,
                'supporting_sources': source_ids,
                'consensus_strength': len(source_ids) / len(sources),
                'source_count': len(source_ids)
            })
    
    # Sort by consensus strength
    consensus_points.sort(key=lambda x: x['consensus_strength'], reverse=True)
    return consensus_points[:10]  # Top 10 consensus points
```

#### Credibility and Bias Analysis

##### Source Credibility Assessment
```python
def analyze_source_credibility(self, sources: List[DocumentChunk]) -> Dict[str, Any]:
    """Analyze credibility and authority of sources"""
    
    credibility_analysis = {
        'overall_credibility': 0.0,
        'source_types': {},
        'authority_indicators': {},
        'credibility_scores': {},
        'recommendations': []
    }
    
    if not sources:
        return credibility_analysis
    
    individual_scores = []
    source_type_counts = Counter()
    
    for source in sources:
        # Calculate individual credibility score
        cred_score = self._calculate_source_credibility(source)
        individual_scores.append(cred_score)
        credibility_analysis['credibility_scores'][source.id] = cred_score
        
        # Identify source type
        source_type = self._identify_source_type(source)
        source_type_counts[source_type] += 1
        
        # Find authority indicators
        authority_indicators = self._find_authority_indicators(source)
        if authority_indicators:
            credibility_analysis['authority_indicators'][source.id] = authority_indicators
    
    # Calculate overall credibility
    credibility_analysis['overall_credibility'] = statistics.mean(individual_scores)
    credibility_analysis['source_types'] = dict(source_type_counts)
    
    # Generate recommendations
    credibility_analysis['recommendations'] = self._generate_credibility_recommendations(
        individual_scores, source_type_counts
    )
    
    return credibility_analysis
```

##### Source Type Identification
```python
def _identify_source_type(self, source: DocumentChunk) -> str:
    """Identify the type/category of a source"""
    
    content_lower = source.content.lower()
    
    reliability_indicators = {
        'academic': ['peer-reviewed', 'journal', 'research', 'study', 'university'],
        'official': ['government', 'official', 'policy', 'regulation', 'statute'],
        'industry': ['industry report', 'white paper', 'technical specification'],
        'news': ['news', 'article', 'report', 'press release'],
        'blog': ['blog', 'opinion', 'personal', 'thoughts']
    }
    
    # Check for academic indicators
    if any(indicator in content_lower for indicator in reliability_indicators['academic']):
        return 'academic'
    
    # Check for official/government indicators
    if any(indicator in content_lower for indicator in reliability_indicators['official']):
        return 'official'
    
    # Check for industry indicators
    if any(indicator in content_lower for indicator in reliability_indicators['industry']):
        return 'industry'
    
    # Check for news indicators
    if any(indicator in content_lower for indicator in reliability_indicators['news']):
        return 'news'
    
    # Check for blog/opinion indicators
    if any(indicator in content_lower for indicator in reliability_indicators['blog']):
        return 'blog'
    
    # File type based classification
    if source.file_type in ['pdf', 'docx']:
        return 'academic'  # Assume formal documents are academic
    elif source.file_type in ['html', 'txt']:
        return 'news'  # Assume web content is news/blog
    
    return 'unknown'
```

##### Authority Verification
```python
def _find_authority_indicators(self, source: DocumentChunk) -> List[str]:
    """Find indicators of source authority and expertise"""
    
    authority_indicators = []
    content_lower = source.content.lower()
    
    # Authority signals by category
    authority_signals = {
        'academic_credentials': ['phd', 'professor', 'dr.', 'university', 'institute'],
        'official_status': ['government', 'ministry', 'department', 'official'],
        'expertise_indicators': ['expert', 'specialist', 'authority', 'leading researcher'],
        'publication_quality': ['peer-reviewed', 'journal', 'published', 'research'],
        'citations': ['cited', 'references', 'bibliography', 'doi:']
    }
    
    for category, signals in authority_signals.items():
        if any(signal in content_lower for signal in signals):
            authority_indicators.append(category)
    
    return authority_indicators
```

#### Alternative Perspective Discovery

##### Alternative Viewpoint Search
```python
async def find_alternative_perspectives(
    self, 
    query: str, 
    existing_sources: List[DocumentChunk],
    max_alternatives: int = 3
) -> List[DocumentChunk]:
    """Find sources that provide alternative perspectives on the topic"""
    
    # Identify dominant perspective in existing sources
    dominant_perspective = self._identify_dominant_perspective(existing_sources)
    
    # Generate queries for alternative perspectives
    alternative_queries = self._generate_alternative_queries(
        query, dominant_perspective
    )
    
    # Search for alternative viewpoints
    alternative_sources = []
    existing_source_ids = {source.id for source in existing_sources}
    
    for alt_query in alternative_queries:
        filters = SearchFilters(max_results=5, similarity_threshold=0.6)
        chunks = await self.db_manager.hybrid_search(alt_query, filters)
        
        # Filter out existing sources and add different perspectives
        for chunk in chunks:
            if (chunk.id not in existing_source_ids and 
                len(alternative_sources) < max_alternatives):
                
                # Score how different this perspective is
                perspective_score = self._score_perspective_difference(
                    chunk, existing_sources
                )
                if perspective_score > 0.3:  # Threshold for "different enough"
                    chunk.perspective_score = perspective_score
                    alternative_sources.append(chunk)
                    existing_source_ids.add(chunk.id)
    
    return alternative_sources
```

##### Perspective Analysis
```python
def _identify_dominant_perspective(self, sources: List[DocumentChunk]) -> Dict[str, Any]:
    """Identify the dominant perspective in existing sources"""
    
    # Analyze sentiment and viewpoint indicators
    perspective_indicators = {
        'positive': ['good', 'effective', 'successful', 'beneficial', 'advantage'],
        'negative': ['bad', 'ineffective', 'failed', 'harmful', 'disadvantage'],
        'neutral': ['neutral', 'objective', 'balanced', 'unbiased'],
        'critical': ['criticism', 'problem', 'issue', 'concern', 'limitation'],
        'supportive': ['support', 'recommend', 'endorse', 'favor', 'promote']
    }
    
    perspective_counts = Counter()
    
    for source in sources:
        content_lower = source.content.lower()
        for perspective, indicators in perspective_indicators.items():
            count = sum(1 for indicator in indicators if indicator in content_lower)
            perspective_counts[perspective] += count
    
    dominant_perspective = perspective_counts.most_common(1)[0] if perspective_counts else ('neutral', 0)
    
    return {
        'dominant_perspective': dominant_perspective[0],
        'perspective_distribution': dict(perspective_counts),
        'confidence': dominant_perspective[1] / max(sum(perspective_counts.values()), 1)
    }
```

## 🔗 Tool Integration Patterns

### 1. Sequential Analysis Chain
```python
# Analysis -> Search -> Triangulation
analysis = await analysis_toolkit.analyze_query(question)
search_results = await search_toolkit.adaptive_search(question, filters)
verification = await triangulation_toolkit.verify_information_consistency(search_results.chunks)
```

### 2. Parallel Processing
```python
# Multiple analysis types in parallel
tasks = [
    analysis_toolkit.analyze_content_patterns(chunks),
    triangulation_toolkit.analyze_source_credibility(chunks),
    search_toolkit.analyze_search_effectiveness(query, chunks)
]
results = await asyncio.gather(*tasks)
```

### 3. Feedback-Driven Refinement
```python
# Use analysis results to refine search
effectiveness = analysis_toolkit.analyze_search_effectiveness(query, results)
if effectiveness['relevance_score'] < 0.6:
    # Apply suggested improvements
    improved_results = await search_toolkit.search_with_expansion(
        query, filters, effectiveness['improvement_suggestions']
    )
```

## 🛠️ Customization and Extension

### Adding New Analysis Tools
```python
class CustomAnalysisToolkit(AnalysisToolkit):
    def analyze_domain_specificity(self, content: str, domain: str) -> float:
        """Custom domain-specific analysis"""
        domain_indicators = self._get_domain_indicators(domain)
        content_lower = content.lower()
        
        matches = sum(1 for indicator in domain_indicators if indicator in content_lower)
        return min(matches / len(domain_indicators), 1.0)
```

### Custom Search Strategies
```python
class CustomSearchToolkit(SearchToolkit):
    async def domain_aware_search(
        self, 
        query: str, 
        domain: str, 
        filters: SearchFilters
    ) -> SearchResults:
        """Domain-aware search implementation"""
        
        # Enhance query with domain-specific terms
        domain_terms = self._get_domain_terms(domain)
        enhanced_query = f"{query} {' '.join(domain_terms[:3])}"
        
        # Adjust filters for domain
        domain_filters = self._adapt_filters_for_domain(filters, domain)
        
        return await self.hybrid_search(enhanced_query, domain_filters)
```

### Custom Triangulation Methods
```python
class CustomTriangulationToolkit(TriangulationToolkit):
    async def domain_specific_verification(
        self, 
        sources: List[DocumentChunk], 
        domain: str
    ) -> Dict[str, Any]:
        """Domain-specific verification logic"""
        
        if domain == "medical":
            return await self._verify_medical_claims(sources)
        elif domain == "legal":
            return await self._verify_legal_information(sources)
        else:
            return await self.verify_information_consistency(sources)
```

## 🎯 Best Practices

### 1. Analysis Quality
- **Use multiple analysis dimensions** for comprehensive assessment
- **Validate analysis results** against known benchmarks
- **Combine quantitative and qualitative** assessment methods
- **Provide actionable insights** not just metrics

### 2. Search Optimization
- **Implement fallback strategies** for failed searches
- **Monitor search effectiveness** and adapt strategies
- **Balance precision and recall** based on use case
- **Cache expensive operations** like similarity calculations

### 3. Triangulation Accuracy
- **Use multiple verification methods** for critical information
- **Weight sources by credibility** and authority
- **Identify and flag potential biases** clearly
- **Provide confidence levels** for verification results

### 4. Performance Considerations
- **Implement async processing** for I/O operations
- **Batch similar operations** when possible
- **Cache analysis results** for repeated queries
- **Set reasonable timeouts** for external operations

The tools folder provides the analytical intelligence that enables the agentic RAG system to make informed decisions about search strategies, content quality, and information reliability, supporting the sophisticated reasoning capabilities that distinguish agentic systems from traditional RAG implementations.