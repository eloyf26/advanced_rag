"""
Analysis Tools for Agentic RAG Agent
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import statistics

from config import RAGConfig
from models.response_models import QueryAnalysisResponse, DocumentChunk

logger = logging.getLogger(__name__)


class AnalysisToolkit:
    """
    Collection of analysis tools for query and content analysis
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Analysis patterns and indicators
        self.complexity_indicators = {
            'high': [
                'comprehensive', 'detailed', 'in-depth', 'thorough', 'extensive',
                'systematic', 'methodical', 'analytical', 'critical'
            ],
            'medium': [
                'explain', 'describe', 'compare', 'contrast', 'analyze',
                'evaluate', 'assess', 'discuss'
            ],
            'low': [
                'what', 'when', 'where', 'who', 'define', 'list', 'name'
            ]
        }
        
        self.domain_keywords = {
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
    
    async def analyze_query(self, question: str) -> QueryAnalysisResponse:
        """
        Comprehensive query analysis
        """
        try:
            # Basic analysis
            complexity_score = self._calculate_complexity_score(question)
            query_type = self._classify_query_type(question)
            key_concepts = self._extract_key_concepts(question)
            
            # Domain analysis
            predicted_domains = self._predict_domains(question)
            predicted_sources = self._predict_source_types(question, predicted_domains)
            
            # Processing estimates
            estimated_time = self._estimate_processing_time(complexity_score, len(key_concepts))
            recommended_strategy = self._recommend_strategy(query_type, complexity_score)
            
            # Optimization analysis
            optimization_suggestions = self._generate_optimization_suggestions(question)
            potential_challenges = self._identify_potential_challenges(question, complexity_score)
            
            # Filter suggestions
            suggested_filters = self._suggest_filters(question, predicted_domains)
            
            return QueryAnalysisResponse(
                query=question,
                complexity_score=complexity_score,
                estimated_processing_time=estimated_time,
                recommended_strategy=recommended_strategy,
                query_type=query_type,
                key_concepts=key_concepts,
                predicted_sources=predicted_sources,
                suggested_filters=suggested_filters,
                optimization_suggestions=optimization_suggestions,
                potential_challenges=potential_challenges
            )
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            
            # Return basic analysis on error
            return QueryAnalysisResponse(
                query=question,
                complexity_score=0.5,
                estimated_processing_time=10.0,
                recommended_strategy="hybrid",
                query_type="general",
                key_concepts=[],
                predicted_sources=[],
                suggested_filters={},
                optimization_suggestions=[],
                potential_challenges=["Analysis failed due to error"]
            )
    
    def analyze_content_patterns(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze patterns in retrieved content
        """
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
    
    def analyze_search_effectiveness(
        self, 
        query: str, 
        results: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Analyze how effectively the search results match the query
        """
        effectiveness = {
            'relevance_score': self._calculate_relevance_score(query, results),
            'coverage_score': self._calculate_coverage_score(query, results),
            'diversity_score': self._calculate_diversity_score(results),
            'completeness_assessment': self._assess_completeness(query, results),
            'improvement_suggestions': self._suggest_search_improvements(query, results)
        }
        
        return effectiveness
    
    def _calculate_complexity_score(self, question: str) -> float:
        """
        Calculate question complexity score (0-1)
        """
        complexity_factors = []
        question_lower = question.lower()
        
        # Length factor
        word_count = len(question.split())
        length_factor = min(word_count / 25.0, 1.0)
        complexity_factors.append(length_factor)
        
        # Complexity indicators
        high_complexity_count = sum(
            1 for indicator in self.complexity_indicators['high']
            if indicator in question_lower
        )
        medium_complexity_count = sum(
            1 for indicator in self.complexity_indicators['medium']
            if indicator in question_lower
        )
        
        indicator_factor = min((high_complexity_count * 0.3 + medium_complexity_count * 0.2), 1.0)
        complexity_factors.append(indicator_factor)
        
        # Multi-part question factor
        conjunctions = ['and', 'or', 'also', 'additionally', 'furthermore']
        multi_part_factor = min(
            sum(1 for conj in conjunctions if conj in question_lower) / 3.0,
            1.0
        )
        complexity_factors.append(multi_part_factor)
        
        # Question marks (multiple questions)
        question_mark_factor = min(question.count('?') / 2.0, 1.0)
        complexity_factors.append(question_mark_factor)
        
        # Technical terminology
        technical_terms = self._count_technical_terms(question)
        technical_factor = min(technical_terms / 5.0, 1.0)
        complexity_factors.append(technical_factor)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.2, 0.1, 0.2]
        return sum(w * f for w, f in zip(weights, complexity_factors))
    
    def _classify_query_type(self, question: str) -> str:
        """
        Classify the type of query
        """
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
        
        elif any(question_lower.startswith(pattern) for pattern in [
            'list', 'name', 'identify', 'what are'
        ]):
            return 'enumerative'
        
        elif any(question_lower.startswith(pattern) for pattern in [
            'when', 'where', 'who'
        ]):
            return 'factual'
        
        else:
            return 'general'
    
    def _extract_key_concepts(self, question: str) -> List[str]:
        """
        Extract key concepts from the question
        """
        # Remove stop words and extract meaningful terms
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'about', 'can', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', question.lower())
        
        # Filter meaningful concepts
        concepts = []
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                not word.isdigit()):
                concepts.append(word)
        
        # Group related concepts and remove duplicates
        unique_concepts = list(dict.fromkeys(concepts))  # Preserve order
        
        return unique_concepts[:10]  # Return top 10 concepts
    
    def _predict_domains(self, question: str) -> List[str]:
        """
        Predict relevant domains for the question
        """
        question_lower = question.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domains sorted by relevance
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in sorted_domains]
    
    def _predict_source_types(self, question: str, domains: List[str]) -> List[str]:
        """
        Predict relevant source file types
        """
        predicted_sources = []
        question_lower = question.lower()
        
        # Domain-based predictions
        if 'technical' in domains:
            predicted_sources.extend(['py', 'js', 'java', 'cpp', 'md'])
        
        if 'business' in domains:
            predicted_sources.extend(['xlsx', 'docx', 'pdf'])
        
        if 'academic' in domains:
            predicted_sources.extend(['pdf', 'docx'])
        
        if 'practical' in domains:
            predicted_sources.extend(['md', 'txt', 'html'])
        
        # Content-based predictions
        if any(term in question_lower for term in ['data', 'statistics', 'numbers']):
            predicted_sources.extend(['csv', 'xlsx'])
        
        if any(term in question_lower for term in ['code', 'programming', 'algorithm']):
            predicted_sources.extend(['py', 'js', 'java'])
        
        if any(term in question_lower for term in ['document', 'report', 'paper']):
            predicted_sources.extend(['pdf', 'docx'])
        
        # Remove duplicates and return
        return list(dict.fromkeys(predicted_sources))
    
    def _estimate_processing_time(self, complexity_score: float, concept_count: int) -> float:
        """
        Estimate processing time in seconds
        """
        base_time = 3.0  # Base processing time
        complexity_multiplier = 1 + complexity_score * 2
        concept_multiplier = 1 + (concept_count / 20.0)
        
        estimated_time = base_time * complexity_multiplier * concept_multiplier
        
        return min(estimated_time, 30.0)  # Cap at 30 seconds
    
    def _recommend_strategy(self, query_type: str, complexity_score: float) -> str:
        """
        Recommend search strategy based on analysis
        """
        # High complexity questions benefit from hybrid search
        if complexity_score > 0.7:
            return "hybrid"
        
        # Strategy based on query type
        strategy_mapping = {
            'definitional': 'semantic',  # Conceptual understanding
            'procedural': 'hybrid',      # Need both concepts and specifics
            'comparative': 'hybrid',     # Need comprehensive coverage
            'causal': 'semantic',        # Conceptual relationships
            'analytical': 'hybrid',      # Comprehensive analysis
            'enumerative': 'keyword',    # Specific items/lists
            'factual': 'keyword'         # Specific facts
        }
        
        return strategy_mapping.get(query_type, "hybrid")
    
    def _generate_optimization_suggestions(self, question: str) -> List[str]:
        """
        Generate suggestions for query optimization
        """
        suggestions = []
        question_lower = question.lower()
        
        # Length-based suggestions
        word_count = len(question.split())
        if word_count < 3:
            suggestions.append("Consider adding more specific terms to improve search precision")
        elif word_count > 20:
            suggestions.append("Consider breaking down the question into smaller, more focused queries")
        
        # Ambiguity detection
        ambiguous_terms = ['it', 'this', 'that', 'they', 'those', 'thing', 'stuff']
        if any(term in question_lower for term in ambiguous_terms):
            suggestions.append("Replace ambiguous terms (it, this, that) with specific nouns")
        
        # Specificity suggestions
        if any(term in question_lower for term in ['best', 'good', 'better']):
            suggestions.append("Define criteria for 'best' or 'good' to get more targeted results")
        
        # Context suggestions
        if question_lower.startswith(('how', 'what')) and 'for' not in question_lower:
            suggestions.append("Consider adding context (e.g., 'for beginners', 'in Python', 'in 2024')")
        
        # Question type suggestions
        if '?' not in question:
            suggestions.append("Rephrase as a clear question for better results")
        
        return suggestions
    
    def _identify_potential_challenges(self, question: str, complexity_score: float) -> List[str]:
        """
        Identify potential challenges in processing the question
        """
        challenges = []
        question_lower = question.lower()
        
        # Complexity-based challenges
        if complexity_score > 0.8:
            challenges.append("High complexity may require multiple search iterations")
        
        # Ambiguity challenges
        pronouns = ['it', 'this', 'that', 'they', 'them']
        if any(pronoun in question_lower for pronoun in pronouns):
            challenges.append("Ambiguous references may affect search accuracy")
        
        # Scope challenges
        if any(term in question_lower for term in ['everything', 'all', 'comprehensive']):
            challenges.append("Very broad scope may result in overwhelming results")
        
        # Temporal challenges
        if any(term in question_lower for term in ['latest', 'recent', 'current', 'new']):
            challenges.append("Temporal requirements may limit available sources")
        
        # Domain challenges
        technical_count = self._count_technical_terms(question)
        if technical_count > 3:
            challenges.append("High technical content may require specialized sources")
        
        return challenges
    
    def _suggest_filters(self, question: str, domains: List[str]) -> Dict[str, Any]:
        """
        Suggest appropriate filters for the search
        """
        filters = {}
        question_lower = question.lower()
        
        # File type filters based on domains
        if domains:
            file_type_suggestions = []
            for domain in domains[:2]:  # Top 2 domains
                if domain == 'technical':
                    file_type_suggestions.extend(['py', 'js', 'md'])
                elif domain == 'business':
                    file_type_suggestions.extend(['xlsx', 'docx', 'pdf'])
                elif domain == 'academic':
                    file_type_suggestions.extend(['pdf'])
            
            if file_type_suggestions:
                filters['file_types'] = list(set(file_type_suggestions))
        
        # Date filters
        if any(term in question_lower for term in ['recent', 'latest', 'current', 'new']):
            filters['date_after'] = 'recent'  # Would be converted to actual date
        
        # Similarity threshold
        if any(term in question_lower for term in ['exactly', 'precisely', 'specific']):
            filters['similarity_threshold'] = 0.8
        elif any(term in question_lower for term in ['broadly', 'generally', 'overview']):
            filters['similarity_threshold'] = 0.6
        
        return filters
    
    def _count_technical_terms(self, text: str) -> int:
        """
        Count technical terms in the text
        """
        technical_indicators = [
            'algorithm', 'framework', 'architecture', 'implementation',
            'methodology', 'optimization', 'configuration', 'integration',
            'deployment', 'scalability', 'performance', 'efficiency'
        ]
        
        text_lower = text.lower()
        return sum(1 for term in technical_indicators if term in text_lower)
    
    def _analyze_content_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze distribution of content across chunks
        """
        if not chunks:
            return {}
        
        # Content length distribution
        content_lengths = [len(chunk.content) for chunk in chunks]
        
        # File type distribution
        file_types = [chunk.file_type for chunk in chunks]
        file_type_counts = Counter(file_types)
        
        # Source distribution
        sources = [chunk.file_name for chunk in chunks]
        source_counts = Counter(sources)
        
        return {
            'content_length_stats': {
                'mean': statistics.mean(content_lengths),
                'median': statistics.median(content_lengths),
                'min': min(content_lengths),
                'max': max(content_lengths)
            },
            'file_type_distribution': dict(file_type_counts),
            'source_distribution': dict(source_counts),
            'total_content_length': sum(content_lengths)
        }
    
    def _analyze_source_diversity(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze diversity of sources
        """
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
    
    def _analyze_topic_coverage(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze topic coverage across chunks
        """
        if not chunks:
            return {}
        
        # Collect all keywords
        all_keywords = []
        for chunk in chunks:
            if chunk.keywords:
                all_keywords.extend(chunk.keywords)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # Identify main topics (most frequent keywords)
        main_topics = [keyword for keyword, count in keyword_counts.most_common(10)]
        
        return {
            'total_keywords': len(all_keywords),
            'unique_keywords': len(set(all_keywords)),
            'main_topics': main_topics,
            'keyword_frequency': dict(keyword_counts.most_common(20))
        }
    
    def _analyze_quality_indicators(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze quality indicators of the chunks
        """
        if not chunks:
            return {}
        
        # Similarity scores
        similarity_scores = [chunk.similarity_score for chunk in chunks if chunk.similarity_score > 0]
        
        # Content quality indicators
        has_titles = sum(1 for chunk in chunks if chunk.title)
        has_summaries = sum(1 for chunk in chunks if chunk.summary)
        has_keywords = sum(1 for chunk in chunks if chunk.keywords)
        
        quality_metrics = {
            'avg_similarity_score': statistics.mean(similarity_scores) if similarity_scores else 0,
            'min_similarity_score': min(similarity_scores) if similarity_scores else 0,
            'max_similarity_score': max(similarity_scores) if similarity_scores else 0,
            'chunks_with_titles': has_titles,
            'chunks_with_summaries': has_summaries,
            'chunks_with_keywords': has_keywords,
            'metadata_completeness': (has_titles + has_summaries + has_keywords) / (len(chunks) * 3)
        }
        
        return quality_metrics
    
    def _identify_content_gaps(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Identify potential gaps in content coverage
        """
        gaps = []
        
        if not chunks:
            return ["No content retrieved"]
        
        # Check for source diversity gaps
        file_types = set(chunk.file_type for chunk in chunks)
        if len(file_types) < 2:
            gaps.append("Limited file type diversity")
        
        # Check for content depth gaps
        avg_content_length = sum(len(chunk.content) for chunk in chunks) / len(chunks)
        if avg_content_length < 200:
            gaps.append("Content chunks may be too brief")
        
        # Check for recency gaps (would need timestamp analysis)
        # This is a placeholder - actual implementation would check dates
        gaps.append("Consider checking for more recent sources")
        
        return gaps
    
    def _analyze_content_redundancy(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze redundancy in content
        """
        if len(chunks) < 2:
            return {'redundancy_score': 0, 'duplicate_sources': []}
        
        # Check for duplicate sources
        source_counts = Counter(chunk.file_name for chunk in chunks)
        duplicate_sources = [source for source, count in source_counts.items() if count > 1]
        
        # Simple content similarity check (placeholder)
        # In practice, this would use more sophisticated similarity measures
        redundancy_score = len(duplicate_sources) / len(chunks)
        
        return {
            'redundancy_score': redundancy_score,
            'duplicate_sources': duplicate_sources,
            'total_unique_sources': len(set(chunk.file_name for chunk in chunks))
        }
    
    def _calculate_relevance_score(self, query: str, results: List[DocumentChunk]) -> float:
        """
        Calculate overall relevance score for search results
        """
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
    
    def _calculate_coverage_score(self, query: str, results: List[DocumentChunk]) -> float:
        """
        Calculate how well results cover the query topic
        """
        if not results:
            return 0.0
        
        # Extract query concepts
        query_concepts = self._extract_key_concepts(query)
        
        if not query_concepts:
            return 0.5
        
        # Check coverage of each concept
        covered_concepts = 0
        for concept in query_concepts:
            if any(concept.lower() in chunk.content.lower() for chunk in results):
                covered_concepts += 1
        
        return covered_concepts / len(query_concepts)
    
    def _calculate_diversity_score(self, results: List[DocumentChunk]) -> float:
        """
        Calculate diversity score for search results
        """
        if not results:
            return 0.0
        
        diversity_factors = []
        
        # Source diversity
        unique_sources = len(set(chunk.file_name for chunk in results))
        source_diversity = unique_sources / len(results)
        diversity_factors.append(source_diversity)
        
        # File type diversity
        unique_types = len(set(chunk.file_type for chunk in results))
        type_diversity = min(unique_types / 3.0, 1.0)  # Normalize to 3 types
        diversity_factors.append(type_diversity)
        
        # Content length diversity
        content_lengths = [len(chunk.content) for chunk in results]
        if len(set(content_lengths)) > 1:
            length_diversity = len(set(content_lengths)) / len(content_lengths)
        else:
            length_diversity = 0.5
        diversity_factors.append(length_diversity)
        
        return statistics.mean(diversity_factors)
    
    def _assess_completeness(self, query: str, results: List[DocumentChunk]) -> str:
        """
        Assess completeness of search results
        """
        if not results:
            return "No results found"
        
        coverage_score = self._calculate_coverage_score(query, results)
        diversity_score = self._calculate_diversity_score(results)
        
        # Combine scores for completeness assessment
        completeness_score = (coverage_score + diversity_score) / 2
        
        if completeness_score > 0.8:
            return "Comprehensive coverage"
        elif completeness_score > 0.6:
            return "Good coverage with minor gaps"
        elif completeness_score > 0.4:
            return "Partial coverage, some important aspects may be missing"
        else:
            return "Limited coverage, significant gaps likely"
    
    def _suggest_search_improvements(self, query: str, results: List[DocumentChunk]) -> List[str]:
        """
        Suggest improvements for search effectiveness
        """
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
            suggestions.append("Search across additional file types for more diverse perspectives")
        
        return suggestions