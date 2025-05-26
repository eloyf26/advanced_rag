"""
Query Planning Agent for Agentic RAG
"""

import re
import logging
from typing import List, Dict, Any
from datetime import datetime

from config import RAGConfig
from models.response_models import QueryPlan

logger = logging.getLogger(__name__)


class QueryPlanningAgent:
    """
    Specialized agent for query analysis and execution planning
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Query classification patterns
        self.analytical_patterns = [
            r'\b(compare|contrast|analyze|evaluate|assess|examine)\b',
            r'\b(explain why|how does|what are the implications)\b',
            r'\b(advantages|disadvantages|pros|cons)\b',
            r'\b(differences|similarities|relationship)\b'
        ]
        
        self.factual_patterns = [
            r'\b(what is|when did|who is|where is|define)\b',
            r'\b(how many|how much|which)\b',
            r'\b(list|name|identify)\b'
        ]
        
        self.procedural_patterns = [
            r'\b(how to|steps|process|procedure)\b',
            r'\b(guide|tutorial|instructions)\b',
            r'\b(implement|create|build)\b'
        ]
        
        # Domain indicators
        self.domain_indicators = {
            'technical': ['algorithm', 'implementation', 'code', 'programming', 'software'],
            'business': ['strategy', 'market', 'revenue', 'profit', 'customer'],
            'scientific': ['research', 'study', 'experiment', 'hypothesis', 'data'],
            'legal': ['law', 'regulation', 'compliance', 'policy', 'contract'],
            'medical': ['treatment', 'diagnosis', 'patient', 'medical', 'health']
        }
    
    async def create_query_plan(self, question: str) -> QueryPlan:
        """
        Create a comprehensive execution plan for the query
        """
        try:
            # Step 1: Classify query type
            query_type = self._classify_query_type(question)
            
            # Step 2: Analyze complexity
            complexity_score = self._calculate_complexity(question)
            
            # Step 3: Generate sub-queries
            sub_queries = self._generate_sub_queries(question, query_type)
            
            # Step 4: Determine search strategy
            search_strategy = self._determine_search_strategy(question, query_type)
            
            # Step 5: Predict relevant sources
            expected_sources = self._predict_source_types(question)
            
            # Step 6: Create reasoning
            reasoning = self._create_planning_reasoning(
                question, query_type, complexity_score, len(sub_queries)
            )
            
            # Step 7: Estimate processing parameters
            estimated_time = self._estimate_processing_time(complexity_score, len(sub_queries))
            recommended_iterations = self._recommend_iterations(complexity_score, query_type)
            
            return QueryPlan(
                original_query=question,
                sub_queries=sub_queries,
                search_strategy=search_strategy,
                reasoning=reasoning,
                expected_sources=expected_sources,
                complexity_score=complexity_score,
                estimated_time=estimated_time,
                recommended_iterations=recommended_iterations
            )
            
        except Exception as e:
            logger.error(f"Error in query planning: {e}")
            
            # Return fallback plan
            return QueryPlan(
                original_query=question,
                sub_queries=[question],
                search_strategy="hybrid",
                reasoning="Fallback plan due to planning error",
                expected_sources=[],
                complexity_score=0.5,
                estimated_time=10.0,
                recommended_iterations=2
            )
    
    def _classify_query_type(self, question: str) -> str:
        """
        Classify the query into different types
        """
        question_lower = question.lower()
        
        # Check for analytical queries
        for pattern in self.analytical_patterns:
            if re.search(pattern, question_lower):
                return "analytical"
        
        # Check for factual queries
        for pattern in self.factual_patterns:
            if re.search(pattern, question_lower):
                return "factual"
        
        # Check for procedural queries
        for pattern in self.procedural_patterns:
            if re.search(pattern, question_lower):
                return "procedural"
        
        # Default classification
        return "general"
    
    def _calculate_complexity(self, question: str) -> float:
        """
        Calculate query complexity score (0-1)
        """
        complexity_factors = []
        
        # Length factor
        word_count = len(question.split())
        length_factor = min(word_count / 20.0, 1.0)
        complexity_factors.append(length_factor)
        
        # Multi-part question factor
        question_marks = question.count('?')
        and_or_count = len(re.findall(r'\b(and|or|also|furthermore|moreover)\b', question.lower()))
        multi_part_factor = min((question_marks + and_or_count) / 3.0, 1.0)
        complexity_factors.append(multi_part_factor)
        
        # Technical complexity factor
        technical_terms = [
            'algorithm', 'implementation', 'architecture', 'framework',
            'methodology', 'systematic', 'comprehensive', 'detailed'
        ]
        technical_count = sum(1 for term in technical_terms if term in question.lower())
        technical_factor = min(technical_count / 3.0, 1.0)
        complexity_factors.append(technical_factor)
        
        # Analytical complexity factor
        analytical_terms = [
            'compare', 'contrast', 'analyze', 'evaluate', 'assess',
            'implications', 'consequences', 'advantages', 'disadvantages'
        ]
        analytical_count = sum(1 for term in analytical_terms if term in question.lower())
        analytical_factor = min(analytical_count / 2.0, 1.0)
        complexity_factors.append(analytical_factor)
        
        # Domain specificity factor
        domain_specificity = self._calculate_domain_specificity(question)
        complexity_factors.append(domain_specificity)
        
        # Calculate weighted average
        weights = [0.2, 0.2, 0.2, 0.3, 0.1]
        complexity_score = sum(w * f for w, f in zip(weights, complexity_factors))
        
        return min(complexity_score, 1.0)
    
    def _calculate_domain_specificity(self, question: str) -> float:
        """
        Calculate how domain-specific the question is
        """
        question_lower = question.lower()
        domain_matches = 0
        total_terms = 0
        
        for domain, terms in self.domain_indicators.items():
            for term in terms:
                total_terms += 1
                if term in question_lower:
                    domain_matches += 1
        
        return min(domain_matches / max(total_terms, 1) * 5, 1.0)
    
    def _generate_sub_queries(self, question: str, query_type: str) -> List[str]:
        """
        Generate sub-queries based on the main question and its type
        """
        sub_queries = [question]  # Always include the original
        
        if query_type == "analytical":
            sub_queries.extend(self._generate_analytical_sub_queries(question))
        elif query_type == "factual":
            sub_queries.extend(self._generate_factual_sub_queries(question))
        elif query_type == "procedural":
            sub_queries.extend(self._generate_procedural_sub_queries(question))
        else:
            sub_queries.extend(self._generate_general_sub_queries(question))
        
        # Remove duplicates and limit
        unique_sub_queries = []
        seen = set()
        for sq in sub_queries:
            if sq.lower() not in seen:
                unique_sub_queries.append(sq)
                seen.add(sq.lower())
        
        return unique_sub_queries[:5]  # Limit to 5 sub-queries
    
    def _generate_analytical_sub_queries(self, question: str) -> List[str]:
        """
        Generate sub-queries for analytical questions
        """
        sub_queries = []
        
        # Extract main entities/concepts
        entities = self._extract_entities(question)
        
        if 'compare' in question.lower() or 'contrast' in question.lower():
            for entity in entities[:2]:
                sub_queries.append(f"What are the key characteristics of {entity}?")
                sub_queries.append(f"What are the advantages and disadvantages of {entity}?")
        
        if 'analyze' in question.lower() or 'evaluate' in question.lower():
            for entity in entities[:2]:
                sub_queries.append(f"What factors influence {entity}?")
                sub_queries.append(f"What are the implications of {entity}?")
        
        return sub_queries
    
    def _generate_factual_sub_queries(self, question: str) -> List[str]:
        """
        Generate sub-queries for factual questions
        """
        sub_queries = []
        entities = self._extract_entities(question)
        
        for entity in entities[:2]:
            sub_queries.append(f"Definition of {entity}")
            sub_queries.append(f"Background information about {entity}")
        
        return sub_queries
    
    def _generate_procedural_sub_queries(self, question: str) -> List[str]:
        """
        Generate sub-queries for procedural questions
        """
        sub_queries = []
        entities = self._extract_entities(question)
        
        for entity in entities[:2]:
            sub_queries.append(f"Prerequisites for {entity}")
            sub_queries.append(f"Step-by-step guide for {entity}")
            sub_queries.append(f"Best practices for {entity}")
        
        return sub_queries
    
    def _generate_general_sub_queries(self, question: str) -> List[str]:
        """
        Generate sub-queries for general questions
        """
        entities = self._extract_entities(question)
        sub_queries = []
        
        for entity in entities[:2]:
            sub_queries.append(f"Overview of {entity}")
            sub_queries.append(f"Examples of {entity}")
        
        return sub_queries
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        Extract key entities/concepts from the question
        """
        # Simple entity extraction - in production, use NER
        words = question.split()
        
        # Filter out common words
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'can', 'could',
            'should', 'would', 'might', 'may', 'must', 'will', 'shall'
        }
        
        entities = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if (len(clean_word) > 3 and 
                clean_word not in stop_words and
                not clean_word.isdigit()):
                entities.append(clean_word)
        
        return entities[:5]  # Return top 5 entities
    
    def _determine_search_strategy(self, question: str, query_type: str) -> str:
        """
        Determine the optimal search strategy
        """
        question_lower = question.lower()
        
        # Exact match indicators suggest keyword search
        exact_indicators = ['exactly', 'precisely', 'specific', 'exact']
        if any(indicator in question_lower for indicator in exact_indicators):
            return "keyword"
        
        # Conceptual indicators suggest semantic search
        conceptual_indicators = ['similar', 'like', 'related', 'concept', 'idea']
        if any(indicator in question_lower for indicator in conceptual_indicators):
            return "semantic"
        
        # For most analytical and complex queries, use hybrid
        if query_type in ["analytical", "procedural"]:
            return "hybrid"
        
        # For simple factual queries, semantic might be sufficient
        if query_type == "factual":
            return "semantic"
        
        # Default to hybrid for comprehensive coverage
        return "hybrid"
    
    def _predict_source_types(self, question: str) -> List[str]:
        """
        Predict relevant source file types based on question content
        """
        question_lower = question.lower()
        predicted_sources = []
        
        # Technical/code-related
        if any(term in question_lower for term in [
            'code', 'programming', 'algorithm', 'implementation', 'function',
            'script', 'development', 'software'
        ]):
            predicted_sources.extend(['py', 'js', 'java', 'cpp', 'sql'])
        
        # Research/academic
        if any(term in question_lower for term in [
            'research', 'study', 'analysis', 'academic', 'paper',
            'findings', 'methodology', 'literature'
        ]):
            predicted_sources.extend(['pdf', 'docx'])
        
        # Data/statistics
        if any(term in question_lower for term in [
            'data', 'statistics', 'numbers', 'metrics', 'analysis',
            'chart', 'graph', 'table', 'dataset'
        ]):
            predicted_sources.extend(['csv', 'xlsx'])
        
        # Documentation
        if any(term in question_lower for term in [
            'documentation', 'guide', 'manual', 'instructions',
            'tutorial', 'how-to', 'readme'
        ]):
            predicted_sources.extend(['md', 'txt', 'html'])
        
        # Business/reports
        if any(term in question_lower for term in [
            'report', 'business', 'strategy', 'plan', 'proposal',
            'presentation', 'meeting', 'summary'
        ]):
            predicted_sources.extend(['docx', 'pdf', 'pptx'])
        
        # Remove duplicates and return
        return list(set(predicted_sources))
    
    def _create_planning_reasoning(
        self, 
        question: str, 
        query_type: str, 
        complexity_score: float, 
        sub_query_count: int
    ) -> str:
        """
        Create reasoning explanation for the planning decisions
        """
        reasoning_parts = []
        
        # Query classification reasoning
        reasoning_parts.append(f"Query classified as '{query_type}' type")
        
        # Complexity reasoning
        if complexity_score > 0.7:
            reasoning_parts.append("High complexity detected - will use iterative search with multiple rounds")
        elif complexity_score > 0.4:
            reasoning_parts.append("Medium complexity - will use targeted search with sub-queries")
        else:
            reasoning_parts.append("Low complexity - single focused search should suffice")
        
        # Sub-query reasoning
        if sub_query_count > 3:
            reasoning_parts.append(f"Generated {sub_query_count} sub-queries for comprehensive coverage")
        elif sub_query_count > 1:
            reasoning_parts.append(f"Created {sub_query_count} focused sub-queries")
        else:
            reasoning_parts.append("Original query is sufficiently focused")
        
        # Strategy reasoning
        reasoning_parts.append("Selected hybrid search for optimal recall and precision")
        
        return ". ".join(reasoning_parts) + "."
    
    def _estimate_processing_time(self, complexity_score: float, sub_query_count: int) -> float:
        """
        Estimate processing time in seconds
        """
        base_time = 5.0  # Base processing time
        complexity_multiplier = 1 + complexity_score
        sub_query_multiplier = 1 + (sub_query_count - 1) * 0.3
        
        estimated_time = base_time * complexity_multiplier * sub_query_multiplier
        
        return min(estimated_time, 60.0)  # Cap at 60 seconds
    
    def _recommend_iterations(self, complexity_score: float, query_type: str) -> int:
        """
        Recommend number of search iterations
        """
        base_iterations = 1
        
        if complexity_score > 0.7:
            base_iterations = 3
        elif complexity_score > 0.4:
            base_iterations = 2
        
        # Analytical queries might need more iterations
        if query_type == "analytical":
            base_iterations += 1
        
        return min(base_iterations, self.config.max_iterations)