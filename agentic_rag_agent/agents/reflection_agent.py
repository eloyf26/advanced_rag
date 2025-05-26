"""
Self-Reflection Agent for Agentic RAG
"""

import re
import logging
from typing import List, Dict, Set
from collections import Counter

from config import RAGConfig
from models.response_models import ReflectionResult, DocumentChunk

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """
    Agent responsible for self-reflection and quality assessment
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Quality assessment criteria
        self.quality_indicators = {
            'length': {'min': 50, 'optimal': 200, 'max': 800},
            'structure': ['introduction', 'main_points', 'conclusion'],
            'evidence': ['citations', 'examples', 'data_points'],
            'clarity': ['clear_language', 'logical_flow', 'coherence']
        }
        
        # Common question words for completeness checking
        self.question_words = {
            'what': ['definition', 'description', 'explanation'],
            'how': ['process', 'method', 'procedure', 'steps'],
            'why': ['reason', 'cause', 'explanation', 'justification'],
            'when': ['time', 'date', 'period', 'timeline'],
            'where': ['location', 'place', 'position'],
            'who': ['person', 'people', 'organization', 'entity'],
            'which': ['selection', 'choice', 'option', 'alternative']
        }
    
    async def reflect_on_answer(
        self,
        question: str,
        answer: str,
        sources: List[DocumentChunk]
    ) -> ReflectionResult:
        """
        Perform comprehensive reflection on answer quality
        """
        try:
            # Assess different quality dimensions
            quality_score = self._assess_answer_quality(question, answer, sources)
            completeness_score = self._assess_completeness(question, answer)
            accuracy_assessment = self._assess_accuracy(answer, sources)
            
            # Identify missing information
            missing_info = self._identify_missing_information(question, answer, sources)
            
            # Generate follow-up suggestions
            follow_ups = self._generate_follow_up_suggestions(question, answer, sources)
            
            # Determine if more search is needed
            needs_more_search = self._needs_additional_search(
                quality_score, completeness_score, sources, missing_info
            )
            
            # Additional detailed assessments
            reasoning_quality = self._assess_reasoning_quality(answer)
            source_diversity = self._assess_source_diversity(sources)
            factual_consistency = self._assess_factual_consistency(answer, sources)
            
            return ReflectionResult(
                quality_score=quality_score,
                completeness_score=completeness_score,
                accuracy_assessment=accuracy_assessment,
                missing_information=missing_info,
                suggested_follow_ups=follow_ups,
                needs_more_search=needs_more_search,
                reasoning_quality=reasoning_quality,
                source_diversity=source_diversity,
                factual_consistency=factual_consistency
            )
            
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            
            # Return conservative assessment on error
            return ReflectionResult(
                quality_score=0.5,
                completeness_score=0.5,
                accuracy_assessment="Unable to assess due to error",
                missing_information=["Assessment failed"],
                suggested_follow_ups=[],
                needs_more_search=True,
                reasoning_quality=0.5,
                source_diversity=0.5,
                factual_consistency=0.5
            )
    
    def _assess_answer_quality(
        self, 
        question: str, 
        answer: str, 
        sources: List[DocumentChunk]
    ) -> float:
        """
        Assess overall answer quality across multiple dimensions
        """
        quality_factors = []
        
        # 1. Length appropriateness
        length_score = self._assess_length_quality(answer)
        quality_factors.append(length_score)
        
        # 2. Question-answer relevance
        relevance_score = self._assess_relevance(question, answer)
        quality_factors.append(relevance_score)
        
        # 3. Source utilization
        source_utilization = self._assess_source_utilization(answer, sources)
        quality_factors.append(source_utilization)
        
        # 4. Language quality
        language_quality = self._assess_language_quality(answer)
        quality_factors.append(language_quality)
        
        # 5. Structure and organization
        structure_score = self._assess_structure(answer)
        quality_factors.append(structure_score)
        
        # Calculate weighted average
        weights = [0.15, 0.3, 0.25, 0.15, 0.15]
        return sum(w * s for w, s in zip(weights, quality_factors))
    
    def _assess_length_quality(self, answer: str) -> float:
        """
        Assess if answer length is appropriate
        """
        word_count = len(answer.split())
        
        if word_count < self.quality_indicators['length']['min']:
            return 0.3  # Too short
        elif word_count > self.quality_indicators['length']['max']:
            return 0.7  # Too long
        elif word_count <= self.quality_indicators['length']['optimal']:
            return 1.0  # Optimal length
        else:
            # Gradually decrease score for longer answers
            excess = word_count - self.quality_indicators['length']['optimal']
            max_excess = self.quality_indicators['length']['max'] - self.quality_indicators['length']['optimal']
            return 1.0 - (excess / max_excess) * 0.3
    
    def _assess_relevance(self, question: str, answer: str) -> float:
        """
        Assess how well the answer addresses the question
        """
        question_words = set(self._extract_meaningful_words(question.lower()))
        answer_words = set(self._extract_meaningful_words(answer.lower()))
        
        # Calculate word overlap
        overlap = len(question_words & answer_words)
        relevance = overlap / max(len(question_words), 1)
        
        # Bonus for addressing question type
        question_type = self._identify_question_type(question)
        if self._answer_addresses_question_type(answer, question_type):
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _assess_source_utilization(self, answer: str, sources: List[DocumentChunk]) -> float:
        """
        Assess how well the answer utilizes available sources
        """
        if not sources:
            return 0.3  # Low score if no sources available
        
        utilization_factors = []
        
        # 1. Source diversity
        source_types = set(source.file_type for source in sources)
        diversity_score = min(len(source_types) / 3.0, 1.0)
        utilization_factors.append(diversity_score)
        
        # 2. Content incorporation
        source_content_words = set()
        for source in sources:
            source_content_words.update(self._extract_meaningful_words(source.content.lower()))
        
        answer_words = set(self._extract_meaningful_words(answer.lower()))
        incorporation = len(answer_words & source_content_words) / max(len(source_content_words), 1)
        utilization_factors.append(incorporation)
        
        # 3. Number of sources referenced
        source_count_score = min(len(sources) / 5.0, 1.0)
        utilization_factors.append(source_count_score)
        
        return sum(utilization_factors) / len(utilization_factors)
    
    def _assess_language_quality(self, answer: str) -> float:
        """
        Assess language quality and clarity
        """
        quality_factors = []
        
        # 1. Sentence structure variety
        sentences = answer.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            length_variance = len(set(sentence_lengths)) / len(sentence_lengths)
            quality_factors.append(min(length_variance * 2, 1.0))
        else:
            quality_factors.append(0.5)
        
        # 2. Vocabulary diversity
        words = self._extract_meaningful_words(answer.lower())
        if words:
            vocabulary_diversity = len(set(words)) / len(words)
            quality_factors.append(vocabulary_diversity)
        else:
            quality_factors.append(0.5)
        
        # 3. Absence of repetition
        word_counts = Counter(words)
        max_repetition = max(word_counts.values()) if word_counts else 1
        repetition_score = max(0, 1.0 - (max_repetition - 1) * 0.1)
        quality_factors.append(repetition_score)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _assess_structure(self, answer: str) -> float:
        """
        Assess structural organization of the answer
        """
        structure_indicators = 0
        total_indicators = 4
        
        # 1. Has clear introduction
        first_sentence = answer.split('.')[0] if '.' in answer else answer
        if any(word in first_sentence.lower() for word in ['based on', 'according to', 'to answer']):
            structure_indicators += 1
        
        # 2. Uses transition words
        transition_words = ['however', 'furthermore', 'additionally', 'moreover', 'therefore', 'consequently']
        if any(word in answer.lower() for word in transition_words):
            structure_indicators += 1
        
        # 3. Has enumeration or listing
        if any(pattern in answer for pattern in ['1.', '2.', '•', '-', 'first', 'second', 'finally']):
            structure_indicators += 1
        
        # 4. Has conclusion indicators
        conclusion_words = ['in conclusion', 'to summarize', 'overall', 'in summary']
        if any(word in answer.lower() for word in conclusion_words):
            structure_indicators += 1
        
        return structure_indicators / total_indicators
    
    def _assess_completeness(self, question: str, answer: str) -> float:
        """
        Assess how completely the answer addresses all aspects of the question
        """
        # Identify question components
        question_components = self._extract_question_components(question)
        
        if not question_components:
            return 0.8  # Default score if can't parse components
        
        addressed_components = 0
        for component in question_components:
            if self._component_addressed_in_answer(component, answer):
                addressed_components += 1
        
        base_completeness = addressed_components / len(question_components)
        
        # Adjust based on question complexity
        complexity_adjustment = self._calculate_question_complexity(question)
        
        return min(base_completeness * (1 + complexity_adjustment * 0.1), 1.0)
    
    def _extract_question_components(self, question: str) -> List[str]:
        """
        Extract distinct components/aspects from the question
        """
        components = []
        
        # Split on question words and conjunctions
        split_patterns = [' and ', ' or ', ' also ', ' additionally ', ' furthermore ']
        parts = [question]
        
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(pattern))
            parts = new_parts
        
        # Clean and filter components
        for part in parts:
            clean_part = part.strip().rstrip('?').strip()
            if len(clean_part) > 10:  # Filter out very short components
                components.append(clean_part)
        
        return components if components else [question]
    
    def _component_addressed_in_answer(self, component: str, answer: str) -> bool:
        """
        Check if a question component is addressed in the answer
        """
        component_words = set(self._extract_meaningful_words(component.lower()))
        answer_words = set(self._extract_meaningful_words(answer.lower()))
        
        # Check for word overlap
        overlap_ratio = len(component_words & answer_words) / max(len(component_words), 1)
        
        return overlap_ratio > 0.3
    
    def _assess_accuracy(self, answer: str, sources: List[DocumentChunk]) -> str:
        """
        Assess accuracy level based on source consistency
        """
        if not sources:
            return "Low confidence - no sources available"
        
        # Check source quality indicators
        source_quality_score = self._calculate_source_quality(sources)
        
        # Check factual consistency
        consistency_score = self._assess_factual_consistency(answer, sources)
        
        # Determine accuracy level
        overall_score = (source_quality_score + consistency_score) / 2
        
        if overall_score > 0.8:
            return "High confidence"
        elif overall_score > 0.6:
            return "Medium confidence"
        else:
            return "Low confidence"
    
    def _calculate_source_quality(self, sources: List[DocumentChunk]) -> float:
        """
        Calculate overall quality of sources
        """
        quality_factors = []
        
        # 1. Source diversity
        file_types = set(source.file_type for source in sources)
        diversity_score = min(len(file_types) / 3.0, 1.0)
        quality_factors.append(diversity_score)
        
        # 2. Average similarity scores
        if sources:
            avg_similarity = sum(source.similarity_score for source in sources) / len(sources)
            quality_factors.append(avg_similarity)
        
        # 3. Content length and detail
        avg_content_length = sum(len(source.content) for source in sources) / max(len(sources), 1)
        length_score = min(avg_content_length / 500, 1.0)  # Normalize to 500 chars
        quality_factors.append(length_score)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _assess_factual_consistency(self, answer: str, sources: List[DocumentChunk]) -> float:
        """
        Assess factual consistency between answer and sources
        """
        if not sources:
            return 0.5
        
        # Extract key claims from answer
        answer_claims = self._extract_factual_claims(answer)
        
        if not answer_claims:
            return 0.7  # No specific claims to verify
        
        # Check each claim against sources
        supported_claims = 0
        for claim in answer_claims:
            if self._claim_supported_by_sources(claim, sources):
                supported_claims += 1
        
        return supported_claims / len(answer_claims)
    
    def _extract_factual_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from the answer
        """
        # Simple claim extraction - in production, use more sophisticated NLP
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        claims = []
        for sentence in sentences:
            # Filter for sentences that make factual statements
            if any(indicator in sentence.lower() for indicator in [
                'is', 'are', 'was', 'were', 'has', 'have', 'shows', 'indicates',
                'according to', 'research shows', 'studies indicate'
            ]):
                claims.append(sentence)
        
        return claims[:5]  # Limit to top 5 claims
    
    def _claim_supported_by_sources(self, claim: str, sources: List[DocumentChunk]) -> bool:
        """
        Check if a claim is supported by the sources
        """
        claim_words = set(self._extract_meaningful_words(claim.lower()))
        
        for source in sources:
            source_words = set(self._extract_meaningful_words(source.content.lower()))
            overlap = len(claim_words & source_words) / max(len(claim_words), 1)
            
            if overlap > 0.4:  # 40% word overlap suggests support
                return True
        
        return False
    
    def _identify_missing_information(
        self, 
        question: str, 
        answer: str, 
        sources: List[DocumentChunk]
    ) -> List[str]:
        """
        Identify what information might be missing from the answer
        """
        missing_info = []
        
        # Check for question type completeness
        question_type = self._identify_question_type(question)
        missing_info.extend(self._check_question_type_completeness(question_type, answer))
        
        # Check for source utilization gaps
        if len(sources) < 3:
            missing_info.append("Limited source diversity")
        
        # Check for depth of analysis
        if len(answer.split()) < 100:
            missing_info.append("Answer could be more detailed")
        
        # Check for examples and evidence
        if not self._has_examples_or_evidence(answer):
            missing_info.append("Could include more examples or evidence")
        
        # Check for balanced perspective
        if not self._has_balanced_perspective(answer):
            missing_info.append("Could present multiple perspectives")
        
        return missing_info[:5]  # Limit to top 5 issues
    
    def _check_question_type_completeness(self, question_type: str, answer: str) -> List[str]:
        """
        Check completeness based on question type
        """
        missing = []
        answer_lower = answer.lower()
        
        if question_type == "what":
            if not any(word in answer_lower for word in ['definition', 'refers to', 'means', 'is']):
                missing.append("Missing clear definition or explanation")
        
        elif question_type == "how":
            if not any(word in answer_lower for word in ['step', 'process', 'method', 'procedure']):
                missing.append("Missing process or methodology explanation")
        
        elif question_type == "why":
            if not any(word in answer_lower for word in ['because', 'due to', 'reason', 'cause']):
                missing.append("Missing causal explanation or reasoning")
        
        elif question_type == "compare":
            if not any(word in answer_lower for word in ['difference', 'similarity', 'contrast', 'both']):
                missing.append("Missing comparative analysis")
        
        return missing
    
    def _has_examples_or_evidence(self, answer: str) -> bool:
        """
        Check if answer includes examples or evidence
        """
        evidence_indicators = [
            'for example', 'such as', 'including', 'study shows', 'research indicates',
            'data suggests', 'according to', 'evidence shows', 'instance'
        ]
        
        return any(indicator in answer.lower() for indicator in evidence_indicators)
    
    def _has_balanced_perspective(self, answer: str) -> bool:
        """
        Check if answer presents balanced perspective
        """
        balance_indicators = [
            'however', 'on the other hand', 'alternatively', 'but', 'although',
            'while', 'despite', 'nevertheless', 'different perspective'
        ]
        
        return any(indicator in answer.lower() for indicator in balance_indicators)
    
    def _generate_follow_up_suggestions(
        self, 
        question: str, 
        answer: str, 
        sources: List[DocumentChunk]
    ) -> List[str]:
        """
        Generate intelligent follow-up questions
        """
        suggestions = []
        
        # Based on question type
        question_type = self._identify_question_type(question)
        suggestions.extend(self._generate_type_based_follow_ups(question, question_type))
        
        # Based on sources
        if sources:
            unique_topics = set()
            for source in sources[:3]:
                unique_topics.update(source.keywords[:2])
            
            suggestions.extend([
                f"Can you tell me more about {topic}?" 
                for topic in list(unique_topics)[:2]
            ])
        
        # Based on missing information
        if len(answer.split()) < 150:
            suggestions.append("Can you provide more detailed information?")
        
        if not self._has_examples_or_evidence(answer):
            suggestions.append("Can you provide specific examples?")
        
        # Generic helpful follow-ups
        suggestions.extend([
            "What are the practical implications of this?",
            "Are there any related topics I should explore?"
        ])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _generate_type_based_follow_ups(self, question: str, question_type: str) -> List[str]:
        """
        Generate follow-ups based on question type
        """
        follow_ups = []
        
        if question_type == "what":
            follow_ups.extend([
                "How does this work in practice?",
                "What are some examples of this?"
            ])
        elif question_type == "how":
            follow_ups.extend([
                "What are the best practices for this?",
                "What challenges might I encounter?"
            ])
        elif question_type == "why":
            follow_ups.extend([
                "What are the implications of this?",
                "How does this affect related areas?"
            ])
        elif question_type == "compare":
            follow_ups.extend([
                "Which option would you recommend?",
                "What are the trade-offs involved?"
            ])
        
        return follow_ups
    
    def _needs_additional_search(
        self,
        quality_score: float,
        completeness_score: float,
        sources: List[DocumentChunk],
        missing_info: List[str]
    ) -> bool:
        """
        Determine if additional search is recommended
        """
        # Need more search if quality is low
        if quality_score < self.config.min_confidence_threshold:
            return True
        
        # Need more search if completeness is low
        if completeness_score < 0.6:
            return True
        
        # Need more search if we have very few sources
        if len(sources) < 2:
            return True
        
        # Need more search if there are significant missing information gaps
        critical_gaps = [gap for gap in missing_info if 'missing' in gap.lower()]
        if len(critical_gaps) > 2:
            return True
        
        return False
    
    def _assess_reasoning_quality(self, answer: str) -> float:
        """
        Assess the quality of reasoning in the answer
        """
        reasoning_indicators = []
        
        # Check for logical connectors
        logical_connectors = ['because', 'therefore', 'thus', 'consequently', 'as a result']
        connector_score = min(
            sum(1 for connector in logical_connectors if connector in answer.lower()) / 3.0,
            1.0
        )
        reasoning_indicators.append(connector_score)
        
        # Check for evidence-based reasoning
        evidence_phrases = ['according to', 'research shows', 'studies indicate', 'data suggests']
        evidence_score = min(
            sum(1 for phrase in evidence_phrases if phrase in answer.lower()) / 2.0,
            1.0
        )
        reasoning_indicators.append(evidence_score)
        
        # Check for structured argumentation
        structure_words = ['first', 'second', 'finally', 'in addition', 'furthermore']
        structure_score = min(
            sum(1 for word in structure_words if word in answer.lower()) / 3.0,
            1.0
        )
        reasoning_indicators.append(structure_score)
        
        return sum(reasoning_indicators) / len(reasoning_indicators)
    
    def _assess_source_diversity(self, sources: List[DocumentChunk]) -> float:
        """
        Assess diversity of sources used
        """
        if not sources:
            return 0.0
        
        diversity_factors = []
        
        # File type diversity
        file_types = set(source.file_type for source in sources)
        type_diversity = min(len(file_types) / 4.0, 1.0)  # Normalize to 4 types
        diversity_factors.append(type_diversity)
        
        # Source file diversity
        source_files = set(source.file_name for source in sources)
        file_diversity = min(len(source_files) / 5.0, 1.0)  # Normalize to 5 files
        diversity_factors.append(file_diversity)
        
        # Content length diversity
        content_lengths = [len(source.content) for source in sources]
        if len(set(content_lengths)) > 1:
            length_diversity = len(set(content_lengths)) / len(content_lengths)
        else:
            length_diversity = 0.5
        diversity_factors.append(length_diversity)
        
        return sum(diversity_factors) / len(diversity_factors)
    
    def _identify_question_type(self, question: str) -> str:
        """
        Identify the type of question being asked
        """
        question_lower = question.lower()
        
        if question_lower.startswith(('what is', 'what are', 'define')):
            return "what"
        elif question_lower.startswith(('how to', 'how do', 'how can')):
            return "how"
        elif question_lower.startswith(('why', 'why do', 'why is')):
            return "why"
        elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
            return "compare"
        elif question_lower.startswith(('when', 'when did', 'when do')):
            return "when"
        elif question_lower.startswith(('where', 'where is', 'where can')):
            return "where"
        elif question_lower.startswith(('who', 'who is', 'who are')):
            return "who"
        else:
            return "general"
    
    def _answer_addresses_question_type(self, answer: str, question_type: str) -> bool:
        """
        Check if answer appropriately addresses the question type
        """
        answer_lower = answer.lower()
        
        type_indicators = {
            "what": ["is", "refers to", "means", "definition"],
            "how": ["step", "process", "method", "by"],
            "why": ["because", "due to", "reason", "cause"],
            "when": ["date", "time", "year", "period"],
            "where": ["location", "place", "at", "in"],
            "who": ["person", "people", "organization", "individual"],
            "compare": ["difference", "similarity", "both", "versus"]
        }
        
        indicators = type_indicators.get(question_type, [])
        return any(indicator in answer_lower for indicator in indicators)
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """
        Extract meaningful words (filter out stop words)
        """
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
        }
        
        words = re.findall(r'\w+', text.lower())
        return [word for word in words if len(word) > 2 and word not in stop_words]
    
    def _calculate_question_complexity(self, question: str) -> float:
        """
        Calculate question complexity for completeness adjustment
        """
        complexity_factors = 0
        
        # Multiple question marks
        if question.count('?') > 1:
            complexity_factors += 1
        
        # Conjunctions indicating multiple parts
        conjunctions = ['and', 'or', 'also', 'additionally']
        complexity_factors += sum(1 for conj in conjunctions if conj in question.lower())
        
        # Length factor
        if len(question.split()) > 15:
            complexity_factors += 1
        
        return min(complexity_factors / 3.0, 1.0)