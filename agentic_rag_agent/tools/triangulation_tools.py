"""
Source Triangulation Tools for Agentic RAG Agent
"""

import asyncio
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
import statistics

from models.request_models import SearchFilters
from models.response_models import DocumentChunk
from services.database_manager import RAGDatabaseManager

logger = logging.getLogger(__name__)


class TriangulationToolkit:
    """
    Tools for source triangulation and verification in agentic RAG
    """
    
    def __init__(self, db_manager: RAGDatabaseManager):
        self.db_manager = db_manager
        
        # Verification strategies
        self.verification_strategies = {
            'cross_reference': self._cross_reference_facts,
            'source_diversity': self._verify_source_diversity,
            'consensus_analysis': self._analyze_consensus,
            'contradiction_detection': self._detect_contradictions,
            'authority_verification': self._verify_authority
        }
        
        # Reliability indicators
        self.reliability_indicators = {
            'academic': ['peer-reviewed', 'journal', 'research', 'study', 'university'],
            'official': ['government', 'official', 'policy', 'regulation', 'statute'],
            'industry': ['industry report', 'white paper', 'technical specification'],
            'news': ['news', 'article', 'report', 'press release'],
            'blog': ['blog', 'opinion', 'personal', 'thoughts']
        }
    
    async def triangulate_sources(
        self, 
        query: str, 
        primary_sources: List[DocumentChunk], 
        max_additional: int = 5
    ) -> List[DocumentChunk]:
        """
        Find additional sources to triangulate and verify information
        """
        try:
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
            
            # Score and rank additional sources by triangulation value
            scored_sources = self._score_triangulation_value(
                additional_sources, primary_sources, key_concepts
            )
            
            return scored_sources
            
        except Exception as e:
            logger.error(f"Error in source triangulation: {e}")
            return []
    
    async def verify_information_consistency(
        self, 
        sources: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Verify consistency of information across sources
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error in information verification: {e}")
            return {'consistency_score': 0.0, 'error': str(e)}
    
    async def find_alternative_perspectives(
        self, 
        query: str, 
        existing_sources: List[DocumentChunk],
        max_alternatives: int = 3
    ) -> List[DocumentChunk]:
        """
        Find sources that provide alternative perspectives on the topic
        """
        try:
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
                filters = SearchFilters(
                    max_results=5,
                    similarity_threshold=0.6
                )
                
                chunks = await self.db_manager.hybrid_search(alt_query, filters)
                
                # Filter out existing sources and add new ones
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
            
        except Exception as e:
            logger.error(f"Error finding alternative perspectives: {e}")
            return []
    
    def analyze_source_credibility(self, sources: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze credibility and authority of sources
        """
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
    
    def _extract_key_concepts_from_sources(self, sources: List[DocumentChunk]) -> List[str]:
        """
        Extract key concepts that should be verified across sources
        """
        all_keywords = []
        concept_frequency = Counter()
        
        # Collect keywords from all sources
        for source in sources:
            if source.keywords:
                all_keywords.extend(source.keywords)
        
        # Also extract concepts from content (simplified NER)
        for source in sources:
            content_concepts = self._extract_concepts_from_text(source.content)
            all_keywords.extend(content_concepts)
        
        # Count frequency and return most common
        concept_frequency.update(all_keywords)
        return [concept for concept, _ in concept_frequency.most_common(10)]
    
    def _extract_potential_claims(self, sources: List[DocumentChunk]) -> List[str]:
        """
        Extract factual claims that should be verified
        """
        claims = []
        
        for source in sources:
            # Look for sentences that make factual statements
            sentences = [s.strip() for s in source.content.split('.') if s.strip()]
            
            for sentence in sentences:
                # Simple heuristics for factual claims
                if (any(indicator in sentence.lower() for indicator in [
                    'is', 'are', 'was', 'were', 'shows', 'indicates', 'proves',
                    'research shows', 'studies indicate', 'according to'
                ]) and len(sentence.split()) > 5):
                    claims.append(sentence[:200])  # Limit length
        
        return claims[:5]  # Return top 5 claims
    
    def _generate_verification_queries(
        self, 
        original_query: str, 
        key_concepts: List[str], 
        claims: List[str]
    ) -> List[str]:
        """
        Generate queries to find verification sources
        """
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
        main_topic = original_query.split()[:3]  # First 3 words of original query
        topic_str = ' '.join(main_topic)
        verification_queries.extend([
            f"independent research {topic_str}",
            f"peer review {topic_str}",
            f"fact check {topic_str}"
        ])
        
        return verification_queries[:8]  # Limit to 8 queries
    
    async def _search_for_triangulating_sources(
        self, 
        verification_queries: List[str], 
        primary_sources: List[DocumentChunk], 
        max_additional: int
    ) -> List[DocumentChunk]:
        """
        Search for additional sources for triangulation
        """
        additional_sources = []
        existing_source_ids = {source.id for source in primary_sources}
        
        # Execute verification searches
        for query in verification_queries:
            if len(additional_sources) >= max_additional:
                break
            
            try:
                filters = SearchFilters(
                    max_results=3,
                    similarity_threshold=0.6
                )
                
                chunks = await self.db_manager.hybrid_search(query, filters)
                
                for chunk in chunks:
                    if (chunk.id not in existing_source_ids and 
                        len(additional_sources) < max_additional):
                        additional_sources.append(chunk)
                        existing_source_ids.add(chunk.id)
                        
            except Exception as e:
                logger.error(f"Error in verification search for '{query}': {e}")
                continue
        
        return additional_sources
    
    def _score_triangulation_value(
        self, 
        additional_sources: List[DocumentChunk], 
        primary_sources: List[DocumentChunk], 
        key_concepts: List[str]
    ) -> List[DocumentChunk]:
        """
        Score additional sources by their triangulation value
        """
        scored_sources = []
        
        for source in additional_sources:
            triangulation_score = 0.0
            
            # Factor 1: Concept coverage
            source_concepts = set(source.keywords) if source.keywords else set()
            concept_overlap = len(source_concepts & set(key_concepts)) / max(len(key_concepts), 1)
            triangulation_score += concept_overlap * 0.3
            
            # Factor 2: Source diversity (different file type/source)
            primary_types = set(ps.file_type for ps in primary_sources)
            if source.file_type not in primary_types:
                triangulation_score += 0.2
            
            primary_files = set(ps.file_name for ps in primary_sources)
            if source.file_name not in primary_files:
                triangulation_score += 0.2
            
            # Factor 3: Content quality indicators
            if source.title:
                triangulation_score += 0.1
            if source.summary:
                triangulation_score += 0.1
            if len(source.content) > 200:
                triangulation_score += 0.1
            
            source.triangulation_score = triangulation_score
            scored_sources.append(source)
        
        # Sort by triangulation score
        scored_sources.sort(key=lambda x: x.triangulation_score, reverse=True)
        return scored_sources
    
    async def _analyze_cross_references(self, sources: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Analyze cross-references between sources
        """
        cross_references = []
        
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                # Find common concepts
                concepts1 = set(source1.keywords) if source1.keywords else set()
                concepts2 = set(source2.keywords) if source2.keywords else set()
                common_concepts = concepts1 & concepts2
                
                if common_concepts:
                    cross_ref = {
                        'source1_id': source1.id,
                        'source2_id': source2.id,
                        'source1_file': source1.file_name,
                        'source2_file': source2.file_name,
                        'common_concepts': list(common_concepts),
                        'overlap_score': len(common_concepts) / max(len(concepts1 | concepts2), 1)
                    }
                    cross_references.append(cross_ref)
        
        return cross_references
    
    def _detect_information_contradictions(self, sources: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Detect potential contradictions between sources
        """
        contradictions = []
        
        # Simple contradiction detection based on opposing keywords
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
                            'confidence': 0.6  # Simple heuristic confidence
                        })
                    elif neg_term in content1_lower and pos_term in content2_lower:
                        contradictions.append({
                            'source1_id': source1.id,
                            'source2_id': source2.id,
                            'source1_file': source1.file_name,
                            'source2_file': source2.file_name,
                            'contradiction_type': f"{neg_term} vs {pos_term}",
                            'confidence': 0.6
                        })
        
        return contradictions
    
    def _find_consensus_points(self, sources: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Find points of consensus across sources
        """
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
    
    def _assess_source_reliability(self, sources: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Assess overall reliability of the source collection
        """
        reliability_scores = []
        source_types = Counter()
        
        for source in sources:
            # Calculate individual reliability
            reliability = self._calculate_source_credibility(source)
            reliability_scores.append(reliability)
            
            # Categorize source type
            source_type = self._identify_source_type(source)
            source_types[source_type] += 1
        
        return {
            'average_reliability': statistics.mean(reliability_scores) if reliability_scores else 0,
            'min_reliability': min(reliability_scores) if reliability_scores else 0,
            'max_reliability': max(reliability_scores) if reliability_scores else 0,
            'source_type_distribution': dict(source_types),
            'reliability_variance': statistics.variance(reliability_scores) if len(reliability_scores) > 1 else 0
        }
    
    def _calculate_consistency_score(
        self, 
        cross_refs: List[Dict], 
        contradictions: List[Dict], 
        consensus: List[Dict], 
        reliability: Dict
    ) -> float:
        """
        Calculate overall consistency score
        """
        consistency_factors = []
        
        # Factor 1: Cross-reference density
        total_possible_refs = len(cross_refs) if cross_refs else 0
        cross_ref_factor = min(total_possible_refs / 5.0, 1.0)  # Normalize to 5 references
        consistency_factors.append(cross_ref_factor)
        
        # Factor 2: Contradiction penalty
        contradiction_penalty = len(contradictions) * 0.2
        contradiction_factor = max(0, 1.0 - contradiction_penalty)
        consistency_factors.append(contradiction_factor)
        
        # Factor 3: Consensus strength
        if consensus:
            avg_consensus = statistics.mean([c['consensus_strength'] for c in consensus])
            consistency_factors.append(avg_consensus)
        else:
            consistency_factors.append(0.5)
        
        # Factor 4: Source reliability
        reliability_factor = reliability.get('average_reliability', 0.5)
        consistency_factors.append(reliability_factor)
        
        return statistics.mean(consistency_factors)
    
    def _identify_dominant_perspective(self, sources: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Identify the dominant perspective in existing sources
        """
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
    
    def _generate_alternative_queries(self, original_query: str, dominant_perspective: Dict) -> List[str]:
        """
        Generate queries to find alternative perspectives
        """
        alternative_queries = []
        main_topic = ' '.join(original_query.split()[:4])  # First 4 words
        
        # Perspective-based alternatives
        perspective_modifiers = {
            'positive': ['criticism of', 'problems with', 'limitations of'],
            'negative': ['benefits of', 'advantages of', 'success of'],
            'supportive': ['criticism of', 'opposition to', 'debate about'],
            'critical': ['defense of', 'support for', 'benefits of'],
            'neutral': ['controversy about', 'debate over', 'different views on']
        }
        
        dominant = dominant_perspective.get('dominant_perspective', 'neutral')
        modifiers = perspective_modifiers.get(dominant, ['alternative view of'])
        
        for modifier in modifiers:
            alternative_queries.append(f"{modifier} {main_topic}")
        
        # General alternative perspective queries
        alternative_queries.extend([
            f"opposing view {main_topic}",
            f"different perspective {main_topic}",
            f"counter argument {main_topic}",
            f"alternative approach {main_topic}"
        ])
        
        return alternative_queries[:5]
    
    def _score_perspective_difference(self, candidate: DocumentChunk, existing_sources: List[DocumentChunk]) -> float:
        """
        Score how different a candidate's perspective is from existing sources
        """
        if not existing_sources:
            return 1.0
        
        # Compare keywords and content tone
        candidate_keywords = set(candidate.keywords) if candidate.keywords else set()
        
        difference_scores = []
        for existing in existing_sources:
            existing_keywords = set(existing.keywords) if existing.keywords else set()
            
            # Calculate keyword difference
            if candidate_keywords or existing_keywords:
                overlap = len(candidate_keywords & existing_keywords)
                total = len(candidate_keywords | existing_keywords)
                keyword_similarity = overlap / max(total, 1)
                keyword_difference = 1 - keyword_similarity
            else:
                keyword_difference = 0.5
            
            difference_scores.append(keyword_difference)
        
        return statistics.mean(difference_scores)
    
    def _calculate_source_credibility(self, source: DocumentChunk) -> float:
        """
        Calculate credibility score for a single source
        """
        credibility_factors = []
        
        # Factor 1: Source type reliability
        source_type = self._identify_source_type(source)
        type_scores = {
            'academic': 0.9,
            'official': 0.85,
            'industry': 0.7,
            'news': 0.6,
            'blog': 0.4,
            'unknown': 0.5
        }
        credibility_factors.append(type_scores.get(source_type, 0.5))
        
        # Factor 2: Content quality indicators
        quality_score = 0.5
        if source.title:
            quality_score += 0.1
        if source.summary:
            quality_score += 0.1
        if len(source.content) > 300:
            quality_score += 0.1
        if source.keywords and len(source.keywords) > 3:
            quality_score += 0.1
        
        credibility_factors.append(min(quality_score, 1.0))
        
        # Factor 3: Authority indicators
        authority_score = len(self._find_authority_indicators(source)) * 0.2
        credibility_factors.append(min(authority_score, 1.0))
        
        return statistics.mean(credibility_factors)
    
    def _identify_source_type(self, source: DocumentChunk) -> str:
        """
        Identify the type/category of a source
        """
        content_lower = source.content.lower()
        file_name_lower = source.file_name.lower()
        
        # Check for academic indicators
        if any(indicator in content_lower for indicator in self.reliability_indicators['academic']):
            return 'academic'
        
        # Check for official/government indicators
        if any(indicator in content_lower for indicator in self.reliability_indicators['official']):
            return 'official'
        
        # Check for industry indicators
        if any(indicator in content_lower for indicator in self.reliability_indicators['industry']):
            return 'industry'
        
        # Check for news indicators
        if any(indicator in content_lower for indicator in self.reliability_indicators['news']):
            return 'news'
        
        # Check for blog/opinion indicators
        if any(indicator in content_lower for indicator in self.reliability_indicators['blog']):
            return 'blog'
        
        # File type based classification
        if source.file_type in ['pdf', 'docx']:
            return 'academic'  # Assume formal documents are academic
        elif source.file_type in ['html', 'txt']:
            return 'news'  # Assume web content is news/blog
        
        return 'unknown'
    
    def _find_authority_indicators(self, source: DocumentChunk) -> List[str]:
        """
        Find indicators of source authority and expertise
        """
        authority_indicators = []
        content_lower = source.content.lower()
        
        # Check for various authority signals
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
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """
        Extract concepts from text using simple NLP techniques
        """
        import re
        
        # Simple concept extraction - in production, use more sophisticated NLP
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words (potential proper nouns)
        
        # Filter for meaningful concepts
        stop_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For'}
        concepts = [word.lower() for word in words if word not in stop_words and len(word) > 3]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen:
                unique_concepts.append(concept)
                seen.add(concept)
        
        return unique_concepts[:10]  # Return top 10 concepts
    
    def _generate_credibility_recommendations(
        self, 
        individual_scores: List[float], 
        source_type_counts: Counter
    ) -> List[str]:
        """
        Generate recommendations for improving source credibility
        """
        recommendations = []
        
        avg_credibility = statistics.mean(individual_scores) if individual_scores else 0
        
        if avg_credibility < 0.6:
            recommendations.append("Consider seeking more authoritative sources")
        
        if 'academic' not in source_type_counts:
            recommendations.append("Include academic or peer-reviewed sources for stronger evidence")
        
        if 'official' not in source_type_counts:
            recommendations.append("Consider including official or government sources")
        
        if len(source_type_counts) < 2:
            recommendations.append("Diversify source types for more comprehensive perspective")
        
        if source_type_counts.get('blog', 0) > len(individual_scores) * 0.5:
            recommendations.append("Balance opinion sources with more authoritative references")
        
        return recommendations