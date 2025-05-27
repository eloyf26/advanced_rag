"""
Metadata Extractor Implementation
File: ingestion_service/processors/metadata_extractor.py
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import statistics

from llama_index.core import Document
from llama_index.core.extractors import (
    TitleExtractor, KeywordExtractor, SummaryExtractor,
    QuestionsAnsweredExtractor, EntityExtractor
)
from llama_index.llms.openai import OpenAI

from config import IngestionConfig

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts rich metadata from document chunks including titles, summaries, keywords, and entities
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        # Initialize LLM for metadata extraction
        self.llm = OpenAI(model=config.llm_model, temperature=0.1)
        
        # Initialize extractors
        self.extractors = []
        
        if config.extract_metadata:
            try:
                # Title extractor
                self.title_extractor = TitleExtractor(
                    llm=self.llm,
                    nodes=5  # Consider 5 nodes for title extraction
                )
                self.extractors.append(self.title_extractor)
                
                # Summary extractor
                self.summary_extractor = SummaryExtractor(
                    llm=self.llm,
                    summaries=["prev", "self"],
                    prompt_template="Summarize the following text in 1-2 sentences: {context_str}"
                )
                self.extractors.append(self.summary_extractor)
                
                # Keyword extractor
                self.keyword_extractor = KeywordExtractor(
                    llm=self.llm,
                    keywords=10  # Extract up to 10 keywords
                )
                self.extractors.append(self.keyword_extractor)
                
                # Questions answered extractor
                self.questions_extractor = QuestionsAnsweredExtractor(
                    llm=self.llm,
                    questions=3  # Extract up to 3 questions
                )
                self.extractors.append(self.questions_extractor)
                
                # Entity extractor (if available)
                try:
                    self.entity_extractor = EntityExtractor(
                        prediction_threshold=0.5,
                        label_entities=False,
                        device="cpu"
                    )
                    self.extractors.append(self.entity_extractor)
                except Exception as e:
                    logger.warning(f"Entity extractor not available: {e}")
                    self.entity_extractor = None
                
                logger.info(f"Initialized {len(self.extractors)} metadata extractors")
                
            except Exception as e:
                logger.error(f"Error initializing metadata extractors: {e}")
                self.extractors = []
        
        # Fallback extractors (rule-based)
        self.fallback_enabled = True
        
        # Common patterns for different content types
        self.patterns = {
            'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_numbers': r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'dates': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'numbers': r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent|\s*million|\s*billion|\s*thousand)?\b',
            'monetary_amounts': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)',
            'technical_terms': r'\b(?:API|SDK|HTTP|JSON|XML|SQL|HTML|CSS|JavaScript|Python|algorithm|framework|database|server|client|protocol)\b',
            'academic_citations': r'\[?\d+\]?|\(\d{4}\)|\w+\s+et\s+al\.|\w+\s+\(\d{4}\)'
        }
        
        logger.info("Metadata extractor initialized")
    
    async def extract_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from a list of documents
        """
        if not documents:
            return documents
        
        try:
            # Process documents in batches to avoid overwhelming the LLM
            batch_size = 5
            enhanced_documents = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_results = await self._process_batch(batch)
                enhanced_documents.extend(batch_results)
            
            logger.debug(f"Enhanced {len(enhanced_documents)} documents with metadata")
            return enhanced_documents
            
        except Exception as e:
            logger.error(f"Error in metadata extraction: {e}")
            # Return original documents with fallback metadata
            return [await self._add_fallback_metadata(doc) for doc in documents]
    
    async def _process_batch(self, documents: List[Document]) -> List[Document]:
        """
        Process a batch of documents for metadata extraction
        """
        enhanced_docs = []
        
        for document in documents:
            try:
                # Apply LLM-based extractors if available
                if self.extractors:
                    enhanced_doc = await self._apply_llm_extractors(document)
                else:
                    enhanced_doc = document
                
                # Apply rule-based extractors
                enhanced_doc = await self._add_fallback_metadata(enhanced_doc)
                
                # Add content analysis
                enhanced_doc = self._add_content_analysis(enhanced_doc)
                
                enhanced_docs.append(enhanced_doc)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                # Add basic fallback metadata
                enhanced_docs.append(await self._add_fallback_metadata(document))
        
        return enhanced_docs
    
    async def _apply_llm_extractors(self, document: Document) -> Document:
        """
        Apply LLM-based metadata extractors
        """
        try:
            # Convert Document to BaseNode for extractors
            from llama_index.core.schema import TextNode
            
            node = TextNode(
                text=document.text,
                metadata=document.metadata.copy()
            )
            
            # Apply each extractor
            for extractor in self.extractors:
                try:
                    nodes = await asyncio.to_thread(extractor.extract, [node])
                    if nodes:
                        node = nodes[0]  # Get the enhanced node
                except Exception as e:
                    logger.warning(f"Extractor {type(extractor).__name__} failed: {e}")
                    continue
            
            # Convert back to Document
            enhanced_document = Document(
                text=document.text,
                metadata=node.metadata
            )
            
            return enhanced_document
            
        except Exception as e:
            logger.error(f"Error applying LLM extractors: {e}")
            return document
    
    async def _add_fallback_metadata(self, document: Document) -> Document:
        """
        Add rule-based metadata extraction as fallback
        """
        try:
            text = document.text
            metadata = document.metadata.copy()
            
            # Extract entities using pattern matching
            entities = self._extract_entities_fallback(text)
            if entities:
                metadata['entities'] = entities
            
            # Extract keywords using frequency analysis
            keywords = self._extract_keywords_fallback(text)
            if keywords:
                metadata['keywords'] = keywords
            
            # Extract title if not already present
            if 'title' not in metadata or not metadata['title']:
                title = self._extract_title_fallback(text)
                if title:
                    metadata['title'] = title
            
            # Generate summary if not present
            if 'summary' not in metadata or not metadata['summary']:
                summary = self._generate_summary_fallback(text)
                if summary:
                    metadata['summary'] = summary
            
            # Extract structured information
            structured_info = self._extract_structured_info(text)
            metadata.update(structured_info)
            
            return Document(text=text, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error in fallback metadata extraction: {e}")
            return document
    
    def _extract_entities_fallback(self, text: str) -> List[str]:
        """
        Extract entities using pattern matching and heuristics
        """
        entities = set()
        
        # Extract using patterns
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(matches)
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'And', 'Or', 'But',
            'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'About'
        }
        
        filtered_entities = [word for word in capitalized_words 
                           if word not in common_words and len(word) > 2]
        
        entities.update(filtered_entities[:10])  # Limit to 10 entities
        
        return list(entities)[:15]  # Return up to 15 entities
    
    def _extract_keywords_fallback(self, text: str) -> List[str]:
        """
        Extract keywords using frequency analysis and TF-IDF-like scoring
        """
        # Clean and tokenize text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
            'very', 'can', 'will', 'just', 'should', 'now', 'also', 'may',
            'might', 'must', 'could', 'would', 'have', 'has', 'had', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did'
        }
        
        # Filter words
        filtered_words = [word for word in words 
                         if word not in stop_words and len(word) > 3]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        # Calculate simple importance score (frequency + length bonus)
        scored_words = []
        for word, freq in word_freq.items():
            score = freq + (len(word) - 3) * 0.1  # Bonus for longer words
            scored_words.append((word, score))
        
        # Sort by score and return top keywords
        scored_words.sort(key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in scored_words[:15]]
        
        return keywords
    
    def _extract_title_fallback(self, text: str) -> Optional[str]:
        """
        Extract title using heuristics
        """
        lines = text.split('\n')
        
        # Look for title patterns
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line:
                continue
                
            # Check for markdown headers
            if line.startswith('#'):
                return line.lstrip('#').strip()
            
            # Check for title-like patterns (short, capitalized)
            if (len(line) < 100 and 
                len(line.split()) <= 10 and
                line[0].isupper() and
                not line.endswith('.')):
                return line
        
        # Fallback: first sentence if it's short enough
        sentences = re.split(r'[.!?]+', text)
        if sentences and len(sentences[0]) < 150:
            return sentences[0].strip()
        
        return None
    
    def _generate_summary_fallback(self, text: str) -> Optional[str]:
        """
        Generate summary using extractive methods
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 2:
            return ' '.join(sentences)
        
        # Simple extractive summarization
        # Score sentences by keyword density and position
        keywords = self._extract_keywords_fallback(text)[:10]
        
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # Keyword density score
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    score += 1
            
            # Position score (first and last sentences are important)
            if i == 0:
                score += 2
            elif i == len(sentences) - 1:
                score += 1
            elif i < len(sentences) * 0.3:  # First third
                score += 1
            
            # Length penalty for very long sentences
            if len(sentence) > 200:
                score -= 1
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top 2-3 sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s for s, score in sentence_scores[:2]]
        
        return ' '.join(top_sentences)
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information like emails, dates, numbers, etc.
        """
        structured_info = {}
        
        # Extract emails
        emails = re.findall(self.patterns['email_addresses'], text)
        if emails:
            structured_info['email_addresses'] = list(set(emails))
        
        # Extract phone numbers
        phones = re.findall(self.patterns['phone_numbers'], text)
        if phones:
            structured_info['phone_numbers'] = list(set(phones))
        
        # Extract URLs
        urls = re.findall(self.patterns['urls'], text)
        if urls:
            structured_info['urls'] = list(set(urls))
        
        # Extract dates
        dates = re.findall(self.patterns['dates'], text)
        if dates:
            structured_info['dates'] = list(set(dates))
        
        # Extract monetary amounts
        amounts = re.findall(self.patterns['monetary_amounts'], text)
        if amounts:
            structured_info['monetary_amounts'] = list(set(amounts))
        
        # Extract technical terms
        tech_terms = re.findall(self.patterns['technical_terms'], text, re.IGNORECASE)
        if tech_terms:
            structured_info['technical_terms'] = list(set([term.lower() for term in tech_terms]))
        
        return structured_info
    
    def _add_content_analysis(self, document: Document) -> Document:
        """
        Add content analysis metadata
        """
        text = document.text
        metadata = document.metadata.copy()
        
        # Basic statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        metadata.update({
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': word_count / max(sentence_count, 1),
            'avg_chars_per_word': char_count / max(word_count, 1)
        })
        
        # Content type analysis
        content_type = self._analyze_content_type(text)
        metadata['content_type'] = content_type
        
        # Language complexity analysis
        complexity = self._analyze_language_complexity(text)
        metadata.update(complexity)
        
        # Reading time estimate (average 200 words per minute)
        reading_time_minutes = word_count / 200
        metadata['estimated_reading_time_minutes'] = round(reading_time_minutes, 1)
        
        return Document(text=text, metadata=metadata)
    
    def _analyze_content_type(self, text: str) -> str:
        """
        Analyze and classify content type
        """
        text_lower = text.lower()
        
        # Check for different content type indicators
        if any(pattern in text_lower for pattern in ['def ', 'class ', 'import ', 'function']):
            return 'code'
        elif any(pattern in text_lower for pattern in ['abstract', 'methodology', 'references', 'bibliography']):
            return 'academic'
        elif any(pattern in text_lower for pattern in ['table', 'figure', 'chart', 'graph']):
            return 'analytical'
        elif re.search(r'^\s*#', text, re.MULTILINE):
            return 'documentation'
        elif re.search(r'[|,]\s*[|,]', text):
            return 'tabular'
        elif any(pattern in text_lower for pattern in ['email', 'subject:', 'from:', 'to:']):
            return 'communication'
        elif any(pattern in text_lower for pattern in ['recipe', 'ingredients', 'instructions']):
            return 'procedural'
        else:
            return 'general'
    
    def _analyze_language_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze language complexity metrics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {'language_complexity': 'unknown'}
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable estimation (simple heuristic)
        def estimate_syllables(word):
            word = word.lower().strip('.,!?;:"')
            if not word:
                return 0
            count = 0
            vowels = 'aeiouy'
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count = 1
            return count
        
        # Calculate average syllables per word
        total_syllables = sum(estimate_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease approximation
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Classify complexity
        if flesch_score >= 90:
            complexity = 'very_easy'
        elif flesch_score >= 80:
            complexity = 'easy'
        elif flesch_score >= 70:
            complexity = 'fairly_easy'
        elif flesch_score >= 60:
            complexity = 'standard'
        elif flesch_score >= 50:
            complexity = 'fairly_difficult'
        elif flesch_score >= 30:
            complexity = 'difficult'
        else:
            complexity = 'very_difficult'
        
        return {
            'language_complexity': complexity,
            'flesch_score': round(flesch_score, 1),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'total_syllables': total_syllables
        }
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get metadata extraction statistics
        """
        return {
            'llm_extractors_enabled': len(self.extractors) > 0,
            'extractor_count': len(self.extractors),
            'fallback_enabled': self.fallback_enabled,
            'available_patterns': list(self.patterns.keys()),
            'llm_model': self.config.llm_model if hasattr(self.config, 'llm_model') else 'unknown'
        }
    
    async def extract_questions_answered(self, text: str) -> List[str]:
        """
        Extract what questions this text might answer
        """
        try:
            if self.questions_extractor:
                # Use LLM-based extraction
                from llama_index.core.schema import TextNode
                node = TextNode(text=text)
                enhanced_nodes = await asyncio.to_thread(self.questions_extractor.extract, [node])
                if enhanced_nodes and 'questions_this_excerpt_can_answer' in enhanced_nodes[0].metadata:
                    return enhanced_nodes[0].metadata['questions_this_excerpt_can_answer']
            
            # Fallback: generate questions based on content patterns
            return self._generate_questions_fallback(text)
            
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []
    
    def _generate_questions_fallback(self, text: str) -> List[str]:
        """
        Generate potential questions using pattern matching
        """
        questions = []
        text_lower = text.lower()
        
        # Look for definition patterns
        if 'is' in text_lower or 'are' in text_lower:
            # Find subjects that might be defined
            definition_patterns = [
                r'(\w+)\s+is\s+',
                r'(\w+)\s+are\s+',
                r'(\w+)\s+refers\s+to',
                r'(\w+)\s+means'
            ]
            for pattern in definition_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:3]:  # Limit to 3
                    questions.append(f"What is {match}?")
        
        # Look for process/method patterns
        if any(word in text_lower for word in ['step', 'process', 'method', 'procedure']):
            # Extract the main topic
            words = text.split()[:20]  # First 20 words
            topic_words = [w for w in words if len(w) > 4 and w.isalpha()]
            if topic_words:
                questions.append(f"How does {topic_words[0]} work?")
        
        # Look for comparison patterns
        if any(word in text_lower for word in ['versus', 'compared to', 'difference', 'similarity']):
            questions.append("What are the key differences mentioned?")
        
        # Look for numerical data
        if re.search(r'\d+%|\d+\s+percent', text):
            questions.append("What are the key statistics or percentages?")
        
        return questions[:5]  # Limit to 5 questions
    
    async def extract_semantic_tags(self, text: str) -> List[str]:
        """
        Extract semantic tags that describe the content
        """
        tags = set()
        text_lower = text.lower()
        
        # Domain-specific tags
        domain_keywords = {
            'technology': ['software', 'hardware', 'algorithm', 'programming', 'computer', 'digital', 'api', 'database'],
            'business': ['revenue', 'profit', 'market', 'customer', 'strategy', 'sales', 'business', 'company'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'data', 'analysis', 'scientific', 'method'],
            'education': ['learning', 'teaching', 'student', 'education', 'academic', 'course', 'curriculum'],
            'health': ['medical', 'health', 'patient', 'treatment', 'therapy', 'diagnosis', 'clinical'],
            'finance': ['financial', 'investment', 'money', 'economic', 'banking', 'credit', 'loan', 'budget'],
            'legal': ['law', 'legal', 'court', 'regulation', 'compliance', 'contract', 'policy'],
            'engineering': ['engineering', 'design', 'construction', 'mechanical', 'electrical', 'civil']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.add(domain)
        
        # Content type tags
        if re.search(r'```|def\s+\w+|class\s+\w+', text):
            tags.add('code')
        if re.search(r'\|.*\|.*\|', text):
            tags.add('table')
        if re.search(r'^\d+\.', text, re.MULTILINE):
            tags.add('list')
        if any(word in text_lower for word in ['figure', 'chart', 'graph', 'diagram']):
            tags.add('visual')
        
        # Format tags
        if re.search(r'^#{1,6}\s', text, re.MULTILINE):
            tags.add('markdown')
        if re.search(r'<[^>]+>', text):
            tags.add('html')
        if '@' in text and '.' in text:
            tags.add('contact_info')
        
        return list(tags)[:10]  # Limit to 10 tags