"""
LLM Integration for Agentic RAG Service
File: agentic_rag_agent/services/llm_service.py
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
from datetime import datetime

from config import RAGConfig
from models.response_models import DocumentChunk, QueryPlan

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for LLM interactions with proper prompt engineering
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = openai.AsyncOpenAI()
        
        # Prompt templates
        self.prompts = {
            'answer_generation': """
Based on the provided sources, answer the user's question comprehensively and accurately.

User Question: {question}

Retrieved Sources:
{sources}

Instructions:
- Provide a clear, well-structured answer based on the sources
- Cite specific sources when making claims (use source numbers)
- If information is incomplete or conflicting, acknowledge this
- Be factual and avoid speculation beyond what the sources support
- Structure your response logically with clear sections if appropriate
- If the sources don't contain relevant information, state this clearly

Answer:""",
            
            'simple_answer_generation': """
Based on the following source, provide a concise answer to the user's question.

Question: {question}

Source: {source}

Provide a clear, direct answer based on the source material:""",
            
            'follow_up_generation': """
Based on the conversation context, generate 3-5 intelligent follow-up questions that would help the user explore this topic further.

Original Question: {question}
Answer Provided: {answer}
Sources Used: {source_count} sources

Generate follow-up questions that are:
- Specific and actionable
- Build naturally on the information provided
- Help explore different aspects of the topic
- Suitable for further research

Follow-up Questions:""",
            
            'query_enhancement': """
Enhance the user's query to improve search results by adding relevant context and synonyms.

Original Query: {query}
Context: {context}

Enhanced Query:""",
            
            'summary_generation': """
Summarize the following content in 1-2 concise sentences that capture the main points.

Content: {content}

Summary:""",
            
            'title_extraction': """
Extract or generate an appropriate title for the following content. The title should be concise and descriptive.

Content: {content}

Title:""",
            
            'content_analysis': """
Analyze the following content and determine:
1. Main topic/subject
2. Content type (technical, academic, business, etc.)
3. Key themes or concepts
4. Target audience level

Content: {content}

Analysis:"""
        }
        
        logger.info("LLM Service initialized")
    
    async def generate_answer(
        self, 
        question: str, 
        sources: List[DocumentChunk], 
        query_plan: QueryPlan
    ) -> str:
        """
        Generate comprehensive answer using retrieved sources
        """
        try:
            if not sources:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
            
            # Prepare sources context
            sources_text = self._format_sources_for_prompt(sources[:8])  # Use top 8 sources
            
            # Create prompt
            prompt = self.prompts['answer_generation'].format(
                question=question,
                sources=sources_text
            )
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a knowledgeable research assistant that provides accurate, well-sourced answers based on retrieved documents. Always cite your sources and be precise in your responses."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Post-process answer
            return self._post_process_answer(answer, sources)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the answer. The sources contain information about {question.lower()}, but I cannot process it properly at the moment."
    
    async def generate_simple_answer(
        self, 
        question: str, 
        top_source: DocumentChunk
    ) -> str:
        """
        Generate simple answer for non-agentic mode
        """
        try:
            if not top_source:
                return "I couldn't find relevant information to answer your question."
            
            # Truncate source if too long
            source_text = top_source.content[:1500] + "..." if len(top_source.content) > 1500 else top_source.content
            
            prompt = self.prompts['simple_answer_generation'].format(
                question=question,
                source=f"Source: {top_source.file_name}\n\n{source_text}"
            )
            
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise, accurate answers based on provided sources."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating simple answer: {e}")
            return f"Based on the information in {top_source.file_name}, I found relevant content but encountered an error processing it."
    
    async def generate_follow_ups(
        self, 
        question: str, 
        answer: str, 
        sources: List[DocumentChunk]
    ) -> List[str]:
        """
        Generate intelligent follow-up questions
        """
        try:
            prompt = self.prompts['follow_up_generation'].format(
                question=question,
                answer=answer[:800],  # Truncate long answers
                source_count=len(sources)
            )
            
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at generating insightful follow-up questions that help users explore topics more deeply."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300
            )
            
            # Parse follow-up questions
            follow_ups_text = response.choices[0].message.content.strip()
            follow_ups = self._parse_follow_ups(follow_ups_text)
            
            return follow_ups[:5]  # Limit to 5
            
        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return [
                "Can you provide more specific details?",
                "What are the practical implications?",
                "Are there any related topics to explore?"
            ]
    
    async def enhance_query(self, query: str, context: str = "") -> str:
        """
        Enhance query for better search results
        """
        try:
            prompt = self.prompts['query_enhancement'].format(
                query=query,
                context=context or "No additional context provided"
            )
            
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at optimizing search queries by adding relevant context and synonyms while maintaining the original intent."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            
            # Ensure the enhanced query isn't too different from original
            if len(enhanced_query) > len(query) * 3:
                return query  # Fallback to original if enhancement is too long
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query
    
    async def generate_summary(self, content: str) -> str:
        """
        Generate summary of content
        """
        try:
            # Truncate very long content
            content_text = content[:2000] + "..." if len(content) > 2000 else content
            
            prompt = self.prompts['summary_generation'].format(content=content_text)
            
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are expert at creating concise, informative summaries that capture the essential information."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary could not be generated."
    
    async def extract_title(self, content: str) -> str:
        """
        Extract or generate title for content
        """
        try:
            # Use first part of content for title extraction
            content_sample = content[:1000]
            
            prompt = self.prompts['title_extraction'].format(content=content_sample)
            
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are expert at creating descriptive, concise titles that accurately represent content."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=50
            )
            
            title = response.choices[0].message.content.strip()
            
            # Clean up title (remove quotes, limit length)
            title = title.strip('"\'').strip()
            if len(title) > 100:
                title = title[:97] + "..."
            
            return title
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return "Untitled Document"
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for type, themes, and audience
        """
        try:
            content_sample = content[:1500]
            
            prompt = self.prompts['content_analysis'].format(content=content_sample)
            
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert content analyst. Provide structured analysis in the format requested."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Parse analysis (simple parsing, could be enhanced)
            analysis = {
                'full_analysis': analysis_text,
                'content_type': 'general',
                'complexity': 'medium',
                'domain': 'general'
            }
            
            # Extract specific fields if possible
            if 'technical' in analysis_text.lower():
                analysis['content_type'] = 'technical'
            elif 'academic' in analysis_text.lower():
                analysis['content_type'] = 'academic'
            elif 'business' in analysis_text.lower():
                analysis['content_type'] = 'business'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {
                'full_analysis': 'Analysis could not be completed.',
                'content_type': 'unknown',
                'complexity': 'unknown',
                'domain': 'unknown'
            }
    
    async def test_llm_connection(self) -> bool:
        """
        Test LLM service connectivity
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "user", "content": "Respond with 'OK' if you can process this message."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            return "ok" in response.choices[0].message.content.lower()
            
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False
    
    def _format_sources_for_prompt(self, sources: List[DocumentChunk]) -> str:
        """
        Format sources for inclusion in prompts
        """
        formatted_sources = []
        
        for i, source in enumerate(sources, 1):
            # Include title if available
            title_info = f" - {source.title}" if source.title else ""
            
            # Truncate long content
            content = source.content
            if len(content) > 800:
                content = content[:800] + "..."
            
            formatted_source = f"""
Source {i} ({source.file_name}{title_info}):
{content}
Relevance Score: {source.similarity_score:.2f}
"""
            formatted_sources.append(formatted_source)
        
        return "\n".join(formatted_sources)
    
    def _post_process_answer(self, answer: str, sources: List[DocumentChunk]) -> str:
        """
        Post-process generated answer
        """
        # Ensure source citations are properly formatted
        # Add source information if not present
        if not any(f"source {i}" in answer.lower() for i in range(1, 6)):
            if sources:
                answer += f"\n\nBased on information from {len(sources)} source(s) including {sources[0].file_name}"
                if len(sources) > 1:
                    answer += f" and others"
                answer += "."
        
        return answer
    
    def _parse_follow_ups(self, follow_ups_text: str) -> List[str]:
        """
        Parse follow-up questions from LLM response
        """
        # Split by lines and clean up
        lines = follow_ups_text.split('\n')
        follow_ups = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering and bullet points
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            
            # Ensure it ends with a question mark
            if not line.endswith('?'):
                line += '?'
            
            if len(line) > 10:  # Minimum length filter
                follow_ups.append(line)
        
        return follow_ups


# Update the agentic_rag_service.py to use the LLM service
"""
Updates to agentic_rag_agent/agents/agentic_rag_service.py
Replace the placeholder LLM implementations with actual calls to LLMService
"""

import re

class AgenticRAGServiceUpdates:
    """
    Updates to integrate LLMService into AgenticRAGService
    """
    
    def __init__(self, config: RAGConfig):
        # Add to __init__ method
        from services.llm_service import LLMService
        self.llm_service = LLMService(config)
    
    async def _generate_answer(
        self, 
        question: str, 
        sources: List[DocumentChunk], 
        query_plan: QueryPlan
    ) -> str:
        """
        Replace the placeholder implementation in agentic_rag_service.py
        """
        return await self.llm_service.generate_answer(question, sources, query_plan)
    
    async def _generate_simple_answer(self, question: str, sources: List[DocumentChunk]) -> str:
        """
        Replace the placeholder implementation in agentic_rag_service.py
        """
        if not sources:
            return "I couldn't find relevant information to answer your question."
        
        return await self.llm_service.generate_simple_answer(question, sources[0])
    
    async def _generate_follow_ups(
        self, 
        question: str, 
        sources: List[DocumentChunk], 
        reflection: Any
    ) -> List[str]:
        """
        Replace the placeholder implementation in agentic_rag_service.py
        """
        # Get answer from last generation (would need to be stored)
        answer = "Previous answer context"  # This would come from actual answer
        
        # Use reflection follow-ups first
        follow_ups = list(reflection.suggested_follow_ups)
        
        # Generate additional LLM-based follow-ups
        llm_follow_ups = await self.llm_service.generate_follow_ups(question, answer, sources)
        follow_ups.extend(llm_follow_ups)
        
        # Add source-based follow-ups
        if sources:
            unique_topics = set()
            for source in sources[:3]:
                if source.keywords:
                    unique_topics.update(source.keywords[:2])
            
            follow_ups.extend([
                f"Tell me more about {topic}" for topic in list(unique_topics)[:2]
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_follow_ups = []
        for follow_up in follow_ups:
            if follow_up.lower() not in seen:
                unique_follow_ups.append(follow_up)
                seen.add(follow_up.lower())
        
        return unique_follow_ups[:5]  # Limit to 5 suggestions
    
    async def _test_llm_service(self) -> bool:
        """
        Replace the placeholder implementation in agentic_rag_service.py
        """
        return await self.llm_service.test_llm_connection()


# Example usage and integration instructions
INTEGRATION_INSTRUCTIONS = """
To integrate the LLM service into your existing agentic RAG service:

1. Add the import to agentic_rag_service.py:
   from services.llm_service import LLMService

2. Initialize in __init__ method:
   self.llm_service = LLMService(config)

3. Replace placeholder methods with:
   - _generate_answer -> self.llm_service.generate_answer
   - _generate_simple_answer -> self.llm_service.generate_simple_answer  
   - _generate_follow_ups -> Use combination of reflection + LLM service
   - _test_llm_service -> self.llm_service.test_llm_connection

4. The LLM service provides proper prompt engineering and error handling
   for all LLM interactions.
"""