"""
Context-Aware Chunker Implementation
File: ingestion_service/processors/context_aware_chunker.py
"""
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import Document, Settings
from llama_index.core.node_parser import (
    SentenceSplitter, 
    HierarchicalNodeParser,
    CodeSplitter,
    SemanticSplitterNodeParser
)
from llama_index.core.schema import BaseNode

from config import IngestionConfig

logger = logging.getLogger(__name__)


class ContextAwareChunker:
    """
    Intelligent chunker that adapts strategy based on content type and structure
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        # Initialize different chunking strategies
        self.sentence_splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )
        
        self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128] if config.enable_hierarchical_chunking else [config.chunk_size]
        )
        
        self.code_splitter = CodeSplitter(
            language="python",  # Will be detected dynamically
            chunk_lines=config.code_chunk_lines,
            chunk_lines_overlap=5,
            max_chars=config.chunk_size
        )
        
        # Semantic splitter (if enabled)
        if config.enable_semantic_chunking:
            try:
                self.semantic_splitter = SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=Settings.embed_model  # Will use default
                )
            except Exception as e:
                logger.warning(f"Semantic splitter initialization failed: {e}")
                self.semantic_splitter = None
        else:
            self.semantic_splitter = None
        
        # Content type patterns
        self.content_patterns = {
            'code': {
                'patterns': [
                    r'(?:def|class|function|import|from|#include|using namespace)',
                    r'(?:public|private|protected|static|const)',
                    r'(?:\{|\}|;|\(\)|=>|->)',
                    r'(?:if\s*\(|for\s*\(|while\s*\(|switch\s*\()'
                ],
                'min_matches': 2
            },
            'structured': {
                'patterns': [
                    r'^#{1,6}\s+',  # Markdown headers
                    r'^\d+\.\s+',   # Numbered lists
                    r'^\*\s+|-\s+', # Bullet points
                    r'^\|.*\|',     # Tables
                    r'<[^>]+>',     # HTML/XML tags
                ],
                'min_matches': 3
            },
            'tabular': {
                'patterns': [
                    r'\t.*\t',      # Tab-separated
                    r',.*,.*,',     # CSV-like
                    r'\|.*\|.*\|',  # Pipe-separated tables
                    r'^\s*\d+\s+\w+\s+\w+',  # Columnar data
                ],
                'min_matches': 2
            },
            'academic': {
                'patterns': [
                    r'(?:Abstract|Introduction|Methodology|Results|Discussion|Conclusion)',
                    r'(?:Figure \d+|Table \d+|Equation \d+)',
                    r'(?:\[?\d+\]|\(\d{4}\))',  # Citations
                    r'(?:et al\.|ibid\.|op\. cit\.)',
                ],
                'min_matches': 2
            }
        }
        
        logger.info("Context-aware chunker initialized")
    
    async def chunk_document(self, document: Document) -> List[Document]:
        """
        Main chunking method that selects appropriate strategy based on content
        """
        try:
            # Analyze content type
            content_type = self._analyze_content_type(document.text)
            file_type = document.metadata.get('file_type', '').lower()
            
            logger.debug(f"Chunking document with content_type={content_type}, file_type={file_type}")
            
            # Select chunking strategy
            strategy = self._select_chunking_strategy(content_type, file_type)
            
            # Apply chunking strategy
            chunks = await self._apply_chunking_strategy(document, strategy, content_type)
            
            # Post-process chunks
            enhanced_chunks = self._post_process_chunks(chunks, document)
            
            logger.debug(f"Created {len(enhanced_chunks)} chunks using {strategy} strategy")
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error in chunking document: {e}")
            # Fallback to simple sentence splitting
            return self._fallback_chunking(document)
    
    def _analyze_content_type(self, text: str) -> str:
        """
        Analyze text to determine content type
        """
        text_sample = text[:2000]  # Analyze first 2000 characters
        
        for content_type, config in self.content_patterns.items():
            matches = 0
            for pattern in config['patterns']:
                if re.search(pattern, text_sample, re.MULTILINE | re.IGNORECASE):
                    matches += 1
            
            if matches >= config['min_matches']:
                return content_type
        
        return 'general'
    
    def _select_chunking_strategy(self, content_type: str, file_type: str) -> str:
        """
        Select appropriate chunking strategy based on content and file type
        """
        # File type specific strategies
        if file_type in ['py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go']:
            return 'code'
        elif file_type in ['csv', 'tsv']:
            return 'tabular'
        elif file_type in ['json', 'xml', 'yaml', 'yml']:
            return 'structured'
        
        # Content type strategies
        if content_type == 'code':
            return 'code'
        elif content_type == 'structured':
            return 'hierarchical'
        elif content_type == 'tabular':
            return 'tabular'
        elif content_type == 'academic' and self.config.enable_semantic_chunking:
            return 'semantic'
        else:
            return 'sentence'
    
    async def _apply_chunking_strategy(
        self, 
        document: Document, 
        strategy: str, 
        content_type: str
    ) -> List[BaseNode]:
        """
        Apply the selected chunking strategy
        """
        try:
            if strategy == 'code':
                return self._chunk_code_content(document)
            elif strategy == 'hierarchical':
                return self._chunk_hierarchical_content(document)
            elif strategy == 'tabular':
                return self._chunk_tabular_content(document)
            elif strategy == 'semantic' and self.semantic_splitter:
                return self._chunk_semantic_content(document)
            else:
                return self._chunk_sentence_content(document)
                
        except Exception as e:
            logger.error(f"Error applying {strategy} chunking strategy: {e}")
            # Fallback to sentence chunking
            return self._chunk_sentence_content(document)
    
    def _chunk_code_content(self, document: Document) -> List[BaseNode]:
        """
        Chunk code content preserving function/class boundaries
        """
        file_type = document.metadata.get('file_type', 'python').lower()
        
        # Map file types to languages
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'cs': 'csharp',
            'php': 'php',
            'rb': 'ruby',
            'go': 'go'
        }
        
        language = language_map.get(file_type, 'python')
        
        # Update code splitter for specific language
        self.code_splitter = CodeSplitter(
            language=language,
            chunk_lines=self.config.code_chunk_lines,
            chunk_lines_overlap=5,
            max_chars=self.config.chunk_size
        )
        
        return self.code_splitter.get_nodes_from_documents([document])
    
    def _chunk_hierarchical_content(self, document: Document) -> List[BaseNode]:
        """
        Chunk structured content preserving hierarchy
        """
        return self.hierarchical_parser.get_nodes_from_documents([document])
    
    def _chunk_tabular_content(self, document: Document) -> List[BaseNode]:
        """
        Chunk tabular content preserving table structure
        """
        text = document.text
        chunks = []
        
        # Split by double newlines (table separation)
        sections = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        for section in sections:
            # If adding this section would exceed chunk size, save current chunk
            if len(current_chunk) + len(section) > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Convert to Document objects
        nodes = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                text=chunk_text,
                metadata={**document.metadata, 'chunk_type': 'tabular', 'chunk_index': i}
            )
            nodes.append(chunk_doc)
        
        return nodes
    
    def _chunk_semantic_content(self, document: Document) -> List[BaseNode]:
        """
        Chunk content using semantic boundaries
        """
        try:
            return self.semantic_splitter.get_nodes_from_documents([document])
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to sentence chunking: {e}")
            return self._chunk_sentence_content(document)
    
    def _chunk_sentence_content(self, document: Document) -> List[BaseNode]:
        """
        Default sentence-based chunking
        """
        return self.sentence_splitter.get_nodes_from_documents([document])
    
    def _post_process_chunks(self, chunks: List[BaseNode], original_document: Document) -> List[Document]:
        """
        Post-process chunks to add context and metadata
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Convert to Document if it's a BaseNode
            if not isinstance(chunk, Document):
                chunk_doc = Document(
                    text=chunk.text,
                    metadata=chunk.metadata if hasattr(chunk, 'metadata') else {}
                )
            else:
                chunk_doc = chunk
            
            # Add chunk-specific metadata
            chunk_doc.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'parent_node_id': original_document.metadata.get('document_id'),
                'chunk_type': chunk_doc.metadata.get('chunk_type', 'standard'),
                'word_count': len(chunk_doc.text.split()),
                'char_count': len(chunk_doc.text)
            })
            
            # Add context previews
            if i > 0:
                prev_chunk = chunks[i-1]
                prev_text = prev_chunk.text if hasattr(prev_chunk, 'text') else str(prev_chunk)
                chunk_doc.metadata['previous_chunk_preview'] = prev_text[-100:]  # Last 100 chars
            
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                next_text = next_chunk.text if hasattr(next_chunk, 'text') else str(next_chunk)
                chunk_doc.metadata['next_chunk_preview'] = next_text[:100]  # First 100 chars
            
            # Inherit original document metadata
            chunk_doc.metadata.update({
                k: v for k, v in original_document.metadata.items() 
                if k not in chunk_doc.metadata
            })
            
            processed_chunks.append(chunk_doc)
        
        return processed_chunks
    
    def _fallback_chunking(self, document: Document) -> List[Document]:
        """
        Fallback chunking strategy when all else fails
        """
        try:
            chunks = self.sentence_splitter.get_nodes_from_documents([document])
            return self._post_process_chunks(chunks, document)
        except Exception as e:
            logger.error(f"Even fallback chunking failed: {e}")
            # Ultimate fallback - just split by character count
            return self._simple_character_split(document)
    
    def _simple_character_split(self, document: Document) -> List[Document]:
        """
        Simple character-based splitting as ultimate fallback
        """
        text = document.text
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_doc = Document(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        'chunk_index': len(chunks),
                        'chunk_type': 'character_split',
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    }
                )
                chunks.append(chunk_doc)
            
            start = max(end - overlap, start + 1)  # Ensure progress
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """
        Get statistics about chunking performance
        """
        return {
            'strategies_available': [
                'sentence', 'hierarchical', 'code', 'tabular', 
                'semantic' if self.semantic_splitter else None
            ],
            'config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'enable_semantic_chunking': self.config.enable_semantic_chunking,
                'enable_hierarchical_chunking': self.config.enable_hierarchical_chunking,
                'code_chunk_lines': self.config.code_chunk_lines
            }
        }