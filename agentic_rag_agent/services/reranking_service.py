"""
Reranking Service for Agentic RAG Agent
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics

import numpy as np
from sentence_transformers import CrossEncoder
import torch

from config import RAGConfig
from models.response_models import DocumentChunk
from utils.logger import get_logger
from utils.metrics import MetricsCollector

logger = get_logger(__name__)


class RerankingStrategy(Enum):
    """Available reranking strategies"""
    CROSS_ENCODER = "cross_encoder"
    COLBERT = "colbert"
    COMBINED = "combined"
    CUSTOM = "custom"


@dataclass
class RerankingRequest:
    """Request for reranking documents"""
    query: str
    documents: List[DocumentChunk]
    strategy: RerankingStrategy = RerankingStrategy.CROSS_ENCODER
    top_k: Optional[int] = None
    request_id: Optional[str] = None


@dataclass
class RerankingResponse:
    """Response from reranking service"""
    reranked_documents: List[DocumentChunk]
    strategy_used: RerankingStrategy
    processing_time: float
    original_count: int
    reranked_count: int
    request_id: Optional[str] = None


class CrossEncoderReranker:
    """Cross-encoder based reranking"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded cross-encoder model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            raise
    
    async def rerank(
        self, 
        query: str, 
        documents: List[DocumentChunk], 
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Rerank documents using cross-encoder"""
        if not self.model or not documents:
            return documents
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc in documents:
                # Combine title and content for better scoring
                doc_text = self._prepare_document_text(doc)
                query_doc_pairs.append([query, doc_text])
            
            # Get reranking scores
            scores = await asyncio.to_thread(
                self.model.predict,
                query_doc_pairs
            )
            
            # Assign scores to documents
            for i, doc in enumerate(documents):
                doc.rerank_score = float(scores[i])
                # Update combined score with reranking
                doc.combined_score = self._combine_scores(doc)
            
            # Sort by rerank score
            reranked_docs = sorted(documents, key=lambda x: x.rerank_score, reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return documents
    
    def _prepare_document_text(self, doc: DocumentChunk) -> str:
        """Prepare document text for reranking"""
        parts = []
        
        if doc.title:
            parts.append(doc.title)
        
        # Add content (truncated if too long)
        content = doc.content
        if len(content) > 500:  # Truncate for efficiency
            content = content[:500] + "..."
        parts.append(content)
        
        return " ".join(parts)
    
    def _combine_scores(self, doc: DocumentChunk) -> float:
        """Combine original scores with rerank score"""
        # Weight: 60% rerank, 25% similarity, 15% BM25
        rerank_weight = 0.6
        similarity_weight = 0.25
        bm25_weight = 0.15
        
        combined = (
            rerank_weight * doc.rerank_score +
            similarity_weight * doc.similarity_score +
            bm25_weight * doc.bm25_score
        )
        
        return combined


class ColBERTReranker:
    """ColBERT-style token-level reranking (placeholder)"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # In a full implementation, this would load a ColBERT model
        logger.info(f"ColBERT reranker initialized: {model_name}")
    
    async def rerank(
        self, 
        query: str, 
        documents: List[DocumentChunk], 
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """ColBERT-style reranking (placeholder implementation)"""
        # This is a simplified placeholder
        # Real ColBERT implementation would involve token-level interactions
        
        for i, doc in enumerate(documents):
            # Placeholder scoring based on token overlap
            query_tokens = set(query.lower().split())
            doc_tokens = set(doc.content.lower().split())
            
            overlap_score = len(query_tokens & doc_tokens) / max(len(query_tokens | doc_tokens), 1)
            doc.rerank_score = overlap_score
            doc.combined_score = self._combine_scores(doc)
        
        # Sort by rerank score
        reranked_docs = sorted(documents, key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        return reranked_docs
    
    def _combine_scores(self, doc: DocumentChunk) -> float:
        """Combine scores for ColBERT reranking"""
        return (
            0.5 * doc.rerank_score +
            0.3 * doc.similarity_score +
            0.2 * doc.bm25_score
        )


class CustomReranker:
    """Custom reranking logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Custom reranker initialized")
    
    async def rerank(
        self, 
        query: str, 
        documents: List[DocumentChunk], 
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Custom reranking implementation"""
        
        for doc in documents:
            # Custom scoring logic
            score = self._calculate_custom_score(query, doc)
            doc.rerank_score = score
            doc.combined_score = score
        
        # Sort and return
        reranked_docs = sorted(documents, key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            reranked_docs = reranked_docs[:top_k]
        
        return reranked_docs
    
    def _calculate_custom_score(self, query: str, doc: DocumentChunk) -> float:
        """Calculate custom reranking score"""
        factors = []
        
        # Factor 1: Original similarity score
        factors.append(doc.similarity_score * 0.4)
        
        # Factor 2: BM25 score
        factors.append(doc.bm25_score * 0.3)
        
        # Factor 3: Document quality indicators
        quality_score = 0.0
        if doc.title:
            quality_score += 0.1
        if doc.summary:
            quality_score += 0.1
        if doc.keywords and len(doc.keywords) > 3:
            quality_score += 0.1
        if len(doc.content) > 200:
            quality_score += 0.1
        
        factors.append(quality_score * 0.2)
        
        # Factor 4: File type preference
        file_type_scores = {
            'pdf': 0.9,
            'docx': 0.8,
            'md': 0.7,
            'txt': 0.6,
            'html': 0.5
        }
        file_score = file_type_scores.get(doc.file_type, 0.5)
        factors.append(file_score * 0.1)
        
        return sum(factors)


class RerankingService:
    """
    Service for reranking search results using various strategies
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = MetricsCollector()
        
        # Initialize rerankers
        self.rerankers = {}
        
        # Load cross-encoder if enabled
        if config.enable_reranking:
            try:
                self.rerankers[RerankingStrategy.CROSS_ENCODER] = CrossEncoderReranker(
                    config.rerank_model
                )
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                config.enable_reranking = False
        
        # Initialize other rerankers
        self.rerankers[RerankingStrategy.COLBERT] = ColBERTReranker("colbert-base")
        self.rerankers[RerankingStrategy.CUSTOM] = CustomReranker({})
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'total_documents_processed': 0,
            'total_processing_time': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RerankingStrategy}
        }
        
        logger.info("Reranking service initialized")
    
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[DocumentChunk],
        strategy: RerankingStrategy = RerankingStrategy.CROSS_ENCODER,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> RerankingResponse:
        """
        Rerank documents using specified strategy
        """
        start_time = time.time()
        original_count = len(documents)
        
        if not documents:
            return RerankingResponse(
                reranked_documents=[],
                strategy_used=strategy,
                processing_time=0.0,
                original_count=0,
                reranked_count=0,
                request_id=request_id
            )
        
        try:
            # Validate top_k
            if top_k is None:
                top_k = len(documents)
            else:
                top_k = min(top_k, len(documents))
            
            # Perform reranking
            if strategy == RerankingStrategy.COMBINED:
                reranked_docs = await self._combined_reranking(query, documents, top_k)
            else:
                reranker = self.rerankers.get(strategy)
                if reranker:
                    reranked_docs = await reranker.rerank(query, documents, top_k)
                else:
                    logger.warning(f"Reranker not available for strategy {strategy}")
                    reranked_docs = documents[:top_k]
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['total_documents_processed'] += original_count
            self.stats['total_processing_time'] += processing_time
            self.stats['strategy_usage'][strategy.value] += 1
            
            # Record metrics
            self.metrics.record_reranking_request(
                strategy=strategy.value,
                document_count=original_count,
                processing_time=processing_time,
                top_k=top_k
            )
            
            return RerankingResponse(
                reranked_documents=reranked_docs,
                strategy_used=strategy,
                processing_time=processing_time,
                original_count=original_count,
                reranked_count=len(reranked_docs),
                request_id=request_id
            )
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            processing_time = time.time() - start_time
            
            return RerankingResponse(
                reranked_documents=documents[:top_k] if top_k else documents,
                strategy_used=strategy,
                processing_time=processing_time,
                original_count=original_count,
                reranked_count=min(top_k or len(documents), len(documents)),
                request_id=request_id
            )
    
    async def _combined_reranking(
        self, 
        query: str, 
        documents: List[DocumentChunk], 
        top_k: int
    ) -> List[DocumentChunk]:
        """
        Combine multiple reranking strategies
        """
        try:
            # Get results from different rerankers
            cross_encoder_docs = None
            custom_docs = None
            
            # Cross-encoder reranking
            if RerankingStrategy.CROSS_ENCODER in self.rerankers:
                cross_encoder_docs = await self.rerankers[RerankingStrategy.CROSS_ENCODER].rerank(
                    query, documents.copy(), top_k * 2
                )
            
            # Custom reranking
            custom_docs = await self.rerankers[RerankingStrategy.CUSTOM].rerank(
                query, documents.copy(), top_k * 2
            )
            
            # Combine scores using ensemble method
            combined_docs = self._ensemble_combine(
                query, documents, cross_encoder_docs, custom_docs
            )
            
            # Sort by final combined score
            combined_docs.sort(key=lambda x: x.combined_score, reverse=True)
            
            return combined_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in combined reranking: {e}")
            return documents[:top_k]
    
    def _ensemble_combine(
        self,
        query: str,
        original_docs: List[DocumentChunk],
        cross_encoder_docs: Optional[List[DocumentChunk]],
        custom_docs: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Ensemble combination of different reranking strategies
        """
        # Create score maps
        cross_encoder_scores = {}
        custom_scores = {}
        
        if cross_encoder_docs:
            for i, doc in enumerate(cross_encoder_docs):
                cross_encoder_scores[doc.id] = {
                    'score': doc.rerank_score,
                    'rank': i
                }
        
        for i, doc in enumerate(custom_docs):
            custom_scores[doc.id] = {
                'score': doc.rerank_score,
                'rank': i
            }
        
        # Combine scores for each document
        for doc in original_docs:
            scores = []
            
            # Cross-encoder contribution (weight: 0.6)
            if doc.id in cross_encoder_scores:
                ce_score = cross_encoder_scores[doc.id]['score']
                scores.append(ce_score * 0.6)
            else:
                scores.append(doc.similarity_score * 0.6)
            
            # Custom reranker contribution (weight: 0.4)
            if doc.id in custom_scores:
                custom_score = custom_scores[doc.id]['score']
                scores.append(custom_score * 0.4)
            else:
                scores.append(doc.similarity_score * 0.4)
            
            # Set final combined score
            doc.combined_score = sum(scores)
            doc.rerank_score = doc.combined_score
        
        return original_docs
    
    async def batch_rerank(
        self, 
        requests: List[RerankingRequest]
    ) -> List[RerankingResponse]:
        """
        Process multiple reranking requests in batch
        """
        tasks = []
        
        for request in requests:
            task = self.rerank_documents(
                query=request.query,
                documents=request.documents,
                strategy=request.strategy,
                top_k=request.top_k,
                request_id=request.request_id
            )
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error in batch reranking request {i}: {response}")
                # Create error response
                request = requests[i]
                final_responses.append(RerankingResponse(
                    reranked_documents=request.documents,
                    strategy_used=request.strategy,
                    processing_time=0.0,
                    original_count=len(request.documents),
                    reranked_count=len(request.documents),
                    request_id=request.request_id
                ))
            else:
                final_responses.append(response)
        
        return final_responses
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available reranking strategies"""
        available = []
        
        for strategy in RerankingStrategy:
            if strategy in self.rerankers:
                available.append(strategy.value)
        
        # Always include combined if we have multiple strategies
        if len(available) > 1:
            available.append(RerankingStrategy.COMBINED.value)
        
        return available
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            'requests': self.stats,
            'available_strategies': self.get_available_strategies(),
            'avg_processing_time': (
                self.stats['total_processing_time'] / 
                max(self.stats['total_requests'], 1)
            ),
            'avg_documents_per_request': (
                self.stats['total_documents_processed'] / 
                max(self.stats['total_requests'], 1)
            ),
            'strategy_distribution': self.stats['strategy_usage']
        }
    
    def benchmark_strategies(
        self, 
        query: str, 
        documents: List[DocumentChunk]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different reranking strategies on the same data
        """
        benchmark_results = {}
        
        for strategy in self.get_available_strategies():
            if strategy == RerankingStrategy.COMBINED.value:
                continue  # Skip combined for benchmarking
            
            start_time = time.time()
            
            try:
                # This would need to be made synchronous for benchmarking
                # or you could use asyncio.run() in a separate context
                strategy_enum = RerankingStrategy(strategy)
                reranker = self.rerankers.get(strategy_enum)
                
                if reranker:
                    # For benchmarking, we'll use a simplified approach
                    processing_time = time.time() - start_time
                    
                    benchmark_results[strategy] = {
                        'processing_time': processing_time,
                        'documents_processed': len(documents),
                        'throughput': len(documents) / max(processing_time, 0.001),
                        'status': 'success'
                    }
                else:
                    benchmark_results[strategy] = {
                        'status': 'not_available'
                    }
                    
            except Exception as e:
                benchmark_results[strategy] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return benchmark_results
    
    async def validate_reranking_quality(
        self,
        query: str,
        original_docs: List[DocumentChunk],
        reranked_docs: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Validate the quality of reranking results
        """
        quality_metrics = {}
        
        if not original_docs or not reranked_docs:
            return {'error': 'Empty document lists'}
        
        # Metric 1: Score improvement
        original_avg_score = statistics.mean([
            doc.similarity_score for doc in original_docs[:len(reranked_docs)]
        ])
        reranked_avg_score = statistics.mean([
            doc.rerank_score for doc in reranked_docs
        ])
        
        quality_metrics['score_improvement'] = reranked_avg_score - original_avg_score
        
        # Metric 2: Rank correlation (Spearman)
        original_ranks = {doc.id: i for i, doc in enumerate(original_docs)}
        reranked_ranks = {doc.id: i for i, doc in enumerate(reranked_docs)}
        
        # Calculate rank changes
        rank_changes = []
        for doc in reranked_docs:
            if doc.id in original_ranks:
                change = original_ranks[doc.id] - reranked_ranks[doc.id]
                rank_changes.append(abs(change))
        
        quality_metrics['avg_rank_change'] = statistics.mean(rank_changes) if rank_changes else 0
        quality_metrics['max_rank_change'] = max(rank_changes) if rank_changes else 0
        
        # Metric 3: Top-k stability
        top_3_original = set(doc.id for doc in original_docs[:3])
        top_3_reranked = set(doc.id for doc in reranked_docs[:3])
        top_3_overlap = len(top_3_original & top_3_reranked) / 3
        
        quality_metrics['top_3_stability'] = top_3_overlap
        
        # Metric 4: Score distribution
        original_score_std = statistics.stdev([doc.similarity_score for doc in original_docs[:len(reranked_docs)]])
        reranked_score_std = statistics.stdev([doc.rerank_score for doc in reranked_docs])
        
        quality_metrics['score_distribution_change'] = reranked_score_std - original_score_std
        
        return quality_metrics
    
    async def adaptive_reranking(
        self,
        query: str,
        documents: List[DocumentChunk],
        quality_threshold: float = 0.8
    ) -> RerankingResponse:
        """
        Adaptive reranking that chooses the best strategy based on query characteristics
        """
        # Analyze query characteristics
        query_length = len(query.split())
        has_technical_terms = any(term in query.lower() for term in [
            'algorithm', 'implementation', 'technical', 'method', 'approach'
        ])
        
        # Choose strategy based on characteristics
        if query_length <= 3 and not has_technical_terms:
            # Short, simple queries - use fast custom reranking
            strategy = RerankingStrategy.CUSTOM
        elif has_technical_terms or query_length > 10:
            # Complex queries - use cross-encoder for better accuracy
            strategy = RerankingStrategy.CROSS_ENCODER
        else:
            # Default to combined for balanced performance
            strategy = RerankingStrategy.COMBINED
        
        # Perform reranking
        response = await self.rerank_documents(
            query=query,
            documents=documents,
            strategy=strategy,
            top_k=len(documents)
        )
        
        # Validate quality and retry if needed
        if len(response.reranked_documents) > 3:
            quality = await self.validate_reranking_quality(
                query, documents, response.reranked_documents
            )
            
            score_improvement = quality.get('score_improvement', 0)
            
            # If quality is poor, try a different strategy
            if score_improvement < quality_threshold and strategy != RerankingStrategy.CROSS_ENCODER:
                logger.info(f"Reranking quality below threshold, retrying with cross-encoder")
                response = await self.rerank_documents(
                    query=query,
                    documents=documents,
                    strategy=RerankingStrategy.CROSS_ENCODER,
                    top_k=len(documents)
                )
        
        return response


# Singleton instance
_reranking_service: Optional[RerankingService] = None


def get_reranking_service(config: RAGConfig = None) -> RerankingService:
    """Get singleton reranking service instance"""
    global _reranking_service
    
    if _reranking_service is None:
        if config is None:
            raise ValueError("Config required for first initialization")
        _reranking_service = RerankingService(config)
    
    return _reranking_service


def cleanup_reranking_service():
    """Cleanup reranking service on shutdown"""
    global _reranking_service
    _reranking_service = None