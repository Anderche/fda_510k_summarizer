"""
Query Pipeline Module

End-to-end query processing from user input to response generation.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_retrieval import RAGRetriever
from llm_integration import LLMGenerator, SimpleLLMGenerator
from reference_formatter import format_multiple_references
from query_enhancement import QueryEnhancer


class QueryPipeline:
    """
    Complete query processing pipeline.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator, 
                 llm_generator: Optional[LLMGenerator] = None, product_code: Optional[str] = None,
                 use_query_expansion: bool = True, use_multi_query: bool = True):
        """
        Initialize query pipeline.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            llm_generator: LLM generator instance (optional, will use SimpleLLMGenerator if None)
            product_code: Product code for context in responses
            use_query_expansion: Whether to use query expansion
            use_multi_query: Whether to use multi-query generation
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_generator = llm_generator or SimpleLLMGenerator()
        self.product_code = product_code
        self.retriever = RAGRetriever(vector_store, embedding_generator)
        self.use_query_expansion = use_query_expansion
        self.use_multi_query = use_multi_query
        self.query_enhancer = QueryEnhancer(embedding_generator, vector_store)
    
    def _merge_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from multiple queries.
        
        Args:
            all_results: List of result dictionaries with 'retrieved_chunks'
            
        Returns:
            Merged and deduplicated chunks
        """
        seen_chunk_ids: Set[str] = set()
        merged_chunks: List[Dict[str, Any]] = []
        chunk_scores: Dict[str, float] = {}
        
        for result in all_results:
            for chunk in result.get('retrieved_chunks', []):
                # Create unique ID from chunk metadata
                metadata = chunk.get('metadata', {})
                chunk_id = f"{metadata.get('k_number', 'unknown')}_{chunk.get('chunk_index', 'unknown')}"
                
                similarity = chunk.get('similarity_score', 0.0)
                
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_chunks.append(chunk)
                    chunk_scores[chunk_id] = similarity
                else:
                    # Take max similarity if duplicate
                    chunk_scores[chunk_id] = max(chunk_scores[chunk_id], similarity)
        
        # Sort by similarity score (using the max score for each chunk)
        merged_chunks.sort(key=lambda c: chunk_scores.get(
            f"{c.get('metadata', {}).get('k_number', 'unknown')}_{c.get('chunk_index', 'unknown')}", 
            0.0
        ), reverse=True)
        
        return merged_chunks
    
    def process_query(self, query: str, k: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """
        Process a user query and generate a response with query enhancement.
        
        Args:
            query: User's query/question
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with 'query', 'retrieved_chunks', 'response', and 'metadata'
        """
        all_retrieved = []
        queries_used = [query]
        
        # Step 1: Query expansion
        expanded_query = query
        if self.use_query_expansion:
            expanded_query = self.query_enhancer.expand_query(query, top_n=5, similarity_threshold=0.6)
            if expanded_query != query:
                queries_used.append(f"Expanded: {expanded_query}")
        
        # Step 2: Multi-query generation
        query_variants = [expanded_query]
        if self.use_multi_query:
            if hasattr(self.llm_generator, 'client'):
                variants = self.query_enhancer.generate_llm_query_variants(expanded_query, self.llm_generator, num_variants=3)
            else:
                variants = self.query_enhancer.generate_query_variants(expanded_query, num_variants=3)
            query_variants.extend([v for v in variants if v != expanded_query])
            queries_used.extend([f"Variant: {v}" for v in variants if v != expanded_query])
        
        # Step 3: Retrieve for each query variant
        for q in query_variants:
            retrieved = self.retriever.retrieve_with_context(q, k=k*2, min_similarity=min_similarity)
            if retrieved:
                all_retrieved.append({'retrieved_chunks': retrieved})
        
        # Step 4: Merge and deduplicate results
        merged_chunks = self._merge_results(all_retrieved)
        retrieved = merged_chunks[:k]  # Take top k after merging
        
        if not retrieved:
            return {
                'query': query,
                'retrieved_chunks': [],
                'references': [],
                'response': "No relevant documents found for your query.",
                'metadata': {
                    'num_retrieved': 0,
                    'k': k,
                    'queries_used': queries_used
                }
            }
        
        # Step 5: Fetch linked full chunks as context
        context_chunks = [chunk for chunk in retrieved]
        
        # Step 6: Augment LLM prompt with context and generate response
        response = self.llm_generator.generate_response(
            query,
            context_chunks,
            product_code=self.product_code
        )
        
        # Format references for each retrieved chunk
        references = format_multiple_references(retrieved, format_type="dict")
        
        return {
            'query': query,
            'retrieved_chunks': retrieved,
            'references': references,
            'response': response,
            'metadata': {
                'num_retrieved': len(retrieved),
                'k': k,
                'similarity_scores': [chunk.get('similarity_score', 0.0) for chunk in retrieved],
                'queries_used': queries_used,
                'expansion_used': self.use_query_expansion,
                'multi_query_used': self.use_multi_query
            }
        }
    
    def set_product_code(self, product_code: str):
        """Update product code for context."""
        self.product_code = product_code

