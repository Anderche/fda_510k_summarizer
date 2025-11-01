"""
Query Pipeline Module

End-to-end query processing from user input to response generation.
"""

from typing import Dict, List, Optional, Any
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_retrieval import RAGRetriever
from llm_integration import LLMGenerator, SimpleLLMGenerator


class QueryPipeline:
    """
    Complete query processing pipeline.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator, 
                 llm_generator: Optional[LLMGenerator] = None, product_code: Optional[str] = None):
        """
        Initialize query pipeline.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            llm_generator: LLM generator instance (optional, will use SimpleLLMGenerator if None)
            product_code: Product code for context in responses
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_generator = llm_generator or SimpleLLMGenerator()
        self.product_code = product_code
        self.retriever = RAGRetriever(vector_store, embedding_generator)
    
    def process_query(self, query: str, k: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User's query/question
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with 'query', 'retrieved_chunks', 'response', and 'metadata'
        """
        # Step 1: Embed query
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Step 2: Retrieve top-k summary vectors
        retrieved = self.retriever.retrieve_with_context(query, k=k, min_similarity=min_similarity)
        
        if not retrieved:
            return {
                'query': query,
                'retrieved_chunks': [],
                'response': "No relevant documents found for your query.",
                'metadata': {
                    'num_retrieved': 0,
                    'k': k
                }
            }
        
        # Step 3: Fetch linked full chunks as context
        context_chunks = [chunk for chunk in retrieved]
        
        # Step 4: Augment LLM prompt with context and generate response
        response = self.llm_generator.generate_response(
            query,
            context_chunks,
            product_code=self.product_code
        )
        
        return {
            'query': query,
            'retrieved_chunks': retrieved,
            'response': response,
            'metadata': {
                'num_retrieved': len(retrieved),
                'k': k,
                'similarity_scores': [chunk.get('similarity_score', 0.0) for chunk in retrieved]
            }
        }
    
    def set_product_code(self, product_code: str):
        """Update product code for context."""
        self.product_code = product_code

