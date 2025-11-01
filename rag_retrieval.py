"""
RAG Retrieval Module

Implements retrieval logic for finding relevant documents based on queries.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class RAGRetriever:
    """
    Retrieval component for RAG system.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User query text
            k: Number of results to retrieve
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        # Embed query
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
    def retrieve_with_context(self, query: str, k: int = 5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with similarity filtering and formatted output.
        
        Args:
            query: User query text
            k: Number of results to retrieve
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        results = self.retrieve(query, k=k)
        
        # Filter by minimum similarity and format
        filtered_results = []
        for chunk_data, similarity in results:
            if similarity >= min_similarity:
                chunk_with_score = {
                    **chunk_data,
                    'similarity_score': similarity
                }
                filtered_results.append(chunk_with_score)
        
        return filtered_results

