"""
RAG Retrieval Module

Implements retrieval logic using LangChain retrievers for finding relevant documents.
Supports section-based retrieval for enhanced precision.
"""

from typing import List, Dict, Tuple, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class RAGRetriever:
    """
    Retrieval component for RAG system with section-based filtering using LangChain.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance (for backward compat)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
        # Get LangChain vectorstore
        self.langchain_vectorstore = vector_store.get_vectorstore()
        
        # Create LangChain retriever if vectorstore exists
        if self.langchain_vectorstore:
            self.retriever = self.langchain_vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
        else:
            self.retriever = None
    
    def find_relevant_sections(self, query: str, top_n: int = 3, min_similarity: float = 0.3) -> List[str]:
        """
        Find relevant sections for a query by embedding similarity and keyword matching.
        
        Args:
            query: User query
            top_n: Number of top sections to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of section names
        """
        available_sections = self.vector_store.get_sections()
        if not available_sections:
            return []
        
        query_lower = query.lower()
        
        # Explicit keyword-to-section mapping for better detection
        section_keywords = {
            'Performance/Data': ['performance', 'data', 'testing', 'bench', 'validation', 'verification', 'testing', 'test', 'tests'],
            'Testing': ['testing', 'test', 'tests', 'clinical', 'bench', 'protocol', 'validation', 'verification'],
            'Device Details': ['device', 'description', 'equivalence', 'predicate', 'substantial'],
            'Administrative Info': ['administrative', 'submission', 'filing', 'indication'],
            'Public Info': ['summary', 'public', 'statement', '510k summary']
        }
        
        # Embed query
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Embed each section name and compute similarity
        section_similarities = []
        for section_name in available_sections:
            # Check explicit keyword matches first
            keyword_boost = 0.0
            keywords = section_keywords.get(section_name, [])
            for keyword in keywords:
                if keyword in query_lower:
                    keyword_boost += 0.2  # Boost for each matching keyword
            
            section_embedding = self.embedding_generator.embed_text(section_name)
            # Compute cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, section_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
            )
            
            # Apply keyword boost
            similarity = min(similarity + keyword_boost, 1.0)
            
            if similarity >= min_similarity:
                section_similarities.append((section_name, float(similarity)))
        
        # Sort by similarity and return top N
        section_similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in section_similarities[:top_n]]
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k most relevant chunks for a query using LangChain retriever.
        
        Args:
            query: User query text
            k: Number of results to retrieve
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.retriever is None:
            return []
        
        # Update retriever search_kwargs
        if hasattr(self.retriever, 'search_kwargs'):
            self.retriever.search_kwargs["k"] = k
        
        # Retrieve documents - use vectorstore directly for better compatibility
        # LangChain retrievers have changed API, so we'll use vectorstore directly
        if self.langchain_vectorstore:
            docs_with_scores = self.langchain_vectorstore.similarity_search_with_score(query, k=k)
            # Extract documents and use scores directly
            results = []
            for doc, score in docs_with_scores:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                chunk_idx = metadata.get('chunk_index')
                
                if chunk_idx is not None and chunk_idx < len(self.vector_store.chunk_mappings):
                    chunk_mapping = self.vector_store.chunk_mappings[chunk_idx]
                    chunk_data = chunk_mapping['chunk_data']
                    # Convert distance to similarity
                    similarity = max(0.0, 1.0 / (1.0 + float(score)))
                    results.append((chunk_data, similarity))
            
            return results[:k]
        else:
            return []
    
    def retrieve_with_context(self, query: str, k: int = 5, min_similarity: float = 0.0, 
                              use_section_filtering: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with similarity filtering and optional section-based filtering.
        
        Args:
            query: User query text
            k: Number of results to retrieve
            min_similarity: Minimum similarity score threshold
            use_section_filtering: Whether to first filter by relevant sections
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Step 1: Find relevant sections if enabled
        relevant_sections = []
        if use_section_filtering:
            relevant_sections = self.find_relevant_sections(query, top_n=3, min_similarity=0.25)
        
        # Step 2: Retrieve from sections if found, otherwise full search
        if relevant_sections:
            # Use section-filtered search
            results = self.vector_store.search_by_sections_text(query, relevant_sections, k=k*2)
        else:
            # Use full search
            results = self.retrieve(query, k=k*2)
        
        # Step 3: Filter by minimum similarity and format
        filtered_results = []
        for chunk_data, similarity in results:
            if similarity >= min_similarity:
                chunk_with_score = {
                    **chunk_data,
                    'similarity_score': similarity
                }
                filtered_results.append(chunk_with_score)
        
        # Return top k results
        return filtered_results[:k]
