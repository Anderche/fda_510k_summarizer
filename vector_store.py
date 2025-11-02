"""
Vector Database Module

Manages FAISS vector store for embeddings with chunk mappings.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Any


class VectorStore:
    """
    FAISS-based vector store with chunk mappings.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        # FAISS index - using L2 distance (inner product with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity (with normalized vectors)
        self.chunk_mappings: List[Dict[str, Any]] = []  # Maps index to chunk metadata
        self.is_normalized = False
        # Section index: maps section names to chunk indices
        self.section_index: Dict[str, List[int]] = {}  # section_name -> list of chunk indices
    
    def normalize_index(self):
        """Normalize all vectors in the index for cosine similarity."""
        if not self.is_normalized and self.index.ntotal > 0:
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            faiss.normalize_L2(vectors)
            self.index.reset()
            self.index.add(vectors)
            self.is_normalized = True
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Add embeddings and their chunk mappings to the store.
        
        Args:
            embeddings: Numpy array of embeddings (n, embedding_dim)
            chunks: List of chunk dictionaries corresponding to embeddings
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of chunks ({len(chunks)})")
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        # Reshape if needed
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        self.is_normalized = True
        
        # Add chunk mappings and build section index
        for i, chunk in enumerate(chunks):
            chunk_idx = start_idx + i
            self.chunk_mappings.append({
                'chunk_index': chunk_idx,
                'chunk_data': chunk,
                'chunk_id': chunk.get('metadata', {}).get('k_number', 'unknown') + f"_chunk_{chunk.get('chunk_index', i)}"
            })
            
            # Index by section header if present
            section_header = chunk.get('metadata', {}).get('section_header')
            if section_header:
                if section_header not in self.section_index:
                    self.section_index[section_header] = []
                self.section_index[section_header].append(chunk_idx)
    
    def add_sub_summary_embeddings(self, embeddings: np.ndarray, sub_summaries: List[Dict[str, Any]], linked_chunks: List[Dict[str, Any]]):
        """
        Add sub-summary embeddings with links to full chunks.
        
        Args:
            embeddings: Numpy array of sub-summary embeddings
            sub_summaries: List of sub-summary dictionaries
            linked_chunks: List of corresponding full chunk dictionaries
        """
        if len(embeddings) != len(sub_summaries) or len(sub_summaries) != len(linked_chunks):
            raise ValueError("Number of embeddings, sub-summaries, and linked chunks must match")
        
        # Normalize embeddings
        embeddings = embeddings.astype('float32')
        # Reshape if needed
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        self.is_normalized = True
        
        # Add mappings linking sub-summaries to full chunks
        for i, (sub_summary, linked_chunk) in enumerate(zip(sub_summaries, linked_chunks)):
            self.chunk_mappings.append({
                'chunk_index': start_idx + i,
                'chunk_data': linked_chunk,  # Store the full chunk data
                'sub_summary': sub_summary,  # Store the sub-summary text
                'chunk_id': linked_chunk.get('metadata', {}).get('k_number', 'unknown') + f"_chunk_{linked_chunk.get('chunk_index', i)}",
                'is_sub_summary': True
            })
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_mappings):
                chunk_mapping = self.chunk_mappings[idx]
                # Convert distance to similarity (for inner product, higher is better)
                similarity = float(dist)
                results.append((chunk_mapping['chunk_data'], similarity))
        
        return results
    
    def save(self, filepath: str):
        """
        Save vector store to disk.
        
        Args:
            filepath: Path to save the index (without extension)
        """
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save mappings
        with open(f"{filepath}.mappings", 'wb') as f:
            pickle.dump(self.chunk_mappings, f)
        
        # Save metadata (including section index)
        metadata = {
            'embedding_dim': self.embedding_dim,
            'is_normalized': self.is_normalized,
            'total_vectors': self.index.ntotal,
            'section_index': self.section_index
        }
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Vector store saved to {filepath}.*")
    
    def load(self, filepath: str):
        """
        Load vector store from disk.
        
        Args:
            filepath: Path to load the index from (without extension)
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load mappings
        with open(f"{filepath}.mappings", 'rb') as f:
            self.chunk_mappings = pickle.load(f)
        
        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            metadata = pickle.load(f)
            self.embedding_dim = metadata['embedding_dim']
            self.is_normalized = metadata['is_normalized']
            # Load section index if present (for backwards compatibility)
            self.section_index = metadata.get('section_index', {})
        
        print(f"Vector store loaded from {filepath}.* ({self.index.ntotal} vectors, {len(self.section_index)} sections)")
    
    def get_sections(self) -> List[str]:
        """
        Get list of all section headers in the index.
        
        Returns:
            List of section header names
        """
        return list(self.section_index.keys())
    
    def get_chunks_by_section(self, section_name: str) -> List[int]:
        """
        Get chunk indices for a specific section.
        
        Args:
            section_name: Name of the section
            
        Returns:
            List of chunk indices
        """
        return self.section_index.get(section_name, [])
    
    def search_by_sections(self, query_embedding: np.ndarray, section_names: List[str], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for chunks within specific sections only.
        
        Args:
            query_embedding: Query embedding vector
            section_names: List of section names to search in
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.index.ntotal == 0:
            return []
        
        # Collect all chunk indices from specified sections
        section_chunk_indices = set()
        for section_name in section_names:
            section_chunk_indices.update(self.section_index.get(section_name, []))
        
        if not section_chunk_indices:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Extract vectors only for relevant sections
        section_indices_list = sorted(list(section_chunk_indices))
        if not section_indices_list:
            return []
        
        # Reconstruct vectors for section chunks individually
        section_vectors = []
        for idx in section_indices_list:
            if 0 <= idx < self.index.ntotal:
                vec = self.index.reconstruct(int(idx))
                section_vectors.append(vec)
        
        if not section_vectors:
            return []
        
        section_vectors = np.array(section_vectors).astype('float32')
        faiss.normalize_L2(section_vectors)
        
        # Create temporary index for section chunks only
        temp_index = faiss.IndexFlatIP(self.embedding_dim)
        temp_index.add(section_vectors)
        
        # Search in section chunks only
        search_k = min(k, len(section_indices_list))
        distances, local_indices = temp_index.search(query_embedding, search_k)
        
        # Map back to original indices and chunk data
        results = []
        for dist, local_idx in zip(distances[0], local_indices[0]):
            if 0 <= local_idx < len(section_indices_list):
                original_idx = section_indices_list[local_idx]
                if original_idx < len(self.chunk_mappings):
                    chunk_mapping = self.chunk_mappings[original_idx]
                    similarity = float(dist)
                    results.append((chunk_mapping['chunk_data'], similarity))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'is_normalized': self.is_normalized,
            'total_chunks': len(self.chunk_mappings),
            'total_sections': len(self.section_index)
        }

