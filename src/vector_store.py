"""
Vector Database Module

Manages FAISS vector store using LangChain for embeddings with chunk mappings.
"""

import os
import pickle
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to deprecated location
    from langchain_community.embeddings import HuggingFaceEmbeddings
from embeddings import EmbeddingGenerator
import faiss


class VectorStore:
    """
    LangChain FAISS-based vector store with chunk mappings for backward compatibility.
    """
    
    def __init__(self, embedding_dim: int = 384, embedding_model: Optional[EmbeddingGenerator] = None):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default for all-MiniLM-L6-v2)
            embedding_model: EmbeddingGenerator instance (optional, will create LangChain compatible if None)
        """
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        
        # Create LangChain-compatible embeddings wrapper
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore: Optional[FAISS] = None
        self.chunk_mappings: List[Dict[str, Any]] = []  # Maps index to chunk metadata for backward compat
        self.section_index: Dict[str, List[int]] = {}  # section_name -> list of chunk indices
        self.source_type: Optional[str] = None
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Add embeddings and their chunk mappings to the store.
        
        Args:
            embeddings: Numpy array of embeddings (n, embedding_dim) - ignored, will compute from chunks
            chunks: List of chunk dictionaries corresponding to embeddings
        """
        # Convert chunks to LangChain Documents
        documents = []
        for i, chunk in enumerate(chunks):
            # Create LangChain Document
            metadata = chunk.get('metadata', {}).copy()
            # Ensure chunk_index is in metadata
            metadata['chunk_index'] = chunk.get('chunk_index', i)
            metadata['k_number'] = metadata.get('k_number', 'unknown')
            
            doc = Document(
                page_content=chunk.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)
            
            # Maintain backward compatibility mappings
            self.chunk_mappings.append({
                'chunk_index': i,
                'chunk_data': chunk,
                'chunk_id': metadata.get('k_number', 'unknown') + f"_chunk_{chunk.get('chunk_index', i)}"
            })
            
            # Index by section header if present
            section_header = metadata.get('section_header')
            if section_header:
                if section_header not in self.section_index:
                    self.section_index[section_header] = []
                self.section_index[section_header].append(i)
        
        # Add to LangChain FAISS vectorstore
        if self.vectorstore is None:
            # Create new vectorstore
            self.vectorstore = FAISS.from_documents(documents, self.langchain_embeddings)
        else:
            # Add to existing vectorstore
            self.vectorstore.add_documents(documents)
        
        print(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunk_mappings)} chunks.")
    
    def add_sub_summary_embeddings(self, embeddings: np.ndarray, sub_summaries: List[Dict[str, Any]], linked_chunks: List[Dict[str, Any]]):
        """
        Add sub-summary embeddings with links to full chunks.
        
        Args:
            embeddings: Numpy array of sub-summary embeddings (ignored, computed from text)
            sub_summaries: List of sub-summary dictionaries
            linked_chunks: List of corresponding full chunk dictionaries
        """
        if len(sub_summaries) != len(linked_chunks):
            raise ValueError("Number of sub-summaries and linked chunks must match")
        
        # Convert to LangChain Documents with special metadata
        documents = []
        start_idx = len(self.chunk_mappings)
        
        for i, (sub_summary, linked_chunk) in enumerate(zip(sub_summaries, linked_chunks)):
            metadata = linked_chunk.get('metadata', {}).copy()
            metadata['chunk_index'] = linked_chunk.get('chunk_index', start_idx + i)
            metadata['is_sub_summary'] = True
            metadata['sub_summary_text'] = sub_summary.get('text', '')
            
            doc = Document(
                page_content=sub_summary.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)
            
            # Maintain backward compatibility mappings
            self.chunk_mappings.append({
                'chunk_index': start_idx + i,
                'chunk_data': linked_chunk,  # Store the full chunk data
                'sub_summary': sub_summary,  # Store the sub-summary text
                'chunk_id': linked_chunk.get('metadata', {}).get('k_number', 'unknown') + f"_chunk_{linked_chunk.get('chunk_index', start_idx + i)}",
                'is_sub_summary': True
            })
        
        # Add to vectorstore
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.langchain_embeddings)
        else:
            self.vectorstore.add_documents(documents)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector (converted to text search)
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.vectorstore is None or len(self.chunk_mappings) == 0:
            return []
        
        # For backward compatibility, we need to search using text query
        # We'll use a dummy query and filter results by embedding similarity
        # This is a limitation - we'd need the original query text
        # For now, we'll search all and return top k
        # In practice, callers should use search_by_query_text instead
        
        # This is a fallback - ideally use search_by_query_text
        docs_with_scores = self.vectorstore.similarity_search_with_score("", k=k*2)
        
        results = []
        for doc, score in docs_with_scores:
            # Find corresponding chunk_data
            metadata = doc.metadata
            chunk_idx = metadata.get('chunk_index')
            if chunk_idx is not None and chunk_idx < len(self.chunk_mappings):
                chunk_mapping = self.chunk_mappings[chunk_idx]
                chunk_data = chunk_mapping['chunk_data']
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = max(0.0, 1.0 - float(score))
                results.append((chunk_data, similarity))
        
        return results[:k]
    
    def search_by_query_text(self, query_text: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using query text (preferred method).
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.vectorstore is None or len(self.chunk_mappings) == 0:
            return []
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(query_text, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            chunk_idx = metadata.get('chunk_index')
            if chunk_idx is not None and chunk_idx < len(self.chunk_mappings):
                chunk_mapping = self.chunk_mappings[chunk_idx]
                chunk_data = chunk_mapping['chunk_data']
                # Convert distance to similarity (lower distance = higher similarity)
                # LangChain FAISS uses L2 distance, normalize to [0, 1]
                similarity = max(0.0, 1.0 / (1.0 + float(score)))
                results.append((chunk_data, similarity))
        
        return results
    
    def save(self, filepath: str):
        """
        Save vector store to disk.
        
        Args:
            filepath: Path to save the index (without extension)
        """
        if self.vectorstore is None:
            raise ValueError("Cannot save empty vector store")
        
        # Save LangChain FAISS vectorstore
        self.vectorstore.save_local(filepath)
        
        # Save mappings and metadata separately
        with open(f"{filepath}.mappings", 'wb') as f:
            pickle.dump(self.chunk_mappings, f)
        
        metadata = {
            'embedding_dim': self.embedding_dim,
            'total_vectors': len(self.chunk_mappings),
            'section_index': self.section_index,
            'source_type': getattr(self, 'source_type', None)
        }
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load vector store from disk.
        Supports both old format (.index, .mappings, .meta) and new LangChain format.
        
        Args:
            filepath: Path to load the index from (directory containing FAISS index)
        """
        old_format_index = f"{filepath}.index"
        new_format_dir = filepath
        
        # Check if old format exists - prioritize old format over new empty format
        if os.path.exists(old_format_index):
            print(f"Detected old format index at {old_format_index}, loading and converting...")
            self._load_old_format(filepath)
        elif os.path.exists(new_format_dir) and os.path.isdir(new_format_dir):
            # Try loading new LangChain format
            try:
                self.vectorstore = FAISS.load_local(filepath, self.langchain_embeddings, allow_dangerous_deserialization=True)
                
                # Load mappings
                mappings_path = f"{filepath}.mappings"
                if os.path.exists(mappings_path):
                    with open(mappings_path, 'rb') as f:
                        self.chunk_mappings = pickle.load(f)
                
                # Load metadata
                meta_path = f"{filepath}.meta"
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        metadata = pickle.load(f)
                        self.embedding_dim = metadata.get('embedding_dim', 384)
                        self.section_index = metadata.get('section_index', {})
                        self.source_type = metadata.get('source_type', None)
                
                print(f"Vector store loaded from {filepath} ({len(self.chunk_mappings)} chunks, {len(self.section_index)} sections)")
            except Exception as e:
                print(f"Error loading new format, trying old format: {e}")
                self._load_old_format(filepath)
        else:
            raise FileNotFoundError(f"Index not found at {filepath} or {old_format_index}")
    
    def _load_old_format(self, filepath: str):
        """Load old format index (.index, .mappings, .meta files)."""
        try:
            import faiss
        except ImportError as e:
            print(f"Error: faiss not available for loading old format: {e}")
            raise
        
        # Load old FAISS index
        old_index_path = f"{filepath}.index"
        old_index = faiss.read_index(old_index_path)
        
        # Load mappings
        mappings_path = f"{filepath}.mappings"
        if os.path.exists(mappings_path):
            with open(mappings_path, 'rb') as f:
                self.chunk_mappings = pickle.load(f)
        
        # Load metadata
        meta_path = f"{filepath}.meta"
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                self.embedding_dim = metadata.get('embedding_dim', 384)
                self.section_index = metadata.get('section_index', {})
                self.source_type = metadata.get('source_type', None)
        
        # Convert old format to LangChain format
        # Extract documents from chunk_mappings
        documents = []
        for mapping in self.chunk_mappings:
            chunk_data = mapping.get('chunk_data', {})
            metadata_dict = chunk_data.get('metadata', {}).copy()
            metadata_dict['chunk_index'] = mapping.get('chunk_index', len(documents))
            
            doc = Document(
                page_content=chunk_data.get('text', ''),
                metadata=metadata_dict
            )
            documents.append(doc)
        
        # Create new LangChain vectorstore from documents
        if documents:
            print("Converting old format to LangChain format...")
            self.vectorstore = FAISS.from_documents(documents, self.langchain_embeddings)
            print(f"Vector store loaded and converted from {filepath} ({len(self.chunk_mappings)} chunks, {len(self.section_index)} sections)")
        else:
            raise ValueError("No documents found in old format index")
    
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
        Note: This uses query_text parameter for LangChain search.
        
        Args:
            query_embedding: Query embedding vector (will need query text, using empty string as fallback)
            section_names: List of section names to search in
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.vectorstore is None or len(self.chunk_mappings) == 0:
            return []
        
        # Collect chunk indices from specified sections
        section_chunk_indices = set()
        for section_name in section_names:
            section_chunk_indices.update(self.section_index.get(section_name, []))
        
        if not section_chunk_indices:
            return []
        
        # Search and filter by section
        # Use a generic query since we only have embeddings
        # In practice, callers should pass query_text
        docs_with_scores = self.vectorstore.similarity_search_with_score("", k=k*3)
        
        results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            chunk_idx = metadata.get('chunk_index')
            if chunk_idx in section_chunk_indices and chunk_idx < len(self.chunk_mappings):
                chunk_mapping = self.chunk_mappings[chunk_idx]
                chunk_data = chunk_mapping['chunk_data']
                similarity = max(0.0, 1.0 / (1.0 + float(score)))
                results.append((chunk_data, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def search_by_sections_text(self, query_text: str, section_names: List[str], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for chunks within specific sections using query text.
        
        Args:
            query_text: Query text
            section_names: List of section names to search in
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if self.vectorstore is None or len(self.chunk_mappings) == 0:
            return []
        
        # Collect chunk indices from specified sections
        section_chunk_indices = set()
        for section_name in section_names:
            section_chunk_indices.update(self.section_index.get(section_name, []))
        
        if not section_chunk_indices:
            return []
        
        # Search and filter by section
        docs_with_scores = self.vectorstore.similarity_search_with_score(query_text, k=k*3)
        
        results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            chunk_idx = metadata.get('chunk_index')
            if chunk_idx in section_chunk_indices and chunk_idx < len(self.chunk_mappings):
                chunk_mapping = self.chunk_mappings[chunk_idx]
                chunk_data = chunk_mapping['chunk_data']
                similarity = max(0.0, 1.0 / (1.0 + float(score)))
                results.append((chunk_data, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        total_vectors = len(self.chunk_mappings) if self.chunk_mappings else 0
        if self.vectorstore:
            # Try to get count from vectorstore
            try:
                # FAISS stores index size
                if hasattr(self.vectorstore, 'index'):
                    total_vectors = self.vectorstore.index.ntotal
            except:
                pass
        
        return {
            'total_vectors': total_vectors,
            'embedding_dim': self.embedding_dim,
            'is_normalized': True,  # LangChain embeddings are normalized
            'total_chunks': len(self.chunk_mappings),
            'total_sections': len(self.section_index)
        }

    def get_vectorstore(self) -> Optional[FAISS]:
        """
        Get the underlying LangChain FAISS vectorstore.
        
        Returns:
            LangChain FAISS vectorstore instance
        """
        return self.vectorstore
