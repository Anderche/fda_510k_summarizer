"""
Embedding Generation Module

Generates vector embeddings for chunks and sub-summaries using sentence transformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np


class EmbeddingGenerator:
    """
    Generates embeddings using sentence transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Embedding model loaded successfully.")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        return self.embed_texts(texts)
    
    def embed_sub_summaries(self, sub_summaries: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for sub-summaries.
        
        Args:
            sub_summaries: List of sub-summary dictionaries with 'text' key
            
        Returns:
            Numpy array of embeddings
        """
        texts = [sub['text'] for sub in sub_summaries]
        return self.embed_texts(texts)

