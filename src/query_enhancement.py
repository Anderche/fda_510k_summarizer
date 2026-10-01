"""
Query Enhancement Module

Implements query expansion and multi-query generation to improve RAG retrieval.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class QueryEnhancer:
    """
    Enhances queries using expansion and multi-query generation.
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore,
                 term_cache_path: Optional[str] = None):
        """
        Initialize query enhancer.
        
        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store to extract terms from
            term_cache_path: Optional .npz path to load/save term embeddings
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.term_cache_path = term_cache_path
        self._term_cache: Dict[str, np.ndarray] = {}
        self._corpus_terms: List[Tuple[str, np.ndarray]] = []
        self._cache_initialized = False
    
    def _load_term_cache(self) -> bool:
        """Load term embeddings from term_cache_path. Returns True on success."""
        if not self.term_cache_path or not os.path.exists(self.term_cache_path):
            return False
        try:
            data = np.load(self.term_cache_path, allow_pickle=False)
            terms_list = [str(t) for t in data['terms']]
            term_embeddings = data['embeddings']
        except Exception as e:
            print(f"Warning: Could not load term cache from {self.term_cache_path}: {e}")
            return False
        self._corpus_terms = list(zip(terms_list, term_embeddings))
        self._term_cache = dict(self._corpus_terms)
        print(f"  Loaded {len(self._corpus_terms)} cached terms from {self.term_cache_path}")
        return True
    
    def _save_term_cache(self, terms_list: List[str], term_embeddings: np.ndarray):
        """Save term embeddings to term_cache_path (best effort)."""
        if not self.term_cache_path:
            return
        try:
            np.savez(self.term_cache_path, terms=np.array(terms_list), embeddings=term_embeddings)
        except Exception as e:
            print(f"Warning: Could not save term cache to {self.term_cache_path}: {e}")
    
    def _initialize_term_cache(self, max_terms: int = 500):
        """
        Initialize cache of terms from corpus for query expansion.
        Extracts key phrases from chunks and embeds them.
        """
        if self._cache_initialized:
            return
        
        # Check if vectorstore is initialized
        if self.vector_store.vectorstore is None or len(self.vector_store.chunk_mappings) == 0:
            self._cache_initialized = True
            return
        
        if self._load_term_cache():
            self._cache_initialized = True
            return
        
        print("Initializing query expansion cache...")
        terms = set()
        
        # Extract terms/phrases from chunks
        num_chunks_to_scan = min(max_terms, len(self.vector_store.chunk_mappings))
        for mapping in self.vector_store.chunk_mappings[:num_chunks_to_scan]:
            chunk_data = mapping.get('chunk_data', {})
            text = chunk_data.get('text', '')
            
            # Extract meaningful phrases (2-4 words)
            phrases = self._extract_phrases(text)
            terms.update(phrases)
            
            # Also extract important single terms (camelCase, ALL_CAPS, or capitalized)
            important_words = self._extract_important_words(text)
            terms.update(important_words)
        
        # Embed all terms
        terms_list = list(terms)[:max_terms]
        if terms_list:
            term_embeddings = self.embedding_generator.embed_texts(terms_list, show_progress=False)
            self._corpus_terms = list(zip(terms_list, term_embeddings))
            for term, emb in zip(terms_list, term_embeddings):
                self._term_cache[term] = emb
            self._save_term_cache(terms_list, term_embeddings)
        
        self._cache_initialized = True
        print(f"  Cached {len(self._corpus_terms)} terms/phrases for expansion")
    
    def _extract_phrases(self, text: str, min_words: int = 2, max_words: int = 4) -> Set[str]:
        """
        Extract meaningful phrases from text.
        """
        # Simple approach: extract n-grams
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        phrases = set()
        
        for n in range(min_words, max_words + 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                phrases.add(phrase)
        
        return phrases
    
    def _extract_important_words(self, text: str) -> Set[str]:
        """
        Extract important single terms (capitalized, camelCase, ALL_CAPS).
        """
        words = set()
        
        # CamelCase or ALL_CAPS
        camel_case = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', text)
        words.update([w.lower() for w in camel_case])
        
        # Capitalized words (likely important nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        words.update([w.lower() for w in capitalized])
        
        return words
    
    def expand_query(self, query: str, top_n: int = 5, similarity_threshold: float = 0.6) -> str:
        """
        Expand query by finding semantically similar terms from corpus.
        
        Args:
            query: Original query
            top_n: Number of similar terms to add
            similarity_threshold: Minimum similarity to include term
            
        Returns:
            Expanded query with additional terms
        """
        self._initialize_term_cache()
        
        if not self._corpus_terms:
            return query  # No expansion possible
        
        # Embed query
        query_embedding = self.embedding_generator.embed_text(query)
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Find similar terms
        similarities = []
        for term, term_emb in self._corpus_terms:
            term_emb_norm = term_emb.astype('float32') / np.linalg.norm(term_emb)
            similarity = np.dot(query_embedding, term_emb_norm)
            if similarity >= similarity_threshold:
                similarities.append((term, float(similarity)))
        
        # Sort by similarity and get top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_terms = [term for term, _ in similarities[:top_n]]
        
        if top_terms:
            expanded_query = f"{query} {' '.join(top_terms)}"
            return expanded_query
        
        return query
    
    def generate_query_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate query variants using LLM.
        
        Args:
            query: Original query
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants including original
        """
        # For now, return simple variations if no LLM
        # This will be enhanced in query_pipeline with actual LLM
        variants = [query]
        
        # Simple rule-based variants
        query_lower = query.lower()
        
        # Add variations for common patterns
        if "what" in query_lower:
            variants.append(query.replace("what", "which", 1))
            variants.append(query.replace("what", "describe", 1))
        
        if "how" in query_lower:
            variants.append(query.replace("how", "what methods", 1))
        
        # Add question variations
        if "?" not in query:
            variants.append(query + "?")
        
        return variants[:num_variants]
    
    def generate_llm_query_variants(self, query: str, llm_generator, num_variants: int = 3) -> List[str]:
        """
        Generate query variants using LLM.
        
        Args:
            query: Original query
            llm_generator: LLM generator instance
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants including original
        """
        variants = [query]
        
        try:
            prompt = f"""Generate {num_variants} alternative phrasings of this question that would help retrieve relevant information from FDA 510(k) documents:

Original question: {query}

Generate {num_variants} different ways to ask the same question. Focus on:
- Using different terminology (synonyms, related terms)
- Different question structures
- Different emphasis

Return only the {num_variants} questions, one per line, no numbering."""
            
            # Try to use LLM if available
            if hasattr(llm_generator, 'client'):
                try:
                    message = llm_generator.client.messages.create(
                        model=llm_generator.model_name,
                        max_tokens=200,
                        temperature=0.8,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    if message.content and len(message.content) > 0:
                        response_text = message.content[0].text.strip()
                        # Parse response to extract questions
                        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                        # Filter out non-question lines
                        generated = [q for q in lines if ('?' in q or len(q.split()) > 5)][:num_variants]
                        variants.extend(generated)
                except Exception as e:
                    # Fallback to simple variants
                    pass
        except Exception:
            pass
        
        return variants[:num_variants + 1]  # Include original + variants

