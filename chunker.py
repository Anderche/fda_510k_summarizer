"""
Document Chunking Module

Splits documents into semantically meaningful chunks using improved sentence-aware chunking.
Maintains context and metadata for each chunk.
"""

import tiktoken
import nltk
import re
from typing import List, Dict, Optional, Any

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class DocumentChunker:
    """
    Splits documents into chunks with improved sentence-aware chunking.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Model name for tokenizer (used for token counting)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK for better accuracy.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to regex-based splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences (likely noise)
        filtered = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return filtered
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using sentence-aware chunking.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with 'text', 'metadata', 'chunk_index', and 'token_count'
        """
        metadata = metadata or {}
        
        # First split into paragraphs to maintain document structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            # Split paragraph into sentences
            sentences = self.split_sentences(para)
            
            if not sentences:
                continue
            
            for sentence in sentences:
                sent_tokens = self.count_tokens(sentence)
                
                # If single sentence exceeds chunk size, add current chunk and split sentence
                if sent_tokens > self.chunk_size:
                    # Save current chunk if it has content
                    if current_chunk_sentences:
                        chunk_text = ' '.join(current_chunk_sentences)
                        chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                        chunk_index += 1
                        current_chunk_sentences = []
                        current_length = 0
                    
                    # Split large sentence into smaller parts (by words)
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0
                    
                    for word in words:
                        word_tokens = self.count_tokens(word + ' ')
                        if temp_length + word_tokens > self.chunk_size and temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                            chunk_index += 1
                            temp_chunk = []
                            temp_length = 0
                        temp_chunk.append(word)
                        temp_length += word_tokens
                    
                    if temp_chunk:
                        current_chunk_sentences = temp_chunk
                        current_length = temp_length
                
                # Check if adding this sentence would exceed chunk size
                elif current_length + sent_tokens > self.chunk_size and current_chunk_sentences:
                    # Create chunk from current sentences
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                    chunk_index += 1
                    
                    # Handle overlap: keep last few sentences for context
                    if self.chunk_overlap > 0:
                        overlap_sentences = self._get_overlap_sentences(current_chunk_sentences, self.chunk_overlap)
                        current_chunk_sentences = overlap_sentences
                        current_length = sum(self.count_tokens(s) for s in overlap_sentences)
                    else:
                        current_chunk_sentences = []
                        current_length = 0
                    
                    current_chunk_sentences.append(sentence)
                    current_length += sent_tokens
                
                else:
                    # Add sentence to current chunk
                    current_chunk_sentences.append(sentence)
                    current_length += sent_tokens
        
        # Add remaining chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
        
        # Filter out chunks that are too short (likely noise)
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk['token_count'] >= 10 and len(chunk['text'].strip()) >= 50
        ]
        
        return filtered_chunks
    
    def _create_chunk(self, text: str, metadata: Dict, chunk_index: int) -> Dict[str, Any]:
        """Create a chunk dictionary."""
        return {
            'text': text.strip(),
            'metadata': {**metadata, 'chunk_index': chunk_index},
            'chunk_index': chunk_index,
            'token_count': self.count_tokens(text)
        }
    
    def _get_overlap_sentences(self, sentences: List[str], target_tokens: int) -> List[str]:
        """Get last N sentences that fit within target token count."""
        if not sentences:
            return []
        
        overlap = []
        overlap_length = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sent_tokens = self.count_tokens(sentence)
            if overlap_length + sent_tokens <= target_tokens:
                overlap.insert(0, sentence)
                overlap_length += sent_tokens
            else:
                break
        
        return overlap if overlap else [sentences[-1]]  # Return at least last sentence
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of documents with 'k_number', 'file_path', and 'text' keys
            
        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        
        for doc in documents:
            metadata = {
                'k_number': doc.get('k_number'),
                'file_path': doc.get('file_path'),
                'source_document': doc.get('k_number')
            }
            
            chunks = self.chunk_text(doc['text'], metadata)
            
            # Filter chunks that don't contain meaningful content
            # (e.g., mostly headers, numbers, or very repetitive)
            for chunk in chunks:
                text = chunk['text']
                
                # Skip if mostly numbers or special characters
                alpha_chars = len(re.findall(r'[A-Za-z]', text))
                if alpha_chars < len(text) * 0.4:
                    continue
                
                # Skip if very repetitive (likely formatting artifacts)
                words = text.split()
                if len(words) > 10:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.3:  # More than 70% repetitive
                        continue
                
                all_chunks.append(chunk)
        
        return all_chunks
