"""
LLM Integration Module

Handles LLM interactions for sub-summary generation and query responses.
Uses Hugging Face Transformers with local models (Llama-2 or similar).
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Optional, Any
import torch


class LLMGenerator:
    """
    LLM wrapper for text generation tasks.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: Optional[str] = None):
        """
        Initialize LLM generator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading LLM model: {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
            
            print(f"LLM model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load {model_name}. Error: {e}")
            print("Falling back to a simpler approach. For production, ensure model is available.")
            self.model = None
            self.tokenizer = None
            self.generator = None
    
    def generate_sub_summary(self, chunk_text: str, k_number: Optional[str] = None) -> str:
        """
        Generate a sub-summary for a chunk.
        
        Args:
            chunk_text: Text from the chunk
            k_number: Optional K-number for context
            
        Returns:
            Generated sub-summary text
        """
        if not self.generator:
            # Fallback: return a simple template-based summary
            return f"Key information from this 510(k) document: {chunk_text[:200]}..."
        
        prompt = f"""Summarize the key information from this FDA 510(k) document excerpt:

{chunk_text[:1000]}

Summary:"""
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_full_text=False
            )
            return result[0]['generated_text'].strip()
        except Exception as e:
            print(f"Error generating sub-summary: {e}")
            return f"Key information from this 510(k) document: {chunk_text[:200]}..."
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], product_code: Optional[str] = None) -> str:
        """
        Generate a response to a user query based on retrieved context.
        
        Args:
            query: User's query/question
            context_chunks: List of retrieved chunk dictionaries with 'text' key
            product_code: Optional product code for context
            
        Returns:
            Generated response text
        """
        if not self.generator:
            # Fallback: return a simple response
            context_text = "\n\n".join([chunk.get('text', '')[:500] for chunk in context_chunks[:3]])
            return f"Based on the retrieved 510(k) documents: {context_text[:500]}..."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):
            chunk_text = chunk.get('text', '')
            k_number = chunk.get('metadata', {}).get('k_number', 'Unknown')
            context_parts.append(f"Document {i} (K-number: {k_number}):\n{chunk_text[:800]}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on these FDA 510(k) submission summaries for product code {product_code or 'specified devices'}, answer the following question:

Question: {query}

Relevant 510(k) Summaries:
{context}

Answer:"""
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_full_text=False
            )
            return result[0]['generated_text'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            context_text = "\n\n".join([chunk.get('text', '')[:300] for chunk in context_chunks[:3]])
            return f"Based on the retrieved FDA 510(k) documents:\n\n{context_text}"


class SimpleLLMGenerator:
    """
    Lightweight fallback LLM generator that doesn't require a large model.
    Useful for development and testing when full LLM isn't available.
    """
    
    def generate_sub_summary(self, chunk_text: str, k_number: Optional[str] = None) -> str:
        """Generate a simple template-based sub-summary."""
        prefix = f"Key safety and effectiveness data from FDA 510(k) {k_number or 'document'}: " if k_number else "Key information from this 510(k) document: "
        # Extract first sentence or first 200 chars
        summary_text = chunk_text.split('.')[0][:200] if '.' in chunk_text else chunk_text[:200]
        return prefix + summary_text
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], product_code: Optional[str] = None) -> str:
        """Generate a simple response based on context."""
        response_parts = [f"Based on FDA 510(k) summaries for product code {product_code or 'the specified devices'}:"]
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            chunk_text = chunk.get('text', '')
            k_number = chunk.get('metadata', {}).get('k_number', 'Unknown')
            response_parts.append(f"\n\nRelevant information from K-number {k_number}:\n{chunk_text[:400]}")
        
        response_parts.append(f"\n\nRegarding your question '{query}', the above documents contain relevant information that may help answer it.")
        
        return "\n".join(response_parts)

