"""
LLM Integration Module

Handles LLM interactions for sub-summary generation and query responses.
Uses Claude API via Anthropic SDK.
"""

import os
from typing import List, Dict, Optional, Any
from reference_formatter import format_reference_string

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")

try:
    from dotenv import load_dotenv
    import os
    # Load .env file from multiple locations
    # 1. Try script's directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    # 2. Try current working directory
    cwd_env = os.path.join(os.getcwd(), '.env')
    if os.path.exists(cwd_env):
        load_dotenv(dotenv_path=cwd_env, override=True)
    # 3. Try loading without path (dotenv searches automatically)
    load_dotenv(override=True)
except ImportError:
    pass  # dotenv not installed, skip loading .env
except Exception as e:
    print(f"Warning: Error loading .env file: {e}")


class LLMGenerator:
    """
    LLM wrapper using Claude API for text generation tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-sonnet-20240229"):
        """
        Initialize LLM generator with Claude API.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Claude model name (default: claude-3-5-sonnet-20241022)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        # Try to get API key from parameter, then env var
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            # Provide helpful debug info
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cwd = os.getcwd()
            env_locations = [
                os.path.join(script_dir, '.env'),
                os.path.join(cwd, '.env'),
                '.env'
            ]
            found_env = [loc for loc in env_locations if os.path.exists(loc)]
            
            error_msg = (
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "  - Set it in .env file (ANTHROPIC_API_KEY=your-key)\n"
                "  - Or export ANTHROPIC_API_KEY='your-key-here'\n"
            )
            if found_env:
                error_msg += f"  - Found .env files at: {', '.join(found_env)}\n"
            else:
                error_msg += f"  - Searched for .env in: {', '.join(env_locations)}\n"
            
            raise ValueError(error_msg)
        
        self.model_name = model_name
        self.client = Anthropic(api_key=self.api_key)
        
        print(f"Claude API initialized with model: {model_name}")
    
    def generate_sub_summary(self, chunk_text: str, k_number: Optional[str] = None) -> str:
        """
        Generate a sub-summary for a chunk.
        
        Args:
            chunk_text: Text from the chunk
            k_number: Optional K-number for context
            
        Returns:
            Generated sub-summary text
        """
        # Truncate chunk if too long (Claude has context limits)
        max_chunk_length = 8000  # Leave room for prompt
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
        
        system_prompt = "You are a helpful assistant that summarizes FDA 510(k) medical device submission documents."
        
        user_prompt = f"""Summarize the key information from this FDA 510(k) document excerpt{f' (K-number: {k_number})' if k_number else ''}:

{chunk_text}

Provide a concise summary (2-3 sentences) highlighting the most important information:"""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=200,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract text from response
            if message.content and len(message.content) > 0:
                return message.content[0].text.strip()
            else:
                return f"Key information from this 510(k) document: {chunk_text[:200]}..."
        
        except Exception as e:
            error_str = str(e)
            # Check for specific error types
            if '404' in error_str or 'not_found' in error_str.lower():
                raise ValueError(f"Model '{self.model_name}' not found. Available models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307. Try: --llm-model claude-3-sonnet-20240229")
            elif '401' in error_str or 'unauthorized' in error_str.lower():
                raise ValueError("API key invalid or missing. Check your ANTHROPIC_API_KEY.")
            else:
                raise e
    
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
        # Build context from chunks with references
        context_parts = []
        total_length = 0
        max_context_length = 200000  # Claude 3.5 Sonnet has 200k context
        
        for i, chunk in enumerate(context_chunks[:10], 1):  # Limit to top 10 chunks
            chunk_text = chunk.get('text', '')
            k_number = chunk.get('metadata', {}).get('k_number', 'Unknown')
            
            # Format reference
            ref_str = format_reference_string(chunk, include_link=False)
            
            chunk_entry = f"Document {i} ({ref_str}):\n{chunk_text}"
            
            # Check if adding this chunk would exceed context limit
            if total_length + len(chunk_entry) > max_context_length:
                # Truncate this chunk to fit
                remaining = max_context_length - total_length - 500  # Safety margin
                chunk_entry = f"Document {i} ({ref_str}):\n{chunk_text[:remaining]}..."
            
            context_parts.append(chunk_entry)
            total_length += len(chunk_entry)
            
            if total_length >= max_context_length:
                break
        
        context = "\n\n".join(context_parts)
        
        system_prompt = f"""You are an expert assistant helping users understand FDA 510(k) medical device submissions. 
You answer questions based on retrieved 510(k) summary documents. Provide accurate, helpful answers based solely on the provided context.
When referencing information, include the source reference (file, page, paragraph) from the document metadata.
Product code context: {product_code or 'Not specified'}"""

        user_prompt = f"""Based on these FDA 510(k) submission summaries, answer the following question:

Question: {query}

Relevant 510(k) Summaries:
{context}

Please provide a clear, comprehensive answer based on the information above. When citing specific information, reference the source document (file name, page, and paragraph) from the document metadata provided above. If the answer is not fully covered in the provided documents, indicate what information is missing."""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract text from response
            if message.content and len(message.content) > 0:
                return message.content[0].text.strip()
            else:
                # Fallback response
                context_text = "\n\n".join([chunk.get('text', '')[:300] for chunk in context_chunks[:3]])
                return f"Based on the retrieved FDA 510(k) documents:\n\n{context_text}"
        
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            context_text = "\n\n".join([chunk.get('text', '')[:300] for chunk in context_chunks[:3]])
            return f"Based on the retrieved FDA 510(k) documents:\n\n{context_text}"


class SimpleLLMGenerator:
    """
    Lightweight fallback LLM generator that doesn't require an API.
    Useful for development and testing when Claude API isn't available.
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
            ref_str = format_reference_string(chunk, include_link=False)
            response_parts.append(f"\n\nRelevant information from {ref_str}:\n{chunk_text[:400]}")
        
        response_parts.append(f"\n\nRegarding your question '{query}', the above documents contain relevant information that may help answer it.")
        
        return "\n".join(response_parts)
