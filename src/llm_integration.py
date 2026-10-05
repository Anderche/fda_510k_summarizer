"""
LLM Integration Module

Handles LLM interactions for sub-summary generation and query responses.
Uses Claude API via Anthropic SDK.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Iterator, List, Dict, Optional, Any, Tuple
from reference_formatter import get_guidance_info

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests package not installed. FDA product code verification will be disabled.")

FDA_API_TIMEOUT_SECONDS = 3
DEFAULT_LLM_MODEL = "claude-haiku-4-5"
GUIDANCE_MAX_TOKENS = 4096  # full guidance answers, including lists
RESPONSE_MAX_TOKENS = 4096  # full 510(k) answers plus citations
REFINED_MAX_TOKENS = 1000  # refined summary of 200-300 words
REFINED_MAX_CHUNKS = 8
REFINED_MAX_CHUNK_CHARS = 600
REFINED_MAX_SUB_SUMMARIES = 20

_LIST_FORMAT_RE = re.compile(
    r"\b("
    r"list(?:s|ing|ed)?|"
    r"itemi[sz]e(?:d)?|itemi[sz]ation|"
    r"bullet(?:s|ed)?(?:\s+points?)?|"
    r"enumerat(?:e|ed|ion)|"
    r"numbered(?:\s+(?:list|items?))?|"
    r"(?:as|in)\s+(?:a\s+)?(?:list|bullets?|points?)|"
    r"(?:key|bullet)\s+points?|"
    r"step[- ]by[- ]step|"
    r"outline"
    r")\b",
    re.IGNORECASE,
)


def query_requests_structured_format(query: str) -> bool:
    """True when the user asked for a list, bullets, steps, or similar structure."""
    return bool(_LIST_FORMAT_RE.search(query or ""))


_RESPONSE_PREFIX_RE = re.compile(r'^(?:REFINED\s+)?SUMMARY:\s*', re.IGNORECASE)


def strip_response_prefix(text: str) -> str:
    """Remove a leading SUMMARY: / REFINED SUMMARY: label if the model still emits one."""
    if not text:
        return ''
    return _RESPONSE_PREFIX_RE.sub('', text.lstrip(), count=1).lstrip()


def _leading_prefix_pending(text: str) -> bool:
    """True while streamed text might still be a SUMMARY: prefix in progress."""
    stripped = text.lstrip()
    if not stripped:
        return True
    upper = stripped.upper()
    for prefix in ('SUMMARY:', 'REFINED SUMMARY:'):
        if prefix.startswith(upper) and len(upper) < len(prefix):
            return True
    return False


def format_source_header(index: int, chunk: Dict[str, Any]) -> str:
    """Numbered source label used in prompts, e.g. [1] Title, p. 12."""
    metadata = chunk.get('metadata') or {}
    file_name = metadata.get('file_name') or 'Unknown'
    page_num = metadata.get('page_num')
    info = get_guidance_info(file_name)
    title = (info or {}).get('title') or metadata.get('k_number') or file_name
    page_bit = f", p. {page_num}" if page_num else ""
    return f"[{index}] {title}{page_bit}"


def citation_instructions() -> str:
    return (
        "CITE:\n"
        "- Cite sources inline with [n] matching the numbered documents above.\n"
        "- Place each citation immediately after the claim it supports.\n"
        "- Do not start with SUMMARY: or restate the user query.\n"
        "- Write a complete answer from the provided documents; do not omit relevant requirements."
    )


def response_format_instructions(query: str) -> str:
    """Prompt text that makes the model follow the user's requested layout."""
    if query_requests_structured_format(query):
        return (
            "FORMAT:\n"
            "- The user asked for a list or itemized answer. You MUST use a Markdown list.\n"
            "- Use \"- item\" or \"1. item\", one item per line. Put each distinct point on its own line.\n"
            "- Do not collapse items into a paragraph. Keep items short and grounded in the documents.\n"
            "- Prefer this list format over any other response structure in these instructions."
        )
    return (
        "FORMAT:\n"
        "- Follow any structure requested in the User Query (lists, bullets, numbered items, steps).\n"
        "- If asked to list or itemize, use Markdown lists (\"- item\" or \"1. item\"), one item per line.\n"
        "- If no format is requested, write a complete answer from the provided documents; "
        "do not omit relevant requirements."
    )

_fda_session = requests.Session() if REQUESTS_AVAILABLE else None


@lru_cache(maxsize=256)
def _fetch_product_code_info(product_code: str) -> Optional[Dict[str, str]]:
    """
    Look up a product code in the openFDA classification endpoint.
    
    Network errors are raised (not returned) so that lru_cache does not cache them.
    """
    url = f"https://api.fda.gov/device/classification.json?search=product_code:{product_code}&limit=1"
    response = _fda_session.get(url, timeout=FDA_API_TIMEOUT_SECONDS)
    if response.status_code == 404:
        # openFDA answers 404 when a search has no matches
        return None
    response.raise_for_status()
    
    data = response.json()
    
    # Check if results exist - data['results'] should be an array
    if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
        result = data['results'][0]
        
        # Extract device information
        device_name = result.get('device_name', '')
        medical_specialty = result.get('medical_specialty_description', '')
        
        # Only return if we have at least device_name
        if device_name:
            return {
                'device_name': device_name,
                'medical_specialty_description': medical_specialty if medical_specialty else 'Not specified'
            }
        # Log if we got results but no device_name
        print(f"Warning: FDA API returned results for product code {product_code} but no device_name found")
    else:
        # Check if API returned no results
        if 'meta' in data and 'results' in data.get('meta', {}):
            meta_results = data['meta'].get('results', {})
            total = meta_results.get('total', 0)
            if total != 0:
                print(f"Warning: FDA API returned meta.total={total} but no results array found")
    
    return None

try:
    import nltk
    from nltk.corpus import stopwords
    # Download stopwords if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    NLTK_AVAILABLE = False
    # Fallback stopwords list (common English stopwords)
    ENGLISH_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how'
    }

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
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize LLM generator with Claude API.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Claude model name (default: claude-haiku-4-5)
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
                raise ValueError(f"Model '{self.model_name}' not found. Try --llm-model {DEFAULT_LLM_MODEL}")
            elif '401' in error_str or 'unauthorized' in error_str.lower():
                raise ValueError("API key invalid or missing. Check your ANTHROPIC_API_KEY.")
            else:
                raise e
    
    def _complete(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        """Run a single Claude completion and return its stripped text ('' if empty)."""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        if message.content and len(message.content) > 0:
            return message.content[0].text.strip()
        return ''
    
    def _stream_text(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        """Stream text deltas from a Claude completion."""
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        ) as stream:
            yield from stream.text_stream
    
    def _stream_with_fallback(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float,
                              fallback: Optional[Callable[[], Optional[str]]] = None) -> Iterator[str]:
        """
        Stream a completion, stripping a leading SUMMARY: label if the model still emits one.
        If nothing was streamed (error or empty output), yield the fallback text instead.
        """
        emitted = False
        pending = ''
        
        try:
            for text in self._stream_text(system_prompt, user_prompt, max_tokens, temperature):
                if not emitted:
                    pending += text
                    if _leading_prefix_pending(pending):
                        continue
                    text = strip_response_prefix(pending)
                if text:
                    emitted = True
                    yield text
            if not emitted and pending.strip():
                emitted = True
                yield strip_response_prefix(pending)
        except Exception as e:
            print(f"Error streaming response: {e}")
        
        if not emitted and fallback is not None:
            fallback_text = fallback()
            if fallback_text:
                yield fallback_text
    
    def _build_guidance_prompts(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Build (system_prompt, user_prompt) for FDA guidance documents."""
        # Build context from chunks with references
        context_parts = []
        total_length = 0
        max_context_length = 50000  # Smaller context for simplified prompt
        
        for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to top 5 chunks
            chunk_text = chunk.get('text', '')
            header = format_source_header(i, chunk)
            chunk_entry = f"{header}:\n{chunk_text}"
            
            # Check if adding this chunk would exceed context limit
            if total_length + len(chunk_entry) > max_context_length:
                remaining = max_context_length - total_length - 500
                chunk_entry = f"{header}:\n{chunk_text[:remaining]}..."
            
            context_parts.append(chunk_entry)
            total_length += len(chunk_entry)
            
            if total_length >= max_context_length:
                break
        
        context = "\n\n".join(context_parts)
        
        system_prompt = "You are a regulatory affairs consultant whose task is to research the AI regulations, standards, and guidances."
        
        user_prompt = f"""User Query: {query}

RELEVANT DOCUMENTS:
{context}

INSTRUCTIONS:
Provide a focused response that directly answers the query based on the FDA AI guidance documents provided. Focus on regulatory requirements, standards, and guidance specific to artificial intelligence in medical devices. Use metadata from documents (file names, sections) but do not reference product codes.
{response_format_instructions(query)}
{citation_instructions()}"""

        return system_prompt, user_prompt
    
    def _generate_guidance_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a simplified response for FDA guidance documents.
        
        Args:
            query: User's query/question
            context_chunks: List of retrieved chunk dictionaries with 'text' key
            
        Returns:
            Generated response text
        """
        system_prompt, user_prompt = self._build_guidance_prompts(query, context_chunks)
        try:
            response_text = self._complete(system_prompt, user_prompt, GUIDANCE_MAX_TOKENS, 0.7)
        except Exception as e:
            print(f"Error generating guidance response: {e}")
            return self._generate_fallback_summary(query, context_chunks)
        
        if not response_text:
            return self._generate_fallback_summary(query, context_chunks)
        return strip_response_prefix(response_text)
    
    def _generate_fallback_summary(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a simple fallback summary when LLM fails.
        
        Args:
            query: User's query
            context_chunks: List of retrieved chunks
            
        Returns:
            Fallback summary text
        """
        if not context_chunks:
            return "No relevant documents found to answer your question."
        
        # Extract key information from top 3 chunks
        summary_parts = ["Based on the retrieved FDA documents:"]
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            chunk_text = chunk.get('text', '').strip()
            # Take first 200 chars or first sentence
            if len(chunk_text) > 200:
                # Try to break at sentence
                first_sentence = chunk_text.split('.')[0] if '.' in chunk_text else chunk_text[:200]
                chunk_text = first_sentence[:200] + "..."
            if chunk_text:
                summary_parts.append(f"\n[{i}] {chunk_text}")
        
        return "\n".join(summary_parts)
    
    def _extract_product_codes(self, query: str) -> List[str]:
        """
        Extract potential 3-character product codes from query.
        
        Args:
            query: User query string
            
        Returns:
            List of potential 3-character product codes (filtered for non-stopwords)
        """
        # Find all 3-character sequences (alphanumeric, case-insensitive)
        # Pattern: word boundaries with exactly 3 alphanumeric characters
        pattern = r'\b[A-Za-z0-9]{3}\b'
        matches = re.findall(pattern, query)
        
        # Filter out stopwords and common English words (including common 3-letter words)
        common_three_letter_words = {'new', 'the', 'for', 'and', 'but', 'you', 'can', 'may', 'not', 
                                     'was', 'are', 'has', 'had', 'did', 'say', 'get', 'use', 'way',
                                     'see', 'him', 'her', 'she', 'all', 'how', 'why', 'who', 'two',
                                     'one', 'yes', 'now', 'old', 'any', 'day', 'way', 'try', 'put'}
        
        potential_codes = []
        for match in matches:
            match_upper = match.upper()
            match_lower = match.lower()
            # Exclude if it's a stopword, common English word, or common 3-letter word
            if (match_upper not in ENGLISH_STOPWORDS and 
                match_lower not in ENGLISH_STOPWORDS and
                match_lower not in common_three_letter_words):
                # Also exclude if it's all digits (product codes typically have letters)
                if not match.isdigit():
                    potential_codes.append(match_upper)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in potential_codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        
        return unique_codes
    
    def _verify_product_code(self, product_code: str) -> Optional[Dict[str, str]]:
        """
        Verify product code via FDA Open API and extract device information.
        
        Args:
            product_code: 3-character product code to verify
            
        Returns:
            Dictionary with 'device_name' and 'medical_specialty_description' if found,
            None otherwise
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            return _fetch_product_code_info(product_code)
        except requests.exceptions.RequestException as e:
            # Network or HTTP errors
            print(f"Warning: Failed to fetch FDA product code data for {product_code}: {e}")
        except (KeyError, TypeError, ValueError) as e:
            # JSON parsing or structure errors
            print(f"Warning: Error parsing FDA API response for {product_code}: {e}")
        except Exception as e:
            # Any other unexpected errors
            print(f"Warning: Unexpected error verifying product code {product_code}: {e}")
        
        return None
    
    def _find_verified_device_info(self, query: str, product_code: Optional[str]) -> Optional[Dict[str, str]]:
        """Verify the provided product code, else the first verifiable code found in the query."""
        # FIRST: Check provided product_code parameter (highest priority)
        if product_code:
            device_info = self._verify_product_code(product_code.upper())
            if device_info:
                print(f"✓ Verified product code '{product_code.upper()}' via FDA API: {device_info['device_name']} ({device_info['medical_specialty_description']})")
                return device_info
        
        # SECOND: Extract and verify codes from query (only if product_code wasn't provided or didn't verify)
        extracted_codes = self._extract_product_codes(query)
        if not extracted_codes:
            return None
        with ThreadPoolExecutor(max_workers=min(len(extracted_codes), 4)) as pool:
            results = list(pool.map(self._verify_product_code, extracted_codes))
        for code, device_info in zip(extracted_codes, results):
            if device_info:
                print(f"✓ Verified product code '{code}' via FDA API (from query): {device_info['device_name']} ({device_info['medical_specialty_description']})")
                return device_info  # Use first verified code
        return None
    
    def _build_response_prompts(self, query: str, context_chunks: List[Dict[str, Any]], product_code: Optional[str] = None,
                                query_ner: Optional[str] = None, query_tfidf: Optional[str] = None,
                                context_sections: Optional[str] = None) -> Tuple[str, str]:
        """Build (system_prompt, user_prompt) for 510(k) documents."""
        # Build context from chunks with references
        context_parts = []
        total_length = 0
        max_context_length = 200000  # Claude 3.5 Sonnet has 200k context
        
        for i, chunk in enumerate(context_chunks[:10], 1):  # Limit to top 10 chunks
            chunk_text = chunk.get('text', '')
            header = format_source_header(i, chunk)
            
            chunk_entry = f"{header}:\n{chunk_text}"
            
            # Check if adding this chunk would exceed context limit
            if total_length + len(chunk_entry) > max_context_length:
                # Truncate this chunk to fit
                remaining = max_context_length - total_length - 500  # Safety margin
                chunk_entry = f"{header}:\n{chunk_text[:remaining]}..."
            
            context_parts.append(chunk_entry)
            total_length += len(chunk_entry)
            
            if total_length >= max_context_length:
                break
        
        context = "\n\n".join(context_parts)
        
        # Extract and verify product codes - PRIORITIZE provided product_code parameter
        verified_device_info = self._find_verified_device_info(query, product_code)
        
        # Build device context string for prompt
        device_context_section = ""
        if verified_device_info:
            device_context_section = f"""

VERIFIED FDA DEVICE CLASSIFICATION (CRITICAL - USE THIS EXACT INFORMATION):
- Product Code: {product_code.upper() if product_code else 'N/A'}
- Device Name: {verified_device_info['device_name']}
- Medical Specialty: {verified_device_info['medical_specialty_description']}

IMPORTANT: You MUST use ONLY the device information provided above. Do NOT infer, guess, or assume the device type based on the product code or other context. The device name and medical specialty above are verified from the FDA database."""
        
        system_prompt = f"""You are an expert assistant specializing in Food and Drug Administration (FDA) Center for Devices and Radiological Health (CDRH) 510(k) medical device submission regulations. 
You provide responses based on retrieved 510(k) summary documents, FDA regulations, medical specialties, and product codes.
Product code context: {product_code or 'Not specified'}"""

        # Format NER and TF-IDF for prompt
        ner_info = query_ner if query_ner else 'None extracted'
        tfidf_info = query_tfidf if query_tfidf else 'None extracted'
        sections_info = context_sections if context_sections else 'N/A'

        user_prompt = f"""User Query: {query}{device_context_section}

Query Analysis:
- Named Entities (NER): {ner_info}
- TF-IDF Key Terms: {tfidf_info}

The following context documents with sections {sections_info}:

RELEVANT DOCUMENTS:
{context}

INSTRUCTIONS:
Produce a response with the following structure:

1. FDA CDRH 510(k) Specific Response (55% weighting):
   - Focus on Food and Drug Administration CDRH 510(k) regulations
   - Reference the VERIFIED device name and medical specialty provided above (if available)
   - DO NOT infer or guess device types - use ONLY the verified FDA classification information
   - Cite FDA guidelines, regulatory requirements, and submission standards
   - Include device classification, substantial equivalence, and predicate device information
   - Address compliance with 21 CFR regulations where applicable

2. Generalized LLM Response (45% weighting):
   - Provide broader context and general medical device information
   - Explain concepts in accessible terms
   - Connect to related regulatory frameworks beyond strict 510(k) requirements
   - Offer practical guidance and best practices

Start the answer with the direct response; do not restate the query.
{response_format_instructions(query)}
{citation_instructions()}

CRITICAL: If verified device information was provided above, you MUST reference the exact device name and medical specialty in your response. Do NOT substitute or infer different device types."""

        return system_prompt, user_prompt
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], product_code: Optional[str] = None,
                         query_ner: Optional[str] = None, query_tfidf: Optional[str] = None, 
                         context_sections: Optional[str] = None, source_type: Optional[str] = None) -> str:
        """
        Generate a response to a user query based on retrieved context.
        
        Args:
            query: User's query/question
            context_chunks: List of retrieved chunk dictionaries with 'text' key
            product_code: Optional product code for context (not used for guidance documents)
            query_ner: NER extraction results from query
            query_tfidf: TF-IDF terms from query
            context_sections: Sections found in retrieved context
            source_type: Source type ('corpus_ai_guidances' or '510k')
            
        Returns:
            Generated response text
        """
        # Use simplified prompt for guidance documents
        if source_type == 'corpus_ai_guidances':
            return self._generate_guidance_response(query, context_chunks)
        
        system_prompt, user_prompt = self._build_response_prompts(
            query, context_chunks, product_code, query_ner, query_tfidf, context_sections
        )
        try:
            response_text = self._complete(system_prompt, user_prompt, RESPONSE_MAX_TOKENS, 0.7)
        except Exception as e:
            print(f"Error generating response: {e}")
            # Generate a simple summary from top chunks as fallback
            return self._generate_fallback_summary(query, context_chunks)
        
        return strip_response_prefix(response_text) or self._generate_fallback_summary(query, context_chunks)
    
    def stream_response(self, query: str, context_chunks: List[Dict[str, Any]], product_code: Optional[str] = None,
                        query_ner: Optional[str] = None, query_tfidf: Optional[str] = None,
                        context_sections: Optional[str] = None, source_type: Optional[str] = None) -> Iterator[str]:
        """Streaming version of generate_response; yields text deltas."""
        def fallback() -> str:
            return self._generate_fallback_summary(query, context_chunks)
        
        if source_type == 'corpus_ai_guidances':
            system_prompt, user_prompt = self._build_guidance_prompts(query, context_chunks)
            yield from self._stream_with_fallback(system_prompt, user_prompt, GUIDANCE_MAX_TOKENS, 0.7,
                                                  fallback=fallback)
            return
        
        system_prompt, user_prompt = self._build_response_prompts(
            query, context_chunks, product_code, query_ner, query_tfidf, context_sections
        )
        yield from self._stream_with_fallback(system_prompt, user_prompt, RESPONSE_MAX_TOKENS, 0.7, fallback=fallback)
    
    def generate_refined_summary(self, query: str, context_chunks: List[Dict[str, Any]], 
                                  vector_store: Any, ner_tfidf_extractor: Any,
                                  product_code: Optional[str] = None) -> Optional[str]:
        """
        Generate a refined summary based on most relevant section and sub-summaries.
        
        Args:
            query: User's query/question
            context_chunks: List of retrieved chunk dictionaries
            vector_store: VectorStore instance to look up sub-summaries
            ner_tfidf_extractor: NERTFIDFExtractor instance for keyword extraction
            product_code: Optional product code for context
            
        Returns:
            Refined summary text, or None if unable to generate
        """
        prompts = self._build_refined_prompts(query, context_chunks, vector_store, ner_tfidf_extractor, product_code)
        if prompts is None:
            return None
        try:
            return strip_response_prefix(self._complete(*prompts, REFINED_MAX_TOKENS, 0.6)) or None
        except Exception as e:
            print(f"Error generating refined summary: {e}")
            return None
    
    def stream_refined_summary(self, query: str, context_chunks: List[Dict[str, Any]],
                               vector_store: Any, ner_tfidf_extractor: Any,
                               product_code: Optional[str] = None) -> Iterator[str]:
        """Streaming version of generate_refined_summary; yields nothing if no summary can be built."""
        prompts = self._build_refined_prompts(query, context_chunks, vector_store, ner_tfidf_extractor, product_code)
        if prompts is None:
            return
        yield from self._stream_with_fallback(*prompts, REFINED_MAX_TOKENS, 0.6)
    
    def _build_refined_prompts(self, query: str, context_chunks: List[Dict[str, Any]],
                               vector_store: Any, ner_tfidf_extractor: Any,
                               product_code: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Build (system_prompt, user_prompt) for the refined summary, or None if there is no usable context."""
        if not context_chunks:
            return None
        
        # Step 1: Find context_max_section (most common section)
        from collections import Counter
        section_counts = Counter()
        for chunk in context_chunks:
            section = chunk.get('metadata', {}).get('section_header')
            if section:
                section_counts[section] += 1
        
        if not section_counts:
            return None
        
        context_max_section = section_counts.most_common(1)[0][0]
        
        # Step 2: Extract top 3 NER and top 3 TF-IDF from all context chunks
        # Combine all chunk texts for NER/TF-IDF extraction
        all_chunk_texts = [chunk.get('text', '') for chunk in context_chunks if chunk.get('text')]
        combined_text = ' '.join(all_chunk_texts)
        
        # Extract NER and TF-IDF from combined context
        query_features = ner_tfidf_extractor.extract_query_features(combined_text)
        ner_results = query_features.get('ner', {})
        tfidf_terms = query_features.get('tfidf', [])
        
        # Get top 3 NER entities (flatten and take most relevant)
        all_ner_entities = []
        for entity_type, entities in ner_results.items():
            if entities:
                all_ner_entities.extend([(e, entity_type) for e in entities[:3]])
        top_ner = [e[0] for e in all_ner_entities[:3]]
        
        # Get top 3 TF-IDF terms
        top_tfidf = tfidf_terms[:3] if tfidf_terms else []
        
        user_prompt_relevant_keywords = {
            'ner': top_ner,
            'tfidf': top_tfidf
        }
        
        # Step 3: Get ALL chunks from context_max_section (not just similarity-filtered ones)
        # Use vector_store to get all chunk indices for this section
        section_chunk_indices = vector_store.get_chunks_by_section(context_max_section)
        
        if not section_chunk_indices:
            return None
        
        # Step 4: Get all chunks and their sub-summaries from context_max_section
        all_section_chunks = []
        sub_summaries = []
        full_chunk_texts = []
        
        # First, get chunks from vector_store mappings
        for chunk_idx in section_chunk_indices:
            if chunk_idx < len(vector_store.chunk_mappings):
                chunk_mapping = vector_store.chunk_mappings[chunk_idx]
                # Get full chunk data (non-sub-summary mappings)
                if not chunk_mapping.get('is_sub_summary', False):
                    chunk_data = chunk_mapping.get('chunk_data', {})
                    if chunk_data:
                        all_section_chunks.append(chunk_data)
                        # Also get full text for context
                        chunk_text = chunk_data.get('text', '')
                        if chunk_text:
                            full_chunk_texts.append(chunk_text)
        
        # Now find sub-summaries for all section chunks
        for chunk_data in all_section_chunks:
            if len(sub_summaries) >= REFINED_MAX_SUB_SUMMARIES:
                break
            chunk_idx = chunk_data.get('chunk_index')
            if chunk_idx is not None:
                sub_summary_text = vector_store.get_sub_summary_text(chunk_idx)
                if sub_summary_text:
                    sub_summaries.append(sub_summary_text)
        
        if not sub_summaries and not full_chunk_texts:
            return None
        
        # Step 5: Build comprehensive refined context
        ner_keywords_str = ', '.join(user_prompt_relevant_keywords['ner']) if user_prompt_relevant_keywords['ner'] else 'None'
        tfidf_keywords_str = ', '.join(user_prompt_relevant_keywords['tfidf']) if user_prompt_relevant_keywords['tfidf'] else 'None'
        
        # Combine sub-summaries and full chunk texts for rich context
        refined_context_parts = []
        
        if sub_summaries:
            sub_summaries_text = '\n\n'.join([f"Sub-summary {i+1}: {ss}" for i, ss in enumerate(sub_summaries)])
            refined_context_parts.append(f"SUB-SUMMARIES FROM {context_max_section} SECTION:\n{sub_summaries_text}")
        
        # Include full chunk texts (limit to avoid exceeding token limits)
        if full_chunk_texts:
            # Limit number of chunks and length to stay within context limits
            max_chunks = min(len(full_chunk_texts), REFINED_MAX_CHUNKS)
            max_chunk_length = REFINED_MAX_CHUNK_CHARS
            
            chunk_texts_formatted = []
            for i, chunk_text in enumerate(full_chunk_texts[:max_chunks], 1):
                # Truncate if too long
                if len(chunk_text) > max_chunk_length:
                    chunk_text = chunk_text[:max_chunk_length] + "..."
                chunk_texts_formatted.append(f"Chunk {i}:\n{chunk_text}")
            
            full_context_text = '\n\n'.join(chunk_texts_formatted)
            refined_context_parts.append(f"FULL CHUNK TEXTS FROM {context_max_section} SECTION:\n{full_context_text}")
        
        refined_context = '\n\n' + '\n\n'.join(refined_context_parts)
        
        system_prompt = f"""You are an expert assistant specializing in FDA CDRH 510(k) medical device submission regulations.
Product code context: {product_code or 'Not specified'}"""
        
        user_prompt = f"""User Query: {query}

Context Analysis: The user's query is most relevant to the '{context_max_section}' section.

Relevant Keywords Extracted:
- Top Named Entities: {ner_keywords_str}
- Top TF-IDF Terms: {tfidf_keywords_str}

Comprehensive Context from {context_max_section} Section:
{refined_context}

INSTRUCTIONS:
Generate a detailed, refined summary (200-300 words) that:
1. Directly and comprehensively answers the user's query
2. Synthesizes information from ALL available chunks in the '{context_max_section}' section
3. Incorporates the relevant keywords naturally throughout the response
4. Provides actionable, regulatory-focused information with specific details
5. References specific findings, requirements, or data points from the section content
6. Maintains accuracy and cites relevant FDA regulatory context where appropriate

Start the answer with the direct response; do not restate the query.
{response_format_instructions(query)}
{citation_instructions()}"""
        
        return system_prompt, user_prompt


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
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], product_code: Optional[str] = None,
                         query_ner: Optional[str] = None, query_tfidf: Optional[str] = None,
                         context_sections: Optional[str] = None, source_type: Optional[str] = None) -> str:
        """Generate a simple response based on context."""
        if not context_chunks:
            return "No relevant documents found to answer your question."
        
        # Use simplified response for guidance documents
        if source_type == 'corpus_ai_guidances':
            response_parts = [f"Based on FDA AI guidance documents:"]
            
            # Extract key information from top 3 chunks
            for i, chunk in enumerate(context_chunks[:3], 1):
                chunk_text = chunk.get('text', '').strip()
                header = format_source_header(i, chunk)
                # Limit chunk text to 200 chars for summary
                if len(chunk_text) > 200:
                    # Try to break at sentence
                    first_sentence = chunk_text.split('.')[0] if '.' in chunk_text else chunk_text[:200]
                    chunk_text = first_sentence[:200] + "..."
                if chunk_text:
                    response_parts.append(f"\n{header} {chunk_text}")
            
            return "\n".join(response_parts)
        
        # Standard 510k response
        response_parts = [f"Based on FDA CDRH 510(k) summaries for product code {product_code or 'the specified devices'}:"]
        
        if query_ner:
            response_parts.append(f"\nQuery entities: {query_ner}")
        if query_tfidf:
            response_parts.append(f"\nKey terms: {query_tfidf}")
        if context_sections:
            response_parts.append(f"\nRelevant sections: {context_sections}")
        
        # Extract key information from top 3 chunks
        for i, chunk in enumerate(context_chunks[:3], 1):
            chunk_text = chunk.get('text', '').strip()
            # Limit chunk text to 200 chars for summary
            if len(chunk_text) > 200:
                # Try to break at sentence
                first_sentence = chunk_text.split('.')[0] if '.' in chunk_text else chunk_text[:200]
                chunk_text = first_sentence[:200] + "..."
            if chunk_text:
                response_parts.append(f"\n{chunk_text}")
        
        return "\n".join(response_parts)
    
    def stream_response(self, *args, **kwargs) -> Iterator[str]:
        """Yield the template response in one piece (same signature as generate_response)."""
        yield self.generate_response(*args, **kwargs)

    def stream_refined_summary(self, *args, **kwargs) -> Iterator[str]:
        """Simple generator has no refined-summary path."""
        return
        yield  # pragma: no cover — keeps this a generator
