"""
LLM Integration Module

Handles LLM interactions for sub-summary generation and query responses.
Uses Claude API via Anthropic SDK.
"""

import os
import re
from typing import List, Dict, Optional, Any
from reference_formatter import format_reference_string

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests package not installed. FDA product code verification will be disabled.")

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
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-haiku-20240307"):
        """
        Initialize LLM generator with Claude API.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Claude model name (default: claude-3-haiku-20240307 - simplest/cheapest)
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
    
    def _generate_guidance_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a simplified response for FDA guidance documents.
        
        Args:
            query: User's query/question
            context_chunks: List of retrieved chunk dictionaries with 'text' key
            
        Returns:
            Generated response text (~80 words)
        """
        # Build context from chunks with references
        context_parts = []
        total_length = 0
        max_context_length = 50000  # Smaller context for simplified prompt
        
        for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to top 5 chunks
            chunk_text = chunk.get('text', '')
            file_name = chunk.get('metadata', {}).get('file_name', 'Unknown')
            
            chunk_entry = f"Document {i} ({file_name}):\n{chunk_text}"
            
            # Check if adding this chunk would exceed context limit
            if total_length + len(chunk_entry) > max_context_length:
                remaining = max_context_length - total_length - 500
                chunk_entry = f"Document {i} ({file_name}):\n{chunk_text[:remaining]}..."
            
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
Provide a concise, focused response (~80 words) that directly answers the query based on the FDA AI guidance documents provided. Focus on regulatory requirements, standards, and guidance specific to artificial intelligence in medical devices. Use metadata from documents (file names, sections) but do not reference product codes."""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=200,  # Limit to ~80 words
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract text from response
            if message.content and len(message.content) > 0:
                response_text = message.content[0].text.strip()
                # Ensure it starts with SUMMARY:
                if not response_text.strip().startswith("SUMMARY:"):
                    response_text = "SUMMARY:\n" + response_text
                return response_text
            else:
                return self._generate_fallback_summary(query, context_chunks)
        
        except Exception as e:
            print(f"Error generating guidance response: {e}")
            return self._generate_fallback_summary(query, context_chunks)
    
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
            return "SUMMARY:\nNo relevant documents found to answer your question."
        
        # Extract key information from top 3 chunks
        summary_parts = []
        summary_parts.append(f"Based on the retrieved FDA 510(k) documents regarding your question about {query.lower()}:")
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            chunk_text = chunk.get('text', '').strip()
            # Take first 200 chars or first sentence
            if len(chunk_text) > 200:
                # Try to break at sentence
                first_sentence = chunk_text.split('.')[0] if '.' in chunk_text else chunk_text[:200]
                chunk_text = first_sentence[:200] + "..."
            if chunk_text:
                summary_parts.append(f"\n{chunk_text}")
        
        summary = "\n".join(summary_parts)
        # Ensure it starts with SUMMARY:
        if not summary.startswith("SUMMARY:"):
            summary = "SUMMARY:\n" + summary
        
        return summary
    
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
            url = f"https://api.fda.gov/device/classification.json?search=product_code:{product_code}&limit=1"
            response = requests.get(url, timeout=10)
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
                else:
                    # Log if we got results but no device_name
                    print(f"Warning: FDA API returned results for product code {product_code} but no device_name found")
            else:
                # Check if API returned no results
                if 'meta' in data and 'results' in data.get('meta', {}):
                    meta_results = data['meta'].get('results', {})
                    total = meta_results.get('total', 0)
                    if total == 0:
                        # No results found for this product code - this is normal
                        pass
                    else:
                        print(f"Warning: FDA API returned meta.total={total} but no results array found")
                        
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
        
        # Extract and verify product codes - PRIORITIZE provided product_code parameter
        verified_device_info = None
        
        # FIRST: Check provided product_code parameter (highest priority)
        if product_code:
            device_info = self._verify_product_code(product_code.upper())
            if device_info:
                verified_device_info = device_info
                print(f"✓ Verified product code '{product_code.upper()}' via FDA API: {device_info['device_name']} ({device_info['medical_specialty_description']})")
        
        # SECOND: Extract and verify codes from query (only if product_code wasn't provided or didn't verify)
        if not verified_device_info:
            extracted_codes = self._extract_product_codes(query)
            for code in extracted_codes:
                device_info = self._verify_product_code(code)
                if device_info:
                    verified_device_info = device_info
                    print(f"✓ Verified product code '{code}' via FDA API (from query): {device_info['device_name']} ({device_info['medical_specialty_description']})")
                    break  # Use first verified code
        
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

FORMAT YOUR RESPONSE AS:
SUMMARY:
[Your response combining 55% FDA CDRH 510(k)-specific regulatory content with 45% generalized medical device guidance - must be under 200 words total]

CRITICAL: If verified device information was provided above, you MUST reference the exact device name and medical specialty in your response. Do NOT substitute or infer different device types."""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1500,  # Increased for summary + references
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
                # Fallback: generate summary from chunks
                return self._generate_fallback_summary(query, context_chunks)
        
        except Exception as e:
            print(f"Error generating response: {e}")
            # Generate a simple summary from top chunks as fallback
            return self._generate_fallback_summary(query, context_chunks)
    
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
            chunk_idx = chunk_data.get('chunk_index')
            if chunk_idx is not None:
                # Find matching sub-summary in vector_store
                for mapping in vector_store.chunk_mappings:
                    if (mapping.get('is_sub_summary') and 
                        mapping.get('chunk_data', {}).get('chunk_index') == chunk_idx):
                        sub_summary_data = mapping.get('sub_summary', {})
                        sub_summary_text = sub_summary_data.get('text', '')
                        if sub_summary_text:
                            sub_summaries.append(sub_summary_text)
                        break
        
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
            max_chunks = min(len(full_chunk_texts), 15)  # Limit to top 15 chunks
            max_chunk_length = 1000  # Limit each chunk to 1000 chars
            
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

FORMAT YOUR RESPONSE AS:
REFINED SUMMARY:
[Your detailed refined summary here]"""
        
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,  # Increased for detailed 200-300 word summary
                temperature=0.6,  # Lower temperature for more focused responses
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract text from response
            if message.content and len(message.content) > 0:
                return message.content[0].text.strip()
        except Exception as e:
            print(f"Error generating refined summary: {e}")
        
        return None


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
            return "SUMMARY:\nNo relevant documents found to answer your question."
        
        # Use simplified response for guidance documents
        if source_type == 'corpus_ai_guidances':
            response_parts = ["SUMMARY:"]
            response_parts.append(f"Based on FDA AI guidance documents, regarding your question '{query}':")
            
            # Extract key information from top 3 chunks
            for i, chunk in enumerate(context_chunks[:3], 1):
                chunk_text = chunk.get('text', '').strip()
                file_name = chunk.get('metadata', {}).get('file_name', 'Unknown')
                # Limit chunk text to 200 chars for summary
                if len(chunk_text) > 200:
                    # Try to break at sentence
                    first_sentence = chunk_text.split('.')[0] if '.' in chunk_text else chunk_text[:200]
                    chunk_text = first_sentence[:200] + "..."
                if chunk_text:
                    response_parts.append(f"\n[{file_name}] {chunk_text}")
            
            return "\n".join(response_parts)
        
        # Standard 510k response
        response_parts = ["SUMMARY:"]
        response_parts.append(f"Based on FDA CDRH 510(k) summaries for product code {product_code or 'the specified devices'}, regarding your question '{query}':")
        
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
