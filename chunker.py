"""
Document Chunking Module

Splits documents into semantically meaningful chunks using improved sentence-aware chunking.
Maintains context and metadata for each chunk.
Extracts section headers for enhanced retrieval.
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
    Extracts section headers dynamically from documents for enhanced retrieval.
    """
    
    # Standardized section categories
    SECTION_CATEGORIES = {
        'Administrative Info': [
            'cover letter', 'fda forms', 'indications for use', 'indication', 
            'administrative', 'submission', 'filing'
        ],
        'Public Info': [
            '510k summary', '510(k) summary', 'statement', 'truthful', 
            'accuracy', 'public', 'summary', '510k statement'
        ],
        'Device Details': [
            'executive summary', 'device description', 'substantial equivalence',
            'predicate device', 'device details', 'equivalence', 'description'
        ],
        'Performance/Data': [
            'labeling', 'declaration of conformity', 'sterilization', 'shelf life',
            'biocompatibility', 'software', 'cybersecurity', 'performance data',
            'performance testing', 'human factors', 'usability', 'electrical safety',
            'mechanical safety', 'emc', 'electromagnetic compatibility', 'packaging'
        ],
        'Testing': [
            'bench testing', 'animal testing', 'clinical testing', 'protocol',
            'test results', 'testing report', 'validation', 'verification',
            'clinical study', 'bench test', 'animal study', 'clinical data'
        ]
    }
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk (reduced for more chunks per PDF)
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
    
    def map_to_standard_section(self, detected_header: str) -> Optional[str]:
        """
        Map detected section header to one of the 5 standardized categories.
        
        Args:
            detected_header: Section header text detected from PDF
            
        Returns:
            Standardized section name or None if no match
        """
        if not detected_header:
            return None
        
        header_lower = detected_header.lower().strip()
        
        # Score each category based on keyword matches
        category_scores = {}
        for category, keywords in self.SECTION_CATEGORIES.items():
            score = 0
            for keyword in keywords:
                if keyword in header_lower:
                    # Exact match gets higher score
                    if keyword == header_lower or header_lower.startswith(keyword):
                        score += 10
                    else:
                        score += 1
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or None if no match
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _infer_section_from_content(self, chunk_text: str) -> Optional[str]:
        """
        Infer section category from chunk content when no header is found.
        
        Args:
            chunk_text: Text content of the chunk
            
        Returns:
            Standardized section name or None
        """
        if not chunk_text:
            return None
        
        text_lower = chunk_text.lower()
        
        # Score each category based on content keywords
        category_scores = {}
        for category, keywords in self.SECTION_CATEGORIES.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of keyword in text
                count = text_lower.count(keyword)
                if count > 0:
                    score += count * 2  # Weighted by frequency
        
            # Special patterns for each category
            if category == 'Administrative Info':
                if any(term in text_lower for term in ['fda form', 'cover letter', 'indication for use', 
                                                       'owner:', 'submitter:', 'contact:', '510(k) owner',
                                                       'we have reviewed', 'premarket notification', 
                                                       'comply with', 'registration and listing', 
                                                       'digitally signed', 'form approved']):
                    score += 5
            elif category == 'Public Info':
                if '510(k) summary' in text_lower or '510k summary' in text_lower or 'summary (21 cfr' in text_lower:
                    score += 10
                if 'statement' in text_lower and ('truthful' in text_lower or 'accuracy' in text_lower):
                    score += 5
            elif category == 'Device Details':
                if any(term in text_lower for term in ['device description', 'substantial equivalence', 
                                                        'predicate device', 'intended use', 'device name',
                                                        'forceps', 'grasper', 'retractor', 'instrument']):
                    score += 5
            elif category == 'Performance/Data':
                if any(term in text_lower for term in ['labeling', 'sterilization', 'biocompatibility', 
                                                       'software', 'cybersecurity', 'human factors', 
                                                       'packaging', 'shelf life', 'non-clinical testing']):
                    score += 5
            elif category == 'Testing':
                if any(term in text_lower for term in ['bench test', 'animal test', 'clinical test', 
                                                       'test protocol', 'test result', 'validation', 
                                                       'verification', 'study', 'bench verification',
                                                       'testing conducted', 'test cases', 'findings']):
                    score += 8
        
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score if above threshold
        if category_scores:
            best_category, best_score = max(category_scores.items(), key=lambda x: x[1])
            if best_score >= 2:  # Lower threshold to catch more chunks
                return best_category
        
        # Final fallback: if text mentions common FDA terms but no strong category match, default to Administrative Info
        if any(term in text_lower for term in ['510(k)', 'fda', 'cfr', 'device', 'submission']):
            return 'Administrative Info'
        
        return None
    
    def extract_section_headers(self, text: str) -> Dict[str, List[int]]:
        """
        Dynamically extract section headers from text using heuristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping section names to line numbers
        """
        sections = {}
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, start=1):
            line_stripped = line.strip()
            
            # Skip empty or very long lines
            if not line_stripped or len(line_stripped) < 3 or len(line_stripped) > 80:
                continue
            
            # Heuristic 1: All uppercase lines (likely headers)
            if line_stripped.isupper() and len(line_stripped) > 3 and not line_stripped.isdigit():
                # Check if next line has content (header followed by content)
                if line_num < len(lines) and lines[line_num].strip():
                    header_text = line_stripped
                    # Normalize: title case for consistency
                    if header_text not in sections:
                        sections[header_text] = []
                    sections[header_text].append(line_num)
                continue
            
            # Heuristic 2: Numbered headers (e.g., "1. PERFORMANCE DATA" or "1. Performance Data")
            numbered_pattern = re.match(r'^\s*(\d+[\.\)])\s*(.+)$', line_stripped)
            if numbered_pattern:
                content = numbered_pattern.group(2).strip()
                # Check if content looks like a header (short, capitalized, possibly all caps)
                if (3 <= len(content) <= 60 and 
                    (content.isupper() or content.istitle() or content[0].isupper())):
                    # Check next line has content
                    if line_num < len(lines) and lines[line_num].strip():
                        if content not in sections:
                            sections[content] = []
                        sections[content].append(line_num)
                continue
            
            # Heuristic 3: Title case short lines followed by blank line or content
            if (line_stripped.istitle() and 4 <= len(line_stripped) <= 50 and
                not line_stripped.endswith('.') and  # Not a sentence
                not any(char.isdigit() for char in line_stripped[:3]) and  # Not starting with number
                len(line_stripped.split()) <= 8):  # Not too many words
                
                # Check if next line is blank or content (not another header-like line)
                if line_num < len(lines):
                    next_line = lines[line_num].strip()
                    # Next line should be content (not empty and not another header)
                    if next_line and not (next_line.isupper() or next_line.istitle()):
                        if line_stripped not in sections:
                            sections[line_stripped] = []
                        sections[line_stripped].append(line_num)
                continue
            
            # Heuristic 4: Lines ending with colon (common header pattern)
            if line_stripped.endswith(':') and 4 <= len(line_stripped) <= 60:
                header_candidate = line_stripped[:-1].strip()
                # Must be title case or all caps
                if (header_candidate.isupper() or header_candidate.istitle() or 
                    header_candidate[0].isupper()):
                    if len(header_candidate.split()) <= 8:
                        if header_candidate not in sections:
                            sections[header_candidate] = []
                        sections[header_candidate].append(line_num)
        
        # Deduplicate similar headers (e.g., "PERFORMANCE DATA" and "Performance Data")
        normalized_sections = {}
        for header, line_nums in sections.items():
            # Normalize to title case for grouping
            normalized = header.strip().title()
            if normalized not in normalized_sections:
                normalized_sections[normalized] = []
            normalized_sections[normalized].extend(line_nums)
        
        # Keep original case for most common occurrence
        final_sections = {}
        for normalized, all_lines in normalized_sections.items():
            # Find original header with most occurrences
            original_header = max([h for h in sections.keys() if h.strip().title() == normalized],
                                 key=lambda h: len(sections[h]))
            final_sections[original_header] = sorted(set(all_lines))
        
        return final_sections
    
    def find_section_for_text(self, text: str, text_start_line: int, sections: Dict[str, List[int]]) -> Optional[str]:
        """
        Find which section a given text belongs to based on line numbers.
        
        Args:
            text: Text to find section for
            text_start_line: Starting line number of the text
            sections: Dictionary of sections to line numbers
            
        Returns:
            Section name or None
        """
        # Find the section whose line number is closest but before text_start_line
        best_section = None
        best_distance = float('inf')
        
        for section_name, line_numbers in sections.items():
            for line_num in line_numbers:
                if line_num <= text_start_line:
                    distance = text_start_line - line_num
                    if distance < best_distance:
                        best_distance = distance
                        best_section = section_name
        
        return best_section
    
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
    
    def _map_chunk_to_source(self, chunk_text: str, pages_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Map a chunk back to its source page and paragraph.
        
        Args:
            chunk_text: Text content of the chunk
            pages_data: List of page data with paragraphs
            
        Returns:
            Dictionary with 'page_num' and 'para_index', or None if not found
        """
        # Normalize chunk text for matching (remove extra whitespace)
        chunk_normalized = ' '.join(chunk_text.split())
        chunk_words = set(chunk_normalized.lower().split())
        chunk_word_list = chunk_normalized.lower().split()
        
        # If chunk is too short, return None
        if len(chunk_words) < 5:
            return None
        
        best_match = None
        best_score = 0
        
        # Search through all pages and paragraphs
        for page_data in pages_data:
            for para in page_data['paragraphs']:
                para_normalized = ' '.join(para['text'].split())
                para_words = set(para_normalized.lower().split())
                para_word_list = para_normalized.lower().split()
                
                # Skip very short paragraphs
                if len(para_words) < 3:
                    continue
                
                # Calculate Jaccard similarity (word overlap)
                if chunk_words and para_words:
                    overlap = len(chunk_words & para_words)
                    total = len(chunk_words | para_words)
                    jaccard_score = overlap / total if total > 0 else 0
                    
                    # Also calculate word order similarity for better matching
                    # Check how many consecutive word sequences from chunk appear in paragraph
                    order_score = 0.0
                    if len(chunk_word_list) >= 2:
                        # Check for 2-3 word sequences
                        sequences_found = 0
                        for seq_len in [3, 2]:  # Check 3-word, then 2-word sequences
                            for i in range(len(chunk_word_list) - seq_len + 1):
                                seq = ' '.join(chunk_word_list[i:i+seq_len])
                                if seq in para_normalized.lower():
                                    sequences_found += 1
                                    break
                        if sequences_found > 0:
                            order_score = min(0.3, sequences_found * 0.1)
                    
                    # Combined score
                    score = jaccard_score + order_score
                    
                    # Boost score if chunk is substring of paragraph (or vice versa)
                    if chunk_normalized.lower() in para_normalized.lower():
                        score = max(score, 0.9)
                    elif para_normalized.lower() in chunk_normalized.lower():
                        score = max(score, 0.85)
                    
                    # Boost if first few words match (likely same source)
                    if len(chunk_word_list) >= 3 and len(para_word_list) >= 3:
                        chunk_start = ' '.join(chunk_word_list[:3])
                        para_start = ' '.join(para_word_list[:3])
                        if chunk_start == para_start:
                            score += 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'page_num': page_data['page_num'],
                            'para_index': para['para_index']
                        }
        
        # Return match if score is reasonable
        # Lowered threshold to 0.15 (15%) since chunks may span paragraphs
        if best_match and best_score >= 0.15:
            return best_match
        
        # Fallback: find paragraph with most word overlap (even if < 15%)
        if best_match and best_score > 0:
            return best_match
        
        # Final fallback: search for paragraph containing chunk's first substantial sentence
        chunk_sentences = chunk_text.split('.')
        if chunk_sentences:
            first_sentence = chunk_sentences[0].strip()
            if len(first_sentence) > 20:
                first_sent_words = set(first_sentence.lower().split())
                for page_data in pages_data:
                    for para in page_data['paragraphs']:
                        para_normalized = ' '.join(para['text'].split())
                        para_words = set(para_normalized.lower().split())
                        # Check if first sentence words appear in paragraph
                        if first_sent_words and para_words:
                            overlap = len(first_sent_words & para_words)
                            if overlap >= min(3, len(first_sent_words) * 0.5):  # At least 50% of first sentence words
                                return {
                                    'page_num': page_data['page_num'],
                                    'para_index': para['para_index']
                                }
        
        return None
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents with page and paragraph tracking, including section headers.
        
        Args:
            documents: List of documents with 'k_number', 'file_path', 'text', and optionally 'pages_data' keys
            
        Returns:
            List of all chunks with metadata including page, paragraph, and section info
        """
        all_chunks = []
        
        for doc in documents:
            metadata = {
                'k_number': doc.get('k_number'),
                'file_path': doc.get('file_path'),
                'file_name': doc.get('file_name', doc.get('file_path', '').split('/')[-1] if doc.get('file_path') else ''),
                'source_document': doc.get('k_number')
            }
            
            # Extract section headers from document text and pages_data (which may have header flags)
            doc_text = doc['text']
            pages_data = doc.get('pages_data', [])
            
            # First, try to get headers from pages_data (if extracted with formatting)
            sections_from_formatting = {}
            if pages_data:
                line_num = 1
                for page_data in pages_data:
                    for para in page_data.get('paragraphs', []):
                        if para.get('is_header', False):
                            header_text = para['text'].strip()
                            if header_text and 3 <= len(header_text) <= 80:
                                # Map to standardized section
                                std_section = self.map_to_standard_section(header_text)
                                if std_section:
                                    if std_section not in sections_from_formatting:
                                        sections_from_formatting[std_section] = []
                                    sections_from_formatting[std_section].append(line_num)
                        line_num += 1
            
            # Also extract headers from raw text using heuristics (fallback/complement)
            sections_from_text = self.extract_section_headers(doc_text)
            
            # Merge both sources, mapping all to standardized sections
            sections = sections_from_formatting.copy()
            for header, line_nums in sections_from_text.items():
                std_section = self.map_to_standard_section(header)
                if std_section:
                    if std_section not in sections:
                        sections[std_section] = []
                    sections[std_section].extend(line_nums)
                    sections[std_section] = sorted(set(sections[std_section]))
            
            # Create line number mapping for text
            lines = doc_text.split('\n')
            line_start_positions = {}
            current_pos = 0
            for i, line in enumerate(lines, start=1):
                line_start_positions[i] = current_pos
                current_pos += len(line) + 1  # +1 for newline
            
            chunks = self.chunk_text(doc['text'], metadata)
            pages_data = doc.get('pages_data', [])
            
            # Map each chunk to its source page, paragraph, and section
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
                
                # Map chunk to source page and paragraph, and find section header
                if pages_data:
                    source_info = self._map_chunk_to_source(text, pages_data)
                    if source_info:
                        chunk['metadata']['page_num'] = source_info['page_num']
                        chunk['metadata']['para_index'] = source_info['para_index']
                        
                        # Try to find section header from pages_data directly (more reliable)
                        current_section = None
                        # Search current page first, then previous pages
                        for page_data in reversed(pages_data):
                            if page_data['page_num'] <= source_info['page_num']:
                                # On current page, look backwards; on previous pages, look for last header
                                for para in reversed(page_data.get('paragraphs', [])):
                                    if page_data['page_num'] == source_info['page_num']:
                                        # Current page: check if before current paragraph
                                        if para['para_index'] <= source_info['para_index']:
                                            if para.get('is_header', False):
                                                header_text = para['text'].strip()
                                                std_section = self.map_to_standard_section(header_text)
                                                if std_section:
                                                    current_section = std_section
                                                    break
                                    else:
                                        # Previous page: get last header
                                        if para.get('is_header', False):
                                            header_text = para['text'].strip()
                                            std_section = self.map_to_standard_section(header_text)
                                            if std_section:
                                                current_section = std_section
                                                break
                                
                                if current_section:
                                    break
                        
                        if current_section:
                            chunk['metadata']['section_header'] = current_section
                        
                        # Fallback: find section based on line number in document
                        if 'section_header' not in chunk['metadata']:
                            chunk_start_in_doc = doc_text.find(text[:100])
                            if chunk_start_in_doc >= 0:
                                estimated_line = 1
                                for line_num, pos in line_start_positions.items():
                                    if pos <= chunk_start_in_doc:
                                        estimated_line = line_num
                                    else:
                                        break
                                
                                section = self.find_section_for_text(text, estimated_line, sections)
                                if section:
                                    # Map to standardized section
                                    std_section = self.map_to_standard_section(section)
                                    if std_section:
                                        chunk['metadata']['section_header'] = std_section
                
                # Final fallback 1: find section from chunk text position
                if 'section_header' not in chunk['metadata']:
                    chunk_start_in_doc = doc_text.find(text[:100])
                    if chunk_start_in_doc >= 0:
                        estimated_line = 1
                        for line_num, pos in line_start_positions.items():
                            if pos <= chunk_start_in_doc:
                                estimated_line = line_num
                            else:
                                break
                        section = self.find_section_for_text(text, estimated_line, sections)
                        if section:
                            # Map to standardized section
                            std_section = self.map_to_standard_section(section)
                            if std_section:
                                chunk['metadata']['section_header'] = std_section
                
                # Final fallback 2: infer section from chunk content keywords
                if 'section_header' not in chunk['metadata']:
                    inferred_section = self._infer_section_from_content(text)
                    if inferred_section:
                        chunk['metadata']['section_header'] = inferred_section
                
                all_chunks.append(chunk)
        
        return all_chunks
