"""
PDF Text Extraction Module

Extracts text content from PDF files with improved extraction methods.
Handles complex layouts, filters headers/footers, and provides cleaned text.
"""

import fitz  # PyMuPDF
import pdfplumber
import os
import re
from typing import List, Dict, Optional, Any


def extract_text_from_pdf(pdf_path: str, use_pdfplumber: bool = True) -> Optional[Dict[str, Any]]:
    """
    Extract text from a PDF file using best available method, with page and paragraph tracking.
    
    Args:
        pdf_path: Path to the PDF file
        use_pdfplumber: Use pdfplumber as primary (better for tables/complex layouts)
        
    Returns:
        Dictionary with 'text', 'pages_data' keys, or None if extraction fails.
        'pages_data' is a list of dicts with 'page_num', 'paragraphs' (list of dicts with 'text', 'para_index')
    """
    pages_data = None
    
    # Try pdfplumber first (better for complex layouts)
    if use_pdfplumber:
        try:
            pages_data = _extract_with_pdfplumber(pdf_path)
        except Exception as e:
            print(f"  pdfplumber extraction failed for {os.path.basename(pdf_path)}, trying PyMuPDF: {e}")
            pages_data = None
    
    # Fallback to PyMuPDF
    if not pages_data:
        try:
            pages_data = _extract_with_pymupdf(pdf_path)
        except Exception as e:
            print(f"  PyMuPDF extraction also failed: {e}")
            return None
    
    if pages_data:
        # Reconstruct full text and apply cleaning
        full_text_parts = []
        for page_data in pages_data:
            page_text_parts = []
            for para in page_data['paragraphs']:
                cleaned_para = clean_text(para['text'])
                if cleaned_para:
                    para['text'] = cleaned_para
                    page_text_parts.append(cleaned_para)
            
            if page_text_parts:
                full_text_parts.append('\n\n'.join(page_text_parts))
        
        full_text = '\n\n'.join(full_text_parts)
        full_text = filter_headers_footers(full_text)
        
        # Re-apply paragraph splitting after filtering for cleaner structure
        # But preserve page info
        return {
            'text': full_text,
            'pages_data': pages_data
        }
    
    return None


def _extract_with_pdfplumber(pdf_path: str) -> Optional[List[Dict[str, Any]]]:
    """Extract text using pdfplumber (better for tables and complex layouts) with page/paragraph tracking."""
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # Split into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
                
                page_paragraphs = []
                for para_index, para_text in enumerate(paragraphs, start=1):
                    page_paragraphs.append({
                        'text': para_text,
                        'para_index': para_index
                    })
                
                if page_paragraphs:
                    pages_data.append({
                        'page_num': page_num,
                        'paragraphs': page_paragraphs
                    })
    
    if not pages_data:
        return None
    
    return pages_data


def _extract_with_pymupdf(pdf_path: str) -> Optional[List[Dict[str, Any]]]:
    """Extract text using PyMuPDF (fallback method) with page/paragraph tracking."""
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
            
            page_paragraphs = []
            for para_index, para_text in enumerate(paragraphs, start=1):
                page_paragraphs.append({
                    'text': para_text,
                    'para_index': para_index
                })
            
            if page_paragraphs:
                pages_data.append({
                    'page_num': page_num + 1,  # 1-indexed
                    'paragraphs': page_paragraphs
                })
    
    doc.close()
    
    if not pages_data:
        return None
    
    return pages_data


def filter_headers_footers(text: str) -> str:
    """
    Filter out common header/footer patterns from FDA 510(k) documents.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Filtered text with headers/footers removed
    """
    lines = text.split('\n')
    filtered_lines = []
    
    # Patterns to filter (headers/footers)
    header_footer_patterns = [
        r'^U\.S\. Food & Drug Administration',
        r'^10903 New Hampshire Avenue',
        r'^Silver Spring, MD',
        r'^www\.fda\.gov',
        r'^D o c  I D #',
        r'^Page \d+ of \d+',
        r'^PSC Publishing Services',
        r'^FORM FDA \d+',
        r'^DEPARTMENT OF HEALTH AND HUMAN SERVICES',
        r'^Food and Drug Administration',
        r'^\d{4}-\d{2}-\d{2}',  # Dates at start of line
        r'^Re:\s*K\d+',  # "Re: K200649" at start
        r'^\d{2}/\d{2}/\d{4}',  # Date formats
        r'^Sincerely yours,',
        r'^Enclosure',
        r'^EF$',  # End of form marker
        r'^\d+\.\d+\.\d+',  # Version numbers like "1.2.3"
    ]
    
    # Compile patterns
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in header_footer_patterns]
    
    # Also filter very short lines that are likely noise
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            filtered_lines.append('')
            continue
        
        # Skip if matches header/footer pattern
        if any(pattern.match(line) for pattern in compiled_patterns):
            continue
        
        # Skip very short lines that are likely page numbers or form IDs
        if len(line) < 5 and (line.isdigit() or re.match(r'^[A-Z]\d+$', line)):
            continue
        
        # Skip lines that are just addresses (multiple short words)
        words = line.split()
        if len(words) > 3 and all(len(w) < 15 for w in words) and any(
            word.lower() in ['road', 'avenue', 'street', 'city', 'state', 'zip'] for word in words
        ):
            continue
        
        filtered_lines.append(line)
    
    # Remove excessive blank lines
    cleaned = []
    prev_empty = False
    for line in filtered_lines:
        if line:
            cleaned.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned.append('')
            prev_empty = True
    
    # Remove trailing empty lines
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    
    return '\n'.join(cleaned)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated line breaks
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    
    # Split by paragraphs
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Skip paragraphs that are too short (likely noise)
        if len(para) < 20:
            continue
        
        # Skip paragraphs that are mostly numbers/special chars
        if len(re.findall(r'[A-Za-z]', para)) < len(para) * 0.3:
            continue
        
        cleaned_paragraphs.append(para)
    
    return '\n\n'.join(cleaned_paragraphs)


def extract_text_from_corpus(corpus_dir: str, use_pdfplumber: bool = True) -> List[Dict[str, Any]]:
    """
    Extract text from all PDFs in a corpus directory with page and paragraph tracking.
    
    Args:
        corpus_dir: Path to corpus directory containing PDF files
        use_pdfplumber: Use pdfplumber as primary extraction method
        
    Returns:
        List of dictionaries with 'k_number', 'file_path', 'text', and 'pages_data' keys
    """
    if not os.path.exists(corpus_dir):
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")
    
    extracted_documents = []
    pdf_files = [f for f in os.listdir(corpus_dir) if f.endswith('.pdf')]
    
    print(f"Extracting text from {len(pdf_files)} PDFs in {corpus_dir}...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(corpus_dir, pdf_file)
        k_number = os.path.splitext(pdf_file)[0]
        
        result = extract_text_from_pdf(pdf_path, use_pdfplumber=use_pdfplumber)
        
        if result and result.get('text'):
            text = result['text']
            pages_data = result.get('pages_data', [])
            
            # Additional filtering for very short documents
            if len(text.strip()) < 100:
                print(f"  ⚠ Extracted text from {pdf_file} is very short ({len(text)} chars), may be low quality")
            
            extracted_documents.append({
                'k_number': k_number,
                'file_path': pdf_path,
                'file_name': pdf_file,
                'text': text,
                'pages_data': pages_data
            })
            print(f"  ✓ Extracted {len(text)} characters from {pdf_file} ({len(pages_data)} pages)")
        else:
            print(f"  ✗ Failed to extract text from {pdf_file}")
    
    print(f"Successfully extracted text from {len(extracted_documents)}/{len(pdf_files)} PDFs")
    return extracted_documents
