"""
PDF Text Extraction Module

Extracts text content from PDF files with improved extraction methods.
Handles complex layouts, filters headers/footers, and provides cleaned text.
"""

import fitz  # PyMuPDF
import pdfplumber
import os
import re
from typing import List, Dict, Optional


def extract_text_from_pdf(pdf_path: str, use_pdfplumber: bool = True) -> Optional[str]:
    """
    Extract text from a PDF file using best available method.
    
    Args:
        pdf_path: Path to the PDF file
        use_pdfplumber: Use pdfplumber as primary (better for tables/complex layouts)
        
    Returns:
        Extracted text as string, or None if extraction fails
    """
    text = None
    
    # Try pdfplumber first (better for complex layouts)
    if use_pdfplumber:
        try:
            text = _extract_with_pdfplumber(pdf_path)
        except Exception as e:
            print(f"  pdfplumber extraction failed for {os.path.basename(pdf_path)}, trying PyMuPDF: {e}")
            text = None
    
    # Fallback to PyMuPDF
    if not text:
        try:
            text = _extract_with_pymupdf(pdf_path)
        except Exception as e:
            print(f"  PyMuPDF extraction also failed: {e}")
            return None
    
    if text:
        text = clean_text(text)
        text = filter_headers_footers(text)
        return text
    
    return None


def _extract_with_pdfplumber(pdf_path: str) -> Optional[str]:
    """Extract text using pdfplumber (better for tables and complex layouts)."""
    text_parts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(text)
    
    if not text_parts:
        return None
    
    return "\n\n".join(text_parts)


def _extract_with_pymupdf(pdf_path: str) -> Optional[str]:
    """Extract text using PyMuPDF (fallback method)."""
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
    
    doc.close()
    
    if not text_parts:
        return None
    
    return "\n\n".join(text_parts)


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


def extract_text_from_corpus(corpus_dir: str, use_pdfplumber: bool = True) -> List[Dict[str, str]]:
    """
    Extract text from all PDFs in a corpus directory.
    
    Args:
        corpus_dir: Path to corpus directory containing PDF files
        use_pdfplumber: Use pdfplumber as primary extraction method
        
    Returns:
        List of dictionaries with 'k_number', 'file_path', and 'text' keys
    """
    if not os.path.exists(corpus_dir):
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")
    
    extracted_documents = []
    pdf_files = [f for f in os.listdir(corpus_dir) if f.endswith('.pdf')]
    
    print(f"Extracting text from {len(pdf_files)} PDFs in {corpus_dir}...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(corpus_dir, pdf_file)
        k_number = os.path.splitext(pdf_file)[0]
        
        text = extract_text_from_pdf(pdf_path, use_pdfplumber=use_pdfplumber)
        
        if text:
            # Additional filtering for very short documents
            if len(text.strip()) < 100:
                print(f"  ⚠ Extracted text from {pdf_file} is very short ({len(text)} chars), may be low quality")
            
            extracted_documents.append({
                'k_number': k_number,
                'file_path': pdf_path,
                'text': text
            })
            print(f"  ✓ Extracted {len(text)} characters from {pdf_file}")
        else:
            print(f"  ✗ Failed to extract text from {pdf_file}")
    
    print(f"Successfully extracted text from {len(extracted_documents)}/{len(pdf_files)} PDFs")
    return extracted_documents
