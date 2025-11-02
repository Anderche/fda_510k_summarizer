"""
Reference Formatting Module

Formats references with file, page, paragraph, and PDF links.
"""

import os
from typing import Dict, Any, Optional


def format_reference(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a reference from chunk metadata.
    
    Args:
        chunk_data: Chunk dictionary with metadata
        
    Returns:
        Dictionary with formatted reference information
    """
    metadata = chunk_data.get('metadata', {})
    file_path = metadata.get('file_path', '')
    file_name = metadata.get('file_name', '')
    k_number = metadata.get('k_number', 'Unknown')
    page_num = metadata.get('page_num')
    para_index = metadata.get('para_index')
    
    # Get file name if not in metadata
    if not file_name and file_path:
        file_name = os.path.basename(file_path)
    
    # Generate PDF link (file:// URL for local files)
    pdf_link = None
    if file_path and os.path.exists(file_path):
        # Use file:// protocol for local files
        abs_path = os.path.abspath(file_path)
        # Encode path properly
        if os.name == 'nt':  # Windows
            # Windows: file:///C:/path/to/file
            abs_path = abs_path.replace('\\', '/')
            if not abs_path.startswith('/'):
                abs_path = '/' + abs_path
        else:  # Unix-like
            # Unix: file:///path/to/file
            if not abs_path.startswith('/'):
                abs_path = '/' + abs_path
        pdf_link = f"file://{abs_path}"
    
    return {
        'file_name': file_name,
        'k_number': k_number,
        'page_num': page_num,
        'para_index': para_index,
        'pdf_link': pdf_link,
        'file_path': file_path
    }


def format_reference_string(chunk_data: Dict[str, Any], include_link: bool = True) -> str:
    """
    Format a reference as a human-readable string.
    
    Args:
        chunk_data: Chunk dictionary with metadata
        include_link: Whether to include the PDF link
        
    Returns:
        Formatted reference string
    """
    ref = format_reference(chunk_data)
    
    parts = []
    
    # File name
    if ref['file_name']:
        parts.append(f"File: {ref['file_name']}")
    elif ref['k_number']:
        parts.append(f"K-number: {ref['k_number']}")
    
    # Page number
    if ref['page_num']:
        parts.append(f"Page {ref['page_num']}")
    
    # Paragraph
    if ref['para_index']:
        parts.append(f"Paragraph {ref['para_index']}")
    
    ref_str = " | ".join(parts) if parts else "Unknown source"
    
    # Add link if requested
    if include_link and ref['pdf_link']:
        ref_str += f" | Link: {ref['pdf_link']}"
    
    return ref_str


def format_reference_markdown(chunk_data: Dict[str, Any]) -> str:
    """
    Format a reference as Markdown with clickable link.
    
    Args:
        chunk_data: Chunk dictionary with metadata
        
    Returns:
        Markdown-formatted reference string
    """
    ref = format_reference(chunk_data)
    
    parts = []
    
    # File name or K-number
    if ref['file_name']:
        file_display = ref['file_name']
    elif ref['k_number']:
        file_display = f"K{ref['k_number']}"
    else:
        file_display = "Unknown"
    
    parts.append(f"**{file_display}**")
    
    # Page number
    if ref['page_num']:
        parts.append(f"Page {ref['page_num']}")
    
    # Paragraph
    if ref['para_index']:
        parts.append(f"Paragraph {ref['para_index']}")
    
    ref_text = " | ".join(parts) if parts else "Unknown source"
    
    # Add clickable link if available
    if ref['pdf_link']:
        ref_text = f"[{ref_text}]({ref['pdf_link']})"
    
    return ref_text


def format_multiple_references(chunks: list, format_type: str = "string") -> list:
    """
    Format multiple references.
    
    Args:
        chunks: List of chunk dictionaries
        format_type: Format type - "string", "markdown", or "dict"
        
    Returns:
        List of formatted references
    """
    if format_type == "dict":
        return [format_reference(chunk) for chunk in chunks]
    elif format_type == "markdown":
        return [format_reference_markdown(chunk) for chunk in chunks]
    else:  # string
        return [format_reference_string(chunk) for chunk in chunks]

