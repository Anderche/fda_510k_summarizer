"""
Test Vector Store Module

Tests and demonstrates vector store functionality including metadata access.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from vector_store import VectorStore


def print_metadata(index_dir: str = "rag_index"):
    """
    Print section headers and metadata statistics from the index.
    
    Args:
        index_dir: Directory containing RAG index files
    """
    index_path = os.path.join(index_dir, "vector_index")
    
    # LangChain FAISS saves as a directory, old format saved as .index file
    if not os.path.exists(index_path) and not os.path.exists(f"{index_path}.index"):
        print(f"Error: Index not found at {index_path}")
        print("Please build the index first using: python build_rag_index.py corpus_NAY/")
        return
    
    print("=" * 80)
    print("RAG Index Metadata")
    print("=" * 80)
    
    # Load vector store
    vector_store = VectorStore()
    vector_store.load(index_path)
    
    # Print statistics
    stats = vector_store.get_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Total sections: {stats.get('total_sections', 0)}")
    print(f"  Is normalized: {stats.get('is_normalized', False)}")
    
    # Print section headers
    sections = vector_store.get_sections()
    print(f"\n{'=' * 80}")
    print(f"Section Headers ({len(sections)} total):")
    print(f"{'=' * 80}")
    if sections:
        for i, section in enumerate(sorted(sections), 1):
            chunk_indices = vector_store.get_chunks_by_section(section)
            chunk_count = len(chunk_indices)
            print(f"  {i:2d}. {section} ({chunk_count} chunks)")
    else:
        print("  No section headers found in index.")
    
    # Count sections including N/A
    from collections import Counter
    section_counts_all = Counter()
    for mapping in vector_store.chunk_mappings:
        if not mapping.get('is_sub_summary', False):
            section = mapping.get('chunk_data', {}).get('metadata', {}).get('section_header', 'N/A')
            section_counts_all[section] += 1
    
    print(f"\n{'=' * 80}")
    print(f"Section Value Counts (including N/A):")
    print(f"{'=' * 80}")
    for section, count in sorted(section_counts_all.items(), key=lambda x: x[1], reverse=True):
        print(f"  {section}: {count} chunks")
    
    # Check for sub-summaries
    sub_summary_count = sum(1 for m in vector_store.chunk_mappings if m.get('is_sub_summary', False))
    chunk_count = len(vector_store.chunk_mappings) - sub_summary_count
    print(f"\n{'=' * 80}")
    print(f"Embedding Breakdown:")
    print(f"{'=' * 80}")
    print(f"  Chunk embeddings: {chunk_count}")
    print(f"  Sub-summary embeddings: {sub_summary_count}")
    print(f"  Total vectors: {stats['total_vectors']}")
    
    # Print sample chunk metadata
    print(f"\n{'=' * 80}")
    print("Sample Chunk Metadata (first 10 chunks):")
    print(f"{'=' * 80}")
    
    sample_chunks = [m for m in vector_store.chunk_mappings if not m.get('is_sub_summary', False)][:10]
    for i, mapping in enumerate(sample_chunks, 1):
        chunk_data = mapping.get('chunk_data', {})
        metadata = chunk_data.get('metadata', {})
        
        print(f"\n  Chunk {i} (Index: {mapping.get('chunk_index', 'N/A')}):")
        print(f"    K-number: {metadata.get('k_number', 'N/A')}")
        print(f"    File: {metadata.get('file_name', 'N/A')}")
        print(f"    Page: {metadata.get('page_num', 'N/A')}")
        print(f"    Paragraph: {metadata.get('para_index', 'N/A')}")
        print(f"    Section: {metadata.get('section_header', 'N/A')}")
        print(f"    Chunk index: {chunk_data.get('chunk_index', 'N/A')}")
        print(f"    Token count: {chunk_data.get('token_count', 'N/A')}")
        text_preview = chunk_data.get('text', '')[:100]
        if text_preview:
            print(f"    Text preview: {text_preview}...")
        
        # Check if this chunk has a sub-summary
        sub_summary_mapping = next((m for m in vector_store.chunk_mappings 
                                   if m.get('is_sub_summary') and 
                                   m.get('chunk_data', {}).get('chunk_index') == chunk_data.get('chunk_index')), None)
        if sub_summary_mapping:
            sub_summary_text = sub_summary_mapping.get('sub_summary', {}).get('text', '')
            if sub_summary_text:
                print(f"    Sub-summary: {sub_summary_text[:150]}...")
    
    # Print section distribution (including N/A)
    print(f"\n{'=' * 80}")
    print("Section Distribution:")
    print(f"{'=' * 80}")
    # Ensure N/A is always in the counts (add with 0 if missing)
    if 'N/A' not in section_counts_all:
        section_counts_all['N/A'] = 0
    
    # Build section_counts dict grouped by count
    section_counts = {}
    for section, count in sorted(section_counts_all.items(), key=lambda x: x[1], reverse=True):
        if count not in section_counts:
            section_counts[count] = []
        section_counts[count].append(section)
    
    # Print sorted by count (descending)
    for count in sorted(section_counts.keys(), reverse=True):
        sections_with_count = section_counts[count]
        for section in sections_with_count:
            print(f"  {count} chunks: {section}")
    
    print("\n" + "=" * 80)


def test_vector_store_basic(index_dir: str = "rag_index"):
    """
    Test basic vector store functionality.
    
    Args:
        index_dir: Directory containing RAG index files
    """
    index_path = os.path.join(index_dir, "vector_index")
    
    if not os.path.exists(f"{index_path}.index"):
        print(f"Error: Index not found at {index_path}")
        return
    
    print("=" * 80)
    print("Testing Vector Store")
    print("=" * 80)
    
    # Load vector store
    vector_store = VectorStore()
    vector_store.load(index_path)
    
    # Test get_sections
    sections = vector_store.get_sections()
    print(f"\n✓ get_sections() returned {len(sections)} sections")
    
    # Test get_chunks_by_section
    if sections:
        test_section = sections[0]
        chunk_indices = vector_store.get_chunks_by_section(test_section)
        print(f"✓ get_chunks_by_section('{test_section}') returned {len(chunk_indices)} chunks")
    
    # Test get_stats
    stats = vector_store.get_stats()
    print(f"✓ get_stats() returned stats with {stats['total_vectors']} vectors")
    
    # Test search capability
    from embeddings import EmbeddingGenerator
    embedding_generator = EmbeddingGenerator()
    test_query = "performance testing"
    query_embedding = embedding_generator.embed_text(test_query)
    results = vector_store.search(query_embedding, k=3)
    print(f"✓ search() returned {len(results)} results for query '{test_query}'")
    
    if results:
        print(f"  Top result similarity: {results[0][1]:.3f}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test vector store and print metadata')
    parser.add_argument('index_dir', nargs='?', default='rag_index', help='Directory containing RAG index files (default: rag_index)')
    parser.add_argument('--test', action='store_true', help='Run basic tests')
    
    args = parser.parse_args()
    
    if args.test:
        test_vector_store_basic(args.index_dir)
    else:
        print_metadata(args.index_dir)

