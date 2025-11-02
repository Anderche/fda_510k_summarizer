"""
Print Metadata Script

Utility to print section headers and metadata from a RAG index.
"""

import os
import argparse
from vector_store import VectorStore


def print_metadata(index_dir: str):
    """
    Print section headers and metadata statistics from the index.
    
    Args:
        index_dir: Directory containing RAG index files
    """
    index_path = os.path.join(index_dir, "vector_index")
    
    if not os.path.exists(f"{index_path}.index"):
        print(f"Error: Index not found at {index_path}")
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
    
    # Print section headers
    sections = vector_store.get_sections()
    print(f"\n{'=' * 80}")
    print(f"Section Headers ({len(sections)} total):")
    print(f"{'=' * 80}")
    if sections:
        for i, section in enumerate(sorted(sections), 1):
            chunk_count = len(vector_store.get_chunks_by_section(section))
            print(f"  {i:2d}. {section} ({chunk_count} chunks)")
    else:
        print("  No section headers found in index.")
    
    # Print sample chunk metadata
    print(f"\n{'=' * 80}")
    print("Sample Chunk Metadata (first 10 chunks):")
    print(f"{'=' * 80}")
    
    sample_chunks = vector_store.chunk_mappings[:10]
    for i, mapping in enumerate(sample_chunks, 1):
        chunk_data = mapping.get('chunk_data', {})
        metadata = chunk_data.get('metadata', {})
        
        print(f"\n  Chunk {i}:")
        print(f"    K-number: {metadata.get('k_number', 'N/A')}")
        print(f"    File: {metadata.get('file_name', 'N/A')}")
        print(f"    Page: {metadata.get('page_num', 'N/A')}")
        print(f"    Paragraph: {metadata.get('para_index', 'N/A')}")
        print(f"    Section: {metadata.get('section_header', 'N/A')}")
        print(f"    Chunk index: {chunk_data.get('chunk_index', 'N/A')}")
        text_preview = chunk_data.get('text', '')[:100]
        print(f"    Text preview: {text_preview}...")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print metadata from RAG index')
    parser.add_argument('index_dir', help='Directory containing RAG index files')
    
    args = parser.parse_args()
    print_metadata(args.index_dir)

