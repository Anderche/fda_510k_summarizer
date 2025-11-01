"""
Build RAG Index Script

Orchestrates the complete pipeline to build the RAG index from corpus PDFs.
"""

import os
import argparse
from pdf_extractor import extract_text_from_corpus
from chunker import DocumentChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from llm_integration import LLMGenerator, SimpleLLMGenerator


def build_rag_index(corpus_dir: str, output_dir: str = "rag_index", 
                   chunk_size: int = 500, chunk_overlap: int = 50,
                   generate_sub_summaries: bool = True,
                   use_llm: bool = False, llm_model: str = "meta-llama/Llama-2-7b-chat-hf"):
    """
    Build complete RAG index from corpus.
    
    Args:
        corpus_dir: Directory containing PDF files
        output_dir: Directory to save index files
        chunk_size: Target chunk size in tokens
        chunk_overlap: Chunk overlap in tokens
        generate_sub_summaries: Whether to generate sub-summaries for multi-vector RAG
        use_llm: Whether to use full LLM (requires model download) or simple generator
        llm_model: LLM model name if use_llm is True
    """
    print("=" * 80)
    print("Building RAG Index")
    print("=" * 80)
    
    # Step 1: Extract text from PDFs
    print("\n[Step 1/6] Extracting text from PDFs...")
    documents = extract_text_from_corpus(corpus_dir, use_pdfplumber=True)
    
    if not documents:
        print("No documents extracted. Exiting.")
        return
    
    # Step 2: Chunk documents
    print(f"\n[Step 2/6] Chunking {len(documents)} documents...")
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Step 3: Generate embeddings for chunks
    print(f"\n[Step 3/6] Generating embeddings for {len(chunks)} chunks...")
    embedding_generator = EmbeddingGenerator()
    chunk_embeddings = embedding_generator.embed_chunks(chunks)
    
    # Step 4: Initialize vector store and add chunk embeddings
    print(f"\n[Step 4/6] Adding chunk embeddings to vector store...")
    vector_store = VectorStore(embedding_dim=chunk_embeddings.shape[1])
    vector_store.add_embeddings(chunk_embeddings, chunks)
    
    # Step 5: Generate sub-summaries and embeddings (multi-vector RAG)
    if generate_sub_summaries:
        print(f"\n[Step 5/6] Generating sub-summaries for multi-vector RAG...")
        
        # Initialize LLM generator
        if use_llm:
            print(f"  Using LLM model: {llm_model}")
            llm = LLMGenerator(model_name=llm_model)
        else:
            print("  Using simple template-based sub-summary generator")
            llm = SimpleLLMGenerator()
        
        sub_summaries = []
        linked_chunks = []
        
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            k_number = chunk.get('metadata', {}).get('k_number', None)
            sub_summary_text = llm.generate_sub_summary(chunk['text'], k_number=k_number)
            
            sub_summaries.append({
                'text': sub_summary_text,
                'source_chunk_index': chunk.get('chunk_index', i)
            })
            linked_chunks.append(chunk)
        
        print(f"  Generated {len(sub_summaries)} sub-summaries")
        
        # Generate embeddings for sub-summaries
        print(f"  Generating embeddings for sub-summaries...")
        sub_summary_embeddings = embedding_generator.embed_sub_summaries(sub_summaries)
        
        # Add sub-summary embeddings to vector store
        print(f"  Adding sub-summary embeddings to vector store...")
        vector_store.add_sub_summary_embeddings(sub_summary_embeddings, sub_summaries, linked_chunks)
    
    else:
        print(f"\n[Step 5/6] Skipping sub-summary generation (multi-vector RAG disabled)")
    
    # Step 6: Save vector store
    print(f"\n[Step 6/6] Saving vector store...")
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "vector_index")
    vector_store.save(index_path)
    
    # Print statistics
    stats = vector_store.get_stats()
    print("\n" + "=" * 80)
    print("RAG Index Built Successfully!")
    print("=" * 80)
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Total vectors in index: {stats['total_vectors']}")
    print(f"Index saved to: {index_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build RAG index from corpus PDFs')
    parser.add_argument('corpus_dir', help='Directory containing PDF files')
    parser.add_argument('--output-dir', default='rag_index', help='Output directory for index files')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size in tokens (default: 500)')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap in tokens (default: 50)')
    parser.add_argument('--no-sub-summaries', action='store_true', help='Disable sub-summary generation')
    parser.add_argument('--use-llm', action='store_true', help='Use full LLM for sub-summaries (requires model download)')
    parser.add_argument('--llm-model', default='meta-llama/Llama-2-7b-chat-hf', help='LLM model name')
    
    args = parser.parse_args()
    
    build_rag_index(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        generate_sub_summaries=not args.no_sub_summaries,
        use_llm=args.use_llm,
        llm_model=args.llm_model
    )

