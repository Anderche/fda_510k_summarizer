"""
Build RAG Index Script

Orchestrates the complete pipeline to build the RAG index from corpus PDFs.
"""

import os
import argparse
import yaml

# Load .env file if available
import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from dotenv import load_dotenv
    import os
    # Load from project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
    # Also try loading from current working directory
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading .env

from pdf_extractor import extract_text_from_corpus
from chunker import DocumentChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from llm_integration import LLMGenerator, SimpleLLMGenerator


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to config.yaml in config directory)
        
    Returns:
        Dictionary with configuration settings
    """
    if config_path is None:
        # Default to config.yaml in config directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        config_path = project_root / "config" / "config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def detect_source_type(corpus_dir: str) -> str:
    """
    Detect source type from directory name.
    
    Args:
        corpus_dir: Directory path
        
    Returns:
        Source type: 'corpus_ai_guidances' or '510k'
    """
    dir_name = os.path.basename(os.path.normpath(corpus_dir))
    if dir_name == 'corpus_ai_guidances':
        return 'corpus_ai_guidances'
    return '510k'


def build_rag_index(corpus_dir: str = None, output_dir: str = "rag_index", 
                   chunk_size: int = 800, chunk_overlap: int = 100,
                   generate_sub_summaries: bool = True,
                   use_llm: bool = False, llm_model: str = "claude-3-sonnet-20240229", api_key: str = None,
                   source_type: str = None):
    """
    Build complete RAG index from corpus.
    
    Args:
        corpus_dir: Directory containing PDF files (optional if config.yaml exists)
        output_dir: Directory to save index files
        chunk_size: Target chunk size in tokens
        chunk_overlap: Chunk overlap in tokens
        generate_sub_summaries: Whether to generate sub-summaries for multi-vector RAG
        use_llm: Whether to use full LLM (requires model download) or simple generator
        llm_model: LLM model name if use_llm is True
        api_key: API key for LLM
        source_type: Source type ('corpus_ai_guidances' or '510k'), auto-detected if None
    """
    # Load config if corpus_dir not provided
    config = load_config()
    if not corpus_dir:
        corpus_dir = config.get('directory_to_vectorize')
        if not corpus_dir:
            raise ValueError("corpus_dir must be provided either as argument or in config.yaml as 'directory_to_vectorize'")
    
    # Detect source type
    if not source_type:
        source_type = detect_source_type(corpus_dir)
    
    print("=" * 80)
    print("Building RAG Index")
    print("=" * 80)
    print(f"Source directory: {corpus_dir}")
    print(f"Source type: {source_type}")
    
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
    # Store source type in vector store metadata
    vector_store.source_type = source_type
    vector_store.add_embeddings(chunk_embeddings, chunks)
    
    # Step 5: Generate sub-summaries and embeddings (multi-vector RAG)
    if generate_sub_summaries:
        print(f"\n[Step 5/6] Generating sub-summaries for multi-vector RAG...")
        
        # Initialize LLM generator
        if use_llm:
            print(f"  Using Claude API model: {llm_model}")
            try:
                # Re-ensure .env is loaded
                try:
                    from dotenv import load_dotenv
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    env_path = os.path.join(script_dir, '.env')
                    if os.path.exists(env_path):
                        load_dotenv(dotenv_path=env_path, override=True)
                    load_dotenv(override=True)
                except Exception as e:
                    print(f"  Warning loading .env: {e}")
                llm = LLMGenerator(model_name=llm_model, api_key=api_key)
                # Test the model with a simple call
                test_result = llm.generate_sub_summary("Test", k_number="TEST")
                if not test_result or "Error" in test_result:
                    raise ValueError("Model test failed")
            except Exception as e:
                error_msg = str(e)
                print(f"  Error initializing Claude API: {error_msg}")
                if 'not found' in error_msg.lower() or '404' in error_msg:
                    print(f"  Suggested fix: Use --llm-model claude-3-sonnet-20240229")
                print(f"  Falling back to simple template-based generator")
                print(f"  (You can skip LLM with --no-sub-summaries to build faster)")
                llm = SimpleLLMGenerator()
        else:
            print("  Using simple template-based sub-summary generator")
            llm = SimpleLLMGenerator()
        
        sub_summaries = []
        linked_chunks = []
        
        failed_count = 0
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            try:
                k_number = chunk.get('metadata', {}).get('k_number', None)
                sub_summary_text = llm.generate_sub_summary(chunk['text'], k_number=k_number)
                
                sub_summaries.append({
                    'text': sub_summary_text,
                    'source_chunk_index': chunk.get('chunk_index', i)
                })
                linked_chunks.append(chunk)
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # Only show first few errors
                    print(f"  Warning: Failed to generate sub-summary for chunk {i+1}: {e}")
                # Fallback: use chunk text itself as sub-summary
                sub_summaries.append({
                    'text': f"Key information from this 510(k) document: {chunk['text'][:200]}...",
                    'source_chunk_index': chunk.get('chunk_index', i)
                })
                linked_chunks.append(chunk)
        
        if failed_count > 0:
            print(f"  Warning: {failed_count} sub-summaries failed to generate, using fallback")
        
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
    parser.add_argument('corpus_dir', nargs='?', default=None, help='Directory containing PDF files (optional if in config.yaml)')
    parser.add_argument('--output-dir', default='rag_index', help='Output directory for index files')
    parser.add_argument('--chunk-size', type=int, default=800, help='Chunk size in tokens (default: 800 for more chunks per PDF)')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap in tokens (default: 100)')
    parser.add_argument('--no-sub-summaries', action='store_true', help='Disable sub-summary generation')
    parser.add_argument('--use-llm', action='store_true', help='Use Claude API for sub-summaries (requires ANTHROPIC_API_KEY)')
    parser.add_argument('--llm-model', default='claude-3-sonnet-20240229', help='Claude model name')
    parser.add_argument('--api-key', default=None, help='Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    build_rag_index(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        generate_sub_summaries=not args.no_sub_summaries,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        api_key=args.api_key
    )

