"""
Query RAG System Script

Interactive script to query the built RAG index.
"""

import os
import argparse

# Load .env file if available
try:
    from dotenv import load_dotenv
    import os
    # Load from project root
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(dotenv_path=env_path)
    # Also try loading from current working directory
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading .env
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from query_pipeline import QueryPipeline
from llm_integration import LLMGenerator, SimpleLLMGenerator


def load_rag_system(index_dir: str, product_code: str = None, use_llm: bool = False, 
                    llm_model: str = "claude-3-5-sonnet-20241022", api_key: str = None):
    """
    Load RAG system from saved index.
    
    Args:
        index_dir: Directory containing index files
        product_code: Product code for context
        use_llm: Whether to use full LLM
        llm_model: LLM model name if use_llm is True
        
    Returns:
        QueryPipeline instance
    """
    print("Loading RAG system...")
    
    # Load vector store
    index_path = os.path.join(index_dir, "vector_index")
    vector_store = VectorStore()
    vector_store.load(index_path)
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Initialize LLM generator
    if use_llm:
        try:
            print(f"Initializing Claude API with model: {llm_model}")
            # Re-ensure .env is loaded before initializing
            try:
                from dotenv import load_dotenv
                script_dir = os.path.dirname(os.path.abspath(__file__))
                env_path = os.path.join(script_dir, '.env')
                if os.path.exists(env_path):
                    load_dotenv(dotenv_path=env_path, override=True)
                    print(f"Loaded .env from: {env_path}")
                cwd_env = os.path.join(os.getcwd(), '.env')
                if os.path.exists(cwd_env):
                    load_dotenv(dotenv_path=cwd_env, override=True)
                    print(f"Loaded .env from: {cwd_env}")
                load_dotenv(override=True)
                # Debug: check if key was loaded
                api_key_from_env = os.getenv("ANTHROPIC_API_KEY")
                if api_key_from_env:
                    print(f"API key loaded from environment (length: {len(api_key_from_env)})")
                else:
                    print("Warning: ANTHROPIC_API_KEY not found in environment after loading .env")
            except Exception as e:
                print(f"Warning: Error loading .env: {e}")
            llm_generator = LLMGenerator(model_name=llm_model, api_key=api_key)
        except Exception as e:
            print(f"Error initializing Claude API: {e}")
            print("Falling back to simple template-based generator")
            llm_generator = SimpleLLMGenerator()
    else:
        print("Using simple template-based generator (no Claude API)")
        llm_generator = SimpleLLMGenerator()
    
    # Create query pipeline
    pipeline = QueryPipeline(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        llm_generator=llm_generator,
        product_code=product_code
    )
    
    print(f"RAG system loaded. Vector store contains {vector_store.get_stats()['total_vectors']} vectors.")
    
    return pipeline


def interactive_query(pipeline: QueryPipeline):
    """Interactive query interface."""
    print("\n" + "=" * 80)
    print("FDA 510(k) RAG Query System")
    print("=" * 80)
    print("Enter your questions about FDA 510(k) submissions.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nProcessing query...")
        result = pipeline.process_query(query, k=5)
        
        print("\n" + "-" * 80)
        print("Response:")
        print("-" * 80)
        print(result['response'])
        print("\n" + "-" * 80)
        print(f"Retrieved {result['metadata']['num_retrieved']} relevant documents")
        if result['retrieved_chunks']:
            print("Top sources:")
            for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
                k_number = chunk.get('metadata', {}).get('k_number', 'Unknown')
                similarity = chunk.get('similarity_score', 0.0)
                print(f"  {i}. K-number: {k_number}, Similarity: {similarity:.3f}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query FDA 510(k) RAG system')
    parser.add_argument('index_dir', help='Directory containing RAG index files')
    parser.add_argument('--product-code', help='Product code for context')
    parser.add_argument('--use-llm', action='store_true', help='Use Claude API for responses (requires ANTHROPIC_API_KEY)')
    parser.add_argument('--llm-model', default='claude-3-5-sonnet-20241022', help='Claude model name')
    parser.add_argument('--api-key', default=None, help='Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')
    parser.add_argument('--query', help='Single query to process (non-interactive mode)')
    parser.add_argument('-k', type=int, default=5, help='Number of chunks to retrieve')
    
    args = parser.parse_args()
    
    pipeline = load_rag_system(
        index_dir=args.index_dir,
        product_code=args.product_code,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        api_key=args.api_key
    )
    
    if args.query:
        # Single query mode
        result = pipeline.process_query(args.query, k=args.k)
        print("\nQuery:", result['query'])
        print("\nResponse:")
        print(result['response'])
        print(f"\nRetrieved {result['metadata']['num_retrieved']} documents")
    else:
        # Interactive mode
        interactive_query(pipeline)

