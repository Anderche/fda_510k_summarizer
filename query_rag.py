"""
Query RAG System Script

Interactive script to query the built RAG index.
"""

import os
import argparse
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from query_pipeline import QueryPipeline
from llm_integration import LLMGenerator, SimpleLLMGenerator


def load_rag_system(index_dir: str, product_code: str = None, use_llm: bool = False, 
                    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"):
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
        llm_generator = LLMGenerator(model_name=llm_model)
    else:
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
    parser.add_argument('--use-llm', action='store_true', help='Use full LLM for responses')
    parser.add_argument('--llm-model', default='meta-llama/Llama-2-7b-chat-hf', help='LLM model name')
    parser.add_argument('--query', help='Single query to process (non-interactive mode)')
    parser.add_argument('-k', type=int, default=5, help='Number of chunks to retrieve')
    
    args = parser.parse_args()
    
    pipeline = load_rag_system(
        index_dir=args.index_dir,
        product_code=args.product_code,
        use_llm=args.use_llm,
        llm_model=args.llm_model
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

