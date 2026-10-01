"""
Evaluate RAG system from CSV file.

Reads queries and expected responses from eval_template.csv,
processes queries through RAG pipeline, and computes ROUGE/BLEU metrics.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project directories to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
config_path = project_root / "config"
sys.path.insert(0, str(config_path))
sys.path.insert(0, str(src_path))

from eval_engine import EvalEngine
from query_rag import load_rag_system
from llm_integration import DEFAULT_LLM_MODEL


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system from CSV file')
    csv_default = project_root / "config" / "eval_template.csv"
    parser.add_argument('csv_path', default=str(csv_default), nargs='?',
                       help='Path to CSV file with query and expected_response columns')
    parser.add_argument('--index-dir', default='rag_index',
                       help='Directory containing RAG index files')
    parser.add_argument('--product-code', help='Product code for context')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use Claude API for responses (requires ANTHROPIC_API_KEY)')
    parser.add_argument('--llm-model', default=DEFAULT_LLM_MODEL,
                       help='Claude model name')
    parser.add_argument('--api-key', default=None,
                       help='Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')
    parser.add_argument('-k', type=int, default=5,
                       help='Number of chunks to retrieve for each query')
    parser.add_argument('--metrics', nargs='+', choices=['rouge', 'bleu'],
                       default=['rouge', 'bleu'],
                       help='Metrics to compute (default: rouge bleu)')
    
    args = parser.parse_args()
    
    # Load RAG system
    print(f"Loading RAG system from {args.index_dir}...")
    pipeline = load_rag_system(
        index_dir=args.index_dir,
        product_code=args.product_code,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        api_key=args.api_key
    )
    
    # Initialize evaluation engine
    print(f"Initializing evaluation engine...")
    engine = EvalEngine()
    
    # Run evaluation from CSV
    print(f"\nEvaluating from CSV: {args.csv_path}")
    print("=" * 80)
    
    try:
        results = engine.evaluate_from_csv(
            csv_path=args.csv_path,
            query_pipeline=pipeline,
            k=args.k,
            metrics=args.metrics
        )
        
        # Print results
        print(engine.format_results(results))
        
        # Print individual query results
        if 'queries' in results:
            print("\n" + "=" * 80)
            print("Individual Query Results:")
            print("=" * 80)
            for i, (query, candidate, ref) in enumerate(zip(
                results['queries'],
                results['candidates'],
                [r for r in results['individual_scores']]
            ), 1):
                print(f"\n[{i}] Query: {query}")
                print(f"    Candidate: {candidate[:200]}...")
                if 'rouge' in ref and 'error' not in ref['rouge']:
                    rouge_l_f = ref['rouge'].get('rougeL', {}).get('fmeasure', 0.0)
                    print(f"    ROUGE-L F1: {rouge_l_f:.4f}")
                if 'bleu' in ref and 'error' not in ref['bleu']:
                    bleu = ref['bleu'].get('bleu', 0.0)
                    print(f"    BLEU: {bleu:.4f}")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

