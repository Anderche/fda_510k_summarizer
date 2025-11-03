"""
Flask Web Application for FDA 510(k) Chatbot

Provides a REST API endpoint for chat queries with conversation history support.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import List, Dict, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    load_dotenv(override=True)
except ImportError:
    pass

from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from query_pipeline import QueryPipeline
from llm_integration import LLMGenerator, SimpleLLMGenerator
from reference_formatter import format_reference_string

app = Flask(__name__, static_folder='static')
CORS(app)

# Global pipeline instance (loaded on startup)
pipeline: Optional[QueryPipeline] = None


def load_pipeline(index_dir: str = "rag_index", product_code: Optional[str] = None, 
                  use_llm: bool = True, llm_model: str = "claude-3-haiku-20240307"):
    """
    Load RAG system from saved index.
    
    Args:
        index_dir: Directory containing index files
        product_code: Product code for context
        use_llm: Whether to use full LLM
        llm_model: LLM model name if use_llm is True
    """
    global pipeline
    
    print(f"Loading RAG system from {index_dir}...")
    
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
            llm_generator = LLMGenerator(model_name=llm_model)
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
    
    stats = vector_store.get_stats()
    print(f"RAG system loaded. Vector store contains {stats['total_vectors']} vectors.")


def enhance_query_with_history(query: str, conversation_history: List[Dict[str, str]]) -> str:
    """
    Enhance query with conversation history for better context.
    
    Args:
        query: Current user query
        conversation_history: List of previous messages in format [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    
    Returns:
        Enhanced query string
    """
    if not conversation_history:
        return query
    
    # Build context from recent conversation (last 2 exchanges)
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    
    context_parts = []
    for msg in recent_history:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'user':
            context_parts.append(f"Previous question: {content}")
        elif role == 'assistant':
            # Extract summary from response if available
            response_text = content
            if 'SUMMARY:' in response_text:
                # Extract text after SUMMARY:
                summary = response_text.split('SUMMARY:')[-1].strip()[:200]
                context_parts.append(f"Previous answer summary: {summary}")
    
    if context_parts:
        context = " ".join(context_parts)
        # Append context to query for better retrieval
        enhanced_query = f"{query} [Context: {context}]"
        return enhanced_query
    
    return query


@app.route('/')
def index():
    """Serve the main chat interface."""
    return send_from_directory('static', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat queries with conversation history support.
    
    Expected JSON:
    {
        "message": "user query",
        "conversation_history": [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"}
        ]
    }
    
    Returns:
    {
        "response": "assistant response",
        "references": [...],
        "metadata": {...}
    }
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    user_message = data['message'].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    conversation_history = data.get('conversation_history', [])
    
    # Enhance query with conversation history for better context
    enhanced_query = enhance_query_with_history(user_message, conversation_history)
    
    try:
        # Process query
        result = pipeline.process_query(enhanced_query, k=5)
        
        # Extract response
        response_text = result['response']
        
        # Format references
        references = []
        for chunk in result.get('retrieved_chunks', [])[:5]:
            ref_str = format_reference_string(chunk, include_link=True)
            similarity = chunk.get('similarity_score', 0.0)
            references.append({
                'text': ref_str,
                'similarity': round(similarity, 3)
            })
        
        # Build response
        response_data = {
            "response": response_text,
            "references": references,
            "metadata": {
                "num_retrieved": result['metadata'].get('num_retrieved', 0),
                "has_refined_summary": bool(result.get('refined_summary'))
            }
        }
        
        # Include refined summary if available
        if result.get('refined_summary'):
            response_data['refined_summary'] = result['refined_summary']
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    })


if __name__ == '__main__':
    # Determine which index to use (check config.yaml or default)
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    index_dir = "rag_index"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                directory_to_vectorize = config.get('directory_to_vectorize', '')
                if directory_to_vectorize == 'corpus_ai_guidances':
                    index_dir = "rag_index_guidances"
        except Exception as e:
            print(f"Warning: Could not read config.yaml: {e}")
    
    # Load pipeline
    use_llm = os.getenv('ANTHROPIC_API_KEY') is not None
    load_pipeline(index_dir=index_dir, use_llm=use_llm)
    
    # Run Flask app
    # Default to 5001 to avoid conflict with macOS AirPlay on port 5000
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*60}")
    print(f"FDA 510(k) Chatbot Server")
    print(f"{'='*60}")
    print(f"Server running on http://localhost:{port}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

