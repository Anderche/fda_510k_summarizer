"""
FastAPI Web Server for FDA 510(k) Summarizer RAG System

Deployable web API for querying FDA 510(k) documents and guidance materials.
"""

import json
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Iterator, Tuple

# Import the RAG system components
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from query_rag import load_rag_system
from query_pipeline import QueryPipeline
from reference_formatter import format_multiple_references, pdf_page_url
from llm_integration import DEFAULT_LLM_MODEL, strip_response_prefix

# Global variables for loaded system
pipeline: Optional[QueryPipeline] = None
INDEX_DIR: Optional[str] = None
PRODUCT_CODE: Optional[str] = None
USE_LLM: bool = True

# Loaded pipelines keyed by index_dir, so each index is only loaded once
_pipelines: Dict[str, QueryPipeline] = {}
_pipelines_lock = threading.Lock()

# LRU cache of finished responses keyed by (index_dir, normalized query, k)
RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", "128"))
_response_cache: "OrderedDict[Tuple[str, str, int], Dict[str, Any]]" = OrderedDict()
_response_cache_lock = threading.Lock()


def _load_pipeline(index_dir: str) -> QueryPipeline:
    return load_rag_system(
        index_dir=index_dir,
        product_code=PRODUCT_CODE,
        use_llm=USE_LLM,
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    )


def get_pipeline(index_dir: Optional[str]) -> QueryPipeline:
    """Return the pipeline for index_dir, loading and caching it on first use."""
    if not index_dir or index_dir == INDEX_DIR:
        return pipeline
    with _pipelines_lock:
        if index_dir not in _pipelines:
            loaded = _load_pipeline(index_dir)
            loaded.warm_up()
            _pipelines[index_dir] = loaded
        return _pipelines[index_dir]


def _cache_key(index_dir: str, query: str, k: int) -> Tuple[str, str, int]:
    return (index_dir, " ".join(query.lower().split()), k)


def _cache_get(key: Tuple[str, str, int]) -> Optional[Dict[str, Any]]:
    with _response_cache_lock:
        cached = _response_cache.get(key)
        if cached is not None:
            _response_cache.move_to_end(key)
        return cached


def _cache_put(key: Tuple[str, str, int], value: Dict[str, Any]):
    if RESPONSE_CACHE_SIZE <= 0:
        return
    with _response_cache_lock:
        _response_cache[key] = value
        _response_cache.move_to_end(key)
        while len(_response_cache) > RESPONSE_CACHE_SIZE:
            _response_cache.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global pipeline, INDEX_DIR, PRODUCT_CODE, USE_LLM
    
    # Determine which index to load
    INDEX_DIR = os.getenv("INDEX_DIR", "rag_index")
    
    # Check if we should load guidance index
    if os.path.exists("rag_index_guidances"):
        # Load guidance index by default if it exists and no regular index
        if not os.path.exists(INDEX_DIR) or INDEX_DIR == "rag_index":
            INDEX_DIR = "rag_index_guidances"
            print(f"Loading guidance index: {INDEX_DIR}")
    
    PRODUCT_CODE = os.getenv("PRODUCT_CODE")
    USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
    
    # Check if ANTHROPIC_API_KEY is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if USE_LLM and not api_key:
        print("Warning: USE_LLM=true but ANTHROPIC_API_KEY not set. Will use simple generator.")
        USE_LLM = False
    
    try:
        print(f"Loading RAG system from {INDEX_DIR}...")
        pipeline = _load_pipeline(INDEX_DIR)
        _pipelines[INDEX_DIR] = pipeline
        pipeline.warm_up()
        print("RAG system loaded successfully!")
    except Exception as e:
        print(f"Error loading RAG system: {e}")
        pipeline = None
    
    yield
    
    # Shutdown (if needed in the future)
    # Cleanup code can go here


app = FastAPI(
    title="FDA 510(k) Summarizer API",
    description="RAG system for querying FDA 510(k) medical device documents",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about FDA documents")
    index_dir: Optional[str] = Field(None, description="Which index to use (defaults to configured index)")
    k: int = Field(5, description="Number of results to retrieve")
    use_guidances: bool = Field(False, description="Use AI guidance documents instead of 510k documents")


class Reference(BaseModel):
    display_title: Optional[str] = None
    file_name: Optional[str] = None
    k_number: Optional[str] = None
    page_num: Optional[int] = None
    para_index: Optional[int] = None
    pdf_link: Optional[str] = None
    guidance_type: Optional[str] = None
    text: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    response: str
    refined_summary: Optional[str] = None
    references: List[Reference]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_dir: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>FDA 510(k) Summarizer API</h1><p>Frontend coming soon. Use /api/info for API documentation.</p>")


@app.get("/api/health", response_model=HealthResponse)
async def api_health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "not_initialized",
        index_loaded=pipeline is not None,
        index_dir=INDEX_DIR
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (alias for /api/health)"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "not_initialized",
        index_loaded=pipeline is not None,
        index_dir=INDEX_DIR
    )


def _format_references(references: List[Dict[str, Any]]) -> List[Reference]:
    return [
        Reference(
            display_title=ref_dict.get('display_title'),
            file_name=ref_dict.get('file_name'),
            k_number=ref_dict.get('k_number'),
            page_num=ref_dict.get('page_num'),
            para_index=ref_dict.get('para_index'),
            pdf_link=pdf_page_url(ref_dict.get('pdf_link'), ref_dict.get('page_num')),
            guidance_type=ref_dict.get('guidance_type'),
            text=ref_dict.get('text'),
        )
        for ref_dict in references
    ]


def _resolve_pipeline(request: QueryRequest) -> Tuple[str, QueryPipeline]:
    """Return (index_dir, pipeline) for a request, raising HTTPException on failure."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check server logs."
        )
    
    index_dir = request.index_dir if request.index_dir else INDEX_DIR
    try:
        return index_dir, get_pipeline(index_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load index {index_dir}: {str(e)}"
        )


# Plain `def` so FastAPI runs the blocking RAG/LLM work in its threadpool instead of the event loop
@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    
    Args:
        request: Query request with the question and options
        
    Returns:
        Query response with answer and references
    """
    index_dir, query_pipeline = _resolve_pipeline(request)
    
    key = _cache_key(index_dir, request.query, request.k)
    cached = _cache_get(key)
    if cached is not None:
        return QueryResponse(**{**cached, 'metadata': {**cached['metadata'], 'cache_hit': True}})
    
    try:
        # Process the query
        result = query_pipeline.process_query(
            query=request.query,
            k=request.k,
            min_similarity=0.0
        )
        
        # Build response
        response = QueryResponse(
            query=result['query'],
            response=strip_response_prefix(result.get('response', result.get('summary', ''))),
            refined_summary=result.get('refined_summary'),
            references=_format_references(result.get('references', [])),
            metadata=result.get('metadata', {})
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
    
    if result.get('references'):
        _cache_put(key, response.model_dump())
    return response


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_query_events(request: QueryRequest, index_dir: str, query_pipeline: QueryPipeline) -> Iterator[str]:
    """
    Server-Sent Events for one query:
    references -> interleaved response/refined text deltas -> done (with metadata).
    """
    key = _cache_key(index_dir, request.query, request.k)
    cached = _cache_get(key)
    if cached is not None:
        yield _sse("references", {"references": cached['references']})
        yield _sse("response", {"text": strip_response_prefix(cached['response'])})
        yield _sse("done", {"metadata": {**cached['metadata'], 'cache_hit': True}})
        return
    
    timings: Dict[str, float] = {}
    total_start = time.perf_counter()
    try:
        retrieval = query_pipeline.retrieve_only(request.query, k=request.k)
    except Exception as e:
        yield _sse("error", {"detail": f"Error processing query: {str(e)}"})
        return
    timings['retrieval'] = round((time.perf_counter() - total_start) * 1000, 1)
    
    retrieved = retrieval['retrieved_chunks']
    references = [ref.model_dump() for ref in _format_references(
        format_multiple_references(retrieved, format_type="dict")
    )]
    yield _sse("references", {"references": references})
    
    metadata: Dict[str, Any] = {
        'num_retrieved': len(retrieved),
        'k': request.k,
        'method': 'custom_pipeline_stream',
        'similarity_scores': [chunk.get('similarity_score', 0.0) for chunk in retrieved],
        'queries_used': retrieval['queries_used'],
        'sections_matched': retrieval['sections_used'],
        'timings_ms': timings
    }
    
    if not retrieved:
        yield _sse("response", {"text": "No relevant documents found for your query."})
        timings['total'] = round((time.perf_counter() - total_start) * 1000, 1)
        yield _sse("done", {"metadata": metadata})
        return
    
    # Stream the main answer only; skip the extra refined-summary call in the demo UI
    events: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
    producers = {
        'response': lambda: query_pipeline.stream_answer(request.query, retrieval),
    }
    
    def produce(name: str, make_stream):
        start = time.perf_counter()
        try:
            for text in make_stream():
                if f'{name}_first_token' not in timings:
                    timings[f'{name}_first_token'] = round((time.perf_counter() - start) * 1000, 1)
                events.put((name, text))
        except Exception as e:
            print(f"Error streaming {name}: {e}")
        finally:
            timings[f'llm_{name}'] = round((time.perf_counter() - start) * 1000, 1)
            events.put((name, None))
    
    for name, make_stream in producers.items():
        threading.Thread(target=produce, args=(name, make_stream), daemon=True).start()
    
    collected = {name: [] for name in producers}
    remaining = len(producers)
    while remaining:
        name, text = events.get()
        if text is None:
            remaining -= 1
            continue
        collected[name].append(text)
        yield _sse(name, {"text": text})
    
    timings['total'] = round((time.perf_counter() - total_start) * 1000, 1)
    
    response_text = strip_response_prefix("".join(collected['response']))
    if response_text:
        _cache_put(key, {
            'query': request.query,
            'response': response_text,
            'refined_summary': None,
            'references': references,
            'metadata': metadata
        })
    yield _sse("done", {"metadata": metadata})


@app.post("/query/stream")
def query_documents_stream(request: QueryRequest):
    """
    Stream a query answer as Server-Sent Events.
    
    Events: `references`, `response` (text delta), `done` (metadata), `error`.
    """
    index_dir, query_pipeline = _resolve_pipeline(request)
    return StreamingResponse(
        _stream_query_events(request, index_dir, query_pipeline),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "name": "FDA 510(k) Summarizer API",
        "version": "1.0.0",
        "description": "RAG system for querying FDA medical device documents",
        "index_dir": INDEX_DIR,
        "product_code": PRODUCT_CODE,
        "use_llm": USE_LLM,
        "initialized": pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )

