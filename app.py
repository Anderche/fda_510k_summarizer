"""
FastAPI Web Server for FDA 510(k) Summarizer RAG System

Deployable web API for querying FDA 510(k) documents and guidance materials.
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Import the RAG system components
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from query_rag import load_rag_system
from query_pipeline import QueryPipeline
from reference_formatter import format_multiple_references

# Global variables for loaded system
pipeline: Optional[QueryPipeline] = None
INDEX_DIR: Optional[str] = None
PRODUCT_CODE: Optional[str] = None
USE_LLM: bool = True


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
        pipeline = load_rag_system(
            index_dir=INDEX_DIR,
            product_code=PRODUCT_CODE,
            use_llm=USE_LLM,
            llm_model=os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
        )
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


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    
    Args:
        request: Query request with the question and options
        
    Returns:
        Query response with answer and references
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check server logs."
        )
    
    # Determine which index to use
    index_dir = request.index_dir if request.index_dir else INDEX_DIR
    
    # Check if we need to load a different index
    if request.index_dir and request.index_dir != INDEX_DIR:
        try:
            # Load the requested index temporarily
            temp_pipeline = load_rag_system(
                index_dir=index_dir,
                product_code=PRODUCT_CODE,
                use_llm=USE_LLM,
                llm_model=os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
            )
            query_pipeline = temp_pipeline
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load index {index_dir}: {str(e)}"
            )
    else:
        query_pipeline = pipeline
    
    try:
        # Process the query
        result = query_pipeline.process_query(
            query=request.query,
            k=request.k,
            min_similarity=0.0
        )
        
        # Format references
        formatted_refs = []
        for ref_dict in result.get('references', []):
            formatted_refs.append(Reference(
                display_title=ref_dict.get('display_title'),
                file_name=ref_dict.get('file_name'),
                k_number=ref_dict.get('k_number'),
                page_num=ref_dict.get('page_num'),
                para_index=ref_dict.get('para_index'),
                pdf_link=ref_dict.get('pdf_link')
            ))
        
        # Build response
        response = QueryResponse(
            query=result['query'],
            response=result.get('response', result.get('summary', '')),
            refined_summary=result.get('refined_summary'),
            references=formatted_refs,
            metadata=result.get('metadata', {})
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
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

