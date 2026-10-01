# FDA 510(k) Summarizer - RAG System

A Retrieval-Augmented Generation (RAG) system for querying FDA 510(k) medical device submission documents. This application allows you to download, process, and intelligently query FDA 510(k) summaries using natural language questions.

## Overview

This system enables medical device specialists to:
- Download FDA 510(k) summary PDFs from the openFDA API
- Extract and process text from PDF documents
- Build a vector knowledge base using embeddings and FAISS
- Query documents using natural language with AI-powered responses via Claude API

## Architecture

The system consists of several key components:

1. **Data Fetcher** - Downloads PDFs from openFDA API (`get_510k_summary.py`)
2. **Text Extraction** - Extracts text from PDFs with header/footer filtering (`pdf_extractor.py`)
3. **Document Chunking** - Splits documents into semantic chunks (`chunker.py`)
4. **Embeddings** - Generates vector embeddings using sentence transformers (`embeddings.py`)
5. **Vector Store** - Stores embeddings in FAISS with chunk mappings (`vector_store.py`) *(FAISS used over Redis for local/offline use)*
6. **LLM Integration** - Generates responses using Claude API (`llm_integration.py`)
7. **RAG Pipeline** - End-to-end query processing (`query_pipeline.py`, `rag_retrieval.py`)

## Installation

1. Clone or navigate to the project directory:
```bash
cd app_fda_510k_summarizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Anthropic API key (required for Claude):
   
   Create a `.env` file in the project root:
   ```bash
   ANTHROPIC_API_KEY=your-api-key-here
   ```
   
   Get your API key from: https://console.anthropic.com/

## Usage

### Step 1: Download PDFs (Optional - if you don't have corpus yet)

If you need to download 510(k) PDFs first:
```bash
python get_510k_summary.py
```

Follow the prompts to:
- Select product code or device name
- Download PDFs to `corpus_{PRODUCT_CODE}/`

### Step 2: Build RAG Index

Build the vector knowledge base from your corpus:

```bash
python build_rag_index.py corpus_DXN --output-dir rag_index --use-llm
```

**Note:** `config.yaml` specifies `directory_to_vectorize`. For `corpus_ai_guidances`, system uses simplified regulatory consultant prompts with metadata only, no product codes.

**Parameters:**
- `corpus_DXN` - Directory containing PDF files
- `--output-dir` - Output directory for index files (default: `rag_index`)
- `--use-llm` - Use Claude API for generating sub-summaries (recommended for better quality)
- `--chunk-size` - Chunk size in tokens (default: 500)
- `--chunk-overlap` - Chunk overlap in tokens (default: 50)
- `--llm-model` - Claude model name (default: `claude-3-5-sonnet`)
- `--no-sub-summaries` - Disable multi-vector RAG (faster but less accurate)

**Example:**
```bash
# With Claude API (recommended)
python build_rag_index.py corpus_DXN --output-dir rag_index --use-llm

# Without Claude API (faster, template-based)
python build_rag_index.py corpus_DXN --output-dir rag_index
```

### Step 3: Query the RAG System

Query your knowledge base interactively:

```bash
python query_rag.py rag_index --product-code DXN --use-llm
```

**Parameters:**
- `rag_index` - Directory containing RAG index files
- `--product-code` - Product code for context in responses
- `--use-llm` - Use Claude API for generating responses (recommended)
- `--llm-model` - Claude model name (default: `claude-3-5-sonnet`)
- `--query` - Single query (non-interactive mode)
- `-k` - Number of chunks to retrieve (default: 5)

**Examples:**

Interactive mode:
```bash
python query_rag.py rag_index --product-code DXN --use-llm
```

Single query:
```bash
python query_rag.py rag_index --product-code DXN --use-llm --query "What tests are needed for 510k submission?"
```

## Quick Start Commands

**Build RAG Index:**
```bash
python build_rag_index.py corpus_DXN --output-dir rag_index --use-llm
```

**Test/Query:**
```bash
python query_rag.py rag_index --product-code DXN --use-llm
```

## How It Works

1. **Index Building:**
   - Extracts text from PDFs (using pdfplumber with PyMuPDF fallback)
   - Filters headers, footers, and noise
   - Chunks documents using sentence-aware splitting
   - Generates embeddings for chunks and sub-summaries
   - Stores in FAISS vector database with metadata mappings

2. **Query Processing:**
   - Embeds user query
   - Retrieves top-k similar chunks via vector search
   - Augments Claude prompt with retrieved context
   - Generates contextualized response

## Features

- **Multi-Vector RAG**: Uses sub-summaries for enhanced retrieval accuracy
- **Intelligent Text Extraction**: Handles complex PDF layouts and filters noise
- **Semantic Chunking**: Sentence-aware chunking preserves document structure
- **Claude API Integration**: High-quality AI responses via Anthropic's Claude
- **Fallback Mode**: Works without Claude API using template-based generation
- **Evaluation Engine**: ROUGE and BLEU metrics for evaluating generated summaries
- **FDA API Product Code Verification**: Automatic product code detection and FDA device classification lookup

### FDA API Product Code Verification

The system includes robust FDA Open API integration for automatic product code verification and device classification. When queries contain 3-character product codes (e.g., "NAY"), the system automatically validates these codes via the FDA's device classification API at `https://api.fda.gov/device/classification.json`. 

The verification process features comprehensive error handling with separate exception types for network failures, parsing errors, and missing data. The implementation validates that API responses contain properly structured data arrays before extraction, ensuring robust handling of various response scenarios. When device information is successfully retrieved, the system extracts both `device_name` and `medical_specialty_description` fields, enriching query responses with specific device classification context.

The code validates that `data['results']` is a list with items, extracts device information from the first result, and only returns data when `device_name` is present. Diagnostic warning messages provide clear feedback for different failure scenarios, helping users understand when API calls fail due to network issues, invalid product codes, or unexpected response structures. This enhancement significantly improves response accuracy by providing targeted regulatory context based on verified FDA device classifications.

## Evaluation

The system includes an evaluation engine for measuring summary quality using ROUGE and BLEU metrics.

### Using the Evaluation Engine

```python
from eval_engine import EvalEngine

# Initialize engine
engine = EvalEngine()

# Evaluate a single candidate-reference pair
candidate = "Generated summary text here..."
reference = "Reference/ground truth text here..."
results = engine.evaluate(candidate, reference)
print(engine.format_results(results))

# Batch evaluation
candidates = ["summary1", "summary2", "summary3"]
references = ["ref1", "ref2", "ref3"]
batch_results = engine.evaluate_batch(candidates, references)
print(engine.format_results(batch_results))
```

### Metrics

- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum with precision, recall, and F-measure
- **BLEU**: BLEU-1 through BLEU-4 n-gram precision scores and overall BLEU score

### CSV-Based Evaluation

The evaluation engine supports CSV-based evaluation where queries are processed through the RAG pipeline and compared against expected responses.

**Instructions:**
1. Create `eval_template.csv` with `query` and `expected_response` columns
2. Add your test queries and expected answers (one per row)
3. Run: `python run_eval_from_csv.py eval_template.csv --index-dir rag_index --use-llm`
4. Review ROUGE/BLEU scores comparing generated vs expected responses

**Example CSV format:**
```csv
query,expected_response
What tests are needed for 510k submission?,A 510(k) submission requires non-clinical bench testing...
What is substantial equivalence?,Substantial equivalence means the device has...
```

### Testing

Run the evaluation engine test script:
```bash
python test_eval_engine.py
```

## Configuration

### Config File

`config.yaml` specifies `directory_to_vectorize`. For `corpus_ai_guidances`, the system uses simplified regulatory consultant prompts (~80 words) with metadata only, excluding product codes.

### Environment Variables

The system reads from `.env` file or environment:

- `ANTHROPIC_API_KEY` - Required for Claude API access
- `LLM_MODEL` - Claude model name (default: `claude-3-haiku-20240307`)
- `USE_LLM` - `true`/`false` (default: `true`)
- `INDEX_DIR` - FAISS index directory to load at startup
- `REFINED_SUMMARY` - Set to `false` to skip the second Claude call during demos (default: `true`)
- `RESPONSE_CACHE_SIZE` - In-memory LRU of finished answers (default: `128`; `0` disables)

### Railway cold starts (free)

Railway may sleep the app when idle. A free [UptimeRobot](https://uptimerobot.com/) HTTP monitor against `/health` every 5 minutes will keep it awake. Keeping the replica awake spends Railway trial/usage hours, so if credits are tight leave sleeping on and accept one cold start after idle.

### Claude Models

Default: `claude-3-sonnet-20240229`

If you get 404 errors, try these known working models:
- `claude-3-sonnet-20240229` (recommended fallback)
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

**Note:** If models aren't available, the system will automatically fallback to template-based generation.

## File Structure

```
app_fda_510k_summarizer/
├── get_510k_summary.py      # Download PDFs from openFDA
├── build_rag_index.py        # Build RAG index
├── query_rag.py              # Query RAG system
├── eval_engine.py            # ROUGE/BLEU evaluation engine
├── eval_template.csv         # Template CSV for evaluation (query, expected_response)
├── run_eval_from_csv.py      # Run evaluation from CSV file
├── test_eval_engine.py       # Evaluation engine test script
├── pdf_extractor.py          # PDF text extraction
├── chunker.py                # Document chunking
├── embeddings.py             # Vector embeddings
├── vector_store.py           # FAISS vector store
├── llm_integration.py        # Claude API integration
├── rag_retrieval.py          # RAG retrieval logic
├── query_pipeline.py         # End-to-end query pipeline
├── requirements.txt          # Dependencies
├── .env                      # API keys (not in git)
└── corpus_*/                 # PDF corpus directories
```

## Troubleshooting

**No results found:**
- Ensure corpus contains PDFs with extractable text
- Check that chunks contain meaningful content (not just headers)

**Claude API errors:**
- Verify `ANTHROPIC_API_KEY` is set in `.env` file
- Check API key is valid and has credits
- System will fallback to simple generator if API fails

**Memory issues:**
- Use `--no-sub-summaries` to reduce memory usage
- Reduce `--chunk-size` for smaller vectors
- Process smaller corpora in batches

**Poor quality responses:**
- Ensure `--use-llm` flag is used for Claude API
- Rebuild index with `--use-llm` for better sub-summaries
- Increase `-k` to retrieve more context chunks

## License

This project is for educational/research purposes. Ensure compliance with FDA data usage policies and Anthropic API terms of service.

