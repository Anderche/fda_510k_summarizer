# System Architecture

## Overview

This document describes the architecture for a 510(k) summarizer system that uses Multi-Vector RAG (Retrieval-Augmented Generation) to help users query FDA 510(k) submission summaries.

---

## Components

### 1. Data Fetcher

The data fetcher component handles the initial data collection and processing:

- **Query openFDA**: Retrieves 510(k) records filtered by product code
- **Filter summaries**: Identifies records that contain summaries
- **Download PDFs**: Fetches PDF documents from FDA sources
- **Extract text**: Converts PDF content to plain text format
- **Build corpus**: Assembles the text corpus for downstream processing

### 2. Corpus Processing for Multi-Vector RAG

The corpus processing pipeline transforms raw text into a searchable vector database:

1. **Chunking**: 
   - Split summary texts into manageable segments (e.g., 500-token chunks)
   - Maintains context and structure of original documents

2. **Sub-summary Generation**:
   - Generate sub-summaries or hypotheticals using an LLM
   - Example format: `"Key safety data from this 510(k): [summary]"`
   - Creates additional semantic representations for better retrieval

3. **Embedding**:
   - Embed both chunks and sub-summaries using a semantic model
   - Recommended model: `all-MiniLM-L6-v2` or similar

4. **Vector Storage**:
   - Store embeddings in a vector database (FAISS)
   - Maintain mappings: `summary vectors → full chunk IDs`
   - Enables efficient similarity search and retrieval

### 3. Query Pipeline

The query pipeline processes user questions and generates responses:

#### Process Flow

```
User Query
    ↓
Embed Query
    ↓
Retrieve top-k summary vectors
    ↓
Fetch linked full chunks as context
    ↓
Augment LLM prompt with context
    ↓
Generate response
```

#### Implementation Details

1. **Query Embedding**: 
   - Convert user query to embedding space
   - Use same embedding model as corpus processing

2. **Retrieval**:
   - Retrieve top-k summary vectors using similarity search
   - Fetch corresponding full chunks based on vector mappings

3. **Context Augmentation**:
   - Build prompt template with retrieved context
   - Example format: `"Based on these approved 510(k) summaries for product code [code], advise on writing a submission: [query]"`

4. **Response Generation**:
   - Use open-source LLM for generation
   - Recommended: `Llama-2-7B` via Hugging Face Transformers
   - Generate contextualized responses based on retrieved documents

### 4. Deployment

The system can be deployed in multiple ways:

- **Script-based**: Run as a standalone Python script for batch processing
- **Web Application**: Deploy as an interactive web app using:
  - [Streamlit](https://streamlit.io/) - Fast prototyping and deployment
  - [Gradio](https://gradio.app/) - Easy ML model interfaces
  - Enables real-time specialist queries and exploration

---

## Technical Stack

### Recommended Tools & Libraries

- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM**: Llama-2-7B or similar via Hugging Face Transformers
- **PDF Processing**: PyPDF2, pdfplumber, or similar
- **API Integration**: requests, httpx for openFDA queries
- **Web Framework**: Streamlit or Gradio (optional)

---

## Data Flow Summary

```
openFDA API
    ↓
PDF Downloads
    ↓
Text Extraction
    ↓
Chunking + Sub-summary Generation
    ↓
Embedding (Chunks + Sub-summaries)
    ↓
FAISS Vector Store
    ↓
Query Processing → Retrieval → LLM Response
```
