"""
Query Pipeline Module

End-to-end query processing using LangChain retrieval chains.
"""

import os
# Set tokenizer parallelism early to avoid warnings
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to load from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterator, List, Optional, Any, Set, Tuple
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_retrieval import RAGRetriever
from llm_integration import GUIDANCE_MAX_TOKENS, LLMGenerator, RESPONSE_MAX_TOKENS, SimpleLLMGenerator
from reference_formatter import format_multiple_references
from query_enhancement import QueryEnhancer
from ner_tfidf_extractor import NERTFIDFExtractor

try:
    from langchain.chains import RetrievalQA
    from langchain.chains.retrieval_qa.base import BaseRetrievalQAChain
    from langchain_core.prompts import PromptTemplate
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not fully available. Some features may be limited.")


class QueryPipeline:
    """
    Complete query processing pipeline using LangChain retrieval chains.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator, 
                 llm_generator: Optional[LLMGenerator] = None, product_code: Optional[str] = None,
                 use_query_expansion: bool = True, use_multi_query: bool = True,
                 use_langchain_chain: bool = True, term_cache_path: Optional[str] = None):
        """
        Initialize query pipeline.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            llm_generator: LLM generator instance (optional, will use SimpleLLMGenerator if None)
            product_code: Product code for context in responses
            use_query_expansion: Whether to use query expansion
            use_multi_query: Whether to use multi-query generation
            use_langchain_chain: Whether to use LangChain RetrievalQA chain
            term_cache_path: Optional .npz path for persisted query-expansion term embeddings
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_generator = llm_generator or SimpleLLMGenerator()
        self.product_code = product_code
        self.retriever = RAGRetriever(vector_store, embedding_generator)
        self.use_query_expansion = use_query_expansion
        self.use_multi_query = use_multi_query
        self.use_langchain_chain = use_langchain_chain and LANGCHAIN_AVAILABLE
        self.query_enhancer = QueryEnhancer(embedding_generator, vector_store, term_cache_path=term_cache_path)
        if self.use_query_expansion:
            self.query_enhancer._initialize_term_cache()
        self.ner_tfidf_extractor = NERTFIDFExtractor()
        self.refined_summary_enabled = os.getenv("REFINED_SUMMARY", "true").lower() == "true"
        
        # Initialize LangChain chain if enabled and available
        self.langchain_qa_chain: Optional[BaseRetrievalQAChain] = None
        if self.use_langchain_chain:
            self._initialize_langchain_chain()
    
    def _initialize_langchain_chain(self):
        """Initialize LangChain RetrievalQA chain."""
        try:
            # Get LangChain retriever
            langchain_retriever = self.retriever.retriever
            if langchain_retriever is None:
                print("Warning: LangChain retriever not available, falling back to custom pipeline")
                self.use_langchain_chain = False
                return
            
            # Create LLM for chain
            llm = None
            if hasattr(self.llm_generator, 'client') and hasattr(self.llm_generator, 'model_name'):
                try:
                    source_type = getattr(self.vector_store, 'source_type', None)
                    llm = ChatAnthropic(
                        model=self.llm_generator.model_name,
                        temperature=0.7,
                        max_tokens=GUIDANCE_MAX_TOKENS if source_type == 'corpus_ai_guidances' else RESPONSE_MAX_TOKENS
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize LangChain Anthropic LLM: {e}")
                    self.use_langchain_chain = False
                    return
            
            if llm is None:
                print("Warning: LLM not available for LangChain chain, falling back to custom pipeline")
                self.use_langchain_chain = False
                return
            
            # Create custom prompt template
            source_type = getattr(self.vector_store, 'source_type', None)
            if source_type == 'corpus_ai_guidances':
                prompt_template = """You are a regulatory affairs consultant specializing in FDA AI regulations, standards, and guidances.

Use the following pieces of context from FDA AI guidance documents to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Follow any formatting requested in the question. If the question asks to list, itemize, enumerate, or give steps, answer with a Markdown list ("- item" or "1. item"), one item per line. Do not collapse those items into a paragraph. Otherwise write a concise, focused answer based on the context. Focus on regulatory requirements, standards, and guidance specific to artificial intelligence in medical devices.

Answer:"""
            else:
                prompt_template = """You are a helpful assistant that answers questions about FDA 510(k) medical device submissions based on retrieved document excerpts.

Use the following pieces of context from FDA 510(k) documents to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Follow any formatting requested in the question. If the question asks to list, itemize, enumerate, or give steps, answer with a Markdown list ("- item" or "1. item"), one item per line. Otherwise provide a clear and concise answer based on the context. Include relevant details from the documents.

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create RetrievalQA chain
            self.langchain_qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=langchain_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("LangChain RetrievalQA chain initialized successfully")
            
        except Exception as e:
            print(f"Warning: Failed to initialize LangChain chain: {e}")
            self.use_langchain_chain = False
    
    def _merge_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from multiple queries.
        
        Args:
            all_results: List of result dictionaries with 'retrieved_chunks'
            
        Returns:
            Merged and deduplicated chunks
        """
        seen_chunk_ids: Set[str] = set()
        merged_chunks: List[Dict[str, Any]] = []
        chunk_scores: Dict[str, float] = {}
        
        for result in all_results:
            for chunk in result.get('retrieved_chunks', []):
                # Create unique ID from chunk metadata
                metadata = chunk.get('metadata', {})
                chunk_id = f"{metadata.get('k_number', 'unknown')}_{chunk.get('chunk_index', 'unknown')}"
                
                similarity = chunk.get('similarity_score', 0.0)
                
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_chunks.append(chunk)
                    chunk_scores[chunk_id] = similarity
                else:
                    # Take max similarity if duplicate
                    chunk_scores[chunk_id] = max(chunk_scores[chunk_id], similarity)
        
        # Sort by similarity score (using the max score for each chunk)
        merged_chunks.sort(key=lambda c: chunk_scores.get(
            f"{c.get('metadata', {}).get('k_number', 'unknown')}_{c.get('chunk_index', 'unknown')}", 
            0.0
        ), reverse=True)
        
        return merged_chunks
    
    def process_query(self, query: str, k: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """
        Process a user query and generate a response using LangChain retrieval chain or custom pipeline.
        
        Args:
            query: User's query/question
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with 'query', 'retrieved_chunks', 'response', and 'metadata'
        """
        # Use LangChain chain if available
        if self.use_langchain_chain and self.langchain_qa_chain:
            return self._process_with_langchain_chain(query, k, min_similarity)
        else:
            # Fallback to custom pipeline
            return self._process_with_custom_pipeline(query, k, min_similarity)
    
    @property
    def refined_summary_supported(self) -> bool:
        """Whether a refined summary (second LLM call) will be generated."""
        return self.refined_summary_enabled and hasattr(self.llm_generator, 'generate_refined_summary')
    
    def _generate_refined_summary(self, query: str, context_chunks: List[Dict[str, Any]]) -> Optional[str]:
        """Generate the refined summary, returning None on any failure."""
        if not self.refined_summary_supported or not context_chunks:
            return None
        try:
            return self.llm_generator.generate_refined_summary(
                query=query,
                context_chunks=context_chunks,
                vector_store=self.vector_store,
                ner_tfidf_extractor=self.ner_tfidf_extractor,
                product_code=self.product_code
            )
        except Exception as e:
            print(f"Error generating refined summary: {e}")
            return None
    
    def _answer_with_refined_summary(self, answer_fn: Callable[[], str], query: str,
                                     context_chunks: List[Dict[str, Any]],
                                     timings: Dict[str, float]) -> Tuple[str, Optional[str]]:
        """Run the main answer and the refined summary LLM calls concurrently."""
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            start = time.perf_counter()
            try:
                return fn()
            finally:
                timings[name] = round((time.perf_counter() - start) * 1000, 1)
        
        with ThreadPoolExecutor(max_workers=2) as pool:
            refined_future = None
            if self.refined_summary_supported and context_chunks:
                refined_future = pool.submit(timed, 'llm_refined',
                                             lambda: self._generate_refined_summary(query, context_chunks))
            answer = timed('llm_response', answer_fn)
            refined_summary = refined_future.result() if refined_future else None
        return answer, refined_summary
    
    def _process_with_langchain_chain(self, query: str, k: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """Process query using LangChain RetrievalQA chain."""
        try:
            timings: Dict[str, float] = {}
            total_start = time.perf_counter()
            
            # Retrieve first so the refined summary can run alongside the answer
            retriever = self.retriever.retriever
            if retriever is None:
                raise ValueError("LangChain retriever is not available")
            if hasattr(retriever, 'search_kwargs'):
                retriever.search_kwargs["k"] = k
            source_docs = retriever.invoke(query)
            timings['retrieval'] = round((time.perf_counter() - total_start) * 1000, 1)
            
            # Convert LangChain documents to chunk format
            retrieved_chunks = []
            for doc in source_docs:
                metadata = doc.metadata.copy()
                chunk_idx = metadata.get('chunk_index')
                
                if chunk_idx is not None and chunk_idx < len(self.vector_store.chunk_mappings):
                    chunk_mapping = self.vector_store.chunk_mappings[chunk_idx]
                    chunk_data = chunk_mapping['chunk_data'].copy()
                    chunk_data['similarity_score'] = 0.5  # Default, could compute actual similarity
                    retrieved_chunks.append(chunk_data)
            
            # Format references
            references = format_multiple_references(retrieved_chunks, format_type="dict")
            
            def answer() -> str:
                result = self.langchain_qa_chain.combine_documents_chain.invoke(
                    {"input_documents": source_docs, "question": query}
                )
                if isinstance(result, str):
                    return result
                return result.get('output_text') or result.get('text') or ''
            
            response_text, refined_summary = self._answer_with_refined_summary(
                answer, query, retrieved_chunks, timings
            )
            timings['total'] = round((time.perf_counter() - total_start) * 1000, 1)
            
            # Ensure response starts with SUMMARY:
            if not response_text.strip().startswith("SUMMARY:"):
                response_text = "SUMMARY:\n" + response_text
            
            return {
                'query': query,
                'retrieved_chunks': retrieved_chunks,
                'references': references,
                'response': response_text,
                'refined_summary': refined_summary,
                'metadata': {
                    'num_retrieved': len(retrieved_chunks),
                    'k': k,
                    'method': 'langchain_retrieval_chain',
                    'similarity_scores': [chunk.get('similarity_score', 0.0) for chunk in retrieved_chunks],
                    'timings_ms': timings
                }
            }
            
        except Exception as e:
            print(f"Error in LangChain chain: {e}, falling back to custom pipeline")
            return self._process_with_custom_pipeline(query, k, min_similarity)
    
    def retrieve_only(self, query: str, k: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """
        Run the retrieval half of the custom pipeline without calling the LLM.
        
        Returns:
            Dictionary with 'retrieved_chunks', 'query_ner', 'query_tfidf',
            'context_sections', 'queries_used' and 'sections_used'
        """
        all_retrieved = []
        queries_used = [query]
        
        # Step 1: Extract NER and TF-IDF from query
        query_features = self.ner_tfidf_extractor.extract_query_features(query)
        query_ner = query_features['ner_text']
        query_tfidf = query_features['tfidf_text']
        queries_used.append(f"NER: {query_ner}")
        queries_used.append(f"TF-IDF: {query_tfidf}")
        
        # Step 2: Use NER/TF-IDF to find relevant sections via embedding similarity
        section_query = f"{query} {query_ner} {query_tfidf}"
        relevant_sections = self.retriever.find_relevant_sections(section_query, top_n=3, min_similarity=0.25)
        sections_used = []
        if relevant_sections:
            sections_used = relevant_sections
            queries_used.append(f"Matched sections: {', '.join(relevant_sections)}")
            
        # Step 3: Search sections or full index (using query text for LangChain)
        if relevant_sections:
            # Use text-based search which works better with LangChain
            results = self.vector_store.search_by_sections_text(query, relevant_sections, k=k*2)
            filtered_results = []
            for chunk_data, similarity in results:
                if similarity >= min_similarity:
                    chunk_section = chunk_data.get('metadata', {}).get('section_header')
                    if relevant_sections and chunk_section == relevant_sections[0]:
                        similarity = min(similarity * 1.15, 1.0)
                    chunk_with_score = {
                        **chunk_data,
                        'similarity_score': similarity
                    }
                    filtered_results.append(chunk_with_score)
            
            filtered_results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
            all_retrieved.append({'retrieved_chunks': filtered_results})
        else:
            all_retrieved.append({'retrieved_chunks': self.retriever.retrieve_with_context(
                query, k=k*2, min_similarity=min_similarity, use_section_filtering=False
            )})
        
        # Step 4: Merge and deduplicate results
        merged_chunks = self._merge_results(all_retrieved)
        retrieved = merged_chunks[:k]
        
        context_sections = []
        for chunk in retrieved:
            section = chunk.get('metadata', {}).get('section_header')
            if section and section not in context_sections:
                context_sections.append(section)
        
        return {
            'retrieved_chunks': retrieved,
            'query_ner': query_ner,
            'query_tfidf': query_tfidf,
            'context_sections': ', '.join(context_sections) if context_sections else 'N/A',
            'queries_used': queries_used,
            'sections_used': sections_used
        }
    
    def warm_up(self):
        """Exercise retrieval once (no LLM calls) so the first real query skips model/index warm-up."""
        start = time.perf_counter()
        try:
            self.retrieve_only("warmup", k=1)
            retriever = self.retriever.retriever
            if retriever is not None:
                if hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs["k"] = 1
                retriever.invoke("warmup")
            self.vector_store.get_sub_summary_text(None)
        except Exception as e:
            print(f"Warning: warm-up failed: {e}")
            return
        print(f"Pipeline warm-up finished in {(time.perf_counter() - start) * 1000:.0f} ms")
    
    def stream_answer(self, query: str, retrieval: Dict[str, Any]) -> Iterator[str]:
        """Stream the main answer for a retrieve_only() result."""
        yield from self.llm_generator.stream_response(
            query,
            retrieval['retrieved_chunks'],
            product_code=self.product_code,
            query_ner=retrieval['query_ner'],
            query_tfidf=retrieval['query_tfidf'],
            context_sections=retrieval['context_sections'],
            source_type=getattr(self.vector_store, 'source_type', None)
        )
    
    def stream_refined(self, query: str, retrieval: Dict[str, Any]) -> Iterator[str]:
        """Stream the refined summary for a retrieve_only() result (yields nothing if disabled)."""
        if not self.refined_summary_supported or not retrieval['retrieved_chunks']:
            return
        if not hasattr(self.llm_generator, 'stream_refined_summary'):
            refined_summary = self._generate_refined_summary(query, retrieval['retrieved_chunks'])
            if refined_summary:
                yield refined_summary
            return
        yield from self.llm_generator.stream_refined_summary(
            query=query,
            context_chunks=retrieval['retrieved_chunks'],
            vector_store=self.vector_store,
            ner_tfidf_extractor=self.ner_tfidf_extractor,
            product_code=self.product_code
        )
    
    def _process_with_custom_pipeline(self, query: str, k: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """Process query using custom pipeline (original implementation)."""
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()
        
        retrieval = self.retrieve_only(query, k=k, min_similarity=min_similarity)
        timings['retrieval'] = round((time.perf_counter() - total_start) * 1000, 1)
        retrieved = retrieval['retrieved_chunks']
        queries_used = retrieval['queries_used']
        
        if not retrieved:
            return {
                'query': query,
                'retrieved_chunks': [],
                'references': [],
                'response': "No relevant documents found for your query.",
                'metadata': {
                    'num_retrieved': 0,
                    'k': k,
                    'queries_used': queries_used,
                    'timings_ms': timings
                }
            }
        
        # Generate response and refined summary concurrently
        source_type = getattr(self.vector_store, 'source_type', None)
        
        def answer() -> str:
            return self.llm_generator.generate_response(
                query,
                retrieved,
                product_code=self.product_code,
                query_ner=retrieval['query_ner'],
                query_tfidf=retrieval['query_tfidf'],
                context_sections=retrieval['context_sections'],
                source_type=source_type
            )
        
        response, refined_summary = self._answer_with_refined_summary(answer, query, retrieved, timings)
        timings['total'] = round((time.perf_counter() - total_start) * 1000, 1)
        
        # Format references
        references = format_multiple_references(retrieved, format_type="dict")
        
        return {
            'query': query,
            'retrieved_chunks': retrieved,
            'references': references,
            'response': response,
            'refined_summary': refined_summary,
            'metadata': {
                'num_retrieved': len(retrieved),
                'k': k,
                'method': 'custom_pipeline',
                'similarity_scores': [chunk.get('similarity_score', 0.0) for chunk in retrieved],
                'queries_used': queries_used,
                'sections_matched': retrieval['sections_used'],
                'expansion_used': False,
                'multi_query_used': self.use_multi_query,
                'timings_ms': timings
            }
        }
    
    def _find_section_start(self, section_name: str) -> Optional[Dict[str, Any]]:
        """
        Find the first chunk (start location) of a section.
        
        Args:
            section_name: Name of the section to find
            
        Returns:
            Dictionary with metadata of the first chunk in the section, or None
        """
        section_chunk_indices = self.vector_store.get_chunks_by_section(section_name)
        if not section_chunk_indices:
            return None
        
        first_chunk_idx = min(section_chunk_indices)
        
        if first_chunk_idx < len(self.vector_store.chunk_mappings):
            chunk_mapping = self.vector_store.chunk_mappings[first_chunk_idx]
            chunk_data = chunk_mapping.get('chunk_data', {})
            metadata = chunk_data.get('metadata', {})
            
            return {
                'page_num': metadata.get('page_num'),
                'para_index': metadata.get('para_index'),
                'section_header': section_name
            }
        
        return None
    
    def set_product_code(self, product_code: str):
        """Update product code for context."""
        self.product_code = product_code
