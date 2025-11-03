"""
Query Pipeline Module

End-to-end query processing from user input to response generation.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag_retrieval import RAGRetriever
from llm_integration import LLMGenerator, SimpleLLMGenerator
from reference_formatter import format_multiple_references
from query_enhancement import QueryEnhancer
from ner_tfidf_extractor import NERTFIDFExtractor


class QueryPipeline:
    """
    Complete query processing pipeline.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator, 
                 llm_generator: Optional[LLMGenerator] = None, product_code: Optional[str] = None,
                 use_query_expansion: bool = True, use_multi_query: bool = True):
        """
        Initialize query pipeline.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            llm_generator: LLM generator instance (optional, will use SimpleLLMGenerator if None)
            product_code: Product code for context in responses
            use_query_expansion: Whether to use query expansion
            use_multi_query: Whether to use multi-query generation
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_generator = llm_generator or SimpleLLMGenerator()
        self.product_code = product_code
        self.retriever = RAGRetriever(vector_store, embedding_generator)
        self.use_query_expansion = use_query_expansion
        self.use_multi_query = use_multi_query
        self.query_enhancer = QueryEnhancer(embedding_generator, vector_store)
        self.ner_tfidf_extractor = NERTFIDFExtractor()
    
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
        Process a user query and generate a response with query enhancement.
        
        Args:
            query: User's query/question
            k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with 'query', 'retrieved_chunks', 'response', and 'metadata'
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
        # Combine NER entities and TF-IDF terms for section matching
        section_query = f"{query} {query_ner} {query_tfidf}"
        relevant_sections = self.retriever.find_relevant_sections(section_query, top_n=3, min_similarity=0.25)
        sections_used = []
        if relevant_sections:
            sections_used = relevant_sections
            queries_used.append(f"Matched sections: {', '.join(relevant_sections)}")
            
            # Find section start locations (first chunk of each section)
            for section_name in relevant_sections[:1]:  # Focus on top section
                section_start = self._find_section_start(section_name)
                if section_start:
                    queries_used.append(f"Section '{section_name}' starts at page {section_start.get('page_num', 'N/A')}, paragraph {section_start.get('para_index', 'N/A')}")
        
        # Step 3: Embed query and search sections (using metadata[section])
        query_embedding = self.embedding_generator.embed_text(query)
        if relevant_sections:
            # Search within matched sections only
            results = self.vector_store.search_by_sections(query_embedding, relevant_sections, k=k*2)
            filtered_results = []
            for chunk_data, similarity in results:
                if similarity >= min_similarity:
                    chunk_section = chunk_data.get('metadata', {}).get('section_header')
                    # Boost similarity for chunks from the most relevant section
                    if relevant_sections and chunk_section == relevant_sections[0]:
                        similarity = min(similarity * 1.15, 1.0)  # Boost by 15%, cap at 1.0
                    chunk_with_score = {
                        **chunk_data,
                        'similarity_score': similarity
                    }
                    filtered_results.append(chunk_with_score)
            
            # Re-sort by boosted similarity
            filtered_results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
            all_retrieved.append({'retrieved_chunks': filtered_results})
        else:
            # Fallback: full search if no sections matched
            all_retrieved.append({'retrieved_chunks': self.retriever.retrieve_with_context(
                query, k=k*2, min_similarity=min_similarity, use_section_filtering=False
            )})
        
        # Step 4: Query expansion (optional)
        expanded_query = query
        if self.use_query_expansion:
            expanded_query = self.query_enhancer.expand_query(query, top_n=5, similarity_threshold=0.6)
            if expanded_query != query:
                queries_used.append(f"Expanded: {expanded_query}")
                # Additional retrieval with expanded query
                if relevant_sections:
                    expanded_embedding = self.embedding_generator.embed_text(expanded_query)
                    exp_results = self.vector_store.search_by_sections(expanded_embedding, relevant_sections, k=k)
                    exp_filtered = []
                    for chunk_data, similarity in exp_results:
                        if similarity >= min_similarity:
                            exp_filtered.append({**chunk_data, 'similarity_score': similarity})
                    if exp_filtered:
                        all_retrieved.append({'retrieved_chunks': exp_filtered})
        
        # Step 5: Merge and deduplicate results
        merged_chunks = self._merge_results(all_retrieved)
        retrieved = merged_chunks[:k]  # Take top k after merging
        
        if not retrieved:
            return {
                'query': query,
                'retrieved_chunks': [],
                'references': [],
                'response': "No relevant documents found for your query.",
                'metadata': {
                    'num_retrieved': 0,
                    'k': k,
                    'queries_used': queries_used
                }
            }
        
        # Step 5: Fetch linked full chunks as context
        context_chunks = [chunk for chunk in retrieved]
        
        # Step 6: Collect section information from context chunks
        context_sections = []
        for chunk in context_chunks:
            section = chunk.get('metadata', {}).get('section_header')
            if section and section not in context_sections:
                context_sections.append(section)
        
        # Step 7: Augment LLM prompt with context, NER, TF-IDF, and sections
        # Get source type from vector store
        source_type = getattr(self.vector_store, 'source_type', None)
        response = self.llm_generator.generate_response(
            query,
            context_chunks,
            product_code=self.product_code,
            query_ner=query_ner,
            query_tfidf=query_tfidf,
            context_sections=', '.join(context_sections) if context_sections else 'N/A',
            source_type=source_type
        )
        
        # Step 8: Generate refined summary based on most relevant section and sub-summaries
        refined_summary = None
        if hasattr(self.llm_generator, 'generate_refined_summary'):
            try:
                refined_summary = self.llm_generator.generate_refined_summary(
                    query,
                    retrieved,  # Use all retrieved chunks for analysis
                    vector_store=self.vector_store,
                    ner_tfidf_extractor=self.ner_tfidf_extractor,
                    product_code=self.product_code
                )
            except Exception as e:
                print(f"Warning: Failed to generate refined summary: {e}")
        
        # Format references for each retrieved chunk
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
                'similarity_scores': [chunk.get('similarity_score', 0.0) for chunk in retrieved],
                'queries_used': queries_used,
                'sections_matched': sections_used,
                'expansion_used': self.use_query_expansion,
                'multi_query_used': self.use_multi_query
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
        
        # Get the first chunk index (assuming indices are in order)
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

