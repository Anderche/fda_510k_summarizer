"""Tests for concurrent LLM calls and timings (heavy deps stubbed)."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _name in (
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
    "langchain_core",
    "langchain_core.documents",
    "langchain_core.retrievers",
    "langchain_core.prompts",
    "langchain_core.messages",
    "langchain_huggingface",
    "langchain",
    "langchain.chains",
    "langchain.chains.retrieval_qa",
    "langchain.chains.retrieval_qa.base",
    "langchain_anthropic",
    "embeddings",
    "vector_store",
    "rag_retrieval",
    "llm_integration",
    "reference_formatter",
    "query_enhancement",
    "ner_tfidf_extractor",
    "faiss",
    "sentence_transformers",
    "nltk",
    "sklearn",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
):
    sys.modules.setdefault(_name, MagicMock())

from query_pipeline import QueryPipeline  # noqa: E402


def test_answer_and_refined_summary_run_concurrently():
    pipeline = QueryPipeline.__new__(QueryPipeline)
    pipeline.refined_summary_enabled = True
    pipeline.llm_generator = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.ner_tfidf_extractor = MagicMock()
    pipeline.product_code = None

    def slow_refined(*_args, **_kwargs):
        time.sleep(0.08)
        return "refined"

    pipeline.llm_generator.generate_refined_summary.side_effect = slow_refined

    def slow_answer():
        time.sleep(0.08)
        return "answer"

    timings = {}
    started = time.perf_counter()
    answer, refined = pipeline._answer_with_refined_summary(
        slow_answer, "what is a predicate?", [{"text": "chunk"}], timings
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert answer == "answer"
    assert refined == "refined"
    assert "llm_response" in timings
    assert "llm_refined" in timings
    # Sequential would be ~160ms; overlapping should land well under 140ms
    assert elapsed_ms < 140
    assert timings["llm_response"] >= 70
    assert timings["llm_refined"] >= 70


def test_retrieve_only_skips_query_expansion():
    pipeline = QueryPipeline.__new__(QueryPipeline)
    pipeline.ner_tfidf_extractor = MagicMock()
    pipeline.ner_tfidf_extractor.extract_query_features.return_value = {
        'ner_text': '',
        'tfidf_text': '',
    }
    pipeline.retriever = MagicMock()
    pipeline.retriever.find_relevant_sections.return_value = []
    pipeline.retriever.retrieve_with_context.return_value = []
    pipeline.vector_store = MagicMock()
    pipeline.query_enhancer = MagicMock()

    result = pipeline.retrieve_only("what is a 510k", k=5)

    pipeline.query_enhancer.expand_query.assert_not_called()
    assert result['retrieved_chunks'] == []
    assert result['queries_used'][0] == "what is a 510k"


def test_custom_pipeline_records_timings(monkeypatch):
    pipeline = QueryPipeline.__new__(QueryPipeline)
    pipeline.refined_summary_enabled = False
    pipeline.llm_generator = MagicMock()
    pipeline.llm_generator.generate_response.return_value = "SUMMARY:\nok"
    pipeline.product_code = None
    pipeline.use_multi_query = False
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.source_type = None

    monkeypatch.setattr(pipeline, "retrieve_only", lambda query, k=5, min_similarity=0.0: {
        'retrieved_chunks': [{'text': 'chunk', 'metadata': {}, 'similarity_score': 0.9}],
        'query_ner': '',
        'query_tfidf': '',
        'context_sections': 'N/A',
        'queries_used': [query],
        'sections_used': [],
    })
    monkeypatch.setattr(
        sys.modules['query_pipeline'],
        "format_multiple_references",
        lambda *_args, **_kwargs: [],
    )

    result = pipeline._process_with_custom_pipeline("what is a predicate?", k=1)
    assert 'timings_ms' in result['metadata']
    assert 'retrieval' in result['metadata']['timings_ms']
    assert 'llm_response' in result['metadata']['timings_ms']
    assert 'total' in result['metadata']['timings_ms']


def test_three_sample_queries_report_timings(monkeypatch):
    """Stand-in for live demo queries when the embedding stack is not installed."""
    pipeline = QueryPipeline.__new__(QueryPipeline)
    pipeline.refined_summary_enabled = True
    pipeline.llm_generator = MagicMock()
    pipeline.llm_generator.generate_response.return_value = "SUMMARY:\nsample"
    pipeline.llm_generator.generate_refined_summary.return_value = "REFINED SUMMARY:\nsample"
    pipeline.product_code = None
    pipeline.use_multi_query = False
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.source_type = None
    monkeypatch.setattr(
        sys.modules['query_pipeline'],
        "format_multiple_references",
        lambda *_args, **_kwargs: [],
    )

    queries = [
        "What is substantial equivalence?",
        "What testing is required for a 510k submission?",
        "What is a predicate device?",
    ]

    def retrieve(query, k=5, min_similarity=0.0):
        time.sleep(0.01)
        return {
            'retrieved_chunks': [{'text': query, 'metadata': {}, 'similarity_score': 0.8}],
            'query_ner': '',
            'query_tfidf': '',
            'context_sections': 'N/A',
            'queries_used': [query],
            'sections_used': [],
        }

    monkeypatch.setattr(pipeline, "retrieve_only", retrieve)

    reports = []
    for query in queries:
        result = pipeline._process_with_custom_pipeline(query, k=3)
        timings = result['metadata']['timings_ms']
        reports.append((query, timings))
        assert timings['retrieval'] >= 0
        assert timings['llm_response'] >= 0
        assert timings['llm_refined'] >= 0
        assert timings['total'] >= timings['retrieval']

    assert len(reports) == 3



def test_three_sample_queries_report_timings(monkeypatch):
    """Stand-in for live demo queries when the embedding stack is not installed."""
    pipeline = QueryPipeline.__new__(QueryPipeline)
    pipeline.refined_summary_enabled = True
    pipeline.llm_generator = MagicMock()
    pipeline.llm_generator.generate_response.return_value = "SUMMARY:\nsample"
    pipeline.llm_generator.generate_refined_summary.return_value = "REFINED SUMMARY:\nsample"
    pipeline.product_code = None
    pipeline.use_multi_query = False
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.source_type = None
    monkeypatch.setattr(
        sys.modules['query_pipeline'],
        "format_multiple_references",
        lambda *_args, **_kwargs: [],
    )

    queries = [
        "What is substantial equivalence?",
        "What testing is required for a 510k submission?",
        "What is a predicate device?",
    ]

    def retrieve(query, k=5, min_similarity=0.0):
        time.sleep(0.01)
        return {
            'retrieved_chunks': [{'text': query, 'metadata': {}, 'similarity_score': 0.8}],
            'query_ner': '',
            'query_tfidf': '',
            'context_sections': 'N/A',
            'queries_used': [query],
            'sections_used': [],
        }

    monkeypatch.setattr(pipeline, "retrieve_only", retrieve)

    reports = []
    for query in queries:
        result = pipeline._process_with_custom_pipeline(query, k=3)
        timings = result['metadata']['timings_ms']
        reports.append((query, timings))
        assert timings['retrieval'] >= 0
        assert timings['llm_response'] >= 0
        assert timings['llm_refined'] >= 0
        assert timings['total'] >= timings['retrieval']

    assert len(reports) == 3

