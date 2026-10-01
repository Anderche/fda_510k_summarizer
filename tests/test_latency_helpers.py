"""Unit tests for zero-budget latency helpers (no index or API required)."""

import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

# VectorStore / QueryEnhancer import LangChain and sentence-transformers at
# module load; stub those so these tests run in a slim environment.
for _name in (
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
    "langchain_core",
    "langchain_core.documents",
    "langchain_huggingface",
    "embeddings",
    "faiss",
    "sentence_transformers",
):
    sys.modules.setdefault(_name, MagicMock())

from query_enhancement import QueryEnhancer
from vector_store import VectorStore


def test_sub_summary_lookup_is_o1():
    store = VectorStore.__new__(VectorStore)
    store.chunk_mappings = [
        {'is_sub_summary': False, 'chunk_data': {'chunk_index': 0, 'text': 'full'}},
        {
            'is_sub_summary': True,
            'chunk_data': {'chunk_index': 0},
            'sub_summary': {'text': 'short summary of chunk 0'},
        },
        {
            'is_sub_summary': True,
            'chunk_data': {'chunk_index': 2},
            'sub_summary': {'text': 'short summary of chunk 2'},
        },
    ]
    store._sub_summary_by_chunk = None

    assert store.get_sub_summary_text(0) == 'short summary of chunk 0'
    assert store.get_sub_summary_text(2) == 'short summary of chunk 2'
    assert store.get_sub_summary_text(99) == ''
    # Second call uses the built dict
    assert store._sub_summary_by_chunk[0] == 'short summary of chunk 0'


def test_term_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "term_cache.npz"
    enhancer = QueryEnhancer(embedding_generator=None, vector_store=None, term_cache_path=str(cache_path))
    terms = ["substantial equivalence", "predicate device"]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    enhancer._save_term_cache(terms, embeddings)

    loaded = QueryEnhancer(embedding_generator=None, vector_store=None, term_cache_path=str(cache_path))
    assert loaded._load_term_cache() is True
    assert [term for term, _ in loaded._corpus_terms] == terms
    np.testing.assert_array_almost_equal(loaded._term_cache["predicate device"], embeddings[1])


def test_response_cache_lru_eviction():
    # Mirror the OrderedDict policy used in app.py without importing the FastAPI app
    cache: OrderedDict = OrderedDict()
    size = 2

    def put(key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > size:
            cache.popitem(last=False)

    put(("idx", "q1", 5), {"response": "a"})
    put(("idx", "q2", 5), {"response": "b"})
    put(("idx", "q3", 5), {"response": "c"})
    assert ("idx", "q1", 5) not in cache
    assert cache[("idx", "q2", 5)]["response"] == "b"
    cache.move_to_end(("idx", "q2", 5))
    put(("idx", "q4", 5), {"response": "d"})
    assert ("idx", "q2", 5) in cache
    assert ("idx", "q3", 5) not in cache


def test_rebuild_sub_summary_index_on_mappings():
    store = VectorStore.__new__(VectorStore)
    store.chunk_mappings = [
        {'is_sub_summary': False, 'chunk_data': {'chunk_index': 1}},
        {
            'is_sub_summary': True,
            'chunk_data': {'chunk_index': 1},
            'sub_summary': {'text': 'built on load'},
        },
    ]
    store._sub_summary_by_chunk = {}
    store._rebuild_sub_summary_index()
    assert store._sub_summary_by_chunk[1] == 'built on load'
    assert store.get_sub_summary_text(1) == 'built on load'
