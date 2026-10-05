"""Formatting instructions honor list/itemize requests in the user query."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_integration import (  # noqa: E402
    LLMGenerator,
    SimpleLLMGenerator,
    citation_instructions,
    format_context_from_chunks,
    query_requests_structured_format,
    response_format_instructions,
    strip_response_prefix,
)


def test_detects_list_and_itemize_queries():
    assert query_requests_structured_format("List the PCCP elements")
    assert query_requests_structured_format("Please itemize labeling recommendations")
    assert query_requests_structured_format("Give bullet points for validation")
    assert query_requests_structured_format("Enumerate the submission steps")
    assert query_requests_structured_format("Outline the change-control plan as a list")
    assert not query_requests_structured_format("What is a Predetermined Change Control Plan?")
    assert not query_requests_structured_format("How should AI-enabled device software be validated?")


def test_list_query_prompt_requires_markdown_list():
    text = response_format_instructions("List the labeling recommendations")
    assert "MUST use a Markdown list" in text
    assert "- item" in text


def test_plain_query_prompt_allows_prose():
    text = response_format_instructions("What is a PCCP?")
    assert "MUST use a Markdown list" not in text
    assert "complete answer from the provided documents" in text
    assert "concise prose summary" not in text


def test_guidance_prompt_includes_list_format_rule():
    gen = LLMGenerator.__new__(LLMGenerator)
    _, user_prompt = gen._build_guidance_prompts(
        "List the marketing submission recommendations",
        [{"text": "Submit a PCCP when the device uses AI.", "metadata": {"file_name": "guidance.pdf", "page_num": 12}}],
    )
    assert "MUST use a Markdown list" in user_prompt
    assert "Do not collapse items into a paragraph" in user_prompt
    assert "[1] guidance.pdf, p. 12" in user_prompt
    assert "Cite sources inline with [n]" in user_prompt
    assert "Start the answer with SUMMARY:" not in user_prompt
    assert "about 250 words" in user_prompt


def test_reference_text_is_trimmed_to_300_chars():
    from reference_formatter import format_reference

    ref = format_reference({"text": "x" * 500, "metadata": {"file_name": "doc.pdf", "page_num": 1}})
    assert ref["text"] == "x" * 300


def test_guidance_prompt_does_not_force_summary_label():
    gen = LLMGenerator.__new__(LLMGenerator)
    _, user_prompt = gen._build_guidance_prompts(
        "What is a Predetermined Change Control Plan?",
        [{"text": "A PCCP describes planned changes.", "metadata": {"file_name": "Guidance-Predetermined-Change-Control-AI.pdf", "page_num": 4}}],
    )
    assert "Start the answer with SUMMARY:" not in user_prompt
    assert "[1] Marketing Submission Recommendations" in user_prompt
    assert citation_instructions() in user_prompt


def test_strip_response_prefix_removes_labels():
    assert strip_response_prefix("SUMMARY:\nHello") == "Hello"
    assert strip_response_prefix("REFINED SUMMARY:\nHello") == "Hello"
    assert strip_response_prefix("Hello") == "Hello"
    assert strip_response_prefix("") == ""


def test_prompt_context_includes_every_chunk_in_full():
    chunks = [
        {"text": f"FULL CHUNK TEXT {i} " + ("x" * 200), "metadata": {"file_name": f"doc{i}.pdf", "page_num": i}}
        for i in range(1, 7)
    ]
    gen = LLMGenerator.__new__(LLMGenerator)
    _, guidance_prompt = gen._build_guidance_prompts("List the PCCP elements", chunks)
    _, response_prompt = gen._build_response_prompts("What is substantial equivalence?", chunks)
    joined = format_context_from_chunks(chunks)
    for i in range(1, 7):
        marker = f"FULL CHUNK TEXT {i}"
        assert marker in joined
        assert marker in guidance_prompt
        assert marker in response_prompt
        assert f"[{i}]" in guidance_prompt
        assert f"[{i}]" in response_prompt
    assert "..." not in joined


def test_fallback_and_simple_responses_include_full_chunk_text():
    long_text = "A" * 250 + " END OF CHUNK"
    chunks = [
        {"text": long_text, "metadata": {"file_name": "doc.pdf", "page_num": 3}},
        {"text": "SECOND CHUNK FULL TEXT", "metadata": {"file_name": "doc.pdf", "page_num": 4}},
        {"text": "THIRD", "metadata": {"file_name": "doc.pdf", "page_num": 5}},
        {"text": "FOURTH", "metadata": {"file_name": "doc.pdf", "page_num": 6}},
    ]
    gen = LLMGenerator.__new__(LLMGenerator)
    fallback = gen._generate_fallback_summary("query", chunks)
    simple = SimpleLLMGenerator().generate_response(
        "query", chunks, source_type="corpus_ai_guidances"
    )
    for text in (fallback, simple):
        assert long_text in text
        assert "SECOND CHUNK FULL TEXT" in text
        assert "FOURTH" in text
        assert "..." not in text

