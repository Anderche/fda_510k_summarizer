"""Helpers for citation links, grouping, and prefix stripping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reference_formatter import (  # noqa: E402
    format_reference,
    guidance_status,
    group_references,
    pdf_page_url,
)


def test_pdf_page_url_appends_fragment():
    url = "https://www.fda.gov/media/184856/download"
    assert pdf_page_url(url, 12) == f"{url}#page=12"
    assert pdf_page_url(f"{url}#page=3", 12) == f"{url}#page=12"
    assert pdf_page_url(url, None) == url
    assert pdf_page_url(None, 1) is None


def test_guidance_status_draft_vs_final():
    assert guidance_status("DRAFT Guidance for Industry") == "Draft"
    assert guidance_status("FDA Guidance - August 18, 2025") == "Final"
    assert guidance_status(None) is None


def test_group_references_merges_pages_and_keeps_slots():
    refs = [
        {
            "display_title": "Doc A",
            "file_name": "a.pdf",
            "page_num": 12,
            "pdf_link": "https://ex.com/a.pdf",
            "guidance_type": "DRAFT Guidance",
        },
        {
            "display_title": "Doc B",
            "file_name": "b.pdf",
            "page_num": 4,
            "pdf_link": "https://ex.com/b.pdf",
            "guidance_type": "FDA Guidance",
        },
        {
            "display_title": "Doc A",
            "file_name": "a.pdf",
            "page_num": 18,
            "pdf_link": "https://ex.com/a.pdf",
            "guidance_type": "DRAFT Guidance",
        },
        {
            "display_title": "Doc A",
            "file_name": "a.pdf",
            "page_num": 12,
            "pdf_link": "https://ex.com/a.pdf",
            "guidance_type": "DRAFT Guidance",
        },
    ]
    groups = group_references(refs)
    assert len(groups) == 2
    assert groups[0]["title"] == "Doc A"
    assert [page["page_num"] for page in groups[0]["pages"]] == [12, 18]
    assert groups[0]["pages"][0]["slots"] == [1, 4]
    assert groups[0]["pages"][0]["pdf_link"] == "https://ex.com/a.pdf#page=12"
    assert groups[1]["pages"][0]["slots"] == [2]
    assert guidance_status(groups[0]["guidance_type"]) == "Draft"
    assert guidance_status(groups[1]["guidance_type"]) == "Final"


def test_format_reference_includes_full_chunk_text():
    chunk = {
        "text": "Full retrieved vector text without cutoff.",
        "metadata": {
            "file_name": "guidance-ai-enabled-device-software-functions.pdf",
            "page_num": 12,
            "para_index": 3,
        },
    }
    ref = format_reference(chunk)
    assert ref["text"] == "Full retrieved vector text without cutoff."
    assert ref["page_num"] == 12
    assert ref["display_title"].startswith("Artificial Intelligence-Enabled")
