"""Assert lean-LLM and network-timeout constants without importing heavy deps."""

import ast
from pathlib import Path


def _module_constants(filename: str) -> dict:
    src = Path(__file__).resolve().parents[1] / "src" / filename
    values = {}
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                values[node.targets[0].id] = node.value.value
    return values


def test_lean_llm_and_timeout_limits():
    values = _module_constants("llm_integration.py")
    assert values["FDA_API_TIMEOUT_SECONDS"] == 3
    assert values["DEFAULT_LLM_MODEL"] == "claude-haiku-4-5"
    assert values["GUIDANCE_MAX_TOKENS"] == 8192
    assert values["RESPONSE_MAX_TOKENS"] == 8192
    assert values["REFINED_MAX_TOKENS"] == 1000
    assert values["REFINED_MAX_CHUNKS"] == 8
    assert values["REFINED_MAX_CHUNK_CHARS"] == 600
