"""
Cache compiled llguidance grammar objects.

``grammar_from_json_schema`` / ``grammar_from_lark`` are expensive; the compiled
grammar is treated as immutable — only ``LLMatcher`` holds parse state.
"""

from __future__ import annotations

import json
from typing import Any

from llguidance import LLMatcher

_grammar_obj_cache: dict[str, Any] = {}


def _cache_key_for_grammar_string(grammar: str) -> str:
    """Stable key: canonical JSON for schema strings, else raw lark source."""
    try:
        obj = json.loads(grammar)
        return "json:" + json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return "lark:" + grammar


def get_cached_grammar(grammar: str) -> Any:
    """
    Return a shared compiled grammar object for ``grammar`` (JSON schema string or Lark).

    Multiple ``LLMatcher`` instances may reference the same object; only matchers
    are stateful.
    """
    key = _cache_key_for_grammar_string(grammar)
    if key in _grammar_obj_cache:
        return _grammar_obj_cache[key]

    try:
        obj = json.loads(grammar)
        canonical = json.dumps(obj, sort_keys=True)
        grm = LLMatcher.grammar_from_json_schema(canonical)
    except (json.JSONDecodeError, TypeError):
        grm = LLMatcher.grammar_from_lark(grammar)

    is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
    assert not is_err, "Grammar is not valid"

    _grammar_obj_cache[key] = grm
    return grm
