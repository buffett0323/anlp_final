"""
Evaluation metrics for SDSD: Parse Rate, Constraint Satisfaction, NFE, TTFT.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    parse_rate: float = 0.0
    constraint_satisfaction: float = 0.0
    nfe_total: int = 0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    n_samples: int = 0


def check_json_parse(text: str) -> bool:
    """Check if text is valid JSON. Handles truncated output."""
    text = text.strip()
    # Try to extract JSON object/array
    start = text.find("{")
    if start < 0:
        start = text.find("[")
    if start < 0:
        return False
    depth = 0
    in_string = False
    escape = False
    end = start
    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if in_string:
            if c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c in '{[':
            depth += 1
        elif c in '}]':
            depth -= 1
            if depth == 0:
                end = i
                break
    try:
        json.loads(text[start : end + 1])
        return True
    except json.JSONDecodeError:
        return False


def check_constraint_regex(text: str, pattern: str) -> bool:
    """Check if text matches regex constraint."""
    return bool(re.search(pattern, text, re.DOTALL))


def compute_perplexity(log_probs: list[float]) -> float:
    """Gen PPL from log probabilities. Lower is better."""
    if not log_probs:
        return float("inf")
    return math.exp(-sum(log_probs) / len(log_probs))
