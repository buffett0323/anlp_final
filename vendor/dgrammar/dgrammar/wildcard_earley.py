"""
Wildcard-aware Earley verifier — Stage 1 (correctness focus).

Each MASK token in the input can be treated as:
  A) Any single terminal symbol, OR
  B) ε (empty string — the MASK is skipped without advancing the grammar dot)

This matches the semantics described in wildcard_detail.md:
  "MASK = ε 或 (S)"  →  MASK covers ε-productions and any single-token expansion.

The Earley chart is extended with masks_used tracking:
  chart[i][(rule, dot, origin)] = set of masks_used counts that can reach this state
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

MASK = "MASK"  # sentinel for wildcard / masked token positions


# ── Grammar ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: tuple[str, ...]  # empty tuple = ε production

    def __repr__(self) -> str:
        body = " ".join(self.rhs) if self.rhs else "ε"
        return f"{self.lhs} → {body}"


class Grammar:
    def __init__(self, start: str, rules: list[Rule]) -> None:
        self.start = start
        self.rules = rules
        self._by_lhs: dict[str, list[Rule]] = defaultdict(list)
        for r in rules:
            self._by_lhs[r.lhs].append(r)
        self.nonterminals: set[str] = {r.lhs for r in rules}

    def rules_for(self, symbol: str) -> list[Rule]:
        return self._by_lhs.get(symbol, [])

    def is_terminal(self, symbol: str) -> bool:
        return symbol not in self.nonterminals

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals


# ── Chart types ───────────────────────────────────────────────────────────────

# (rule, dot_position, origin_position)
_StateKey = tuple[Rule, int, int]
_Chart = list[dict[_StateKey, set[int]]]  # index → state → set[masks_used]


# ── Earley fixpoint ───────────────────────────────────────────────────────────


def _fixpoint(pos: int, chart: _Chart, grammar: Grammar, total_masks: int) -> None:
    """Run Earley prediction + completion to fixpoint at chart[pos]."""
    changed = True
    while changed:
        changed = False
        for (rule, dot, origin), masks_set in list(chart[pos].items()):

            # Prediction: expand non-terminal after dot
            if dot < len(rule.rhs) and grammar.is_nonterminal(rule.rhs[dot]):
                for new_rule in grammar.rules_for(rule.rhs[dot]):
                    state: _StateKey = (new_rule, 0, pos)
                    before = len(chart[pos][state])
                    chart[pos][state] |= masks_set
                    if len(chart[pos][state]) > before:
                        changed = True

            # Completion: rule fully matched — advance waiting states
            if dot == len(rule.rhs):
                for (prev_rule, prev_dot, prev_orig), prev_masks in list(
                    chart[origin].items()
                ):
                    if (
                        prev_dot < len(prev_rule.rhs)
                        and prev_rule.rhs[prev_dot] == rule.lhs
                    ):
                        new_state: _StateKey = (prev_rule, prev_dot + 1, prev_orig)
                        new_masks = {
                            m1 + m2
                            for m1 in prev_masks
                            for m2 in masks_set
                            if m1 + m2 <= total_masks
                        }
                        if new_masks:
                            before = len(chart[pos][new_state])
                            chart[pos][new_state] |= new_masks
                            if len(chart[pos][new_state]) > before:
                                changed = True


# ── Main verifier ─────────────────────────────────────────────────────────────


def wildcard_earley_verify(
    incomplete_prefix: list[str],
    grammar: Grammar,
    total_masks: int,
) -> bool:
    """
    Return True iff there exists an assignment of MASK tokens (to terminals or ε)
    such that the resulting token sequence is a complete parse of grammar.start.

    Parameters
    ----------
    incomplete_prefix:
        Token list; entries that equal MASK are wildcard positions.
    grammar:
        Context-free grammar.
    total_masks:
        Number of MASK entries in incomplete_prefix.
        The verifier checks that exactly this many masks are accounted for.
    """
    n = len(incomplete_prefix)
    chart: _Chart = [defaultdict(set) for _ in range(n + 1)]

    # Seed chart[0] with the start symbol's rules
    for rule in grammar.rules_for(grammar.start):
        chart[0][(rule, 0, 0)].add(0)

    _fixpoint(0, chart, grammar, total_masks)

    for i, token in enumerate(incomplete_prefix):
        if token == MASK:
            for (rule, dot, origin), masks_set in list(chart[i].items()):
                # After consuming this MASK, masks_used increments by 1
                new_masks = {m + 1 for m in masks_set if m + 1 <= total_masks}
                if not new_masks:
                    continue
                # Option A: MASK = ε  →  grammar dot unchanged, input advances
                chart[i + 1][(rule, dot, origin)] |= new_masks
                # Option B: MASK = any terminal  →  dot advances past terminal
                if dot < len(rule.rhs) and grammar.is_terminal(rule.rhs[dot]):
                    chart[i + 1][(rule, dot + 1, origin)] |= new_masks
        else:
            # Concrete token: only states expecting this exact token advance
            for (rule, dot, origin), masks_set in list(chart[i].items()):
                if dot < len(rule.rhs) and rule.rhs[dot] == token:
                    chart[i + 1][(rule, dot + 1, origin)] |= masks_set

        _fixpoint(i + 1, chart, grammar, total_masks)

    # Accept: chart[n] contains a complete parse of the start symbol from pos 0
    # using exactly total_masks wildcard slots.
    for (rule, dot, origin), masks_set in chart[n].items():
        if (
            rule.lhs == grammar.start
            and dot == len(rule.rhs)
            and origin == 0
            and total_masks in masks_set
        ):
            return True

    return False
