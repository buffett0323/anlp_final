"""
Sparse DINGO: O(K) Dynamic Programming for Constrained Decoding.

Optimizes DINGO's transition cost computation from O(N) to O(K) by using
STATIC's CSR sparse indexing. Instead of scanning the full vocabulary V (size N)
for V_i(q, q') = max_{t∈V} v_i(t) s.t. q ∈ δ(q', t), we only iterate over the
K valid tokens stored in the CSR slice for state q'.

Reference: 
- DINGO (Suresh et al., NeurIPS 2025) - Constrained Inference for Diffusion LLMs
- STATIC (Su et al., 2026) - Sparse Transition Matrix for Constrained Decoding
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .csr_dfa import CSRTransitionMatrix


@dataclass 
class DINGOResult:
    """Result of DINGO constrained decoding."""
    tokens: list[int]
    final_state: int
    probability: float
    success: bool


def _log_prob(p: float) -> float:
    return math.log(p) if p > 0.0 else float("-inf")


def compute_transition_costs_sparse(
    csr: CSRTransitionMatrix,
    prob_vector: list[float],
    vocab_size: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """
    Compute V_i(q, q') and T_i(q, q') using O(|Q| * K) instead of O(|Q| * N).
    
    V_i(q, q') = max_{t∈V} v_i(t) s.t. q ∈ δ(q', t)
    T_i(q, q') = argmax token for the above
    
    Instead of iterating over all N tokens for each (q', q), we use the CSR slice:
    For source state q', only K tokens have valid transitions. We iterate only over those.
    
    Complexity: O(|Q| * K) vs original O(|Q| * N)
    """
    Vi: dict[tuple[int, int], float] = {}
    Ti: dict[tuple[int, int], int] = {}
    
    for q_prime in range(csr.num_states):
        # O(K): Get only valid (token, next_state) pairs for q'
        transitions = csr.get_transitions(q_prime)
        
        # Group by target state q, keep max probability token
        best_for_target: dict[int, tuple[float, int]] = {}
        
        for t, q_next in transitions:
            if t < len(prob_vector):
                prob = prob_vector[t]
            else:
                prob = 0.0
            
            if q_next not in best_for_target or prob > best_for_target[q_next][0]:
                best_for_target[q_next] = (prob, t)
        
        for q, (max_prob, best_tok) in best_for_target.items():
            Vi[(q, q_prime)] = max_prob
            Ti[(q, q_prime)] = best_tok
    
    return Vi, Ti


def sparse_dingo_dp(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    block_length: int | None = None,
    initial_log_probs: dict[int, float] | None = None,
) -> DINGOResult:
    """
    DINGO Dynamic Programming: forward pass is O(|reachable| × K) per mask step
    (iterate outgoing transitions from each reachable state), not O(|Q|²).

    Args:
        csr: CSR format DFA (STATIC sparse indexing)
        prob_vectors: v_1, ..., v_d - probability vectors at each position
        start_state: q0 (used only if ``initial_log_probs`` is None)
        live_states: Ql - states that can reach accepting states
        block_length: d (default: len(prob_vectors))
        initial_log_probs: optional log π0(q); if set, overrides single-state init

    Returns:
        DINGOResult with optimal token sequence
    """
    d = block_length if block_length is not None else len(prob_vectors)
    if d == 0:
        return DINGOResult(tokens=[], final_state=start_state, probability=1.0, success=True)

    neg_inf = float("-inf")
    if initial_log_probs is not None:
        W: dict[int, float] = {}
        for q, lv in initial_log_probs.items():
            if 0 <= q < csr.num_states and math.isfinite(lv):
                W[q] = max(W.get(q, neg_inf), lv)
    else:
        W = {start_state: 0.0}

    pr_rows: list[dict[int, tuple[int | None, int | None]]] = []

    for i in range(d):
        pv = prob_vectors[i]
        W_new: dict[int, float] = {}
        row: dict[int, tuple[int | None, int | None]] = {}
        for q_prev, score in W.items():
            if score == neg_inf:
                continue
            for tok, q_next in csr.get_transitions(q_prev):
                if tok < 0:
                    continue
                lp = _log_prob(pv[tok]) if tok < len(pv) else neg_inf
                cand = score + lp
                if cand > W_new.get(q_next, neg_inf):
                    W_new[q_next] = cand
                    row[q_next] = (q_prev, tok)
        W = W_new
        pr_rows.append(row)

    if not W:
        return DINGOResult(
            tokens=[],
            final_state=start_state,
            probability=0.0,
            success=False,
        )

    q_max = -1
    max_log = neg_inf
    for q in live_states:
        lv = W.get(q, neg_inf)
        if lv > max_log:
            max_log = lv
            q_max = q

    if q_max < 0 or not math.isfinite(max_log):
        return DINGOResult(
            tokens=[],
            final_state=start_state,
            probability=0.0,
            success=False,
        )

    tokens: list[int] = []
    q_curr = q_max
    for row in reversed(pr_rows):
        prev_q, t = row.get(q_curr, (None, None))
        if prev_q is None and t is None:
            return DINGOResult(
                tokens=[],
                final_state=start_state,
                probability=0.0,
                success=False,
            )
        if t is not None:
            tokens.append(t)
        q_curr = prev_q if prev_q is not None else q_curr

    tokens.reverse()
    return DINGOResult(
        tokens=tokens,
        final_state=q_max,
        probability=math.exp(max_log),
        success=True,
    )
