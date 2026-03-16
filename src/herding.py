"""
Herding for Constrained Decoding: Deterministic Path with Momentum

When constraints force a token different from the model's top prediction (e.g.,
grammar requires " but model wanted "password"), the probability mass is lost
("evaporated" by the sampling wall). Herding preserves this via a residual
weight vector w that accumulates blocked intentions.

Update rule: x* = argmax_{x ∈ V_valid} (w_t + p_t)ᵀ e_x
Herding update: w_{t+1} = w_t + p_t - e_{x*}

When the grammar later allows string input, the accumulated momentum in w
deterministically favors tokens like "password" that were previously blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .csr_dfa import CSRTransitionMatrix


@dataclass
class HerdingResult:
    """Result of Herding constrained decoding."""
    tokens: list[int]
    final_state: int
    momentum_trace: list[list[float]]  # w at each step (for analysis)
    success: bool


def herding_decode(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    block_length: int | None = None,
) -> HerdingResult:
    """
    Greedy constrained decoding with Herding momentum.
    
    At each step i:
      score(t) = w_i[t] + p_i[t]   for t ∈ V_valid(q)
      t* = argmax score over valid t
      w_{i+1} = w_i + p_i - e_{t*}
      q_{i+1} = δ(q_i, t*)
    
    O(K) per step via CSR sparse lookup.
    """
    d = block_length if block_length is not None else len(prob_vectors)
    if d == 0:
        return HerdingResult(
            tokens=[], final_state=start_state,
            momentum_trace=[], success=True,
        )
    
    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size
    w = [0.0] * vocab_size
    tokens: list[int] = []
    momentum_trace: list[list[float]] = [w.copy()]
    q = start_state
    
    for i in range(d):
        p = prob_vectors[i]
        if len(p) < vocab_size:
            p = p + [0.0] * (vocab_size - len(p))
        
        transitions = csr.get_transitions(q)
        if not transitions:
            return HerdingResult(
                tokens=tokens, final_state=q,
                momentum_trace=momentum_trace,
                success=False,
            )
        
        best_t: int | None = None
        best_q_next: int | None = None
        best_score = float("-inf")
        
        for t, q_next in transitions:
            score = w[t] + p[t]
            if score > best_score:
                best_score = score
                best_t = t
                best_q_next = q_next
        
        if best_t is None or best_q_next is None:
            return HerdingResult(
                tokens=tokens, final_state=q,
                momentum_trace=momentum_trace,
                success=False,
            )
        
        tokens.append(best_t)
        
        # Herding update: w_new = w + p - e_{t*}
        for j in range(vocab_size):
            w[j] = w[j] + p[j]
        w[best_t] -= 1.0
        
        q = best_q_next
        momentum_trace.append(w.copy())
    
    return HerdingResult(
        tokens=tokens,
        final_state=q,
        momentum_trace=momentum_trace,
        success=q in live_states,
    )
