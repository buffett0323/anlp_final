"""
Self-Speculative Decoding Tree

Reduces NFE (Number of Function Evaluations) by generating multiple tokens per
forward pass. STATIC optimizes "width" (N→K); Speculative Tree optimizes "depth"
(fewer forward passes).

Pipeline:
1. Drafting: dLLM predicts draft_length positions in one parallel forward pass
2. Tree Construction: CSR picks K valid tokens per node → grammar-correct tree
3. Deterministic Picking: Herding (w + p) selects best branch at each node
4. Parallel Verification: One forward pass verifies the entire tree
5. Bonus Tokens: Output draft_length tokens per NFE (e.g., 3 tokens = 3× NFE reduction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .csr_dfa import CSRTransitionMatrix


@dataclass
class SpeculativeNode:
    """Node in the grammar-correct speculative tree."""
    state: int
    token: int | None  # None for root
    depth: int
    path: list[int]
    children: list[SpeculativeNode] = field(default_factory=list)


@dataclass
class SpeculativeResult:
    """Result of self-speculative decoding."""
    tokens: list[int]
    final_state: int
    draft_length: int
    nfe_used: int  # 1 = one forward pass for entire draft
    success: bool


def build_speculative_tree(
    csr: CSRTransitionMatrix,
    start_state: int,
    draft_length: int,
    live_states: set[int],
) -> SpeculativeNode:
    """
    Build grammar-correct tree using CSR: only K valid tokens per node.
    Tree depth = draft_length. O(K^d) nodes for full tree.
    """
    root = SpeculativeNode(state=start_state, token=None, depth=0, path=[])
    frontier = [root]

    for _ in range(draft_length):
        next_frontier = []
        for node in frontier:
            transitions = csr.get_transitions(node.state)
            for t, q_next in transitions:
                child = SpeculativeNode(
                    state=q_next,
                    token=t,
                    depth=node.depth + 1,
                    path=node.path + [t],
                )
                node.children.append(child)
                next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break

    return root




def select_path_herding(
    root: SpeculativeNode,
    prob_vectors: list[list[float]],
    vocab_size: int,
) -> tuple[list[int], int, list[list[float]]]:
    """
    Traverse tree using Herding: at each node, pick child with max (w + p).
    Returns (tokens, final_state, momentum_trace).
    """
    w = [0.0] * vocab_size
    momentum_trace = [w.copy()]
    tokens: list[int] = []
    node = root

    for i in range(len(prob_vectors)):
        if not node.children:
            break

        p = prob_vectors[i]
        if len(p) < vocab_size:
            p = p + [0.0] * (vocab_size - len(p))

        best_child: SpeculativeNode | None = None
        best_score = float("-inf")

        for child in node.children:
            t = child.token
            if t is None:
                continue
            score = w[t] + p[t]
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None or best_child.token is None:
            break

        t_star = best_child.token
        tokens.append(t_star)

        # Herding update: w_new = w + p - e_{t*}
        for j in range(vocab_size):
            w[j] = w[j] + p[j]
        w[t_star] -= 1.0
        momentum_trace.append(w.copy())

        node = best_child

    return tokens, node.state, momentum_trace


def speculative_decode_lazy(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int | None = None,
) -> SpeculativeResult:
    """
    Lazy speculative decode: no full tree materialization.
    Equivalent to herding_decode but returns SpeculativeResult with NFE=1.
    O(d * K) memory vs O(K^d) for full tree.
    """
    from herding import herding_decode

    d = draft_length if draft_length is not None else len(prob_vectors)
    result = herding_decode(csr, prob_vectors, start_state, live_states, block_length=d)
    return SpeculativeResult(
        tokens=result.tokens,
        final_state=result.final_state,
        draft_length=len(result.tokens),
        nfe_used=1,
        success=result.success,
    )


def speculative_decode(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int | None = None,
    use_lazy: bool = True,
) -> SpeculativeResult:
    """
    Self-Speculative Decoding: one forward pass → draft_length tokens.

    Assumes prob_vectors come from one parallel model forward (dLLM predicts
    draft_length positions at once). Builds grammar tree via CSR, selects
    path with Herding, returns bonus tokens.

    use_lazy=True: O(d*K) memory, no full tree (recommended for large K).
    use_lazy=False: builds full tree for analysis.
    """
    d = draft_length if draft_length is not None else len(prob_vectors)
    if d == 0:
        return SpeculativeResult(
            tokens=[], final_state=start_state,
            draft_length=0, nfe_used=1, success=True,
        )

    if use_lazy:
        return speculative_decode_lazy(
            csr, prob_vectors, start_state, live_states, d
        )

    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size

    # 1. Tree construction (CSR: K valid tokens per node)
    root = build_speculative_tree(csr, start_state, d, live_states)

    # 2. Deterministic picking with Herding
    tokens, final_state, _ = select_path_herding(root, prob_vectors, vocab_size)

    return SpeculativeResult(
        tokens=tokens,
        final_state=final_state,
        draft_length=len(tokens),
        nfe_used=1,
        success=final_state in live_states and len(tokens) == d,
    )
