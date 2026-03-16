"""
Intent Recovery Ablation: Compare B2 (STATIC+DINGO) vs SDSD/Herding

1. At perturb_step: probs favor low-prob token so both decoders pick it (blocking intent)
2. After perturb: probs favor low-prob again, but Herding's momentum w boosts high-prob
3. Measure recovery steps: first step after perturb where high-prob token is chosen
4. Expected: Herding recovers faster due to momentum preservation

DFA: state 0 --0--> 0, 0 --1--> 1 (accept). Must emit 1 to reach accept.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csr_dfa import build_csr_from_transition_dict
from sparse_dingo import sparse_dingo_dp
from herding import herding_decode


def run_intent_recovery(
    vocab_size: int = 100,
    block_length: int = 5,
    perturb_step: int = 0,
    high_prob_token: int = 1,
    low_prob_token: int = 0,
) -> dict:
    """
    DFA: 0 --0--> 0, 0 --1--> 1 (accept). Need token 1 to reach accept.
    At perturb_step: p[low]=0.9, p[high]=0.1 -> both pick 0 (block intent)
    At perturb+1: p[low]=0.59, p[high]=0.41 -> B2 picks 0 (future p[high] high),
                 Herding picks 1 (w[high]=0.1 boosts score)
    At perturb+2: p[low]=0.16, p[high]=0.84 -> B2 picks 1 (recovery)
    """
    transitions = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    csr = build_csr_from_transition_dict(transitions, num_states=2, vocab_size=vocab_size)
    start_state, live_states = 0, {1}

    prob_vectors = []
    for i in range(block_length):
        p = [0.0] * vocab_size
        if i == perturb_step:
            p[low_prob_token] = 0.9
            p[high_prob_token] = 0.1
        elif i == perturb_step + 1:
            p[low_prob_token] = 0.59
            p[high_prob_token] = 0.41
        elif i >= perturb_step + 2:
            p[low_prob_token] = 0.16
            p[high_prob_token] = 0.84
        else:
            p[low_prob_token] = 0.5
            p[high_prob_token] = 0.5
        prob_vectors.append(p)

    r_b2 = sparse_dingo_dp(csr, prob_vectors, start_state, live_states)
    r_herding = herding_decode(csr, prob_vectors, start_state, live_states)

    def recovery_steps(tokens: list[int], target: int, after: int) -> int:
        for i, t in enumerate(tokens):
            if i > after and t == target:
                return i - after
        return -1

    b2_rec = recovery_steps(r_b2.tokens, high_prob_token, perturb_step)
    herding_rec = recovery_steps(r_herding.tokens, high_prob_token, perturb_step)

    return {
        "B2_tokens": r_b2.tokens,
        "Herding_tokens": r_herding.tokens,
        "B2_success": r_b2.success,
        "Herding_success": r_herding.success,
        "B2_recovery_steps": b2_rec,
        "Herding_recovery_steps": herding_rec,
    }


def main():
    result = run_intent_recovery()
    print("Intent Recovery Ablation")
    print("  B2 (STATIC+DINGO):", result["B2_tokens"], "| recovery steps:", result["B2_recovery_steps"])
    print("  Herding (SDSD):   ", result["Herding_tokens"], "| recovery steps:", result["Herding_recovery_steps"])
    print("  B2 success:", result["B2_success"], "| Herding success:", result["Herding_success"])
    if result["Herding_recovery_steps"] < result["B2_recovery_steps"]:
        print("  -> Herding recovers faster (momentum preserves blocked intent)")


if __name__ == "__main__":
    main()
