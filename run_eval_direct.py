#!/usr/bin/env python3
"""Run evaluation directly without eval package - bypasses potential import issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from test_dllm_sdsd import get_device, get_synthetic_logits, build_permissive_dfa
from sparse_dingo import sparse_dingo_dp
from baseline_dingo import baseline_dingo_dp
from herding import herding_decode
from speculative_tree import speculative_decode
import time

def main():
    device, has_gpu = get_device()
    vocab_size = 32000
    block_length = 16
    num_samples = 5
    
    csr, start_state, live_states = build_permissive_dfa(vocab_size)
    trans_fn = lambda q, t: 1 if t < 100 else None
    num_states = 2
    
    all_results = {"B1": [], "B2": [], "B3": [], "SDSD": []}
    
    for i in range(num_samples):
        prob_vectors = get_synthetic_logits(vocab_size, block_length, seed=42 + i)
        
        t0 = time.perf_counter()
        r1 = baseline_dingo_dp(num_states, vocab_size, trans_fn, prob_vectors, start_state, live_states)
        all_results["B1"].append({"latency_ms": (time.perf_counter()-t0)*1000, "success": r1.success})
        
        t0 = time.perf_counter()
        r2 = sparse_dingo_dp(csr, prob_vectors, start_state, live_states)
        all_results["B2"].append({"latency_ms": (time.perf_counter()-t0)*1000, "success": r2.success})
        
        t0 = time.perf_counter()
        r3 = herding_decode(csr, prob_vectors, start_state, live_states)
        all_results["B3"].append({"latency_ms": (time.perf_counter()-t0)*1000, "success": r3.success})
        
        t0 = time.perf_counter()
        r4 = speculative_decode(csr, prob_vectors, start_state, live_states, draft_length=block_length)
        all_results["SDSD"].append({"latency_ms": (time.perf_counter()-t0)*1000, "success": r4.success, "nfe": r4.nfe_used})
    
    print("Evaluation complete!")
    for name, entries in all_results.items():
        avg = sum(e["latency_ms"] for e in entries) / len(entries)
        print(f"  {name}: {avg:.3f} ms avg")
    return all_results

if __name__ == "__main__":
    main()
