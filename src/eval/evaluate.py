"""
SDSD Evaluation: B1, B2, B3, SDSD on LLaDA / Dream

Baselines:
  B1: DINGO O(N) - baseline_dingo_dp
  B2: STATIC + DINGO O(K) - sparse_dingo_dp
  B3: DINGO + Herding - herding_decode (no speculative)
  SDSD: STATIC + Herding + Speculative Tree - speculative_decode

Metrics: Parse Rate, Constraint Satisfaction, NFE, TTFT, Latency
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csr_dfa import build_csr_from_transition_dict
from sparse_dingo import sparse_dingo_dp
from baseline_dingo import baseline_dingo_dp
from herding import herding_decode
from speculative_tree import speculative_decode

# Import model loading from test_dllm_sdsd
from test_dllm_sdsd import (
    get_device,
    load_dream_model,
    load_llada_model,
    get_block_logits_dream,
    get_block_logits_llada,
    get_synthetic_logits,
    build_permissive_dfa,
    build_simple_json_dfa,
)
# Metrics (check_json_parse, EvalMetrics) - loaded lazily when needed
def _load_metrics():
    import importlib.util
    p = Path(__file__).resolve().parent / "metrics.py"
    spec = importlib.util.spec_from_file_location("eval_metrics", p)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("metrics.py loader is None")
    spec.loader.exec_module(mod)
    return mod.check_json_parse, mod.EvalMetrics


@dataclass
class DecoderResult:
    tokens: list[int]
    final_state: int
    success: bool
    nfe: int = 1
    ttft_ms: float = 0.0
    latency_ms: float = 0.0


def decode_b1(csr, trans_fn, num_states, vocab_size, prob_vectors, start_state, live_states) -> DecoderResult:
    """B1: DINGO O(N)."""
    t0 = time.perf_counter()
    result = baseline_dingo_dp(num_states, vocab_size, trans_fn, prob_vectors, start_state, live_states)
    latency = (time.perf_counter() - t0) * 1000
    return DecoderResult(
        tokens=result.tokens, final_state=result.final_state, success=result.success,
        nfe=1, ttft_ms=latency, latency_ms=latency,
    )


def decode_b2(csr, prob_vectors, start_state, live_states) -> DecoderResult:
    """B2: STATIC + DINGO O(K)."""
    t0 = time.perf_counter()
    result = sparse_dingo_dp(csr, prob_vectors, start_state, live_states)
    latency = (time.perf_counter() - t0) * 1000
    return DecoderResult(
        tokens=result.tokens, final_state=result.final_state, success=result.success,
        nfe=1, ttft_ms=latency, latency_ms=latency,
    )


def decode_b3(csr, prob_vectors, start_state, live_states) -> DecoderResult:
    """B3: DINGO + Herding (no speculative)."""
    t0 = time.perf_counter()
    result = herding_decode(csr, prob_vectors, start_state, live_states)
    latency = (time.perf_counter() - t0) * 1000
    return DecoderResult(
        tokens=result.tokens, final_state=result.final_state, success=result.success,
        nfe=1, ttft_ms=latency, latency_ms=latency,
    )


def decode_sdsd(csr, prob_vectors, start_state, live_states, draft_len: int) -> DecoderResult:
    """SDSD: STATIC + Herding + Speculative Tree."""
    t0 = time.perf_counter()
    result = speculative_decode(csr, prob_vectors, start_state, live_states, draft_length=draft_len)
    latency = (time.perf_counter() - t0) * 1000
    return DecoderResult(
        tokens=result.tokens, final_state=result.final_state, success=result.success,
        nfe=result.nfe_used, ttft_ms=latency, latency_ms=latency,
    )


def run_eval_sample(
    prob_vectors: list[list[float]],
    csr,
    trans_fn,
    num_states: int,
    vocab_size: int,
    start_state: int,
    live_states: set[int],
    tokenizer=None,
) -> dict:
    """Run all decoders on one sample, return metrics."""
    draft_len = len(prob_vectors)
    results = {}

    # B1 needs transition fn
    results["B1"] = decode_b1(csr, trans_fn, num_states, vocab_size, prob_vectors, start_state, live_states)
    results["B2"] = decode_b2(csr, prob_vectors, start_state, live_states)
    results["B3"] = decode_b3(csr, prob_vectors, start_state, live_states)
    results["SDSD"] = decode_sdsd(csr, prob_vectors, start_state, live_states, draft_len)

    return results


def run_evaluation(
    model_name: str,
    num_samples: int = 10,
    block_length: int = 16,
    mock: bool = False,
) -> dict:
    """Run full evaluation across B1, B2, B3, SDSD."""
    device, has_gpu = get_device()
    if mock or not has_gpu:
        vocab_size = 32000
        csr, start_state, live_states = build_permissive_dfa(vocab_size)
        def trans_fn(q, t):
            if t < 100:
                return 1
            return None
        num_states = 2
        tokenizer = None
    else:
        if model_name == "dream":
            model, tokenizer = load_dream_model(device)
            get_logits = lambda p, bl: get_block_logits_dream(model, tokenizer, p, bl, device)
        else:
            model, tokenizer = load_llada_model(device)
            get_logits = lambda p, bl: get_block_logits_llada(model, tokenizer, p, bl, device)
        vocab_size = tokenizer.vocab_size if tokenizer else 32000
        csr, start_state, live_states = build_permissive_dfa(vocab_size)
        def trans_fn(q, t):
            if t < min(100, vocab_size):
                return 1
            return None
        num_states = 2

    prompt = "Generate a JSON object with key 'name' and value 'test'."
    all_results = { "B1": [], "B2": [], "B3": [], "SDSD": [] }

    for i in range(num_samples):
        if mock or not has_gpu:
            prob_vectors = get_synthetic_logits(vocab_size, block_length, seed=42 + i)
        else:
            prob_vectors, _ = get_logits(prompt, block_length)

        sample_results = run_eval_sample(
            prob_vectors, csr, trans_fn, num_states, vocab_size,
            start_state, live_states, tokenizer,
        )
        for name, r in sample_results.items():
            all_results[name].append({
                "latency_ms": r.latency_ms,
                "nfe": r.nfe,
                "ttft_ms": r.ttft_ms,
                "success": r.success,
                "n_tokens": len(r.tokens),
            })

    # Aggregate
    summary = {}
    for name in ["B1", "B2", "B3", "SDSD"]:
        entries = all_results[name]
        summary[name] = {
            "latency_ms_mean": sum(e["latency_ms"] for e in entries) / len(entries) if entries else 0,
            "nfe_mean": sum(e["nfe"] for e in entries) / len(entries) if entries else 0,
            "ttft_ms_mean": sum(e["ttft_ms"] for e in entries) / len(entries) if entries else 0,
            "success_rate": sum(1 for e in entries if e["success"]) / len(entries) if entries else 0,
            "n_tokens_mean": sum(e["n_tokens"] for e in entries) / len(entries) if entries else 0,
        }

    return {
        "model": model_name,
        "mock": mock,
        "num_samples": num_samples,
        "block_length": block_length,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="SDSD Evaluation")
    parser.add_argument("--model", choices=["dream", "llada"], default="dream")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data (no GPU)")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Running SDSD Evaluation...")
    print(f"  Model: {args.model}, Mock: {args.mock}, Samples: {args.samples}")

    results = run_evaluation(args.model, args.samples, args.block_length, args.mock)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, s in results["summary"].items():
        print(f"\n{name}:")
        print(f"  Latency: {s['latency_ms_mean']:.3f} ms")
        print(f"  NFE: {s['nfe_mean']:.2f}")
        print(f"  TTFT: {s['ttft_ms_mean']:.3f} ms")
        print(f"  Success: {s['success_rate']*100:.1f}%")
        print(f"  Tokens: {s['n_tokens_mean']:.1f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
