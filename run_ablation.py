#!/usr/bin/env python3
"""
SDSD Ablation Experiment Runner

Runs all 5 methods (Baseline, Ablation 1–3, SDSD) and outputs the comparison table.

Usage:
  python run_ablation.py --model dream --mock             # Synthetic (no GPU)
  python run_ablation.py --model dream --samples 20       # Dream-7B
  python run_ablation.py --model llada --samples 20       # LLaDA-8B
  python run_ablation.py --model dream --dataset json-mode-eval  # With dataset
  python run_ablation.py --output results/ablation_table.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Mock mode: force CPU to avoid slow CUDA init (torch.cuda.is_available() can take minutes)
if "--mock" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from csr_dfa import build_csr_from_transition_dict
from baseline_dingo import baseline_dingo_dp
from sparse_dingo import sparse_dingo_dp
from herding import herding_decode
from speculative_tree import speculative_decode, speculative_decode_argmax

from test_dllm_sdsd import (
    get_device,
    load_dream_model,
    load_llada_model,
    get_block_logits_dream,
    get_block_logits_llada,
    get_logits_for_position_dream,
    get_logits_for_position_llada,
    get_synthetic_logits,
    get_synthetic_logits_for_position,
    build_permissive_dfa,
    build_simple_json_dfa,
)


def _trans_fn_permissive(q, t, vocab_limit=100):
    if t < vocab_limit:
        return 1
    return None


def _run_sequential_decode(
    get_logits_fn,
    csr,
    trans_fn,
    num_states,
    vocab_size,
    start_state,
    live_states,
    block_length,
    prompt,
    seed,
) -> tuple[list[int], int, bool, float]:
    """
    Sequential decoding: 1 model forward per token. NFE = block_length.
    Returns (tokens, nfe, success, latency_ms).
    """
    tokens = []
    q = start_state
    nfe = 0
    t0 = time.perf_counter()

    for i in range(block_length):
        prob_i = get_logits_fn(prompt, tokens, seed + i)
        nfe += 1
        r = baseline_dingo_dp(num_states, vocab_size, trans_fn, [prob_i], q, live_states)
        if not r.tokens:
            break
        tokens.append(r.tokens[0])
        q = r.final_state

    latency = (time.perf_counter() - t0) * 1000
    success = q in live_states and len(tokens) == block_length
    return tokens, nfe, success, latency


def _run_sequential_sparse(
    get_logits_fn,
    csr,
    start_state,
    live_states,
    block_length,
    prompt,
    seed,
) -> tuple[list[int], int, bool, float]:
    """Sequential STATIC+DINGO: NFE = block_length, O(K) decode per step."""
    tokens = []
    q = start_state
    nfe = 0
    t0 = time.perf_counter()

    for i in range(block_length):
        prob_i = get_logits_fn(prompt, tokens, seed + i)
        nfe += 1
        r = sparse_dingo_dp(csr, [prob_i], q, live_states)
        if not r.tokens:
            break
        tokens.append(r.tokens[0])
        q = r.final_state

    latency = (time.perf_counter() - t0) * 1000
    success = q in live_states and len(tokens) == block_length
    return tokens, nfe, success, latency


def _run_sequential_herding(
    get_logits_fn,
    csr,
    start_state,
    live_states,
    block_length,
    prompt,
    seed,
) -> tuple[list[int], int, bool, float]:
    """Sequential Herding: NFE = block_length."""
    tokens = []
    q = start_state
    nfe = 0
    t0 = time.perf_counter()

    for i in range(block_length):
        prob_i = get_logits_fn(prompt, tokens, seed + i)
        nfe += 1
        r = herding_decode(csr, [prob_i], q, live_states, block_length=1)
        if not r.tokens:
            break
        tokens.append(r.tokens[0])
        q = r.final_state

    latency = (time.perf_counter() - t0) * 1000
    success = q in live_states and len(tokens) == block_length
    return tokens, nfe, success, latency


def run_one_sample(
    get_logits_fn,
    get_block_logits_fn,
    csr,
    trans_fn,
    num_states,
    vocab_size,
    start_state,
    live_states,
    block_length,
    prompt,
    seed,
) -> dict:
    """
    Run all 5 methods. Sequential (Baseline, A1, A2): NFE=block_length.
    Block (A3, SDSD): NFE=1.
    """
    results = {}

    # Baseline: Sequential DINGO O(N), NFE = block_length
    tokens1, nfe1, succ1, lat1 = _run_sequential_decode(
        get_logits_fn, csr, trans_fn, num_states, vocab_size,
        start_state, live_states, block_length, prompt, seed,
    )
    results["Baseline"] = {"tokens": tokens1, "success": succ1, "latency_ms": lat1, "nfe": nfe1}

    # Ablation 1: Sequential STATIC+DINGO O(K), NFE = block_length
    tokens2, nfe2, succ2, lat2 = _run_sequential_sparse(
        get_logits_fn, csr, start_state, live_states, block_length, prompt, seed,
    )
    results["Ablation1"] = {"tokens": tokens2, "success": succ2, "latency_ms": lat2, "nfe": nfe2}

    # Ablation 2: Sequential Herding, NFE = block_length
    tokens3, nfe3, succ3, lat3 = _run_sequential_herding(
        get_logits_fn, csr, start_state, live_states, block_length, prompt, seed,
    )
    results["Ablation2"] = {"tokens": tokens3, "success": succ3, "latency_ms": lat3, "nfe": nfe3}

    # Ablation 3 & SDSD: Block decode, NFE = 1 (include model forward in latency)
    t0 = time.perf_counter()
    try:
        prob_vectors = get_block_logits_fn(prompt, block_length, seed)
    except TypeError:
        prob_vectors = get_block_logits_fn(prompt, block_length)
    if isinstance(prob_vectors, tuple):
        prob_vectors = prob_vectors[0]
    lat_model = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r4 = speculative_decode_argmax(csr, prob_vectors, start_state, live_states, draft_length=block_length)
    lat4 = lat_model + (time.perf_counter() - t0) * 1000
    results["Ablation3"] = {
        "tokens": r4.tokens,
        "success": r4.success,
        "latency_ms": lat4,
        "nfe": r4.nfe_used,
    }

    t0 = time.perf_counter()
    r5 = speculative_decode(csr, prob_vectors, start_state, live_states, draft_length=block_length)
    lat5 = lat_model + (time.perf_counter() - t0) * 1000
    results["SDSD"] = {
        "tokens": r5.tokens,
        "success": r5.success,
        "latency_ms": lat5,
        "nfe": r5.nfe_used,
    }

    return results


def run_ablation(
    model_name: str,
    num_samples: int = 10,
    block_length: int = 16,
    mock: bool = False,
    dataset: str | None = None,
    dataset_limit: int = 50,
) -> dict:
    """Run full ablation study."""
    device, has_gpu = get_device()
    vocab_size = 64 if mock else 32000  # Small vocab for mock (B1 is O(N), ~16*2*64=2K per sample)
    use_json_dfa = dataset == "json-mode-eval"

    if mock or not has_gpu:
        prob_sources = []
        for i in range(num_samples):
            prob_sources.append(("synthetic", "Generate a JSON object with key 'name' and value 'test'.", i))
        tokenizer = None

        def get_logits_fn(prompt, prefix_tokens, seed):
            return get_synthetic_logits_for_position(vocab_size, prefix_tokens, seed)

        def get_block_logits_fn(prompt, bl, seed=42):
            return get_synthetic_logits(vocab_size, bl, seed=seed)

    else:
        if model_name == "dream":
            model, tokenizer = load_dream_model(device)
            get_logits_fn = lambda p, prefix, _: get_logits_for_position_dream(model, tokenizer, p, prefix, device)[0]
            get_block_logits_fn = lambda p, bl: get_block_logits_dream(model, tokenizer, p, bl, device)
        else:
            model, tokenizer = load_llada_model(device)
            get_logits_fn = lambda p, prefix, _: get_logits_for_position_llada(model, tokenizer, p, prefix, device)[0]
            get_block_logits_fn = lambda p, bl: get_block_logits_llada(model, tokenizer, p, bl, device)
        vocab_size = tokenizer.vocab_size if tokenizer else vocab_size

        if dataset:
            try:
                from datasets import get_dataset
                samples = get_dataset(dataset)[:dataset_limit]
                prob_sources = [(dataset, s.prompt, i) for i, s in enumerate(samples)]
            except Exception as e:
                print(f"Dataset load failed: {e}, using default prompt")
                prob_sources = [("default", "Generate a JSON object with key 'name' and value 'test'.", i) for i in range(num_samples)]
        else:
            prob_sources = [("default", "Generate a JSON object with key 'name' and value 'test'.", i) for i in range(num_samples)]

    if use_json_dfa:
        csr, start_state, live_states = build_simple_json_dfa(vocab_size)
        num_states = 10

        def trans_fn(q, t):
            for (tt, qn) in csr.get_transitions(q):
                if tt == t:
                    return qn
            return None
    else:
        # Real model: accept full vocab (100 is too small for 152K vocab - model tokens rarely in 0-99)
        csr, start_state, live_states = build_permissive_dfa(vocab_size, valid_tokens=list(range(vocab_size)))
        num_states = 2

        def trans_fn(q, t):
            if t < vocab_size:
                return 1
            return None

    all_results = {m: [] for m in ["Baseline", "Ablation1", "Ablation2", "Ablation3", "SDSD"]}

    for si, (src, prompt, idx) in tqdm(enumerate(prob_sources[:num_samples]), total=num_samples):
        seed = 42 + idx
        sample_res = run_one_sample(
            get_logits_fn,
            get_block_logits_fn,
            csr,
            trans_fn,
            num_states,
            vocab_size,
            start_state,
            live_states,
            block_length,
            prompt,
            seed,
        )
        for name, r in sample_res.items():
            all_results[name].append(r)

    # Aggregate
    summary = {}
    for name in ["Baseline", "Ablation1", "Ablation2", "Ablation3", "SDSD"]:
        entries = all_results[name]
        n = len(entries)
        summary[name] = {
            "ttft_ms": sum(e["latency_ms"] for e in entries) / n if n else 0,
            "throughput_tok_s": (
                sum(len(e["tokens"]) for e in entries) / max(1e-6, sum(e["latency_ms"] for e in entries) / 1000)
                if n else 0
            ),
            "nfe_avg": sum(e["nfe"] for e in entries) / n if n else 0,
            "parse_rate": sum(1 for e in entries if e["success"]) / n * 100 if n else 0,
            "n_samples": n,
        }

    return {
        "model": model_name,
        "mock": mock,
        "num_samples": num_samples,
        "block_length": block_length,
        "dataset": dataset or "default",
        "summary": summary,
    }


def print_table(summary: dict, model_name: str):
    """Print ablation comparison table."""
    methods = [
        ("Baseline", "DINGO ($O(N)$)", "$O(N)$"),
        ("Ablation1", "STATIC + DINGO", "$O(K)$"),
        ("Ablation2", "DINGO + Herding", "$O(N)$"),
        ("Ablation3", "STATIC + Spec-Tree", "$O(K)$"),
        ("SDSD", "STATIC + Herding + Tree", "$O(K)$"),
    ]
    print(f"\n{'='*100}")
    print(f"  SDSD Ablation Table — {model_name.upper()}")
    print(f"{'='*100}")
    print(f"{'Method':<12} {'Technique':<28} {'Complexity':<10} {'TTFT(ms)':<12} {'Throughput':<14} {'NFE':<10} {'Parse%':<10}")
    print("-" * 100)
    for key, tech, compl in methods:
        s = summary.get(key, {})
        print(f"{key:<12} {tech:<28} {compl:<10} {s.get('ttft_ms', 0):<12.2f} {s.get('throughput_tok_s', 0):<14.2f} {s.get('nfe_avg', 0):<10.2f} {s.get('parse_rate', 0):<10.1f}")
    print(f"{'='*100}\n")


def main():
    print("Starting ablation...", flush=True)
    parser = argparse.ArgumentParser(description="SDSD Ablation Experiment")
    parser.add_argument("--model", choices=["dream", "llada"], default="dream")
    parser.add_argument("--mock", action="store_true", help="Synthetic data (no GPU)")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None, help="json-mode-eval, humaneval, mbpp, gsm-symbolic")
    parser.add_argument("--dataset-limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/ablation_results.json")
    args = parser.parse_args()

    print("Running SDSD Ablation...")
    print(f"  Model: {args.model}, Mock: {args.mock}, Samples: {args.samples}")

    results = run_ablation(
        args.model, args.samples, args.block_length,
        mock=args.mock, dataset=args.dataset, dataset_limit=args.dataset_limit,
    )

    print_table(results["summary"], args.model)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
