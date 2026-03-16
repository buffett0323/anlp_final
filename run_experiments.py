#!/usr/bin/env python3
"""
Run SDSD experiments: B1/B2/B3/SDSD evaluation + Intent Recovery ablation.

Usage:
  python run_experiments.py --model dream --mock          # Synthetic (no GPU)
  python run_experiments.py --model dream                 # Dream-7B (GPU)
  python run_experiments.py --model llada                 # LLaDA-8B (GPU)
  python run_experiments.py --intent-recovery-only        # Intent recovery only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="SDSD Experiments")
    parser.add_argument("--model", choices=["dream", "llada"], default="dream")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data (no GPU)")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--intent-recovery-only", action="store_true")
    parser.add_argument("--output", type=str, default="results/experiment_results.json")
    args = parser.parse_args()

    results = {}

    # Intent Recovery Ablation
    print("\n" + "=" * 60)
    print("INTENT RECOVERY ABLATION")
    print("=" * 60)
    try:
        from eval.intent_recovery import run_intent_recovery
        ir = run_intent_recovery()
        results["intent_recovery"] = ir
        print("  B2 (STATIC+DINGO):", ir["B2_tokens"], "| recovery steps:", ir["B2_recovery_steps"])
        print("  Herding (SDSD):   ", ir["Herding_tokens"], "| recovery steps:", ir["Herding_recovery_steps"])
        if ir["Herding_recovery_steps"] < ir["B2_recovery_steps"]:
            print("  -> Herding recovers faster (momentum preserves blocked intent)")
    except Exception as e:
        print(f"  Error: {e}")
        results["intent_recovery"] = {"error": str(e)}

    if args.intent_recovery_only:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        return

    # Full evaluation
    print("\n" + "=" * 60)
    print("B1/B2/B3/SDSD EVALUATION")
    print("=" * 60)
    try:
        from eval.evaluate import run_evaluation
        ev = run_evaluation(args.model, args.samples, args.block_length, args.mock)
        results["evaluation"] = ev
        for name, s in ev["summary"].items():
            print(f"\n{name}:")
            print(f"  Latency: {s['latency_ms_mean']:.3f} ms")
            print(f"  NFE: {s['nfe_mean']:.2f}")
            print(f"  Success: {s['success_rate']*100:.1f}%")
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        results["evaluation"] = {"error": str(e)}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ensure results are JSON-serializable
    import math
    def _make_serializable(obj):
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        return str(obj)
    with open(out, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
