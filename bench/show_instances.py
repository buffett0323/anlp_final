"""
Show GT (schema) / LAVE / DGrammar / DPGrammar side-by-side for given IDs.

Usage:
    python bench/show_instances.py o5395,o14478,o21285,o45752
    python bench/show_instances.py o5395,o14478,o21285,o45752 --full
    python bench/show_instances.py o5395 --diff
    python bench/show_instances.py o21285 --dp-tag pwdp
    python bench/show_instances.py          # all validity-divergent cases
    python bench/show_instances.py --no-schema
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RESULTS  = Path(__file__).parent.parent / "results"

DG_PATTERN   = "v2_async_ac4_timed_jsb_medium_s0_t128*.jsonl"
DP_PATTERN   = "dp_jsb_medium_s0_t128*.jsonl"
LAVE_PATTERN = "lave_timed_jsb_medium_s0_t128*.jsonl"

_HF_DATASET = "epfl-dlab/JSONSchemaBench"
_HF_SUBSET  = "Github_medium"

SEP  = "=" * 74
RULE = "─" * 74


# ── loading ───────────────────────────────────────────────────────────────────

def load_results(pattern: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for p in sorted(RESULTS.glob(pattern)):
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = r.get("instance_id")
            if iid:
                records[iid] = r
    return records


def load_schemas() -> dict[str, str]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("[WARN] 'datasets' not installed — schema display disabled.")
        return {}
    schemas: dict[str, str] = {}
    try:
        ds = hf_load(_HF_DATASET, name=_HF_SUBSET)
        for split in ds:
            for row in ds[split]:
                schemas[row["unique_id"]] = row["json_schema"]
    except Exception as e:
        print(f"[WARN] Could not load schemas: {e}")
    return schemas


# ── formatting ────────────────────────────────────────────────────────────────

def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + l for l in text.splitlines())


def _pretty_json(s: str) -> str:
    """Try to pretty-print a JSON string; return original if it fails."""
    try:
        return json.dumps(json.loads(s), indent=2, ensure_ascii=False)
    except Exception:
        return s


def fmt_schema(schema_str: str, max_lines: int | None = 40) -> str:
    try:
        pretty = json.dumps(json.loads(schema_str), indent=2, ensure_ascii=False)
    except Exception:
        pretty = schema_str
    lines = pretty.splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more lines)"]
    return _indent("\n".join(lines))


def fmt_output(r: dict | None, max_lines: int | None = 40) -> str:
    if r is None:
        return "    (no result)"
    valid     = r.get("valid")
    resamples = r.get("resamples", "?")
    t         = r.get("time_taken")
    t_str     = f"{t:.1f}s" if isinstance(t, (int, float)) else "?"
    fwd       = r.get("timing", {}).get("forward_count", "?")
    header    = f"    valid={valid}  resamples={resamples}  time={t_str}  fwd_passes={fwd}"

    ext   = _pretty_json(r.get("extracted") or "")
    lines = ext.splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more lines)"]
    return header + "\n" + _indent("\n".join(lines))


def diff_outputs(a: str, b: str, label_a: str = "A", label_b: str = "B") -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fa:
        fa.write(_pretty_json(a))
        fa_name = fa.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fb:
        fb.write(_pretty_json(b))
        fb_name = fb.name
    try:
        res = subprocess.run(
            ["diff", "-u", "--label", label_a, "--label", label_b, fa_name, fb_name],
            capture_output=True, text=True,
        )
        return res.stdout or "(no diff — outputs are identical)"
    finally:
        os.unlink(fa_name)
        os.unlink(fb_name)


# ── display ───────────────────────────────────────────────────────────────────

def show(
    lave: dict, dg: dict, dp: dict, schemas: dict,
    iid: str,
    max_lines: int | None = 40,
    do_diff: bool = False,
    show_schema: bool = True,
):
    lave_r   = lave.get(iid)
    dg_r     = dg.get(iid)
    dp_r     = dp.get(iid)
    schema_s = schemas.get(iid)

    print(f"\n{SEP}")
    print(f"  Instance: {iid}")

    if show_schema:
        print(RULE)
        print("  GT (JSON Schema — any valid instance is acceptable):")
        if schema_s:
            print(fmt_schema(schema_s, max_lines=max_lines))
        else:
            print("    (schema not loaded)")

    print(RULE)
    print("  LAVE:")
    print(fmt_output(lave_r, max_lines=max_lines))

    print(RULE)
    print("  DGrammar:")
    print(fmt_output(dg_r, max_lines=max_lines))

    print(RULE)
    print("  DPGrammar:")
    print(fmt_output(dp_r, max_lines=max_lines))

    if do_diff and dg_r and dp_r:
        print(RULE)
        print("  Diff (DGrammar → DPGrammar):")
        print(_indent(diff_outputs(
            dg_r.get("extracted", ""),
            dp_r.get("extracted", ""),
            label_a="DGrammar", label_b="DPGrammar",
        ), "    "))

    print(SEP)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = list(sys.argv[1:])

    full      = "--full"      in args; args = [a for a in args if a != "--full"]
    do_diff   = "--diff"      in args; args = [a for a in args if a != "--diff"]
    no_schema = "--no-schema" in args; args = [a for a in args if a != "--no-schema"]

    dp_tag = ""
    if "--dp-tag" in args:
        idx    = args.index("--dp-tag")
        dp_tag = args[idx + 1]
        args   = args[:idx] + args[idx + 2:]

    max_lines = None if full else 40

    dp_pattern = (
        f"dp_jsb_medium_s0_t128*_{dp_tag}*.jsonl" if dp_tag else DP_PATTERN
    )

    lave = load_results(LAVE_PATTERN)
    dg   = load_results(DG_PATTERN)
    dp   = load_results(dp_pattern)

    print(f"Loaded LAVE      : {len(lave)} instances")
    print(f"Loaded DGrammar  : {len(dg)} instances")
    print(f"Loaded DPGrammar : {len(dp)} instances  (pattern: {dp_pattern})")

    schemas: dict[str, str] = {}
    if not no_schema:
        print("Loading schemas from HuggingFace …")
        schemas = load_schemas()
        print(f"Loaded {len(schemas)} schemas")

    if args:
        ids = args[0].split(",")
    else:
        all_ids = set(lave) | set(dg) | set(dp)
        ids = sorted(
            iid for iid in all_ids
            if len({
                dg.get(iid, {}).get("valid"),
                dp.get(iid, {}).get("valid"),
                lave.get(iid, {}).get("valid"),
            }) > 1
        )
        if not ids:
            print("\nAll methods agree on validity for every instance.")
            return
        print(f"\nFound {len(ids)} instances where methods disagree:")
        for iid in ids:
            print(f"  {iid}  lave={lave.get(iid,{}).get('valid')}  "
                  f"dg={dg.get(iid,{}).get('valid')}  "
                  f"dp={dp.get(iid,{}).get('valid')}")

    for iid in ids:
        show(lave, dg, dp, schemas, iid,
             max_lines=max_lines, do_diff=do_diff, show_schema=not no_schema)


if __name__ == "__main__":
    main()
