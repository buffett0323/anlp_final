"""
Auto-scan results/ and write a comparison.md.

Groups JSONL files by stripping trailing _offNNN shards, merges records
(dedup by instance_id, last shard wins), computes stats, and writes markdown.

Usage:
    python bench/compare_results.py
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_MD   = RESULTS_DIR / "comparison.md"

# Human-readable display names keyed on the method field inside records.
# Falls back to the raw method string if not listed here.
METHOD_LABELS = {
    "lave":             "LAVE",
    "dgrammar_v2_async": "Dgrammar",
    "dgrammar_dp":      "DPGrammar",
}

# Preferred display order (others appended alphabetically).
METHOD_ORDER = ["lave", "dgrammar_v2_async", "dgrammar_dp"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _base_name(stem: str) -> str:
    """Strip trailing _offNNN from a file stem."""
    return re.sub(r"_off\d+$", "", stem)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def avg(lst):
    lst = [x for x in lst if x is not None and not math.isnan(x)]
    return sum(lst) / len(lst) if lst else float("nan")


def pct(count, total):
    return count / total * 100 if total else float("nan")


def compute_stats(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}

    n = len(records)
    valid_count = sum(1 for r in records if r.get("valid"))

    times      = [r["time_taken"]  for r in records if "time_taken"  in r]
    resamples  = [r["resamples"]   for r in records if "resamples"   in r]

    def tget(r, *keys):
        t = r.get("timing", {})
        for k in keys:
            if k in t:
                return t[k]
        return None

    fwd_counts   = [x for r in records if (x := tget(r, "forward_count"))       is not None]
    con_pcts     = [x for r in records if (x := tget(r, "constraint_pct"))       is not None]
    eff_pcts     = [x for r in records if (x := tget(r, "effective_constraint_pct")) is not None]
    pt_total     = [x for r in records if (x := tget(r, "per_token_total_ms"))   is not None]
    pt_con       = [x for r in records if (x := tget(r, "per_token_constraint_ms")) is not None]
    mask_ms      = [x for r in records if (x := tget(r, "mask_compute_total_ms")) is not None]
    fwd_ms       = [x for r in records if (x := tget(r, "forward_total_ms", "total_forward_ms")) is not None]

    return {
        "n":                    n,
        "valid_count":          valid_count,
        "valid_rate":           valid_count / n,
        "avg_time_s":           avg(times),
        "avg_resamples":        avg(resamples),
        "avg_fwd_count":        avg(fwd_counts),
        "avg_constraint_pct":   avg(con_pcts),
        "avg_eff_constraint_pct": avg(eff_pcts),
        "avg_pt_total_ms":      avg(pt_total),
        "avg_pt_con_ms":        avg(pt_con),
        "avg_mask_ms":          avg(mask_ms),
        "avg_fwd_ms":           avg(fwd_ms),
    }


# ── load & merge ───────────────────────────────────────────────────────────────

def load_all() -> dict[str, dict[str, dict]]:
    """
    Returns:
        { base_name -> { instance_id -> record } }
    """
    groups: dict[str, dict[str, dict]] = defaultdict(dict)

    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        base = _base_name(path.stem)
        records = load_jsonl(path)
        for rec in records:
            iid = rec.get("instance_id")
            if iid:
                groups[base][iid] = rec  # last shard wins on duplicate

    return groups


# ── markdown helpers ───────────────────────────────────────────────────────────

def _fmt(val, fmt):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return "—"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep   = "| " + " | ".join("-" * w for w in widths) + " |"
    head  = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    groups = load_all()

    if not groups:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    # Identify method key for each group from first record's "method" field.
    group_method: dict[str, str] = {}
    for base, recs_by_id in groups.items():
        first = next(iter(recs_by_id.values()), {})
        group_method[base] = first.get("method", base)

    # Sort groups: preferred order first, then alphabetical.
    def sort_key(base):
        m = group_method[base]
        try:
            return (METHOD_ORDER.index(m), base)
        except ValueError:
            return (len(METHOD_ORDER), base)

    sorted_bases = sorted(groups.keys(), key=sort_key)

    # Compute stats per group.
    stats: dict[str, dict] = {}
    records_by_group: dict[str, list[dict]] = {}
    for base in sorted_bases:
        recs = list(groups[base].values())
        records_by_group[base] = recs
        stats[base] = compute_stats(recs)

    # Display labels.
    def label(base):
        m = group_method[base]
        return METHOD_LABELS.get(m, m)

    labels = [label(b) for b in sorted_bases]

    # ── Build markdown ─────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Results Comparison")
    lines.append("")
    lines.append(f"**Dataset:** jsb\\_medium &nbsp;|&nbsp; **Seed:** 0 &nbsp;|&nbsp; **Steps:** 128")
    lines.append("")

    # ── Summary table ──────────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")

    metrics = [
        ("n",                     "N instances",              "{:d}"),
        ("valid_count",           "Valid (count)",             "{:d}"),
        ("valid_rate",            "Validity (%)",              "{:.1%}"),
        ("avg_time_s",            "Avg time (s)",              "{:.2f}"),
        ("avg_resamples",         "Avg resamples",             "{:.2f}"),
        ("avg_fwd_count",         "Avg fwd passes",            "{:.1f}"),
        ("avg_constraint_pct",    "Constraint overhead (%)",   "{:.2f}"),
        ("avg_eff_constraint_pct","Eff. constraint (%)",       "{:.2f}"),
        ("avg_pt_total_ms",       "Per-token total (ms)",      "{:.2f}"),
        ("avg_pt_con_ms",         "Per-token constraint (ms)", "{:.3f}"),
        ("avg_mask_ms",           "Avg mask compute (ms)",     "{:.1f}"),
        ("avg_fwd_ms",            "Avg forward total (ms)",    "{:.1f}"),
    ]

    headers = ["Metric"] + labels
    rows = []
    for key, display, fmt in metrics:
        row = [display]
        for base in sorted_bases:
            row.append(_fmt(stats[base].get(key), fmt))
        rows.append(row)

    lines.append(md_table(headers, rows))
    lines.append("")

    # ── Per-instance validity agreement ───────────────────────────────────────
    lines.append("## Per-instance Validity Agreement")
    lines.append("")
    lines.append(
        "Each row shows a validity pattern across methods "
        f"({', '.join(labels)}) and how many instances share that pattern."
    )
    lines.append("")

    all_ids: set[str] = set()
    for recs in records_by_group.values():
        all_ids |= {r["instance_id"] for r in recs}

    rec_lookup: dict[str, dict[str, dict]] = {
        base: {r["instance_id"]: r for r in recs}
        for base, recs in records_by_group.items()
    }

    agreement: dict[tuple, list[str]] = defaultdict(list)
    for iid in sorted(all_ids):
        pattern = tuple(
            rec_lookup[base].get(iid, {}).get("valid", None)
            for base in sorted_bases
        )
        agreement[pattern].append(iid)

    def pattern_str(pat):
        def sym(v):
            if v is True:  return "✓"
            if v is False: return "✗"
            return "—"
        return "  ".join(sym(v) for v in pat)

    agree_headers = ["Pattern (" + " / ".join(labels) + ")", "Count", "%", "Example IDs"]
    agree_rows = []
    for pat, ids in sorted(agreement.items(), key=lambda x: -len(x[1])):
        count = len(ids)
        pct_val = count / len(all_ids) * 100
        examples = ", ".join(ids[:3]) + ("…" if len(ids) > 3 else "")
        agree_rows.append([
            pattern_str(pat),
            str(count),
            f"{pct_val:.1f}%",
            examples,
        ])

    lines.append(md_table(agree_headers, agree_rows))
    lines.append("")
    lines.append(f"Total unique instances across all files: **{len(all_ids)}**")
    lines.append("")

    # ── Instances where methods disagree ──────────────────────────────────────
    if len(sorted_bases) > 1:
        lines.append("## Disagreement Cases")
        lines.append("")
        lines.append("Instances valid for some methods but not others (sorted by instance ID):")
        lines.append("")

        dis_headers = ["Instance ID"] + labels
        dis_rows = []
        for iid in sorted(all_ids):
            valids = [rec_lookup[base].get(iid, {}).get("valid", None) for base in sorted_bases]
            truthy = [v for v in valids if v is not None]
            if truthy and not all(v == truthy[0] for v in truthy):
                def sym(v):
                    if v is True:  return "✓"
                    if v is False: return "✗"
                    return "—"
                dis_rows.append([iid] + [sym(v) for v in valids])

        if dis_rows:
            lines.append(md_table(dis_headers, dis_rows))
        else:
            lines.append("_No disagreements found._")
        lines.append("")

    # ── File inventory ─────────────────────────────────────────────────────────
    lines.append("## File Inventory")
    lines.append("")
    inv_headers = ["Group (base name)", "Method", "Shards", "Total records"]
    inv_rows = []
    for base in sorted_bases:
        shards = sorted(
            p for p in RESULTS_DIR.glob("*.jsonl")
            if _base_name(p.stem) == base
        )
        n_total = sum(sum(1 for _ in open(s)) for s in shards)
        inv_rows.append([
            base,
            group_method[base],
            str(len(shards)),
            str(n_total),
        ])
    lines.append(md_table(inv_headers, inv_rows))
    lines.append("")

    md_content = "\n".join(lines)
    OUTPUT_MD.write_text(md_content)
    print(f"Written: {OUTPUT_MD}")
    print()

    # Also print summary to stdout.
    for base, lbl in zip(sorted_bases, labels):
        s = stats[base]
        print(f"  {lbl:20s}  n={s['n']:4d}  valid={s['valid_rate']:.1%}  "
              f"time={s['avg_time_s']:.1f}s  resamples={s['avg_resamples']:.1f}")


if __name__ == "__main__":
    main()
