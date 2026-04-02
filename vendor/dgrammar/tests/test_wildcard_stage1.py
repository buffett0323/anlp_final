"""
Stage 1 correctness tests for wildcard_earley_verify.

Grammar under test:  S → ( S ) | ε
Language:            { (^n )^n | n ≥ 0 }  =  { ε, (), (()), ((())) … }

The four cases from wildcard_detail.md plus basic sanity checks.

Run directly:   python tests/test_wildcard_stage1.py
Run via pytest: pytest tests/test_wildcard_stage1.py -v
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dgrammar.wildcard_earley import Grammar, Rule, MASK, wildcard_earley_verify


# ── Grammar factory ───────────────────────────────────────────────────────────


def balanced_paren_grammar() -> Grammar:
    """S → ( S ) | ε"""
    return Grammar(
        start="S",
        rules=[
            Rule("S", ("(", "S", ")")),
            Rule("S", ()),  # S → ε
        ],
    )


# ── Helper ────────────────────────────────────────────────────────────────────


def verify(tokens: list[str]) -> bool:
    g = balanced_paren_grammar()
    total = tokens.count(MASK)
    return wildcard_earley_verify(tokens, g, total)


# ── Cases from wildcard_detail.md (Stage 1) ───────────────────────────────────


def test_case1_open_mask_close():
    """( * )  →  True   (MASK as ε → effective sequence is "()")"""
    assert verify(["(", MASK, ")"]) is True


def test_case2_open_three_masks_close():
    """( * * * )  →  True   (all MASKs as ε → effective sequence is "()")"""
    assert verify(["(", MASK, MASK, MASK, ")"]) is True


def test_case3_parens_mask_parens_mask():
    """( ) * ( ) *  →  False   (()() is not in L(S))"""
    assert verify(["(", ")", MASK, "(", ")", MASK]) is False


def test_case4_two_masks_then_close():
    """* * )  →  False   (no assignment makes X Y ) a valid S)"""
    assert verify([MASK, MASK, ")"]) is False


# ── Basic sanity checks ───────────────────────────────────────────────────────


def test_empty_is_valid():
    """ε  →  True   (S → ε)"""
    assert verify([]) is True


def test_single_pair():
    """()  →  True"""
    assert verify(["(", ")"]) is True


def test_double_nested():
    """(())  →  True"""
    assert verify(["(", "(", ")", ")"]) is True


def test_unbalanced_close():
    """)  →  False"""
    assert verify([")"]) is False


def test_unbalanced_open():
    """(  →  False   (incomplete; no wildcards to close it)"""
    assert verify(["("]) is False


def test_mask_only_single():
    """*  →  True   (MASK as ε → ε, which is valid S)"""
    assert verify([MASK]) is True


def test_mask_only_two():
    """* *  →  True   (both MASKs as ε → ε, which is valid S)"""
    assert verify([MASK, MASK]) is True


def test_open_mask_mask_close():
    """( * * )  →  True   (both MASKs as ε → "()")"""
    assert verify(["(", MASK, MASK, ")"]) is True


# ── Runner ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    passed = failed = 0
    for name, fn in tests:
        doc = (inspect.getdoc(fn) or name).strip()
        try:
            fn()
            print(f"  PASS  {doc}")
            passed += 1
        except AssertionError:
            print(f"  FAIL  {doc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR {doc}  ({exc})")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
