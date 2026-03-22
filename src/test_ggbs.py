"""Unit tests for GGBS (no GPU; mock TokenChecker with matcher rollback)."""

from __future__ import annotations

import torch

from ggbs import segmented_ggbs_beam


class _MockMatcher:
    """Incremental consume matching _MockChecker depth rules."""

    def __init__(self, parent: "_MockChecker"):
        self._p = parent

    def try_consume_tokens(self, token_ids: list[int]) -> int:
        n = 0
        for t in token_ids:
            if t != self._p._depth % 3:
                return n
            self._p._depth += 1
            n += 1
        return n

    def rollback(self, count: int) -> bool:
        self._p._depth -= count
        return True


class _MockChecker:
    """Accepts only token d in position d mod 3 (chain)."""

    def __init__(self) -> None:
        self._depth = 0
        self.matcher = _MockMatcher(self)

    def clone(self) -> "_MockChecker":
        c = _MockChecker()
        c._depth = self._depth
        c.matcher = _MockMatcher(c)
        return c

    def reset(self) -> None:
        self._depth = 0

    def consume_tokens(self, token_ids: list[int]) -> bool:
        for t in token_ids:
            if self.matcher.try_consume_tokens([t]) != 1:
                return False
        return True

    def compute_mask(self, vocab_size: int = 126464) -> torch.Tensor:
        m = torch.ones(vocab_size, dtype=torch.bool)
        ok = self._depth % 3
        if ok < vocab_size:
            m[ok] = False
        return m


def test_ggbs_finds_chain():
    checker = _MockChecker()
    segments = [{"type": "mask", "probs": [[0.1, 0.9, 0.0]]}]
    r = segmented_ggbs_beam(
        checker,
        prefix_tokens=[],
        segments=segments,
        mask_positions=[0],
        suffix_tokens=[],
        vocab_size=3,
        beam_size=4,
    )
    assert r["success"]
    assert r["tokens"] == [0]


def test_suffix_prune_last_mask():
    """Last mask must allow consuming suffix [1]."""
    checker = _MockChecker()
    segments = [{"type": "mask", "probs": [[0.5, 0.5, 0.0]]}]
    r = segmented_ggbs_beam(
        checker,
        prefix_tokens=[],
        segments=segments,
        mask_positions=[5],
        suffix_tokens=[1],
        vocab_size=3,
        beam_size=4,
    )
    assert r["success"]
    assert r["tokens"] == [0]


if __name__ == "__main__":
    test_ggbs_finds_chain()
    test_suffix_prune_last_mask()
    print("ok")
