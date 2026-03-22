"""
Grammar-Guided Beam Search (GGBS) on incremental llguidance / TokenChecker states.

Beams carry ``full_tokens`` (committed prefix through the gap). Each step does
one ``_checker_at_path`` (``clone`` + ``reset`` + ``consume_tokens(path)``) per
beam—O(|path|) per beam, not per vocabulary item. Mask expansion then uses the
matcher’s ``try_consume_tokens`` + ``rollback`` per candidate token (no full
``compute_mask`` over |V|). If ``matcher`` is missing, the code falls back to
``replay_checker`` for that candidate only.

Suffix feasibility on the last mask runs on the sorted candidate list (bounded
pool, then full scan if needed), not on every valid token before scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import torch

from bidirectional_dingo import FixedSeg, MaskSeg


class _CheckerProto(Protocol):
    def clone(self) -> Any: ...
    def reset(self) -> None: ...
    def consume_tokens(self, token_ids: list[int]) -> bool: ...
    def compute_mask(self, vocab_size: int = 126464) -> torch.Tensor: ...  # optional fallback


class _MatcherProto(Protocol):
    def try_consume_tokens(self, token_ids: list[int]) -> int: ...
    def rollback(self, count: int) -> bool: ...


class GGBSResult(TypedDict):
    tokens: list[int]
    mask_positions: list[int]
    success: bool


def replay_checker(checker: _CheckerProto, token_ids: list[int]) -> Any | None:
    """Fresh matcher advanced along ``token_ids``; ``None`` if consume fails."""
    c = checker.clone()
    c.reset()
    if not token_ids:
        return c
    if not c.consume_tokens(token_ids):
        return None
    return c


def valid_next_tokens(checker: _CheckerProto, vocab_size: int) -> list[int]:
    """Full-vocabulary legality via llguidance mask (expensive; avoid in hot paths)."""
    bias = checker.compute_mask(vocab_size=vocab_size)
    n = min(vocab_size, int(bias.shape[0]))
    return [t for t in range(n) if not bool(bias[t].item())]


def _valid_tokens_approach_a_scan(
    chk: Any,
    template: _CheckerProto,
    full_tokens: list[int],
    vocab_size: int,
) -> list[int]:
    """
    Approach A: legality by single-token probes only (no ``compute_mask``).

    ``TokenChecker.clone()`` does not duplicate matcher state; probes use
    ``try_consume`` + ``rollback`` on ``chk`` (same as anlp_final GGBS helpers).
    """
    out: list[int] = []
    for tok in range(vocab_size):
        if _grammar_allows_single_token(chk, template, full_tokens, tok):
            out.append(tok)
    return out


def token_is_valid_after_prefix(
    checker: _CheckerProto, prefix_tokens: list[int], tok: int
) -> bool:
    """Whether ``tok`` is allowed immediately after ``prefix_tokens`` (no full vocab mask)."""
    c = replay_checker(checker, prefix_tokens)
    if c is None:
        return False
    matcher = getattr(c, "matcher", None)
    if matcher is not None:
        cnt = matcher.try_consume_tokens([tok])
        if cnt == 1:
            matcher.rollback(1)
            return True
        return False
    return replay_checker(checker, prefix_tokens + [tok]) is not None


def _grammar_allows_single_token(
    chk: Any,
    template: _CheckerProto,
    full_tokens: list[int],
    tok: int,
) -> bool:
    """Single-token probe without full-vocab mask (Approach A)."""
    matcher = getattr(chk, "matcher", None)
    if matcher is not None:
        cnt = matcher.try_consume_tokens([tok])
        if cnt == 1:
            matcher.rollback(1)
            return True
        return False
    return replay_checker(template, full_tokens + [tok]) is not None


def grammar_valid_tokens_topk(
    chk: Any,
    template: _CheckerProto,
    full_tokens: list[int],
    pv: list[float],
    vocab_size: int,
    beam_size: int,
    *,
    grammar_topk: int,
    grammar_max_topk: int,
) -> list[int]:
    """
    Grammar-valid token ids ranked by model probability, probing top-K logits first.

    Expands K (doubling) until enough candidates or ``grammar_max_topk``. If still
    empty, falls back to a linear ``try_consume`` scan (no ``compute_mask``).
    """
    n = min(len(pv), vocab_size)
    if n <= 0:
        return []

    pv_t = torch.as_tensor(pv[:n], dtype=torch.float32)
    k_eff = min(max(1, grammar_topk), n)
    max_k = min(max(grammar_topk, grammar_max_topk), n)

    accepted: list[int] = []
    accepted_set: set[int] = set()
    rejected: set[int] = set()

    while k_eff <= max_k:
        _, idx = torch.topk(pv_t, k=min(k_eff, n), largest=True)
        for tok in idx.tolist():
            if tok in accepted_set or tok in rejected:
                continue
            if _grammar_allows_single_token(chk, template, full_tokens, tok):
                accepted_set.add(tok)
                accepted.append(tok)
            else:
                rejected.add(tok)
        need = max(beam_size * 2, beam_size)
        if len(accepted) >= need or k_eff >= n:
            break
        next_k = min(k_eff * 2, n, max_k)
        if next_k <= k_eff:
            break
        k_eff = next_k

    if not accepted:
        accepted = _valid_tokens_approach_a_scan(chk, template, full_tokens, n)

    accepted.sort(key=lambda t: -float(pv[t]) if t < len(pv) else 0.0)
    return accepted


def _log_p(p: float) -> float:
    return math.log(p) if p > 0.0 else float("-inf")


def _checker_at_path(checker: _CheckerProto, full_tokens: list[int]) -> Any | None:
    """One replay from grammar start to ``full_tokens`` (used once per beam per step)."""
    c = checker.clone()
    c.reset()
    if not full_tokens:
        return c
    if not c.consume_tokens(full_tokens):
        return None
    return c


def _suffix_ok_after_path(
    template: _CheckerProto, full_tokens: list[int], suffix: list[int]
) -> bool:
    """Whether ``full_tokens`` + ``suffix`` is consumable (state after last mask + suffix)."""
    if not suffix:
        return True
    c = replay_checker(template, full_tokens)
    if c is None:
        return False
    for s in suffix:
        if not c.consume_tokens([s]):
            return False
    return True


@dataclass
class _Beam:
    log_prob: float
    mask_tokens: list[int]
    full_tokens: list[int]


def segmented_ggbs_beam(
    checker: _CheckerProto,
    *,
    prefix_tokens: list[int],
    segments: list[FixedSeg | MaskSeg],
    mask_positions: list[int],
    suffix_tokens: list[int],
    vocab_size: int,
    beam_size: int,
    suffix_survive_mult: int = 3,
    grammar_topk: int = 256,
    grammar_max_topk: int = 8192,
) -> GGBSResult:
    """
    Beam search over alternating fixed / mask segments; each mask column ranks
    grammar-valid tokens that appear in the model’s top-``grammar_topk`` (expanded
    up to ``grammar_max_topk``) by log-prob, not the full legal vocabulary.
    """
    if not mask_positions:
        return GGBSResult(tokens=[], mask_positions=[], success=False)

    init = _checker_at_path(checker, prefix_tokens)
    if init is None:
        return GGBSResult(tokens=[], mask_positions=[], success=False)

    beams: list[_Beam] = [
        _Beam(log_prob=0.0, mask_tokens=[], full_tokens=list(prefix_tokens))
    ]

    flat: list[tuple[str, Any]] = []
    mi = 0
    for seg in segments:
        if seg["type"] == "fixed":
            for t in seg["tokens"]:
                flat.append(("fixed", int(t)))
        else:
            for _, pv in enumerate(seg["probs"]):
                if mi >= len(mask_positions):
                    break
                flat.append(("mask", (pv, mi)))
                mi += 1

    last_mask_order_idx = len(mask_positions) - 1

    for kind, payload in flat:
        if kind == "fixed":
            t = int(payload)
            new_beams: list[_Beam] = []
            for beam in beams:
                chk = _checker_at_path(checker, beam.full_tokens)
                if chk is None:
                    continue
                matcher = getattr(chk, "matcher", None)
                if matcher is not None:
                    if matcher.try_consume_tokens([t]) != 1:
                        continue
                elif replay_checker(checker, beam.full_tokens + [t]) is None:
                    continue
                new_beams.append(
                    _Beam(
                        log_prob=beam.log_prob,
                        mask_tokens=list(beam.mask_tokens),
                        full_tokens=beam.full_tokens + [t],
                    )
                )
            beams = new_beams
        else:
            pv, order_idx = payload
            is_last_mask = order_idx == last_mask_order_idx
            candidates: list[_Beam] = []

            for beam in beams:
                chk = _checker_at_path(checker, beam.full_tokens)
                if chk is None:
                    continue
                valid = grammar_valid_tokens_topk(
                    chk,
                    checker,
                    beam.full_tokens,
                    pv,
                    vocab_size,
                    beam_size,
                    grammar_topk=grammar_topk,
                    grammar_max_topk=grammar_max_topk,
                )
                scored: list[tuple[int, float]] = []
                for tok in valid:
                    lp = _log_p(pv[tok] if tok < len(pv) else 0.0)
                    if not math.isfinite(lp):
                        continue
                    scored.append((tok, beam.log_prob + lp))
                scored.sort(key=lambda x: -x[1])

                for tok, sc in scored[:beam_size]:
                    candidates.append(
                        _Beam(
                            log_prob=sc,
                            mask_tokens=beam.mask_tokens + [tok],
                            full_tokens=beam.full_tokens + [tok],
                        )
                    )

            candidates.sort(key=lambda b: -b.log_prob)

            if is_last_mask and suffix_tokens:
                pool = min(len(candidates), max(beam_size * suffix_survive_mult, beam_size))
                surviving: list[_Beam] = []
                for b in candidates[:pool]:
                    if _suffix_ok_after_path(checker, b.full_tokens, suffix_tokens):
                        surviving.append(b)
                    if len(surviving) >= beam_size:
                        break
                if not surviving:
                    for b in candidates[pool:]:
                        if _suffix_ok_after_path(checker, b.full_tokens, suffix_tokens):
                            surviving.append(b)
                        if len(surviving) >= beam_size:
                            break
                if not surviving:
                    return GGBSResult(tokens=[], mask_positions=[], success=False)
                beams = surviving
            else:
                beams = candidates[:beam_size]

        if not beams:
            return GGBSResult(tokens=[], mask_positions=[], success=False)

    if not beams:
        return GGBSResult(tokens=[], mask_positions=[], success=False)

    best = max(beams, key=lambda b: b.log_prob)
    if not math.isfinite(best.log_prob):
        return GGBSResult(tokens=[], mask_positions=[], success=False)

    if len(best.mask_tokens) != len(mask_positions):
        return GGBSResult(tokens=[], mask_positions=[], success=False)

    return GGBSResult(
        tokens=best.mask_tokens,
        mask_positions=list(mask_positions),
        success=True,
    )
