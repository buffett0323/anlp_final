import copy
import json
import time
from typing import Tuple, List, Optional, Any, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import frozendict
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import concurrent.futures
from transformers import AutoTokenizer, AutoModel

from constrained_diffusion.constrain_utils import (
    EOS,
    EOSType,
)

from constrained_diffusion.checker_tokenizer import Checker
import heapq
import random

eos_id = 126081
eot_id = 126348
mask_id = 126336

cache_seq: List[int] = []

# Max mask columns in one bidirectional gap (same order of magnitude as SDSD).
MAX_BIDI_GAP = 48


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.
    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    if steps >= mask_num.max().item():
        num_transfer_tokens = torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        for i in range(mask_num.size(0)):
            for j in range(mask_num[i].item()):
                num_transfer_tokens[i, j] = 1
        return num_transfer_tokens
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_end_index(all_token_ids, num_block: int, block_length: int, prompt):
    end_index = 0
    for i in range(prompt.shape[1] + (num_block + 1) * block_length - 1, -1, -1):
        w = all_token_ids[i]
        if w not in (eos_id, eot_id, mask_id):
            end_index = i
            break
    return end_index


def min_eos_or_eot_index(all_token_ids, num_block: int, block_length: int, prompt):
    exist_mask = False
    for i in range(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length):
        w = all_token_ids[i]
        if w in (eos_id, eot_id):
            return i, exist_mask
        if w == mask_id:
            exist_mask = True
    return -1, exist_mask


def compute_logits_mask(
    checker: Checker,
):
    logits_mask = checker.compute_mask()
    logits_mask_long = torch.zeros(126464)
    logits_mask_long[0:len(logits_mask)] = logits_mask
    logits_mask = logits_mask_long
    logits_mask[mask_id] = 0
    if logits_mask[eos_id] == 1:
        logits_mask[eot_id] = 1
    if logits_mask[eot_id] == 1:
        logits_mask[eos_id] = 1
    return logits_mask


def _grammar_allows_token_lave(c_hyp: Checker, tid: int) -> bool:
    """
    Single-token legality without ``compute_mask`` (Approach A).

    ``Checker.clone()`` does not copy matcher state, so we probe with
    ``try_consume_tokens`` + ``rollback`` on the already-replayed ``c_hyp`` —
    same effect as “clone state + consume one token” would have if deep copy
    existed.
    """
    # Guard: token IDs >= llguidance vocab_size cause the matcher to enter a
    # stopped state (emitting a Rust-level warning), corrupting all subsequent
    # probes on the same clone.  Treat them as grammar-disallowed.
    if tid >= c_hyp.tokenizer.vocab_size:
        return False
    m = c_hyp.matcher
    cnt = m.try_consume_tokens([tid])
    if cnt == 1:
        m.rollback(1)
        return True
    return False


def _lave_ge_gt(c_hyp: Checker) -> Tuple[bool, bool]:
    """Grammar legality for eos and eot ids (for eos/eot mask symmetry)."""
    return (
        _grammar_allows_token_lave(c_hyp, eos_id),
        _grammar_allows_token_lave(c_hyp, eot_id),
    )


def _lave_tid_allowed_after_symmetry(
    c_hyp: Checker,
    tid: int,
    ge_gt: Optional[Tuple[bool, bool]] = None,
) -> bool:
    """
    Match ``compute_logits_mask`` semantics for eos/eot mutual visibility.
    Pass ``ge_gt`` from ``_lave_ge_gt`` when scanning many ids to avoid recomputing.
    """
    if tid == mask_id:
        return False
    if ge_gt is None:
        ge, gt = _lave_ge_gt(c_hyp)
    else:
        ge, gt = ge_gt
    if tid == eos_id or tid == eot_id:
        return ge or gt
    return _grammar_allows_token_lave(c_hyp, tid)


def _ensure_anlp_src_for_bidi() -> bool:
    """Add anlp_final/src to sys.path so bidirectional_dingo / csr_dfa can load."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[7]
    src = root / "src"
    if not src.is_dir():
        return False
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)
    return True


def _lave_lm_tensor(checker: Checker, device: torch.device) -> torch.Tensor:
    """Single LAVE mask at current matcher state (same for all gap rows)."""
    lm = compute_logits_mask(checker).to(device)
    return lm


def _lave_masked_probs_row(
    logits_row: torch.Tensor,
    lm: torch.Tensor,
) -> tuple[list[float], torch.Tensor]:
    """Softmax over LAVE-masked row; returns probs list + masked_logits for argmax fallback."""
    vs = logits_row.shape[0]
    if lm.shape[0] < vs:
        lrow = torch.cat(
            [
                lm,
                torch.zeros(
                    vs - lm.shape[0],
                    device=lm.device,
                    dtype=lm.dtype,
                ),
            ]
        )
    else:
        lrow = lm[:vs]
    masked_logits = logits_row.clone().float()
    masked_logits[lrow == 0] = float("-inf")
    if not torch.isfinite(masked_logits).any():
        if lrow.sum() <= 0:
            raise RuntimeError(
                "LAVE compute_mask has no legal token ids at this position"
            )
        legal = torch.nonzero(lrow > 0, as_tuple=False).squeeze(-1)
        pick = int(legal[0].item())
        out = [0.0] * vs
        out[pick] = 1.0
        return out, masked_logits
    probs = F.softmax(masked_logits.to(torch.float64), dim=-1).tolist()
    if len(probs) < vs:
        probs.extend([0.0] * (vs - len(probs)))
    return probs, masked_logits


def _build_gap_segments_lave(
    x: torch.Tensor,
    focus_pos: int,
    seq_len: int,
    mask_id: int,
    logits_with_noise: torch.Tensor,
    lm: torch.Tensor,
    max_masks: int,
) -> tuple[list[Any], list[int], int]:
    """
    Like SDSD ``_build_bidi_segments_from_x``, but each mask column uses
    LAVE-masked softmax from ``logits_with_noise`` (same ``lm`` for all rows).
    """
    segments: list[Any] = []
    mask_positions: list[int] = []
    pos = focus_pos
    mask_count = 0
    vs = logits_with_noise.shape[-1]
    while pos < seq_len and mask_count < max_masks:
        tid = int(x[0, pos].item())
        if tid == mask_id:
            row = logits_with_noise[0, pos]
            probs, _ = _lave_masked_probs_row(row, lm)
            segments.append({"type": "mask", "probs": [probs]})
            mask_positions.append(pos)
            mask_count += 1
            pos += 1
        else:
            fixed: List[int] = []
            while pos < seq_len and int(x[0, pos].item()) != mask_id:
                fixed.append(int(x[0, pos].item()))
                pos += 1
            if fixed:
                segments.append({"type": "fixed", "tokens": fixed})
    return segments, mask_positions, pos


def _suffix_after_gap(x: torch.Tensor, pos_after: int, seq_len: int, mask_id: int) -> List[int]:
    suffix: List[int] = []
    q = pos_after
    while q < seq_len:
        tid = int(x[0, q].item())
        if tid == mask_id:
            break
        suffix.append(tid)
        q += 1
    return suffix


def _suffix_after_focus(x: torch.Tensor, focus_pos: int, seq_len: int, mask_id: int) -> List[int]:
    """Fixed tokens strictly after ``focus_pos`` until the next mask (single-column BiDi suffix)."""
    return _suffix_after_gap(x, focus_pos + 1, seq_len, mask_id)


def _lrow_for_vs(lm: torch.Tensor, vs: int, device: torch.device) -> torch.Tensor:
    if lm.shape[0] >= vs:
        return lm[:vs]
    return torch.cat(
        [lm, torch.zeros(vs - lm.shape[0], device=device, dtype=lm.dtype)],
        dim=0,
    )


def _lave_argmax_on_row(row: torch.Tensor, lm: torch.Tensor) -> int:
    """Argmax on one position; illegal dims (lm==0) masked to -inf."""
    vs = row.shape[0]
    device = row.device
    lrow = _lrow_for_vs(lm, vs, device)
    masked = row.clone().float()
    masked[lrow == 0] = float("-inf")
    if not torch.isfinite(masked).any():
        legal = torch.nonzero(lrow > 0, as_tuple=False).squeeze(-1)
        return int(legal[0].item())
    return int(torch.argmax(masked).item())


def _lave_argmax_at_focus(
    logits_with_noise: torch.Tensor,
    focus_pos: int,
    lm: torch.Tensor,
) -> int:
    return _lave_argmax_on_row(logits_with_noise[0, focus_pos], lm)


def _clamp_csr_pick_to_lave(
    tid: int,
    lm: torch.Tensor,
    logits_row: torch.Tensor,
) -> int:
    """
    CSR / segmented BiDi can disagree with llguidance at this step. If ``tid`` is
    not allowed by ``compute_logits_mask`` (1=allowed), fall back to LAVE-legal argmax.
    """
    vs = logits_row.shape[0]
    lrow = _lrow_for_vs(lm, vs, logits_row.device)
    if 0 <= tid < lrow.shape[0] and lrow[tid].item() > 0:
        return tid
    return _lave_argmax_on_row(logits_row, lm)


def bidi_frontier_pick(
    logits_with_noise: torch.Tensor,
    x: torch.Tensor,
    focus_pos: int,
    checker: Checker,
    csr: Any,
    live_states: Set[int],
    mask_id: int,
    gen_start: int,
    json_start_state: int,
    bidi_ctx: dict,
    bidi_segmented_gap: bool = False,
) -> int:
    """
    Frontier picker: LAVE-masked logits + optional JSON-CSR bidirectional DP.

    **CSR vs LAVE:** The CSR from ``build_json_dfa_from_tokenizer`` is a loose
    JSON-shaped DFA; llguidance's matcher is the real grammar. We always mask
    logits with ``compute_logits_mask`` and **clamp** CSR picks to LAVE-legal
    tokens. ``check()`` can still reject (beam / future tokens); use
    ``bidi_segmented_gap=False`` (default) to avoid joint multi-mask DP that
    uses one mask for all columns and spikes retries.
    """
    if not _ensure_anlp_src_for_bidi():
        lm = _lave_lm_tensor(checker, logits_with_noise.device)
        return _lave_argmax_at_focus(logits_with_noise, focus_pos, lm)

    from bidirectional_dingo import bidirectional_gap_dingo, segmented_bidirectional_dingo
    from csr_dfa import dfa_run_csr

    device = logits_with_noise.device
    lm = _lave_lm_tensor(checker, device)
    vs = logits_with_noise.shape[-1]

    prefix_list = [
        int(x[0, j].item())
        for j in range(gen_start, focus_pos)
        if int(x[0, j].item()) != mask_id
    ]
    prefix_key = tuple(prefix_list)
    if bidi_ctx.get("prefix_key") == prefix_key and bidi_ctx.get("q_left") is not None:
        q_left = bidi_ctx["q_left"]
    else:
        q_left = dfa_run_csr(csr, json_start_state, prefix_list)
        bidi_ctx["prefix_key"] = prefix_key
        bidi_ctx["q_left"] = q_left

    if q_left is None:
        bidi_ctx["prefix_key"] = None
        bidi_ctx["q_left"] = None
        return _lave_argmax_at_focus(logits_with_noise, focus_pos, lm)

    seq_len = x.shape[1]
    if focus_pos >= seq_len or int(x[0, focus_pos].item()) != mask_id:
        return _lave_argmax_at_focus(logits_with_noise, focus_pos, lm)

    row_focus = logits_with_noise[0, focus_pos]
    suffix_single = _suffix_after_focus(x, focus_pos, seq_len, mask_id)

    if bidi_segmented_gap:
        segments, mask_positions, pos_after = _build_gap_segments_lave(
            x, focus_pos, seq_len, mask_id, logits_with_noise, lm, MAX_BIDI_GAP
        )
        if segments and mask_positions:
            suffix_seg = _suffix_after_gap(x, pos_after, seq_len, mask_id)
            res = segmented_bidirectional_dingo(
                csr, q_left, segments, suffix_seg, live_states
            )
            if res.success and res.tokens and len(res.tokens) == len(mask_positions):
                tid = int(res.tokens[0])
                return _clamp_csr_pick_to_lave(tid, lm, row_focus)

    # Default: one mask column + suffix to next mask (aligned with LAVE step semantics).
    masked_probs, _ = _lave_masked_probs_row(row_focus, lm)
    if len(masked_probs) < vs:
        masked_probs.extend([0.0] * (vs - len(masked_probs)))

    res1 = bidirectional_gap_dingo(
        csr, q_left, [masked_probs], suffix_single, live_states
    )
    if res1.success and res1.tokens:
        tid = int(res1.tokens[0])
        return _clamp_csr_pick_to_lave(tid, lm, row_focus)

    return _lave_argmax_at_focus(logits_with_noise, focus_pos, lm)


def _replay_checker_upto_pos(
    checker: Checker,
    prompt_ids: torch.Tensor,
    input_len: int,
    seq: List[int],
    pos: int,
) -> Optional[Checker]:
    """
    Fresh Checker with matcher advanced to the state *before* filling absolute index ``pos``,
    following the hypothesis ``seq`` (used for grammar-masked beam expansion in validate_ggbs).
    """
    c = checker.clone()
    c.reset()
    suf = prompt_ids[0][input_len:].tolist()
    if suf:
        cnt = c.matcher.try_consume_tokens(suf)
        if cnt != len(suf):
            c.matcher.rollback(cnt)
            return None
        c.tokens.extend(suf)
    plen = int(prompt_ids.shape[1])
    if pos <= plen:
        return c
    for j in range(plen, pos):
        t = seq[j]
        if t == mask_id:
            return None
        # Skip tokens outside llguidance's vocab (they would stop the matcher).
        if t >= c.tokenizer.vocab_size:
            return None
        cnt = c.matcher.try_consume_tokens([t])
        if cnt != 1:
            if cnt > 0:
                c.matcher.rollback(cnt)
            return None
        c.tokens.append(t)
    return c


def _ggbs_state_before_mask(
    checker: Checker,
    prompt_ids: torch.Tensor,
    input_len: int,
    seq: List[int],
    pos: int,
    c_after: Optional[Checker],
    last_pos: int,
) -> Optional[Checker]:
    """
    Matcher state *before* consuming ``seq[pos]`` at a mask column.

    If ``c_after`` is the state after consuming ``seq[last_pos]`` and
    ``last_pos == pos - 1``, reuse it via ``deepcopy`` (no O(pos) replay).
    Otherwise fall back to ``_replay_checker_upto_pos``.
    """
    if c_after is None or last_pos < 0:
        return _replay_checker_upto_pos(checker, prompt_ids, input_len, seq, pos)
    if last_pos == pos - 1:
        try:
            return copy.deepcopy(c_after)
        except Exception:
            return _replay_checker_upto_pos(checker, prompt_ids, input_len, seq, pos)
    try:
        c = copy.deepcopy(c_after)
        for j in range(last_pos + 1, pos):
            t = seq[j]
            if t == mask_id:
                return None
            if t >= c.tokenizer.vocab_size:
                return None
            cnt = c.matcher.try_consume_tokens([t])
            if cnt != 1:
                if cnt > 0:
                    c.matcher.rollback(cnt)
                return None
            c.tokens.append(t)
        return c
    except Exception:
        return _replay_checker_upto_pos(checker, prompt_ids, input_len, seq, pos)


def _ggbs_state_after_fixed_pos(
    checker: Checker,
    prompt_ids: torch.Tensor,
    input_len: int,
    seq: List[int],
    pos: int,
    c_after: Optional[Checker],
    last_pos: int,
) -> Optional[Checker]:
    """Matcher state after consuming fixed token ``seq[pos]``."""
    if c_after is None or last_pos < 0:
        return _replay_checker_upto_pos(checker, prompt_ids, input_len, seq, pos + 1)
    if last_pos == pos - 1:
        try:
            c = copy.deepcopy(c_after)
        except Exception:
            return _replay_checker_upto_pos(checker, prompt_ids, input_len, seq, pos + 1)
        t = seq[pos]
        if t == mask_id or t >= c.tokenizer.vocab_size:
            return None
        cnt = c.matcher.try_consume_tokens([t])
        if cnt != 1:
            if cnt > 0:
                c.matcher.rollback(cnt)
            return None
        c.tokens.append(t)
        return c
    try:
        c = copy.deepcopy(c_after)
        for j in range(last_pos + 1, pos + 1):
            t = seq[j]
            if t == mask_id or t >= c.tokenizer.vocab_size:
                return None
            cnt = c.matcher.try_consume_tokens([t])
            if cnt != 1:
                if cnt > 0:
                    c.matcher.rollback(cnt)
                return None
            c.tokens.append(t)
        return c
    except Exception:
        return _replay_checker_upto_pos(checker, prompt_ids, input_len, seq, pos + 1)


def _ggbs_state_after_mask_choice(
    c_hyp: Checker,
    tid: int,
    checker: Checker,
    prompt_ids: torch.Tensor,
    input_len: int,
    seq: List[int],
    pos: int,
) -> Optional[Checker]:
    """State after placing ``tid`` at mask ``pos`` (after consuming ``seq[pos]``)."""
    try:
        c = copy.deepcopy(c_hyp)
    except Exception:
        new_seq = seq.copy()
        new_seq[pos] = tid
        return _replay_checker_upto_pos(checker, prompt_ids, input_len, new_seq, pos + 1)
    if tid >= c.tokenizer.vocab_size:
        return None
    cnt = c.matcher.try_consume_tokens([tid])
    if cnt != 1:
        if cnt > 0:
            c.matcher.rollback(cnt)
        return None
    c.tokens.append(tid)
    return c


def validate_ggbs(
    checker: Checker,
    all_token_ids: List[int],
    p: torch.Tensor,
    index_to_consume: int,
    last_token_index: int,
    min_eos_eot_index: int,
    trace: bool = False,
    top_k_per_mask: int = 10,
    top_n_beam: int = 30,
    random_n_beam: int = 20,
    prompt_ids: Optional[torch.Tensor] = None,
    input_len: int = 0,
):
    """
    Same contract as ``validate``, but at mask positions:
    (1) candidates are filtered by Approach A (top-``k`` logits × single-token
        ``try_consume`` + ``rollback`` probes; no ``compute_mask``) using a matcher
        at the beam prefix (see carried checker state below);
    (2) no ``random.sample`` on the remainder — deterministic top-scoring remainder beams are kept.

    Beams carry ``(seq, score, c_after, last_pos)``: ``c_after`` is matcher state
    after consuming ``seq[last_pos]``. When ``last_pos == pos - 1``, the state
    before mask ``pos`` is obtained via ``copy.deepcopy`` (O(1) vs O(pos) replay).
    Child beams use ``deepcopy`` + ``consume_tokens``; if deepcopy fails, falls back
    to ``_replay_checker_upto_pos``.
    """
    import heapq

    global cache_seq

    if prompt_ids is None:
        raise ValueError("validate_ggbs requires prompt_ids and input_len")

    if top_n_beam == 1 and random_n_beam == 0:
        beams = [(all_token_ids.copy(), 0.0)]
        for pos in range(index_to_consume, last_token_index + 1):
            if all_token_ids[pos] == mask_id:
                new_beams: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    p[0, pos][mask_id] = 0
                    p[0, pos][eos_id] = 0
                    p[0, pos][eot_id] = 0
                    top_probs, top_idxs = torch.topk(p[0, pos], 1)
                    for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
                        new_seq = seq.copy()
                        new_seq[pos] = tid
                        new_score = score + prob
                        new_beams.append((new_seq, new_score))
                top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
                beams = top_beams
            else:
                for seq, score in beams:
                    if checker.validate_tokens(seq[index_to_consume : pos + 1]) == len(seq[index_to_consume : pos + 1]):
                        continue
                    else:
                        beams.remove((seq, score))
                if len(beams) == 0:
                    return False
    else:
        # (seq, score, c_after, last_pos): carry matcher state after seq[last_pos] to avoid O(pos) replay.
        beams = [(all_token_ids.copy(), 0.0, None, index_to_consume - 1)]
        vs = p.shape[-1]
        for pos in range(index_to_consume, last_token_index + 1):
            if all_token_ids[pos] == mask_id:
                new_beams = []
                for seq, score, c_after, last_pos in beams:
                    c_hyp = _ggbs_state_before_mask(
                        checker, prompt_ids, input_len, seq, pos, c_after, last_pos
                    )
                    if c_hyp is None:
                        continue
                    k_scan = min(max(top_k_per_mask * 6, top_n_beam * 2), vs)
                    top_probs, top_idxs = torch.topk(p[0, pos], k_scan)
                    ge_gt = _lave_ge_gt(c_hyp)
                    n_added = 0
                    for prob, tid in zip(
                        top_probs.detach().cpu().tolist(),
                        top_idxs.detach().cpu().tolist(),
                    ):
                        tid = int(tid)
                        if not _lave_tid_allowed_after_symmetry(c_hyp, tid, ge_gt):
                            continue
                        new_seq = seq.copy()
                        new_seq[pos] = tid
                        new_score = score + prob
                        c_new = _ggbs_state_after_mask_choice(
                            c_hyp, tid, checker, prompt_ids, input_len, seq, pos
                        )
                        if c_new is None:
                            continue
                        new_beams.append((new_seq, new_score, c_new, pos))
                        n_added += 1
                        if n_added >= top_k_per_mask * 3:
                            break
                    if n_added == 0:
                        row = p[0, pos].detach().cpu()
                        best_tid = None
                        best_p = -1.0
                        k_fb = min(vs, 8192)
                        _, idx_fb = torch.topk(row, k=k_fb)
                        for tid in idx_fb.tolist():
                            tid = int(tid)
                            if not _lave_tid_allowed_after_symmetry(c_hyp, tid, ge_gt):
                                continue
                            pv = float(row[tid].item())
                            if pv > best_p:
                                best_p = pv
                                best_tid = tid
                        if best_tid is None:
                            for tid in range(vs):
                                if not _lave_tid_allowed_after_symmetry(c_hyp, tid, ge_gt):
                                    continue
                                pv = float(row[tid].item())
                                if pv > best_p:
                                    best_p = pv
                                    best_tid = tid
                        if best_tid is not None:
                            new_seq = seq.copy()
                            new_seq[pos] = best_tid
                            c_new = _ggbs_state_after_mask_choice(
                                c_hyp, best_tid, checker, prompt_ids, input_len, seq, pos
                            )
                            if c_new is not None:
                                new_beams.append((new_seq, score + best_p, c_new, pos))

                if not new_beams:
                    return False
                sorted_nb = sorted(new_beams, key=lambda x: -x[1])
                top_beams = sorted_nb[:top_n_beam]
                remaining = sorted_nb[top_n_beam:]
                extra = remaining[: min(random_n_beam, len(remaining))]
                beams = top_beams + extra
            else:
                # Fixed column: advance carried matcher only (no global checker.validate_tokens
                # over seq[0:pos+1], which re-walks O(pos) per beam).
                new_beams = []
                for seq, score, c_after, last_pos in beams:
                    c_new = _ggbs_state_after_fixed_pos(
                        checker, prompt_ids, input_len, seq, pos, c_after, last_pos
                    )
                    if c_new is None:
                        continue
                    new_beams.append((seq, score, c_new, pos))
                beams = new_beams
                if len(beams) == 0:
                    return False

    assert len(beams) > 0, "No valid beams after processing tokens."
    largest_seq = max(beams, key=lambda x: x[1])[0]
    if min_eos_eot_index == -1:
        if trace:
            print(f"Pre-check passed. No EOS/EOT.")
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: last_token_index+1])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: last_token_index+1])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.validate_tokens(
        largest_seq[index_to_consume : last_token_index + 1]
    ), f"Should be valid sequence, {checker.dbg_tokens(largest_seq[index_to_consume:last_token_index + 1])}, {last_token_index - index_to_consume + 1}"
    checker.consume_tokens(largest_seq[index_to_consume : last_token_index + 1])
    assert checker.is_error() == False, "Should not be error here."

    for pos in range(last_token_index + 1, len(largest_seq)):
        largest_seq[pos] = eos_id

    for pos in range(last_token_index + 1, min_eos_eot_index):
        ge = _grammar_allows_token_lave(checker, eos_id)
        gt = _grammar_allows_token_lave(checker, eot_id)
        if ge or gt:
            checker.rollback(pos - index_to_consume)
            if trace:
                print(f"Pre-check passed. EOS/EOT found)).")
            if trace:
                print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: pos])}")
                print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: pos])}")
            cache_seq = largest_seq.copy()
            return True

        row = p[0, pos]
        vs_row = int(row.shape[0])
        k_pick = min(512, vs_row)
        _, idxs = torch.topk(row, k=k_pick)
        pick_tid = None
        pick_prob = -1.0
        ge_gt_eos = _lave_ge_gt(checker)
        for tid in idxs.tolist():
            tid = int(tid)
            if not _lave_tid_allowed_after_symmetry(checker, tid, ge_gt_eos):
                continue
            pr = float(row[tid].item())
            if pr > pick_prob:
                pick_prob = pr
                pick_tid = tid
        if pick_tid is None:
            for tid in range(vs_row):
                if not _lave_tid_allowed_after_symmetry(checker, tid, ge_gt_eos):
                    continue
                pr = float(row[tid].item())
                if pr > pick_prob:
                    pick_prob = pr
                    pick_tid = tid
        prob, tid = pick_prob, pick_tid
        assert prob > 1e-9, "Should not reach here."
        checker.consume_tokens([tid])
        largest_seq[pos] = tid
    if trace:
        print(f"is_stoped: {checker.is_stoped()}")
    if checker.is_accepting():
        if trace:
            print(f"Pre-check passed. Accepting state reached.")
        assert checker.rollback(min_eos_eot_index - index_to_consume)
        if trace:
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: min_eos_eot_index])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: min_eos_eot_index])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.rollback(min_eos_eot_index - index_to_consume), "Should rollback"
    return False


def validate(
    checker: Checker,
    all_token_ids: List[int],
    p: torch.Tensor,
    index_to_consume: int,
    last_token_index: int,
    min_eos_eot_index: int,          
    trace: bool = False,
    top_k_per_mask: int = 10,
    top_n_beam: int = 30,
    random_n_beam: int = 20,
):
    import heapq
    import random
    global cache_seq
    
    beams = [(all_token_ids.copy(), 0.0)]
    # temp_consume_index = index_to_consume
    if top_n_beam == 1 and random_n_beam == 0:
        for pos in range(index_to_consume, last_token_index + 1):
            if all_token_ids[pos] == mask_id:
                new_beams: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    p[0, pos][mask_id] = 0
                    p[0, pos][eos_id] = 0
                    p[0, pos][eot_id] = 0
                    top_probs, top_idxs = torch.topk(p[0, pos], 1)
                    for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
                        new_seq = seq.copy()
                        new_seq[pos] = tid
                        new_score = score + prob
                        new_beams.append((new_seq, new_score))
                top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
                beams = top_beams
            else:
                for seq, score in beams:
                    if checker.validate_tokens(seq[index_to_consume:pos+1]) == len(seq[index_to_consume:pos+1]):
                        continue
                    else:
                        beams.remove((seq, score))
                if len(beams) == 0:
                    return False
    else:
        for pos in range(index_to_consume, last_token_index + 1):
            if all_token_ids[pos] == mask_id:
                new_beams: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    top_probs, top_idxs = torch.topk(p[0, pos], top_k_per_mask)
                    for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
                        new_seq = seq.copy()
                        new_seq[pos] = tid
                        new_score = score + prob
                        new_beams.append((new_seq, new_score))
                top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
                remaining = [b for b in new_beams if b not in top_beams]
                random_beams = random.sample(remaining, min(random_n_beam, len(remaining)))
                beams = top_beams + random_beams
            else:
                beams = [
                    (seq, score)
                    for seq, score in beams
                    if checker.validate_tokens(seq[index_to_consume:pos+1])
                ]
                if len(beams) == 0:
                    return False
        
    assert len(beams) > 0, "No valid beams after processing tokens."
    largest_seq = max(beams, key=lambda x: x[1])[0]
    if min_eos_eot_index == -1:
        if trace:
            print(f"Pre-check passed. No EOS/EOT.")
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: last_token_index+1])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: last_token_index+1])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.validate_tokens(largest_seq[index_to_consume:last_token_index + 1]), f"Should be valid sequence, {checker.dbg_tokens(largest_seq[index_to_consume:last_token_index + 1])}, {last_token_index - index_to_consume + 1}"
    checker.consume_tokens(largest_seq[index_to_consume:last_token_index + 1])
    assert checker.is_error() == False, "Should not be error here."

    for pos in range(last_token_index + 1, len(largest_seq)):
        largest_seq[pos] = eos_id

    for pos in range(last_token_index + 1, min_eos_eot_index):
        ge = _grammar_allows_token_lave(checker, eos_id)
        gt = _grammar_allows_token_lave(checker, eot_id)
        if ge or gt:
            checker.rollback(pos - index_to_consume)
            if trace:
                print(f"Pre-check passed. EOS/EOT found)).")
            if trace:
                print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: pos])}")
                print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: pos])}")
            cache_seq = largest_seq.copy()
            return True

        row = p[0, pos]
        vs_row = int(row.shape[0])
        k_pick = min(512, vs_row)
        _, idxs = torch.topk(row, k=k_pick)
        pick_tid = None
        pick_prob = -1.0
        ge_gt_eos = _lave_ge_gt(checker)
        for tid in idxs.tolist():
            tid = int(tid)
            if not _lave_tid_allowed_after_symmetry(checker, tid, ge_gt_eos):
                continue
            pr = float(row[tid].item())
            if pr > pick_prob:
                pick_prob = pr
                pick_tid = tid
        if pick_tid is None:
            for tid in range(vs_row):
                if not _lave_tid_allowed_after_symmetry(checker, tid, ge_gt_eos):
                    continue
                pr = float(row[tid].item())
                if pr > pick_prob:
                    pick_prob = pr
                    pick_tid = tid
        prob, tid = pick_prob, pick_tid
        assert prob > 1e-9, "Should not reach here."
        checker.consume_tokens([tid])
        largest_seq[pos] = tid
    if trace:
        print(f"is_stoped: {checker.is_stoped()}")
    if checker.is_accepting():
        if trace:
            print(f"Pre-check passed. Accepting state reached.")
        assert checker.rollback(min_eos_eot_index - index_to_consume)
        if trace:
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: min_eos_eot_index])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: min_eos_eot_index])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.rollback(min_eos_eot_index - index_to_consume), "Should rollback"
    return False

    
def check(
    checker: Checker,
    
    all_token_ids: List[int],
    prompt_ids,
    p: torch.Tensor,

    index_to_consume: int,
    block_num: int,
    block_length: int,
    top_k_per_mask: int = 5,
    top_n_beam: int = 30,
    random_n_beam: int = 20,
    trace: bool = False,
    input_len: int = 0,
    use_ggbs_validate: bool = False,
):
    last_token_index = get_end_index(all_token_ids, block_num, block_length, prompt_ids)
    min_eos_eot_index, _ = min_eos_or_eot_index(all_token_ids, block_num, block_length, prompt_ids)

    if min_eos_eot_index != -1 and min_eos_eot_index < last_token_index:
        return False
    
    if use_ggbs_validate:
        accept = validate_ggbs(
            checker=checker,
            all_token_ids=all_token_ids,
            p=p,
            index_to_consume=index_to_consume,
            last_token_index=last_token_index,
            min_eos_eot_index=min_eos_eot_index,
            trace=trace,
            top_k_per_mask=top_k_per_mask,
            top_n_beam=top_n_beam,
            random_n_beam=random_n_beam,
            prompt_ids=prompt_ids,
            input_len=input_len,
        )
    else:
        accept = validate(
            checker=checker,
            all_token_ids=all_token_ids,
            p=p,
            index_to_consume=index_to_consume,
            last_token_index=last_token_index,
            min_eos_eot_index=min_eos_eot_index,
            trace=trace,
            top_k_per_mask=top_k_per_mask,
            top_n_beam=top_n_beam,
            random_n_beam=random_n_beam,
        )
    if accept:
        return True
    else:
        return False


@torch.no_grad()
def generate(
    model, 
    tokenizer,
    prompt_ids,
    input_len: int,
    grammar: str,
    
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str = "low_confidence",
    trace: bool = False,
    change_logits: bool = False,

    top_k_per_mask: int = 5,
    top_n_beam: int = 3,
    random_n_beam: int = 3,
    max_retry_num_total: int = 5,
    # Optional: BiDi frontier on LAVE-masked logits (same NFE / check() as LAVE).
    use_bidi_frontier: bool = False,
    bidi_csr: Any = None,
    bidi_live_states: Optional[Set[int]] = None,
    bidi_json_start_state: int = 0,
    # If True, run segmented BiDi over a multi-mask gap (can disagree with LAVE check()).
    bidi_segmented_gap: bool = False,
    # If True, replace validate()'s random beam sampling with grammar-masked candidates + deterministic remainder.
    use_ggbs_validate: bool = False,
):    
    checker = Checker(grammar=grammar, model_name="LLaDA")
    if checker.consume_tokens(prompt_ids[0][input_len:].tolist()) == False:
        raise ValueError("Prompt does not conform to grammar.")
    index_to_consume = prompt_ids.shape[1]

    # Prefix -> DFA state cache for BiDi (JSON CSR); LAVE loop unchanged otherwise.
    bidi_ctx: dict = {}

    start_time = time.monotonic()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks  

    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt_ids.shape[1]] = prompt_ids.clone()  

    all_token_ids = x[0].detach().cpu().tolist()
    global cache_seq
    cache_seq = all_token_ids.copy()

    total_retry_num = 0
    complete = False
    for block_num in range(num_blocks):
        if complete:
            break
        block_mask_index = (
            x[
                :,
                prompt_ids.shape[1] + block_num * block_length : prompt_ids.shape[1]
                + (block_num + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for step_num in range(steps):
            # If the generation is complete, we can stop early.
            if complete:
                break
            # if start_ar:
                # break
            start_ar = False

            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature)

            # greedy constrained sampling for multi-step
            token_num_to_decode = num_transfer_tokens[0, step_num]
            token_num = 0
            while token_num < token_num_to_decode:
                # If the generation is complete, we can stop early.
                if complete:
                    # print("test1")
                    break
                if start_ar:
                    break

                if trace:
                    print(f"\033[38;2;165;42;42m[Block {block_num} / {num_blocks}, step {step_num} / {steps}, token {token_num} / {num_transfer_tokens[0, step_num]}]\033[0m")
                
                mask_indexs = x == mask_id

                token_found = False
                one_token_retry_num = 0

                while not token_found:            
                    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                    
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )  # b, l
                    else:
                        raise NotImplementedError(remasking)

                    x0_p[
                        :, prompt_ids.shape[1] + (block_num + 1) * block_length :
                    ] = -np.inf
                    x0 = torch.where(mask_indexs, x0, x)
                    confidences = torch.where(mask_indexs, x0_p, -np.inf)
                    # TODO 
                    if False:
                        confs, select_indexs = torch.topk(
                                                        input=confidences[0], 
                                                        k=token_num_to_decode - token_num
                                                    )
                        assert select_indexs.shape[0] > 0, "No tokens to transfer"
                        pos_min = torch.argmin(select_indexs)
                        select_index = select_indexs[pos_min]
                        conf = confs[pos_min]
                    conf, select_index = torch.topk(
                                                    input=confidences[0], 
                                                    k=1
                                                )

                    index_of_new_token = select_index.item()
                    gen_start = prompt_ids.shape[1]
                    if (
                        use_bidi_frontier
                        and bidi_csr is not None
                        and bidi_live_states is not None
                    ):
                        picked = bidi_frontier_pick(
                            logits_with_noise,
                            x,
                            index_of_new_token,
                            checker,
                            bidi_csr,
                            bidi_live_states,
                            mask_id,
                            gen_start,
                            bidi_json_start_state,
                            bidi_ctx,
                            bidi_segmented_gap=bidi_segmented_gap,
                        )
                        new_token_vocab_index = torch.tensor(
                            picked, device=x0.device, dtype=x0.dtype
                        )
                        x0[0][index_of_new_token] = new_token_vocab_index
                    else:
                        new_token_vocab_index = x0[0][index_of_new_token]
                    if (
                        use_bidi_frontier
                        and bidi_csr is not None
                        and bidi_live_states is not None
                    ):
                        conf = p[0, index_of_new_token, new_token_vocab_index].reshape(1)
                    else:
                        assert (
                            logits_with_noise[0][index_of_new_token][
                                new_token_vocab_index
                            ]
                            != -np.inf
                        ), "No valid token found"

                    all_token_ids[index_of_new_token] = new_token_vocab_index.item()
                    # if has_constrain:
                    if trace:
                        print(f"Checking token at position {index_of_new_token}, token id: {new_token_vocab_index.item()}")
                        print(f"cache_seq[index_of_new_token]: {cache_seq[index_of_new_token]}")
                        print(f"new_token_vocab_index.item(): {new_token_vocab_index.item()}")

                    if cache_seq[index_of_new_token] == new_token_vocab_index.item():
                        is_accept = True
                        if trace:
                            print("cache_seq hit, accept directly.")
                    else:
                        is_accept = check(
                            checker=checker,
                            all_token_ids=all_token_ids,
                            prompt_ids=prompt_ids,
                            p=p,
                            index_to_consume=index_to_consume,
                            block_num=block_num,
                            block_length=block_length,
                            top_k_per_mask=top_k_per_mask,
                            top_n_beam=top_n_beam,
                            random_n_beam=random_n_beam,
                            trace=trace,
                            input_len=input_len,
                            use_ggbs_validate=use_ggbs_validate,
                        )

                    if trace:                
                        new_word = tokenizer.decode(new_token_vocab_index)
                        if new_word is EOS:
                            new_word = "<EOS>"
                        if is_accept:
                            print(
                                f"+++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
                            )
                        else:
                            print(
                                f"--- Reject {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
                            )
                    
                    if not is_accept:
                        logits_with_noise[0][index_of_new_token][
                            new_token_vocab_index
                        ] = -np.inf
                        if change_logits:
                            logits[0][index_of_new_token][
                                new_token_vocab_index
                            ] = -np.inf

                        # generated_words[index_of_new_token] = None
                        all_token_ids[index_of_new_token] = mask_id

                        one_token_retry_num += 1
                        total_retry_num += 1
                        # TODO
                        if one_token_retry_num >= max_retry_num_total:
                            if trace:
                                print("Too many retries for one token, start autoregressive generation.")
                            start_ar = True
                            # print("Rollback to index:", index_to_consume - (block_num * block_length + prompt_ids.shape[1]))
                            # r = checker.rollback(index_to_consume - (block_num * block_length + prompt_ids.shape[1]))
                            if trace:
                                print(f"index_to_consume={index_to_consume},{block_num * block_length + prompt_ids.shape[1]}")
                            # checker.reset()
                            break
                    else:
                        token_found = True
                        token_num += 1
                        one_token_retry_num = 0
                        transfer_index = torch.zeros_like(
                            x0, dtype=torch.bool, device=x0.device
                        )
                        transfer_index[0, select_index] = True
                        if new_token_vocab_index in (eos_id, eot_id):
                            transfer_index[0, select_index:] = True
                            val = x0[0, select_index].clone()
                            x0[0, select_index:] = val
                            # all_token_ids[select_index:] = [val.item()] * (len(all_token_ids) - select_index)
                            for idx in range(select_index, len(all_token_ids)):
                                all_token_ids[idx] = val.item()
                        x[transfer_index] = x0[transfer_index]
                        # if EOS in generated_words:
                        min_eos_eot_index, exist_mask = min_eos_or_eot_index(all_token_ids, block_num, block_length, prompt_ids)
                        if min_eos_eot_index != -1 and not exist_mask:
                            complete = True
                            break 

                        # if has_constrain:
                        for idx in range(index_to_consume, prompt_ids.shape[1] + (block_num + 1) * block_length):
                            if all_token_ids[idx] in (mask_id, eos_id, eot_id):
                                if idx > index_to_consume:
                                    if trace:
                                        print(f"Consume tokens: {checker.dbg_tokens(all_token_ids[index_to_consume:idx])}, index_to_consume={index_to_consume} -> {idx}")
                                    tokens_to_consume = all_token_ids[index_to_consume:idx]
                                    assert checker.consume_tokens(tokens_to_consume)
                                    index_to_consume = idx
                                    if checker.is_stoped():
                                        x[0, index_to_consume] = eos_id 
                                        complete = True
                                        if trace:
                                            print("Grammar matched complete sequence. Inserting EOS.")
                                        break
                                break

            if not start_ar:
                continue
            if trace:
                print("Start autoregressive generation.")
            
            # init
            a = prompt_ids.shape[1] + block_num * block_length
            b = prompt_ids.shape[1] + (block_num + 1) * block_length
            for i in range(b, len(cache_seq)):
                cache_seq[i] = mask_id
            index_cache = b
            for i in range(a, b):
                if cache_seq[i] in (eos_id, eot_id, mask_id):
                    if trace:
                        print(f"All cache_seq from {i} to {b} are mask_id, set index_cache to {i}")
                    index_cache = i
                    break
            free_num = 0
            for i in range(a, index_cache):
                if x[0, i] == mask_id:
                    free_num += 1

            x[:, a:b] = mask_id
            if b != x.shape[1]:
                x[0, b:b+2] = mask_id
                all_token_ids[b] = mask_id
                all_token_ids[b+1] = mask_id
            for i in range(a, b):
                all_token_ids[i] = mask_id

            
            for i in range(a, index_cache):
                assert cache_seq[i] != mask_id, "Cache seq should have value here."
                assert cache_seq[i] != eos_id and cache_seq[i] != eot_id, "Should not be EOS/EOT in cache seq here."
                x[0, i] = cache_seq[i]
                all_token_ids[i] = cache_seq[i]
            tokens_to_consume = all_token_ids[index_to_consume:index_cache]

            assert checker.consume_tokens(tokens_to_consume), f"Should be valid sequence in AR generation, {checker.dbg_tokens(tokens_to_consume)}"

            if trace:
                print(f"index_to_consume: {index_to_consume} -> index_cache: {index_cache}")
            index_to_consume = index_cache

            if checker.is_stoped():
                x[0, index_to_consume] = eos_id
                return x, total_retry_num, start_time

            min_decode = num_transfer_tokens[0, step_num] - token_num
            
            if trace:
                print(f"min_decode before adjustment: {min_decode}, free_num: {free_num}, num_transfer_tokens: {num_transfer_tokens[0, step_num]}, token_num: {token_num}")
            if min_decode > free_num:
                min_decode = min_decode - free_num
                logits = model(x).logits
                logits_with_noise = add_gumbel_noise(logits, temperature)
                for i in range(min_decode):
                    index_of_new_token = index_cache + i
                    one_logits = logits_with_noise[0, index_of_new_token]
                    logits_mask = compute_logits_mask(checker)
                    one_logits[logits_mask == 0] = -float('inf')
                    token_id = torch.argmax(one_logits)
                    assert logits_mask[token_id] == 1, "Selected token should be valid."
                    if trace:
                        conf = F.softmax(one_logits, dim=-1)[token_id]
                        new_word = tokenizer.decode(token_id.item())
                        if new_word is EOS:
                            new_word = "<EOS>"
                        print(
                            f"index_cache:{index_cache} +++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({token_id.item()}), confidence={conf.item():.6f}"
                        )
                    if token_id.item() in (eos_id, eot_id):
                        x[0, index_of_new_token] = token_id
                        return x, total_retry_num, start_time
                    else:
                        x[0, index_of_new_token] = token_id
                        all_token_ids[index_of_new_token] = token_id.item()
                        cache_seq[index_of_new_token] = token_id.item()
                        checker.consume_tokens([token_id.item()])
                        if trace:
                            print(f"Consume token: {checker.dbg_tokens([token_id.item()])}, index_to_consume={index_to_consume} -> {index_of_new_token + 1}")
                        index_to_consume = index_of_new_token + 1
                        if checker.is_stoped():
                            x[0, index_of_new_token + 1] = eos_id
                            return x, total_retry_num, start_time
            else:
                total_decode = 0
                for i in range(step_num + 1, steps):
                    total_decode += num_transfer_tokens[0, i]
                sub_num = free_num - min_decode
                if trace:
                    print(f"total_decode: {total_decode}, sub_num: {sub_num}, steps: {steps}, step_num: {step_num}")
                if total_decode <= sub_num:
                    for i in range(step_num + 1, steps):
                        num_transfer_tokens[0, i] = 0
                else:
                    total_decode_new = total_decode - sub_num
                    number = steps - step_num - 1
                    for i in range(step_num + 1, steps):
                        num_transfer_tokens[0, i] = total_decode_new // number
                    remainder = total_decode_new % number
                    for i in range(step_num + 1, step_num + 1 + remainder):
                        num_transfer_tokens[0, i] += 1
            
    return x, total_retry_num, start_time