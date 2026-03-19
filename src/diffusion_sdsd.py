"""
Diffusion loop with SDSD constraint at frontier.

Same structure as dgrammar.generate, but at the frontier (and violator retry)
uses our DINGO/Herding instead of argmax. Enables SDSD constrained decoding
in the diffusion paradigm.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens, extend_prefix

from csr_dfa import build_csr_from_transition_dict
from sparse_dingo import sparse_dingo_dp
from herding import herding_decode
from baseline_dingo import baseline_dingo_dp


def _make_frontier_csr(valid_tokens: list[int], vocab_size: int):
    """Build minimal CSR: state 0 -> state 1 for each valid token."""
    if not valid_tokens:
        return None
    transitions = {(0, t): 1 for t in valid_tokens}
    return build_csr_from_transition_dict(transitions, num_states=2, vocab_size=vocab_size)


def _frontier_picker_base(
    logits_1d: torch.Tensor,
    checker: TokenChecker,
    vocab_size: int,
    decode_fn,
) -> int:
    """Generic frontier picker: get valid tokens, run decode_fn, return best token."""
    bias = checker.compute_mask(vocab_size=vocab_size)
    if hasattr(logits_1d, "cpu"):
        logits_1d = logits_1d.cpu().float()
    # Keep mask/logits on the same device for in-place masking.
    bias = bias.to(logits_1d.device)
    valid_tokens = [t for t in range(min(vocab_size, len(bias))) if not bool(bias[t].item())]
    if not valid_tokens:
        return -1
    csr = _make_frontier_csr(valid_tokens, vocab_size)
    if csr is None:
        return -1
    # add_gumbel_noise can produce +/-inf (exp(logits)/noise). If fed directly to
    # softmax, this may create NaNs and collapse DINGO/Herding to fallback tokens.
    logits_1d = torch.nan_to_num(logits_1d, nan=-1e9, posinf=1e9, neginf=-1e9)
    # Critical: mask invalid tokens *before* softmax. Otherwise, one huge invalid
    # logit can push all valid probabilities to 0 (underflow), causing fallback.
    masked_logits = logits_1d.clone()
    if bias.shape[0] > masked_logits.shape[0]:
        bias = bias[: masked_logits.shape[0]]
    elif bias.shape[0] < masked_logits.shape[0]:
        pad = torch.ones(
            masked_logits.shape[0] - bias.shape[0],
            dtype=torch.bool,
            device=masked_logits.device,
        )
        bias = torch.cat([bias, pad])
    masked_logits[bias] = float("-inf")
    if torch.isneginf(masked_logits).all():
        return -1
    probs = F.softmax(masked_logits, dim=-1)
    if probs.dim() > 1:
        probs = probs[0]
    prob_list = probs.tolist()
    if len(prob_list) < vocab_size:
        prob_list.extend([0.0] * (vocab_size - len(prob_list)))
    result = decode_fn(csr, [prob_list], 0, {1}, block_length=1)
    if not result.success or not result.tokens:
        return valid_tokens[0] if valid_tokens else -1
    return result.tokens[0]


def make_frontier_picker(method: str):
    """Create frontier_picker for sdsd, ablation1, ablation2, ablation3, baseline, argmax."""

    def _argmax(logits_1d, checker, vocab_size):
        """Dgrammar-style: mask invalid, argmax. Use for debugging (verify loop works)."""
        bias = checker.compute_mask(vocab_size=vocab_size)
        if hasattr(logits_1d, "device"):
            logits = logits_1d.clone().float()
        else:
            logits = torch.tensor(logits_1d, dtype=torch.float32)
        bias = bias.to(logits.device)
        if bias.shape[0] > logits.shape[0]:
            bias = bias[: logits.shape[0]]
        elif bias.shape[0] < logits.shape[0]:
            pad = torch.ones(logits.shape[0] - bias.shape[0], dtype=torch.bool, device=logits.device)
            bias = torch.cat([bias, pad])
        logits = logits.clone()
        logits[bias] = float("-inf")
        return int(logits.argmax().item())

    def _sparse_dingo(logits_1d, checker, vocab_size):
        return _frontier_picker_base(logits_1d, checker, vocab_size, sparse_dingo_dp)

    def _herding(logits_1d, checker, vocab_size):
        return _frontier_picker_base(logits_1d, checker, vocab_size, herding_decode)

    def _baseline(logits_1d, checker, vocab_size):
        bias = checker.compute_mask(vocab_size=vocab_size)
        if hasattr(logits_1d, "cpu"):
            logits_1d = logits_1d.cpu().float()
        bias = bias.to(logits_1d.device)
        valid_tokens = [t for t in range(min(vocab_size, len(bias))) if not bool(bias[t].item())]
        if not valid_tokens:
            return -1

        def trans_fn(q, t):
            return 1 if (q == 0 and t in valid_tokens) else None

        logits_1d = torch.nan_to_num(logits_1d, nan=-1e9, posinf=1e9, neginf=-1e9)
        masked_logits = logits_1d.clone()
        if bias.shape[0] > masked_logits.shape[0]:
            bias = bias[: masked_logits.shape[0]]
        elif bias.shape[0] < masked_logits.shape[0]:
            pad = torch.ones(
                masked_logits.shape[0] - bias.shape[0],
                dtype=torch.bool,
                device=masked_logits.device,
            )
            bias = torch.cat([bias, pad])
        masked_logits[bias] = float("-inf")
        if torch.isneginf(masked_logits).all():
            return -1
        probs = F.softmax(masked_logits, dim=-1)
        if probs.dim() > 1:
            probs = probs[0]
        prob_list = probs.tolist()
        if len(prob_list) < vocab_size:
            prob_list.extend([0.0] * (vocab_size - len(prob_list)))
        result = baseline_dingo_dp(2, vocab_size, trans_fn, [prob_list], 0, {1}, block_length=1)
        if not result.success or not result.tokens:
            return valid_tokens[0] if valid_tokens else -1
        return result.tokens[0]

    if method in ("argmax", "dgrammar"):
        return _argmax
    if method in ("sdsd", "ablation3"):
        return _sparse_dingo
    if method == "ablation1":
        return _sparse_dingo
    if method == "ablation2":
        return _herding
    if method == "baseline":
        return _baseline
    return _sparse_dingo


def generate_diffusion_sdsd(
    model,
    prompt: torch.Tensor,
    tokenizer,
    checker: TokenChecker,
    prompt_len: int,
    frontier_picker: Callable[[torch.Tensor, TokenChecker, int], int],
    steps: int = 128,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.2,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_id: int = 126081,
    eot_id: int = 126348,
    max_batch_size: int = 8,
    max_resamples: int = 100,
):
    """
    Diffusion generation with SDSD constraint at frontier.

    frontier_picker(logits_1d, checker, vocab_size) -> token_id
    Uses our DINGO/Herding to pick among grammar-valid tokens.
    """
    start_time = time.monotonic()
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_violations = 0
    total_remasks = 0
    total_grammar_checks = 0
    resamples = []
    current_batch = 1
    gen_start = prompt.shape[1]
    consume_idx = gen_start

    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        if not checker.consume_tokens(prefix_tokens):
            pass  # warning only

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed_this_step = 0
            vocab_size = logits_with_noise.shape[-1]

            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                x0_p[:, block_end:] = -np.inf

                # SDSD frontier: use our picker instead of argmax
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    tok = frontier_picker(
                        logits_with_noise[0, consume_idx], checker, vocab_size
                    )
                    if tok < 0:
                        bias = checker.compute_mask(vocab_size=vocab_size)
                        logits_with_noise[0, consume_idx, bias] = -np.inf
                        tok = torch.argmax(logits_with_noise[0, consume_idx]).item()
                    x0[0, consume_idx] = tok

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break

                remaining = n_scheduled - tokens_placed_this_step
                batch_k = min(current_batch, remaining, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)

                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()
                    if logits_with_noise[0, pos, vocab_idx] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                tokens_placed_this_step += len(positions)

                total_grammar_checks += 1
                new_idx, violator = extend_prefix(checker, x, consume_idx, mask_id)

                if violator < 0:
                    consume_idx = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    total_violations += 1
                    consume_idx = new_idx

                    if checker.is_accepting():
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    bad_token = x[0, violator].item()
                    logits_with_noise[0, violator, bad_token] = -np.inf
                    x[0, violator] = mask_id
                    total_remasks += 1
                    tokens_placed_this_step -= 1
                    resamples.append((violator, time.monotonic() - start_time))

                    if len(resamples) >= max_resamples:
                        yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                        return

                    # SDSD violator retry: use our picker instead of argmax
                    found = False
                    while len(resamples) < max_resamples:
                        next_vocab = frontier_picker(
                            logits_with_noise[0, violator], checker, vocab_size
                        )
                        if next_vocab < 0 or logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break

                        total_grammar_checks += 1
                        c = checker.matcher.try_consume_tokens([next_vocab])
                        if c == 1:
                            x[0, violator] = next_vocab
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            found = True
                            further_idx, further_viol = extend_prefix(
                                checker, x, consume_idx, mask_id
                            )
                            consume_idx = further_idx
                            break

                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        total_remasks += 1
                        resamples.append((violator, time.monotonic() - start_time))

                    current_batch = 1

                if not complete and checker.is_accepting():
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

                if not complete:
                    gen_ids = x[0, gen_start:].tolist()
                    if eos_id in gen_ids or eot_id in gen_ids:
                        eos_pos = None
                        for j, tid in enumerate(gen_ids):
                            if tid in (eos_id, eot_id):
                                eos_pos = j
                                break
                        if eos_pos is not None and mask_id not in gen_ids[:eos_pos]:
                            for j in range(eos_pos, len(gen_ids)):
                                x[0, gen_start + j] = x[0, gen_start + eos_pos]
                            complete = True

            yield x, resamples, False, total_violations, total_remasks, total_grammar_checks

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next(
            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
        )
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks
