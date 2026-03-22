"""Run Dgrammar (llguidance) with async overlap: compute_mask runs in parallel with forward pass."""

import json
import time
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from dgrammar.checker import TokenChecker
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens


class TimingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.grammar_check_times = []
        self.token_select_times = []
        self.mask_compute_times = []
        self.mask_wait_times = []  # time waiting for async result after forward
        self.resample_count = 0
        self.tokens_unmasked = 0
        self.batch_sizes = []
        self.overlap_count = 0  # how many mask computes were overlapped

    def summary(self):
        fwd = self.forward_times
        gc = self.grammar_check_times
        ts = self.token_select_times
        mc = self.mask_compute_times
        mw = self.mask_wait_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "grammar_check_count": len(gc),
            "grammar_check_total_ms": sum(gc) * 1000,
            "grammar_check_mean_ms": (sum(gc) / len(gc) * 1000) if gc else 0,
            "grammar_check_median_ms": (sorted(gc)[len(gc) // 2] * 1000) if gc else 0,
            "grammar_check_p95_ms": (sorted(gc)[int(len(gc) * 0.95)] * 1000) if len(gc) >= 20 else 0,
            "mask_compute_count": len(mc),
            "mask_compute_total_ms": sum(mc) * 1000,
            "mask_compute_mean_ms": (sum(mc) / len(mc) * 1000) if mc else 0,
            "mask_wait_count": len(mw),
            "mask_wait_total_ms": sum(mw) * 1000,
            "mask_wait_mean_ms": (sum(mw) / len(mw) * 1000) if mw else 0,
            "overlap_count": self.overlap_count,
            "token_select_count": len(ts),
            "token_select_total_ms": sum(ts) * 1000,
            "token_select_mean_ms": (sum(ts) / len(ts) * 1000) if ts else 0,
            "resample_count": self.resample_count,
            "tokens_unmasked": self.tokens_unmasked,
            "avg_batch_size": (sum(self.batch_sizes) / len(self.batch_sizes)) if self.batch_sizes else 0,
        }


STATS = TimingStats()


def compute_mask_async(checker, vocab_size):
    """Run compute_mask in a background thread, return (result, compute_time)."""
    result = [None, 0.0]

    def _run():
        t0 = time.perf_counter()
        result[0] = checker.compute_mask(vocab_size=vocab_size)
        result[1] = time.perf_counter() - t0

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, result


def extend_prefix_timed(checker, x, consume_idx, mask_id):
    """extend_prefix with timing."""
    t0 = time.perf_counter()
    tokens_to_consume = []
    pos = consume_idx
    while pos < x.shape[1]:
        tid = x[0, pos].item()
        if tid == mask_id:
            break
        tokens_to_consume.append(tid)
        pos += 1

    if not tokens_to_consume:
        STATS.grammar_check_times.append(time.perf_counter() - t0)
        return consume_idx, -1

    count = checker.matcher.try_consume_tokens(tokens_to_consume)
    STATS.grammar_check_times.append(time.perf_counter() - t0)

    if count == len(tokens_to_consume):
        return consume_idx + count, -1
    else:
        return consume_idx + count, consume_idx + count


@torch.no_grad()
def autocomplete_greedy(model, x, checker, consume_idx, gen_start, mask_id, eos_id,
                        refresh_interval=8):
    """Grammar-guided greedy completion from consume_idx forward.

    Diffusion models give logits for all positions at once. We exploit this by
    doing one forward pass and completing multiple positions from the same logits.
    Logits are refreshed every `refresh_interval` steps to avoid staleness.

    Returns (x, autocomplete_steps, autocomplete_mask_ms, autocomplete_fwd_ms).
    """
    ac_steps = 0
    ac_fwd_ms = 0.0
    ac_mask_ms = 0.0
    seq_len = x.shape[1]
    steps_since_refresh = refresh_interval  # force initial forward

    while consume_idx < seq_len:
        if checker.is_accepting():
            for j in range(consume_idx, seq_len):
                x[0, j] = eos_id
            break

        # Refresh logits periodically
        if steps_since_refresh >= refresh_interval:
            t_fwd = time.perf_counter()
            logits = model(x).logits
            ac_fwd_ms += (time.perf_counter() - t_fwd) * 1000
            steps_since_refresh = 0

        # compute_mask at frontier
        t_mc = time.perf_counter()
        bias = checker.compute_mask(vocab_size=logits.shape[-1])
        ac_mask_ms += (time.perf_counter() - t_mc) * 1000

        logits[0, consume_idx, bias] = -np.inf
        best = torch.argmax(logits[0, consume_idx]).item()

        if logits[0, consume_idx, best] == -np.inf:
            break

        x[0, consume_idx] = best

        c = checker.matcher.try_consume_tokens([best])
        if c == 1:
            consume_idx += 1
            ac_steps += 1
            steps_since_refresh += 1
        else:
            break

        # Extend past already-placed non-mask tokens
        while consume_idx < seq_len:
            tid = x[0, consume_idx].item()
            if tid == mask_id:
                break
            c = checker.matcher.try_consume_tokens([tid])
            if c == 1:
                consume_idx += 1
                ac_steps += 1
                steps_since_refresh += 1
            else:
                x[0, consume_idx] = mask_id
                break

    return x, ac_steps, ac_mask_ms, ac_fwd_ms


@torch.no_grad()
def generate_async_timed(
    model, prompt, tokenizer, checker,
    prompt_len, steps=128, gen_length=256,
    block_length=32, temperature=0.0,
    remasking="low_confidence", mask_id=126336,
    eos_id=126081, eot_id=126348,
    max_batch_size=8, max_resamples=100,
):
    start_time = time.monotonic()

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

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
        checker.consume_tokens(prefix_tokens)

    # Pending async mask result from previous iteration
    pending_mask = None  # (thread, result_list, vocab_size)

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            # --- Check if we should precompute mask async BEFORE forward ---
            # Grammar state is known now. If frontier is a mask, start compute_mask.
            mask_index_pre = x == mask_id
            need_mask = (
                consume_idx < x.shape[1]
                and mask_index_pre[0, consume_idx]
                and pending_mask is None
            )
            if need_mask:
                vocab_size = 126464  # will be corrected after logits are available
                thread, result_holder = compute_mask_async(checker, vocab_size)
                pending_mask = (thread, result_holder)

            # --- Model forward (timed) ---
            t_fwd = time.perf_counter()
            logits = model(x).logits
            STATS.forward_times.append(time.perf_counter() - t_fwd)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                # --- Token selection (timed) ---
                t_sel = time.perf_counter()
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

                # Frontier masking: use async result if available
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    if pending_mask is not None:
                        # Wait for async result
                        t_wait = time.perf_counter()
                        thread, result_holder = pending_mask
                        thread.join()
                        wait_time = time.perf_counter() - t_wait
                        STATS.mask_wait_times.append(wait_time)
                        STATS.mask_compute_times.append(result_holder[1])
                        STATS.overlap_count += 1
                        bias = result_holder[0]
                        pending_mask = None
                        # Truncate/pad if vocab sizes differ
                        actual_vocab = logits_with_noise.shape[-1]
                        if bias.shape[0] > actual_vocab:
                            bias = bias[:actual_vocab]
                        elif bias.shape[0] < actual_vocab:
                            pad = torch.ones(actual_vocab - bias.shape[0], dtype=torch.bool)
                            bias = torch.cat([bias, pad])
                    else:
                        # Fallback: compute synchronously (e.g. after resampling changed state)
                        t_mask = time.perf_counter()
                        bias = checker.compute_mask(vocab_size=logits_with_noise.shape[-1])
                        STATS.mask_compute_times.append(time.perf_counter() - t_mask)

                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    x0[0, consume_idx] = torch.argmax(logits_with_noise[0, consume_idx])

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
                STATS.token_select_times.append(time.perf_counter() - t_sel)
                STATS.batch_sizes.append(batch_k)

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
                STATS.tokens_unmasked += len(positions)

                # --- Grammar check (timed) ---
                total_grammar_checks += 1
                new_idx, violator = extend_prefix_timed(checker, x, consume_idx, mask_id)

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
                    STATS.resample_count += 1
                    tokens_placed_this_step -= 1
                    STATS.tokens_unmasked -= 1
                    resamples.append((violator, time.monotonic() - start_time))

                    if len(resamples) >= max_resamples:
                        yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                        return

                    found = False
                    while len(resamples) < max_resamples:
                        next_vocab = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break
                        t_gc_retry = time.perf_counter()
                        total_grammar_checks += 1
                        c = checker.matcher.try_consume_tokens([next_vocab])
                        STATS.grammar_check_times.append(time.perf_counter() - t_gc_retry)
                        if c == 1:
                            x[0, violator] = next_vocab
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            STATS.tokens_unmasked += 1
                            found = True
                            further_idx, further_viol = extend_prefix_timed(
                                checker, x, consume_idx, mask_id
                            )
                            total_grammar_checks += 1
                            if further_viol < 0:
                                consume_idx = further_idx
                            else:
                                consume_idx = further_idx
                            break
                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        total_remasks += 1
                        STATS.resample_count += 1
                        resamples.append((violator, time.monotonic() - start_time))
                    current_batch = 1

                # Check completion
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
                        eos_pos = next((j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None)
                        if eos_pos is not None and mask_id not in gen_ids[:eos_pos]:
                            for j in range(eos_pos, len(gen_ids)):
                                x[0, gen_start + j] = x[0, gen_start + eos_pos]
                            complete = True

            yield x, resamples, False, total_violations, total_remasks, total_grammar_checks

    # Clean up any pending async
    if pending_mask is not None:
        thread, _ = pending_mask
        thread.join()
        pending_mask = None

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if eos_id in gen_ids or eot_id in gen_ids:
        eos_pos = next((j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None)
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]
    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    block_ar = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    tag = "v2_async_ac4_fullpar_timed" if not block_ar else "v2_async_ac4_timed"
    ds_safe = dataset_name.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    output_file = f"results/{tag}_{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]
    bl = 32 if block_ar else 256
    print(f"Dgrammar timed: {len(instances)} instances, seed={seed}, T={steps}, block_length={bl}")

    cached_checker = None

    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            print(f"  Skipping {instance.instance_id()}: no schema")
            continue

        try:
            if cached_checker is None or dataset.different_grammar_per_instance:
                cached_checker = TokenChecker(schema_str)
            checker = cached_checker.clone()
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        STATS.reset()
        torch.manual_seed(seed)
        start_time = time.monotonic()

        out = None
        resamples = []
        valid = False
        total_violations = 0
        total_remasks = 0
        total_grammar_checks = 0

        ac_steps = 0
        ac_mask_ms = 0.0
        ac_fwd_ms = 0.0

        for out, resamples, valid, violations, remasks, grammar_checks in generate_async_timed(
            model, prompt_ids, tokenizer, checker=checker,
            prompt_len=prompt_len, steps=steps, gen_length=256,
            block_length=bl, temperature=0.2, remasking="low_confidence",
            max_batch_size=8, max_resamples=100,
        ):
            total_violations = violations
            total_remasks = remasks
            total_grammar_checks = grammar_checks

        # Autocompletion fallback: if generation is incomplete, greedily complete
        if out is not None and not valid:
            gen_start = prompt_ids.shape[1]
            # Find consume_idx: first mask position or end of valid prefix
            gen_ids = out[0, gen_start:].tolist()
            mask_id_val = 126336
            # consume_idx relative to sequence start
            first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id_val), len(gen_ids))
            consume_idx_ac = gen_start + first_mask

            out, ac_steps, ac_mask_ms, ac_fwd_ms = autocomplete_greedy(
                model, out, checker, consume_idx_ac, gen_start,
                mask_id=mask_id_val, eos_id=126081,
            )
            # Re-check validity after autocompletion
            gen_ids_ac = out[0, gen_start:].tolist()
            eos_id_val = 126081
            eot_id_val = 126348
            if eos_id_val in gen_ids_ac or eot_id_val in gen_ids_ac:
                eos_pos = next((j for j, t in enumerate(gen_ids_ac) if t in (eos_id_val, eot_id_val)), None)
                valid = eos_pos is not None and mask_id_val not in gen_ids_ac[:eos_pos]

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]

        extracted = instance.extract_result(suffix_str + start_line + code)

        timing = STATS.summary()
        timing["autocomplete_steps"] = ac_steps
        timing["autocomplete_mask_ms"] = ac_mask_ms
        timing["autocomplete_fwd_ms"] = ac_fwd_ms
        # For async, effective constraint = gc + token_select + mask_wait + ac_mask
        # mask_compute was hidden behind forward pass
        effective_constraint_ms = (
            timing["grammar_check_total_ms"]
            + timing["token_select_total_ms"]
            + timing["mask_wait_total_ms"]
            + ac_mask_ms
        )
        total_constraint_ms = (
            timing["grammar_check_total_ms"]
            + timing["token_select_total_ms"]
            + timing["mask_compute_total_ms"]
        )
        total_forward_ms = timing["forward_total_ms"]

        result = {
            "instance_id": instance.instance_id(),
            "method": "dgrammar_v2_async",
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": len(resamples),
            "timing": {
                **timing,
                "total_constraint_ms": total_constraint_ms,
                "effective_constraint_ms": effective_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100)
                    if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "effective_constraint_pct": (effective_constraint_ms / (effective_constraint_ms + total_forward_ms) * 100)
                    if (effective_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": (effective_constraint_ms / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "per_token_total_ms": (elapsed * 1000 / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "mask_time_saved_ms": timing["mask_compute_total_ms"] - timing["mask_wait_total_ms"],
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        gc_mean = timing["grammar_check_mean_ms"]
        fwd_mean = timing["forward_mean_ms"]
        eff_pct = result["timing"]["effective_constraint_pct"]
        avg_batch = timing["avg_batch_size"]
        wait_mean = timing["mask_wait_mean_ms"]
        mc_mean = timing["mask_compute_mean_ms"]
        saved = result["timing"]["mask_time_saved_ms"]
        ac_info = f", AC={ac_steps}steps/{ac_mask_ms:.0f}ms" if ac_steps > 0 else ""
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms(x{timing['forward_count']}), "
            f"gc={gc_mean:.3f}ms(x{timing['grammar_check_count']}), "
            f"mask={mc_mean:.2f}ms(x{timing['mask_compute_count']}), "
            f"wait={wait_mean:.2f}ms(x{timing['mask_wait_count']}), "
            f"overlap={timing['overlap_count']}, saved={saved:.0f}ms, "
            f"eff_constraint={eff_pct:.1f}%, batch={avg_batch:.1f}{ac_info}"
        )


if __name__ == "__main__":
    main()
