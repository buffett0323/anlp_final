# STATIC CSR Alignment Verification

This document verifies that our `sparse_dingo.py` and `csr_dfa.py` implementations are aligned with the official STATIC implementation in `vendor/static-constraint-decoding`.

## Summary: **Aligned** ✓

Our CSR format and O(K) sparse lookup match STATIC's design. The main difference is the **algorithm** (DINGO DP vs beam search) and **input** (DFA vs SID trie).

---

## CSR Format Comparison

### STATIC (vendor/static-constraint-decoding)

| Component | Description |
|-----------|-------------|
| `packed_csr` | Flat array of `[token, next_state]` pairs. Shape `(num_transitions + V, 2)`. |
| `csr_indptr` | Row pointers. For state `s`: `packed_csr[indptr[s]:indptr[s+1]]` = transitions from `s`. |
| Lookup | Given state `s`, get `(token, next_state)` pairs in O(K) where K = valid transitions from `s`. |

**Source:** `csr_utils.py` lines 155–201, `decoding_pt.py` lines 84–86.

### Ours (src/csr_dfa.py)

| Component | Description |
|-----------|-------------|
| `column_indices` | Token IDs (equivalent to `packed_csr[:, 0]`) |
| `values` | Next state IDs (equivalent to `packed_csr[:, 1]`) |
| `row_pointers` | Same as `indptr` (without STATIC's padding). For state `q`: `row_pointers[q]:row_pointers[q+1]` = slice. |
| `get_transitions(q)` | Returns `(token, next_state)` pairs in O(K). |

**Mapping:** Our `(column_indices[i], values[i])` = STATIC's `packed_csr[i]` = `[token, next_state]`.

---

## Sparse Lookup Logic

### STATIC (`generate_and_apply_logprobs_mask`)

```python
starts = csr_indptr[flat_states]
actual_lens = csr_indptr[flat_states + 1] - starts
gathered_vals = packed_csr[gather_indices]  # [token, next_state] per row
candidate_token_ids = gathered_vals[..., 0]
candidate_next_states = gathered_vals[..., 1]
```

**Purpose:** Beam search – for each state, fetch valid (token, next_state) pairs, apply mask to logprobs.

### Ours (`compute_transition_costs_sparse`)

```python
for q_prime in range(csr.num_states):
    transitions = csr.get_transitions(q_prime)  # (token, next_state) pairs
    for t, q_next in transitions:
        # Compute Vi(q_next, q_prime) = max prob of q_prime -> q_next
```

**Purpose:** DINGO DP – for each source state q', iterate only over K valid tokens instead of N.

**Same idea:** O(K) lookup per state instead of O(N) vocabulary scan.

---

## Differences (Not Misalignments)

| Aspect | STATIC | Ours |
|--------|--------|------|
| **Input** | SIDs (prefix tree from corpus) | DFA transition function (any grammar) |
| **Index builder** | `build_static_index` (trie → hybrid dense/sparse) | `build_csr_from_dfa` (δ: Q×V→Q → CSR) |
| **Decoding** | Beam search (autoregressive) | DINGO DP (optimal path) |
| **Dense head** | Yes (first d layers) | No (full CSR for general DFA) |
| **Padding** | `indptr` has length num_states+2 for GPU OOB safety | `row_pointers` length num_states+1 |

---

## Verification

- **CSR format:** Same logical structure – (token, next_state) pairs per row, indexed by row pointers.
- **O(K) lookup:** Both use `indptr[s]:indptr[s+1]` (or equivalent) to get only valid transitions.
- **Transition semantics:** Both store δ(q, t) = q' as (t, q') in the row for state q.

---

## References

- STATIC: `vendor/static-constraint-decoding/static_decoding/csr_utils.py`, `decoding_pt.py`
- Ours: `src/csr_dfa.py`, `src/sparse_dingo.py`
