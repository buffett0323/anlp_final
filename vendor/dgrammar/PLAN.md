# Dgrammar: Research & Implementation Plan

## 1. One-sentence Contribution

Dgrammar 是第一個在 grammar-constrained decoding 中保留 dLLM parallel decoding 能力的方法，
透過 selective remasking（而非 rejection sampling）將 grammar violations 轉化為 refinement targets。

---

## 2. Problem Statement

| 方法 | 每步 tokens | 機制 | 問題 |
|---|---|---|---|
| Mündler et al. | 1 | reject + resample | 退化成 sequential；走到死胡同 |
| LAVE | 1 | lookahead + verify | 退化成 sequential；lookahead 有 approximation error |
| **Dgrammar** | **K (4-8)** | **parse + selective remask** | **保留 parallel advantage** |

核心 insight：LAVE 和 Mündler 都把 grammar constraint 變成了 accept/reject 問題，
但 dLLM 天生就有 iterative refinement 能力 — grammar violation 應該觸發 refinement，不是 rejection。

---

## 3. Method（四步 decode loop）

```
Input:  prompt, grammar G, dLLM model, total steps T
Output: grammar-compliant text

1. Initialize: x = [MASK] * L （全 mask）

2. For step t = 1..T:
   a. Forward pass: logits = model(prompt, x)
   b. Standard unmask: 選 top-K 高信心 mask 位置 → unmask 為 predicted tokens
      （K = adaptive，基於 confidence threshold，跟 Fast-dLLM 一致）
   c. Grammar check: 用 incremental parser 檢查當前 x
   d. If violations found:
      - 識別哪些「剛 unmask 的 K 個位置」違反 grammar
      - 只 remask 那些 violating positions → 回到 [MASK]
      - 合法 positions 保留
   e. If no violations: 繼續

3. Final check: 如果最終 output 仍有 violation，fallback 到 per-token mode（LAVE-like）
```

### 關鍵設計決策

**Q: Incremental parser 用什麼？**

三個 candidate，按 task 選擇：

| Task | Parser | 理由 |
|---|---|---|
| JSON | jsonschema validator | 成熟、精確、fast |
| C++ / Code | tree-sitter (incremental) | 業界標準，支援 partial parse |
| SMILES | rdkit / 自訂 CFG | domain-specific |
| General CFG | Earley parser (Rust) | 通用，O(n³) worst case 但實際常 O(n²) |

**實際策略：先用 tree-sitter（C++）和 jsonschema（JSON）跑主實驗，
Earley parser 作為 general-purpose fallback。**

**Q: 怎麼判斷「哪個 token 違反」？**

三層策略（由快到慢）：
1. **Fast path**: parser error 通常報 error position → 直接用
2. **Medium path**: 如果 error position 不明確，逐個替換成 [MASK] 測試
3. **Slow fallback**: 全部 K 個 positions 都 remask（最保守）

**Q: Convergence 怎麼保證？**

- 設定 max_remask_attempts per step（如 3 次）
- 如果 3 次都修不好，降級為 per-token mode（greedy sequential）
- 實驗中記錄 「降級率」作為 metric

---

## 4. Architecture（Rust + Python）

```
┌──────────────────────────────────────────────┐
│  Python (PyTorch)                             │
│  ┌──────────────────────────────┐             │
│  │  decode.py                    │             │
│  │  - dLLM forward pass          │             │
│  │  - unmask top-K               │             │
│  │  - call grammar_check()       │  ◄── 主 loop│
│  │  - selective remask            │             │
│  └──────────┬───────────────────┘             │
│             │ FFI call                         │
│  ┌──────────▼───────────────────┐             │
│  │  dgrammar_core (Rust → PyO3)  │             │
│  │  - Earley parser              │             │
│  │  - find_violations()          │  ◄── hot path│
│  │  - incremental state mgmt     │             │
│  └──────────────────────────────┘             │
└──────────────────────────────────────────────┘
```

**為什麼 Rust：**
- grammar check 是 inner loop（每 denoising step 調一次）
- 128 token 序列的 Earley parse: Python ~5ms, Rust ~0.1ms
- 加上 violation detection（每個 candidate 重新 parse），Python 會成為瓶頸
- tree-sitter 本身就是 Rust/C，binding 自然

**Python 負責：**
- model inference（PyTorch）
- token management（mask/unmask logic）
- evaluation harness
- 跟 Dream/LLaDA 的 generation utils 整合

---

## 5. Codebase 依賴

### 要 fork/修改的
- `dream_generation_utils.py`（Dream-7B 的 decode loop）
  - 插入 grammar check 在 unmask 之後
  - 插入 selective remask logic
  - ~80 行改動
- 或 `llada` 的等效 generation code

### 要寫的（新 code）
| 模組 | 語言 | 行數估計 | 功能 |
|---|---|---|---|
| `dgrammar_core/earley.rs` | Rust | ~300 | Earley parser + incremental state |
| `dgrammar_core/violations.rs` | Rust | ~150 | violation detection logic |
| `dgrammar_core/lib.rs` | Rust | ~100 | PyO3 bindings |
| `dgrammar/decode.py` | Python | ~200 | 主 decode loop (integrates with Dream/LLaDA) |
| `dgrammar/grammar.py` | Python | ~100 | grammar definition helpers (JSON schema → CFG) |
| `dgrammar/eval.py` | Python | ~150 | evaluation harness |
| `dgrammar/baselines.py` | Python | ~100 | LAVE / unconstrained baseline runners |
| **Total** | | **~1100** | |

### 第三方依賴
- `tree-sitter` + `tree-sitter-cpp` (Python binding，用於 C++ parsing)
- `jsonschema` (Python，用於 JSON validation)
- `transformers` + Dream/LLaDA model weights
- `maturin` (Rust→Python build tool)
- `pyo3` (Rust crate for Python bindings)

---

## 6. Experiments

### 6.1 Benchmarks（跟 LAVE 完全對齊，確保可比）

| Benchmark | Task | Grammar | # Problems | Metric |
|---|---|---|---|---|
| CPP-Bench | C++ code generation | C++ CFG | 164 | syntactic@k, pass@k |
| JSON-Bench | JSON generation | JSON Schema | 272 | syntactic@k, functional@k |
| SMILES-Eval | Chemical expression | SMILES grammar | 167 | syntactic@k, validity |

### 6.2 Models

| Model | Size | Quantization | GPU |
|---|---|---|---|
| Dream-7B-Instruct | 7B | int4 (bitsandbytes) | 3090 24GB |
| DiffuCoder-7B | 7B | int4 | 3090 24GB |
| LLaDA-8B-Instruct | 8B | int4 | 3090 24GB |
| LLaDA-1.5 | 8B | int4 | 3090 24GB |

### 6.3 Baselines

1. **No-CD**: 無 constrained decoding（unconstrained dLLM）
2. **Mündler (FS-CD)**: per-token rejection sampling
3. **LAVE**: per-token + lookahead verify
4. **AR + GBNF**: llama.cpp 的 grammar-constrained AR decoding（upper bound reference）

### 6.4 Metrics

**Primary:**
- **syntactic@k** (k=1,3,5,10): k 次 sampling 中有幾次 syntactically valid
- **Throughput** (tokens/sec): 包含 grammar checking overhead
- **Total denoising steps**: 到達 valid output 的總步數

**Secondary:**
- **pass@k**: functional correctness（code 能跑）
- **Degradation rate**: 降級到 per-token mode 的比例
- **Violation distribution**: 每步平均有幾個 violations，幾次 remask 能修好

### 6.5 Ablations

1. **K 值的影響**: K=1 (=LAVE-like) vs K=2,4,8,16
2. **Violation detection 策略**: parser error position vs per-token check vs conservative
3. **Max remask attempts**: 1,2,3,5 次
4. **Grammar complexity**: 簡單 grammar (JSON) vs 複雜 grammar (C++)

---

## 7. Expected Results（假設）

| Method | syntactic@1 (CPP) | syntactic@1 (JSON) | Throughput vs No-CD |
|---|---|---|---|
| No-CD | ~77% | ~60% | 1.0× |
| Mündler (FS-CD) | ~85% | ~80% | 0.3-0.5× (很慢) |
| LAVE | ~97% | ~95% | 0.5-0.7× (per-token overhead) |
| **Dgrammar** | **~93-97%** | **~92-96%** | **0.7-0.9×** |

**核心 claim:**
- syntactic correctness 接近 LAVE（可能略低，因為 parallel unmask 的 approximation）
- throughput 顯著高於 LAVE（2-3× faster，因為每步 K tokens vs 1 token）
- 整體 speed-accuracy tradeoff 優於所有 baselines

**如果 syntactic 比 LAVE 低太多？**
- 這本身是有價值的 finding：quantify parallel decoding vs grammar constraint 的 tradeoff
- Paper 可以改寫為 "Understanding the Parallelism-Constraint Tradeoff in dLLM Decoding"

---

## 8. Timeline

| 週 | 任務 | Deliverable |
|---|---|---|
| **Week 1** | Rust parser core + PyO3 binding + unit tests | `dgrammar_core` 可 import from Python |
| **Week 2** | Dream-7B decode loop 整合 + JSON-Bench 跑通 | 第一個 end-to-end 結果 |
| **Week 3** | CPP-Bench + SMILES-Eval + baselines 對齊 | 主實驗 Table 完成 |
| **Week 4** | Ablations + throughput profiling + paper writing | Draft v1 |

---

## 9. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Selective remask 不 converge | Medium | High | fallback to per-token; measure degradation rate |
| Grammar check overhead 太大 | Low | Medium | Rust 已經很快；可 batch check |
| syntactic@1 比 LAVE 差太多 | Medium | Medium | 改寫 story 為 tradeoff analysis |
| LAVE code 未公開無法 reproduce | Medium | Medium | 自己 implement LAVE 作為 baseline |
| Concurrent work 搶先 | Low | High | 已確認空白；加快 timeline |

---

## 10. Paper Outline

```
1. Introduction
   - dLLM 的 parallel advantage
   - Constrained decoding 的需求
   - 現有方法的 per-token bottleneck
   - Dgrammar: detect-and-repair paradigm

2. Background
   - Masked diffusion LLMs (LLaDA, Dream)
   - Constrained decoding for AR LLMs
   - Earley parsing basics

3. Method
   - 3.1 Problem formulation
   - 3.2 Dgrammar decode loop (Algorithm 1)
   - 3.3 Violation detection strategies
   - 3.4 Convergence analysis (informal)
   - 3.5 Complexity analysis (vs LAVE, Mündler)

4. Experiments
   - 4.1 Setup (models, benchmarks, baselines)
   - 4.2 Main results (Table 1: syntactic@k, Table 2: throughput)
   - 4.3 Ablation studies
   - 4.4 Case study: when does Dgrammar fail?

5. Related Work
   - Constrained decoding for AR LLMs
   - dLLM inference acceleration
   - Constrained decoding for dLLMs (Mündler, DINGO, LAVE)

6. Conclusion
```

**Target venue**: ACL 2026 / EMNLP 2026 / NeurIPS 2026
**Backup**: NAACL Findings / ICLR 2027 Workshop