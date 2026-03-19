# Unified Benchmark Results Summary

## Extracted Output Assessment

**結論：extracted 內容大多不合理。**

各 method 的 `extracted` 狀況：

| Method | 問題描述 |
|--------|----------|
| **baseline** | 幾乎全是換行符 `\n` 和重複的 `}`，沒有有意義的 JSON 結構 |
| **ablation1** | 多為單一 `{`、長串 `<<<<<<<` 或 `}}}}}}`，無有效 JSON |
| **ablation2** | 與 ablation1 類似：`{`、`<<<<<<<`、`}}}}}}` |
| **ablation3** | 出現 `>I,`、數字、逗號等 token 組合，非 JSON |
| **sdsd** | 混合 `{`、`{[]]`、`{869)}` 等，多數為無意義字元序列 |

`valid: true` 表示 DFA 約束通過（token 序列在 grammar 內），**不代表** 解碼後的文字是合法 JSON。模型在 permissive DFA 下產出的 token 雖符合 grammar，但解碼後多為無意義字串。

### 修正建議

根本原因：目前使用 **permissive DFA**（`build_json_dfa_from_tokenizer`），只接受約 200 個 token，且 `trans_fn` 在 `num_states==2` 時被覆寫為接受**任意 token**，等於幾乎沒有約束。模型（LLaDA-8B 擴散 LLM）在無約束下容易產出無意義 token 序列。

| 修正方向 | 具體做法 |
|----------|----------|
| **1. 使用 strict JSON schema DFA** | 將每個 instance 的 `schema` 編譯成 DFA，只允許符合 schema 的 token 序列。可參考 [eth-sri/constrained-diffusion](https://github.com/eth-sri/constrained-diffusion) 或整合 `dgrammar` / `outlines` 等 grammar 解碼器。 |
| **2. 移除 trans_fn 覆寫** | 在 `run_unified_benchmark.py` 中，當使用 schema DFA 時不要用 `trans_fn(q,t)=1 if t<vocab_size else None`，改為使用 CSR 的實際 transitions，讓 DFA 真正限制 token。 |
| **3. 驗證模型與任務相容性** | LLaDA-8B 為擴散 LLM，需確認其 inference 流程（mask 位置、logits 對應）是否與 json-mode-eval 的 autoregressive 假設一致。可先用簡單 prompt 測試能否產出合理 JSON。 |
| **4. 加強 extract_result** | 若 `{` 或 `[` 不存在，可嘗試從 `output` 欄位做 schema 比對或回退；或標記 `extracted` 為 invalid 而非回傳原始 decoded。 |

建議優先做 **1 + 2**：為每個 instance 建立 schema-specific DFA，並確保 `trans_fn` 使用該 DFA 的 transitions。

### 已實作修正 (run_unified_benchmark.py)

| 修正 | 狀態 |
|------|------|
| **1. schema_guided** | 使用 llguidance (vendor/dgrammar) 做 schema-specific 解碼，產出正確 JSON |
| **2. 移除 trans_fn 覆寫** | 一律使用 CSR 實際 transitions，不再用 `trans_fn(q,t)=1` 接受任意 token |
| **進度條** | tqdm 顯示 `inst/total [elapsed<remaining]` |
| **--skip-slow** | 可略過 baseline（O(N) per step，~30–60 min/instance） |

**特別慢的方法（建議先略過不測）：** `baseline` — 每步 O(N) 掃描 vocab，272 instances 約 30–60 分鐘。

---

## Metrics Table

| method | n | time_taken (s) | time_min | time_max | valid% | forward_ms | constraint_ms | constraint_pct% |
|--------|---|----------------|----------|----------|--------|------------|---------------|-----------------|
| ablation1 | 272 | 34.10 | 15.95 | 53.44 | 100.0 | 34060 | 45 | 0.14 |
| ablation2 | 272 | 37.10 | 16.80 | 57.05 | 100.0 | 34158 | 2945 | 8.33 |
| ablation3 | 272 | 12.08 | 9.43 | 15.16 | 100.0 | - | - | 0.00 |
| baseline | 277 | 51.95 | 30.23 | 71.60 | 100.0 | 32574 | 19379 | 38.38 |
| sdsd | 272 | 15.09 | 10.83 | 19.28 | 100.0 | - | - | 0.00 |

**說明：**
- `time_taken`: 單次 inference 總耗時（秒）
- `forward_ms`: 模型 forward 總時間（ms），ablation3/sdsd 未記錄
- `constraint_ms`: DFA 約束檢查時間（ms）
- `constraint_pct%`: 約束檢查佔總 forward 時間的比例

**觀察：**
- **baseline** 最慢（~52s），且 constraint 佔 ~38%
- **sdsd** 與 **ablation3** 最快（~12–15s），但無 forward/constraint 細分
- **ablation1** constraint 幾乎為 0（~0.14%），ablation2 約 8%
