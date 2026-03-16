# SDSD 消融實驗設計 (Ablation Experiment Design)

## 1. 實驗設置 (Experimental Settings)

### 模型基座 (Backbones)

| 模型 | 類型 | VRAM | Transformers |
|------|------|------|--------------|
| **LLaDA-8B-Instruct** | 離散擴散 (Discrete Diffusion) | ~16GB | 4.38.2 |
| **Dream-7B-Instruct** | 擴散 LLM | ~20GB | ≥4.46 |

### 硬體環境

- **GPU**: 建議 24GB+ VRAM
- **優化**: CSR 稀疏索引 (O(K) 過渡成本)，可考慮 JAX/XLA 靜態編譯優化 CSR 算子

### 超參數設定

| 參數 | 符號 | 建議值 | 說明 |
|------|------|--------|------|
| 擴散步數 | $T$ | 128, 256, 512 | 離散擴散步數 (LLaDA) |
| 投機塊大小 | $\gamma$ | 8, 16 | Speculative Tree draft length |
| 解碼溫度 | Temp | 0.0 (Argmax), 0.4 | 品質評估常用 |
| Herding 延遲 | $\delta$ | 0.0005 ~ 0.01 | 動量衰減 (若可調) |

---

## 2. 評估指標 (Metrics)

### 效率指標 (Efficiency)

| 指標 | 說明 | 預期 |
|------|------|------|
| **TTFT (ms)** | Time to First Token，CSR 消除 O(N) 掃描後的啟動延遲 | STATIC 顯著降低 |
| **NFE** | Number of Function Evaluations，目標模型 Forward pass 次數 | Spec-Tree 減少 40–65% |
| **Throughput (tok/s)** | 每秒生成 token 數 | 隨 STATIC 穩定 |

#### NFE 測量語義 (NFE Measurement Semantics)

- **Sequential 方法 (Baseline, Ablation 1, Ablation 2)**：每生成 1 個 token 呼叫一次 `model.forward()` → **NFE = block_length**。
- **Block 方法 (Ablation 3, SDSD)**：一次 `model.forward()` 取得整塊 logits，再於 CPU 上建樹解碼 → **NFE = 1**。
- 實驗中必須將模型推論納入循環，才能正確反映 SDSD 的 NFE 優勢；若僅傳入預算好的 `prob_vectors`，NFE 會恆為 1，無法區分各方法。

### 可靠性指標 (Reliability)

| 指標 | 說明 |
|------|------|
| **Parse Rate (%)** | 輸出符合 JSON/CFG 語法並可被 Parser 解析的比例 |

### 品質與意圖指標 (Quality & Intent)

| 指標 | 說明 |
|------|------|
| **Gen PPL** | 生成文本困惑度 |
| **Intent Recovery (steps)** | 人為干擾後，高機率 token 恢復 Top-1 的步數 |
| **Pass@1 (%)** | 程式碼生成功能正確性 (HumanEval/MBPP) |

---

## 3. 數據集 (Datasets)

| 數據集 | 用途 | Hugging Face ID |
|--------|------|-----------------|
| **JSON-Mode-Eval** | 複雜嵌套 JSON Schema | `NousResearch/json-mode-eval` |
| **HumanEval** | 程式碼生成 (CFG) | `openai/openai_humaneval` |
| **MBPP** | Python 程式碼生成 | `google-research-datasets/mbpp` |
| **GSM-Symbolic** | 符號數學推理結構穩定性 | `apple/GSM-Symbolic` |

### 數據集下載方式

見下方「數據集下載指南」。

---

## 4. 消融對照表 (Ablation Comparison Table)

### LLaDA-8B-Instruct

| 方法代號 | 技術組件 | 複雜度 | TTFT (ms) | Throughput (tok/s) | NFE (avg) | Parse Rate (%) | Pass@1 (%) | Intent Recovery (steps) |
|----------|----------|--------|-----------|--------------------|-----------|----------------|-----------|--------------------------|
| **Baseline** | 原始 DINGO ($O(N)$) | $O(N)$ | | | 100% | ~100% | | N/A |
| **Ablation 1** | STATIC + DINGO | $O(K)$ | | | 100% | ~100% | | N/A |
| **Ablation 2** | DINGO + Herding | $O(N)$ | | | 100% | 100% | | **低 (預期)** |
| **Ablation 3** | STATIC + Spec-Tree | $O(K)$ | | | **低 (預期)** | ~100% | | N/A |
| **Ours (SDSD)** | **STATIC + Herding + Tree** | **$O(K)$** | **極低** | **極高** | **極低** | **100%** | **高** | **低** |

### Dream-7B-Instruct

| 方法代號 | 技術組件 | 複雜度 | TTFT (ms) | Throughput (tok/s) | NFE (avg) | Parse Rate (%) | Pass@1 (%) | Intent Recovery (steps) |
|----------|----------|--------|-----------|--------------------|-----------|----------------|-----------|--------------------------|
| **Baseline** | 原始 DINGO ($O(N)$) | $O(N)$ | | | 100% | ~100% | | N/A |
| **Ablation 1** | STATIC + DINGO | $O(K)$ | | | 100% | ~100% | | N/A |
| **Ablation 2** | DINGO + Herding | $O(N)$ | | | 100% | 100% | | **低 (預期)** |
| **Ablation 3** | STATIC + Spec-Tree | $O(K)$ | | | **低 (預期)** | ~100% | | N/A |
| **Ours (SDSD)** | **STATIC + Herding + Tree** | **$O(K)$** | **極低** | **極高** | **極低** | **100%** | **高** | **低** |

---

## 5. 實驗邏輯檢核

1. **STATIC 的關鍵**: TTFT 顯著下降且 Throughput 在 $N$ 很大時仍穩定 → 證明 $O(N) \to O(K)$ 成功。
2. **Herding 的關鍵**: Intent Recovery 欄位。SDSD 應能靠權重向量 $w$ 更快找回「原本想說的話」。
3. **Speculative Tree 的關鍵**: NFE 減少 40–65% 以上 → 證明投機並行驗證在結構化場景下高效。

---

## 6. 方法與代碼對應

| 方法 | 代碼模組 | 函數 |
|------|----------|------|
| Baseline | `baseline_dingo.py` | `baseline_dingo_dp` |
| Ablation 1 | `sparse_dingo.py` | `sparse_dingo_dp` |
| Ablation 2 | `herding.py` | `herding_decode` |
| Ablation 3 | `speculative_tree.py` | `speculative_decode_argmax` |
| SDSD | `speculative_tree.py` + `herding.py` | `speculative_decode` (herding path) |

## 7. 執行方式

```bash
# Mock 模式（無 GPU，合成 logits）
python run_ablation.py --model dream --mock --samples 20

# Dream-7B
python run_ablation.py --model dream --samples 20

# LLaDA-8B（需 transformers==4.38.2）
python run_ablation.py --model llada --samples 20

# 使用數據集
python run_ablation.py --model dream --dataset json-mode-eval --dataset-limit 50

# 輸出
python run_ablation.py --model dream --output results/ablation_table.json
```
