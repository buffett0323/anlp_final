讓我從幾個不同角度來分析可能的優化方向。

---

## 方向一：Block-level Joint Verification

**現在 LAVE 的問題**：
```
每個 denoising step 會 unmask 一個 block（預設 32 個位置），
但 LAVE 是逐一 verify 每個 token，
沒有利用「這 32 個 token 是同一個 step 產生的」這個資訊
```

**可以嘗試的做法**：
```
既然知道這個 step 要 unmask 的所有位置，
可以對整個 block 做一次 joint verification，
而不是 32 次獨立的 verification

好處：
- 可以更早發現 block 內的 token 組合不合法
- 減少 verify 次數
```

---

## 方向二：Deterministic Wildcard Verification

**現在 LAVE 的問題**：
```
Lookahead 是 random sampling，
N=10 次不保證找到所有合法路徑，
有 ~2% 的 false negative
```

**可以嘗試的做法**：
```
用 wildcard grammar intersection 取代 random sampling，
把 MASK 當成 wildcard 在 parser chart 上搜尋

理論上可以達到 0% false negative，
也就是 100% syntactic correctness
```

這個方向的核心挑戰就是之前討論的 remaining_masks 追蹤問題。

---

## 方向三：Incremental Parsing Cache

**現在 LAVE 的問題**：
```
每次 verify 都需要重新 parse 整個 prefix，
但相鄰兩個 step 的差異只有幾個新 unmask 的 token
```

**可以嘗試的做法**：
```
維護一個 incremental parser state，
每個 step 只更新新增的 token 部分，
不重新 parse 已經確認合法的 prefix

好處：
- 大幅減少 parsing overhead
- 特別對長 sequence 有幫助
```

---

## 方向四：Grammar-Guided Token Filtering

**現在 LAVE 的問題**：
```
LAVE 是 propose 之後再 verify，
如果 reject 了還要重新 propose，
在困難的 context 下會一直 reject → 觸發 Cache-Enhanced Recovery
```

**可以嘗試的做法**：
```
在 propose 之前就用 grammar 來 filter vocab，
只從合法的 token 集合裡 sample，
而不是 propose 完再 reject

類似 AR LLM 的 constrained decoding 做法，
但需要適配 dLLM 的 non-autoregressive 特性
```

---


## Commands
### Single experiment
modal run bench/modal_lave_improved_bench.py --experiment dir4

### All five in parallel (spawns all chunks for all experiments at once)
modal run bench/modal_lave_improved_bench.py --run-all

Results land in the same dgrammar-results volume as the other benchmarks, named
lave_{experiment}_timed_{dataset}_s{seed}_t{steps}.jsonl.