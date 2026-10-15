# 把測試時計算蒸餾進策略：Verify, Act, Distill (Turning Test-Time Compute into Test-Time Adaptation for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-10-15
>
> **論文**: Verify, Act, Distill: Turning Test-Time Compute into Test-Time Adaptation for Vision-Language-Action Models
> **鏈接**: https://arxiv.org/abs/2610.14885
> **核心定位**: 現有 VLA 的測試時計算（TTC，多候選打分聚合）每次執行都要重算一遍，因為策略被凍結；本文把「驗證器聚合出的好動作」直接當老師數據蒸餾回 VLA 的 LoRA，讓一次 TTC 的收益變成永久能力。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 把 Agg-8 執行中產生的加權聚合動作當 teacher，用 10 條 episode/task 蒸餾進 LoRA，單候選成功率從 54.1% → 70.6%，保留約 74% 的 TTC 增益 |
| 適合精讀 | 如果你在做 VLA 部署後自適應、推理效率優化、或 TTC/verifier-reranking 系統，重點看 §3.2–§3.3 與 §4.6 |
| 可以跳過 | 如果你只關心純預訓練架構創新或大規模真機 benchmark，這篇距離中等（全模擬、單架構、單 seed） |
| 落地可行性 | 中高（不需 RL、不需額外 rollout，只多 200 步 LoRA 更新；但依賴固定 verifier 且需至少一次成功 episode）|
| 主要風險 | verifier 把策略拉向示範動作風格，可能降低候選多樣性；10k checkpoint 上「再放大」效應消失 |

💡 **X-Ray 開場**

VLA 生成式策略每次推理都能吐出多個動作候選，用一個學好的 verifier 給它們打分、加權平均後執行——這就是測試時計算（TTC），能顯著提升成功率。但問題是：策略本身沒被更新，所以下次遇到同一個任務，這筆「生成 + 打分」的算力還得重花一遍，好決策永遠進不了參數。本文的想法很樸素：既然執行時已經算出了「哪個動作更好」這個監督信號，那就把它回收，用標準 flow-matching loss 把聚合後的好動作蒸餾回 VLA。對 VLA 研究者的意義是——TTC 不再只是「一次性選擇器」，而可以變成一個自產自用的學習信號源，TTC 的算力同時在「改進未來的推理」。

📍 **研究全景時間線**

```
[2023] Diffusion Policy / Flow Matching 生成式動作策略
   → [2024] Snell 測試時計算縮放（LLM 領域外溢）
   → [2025] V-GPS 用 value function 重排候選；RoboMonkey 用 learned verifier 打分
   → [2026] 大量 reranking/guidance/tree-search（RoVer, TACO, CoVer, MG-Select…）
   → [2026] PA-RL / PLD / Q-Planning：把改進動作蒸餾回策略，但需 critic / residual RL / evaluator 更新
   → [本文 2610.14885] ← 當前位置
      把「固定 verifier 下的候選聚合結果」直接當 teacher，
      無 RL、無專用 rollout，實現 TTC→TTA（測試時自適應）
   局限：全模擬、單一 VLA 架構、單 seed、per-task LoRA 不跨任務遷移
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

框架叫 **Verify, Act, Distill**，三個階段循環：

1. **Verify**：固定 verifier 給 N 個候選打分。
2. **Act**：分數標準化 → softmax 權重 → 加權平均成一個 plan，執行前 H₂ 步。
3. **Distill**：把「成功 episode」裡的 (observation, 聚合後 chunk) 存進 buffer，用標準 flow-matching loss 更新 LoRA。

### 1.1 系統組件對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 頻率/時序 | 訓練/推理差異 |
|------|------|------|-----------|----------------|
| VLA 策略 π_θ0 (SmolVLA) | 噪聲 z, 觀測 o_t（2 相機 + 8 維本體 + 語言） | 歸一化動作 chunk A ∈ ℝ^(H₁×d)，H₁=50, d=32 | 每個 replanning point b 生成 | 預訓練凍結；只更新 action expert 的 LoRA φ |
| Verifier V_act | o_b 的 frozen VLM connector 特徵 + 候選前 H₂=10 步動作 | 標量分數 s_i | 每個 replanning point（N=8 次調用） | 預先訓練 60k 步，適配期凍結 |
| 候選聚合 Best-of-N | N 個候選 + 分數 | 加權平均 A̅_b | 每個 replanning point | 純推理，無參數 |
| LoRA 蒸餾 | buffer 𝒟（成功 episode 的 (o, A̅)） | LoRA 參數 φ 更新 | 每 task 收集 K=10 episode 後更新一次（200 步） | 只在適配期訓練，評估期不動 |
| 執行接口 g | 歸一化動作 | 環境動作（去歸一化 + 座標變換） | 每次執行前 H₂=10 步 | — |

### 1.2 關鍵機制 (Key Mechanism)

- **監督信號來自哪裡是關鍵差異**：PA-RL 用 critic 選/優化動作，PLD 用 residual RL specialist 蒸餾 recovery 動作，Q-Planning 更新 evaluator 但策略凍結。本文**直接用固定 verifier 下的聚合結果當 teacher**，把「生成 teacher 數據的過程」等同於「普通的候選評估與執行」——零額外成本。
- **只更新 LoRA**：θ₀（VLA 主體）與 verifier 全程凍結，只更新 action expert 的 LoRA（rank 8、scaling 16、0.37M 可訓練參數），回歸到原始輸入輸出維度與目標。
- **老師是完整 H₁ 步 chunk，不是只執行了的 H₂ 步**：權重 w_i 在各時間步共享，所以聚合出的是一個「連貫 plan」，而非各候選拼湊的補丁；且與 SmolVLA 預訓練的輸入/輸出維度一致。
- **成功過濾**：只有 episode 級成功標籤 y_e=1 的資料進 buffer；若 10 條全失敗則跳過更新，保留 base policy。

⚡ **Eureka Moment**：TTC 執行時已經免費產出了一批「高質量動作監督」，把它當 teacher 蒸餾回提出候選的策略本身——驗證不只選擇執行動作，還可以成為學習信號，讓 TTC 的收益被「保留」甚至「再放大」。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
觀測 o_t ─┬─► VLA π_θ0 ──► N 個候選 A^(1..N)（N=8）
          │                    │
          └─► frozen VLM conn. 特徵 ──► Verifier V_act ──► 分數 s_i
                               │
                     softmax 加權 (λ=0.3)
                               │
                     A̅_b = Σ w_i A^(i)  ← 聚合 plan
                               │
                  執行前 H₂=10 步 g(A̅_0:H2)
                               │
                    ┌──────────┴───────────┐
               episode 成功? y_e=1         y_e=0 → 丟棄
                    │
        buffer 𝒟 = {(o_b, A̅_b)}   (K=10 episodes)
                    │
        flow-matching loss 更新 LoRA φ (200 步)
                    │
        ┌───────────┴────────────┐
   單候選推理（免 verifier）   再套 Agg-8（再放大）
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
TTA 增益 = 把 Agg-N 執行時算出的 A̅ 當 teacher，用 flow-matching loss 寫回 LoRA
```

**目標**：讓 verifier-free 的單候選生成，盡可能逼近（並保留）多候選聚合的收益。

**候選加權聚合**（論文 Eq.1）：

```
w_i = exp(u_i) / Σ_j exp(u_j)
u_i = (s_i - max_j s_j) / [λ · (σ_s + δ)]
A̅_b = Σ_i w_i · A^(i)
```

變數說明：

| 符號 | 含義 |
|------|------|
| s_i | verifier 對第 i 個候選的分數 |
| σ_s | 該組內分數標準差 |
| λ | softmax 溫度，默認 0.3 |
| δ | 數值穩定項 1e-8 |
| A̅_b | 加權聚合後的執行 plan（歸一化空間）|
| N | 候選數，默認 8 |

**蒸餾緩衝區**（Eq.2）：

```
𝒟 = ∪_{e: y_e=1} { (o_b, A̅_b) : b 是 episode e 的 replanning point }
```

**Flow-matching 蒸餾損失**（Eq.3）：

```
A_τ = (1 - τ)·A̅ + τ·ε,   τ ~ U(0,1),  ε ~ N(0, I)
L(φ) = E_{(o,A̅)~𝒟, τ, ε} [ ‖ v_{θ0,φ}(A_τ, τ | o) - (ε - A̅) ‖_F² ]
```

> 符號與原文保持一致：損失在 data-to-noise 方向定義，推理時反向積分；v 是預測速度場的網絡，F 為 Frobenius 範數。

**直覺**：你已經在執行時算出了一個「比任何單一候選都更靠譜」的 A̅。那就讓策略學會「直接生成 A̅ 這種質量的動作」——用標準 flow-matching 目標把 A̅ 拉成去噪目標即可。不需要 reward、不需要 gradient through critic，因為 teacher 本身就是連續動作，監督是稠密的（每個 replanning point 一個樣本）。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

假設某個 replanning point，VLA 吐出 3 個單步候選（簡化 N=3，實際 N=8），verifier 給出分數：

| 候選 | 分數 s_i | 動作 A^(i)（1D 簡化）|
|------|----------|----------------------|
| A1 | 0.80 | 1.0 |
| A2 | 0.50 | 0.4 |
| A3 | 0.30 | -0.2 |

步驟 1：算 u_i。max s = 0.80，取 σ_s 為組內標準差（此例約 0.205），λ=0.3，δ≈0：

```
分母 = 0.3 × (0.205) ≈ 0.0615
u_1 = (0.80 - 0.80)/0.0615 = 0
u_2 = (0.50 - 0.80)/0.0615 ≈ -4.88
u_3 = (0.30 - 0.80)/0.0615 ≈ -8.13
```

步驟 2：softmax 權重：

```
exp(0)=1, exp(-4.88)≈0.0076, exp(-8.13)≈0.00029
Σ ≈ 1.0079
w ≈ [0.992, 0.0075, 0.0003]
```

步驟 3：加權平均：

```
A̅ ≈ 0.992·1.0 + 0.0075·0.4 + 0.0003·(-0.2) ≈ 0.995
```

觀察：由於 λ=0.3 很小（低溫），softmax 幾乎退化成 argmax，聚合動作 ≈ 最高分候選。這解釋了論文附錄對比 **Agg-8 vs Sel-8（argmax）** 的必要性——低溫下兩者接近，但聚合仍保留了「多候選共識」的平滑性。

步驟 4：若該 episode 成功，把 (o_b, A̅≈0.995) 存進 buffer，訓練時對 A_τ = (1-τ)·0.995 + τ·ε 回歸速度場目標 (ε - 0.995)。K=10 條 episode 累積後做 200 步 Adam 更新。

## 4. 工程視角 (Engineering View)

| 面向 | 事實（論文陳述） | 工程含義 |
|------|------------------|----------|
| 額外推理成本 | 蒸餾後單候選推理**不需要** verifier 及其 observation encoder | 部署時可省掉 verifier 的算力與內存 |
| 額外訓練成本 | 每 task 收集 K=10 episode + 200 步 LoRA | 0.37M 可訓練參數，極輕量 |
| TTC 調用開銷 | N=8 時每 replanning point 調用 verifier 8 次 | 保留 Agg 則維持原成本，換取更高上限 |
| 執行時序 | H₁=50 生成、H₂=10 執行（半分開區間） | replanning 頻率由 H₂ 決定；teacher 含完整 50 步 |
| 延遲聲明 | 論文明確說「real hardware latency 尚待檢驗」，成本論證是**數候選數與評估器調用次數**，非實測 | 真機部署前需自行量測 |
| 內存 | LoRA rank 8 / scaling 16 / batch 8 | 邊緣端適配友好 |
| 硬件 | 實驗跑在 RTX PRO 6000 (96GB) + RTX 6000 Ada (48GB) | 訓練側需求，非部署側 |

**Trade-off**：蒸餾把 TTC 的收益「凝固」進參數，代價是可能把策略拉向 verifier 訓練所用的示範動作分佈（style bias），並可能壓縮候選多樣性——後文 §4.3 的 10k 現象與此一致。

## 5. 數據與評測 (Data & Eval)

- **基準**：LIBERO 四套件（Spatial / Object / Goal / Long，各 10 任務 = 40 任務）。
- **策略**：SmolVLA，主實驗用 **5k checkpoint** 作為「尚未充分適配」的代理，另測 10k 與 20k。
- **訓練數據**：SmolVLA 在 10,000 條由 π_0.5 收集的 rollout 中 5,225 條成功 episode 上訓練。
- **Verifier 訓練數據**：official LIBERO 示範（四套件）每狀態生成 8 個候選 + 新收集的成功 rollout；用 all-pairs Bradley–Terry loss 學「與記錄動作 RMSE 更小者更優」。訓練 60k 步、batch 16、lr 1e-4。
- **適配/評估協議**：每 task 用 reset seeds 0–9 適配，held-out seeds 10–59（每 task 50 episode = 每套件 500）評估；Base 與 Agg-8 用同一 seed 集。
- **報表指標**：套件均值、四套件宏平均、per-task 配對差（mean ± SEM）、改善/持平/退化任務計數。
- **分佈偏移**：LIBERO-plus，每套件 105 任務（7 類擾動各 15），適配 seed 0–9、評估 seed 10–19；因算力，蒸餾策略只評每套件前 35 任務（140 任務）。

**主要結果（Table 1，LIBERO clean，SmolVLA 5k）**：

| 方法 | 單候選成功率 | +Agg-8 成功率 |
|------|--------------|----------------|
| Base | 54.1% | 76.4% |
| Self（自模仿成功軌跡，無 verifier）| 63.7% | 80.8% |
| **Distill（本文）** | **70.6%** | **82.3%** |
| N-GRPO（noise-space，用 verifier 排序當 reward）| 58.1% | 79.2% |

- **保留率**：(70.6 − 54.1) / (76.4 − 54.1) ≈ **74%**。
- per-task 配對差 vs Base：**+16.5 ± 2.7** 點，34 改善 / 1 持平 / 5 退化。
- Distill + Agg-8 vs 原 Agg-8：**+5.8 ± 2.3** 點；Long 套件 44.0% → **57.8%**（24 改善 / 6 持平 / 10 退化）。
- **10k checkpoint**：單候選 83.2% → 86.2%（+3.0 ± 1.4）；但 Distill+Agg-8 89.8% 未超過原 Agg-8 90.4%，**再放大效應消失**。verifier 驗證配對準確率從 5k 的 0.82 降到 10k 的 0.73。
- **LIBERO-plus**（10k，共同 35 任務子集）：Base 20.7% → Distill 單候選 24.5%（+3.9 ± 1.4，43/74/23），Distill+Agg-8 25.4%（vs 原 Agg-8 僅 +0.6 ± 1.2）。且 140 任務中 **66 個在 10 episode 內完全無成功**，無法更新。
- **消融**：去掉成功過濾 → 64.4% / 78.8%（明顯更差，過濾貢獻 +6.2 ± 1.6 / +3.5 ± 1.7）；用最高分候選代替聚合 chunk 當 teacher → 67.8% / 82.5%（單候選反而 −2.8 ± 2.0）。
- **適應預算**：Long 套件 5k，僅 K=2 就到 29.6%，K=50 到 34.6%；K=10 下 Distill 27.0% > N-GRPO 25.6% > DSRL-style SAC 19.4%。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 把 TTC 的一次性增益「固化」進免 verifier 的單候選推理（省算力）。
- 在計算允許時，蒸餾後的策略再套 Agg-8 可進一步提升上限（尤其 Long 長程任務）。
- 樣本效率高：幾條 rollout 就能適配（K=2 已見效）。

**不能做 / 失敗模式**：
- **依賴 verifier 質量**：verifier 與示範動作距離綁定，可能把策略拉向示範風格、壓縮候選多樣性（10k 時再放大失效疑與此相關，但作者說無法僅憑 verifier 準確率下降斷定）。
- **依賴至少一次成功**：LIBERO-plus 140 任務中 66 個完全無成功 → 無法更新；真機上成功信號來自檢測器，其誤差會直接污染 buffer。
- **無條件更新、無回退**：沒有驗證步或 fallback，40 任務中單候選有 5 個退化、再聚合時有 10 個退化。
- **僅模擬、單架構、單 seed**：per-task LoRA 不跨任務遷移；成本論證用候選/評估器調用次數而非實測延遲。

### 6.1 隱含假設 (Hidden Assumptions)

- **成就假定**：假設 verifier 對「低質策略的候選」仍具足夠判別力（5k 準確率 0.82），但這是一個為每類 checkpoint 單獨訓練的 verifier——「no additional data」只對適配階段成立，verifier 訓練本身用了新收集的成功 rollout。
- **成功標籤可靠**：訓練與評估都用 episode 級二元成功；真機場景下這個前提未必成立。
- **示範動作即最優**：verifier 以「與記錄動作的 RMSE」為排序依據，隱含「接近示範即更好」——在示範本身非最優或多模態的任務上可能引入偏差。
- **候選多樣性充足**：整個方法靠生成多樣候選來產生 TTC 收益，但未量化蒸餾後多樣性的變化。
- **per-task 適用**：假定每個任務可獨立收集 K 條 episode 並獨立更新，未驗證跨任務/連續部署場景。

## 7. 與相關工作對比 (Comparison)

| 方法 | 監督來源 | 是否需 RL | 更新對象 | 適用場景 |
|------|----------|-----------|----------|----------|
| V-GPS / RoboMonkey | 訓練好的 verifier / value function | 否 | 不更新策略（僅執行時選擇）| 一次性 TTC 提升 |
| PA-RL | Q-function 選擇 + 局部優化 | 是（critic）| 策略 | 需 critic 的場景 |
| PLD | residual RL specialist 的 recovery 動作 | 是 | VLA | 有專用 RL specialist |
| Q-Planning | 經驗更新 evaluator | 部分 | evaluator（策略凍結）| 持續改進 evaluator |
| N-GRPO（本文對照）| verifier 排序作 reward | 是 | noise 擾動分佈 | noise-space RL |
| **Verify, Act, Distill** | **固定 verifier 下的聚合動作** | **否** | **LoRA（action expert）** | **部署後輕量自適應** |

**面試 Tip**：被問到「這篇跟 RoboMonkey 差在哪」——一句話答：**RoboMonkey 用 verifier 選動作執行，本文用 verifier 的聚合結果當 teacher 反過來訓練產生候選的策略**，把推理時算力變成學習信號，實現 TTC→TTA，且不需 RL 或額外 rollout。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做 VLA 部署後自適應 / test-time training 的研究者（§3.2–§3.3 是核心）。
  2. 要評估「能否把 verifier-reranking 收益固化到策略」的系統工程師（看 §4.2、§4.6 的預算與 N 依賴）。
  3. 研究 TTC 縮放在機器人策略上邊界的人（§4.3 的 10k 現象值得深挖）。
- **建議章節路徑**：先讀 §1 引言 + Figure 1（30 分鐘懂全貌）→ §3.2/§3.3（公式與機制）→ §4.2 主結果 + §4.6 預算/N 依賴 → §5 局限（別跳，理解前提條件）→ 有餘力再看 §4.3/§4.5 的分佈偏移與 10k 現象。
- **不值得精讀的理由**：若你不做機器人學習、或已熟悉 flow-matching 蒸餾與 candidate reranking，讀摘要 + Figure 1 + Table 1 即可；本文的理論新意在於「監督來源的重新定位」而非全新數學框架。

---
[← Back to Theory](./README.md)

**關鍵引用鏈接**：
- 論文：https://arxiv.org/abs/2610.14885
- 引用之 verifier 方法 RoboMonkey (Kwok et al., 2025)：https://arxiv.org/abs/2502.02538
- 蒸餾對照 PA-RL (Mark et al., 2024)：https://arxiv.org/abs/2410.06158
- 基準 LIBERO (Liu et al., 2023)：https://arxiv.org/abs/2306.03310
