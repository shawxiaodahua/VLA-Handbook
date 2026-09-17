# 把幾何接進動作專家：sensVLA 的空間接地 VLA 架構 (sensVLA: Spatially-Grounded Vision-Language-Action Model for Autonomous Wheel Loader)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-17
>
> **论文**: sensVLA: Spatially-Grounded Vision-Language-Action Model for Autonomous Wheel Loader
> **链接**: https://arxiv.org/abs/2609.17021 (ICRA 2026 Workshop: From Data to Decisions: VLA Pipelines for Real Robots)
> **核心定位**: 為重型工程機械（輪式裝載機）提出的 VLA 架構——把 LiDAR 的 BEV 幾何特徵**繞過語言解碼器**，直接用一條專屬 cross-attention 通路餵給可訓練的動作專家，從而在「攝影機失效」時仍能維持度量一致的動作預測。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 把 BEV 幾何直接接進 action expert（而非塞進 VLM 解碼器），在 loading 場景把縱向速度 RMSE 降 28%、位移誤差降 9.4%，且在攝影機被移除時退化幅度小 29% |
| 適合精讀 | 如果你在做**重型/戶外/多執行器**機器人（裝載、挖掘、礦卡），或想解決「RGB 對度量幾何不敏感」與「單模態失效」問題，重點看 §IV 與 §V-B |
| 可以跳過 | 如果你只關心室內桌面操作、或已經非常熟悉 π0 式的 flow-matching 動作頭，這篇距離中等（新意在「模態路由位置」而非動作生成本身） |
| 落地可行性 | 中（架構與訓練配方清晰可抄；但依賴私有 LiDAR 資料集與凍結的 in-house PointPillars，復現門檻在資料） |
| 主要風險 | 單一機型（輪式裝載機）+ 私有資料，泛化性未驗證；文中超參數有自相矛盾處（見 §6.1） |

💡 **X-Ray 開場**
這篇論文問的是：當一台自駕裝載機在堆料場開來開去時，語言模型負責「懂任務」，那「懂距離」該由誰負責？作者發現，如果把 LiDAR 的鳥瞰圖（BEV）幾何特徵塞進 VLM 的語言解碼器會浪費 token 預算又損失空間解析度，於是改為把它直接餵給負責出動作的 transformer「動作專家」。結果不但 loading 場景更準，還帶來一個意外的紅利：把攝影機整個拿掉，模型也不會崩。

📍 **研究全景時間線**

```
2023  RT-1 / RT-2 / PaLM-E          →  2024 OpenVLA / Octo / Diffusion Policy
「VLM 能否驅動機器人」的開端               「開源通用策略 + 連續動作頭」成熟
                                                  │
2024  π0 (flow-matching VLA)       →  2026  sensVLA ← 本文
「用 flow matching 生成連續動作塊」          「幾何不進語言解碼器，改走獨立 cross-attn 通道」
                                                  │
                                    局限：單一機型、私有資料、BEV 編碼器凍結
```

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

sensVLA 是一次「**模態路由**」的設計實驗。輸入四種模態（前後 RGB、文字 prompt、本體感知狀態、時序累積的 LiDAR 點雲），輸出未來 10 步、每步 6 維的連續動作塊。

### 1.1 系統對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 是否訓練 | 頻率/時序 |
|------|------|------|----------|-----------|
| Qwen3-VL-2B VLM | 前+後 RGB + 文字 + 當前狀態 token | 任務語義上下文 C_vlm | 凍結主幹 + LoRA 適配（最後 8 層, r=16） | 每次前向一次；解碼器截斷到前 14 個 block |
| PointPillars BEV 編碼器 | 0.5s 累積的前+後 LiDAR 掃描（前雷達座標系） | BEV 特徵圖 B → 32 個空間 token | **完全凍結**（in-house 預訓練） | 0.5s 窗口；多尺度特徵金字塔上採樣拼接 |
| 狀態歷史編碼器 | 過去 10 步本體感知 | 歷史上下文（壓縮成 token） | 可訓練 | 10 步窗口 |
| 動作專家 (Action Expert) | 加噪動作塊 x_t + VLM 上下文 + BEV token + 狀態 token | 速度場（velocity field）→ 動作塊 | **完全可訓練** | 10 步 chunk，stride 5，等效 2s horizon @ 25Hz |
| Flow-matching 頭 | 速度場積分 | 6 維連續動作 | 併入專家訓練 | 推理時 10 步 midpoint ODE solver (t: 1→0) |

### 1.2 關鍵機制 (Key Mechanism)

1. **幾何不走語言解碼器。** BEV 特徵的存在意義是保留空間解析度；若塞進 VLM 的 token 序列，既會和語言 token 搶預算，又會被 LoRA 的低秩容量限制。作者選擇一條**平行**通路。
2. **異質 cross-attention（Heterogeneous Cross-Attention）：** 動作專家的不同層各自 attend 不同來源——第 1 個 cross-attn 看 VLM 上下文（語義接地），第 2 個看 BEV token（空間接地，帶近恆等初始化的 gated residual），第 3 個看兩者拼接（讓 softmax 逐 query 自選模態）。這樣在不增加多少參數的前提下把空間接地「插」進決策層。
3. **雙向 + 因果混合注意力：** 專家第 1 層是雙向 self-attention（讓 10 個動作位自由交換資訊），其餘為因果層，兼顧動作塊內部的協調與時序自回歸。
4. **本體感知不膨脹 VLM 序列：** 10 步狀態歷史用卷積/重採樣壓成少數 token，低頻慣性上下文不佔用 VLM 的序列長度。
5. **分階段 PEFT 訓練：** 首個 epoch 凍結 LoRA、只暖機動作專家；後續解凍。配合 BEV token dropout 與狀態歷史 dropout 抑制小樣本過擬合。

⚡ **Eureka Moment**：**「空間接地應該發生在決策層，而不是感知層或語言層」**——把 BEV 特徵從感知棧裡拉出來、直接接到動作專家，同時收穫了準確度（loading 場景）與容錯性（攝影機失效）兩個紅利。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
 前+後 RGB ─┐
 文字 prompt ─┼─► Qwen3-VL-2B (frozen, 前14 block) ─► C_vlm ─┐
 當前狀態 s ─┘   (LoRA r=16 @ last 8 layers)                    │
                                                                ├─► Action Expert
 前+後 LiDAR ─► PointPillars (frozen) ─► BEV 特徵圖 B            │   (hidden=384,
               (0.5s 累積, 前雷達座標系)  └► flatten ─► 32 tokens ┘    8 heads, 5 layers)
                                                                        │
 過去10步狀態 ─► Conv 編碼 ─► 歷史 token ────────────────────────────────┤
                                                                        ▼
                          加噪動作塊 x_t (10×6) + 時間嵌入 t ─► velocity field
                                                                        │
                                         midpoint ODE (t: 1→0, 10 步) ───┘
                                                                        ▼
                                                          10 步 × 6 維連續動作
```

---

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
flow matching 動作頭：  x_t = (1-t)·noise + t·action ,  專家回歸速度場 v_θ(x_t, t, c)
```

**目標**：在有限的本機資料下，學一個能把「多模態觀測 + 任務語義」映射到「連續多執行器動作塊」的策略，而不引入離散化偽影（no VQ/discretisation）。

**核心公式（用純文字/代碼塊表達，非 LaTeX）**：

```
(1) 線性插值構成加噪輸入：
    x_t = (1 - t) * x_0 + t * x_1
    其中 x_0 ~ noise, x_1 = ground-truth action chunk, t ~ Beta(2, 5)

(2) 速度回歸目標（訓練損失，逐分量加權 MSE）：
    L = E_{t, x_0, x_1} [ Σ_k w_k · || v_θ(x_t, t, c)_k − (x_1 − x_0)_k ||² ]
    c = [C_vlm ; BEV_tokens ; h_state]   ← 條件上下文

(3) 推理（積分速度場生成動作）：
    x_1 = x_0 + ∫₀¹ v_θ(x_t, t, c) dt     ≈  midpoint ODE, 10 步, t 從 1 積到 0
```

**變量說明**：

| 符號 | 含義 |
|------|------|
| x_0 | 採樣的噪聲（起點） |
| x_1 | 真實動作塊（10 步 × 6 維） |
| t | 插值時間，Beta(2,5) 偏向前段採樣 |
| v_θ | 動作專家預測的速度場 |
| w_k | 六個動作維度各自的損失權重（平衡異質物理單位） |
| C_vlm | VLM 最後一層隱狀態（任務條件上下文） |
| B | PointPillars 產生的 BEV 特徵 token 序列（32 個） |
| h | 狀態歷史壓縮後的 token |

**直覺**：不要把動作「一次性回歸」出來，而是學一個「從噪聲把動作搬運過去」的速度場。積分路徑越平滑，越能表達多峰、多執行器耦合的連續控制，且天然避免「動作被離散化成 token」的量化誤差。**相較擴散策略，flow matching 用更少的採樣步（此處 10 步）達到相近品質**，對 25Hz 實時控制很關鍵。

> 符號與本文/相關文檔保持一致：本文符號系統沿用 π0 / flow matching 的約定（x_0 = 噪聲，x_1 = 資料）。

---

## 3. 帶數字走一遍：玩具例子 (Worked Example)

假設單一維度（比如縱向速度 vx），我們走一遍 flow matching 的閉環。

**訓練時**（假設 ground-truth 動作 = 2.0 m/s，噪聲 = −1.0 m/s）：

```
採樣 t = 0.3 （Beta(2,5) 傾向早期）
x_t = (1-0.3)·(-1.0) + 0.3·(2.0) = -0.7 + 0.6 = -0.1 m/s
目標速度  (x_1 - x_0) = 2.0 - (-1.0) = 3.0 m/s
若 v_θ(x_t, t, c) = 2.4  →  損失 = (2.4 - 3.0)² = 0.36
```

**推理時**（從純噪聲開始，積分 10 步）：

```
x_0 = -1.0
每步用 midpoint 估計速度，逐步把 x 從噪聲「推」向真實動作：
  step 1: v ≈ 3.0 → x ≈ -1.0 + 0.1·3.0 = -0.7
  ...
  step 10: x → 2.0 (收斂到動作)
```

**可計算的閉環**：只要速度場學對了，整條 ODE 路徑是確定性的、可微分、可預測的——這正是工程上要的「可重現、可調步數」。

再套一個**架構對比的具體推論**：作者報告在 loading 場景 vx RMSE 從 0.996 → 0.717 m/s。若平均動作約 2 m/s、控制頻率 25Hz，則單步時間約 40ms，0.7 m/s 的 RMSE 對應每步位置誤差量級約 0.7 × 0.04 ≈ 2.8cm——這與報告的位移誤差（dx RMSE 8.4cm @ 2s horizon 累積）量級自洽。

---

## 4. 工程視角 (Engineering View)

這一節是本文對 Ken 這類工程導向讀者最有價值的地方。

| 工程維度 | 設計選擇 | 含義 |
|----------|----------|------|
| 控制頻率 | 動作塊 10 步、stride 5、@25Hz → 2s horizon | 每 0.2s 重規劃一次（stride 5 = 5×40ms），滑動執行保證閉環 |
| 推理延遲 | 凍結 PointPillars（比 BEVFormer 等更輕）、VLM 解碼器截斷到 14 層、LoRA 低秩 | 三個動作都為**實時性**服務；PointPillars 選型明確理由是「comparable performance at much lower runtime」 |
| 記憶體 | VLM 主幹凍結 + LoRA + 截斷解碼器；BEV 走獨立通道不佔 VLM token 預算 | 在車載/嵌入式算力下可行；也縮小了「語言解碼器 vs 動作專家」的優化落差 |
| 動作離散化 | flow matching 連續生成，10 步 ODE | 無 VQ 量化誤差、無離散 token 的抖動；代價是推理需迭代 10 步 |
| 多執行器耦合 | 6 維動作（vx、轉向、body-frame dx/dy、arm rate、bucket rate）一起出 | 底盤與液壓臂統一決策，避免模組化管線的級聯誤差 |
| 感測冗餘 | 前+後雙向 LiDAR/RGB | 裝載機在堆料與卸料間頻繁倒車，前視單向無法覆蓋 hauling 時的後方危險 |
| 資料同步 | 各感測器異步儲存，訓練時對齊控制時間戳；丟棄缺包/標定不完整的樣本 | 真實機具資料的工程現實：清洗/對齊是隱形成本 |

**核心 trade-off 一句話**：用「凍結骨幹 + 輕量前端的 PEFT」換取資料效率，用「獨立 BEV cross-attn 通道」換取空間解析度與容錯，代價是必須接受一個凍結的 in-house BEV 編碼器（無法端到端優化幾何表徵）。

---

## 5. 數據與評測 (Data & Eval)

**資料組成**：私有真實資料集，來自生產用輪式裝載機（proprietary）。200K 訓練動作塊 / 40K 驗證動作塊。每樣本含前後 RGB、LiDAR（0.5s 累積）、IMU、短文字任務 prompt、本體狀態、動作目標（10 步 chunk）。

**標註類型**：模仿學習（imitation learning）——動作目標來自人類操作記錄，監督為 10 步動作塊，條件含當前觀測 + 10 步狀態歷史。

**場景配比**：兩類——(a) 從料堆裝載物料（loading）、(b) 地下隧道行駛（hauling/driving）。

**評測設置**：
- **切分方式**：按**錄製序列**而非按幀切分，防止時序近鄰洩漏到驗證集（這是本文資料處理上的一個好細節）。
- **基線**：同一架構 + 同一訓練配置，但**移除 BEV 通路**，動作專家只條件於前後 RGB + 文字 + 本體感知（camera-only baseline）。
- **指標**：per-step 物理 RMSE（vx、dx、dy、轉向等六維），以及 2s horizon 上的累積位移誤差；另有 z-normalised 動作空間的聚合 RMSE 用於魯棒性對比。

**主要結果（來自論文 Table I）**：

| 場景 | 指標 | Baseline | sensVLA | 變化 |
|------|------|----------|---------|------|
| 聚合 (40K chunks) | vx RMSE | 1.134 m/s | 0.886 m/s | −21.9% |
| 聚合 | dx RMSE | 9.52 cm | 8.98 cm | −5.7% |
| 聚合 | 轉向 / dy | — | — | 統計上無差異 |
| Loading 組 | vx RMSE | 0.996 m/s | 0.717 m/s | **−28.0%** |
| Loading 組 | dx RMSE | 9.27 cm | 8.40 cm | −9.4% |
| Driving 組 | vx RMSE | 1.281 m/s | 1.054 m/s | −17.7% |
| Driving 組 | dx RMSE | — | — | 基本持平 |
| Loading, 2s rollout | 末端位移誤差 | 0.538 m | 0.521 m | −3.1% |

**魯棒性（來自論文 Table II，z-normalised 聚合 RMSE，相對各自無擾動 anchor 的倍率，越低越好）**：

| 擾動 | Baseline 退化 | sensVLA 退化 | 備註 |
|------|---------------|-------------|------|
| img-blur（輕度光度擾動） | ~1.00× | ~1.00× | 兩者均無影響 |
| **no-cam（移除攝影機）** | **2.34×** | **1.67×** | vx RMSE：baseline 1.13→9.05 m/s，sensVLA 0.89→2.47 m/s（崩潰幅度小 3.7×） |
| no-bev（移除 BEV，健全性檢查） | — | 1.18× | 證明 BEV 被主動使用但非唯一承載模態 |

**關鍵解讀**：增益集中在 loading 場景——正是「料堆面幾何、鏟斗進場走廊」等度量線索能消歧、而純 RGB 難以捕捉的地方。

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 在**結構化重複**的裝載/行駛循環中輸出連續多執行器動作（底盤 + 液壓臂統一決策）。
- 在**攝影機退化或移除**時，仍保持度量一致的運動預測（BEV 提供冗餘路徑）。
- 用有限本機資料（LoRA + 分階段訓練）適配 2B VLM 做控制。

**不能做 / 未驗證**：
- **單一機型泛化未驗證**：只在輪式裝載機上測過；論文未聲稱可遷移到挖掘機、雙臂或人形。
- **場景局限**：僅兩類場景（loading + 地下隧道 driving）；開放式堆場、極端天氣、密集人車混流未測。
- **相機完全失效是模擬的**（simulated camera loss），非真實硬體拔除；且 baseline 的 「collapse」 數字來自該模擬設置。
- **BEV 編碼器凍結**：幾何表徵無法端到端優化，若 in-house PointPillars 本身有偏，架構無法補救（作者列為 future work）。
- **私有資料不可復現**：無公開 benchmark（LIBERO 等），外部無法直接對比，只能對比自身 camera-only baseline。

### 6.1 隱含假設 (Hidden Assumptions)

論文未明說、但推論成立的前提：

1. **LiDAR 在真實部署中始終可用的幾何退化不嚴重**——但地下隧道粉塵、雨天點雲噪聲等未討論。
2. **私有資料集足以訓練 2B VLM 的 LoRA**——200K chunk 相對 VLM 規模不大，作者用 PEFT + dropout 緩解，但「資料效率」的邊界未給出。
3. **「聚合 parity + loading 增益」值得**——作者誠實承認聚合層面只是 parity，主要價值在切片與魯棒性，讀者需自行判斷這是否符合任務需求。
4. **超參數描述自洽**——實測文中存在不一致：§IV-C 說動作專家是「4-layer, 8-head」，§V-A 卻說「5 transformer layers (1 bidirectional + 4 causal)」；狀態歷史 §IV-B 說 mean-pool 成**單一** token，§V-A 卻說壓成 **8 個 Perceiver token**。這類矛盾需向作者確認或看官方代碼，引用時應標注。

---

## 7. 與相關工作對比 (Comparison)

| 方法 | 關注點 | 架構 | 訓練方式 | 適用場景 |
|------|--------|------|----------|----------|
| RT-2 / OpenVLA | 通用 VLA，離散動作 token | VLM 直接出 token | 全量/指令微調 | 桌面操作、跨任務 |
| π0 | 連續動作生成 | VLM + flow-matching 動作頭 | 大規模預訓練 | 通用機器人控制 |
| Diffusion Policy | 多峰動作分佈 | 擴散解碼 | 模仿學習 | 精細操作 |
| 模組化裝載機系統 (Näslund 等) | 工程機械自動化 | 手工分模組管線 | 規則/控制論 | 地下裝載 |
| **sensVLA (本文)** | **幾何接地接在哪一層** | **凍結 VLM + 獨立 BEV cross-attn + flow-matching 專家** | **分階段 PEFT + 模仿學習** | **重型工程機械（裝載機）** |

**差異化定位**：sensVLA 的新意**不在動作生成機制**（flow matching 已是 π0 的路線），而在 **「模態路由的拓撲」**——把幾何感知從語言解碼器裡解耦出來、直連決策層。這是對「多模態 VLA 該在哪一層融合」這個問題的一次具體、可測的架構回答。

**🎯 面試 Tip**：被問到「VLA 裡多模態該怎麼融合」時，除了說「早融合/晚融合」，可以補一句：**sensVLA 的答案是「按模態性質分工——語義走語言骨幹，度量幾何走獨立通道接到動作專家，並用 gated residual 讓它在訓練中自己決定要不要放大」**，同時點出這帶來魯棒性紅利（單模態失效時仍有 fallback）。

---

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做**戶外/重型/多執行器**機器人（裝載、挖掘、礦卡、農業機械）的研究者與工程師。
  2. 想評估「BEV 幾何 vs 純 RGB」在實際控制中的增益與容錯價值的多模態具身 Agent 研究者。
  3. 需要在**車載算力受限**下設計 PEFT + 實時 VLA 管線的工程師。
- **建議章節路徑**：先讀 §IV-B（BEV 編碼與路由動機）→ 再看 §IV-C（異質 cross-attention + flow-matching 頭）→ 然後 §V-B 兩張表（Table I 增益切片 / Table II 魯棒性）→ 可跳 §III 資料管線細節（除非你要復現清洗流程）。
- **不值得精讀的理由**：如果你不做機器人學習、或已熟悉 π0 式 flow-matching 動作頭且不關心感測冗餘，讀摘要 + 本文明確的「聚合 parity、loading 增益」即可，其餘是工程機械特化的細節。

---
[← Back to Theory](./README.md)

**關鍵引用**：
- 論文: https://arxiv.org/abs/2609.17021 （ICRA 2026 Workshop, arXiv:2609.17021v1, 15 Sep 2026）
- 基礎工作: π0 (arXiv:2410.24164)、OpenVLA (CoRL 2024)、PointPillars (CVPR 2019)、Flow Matching (arXiv:2210.02747)、LoRA (ICLR 2022)、Qwen3-VL (arXiv:2511.21631)
