# Workspace Models：用顯著性監督把 VLM 記憶「攤銷」進一個輕量 token (Workspace Models: Lightweight Robotic Memory via Saliency-Driven Supervision)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-09-20
>
> **論文**: Workspace Models: Lightweight Robotic Memory via Saliency-Driven Supervision
> **連結**: https://arxiv.org/abs/2609.20820
> **發表**: CoRL 2026（26 pages, 11 figures），arXiv v1 於 2026-09-17 提交
> **作者**: Nitish Dashora, Douglas Chen, Idan Shenfeld, John Marangola, Pulkit Agrawal (MIT), Max Simchowitz (CMU)
> **官方代碼**: 未提供 repo_url（待補）
> **核心定位**: 把「測試時一直在線查 VLM 做記憶摘要」這件事，改成「訓練時用 VLM 標顯著性、蒸餾進一個輕量 latent token」，測試時只查那個 token——延遲大降，成功率反而更高。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用「訓練時 VLM 顯著性標註 + 集合重建蒸餾」訓出的 workspace token，可作為觀測的 drop-in 替代品；測試期不需 VLM |
| 適合精讀 | 如果你在做**長程操作策略**或**在線 VLM/LLM 推理拖慢控制迴路**的系統，重點看 §3（訓練目標）與 §5（為什麼 latent 比 frame selection 好） |
| 可以跳過 | 如果你只關心單幀反應式 VLA（如 OpenVLA/pi0 類）且任務無需歷史記憶，這篇距離中等 |
| 落地可行性 | **中—高**：架構是 DinoV3 + 小 transformer + Diffusion Policy，全是現成組件；但 label pipeline 依賴 Qwen3-VL + MolmoPoint 兩個 VLM，前期數據標註成本高 |
| 主要風險 | 顯著性由 VLM 判定，未必與「控制真正相關」對齊（作者自己承認）；且編碼器是全自回歸 transformer，序列長時計算二次增長 |

💡 **X-Ray 開場**（2-3 句，非專家也能讀懂）

長程操作機器人需要「記住過去」，但把整段歷史餵給策略會讓模型學到假相關而崩掉；流行做法是部署時一直在線問 VLM「哪些幀重要」。這篇反過來：**訓練時**用 VLM 標出重要畫面，逼一個小編碼器把「重要資訊」壓進一個 token，之後機器人只看這個 token 就能回憶。結果不只是更快（延遲 ~5×→~1.2×），成功率還更高（跨任務 91.5% vs 66.8%）。

📍 **研究全景時間線**

```
2019-2023  行為克隆擴散策略(DP)、RT-2/OpenVLA 單幀反應式策略
   │        歷史條件化被指出有 causal confusion / copycat 問題
   ▼
2025-2026  VLM-in-the-loop 記憶：選關鍵幀 (BPP/Mark 2026)、
   │        語言記憶 (MEM/Torne 2026)、經歷檢索 (MemER/Sridhar 2025)
   │        痛點：在線查詢延遲高、成本高、脆
   ▼
[本文 2026] Workspace Models ← 當前位置
   │        「把在線推理攤銷到訓練期」→ 離線蒸餾 latent memory token
   │        限制：全自回歸編碼器(O(T²))、VLM 監督非控制接地、未比語言層級方法
   ▼
下一步？   多相機、語言推理蒸餾、dynamics 預測、RL 接地
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

方法一句話：**訓練一個能壓縮歷史的 latent 記憶編碼器（workspace encoder），用 VLM 給的顯著性當監督訊號，再讓凍結的它去餵一個反應式 Diffusion Policy。**

### 1.1 系統對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 訓練/推理 | 備註 |
|------|------|------|-----------|------|
| Tokenizer | 觀測 o_t = [I_t, x_t] | 影像走 DinoV3 → 單層 cross-attention pooling 成 1 token；proprio 走 MLP | 凍結（DinoV3 全凍） | 加 learnable 位置嵌入 + 跨時間 sinusoidal 嵌入 |
| Workspace Encoder f_φ | 序列 [(ō_t′, z_t′)]，z 為 N(0,I) padding slot | workspace token w_1:t | 訓練 | 因果遮罩 transformer，2–3 層 |
| Workspace Decoder g | w_t + 可學 query tokens | m 個槽位 {(p̂_{t;j}, β_{t;j})} | 訓練（DETR 式） | m=8，Hungarian 配對 |
| VLM 標註管線 | 每一幀 RGB | 顯著集 S_t（Dino patch 索引集合） | **僅訓練期** | Qwen3-VL-8B 判事件 + MolmoPoint-8B 指點 |
| Policy π_θ | w_{t−C:t} + x_{t−C:t} | 動作 chunk a_{t:t+H} | 訓練（編碼器凍結） | Diffusion Policy（1D U-Net + FiLM），C=2 |

### 1.2 關鍵機制 (Key Mechanism)

- **為什麼是 latent 而不是選幀？** 選幀（frame selection）在訓練集裡產生離散的「尖銳」輸入分佈，測試時一有偏移（aliasing）就崩；latent token 隨時間平滑變化，天然抗偏移。
- **為什麼用集合重建（set reconstruction）？** 顯著資訊數量可變，用 DETR 式「固定槽位 + Hungarian 配對 + 存在機率」可以表達 0~m 個元素，不需要固定長度。
- **為什麼標「patch」而不是「整張圖」或「點」？** 消融顯示 patch 監督最佳（平均 94），整圖監督會在 CubeDrop 崩到 0%（混入無關資訊），純點監督有損（丢掉旋轉等抓取相關細節）。
- **為什麼離線標註反而更強？** 訓練期能看完整軌跡，用中位數選事件時間，比測試期只能看過去的 rising-edge 標註更抗噪、更一致。

⚡ **Eureka Moment**：**把「昂貴推理」從測試期挪到訓練期——用強模型當「顯著性老師」離線標籤，蒸餾進一個訓練後可零成本反覆查詢的 latent token。** 這是一個新的 scaling 軸（作者稱 saliency-driven supervision），且攤銷後的效果居然**優於**把強模型留在迴路裡。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
訓練期（Workspace 學習）
  π 幀影像 ──► DinoV3 patch 特徵 ──┐
                                  ├─► Workspace Encoder (causal xfmr) ─► w_t
  z ~ N(0,I) padding slot ────────┘            │
                                               ▼
                                        Workspace Decoder (m slots)
                                               │  Hungarian matching
                                               ▼
  Qwen3-VL(判事件) + MolmoPoint(指點) ──► 顯著集 S_t ──► 重建損失 L_feat / L_active
  （監督訊號，僅訓練期）

部署期（推理，零 VLM）
  影像 ──► DinoV3 ──► 凍結 Encoder ──► w_t ──┐
  proprio x_t ───────────────────────────────┼─► Diffusion Policy ─► 動作 chunk
  短歷史 w_{t−C:t}, x_{t−C:t} ───────────────┘
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
w_t = f_φ(o_{1:t})      # 一個因果編碼器，把歷史壓成一個記憶 token
L   = E_τ [ Σ_t ( λ1·L_feat + λ2·L_active ) ]   # 逼 token 記住「顯著集」的全部資訊
```

**目標**：讓 latent token w_t 承載「完成任務所需的少量過去資訊」（顯著集 S_t），使下游策略只看 w 就能決策。

**顯著集的定義**：

```
S_t = { p_{t;i} }_{i=1..m}      # 一組 Dino patch token
p_{t;i} = 位於位置 ℓ_i、取自時刻 t_i 的一塊 Dino patch
（patch 本身已含其位置 ℓ_i 的編碼；|S_t| 可小於上限 m）
```

**Hungarian 配對成本**（決定哪個槽位對應哪塊 patch）：

```
c_t(i,j) = y(i)·|| p_{t;i} − p̂_{t;j} ||²  +  λ0 · CrossEnt( β_{t;j}, y(i) )
其中 y(i) := I{ i ≠ 0 }，i=0 為「空 patch」位置（未佔用槽位映射到此）
```

**兩個損失**：

```
L_feat   = (1/|S_t|) · Σ_j  y(σ_t(j)) · || p̂_{t;j} − p_{t;σ_t(j)} ||²
L_active = (1/m)    · Σ_j  CrossEnt( β_{t;j}, y(σ_t(j)) )
CrossEnt(β; y) := I{y=1}·log(1/β) + I{y=0}·log(1/(1−β))
```

**下游策略目標**：

```
π_θ(a_{t:t+H} | w_{t−C:t}, x_{t−C:t})      # Diffusion Policy，編碼器 f_φ 凍結
```

| 符號 | 含義 |
|------|------|
| o_t = [I_t, x_t] | 時刻 t 的影像 + proprio 狀態 |
| w_t | workspace token（記住歷史的緊湊表徵） |
| S_t | VLM 判定的顯著 patch 集合（當前 + 歷史） |
| m | 槽位上限（本文 m=8） |
| p̂, β | 解碼器輸出的 patch 重建 / 槽位存在機率 |
| σ_t | Hungarian 配對（不回傳梯度） |
| λ0, λ1, λ2 | 配對成本 / feat 損失 / active 損失 權重 |
| C, H | 條件歷史長度 / 動作預測視窗 |

> 符號與原文一致：f_φ 為編碼器、g 為解碼器、β 為槽位存在機率、σ_t 為配對。原文 y(i) 為「i 是否為非空位置」的指示函數。

**直覺**：這是一個「以重建為代理的資訊瓶頸」——你不知道該記什麼，就讓一個見過全軌跡的 VLM 告訴你哪些 patch 重要，然後強迫單一 token 能重建出這一整組 patch。梯度不回傳經過 σ_t，避免配對的不可導問題（DETR 的標準做法）。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

**(A) 論文自帶的「為什麼需要記憶」插圖（§2）**

一維點目標導航：起點 x0 = 0，目標 g ∈ {−1, +1}，但在某個**顯著時刻** t_s ~ U(1, L) 只展示一次 g。要拿到 g，策略必須保有至少 L 步的歷史。

- 觀察（論文 Figure 2）：condition on **full history** 時，成功率隨 L 增大而**下降**，而且即使增加示範數量也**飽和**。
- 工程含義：機器人資料本就稀缺（本文任務僅 75–300 條軌跡），「照單全收餵全歷史」在資料量上根本不可行 → 必須做資訊選擇。

**(B) 走一遍重建損失（假設數值，僅示意可計算閉環）**

設 m = 3 個槽位，該時刻顯著集只有 2 塊 patch：p_1（t=50，鹽已加入）、p_2（t=150，刀已歸位）。解碼器輸出 3 個槽位：

| 槽位 j | β_{t;j}（存在機率） | 配對 σ_t(j) | 重建平方誤差 ||p̂−p||² |
|--------|---------------------|-------------|------------------------|
| 1 | 0.90 | p_1（y=1） | 0.04 |
| 2 | 0.70 | p_2（y=1） | 0.09 |
| 3 | 0.20 | 空 i=0（y=0） | — |

計算是：

```
L_feat   = (1/2)·(0.04 + 0.09) = 0.065
L_active = (1/3)·[ log(1/0.90) + log(1/0.70) + log(1/(1−0.20)) ]
         = (1/3)·[ 0.105 + 0.357 + 0.223 ] = 0.228
L_t      = λ1·0.065 + λ2·0.228
```

若取 DrawerRecall 的權重（Table 4: feat weight = 2.0, existence weight = 0.01）：

```
L_t ≈ 2.0×0.065 + 0.01×0.228 = 0.130 + 0.0023 ≈ 0.132
```

**可讀出的設計含義**：在多數任務裡 existence 權重被壓到 0.01（僅 CubeDrop 用 1.0），所以**梯度幾乎全由 patch 重建驅動**，存在機率只是個輕量開關。這也解釋了為什麼 latent 內容品質主要由「選哪些 patch 當監督」決定。

## 4. 工程視角 (Engineering View)

**延遲（Table 1，相對 VanillaDP 正規化）**：

| 方法 | 延遲 (ms) | 相對倍數 |
|------|-----------|----------|
| VanillaDP | 41.3 | 1.00× |
| HistoryDP | 41.3 | 1.00× |
| Keyframe | 302 | 7.31× |
| Keyframe+Batch | 199.5 | 4.83× |
| **Wksp** | 94.9 | 2.3× |
| **Wksp+Batch** | 52.9 | 1.28× |
| **Wksp+Batch+KV** | 49.2 | 1.19× |

- **工程含義**：加上 batching（chunk 間堆疊）與 KV-cache 後，Wksp 幾乎退回純 CNN 感知棧的成本（~1.2×），而 Keyframe 即便優化仍 ~5×。若 Keyframe 用 API 查詢會更糟。
- **控制迴路**：仿真 ManiSkill3 跑 100 Hz，策略與資料收集在 20 Hz（PD 控制，7D 動作含末端位姿 + 夾爪）。
- **真機**：Franka FR3 + AgileX 平行夾爪，operational space control 在 **1 kHz** 跑；teleop 資料 50 Hz（6D 旋轉 + 3D 位置 + 連續夾爪）。6D 旋轉預測先做 **Gram-Schmidt 正交化**才送進控制器。
- **模組邊界**：VLM（Qwen/MolmoPoint）只在離線標註階段出現，**不在部署路徑上**——這是整個設計的關鍵取捨：用可接受的離線標註成本，換取在線零 VLM。
- **計算瓶頸**：編碼器是全自回歸 transformer，**O(T²)**，長序列會吃緊（作者列為 limitation，建議改 recurrent / block-diagonal mask / KV-cache）。
- **模型規模很小**：編碼器 2–3 層 / 隱藏 512–768；解碼器 2 層；槽位 8；patch 數 49（HnH 為 196）；workspace token 數 = 1。這是一個「小而穩」的設計，不是大模型路線。

## 5. 數據與評測 (Data & Eval)

**四個記憶密集任務（3 仿真 + 1 真機）**：

| 任務 | 環境 | 記憶軸向 | 訓練示範 | 評測量 |
|------|------|----------|----------|--------|
| CubeDrop | 仿真 | 部分可觀測下**計數**：把正好 5 塊方塊放入碗（放入後消失），再按綠鈕 | 272 條 | N=100 |
| DrawerRecall | 仿真 | **空間回憶**：另一機器人把方塊藏進三抽屜之一後關閉，需開對抽屜 | 272 條 | N=100 |
| BalanceBar | 仿真 | **上下文自適應**：未知質心 (3 種 CoM)，需用歷史判斷並在第二次抓對位置 | 272 條 | N=100 |
| HalfAndHalf | 真機 | 部分可觀測下**計數**：把 N∈{2,4} 方塊均分到兩個看不進去的箱子 | 75 示範→300 軌跡 | N=20 |

**主要結果**：

- 跨任務平均成功率 **Wksp 91.5% ± 2.2** vs 次佳 Keyframe **66.8% ± 3.2**（Figure 6）——仿真相機與真機皆成立。
- **最反直覺的發現**：Wksp 不只延遲更低，**成功率也更高**（原本只預期贏在延遲）。
- 基線設定：VanillaDP（當前 + 前一幀）、HistoryDP（跨越歷史幀，且**未跨步的全歷史成功率為 0%**）；Keyframe 仿 BPP（Mark et al. 2026），用 VLM 選幀——所有基線都逐任務調參以「強化對手」。

**監督目標消融（Table 6，CD/DR/BB，平均）**：

| 監督目標 | CD | DR | BB | 平均 |
|----------|----|----|----|------|
| Point | 63 | 100 | 88 | 84 |
| **Patch** | **92** | **100** | **89** | **94** |
| Image | 0 | 100 | 91 | 64 |

→ Patch 監督最佳；整圖監督在 CubeDrop（最需精細感知）**崩到 0%**。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 在計數 / 空間回憶 / 上下文自適應三類記憶任務上、仿真與真機皆達高成功率。
- 以近似 CNN 的成本（~1.2×）提供長程記憶，**部署時不需任何 VLM**。
- 對「幀選擇時機噪聲」更魯棒（離線中位數標註 + 自注意力平滑）。

**不能做 / 弱點**：
- 論文明確限於**單臂桌面操作**（Franka FR3 / ManiSkill3），**未驗證移動、雙臂、人形**——不要外推。
- 全自回歸編碼器 **O(T²)**，極長序列吃力。
- 顯著性由 VLM 決定，**未必與控制真正相關對齊**（作者建議用 RL reward 接地）。
- **未與語言摘要類層級方法（MEM 等）比較**，也未測多任務通用策略。
- Table 2 的記憶/控制失敗率在 HTML 渲染下部分格值疑似折疊（如 Bar 行 0%/100%），**具體數字請以原文 PDF 為準**；可引用的定性結論是：frame-stacking 類方法（HistoryDP/Keyframe）記憶尚可但**控制失敗更多**（如抓偏、抓在中間、卡住），Wksp 較能兼顧記憶與控制。

### 6.1 隱含假設 (Hidden Assumptions)

1. **VLM 指出的「顯著 patch」= 控制所需資訊**——全篇最強的隱含前提，論文本質上把「資訊選擇」外包給 VLM 的常識。
2. **事件是時空局部、稀疏的**——用少量 patch 就能表達顯著集；若關鍵資訊是全局/稠密的（如整場景動力學），此表徵會失真。
3. **訓練分佈覆蓋測試分佈的顯著事件**——saliency 監督靠 ERM 平均，若測試出現訓練沒見過的顯著型態，latent 未必泛化。
4. **proprio 直接餵給策略（不進 workspace）**——即假設「當前狀態」不需要被壓縮/記憶化，只壓縮視覺歷史。
5. **中位數標註與測試一致性**——離線用全軌跡標註帶來平滑優勢，但也假設測試期的事件時序分佈與訓練相似。

## 7. 與相關工作對比 (Comparison)

| 方法 | 歷史表徵 | 測試期算力 | 機制 | 適用場景 |
|------|----------|-----------|------|----------|
| VanillaDP | 當前+前一幀 | 最低 | 無記憶 | 反應式、短視界 |
| HistoryDP | 跨步歷史幀 | 最低 | 樸素幀堆疊 | 簡單記憶；全歷史 → 0% |
| Keyframe (BPP/Mark 2026) | VLM 選關鍵幀 | 高（~5–7×） | 在線 VLM 選幀 | 記憶密集，但延遲/脆 |
| MEM (Torne 2026) | 語言記憶/摘要 | 高 | 在線 VLM 語言摘要 | 層級任務分解 |
| MemER (Sridhar 2025) | 經歷檢索 + 視覺 keyframe | 高 | 在線檢索 | 大規模記憶 |
| **Workspace（本文）** | **單一 latent token** | **低（~1.2×）** | **離線蒸餾顯著性** | 記憶密集且約束時延 |

概念類比：**Cartridges（離線蒸餾 KV-cache）之於推理**，正如 **Workspace token 之於機器人記憶**——都是把在線計算攤銷成訓練期產物。

🎯 **面試 Tip**：被問「為何不直接讓 VLM 在線選關鍵幀？」→ 答：**amortization**。把 VLM 推理移到訓練期做，測試期只查一個平滑 latent token，延遲從 ~5× 降到 ~1.2×；而且因為離線中位數標註更抗噪 + 自注意力跨時間平滑，成功率反而更高（91.5% vs 66.8%）——強模型留在迴路裡反而更差。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做**長程操作 / 記憶密集**策略的研究者（計數、空間回憶、in-context 適應）。
  2. 正在評估「在線 VLM/LLM 拖慢控制」的工程師——本文的攤銷範式可直接借鑑。
  3. 研究**表徵學習 / 資訊瓶頸 / 集合重建（DETR）**如何服務機器人策略的人。
- **建議章節路徑**：先讀 §3（方法與損失）→ 再看 §5（為何 latent 勝過選幀，含 aliasing 與平滑性論證）→ §4 + Table 1/6 看數字 → 可跳 Appendix A（延伸相關工作）與 Appendix C 的 prompt 細節（除非你要復現標註管線）。
- **不值得精讀的理由**：若你不做機器人學習、或已熟悉「VLM 監督 + latent memory」這類做法、或任務本身是單幀反應式（無需歷史），讀摘要 + Figure 6 即可掌握。

---
[← Back to Theory](./README.md)
