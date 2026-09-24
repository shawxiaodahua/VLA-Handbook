# 學習「何時」與「由哪個」RL 專家接管 VLA 策略 (RouteRLT: Learning When and Which RL Specialist Should Control a Vision-Language-Action Policy)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-09-24
>
> **論文**: RouteRLT: Learning When and Which RL Specialist Should Control a Vision-Language-Action Policy
> **鏈接**: https://arxiv.org/abs/2609.26467
> **核心定位**: 把「通用 VLA 何時交出控制權給哪個 RL 專家」從一個需要人工/特權信號的設計決策，變成一個由凍結 VLA 自身潛在表徵（latent）即時預測的學習問題——不改動骨幹、不需要部署期的階段標籤，就能在精密階段局部注入 RL 精度。

---

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 用凍結 VLA 的內部 latent 訓練一個輕量 router，逐控制步預測「該由 base VLA 還是某個 RL 專家控制」，在 LIBERO 上把全任務成功率推到 92.22%，在真實 Trossen 電纜插孔上把最終插入成功率從 6.7% 拉到 35.0% |
| 適合精讀 | 如果你在做「通用策略 + 局部精修」的系統（contact-rich 精密階段、工控插裝、電纜整理），重點看 §III-C（router）與 §III-D（chunk 邊界執行） |
| 可以跳過 | 如果你只關心「端到端全模型微調」或大規模多任務泛化，這篇距離中等——它刻意只做模組化路由 |
| 落地可行性 | 中高：不碰骨幹、專家輕量、可複用 RLT 的 latent 讀出，但依賴高質量階段標註來訓練 router，且需要工程化的 chunk 邊界管理 |
| 主要風險 | 路由資料在「特權邊界」下採集、部署時走學習路由，存在分佈漂移；硬體只測一台機器人、一個接頭與孔型 |

💡 **X-Ray 開場**
一個通用 VLA 能把連接器搬到插孔邊，卻常在最後「對準並插進去」的幾百毫秒內失手——失敗高度集中在少數幾個精密階段。RouteRLT 問的是：既然只需要在某幾個階段精修，為什麼要整條軌跡都微調？它訓練幾個「階段專家」RL 小策略，再用一個只讀 VLA 內部 latent 的輕量分類器，即時決定「現在該誰上」，並在動作分塊（chunk）切換的瞬間清掉過期動作。對 VLA 研究者的意義：**控制權分配本身可以是一個可學習、可在部署期獨立運行的模組，而不必綁定在人類介入或特權狀態上。**

📍 **研究全景時間線**

```
[2017] DAgger/SafeDAgger         [2021] LazyDAgger          [2025] HIL-SERL / Sirius
人工示範介入、動作差異觸發        非對稱進出閾值抑制抖動      操作員介入觸發，特權信號
        │                              │                         │
        └──────────────┬───────────────┴─────────────┬───────────┘
                       ▼                             ▼
             [2025] RLT: RL Token          [2025] 潛在空間 RL / 價值引導
             凍結 VLA → latent 讀出 → 輕量      (Latent RL, Value Guidance)
             actor-critic，但控制範圍固定      只改「更新什麼」，沒解「何時切換」
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                     [2026-09] RouteRLT（本文）← 當前位置
                     從 VLA latent 逐查詢預測「誰控制」，多專家、chunk 中即時生效
                     局限：SmolVLA 單一骨幹、單機單孔型、router 與專家解耦訓練
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

RouteRLT 的骨架是一個 **frozen generalist + specialist bank + learned router** 的三件套。通用策略（SmolVLA）凍結不動，RL 專家各自只負責一個精密階段，router 決定每一刻的「控制權歸屬」。

### 1.1 系統對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 時序/頻率 | 訓練 vs 部署 |
|------|------|------|-----------|--------------|
| Frozen VLA（π_VLA） | 多視角圖像 I_t、語言 ℓ、本體覺 q_t | H 步動作提案 ã_t^VLA（取前 C_VLA 步排隊） | 每次被查詢時推理 | 部署期凍結；僅前期在示範上微調 |
| Representation Encoder（RLT latent） | 圖像–語言 prefix + 附加 token e_rl | 緊湊狀態 z_t ∈ R^{d_z} | 每次控制查詢 | 表徵學習階段訓練一次後凍結 |
| Phase Selector f_φ（router） | 因果窗口 Z_t（W 個 z_t） | 控制器後驗 p_t（二分類 sigmoid 或多類 softmax） | 每 Δ_r 控制步查詢一次 | 在標註路由資料上訓練，早停於 val AUPRC |
| Router Stabilizer | p_t 序列 | 穩定後的控制權 c_t（PRE/ACTIVE/POST 狀態機） | 與 selector 同步 | 網格搜索 η，選於驗證準則 |
| Action-Boundary Manager | c_t 變化事件、控制器輸出 chunk | 只送當前單步動作 a_t 給機器人 | 每次控制步 | 規則式，無學習 |
| RL Specialists {π_k} | z_t、q_t、VLA 參考 ã_t | C 步動作 chunk â_t^k | 被選中時 | offline-to-online actor–critic，VLA 全程凍結 |

**關鍵設計取向**：router 只吃 z_t，**不碰原始觀測、獎勵或任何特權狀態**；本體覺與 VLA 參考動作只餵給專家，不餵給 router。這讓「誰控制」的判斷成為一個純 latent 分類問題，天然可在部署期獨立運行。

### 1.2 關鍵機制 (Key Mechanism)

- **為何要路由，而不是全軌跡 RL 精修**：失敗在時間軸上分佈極不均勻，少數階段決定成敗。把 RL 容量全押在這些階段，既省算力，又避免覆寫 VLA 已被驗證的長程行為。
- **為何用 latent 而非像素/狀態**：z_t 由 RLT 式的「附加 token 讀出 + 平行解碼器重建」得到，是凍結 VLA 自己對當前情境的壓縮理解；分類器因此不需要額外觀測編碼器。
- **為何需要 stabilizer**：逐幀分類 p_t 會在階段邊界附近抖動，直接切換會造成高頻抖振。這裡沿用 LazyDAgger 的非對稱進出閾值思想：進（entry）與出（exit）用不同門檻與滯留計數。
- **為何需要 action-boundary manager**：VLA 與專家都輸出 chunk，切換若發生在 chunk 中間，殘留的舊 chunk 動作會與新控制者衝突。做法是**切換瞬間作廢未執行的 chunk 後綴**，立刻向新控制者查詢。

⚡ **Eureka Moment**：把「通用策略何時該讓位給專家」從一個依賴人工/特權信號的外生決策，改寫成**從凍結 VLA 自身 latent 中逐查詢預測的內生分類問題**——控制權本身成了一個可以學、且成本極低的輕量模組。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
        觀測 I_t / 語言 ℓ / 本體覺 q_t
                    │
                    ▼
        ┌───────────────────────┐
        │  Frozen VLA (SmolVLA)  │──────► ã_t^VLA (H 步提案)
        │  + RLT latent readout  │
        └───────────┬───────────┘
                    │ z_t (緊湊狀態)
                    ▼
        ┌───────────────────────┐        ┌──────────────────────┐
        │ Phase Selector f_φ     │  p_t   │ Router Stabilizer    │  c_t
        │ 窗口 Z_t=[z_{t-Δ},z_t] │───────►│ PRE/ACTIVE/POST 狀態機│──────►
        └───────────────────────┘        └──────────────────────┘
                                                      │ (控制權 c_t)
                                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │ Controller Bank  𝒞 = { π_VLA } ∪ { π_1 … π_M }            │
        │   c_t = π_VLA → 排隊 VLA 前 C_VLA 步動作                   │
        │   c_t = π_k   → 以 (z_t, q_t, ã_t) 採樣 C 步 â_t^k         │
        └───────────────────────────┬──────────────────────────────┘
                                    │ 生成 chunk
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │ Action-Boundary Manager                                   │
        │   c_t ≠ c_{t-1} → 清空佇列 𝒬（作廢過期後綴），立即查詢新控制者 │
        │   每步僅執行 a_t = popfront(𝒬)                            │
        └───────────────────────────┬──────────────────────────────┘
                                    ▼
                              機器人執行單步動作
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
路由的本質 = 用凍結 VLA 的 latent 學一個「誰控制」的分類器
p_t = P( specialist 應接管 | 窗口 Z_t ) = σ( f_φ(Z_t) )
```

**目標**：在部署期不存取階段標籤、不讀特權狀態的前提下，用學習到的路由恢復「特權階段邊界路由」的表現。

**① 專家策略**（每個階段專家 k 的動作分佈）：

```
â_t^k ~ π_k( · | z_t, q_t, ã_t )
```
- z_t：凍結 VLA 的緊湊狀態（共享感知輸入）
- q_t = [q_t^joint, q_t^gripper]：本體覺
- ã_t：base VLA 產生的參考 chunk（前 C 步）
- â_t^k：專家 k 生成的 C × d_a 動作 chunk

**② 專家 actor 損失**（回報 − 參考錨定 + 時間平滑）：

```
L_actor = -λ_Q(n)·Q1(s_t, â_t^k)
          + (λ_ref(n) / (C·d_a)) · ||â_t^k - ã_t||²
          + λ_Δ · L_Δ(â_t^k),        其中 s_t = (z_t, q_t)

L_Δ = [ (C-1)·d_a ]^{-1} · Σ_{j=1}^{C-1} ||â_{t,j+1}^k - â_{t,j}^k||²
```
- λ_Q(n)：回報權重，隨訓練步 n 增大
- λ_ref(n)：參考錨定權重，隨 n 減小（早期貼近 VLA，critic 可信後才允許偏離）
- λ_Δ：相鄰動作差的平滑強度
- **reference dropout**：訓練時以機率 p_drop 把 ã_t 歸零，但偏差懲罰永遠對真實 ã_t 計算——強迫專家學出獨立的 (z_t, q_t) → â_t^k 通路，而不是退化成「VLA 動作 + 擾動」。

**③ Phase Selector（窗口分類）**：

```
Z_t = [ z_{t-(W-1)Δ} ; … ; z_{t-Δ} ; z_t ]           ← 因果窗口，最舊在前
p_t = σ( f_φ(Z_t) ) = P( 專家應持控制權 | Z_t )

內部：每個 latent 經共享 Linear(d_z → 64) + ReLU，權重跨 W 個位置共享；
      拼接後映射到單一 logit。訓練用類別平衡 BCE，正類權重 (N-N_+)/N_+。
多專家版：輸出 |𝒞| 路 softmax，[p_t]_c = P( 控制者 = c | Z_t )。
```

**④ Stabilizer（三段狀態機）**：

```
ψ_t ∈ { PRE, ACTIVE, POST }，PRE/POST 選 π_VLA，ACTIVE 選專家。

n_t^+ = 連續滿足 p ≥ τ_enter 的步數
n_t^- = 連續滿足 p <  τ_exit  的步數
m_t   = 距最近一次進入的步數

進入守衛 E_t^+：ψ_{t-1} ≠ ACTIVE 且 n_t^+ ≥ d_enter
離開守衛 E_t^-：ψ_{t-1} = ACTIVE 且 n_t^- ≥ d_exit 且 m_t ≥ d_occ

              ┌ ACTIVE,  若 E_t^+
ψ_t = ────────┤ POST,    若 E_t^-
              └ ψ_{t-1},  其他情況

η = (τ_enter, τ_exit, d_enter, d_exit, d_occ)
```
- 注意 **d_occ 是非對稱的**：只約束「離開」，絕不延遲「進入」。
- 計數器數的是「連續確認的決策次數」，不是實際經過時間。

**⑤ 路由決策的時序解析度**：每次控制查詢（t mod Δ_r = 0）重新預測 p_t；若 c_t 改變，立刻作廢未執行的 chunk 後綴——切換在「下一個動作」生效，而不是等下一個 replan 邊界。

> 符號與本文/相關文檔保持一致：π_VLA = base 通用策略；π_k = 第 k 個階段專家；z_t = RLT latent；ã_t = VLA 參考 chunk；â_t^k = 專家 chunk；c_t = 當前控制權；Z_t = latent 因果窗口；C_VLA / C = VLA 與專家的 chunk 長度；Δ_r = router 查詢間隔。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

用一個「單一精密階段」的二分類玩具設定走一遍端到端，數值為合理假設：

```
設定：W = 2，Δ = 10 步（0.5 s @20 Hz），Δ_r = 1（每步查詢）
門檻：τ_enter = 0.6, τ_exit = 0.3, d_enter = 2, d_exit = 3, d_occ = 5
```

| 步 t | p_t | n_t^+ | n_t^- | m_t | ψ_t | c_t |
|------|-----|-------|-------|-----|-----|-----|
| 40 | 0.12 | 0 | 4 | — | PRE | π_VLA |
| 41 | 0.55 | 1 | 4 | — | PRE | π_VLA |
| 42 | 0.72 | 2 | 0 | — | **ACTIVE**（E⁺ 成立）| π_1 |
| 43 | 0.80 | 3 | 0 | 1 | ACTIVE | π_1 |
| 44 | 0.65 | 4 | 0 | 2 | ACTIVE | π_1 |
| 45 | 0.85 | 5 | 0 | 3 | ACTIVE | π_1 |
| 46 | 0.58 | 6 | 0 | 4 | ACTIVE | π_1 |
| 47 | 0.22 | 0 | 1 | 5 | ACTIVE（m<d_occ 不允許出）| π_1 |
| 48 | 0.18 | 0 | 2 | 6 | ACTIVE | π_1 |
| 49 | 0.10 | 0 | 3 | 7 | **POST**（E⁻ 成立）| π_VLA |

**一次切換的代價核算**：t=42 進入專家時，佇列 𝒬 被清空，重新查詢 π_1 得 C 步 chunk；t=49 退回 VLA 時再清一次，取 VLA 前 C_VLA 步。整個階段切換只發生 **2 次**（進、出），與論文觀測到的「2.12 switches/episode 接近理想兩次切換」一致。

**滯留計數的意義**：若 t=43 的 p_t 忽然掉到 0.1，n_t^+ 歸零、但 ψ 不會立刻掉出 ACTIVE——它需要 n_t^- ≥ d_exit 且 m_t ≥ d_occ。這正是「抑制瞬時抖振」的機制：單幀噪聲不足以改變控制權。

**若把 W 設為 1**（只看當前 latent）：階段邊界的 p_t 方差上升，需要的 d_enter/d_exit 更大才能壓住抖動，代價是切換延遲變長——這解釋了為何論文固定用 W=2 並跨位置共享投影權重。

## 4. 工程視角 (Engineering View)

| 議題 | 設計取捨 | 工程含義 |
|------|----------|----------|
| 控制頻率 vs 查詢頻率 | selector 每 Δ_r 步查詢，獨立於控制器執行 horizon | LIBERO 用 Δ_r=1（每步查詢，20 Hz → 50 ms）；硬體放寬到 Δ_r=5 以省算力。router 極輕（Linear→64 + ReLU），不是瓶頸 |
| Chunk 長度對齊 | C_VLA=10、C=5（模擬）；C_VLA=35、C=10（硬體） | 硬體上放大 VLA horizon 是為了在較低控制率下，維持與模擬相同的「牆鐘時間 replan 週期」 |
| 切換代價 | 切換即清空 chunk 後綴並重查 | 每次切換會多一次 VLA 前向（給專家取參考 ã_t）；頻繁抖動=推理開銷與軌跡不穩加劇，因此 stabilizer 是必要的，不只是美觀 |
| 記憶體/算力 | VLA 凍結、專家輕量、表徵編碼器訓練一次 | 可複用 RLT 的 latent 讀出；線上專家微調用 update-to-data ratio = 5，維持在中等算力預算內 |
| 部署期觀測需求 | router 不吃原始觀測/特權狀態 | 部署只需 z_t（由凍結 VLA 順帶產生），無需額外傳感器或人工對齊信號 |
| 離線到線上 | 專家先 offline 擬合、再混 replay buffer 線上微調 | 降低真實機器人上的探索風險；warm-up rollout 先執行 VLA 參考再交棒 |

**工程結論**：這是一個典型的「模組化精修」部署樣式——骨幹不動（可沿用既有 VLA 推理棧），專家可獨立迭代，router 是一個可隨時重訓的小分類器。主要工程債在 **chunk 邊界的一致性管理**，以及 **路由資料的採集成本**（需要階段標註 / 特權邊界來產生控制器標籤）。

## 5. 數據與評測 (Data & Eval)

**模擬設定（LIBERO Object）**
- 任務：multitask pick-and-place——tomato sauce / butter / chocolate pudding，共 3 個任務
- 評測：60 條 held-out episodes，任務順序、初始條件與 300 步 horizon 對所有方法一致，20 Hz
- 控制器 horizon：C_VLA = 10、C = 5；router Δ_r = 1；窗口 W = 2（當前 z_t + 0.5 s 前的 state）
- 每專家 200 條 online-training episodes；λ_Δ = 0
- 指標：full-task success（主）、pickup success、每 1000 環境步的成功數（互動效率）
- 統計：3 個獨立訓練 seed 的均值 ± 樣本標準差

**硬體設定（Trossen Stationary AI，衍生自 ALOHA）**
- 任務：電纜抓取 + 埠插入，含兩個精密階段；插孔間隙 **1 mm**（3D 列印埠）
- 分工：π_1 控抓取、π_2 控對準與坐入，SmolVLA 控接近/搬運/退出
- 參數：Δ_r = 5；λ_Δ = 25（π_1）、0（π_2）；全程 C_VLA = 35、C = 10
- 協議：**operator-aligned handoff**——插入執行在操作員對齊後開始，但專家的選擇仍由系統自動完成
- 樣本量：SmolVLA 與 Oracle-routed RLT 各 30 trials；RouteRLT 用同一組 20 trials（鎖定插入退出協議）報告兩個指標；置信區間為 Wilson 95%

**主要數字（來源：論文 Table II / Fig. 6 / Fig. 7）**

| 指標 | SmolVLA | Oracle-routed RLT | RouteRLT |
|------|---------|-------------------|----------|
| LIBERO full-task success | 83.33%（受控對比）；貢獻段記為 85.00% 基線 | 91.11% | **92.22%** |
| LIBERO 互動效率 | 基準 | — | 每 1000 步成功數 +14.3%（vs SmolVLA@10） |
| 硬體 pickup-to-fixture | 53.3% | 80.0% | 65.0% |
| 硬體 final insertion | 6.7% | 33.3% | **35.0%** |
| 孤立 pickup（標準化入口狀態） | 0.0%（SmolVLA@10） | — | 83.3%（π_1） |
| 孤立 insertion | 8.0% | — | 62.1%（π_2） |

> 注：貢獻段（§I）稱「全任務成功率 85.00% → 92.22%」，而 §IV-A 的受控對比寫「83.33% → 91.11%」；兩者數字不一致，疑似基線取法（是否含 oracle self-route 行）不同。使用時以 Table II 的受控對比為準，並標註此差異。

**路由品質分析（LIBERO，180 episodes 合計）**
- entry recall = 1.0，進入延遲中位數 = 0 步——每個目標 pick-up 區間都被覆蓋到
- 平均 2.12 次控制權切換/episode（接近理想 2 次）
- 49/180 episodes 在特權邊界前 1–2 步提前進入，其中 47 條仍成功——「提前進入」本身不是失敗
- 僅 8 條含真正偽進入/重入，其中 5 條仍完成任務
- 14 次失敗中 10 次源於「pickup 從未完成」，導致兩種 router 都觀察不到 exit

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做什麼**
- 在**精密接觸階段**（抓取閉合、最終對準插入）注入 RL 精度，同時保留通用 VLA 在長程搬運/接近上的能力
- 僅靠凍結 VLA 的 latent 完成自動路由，部署期不需要階段標籤或特權狀態
- 支持**多個**精密階段（硬體上兩個專家自動接力），並能從失敗嘗試中恢復（pickup 未成功時可再次激活 π_1，而非強行進入插入）

**不能做什麼（場景 + 原因）**
- **強骨幹替換下的增益未知**：只用 SmolVLA 骨幹，未與「全骨幹線上微調」比較；更強 VLA 或全模型自適應可能改變特化收益
- **大規模多專家 / 邊界模糊場景未測**：只驗證了 1 個（模擬）與 2 個（硬體）專家；控制器庫更大、階段邊界更模糊時路由難度未知
- **硬體泛化有限**：僅一台機器人、一個接頭、一種埠型；抓取姿態不受控，是插入方差的來源之一
- **非全自主組合**：硬體採用 operator-aligned handoff（插入前需操作員對齊），並未證明完全自主的多專家組合
- **LIBERO 已近飽和**：該設定的 headroom 小，難以區分不同路由變體的精細差異

### 6.1 隱含假設 (Hidden Assumptions)

- **階段結構是已知且可標註的**：系統預設存在清晰的「精密階段」定義，router 的控制器標籤由特權邊界/標註產生。若階段本身模糊或隨任務變化，這套 pipeline 的前提就動搖了。
- **z_t 足以判別「當下是哪個階段」**：假設凍結 VLA 的緊湊 latent 已編碼足夠的階段資訊。消融顯示 image-only latent 仍能抓「何時 pickup」，但抓不住「哪個物體」——也就是說語言資訊對「對象身份」是必要的。
- **分佈一致性**：假設特權路由下採集的路由資料，與學習路由部署時遇到的狀態分佈足夠接近。論文自承存在漂移（spurious transitions 的部分原因）。
- **專家與 router 可解耦訓練**：假設分開訓練的 router 與 specialists 在端到端組合時不會互相拆台；實際上兩者都可能在 OOD 狀態下運行。
- **單步執行的動作一致性**：假設切換瞬間「作廢後綴」足以避免舊 chunk 殘留；依賴嚴格的佇列與評估驗證（每個 eval episode 被驗證不含被取代計畫的動作）。

## 7. 與相關工作對比 (Comparison)

| 方法 | 觸發信號 | 需要特權信號? | 交給誰 | 決策解析度 |
|------|----------|---------------|--------|------------|
| HIL-SERL | 操作員 | 是 | 人類 | 自由 |
| Sirius | 操作員 | 是 | 人類 | 自由 |
| **RLT** | 操作員 | 是 | RL 策略 | 階段（固定） |
| SafeDAgger / LazyDAgger | 動作差異 | 否 | 人類 | 步 |
| EnsembleDAgger | 集成方差 | 否 | 人類 | 步 |
| Sentinel | chunk 不一致 | 否 | （監控） | chunk |
| Recovery RL / ASC | 安全 critic / OOD 狀態 | 否 | 恢復/糾正策略 | 步 |
| AtomicVLA | skill identity | 否 | skill 專家 | 子任務 |
| **RouteRLT（本文）** | **VLA latent** | **否** | **RL 階段專家** | **查詢點（切換於下一步生效）** |

**與 RLT 的關係最直接**：RouteRLT 沿用 RLT 的 latent 讀出與 actor–critic 設計，但把「控制範圍由資料採集信號固定」升級為「由 latent 逐查詢可學」。與 HYDRA/AtomicVLA 等的差異在於：這裡協調的是「既有通用 VLA」與「分開訓練的階段專家」，而非聯合訓練的多專家系統。

🎯 **面試 Tip**：被問到「這篇和 RLT / HIL-SERL 有什麼不同」時，一句話答——**RLT 把精度綁在人工提供的階段邊界上，RouteRLT 把『何時切換』變成從凍結 VLA 自身 latent 學出來的分類問題，並用 chunk 邊界管理器讓切換在下一步立即生效、不需要特權狀態。**

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**
  1. 做「通用策略 + 局部精修」架構的研究者：§III-C（router 分類器）與 §III-D（chunk 邊界管理）是可直接借用的設計。
  2. 要評估「把精密階段特化遷移到新機器人/新裝配任務」可行性的工程師：§IV-B 與 §V 講清了硬體約束與失敗來源。
  3. 做 contact-rich 工控操作（插裝、電纜整理）的系統整合者：1 mm 間隙、53.3%→35.0% 的具體數字給出量級預期。

- **建議章節路徑**：先讀 §I 貢獻 + Fig. 1（把握三件套）→ 再讀 §III-C 與 §III-D（核心機制）→ 接著 §IV-A 的路由分析（entry recall、切換次數）→ §V 局限性。§II 相關工作可略讀（Table I 已濃縮）。

- **不值得精讀的理由**：如果你不做機器人學習、或已熟悉 RLT/latent-space RL 這條線，且只關心端到端大模型微調，讀摘要 + Table I 即可——本文的增量主要在「路由可學 + chunk 即時生效」這一塊。

---

[← Back to Theory](./README.md)

**關鍵引用**
- 論文：https://arxiv.org/abs/2609.26467 （IROS 2026 IARL Workshop）
- 前置工作 RLT：RL Token: Bootstrapping Online RL with Vision-Language-Action Models, arXiv:2604.23073
- 骨幹 SmolVLA：arXiv:2506.01844
- 真機平台：Trossen Stationary AI（衍生自 ALOHA）
