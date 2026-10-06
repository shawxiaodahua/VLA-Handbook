# 分階段操作：以行為相位解耦實現類人機器人操作 (Move-Then-Operate: Behavioral Phasing for Human-Like Robotic Manipulation)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-10-06
>
> **論文**: Move-Then-Operate: Behavioral Phasing for Human-Like Robotic Manipulation
> **鏈接**: https://arxiv.org/abs/2604.23620 (v3, 2026-10-02)
> **代碼**: https://github.com/healenrens/Move-then-Operate
> **核心定位**: 把「粗定位（move）」與「接觸關鍵交互（operate）」用雙專家 + 可學習相位路由器硬切開，在 RoboTwin2 上把單體 π0 的成功率從 ~44.8% 拉到 68.9%（+24.1% 絕對值），並做到「1/10 數據打平 10× 數據baseline」。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 顯式把操作拆成 move/operate 兩個相位、各自訓練一個 flow-matching 專家，能大幅降低長時程操作中的梯度干擾 |
| 適合精讀 | 如果你在做雙臂/長時程操作、接觸密集任務，或任何「低頻粗移 + 高頻精調」耦合的 VLA 系統，重點看 §3、§4.2、§5.3 |
| 可以跳過 | 如果你只關心單步短時程抓取、或單純的視覺泛化 scaling，這篇距離中等 |
| 落地可行性 | 中（架構簡單、推論只激活單專家；但依賴 MLLM 自動標相位，且論文證據集中在單一 sim benchmark）|
| 主要風險 | 路由錯判是災難性失敗：ablation 顯示隨機路由掉到 25.6%、反轉相位掉到 8.9% |

💡 **X-Ray 開場**
這篇解決一個很具體的痛點：機器人示範裡「手伸過去」的動作幅度大、幀數多，而「精細接觸調整」動作幅度小、幀數少，兩者混在同一個損失裡訓練時，小幅度操作信號會被大幅度移動淹沒，導致精調學不好。作者的做法很直接——把兩者硬拆成兩個參數獨立的專家（都建在 π0 上），用一個小的 MLP 路由器按動作塊（chunk）選一個專家。結果是在 RoboTwin2 上成功率大幅提升、且數據/訓練效率都更好。對 VLA 研究者的意義：**inductive bias（結構先驗）在數據稀缺時，可能比堆數據更值錢**。

📍 **研究全景時間線**

```
1899 Woodworth 兩組分動作模型 → 1954 Fitts 定律（速度-精度權衡）
   → 2020 BBN 長尾解耦（頭/尾分支分離）
   → 2023-24 Diffusion Policy / π0 flow-matching VLA（單體策略主流）
   → 2025 雙系統 & MoE VLA（π0.5, GO-1, MoE-DP, FedVLA, SP-VLA）
   → [本文 2026] Move-Then-Operate ← 當前位置（相位解耦，非 RL、非時間縮放）
   → 局限：單一 sim benchmark、依賴 MLLM 標籤、真實機器人驗證未見
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

整體是一個**層級門控策略（hierarchically gated policy）**：共享 VLM backbone 編碼上下文 → 輕量相位路由器推斷離散標籤 z ∈ {Move, Operate} → 只激活對應的單個 CFM 專家生成動作序列。

### 1.1 系統對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 時序粒度 | 訓練/推理差異 |
|------|------|------|----------|----------------|
| VLM Backbone (Φ_VLM) | 指令 I、觀測 O_t、本體狀態 S_t | 隱狀態 F_t ∈ R^{L×D} | 每控制步 | 共享；訓練/推理皆參與 |
| 相位路由器 (MLP_router) | 池化語義摘要 f_t | 相位分佈 p_φ(z\|C_t) | 每動作塊（chunk）| 訓練用 GT 標籤算 CE；推理用貪婪 |
| Move 專家 E_Move | F_t, flow state x_σ | 速度場 v | 每動作塊 | LoRA 參數獨立；僅在 y_t=Move 時更新梯度 |
| Operate 專家 E_Operate | F_t, flow state x_σ | 速度場 v | 每動作塊 | LoRA 參數獨立；僅在 y_t=Operate 時更新梯度 |
| MLLM 自動標註器 | 影片 V、指令 I | 結構化相位排程 S | 離線、每條軌跡一次 | 僅資料準備階段，推理時不使用 |

關鍵設計：**專家共享 base VLA 架構但參數不相交**；路由器決策在整個 flow 積分週期 σ ∈ [0,1] 內保持不變（chunk-level，而非 token-level MoE）。

### 1.2 關鍵機制 (Key Mechanism)

- **硬切換而非軟混合**：每個動作塊只由一個專家生成，避免兩套動力學被平均到「中間的錯誤行為」。
- **參數隔離 → 梯度正交**：用 GT 相位標籤做 teacher-forcing，讓 v_pred 只對匹配專家產生非零梯度（論文明說「orthogonalizing parameter updates」），從根上消除 move/operate 的梯度衝突。
- **路由器只看高層語義**：對 VLM 隱狀態做 masked global average pooling 得到整場景摘要，再過一個 MLP，避免依賴淺層嵌入。
- **對齊人類動作模式**：以 Woodworth 兩組分模型 / Fitts 定律為先驗，用 MLLM 依據末端速度、子任務分解等線索自動打相位標籤。

⚡ **Eureka Moment**：**「把長時程操作看成 move/operate 兩個異質動力學的混合，並用硬參數隔離代替單體聯合優化」**——就是這一刀，讓小幅度精調信號不再被大幅度移動淹沒。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
上下文 C_t = (I, O_t, S_t)
        │
        ▼
  Φ_VLM (共享 backbone) ──► F_t ∈ R^{L×D}
        │                        │
        │ (masked GAP)           │ (共享表徵)
        ▼                        │
    f_t ──► MLP_router ──► z_t   │
        (訓練: y_t 監督)         │
                                 ▼
                ┌──────────── 硬路由 z_t ────────────┐
                ▼                                    ▼
        E_Move (LoRA)                        E_Operate (LoRA)
        (大位移向量場)                        (小幅精調向量場)
                │                                    │
                └────────────► ODE 積分 ────────────┘
                                 │
                                 ▼
                      動作序列 a_t ∈ R^{H×d}
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
p(a_t | C_t) = p(a_t | C_t ; θ_{z_t})   ,  z_t ∈ {Move, Operate}
```

先給目標：把單體動作分佈拆成相位條件分佈；再給各模組公式；最後給直覺。

**（1）相位分解（硬切換策略）**

```
p(a_t | C_t) = p(a_t | C_t ; θ_{z_t})
z_t ∈ Z = {Move, Operate}
```

**（2）條件流匹配（CFM）專家損失**

```
x_σ = (1 - σ)·x_0 + σ·a_t ,   σ ∈ [0, 1]
L_FM(θ_z) = E_{σ, x0, a_t} [ || v_{θ_z}(σ, x_σ, C_t) - (a_t - x_0) ||_2^2 ]
```

**（3）推理：解 ODE 合成動作**

```
a_t = x_0 + ∫_0^1 v_{θ_{z_t}}(σ, x, C_t) dσ
```

**（4）掩碼速度場（訓練時用 GT 標籤 y_t）**

```
v_pred(σ, x_σ, C_t) = Σ_{z ∈ Z} 1[z = y_t] · v_{θ_z}(σ, x_σ, C_t)
```

**（5）動作損失（批內掩碼 MSE）**

```
u_t = a_t - x_0
L_action = E_{σ, x0} [ || M_t ⊙ (v_pred - u_t) ||_2^2 / (|| M_t ||_1 + ε) ]
```

**（6）路由器損失與總目標**

```
f_t = masked-GAP(F_t) ,  F_t = Φ_VLM(C_t)
p_φ(z | C_t) = Softmax( MLP_router(f_t) )
L_router = CE( p_φ(· | f_t), y_t )
L_total  = L_action + λ · L_router
```

**（7）推理時路由器決策**

```
z_t = arg max_z p_φ(z | f_t)      （greedy decoding）
```

> 符號與本文保持一致：C_t=(I,O_t,S_t) 為上下文；a_t ∈ R^{H×d} 為 H 步動作塊；θ_z 為相位專家參數；σ 為 flow 時間（有別於軌跡步 t）；M_t 為批量長度掩碼；y_t 為 GT 相位標籤。

**直覺**：公式（4）是整篇的靈魂——訓練時先驗地知道這段是 move 還是 operate，就直接只更新對應專家，兩套參數的梯度天然不互相污染。路由器則被當成獨立的分類問題單獨訓練（公式 6），推理時才用它做貪婪決策。這種「訓練解耦、推理串接」是避免 router 與 expert 相互拖累的關鍵。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

> 以下為說明用的假設數值（非論文原始數據），目的是展示「淹沒」與「解耦」的機制。取單臂 2-DoF，動作塊 H=2 步，每步 d=2。

假設兩個相位的原始動作目標（單位：歸一化前）：

| 相位 | 步1 動作 | 步2 動作 | 幅度量級 |
|------|----------|----------|----------|
| Move | [10.0, 0.0] | [8.0, 0.5] | ~8-10 |
| Operate | [0.2, -0.1] | [0.1, 0.05] | ~0.1-0.2 |

**單體訓練的困境**：若把兩相位混在一起做 z-score 歸一化，全局均值 μ ≈ 4.6、標準差 s ≈ 4.9（受大位移主導）。則 operate 的關鍵信號 (0.2) 歸一化後 ≈ (0.2-4.6)/4.9 ≈ -0.90，看似有值，但梯度貢獻被 move 樣本主導；模型傾向先擬合誤差平方最大的 move 樣本，operate 的細微差異被視為「噪聲」。

**雙專家解耦後**：
- Move 專家：只在 8-10 量級的樣本上回歸速度場，學習「把末端快速送到目標附近」。
- Operate 專家：只在 0.1-0.2 量級的樣本上回歸，等效於擁有自己的、適合精調尺度的歸一化，能把 0.1 的差異學成有意義的梯度。

**損失走一遍**（單樣本，簡化為標量）：設 x_0 = 0（高斯先驗採樣）、σ = 0.5。
- Move 樣本 a = 10.0 → x_σ = 0.5·10.0 = 5.0，目標速度 u = a - x_0 = 10.0。
- Operate 樣本 a = 0.2 → x_σ = 0.1，目標速度 u = 0.2。

兩者目標速度相差 50 倍。若不隔離，回歸器會把容量幾乎全給 10.0；隔離後，operate 專家在 [0, 0.2] 的小範圍內做回歸，相對誤差才有機會被壓低。這就是論文 Fig.2(b) 所述的「joint learning and normalization cause operate actions to be overshadowed」的數值化版本。

## 4. 工程視角 (Engineering View)

| 面向 | 說明 | 工程含義 |
|------|------|----------|
| 推論算力 | 每個動作塊**只激活一個專家** | 推論 FLOPs ≈ 單體模型（非 2×），但**權重記憶體 ≈ 2×**（兩套 LoRA）|
| 參數效率 | 用 LoRA 注入兩專家 + backbone | 只增少量可訓練參數，保住 π0-base 的通用表徵 |
| 路由粒度 | chunk-level（整個 flow 積分期不變）| 提供時序一致性，避免塊內切換造成的抖動/抖振 |
| 延遲主導項 | ODE 積分步數（flow matching 典型 ~10 步） | 真正的延遲瓶頸仍是積分步，路由本身開銷可忽略 |
| 標註成本 | 離線 MLLM 標註：5 fps、單軌跡 ≤64 幀、兩階段提示 + 驗證器自修正 | 一次性資料準備成本；錯誤標籤會直接污染訓練 |
| 動作維度 | 14-DoF 雙臂；論文明言 gripper 維度不服從高斯假設 | 若做量化/正規化，需對 gripper 維度特殊處理 |

部署約束小結：這是一個「幾乎零推論額外開銷」的架構改造——換來的是**訓練側的相位標籤依賴**與**記憶體翻倍**。對於 RAM/顯存吃緊的邊緣部署（如本研究團隊 2GB 級硬體環境），LoRA + 單專家激活是很友善的設計。

## 5. 數據與評測 (Data & Eval)

- **Benchmark**：RoboTwin2（雙臂操作，強域隨機化）。聚焦 8 個任務：Click Alarmclock、Click Bell、Press Stapler、Place Bread Basket、Place Cans (Plasticbox)、Place Burger Fries、Move Pillbottle Pad、Place Empty Cup（§5.1）。
- **資料量**：每任務 **50 條示範軌跡**（clean scenes）；多工預訓練覆蓋全部 50 任務，每任務同樣 50 條。
- **訓練排程**：先在 50 任務上預訓練 **100k 步**（維持 move/operate 等採樣比），再在 8 個任務上各自微調 **20k 步**。
- **評估**：每任務 **100 次試驗**，報告 clean scene 平均成功率。
- **主要結果（表 1）**：平均成功率 **68.9%**，比單體 π0 baseline 高 **+24.1%**（絕對）；接觸密集任務提升最明顯——Click Bell 比次佳高 **+55%**，Press Stapler 高 **+8%**；長時程搬運 Place Cans 由 64% → 79%。
- **數據效率（表 2）**：以 1/10 數據對比 π0.5* 與 GO-1*（均用 10× 數據），在 Press Stapler / Click Bell 上反而 **+13% / +24%**；但在視覺多樣任務（Place Cans Plasticbox）落後，作者歸因於視覺曝光不足。
- **訓練效率（圖 5/6）**：Click Bell 在 60k 步即近峰值 100%、Click Alarm 89%、Press Stapler 91%；論文稱峰值出現在 **少 40% 訓練步數**；Place Cans Plasticbox 微調前 5k 步由 16% → 73%，Place Burger Fries 15k 步達峰值 96%。

> 證據性質說明：以上為**模擬 benchmark（RoboTwin2）**結果；論文正文未見真實機器人實驗，泛化性結論應限於該 sim 設置。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 長時程、接觸密集操作中，把粗移與精調解耦，顯著提升成功率（RoboTwin2 8 任務）。
- 在數據稀缺（50 demos/任務）時，以架構先驗替代數據規模。
- 用自動化 MLLM 管線生產相位標籤，免去人工標註。

**不能做 / 失敗模式**：
- **路由錯判是災難級的**（表 3）：把學習到的路由器換成隨機選擇，平均成功率由 68.88% → **25.63%**；對抗式「反轉相位」進一步掉到 **8.88%**。短時程任務（如 Press Stapler）反轉仍保有 35%，但長時程/高精度任務（Place Cans Plasticbox 2%、Move Pillbottle Pad 0%）幾乎完全失敗——因為 Move 專家缺精調能力、Operate 專家缺速度與幅度。
- **視覺泛化弱**：Place Cans Plasticbox 面對未見過的物體幾何/朝向時，因 1/10 的視覺曝光而吃力，即使動作基元正確。
- **動作幅度假設**：gripper 維度不服從高斯分佈（Fig.2 註），若照搬分佈假設會失真。
- **標籤依賴**：整體訓練品質綁定 MLLM（Seed 1.6 Vision）標註正確性；論文未報告標註錯誤率對下游的敏感性。

### 6.1 隱含假設 (Hidden Assumptions)

1. **人類兩階段運動模型（Woodworth/Fitts）能乾淨映射到機器人任務**——但真實操作中 move 與 operate 常存在重疊/模糊邊界，論文用離散硬標籤簡化了它。
2. **MLLM 產生的相位標籤基本正確**——論文明確用離散硬標籤而非軟標籤，若標籤有噪，專家會被「教錯」。
3. **π0-base 的共享表徵對兩個相位都足夠通用**——只換 LoRA，若 base 對某相位先天不適配，這層假設會失效。
4. **相位數固定為 2**、且子任務分解深度 |p_i| ≤ 2——更複雜任務可能需要多相位，論文未驗證擴展性。
5. **50 任務 × 50 demos 的分佈足以支撐路由泛化**——路由器的 OOD 表現未被單獨評估。

## 7. 與相關工作對比 (Comparison)

| 方法 | 關注點 | 架構 | 訓練方式 | 適用場景 |
|------|--------|------|----------|----------|
| π0（baseline）| 通用單體 flow VLA | 單一策略網絡 | 單體模仿學習 | 泛化但精調弱 |
| π0.5* / GO-1* | 大數據泛化 | 單體/大規模 | 10× 數據 | 視覺多樣但依賴數據規模 |
| MoE-DP / FedVLA | MoE 任務特化 | 專家混合（軟/門控）| 混合專家 | 多任務異質性 |
| SP-VLA / Action-aware Pruning | 推論加速 | 調度 + token 剪枝 | 加速導向 | 效率優先 |
| STARE-VLA / Mixture of Horizons | 階段式 RL / 多視野 | 時間縮放 | RL/信用分配 | 信用分配 |
| **本文 Move-Then-Operate** | **move/operate 相位解耦** | **雙硬切換專家 + chunk 路由器** | **模仿 + GT 標籤 teacher-forcing** | **長時程、接觸密集、數據稀缺** |

與已有 MoE-VLA 的差異：前人主要針對 RL 信號或時間縮放，且多做 token 級軟路由；本文強調在**以模仿學習為主**的監督下，把「長程移動」與「接觸密集操作」用**硬參數隔離**顯式分開。

**面試 Tip**：被問到「dual-expert VLA 怎麼防止 routing 崩潰」時，答——訓練時用 GT 相位做 teacher-forcing 只更新匹配專家、路由器單獨用 CE 訓練，推理才貪婪選專家；ablation 顯示反轉相位直接掉到 8.88%，證明兩個專家學到的是不兼容的行為，因此路由正確性是整套方法的生命線。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做**雙臂/長時程操作**、接觸密集任務的研究者——§3.1、§4.2、§5.3 是核心。
  2. 想評估**把相位解耦遷移到新機器人平台**可行性的工程師——重點看 §4.1 的 LoRA 參數隔離與 §4.4 的自動標註管線。
  3. 研究**指數先驗 vs 數據規模**權衡的人——§5.2.2 的 1/10 數據對比值得細看。
- **建議章節路徑**：先讀 §1 Introduction 抓動機 → 再看 §3.1 相位分解與 §4.2 路由器 → 然後 §5.3 Ablation（理解失敗模式）→ 可跳 §3.2 CFM 推導（若已熟悉 flow matching）。
- **不值得精讀的理由**：若你不做機器人學習、或已熟悉 MoE/雙專家 VLA，讀摘要 + §5.3 即可掌握全部要點；目前的證據僅限單一 sim benchmark，尚不足以支撐「通用解」的結論。

---
[← Back to Theory](./README.md)

**關鍵引用**：
- 論文：https://arxiv.org/abs/2604.23620
- 代碼：https://github.com/healenrens/Move-then-Operate
- 基礎模型：π0 — A Vision-Language-Action Flow Model for General Robot Control (Black et al., 2024)
- 基準：RoboTwin 2.0 (Chen et al., 2025)
