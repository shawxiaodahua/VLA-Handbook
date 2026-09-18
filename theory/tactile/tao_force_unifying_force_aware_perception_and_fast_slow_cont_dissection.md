# TAO-Force：融合力觉感知与快慢双环控制的接触丰富操作 (TAO-Force: Unifying Force-Aware Perception and Fast–Slow Control for Contact-Rich Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-18
>
> **论文**: TAO-Force: Unifying Force-Aware Perception and Fast–Slow Control for Contact-Rich Manipulation
> **链接**: [arXiv:2609.18497](https://arxiv.org/abs/2609.18497)（[HTML](https://arxiv.org/html/2609.18497v1)）
> **核心定位**: 在一个**冻结**的预训练 VLA（GR00T N1.5）之上，用 F-FiLM 把六维力/力矩反馈注入视觉-语言表征，并用**接触门控的慢-快双环控制器**解决"低频策略 + 高频接触动力学"的错配——不重训 backbone，只加轻量适配层。

---

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 力应该作为"一等模态"注入 VLA 表征（F-FiLM），且动作要交给"接触态感知的快-慢双环"执行，而非直接把位置指令打给机器人 |
| 適合精讀 | 做**接触丰富操作**（剥皮/按压/擦拭/切割）、需要把已有 VLA 迁到力控执行栈的研究者与工程师，重點看 §IV-B、§IV-D、§V-E |
| 可以跳過 | 只关心自由空间抓取取放、或完全不用力传感器的人，这篇距离中等 |
| 落地可行性 | 中——F-FiLM 与接触门控逻辑可复用；但需要具备六维力传感的机器人 + 高频导纳控制接口 |
| 主要風險 | 评测集中在单一 AgiBot G1 平台、4 个桌面任务；OOD 泛化仍会在"换工具"场景明显掉点 |

💡 **X-Ray 开场**
这篇论文问的是：**当机器人要剥黄瓜、按泵头这种"靠力不靠看"的任务时，纯视觉 VLA 为什么不够？**它发现视觉几乎看不出"何时接触、接触多大"，而位置控制又无法对快速变化的接触力做出柔顺响应。于是它做了两件事：把力反馈通过学习到的**仿射调製（F-FiLM）**塞进冻结的视觉-语言表征，再用一个**接触门控**在两套控制器之间切换（自由段走位置、接触段走高频导纳）。对 VLA 研究者的意义是：**力觉泛化的瓶颈可能不在数据量，而在"如何在不动语义先验的前提下把力信息挤进表征"。**

📍 **研究全景时间线**

```
2023  RT-1 / RT-2 —— 端到端视觉-语言-动作
  │
2024  OpenVLA / Octo / RDT-1B —— 开源 VLM-based policy、扩散/流匹配动作头
  │
2024-25  ForceVLA / TA-VLA / FoAR / ManipForce —— 开始把力/触觉并入 VLA（token 融合、MoE、接触门控、频率编码）
  │
2025  π0 / π0.5 / GR00T N1 —— 分层系统 + 连续控制器
  │
2026-09  ★ TAO-Force ← 当前位置
        冻结 backbone + F-FiLM 注入力 + 接触门控快慢双环
  │
局限  单一 AgiBot G1 平台；4 个桌面任务；OOD 换工具后掉点明显
```

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

TAO-Force 建構在 **GR00T N1.5** 之上：其 Eagle 視覺-語言 backbone 由 SigLIP 視覺編碼器 + Qwen 語言模型組成，再接一個 **DiT（diffusion transformer）動作頭**。核心策略是 **凍結 backbone**，只訓練新增的力編碼器、F-FiLM 調製網路與預測頭。

### 1.1 系統對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 頻率/時序 | 訓練/推理差異 |
|------|------|------|-----------|---------------|
| 視覺-語言 backbone（凍結） | 觀測 o_t、語言 l | 表徵 z^vl_t | 策略 5 Hz（觀測緩衝 30 Hz） | 全程凍結，不更新 |
| 力編碼器 g^cur/g^hist/g^state | 當前力、力歷史、機器人狀態 | 條件 token e_t | 與策略同步 | 只此部分可訓練 |
| F-FiLM 調製網路 h^film | e_t | Δγ_t、β_t | 與策略同步 | 零初始化，初期等價恆等 |
| 共享 DiT 解碼器 d^dit | 噪聲 action/force token、z̃^vl、s_t | 兩路特徵 u^a、u^f | 與策略同步 | 分別出速度場 v^a、v^f |
| 接觸分類頭 d^c | z̃^vl_t | 接觸機率 p̂^c_t | 與策略同步 | 輕量分類頭 |
| 動作處理器 ActionProcessor | 動作 chunk | 關節參考 Q^ref | 267 Hz 參考 | 多項式擬合 + minimum-jerk 過渡 |
| 力處理器 ForceProcessor | 力 chunk | 力參考 F^ref | 267 Hz 參考 | chunk 邊界平滑混合 |
| 導納控制器 Admittance | q、F_ext、F_d | q̇^adm → q^cmd | **1000 Hz** | 接觸期才啟用 |

### 1.2 關鍵機制 (Key Mechanism)

- **F-FiLM 而不是簡單拼接**：不像早期方法把力做成 token 直接 concat，F-FiLM 用學到的**仿射變換**去縮放/平移 backbone 已有的視覺-語言表徵。原因是力感測沒有可比擬的大規模預訓練特徵提取器，post-training 時視覺表徵會**主導優化**，導致力資訊被"淹沒"。
- **零初始化**：把調製網路最後一層零初始化，使訓練起點滿足 Δγ=0、β=0，F-FiLM 起始為恆等映射——**先保住語義先驗，再漸進注入力**。
- **接觸因果力監督**：只在當前處於接觸區間時才用示範的力作為目標，避免模型"從視覺/時間模式猜未來力"，導致接觸前就輸出非零力。
- **力注意力模態丟棄**：在力關鍵時段部分遮蓋視覺/本體特徵，逼模型真的去用（而非當作另一個無關 token）。

⚡ **Eureka Moment**：**別把力當成"又一路輸入"，而要把它當成"對已凍結語義表徵的條件調製"——用零初始化的 FiLM 讓力資訊從旁路漸進接管，同時用接觸機率把"預測"與"執行"解耦成慢/快兩個時標。**

### 1.3 資訊流/架構圖 (Flow / Diagram)

```
 ┌─────────────── 慢系統（策略推理, 5 Hz） ────────────────┐
 o_t (3×RGB) ─┐                                          │
 l (語言)   ─┼─► [凍結 VL backbone] ─► z^vl_t ─┐          │
 s_t (狀態) ─┘                                  │          │
 F_ext_t ─► g^cur ─┐                            ▼          │
 F_ext_{t-H:t} ─► g^hist ─┼─► e_t ─► h^film ─► z̃ = (1+Δγ)⊙z^vl + β
 s_t ─► g^state ─┘                                  │      │
                                              ┌─────┴─────┐
                                              ▼           ▼
                                        [共享 DiT]   [接觸分類頭]
                                         ├─► â (action chunk)  ├─► p̂^c_t
                                         └─► F̂^d (force chunk) │
                                                              │
 ┌───────────── 快系統（執行, 1 kHz） ───────────────────────┼──┐
 ActionProcessor(â) ─► Q^ref                          gate g = 1[p̂^c ≥ η]
 ForceProcessor(F̂^d) ─► F^ref                              │
                          Free 段(g=0): q^cmd = q^pos ◄──────┤
                          Contact 段(g=1): q^cmd = q^cmd⁻ + Δt·IK(Admittance) ◄─┘
                          Contact→Free: quintic smoothstep 移除殘差，平滑回歸
```

---

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
z̃ = (1 + Δγ(e)) ⊙ z^vl + β(e)      # 力調製表徵
gate g = 1[ p̂^c ≥ η ]               # 接觸門控：慢位置 ↔ 快導納
```

**先給目標**：讓凍結的 VLA 表徵被"當前力狀態"條件化（感知），再把策略輸出翻譯成不致抖動的高頻力控指令（控制）。

**目標一：導納控制（執行介面）**

```
M(ẍ − ẍ_d) + D(ẋ − ẋ_d) + K(x − x_d) = F_ext − F_d
```

| 符號 | 含義 |
|------|------|
| x, x_d | 實際 / 參考末端位姿 |
| F_ext, F_d | 測得外力（已補償工具重力）/ 期望力 |
| M, D, K | 虛擬質量、阻尼、剛度矩陣 |

直覺：把**力誤差**翻譯成**柔順的末端運動修正**；小剛度 K 當"弱位置錨點"限制漂移，不讓它壓過力調節。

**目標二：策略輸出（感知+生成）**

```
π_{θ,φ}( o_t, l, s_t, F_ext_t ) → ( a_{t:t+K},  F^d_{t:t+K},  p^c_t )
```

- θ：來自預訓練 VLA 的（凍結）參數；φ：新引入的力適配參數。
- 輸出：動作 chunk、期望力 chunk、當前接觸機率。

**F-FiLM 調製**：

```
e_t   = Concat[ g^cur(F_ext_t),  g^hist(F_ext_{t-H:t}),  g^state(s_t) ]
Δγ_t, β_t = h^film(e_t)
z̃^vl_t = (1 + Δγ_t) ⊙ z^vl_t + β_t
```

**接觸因果力目標**：

```
F̄*_{t:t+K} = c_t · F*_{t:t+K}      # c_t ∈ {0,1}：是否在任務相關接觸區間
```

非接觸段 c_t=0 → 力目標歸零，抑制"接觸前亂預測力"。

**總損失**：

```
L = L_act + λ_force · L_force + λ_contact · L_contact

L_act     = E_{t,τ}[ ‖v̂^a − v^{a,*}‖²₂ ]        # 動作流匹配
L_force   = E_{t,τ}[ ‖v̂^f − v̄^{f,*}‖²₂ ]        # 力流匹配（用接觸因果目標）
L_contact = −E_t[ c_t·log p̂^c + (1−c_t)·log(1−p̂^c) ]   # BCE
```

> 符號與本文/相關文檔保持一致：v̂ 為預測速度場，v* 為 ground-truth chunk 誘導的目標速度場；τ∈[0,1] 為流時間。λ_force、λ_contact 為權重（**具體數值論文正文未給出，標 TODO**）。

> TODO: 論文未在主文給出 K（預測時域）、H（力歷史長度）、η（接觸閾值）與 λ 的具體取值；文中提到接觸判別分析時對 p 取 **0.8** 閾值，可作 η 的近似參考。

---

## 3. 帶數字走一遍：玩具例子 (Worked Example)

以論文 **Blind Box Sorting（視覺完全相同的粥罐，按重量分揀）** 的**反事實力注入**為例（論文 Table I）：

設定：固定視覺觀測（同一個罐子看起來一樣），在決策錨點（決策路徑開始分岔的時刻）替換力信號為"空罐"或"滿罐"的力特徵，觀察生成的 Joint-2 動作 chunk 最後 16 幀。

| Episode | 方法 | 注入力 | Trajectory Error | Flip Rate |
|---------|------|--------|------------------|-----------|
| Empty | F-FiLM (Ours) | Empty | 0.0001 | – |
| Empty | F-FiLM (Ours) | Full | **0.0186** | 90% |
| Empty | DePost | Empty | 0.0005 | – |
| Empty | DePost | Full | 0.0056 | 20% |
| Empty | FVLMoE | Empty | 0.0005 | – |
| Empty | FVLMoE | Full | 0.0005 | **0%** |
| Full | F-FiLM (Ours) | Full | 0.0001 | – |
| Full | F-FiLM (Ours) | Empty | **0.0302** | 100% |
| Full | DePost | Full | 0.0002 | – |
| Full | DePost | Empty | 0.0326 | 90% |
| Full | FVLMoE | Full | 0.0001 | – |
| Full | FVLMoE | Empty | 0.0036 | **10%** |

**怎麼讀這張表**：同一張視覺圖，只換力——F-FiLM 每次都**大幅偏離**原軌跡（誤差 0.0186 / 0.0302）並**按注入力改道**（Flip 90% / 100%），說明它**真的在用力**。反觀 FVLMoE 幾乎不動（0.0005 / 0.0036，Flip 0% / 10%），說明它的融合被視覺主導、力信號"進不去"。DePost 則表現不對稱（一個方向 20%、另一方向 90%）。

一句話：**F-FiLM 在兩個交換方向上都響應一致，是三者中最"聽力"的融合方式。**

---

## 4. 工程視角 (Engineering View)

| 層 | 頻率 | 工程含義 |
|----|------|----------|
| 觀測緩衝 | 30 Hz | 相機與狀態入佇列，供策略取樣 |
| VLA 策略推理 | **5 Hz** | 單張 RTX 5070 Laptop GPU（8 GB VRAM）即可跑，動作/力 chunk 以 30 Hz 表達 |
| 位置/力參考 | 267 Hz | ActionProcessor 多項式擬合 + minimum-jerk，ForceProcessor 平滑混合 |
| 導納控制 | **1000 Hz** | 讀 q 與 F_ext，算 q̇^adm，IK 後積分出 q^cmd |

**關鍵 trade-off**：

- **時標分離（慢 5 Hz / 快 1 kHz）**：策略負責"該去哪、該出多大力"，導納迴路負責"當下怎麼柔順地達成"。這是把"低頻 chunk 預測"接到"高頻接觸動力學"的橋。
- **抖動/不連續**：兩個來源——(a) chunk 邊界，(b) 接觸→自由切換殘差。解法分別是 minimum-jerk 過渡 與 **quintic smoothstep**（w(s)=1−10s³+15s⁴−6s⁵）漸進消殘差；若過渡中再次接觸，立即取消融合、恢復力控。
- **內存/算力友好**：凍結 backbone + 輕量適配層，讓部署能塞進 8 GB 顯存——這對"已有 VLA、想加力控"的團隊是低門檻賣點。
- **部署約束**：需要**腕部六軸 F/T 傳感器** + 可接受高頻導納指令的位置控制機器人；力信號要先做**工具重力補償**並轉到控制幀。

---

## 5. 數據與評測 (Data & Eval)

**平台**：AgiBot G1 雙臂人形機器人；3 個 RGB（1 頭部 + 2 腕部）；腕部六軸 F/T 傳感器。

**任務（5 個，每任務 ID/OOD 各 30 trials）**：

| 任務 | 成功判據 | 力的角色 |
|------|----------|----------|
| Blind Box Sorting | 每個罐放對箱（按重量） | 視覺無法判別，**必須**用力 |
| Peeling Cucumber | 剝出長於半根黃瓜的連續皮 | 表面接觸力決定貼合 |
| Pressing Pump Head | 出液且不過度用力 | 過壓即失敗 |
| Wiping Whiteboard Marks | 標記可視清除 | 壓力不足/不穩即失敗 |
| Cutting Cucumber | 切斷 | 力不穩則切不斷 |

**OOD 擾動**：Peeling/Pressing **改變被操作物高度**（改變接觸幾何）；Wiping/Cutting **替換工具**（改變工具幾何→改變接觸時序與力特性）。

**基線**：GR00T N1.7、π0.5（Blind Box 因基線無力輸入而標 N/A）。

**主要結果**：四個接觸豐富任務平均成功率 **ID 80.0% / OOD 64.2%**（來自 Fig.1 描述，論文 Table II 具體逐任務數字未在抓取文本中完整呈現 → 待補）；Blind Box Sorting 兩設定皆 **100.0%**。OOD 掉點比 GR00T N1.7 與 π0.5 更小。

**消融（Table IV，Peeling + Pressing）**：GR00T N1.5 + F-FiLM（無力控） vs GR00T N1.5 + Force Control（無 F-FiLM） vs 完整 TAO-Force → **完整版在每個設定都最高**，說明"上游力調製生成"與"下游柔順執行"**互補且缺一不可**。

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 用力區分**視覺完全相同**的物體（Blind Box 100%）。
- 對接觸幾何變化（高度）與工具變化有較好的 OOD 魯棒性（相對基線掉點更小）。
- 在 8 GB 顯存上以凍結 backbone 方式部署（工程可負擔）。

**不能做/易失敗**：
- **換工具場景**（Wiping / Cutting 的 OOD）：工具幾何改變 → 接觸時序與力特性漂移 → 擦拭壓力不足、接觸不穩、切割無效，成功率**明顯下降**（論文明說）。
- **Peeling**：失敗主因是削皮器**沒接觸到表面**或只削下很短的皮。
- **Pressing Pump Head**：換高度後，失敗主因是**壓得過猛**（力控未能及時收斂）。
- 未在**移動底盤 / 多臂協同 / 人形全身**上驗證——不要把桌面結果外推。

### 6.1 隱含假設 (Hidden Assumptions)

1. **示範質量決定力目標質量**：接觸因果監督依賴示範中"任務相關接觸區間"的 c_t 標註——但論文未明說 c_t 是**自動推斷還是人工標註**，若為人工則擴展成本高。
2. **接觸概率 p̂^c 可分**：整套門控假設存在一個閾值 η 能把自由段與接觸段乾淨切開；論文分析用了 0.8，但**未給部署 η 與誤判的代價分析**（誤判→抖動/切換震盪）。
3. **力與視覺可異步但需時間對齊**：策略在 5 Hz 取樣，假設力歷史 H 幀能覆蓋"當下接觸狀態"；若接觸來得比 H 更急，可能漏檢。
4. **力傳感器標定與重力補償可靠**：F_ext 需補償工具重力並轉幀——傳感器漂移/工具變更都會破壞這一前提（正是換工具 OOD 掉點的根源之一）。
5. **單一平台代表性**：僅 AgiBot G1，F/T 傳感器規格、控制接口的差異性未討論。

---

## 7. 與相關工作對比 (Comparison)

| 方法 | 關注點 | 融合方式 | 執行方式 | 適用場景 |
|------|--------|----------|----------|----------|
| ForceVLA / TA-VLA | 力作為輸入 | token 融合 / 後置融合（如 FVLMoE、DePost） | 位置指令直接執行 | 一般接觸任務，力貢獻有限 |
| ForceMimic / ForceVLA2 / Tactile-VLA | 聯合預測位姿與力 | 混合力-位置控制 | 較高頻低層控制器跟蹤 | 需要力跟蹤的裝配類 |
| Reactive Diffusion / AT-VLA / PhaForce | 慢規劃 + 快修正 | 解耦快慢 | 接近控制率的快流修正 | 動態接觸 |
| **TAO-Force** | **感知+控制雙 gap** | **F-FiLM 凍結表徵調製** | **接觸門控 慢位置/快導納 1 kHz** | **接觸豐富桌面操作** |

**差別一句話**：別家多在"力怎麼進網路"或"怎麼分快慢"上二選一；TAO-Force 主張**兩者都要**，且用零初始化 FiLM 保證"加力不傷語義先驗"。

**面試 Tip**：若被問"為什麼不直接把力 concat 進去？"——答：**力缺大規模預訓練特徵提取器，post-training 時視覺表徵會主導優化，把力信號淹沒；F-FiLM 用零初始化的仿射調製從旁路漸進注入，既保語義先驗又逼模型真用力。**（Table I 的 Flip Rate 就是這個論點的直接證據。）

---

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 在做**接觸豐富操作 / 力-位置混合控製**的 VLA 研究者（重點 §IV-B、§IV-D）。
  2. 手上已有預訓練 VLA（尤其 GR00T 系）想**低成本加力控**的工程師（重點 §IV-A、§IV-C、§V-A 部署參數）。
  3. 研究**多模態異步融合**（力/觸覺 vs 視覺）表徵學習的人（重點 §IV-B 的 F-FiLM 設計動機）。
- **建議章節路徑**：先讀 §I 引言（兩個 gap 的定義）→ §IV-B（F-FiLM + 接觸因果監督）→ §IV-D + Algorithm 1（快慢雙環）→ §V-B/V-E（Table I / IV 證據）；§II 相關工作可略讀（按需查引用）。
- **不值得精讀的理由**：若你不做機器人學習、或只關心自由空間抓取取放、或**沒有力傳感器**的部署環境——讀摘要 + §V 結果即可；其餘設計對你複用價值有限。

---

[← Back to Theory](./README.md)

**關鍵引用**：
- 論文：[arXiv:2609.18497](https://arxiv.org/abs/2609.18497) · [HTML](https://arxiv.org/html/2609.18497v1)
- 基礎模型：GR00T N1.5 (Eagle backbone) · π0.5 · GR00T N1.7
- 力覺 VLA 前作：TA-VLA (DePost)、ForceVLA (FVLMoE)、FoAR、ForceMimic、ForceVLA2、Tactile-VLA
- 執行參考：VLA-RAIL（chunk 稠密化 + minimum-jerk）
