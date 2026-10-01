# DeltaWAM：以 Delta Token 為單位的變更中心世界動作模型 (DeltaWAM: Change-Centric Visual Foresight via Delta Tokens for an Efficient World-Action Model)

> ⚙️ 本文由 Moltbot 自動生成 | 2026-09-30
>
> **論文**: DeltaWAM: Change-Centric Visual Foresight via Delta Tokens for an Efficient World-Action Model
> **鏈接**: https://arxiv.org/abs/2609.33177
> **代碼**: https://github.com/deltawam/DeltaWAM （論文聲稱，未經外部複現）
> **作者**: Wenrui Bao (UCF), Bingxin Xu (USC), Yu Tian (UCF), Yuzhang Shang (UCF, 通訊)
> **核心定位**: 把「預測完整未來畫面」換成「只預測相鄰幀的變化量」——用一個低維 delta token 取代一整張未來 DINO 特徵圖，在不重建全場景的前提下為策略提供動作驅動的動態先驗。

---

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 世界動作模型（WAM）不該逐幀重建未來全場景；把預測單位壓成「單個 delta token（幀間 DINO 特徵差）」，反而給出更強、更省算力的動作先驗 |
| 適合精讀 | 如果你在做「世界模型 + VLA」、latent world model、或想降低 WAM 推理延遲與顯存，重點看 §1.2、§2、§4、§6.1 |
| 可以跳過 | 如果你只關心純 VLA（無預測分支）或大規模機器人預訓練策略的 SOTA 數字，這篇距離中等——它打的是「效率 × 動態先驗」而非絕對成功率 |
| 落地可行性 | 中高。單次前向生成 delta token（無迭代去噪）、142.1 ms/chunk、3.86 GB 峰值顯存、0.725B 參數，單卡可跑；但依賴預訓練好的 DeltaWorld 與 DINOv3 |
| 主要風險 | 高度壓縮的 delta token 會濾掉精細空間細節，導致「精度接觸」任務放置偏移；且 LIBERO 絕對成功率低於最強預訓練 VLA/WAM 基線 |

💡 **X-Ray 開場**（非專家也能讀懂）
想像你預測下一步，不必重畫整張照片，只要說明「哪個東西往哪移了多少」。DeltaWAM 就是這個思路：它不重建未來畫面，只預測「變化」。解決的問題是世界動作模型太貴——逐幀重畫未來場景，大量算力浪費在幾乎不變的背景上。它發現：把變化壓進一個向量，策略不但學得更好（LIBERO 92.8%），還更抗任務擾動。對 VLA 研究者的意義是：**世界模型的關鍵不在「看得全」，而在「抓得準變化」**。

📍 **研究全景時間線**

```
2019-2023  pixel-space 世界模型 (Dreamer / VAE-latent)
   │
2025       Video-based WAM：逐幀生成未來畫面（F1 / Cosmos-Policy），貴、延遲高
   │
2025-2026  Latent WAM：在 DINO/V-JEPA 特徵空間預測（LDA-1B / JEPA-WAM / LaWAM）
   │         但仍逐幀預測「完整」latent 狀態
   │
2026 (Sep) DeltaWorld：把幀間 DINO 差壓成單一 delta token（本文骨幹）[arXiv 2609.x]
   │
2026 (Sep) DeltaWAM ← 本文：delta token + 語言調節 + flow-matching Action DiT
            從「場景重建」轉向「變化預測」；0.725B / 142ms
   │
   局限：僅 LIBERO 級別桌面操作 + 單臂 PiPER；精度接觸任務放置偏移
```

---

## 1. 核心架構/方法總覽 (Overview / Architecture)

DeltaWAM 由三個部件串成：**(1) Delta Tokenizer**（凍結，來自 DeltaWorld）把幀間 DINO 特徵差壓成 delta token；**(2) DeltaWorld Predictor**（自迴歸、經語言調節）滾出未來 delta 序列；**(3) Action DiT**（flow-matching）以「當前 DINO 特徵（空間錨）+ 預測 delta 序列（動態子目標）」為條件生成動作塊。

### 1.1 系統對比概覽 (System Component Comparison)

| 模組 | 輸入 | 輸出 | 訓練/推理 | 備註 |
|------|------|------|-----------|------|
| DINOv3 骨幹 (φ) | 多視角 RGB 幀 O_τ | 稠密 patch 特徵 X_τ ∈ R^{V×N×Dv} | 凍結 | 提供空間語義佈局 |
| Delta Tokenizer (g/h) | (X_{τ-1}, X_τ, z_init) | delta token z_τ ∈ R^{V×1×Dv} | 凍結（來自 DeltaWorld） | 用重建損失訓練；z_init 為可學習聚合 token |
| DeltaWorld Predictor (P_ψ) | 歷史 delta Z_obs、隨機查詢 q、任務嵌入 c | 未來 delta 序列 Ẑ | Stage 1 微調（BoM） | 生成式；每步單次前向，無迭代去噪 |
| 語言編碼器 (T5) | 指令文字 | 任務嵌入 c | 凍結 | 經 Gated AdaLN 調節 Predictor |
| Action DiT (v_θ) | 當前 DINO 特徵 + 預測 delta + 噪聲動作 | 動作塊 a ∈ R^{16×7} | Stage 2 訓練（DeltaWorld 凍結） | flow-matching；含 KV adapter |

### 1.2 關鍵機制 (Key Mechanism)

- **只建模變化，不重建全圖**：物理操作中相鄰幀共享絕大部分視覺上下文，真正需要「預判」的只是動態變化（物體位移、末端運動）。delta token 天生把共享靜態上下文丟掉，只留變化。
- **解耦「錨」與「子目標」**：當前 DINO 特徵是「東西在哪（spatial anchor）」，預測 delta 序列是「東西該怎麼動（dynamic subgoal）」——兩路信號互補餵給策略。
- **語言調節避免幻覺未來**：原始 DeltaWorld 是無條件的，同一場景可能滾出多個物理合理但與任務不符的未來。用 Gated AdaLN 把指令注入 Predictor，讓語言「引導」已學到的物理，而非覆寫（輸出投影零初始化）。
- **單次前向生成**：每個 delta token 只需一次前向，而非擴散式的迭代去噪——這是延遲能壓到 142 ms 的關鍵。
- **BoM（Best-of-Many）多模態**：同一任務有多條合理完成路徑，回歸目標會產生「平均化」的非真實未來；BoM 用 K 個高斯查詢、只監督最接近真值的那條，保留多模態。

⚡ **Eureka Moment**：**世界模型的預測單位不該是「場景」，而該是「變化」**——把幀間 DINO 差壓成一個向量，既暴露了動作相關的動態，又天然丟掉了跨幀冗餘，比預測完整未來 latent 更「動作驅動（action-forcing）」。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
                  ┌─────────────── 訓練 Stage 1（BoM 微調 DeltaWorld）───────────────┐
指令文字 ──T5──► c ─┐                                                               │
                    │                                                               ▼
多視角 RGB ──DINOv3──► X_τ ──Delta Tokenizer g──► z_τ (delta token) ──► DeltaWorld Predictor P_ψ
                                                    │                        │ 自迴歸滾出 Ẑ_{t+1..t+Lw}
                                                    │                        │ (Gated AdaLN 用 c 調節)
                                                    ▼                        ▼
                                              (歷史 Z_obs)           未來 delta 序列 Ẑ（動態子目標）

┌─────────────── 訓練 Stage 2 / 推理（凍結 DeltaWorld，訓練 Action DiT）───────────────┐
當前 DINO 特徵 X_t（空間錨） ──┐                                                      │
                              ├──► Action DiT (flow-matching, v_θ) ──► 動作塊 a (16 步) │
預測 delta 序列 Ẑ（動態子目標）─┘                                                      │
                                                      推理：採樣一條未來軌跡 → 生成 1 個 action chunk
```

---

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本質）：

```
z_τ = g(X_{τ-1}, X_τ, z_init)      # 幀間 DINO 特徵差 → 一個 delta token（預測單位）
```

**目標**：把「逐幀預測完整未來狀態」替換為「自迴歸預測未來的 delta token 序列」，再以此為條件生成動作。

**(a) Delta Tokenization（來自 DeltaWorld）**

```
X_τ = φ(O_τ) ∈ R^{V×N×Dv}                # V 視角, N patch, Dv 特徵維
z_τ = g(X_{τ-1}, X_τ, z_init) ∈ R^{V×1×Dv}   # 每視角壓成 1 個 token
X̂_τ = h(z_τ, X_{τ-1})                    # decoder 重建當前特徵（訓練用重建損失）
```

- φ：凍結 DINOv3；V：相機視角數；N：每視角 patch 數；Dv：特徵維。
- z_init：可學習聚合 token；對首幀用合成參考幀 O_∅ = 0 錨定絕對視覺。
- 時間戳採樣讓同一 token 能表徵「近乎靜止」到「劇烈變化」的轉移。

**(b) 任務條件化世界建模**

```
ẑ_{t+j}^{1:V} = P_ψ(q_{t+j}, Z_obs, Ẑ_{t+1:t+j-1}, c),   j = 1..Lw
```

- q_{t+j}：跨視角共享、採自高斯分佈的隨機查詢（提供多模態）；Z_obs：觀測歷史 delta；c：語言任務嵌入。
- Gated AdaLN 調節每個注意力/MLP 分支：

```
h̃    = (1 + γ(c)) ⊙ LN(h) + β(c)
h_out = h + (1 + η(c)) ⊙ F(h̃)
```

- γ, β, η 為語言相關的 scale / shift / residual-gate，其輸出投影**零初始化**——保證微調是「疊加」而非「覆蓋」預訓練動力學。

**(c) Stage 1 損失（BoM）**

```
E_τ^(k) = (1/V) Σ_v ρ_SL1(z_τ^v, ẑ_τ^(k),v),  k_τ* = argmin_k E_τ^(k)
L_world = 1/(T-1) Σ_{τ=2}^{T} E_τ^(k_τ*)
```

- 每位點採 K 個高斯查詢並行前向；只對最接近真值的候選回傳梯度（Smooth-L1）。候選索引不必跨序列固定。

**(d) Stage 2 損失（flow-matching 動作專家）**

```
a_σ = (1-σ)·a + σ·ε,   ε ~ N(0, I)
L_action = E_{a,ε,σ}[ w(σ) · ‖ v_θ(a_σ, σ; C) − (ε − a) ‖_F² ]
```

- v_θ：Action DiT 預測的速度場；w(σ)：flow-time 權重；C：條件上下文（空間錨 + 預測 delta）。
- 訓練時以概率 p_oracle 條件於「oracle」未來分支 k†（最小軌跡誤差），否則隨機選一條——避免不穩定分支污染訓練。

> 符號與 DeltaWorld 原文保持一致。以上公式均以代碼塊/純文本表達，未使用 LaTeX 數學語法。

**直覺**：世界模型的訓練目標從「重建場景」變成「重建變化」；策略的條件從「完整未來狀態」變成「稠密錨點 + 稀疏變化序列」。變化序列是低維的，所以更難被無關細節淹沒。

---

## 3. 帶數字走一遍：玩具例子 (Worked Example)

假設單相機（V=1）、單任務，取一個 2 步驟的極簡玩具：

1. **觀測**：當前幀 O_t → DINOv3 → X_t（一張 patch 特徵圖，N 個 patch）。
2. **編碼變化**：上一幀 X_{t-1} 與 X_t 之差不變（機器人靜止），則 z_t ≈ 0 向量（近乎靜止轉移）；若末端移動 2 cm，z_t 是一個非零低維向量，主要編碼末端位移方向。
3. **預測**：給定歷史 Z_obs = [.., z_{t-1}, z_t]、隨機查詢 q、任務嵌入 c="把黑碗放到盤子上"，Predictor 自迴歸滾出 Ẑ = [ẑ_{t+1}, ẑ_{t+2}, ..., ẑ_{t+6}]（默認 6 步 horizon，每視角 1 個 token/步）。
4. **動作生成**：Action DiT 以 X_t（空間錨：碗/盤子在哪）+ Ẑ（動態子目標：碗該往盤子方向移）為條件，經 flow-matching（10 個 Euler 步）生成 16 步動作塊，執行前 8 步後重規劃。
5. **對比量化**：若改用「解碼完整未來 DINO」作條件，每 chunk 需要約 14,336 個視覺 token；用 delta token 只需約 2,060 個——**約 1,024× 的未來 token 削減**（論文 Figure 6a 註記），而成功率反而更高（92.8% vs 79.0%）。

這個「可計算閉環」的核心：**變化量維度低、噪聲少 → 策略不必在大量靜態冗餘裡撈信號**。

---

## 4. 工程視角 (Engineering View)

| 指標 | 數值 | 工程含義 |
|------|------|----------|
| 總參數 | 0.725B（trainable 120.3M） | 單卡 A100/H100 可載入，部署門檻低 |
| 推理延遲 | 142.1 ms / action chunk（A100，查詢 100 次） | 約 7 Hz 重規劃頻率，接近實時閉環 |
| 峰值顯存 | 3.86 GB | 邊緣/工作站級 GPU 友好 |
| 算力 | ~0.584 TFLOPs / action | 相比 pixel-space WAM 低一到兩個數量級 |
| 訓練成本 | 256 GPU 小時 / 2×H100 | 訓練門檻大幅低於同類 WAM |
| 生成方式 | 每 delta token 單次前向（非迭代去噪） | 延遲可預測、抖動小 |
| 動作塊 | 16 步 chunk、10 Euler 步、執行 8 步後重規劃 | 重規劃頻率決定閉環響應 vs 抖動折衷 |
| 未來 horizon | 默認 6 步（= 12 個動作步，含時間下採樣） | 更長 horizon 非單調更好，存在預測誤差累積 trade-off |

**關鍵 trade-off**：
- **壓縮 vs 精度**：delta token 濾掉共享上下文換來效率，但也濾掉了精細空間細節 → 精度接觸任務出現「放置偏移」（論文自述 LIBERO-Spatial/Long 的主要失敗模式）。
- **horizon vs 誤差累積**：測試時移除全部未來 token → 成功率歸零（策略退化為亂伸亂抓）；只用 2 步未來 → Spatial/Object/Goal/Long 分別恢復到 76.2/76.4/76.8/48.0；默認 6 步最好但並非越長越好。
- **多模態 vs 穩定**：BoM 保留多種未來，但非 oracle 分支質量不穩，故需要 p_oracle 的概率混合來穩定 Action DiT 訓練。

---

## 5. 數據與評測 (Data & Eval)

**訓練數據**：
- 世界模型預訓練：Kinetics-700（大規模人類活動/物體交互視頻）→ DeltaWorld。
- Stage 1：在 LIBERO 示範上微調 DeltaWorld（任務條件化）。
- Stage 2：在機器人數據上訓練 Action DiT（DeltaWorld 凍結）。
- 真實世界：AgileX PiPER 六自由度機械臂，150 集 pick-and-place，30 Hz，前視 + 腕部 RGB，14 維機器人狀態，7 維動作（6 絕對關節位置 + 1 夾爪）。

**評測協議**：
- 基準：LIBERO 四套件（Spatial / Object / Goal / LIBERO-10 Long）；LIBERO-Pro 的 task-perturbation 維度；附錄另報 LIBERO-Plus。
- 每任務評估 50 次、3 個隨機種子；推理時 DeltaWorld 採樣一條未來軌跡、Action DiT 生成 16 步 chunk。

**主要結果（Table 1，LIBERO）**：

| 方法 | 規模 | Spatial | Object | Goal | Long | 平均 |
|------|------|---------|--------|------|------|------|
| π0.5 | 3.5B | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.6 | 3.3B | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| Fast-WAM | 6B | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| LaWAM | 2.3B | 99.4 | 99.6 | 98.4 | 97.0 | 98.6 |
| **DeltaWAM (Ours)** | **0.725B** | 92.6 | 97.0 | 95.8 | 85.8 | **92.8** |

- DeltaWAM 延遲 142.1 ms，比 LaWAM（187）、F1（399）、Fast-WAM（486）、Cosmos-Policy（1413）、Motus（3231）、LingBot-VA（4482）等更低。
- 論文坦承：DeltaWAM 是**從零在 LIBERO 上訓練**（無大規模機器人預訓練 Emb. PT），絕對成功率低於最強預訓練基線，其核心優勢是「性能 × 極小算力足跡」。

**LIBERO-Pro 任務擾動（Table 2）**：DeltaWAM 平均 17.85%，而代表性基線接近 0%（論文稱基線高成功率可能來自對訓練場景的機械記憶）。

---

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 在 LIBERO 四套件上以 0.725B 實現 92.8% 平均成功率，同時 142 ms 延遲、3.86 GB 顯存。
- 對「任務擾動」（改變目標物/容器/空間關係/指令語義）表現出較強魯棒性（17.85% vs 基線 ~0%），說明其利用了可轉移的動態而非綁定場景佈局。
- 真實 PiPER 機械臂 pick-and-place 部署（論文 Figure 7）。

**不能做 / 失敗模式**：
- **精度接觸任務放置偏移**：LIBERO-Spatial / Long 的失敗多為「東西放上盤子但沒對準中心」——高度壓縮的 delta token 濾掉了精確對齊所需的細粒度空間細節（論文自述）。
- **依賴預測未來**：移除全部未來 token → 四套件成功率全為 0%，策略退化為無目的抓取。這既是優點（真正用了預測）也是風險（預測崩了就全崩）。
- **泛化範圍未證**：實驗限於 LIBERO 級別桌面操作 + 單臂 PiPER；論文未聲稱對移動/雙臂/人形有效，**不要外推**。
- **絕對成功率非 SOTA**：無大規模機器人預訓練，純 LIBERO 訓練。

### 6.1 隱含假設 (Hidden Assumptions)

- **假設「變化是低維、結構化的」**：論文核心前提是相鄰幀差異結構化且低維（物體位移 + 末端運動）。若場景存在大量無關動態（晃動背景、光照劇變、多主體擾動），此假設可能不成立，delta token 會被非任務相關變化污染。
- **假設凍結 DINOv3 特徵已足夠表達控制所需空間信息**：精度接觸失敗暗示 DINO patch 特徵的空間分辨率可能不足，但論文未系統驗證。
- **假設 Kinetics-700 預訓練的動力學可遷移到機器人操作**：跨域遷移效果未做消融（若無 DeltaWorld 預訓練會如何？作者未報）。
- **假設 oracle 分支可作為訓練時的上界信號**：k† 以軌跡誤差選出，但「預測得更準的未來」是否等同於「更利於動作」未經獨立驗證。
- **假設 LIBERO-Pro 的低基準分反映「機械記憶」**：這是作者的解釋性推斷，未提供直接的記憶 vs 泛化證據。

---

## 7. 與相關工作對比 (Comparison)

| 方法 | 未來表示 | 預測空間 | 生成方式 | 預測用途 | 適用場景 |
|------|----------|----------|----------|----------|----------|
| Diffusion Policy / π0 / π0.5 | 無顯式未來 | — | 反應式 / flow | 無 | 通用反應式策略 |
| Cosmos-Policy / Motus / LingBot-VA | 幀序列 / 完整場景 | 像素或完整 latent | 擴散迭代 | 測試時顯式 | 高成功率、高算力 |
| F1 / Fast-WAM | 幀 / 輔助 | 像素 | 擴散 / 輔助損失 | 部分顯式 | 中等效率 |
| LDA-1B | 未來 DINO 狀態 | DINO latent（完整） | 聯合去噪 | 測試時 | latent WAM |
| JEPA-WAM | 未來 V-JEPA latent | latent（完整） | — | 僅訓練輔助 | 語義世界模型 |
| LaWAM | chunk 級 DINO 子目標 | latent（完整） | 需額外 latent action 蒸餾 | 測試時 | 高效 latent WAM |
| **DeltaWAM** | **幀間變化（delta token）** | **DINO 差分，低維** | **單次前向（非迭代）** | **測試時顯式** | **效率 × 動態先驗** |

**共同點**：多數以 DINO/V-JEPA 特徵空間為世界模型狀態空間。**DeltaWAM 的差異**：不預測「完整未來狀態」，而預測「狀態之間的變化」——把濾掉靜態共享上下文當作設計目標，而非副產品。

**面試 Tip**：被問到「這篇和 LaWAM / LDA-1B 差在哪」時，一句話答：**「它們在 latent 空間逐幀預測完整未來狀態；DeltaWAM 只預測幀間變化（delta token），把未來 token 壓縮約 1024×，反而讓策略更依賴動態而非場景——所以更省、更抗任務擾動。」**

---

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做「世界模型 + VLA / WAM」的研究者，尤其關心 latent 世界模型的狀態空間設計與多模態未來預測。
  2. 要評估「把 WAM 部署到有限算力平台（單卡、邊緣）」可行性的工程師——本文的延遲/顯存/算力數字是直接可用的參考。
  3. 對「表徵壓縮如何影響策略魯棒性」感興趣的人——LIBERO-Pro 的擾動實驗與 delta token t-SNE 分析（Figure 5）值得細看。
- **建議章節路徑**：先讀 §1 Introduction（抓住 change-centric 動機）→ §3.1–3.3（delta token + Gated AdaLN 條件化 + 兩階段訓練）→ §4.4（消融：完整未來 DINO vs delta token 的 92.8% vs 79.0% 是全文最有說服力的證據）→ 可跳過 §2.2 的長 related work（可從 §7 表格速覽）。
- **不值得精讀的理由**：若你不做機器人學習、或你只想要最高成功率而不管算力，本文的 92.8% 低於 π0.5/LaWAM 等，讀摘要即可；若你已熟悉 DeltaWorld/Delta Token 與 flow-matching 動作專家，本文增量主要在「把 delta token 接進 WAM 並加語言調節」。

---
[← Back to Theory](./README.md)

**關鍵引用**
- 論文: https://arxiv.org/abs/2609.33177 (arXiv:2609.33177v1, cs.RO, 2026-09-27)
- 代碼（聲稱）: https://github.com/deltawam/DeltaWAM
- 骨幹: DeltaWorld — "A frame is worth one token: efficient generative world modeling with delta tokens" (Kerssies et al., 2026)
- 基準: LIBERO (Liu et al.) / LIBERO-Pro (Zhou et al., 2026) / LIBERO-Plus (Fei et al., 2026)
