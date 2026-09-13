# 冻结世界模型的内部预测状态里，已经写好了失败信号 (FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-13
>
> **论文**: FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model (Haoran Pei, Mingrui Luo, Senbao Wang, Jie Guo, Sheng Zhong, Ruixi Ci — 中科院自动化所)
> **链接**: [arXiv:2609.11445](https://arxiv.org/abs/2609.11445) · [HTML](https://arxiv.org/html/2609.11445v1) · [Code](https://github.com/HaoranPei-casia/FARM)
> **核心定位**: 不训练任何监控网络、不微调世界模型骨干，只用 33,985 个参数的 readout 去"读"冻结 VLA-JEPA 的预测状态，就拿到 85.68/88.59 的池化 AUROC/AUPRC，并在 3 个真实机器人平台上完成跨策略、跨平台迁移。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 冻结预训练机器人世界模型的内部预测状态（WM state）中，已经**可直接解码**出失败信息；一个 34K 参数的浅层 readout 就能把它读出来 |
| 適合精讀 | 你在做世界模型 / VLA 部署监控、需要在线失败检测、或想评估"表示复用 vs 专用监控模型"这条路线时重点看 §2 §4 §5 |
| 可以跳過 | 你只关心策略本身如何提升成功率、或关心触觉感知建模 —— 这篇完全不碰动作生成 |
| 落地可行性 | 高：推理侧只增加 0.2256 ms CUDA 延迟（不含世界模型前向），且支持"换机器人只重训 readout" |
| 主要風險 | 前提是你**已经拥有**一个世界模型并能拿到其内部状态；且训练/适配都需要带 outcome 标签的轨迹 |

💡 **X-Ray 开场**
这篇论文问了一个很朴素的问题：以前做机器人失败监控，要么训一个专门的监控模型（学一套动力学），要么从预测误差、不确定性、OOD 这类"代理信号"里推风险；但这些代理信号不等于失败——策略可以一边很自信、很时间一致，一边原地不动毫无进展。
作者的假设是：既然世界模型为了预测未来，内部状态已经把"观测历史 + 动作效果 + 任务上下文 + 状态演化"揉成了一个 summary，那么**失败信息可能已经在那里面了**，只是没人去读。于是他们冻结整个 VLA-JEPA，只挂一个 34K 参数的 readout，看看能不能解码出逐步失败分数。
结论对做 VLA 部署的人意味着：世界模型不只是"用来规划/控制的模块"，它顺带就是一个**免费的可复用监控接口**。

📍 **研究全景时间线**

```
2018 World Models / 2019 PlaNet        → 潜空间动力学 + 想象控制
2023 DreamerV3 / 2022 TD-MPC          → 潜空间预测规模化、任务导向动力学
2024-25 DINO-WM / LaDi-WM / V-JEPA2-AC → 在预训练表示空间里做动作条件预测
2025 SAFE (NeurIPS)                   → 首个证明：VLA 策略特征可直接被轻量 readout 解码出成功/失败
2025-26 Foresight / FoMo-FD / ContactGuard / Model-Based Runtime Monitoring
                                       → 为监控专门学一套潜动力学 + 分类器（重、且要另训）
[2026-09] FARM ← 当前位置            → 反其道：不训动力学，只读冻结 WM state
局限：依赖已有世界模型 + 需要带标签轨迹
```

## 1. 核心架构/方法总览 (Overview / Architecture)

FARM 的全部系统就三块：冻结的世界模型、一个 32 维瓶颈的 readout、一个因果 risk 聚合规则。没有任何一块是为了"监控"而重新训练的。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 是否训练 | 频率/时序 |
|------|------|------|----------|-----------|
| V-JEPA2 encoder | 观测历史（到当前步 t） | world-state tokens | ❄️ 冻结（含预训练） | 每个 replanning step |
| VLA-JEPA predictor（12 层，temporally causal） | world-state tokens + 任务条件 latent-action tokens | 第 12 层输出（未归一化） | ❄️ 冻结 | 每个 replanning step |
| **WM-state 抽取** | predictor 第 12 层输出，**去掉 action-conditioning tokens** | H^WM ∈ R^{768×1024} | ❄️ 无参数（纯抽取） | 每个 replanning step |
| **FARM readout** | H^WM（768 个 token × 1024 维） | z^F ∈ R^32 → step 分数 s_t ∈ [0,1] | ✅ 唯一训练部分（33,985 参数） | 每个 replanning step |
| 因果 risk 聚合 | s_{1..t} | q_t = max_{τ≤t} s_τ | ❌ 无参数（只存 1 个标量） | 每个 replanning step，递归更新 |

关键对比：SAFE 读的是**策略侧**的高层特征，FARM 读的是**世界模型侧**的预测状态；Foresight / FoMo-FD / ContactGuard 都是"另训一套 latent dynamics + 分类器"，FARM 则把监督完全限制在下游 detector 上。

### 1.2 关键机制 (Key Mechanism)

- **为什么冻结骨干**：如果把骨干也微调了，就无法区分"失败信息本来就存在于预训练表示里"还是"被失败监督新学出来的"。冻结骨干 = 把实验变成一个可证伪的表征可及性（probe）测试。
- **为什么只取第 12 层、且在 LayerNorm 与 output projection **之前**抽**：预归一化状态保留了更原始的 token 级预测结构；作者还对比了其他抽取位置，full WM-state 读出来最强。
- **为什么要丢掉 action-conditioning tokens**：监控关心的是"世界状态演化得好不好"，动作 token 更像是查询/条件而非状态本身，保留会引入动作分布偏置。
- **为什么 shared projection 且只有 32 维**：W_p 在 token 之间共享 → 参数量不随 token 数（768）增长；32 维瓶颈是**故意的**——逼 readout 只能"选择和组合已有信息"，而不是用失败监督重新拟合一个新预测模型。
- **为什么用 attention pooling**：不同执行状态需要强调不同的预测 token（如接触时刻 vs 自由移动），但注意只在当前 step 内做，不跨时间混合，保证时序因果。
- **任务均衡采样**：k ~ Unif(K_tr) → i ~ Unif(D_k) → t ~ Unif(T_i)，让每个任务等权，避免长 rollout 因为状态多而主导损失。

⚡ **Eureka Moment**：失败信息不需要被"引入"——一个为预测而训练的世界模型，其内部状态已经按"任务会成功还是失败"的方式组织好了，你需要的只是一个足够小的解码器，而不是一个更强的监控模型。

### 1.3 信息流/架构图 (Flow / Diagram)

```
观测历史 x_{≤t} ──► [V-JEPA2 encoder] ❄️
                          │
                          ▼ world-state tokens
        + 任务条件 latent-action tokens
                          │
                          ▼
              [VLA-JEPA predictor × 12] ❄️
                          │ 第12层输出（未归一化）
                          ▼
        [去掉 action tokens → H^WM ∈ R^{768×1024}]  ← 抽取，无参数
                          │
              ┌───────────┴───────────┐  每 replanning step 内部
              ▼                       │
   u_j = GELU(LN(W_p h_j + b_p))      │  (共享 W_p, 32×1024)
   α_j = softmax_j(aᵀ u_j)            │
   r   = Σ_j α_j u_j                  │  attention pooling
   z^F = GELU(W_f r + b_f)  ∈ R^32    │
   s_t = σ(w_oᵀ z^F + b_o) ∈ [0,1]    │
              │
              ▼
   q_t = max(q_{t-1}, s_t)  ──►  阈值判定：继续 / 干预 / 恢复 / 重置
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
s_t = σ(w_oᵀ · MLP( Σ_j softmax_j(aᵀu_j) · u_j )),  u_j = GELU(LN(W_p h_j + b_p))
```

目标：在骨干参数 φ 完全冻结的前提下，只优化 readout 参数 θ，让每个 replanning step 的失败分数 s_t 尽量接近轨迹级标签 y。

**（a）冻结抽取 + 逐步打分**

```
H^WM_{i,t} = sg[ F_φ(x_{i,≤t}) ]        # sg = stop-gradient，φ 永不更新
s_{i,t}    = G_θ(H^WM_{i,t}) ∈ [0,1]    # θ 是唯一可训练参数集
```

**（b）readout 内部（省略 i, t 下标）**

```
u_j = GELU( LN( W_p h_j + b_p ) )       # h_j ∈ R^1024, W_p ∈ R^{32×1024}
α_j = softmax_j( aᵀ u_j )               # a ∈ R^32
r   = Σ_{j=1}^{768} α_j u_j  ∈ R^32
z^F = GELU( W_f r + b_f )   ∈ R^32
s   = σ( w_oᵀ z^F + b_o )   ∈ [0,1]
```

**（c）任务均衡的 BCE 损失**

```
L_FARM(θ) = E_{k,i,t}[ ℓ_BCE(s_{i,t}, y_i) ]
ℓ_BCE(s, y) = -y·log s - (1-y)·log(1-s)
采样: k ~ Unif(K_tr) → i ~ Unif(D_k) → t ~ Unif(T_i)
```

**（d）因果轨迹风险（只向前看）**

```
q_{i,t} = max_{τ ≤ t} s_{i,τ}           # 递归实现: q_{i,t} = max(q_{i,t-1}, s_{i,t})
```

| 符号 | 含义 | 取值/维度 |
|------|------|-----------|
| F_φ | WM-state 抽取器（V-JEPA2 encoder + VLA-JEPA predictor 第 12 层） | 冻结 |
| H^WM | 预归一化预测状态（去 action token 后） | 768 × 1024 |
| W_p / a | 共享 token 投影 / attention 向量 | 32×1024 / 32 |
| z^F | 失败感知瓶颈表示 | 32 |
| s_t | step-wise 失败分数 | [0,1] |
| q_t | 因果轨迹风险（单调不减） | [0,1] |
| y | 轨迹结果标签 | 1 = 失败，0 = 成功 |

参数账本（论文 §III-C）：token 投影 32,800 + LayerNorm 64 + attention 向量 32 + 隐层 1,056 + 输出头 33 = **33,985**。

**直觉**：它不学动力学，只学"在已有的 768 个预测 token 里，哪些 token 值得看、怎么看"。attention 权重 α_j 相当于一个内容相关的软选择器——在正常执行时可能平均分配，在失败临近时把权重压到少数"异常 token"上。而 32 维瓶颈则把"解码失败"这件事的容量上限卡死，让结果只能归因于表示本身。

> 符号与本文保持一致：H^WM（world model state）、z^F（failure-aware representation）、sg（stop-gradient）。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设我们手上有一条轨迹，T=4 个 replanning step，readout 已经训好了（截取论文 Fig. 4 三类失败的典型形状量级，数值为演示用）：

| t | 观测到的情况 | s_t（readout 输出） | q_t = max(s_{≤t}) |
|---|--------------|--------------------|--------------------|
| 1 | 正常接近目标 | 0.12 | 0.12 |
| 2 | 正常抓取，接触成功 | 0.09 | 0.12 |
| 3 | 物体滑落，出现异常 | 0.71 | 0.71 |
| 4 | 空手移动到目标位 | 0.88 | **0.88** |

轨迹风险 q_4 = 0.88 → 若阈值设为 0.5，系统在 t=3 就已经越线，比"整条轨迹跑完再判断"提前了一步，有机会在 t=3 触发恢复/重置。

对比另一种聚合方式：
```
mean(s)   = (0.12+0.09+0.71+0.88)/4 = 0.45   ← 低于阈值，漏报！
max(s)    = 0.88                              ← 保留短暂但强烈的失败证据
last(s)   = 0.88                              ← 本例巧合；但若末端"看起来正常"会漏报
```

为什么用 max 而不是 mean：失败往往是**短暂但强烈**的事件（一次碰撞、一次滑落），随后策略可能"看起来正常地"继续执行。mean 会被前面的低分稀释，max 只需存 1 个标量、且天然因果（只用 ≤ t 的分数）。

再看部分历史判别（论文 Fig. 5，池化指标）：

| 已观测比例 r | 截止步 t_i(r) | Pooled AUROC | Pooled AUPRC |
|--------------|---------------|--------------|--------------|
| 25% | ⌊0.25·T_i⌋（至少 1） | 75.07 ± 0.73 | 80.75 ± 0.16 |
| 50% | ⌊0.50·T_i⌋ | —（论文仅给曲线趋势） | — |
| 75% | ⌊0.75·T_i⌋ | 81.63 ± 0.40 | 85.53 ± 0.65 |
| 100% | T_i | 83.41 / 86.80（= 75% 高出 1.78/1.27 反推） | — |

读法：只用 75% 的历史，就已经拿到接近满分历史的判别力，且**不需要为不同前缀重训**。这对在线部署很关键——监控器必须在轨迹还没跑完时就给分。

## 4. 工程视角 (Engineering View)

| 指标 | 数值 | 工程含义 |
|------|------|----------|
| Detector 侧增量延迟 | 0.2256 ms 均值 CUDA，P99 0.2393 ms | 相对典型 VLA 推理（几十毫秒/几十 Hz 控制）几乎可忽略 |
| Detector 吞吐 | 4,432.8 steps/s（batch=1） | 单卡可同时监控多条 rollout 流 |
| Readout 参数量 | 33,985 | 可以忽略显存占用，甚至可作为 checkpoint 附在部署包里随策略分发 |
| 测时口径 | 70 个 [1,768,1024] 张量已驻留 RTX 5090；500 次 warm-up + 10,000 次采样，CUDA event 同步 | **不含**世界模型前向、WM-state 抽取、预处理、传输、控制器开销 |
| 存储开销 | 因果 risk 只需 1 个标量/轨迹 | 可长期在线运行，不膨胀 |

几个部署约束值得注意：

- **前置依赖**：这 0.2256 ms 的前提是"WM state 已经可用"。也就是说你必须已经在跑世界模型；如果世界模型本身是你为了监控才引入的，成本要重新算。
- **状态抽取是有侵入性的**：需要拿到 predictor 第 12 层、未归一化的中间张量，并去掉 action tokens。这要求你对世界模型实现有控制权（repo 明确说"frozen VLA/V-JEPA feature extraction 在包边界之外"）。
- **两种部署模式**（同一架构、同一表示、同一打分规则，只差是否用目标域标签更新那 34K 参数）：
  - **Fixed-readout zero-shot**：源域训练好的 θ_S* 直接用；目标域轨迹既不参与训练也不参与模型选择。
  - **Readout-only adaptation**：从 θ_S* 初始化，在目标域标签上继续训 readout；φ 与 WM-state 接口不动。
- **阈值策略是下游的**：论文只给分数，不给"该继续还是该干预"的策略——这是明确划出去的边界。

## 5. 数据与评测 (Data & Eval)

**仿真基准**：LIBERO 10 任务，500 条轨迹（246 成功 / 254 失败），每任务 50 条。作者把 LIBERO-10 里一个"执行策略几乎从不失败"的任务替换成 LIBERO-Goal 的 drawer-and-placement，最终 = 9 个 LIBERO-10 + 1 个 LIBERO-Goal。

**任务划分（在 detector 训练前就固定）**：
- S1–S7 = source / Seen；U1–U3 = strict-Unseen。
- 固定切分比较实验：每源任务 30 TRAIN / 20 SEEN_EVAL → 210 / 140 条。
- 表征与因果分析：全部 350 条 source 轨迹走 trajectory-grouped 五折 OOF。
- **防捷径**：每个任务只取所有轨迹共有的前 T_k 步（冻结公共 horizon），避免"轨迹长度 / 终止时刻"变成分类捷径。

**主结果**：

| 设置 | 指标 | 结果 |
|------|------|------|
| 五折 OOF（7 任务） | Pooled AUROC / AUPRC | **85.68 / 88.59** |
| 10 任务 Seen（15 个匹配基线） | 四项 Seen 指标 | 全部第一；比 SAFE-MLP 高 **5.42 / 4.88**（Macro AUROC/AUPRC） |
| strict-Unseen 零样本 | — | STAC-Single 第一（任务无关的动作一致性分数）；FARM 高于 SAFE-MLP/LSTM |
| strict-Unseen + Adapt-35（每任务每外折 35 条适配轨迹） | 四项聚合指标 | 全部领先 SAFE-MLP/LSTM，且超过零样本 STAC-Single |
| 源任务覆盖度（FARM-Expanded，446 条） | Macro 从 1→7 任务 | 62.55/56.78 → 85.43/85.67；Pooled 61.71/52.42 → 89.73/88.06 |

**对照实验（表征可及性）**：7 个低阶统计量（全局均值、标准差、RMS 幅度、token 均值的时序 delta 均值/最大值、末端 token 均值范数、token 方差）。最强的单统计量控制是"时间 delta 均值"，7 统计量线性拟合次之；**full WM-state readout 优于两者**——说明增益不只是粗粒度隐状态统计。

**真实机器人（4 组策略/平台）**：

| 平台 / 策略 | 轨迹数 | 备注 |
|-------------|--------|------|
| PIPER X / π*0.6 | 497 | 外部策略；示教任务如 plug insertion |
| PIPER X / VLA-JEPA | 100 | 任务适配后冻结的自主策略；duck manipulation |
| SO-101 / Eval-RL | 70 | 自研 RL 执行策略；blue-cube grasping |
| Franka / π0-FAST-DROID | SAFE 公开 13 任务 778 rollouts | 复用公开人群 |

- FARM-Expanded 在四组零样本设置里都优于 FARM-Core（同表示、同架构，只差源域监督人群 → 迁移取决于"源 readout 怎么训"）。
- Readout-only 适配在每一组都优于 Core 初始化；完整 Franka 人群上 trajectory-level 五折 OOF 到达 **75.78 ± 1.08 AUROC / 74.31 ± 0.96 AUPRC**。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**：
- 多任务逐步失败打分 + 因果轨迹 risk（跨 7 个源任务、10 任务基准）。
- 跨策略 / 跨平台 / 跨相机 / 跨视觉域迁移，只重训 34K 参数。
- 在只观测到 75% 历史时就恢复接近满历史判别力，无需前缀专用重训。
- 亚毫秒级 detector 侧开销，可在线长期跑。

**不能做 / 明确的边界**：
- 论文自述需**访问世界模型内部状态**，且训练/适配需要**带 outcome 标签的轨迹**（没有标签就只能零样本迁移，而零样本在 strict-Unseen 上明显更弱）。
- **不做干预决策**：只输出分数，怎么和 intervention / recovery 策略耦合留作未来工作。
- 跨域零样本并非一致有效：任务语义迁移（strict-Unseen）比匹配 Seen 更难；作者明确写"Zero-shot transfer is not uniform"。
- 失败信息在 predictor 各层的分布未探明（未来工作）。
- 评测域以桌面操作（抓取、插接、抽屉）为主，**不能外推**到移动、双臂协同、人形等未测场景。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设 1：轨迹级标签可以逐帧继承**。每条轨迹的 y 广播到该轨迹所有有效 WM state。这默认"失败在整条轨迹上都可辨识"，但末端才暴露的失败会让早期帧被误标为失败——论文没有专门分析这个标签噪声。
- **假设 2：WM state 的语义在冻结前提下跨域稳定**。VLA-JEPA 是在特定数据/任务上预训练的，其状态空间是否在 SO-101、Franka 这类差异很大的平台上仍线性可分，作者用 adapt 之后的提升间接回答了，但零样本之所以不完美，恰恰是这个假设受损的证据。
- **假设 3：公共 horizon 截断不损伤失败判别**。只取所有轨迹共有的前 T_k 步，避免了长度捷径，但也可能切掉"晚期才出现的失败"的证据。
- **假设 4：world model 状态抽取路径是"无参数"的，因此冻结算得清**。实际上抽取含明确的工程选择（第 12 层、去 action token、预归一化），这些选择本身会影响结果——论文用抽取位置对比做了部分控制，但没有穷举。
- **假设 5：0.2256 ms 的结论只在"WM state 已在 GPU 上"时成立**。这是被明确声明的、但容易被引用时忽略的前提。

## 7. 与相关工作对比 (Comparison)

| 方法 | 表示来源 | 风险信号 | 是否需另训动力学 | 适用场景 |
|------|----------|----------|------------------|----------|
| **FARM（本文）** | 冻结 VLA-JEPA 内部预测状态 | 轻量 readout 直接解码 | ❌ 否（只训 34K readout） | 已有世界模型的部署监控 |
| SAFE (NeurIPS 2025) | VLA 策略高层特征 | MLP / LSTM readout | ❌ 否 | 无世界模型、只有策略特征 |
| Model-Based Runtime Monitoring (ICRA 2024) | 交互模仿学习中联合学潜动力学 | 学到的动力学 + 分类器 | ✅ 是 | 交互式模仿学习 |
| Foresight (2026) | 动作条件世界模型 latent + 因果 Transformer | 长时域失败检测 | ✅ 是 | 长时域操作 |
| FoMo-FD (2026) | flow-matching 世界模型 | 视觉-动作 nonconformity | ✅ 是（flow matching 世界模型） | 手术机器人模仿策略 |
| ContactGuard (2026) | 动作条件潜世界模型 | 预接触不良后果预测 | ✅ 是 | 接触前预警 |
| 距离/OOD 类基线（Mahalanobis、kNN、PCA-KMeans、RND、LogpZO） | 策略特征 | 距离对比 / novelty | ❌ 否 | 无需标签，但代理信号 ≠ 失败 |
| 动作不确定性类（Action Variance、Cluster Entropy、STAC） | 动作 chunk 分布 | 方差 / 熵 / MMD 一致性 | ❌ 否 | 无需标签、任务无关（STAC-Single 在 strict-Unseen 最强） |

**面试 Tip**：被问"FARM 和 SAFE 有什么区别"时，一句话答——**"两者都在问同一个问题（现有表示里是否已含失败信息），但 SAFE 读的是策略特征，FARM 读的是世界模型的预测状态；FARM 的独特点是监督被严格限制在 34K readout 上、骨干完全冻结，所以它是更强的'表征可及性'证据，代价是必须先有世界模型。"**

## 8. 精讀建議 (Reading Guide)

**值得精讀原文的人**：
1. 正在做 world-model-driven VLA 部署、需要在线失败/异常监控的研究者与工程师——重点看 §III（方法）与 §IV-B/IV-C（表征可及性与匹配基线）。
2. 要在多机器人平台间迁移监控能力、不想为每个新平台重训大模型的工程团队——重点看 §IV-D（真实机器人迁移）与 §IV-E（延迟）。
3. 对"探针 / 冻结特征复用"这类方法论感兴趣、想找表征分析范式的人——重点看 §IV-B 的 7 统计量控制实验。

**建議章節路徑**：先讀 §III（方法，含 Fig. 1 的架构图）→ 再看 §IV-B / §IV-D（表征证据 + 真实迁移）→ 可跳 §IV-A 的基线实现细节（除非你要复现 15 个基线）→ §V 结论 30 秒扫过。

**不值得精讀的理由**：如果你不做机器人学习、或者已经很熟悉 SAFE 式的轻量 readout 思路、或者你根本拿不到世界模型的内部状态，那么读摘要（85.68/88.59 池化指标 + 34K 参数 + 0.2256 ms）就够了——这篇的技术贡献是"把表示复用推到世界模型内部状态"，而不是新算法本身。

---
[← Back to Theory](./README.md)

**关键引用**
- FARM: arXiv:2609.11445 · [HTML](https://arxiv.org/html/2609.11445v1) · [Code (MIT)](https://github.com/HaoranPei-casia/FARM)
- VLA-JEPA: arXiv:2602.10098（被复用为冻结骨干）
- V-JEPA 2: arXiv:2506.09985
- SAFE: NeurIPS 2025, doi:10.52202/085713-1337（最接近的表征复用先例）
- LIBERO: NeurIPS 2023, pp. 44776–44791（仿真基准）
- π*0.6: arXiv:2511.14759 · π0: arXiv:2410.24164 · FAST: arXiv:2501.09747 · DROID: RSS 2024
