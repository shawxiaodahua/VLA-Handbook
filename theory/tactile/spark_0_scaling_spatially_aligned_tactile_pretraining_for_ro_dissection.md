# Spark-0：面向机器人灵巧操作的空间对齐触觉预训练 (Spark-0: Scaling Spatially Aligned Tactile Pretraining for Robot Dexterous Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: Spark-0: Scaling Spatially Aligned Tactile Pretraining for Robot Dexterous Manipulation
> **链接**: https://arxiv.org/abs/2610.22342
> **核心定位**: 把触觉从“策略输入”扩展为“可扩展的视觉表征预训练监督”，并用独立触觉通路条件动作专家；在接触密集灵巧操作上同时超过 vision-only 与触觉基线，且一个 checkpoint 支持有/无触觉部署。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 触觉不仅能当推理输入，还能当预训练监督：train-only 的 contact map + 3D 几何对齐教视觉 token “哪里接触”，独立触觉通路教动作专家“接触如何演化”。 |
| 适合精读 | 做触觉预训练、灵巧操作 VLA、跨 embodiment 迁移的人；重点看 §4.2、§4.3、§6.3。 |
| 可以跳过 | 只关心纯视觉 VLA 架构或大模型 scaling 的人，距离中等；读 §1 和 §6.2 即可。 |
| 落地可行性 | 中：算法可复用，单 checkpoint 双模式对部署友好；但 1,000 小时同步触觉数据 + 96 GPU 预训练是硬门槛。 |
| 主要风险 | 真实机器人每任务仅 10 trials，52% vs 36% 的差距未给置信区间；contact map 是手级区域而非精确接触点，精细插入仍弱。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？通用 VLA 靠大规模视觉预训练变强，但接触密集灵巧操作里力、合规、初始滑移很难从图像看见，手还会挡住接触区。

发现了什么？触觉不必在推理时进入 VLM：把触觉做成训练时的空间对齐监督，视觉 token 就能在无触觉部署时保留“哪里接触”的先验；在线触觉则走独立通路，只条件动作专家。

对 VLA 研究者意味着什么？触觉可以是一种“预训练信号”而不只是“推理输入”；但这依赖 1,000 小时级同步触觉数据、专用手套布局和 96 GPU 级预训练，且当前监督粒度是手级区域。

📍 **研究全景时间线**

```
2017 GelSight ── 2023 RT-2 / ACT ── 2025 EgoVLA / VITRA ── 2026 FTP-1 / STAR ── 2026 Spark-0（本文）
  触觉传感器       VLA/动作分块         人类视频 VLA            触觉基础策略/稀疏触觉 VTLA   触觉预训练 + 双角色
局限：单 embodiment（Gen 1 Prototype）；contact map 为手级区域；真实机器人每任务 10 trials；统计不确定性未报告
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| VLM backbone：Qwen3-VL-4B | head 384×384、wrist 256×256、language、离散化 state | hidden states h1..hL、64 个 action queries | 每推理步 | 联合训练；推理保留 |
| Geometry alignment head | layer 25 视觉 token、冻结 DA3 特征 | cosine 对齐损失 L_geo | 每样本每视图 | train-only；推理丢弃 |
| Contact alignment head | head view 的 layer 25 token | contact logits、L_con | 每帧 | train-only；推理丢弃 |
| Tactile encoder：regional | 每手 6 区域 × 20 帧力历史 | 12 个 regional force tokens | 20 帧 ≈ 333 ms @60 Hz | 联合训练；推理保留，缺失用 placeholder |
| Tactile encoder：3D | 每手 320 taxels 的 xyz + pressure，当前帧 | 12 个 3D tokens | 当前帧 | 联合训练；3D 流 dropout 0.15；缺失用 placeholder |
| Action expert：36 层 DiT，width 1024 | 3 帧 state、24 触觉 token、50 行 noisy action、VLM KV | 50×58 action chunk | 每 replan | flow matching 训练；推理 4 Euler steps |
| RTC 部署 | action chunk | 异步重规划 | 剩余 26 步请求；每 24 步重规划；名义延迟 8 步 | 仅推理 | 训练无关 |

### 1.2 关键机制 (Key Mechanism)

- **触觉的两个角色被解耦**：训练时，手套 taxel 生成 head-view contact map，和冻结 DA3 的 3D 几何特征一起监督 VLM layer 25 的视觉 token；推理时这两个 head 全部移除。在线触觉则只进入 action expert，raw touch 从不进入 VLM。这样同一个 checkpoint 既能 vision-only 部署，也能 tactile-conditioned 部署。
- **为什么 contact map 是 train-only**：它提供“哪里接触”的空间监督，但不作为策略输入。部署时没有触觉传感器，视觉 token 仍保留这种先验；这解释了 Ego Humanoid 上 vision-only 成功率从 52.45% 升到 71.72% 的消融结果。
- **为什么 raw touch 不进 VLM**：保持 VLM 的视觉-语言分布不被高频、低维、易缺失的触觉信号扰动；同时让有触觉/无触觉样本共享同一 backbone，一个 checkpoint 覆盖两种模式。
- **为什么触觉 token 是 24 个**：12 个 regional force token 编码 20 帧（≈333 ms）力历史，捕捉接触演化；12 个 3D token 编码当前帧 320 taxels/hand 的 xyz 与压力，捕捉接触几何。两种时间尺度互补。
- **为什么非对称 bias + tactile shortcut**：非对称 bias 只限制 tactile query 对 State/Action key 的注意力，防止触觉被状态/动作淹没；tactile shortcut 在训练时偶尔把 State/Action 行的 cross-attention 输出置零，迫使视觉-语言上下文通过触觉行流向动作，防止 action expert 绕过触觉路径。
- **为什么 learned placeholder**：缺失传感器不等于零压力。用共享 learned placeholder `e∅` 填充缺失流，让 vision-only 和 tactile 样本可以同 batch 训练。

⚡ **Eureka Moment**：触觉的两种角色可以解耦——训练时用触觉教视觉“哪里接触”，推理时用触觉教动作“接触如何演化”；两者共享一个 checkpoint，但互不绑架。

### 1.3 信息流/架构图 (Flow / Diagram)

```
HumanTouch / same-glove robot / sim
  RGB (head, wrists) ──► Qwen3-VL-4B ──► h^(1..36)
        │                                   │
        ├─ 冻结 DA3 教师 ──► L_geo            ├─ layer 25 ──► contact head ──► L_con
        └─ glove taxels ──► contact map ─────┘
  state (3 frames) ─┐
  tactile (24 tokens)├─► Joint-Stream DiT (36 blocks) ──► action chunk 50×58
  action queries ────┘        even blocks: cross-attend to VLM
                              odd blocks: masked self-attn over State/Tactile/Action
                              AdaLN(c) 由 flow time + regional force window 调制
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
L = L_act + 0.2·L_geo + L_con
```

更直觉地说：`触觉 → contact map 监督视觉 token（train-only）+ 触觉 token 条件动作专家（inference）`。

**目标**：让视觉 token 编码“哪里接触”，让动作专家编码“接触如何演化”，同时不要求推理时一定有触觉。

**统一状态与动作空间**：

```
s_t^k = [ ρ(C_t^{-1} T_t^k), q_t^k ]
a_{t+h}^k = [ ρ((T_t^k)^{-1} T_{t+h}^k), q_{t+h}^k ],  h = 0..H-1
```

| 符号 | 含义 |
|------|------|
| k ∈ {L, R} | 左手/右手 |
| T_t^k ∈ SE(3) | 末端执行器位姿 |
| q_t^k ∈ R^20 | WUJI Hand 2 的 20 个关节角 |
| C_t | head camera 位姿 |
| ρ: SE(3) → R^9 | 平移 p + 6D rotation r(R)，ρ^{-1} 正交化恢复旋转 |
| H = 50 | action chunk 长度 |
| 58 | 2 × (9 + 20) 维 state/action |

**Flow matching 动作损失**：

```
a_τ = (1 - τ)·ε + τ·a
L_act = E_{a,τ,ε} || v_θ(a_τ, τ, C) - (a - ε) ||_2^2
```

| 符号 | 含义 |
|------|------|
| a | 归一化后的目标 action chunk |
| ε | Gaussian noise |
| τ ∈ [0,1] | flow time；训练采样 τ = (0.999 - u)/0.999，u ~ Beta(1.5, 1) |
| C | 连续 state、layerwise VLM context、可选 tactile tokens |
| v_θ | 预测速度场 |
| 6 noise draws | 每个样本 6 次噪声采样 |
| 4 Euler steps | 推理积分步数 |

**几何对齐**：

```
L_geo = E_{b,v,j} [ 1 - cos( φ_geo(h_{b,v,j}^{(25)}), z_{b,v,j}^{geo} ) ]
```

| 符号 | 含义 |
|------|------|
| b, v, j | 样本、视图、视觉 token |
| h^{(25)} | VLM 第 25 层 token |
| z^{geo} | 冻结 DA3 教师的 patch 特征，重采样到 token grid |
| φ_geo | LayerNorm–MLP projector，hidden width 2048 |
| λ_geo = 0.2 | 总损失中的权重 |

**接触对齐**：

```
L_con = 0.1·L_focal + 0.1·L_dice
```

| 符号 | 含义 |
|------|------|
| L_focal | focal loss，exponent 2，处理稀疏正样本 |
| L_dice | soft Dice loss，处理区域重叠 |
| y ∈ [0,1] | head-view 12×12 token grid 上的软 contact target |
| 0.1 / 0.1 | 两项权重，论文固定设置 |

**触觉编码与缺失处理**：

```
e_{k,r}^{reg} = f_temp( ψ(F_{k,r,1:20}) )
e_{k,r}^{3d} = f_set( Pool_{i∈R_{k,r}} f_pt( Γ([x_i, ψ(p_i)]) ) )
ẽ_q = m_q·γ_q·e_q + (1 - m_q)·e∅
```

| 符号 | 含义 |
|------|------|
| F_{k,r,1:20} | 手 k、区域 r 的 20 帧力窗口 |
| ψ(p) = log(1+p)/s | 压力变换，s 自动估计，零压力有效 |
| R_{k,r} | 手 k、区域 r 的 taxel 集合 |
| Γ | 四频段 Fourier 特征 |
| γ_reg, γ_3d | 可学习标量门，初始 0.1 |
| m_q ∈ {0,1} | 流是否存在 |
| e∅ | 共享 learned placeholder |

**联合序列与条件**：

```
Z^(0) = E_s(s) || ẽ_reg || ẽ_3d || E_a(a_τ)
c = Embed(τ) + Embed(flatten(ψ(F_L), ψ(F_R)))
```

按论文 token 数相加：3 个 State + 12 个 regional + 12 个 3D + 50 个 Action = 77 个 token。AdaLN 条件 `c` 中，regional force window 是 2×6×20 = 240 维，经 zero-initialized MLP（hidden width 512）映射。

**非对称 bias**：

```
b(g) = -6·(1 - σ(g)),  g 初始化为 1.0
```

只加在 tactile query 对 State/Action key 的注意力上；反方向不加 bias，State/Action 可以无偏读取触觉 token。

**直觉**：L_act 学动作速度场；L_geo 让视觉 token 对齐 3D 场景结构；L_con 让视觉 token 聚焦交互区域；触觉 token 补足视觉遮挡下的压力变化。非对称 bias 和 tactile shortcut 都是为了防止 action expert 忽略触觉路径。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设例子**，不是论文实验，用于验算 flow matching 的 4 步 Euler 推理。

假设一维动作 `a = 0.8`，噪声 `ε = -0.2`，真实速度 `v* = a - ε = 1.0`。模型预测速度恒定为 `v = 0.9`。从 `a_0 = ε = -0.2` 出发，4 步 Euler，步长 `Δτ = 0.25`：

```
a_1 = -0.2 + 0.9×0.25 = 0.025
a_2 = 0.025 + 0.225 = 0.25
a_3 = 0.25 + 0.225 = 0.475
a_4 = 0.475 + 0.225 = 0.7
```

终点 `0.7`，与目标 `0.8` 相差 `-0.1`。若模型速度准确为 `1.0`，则 `a_4 = -0.2 + 1.0 = 0.8`，精确命中。这个例子说明：对恒定速度的直线 flow，4 步 Euler 本身没有离散化误差；误差全部来自速度估计。论文用 6 次噪声采样训练速度场、4 步推理，本质上是在“速度场精度”和“推理步数”之间取平衡。

再验算一个 contact loss 的小例子（假设 2×2 token grid）：目标 `y = [[1,0],[0,0]]`，预测 `p = [[0.8,0.2],[0.1,0.1]]`。Focal loss（exponent 2）逐元素约为 `0.0089, 0.0089, 0.00105, 0.00105`，平均约 `0.00498`；Dice 相似度 `2×0.8/(1.2+1)=0.7273`，Dice loss `0.2727`。总 contact loss `0.1×0.00498 + 0.1×0.2727 ≈ 0.0278`。这说明 Dice 项在稀疏接触图上占主导，focal 项负责压低易分负样本。

## 4. 工程视角 (Engineering View)

| 项目 | 论文设置 | 工程含义 |
|------|----------|----------|
| 预训练 | 400k optimizer steps，12 节点 × 8 GPU = 96 GPU，global batch 576 | 按配置计算约 2.3×10^8 次样本级前向（未计多视图/多帧展开）；不是轻量方案 |
| 微调 | 80k steps，8k warmup | 每个下游基准单独微调；复用预训练归一化统计 |
| 评测频率 | 每 10k steps 评测，每 5k steps 存 checkpoint | 训练中频繁评测，成本高 |
| 触觉时间窗 | 20 帧 @60 Hz ≈ 333 ms | 覆盖约三分之一秒的力历史；适合接触起始/滑移的短时演化 |
| 动作 chunk | H = 50 | 一次预测 50 步 58 维动作 |
| 推理 | 4 Euler steps | 比多步扩散更轻，但速度场误差会直接进入动作 |
| RTC 部署 | 剩余 26 步请求下一 chunk；每 24 步重规划；名义延迟 8 步 | 异步重规划掩盖部分推理延迟；控制频率论文未给，无法换算绝对延迟 |
| 触觉缺失 | learned placeholder；3D 流 dropout 0.15；质量检查：<25% finite 或 >85% 饱和则替换 | 部署时需要传感器健康监测；regional 流无 modality dropout，缺失鲁棒性需实测 |

【条件性分析】论文没有报告单步推理延迟、显存占用和控制频率，因此不能直接断言“可实时”。只能确认：action chunk 为 50 步，RTC 每 24 步重规划，名义延迟 8 步；若控制频率为 10 Hz，重规划周期约 2.4 秒；若为 30 Hz，约 0.8 秒。这两个频率都是假设，不是论文数据。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练数据

| 来源 | 小时 | 采样权重 | 视角 | 作用 |
|------|------|----------|------|------|
| HumanTouch | 1,000 | 0.35 | head + wrists | 人类全手接触监督 |
| EgoDex | 662 | 0.39 | head | 场景/物体/运动多样性 |
| Same-glove robot play | 20 | 0.09 | robot views | 目标 embodiment 与接触对齐 |
| Robot play，tactile removed | 24 | 0.05 | matched views | 场景匹配的视觉对照 |
| Simulation | 25 | 0.05 | simulated | 视觉域随机化 |
| Fourier ActionNet | 153 | 0.04 | head | 机器人任务多样性 |
| AgiBot World | 60 | 0.03 | head + wrists | 多视图机器人多样性 |

论文明确说明：**Exposure is set by sampling weight, not hours**。因此不能把权重当时长比例读。总时长约 1.9k 小时，HumanTouch 占 1,000 小时。

HumanTouch 采集硬件：每手 460 个传感点，遮掉 140 个弯曲区伪压力点，保留 320 个 taxels；25 关节手指跟踪、6-DoF 腕跟踪、head + 双 wrist 相机，全部 60 Hz 同步；100+ 任务、32 名操作者。手套可装到目标机器人手上，因此人手和机器人数据共享 taxel 布局。

### 5.2 评测设置

| 基准 | 任务数 | 每任务 trials | 触觉 | 指标 |
|------|--------|----------------|------|------|
| SparkArena-Sim | 7 | 50 | 无 | success rate / completion score |
| SparkArena-Real | 5 | 10 | 有 | success rate / completion score，paired initial conditions |
| Ego Humanoid short-horizon | 7 | 27 seen / 66 unseen | 无 | SR / PSR，seen/unseen 指视觉配置 |
| Ego Humanoid long-horizon | 5 | 27 seen / 66 unseen | 无 | SR / PSR |

【论文实测】SparkArena-Sim：Spark-0 36.86% SR / 39.74 completion，最强基线 FastWAM 28.29 / 33.51，高 8.57 / 6.23 点。SparkArena-Real：Spark-0 52.00 / 62.80，最强触觉基线 FTP-1 36.00 / 47.80，高 16.00 / 15.00 点。Ego Humanoid：Spark-0 在 8 个 aggregate 指标上均第一，四个设置均值 71.72%。

【条件性分析】SparkArena-Real 每任务 10 trials，5 个任务共 50 trials；52% 与 36% 相差约 8 次成功。论文没有给置信区间或显著性检验，因此“真实机器人上大幅领先”需要外部复现确认。SparkArena-Sim 每任务 50 trials，共 350 trials，统计上更稳，但它测的是 vision-only 策略，不直接验证在线触觉。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力/场景 | 论文证据 | 边界与失败模式 |
|-----------|----------|----------------|
| 无触觉部署的跨 embodiment 迁移 | Ego Humanoid 四个设置均值 71.72%，比 Visual PT 的 52.45% 高 19.27 点 | 仅在 Ego Humanoid 上验证；增益依赖 1,000 小时触觉预训练；100H 只到 59.42% |
| 接触密集真实操作 | SparkArena-Real 52.00% vs FTP-1 36.00% | 有在线触觉；Insert Test Tubes 所有方法 0% 成功；每任务仅 10 trials |
| 长程多阶段任务 | Ego long-horizon seen 68.15% vs FastWAM 49.63%（最强基线，+18.52 点）；unseen 61.82% vs VITRA 39.70% | 长程 unseen 仍有约 38% 失败；PSR 80.53 说明部分完成多、完整完成难 |
| 单 checkpoint 双模式 | 同一权重支持 vision-only 与 tactile-conditioned | 依赖 placeholder 训练分布；regional 流无 modality dropout；传感器故障模式未充分验证 |
| 精细插入/高精度接触 | Insert Test Tubes completion 31.00 最高 | 三管全部插入无人完成；contact map 是手级区域，不是精确接触点 |

### 6.1 隐含假设 (Hidden Assumptions)

1. **人手与机器人手共享触觉布局意味着接触监督可迁移**。论文用 same-glove robot 数据（20 小时，权重 0.09）对齐，但人手 MANO 与 WUJI Hand 2 的形态差异、retargeting 误差没有量化。
2. **contact map 的手级区域粒度足够提供“哪里接触”的先验**。论文 Appendix C.1 明确说 targets encode hand-level contact regions rather than precise contact points；12×12 token grid 上每个 token 覆盖 32 像素。这对滑移/插入的精细接触可能不够。
3. **Ego Humanoid 的增益主要来自触觉预训练，而非其他混杂因素**。消融中 Visual PT 与 1000H 共享 RGB 数据和预训练 compute，只差触觉，因此归因较强；但 Scratch 在 Ego 上反而高于 Visual PT（58.34 vs 52.45），论文未解释这一非单调性。
4. **真实机器人 paired initial conditions 足以控制噪声**。paired 设计减少初始状态差异，但 10 trials/task 的样本量仍小，且没有报告置信区间。
5. **缺失触觉可以用 placeholder 安全替代**。这假设训练时的缺失模式与部署时的传感器失效模式相似；论文的质量检查（有限值比例、饱和比例、stuck taxel）是工程补丁，不是鲁棒性证明。

## 7. 与相关工作对比 (Comparison)

| 方法 | 触觉角色 | 数据规模 | 架构/训练 | 适用场景 | 局限 |
|------|----------|----------|-----------|----------|------|
| STAR | 稀疏触觉表示，策略输入 | 200 小时 bimanual robot 数据 | VTLA，处理触觉稀疏 | 灵巧操作 | 机器人数据规模较小；触觉不做视觉监督 |
| FTP-1 | 触觉基础策略，跨传感器 | 论文未在正文给规模 | tactile foundation policy | 接触密集操作 | SparkArena-Real 36.00%，低于 Spark-0 52.00% |
| N0-VTLA | 预测 latent tactile tokens | 论文未在正文给规模 | VTLA | 触觉-视觉-语言联合 | 触觉是 latent 预测，不是显式空间对齐监督 |
| EgoVLA | 无触觉，人类视频预训练 | egocentric human video | VLA + retargeting | Ego Humanoid short-horizon | long-horizon seen 45.93%，unseen 28.79% |
| FastWAM | 无触觉，world-action model | 论文未在正文给规模 | video + action DiT | SparkArena-Sim | Sim 28.29%，低于 Spark-0 36.86% |
| VITRA | 无触觉，人类视频转 3D hand action | 100 万 human-video clips | VLA 预训练 | 人类视频迁移 | Ego long-horizon unseen 39.70%，低于 Spark-0 61.82% |
| **Spark-0** | **train-only contact/geometry 监督 + 触觉动作条件** | **1.9k 小时混合数据，1,000 小时 HumanTouch** | **Qwen3-VL-4B + 36 层 DiT action expert** | **接触密集灵巧操作；有/无触觉部署** | **单 embodiment；contact map 粗糙；真实 trial 少；算力门槛高** |

**面试 Tip**：被问“这篇的核心贡献”时，不要只说“触觉 VLA”。要说：**它把触觉的两种角色拆开——训练时用触觉监督视觉 token 的空间对齐，推理时用独立触觉通路条件动作专家；因此无触觉也能受益，有触觉更准。证据是同一 checkpoint 在 Ego Humanoid vision-only 上 +19.27 点，在 SparkArena-Real 上 +8.00 点。**

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做触觉预训练/多模态表征的研究者：重点 §4.2、§4.3、Appendix C.1，看 contact map 的生成、layer 25 的选择、非对称 bias 与 tactile shortcut 的设计。
  - 做灵巧操作 VLA 落地的工程师：重点 §4.1、§4.3、§5、§6.3、Appendix D.2，看动作空间、RTC 部署、缺失触觉处理和预训练/微调预算。
  - 做 benchmark/评测的人：重点 §5、Appendix E，看 SparkArena 的 Sim/Real 双轨、paired initial conditions、success/completion/PSR 的定义。
- **建议章节路径**：先读 §1.2 和 §4.2/§4.3 抓住双角色机制 → 再看 §6.2/§6.3 的消融和数字 → 最后看 Appendix B/C 的实现细节。§2 相关工作如果熟悉 VLA 可快速跳过。
- **不值得精读的理由**：如果你不做机器人学习、不关心触觉或多模态预训练，读摘要和 §6.2 即可；论文的主要价值在“触觉作为预训练监督”的机制和消融，而不是通用 VLA 架构创新。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接：https://arxiv.org/abs/2610.22342