# 机器人策略该学什么预测？重审 VLA 学习的监督目标 (What Should Robot Policies Learn to Predict? Rethinking Supervision Targets in Vision-Language-Action Learning)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: What Should Robot Policies Learn to Predict? Rethinking Supervision Targets in Vision-Language-Action Learning
> **链接**: https://arxiv.org/abs/2610.21915
> **核心定位**: 在固定训练预算下，把大量样本从动作预测改分配给 LA（latent action）或 FV（future vision）目标，仍可能提升动作执行；监督目标与样本分配应作为显式设计变量，而不是默认动作主导。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 固定样本预算下，合适的 LA/FV 目标在动作样本仅占 50% 时仍可超过 action-only；FV 总体优于 LA，但最优分配依目标、基准、预算而变 |
| 适合精读 | 正在设计 VLA 训练目标、数据配比或辅助 loss 的研究者/工程师；重点看 §3（router 与损失）和 §4.2–4.3（分配扫描） |
| 可以跳过 | 只关心推理加速、硬件部署或纯 RL 算法的人；本文重点是训练期监督分配 |
| 落地可行性 | 中：机制简单（改路由即可），但需额外冻结目标编码器与离线标签，收益依基准 |
| 主要风险 | 预算匹配只固定样本曝光，不等 FLOPs 或墙钟；跨 LA/FV 家族比较仍混淆目标表示与辅助专家 |

💡 **X-Ray 开场**
这篇论文问三个问题：VLA 训练中，动作预测必须拿大多数样本吗？把样本分给 LA 或 FV 预测，动作执行会变好还是变差？如果会变好，是哪种目标、在什么条件下？结论是：在相同样本预算下，FV 在 LIBERO 与四个真实机器人任务上总体优于 action-only 和 LA；CALVIN 上 LAPA-DINOv3 可与 FV 竞争；但 LIBERO 的 P25 分配全部低于对应 P100 控制，说明不是"越少动作监督越好"。对 VLA 研究者意味着：辅助目标不是免费附加项，而是与动作预测竞争预算的显式设计选择。

📍 **研究全景时间线**

```
2022 CALVIN 基准 → 2023 LIBERO 基准
→ 2025 LAPA / V-JEPA 2 / FLARE / π0 等
→ 2026 villa-X、DINOv3 等目标编码器演进
→ [本文 2026] 固定预算下系统比较 action / LA / FV 的样本分配
```

本文位置：不是提出新编码器，而是把"监督目标选择 + 样本分配"变成受控实验变量。局限：结论主要在 LIBERO、CALVIN 与单一真实机器人平台上验证。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 训练/推理差异 | 备注 |
|------|------|------|----------------|------|
| VLM（MiniCPM-V-4.6） | 当前观测 o_t、语言 ℓ | 隐状态 h_t | 全量微调，FP32 | 本地约 1.30B 参数；文本塔 hidden 1024/24 层，视觉塔 1152/27 层 |
| 动作专家 | 噪声动作 a_t^τ、状态 s_t、h_t、τ_a | 干净动作 chunk | 训练与推理都用 | 预测 H=8 步、7 维动作 |
| 辅助专家（LA） | L_z 个可学习 query Q_LA、h_t | LA token 预测 ẑ_t | 训练参与；推理仍参与 joint self-attention，但不提供辅助目标，且只解码动作专家输出 | L_z = H·K；LAPA K=4，LAPA-DINO K=16 |
| 辅助专家（FV） | f_t、噪声未来特征 f_{t+δ}^τ、clean 动作 a_t、h_t、τ_f | 流速度 v̂ | 训练参与；推理仍参与 joint self-attention，但不提供未来观测或辅助目标，且只解码动作专家输出 | FV 路由样本以 clean demonstrated action 为条件 |
| 样本级 router | 每个训练样本 i | r_i ∈ {0,1} | 仅训练 | r_i ~ Bernoulli(p_A)，每样本只贡献一个损失 |
| 目标编码器（6 个） | o_t、o_{t+δ} | LA codes 或 FV features | 冻结 | DINOv3、V-JEPA 2、Wan2.2 VAE 等 |

### 1.2 关键机制 (Key Mechanism)

- **硬路由替代损失加权**：每个样本只走动作损失或辅助损失之一，而不是在同一 batch 内对两项损失加权求和。好处是 B_A 与 B_q 可直接计数，预算分配变成显式变量；代价是每个 batch 内两路梯度都更稀疏。
- **双流 MMDiT + joint self-attention**：动作专家与辅助专家有各自输入投影和 FFN，但通过联合自注意力交换信息。辅助目标不是外挂 head，而是与动作表征共享注意力。
- **FV 路由仍看 clean action**：FV 样本以 demonstrated action chunk 为条件输入，因此"动作标签被使用"不等于"动作损失预算被占用"。论文在 §3.2 明确这一点，避免把 B_A 误读为所有动作标签的使用次数。
- **source-matched 控制**：每个 FV 配置的 P100 控制保留同一 FV 编码器、输入 f_t、双流结构和训练设置，只把路由改为全动作。LA 因 Q_LA 无源特定特征，共用 full-budget action-only 基线。
- **分配扫描**：p_A ∈ {0.25, 0.50, 0.75, 1.00}，并在 30k/40k/50k/60k optimizer updates 上重复，检验增益是否跨预算成立。

⚡ **Eureka Moment**：在固定样本预算下，"给辅助目标多少样本"是一个可独立调节的设计变量；把动作预测降到 50%，只要目标选得对，动作执行仍可更好——监督目标的选择本身就是策略设计的一部分。

### 1.3 信息流/架构图 (Flow / Diagram)

```
o_t, ℓ ──► VLM ──► h_t ──┬──► 动作专家 ──► â_t（推理时唯一执行输出）
                         │
s_t ─────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │ 训练时 router: r_i ~ Bern(p_A) │
         └───────┬────────────────┬───────┘
             r=1 │                │ r=0
                 ▼                ▼
            L_A（动作损失）    L_q（LA 或 FV 损失）
                 │                │
                 └── joint self-attention ──► L
```

推理路径只走动作路由：4 步采样，τ_a,j = j/4；辅助专家参与联合注意力但不解码，未来观测与目标编码器都不上机器人。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
固定预算 B 下，p_A 决定动作样本占比；每个样本只算一个损失
```

**目标**：在总样本预算 B 固定时，控制动作损失与辅助损失的期望样本数，并比较不同目标 q 的效果。

**预算与路由**：

```
B = N · B_eff = B_A + B_q
r_i ~ Bernoulli(p_A)
E[B_A] = p_A · B,   E[B_q] = (1 - p_A) · B
L = (1/B) Σ_i [ r_i · L_A,i + (1 - r_i) · λ_q · L_q,i ]
```

**LA 目标**：

```
c_u = E_LA^(m)(o_u, o_{u+δ}),  m ∈ {LAPA, LAPA-DINOv2, LAPA-DINOv3}
z_t = (c_t, ..., c_{t+H-1}),   L_z = H · K
L_LA = CE(ẑ_t, z_t)
```

**FV 目标（flow matching）**：

```
f_t = E_FV^(m)(o_t),   f_{t+δ} = E_FV^(m)(o_{t+δ})
a_t^τ = (1 - τ_a) ε_a + τ_a a_t
f_{t+δ}^τ = (1 - τ_f) ε_f + τ_f f_{t+δ}
v = f_{t+δ} - ε_f
L_FV = || v̂(f_t, f_{t+δ}^τ, a_t, h_t, τ_f) - v ||²
```

**变量说明**：

| 符号 | 含义 | 本文取值 |
|------|------|----------|
| B | 总预算（样本出现次数，含重复使用） | 60k updates × 128 = 7.68M |
| p_A | 动作路由概率 | 0.25 / 0.50 / 0.75 / 1.00 |
| H | 动作 horizon | 8 |
| δ | 目标时间偏移 | LA 用 8 帧；FV 用 8 |
| K | 每次视觉过渡的 LA code 数 | LAPA 4；LAPA-DINO 16 |
| λ_q | 辅助损失权重 | LAPA 0.1；LAPA-DINO 0.01；FV 1 |
| τ_a, τ_f | 插值时间 | 训练由 u~Beta(1.5,1) 按 τ=(0.999−min(u,0.999))/0.999 得到；推理 4 步仅用于动作采样器，τ_f 不在推理使用 |

**直觉**：路由把"辅助监督"从加性 loss 变成预算内的机会成本；FV 路由样本仍以 clean action 为条件，所以动作信息通过条件输入进入辅助分支，但不计入 B_A。论文强调匹配 B 只固定样本曝光，不建立等 FLOPs 或等墙钟时间。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设数值，用于验算路由损失的算法，不代表论文实测。

设一个小预算 B = 8 个样本出现次数，λ_q = 1，假设每条动作样本损失 L_A = 0.4，每条辅助样本损失 L_q = 0.1（假设恒定，便于手算）。

- P100（p_A=1.00）：r 全为 1，L = (8×0.4)/8 = 0.40。
- P50（p_A=0.50）：r = [1,0,1,0,1,0,1,0]，L = (4×0.4 + 4×0.1)/8 = 0.25。
- P25（p_A=0.25）：2 个动作样本、6 个辅助样本，L = (2×0.4 + 6×0.1)/8 = 0.175。

验算：P50 的期望动作样本数 = 0.5×8 = 4；P25 = 0.25×8 = 2。对应到论文规模，B = 60,000×128 = 7,680,000；P50 动作样本期望 3,840,000，P25 为 1,920,000。

关键点：损失值下降不等于动作性能提升，因为动作梯度样本在减少。论文正是用下游成功率而非该损失值来判断分配好坏；这个玩具例子只验证路由算术。

## 4. 工程视角 (Engineering View)

- **训练实现**：按 route 分组计算损失，组均值按 microbatch 比例加权，辅助组再乘 λ_q。FV 路由需要未来帧编码，但目标编码器冻结，可离线预计算特征；LA 路由是分类交叉熵，计算更便宜。
- **吞吐与显存**：论文只报样本预算，未报 FLOPs、墙钟或显存。双流 MMDiT 加 VLM 全量微调（FP32）是主要开销；FV 分支多一次冻结编码器前向，LA 分支多一个分类头。
- **推理**：只保留 VLM + 动作专家路径，4 步采样；辅助专家参与联合注意力但不输出，目标编码器不部署到机器人。工程上这与普通 action-only VLA 推理路径接近。
- **超参敏感性**：λ_q 需按目标调（LAPA 0.1，LAPA-DINO 0.01，FV 1）；论文在 LIBERO 上做消融后固定，CALVIN 不再重调。我的判断：跨数据集迁移时 λ_q 可能需重新校准。
- **预算口径**：60k optimizer updates、B_eff=128；P50 不是"一半 FLOPs"，因为 FV 路由仍要跑目标编码器前向。部署评估时应把训练成本与推理成本分开算。

## 5. 数据与评测 (Data & Eval)

| 基准 | 数据组成 | 训练预算 | 评测设置 | 指标 |
|------|----------|----------|----------|------|
| LIBERO | 1,693 episodes / 273,465 frames / 40 tasks / 20 Hz | 60k updates，B_eff=128；半预算 30k | 4 suites × 10 tasks × 50 rollouts = 2,000；evaluation seed 0 | success rate |
| CALVIN | ABC→D 协议 | 60k updates | 1,000 条五任务序列；主文 seed 0，附录 seed 0/1/2 | Avg. Len.、SR-5 |
| 真实机器人 | DOS W1 左臂，203 demos（50/50/50/53） | 30k updates | 4 任务 × 20 trials，每 trial 最高 8 分，累计 160 | partial-credit score |

**P50、full budget 的关键结果（论文 Table 1，seed 0）**：

| 配置 | LIBERO Avg. (%) | CALVIN Avg. Len. | CALVIN SR-5 (%) |
|------|------------------|-------------------|-------------------|
| Action-only，half budget | 93.20 | 3.971 | 57.7 |
| Action-only，full budget | 96.50 | 4.314 | 70.3 |
| LAPA | 94.65 | 3.466 | 42.1 |
| LAPA-DINOv2 | 96.45 | 3.935 | 63.4 |
| LAPA-DINOv3 | 95.90 | 4.434 | 77.0 |
| DINOv3 FV | 97.25 | 4.370 | 75.1 |
| V-JEPA 2 FV | 97.95 | 4.451 | 76.0 |
| Wan2.2 FV | 98.15 | 4.237 | 70.5 |

读数注意：
- 相对 half-budget action-only（93.20 / 3.971），六个 P50 配置在 LIBERO 全部提升；CALVIN 上所有 FV 与 LAPA-DINOv3 提升，LAPA 与 LAPA-DINOv2 不提升。
- 相对 full-budget action-only（96.50 / 4.314），LIBERO 只有三个 FV 超过（最高 98.15），LA 都未超过；CALVIN 上 DINOv3、V-JEPA 2、LAPA-DINOv3 超过。
- 论文正文还给出 source-matched 控制：LIBERO 上 V-JEPA 2 与 Wan2.2 的 P50 超过各自 P100（+1.10、+0.60 个百分点），DINOv3 低 0.20；CALVIN 上三个 FV 在四个预算均超过 source-matched 控制（Table 12），但这些控制本身低于共享 action-only 基线，所以相对该基线的增益不一致。

**真实机器人（Table 15，P50，每任务满分 160）**：

| 配置 | Pick-out | Place-in | Rearrange-cups | Rearrange-objects |
|------|----------|----------|----------------|-------------------|
| Action-only | 116 | 92 | 80 | 124 |
| LAPA | 92 | 92 | 80 | 96 |
| LAPA-DINOv2 | 112 | 80 | 76 | 120 |
| LAPA-DINOv3 | 106 | 88 | 72 | 92 |
| DINOv3 FV | 138 | 100 | 96 | 132 |
| V-JEPA 2 FV | 150 | 108 | 100 | 144 |
| Wan2.2 FV | 156 | 112 | 108 | 148 |

论文报告：所有 FV 配置在四个任务上均高于 action-only 与所有 LA；Wan2.2 全程最高；没有 LA 配置超过 action-only。这是 20 trials/task 的 partial-credit 结果，统计力有限。

**诊断（论文 Table 2，冻结源探针，位置误差 mm）**：

| 编码器 | AgiBot ABS | AgiBot Δ1 | AgiBot Δ9 | AgiBot Δ45 | CALVIN ABS | CALVIN Δ1 | CALVIN Δ9 | CALVIN Δ45 |
|--------|------------|-----------|-----------|------------|------------|-----------|-----------|------------|
| DINOv3 | 76.31 | 25.23 | 13.73 | 14.19 | 52.06 | 69.64 | 75.44 | 58.21 |
| V-JEPA 2 | 101.98 | 20.44 | 11.42 | 12.16 | 54.70 | 64.21 | 68.43 | 55.85 |
| Wan2.2 | 105.55 | 32.28 | 17.20 | 16.65 | 53.86 | 53.44 | 59.72 | 50.01 |
| LAPA-DINOv2 | 155.65 | 14.79 | 8.62 | 8.79 | 62.70 | 56.89 | 59.16 | 54.40 |
| LAPA-DINOv3 | 155.61 | 14.68 | 8.71 | 9.23 | 60.14 | 53.23 | 57.41 | 50.48 |
| LAPA | 164.82 | 30.79 | 15.79 | 15.58 | 127.91 | 139.59 | 109.36 | 95.03 |

作者主张：FV 源在 ABS 位置误差上整体低于 LA 源；LAPA-DINO 在 AgiBotWorld-Beta unseen 的相对运动误差上整体最低；CALVIN 上领先编码器随 stride c 变化。聚类 NMI 显示 LAPA-DINO 与 motion label 关联更强（LAPA-DINOv3 0.412），FV 与 task ID 关联更强（DINOv3 0.828、V-JEPA 2 0.811）。这些是描述性关联，不建立因果。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：
- 在固定样本预算下，对 action / LA / FV 的样本分配做受控比较，且 FV 有 source-matched 控制。
- 在 LIBERO 上，P50 的六个配置全部超过 half-budget action-only；三个 FV 超过 full-budget action-only。
- 在 CALVIN 上，V-JEPA 2 P25 达到 4.522，高于其 source-matched P100 的 4.091，也高于共享 action-only 的 4.314。
- 在四个真实机器人任务上，FV 全面高于 action-only 与 LA。

**不能做什么 / 边界**：
- 不是所有目标都受益：CALVIN 上 LAPA P50 3.466、LAPA-DINOv2 P50 3.935，均低于 half-budget action-only 3.971；LIBERO 上 LA 都未超过 full-budget 96.50。
- 不是越少动作监督越好：LIBERO 上所有测试的 P25 配置都低于其 P100 控制；CALVIN 上 DINOv3 P25 仅 2.107/16.6，远低于其 P100。
- 增益依赖预算：CALVIN 上 FV 相对 source-matched 控制在四个预算都正，但相对共享 action-only 基线，V-JEPA 2 只在四个预算中的三个超过，LA 中只有 LAPA-DINOv3 在全预算超过。
- 诊断不能证明因果：探针、聚类、注意力图都是相关性或定性证据。
- 真实机器人只在一个平台、四个任务、每任务 20 trials 上评估，不能外推到其他本体或开放环境。

### 6.1 隐含假设 (Hidden Assumptions)

- **样本曝光是训练预算的充分代理**：论文明确承认匹配 B 不等 FLOPs 或墙钟；FV 路由的额外编码成本未计入。
- **冻结目标编码器的特征质量足够**：编码器不随策略更新，若目标特征与任务分布不匹配，路由收益可能消失。
- **路由独立于样本难度与任务**：Bernoulli 路由不区分难样本、任务或阶段；论文未做课程式路由。
- **离线 LA 标签与在线观测分布一致**：LIBERO 上 LA code 预计算；若部署域偏移，标签可能失真。
- **辅助信号通过 joint attention 正向迁移到动作专家**：这是架构假设，论文用注意力图与探针间接支持，未做消融切断联合注意力。
- **基准结论可外推到其他机器人形态**：真实机器人仅 DOS W1 左臂，未验证双臂、移动或人形平台。

## 7. 与相关工作对比 (Comparison)

| 方法/线索 | 关注点 | 监督形式 | 预算控制 | 与本文差异 |
|-----------|--------|----------|----------|------------|
| LAPA（Ye et al., 2025） | 潜动作预训练 | LA | 未系统分配 | 本文把 LA 作为可分配目标之一，并固定预算 |
| villa-X（Chen et al., 2026） | 联合建模 latent 与 robot action | LA + action | 未固定预算 | 本文用样本级硬路由，显式计 B_A/B_q |
| FLARE（Zheng et al., 2025） | 未来表征对齐 | FV | 对比更长训练的 action-only | 本文同预算、变化 action share，并加 source-matched 控制 |
| GR-2 / UWM（2024–2025） | 视频/世界模型联合建模 | FV/视频 | 联合目标 | 本文不生成像素，预测冻结编码器特征 |
| 本文 | 监督目标选择与分配 | action / LA / FV | 样本级 router，P25–P100 | 把"加辅助 loss"改写成"预算内分配"问题 |

**面试 Tip**：被问"VLA 要不要加辅助 loss"时，先反问预算口径；然后答：本文表明固定样本曝光下，50% 样本给 FV 仍可超过 action-only，但 LIBERO 的 P25 全部下降，CALVIN 上 LAPA 与 LAPA-DINOv2 在 P50 反而低于半预算基线（LAPA-DINOv3 为 4.434，高于 3.971）——所以是条件性收益，不是免费午餐。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 正在设计 VLA 训练目标、数据配比或辅助 loss 的研究者；
  2. 想在现有 action loss 之外加入 LA/FV，但需要知道"给多少样本"的工程师；
  3. 需要理解"预算匹配 + source-matched 控制"实验设计的人。
- **建议章节路径**：先读 §3.1–3.2（问题定义、router、FV 条件输入）→ §4.2（P50 与分配扫描）→ §4.3（LA vs FV 的基准差异）→ §4.4（探针与聚类诊断）→ 附录 A.2（控制匹配细节）与 B.7（真实机器人协议）。
- **不值得精读的理由**：如果你只关心推理优化、硬件部署，或已经熟悉 LA/FV 且只想要 SOTA 数字，读摘要与 §4.2 的表格即可；本文的贡献在实验设计与分配视角，不在新模型结构。

---
[← Back to Theory](./README.md)

**关键引用**：[论文](https://arxiv.org/abs/2610.21915) · [HTML 全文](https://arxiv.org/html/2610.21915)