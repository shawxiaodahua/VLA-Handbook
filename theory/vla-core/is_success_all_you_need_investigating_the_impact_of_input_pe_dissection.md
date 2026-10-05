# 成功率就够了么？输入扰动如何改变 VLA 在桌面操作中的行为 (Is Success All You Need? Investigating the Impact of Input Perturbations on VLA Behaviour in Tabletop Manipulation Tasks)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-04
>
> **论文**: Is Success All You Need? Investigating the Impact of Input Perturbations on VLA Behaviour in Tabletop Manipulation Tasks
> **链接**: https://arxiv.org/abs/2610.01351
> **核心定位**: 在任务成功率 (TSR) 之外，提出一套 **benchmark-agnostic 的「行为鲁棒性」评估框架**——即使任务仍然成功，成功的轨迹在扰动下也可能被显著改变，而这种退化 TSR 完全看不到。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | TSR 不能完整反映鲁棒性：相同扰动下两个模型 TSR 相近，但成功轨迹的行为质量（平滑度/效率/夹爪动作）可以差出数倍 |
| 適合精讀 | 如果你在做 VLA 鲁棒性评估、部署前验证、或需要设计「模型卡」——重点看 §2 指标定义、§5 结果、§7 对比 |
| 可以跳過 | 如果你只关心新模型架构/训练算法（本文不提出新模型，只用 π0.5 / VLANeXt / OpenVLA-OFT 做被试） |
| 落地可行性 | 高（纯事后分析，只要有轨迹级 state/action 即可复现，代码 + vla-eval harness 已开源）|
| 主要風險 | 仅在 SIM 环境桌面操作上验证；指标高层抽象、语义不精确；只统计成功轨迹 |

💡 **X-Ray 开场**（2-3 句，非专家也能读懂）
这篇论文问的问题很简单：一个机器人「把碗放到盘子上」成功了，但中途它掉了一次又捡起来——这算鲁棒么？现有 VLA 基准只看「成没成」（TSR），看不到「怎么成的」。作者在 LIBERO / LIBERO-Plus 上补了一套**轨迹级行为指标**（抖动、路径、时长、夹爪行程），发现扰动可以让成功轨迹的抖动翻倍甚至翻几倍，而这些在成功率上完全被抹平。对 VLA 研究者的意义：以后报告鲁棒性，只给 TSR 是不够的。

📍 **研究全景时间线**

```
2020 MetaWorld/RLBench          →  2022 BEHAVIOR           →  2023 LIBERO
   (binary TSR 为主)                 (引入效率类指标)            (标准 TSR 基准)
        │                                │                          │
        ▼                                ▼                          ▼
2025 RoboEval  ───────────────►  2025 LIBERO-Plus / LIBERO-PRO  ──►  [本文 2026] ← 当前位置
(jerk/碰撞 安全指标,             (7 维扰动鲁棒性, 但仍只看 TSR)       首次把「扰动」×「轨迹级行为」结合
 但静态无扰动)                                                      局限: 仅 SIM 桌面 / 仅成功轨迹
```

## 1. 核心架构/方法总览 (Overview / Architecture)

本文不训练新模型，而是提出一套**评估方法论 + 开源实现**。整体是「采集轨迹 → 事后分析 → 统计聚合」的流水线。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 备注 |
|------|------|------|-----------|------|
| Rollout 采集 (vla-eval) | LIBERO / LIBERO-Plus 任务 × 扰动条件 | 每 episode 的 state/action 轨迹 | 仿真控制频率 **10 Hz** | 基线每任务 50 rollouts；每个扰动变体 2 rollouts |
| 轨迹数据暴露 | harness 内部轨迹 | 末端位置 x、关节位置 q、夹爪宽度 | 每控制步 | 本文扩展了 harness 以暴露这些量 |
| 行为指标计算 | 上一步轨迹序列 | 每 episode 的 6+1 个指标值 | 每 episode 一次 | 时长/夹爪行程/笛卡尔 jerk/关节 jerk/笛卡尔路径/关节路径 (+P95 jerk) |
| 统计聚合 | 基线 vs 扰动分布 | 任务级 % 变化 → suite 级中位数 | 每任务、每 suite | Mann-Whitney U + Benjamini-Hochberg FDR |
| 变异性分析 | 分布 | MAD 及其 % 变化 | 每任务、每 suite | 衡量成功行为「一致性」 |

### 1.2 关键机制 (Key Mechanism)

- **只评估成功轨迹**：目的不是比较「成没成」，而是揭示「成功后行为退化了多少」——这正是 TSR 掩盖的部分。
- **基线 = 原始 LIBERO**（无扰动）；**条件 = LIBERO-Plus 的 7 类扰动**。
- **任务级 → suite 级聚合**：先算每个任务的指标均值变化百分比，再取 suite 内任务的**中位数**（抗离群）。
- **显著性把关**：每个 suite 内报告「显著上升的任务数 / 10」，并做 FDR 多重比较校正，避免刷假阳性。
- **行为一致性 (MAD)**：不仅看「均值漂移」，还看「成功行为是否变得更不稳定」。

⚡ **Eureka Moment**：**「成功」不是二元的终点，而是一条轨迹**——把评估单位从 episode 的 success bit 下沉到 trajectory 的 state-action 序列，就暴露了 TSR 结构性盲区。扰动可以在 TSR 不变的情况下，让成功轨迹默默变得更抖、更慢、更折腾。

### 1.3 信息流/架构图 (Flow / Diagram)

```
LIBERO 任务 ──┬─► 无扰动 rollout (baseline, 50 ep/task)
              │
              └─► LIBERO-Plus 7 类扰动 rollout (2 ep/variant)
                          │
                          ▼
              vla-eval harness (10 Hz)  ──► 轨迹序列 (x, q, gripper)
                          │
                          ▼
              事后分析 pipeline ──► 每 episode 指标向量
                          │
          ┌───────────────┴────────────────┐
          ▼                                 ▼
   典型行为变化 Δ (%)                 行为变异性 ΔMAD (%)
   (median over tasks)                (median over tasks)
          │                                 │
          └──────────► 与 TSR 退化做 Spearman 相关 ◄──────┘
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
Δ = 100 · (x̄_perturbed − x̄_baseline) / x̄_baseline      # 只看成功轨迹的行为相对漂移
```

**目标**：量化「扰动条件下，成功轨迹的典型行为」相对基线的相对变化，并判断是否统计显著。

**核心公式**（全部用代码块表达，禁止 LaTeX）：

```
# (1) 平均笛卡尔 jerk —— 末端位置 x 的三阶差分（Δt = 1/10 s）
Jerk_cart^Mean = (1/(T-3)) · Σ_{t=1}^{T-3} || (x_{t+3} - 3x_{t+2} + 3x_{t+1} - x_t) / (Δt)^3 ||_2

# (2) P95 笛卡尔 jerk（捕捉上尾/最差情况）
J_cart^95 = Q0.95( { ||(x_{t+3} - 3x_{t+2} + 3x_{t+1} - x_t)/(Δt)^3||_2 }_{t=1..T-3} )

# (3) 平均关节 jerk（同上，换成关节角 q）
Jerk_joint^Mean = (1/(T-3)) · Σ_{t=1}^{T-3} || (q_{t+3} - 3q_{t+2} + 3q_{t+1} - q_t) / (Δt)^3 ||_2

# (4) 路径长度（笛卡尔 / 关节）
L_cart  = Σ_{t=1}^{T-1} || x_{t+1} - x_t ||_2
L_joint = Σ_{t=1}^{T-1} || q_{t+1} - q_t ||_2

# (5) 任务级典型行为变化（x̄ = 成功轨迹的均值）
Δ_{k,m,p} = 100 · (x̄_{k,m,p} - x̄_{k,m,base}) / x̄_{k,m,base}

# (6) suite 级汇总 = 任务级变化的中位数
Δ̃_{m,p} = median_k ( Δ_{k,m,p} )

# (7) 行为变异性 MAD
MAD_{k,m,p} = median_i ( | X_{k,m,p,i} - median_j(X_{k,m,p,j}) | )

# (8) 变异性相对变化
ΔMAD_{k,m,p} = 100 · (MAD_{k,m,p} - MAD_{k,m,base}) / MAD_{k,m,base}
```

**变量说明**：

| 符号 | 含义 | 单位 |
|------|------|------|
| x_t, q_t | t 时刻末端笛卡尔位置 / 关节角 | m / rad |
| Δt | 控制步长（此论文用 10 Hz → Δt = 0.1 s） | s |
| T | 轨迹总控制步数 | — |
| x̄_{·} | 某任务下成功轨迹的指标均值 | 随指标 |
| k, m, p | 任务 / 指标 / 扰动条件下标 | — |
| MAD | 中位数绝对偏差（对离群更稳） | 随指标 |

**直觉**：
- **jerk = 位置的三阶差分**，本质是「加速度变化率」。jerk 大 = 动作急刹急启，机械磨损大、看起来不可预测。
- **Δ 用中位数聚合**：避免单个难任务把整 suite 的结论带偏。
- **MAD 而非标准差**：轨迹分布可能有重尾，MAD 更稳，也呼应「中位数 × 中位数」的稳健风格。
- **只统计成功轨迹**：Δ 的分母是「成功轨迹的均值」，不是全体 rollout——这正是设计意图：**在成功集合内部**看行为漂移。

> 符号与本文正文保持一致。所有公式原文以 LaTeX 呈现，此为 Unicode 纯文本重写。

## 3. 带数字走一遍：玩具例子 (Worked Example)

构造一个 2D、单任务的最小闭环，走一遍「成功但行为退化」的检测流程。

**假设**：一个 pick-and-place 任务，基线成功轨迹平均时长 40 步（4.0 s @10 Hz），末端轨迹平滑。

```
基线 (无扰动):
  平均时长 x̄_dur,base = 4.0 s
  平均笛卡尔 jerk x̄_jerk,base = 20.0 m/s³
  夹爪行程 x̄_grip,base = 30.0 mm

扰动后 (camera):
  平均时长 x̄_dur,pert = 4.9 s     → Δ = 100·(4.9-4.0)/4.0 = +22.5 %
  平均 jerk x̄_jerk,pert = 24.8    → Δ = 100·(24.8-20.0)/20.0 = +24.0 %
  夹爪行程 x̄_grip,pert = 41.3     → Δ = 100·(41.3-30.0)/30.0 ≈ +37.7 %
```

这套数值恰好对应论文 §V-B 中 π0.5 在 LIBERO-Spatial / camera 条件下的报告（+22.5% / +24.1% / +37.8%）。**关键点**：该条件下 π0.5 的 TSR = 69.7%，VLANeXt TSR = 68.0%（几乎一样），但 VLANeXt 的对应 Δ 只有 +4.2% / +3.2% / +2.5%。→ 仅凭 TSR 你完全分不出这两个模型的行为质量差异。

再看变异性（MAD）玩具推演，以 LIBERO-Spatial / camera / duration 为例：

```
VLANeXt : ΔMAD_dur = +58 %
π0.5    : ΔMAD_dur = +142 %
OpenVLA-OFT : ΔMAD_dur = +317 %
```

含义：扰动后不仅成功轨迹「均值」漂移，**成功集合内部的分散度也扩大**——同一任务下不同次成功执行差异更大，即「行为可预测性下降」，这对人机信任与部署安全是坏信号。

## 4. 工程视角 (Engineering View)

| 维度 | 观察 / 含义 |
|------|-------------|
| 计算成本 | 纯事后分析，无额外模型推理；主要开销在 rollout（仿真）与 O(T) 差分计算 |
| 频率/时序 | 用仿真控制频率 10 Hz 计算导数，**刻意不用 wall-clock**，以避免 GPU 争用/推理延迟污染时长指标 |
| 采样预算 | 基线 50 rollouts/task（对齐社区惯例）；扰动变体仅 2 rollouts/variant，object 类仅 1 个初始状态 |
| 指标选择权衡 | 高层、task-agnostic → 可移植性强，但**语义不精确**：夹爪行程上升无法区分「反复抓取失败」还是「抓放重试」 |
| 部署含义 | jerk 关系到机械应力与执行器磨损；时长/效率关系到用户接受度；MAD 关系到可预测性与信任 |
| 迁移边界 | 只要基准能导出轨迹级 state/action（如 MetaWorld），框架即可复用 |
| 复现入口 | 代码: https://github.com/esgi-research-group/vla-reliability （论文脚注 2）；harness: vla-eval (arXiv:2603.13966) |

**工程落地要点**：把此框架接进现有 eval CI，作为「鲁棒性回归测试」——每次模型迭代不仅比 TSR，还比 Δ 向量；若某扰动的 jerk/MAD 突增，即使 TSR 未跌也拦下。

## 5. 数据与评测 (Data & Eval)

**基准**：LIBERO（原始，作基线）+ LIBERO-Plus（7 类扰动）。

**四个 suite**：LIBERO-Spatial、LIBERO-Long、LIBERO-Goal、LIBERO-Object。

**7 类扰动条件**（LIBERO-Plus）：物体布局 (object)、相机视角 (camera)、机器人初始状态 (robot)、语言指令 (language)、光照 (lighting)、背景纹理 (background)、传感器噪声 (sensor) —— 连同无扰动共 8 个条件。

**被试模型**：π0.5、VLANeXt、OpenVLA-OFT（均使用「跨 suite 联合 fine-tune」的 checkpoint，避免 suite-specific fine-tune 带来的不公平）。

**关键结果（数字取自论文 Table II / §V）**：

| 观察 | 具体数字 | 来源 |
|------|----------|------|
| SOTA 模型的扰动鲁棒 TSR | π0.5 85.7% / VLANeXt 83.9%（LIBERO-Plus 总体均值，作者引用） | §IV-A |
| OpenVLA-OFT 只评 Spatial | 因其 TSR 退化到 67.9%，goal/long suite 预计更差 | §IV-A / 脚注 3 |
| Spatial / camera 的 jerk 变化 | π0.5 +24.1%（7/10 任务显著）；OpenVLA-OFT +35.3%（10/10 显著） | §V-A |
| TSR 退化 vs 行为变化的 Spearman 相关 (n=63) | 夹爪行程 ρ=0.56；P95 笛卡尔 jerk ρ=0.54；时长 ρ=0.52；平均笛卡尔 jerk ρ=0.43；平均关节 jerk ρ=0.40 | §V-B |
| 相关性弱/不显著 | 笛卡尔路径 ρ=−0.03 (p=0.80)；关节路径 ρ=0.20 (p=0.11) | §V-B |
| 「同 TSR 不同行为」实例 | Spatial/camera：π0.5 TSR 69.7% vs VLANeXt 68.0%，但 π0.5 的 Δ 远大于 VLANeXt | §V-B |
| 变异性收窄反例 | LIBERO-Goal/lighting：夹爪行程 MAD 下降 28%（π0.5）与 21%（VLANeXt） | §V-C |
| 相机扰动一致性 | camera 使**所有模型与 suite** 在时长/平均笛卡尔 jerk/夹爪行程上 MAD 上升 | §V-C |

**统计方法**：任务级 Mann-Whitney U 双侧检验 + Benjamini-Hochberg FDR 校正（α=0.05）；相关性用 Spearman。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 能做什么
- **发现被 TSR 掩盖的退化**：识别「成功但行为变差」的扰动-模型组合。
- **区分行为鲁棒性**：在 TSR 相近时，比较模型的成功执行质量（如 camera 下 π0.5 明显比 VLANeXt 差）。
- **刻画一致性**：用 MAD 展示成功行为是「更散」还是「更收敛」。

### 失败模式 / 局限（论文自述，§VI）
- **仅仿真**：全部实验在 LIBERO / LIBERO-Plus 的仿真桌面操作上完成，未在真机部署验证；结论能否外推物理机器人**未知**。
- **仅成功轨迹**：刻意排除失败轨迹，因此**无法回答**「扰动是否也让失败轨迹的行为变差」。
- **指标语义粗略**：高层 task-agnostic 指标无法精确定位「为什么」变差（如夹爪行程上升的原因不可分解）。
- **模型样本受限**：只有 TSR 足够高的 3 个模型能评；OpenVLA-OFT 因 TSR 太差只评了 Spatial。
- **扰动变体采样少**：每变体仅 2 rollouts（object 仅 1 个初始状态），分布估计可能有噪声。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设「成功轨迹的行为分布可比」**：把不同扰动下的成功集合直接对比，但成功集合的**组成**可能因扰动而变（论文 §V-C 自己也提出：扰动可能让「靠恢复侥幸成功」的轨迹先失败，从而留下更干净的轨迹——这会**混淆** Δ 的解读）。
- **假设 TSR 是可靠的筛选器**：以 TSR 高来选被试模型，隐含「TSR 可信」前提；但本文主旨恰恰是 TSR 会掩盖问题。
- **假设 10 Hz 差分足以刻画平滑度**：jerk 对采样频率敏感，且仿真动作与真实电机指令之间存在 gap。
- **假设指标跨任务同构可比**：把不同任务的 Δ 直接取中位数，隐含「同一扰动的效应方向在不同任务间同质」——但 Table II 中出现大量符号不一致的单元格，说明该假设较弱。
- **「扰动保成功」前提**：LIBERO-Plus 设计上保证扰动后任务仍可行，真实分布的扰动不一定满足。

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 是否含扰动 | 是否有轨迹级行为指标 | 备注 |
|------|--------|------------|----------------------|------|
| LIBERO (2023) | 终身学习基准 | 否 | 否 | 主报 TSR |
| BEHAVIOR (2022) | 家庭活动 | 否 | 是（效率类：时间/位移/手部位移） | 无扰动 |
| RoboEval (2025) | 操作结构化评估 | 仅空间变化 | 是（jerk/自碰撞/环境碰撞/滑落） | 安全 + 效率指标，但静态 |
| LIBERO-Plus (2025) | 7 维扰动鲁棒性 | 是 | 否 | 只看 TSR 下降 |
| LIBERO-PRO (2025) | 抗记忆的公平评估 | 是（4 维） | 否 | 只看 TSR |
| Colosseum V2 / RobustVLA | VLA 泛化/扰动鲁棒 | 是 | 否 | 以 TSR 度量 |
| **本文 (2026)** | **扰动 × 成功轨迹行为** | **是** | **是** | **首个结合二者**，并开源进 vla-eval |

**定位一句话**：本文 = LIBERO-Plus 的扰动条件 × RoboEval 的行为指标，并补上「只统计成功轨迹」这一关键视角。

🎯 **面试 Tip**：被问到「这篇做了什么」，答：**它证明了 TSR 与行为鲁棒性会脱钩——两个模型 TSR 几乎相同，但成功轨迹的 jerk/时长/夹爪行程变化可以差一个量级（Spatial/camera：π0.5 +24.1% vs VLANeXt +3.2%），所以鲁棒性评估必须下沉到轨迹级。**

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 负责 VLA 评估/基准设计的研究者——需要往「模型卡」里补行为指标。
  2. 要做部署前鲁棒性验证的工程团队——关心 SIM-to-Real 前如何筛模型。
  3. 研究 VLA 失效模式、想在 TSR 之外找信号的博士生。
- **建議章節路徑**：先讀 §III-A/§III-B（指标与统计定义，决定你能复现什么）→ 再看 §V-A/§V-B（核心发现与相关性）→ 可跳 §II 相关工作（除非你要做文献定位）。
- **不值得精讀的理由**：如果你不做机器人学习、或已经熟悉 LIBERO 系列与 jerk 类指标，读摘要 + Table II 即可；本文不提出新模型或新训练方法，纯评估侧贡献。

---

[← Back to Theory](./README.md)

**关键引用**
- 论文: [arXiv:2610.01351](https://arxiv.org/abs/2610.01351)
- 代码: [esgi-research-group/vla-reliability](https://github.com/esgi-research-group/vla-reliability)
- 评测 harness: vla-eval (arXiv:2603.13966)
- 基础基准: LIBERO (NeurIPS 2023) / LIBERO-Plus (CVPR 2026) / RoboEval (arXiv:2507.00435)
