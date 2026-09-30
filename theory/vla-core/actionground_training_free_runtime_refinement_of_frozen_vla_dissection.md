# ActionGround：冻结 VLA 策略的免训练运行时精炼 (ActionGround: Training-Free Runtime Refinement of Frozen VLA Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-30
>
> **论文**: ActionGround: Training-Free Runtime Refinement of Frozen VLA Policies
> **链接**: https://arxiv.org/abs/2609.33256
> **作者**: Madhur Thareja (IIT Madras / Indian AI Research Organisation), Shriram Damodaran (NTU), Addison Lin Wang (NTU)
> **核心定位**: 不重训、不微调、不碰权重，用一个 <1ms 的神经符号运行时夹层，把"任务相位结构 + 刚体动力学"注入任何冻结 VLA 的动作输出——即插即用的后处理式精炼。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 冻结 VLA 的动作里有"物理缺口"，用相位 FSM + Euler-Lagrange 惯量加权修正，可在 4 个 backbone 上稳定提升成功率（最高 +6pp）与稳定性（最高 +19.3pp），单步开销约 0.6ms |
| 適合精讀 | 如果你在做**部署阶段的 VLA 可靠性工程**（不能重训/拿不到权重/要做安全夹层），重点看 §1.2 Branch A 与 §2 的 Eq.3/8/9；想做 world model 之前先拿低成本的"物理后处理"兜底的人 |
| 可以跳過 | 如果你只关心**训练范式创新**（如新 action head、新预训练目标），这篇是纯推理期产物，方法论上不碰训练 |
| 落地可行性 | 中—高（无需权重访问、开销可忽略、通用叠加；但强依赖 URDF/XML 标定精度，且目前只在刚性物体 pick-and-place 上验证）|
| 主要風險 | 5% 的硬上限（c=0.05）决定了它只能"微调"不能"纠偏"，接触密集任务（T4/T9）仍是 0% |

💡 **X-Ray 开场**
VLA 模型被训练成"看图 + 读指令 → 直接吐动作"，但网络里没有任何东西显式知道"现在处于抓取还是搬运阶段"，也不检查这个动作是否违反机械臂的运动方程。ActionGround 的做法是：不动模型，在动作送进仿真器之前加一层"物理关卡"——一个规则状态机按相位改写动作，一个惯性加权的动力学项做逐关节修正。对 VLA 研究者的意义：**在追求更大的端到端模型之前，先看看有多少失败其实是可以在推理期用 10 行物理规则消掉的**。

📍 **研究全景时间线**

```
2022-2023 RT-1/RT-2/PaLM-E 端到端 VLA 范式确立
        → 2024 π0 / CogACT / Diffusion-VLA 动作头多元化（自回归/扩散/flow）
        → 2024-2025 OpenVLA / OpenVLA-OFT chunked 推理弥合短时一致性
        → 2025-2026 力觉 VLA (ForceVLA)、physics-informed 训练期正则 (PINNs/Lagrangian net)
        → [本文 2026-09] ActionGround ← 当前位置：把物理与相位结构搬到推理期，冻结权重
        局限：仅刚性物体 pick-and-place；5% 修正上限；URDF 标定敏感
```

## 1. 核心架构/方法总览 (Overview / Architecture)

ActionGround 是一个**夹在冻结 VLA 与执行器之间的双层校正器**。每一控制步，MuJoCo 状态 s_t = (p^eef_t, q^grip_t, p^obj_t, c_t) 同时喂给两条并行支路：Branch A（符号相位状态机）与 Branch B（惯性加权 Euler-Lagrange 修正）。二者输出合成一个物理候选 a^phys_t，再通过全局封顶混合器与 VLA 原始动作 a^VLA 融合。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 触发/时序 | 训练 vs 推理 |
|------|------|------|-----------|--------------|
| 冻结 VLA π_θ | 图像 I_t + 语言 ℓ | a^VLA ∈ R⁷ | 每步一次前向 | 训练好、**完全不改** |
| Branch A：相位 FSM | 几何状态 s_t | a^A_t（相位特定修正） | 每步，<0.2ms | 纯规则，无参数 |
| Branch B：EL 惯量修正 | q, q̇, q̈, τ（MuJoCo 内部量） | w（软 max 权重）+ a^phys | 每步，<0.4ms | 解析动力学 oracle |
| 全局封顶混合器 | a^VLA, a^phys | a_t（实际执行动作） | 每步 | 单一超参 c=0.05 |
| EL 残差 r_EL | 同上 | 仅日志（一致性诊断） | 每步 | **不用作 gate** |

### 1.2 关键机制 (Key Mechanism)

- **相位条件化，而非均匀平滑**：核心观察是"不同操作相位受不同物理约束支配"。抓取阶段需要高频响应，搬运阶段需要抑制抖动，放置阶段需要减速。之前的 temporal ensembling（EMA）对全程均匀平滑，反而抹平了接触期必要的爆发动作——论文实测 EMA 在三个单步 backbone 上全部降低成功率（OpenVLA 36%→28%）。
- **规则而非学习**：Branch A 用几何谓词划分相位，每条规则是手工设定的（无权重），因此对 backbone 通用，无需按任务/backbone 调参。
- **动力学始终开启，而不是门槛开关**：作者试过用 ‖r_EL‖ 超阈值 ε 才触发的"硬门槛变体"，发现没有稳定收益，于是改为**每步恒定施加**惯性加权修正。r_EL 只做日志诊断，不参与加权——这保证了不会因残差估计不准而误关/误开物理。
- **两路双向耦合**：离散相位 φ_t 选择连续修正律；连续惯量权重 w 又决定该修正被信任的程度。这是一个"离散选律、连续定权"的运行时接口，梯度不回传到 π_θ。

⚡ **Eureka Moment**：**不要把物理塞进训练里去改权重，而是把物理当成推理期的"动作守门员"——用一个 95/5 的封顶混合，让冻结 VLA 保留主导权，物理修正只在每步偷偷累积。** 5% 看似微不足道，但每步都施加，就能在多步轨迹上把成功率推高最多 6pp。

### 1.3 信息流/架构图 (Flow / Diagram)

```
        ┌──────────────┐
  I_t ─▶│  冻结 VLA π_θ │──▶ a^VLA (R^7: Δxyz, Δφθψ, g)
   ℓ  ─▶└──────────────┘          │
                                   │
  s_t (MuJoCo state) ──┬───────────┼─────────────┐
                       ▼           │             │
            ┌────────────────┐     │             │
            │ Branch A: FSM  │     │             │
            │ approach/grasp │     │             │
            │ transport/place│──▶ a^A_t          │
            └────────────────┘     │             │
                       ▼           ▼             │
            ┌────────────────────────────┐       │
            │ Branch B: EL 惯量加权       │       │
            │ w = Softmax(diag(M)^-1)     │       │
            │ a^phys = w⊙a^A + (1-w)⊙a^raw│──────┤
            └────────────────────────────┘       │
                                                 ▼
                              a_t = (1-c) a^VLA + c a^phys,  c = 0.05
                              (|g| > 1.5 时 gripper 硬覆盖)
                                                 │
                                                 ▼
                                          执行器 / 仿真器
                                                 │
                          r_EL 仅日志（不回流，不 gate）▶ 诊断
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
a_t = (1-c)·a_VLA + c·w ⊙ a^A      c=0.05,  w = Softmax(diag(M(q))^{-1})
```

先给目标，再给公式，再给变量说明，最后给直觉。

**目标**：在冻结 VLA 输出的动作 a^VLA 之上，用相位规则 a^A 与动力学权重 w，产生一个仍以 VLA 为主导（95%）的物理一致动作 a_t。

```
Eq.1  a_VLA = π_θ(I_t, ℓ) ∈ R^7
Eq.2  a_VLA = [Δx, Δy, Δz, Δφ, Δθ, Δψ, g]^T   # 前六维 delta EE 位姿，第七维夹爪
Eq.3  a_t = (1 - c)·a_VLA + c·a^phys,  c = 0.05；|g| > 1.5 时硬覆盖夹爪
Eq.9  a^phys = w ⊙ a^A_t + (1 - w) ⊙ a^raw_t
Eq.8  w = Softmax( diag(M(q))^{-1} )
```

相位划分（Eq.4）：

```
phi_t = DetectPhase(s_t):
  approach  : d_xy^bowl ≥ 6 cm
  grasp     : d_xy^bowl < 6 cm  AND  p^bowl,z < z_lift
  transport : p^bowl,z ≥ z_lift AND  d_xy^plate ≥ 6 cm
  place     : p^bowl,z ≥ z_lift AND  d_xy^plate < 6 cm
```

各相位修正律：

```
approach  : 若 d_xy^bowl ≥ 6cm，则 g ← 0（禁止过早闭合夹爪）
grasp     : a^phys = β·(p* - p^eef) + (1-β)·a^raw,  β = 0.5（朝抓取路径点引导，不平滑）
transport : a^pos_t ← α·a^pos_{t-1} + (1-α)·(a^raw,pos + Δz+·ê_z),  α = 0.92, Δz+ = +2cm
place     : a^phys,pos ← min(1, d_xy^plate / d_thresh)·a^raw,pos   （减速斜坡，Eq.5）
```

Euler-Lagrange 残差与加速度估计（Eq.6–7）：

```
Eq.6  r_EL(q, q̇, q̈) = M(q)·q̈ + C(q, q̇)·q̇ + G(q) - τ
Eq.7  q̈ = 2·(q_next - q - q̇·dt) / dt²,   q_next = q + Δq_VLA
```

变量说明：

| 符号 | 含义 | 来源 |
|------|------|------|
| M(q) | 质量/惯量矩阵 | MuJoCo 空间代数 oracle 𝒪(q_t, q̇_t) |
| C(q, q̇) | 科氏/离心项 | 同上 |
| G(q) | 重力项 | 同上 |
| τ | 当前步执行器力矩 | MuJoCo 直接读取（**不由 Eq.6 反解**） |
| q̈ | VLA 动作蕴含的加速度 | 由 Eq.7 纯运动学估计 |
| w | 逐关节修正权重 | Softmax(diag(M)⁻¹)，低惯量关节权重更大 |
| z_lift | 抬起阈值 = 桌面以上 1cm | LIBERO-Spatial 坐标框架，全 backbone 共享 |

**直觉**：r_EL 之所以不是同义反复（tautological），是因为它比较了两个**独立来源**的量——VLA 动作蕴含的加速度（Eq.7，纯运动学），与机械臂当前实际能产生的力矩（τ，从仿真器直接读取）。Eq.8 的 Softmax(diag(M)⁻¹) 让"低惯量关节"获得更大的修正权重，因为同样的指令在低惯量关节上带来更大的动能风险，更需要被拉一把。

> Remark（论文给的"原理性替代方案"）：严格做法应通过机械臂雅可比 J(q) ∈ R^{6×7} 把惯量投影到任务空间，用操作空间惯量矩阵 Λ(q) = (J·M⁻¹·Jᵀ)⁻¹ ∈ R^{6×6} 再取 Softmax(diag(Λ)⁻¹)。作者承认**Eq.8 的关节空间索引直接套用到任务空间七维动作是启发式近似**，Λ 版本留作未来工作，本文所有数字都不用它。

## 3. 带数字走一遍：玩具例子 (Worked Example)

设某步机械臂 7 个关节（简化成 3 个示意：肩、肘、腕），VLA 输出的任务空间动作 a^VLA = [Δx,Δy,Δz, Δφ,Δθ,Δψ, g] = [0.02, 0.00, -0.01, 0, 0, 0, 1.0]（单位：米 / 弧度 / 夹爪）。控制器周期 dt = 50ms（20Hz）。

**Step 1 — 相位判定**：假设当前 d_xy^bowl = 5cm < 6cm 且 p^bowl,z < z_lift，判定为 **grasp**。

**Step 2 — Branch A 抓取律**：设抓取路径点 p* - p^eef = [0.01, 0.01, 0.00]，β = 0.5：

```
a^A = 0.5·[0.01,0.01,0.00] + 0.5·[0.02,0.00,-0.01]
    = [0.005,0.005,0.0] + [0.010,0.0,-0.005]
    = [0.015, 0.005, -0.005]
```

**Step 3 — Branch B 惯量权重**：设 diag(M) = [0.5, 2.0, 0.1]（肩/肘/腕），则 diag(M)⁻¹ = [2.0, 0.5, 10.0]。Softmax → w ≈ [0.012, 0.003, 0.985]（腕关节惯量最小，权重最大）。把它索引对齐到任务空间前三维（启发式）：

```
a^phys = w ⊙ a^A + (1-w) ⊙ a^raw
       ≈ [0.012·0.015, 0.003·0.005, 0.985·(-0.005)] + [0.988·0.02, 0.997·0, 0.015·(-0.01)]
       ≈ [0.0198, 0.0, -0.00493]
```

**Step 4 — 全局封顶混合**：

```
a_t = (1-0.05)·a^VLA + 0.05·a^phys
    = 0.95·[0.02,0,-0.01] + 0.05·[0.0198,0,-0.00493]
    ≈ [0.02, 0.0, -0.0098]
```

**读法**：单步看，输出几乎等于 VLA 原动作（因为 c=0.05）。但注意 z 分量从 -0.01 被轻微朝 -0.0098 拉——这个 ~0.2% 的偏差在每步累积、且随惯量分布动态变化，就构成了论文所说的"可累积的小修正"。而 gripper 分量 g=1.0 未超 1.5 覆盖阈，仍走混合路径。

> 注：玩具例中的惯量数值为示意，非论文实测；论文以 MuJoCo 内部分量真实计算。（[论文 §III-C](https://arxiv.org/html/2609.33256v1)）

## 4. 工程视角 (Engineering View)

| 维度 | 数值/结论 | 工程含义 |
|------|-----------|----------|
| 单步开销 | Branch A ≈ 0.2ms + Branch B ≈ 0.4ms ≈ **0.6ms** | 远低于 20Hz 的 50ms 控制周期；被 VLA 前向（~30–90ms）完全掩盖 |
| 权重访问 | **零** | 可用于闭源/只读/量化后模型，无需梯度 |
| 超参数量 | c=0.05、z_lift=1cm、δ_grasp=6cm、β=0.5、α=0.92、Δz+=2cm | **跨 4 backbone × 10 任务全固定**，无 per-task/per-backbone 调参 |
| 修正上限 | c=0.05（95% 保持 VLA 原样） | 安全但对需要大改的误差（亚厘米精度）无能为力 |
| 依赖 | URDF/XML 精确的关节与惯量参数 | 标定不准 → 必须 pivot 到数据驱动动力学近似，闭式 EL 残差失效 |
| 夹爪处理 | |g|>1.5 硬覆盖 | 避免混合器吃掉刻意的抓取指令 |

**部署约束**：这是一个典型的"旁路夹层"设计——它不改动作维度的接口（仍是 7-DoF），因此对任何符合该接口的 VLA 都能即插即用。代价是它**只能在动作空间做逐分量启发式映射**（Eq.8 的关节→任务空间对齐是近似的），真正的 Jacobian 一致版本（Eq.10 的 Λ(q)）未被采用。

## 5. 数据与评测 (Data & Eval)

- **主基准**：LIBERO-Spatial，10 个空间推理 pick-and-place 任务（T0–T9），7-DoF Franka Emika Panda。
- **协议**：每任务 5 trials × seed 7（每格 50 episodes），20Hz 闭环 MuJoCo，单张 RTX 4090。选 seed 7 因其落在 10-seed pilot 中位数的 ±1% 内——但**全程单 seed 报告**（为跨 4 backbone 可比的取舍）。
- **Backbone**（跨两轴变化：memoryless/chunked；自回归/力条件/flow-matching）：
  - OpenVLA（7B 单步自回归，memoryless）
  - OpenVLA-OFT（chunked，接近天花板参照，92%）
  - Force-VLA（力残差头，读实时 6-DoF 接触 wrench）
  - Generalist-VLA（flow-matching 集成头，π0/GR00T-N1 风格）
  - 后两者共享同一 OpenVLA-7B 权重，只换 action head。
- **四种推理模式**：Baseline（原始）、Temporal（均匀 EMA α=0.85）、ActionGround(Branch A)、ActionGround + EL blend（仅 OpenVLA 报）。
- **指标**：Success（完成率）、Stability（1 − 归一化累积 EE jerk）、Precision（到抓取/放置路径点接近度）、Efficiency（1 − 归一化完成步数）。
- **聚合结果（Table III）**：Success — OpenVLA 36%→38%、Force-VLA 40%→44%、Generalist-VLA 36%→42%、OpenVLA-OFT 92%→92%（持平）；Stability — OpenVLA 20.1%→36.8%、OFT 86.1%→88.9%、Force-VLA 20.0%→38.2%、Generalist 29.9%→49.2%。
- **跨仿真器**：Robosuite Lift + OSC_POSE，Gaussian XY 噪声 σ∈{0,0.05,...,0.40}。ActionGround 均值 jerk 0.064→0.075（+17%），Baseline 0.064→0.176（+175%），≈10× 抖动鲁棒性；σ=0.40 时 jerk 降 58%，reward 11.06 vs 10.89。
- **真实硬件**：Agilex Piper 6-DoF 抓海绵块放陶瓷盘，**仅定性演示**（定量 per-trial 成功率留待未来工作）。另有一个 matched-seed 的 Robosuite 仿真伴跑，Baseline 成功率 35%→95%。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**：

- 在单步 backbone 上同时提升成功率与稳定性，且**不引入 per-task 回归**（Force-VLA、Generalist-VLA 零回归）。
- 在注入动作噪声时把轨迹抖动鲁棒性提升约 10×。
- 对 chunked backbone（OFT）以零成功率代价换取稳定性提升（OFT 上 EMA 是冗余的，ActionGround 仍加稳定）。

**不能做 / 失败模式**：

| 失败场景 | 具体表现 | 原因 |
|----------|----------|------|
| 遮挡目标几何 | T4、T9 在**所有单步 backbone 上恒为 0%** | FSM 相位谓词依赖 bowl 高度，遮挡时高度阈值判断失效 |
| 亚厘米精度 | T5 改善有限 | 5% 封顶太小，无法彻底解决需要训练期整合物理结构的问题 |
| OpenVLA 特定任务 | T3、T6 从 100%→60%（唯一回归） | 相位谓词对某些几何不成立 |
| 相位谓词无记忆 | 一旦高度阈值对某任务几何错误就无法恢复 | DetectPhase 只用瞬时几何，无前一相位记忆、无接触感知 |
| 非刚性/非 pick-and-place | 未验证 | 论文明确限定刚性物体 pick-and-place |

### 6.1 隐含假设 (Hidden Assumptions)

1. **相位的几何划分对所有任务通用**：approach/grasp/transport/place 四相位 + 6cm/1cm 阈值被假设为跨任务成立，但 T4/T9 的 0% 说明书面上这个假设其实是**任务相关**的。
2. **关节→任务空间的索引对齐是无害近似**：Eq.8 把关节空间惯量权重直接按分量套到任务空间七维动作，作者自承是启发式；在运动学耦合强的构型下，这个近似可能把修正加到"错误的轴"。
3. **仿真提供的 τ 与真实执行器一致**：r_EL 的"非同义反复"依赖 τ 是仿真器瞬时执行器力矩；真机部署时这个量的可获取性与噪声水平未验证（真机只做了定性演示）。
4. **单 seed（seed 7）足以代表**：虽有 10-seed pilot 说明 ±1%，但主表全程单 seed，回归项（T3/T6）的统计显著性未给出。
5. **URDF/XML 标定准确**：作者自己把这条列为首要局限——闭式 EL 残差依赖准确的关节与惯量参数。

## 7. 与相关工作对比 (Comparison)

| 方法族 | 关注点 | 架构 | 训练方式 | 适用场景 |
|--------|--------|------|----------|----------|
| Temporal ensembling (ACT/Octo) | 动作流平滑 | 均匀 EMA | 推理期，无参数 | 需要宏观平滑；但抹平接触爆发 |
| CBF-QP safety filter | 安全证书约束 | 每步解 QP，总是开启的 certificate | 推理期 | 安全关键；但每步需优化器 |
| MPPI 精炼 | 采样+重打分 | 对学习代价采样 rollout | 推理期 | 需要 rollout 预算 |
| ForceVLA / FD-VLA | 力觉接触修正 | 力残差头（需训练期访问） | **训练期** | 接触丰富任务 |
| PINNs / Lagrangian net | 物理约束 | 结构嵌入网络 | **训练期**（软/硬约束） | 数据高效、鲁棒；但需梯度/改架构 |
| **ActionGround（本文）** | 相位+动力学一致性 | 符号 FSM + 解析 EL 残差，**单次闭式残差** | **纯推理期**，零权重更新 | 冻结 VLA 的即插即用精炼；刚性 pick-and-place |

**面试 Tip**：被问到"这篇跟 EMA 平滑 / CBF 安全过滤有什么本质区别？"——一句话答：**EMA 是无差别平滑（会牺牲接触期响应），CBF 每步解 QP（有优化器开销），ActionGround 是"相位条件化 + 单次闭式残差"，每步 <1ms、无优化器、且不 gate（始终开启），把物理当诊断而不是开关。**

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做**部署阶段 VLA 可靠性**的工程师——需要在不重训的前提下给现有策略加一层确定性安全/质量兜底。
  2. 想评估"物理后处理 vs world model"性价比的研究者——本文提供了一个几乎零成本的下界参照。
  3. 研究神经符号运行时接口的人——本文的"离散选律、连续定权"双向耦合值得细看。
- **建議章節路徑**：先讀 §III-C（两条支路 + Eq.3/8/9 的混合器）→ 再看 §IV-B1（Table III 的 per-backbone 数字与回归项）→ 可跳 §II 相关工作（除非要写 survey）→ 最後讀 §V Limitations 核对适用边界。
- **不值得精讀的理由**：如果你不做机器人学习、或者已经熟悉 temporal ensembling / CBF 滤波器这类推理期技巧，讀摘要 + Table III 即可；本文方法论上是**工程夹层**而非训练范式创新。

---
[← Back to Theory](./README.md)

**关键引用**
- 论文: https://arxiv.org/abs/2609.33256
- HTML 全文: https://arxiv.org/html/2609.33256v1
- LIBERO 基准: [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
- OpenVLA: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- OpenVLA-OFT: [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)
