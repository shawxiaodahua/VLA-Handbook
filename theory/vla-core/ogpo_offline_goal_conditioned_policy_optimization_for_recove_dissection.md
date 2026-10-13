# OGPO：面向可恢复 VLA 的离线目标条件策略优化 (Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-13
>
> **论文**: OGPO: Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2610.12914v1 （NeurIPS 2026 接收，v1 提交于 2026-10-09）
> **作者**: Xule Gao, Xi Wang, Rui Wang, Changwen Zheng, Chuxiong Sun（中科院软件所 / UCAS）
> **核心定位**: 把 VLA 后训练从「轨迹级动作模仿」改写为「关键状态可达性学习」——只用离线数据、无需在线 rollout，通过 gripper 分段 + 目标条件过程奖励 + AWAC 加权回归，让策略在偏离示范轨迹后还能「走回来」。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 在纯离线设定下，用 goal-conditioned 过程奖励 + AWAC 加权模仿，比标准 SFT 显著更「可恢复」：LIBERO 平均 77.1%→97.1%，Long 套件 43.9%→93.2%。 |
| 適合精讀 | 做 VLA 离线后训练 / 后训练 RL 的研究者；需要在真实机器人上做数据高效 fine-tune、且无法承担在线 rollout 成本的工程师。重点看 §3 方法（分段 + 奖励设计）与 §4 实验（扰动基准）。 |
| 可以跳過 | 如果你只关心 online RL（PPO/GRPO 系）或世界模型路线，这篇只是「另一个离线替代」，距离中等。 |
| 落地可行性 | 中—高（原因：完全离线、无架构改动、只加一个 value head；但对「gripper 事件可表征任务进度」的假设较强）。 |
| 主要風險 | 分段强依赖 gripper 开合事件；离线数据质量决定恢复上限，严重 OOD 偏离仍无能为力。 |

💡 **X-Ray 开场**
一个只会「照着示范录像做动作」的机器人，一旦被抓到一半手滑、或者位置偏了，就会一直错下去——因为它从没学过「怎么回到正轨」。OGPO 的做法是：不再要求它逐步复刻专家动作，而是把每条示范切成语义小目标（伸手→抓→搬→放→关门），然后教它「到达这些小目标的中间状态才是得分点」。于是即便中途出错，策略也有明确的「回到关键状态」信号；而这一切只用离线数据完成，不需要在真实环境里反复试错。

📍 **研究全景时间线**

```
2023  RT-2/RT-1 奠定 VLA 范式 ──▶ 2024-25 OpenVLA / Octo / π0 / π0.5（SFT+flow matching）
        │
        ▼
2025-26  RL 后训练兴起：iRe-VLA、VLA-RL、SimpleVLA-RL、πRL（PPO/GRPO 需在线 rollout）
        │
        ▼
        ├── 世界模型/仿真器路线：World-Env、WoVR、WMPO（降成本但仍需 rollouts）
        │
        ▼
   【本文 OGPO】 首个「SFT 正则 + 离线目标条件 RL」融合框架 ← 当前位置
        （痛点：在线 RL 太贵；纯 SFT 不可恢复）
        │
        ▼
   局限：gripper 分段不可泛化到非抓取型进度；纯离线受数据质量上限制约
```

## 1. 核心架构/方法总览 (Overview / Architecture)

OGPO 不改变 VLA 主干结构，而是在「后训练阶段」插入一个三阶段离线流水线：**分段重标注 → 目标条件过程奖励 → 离线优势加权策略优化**。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 训练/推理差异 |
|------|------|------|----------------|
| 轨迹分段器 (Gripper Segmentation) | 离线轨迹 tau = {(o_t,a_t), l} | 非重叠段 {tau^1..tau^M} | 仅训练前预处理，推理不参与 |
| 目标重标注 (Goal Relabel) | 段 + 指令 + gripper 转换 | 语义局部目标 g^m、完成标签 y^m | 仅训练前预处理；成功轨迹 y=1，失败轨迹用 VLM 判定 |
| Value 网络 V_psi | 观测 o_t + 指令 l | 折扣回报估计 | 训练：n-step TD；推理：**不使用**（不参与动作输出） |
| 策略 π_θ | o_t + l | 动作 a_t | 训练：L_RL + L_SFT；推理：与原 VLA 完全一致 |
| VLM 标注器 (Qwen2.5-VL-72B-Instruct) | 段首/中/尾帧 + 候选目标 | SUCCESS/FAILURE 标签 | 仅离线标注失败轨迹段，一次性成本 |

> 关键工程取舍：goal g_t **只用于生成奖励**，绝不作为额外输入喂给 policy 或 value——避免破坏预训练的视觉-语言表征，也保持推理接口零改动。

### 1.2 关键机制 (Key Mechanism)

- **为什么用 gripper 开合做分段锚点**：在桌面操作任务里，抓/放/运输/对齐这些语义阶段几乎都与「夹爪开合事件」同步，因此 gripper 状态变化是廉价又鲁棒的进度信号（论文 §3.1 / Appendix B.1）。
- **为什么失败轨迹也要用**：失败轨迹里往往前几段是成功的（例如先成功抓取，后面放偏了）。用 VLM 逐段判定，可把「已完成的中间进度」保留为正样本，把真正失败的那段标为负样本，实现更细粒度的信用分配（Appendix B.3-B.5）。
- **为什么奖励要 dense**：段完成信号本是稀疏 0/1 标签；用归一化进度变量 rho 把它铺成段内稠密奖励，让 value 函数能区分「推进中」与「卡住」（§3.2）。
- **为什么保留 SFT**：纯 AWAC 加权更新会漂离预训练动作先验；SFT 项作为粗粒度行为正则，锚定可行动作分布、稳定优化（§3.3，消融中移除 SFT 性能下降证实此点）。

⚡ **Eureka Moment**：离线示范里其实藏着「哪些状态是可达的任务进度点」的隐式证据——把示范从「逐步动作标签」重新解读为「关键状态可达性监督」，就能不碰环境地教会策略**从偏差状态回到正轨**。

### 1.3 信息流/架构图 (Flow / Diagram)

```
离线数据 D_demo (成功) + D_fail (失败)
        │
        ▼
[1] Gripper-Based Segmentation
    边界 B = {t | c_t != c_{t-1}}（保留稳定切换点）∪ {1, T}
    tau  ──▶  {tau^1, tau^2, ..., tau^M}
        │
        ▼
[2] Goal Relabeling + Completion Labeling
    成功段  → 语义目标 g^m, y^m = 1
    失败段  → VLM 判定 (Qwen2.5-VL-72B) → y^m ∈ {0,1}
    得到 D~ = {(o_t, l, a_t, g_t, y_t)}
        │
        ▼
[3] Goal-Conditioned Process Reward
    rho_t = y_t · (t - s_m)/(e_m + 1 - s_m)
    r_t = -λ_step + R_prog·(rho_{t+1} - rho_t) + δ_t·(R_succ·y_t - R_fail·(1-y_t))
        │
        ├──▶ Value 学习：n-step Bellman backup，最小化 TD 误差 → V_psi
        │
        ▼
[4] Offline Goal-conditioned Policy Optimization (AWAC-style)
    A_t = n-step return - V_psi(o_t,l)
    ω_t = sg[ min( exp(A_t / β), ω_max ) ]
    L_total = λ_RL · L_RL(θ) + λ_SFT · L_SFT(θ)
        │
        ▼
   优化后的 VLA 策略 π_θ（推理接口与原来完全一致）
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
L_total = λ_RL · ( -E[ ω_t · log π_θ(a_t | o_t, l) ] ) + λ_SFT · ( -E[ log π_θ(a_t | o_t, l) ] )
其中 ω_t 由「关键状态可达性」过程奖励导出的优势 A_t 决定
```

**目标**：在纯离线设定下，让策略不仅模仿示范动作，还能被优化去「抵达任务进度关键状态」。

**① 归一化进度变量**（论文 Eq. 3）：

```
rho_t^m = y_t · (t - s_m) / (e_m + 1 - s_m),   t ∈ [s_m, e_m+1]
```
- y_t=1 时，rho 在一个段内从 0 线性升到 1（稠密进度信号）；
- y_t=0 时，整段 rho=0（失败段不给予正向进度奖励）。

**② 过程奖励**（论文 Eq. 4）：

```
r_t = -λ_step + R_prog · (rho_{t+1}^m - rho_t^m)
      + δ_t^m · ( R_success^goal · y_t - R_fail^goal · (1 - y_t) )
```
| 变量 | 含义 | 论文取值 |
|------|------|----------|
| λ_step | 每步小惩罚，抑制冗余动作 | 0.01 |
| R_prog | 段内进度奖励系数 | 1.0 |
| R_success^goal | 段末达成目标的奖励 | 1.0 |
| R_fail^goal | 段末未达成的惩罚 | 2.0 |
| δ_t^m | 段末指示符 = I[t = e_m] | {0,1} |

**③ Value 的 n-step TD 目标**（Eq. 6）与 **TD 损失**（Eq. 7）：

```
V_hat_t = Σ_{i=0}^{n-1} γ^i · r_{t+i} + γ^n · V_psibar(o_{t+n}, l)
L_V(ψ)  = E[ ( V_ψ(o_t,l) - V_hat_t )^2 ]
```

**④ 优势与 AWAC 权重**（Eq. 8, 9）：

```
A_t  = ( Σ_{i=0}^{n-1} γ^i r_{t+i} + γ^n V_ψ(o_{t+n}, l) ) - V_ψ(o_t, l)
ω_t  = sg[ min( exp(A_t / β), ω_max ) ]
```
- β 控制优势加权的锐度；ω_max 防止少数高优势样本主导更新；sg(·) 为 stop-gradient。

**⑤ 最终目标**（Eq. 12）：

```
L_total(θ) = λ_RL · L_RL(θ) + λ_SFT(θ) · λ_SFT
L_RL(θ)  = -E[ ω_t · log π_θ(a_t | o_t, l) ]     （优势加权行为回归）
L_SFT(θ) = -E[ log π_θ(a_t | o_t, l) ]           （成功示范上的动作先验）
```

**直觉**：SFT 提供「别偏离可行动作」的粗粒度先验；RL 项用优势权重放大那些「真正推进任务进度」的动作、压低无效动作，于是策略被允许在必要时偏离示范路径，去追求关键状态。

> 符号与本文/相关文档保持一致：π_θ 为 VLA 策略，V_ψ 为 value 网络，τ/tau 为轨迹，o_t 观测，a_t 动作，l 语言指令，g 语义局部目标，y 段完成标签，γ 折扣因子，n 为 n-step 回看步数。
> TODO: 论文正文未给出 γ、n、β、ω_max、λ_RL、λ_SFT 的具体数值（仅给出奖励系数 λ_step/R_prog/R_succ/R_fail），如有需要请查附录 E。

## 3. 带数字走一遍：玩具例子 (Worked Example)

设一条成功示范：「把碗放进抽屉并关上」，共分 3 段（依 gripper 事件切分）。

```
seg1: reach→grasp    (开→合)   s_1=1,  e_1=10   y=1
seg2: move→lay       (合→开)   s_2=11, e_2=25   y=1
seg3: close drawer   (合)      s_3=26, e_3=35   y=1
```

取参数：λ_step=0.01, R_prog=1.0, R_succ=1.0, R_fail=2.0。

**seg1（y=1）**：rho_t = (t-1)/(10+1-1) = (t-1)/10。
- t=5 → rho_5 = 0.4，t=6 → 0.5 → 段内进度奖励 = 1.0×(0.5-0.4) = +0.1；末尾再叠加 -0.01 步惩罚。
- t=10 → 段末 δ=1：r_10 = -0.01 + 1.0×(1.0-0.9) + 1.0×1 = +1.09。

**seg2（y=1）**：rho 重新从 0 升到 1，同理段末得到约 +1.09。

**seg3（y=0，假设关门失败）**：整段 rho=0，段内无进度奖励；段末 δ=1：
```
r_35 = -0.01 + 0 + 1.0×1.0×(0) - 2.0×(1-0) = -2.01
```
→ 失败段末被重罚，value 学会「进入该状态后回报为负」。

**一个失败轨迹的重标注**：整条任务失败，但 seg1/seg2 成功、seg3 失败。VLM 判定后 y=(1,1,0)。于是前两段的动作获得正优势（该学），第三段获得负优势（该压）。价值曲线（Appendix C）显示局部高值转折点恰好对齐「接近目标物体、稳定抓取、拉开抽屉、放置到位」等语义关键状态——证明 value 确实在估计「中间进度」而非仅最终成败。

## 4. 工程视角 (Engineering View)

| 维度 | 情况 | 工程含义 |
|------|------|----------|
| 环境交互 | **零在线 rollout** | 相比 PPO/GRPO 系，无需仿真器/真机回环，训练成本≈纯监督 + 一个 value head 的额外前向 |
| 架构改动 | 仅加一个轻量 value head（挂在 VLA 主干中间表示上） | 推理路径不变，动作头（flow-matching / 回归 / 离散 token）原样保留 → 部署零迁移成本 |
| 一次性标注成本 | 失败轨迹逐段过 Qwen2.5-VL-72B 判定 | 数据预处理阶段的一次性算力；成功轨迹无需 VLM（直接 y=1） |
| 分段鲁棒性 | gripper 切换点需「持续一小段时间」才保留 | 抑制短时抖动造成的噪声边界；对指令抖动容错 |
| 优化稳定性 | SFT 项作为正则 | 防止 AWAC 更新漂离预训练先验，减少灾难性遗忘 |
| 覆盖上限 | 纯离线，受数据分布约束 | 面对严重 OOD 偏离（示范中从未出现过的状态）恢复能力有限 |

**落地要点**：OGPO 是「后训练配方」而非新模型。若你已有一个跑通 SFT 的 VLA（π0/π0.5/OpenVLA 系），接入成本主要是：(1) 写一个基于动作里 gripper 维度的分段器；(2) 准备失败轨迹或用现有失败样本；(3) 接一个 value head + AWAC 损失。控制频率与延迟不受影响，因为推理时 value 完全不参与。

## 5. 数据与评测 (Data & Eval)

| 基准 | 用途 | 扰动/设定 |
|------|------|-----------|
| LIBERO | 主实验（Spatial/Object/Goal/Long 四套件） | 标准成功率 |
| MetaWorld-MT50 | 跨主干泛化（Easy/Hard/Very Hard） | 多任务 |
| LIBERO-Plus | 静态分布漂移鲁棒性 | 相机视角 / 机器人初始态 / 视觉噪声 / 物体布局 |
| **LIBERO-ReAct**（本文提出） | 执行中途扰动下的「可恢复性」 | 动作噪声 / gripper 状态翻转 / 末端位置扰动 |

**LIBERO-ReAct 设置**：每个 episode 随机选 20 步区间，在策略输出动作之后、执行之前注入扰动：
- 动作噪声：机械臂 U[-0.1, 0.1]，夹爪 U[-0.5, 0.5]；
- gripper 翻转：把命令翻成相反状态（该开变关、该关变开）；
- 末端位置扰动：位置运动命令加 U[-0.2, 0.2] 噪声。

**主干与基线**：π0、π0.5；对比 few-shot SFT / full-shot SFT，另列 OpenVLA、SmolVLA、UniVLA、OpenVLA-OFT 作参照，并用 RLinf 的 PPO 在线 RL 作为「在线代表」。所有鲁棒性测试**不做扰动专用微调**。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**：
- 长程任务上大幅提升成功率：LIBERO-Long 43.9%→93.2%（full-SFT 为 92.4%）；
- 跨主干稳定增益：MetaWorld π0 50.8%→81.0%（在线 PPO 85.8%），Hard +28.4%、Very Hard +53.6%；
- 中途扰动恢复：gripper 状态扰动下成功率 76.8%→87.7%；整体鲁棒性分 76.7%→81.8%；
- 训练更高效：π0 主干 3000 步时 LIBERO-Spatial 93.6%（+13.6）、LIBERO-Goal 92.3%（+11.1）。

**不能做/受限**（论文 §5 明说）：
- 「任务进度」必须能被 gripper 事件自然刻画——非抓取型任务（如纯推、纯部署、无夹爪事件的长流程）分段可能失效；
- 纯离线框架，上限受离线数据质量约束，面对严重 OOD 偏离仍难以恢复；
- 恢复能力是「回到示范覆盖过的关键状态」，而非「发现全新解法」。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设 1**：gripper 开合事件与语义阶段一一同步。对双臂/灵巧手/非抓取操作可能不成立（论文自认局限）。
- **假设 2**：VLM 能可靠判定失败轨迹中的段完成度。判定噪声直接进奖励，影响 value 质量——论文未报告 VLM 标注准确率。
- **假设 3**：示范数据里「存在可行 off-demonstration 状态」可被关键状态覆盖。若真实偏差远超示范覆盖，恢复信号无锚点。
- **假设 4**：动作头形式（token/回归/flow-matching）可统一写成 log π_θ(a|o,l) 的似然式目标——对 flow-matching/扩散头，此「似然」实为替代损失，理论一致性待严格说明。> TODO: 待补充 flow-matching 情况下该目标的确切形式与梯度。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 架构/训练 | 需要在线交互 | 适用场景 |
|------|--------|-----------|--------------|----------|
| π0.5 SFT | 轨迹级动作模仿 | 监督回归/flow matching | 否 | 示范覆盖充足、无扰动的稳定场景 |
| PPO / GRPO 系（VLA-RL、SimpleVLA-RL、πRL） | 任务级回报优化 | 在线策略梯度 | **是**（昂贵） | 可大规模仿真的场景 |
| iRe-VLA | 在线 RL + SFT 交替 | 在线迭代 | 是 | 追求稳定性但可回环 |
| 残差 RL / RL 蒸馏 | 修正策略 / 专家蒸馏 | 冻结主干 + 轻量修正 | 部分需要 | 已有强 generalist，需微调 |
| 世界模型系（World-Env、WoVR、WMPO） | 用 WM 降交互成本 | 仿真器/世界模型 rollout | 间接需要 | 可建可靠仿真器 |
| **OGPO（本文）** | **关键状态可达性 / 可恢复性** | **纯离线 AWAC + SFT 正则** | **否** | 无法在线回环、但需离线鲁棒恢复的真机场景 |

**面试 Tip**：被问到时，一句话答——「OGPO 的贡献不在新模型，而在把离线示范从『逐步动作标签』重新解读为『关键状态可达性监督』：用 gripper 分段 + 目标条件过程奖励 + AWAC 加权回归，在纯离线设定下逼近在线 PPO 的恢复能力，代价是强依赖 gripper 事件可表征任务进度。」

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做 VLA 离线后训练/后训练 RL 的研究者——这套「分段 + 过程奖励 + AWAC」配方是可直接迁移的模板；
  2. 在真实机器人上做数据高效 fine-tune、且**无法承担在线 rollout** 的工程团队；
  3. 关心「模仿学习可恢复性 / 复合误差」问题的 IL/RL 交叉研究者。
- **建議章節路徑**：先讀 §3 方法（Eq. 3-4 奖励设计、Eq. 8-12 策略优化）→ 再看 §4.2/§4.3 主结果与消融（重点看 LIBERO-Long 与 LIBERO-ReAct）→ 附錄 B（分段与 VLM 标注细节）可選讀 → §4.1 的基准罗列可快速跳過。
- **不值得精讀的理由**：如果你不做机器人学习、或已熟悉 AWAC/离线 RL + SFT 融合范式，读摘要与 §1 结论即可——方法本质是「已知组件的巧妙组合」，而非全新算法。

---
[← Back to Theory](./README.md)

**关键引用链接**
- 论文: https://arxiv.org/abs/2610.12914v1
- π0.5 (Black et al., 2025): https://proceedings.mlr.press/v305/black25a.html
- π0 (Black et al.): https://arxiv.org/abs/2410.24164
- AWAC (Nair et al., 2020): https://arxiv.org/abs/2006.09359
- LIBERO (Liu et al., 2023): NeurIPS 36, pp. 44776-44791
- LIBERO-Plus (Fei et al., 2025): https://arxiv.org/abs/2510.13626
