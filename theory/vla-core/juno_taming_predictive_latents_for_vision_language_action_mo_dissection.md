> 历史补录：补回 2026-10-08 的深度阅读任务；论文原始收录日期 2026-10-08；实际补写日期 2026-10-17。本文依据当前读取的指定论文版本，不冒充当时已完成的报告。

# Juno：驯服预测隐变量用于 VLA 模型 (Juno: Taming Predictive Latents for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: Juno: Taming Predictive Latents for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2610.09940v1
> **核心定位**: 用一个 action-conditioned JEPA 同时充当控制对齐表征骨干、预测教师和可适应动力学模型，解决 VLA 生命周期三阶段中预测隐变量的失配、干扰与教师失准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 预测隐变量要按“预训练—策略学习—部署”三阶段分别治理：动态 CLS 对齐控制、解耦推理分支避免干扰动作学习、动力学优先 TTT 先修世界模型再重对齐策略。 |
| 适合精读 | 如果你在做 VLA 表征学习、JEPA/world model 与策略联合训练、部署期 TTT，重点看 §3.2、§3.3、§3.4 和 Table 1–4。 |
| 可以跳过 | 如果你只关心纯视觉自监督或纯语言模型，这篇距离中等；可读摘要和 §1.2 即可。 |
| 落地可行性 | 中：方法链路完整，但真实机器人只验证单臂单任务，TTT 依赖部署 rollouts 与成功验证。 |
| 主要风险 | 真实评测规模小（每条件 20 trials）；附录 D.3 与正文 Table 4 对 TTT 更新是否用于该表存在表述冲突，需作者澄清。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？JEPA 式预测隐变量在表征空间预测未来，但直接塞进 VLA 会碰到三件事：预训练表征与具体机器人控制不匹配、预测梯度干扰动作学习、部署时教师本身在强分布偏移下失准。

发现了什么？作者没有换一个大 VLA  backbone，而是把同一个 action-conditioned JEPA 拆成三种角色：控制对齐的表征骨干、策略学习中的预测教师、部署时可适应的动力学模型。

对 VLA 研究者意味着什么？预测隐变量不是“越多越好”的附加 token，而是需要按阶段治理的训练对象；尤其“失败动作不是好策略目标，但仍是有效动力学监督”是本文最值得复述的工程洞见。

📍 **研究全景时间线**

```
2022  JEPA 路线提出（LeCun et al.）
2023  JEPA 图像自监督（Assran et al.）
2025  V-JEPA 2（Assran et al.）
2026  LeWM / VLA-JEPA / Qwen3GR00T / 本文 Juno ← 当前位置
```

> 只列论文引用中可确认年份的少量节点，不罗列完整参考文献表。本文 arXiv 版本标注为 2026-10-07。局限：这些节点只用于定位，不代表完整领域谱系。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 → 输出 | 频率/时序 | 训练 vs 推理差异 |
|------|-------------|-----------|------------------|
| Control-aligned JEPA encoder | 当前观测 o_t → CLS token c_t 与 patch grid P_t | 随控制步/视频帧 | 预训练时用 embodiment-matched trajectories；策略学习时冻结为 E_J；TTT 时 Stage 1 可全参数更新。 |
| Dynamic CLS loss | 冻结 supervisor 的 patch 变化 → 运动加权目标 u_t；可训练 encoder 的 Δc_t | 每个 transition 一次 | 两阶段：先标准 L_JEPA 训练 LeWM，再冻结 patch supervisor，用 L_dyn 把 patch 动态压到 CLS。 |
| Patch-level fusion | JEPA patch 特征 → 投影到 VLA hidden width → 与原生 visual tokens 做 cross-attention + gated residual | 每个观测/控制步 | 推理时只用当前帧 patch；融合模块训练，JEPA encoder 冻结。 |
| Decoupled predictive reasoning branch | K 个 reasoning queries → 未来 latent 预测 Ẑ_t；hidden states 条件化动作解码 | 每个观测/控制步 | 训练时用未来观测的 CLS 目标做蒸馏；推理时移除未来目标路径，只凭当前观测与语言指令运行。 |
| Flow-matching action decoder | 全 hidden sequence → 动作 chunk | 每个 replan | 训练用 flow-matching；推理用 4 个 Euler steps。 |
| World-model adaptation | 部署 buffer D_ttt 的 transition → 更新 f_θ 与 g_φ | Stage 1，离线/部署后 | 使用全部 transition，包括失败 rollout；目标是 action-conditioned dynamics。 |
| Policy re-alignment | 适应后的冻结教师 E_J^ttt + 成功数据 D_ok → LoRA 策略更新 | Stage 2，离线/部署后 | 只用 verified successful executions；backbone/predictive branch 用 LoRA，action head 全可训练。 |

### 1.2 关键机制 (Key Mechanism)

- **Control-Aligned Pretraining**：作者选 LeWM 作为轻量 action-conditioned JEPA，ViT-Tiny 配置约 15M 参数，直接在像素上训练。关键不是“用大视频教师”，而是用与下游控制同 embodiment 的轨迹训练，使 patch 特征对机器人—物体交互更局部响应。
- **Dynamic CLS loss**：LeWM 只用 CLS token 做预测与规划，但 CLS 会稀释局部接触变化。作者用冻结的 patch supervisor 产生 patch 变化，按运动幅度做 soft weighting，再把加权 patch 动态压到 CLS transition 上。这样保留单 CLS 接口，同时让全局状态对局部运动更敏感。
- **Decoupled Predictive Reasoning**：JEPA patch 通过 gated cross-attention 融入 VLA visual tokens；预测分支用 MoT decoder block，与 vision-language 分支共享 causal attention 但分开非 embedding 参数。K 个 reasoning queries 与 K 个未来 CLS 目标一一对应，做 cosine + SmoothL1 对齐。预测分支同时接收未来对齐梯度和动作生成梯度，因此对齐状态被训练成“可解码为动作”，而不只是“可预测未来”。
- **Dynamics-First TTT**：部署时先把世界模型在所有 observed transitions 上适应，包括失败 rollout；然后冻结适应后的教师，只在 verified successful executions 上重对齐策略。失败动作不能当策略目标，但它的后果仍是有效的动力学监督。

⚡ **Eureka Moment**：THES 关键洞见是——失败动作不是好的策略目标，但仍然是有效的动力学监督；同时预测分支与视觉-语言分支分开非嵌入参数但共享 causal attention，并同时接收未来对齐与动作生成梯度，由此对齐状态可解码为动作而不只是预测未来。

### 1.3 信息流/架构图 (Flow / Diagram)

```
预训练阶段
  o_t ──► f_θ ──► [c_t, P_t]
                 │
                 ├─► g_φ(c_t, a_t) ──► ĉ_t+1  对比 c_t+1：L_JEPA
                 │
                 └─► 冻结 patch supervisor p̄_t
                        │
                        ├─ Δp̄_t,i ──► m_t,i ──► α_t,i ──► u_t
                        └─ 可训练 encoder 的 Δc_t ──► L_dyn

策略学习阶段
  o_t ──► 冻结 E_J ──► P_t ──► 投影 + gated cross-attention ──► VLA visual tokens
  VLA tokens + K reasoning queries ──► MoT 分支 ──► Ẑ_t
  未来观测 ──► E_J^cls ──► Z_t⁺
  L_VLA = L_FM + α L_align
  推理时：移除 Z_t⁺ 路径，只用当前观测与指令

部署阶段
  D_ttt = D_ok ∪ D_fail
  Stage 1：所有 transition ──► 更新 f_θ, g_φ ──► E_J^ttt
  Stage 2：冻结 E_J^ttt；D_ok ──► LoRA 策略重对齐
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
L = L_FM + α·L_align；TTT 先在世界模型上对所有转移最小化 L_pred，再在成功数据上重对齐策略。
```

先给目标：让预测隐变量在预训练时贴合控制相关运动，在策略学习时支持动作生成，在部署时能靠自身转移修正教师。再给公式。

**JEPA 世界模型与动态 CLS**

```
[c_t, P_t] = f_θ(o_t)
ĉ_t+1 = g_φ(c_t, a_t)
L_JEPA = L_pred(ĉ_t+1, c_t+1) + λ_reg R(c)

Δp̄_t,i = p̄_t+1,i − p̄_t,i
m_t,i = ||Δp̄_t,i||₂ / sqrt(D)
m̃_t,i = m_t,i / max(M⁻¹ Σ_j m_t,j, ε)
α_t,i = exp(m̃_t,i / τ) / Σ_j exp(m̃_t,j / τ)
u_t = Σ_i α_t,i Δp̄_t,i
Δc_t = c_t+1 − c_t
L_dyn(θ) = E_(o_t,a_t,o_t+1)~D [ ||Δc_t − u_t||₂² ]
L_LeWM+Dyn = L_JEPA + λ_dyn L_dyn
```

**解耦预测推理与动作生成**

```
L_align = (1/(B K)) Σ_b Σ_k [1 − cos(ẑ_b,k, z⁺_b,k) + λ_sl1 SmoothL1_δ(ẑ_b,k, z⁺_b,k)]
L_VLA = L_FM + α L_align
```

**动力学优先 TTT**

```
θ*, φ* = argmin_(θ,φ) E_(o_t,a_t,o_t+1)~D_ttt [ L_pred(g_φ(c_t,a_t), c_t+1) + λ_reg R(c) ]

L_ttt(ψ) = E_D_ok [ L_FM(ψ) + λ_ttt L_align(ψ; Z̃⁺) ]
```

> 符号与本文一致：c_t 是 CLS token，P_t 是 patch-token grid，f_θ 是 encoder，g_φ 是 latent predictor，p̄ 是冻结 patch supervisor 的输出，M 是 patch 数，D 是特征维度，τ 是温度，α_t,i 是运动感知 soft weight，u_t 是运动加权 patch 目标。K 是未来步数/ reasoning queries 数，Z̃⁺ 是适应后教师给出的未来 CLS 目标。D_ttt = D_ok ∪ D_fail；Stage 1 用全部，Stage 2 只用 D_ok。

直觉：L_dyn 不是重建像素，而是把“局部 patch 动了多少”压成“全局 CLS 该怎么变”。L_align 不是让策略只看未来，而是让预测推理分支与视觉-语言分支共享一条 causal attention 图、但分开非嵌入参数；完整 hidden sequence（含视觉-语言与推理状态）再条件化 flow-matching 动作解码器，使对齐状态可动作化。TTT 的非对称性在于：动力学可以从失败中学习，策略只从成功中学习。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设例子**，只用于验算 Dynamic CLS 的加权与损失，不是论文实验数值。

设 M=3 个 patch，D=2，温度 τ=0.5。冻结 supervisor 给出三个 patch 变化：

```
Δp̄_1 = (1, 0)
Δp̄_2 = (0, 1)
Δp̄_3 = (0, 0)
```

先算运动幅度：

```
m_1 = ||(1,0)||₂ / sqrt(2) = 1 / 1.414 ≈ 0.707
m_2 = 0.707
m_3 = 0
均值 = (0.707 + 0.707 + 0) / 3 ≈ 0.471
m̃ = (0.707/0.471, 0.707/0.471, 0) ≈ (1.5, 1.5, 0)
```

再算 soft weights：

```
exp(1.5/0.5) = exp(3) ≈ 20.086
exp(0/0.5) = exp(0) = 1
分母 ≈ 20.086 + 20.086 + 1 = 41.172
α ≈ (0.488, 0.488, 0.024)
u = 0.488·(1,0) + 0.488·(0,1) + 0.024·(0,0) ≈ (0.488, 0.488)
```

若可训练 encoder 给出 Δc=(0.5, 0.5)，则：

```
L_dyn = (0.5 − 0.488)² + (0.5 − 0.488)²
      ≈ 0.000144 + 0.000144
      ≈ 0.00029
```

若 Δc=(0, 0)，则：

```
L_dyn = 0.488² + 0.488² ≈ 0.476
```

验算结论：静态 patch 的权重只有约 0.024，运动 patch 主导目标；CLS 被拉向局部运动方向。这个例子只说明损失形状，不证明论文方法在真实数据上一定更优。

## 4. 工程视角 (Engineering View)

| 项目 | 论文设置 | 工程含义 |
|------|----------|----------|
| SimplerEnv 评测 | 4 个任务；每任务 24 episodes/run；4 runs，共 96 episodes/task；control 5 Hz，simulation 500 Hz；每 episode 限 120 control steps | 120 control steps 在 5 Hz 下约 24 s 的闭环预算；评测频率与控制频率要分开看。 |
| RoboCasa-GR1 评测 | 24 个任务；每任务 50 episodes；720-step horizon；每个 predicted chunk 执行 12 actions | 长 horizon 任务，chunk 执行数与训练 chunk 长度不是同一概念。 |
| 真实机器人冻结评测 | 每条件 20 trials；60 个 adaptation demos（3 个位置各 20）；部署 15 Hz，16-step chunk ≈1.07 s | 16-step chunk 在 15 Hz 下是开环执行约 1.07 s，然后重新观测/规划；延迟与抖动要看 chunk 边界。 |
| SimplerEnv TTT | Stage A：10 epochs；Stage B：400 steps；batch 8×16 / 8×4；action chunk 16，8 future frames at stride 2；推理 4 Euler steps | “epoch”用于世界模型适应，“step”用于策略重对齐，不能混写。 |
| 真实机器人 TTT | 每条件收集 40 rollouts；Stage A 10 epochs；Stage B 400 steps，checkpoints every 50 steps | TTT 需要额外部署数据与离线优化预算；不是单次前向。 |
| 策略训练 | Bridge/Fractal 与 RoboCasa 各 100,000 updates；PiPER fine-tuning 50,000 updates；global batch 128 或 64 | 训练更新步、评测 episode 数、控制时间步要分开记录。 |
| 表征预训练 | 8 devices × batch 256 = global 2,048；ViT-Tiny，224×224，patch 14，768D hidden，192D projector | 轻量 backbone 是卖点之一，但融合与预测分支仍增加策略侧开销。 |

工程含义：Juno 的推理主路径不是“每一步都跑世界模型规划”，而是把 JEPA 特征融合进 VLA 感知，用预测分支蒸馏未来 latent；TTT 是部署后的离线两阶段更新。真实机器人 15 Hz、16-step chunk 的设计意味着每个 chunk 约 1.07 s，控制频率与 chunk 长度共同决定反应速度。若任务需要更快的闭环纠错，需要缩短 chunk 或提高 replan 频率，但论文未系统评估这一 trade-off。

## 5. 数据与评测 (Data & Eval)

**数据组成（论文 Appendix B.1）**

| 数据 | Episodes | Frames | Rate | Pretraining windows |
|------|----------|--------|------|---------------------|
| BridgeData V2 | 53,192 | 1,893,026 | 5 Hz | 1,520,923 |
| Fractal | 87,212 | 3,786,400 | 3 Hz | 3,178,323 |

策略训练时，Bridge/Fractal sampler 先以 1/2 概率选 source，再在 source 内均匀选 episode 与 timestep；因此“等 source 权重”不等于“所有帧均匀”。编码器预训练窗口用 4 个观测、stride 2、offsets 0,2,4,6，不跨 episode；窗口打乱后 source 质量约 32% Bridge / 68% Fractal。策略输入一张 RGB，resize 到 224×224；每个样本含当前图像、指令、连续 16-step action chunk、8 张未来图像（stride 2）用于 latent alignment。未来图像只在训练时作为目标。

RoboCasa-GR1 用 24 个 GR1 Tabletop Tasks 训练一个 multitask policy，等任务采样权重；输入 ego-view RGB 224×224 与任务指令，无 proprioception；动作是 16-step chunk、29 通道，min–max 归一化到 [-1,1]。真实机器人从 SimplerEnv 策略初始化，在 AgileX Cobot Magic 右 PiPER arm 上收集 60 个 demonstrations，每个 3 个固定 cube 位置各 20 个；训练用 top view 224×224、15 fps 导出，动作 7D end-effector delta。

**评测设置与结果（论文实测）**

- SimplerEnv：Juno 平均 68.5%，Qwen3GR00T 60.9%，VLA-JEPA 57.3%，GR00T-N1.6-Bridge 57.1%。Juno-TTT 进一步到 72.7%。任务级：spoon 86.5→94.8，carrot 59.4→62.5，block 31.3→37.5，eggplant 96.9→95.8（TTT 前后）。
- RoboCasa-GR1：Juno 平均 59.6%，Qwen3GR00T 47.8%，Qwen3OFT 48.8%。论文称在 24 个任务中 21 个优于 Qwen3GR00T，16 个取得最高分。
- 真实机器人冻结策略：in-domain Juno 100%（20/20），Qwen3GR00T 40%（8/20）；OOD background 75%（15/20），background+height 70%（14/20），background+height+object 70%（14/20），基线三个 shifted 条件均为 0%（0/20）。
- 真实机器人 TTT：Gaussian noise 下 Juno frozen 40%（8/20）→ Juno-TTT 65%（13/20）；dynamic lighting 下 55%（11/20）→ 70%（14/20）。

> 注意：Table 3 是冻结策略真实评测，Table 4 是 TTT 真实评测。论文正文与附录 D.3 对“TTT 更新是否应用于 Table 4”存在一处表述冲突；本笔记保留 Table 4 的数值作为论文实测，但将冲突列入风险。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**

- 在 SimplerEnv 四个桌面操作任务上，Juno 相对 Qwen3GR00T 平均提升 7.6 个百分点；TTT 再提升 4.2 个百分点。论文实测，非作者泛化主张。
- 在 RoboCasa-GR1 24 任务上，Juno 平均 59.6%，高于 Qwen3GR00T 的 47.8% 与 Qwen3OFT 的 48.8%。论文实测。
- 在真实单臂桌面任务上，冻结 Juno 在 in-domain 20/20 成功；在 background、height、object 累积偏移下保持 70%–75%，基线为 0/20。TTT 在 Gaussian noise 与 dynamic lighting 下分别从 40%→65%、55%→70%。论文实测，但每条件只有 20 trials。

**不能做什么 / 失败模式**

- 只验证了桌面 manipulation：SimplerEnv WidowX、RoboCasa-GR1、AgileX Cobot Magic 右 PiPER arm。没有移动底盘、双臂协同、人形全身控制证据。
- 真实机器人主要验证一个 cube-to-bowl 任务（“把红 cube 放进蓝 bowl”），并在 object shift 中把 cube 换成 grape 且更新指令；评测只用右臂与 top view。不能外推为通用真实机器人泛化。
- TTT 依赖部署 rollouts 与“verified successful executions”。没有成功验证器时，Stage 2 无法按论文方式运行；失败数据只用于 Stage 1 动力学。
- 强分布偏移下教师会失准，Juno-TTT 试图修正，但论文没有给出教师失准的在线检测指标；仍然假设 Stage 1 的 action-conditioned dynamics 监督在失败数据上有效。
- 附录 A 明确说其表征分析不隔离预训练数据、架构与目标三个因素；因此“控制对齐比 pretraining scale 更重要”是作者主张，不是被完全分离的因果结论。
- 真实 TTT 的 40 rollouts/条件、Stage B 400 steps 是离线协议；在线部署时的数据收集成本、失败清理与成功判定延迟未充分评估。

### 6.1 隐含假设 (Hidden Assumptions)

- 失败 rollout 中经过轻量清洗后的 transition，仍然是可靠的 action-conditioned dynamics 监督；无效片段可被清洗掉。
- 部署时能可靠判断“verified successful executions”；否则 Stage 2 的数据分区不成立。
- 冻结适应后的教师 E_J^ttt 在策略重对齐期间保持稳定；LoRA 加 trainable action head 足以在 scarce test-time data 下稳定更新。
- 真实机器人 top view 224×224 足以支撑 cube-to-bowl 任务；更复杂的遮挡/多物体场景未验证。
- action chunk 16 在 15 Hz 下约 1.07 s 的开环执行不会成为动态任务瓶颈；论文未系统评估 chunk 长度与反应速度的 trade-off。
- 论文正文与附录 D.3 的 TTT 表述冲突不影响 Table 4 数值的有效性；这一点需要作者澄清，属于待证事项。

## 7. 与相关工作对比 (Comparison)

| 方法 | 核心思路 | 预测/世界模型用法 | 训练/适配 | 论文报告场景与关键数字 |
|------|----------|------------------|-----------|------------------------|
| Qwen3GR00T | 强 VLA baseline | 无本文式 JEPA 预测分支 | 监督策略训练 | SimplerEnv 60.9%；RoboCasa-GR1 47.8%；真实 in-domain 40%，shifted 0%。 |
| VLA-JEPA | 用 latent world model 增强 VLA | 未来表征对齐 | 策略学习阶段引入预测 | SimplerEnv 57.3%。 |
| GR00T-N1.6-Bridge / GR00T-N1.6 | 通用机器人基础模型 | 非本文控制对齐 JEPA | 预训练+适配 | SimplerEnv 57.1%（Isaac-GR00T-N1.6-Bridge，Table 1）；RoboCasa-GR1 47.6%（GR00T-N1.6，Table 2）。 |
| Qwen3OFT | Qwen3 系列策略变体 | 非本文 JEPA 路线 | 监督微调 | RoboCasa-GR1 48.8%。 |
| Juno | 一个 action-conditioned JEPA 贯穿三阶段 | 控制对齐表征骨干 + 预测教师 + 可适应动力学模型 | 动态 CLS 预训练；解耦预测推理；动力学优先 TTT | SimplerEnv 68.5%，TTT 72.7%；RoboCasa-GR1 59.6%；真实 in-domain 100%，OOD 70%–75%；TTT noise 65%、lighting 70%。 |

面试 Tip：被问到“Juno 和把 JEPA 塞进 VLA 有什么区别”时，一句话答——**它不是把未来 latent 当额外 token，而是把同一个 JEPA 分成表征骨干、预测教师、动力学模型三种角色，并且用“失败动作可做动力学监督、不可做策略目标”的非对称 TTT 把部署数据用起来。**

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：做 VLA 表征学习与 world model 联合训练的研究者；要评估 JEPA/预测隐变量在机器人策略中落地方式的工程师；研究部署期 TTT、分布偏移鲁棒性的读者。
- **建议章节路径**：先读摘要 + Fig. 1 + §3.2–§3.4，建立三阶段机制；再看 Table 1–4 与 Appendix B/D，核对训练预算、评测次数与 TTT 协议；Related Work 可在熟悉 JEPA/VLA 后跳读。
- **不值得精读的理由**：如果你不做机器人学习、不关心部署期适应，或只需要一个 SimplerEnv 排行榜数字，读摘要与 Table 1 即可；真实机器人证据只有单臂单任务、每条件 20 trials，不能支撑通用泛化结论。

---

关键引用：论文链接 https://arxiv.org/abs/2610.09940v1 。论文正文未提供可确认的代码仓库信息，本笔记不引用额外 URL。

[← Back to Theory](./README.md)