# 什么时候停止惩罚动作？面向 VLA 强化学习的 Martingale GRPO 拆解 (When to Stop Penalizing Actions? Martingale GRPO for VLA Reinforcement Learning)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-26
>
> **论文**: When to Stop Penalizing Actions? Martingale GRPO for VLA Reinforcement Learning
> **链接**: https://arxiv.org/abs/2610.32431
> **核心定位**: 它指出 GRPO 把整条失败轨迹共用的负 advantage 施加到所有 action chunk 上，会连带惩罚后续恢复动作；MART-GRPO 用冻结 VLM 定位“最后可见错误且未观察到恢复”的区间，把失败反馈截断到该前缀，并在理想停点下证明这是完整 GRPO 组梯度估计量的条件投影。

> **依据版本说明**：本文依据来源提供的 arXiv:2610.32431v1 HTML 全文（提交日期 2026-10-23）写作。来源记录的 `abstract_verification` 为空；候选摘要与正文摘要逐句对照未发现实质差异，因此本文不以候选摘要替代正文主张。若后续以官方 PDF 复核，应以 PDF 正文表格为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 失败轨迹 ≠ 每个动作都失败；MART-GRPO 保留失败前缀的负反馈、屏蔽后缀，理想停点下保持参考 GRPO 估计量的期望且不增加协方差。 |
| 适合精读 | 如果你在做 VLA 的 RL 精调、credit assignment、GRPO 变体，重点看 §2.2–2.4、§3.2–3.3、§4.2–4.3。 |
| 可以跳过 | 如果你只关心 VLA 预训练架构、真实机器人部署或通用机器人策略，这篇是 RoboTwin 2.0 仿真上的 RL 精调研究。 |
| 落地可行性 | 中高：不改 actor 架构、保留原 GRPO 组 advantage 与 token 归一化；但需要离线事件标注和冻结 VLM 推理。 |
| 主要风险 | 回顾性边界不自动满足“零剩余信用”；VLM 定位错误、策略演化后的分布偏移会把偏差带入梯度。 |

💡 **X-Ray 开场**
这篇论文解决什么问题？GRPO 在失败轨迹上给所有 action chunk 同一个负 advantage，可能把“掉物体后尝试重抓”的恢复动作也一起惩罚。它发现了什么？若在“剩余条件信用为零”的位置停止惩罚，被删掉的梯度贡献条件期望为零，截断后的组梯度是完整估计量的条件投影。对 VLA 研究者意味着什么？可以把失败反馈的时间支撑当作一个可学习、可审计的模块，而不是只能调大或调小整条轨迹的负 advantage。

📍 **研究全景时间线**（仅列来源可确认年份的节点）

```
2024  GRPO 组相对优势用于 LLM 推理 (Shao et al.)
2025  OpenVLA-OFT / VLA-RL 把 VLA 接到 SFT 后的 RL 精调
2026  Temporal GRPO、PCM、μ-GRPO、ESPO、PS-PPO 等改反馈粒度/前缀/停止
2026-10  MART-GRPO v1：用可见错误与恢复线索做失败后缀截断 ← 本文位置
```

本文的局限也在这个时间线里：它没有把 Temporal GRPO、PCM、μ-GRPO、ESPO、PS-PPO 放进同一实验复现，只做了文字对比；策略评测也只在 OpenVLA-OFT + RoboTwin 2.0 的七任务仿真中完成。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块/模型 | 角色 | 输入 | 输出/时序 | 训练/推理 | 记忆机制 |
|-----------|------|------|-----------|-----------|----------|
| OpenVLA-OFT（actor） | 被执行、被 RL 优化的 VLA 策略 | 头相机视图 + 本体状态；任务指令 | action-token 动作；每个 action chunk 为 25 control steps、14 维/step | 从任务特定 SFT checkpoint 开始；actor LoRA 关闭；PPO/GRPO 更新 | 无显式跨 episode memory；action chunk 是执行单元，不是记忆窗口 |
| Qwen3.8-27B（event VLM） | 离线事件定位器，RL 中冻结 | 任务指令、三路相机视频、本体观测、帧到 chunk 对齐信息 | JSON：可见结果、错误类型、错误区间 [l_i,u_i]、是否恢复、证据帧 | 冻结 backbone，只训练一个 LoRA adapter；rank 8、scaling 16 | 读取一条完整 rollout 的视频做回顾性判断；不是策略的持久记忆 |
| MART-GRPO | 训练算法/反馈掩码规则 | 组内终端奖励、失败 rollout 的 VLM 区间 | 失败前缀掩码 w_{i,k}；成功轨迹全保留 | 与 GRPO/PPO 管线结合；掩码 detached | 无独立记忆模块；掩码按 rollout 生成 |

> **时序口径**：若均匀采样 K 帧、间隔 Δt，首末跨度是 (K-1)×Δt；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 H×Δt。论文的 25 step/chunk、每 chunk 最多 6 帧采样、以及 B_i 都是具体实验设置，不能合并成“所有实验共享的记忆窗口”。连续未来 chunk 的预测区间是 future prediction horizon，不是历史上下文。

### 1.2 关键机制 (Key Mechanism)

- **问题定位**：GRPO 的 A_i 是轨迹级系数，失败轨迹的负 A_i 会被施加到该轨迹所有 action chunk。论文用 Hammer 的失败例子说明：掉锤子后策略可能尝试重抓，但整条失败轨迹仍共享同一个负 advantage。
- **理论停点**：在固定策略下，成功概率 P_k 与累计 policy score S_t 都是鞅；剩余条件信用 χ_t 度量“给定已观测历史，未来动作对策略梯度的期望贡献”。当 χ_T=0 时，停止惩罚后的分数是完整分数的条件投影。
- **事件引导的回顾性边界**：VLM 在完整失败视频中定位“最后可见错误且未观察到恢复”的区间，取右端点 B_i=u_i；失败反馈保留 chunk 0 到 B_i，之后屏蔽。
- **语义回退**：若 VLM 预测成功、结果不确定、错误类型不明确、区间为空，或报告已恢复，则保留全轨迹反馈，B_i=K_i-1。
- **保留原归一化**：成功轨迹损失项保留；失败轨迹仍用原始有效 token 数 N_i 归一化，不因截断而重归一化。

⚡ **Eureka Moment**：不是给失败轨迹换一个更小的负 advantage，而是把负 advantage 的**时间支撑**截断到可见错误区间；在理想停点下，这是完整 GRPO 组梯度估计量的条件投影。

### 1.3 信息流/架构图 (Flow / Diagram)

```
任务指令 + 头相机 + 本体状态
        │
        ▼
OpenVLA-OFT actor（action-token；25 step/chunk；14 dim/step）
        │ rollout 执行到终止或预算耗尽
        ▼
终端奖励 R_i ∈ {0,1} ──► 组内标准化 ──► A_i
        │
        ├─ R_i=1：全部 chunk 保留
        └─ R_i=0：完整 rollout 视频 ──► 冻结 VLM（Qwen3.8-27B + LoRA）
                                      │ 输出最后可见错误且无恢复区间 [l_i,u_i]
                                      ▼
                                 B_i = u_i；否则 fallback B_i=K_i-1
                                      │
                                      ▼
                         w_{i,k}=1{k≤B_i}；Ã_{i,k}=w_{i,k}A_i
                                      │
                                      ▼
                         PPO/GRPO loss（原 N_i 归一化）
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
失败反馈掩码：Ã_{i,k} = w_{i,k} · A_i
w_{i,k} = 1（成功轨迹）；w_{i,k}=1{k≤B_i}（失败轨迹）
```

先给目标：在保留成功轨迹损失、组 advantage 和原始 token 归一化的前提下，删除失败轨迹后缀中“条件期望为零”的梯度贡献，使估计量保持参考期望且不增加协方差。再给核心公式。

**GRPO 组 advantage 与掩码**

```
A_i = (R_i - mean(R_g)) / (s_g + eps_num)

Ã_{i,k} = w_{i,k} A_i

w_{i,k} = 1, 若 R_i=1
w_{i,k} = 1{k ≤ B_i}, 若 R_i=0
B_i ∈ {-1,0,...,K_i-1}
```

| 符号 | 含义 |
|------|------|
| R_i | 轨迹 i 的终端二值奖励 |
| g(i) | 轨迹 i 所属的 rollout group |
| s_g | 组内奖励标准差 |
| eps_num | 数值稳定项，论文设为 10^-6 |
| K_i | 轨迹 i 的 action chunk 数 |
| B_i | 失败反馈保留到的最后一个 chunk；-1 表示空失败前缀 |
| N_i | 轨迹 i 的原始有效 action token 数 |

**剩余条件信用**

```
P_k = E[R | F_k]
Z_k = ∇θ log πθ(a_k | h_k)
S_t = Σ_{k=0}^{t-1} Z_k
ΔP_k = P_{k+1} - P_k

χ_t = E[ R · (S_H - S_t) | F_t ]
    = E[ Σ_{k=t}^{H-1} Z_k · ΔP_k | F_t ]
```

直觉：χ_t 是“给定已观测前缀，继续执行对未来策略梯度的期望贡献”。它可以在动作之间相互抵消，所以 χ_t=0 不等于每个后续动作的局部 advantage 都为零。

**理想停点的条件投影**

```
χ_T = 0  ⇔  S̃_T = E[S_H | K_T]
```

其中 S̃_T 是停止后保留的分数：成功轨迹保留完整 S_H，失败轨迹保留到 T 的 S_T。K_T 记录停止后的历史、最终结果，以及成功轨迹的完整记录。

**组级定理**

```
G_g = (1/m) Σ_i (A_i/N_i) S_H^(i)
G_{g,T} = (1/m) Σ_i (A_i/N_i) S̃_{T_i}^(i)

G_{g,T} = E[G_g | K]

Cov(G_g) - Cov(G_{g,T}) = E[D_g D_g^T] ⪰ 0
D_g = G_g - G_{g,T}
```

这说明：在定理条件下，理想掩码保持组估计量的期望，并且协方差不增。注意这是 PPO clipping 之前的参考估计量分析，不是对实际 clipped 优化轨迹的保证。

**回顾性边界的残差**

```
η_B = E[D_B | K_B]
G_B = E[G | K_B] - η_B
eps_B^2 = E[ ||η_B||^2 ]
```

VLM 使用完整视频选择边界，所以 B 可能依赖未来信息。若 η_B=0，才等价于条件投影；否则期望和方差都可能偏移。论文明确指出：可见事件定位本身并不自动保证 η_B=0。

**偏差–方差权衡**

```
E||\bar G_w - μ||^2 - E||\bar G - μ||^2
= ||d||^2 - Δ_V / n

d = E[D]
Δ_V = tr Cov(G) - tr Cov(G_w)
```

严格改善 MSE 的条件是：

```
Δ_V > n ||d||^2
```

直觉：截断带来方差下降，但回顾性选择可能带来期望偏移；batch 越大，方差收益相对固定偏差越小。理想停点 d=0 时，MSE 降低量为 E||D_g||^2/n。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设性验算**，按论文 Appendix C 的两块 Bernoulli 构造改写，不是新实验。目的只是把“零信用停点”和“非零信用停点”的差别算清楚。

设单条轨迹有两个 chunk：chunk 0 无参数相关分数；chunk 1 的动作 a ~ Bernoulli(p)，p=0.5，score Z=a-p。固定 baseline b=0.5，参考估计量 G=(R-b)Z。在 T=1 时移除失败轨迹的后缀 Z。

**情形 A：理想停点，χ_1=0。** 令 R 为独立公平硬币，与动作无关。

```
E[G] = 0
Var(G) = 1/16

失败时 G_w = 0；成功时 G_w = G
E[G_w] = 0
Var(G_w) = 1/32
```

验算：G=(R-0.5)(a-0.5)，四个等概率组合给 G ∈ {+0.25,-0.25}，均值 0，方差 1/16。掩码后失败一半直接置零，成功一半保留原值，方差减半为 1/32。n=64 时，全量 MSE=1/(16×64)=1/1024≈0.0009766；掩码 MSE=1/(32×64)=1/2048≈0.0004883。这里截断确实降方差且不改期望。

**情形 B：非理想停点，χ_1>0。** 令 R=a，即动作决定成功。

```
a=0：R=0，Z=-0.5，G=0.25
a=1：R=1，Z=0.5，G=0.25
E[G]=0.25，Var(G)=0
χ_1 = E[R·Z | F_1] = p(1-p)=0.25 > 0

掩码后：成功保留 0.25，失败置 0
E[G_w]=0.125
偏差 d=-0.125
Var(G_w)=1/64=0.015625
```

n=64 时，掩码 MSE=0.015625+0.015625/64=0.0158691；全量 MSE=0。这里截断虽然删掉了失败项，但期望偏移远大于方差收益，MSE 反而变差。这个例子说明：**“失败轨迹”不是停止惩罚的充分条件；停点必须满足剩余信用为零，或至少让偏差–方差条件成立。**

## 4. 工程视角 (Engineering View)

| 设计选择 | 工程收益 | 代价/风险 |
|----------|----------|-----------|
| 只在失败轨迹上跑 VLM | VLM 不在控制闭环内；成功轨迹直接跳过 | 失败 rollout 多时，离线推理成本随失败数和 chunk 数增长；论文未给延迟数字 |
| 边界在完整 rollout 后确定 | 可利用后续帧判断“是否恢复”，避免过早误判 | 回顾性选择可能引入 η_B≠0；不能把它当成在线可用的因果停点 |
| 保留原 N_i 归一化 | 不改变 GRPO 的 token 权重口径，便于插入现有管线 | 若实现中重归一化，会改变每个保留项的权重，理论不再直接对应 |
| 掩码 detached | 每个 PPO update 内固定，不反传到 VLM | VLM 错误无法通过策略梯度纠正；需要独立的事件模型评估 |
| 语义回退保留全反馈 | 避免 VLM 不确定时误删关键错误信号 | 回退率高时退化为 Full GRPO；论文报告各任务 fallback 0.2%–8.3% |

**训练与评测节拍**：每次 policy update 收集 64 个 accepted groups、每组 8 条 rollout，即 512 条轨迹；混合结果过滤要求组内成功比例在 [0.1,0.9]，不足时 adaptive refill。每个 update 做 1 次 PPO pass、global minibatch 128，因此是 4 个 optimizer steps；总共 40 个 policy updates。评测每 5 个 update 一次（0,5,...,40），每次 128 个 seen-seed episode + 128 个 held-out-seed episode，greedy 解码。这里的“40”是 update 数，不是 epoch；“25 control steps”是 action chunk 的执行步数，不是记忆窗口；“每 5 个 update 评测一次”是评测频率，不是训练轮次。

**部署含义**：部署只用 actor，不需要 VLM。VLM 和掩码只影响训练时的失败反馈。因此额外成本主要在训练管线：事件标注、冻结 VLM 推理、掩码生成。论文没有报告 VLM 推理延迟、吞吐或显存占用；这部分只能做条件性工程判断，不能当作实测。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与微调数据

- **OpenVLA-OFT actor**：论文将其描述为 action-token 实现，来自 SimpleVLA-RL 的管线；RL 从同一任务特定的 SFT checkpoint 开始。预训练细节不是本文贡献，正文没有展开。actor LoRA 在策略优化中关闭。
- **Qwen3.8-27B event VLM**：backbone 冻结，只训练一个 LoRA adapter；训练目标是 response-token 平均负对数似然。语料为 1,151 条 rollout 视频：1,024 条环境失败轨迹 + 127 条成功参考；训练/验证/测试划分为 758/133/260。划分按 task–scene identity 去重，1,045 个唯一 task–scene 身份跨划分零重叠，并排除策略评测用的 held-out scene seeds。
- **标注内容**：可见结果、抓取是否建立、handover 是否完成、错误类型、错误定位区间、是否观察到恢复、最终持有者、证据帧、摘要。错误区间表达“最后可见错误且未观察到恢复”的起始不确定性；允许空区间。
- **开放范围**：提供的全文没有声明代码、模型权重或标注数据是否公开；candidate 的 `repo_url` 为空，不能据此推断存在代码仓库。本文不杜撰仓库或额外 URL。

### 5.2 仿真评测设置

论文在 RoboTwin 2.0 的七个任务上评测：Hammer、Handover、Lift pot、Dual bottles、Container on Plate、Empty cup、Stack bowls。每个任务有不同 episode budget（最大 action steps 100–400，对应 4–16 chunks），每个 chunk 25 个 control steps。策略输入头相机和本体状态；VLM 看三路相机。所有策略评测都在仿真中完成。

**VLM 事件定位实测（Table 8）**：220 条可定位、未恢复的测试失败，task-macro IoU 从 base 的 28.5% 升到 epoch 3 的 50.9%、epoch 6 的 67.8%、epoch 8 的 72.1%。验证集选出的 update 304（epoch 8）达到 68.1% exact endpoint agreement、91.8% within-one-chunk agreement。注意这是与参考标注的协议一致性，不是策略成功率。

**策略性能实测（Table 1）**：update 40 时，七任务平均成功率从 Full GRPO 的 65.01% 升到 MART-GRPO 的 68.64%，正文报告 +3.63 个百分点；seen seeds 从 67.75% 到 68.97%，正文称 +1.23；held-out seeds 从 62.28% 到 68.30%，正文称 +6.03（表 1 舍入后显示为 +1.22 和 +6.02，正文说明均值与差值在舍入前计算）。六个任务提升，Handover +8.98、Hammer +7.03；Container on Plate 的 pooled endpoint 为 -1.95，差异集中在 held-out seeds，两种方法峰值都为 94.92%。

**反馈消融实测（Table 2）**：Hammer 和 Handover 两任务。update 40 时，Uniform scaling 为 77.73% / 40.62%，Random prefix 为 71.88% / 41.41%，MART-GRPO 为 81.25% / 43.75%。MART 相对 Uniform 高 3.52 / 3.12 个百分点，相对 Random prefix 高 9.38 / 2.34 个百分点。两个消融都冻结统计量、保留成功权重和原 token 分母、不使用在线 VLM，并沿用 40 update 预算与同一评测 schedule。

### 5.3 真实机器人设置

提供的全文没有报告真实机器人实验。所有策略结果来自 RoboTwin 2.0 仿真；不能把七任务仿真结论外推到真实机械臂、移动机器人或人形平台。论文的“OpenVLA-OFT”是策略骨干，不等于真实部署验证。

> **证据分级**：Table 1/2/8 的数字是论文实测；Theorem 2、Proposition 1 是固定策略、PPO clipping 之前的理想停点结论；Proposition 4 处理回顾性边界的投影残差（含 η_B≠0），Proposition 7 给出一般截断的偏差–方差权衡；本文关于“若 η_B≠0 则期望可能偏移”的讨论属于条件性分析，不是论文新实验。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**

- 在混合结果组中，对失败轨迹保留从 chunk 0 到所选错误区间右端点 B_i（包含错误起始区间）的负反馈，屏蔽其后的后缀；成功轨迹完全保留。
- 不改变 actor 架构、不引入 critic、不改变组 advantage 公式；可作为 GRPO 失败反馈模块插入现有 VLA RL 精调。
- 在两任务消融中，事件引导截断优于“统一缩小失败 advantage”和“随机前缀”，说明收益不只来自减少负反馈总量。
- 在 Hammer、Handover 等需要恢复行为的任务上提升明显；held-out seeds 的平均提升大于 seen seeds，作者将其作为泛化信号，但仍是仿真内证据。

**不能做什么 / 失败模式**

- **不是所有失败轨迹都该截断**：若停点不满足零剩余信用，截断会改变期望；论文的 Theorem 2 是理想条件，VLM 边界是回顾性的，η_B 未在正文中给出实测估计。
- **VLM 定位错误会传播**：错误区间过短会过早停止惩罚，过长则退化为 Full GRPO；论文用 semantic fallback 降低风险，但 fallback 率高时收益会缩小。
- **策略演化后的分布偏移**：事件模型在固定标注语料上训练后冻结；论文在 limitation 中明确问“可见事件是否随策略演化仍跟踪剩余信用”，没有回答。
- **任务覆盖有限**：只有七任务仿真、单一策略家族 OpenVLA-OFT；没有真实机器人、没有多机器人形态、没有跨 embodiment 验证。
- **对比有限**：Temporal GRPO、PCM、μ-GRPO、ESPO、PS-PPO 只在 related work 中文字对比，未在同一 benchmark 复现；不能声称 MART 优于这些方法。
- **Container on Plate 端点下降**：MART pooled endpoint 比 Full GRPO 低 1.95 个百分点，说明高 SFT 起点、有限 headroom 的任务上收益不稳定。

### 6.1 隐含假设 (Hidden Assumptions)

1. “最后可见错误且未观察到恢复”是剩余条件信用为零或接近零的可靠代理；论文用失败认证（P_T=0）作为充分条件之一，但可见错误不是认证本身。
2. 完整视频中的后续帧能比在线前缀更准确地判断“是否恢复”；这要求 VLM 的事件标注与策略实际失败模式对齐。
3. 冻结 VLM 在策略更新 40 次后仍能定位新策略的失败；论文没有报告跨 checkpoint 的在线事件模型重训。
4. 保留成功轨迹全部损失项、保留原 N_i 归一化，足以让组估计量保持参考口径；若工程实现改变归一化，理论对应关系会变。
5. 混合结果过滤后的 accepted groups 能代表训练分布；adaptive refill 会改变实际生成的轨迹数，论文用 accepted groups 固定预算，但生成成本可能不同。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 反馈改变发生在哪里 | 是否需额外模型/阶段检测 | 与本文的关系 |
|------|--------|--------------------|--------------------------|--------------|
| Full GRPO | 轨迹级组相对 advantage | 所有 action chunk 共享 A_i | 否 | 基线；本文指出其误罚恢复动作 |
| Temporal GRPO | 阶段相对 advantage | 重建 stage-relative 信用 | 需要任务阶段检测与 rollout 对齐 | 文字对比；本文不重建新信用，只改负反馈到达的位置 |
| PCM | chunk 采样/梯度计算分配 | 用 phase-level 成功–失败 divergence 选 chunk | 需要阶段级统计 | 文字对比；本文不采样 chunk，而是掩码失败后缀 |
| μ-GRPO | 负 advantage 更新掩码 | 用 importance-ratio 触发 veto | 否 | LLM RL 的近亲；本文用可见事件而非重要性比率 |
| ESPO | 早停失败 rollout | 提前终止并给终端失败 | 需要停止规则 | 文字对比；本文在完整执行后做回顾性截断 |
| PS-PPO | 前缀采样与重要性加权 | 采样 update prefix，保持参考梯度期望 | 需要前缀采样机制 | 文字对比；本文保留原组 advantage 和归一化，不加截断校正权重 |
| MART-GRPO | 失败反馈的时间支撑 | VLM 定位错误区间，失败前缀保留、后缀屏蔽 | 需要冻结事件 VLM + 离线标注 | 本文方法；理论用鞅/条件投影，实验用七任务仿真 |

> 表中除 Full GRPO 和 MART-GRPO 外，其余方法的描述来自论文 related work 的文字对比；论文没有在同一实验复现它们，因此不写未给出的数值。

**面试 Tip**：被问到时先答“失败轨迹级 advantage 会误罚恢复动作”，再答“理想停点是剩余条件信用为零，截断是完整 GRPO 估计量的条件投影”，最后补“工程上是冻结 VLM 做回顾性事件定位，证据只在 RoboTwin 2.0 七任务仿真和两任务消融”。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA RL 精调、特别是 GRPO/PPO 管线的人：重点 §2.2–2.4 和 §3.2–3.3，理解“掩码改的是时间支撑，不是 advantage 公式”。
  - 做 credit assignment 理论的人：重点 Appendix B.4–B.9，尤其是条件投影、回顾性残差、偏差–方差分解。
  - 要做工程落地的人：重点 §4、§5 和 Appendix D/E，核对训练节拍、事件标注、fallback 和归一化细节。
- **建议章节路径**：先读 §1.2 + §2.2 建立“剩余信用”直觉 → 再看 §3.2–3.3 的 VLM 边界映射 → 然后读 §4.2/4.3 的数字与条件 → 最后按需查 Appendix B.8/B.9 的证明。
- **不值得精读的理由**：如果你不做机器人学习、不关心 RL 精调，或者只想要真实机器人部署结论，读摘要即可；这篇的核心证据是仿真七任务和两任务消融，不是真实平台验证。

---

[← Back to Theory](./README.md)

**关键引用**：[arXiv:2610.32431](https://arxiv.org/abs/2610.32431)