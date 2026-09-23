# 异步回放锚定的在线 VLA 后训练 (Stable and Efficient Real-World Online VLA Post-Training via Asynchronous Replay-Anchored Policy Improvement)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-23
>
> **论文**: Stable and Efficient Real-World Online VLA Post-Training via Asynchronous Replay-Anchored Policy Improvement (RAPolicy)
> **链接**: [arXiv:2609.22888](https://arxiv.org/abs/2609.22888)
> **项目页**: [flyfaerss.github.io/RAPolicy](https://flyfaerss.github.io/RAPolicy)
> **核心定位**: 把"异步 rollout + 回放锚定 (replay anchoring)"合起来做 VLA 在线 RL 后训练——critic 不再用当前策略预测的下一动作去 bootstrap，actor 不再重采样噪声，从而在真实机器人上把 1–2 小时的在线训练预算用出 52%→88% 的联合任务成功率。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 异步执行 + 回放锚定的 critic/actor 更新，是真实世界在线 VLA 后训练"又快又稳"的关键；单任务均 86.3%、联合五任务 52%→88% |
| 適合精讀 | 如果你在做**在线 RL 精调 VLA**、人在回路数据采集、或 async actor-learner 机器人系统，重点看 §III-C / §III-D / §IV-D |
| 可以跳過 | 如果你只关心离线 SFT 数据配方或多模态表征预训练，这篇距离中等 |
| 落地可行性 | **中**（算法改动小、可直接套在 π0.5 one-step flow actor 上；但需 8×3090 级训练端 + 人在回路硬件） |
| 主要風險 | 全部实验在单一 Franka Research 3 + 单臂桌面任务上；每任务仅 20 次（联合 10 次）评测，统计置信度有限 |

💡 **X-Ray 开场**

这篇论文解决一个非常工程化的问题：真实机器人上的 VLA 在部署时部署得不够准（插头插不进、方块叠不稳），而现场产生的"成功/失败/人工接管"数据本可以用来改进策略。RAPolicy 的两个关键发现是——(1) 让采集和训练**并行**而非串行，机器人和训练机都不空等；(2) 把 critic 的 Bellman 目标锚在**已记录的动作**上，而不是让当前策略现编一个下一动作去估值。作者报告在插充电器、叠方块、擦白板等任务上，用 1–2 小时在线预算从弱初始策略（SFT 5%/15%/0%/10%）拉到 100%/95%/70%/80%。对 VLA 研究者的意义是：**在线后训练的瓶颈未必是算力，而是值估计目标选取与 actor 更新的"接地方式"**。

📍 **研究全景时间线**

```
2023  Diffusion Policy / ACT（动作分块起点）
  → 2024  π0 flow-matching VLA（一类生成式动作策略）
  → 2025  π0.5（开放世界泛化 VLA）+ AWAC/IQL 式离线→在线 RL 迁移
  → 2026  batch-online RL 系（ALOE / EXPO-FT：先采集再更新，串行）
  → 2026  RL Token / StructRL（同期在线 VLA 后训练探索）
  → 本文 RAPolicy：异步 actor-learner + replay anchoring ← 当前位置
局限：单臂桌面、单一本体、稀疏二值奖励、短时在线预算
```

## 1. 核心架构/方法总览 (Overview / Architecture)

RAPolicy 站在 π0.5 预训练 VLA 之上做在线微调。系统里同时存在两套参数：**learner 策略 θ**（在训练机上更新）和 **rollout 策略 θ̄**（在机器人上执行）。两者异步推进，每完成一组更新后做一次同步。

### 1.1 系统对比概览

| 模块 | 输入 | 输出 | 频率/时序 | 训练 vs 推理 |
|------|------|------|-----------|--------------|
| Rollout actor (θ̄) | 图像 + 机器人状态 + 语言指令 | 动作块 a_t（H 步 delta-pose），并回存噪声 z_t | 连续采集，与学习并行 | 单步 flow 求解 + 高斯探索噪声 |
| Learner actor (θ) | 状态 s_t、记录动作块 a_t^b、回放存下的 z_t | 噪声干净预测 F_θ(s_t,z_t) | 每 update group 5 次 actor 更新 | 只做一次 action-expert 前向 |
| Critic V / Q_i | 状态（V）；状态+动作块（Q_i） | chunk 级价值与双 Q | 每 update group 20 次 critic 更新，预热 2560 次 | 冻结 VLM + 共享状态骨干 + GRU 动作编码 |
| Online buffer | 所有完成的 rollout episode | 采样批次 | 持续写入 | — |
| Demo buffer | 干预转移 + rollout 中最快的 20 条成功轨迹 | 采样批次 | 持续被更快轨迹替换 | — |

> 采样时 online/demo 两个 buffer 按 **50:50** 混合。

### 1.2 关键機制

- **异步执行**：rollout 与优化并发，机器人不等训练；learner 也不等采集。每个 update group 内使用固定的回放采样集，期间完成的 episode 在下一组前并入。
- **回放锚定 (replay anchoring) 的双侧含义**：
  - critic 侧：Bellman 目标用 V'(s_{t+1})，**不预测下一动作**，因此不依赖当前策略在回放覆盖之外生成的动作值；
  - actor 侧：只对**已记录的动作块** a_t^b 做优势加权条件似然，不取 critic 对动作的梯度。
- **存下的 rollout 潜变量 z_t 复用**：actor 更新时从回放取回当初的 z_t 而非重采样——z_t 来自固定的标准高斯基分布，与策略无关，因此可跨 actor 更新复用，无需保存策略相关中间流状态。
- **单步 flow actor**：直接优化"存下的 z_t → 记录动作 a_t^b"这一条一步映射，不引入独立 one-step 学生网络，也不做多步蒸馏。

⚡ **Eureka Moment**：**把"策略改进"的输入锚死在回放里已有的数据上——critic 不编下一动作、actor 不重采噪声——异步在线 RL 立刻从"容易发散"变成"稳定且便宜"。** 一句话：不是加更多并发算力，而是减少对回放覆盖之外动作的依赖。

### 1.3 信息流/架构图

```
              ┌───────────────── 机器人端 (rollout policy θ̄) ─────────────────┐
              │  观测 s_t → [VLA action expert] → F_θ̄(s_t, z_t) + ε_t → a_t     │
              │                     ↑ z_t ~ N(0,I) 采样一次并回存              │
              └───────────────────────────┬───────────────────────────────────┘
                                          │  转移 (s_t, a_t^b, R_t, d_t, s_{t+1}, z_t)
                                          ▼
              ┌──────────── Online buffer (全部 episode) ───┬── Demo buffer (干预 + 最快 20 成功) ──┐
              │                       50:50 采样 → 分布 D                                             │
              └───────────────────────────┬───────────────────────────────────────────────────────┘
                                          ▼
              ┌──── 训练端 (learner θ) ─── Replay-Anchored 更新 ────┐
              │  Critic:  y_Q = R_t + (1-d_t)·γ·V'(s_{t+1})         │  20 次/组
              │           δ_t = min_i Q'_i(s_t,a_t^b) - V(s_t)      │
              │  Actor :  w_t = exp(clip(A_t/β, -c, c))             │  5 次/组
              │           L = -E[ w_t · log p_θ(a_t^b | s_t, z_t) ] │
              └───────────────────────────┬─────────────────────────┘
                                          │  每组更新后同步 θ → θ̄
                                          ▼
                                    回到机器人端
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
Actor:   θ ← argmax  E_D[ exp(clip((Q(s,a^b) - V(s))/β, -c, c)) · log p_θ(a^b | s, z) ]
Critic:  y_Q = R + (1-d)·γ·V'(s')      # 注意：目标里没有 "当前策略预测的下一动作"
```

**目标**：在持续采集的行为数据分布 D 上，做 (a) chunk 级值学习、(b) 优势加权的一步策略改进。

**公式（critic 侧）** — 论文 Eq. (1)–(3)：

```
Bellman 目标:   y_Q = R_t + (1 - d_t)·γ·V'(s_{t+1})
双 Q 损失:      L_Q = 0.5 · Σ_{i=1..2} E_D[ (Q_i(s_t, a_t^b) - y_Q)^2 ]
残差:           δ_t = min_i Q'_i(s_t, a_t^b) - V(s_t)
V 的 expectile 损失:  L_V = E_D[ w_τ(δ_t) · δ_t^2 ]
               其中 w_τ(δ) = τ (δ ≥ 0), 否则 1 - τ；τ = 0.7（上分位）
```

**公式（actor 侧）** — 论文 Eq. (4)–(7)：

```
残差:           e_t = a_t^b - F_θ(s_t, z_t)
对数似然:       log p_θ(a_t^b | s_t, z_t) = -0.5·e_t^T·Σ^{-1}·e_t - 0.5·log det(2πΣ)
优势:           A_t = min_i Q'_i(s_t, a_t^b) - V(s_t)
权重:           w_t = exp( clip(A_t/β, -c, c) )      β = 0.5, c = 3
actor 损失:      L_actor = -E_D[ w_t · log p_θ(a_t^b | s_t, z_t) ]
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| s_t = (o_t, x_t, ℓ) | 图像、机器人状态、语言指令 |
| a_t | rollout 输出的动作块（Horizon H 步归一化 delta-pose） |
| a_t^b | **记录行为块**：自主转移即 rollout 动作，干预转移即人工纠正动作 |
| z_t | rollout 时采样并回存的初始高斯潜变量 |
| R_t | 执行该动作块对应的累计折扣奖励（论文用 chunk 级） |
| d_t | 终止标志，用于去掉终端 bootstrap |
| γ | 相邻动作块之间的折扣因子 |
| Σ | 探索噪声协方差（ε_t ~ N(0,Σ)） |
| τ / β / c | expectile 分位、优势温度、log 权重截断 |

**直觉**：整个方法是 **IQL 式的"样本内"值学习 + AWAC 式的优势加权回归**，但把生成式策略的落点从"重采样的流匹配路径"换成了"回放里真实发生过的那条噪声→动作映射"。因此值学习不需要 VLA 前向去现编 target 动作（省掉一次昂贵的 VLA 前向），actor 更新也不需要展开多步流求解器（省掉 roll-out 链上的误差累积与显存）。

> 符号与本文/相关文档保持一致：论文记 learner 参数为 θ、rollout 参数为 θ̄；V'、Q'_i 为 EMA 目标网络；D 为 online/demo 两 buffer 50:50 混合后的采样分布。

## 3. 带数字走一遍：玩具例子 (Worked Example)

设动作块 H = 5，状态维度略去，只看一维动作标量，探索方差 Σ = σ² = 1（标量情形）。

假设某一转移：人工把机器人从偏位纠正回正，记录行为块最终值为 a_t^b = [0.20, 0.21, 0.19, 0.20, 0.18]；而当前 learner actor 在存下的 z_t 下预测 F_θ(s_t, z_t) = [0.10, 0.10, 0.10, 0.10, 0.10]。

1. **残差**：e_t = a_t^b - F_θ = 每维约 0.10（向量各处 ~0.09–0.11）。
2. **对数似然**（标量和，σ²=1）：每维 -0.5·e² - 0.5·log(2π)，即每维约 -0.005 - 0.919 = -0.924；五维合计 ≈ **-4.62**。
3. **优势**：设 critic 给出 min_i Q'_i(s_t, a_t^b) = 6.0，V(s_t) = 5.0 → A_t = **+1.0**。
4. **权重**：w_t = exp(clip(1.0 / 0.5, -3, 3)) = exp(clip(2.0, -3, 3)) = exp(2.0) ≈ **7.39**。
5. **该样本 actor 损失贡献**：-w_t · log p = -7.39 × (-4.62) ≈ **+34.1**。

若换成一条失败转移：a_t^b 仍是记录动作，但 min_i Q'_i = 3.0、V = 5.0 → A_t = -2.0 → w_t = exp(clip(-4.0, -3, 3)) = exp(-3) ≈ **0.0498**。同一条样本的似然项几乎不被加权（-0.0498 × (-4.62) ≈ 0.23）。

**可计算闭环**：高的 Q 相对 V 越高 → 权重指数放大 → 该"好动作"被更强地拉向记录行为；失败的记录动作仍留在 replay 里供 critic 学值（因为它有 R_t），但在 actor 侧不推动策略。优势温度 β=0.5、截断 c=3 恰好把权重限制在 [exp(-3), exp(3)] ≈ [0.05, 20.1]，避免单条轨迹主导更新。

## 4. 工程视角 (Engineering View)

| 维度 | RAPolicy 的做法 | 工程含义 |
|------|----------------|----------|
| 等待时间 | rollout 与学习并发，每 update group 内固定采样集 | 机器人与训练机同时忙碌；吞吐由较长一侧决定 |
| 训练端配置 | 本地控制机管机器人；远端 **8× RTX 3090**（1 GPU rollout + 7 GPU 训练） | 单机多卡受控环境，尚非边缘部署方案 |
| 每 critic 更新成本 | 用 V'(s_{t+1}) 直接读缓存的状态表征，**不调 VLA 前向** | 消掉一次 VLA forward → 训练显著更快（论文报告对手需 >3× 时间） |
| actor 更新成本 | 单步 flow + 一次 action-expert 前向 | 不展开多步流求解器，显存/步数都低 |
| 缓存刷新 | 若改用"预测动作" bootstrap，则需每次策略更新后刷新动作缓存 | 缓存失效是另一笔隐性开销来源 |
| 同步频率 | 每 20 critic + 5 actor 更新同步一次 θ→θ̄ | 同步越频繁，θ 与 θ̄ 差异越小，但通信开销上升 |
| 冷启动 | critic 预热 **2560** 次更新 | 保证 actor 拿到可靠优势信号前不误导 |
| 抖动/稳定性 | EMA 目标 + 权重截断 + 上分位 expectile | 三处"减方差"设计共同支撑"稳定提升"的 claim |
| 奖励设计 | 二值稀疏奖励（成功 10，否则 0） | critic 需从稀疏信号里学到进度敏感的价值（见 Fig. 7） |

**部署约束小结**：这套方法要求你有 (a) 一条低延迟的 rollout 通路、(b) 一个能跑 8 卡级的训练端、(c) 人在回路触发干预。它不是"在机器人上本地训练"的方案，而是"机器人现场 + 远程训练集群"的方案。

## 5. 数据与评测 (Data & Eval)

**任务集（Fig. 3）**：

| 场景 | 类型 | 语言指令 | 动作块 H |
|------|------|----------|----------|
| Plug Charger | 精密插接 | "plug the charger into the socket" | 5 |
| Pick Banana | 抓放 | "put the banana into the basket" | 10 |
| Stack Blocks | 精密堆叠 | "stack the yellow block on top of the green block" | 10 |
| Wipe Whiteboard | 接触丰富 + 视觉接地 | "pick up the whiteboard eraser and erase the red writing…" | 10 |
| Joint Multi-Task | 语言条件五任务共享策略 | 5 条（放柠檬/放黄块/放绿块/叠黄/叠绿） | 10 |

**硬件与数据**：Franka Research 3，每任务 2 路 RGB（腕部 + 第三人称）。单任务先 10 条示范 SFT 再做在线 RL，且**不用离线数据初始化 buffer**（专门衡量在线适应效率）；联合设定每任务 30 条示范（共 150）训练共享 SFT 策略，**并用离线数据初始化 buffer**。批次大小 112。

**评测指标（Eq. 8）**：

```
SR_i = N_i^succ / N_i        （无人工干预的成功 episode 比例）
IR_i = N_i^int  / N_i        （至少一次干预的 episode 比例）
```

指标按"在线训练流逝时间"和"累计机器人交互时间"两条轴汇报。最终策略单任务评 20 次、联合每任务评 10 次。

**主要结果**：

| 设定 | 初始 SFT | RAPolicy 最终 | 最强基线 |
|------|---------|---------------|----------|
| 单任务（Table I，20 trials）| 5% / 15% / 0% / 10% | **100% / 95% / 70% / 80%** | EXPO-FT 在 Plug Charger 61.9%、ALOE 14.3%；HIL-SERL 仅 Plug Charger |
| 单任务平均 | — | 相对 EXPO-FT **+72.5%** 相对提升 | — |
| 联合五任务（Table II，10 trials/任务）| 52% | **88%** | HG-DAgger 70% |
| 联合-堆叠两项合计 | 15% (3/20) | **75% (15/20)** | — |
| 联合-放柠檬 | 40% | **90%** | — |

> 注：论文自述 HIL-SERL 因视觉分布更广、horizon 更长，除 Plug Charger 外在有限在线预算内难以学出有效策略，故只报 Plug Charger。这是对基线"不利条件"的披露，读表时需注意。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**：
- 在弱初始策略（SFT 仅 10 条示范、成功率 0–15%）上快速拉起，Plug Charger 30 分钟内收敛并保持 100%。
- 联合五任务共享策略："保强项、补弱项"——已经可靠的任务不掉，弱的堆叠任务大幅补上。
- 稀疏二值奖励下仍学出**进度敏感**的状态价值：Fig. 7 中 V(s_t) 从 4.3 升到 9.6（黄块被抬起移向绿块）；被人工扰动把柠檬推走后 V 下降、随后抓取并移向篮子时回升。
- 更少的机器人交互与更少人工干预（rollout 成功率上升同时干预率下降）。

**不能 / 失败模式**：
- **任务覆盖窄**：全部实验在单臂 Franka Research 3 的桌面操作，未见移动本体、双臂、人形或长时程任务。
- **统计置信度有限**：单任务 20 次、联合每任务 10 次评测，单点百分比的方差不可忽略（尤其 Stack Blocks 60%、Wipe Whiteboard 70%）。
- **对奖励设计敏感**：只有二值稀疏奖励，论文未系统考察稠密/塑形奖励下的行为。
- **异步正确性依赖工程细节**：缓存刷新、同步频率、critic 预热 2560 步都是隐式超参；论文的 ablation 表明一旦退回"预测动作 bootstrap"，结果从 30 分钟收敛掉到 40 分钟 10%（120 分钟也仅约 40%）且耗时 >3×。
- **人在回路假设**：demo buffer 依赖干预转移与"最快 20 条成功轨迹"，缺少人工干预的场景下该收益机制打折。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设"记录动作块"一定优于当前策略生成的动作**：actor 只被拉向 a_t^b。若某条记录行为受人类次优纠正或噪声污染，优势加权虽能压制其权重，但权重公式仍基于同一 critic 的 Q 估计——critic 错了，权重就错。
- **假设存下的 z_t 在 actor 更新后仍是"语义一致"的条件变量**：论文论证 z_t 来自固定基分布、与策略无关；但 F_θ 变了，同一 z_t 对应的动作分布形状也随之变化，该论证对"似然可解释性"的成立程度未做显式量化。
- **假设 rollout 与 learner 的分布差异可被 50:50 buffer 混合 + 优势加权吸收**：未报告 off-policy 校正项（如重要性采样比率）的消融。
- **假设单动作块 Bellman（chunk-level TD）足以刻画任务进度**：H=5/10 的块内信用分配被整体吞掉。
- **单任务"不初始化 buffer"与联合"初始化 buffer"的差异**未被论文当作变量单独讨论，两者结果不可直接横向比较。

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 采集/更新 | 值目标 | actor 更新 | 适用场景 |
|------|--------|-----------|--------|-----------|----------|
| **RAPolicy（本文）** | 稳定 + 高效的在线 VLA 后训练 | **异步并发** | V'(s') 回放锚定，不预测下一动作 | 优势加权一步条件似然（复用 z_t） | 真实机器人 1–2h 在线预算 |
| ALOE | action-level off-policy 评估 + chunk 级 TD | batch-online（本文将其实现为异步以公平比较） | 参考其 off-policy 评估 | 优势加权 **flow-matching** | VLA 后训练 |
| EXPO-FT | 把 EXPO 扩展到 VLA 微调（含干预） | 先采集再更新 | — | 多步流策略 | 样本高效 RL 微调 |
| HIL-SERL | 从零训练紧凑策略 | 人在回路 off-policy RL | — | — | 精密双臂操作（预算充足） |
| FQL (Flow Q-Learning) | Q 最大化 + 多步 flow 蒸馏到独立一步 actor | 离线为主 | Q 目标 | 蒸馏 | 生成式策略 RL |
| IQL / AWAC | 样本内值学习 / 优势加权回归 | — | 本文的数学源头 | 本文的数学源头 | 通用离线/在线 RL |

**面试 Tip**：若被问"RAPolicy 和 AWAC/IQL 到底差在哪？"——一句话答：**数学骨架就是 IQL 的样本内值学习 + AWAC 的优势加权回归，新颖点在于把它接到"异步 actor-learner + 生成式一步 flow VLA"上，并用回放存下的 z_t 把 actor 的似然"锚"回当时的噪声—动作配对**，从而省掉 critic 侧一次 VLA 前向、actor 侧一次多步流展开。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 正在做**在线 RL 后训练 VLA**（尤其 π0/π0.5 系 flow 策略）的研究者——§III-C/§III-D 的公式可直接迁移；
  2. 搭**异步 actor-learner 机器人系统**的工程师——§III-A 的 update group / 同步 / 预热细节决定成败；
  3. 关心"稀疏奖励下 critic 如何学到进度信号"的人——Fig. 7 的价值可视化是少见的一手证据。

- **建議章節路徑**：先讀 §III-C（critic 怎么锚）→ 再看 §III-D（actor 怎么锚、z_t 为什么可复用）→ 然後 §IV-D 的四个 ablation（一步 vs 多步、回放锚定 vs 预测动作、存 z_t vs 重采噪声）→ §IV-B/§IV-C 的结果表 → 可跳 §II 相关工作的文献罗列（除非要追 IQL/AWAC/FQL 谱系）。

- **不值得精讀的理由**：若你不做机器人学习、或已熟悉 IQL/AWAC 式样本内 RL 与 flow 策略的一步化，本文的方法论增量是"组合与工程验证"而非全新算法，读摘要 + Table I/II + Fig. 6 即可抓住全部信息。

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文：[arXiv:2609.22888](https://arxiv.org/abs/2609.22888) · [HTML](https://arxiv.org/html/2609.22888v1)
- 项目页：[flyfaerss.github.io/RAPolicy](https://flyfaerss.github.io/RAPolicy)
- 基础模型：[π0.5 (arXiv:2504.16054)](https://arxiv.org/abs/2504.16054) · [π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- 相关方法：IQL · AWAC · FQL · ALOE (arXiv:2602.12691) · EXPO-FT (arXiv:2605.25477)
