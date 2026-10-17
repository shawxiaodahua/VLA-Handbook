> 历史补录：补回 2026-10-10 的深度阅读任务；论文原始收录日期 2026-10-10；实际补写日期 2026-10-17。本文依据当前读取的指定论文版本，不冒充当时已完成的报告。

# 噪声空间学习：冻结 flow-matching VLA 的轻量适配 (Noise Space Learning for VLA Adaptation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: Noise Space Learning for VLA Adaptation
> **链接**: https://arxiv.org/abs/2610.13943v1
> **核心定位**: 不改 flow-matching VLA 的生成器，只在噪声源头学习 context-conditioned 分布；用 PPO/SAC/AWR 做轻量任务适配，在 LIBERO、RoboTwin 和四个真实机器人任务上以低更新成本取得有竞争力的成功率，但真实安全仍依赖噪声约束与成功样本。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 冻结 VLA，只训练一个 32 维噪声 actor；LIBERO 两设置等权平均 SR：NSL-SAC 83.5%、NSL-PPO 81.6%、NSL-AWR 79.1%，对比 RLT 81.0%、Action-AWR 79.4%（论文表 1）。真实机器人自主 NSL-AWR 71.3%，训练 1.53 min/任务；加人类纠正 83.8%，2.04 min（论文表 3）。 |
| 适合精读 | 做 flow-matching VLA 后训练、RL 微调、真实机器人部署的人；重点看 §3.1–3.2、§4.3–4.7、Appendix A.3/A.7/B.5。 |
| 可以跳过 | 只关心 VLM 预训练，或不使用 flow matching 作为 action generator 的读者；读摘要和 §4.3 表 1 即可。 |
| 落地可行性 | 中高：更新成本极低，VLA 主干不反传；但真实安全需要噪声限幅、成功样本筛选，SAC 在真实机器人上有碰撞风险。 |
| 主要风险 | 固定 generator 的表达力上限；噪声空间安全没有形式保证；AWR 在 π0.5-One 上低于 RLT 和 Action-AWR。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？预训练 VLA 要适配具体任务，全参微调贵且容易破坏先验，LoRA/残差/只调 action head 又各有局限。它发现了什么？对 flow-matching VLA，行为的“源头”是初始噪声；只学一个 context-conditioned 噪声分布，就能在不改生成器的情况下重加权行为分布，而且信息论上这是达到目标行为分布的**最小信息成本**路径。对 VLA 研究者意味着什么？它把后训练从“改网络权重”变成“改噪声采样器”，给了一个低算力、可接 PPO/SAC/AWR 的适配接口；但“冻结生成器”也意味着能力上限，真实部署还要单独处理安全。

📍 **研究全景时间线**

```
2017 PPO ─ 2018 SAC ─ 2019 AWR ─ 2023 RT-2 ─ 2024 Octo / π0
   ─ 2025 π0.5 / DSRL / SmolVLA ─ 2026 NSL（本文）
```

年份取自论文参考文献，仅列与本文机制直接相关的少量节点，不构成完整文献综述。本文位置：把 DSRL/UniSteer 一类的 noise steering 统一成 VLA 适配接口，并用信息论说明“为什么改噪声源是省信息的”。局限：不改变 generator 通道，不能凭空创造通道外行为；真实安全仍需工程约束。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 → 输出 | 时序/频率 | 训练/推理差异 |
|------|-------------|-----------|----------------|
| 冻结 VLA 生成器 G_φ | context s（图像、proprioception、指令）+ 噪声 z → action chunk a（50 步） | 每个 chunk decision 调用一次；执行前 10 步后重规划 | 全程冻结，不反传；论文的 NSL 不更新它 |
| 噪声 actor π_θ^z | pooled VLA hidden features + proprioception → 32 维高斯均值与 log std | 每个 chunk decision 采样一次；噪声重复到 50 步 horizon | 训练；PPO/SAC/AWR 更新；推理用确定性均值（LIBERO 对比） |
| 价值/评论家 | context + z → V 或 Q | 每个梯度 update | PPO/AWR 用 16-bin categorical value MLP；SAC 用 twin critics |
| Replay | transition（s, z, R, s', done/trunc） | 每个 collection batch | SAC/AWR 跨 batch 保留；PPO 每个 collection–update cycle 后清空 |
| 人类纠正反转 | human action chunk → z*（fixed-point inversion） | 每轮人类数据 | 仅 human-guided NSL-AWR；用 UniSteer 的反转过程 |
| 评测 | 500 task–reset pairs（LIBERO）或 40 trials/任务（真实） | 每个 epoch 后一次（LIBERO）；真实每任务 40 trials | NSL 方法用 deterministic actor mean；base policy 用原生 Gaussian 采样 |

### 1.2 关键机制 (Key Mechanism)

- **只改源分布，不改通道**：标准 flow matching 把 ρ=N(0,I) 的噪声通过固定 G_φ 映射为动作；NSL 学 π_θ^z(·|s) 替代 ρ，保持 G_φ 不变。行为变化来自 pushforward 分布 (G_φ,s)_# π_θ^z 的改变。
- **信息成本最小**：论文把噪声当源、生成器当信道，用 KL 散度比较三种干预位置——改噪声源（NSL）、改生成器（generator adaptation）、改输出（residual correction）。Proposition 1 说明：在确定性生成器下，要达到目标动作分布 q，噪声分布的最小 KL 恰等于 D_KL(q || μ_φ^a)，且由密度比 dq/dμ_φ^a 在纤维上取常值实现。
- **轻量更新**：梯度不穿过 VLA；只更新噪声 actor 和小的 value/critic。论文表 1 的 NSL-AWR 更新时间为 1.09 s/batch，对比 Action-AWR 210.20 s/batch、RLT 111.60 s/batch。
- **三种优化器对应不同资源**：AWR 用成功样本加权回归，少交互、低更新预算友好；SAC 用 twin critic 和熵正则，数据与算力充足时最终 SR 最高；PPO 用 on-policy 比率裁剪，吃 rollout，但额外更新增益有限。
- **实现技巧不是装饰**：noise limit、task balancing、success/failure balancing 对 SAC 尤其关键；消融显示 SmolVLA-Full 上 SAC 去掉 noise limit 从 93.4% 掉到 32.0%，全部去掉掉到 0.0%（论文表 2，前 30 epochs 最佳 SR）。

⚡ **Eureka Moment**：VLA 适配不一定要改“生成器”或“动作”，可以只改“初始噪声的分布”；因为 flow generator 是固定的 pushforward 通道，源分布的重加权等价于行为分布的重加权，而噪声空间的 KL 恰好是这种行为改变的最小信息成本。

### 1.3 信息流/架构图 (Flow / Diagram)

```
context s ──► frozen VLA features ──► noise actor π_θ^z(·|s)
                                              │
                                              ▼
                                        z ~ π_θ^z(·|s)
                                              │
                                              ▼
                         frozen flow generator G_φ(s, z) ──► action chunk a (50 steps)
                                              │
                                              ▼
                               execute first 10 steps ──► env ──► reward R, success Y
                                              │
                                              ▼
                         replay (s, z, R, s', done/trunc) ──► PPO / SAC / AWR update
                                              ▲
human correction ──► fixed-point inversion ───┘  (仅 human-guided NSL-AWR)
```

关键边界：训练时梯度只到噪声 actor 和 value/critic，不穿过 frozen VLA；推理时噪声 actor 先采样 z，再调用一次 flow generator 生成 chunk。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
min_{π^z: (G_φ,s)_# π^z = q} D_KL(π^z || ρ) = D_KL(q || (G_φ,s)_# ρ)
```

先给目标：学一个噪声分布 π_θ^z(·|s)，使冻结生成器产生的动作分布更符合任务回报。再给关键公式：

```
J(θ) = E[ Σ_{n≥0} γ^n R_n ]

π_{θ,φ}^a(·|s) = (G_φ,s)_# π_θ^z(·|s)
μ_φ^a(·|s)     = (G_φ,s)_# ρ
```

信息成本与行为下界：

```
C_M = E_{s~d} D_KL(P_M(·|s) || P_0(·|s))
    ≥ B_M = E_{s~d} D_KL(q_M^a(·|s) || μ_φ^a(·|s))

C_NSL = E_{s~d} D_KL(π^z(·|s) || ρ)
```

Proposition 1（最小信息 lift）：

```
min_{π^z: (G_φ,s)_# π^z = q} D_KL(π^z || ρ) = D_KL(q || μ_φ^a)

dπ^{z,*}/dρ(z) = (dq/dμ_φ^a)(G_φ,s(z))
```

Proposition 2（效用约束下的比较）：

```
R_NSL(τ) = R_beh(τ) ≤ R_M(τ),  M ∈ {gen, corr}
```

三种优化器的目标：

```
PPO:
E[ min( r_n(θ) Â_n, clip(r_n(θ), 1-ε, 1+ε) Â_n ) ]
r_n(θ) = π_θ^z(z_n|s_n) / π_old^z(z_n|s_n)

SAC:
E_{s~D, z~π_θ} [ α log π_θ^z(z|s) - min_{i=1,2} Q_{χ_i}^z(s,z) ]

AWR:
max_θ E_{(s,z,Â,Y)~D} [ w log π_θ^z(z|s) ]
w = Y · min{ exp(Â/β), w_max }
```

| 符号 | 含义 |
|------|------|
| s | context：图像观测、proprioception、任务指令 |
| z | 初始噪声；LIBERO 实现为 32 维，重复到 50 步 prediction horizon |
| G_φ(s,z) | 冻结的 flow-matching VLA 生成的 action chunk |
| ρ | 标准高斯先验 N(0,I) |
| π_θ^z(·|s) | 学到的 context-conditioned 噪声分布 |
| (G_φ,s)_# π | 噪声分布经固定生成器 pushforward 得到的动作分布 |
| R_n | 第 n 个 chunk 内累计 reward；LIBERO 用 chunk 间折扣 γ=0.99 |
| C_M | 方法 M 的 intervention information cost |
| B_M | 行为分布改变的信息下界 |
| q_M^a | 方法 M 产生的 action marginal |
| Â | advantage 估计；AWR 用 β 温度和 w_max 截断 |
| Y | 轨迹成功指示；AWR 只用 Y=1 的样本做 actor 回归 |

直觉：噪声是“源”，生成器是“固定信道”。改变源分布可以改变输出分布；KL 链式分解说明，行为改变至少需要 B_M 的信息，而 NSL 的噪声 KL 恰好等于这个下界（确定性生成器、q 绝对连续等条件下）。因此 NSL 不是“省算力的近似”，而是在论文设定下达到信息下界的适配接口。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> **假设例子，仅用于验算论文公式，不是论文实验数据。** 取 1 维、恒等生成器 G(z)=z、ρ=N(0,1)、目标动作分布 q=N(μ,1)。

**验算 1：NSL 的噪声 KL。** 最优噪声分布 π*=N(μ,1)，因为 pushforward 后仍是 N(μ,1)。KL 为：

```
D_KL(N(μ,1) || N(0,1)) = 1/2 · (μ² + 1 - 1 - log 1) = μ²/2
```

μ=1 时成本 0.5 nats；μ=2 时成本 2 nats。与 Proposition 1 的 D_KL(q || μ_φ^a)=μ²/2 一致。

**验算 2：改生成器的确定性成本。** 若用确定性 G_ψ 把 0 映射到 μ，而 G_φ 把 0 映射到 0，两个点质量的 KL 为无穷（论文 Appendix A.3）。若用 σ 松弛为 N(G,σ²I)，条件 KL 为：

```
||G_ψ - G_φ||² / (2σ²)
```

取 μ=1、σ=0.1，成本为 1/(2·0.01)=50；这说明确定性核下“改通道”的成本不可直接与 NSL 的 0.5 比较，论文用 Gaussian relaxation 才得到有限值。这个例子也提示：NSL 的“省信息”结论依赖确定性生成器和绝对连续目标分布。

**验算 3：熵不控制均值漂移。** 取 π=N(m,1)，ρ=N(0,1)。熵为 0.5·log(2πe)，与 m 无关；但 KL 为 m²/2。m=1 时 KL=0.5，m=2 时 KL=2，熵相同。对应论文 Appendix A.7：熵没有均值惩罚，不能当作安全约束；PPO 的 moving reference 例子也说明局部 KL 小不等于全局 KL 小。

**验算 4：AWR 权重。** 取 β=0.5、w_max=20、Y=1。Â=0.5 时 w=exp(1)=2.718；Â=-1 时 w=exp(-2)=0.135；Â=5 时 exp(10)=22026，被 w_max 截到 20。若 Y=0，则 w=0，不参与 actor 回归。论文 Appendix A.6 提醒：过滤和截断会改变未截断的 Gibbs 最优，不能把“成功过滤 AWR”等同于标准 AWR。

## 4. 工程视角 (Engineering View)

| 项目 | NSL-AWR | NSL-PPO | NSL-SAC | Action-AWR | RLT |
|------|---------|---------|---------|------------|-----|
| 每 collection batch 梯度更新时间（s/batch） | 1.09 | 2.34 | 9.93 | 210.20 | 111.60 |
| 真实机器人累计训练时间（min/任务，四任务平均） | 1.53 | — | — | 52.4 | 23.2 |
| 主实验更新预算 | 1× | 2× | 4× | — | — |
| 真实机器人采用 | 是（自主与人类引导） | 否 | 早期 bead placement 实验出现 unintended table contact | 基线 | 基线 |

> 单位说明：s/batch 是“每个 collection batch 的梯度更新时间”，不是每个 epoch 的总训练时间；min 是真实机器人“累计模型更新时间”，按四个任务平均。更新预算 1×/2×/4× 是每个 collection batch 内梯度更新的倍数，不是 epoch 数。

工程含义：

- **吞吐/延迟**：推理时每个 chunk decision 只多一次小 MLP 前向（噪声 actor），VLA 前向次数不变；论文预测 50 步 chunk、执行前 10 步再重规划，最大 episode 520 environment steps。论文没有报告控制频率 Hz，也没有报告量化/抖动指标，不能编造。
- **内存**：replay 存 context、32 维 z、reward、done/trunc；SAC 需要 twin critic 和 target critic；AWR 需要 value MLP（16-bin categorical）和 actor；PPO 需要 on-policy buffer，但每个 cycle 后清空。显存占用主要仍由 frozen VLA 前向决定，论文未给具体显存数字。
- **训练稳定性**：SAC 对 noise limit 极敏感（SmolVLA-Full 消融 93.4→32.0，全部去掉→0.0）；AWR 对组件不敏感，但这也意味着它更依赖成功样本质量；PPO 对 task balancing 更敏感（去掉后 SmolVLA-Full 92.2→87.2，π0.5-One 58.2→50.4）。
- **部署约束**：真实机器人上，NSL-SAC 的噪声分布可能偏离标准高斯，论文报告早期 bead placement 出现非预期桌子接触；Appendix B.5 的诊断显示确定性噪声输出接近 5·tanh 的幅度上限（最终 minibatch 最大绝对值 4.96），actor proposal 的 Q 值比 replay 高 4.43，twin-critic 分歧 0.916 vs 0.151。作者据此选择 AWR 做真实实验。我的条件性分析：如果要在真实机器人上用 SAC，需要额外的动作可行性检查、torque/force 限幅或安全 shielding，不能只靠噪声限幅。
- **成本对比**：NSL-SAC 的 9.93 s/batch 约为 Action-AWR 的 1/21.2、RLT 的 1/11.2（论文 §4.3 文字）；NSL-AWR 的 1.09 s/batch 是表中最轻的更新。真实机器人上 NSL-AWR 1.53 min vs Action-AWR 52.4 min、RLT 23.2 min，这是论文实测，但要注意任务、数据量和实现不同，不能直接外推到其他机器人。

## 5. 数据与评测 (Data & Eval)

| 设置 | 模型/任务 | 数据与初始化 | 训练轮次/采样 | 评测频率/次数 |
|------|-----------|--------------|----------------|----------------|
| LIBERO-10 | SmolVLA、π0.5 | 每个 backbone 用 10,000-step SFT checkpoint；One=每任务 1 个演示（共 10 个），Full=379 个过滤演示 | 主曲线 60 epochs；每个 training epoch 覆盖每任务 50 个 benchmark reset states × 每 task–state pair 8 次 stochastic rollouts = 4,000 trajectories；混合任务 batch 最多 64 对 | 每个 epoch 后评测全部 500 个 task–reset pairs 一次；报告 10 任务平均 SR；消融看前 30 epochs |
| RoboTwin | SmolVLA SFT checkpoint | 8 个任务，Easy/Hard；每个任务独立 noise actor，不共享 RL 参数或 replay | 30 training epochs；R=8 rollouts；更新倍数 AWR 4×、PPO 2×、SAC 4× | epoch-30 SR；best within epochs 1–30；额外消融在 epoch 20 |
| 真实机器人 | 每任务单独 SFT 的 π0.5 | cup stacking 25 demos/30k SFT steps；toy fishing 35/30k；sequential insertion 35/50k；bead placement 20/140k | 在线 collection + update；预测 50 步 chunk，执行前 10 步；7 维 proprioception；主视角 224×224 | 每任务 40 trials；报告 SR 和四任务平均累计训练时间（min） |

数据与评测的关键条件：

- LIBERO 的 NSL run 训练一个共享的 task-conditioned noise actor，覆盖全部 10 个任务；RoboTwin 则是每任务独立 actor。这两种设置不能混为一谈。
- LIBERO 的评测用 500 个 task–reset pairs，每个 epoch 一次；主曲线 60 epochs，消融前 30 epochs。论文表 1 的 “best observed within 60 epochs” 是 60 个 epoch 内的最佳 SR，不是最后一个 epoch。
- 真实机器人每任务 40 trials；训练时间是“累计模型更新时间”，不是 wall-clock 总时间；人类引导的 NSL-AWR 用 UniSteer 的 fixed-point inversion 把人类动作 chunk 反转为噪声，再与机器人样本合并做 AWR。
- RoboTwin 的 50 个训练场景和 50 个 held-out 评测场景使用不相交的 reset seeds；这一点比 LIBERO 的 reset-state 评测更接近泛化测试。论文没有声称跨 embodiment 泛化。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么（论文实测/作者主张）：**

- 在 LIBERO 上，三种 NSL 都超过各自 base policy；NSL-SAC 在两设置等权平均上最高（83.5%），NSL-PPO 81.6%，NSL-AWR 79.1%。Full 设置峰值：SmolVLA 95.8%，π0.5 99.8%（论文 §4.2 文字；注意 π0.5-Full 未进表 1 的完整对比）。
- 在 One 设置上，SmolVLA 从 24.4% 升到 77.0%，π0.5 从 40.4% 升到 71.2%（论文 §4.2）。这说明少量演示下噪声重加权也能带来大幅提升。
- 真实机器人上，自主 NSL-AWR 平均 71.3%，高于 Action-AWR 58.8%，与 RLT 70.0% 接近；加人类纠正后 83.8%，高于 DAgger 66.3% 和 UniSteer 73.8%（论文表 3）。
- 更新成本低：NSL-AWR 1.09 s/batch；真实机器人 1.53 min/任务（自主）、2.04 min（人类引导）。这是论文的核心工程卖点。

**不能做什么/失败模式：**

- **固定 generator 的表达力上限**：作者在 Discussion 明确说 adaptation capacity 受 fixed generator 和 noise actor 表达力限制；NSL 只能重加权已有通道，不能创造通道外行为。
- **SAC 的真实安全风险**：早期真实 bead placement 的 SAC 实验出现 unintended table contact；Appendix B.5 诊断显示噪声接近幅度上限、actor proposal Q 高且 critic 分歧大。作者主张“限制噪声幅度不足以保证解码动作安全”，这是论文实测支持的失败模式。
- **AWR 不是所有设置都优**：π0.5-One 上 NSL-AWR 64.0%，低于 RLT 67.8% 和 Action-AWR 65.8%；真实 sequential insertion 上 RLT 75% 高于 NSL-AWR 70%。选型不能只看平均值。
- **复杂接触任务仍难**：真实 bead placement 的 NSL-AWR 只有 35%，加人类纠正到 55%；RoboTwin 的 place_a2b_left 在所有六次主实验中仍是最弱任务（Easy best 40%，Hard best 34%）。这说明噪声适配对精细接触/长时序任务的提升有限。
- **SAC 对组件极敏感**：SmolVLA-Full 上无 noise limit 从 93.4% 掉到 32.0%，全部去掉掉到 0.0%；π0.5-One 上全部去掉从 66.6% 掉到 39.8%。这些是前 30 epochs 的最佳 SR，说明 SAC 的高性能依赖实现细节，不是“开箱即用”。
- **AWR 的消融反而有时更好**：SmolVLA-Full 上 AWR 全部去掉组件为 90.2%（+3.0 pp），π0.5-One 为 60.2%（+1.2 pp）。作者解释 AWR 对组件不敏感；我的条件性分析是：这些消融只说明在该设置下组件不是瓶颈，不能推断组件一般无用。

### 6.1 隐含假设 (Hidden Assumptions)

1. **成功指示 Y 可靠**：AWR 只用成功样本做 actor 回归；如果真实任务的成功判定有噪声，权重会被污染。
2. **replay 中有成功样本**：Appendix B.1 明确说，如果 replay 没有成功样本，success-filtered actor update 会被跳过；AWR 在冷启动时可能没有学习信号。
3. **噪声空间可达目标行为**：Proposition 1 要求目标 q 绝对连续于 μ_φ^a；固定 generator 的 support 限制仍然存在。高斯先验有数学支撑，不等于高概率覆盖有用区域（Appendix A.4 的 coverage 讨论）。
4. **确定性生成器**：理论的最小信息 lift 依赖确定性 G_φ；随机解码需要扩展（Appendix A.1），构造性等式不一定成立。
5. **人类纠正可反转**：human-guided NSL-AWR 用 fixed-point inversion 从人类动作恢复噪声；这是近似反转，论文没有给出反转误差的定量界。
6. **噪声限幅能帮助安全**：论文自己给出反例——限制幅度不足以保证解码动作安全；安全需要额外的 generator、环境、控制器和安全集假设（Appendix A.7）。
7. **训练与评测 reset states 同分布**：LIBERO 用 benchmark reset states 收集和评测；这适合 benchmark 内适配，但不能直接推断新任务/新 embodiment 的泛化。作者在 Discussion 把新任务、新 embodiment、技能保留列为未来工作。
8. **共享噪声 actor 足够多任务**：LIBERO 用一个共享 task-conditioned actor；RoboTwin 用每任务独立 actor。共享 actor 的容量和多任务干扰没有被单独消融。

## 7. 与相关工作对比 (Comparison)

**仿真对比（LIBERO 两设置等权平均，论文表 1）：**

| 方法 | 改什么 | 是否冻结 VLA | 平均 SR（%） | 更新时间（s/batch） | 适用场景 |
|------|--------|--------------|--------------|----------------------|----------|
| Base policy | 不改 | — | 45.5 | — | 参考 |
| Action-AWR | 生成器适配：冻结 VLM，更新 action expert | VLM 冻结，action expert 更新 | 79.4 | 210.20 | 有较多更新算力，想改生成器 |
| RLT | 残差适配：在动作上加 correction | base 冻结，residual 学习 | 81.0 | 111.60 | 想保留 base，允许残差修正 |
| NSL-AWR | 噪声分布：成功过滤加权回归 | 是 | 79.1 | 1.09 | 少交互、低更新预算、真实机器人 |
| NSL-PPO | 噪声分布：on-policy 比率裁剪 | 是 | 81.6 | 2.34 | 有较多 rollout，更新预算中等 |
| NSL-SAC | 噪声分布：off-policy twin critic | 是 | 83.5 | 9.93 | 仿真大预算，追求最高最终 SR |

**真实机器人对比（四任务平均，论文表 3）：**

| 方法 | 数据来源 | 平均 SR（%） | 累计训练时间（min/任务） |
|------|----------|--------------|---------------------------|
| Initial policy | — | 37.5 | — |
| Action-AWR | 自主 | 58.8 | 52.4 |
| RLT | 自主 | 70.0 | 23.2 |
| NSL-AWR | 自主 | 71.3 | 1.53 |
| DAgger | 人类 | 66.3 | 61.6 |
| UniSteer | 模型+人类 | 73.8 | 49.8 |
| NSL-AWR + human | 模型+人类 | 83.8 | 2.04 |

注意：仿真表和真实表不能混比；真实表的训练时间是四个任务平均的累计模型更新时间，不是每个 trial 的时间。DAgger/UniSteer 是人类引导方法，NSL-AWR+human 也使用人类纠正，因此和自主 NSL-AWR 分开看。

**面试 Tip**：被问“NSL 为什么便宜”，答：梯度不穿过 VLA，只更新 32 维噪声 actor 和小 value/critic；被问“怎么选”，答：少交互/真实机器人选 AWR，仿真大预算选 SAC，PPO 吃 rollout 但额外更新增益有限；真实部署先限噪声，再用成功样本或人类纠正，不能把噪声限幅当安全保证。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 flow-matching VLA 后训练/RL 微调的研究者：重点看 §3.1–3.2 的接口和信息论，以及 §4.4–4.6 的资源消融。
  - 要在真实机器人上做在线适配的工程师：重点看 §4.7、Appendix B.2、B.5；尤其注意 SAC 的碰撞诊断和 AWR 的人类纠正流程。
  - 研究噪声空间/源分布适配的人：重点看 Appendix A.3 的确定性核奇异性、A.4 的 fixed-noise 讨论、A.7 的安全边界。
- **建议章节路径**：先读 §3.1–3.2（理解“只改噪声源”与最小信息成本）→ §4.2–4.3（主结果与更新成本）→ §4.6–4.7（组件消融与真实机器人）→ Appendix A.3/A.7/B.5（理论边界与安全诊断）→ Appendix C（RoboTwin 的独立 actor 设置）。
- **不值得精读的理由**：如果你不做机器人学习、或你的策略不是 flow matching（例如离散 token VLA），这篇的核心机制不直接适用；读摘要和 §4.3 表 1 即可。如果你已经熟悉 DSRL/UniSteer 一类 noise steering，重点看它如何统一接口、以及 PPO/SAC/AWR 的资源对比。

---

[← Back to Theory](./README.md)

关键引用：[Noise Space Learning for VLA Adaptation](https://arxiv.org/abs/2610.13943v1)