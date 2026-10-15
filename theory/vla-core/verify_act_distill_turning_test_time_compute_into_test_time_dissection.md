# 把测试时计算蒸馏进策略：Verify, Act, Distill (Turning Test-Time Compute into Test-Time Adaptation for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-15
>
> **论文**: Verify, Act, Distill: Turning Test-Time Compute into Test-Time Adaptation for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2610.14885
> **核心定位**: 现有 VLA 的测试时计算（TTC，多候选打分聚合）每次执行都要重算一遍，因为策略被冻结；本文把「验证器聚合出的好动作」直接当老师数据蒸馏回 VLA 的 LoRA，让一次 TTC 的收益变成永久能力。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 把 Agg-8 执行中产生的加权聚合动作当 teacher，用 10 条 episode/task 蒸馏进 LoRA，单候选成功率从 54.1% → 70.6%，保留约 74% 的 TTC 增益 |
| 适合精读 | 如果你在做 VLA 部署后自适应、推理效率优化、或 TTC/verifier-reranking 系统，重点看 §3.2–§3.3 与 §4.6 |
| 可以跳过 | 如果你只关心纯预训练架构创新或大规模真机 benchmark，这篇距离中等（全模拟、单架构、单 seed） |
| 落地可行性 | 中高（不需 RL、不需额外 rollout，只多 200 步 LoRA 更新；但依赖固定 verifier 且需至少一次成功 episode）|
| 主要风险 | verifier 把策略拉向示范动作风格，可能降低候选多样性；10k checkpoint 上「再放大」效应消失 |

💡 **X-Ray 开场**

VLA 生成式策略每次推理都能吐出多个动作候选，用一个学好的 verifier 给它们打分、加权平均后执行——这就是测试时计算（TTC），能显著提升成功率。但问题是：策略本身没被更新，所以下次遇到同一个任务，这笔「生成 + 打分」的算力还得重花一遍，好决策永远进不了参数。本文的想法很朴素：既然执行时已经算出了「哪个动作更好」这个监督信号，那就把它回收，用标准 flow-matching loss 把聚合后的好动作蒸馏回 VLA。对 VLA 研究者的意义是——TTC 不再只是「一次性选择器」，而可以变成一个自产自用的学习信号源，TTC 的算力同时在「改进未来的推理」。

📍 **研究全景时间线**

```
[2023] Diffusion Policy / Flow Matching 生成式动作策略
   → [2024] Snell 测试时计算缩放（LLM 领域外溢）
   → [2025] V-GPS 用 value function 重排候选；RoboMonkey 用 learned verifier 打分
   → [2026] 大量 reranking/guidance/tree-search（RoVer, TACO, CoVer, MG-Select…）
   → [2026] PA-RL / PLD / Q-Planning：把改进动作蒸馏回策略，但需 critic / residual RL / evaluator 更新
   → [本文 2610.14885] ← 当前位置
      把「固定 verifier 下的候选聚合结果」直接当 teacher，
      无 RL、无专用 rollout，实现 TTC→TTA（测试时自适应）
   局限：全模拟、单一 VLA 架构、单 seed、per-task LoRA 不跨任务迁移
```

## 1. 核心架构/方法总览 (Overview / Architecture)

框架叫 **Verify, Act, Distill**，三个阶段循环：

1. **Verify**：固定 verifier 给 N 个候选打分。
2. **Act**：分数标准化 → softmax 权重 → 加权平均成一个 plan，执行前 H₂ 步。
3. **Distill**：把「成功 episode」里的 (observation, 聚合后 chunk) 存进 buffer，用标准 flow-matching loss 更新 LoRA。

### 1.1 系统组件对比概览 (System Component Comparison)

| 模组 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| VLA 策略 π_θ0 (SmolVLA) | 噪声 z, 观测 o_t（2 相机 + 8 维本体 + 语言） | 归一化动作 chunk A ∈ ℝ^(H₁×d)，H₁=50, d=32 | 每个 replanning point b 生成 | 预训练冻结；只更新 action expert 的 LoRA φ |
| Verifier V_act | o_b 的 frozen VLM connector 特征 + 候选前 H₂=10 步动作 | 标量分数 s_i | 每个 replanning point（N=8 次调用） | 预先训练 60k 步，适配期冻结 |
| 候选聚合 Best-of-N | N 个候选 + 分数 | 加权平均 A̅_b | 每个 replanning point | 纯推理，无参数 |
| LoRA 蒸馏 | buffer 𝒟（成功 episode 的 (o, A̅)） | LoRA 参数 φ 更新 | 每 task 收集 K=10 episode 后更新一次（200 步） | 只在适配期训练，评估期不动 |
| 执行接口 g | 归一化动作 | 环境动作（去归一化 + 座标变换） | 每次执行前 H₂=10 步 | — |

### 1.2 关键机制 (Key Mechanism)

- **监督信号来自哪里是关键差异**：PA-RL 用 critic 选/优化动作，PLD 用 residual RL specialist 蒸馏 recovery 动作，Q-Planning 更新 evaluator 但策略冻结。本文**直接用固定 verifier 下的聚合结果当 teacher**，把「生成 teacher 数据的过程」等同于「普通的候选评估与执行」——零额外成本。
- **只更新 LoRA**：θ₀（VLA 主体）与 verifier 全程冻结，只更新 action expert 的 LoRA（rank 8、scaling 16、0.37M 可训练参数），回归到原始输入输出维度与目标。
- **老师是完整 H₁ 步 chunk，不是只执行了的 H₂ 步**：权重 w_i 在各时间步共享，所以聚合出的是一个「连贯 plan」，而非各候选拼凑的补丁；且与 SmolVLA 预训练的输入/输出维度一致。
- **成功过滤**：只有 episode 级成功标签 y_e=1 的资料进 buffer；若 10 条全失败则跳过更新，保留 base policy。

⚡ **Eureka Moment**：TTC 执行时已经免费产出了一批「高质量动作监督」，把它当 teacher 蒸馏回提出候选的策略本身——验证不只选择执行动作，还可以成为学习信号，让 TTC 的收益被「保留」甚至「再放大」。

### 1.3 信息流/架构图 (Flow / Diagram)

```
观测 o_t ─┬─► VLA π_θ0 ──► N 个候选 A^(1..N)（N=8）
          │                    │
          └─► frozen VLM conn. 特征 ──► Verifier V_act ──► 分数 s_i
                               │
                     softmax 加权 (λ=0.3)
                               │
                     A̅_b = Σ w_i A^(i)  ← 聚合 plan
                               │
                  执行前 H₂=10 步 g(A̅_0:H2)
                               │
                    ┌──────────┴───────────┐
               episode 成功? y_e=1         y_e=0 → 丢弃
                    │
        buffer 𝒟 = {(o_b, A̅_b)}   (K=10 episodes)
                    │
        flow-matching loss 更新 LoRA φ (200 步)
                    │
        ┌───────────┴────────────┐
   单候选推理（免 verifier）   再套 Agg-8（再放大）
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
TTA 增益 = 把 Agg-N 执行时算出的 A̅ 当 teacher，用 flow-matching loss 写回 LoRA
```

**目标**：让 verifier-free 的单候选生成，尽可能逼近（并保留）多候选聚合的收益。

**候选加权聚合**（论文 Eq.1）：

```
w_i = exp(u_i) / Σ_j exp(u_j)
u_i = (s_i - max_j s_j) / [λ · (σ_s + δ)]
A̅_b = Σ_i w_i · A^(i)
```

变数说明：

| 符号 | 含义 |
|------|------|
| s_i | verifier 对第 i 个候选的分数 |
| σ_s | 该组内分数标准差 |
| λ | softmax 温度，默认 0.3 |
| δ | 数值稳定项 1e-8 |
| A̅_b | 加权聚合后的执行 plan（归一化空间）|
| N | 候选数，默认 8 |

**蒸馏缓冲区**（Eq.2）：

```
𝒟 = ∪_{e: y_e=1} { (o_b, A̅_b) : b 是 episode e 的 replanning point }
```

**Flow-matching 蒸馏损失**（Eq.3）：

```
A_τ = (1 - τ)·A̅ + τ·ε,   τ ~ U(0,1),  ε ~ N(0, I)
L(φ) = E_{(o,A̅)~𝒟, τ, ε} [ ‖ v_{θ0,φ}(A_τ, τ | o) - (ε - A̅) ‖_F² ]
```

> 符号与原文保持一致：损失在 data-to-noise 方向定义，推理时反向积分；v 是预测速度场的网络，F 为 Frobenius 范数。

**直觉**：你已经在执行时算出了一个「比任何单一候选都更靠谱」的 A̅。那就让策略学会「直接生成 A̅ 这种质量的动作」——用标准 flow-matching 目标把 A̅ 拉成去噪目标即可。不需要 reward、不需要 gradient through critic，因为 teacher 本身就是连续动作，监督是稠密的（每个 replanning point 一个样本）。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设某个 replanning point，VLA 吐出 3 个单步候选（简化 N=3，实际 N=8），verifier 给出分数：

| 候选 | 分数 s_i | 动作 A^(i)（1D 简化）|
|------|----------|----------------------|
| A1 | 0.80 | 1.0 |
| A2 | 0.50 | 0.4 |
| A3 | 0.30 | -0.2 |

步骤 1：算 u_i。max s = 0.80，取 σ_s 为组内标准差（此例约 0.205），λ=0.3，δ≈0：

```
分母 = 0.3 × (0.205) ≈ 0.0615
u_1 = (0.80 - 0.80)/0.0615 = 0
u_2 = (0.50 - 0.80)/0.0615 ≈ -4.88
u_3 = (0.30 - 0.80)/0.0615 ≈ -8.13
```

步骤 2：softmax 权重：

```
exp(0)=1, exp(-4.88)≈0.0076, exp(-8.13)≈0.00029
Σ ≈ 1.0079
w ≈ [0.992, 0.0075, 0.0003]
```

步骤 3：加权平均：

```
A̅ ≈ 0.992·1.0 + 0.0075·0.4 + 0.0003·(-0.2) ≈ 0.995
```

观察：由于 λ=0.3 很小（低温），softmax 几乎退化成 argmax，聚合动作 ≈ 最高分候选。这解释了论文附录对比 **Agg-8 vs Sel-8（argmax）** 的必要性——低温下两者接近，但聚合仍保留了「多候选共识」的平滑性。

步骤 4：若该 episode 成功，把 (o_b, A̅≈0.995) 存进 buffer，训练时对 A_τ = (1-τ)·0.995 + τ·ε 回归速度场目标 (ε - 0.995)。K=10 条 episode 累积后做 200 步 Adam 更新。

## 4. 工程视角 (Engineering View)

| 面向 | 事实（论文陈述） | 工程含义 |
|------|------------------|----------|
| 额外推理成本 | 蒸馏后单候选推理**不需要** verifier 及其 observation encoder | 部署时可省掉 verifier 的算力与内存 |
| 额外训练成本 | 每 task 收集 K=10 episode + 200 步 LoRA | 0.37M 可训练参数，极轻量 |
| TTC 调用开销 | N=8 时每 replanning point 调用 verifier 8 次 | 保留 Agg 则维持原成本，换取更高上限 |
| 执行时序 | H₁=50 生成、H₂=10 执行（半分开区间） | replanning 频率由 H₂ 决定；teacher 含完整 50 步 |
| 延迟声明 | 论文明确说「real hardware latency 尚待检验」，成本论证是**数候选数与评估器调用次数**，非实测 | 真机部署前需自行量测 |
| 内存 | LoRA rank 8 / scaling 16 / batch 8 | 边缘端适配友好 |
| 硬件 | 实验跑在 RTX PRO 6000 (96GB) + RTX 6000 Ada (48GB) | 训练侧需求，非部署侧 |

**Trade-off**：蒸馏把 TTC 的收益「凝固」进参数，代价是可能把策略拉向 verifier 训练所用的示范动作分布（style bias），并可能压缩候选多样性——后文 §4.3 的 10k 现象与此一致。

## 5. 数据与评测 (Data & Eval)

- **基准**：LIBERO 四套件（Spatial / Object / Goal / Long，各 10 任务 = 40 任务）。
- **策略**：SmolVLA，主实验用 **5k checkpoint** 作为「尚未充分适配」的代理，另测 10k 与 20k。
- **训练数据**：SmolVLA 在 10,000 条由 π_0.5 收集的 rollout 中 5,225 条成功 episode 上训练。
- **Verifier 训练数据**：official LIBERO 示范（四套件）每状态生成 8 个候选 + 新收集的成功 rollout；用 all-pairs Bradley–Terry loss 学「与记录动作 RMSE 更小者更优」。训练 60k 步、batch 16、lr 1e-4。
- **适配/评估协议**：每 task 用 reset seeds 0–9 适配，held-out seeds 10–59（每 task 50 episode = 每套件 500）评估；Base 与 Agg-8 用同一 seed 集。
- **报表指标**：套件均值、四套件宏平均、per-task 配对差（mean ± SEM）、改善/持平/退化任务计数。
- **分布偏移**：LIBERO-plus，每套件 105 任务（7 类扰动各 15），适配 seed 0–9、评估 seed 10–19；因算力，蒸馏策略只评每套件前 35 任务（140 任务）。

**主要结果（Table 1，LIBERO clean，SmolVLA 5k）**：

| 方法 | 单候选成功率 | +Agg-8 成功率 |
|------|--------------|----------------|
| Base | 54.1% | 76.4% |
| Self（自模仿成功轨迹，无 verifier）| 63.7% | 80.8% |
| **Distill（本文）** | **70.6%** | **82.3%** |
| N-GRPO（noise-space，用 verifier 排序当 reward）| 58.1% | 79.2% |

- **保留率**：(70.6 − 54.1) / (76.4 − 54.1) ≈ **74%**。
- per-task 配对差 vs Base：**+16.5 ± 2.7** 点，34 改善 / 1 持平 / 5 退化。
- Distill + Agg-8 vs 原 Agg-8：**+5.8 ± 2.3** 点；Long 套件 44.0% → **57.8%**（24 改善 / 6 持平 / 10 退化）。
- **10k checkpoint**：单候选 83.2% → 86.2%（+3.0 ± 1.4）；但 Distill+Agg-8 89.8% 未超过原 Agg-8 90.4%，**再放大效应消失**。verifier 验证配对准确率从 5k 的 0.82 降到 10k 的 0.73。
- **LIBERO-plus**（10k，共同 35 任务子集）：Base 20.7% → Distill 单候选 24.5%（+3.9 ± 1.4，43/74/23），Distill+Agg-8 25.4%（vs 原 Agg-8 仅 +0.6 ± 1.2）。且 140 任务中 **66 个在 10 episode 内完全无成功**，无法更新。
- **消融**：去掉成功过滤 → 64.4% / 78.8%（明显更差，过滤贡献 +6.2 ± 1.6 / +3.5 ± 1.7）；用最高分候选代替聚合 chunk 当 teacher → 67.8% / 82.5%（单候选反而 −2.8 ± 2.0）。
- **适应预算**：Long 套件 5k，仅 K=2 就到 29.6%，K=50 到 34.6%；K=10 下 Distill 27.0% > N-GRPO 25.6% > DSRL-style SAC 19.4%。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**：
- 把 TTC 的一次性增益「固化」进免 verifier 的单候选推理（省算力）。
- 在计算允许时，蒸馏后的策略再套 Agg-8 可进一步提升上限（尤其 Long 长程任务）。
- 样本效率高：几条 rollout 就能适配（K=2 已见效）。

**不能做 / 失败模式**：
- **依赖 verifier 质量**：verifier 与示范动作距离绑定，可能把策略拉向示范风格、压缩候选多样性（10k 时再放大失效疑与此相关，但作者说无法仅凭 verifier 准确率下降断定）。
- **依赖至少一次成功**：LIBERO-plus 140 任务中 66 个完全无成功 → 无法更新；真机上成功信号来自检测器，其误差会直接污染 buffer。
- **无条件更新、无回退**：没有验证步或 fallback，40 任务中单候选有 5 个退化、再聚合时有 10 个退化。
- **仅模拟、单架构、单 seed**：per-task LoRA 不跨任务迁移；成本论证用候选/评估器调用次数而非实测延迟。

### 6.1 隐含假设 (Hidden Assumptions)

- **成就假定**：假设 verifier 对「低质策略的候选」仍具足够判别力（5k 准确率 0.82），但这是一个为每类 checkpoint 单独训练的 verifier——「no additional data」只对适配阶段成立，verifier 训练本身用了新收集的成功 rollout。
- **成功标签可靠**：训练与评估都用 episode 级二元成功；真机场景下这个前提未必成立。
- **示范动作即最优**：verifier 以「与记录动作的 RMSE」为排序依据，隐含「接近示范即更好」——在示范本身非最优或多模态的任务上可能引入偏差。
- **候选多样性充足**：整个方法靠生成多样候选来产生 TTC 收益，但未量化蒸馏后多样性的变化。
- **per-task 适用**：假定每个任务可独立收集 K 条 episode 并独立更新，未验证跨任务/连续部署场景。

## 7. 与相关工作对比 (Comparison)

| 方法 | 监督来源 | 是否需 RL | 更新对象 | 适用场景 |
|------|----------|-----------|----------|----------|
| V-GPS / RoboMonkey | 训练好的 verifier / value function | 否 | 不更新策略（仅执行时选择）| 一次性 TTC 提升 |
| PA-RL | Q-function 选择 + 局部优化 | 是（critic）| 策略 | 需 critic 的场景 |
| PLD | residual RL specialist 的 recovery 动作 | 是 | VLA | 有专用 RL specialist |
| Q-Planning | 经验更新 evaluator | 部分 | evaluator（策略冻结）| 持续改进 evaluator |
| N-GRPO（本文对照）| verifier 排序作 reward | 是 | noise 扰动分布 | noise-space RL |
| **Verify, Act, Distill** | **固定 verifier 下的聚合动作** | **否** | **LoRA（action expert）** | **部署后轻量自适应** |

**面试 Tip**：被问到「这篇跟 RoboMonkey 差在哪」——一句话答：**RoboMonkey 用 verifier 选动作执行，本文用 verifier 的聚合结果当 teacher 反过来训练产生候选的策略**，把推理时算力变成学习信号，实现 TTC→TTA，且不需 RL 或额外 rollout。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 VLA 部署后自适应 / test-time training 的研究者（§3.2–§3.3 是核心）。
  2. 要评估「能否把 verifier-reranking 收益固化到策略」的系统工程师（看 §4.2、§4.6 的预算与 N 依赖）。
  3. 研究 TTC 缩放在机器人策略上边界的人（§4.3 的 10k 现象值得深挖）。
- **建议章节路径**：先读 §1 引言 + Figure 1（30 分钟懂全貌）→ §3.2/§3.3（公式与机制）→ §4.2 主结果 + §4.6 预算/N 依赖 → §5 局限（别跳，理解前提条件）→ 有余力再看 §4.3/§4.5 的分布偏移与 10k 现象。
- **不值得精读的理由**：若你不做机器人学习、或已熟悉 flow-matching 蒸馏与 candidate reranking，读摘要 + Figure 1 + Table 1 即可；本文的理论新意在于「监督来源的重新定位」而非全新数学框架。

---
[← Back to Theory](./README.md)

**关键引用链接**：
- 论文：https://arxiv.org/abs/2610.14885
- 引用之 verifier 方法 RoboMonkey (Kwok et al., 2025)：https://arxiv.org/abs/2502.02538
- 蒸馏对照 PA-RL (Mark et al., 2024)：https://arxiv.org/abs/2410.06158
- 基准 LIBERO (Liu et al., 2023)：https://arxiv.org/abs/2306.03310
