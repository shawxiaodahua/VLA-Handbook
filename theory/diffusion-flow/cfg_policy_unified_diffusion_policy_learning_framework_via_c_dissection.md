# CFG-Policy：用 Classifier-free Guidance 统一扩散策略的采样、探索与 RL 微调 (CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance)

> **补录说明**: 初稿生成于 2026-10-15，2026-10-20 重新核验后补录；这不表示论文今天首次发布。


> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-15
>
> **论文**: CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance
> **链接**: https://arxiv.org/abs/2610.14936
> **核心定位**: 把探索与利用拆成两个生成策略，在 score/velocity field 层面用 classifier-free guidance (CFG) 组合，作为即插即用采样接口，不改底层 RL 目标。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在扩散/flow 策略的每步去噪中组合两个生成场：冻结一支保留探索或预训练能力，训练另一支学利用或自适应行为；论文在 5 个 MuJoCo 任务和 4 个 RoboTwin 任务上报告 QSM/QVPO/Flow-SDE+PPO 的提升 |
| 适合精读 | 如果你在做 diffusion/flow policy 的 RL、VLA post-training、或需要探索-利用解耦，重点看 §3.1 组合机制、§3.2/§3.3 两种 regime、§4.2/§4.3 实验 |
| 可以跳过 | 如果你只关心 VLA 的视觉语言 backbone 结构，或只关心纯离线 RL，这篇距离中等 |
| 落地可行性 | 中：只改采样接口，但需要额外一条生成分支；post-training 的 w 加权方向在 Eq.9 与 Algorithm 2/4 间不一致，工程实现前必须统一 |
| 主要风险 | 固定噪声级的 score 组合是精确的，但端点乘积形式依赖 path-consistency 近似；Theorem 1 依赖 π_X 是 π_E 的 sharpened 版本的假设；Table 3 未报告方差/种子数 |

💡 **X-Ray 开场**  
扩散策略擅长表示多模态动作分布，但用 RL 微调后容易模式坍缩、多样性下降；直接微调预训练策略又可能破坏已获得的能力。这篇论文的解法很直接：不要只训练一个策略，而是保留两个生成策略，在采样时用 CFG 把它们的 score/velocity field 组合起来。对 VLA 研究者的含义是：CFG 不再只是条件生成的旋钮，而可以成为 RL 中“保留什么、学习什么”的接口。

📍 **研究全景时间线**

```
[2022] CFG (Ho and Salimans) / Diffusion-QL 离线扩散策略 RL
   ↓
[2023] IDQL、QSM、DIPO
   ↓
[2024] QVPO (Ding et al., 2024)
   ↓
[2025] CFGRL (Frans et al., 2025)：把 CFG 当作从离线数据提取策略的可控 improvement operator；
       π_RL (Chen et al., 2025a)、Flow Policy Optimization (Lyu et al., 2025)、
       RIPT-VLA (Tan et al., 2025)、Adaptive Reinforced Flow Matching (Zhang et al., 2025) 等
   ↓
[2026] ReinFlow (Zhang et al., 2026) 等；
       本文 CFG-Policy（arXiv:2610.14936，2026-10-10）← 当前位置
```

说明：以上节点只列论文参考文献中确认的工作。Flow-SDE 在论文 Table 1 与 §4.1 中被作为基线提及，但参考文献未对其单独标注年份，故不列入时间线；更细的年份或未在论文中确认的节点省略。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

论文把两个学习 regime 统一到同一机制：一个分支冻结，另一个分支可训练；组合发生在生成采样过程中。

| 模块 | Online RL（论文 §3.2） | RL Post-Training（论文 §3.3） |
|------|------------------------|-------------------------------|
| 冻结分支 | 探索策略 π_E，由 RND 预训练 | 预训练策略 π_X^0，来自 SFT checkpoint |
| 可训练分支 | 利用策略 π_X，从任务奖励学习 | steering 策略 π_E^θ，从 RL 目标学习 |
| 组合场 | v_w = (1-w)v_E + w v_X（论文 Eq.5） | 论文 Eq.9 写 v_w^θ = (1-w)v_E^θ + w v_X^0；Algorithm 2/4 与 B.4.2 写 v_w = (1-w)v_X^0 + w v_E^θ，方向不一致 |
| 组合发生位置 | 每个去噪步的 score/velocity field | 每个去噪步的 velocity field |
| 底层 RL 目标 | 保持 QSM / QVPO 原目标 | 保持 Flow-SDE + PPO 原目标 |
| 数据收集 | 组合策略 π_w 作为 behavior policy | 组合策略 π_w 生成 rollout |
| 推理 | 论文 Algorithm 1 说评估时 w=1，执行 π_X | 论文 Algorithm 2 返回 (π_X^0, π_E^θ, w)，推理用组合策略 |

### 1.2 关键机制 (Key Mechanism)

- **探索与利用分离到两个生成策略**：不是在一个策略内部调温度或熵，而是显式保留两个行为不同的生成模型。
- **在 score/velocity field 层面组合**：两个场作用于同一个中间样本 x_t，因此组合影响整条生成轨迹，而不是对两个已完成动作做插值。
- **冻结一支以保留能力**：online RL 冻结探索分支，post-training 冻结预训练分支；梯度只通过可训练分支。
- **不改底层 RL 目标**：组合替换的是 behavior sampler，actor/critic 的原始损失不变。
- **自适应 w**：论文 §3.4 用 TRPO 风格约束优化 guidance weight，目标是在保持与纯利用分支足够路径差异的前提下最大化 critic value surrogate。

⚡ **Eureka Moment**：把 CFG 的 guidance scale 从“条件生成强度”变成“探索-利用旋钮”；冻结一支生成策略，另一支通过 RL 学习，组合发生在整条去噪轨迹上。

### 1.3 信息流/架构图 (Flow / Diagram)

```
Online RL:
  s_t ──► π_E (frozen, RND-pretrained) ──► v_E(x_t, s_t, t)
       ──► π_X (trainable)            ──► v_X(x_t, s_t, t)
                              │
                              ▼
                  v_w = (1-w) v_E + w v_X
                              │
                              ▼
                  DDPM reverse sampler ──► a_t
                              │
                              ▼
                  env step; store (s,a,r,s') in B
                              │
                              ▼
                  update Q and π_X with original QSM/QVPO objective

RL Post-Training:
  s ──► frozen π_X^0 ──► v_X^0(x_t, s, t)
     ──► trainable π_E^θ ──► v_E^θ(x_t, s, t)
                    │
                    ▼
        v_CFG = (1-w) v_base + w v_exp   # 论文 B.4.2 Eq.57
                    │
                    ▼
        Flow-SDE / ODE sampler ──► action chunk
                    │
                    ▼
        PPO + GAE; stop gradient through π_X^0; update π_E^θ
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
v_w = (1-w) v_A + w v_B   # 每一步去噪都作用于同一个 x_t
```

**目标**：构造一个组合策略 π_w，使其在采样时同时受两个生成策略影响；通过 w 控制探索与利用、或预训练能力与自适应行为的相对强度。

**核心公式与变量**：

```
组合场：
v_w(x_t, s, t) = (1-w) v_A(x_t, s, t) + w v_B(x_t, s, t)

固定噪声级的几何组合：
q_{w,t}(x_t|s) ∝ π_{A,t}(x_t|s)^(1-w) · π_{B,t}(x_t|s)^w

端点近似：
π_w^sam(a|s) ≈ q_{w,0}(a|s) ∝ π_A(a|s)^(1-w) · π_B(a|s)^w
```

| 符号 | 含义 |
|------|------|
| v_A, v_B | 两个生成策略的 score 或 velocity field |
| w | guidance weight；论文不同 regime 对“大 w 偏向谁”的写法不一致 |
| x_t | 第 t 步去噪的中间动作样本 |
| s | 当前观测/状态条件 |
| q_{w,t} | 固定噪声级下两个边缘分布的几何组合 |
| π_w^sam | 组合采样器诱导的端点动作分布 |

**直觉**：固定噪声级下，两个 score 的线性组合恰好是几何组合分布的 score，这一步是精确的（论文 Appendix B.1 Eq.18）。但把每个 t 的 q_{w,t} 串成一条完整扩散路径时，前向扩散与几何组合不交换：

```
T_t[p_A^(1-w) p_B^w]  ∝̸  (T_t[p_A])^(1-w) (T_t[p_B])^w
```

因此端点乘积形式是近似，依赖 path-consistency。论文用 Fokker–Planck 推导出残差：

```
∂_t q_{w,t} = L_t^* q_{w,t} + q_{w,t} (R_{w,t} - E_{q_{w,t}}[R_{w,t}])
R_{w,t} = (g_t^2 / 2) w(1-w) ||s_{B,t} - s_{A,t}||_2^2
```

只有 R_{w,t} 的状态依赖变化会影响扩散路径；若 R_{w,t} 对 x_t 恒定，残差会被归一化项抵消。

**Theorem 1 的假设与结论**：

```
假设：
π_X(a|s) = π_E(a|s)^β / Z_β(s), β > 1

则：
q_w(a|s) ∝ π_E(a|s)^(1-w) π_X(a|s)^w
         = π_E(a|s)^(1 + w(β-1))

令 α(w) = 1 + w(β-1)
dH(q_w)/dα ≤ 0, dα/dw = β-1 > 0
所以 dH(q_w)/dw ≤ 0
```

**Adaptive composition**：

```
max_w J_Q(w)  s.t.  D_path(w) ≥ δ

L(w, λ) = J_Q(w) + λ (D_path(w) - δ), λ ≥ 0
```

其中 J_Q(w) 是 critic value surrogate，D_path(w) 是组合采样器与纯利用采样器之间的路径差异估计。论文明确说 D_path 是辅助路径差异，不是最终执行动作分布之间的精确 KL。

> 符号与论文保持一致：v_E 表示探索分支，v_X 表示利用分支，v_X^0 表示冻结预训练分支，v_E^θ 表示可训练 steering 分支。注意 post-training 的 w 加权方向在不同公式/算法中有符号不一致。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，用于验算 Theorem 1 的推导闭环，不是论文实验。

假设动作空间只有两个离散动作，π_E = [0.7, 0.3]，β = 2。

```
Z_β = 0.7^2 + 0.3^2 = 0.49 + 0.09 = 0.58
π_X = [0.49/0.58, 0.09/0.58] = [0.8448, 0.1552]

w=0:
  q = π_E = [0.7, 0.3]
  H = -0.7 ln 0.7 - 0.3 ln 0.3 = 0.6109 nats

w=0.5:
  α = 1 + 0.5*(2-1) = 1.5
  q ∝ [0.7^1.5, 0.3^1.5] = [0.5857, 0.1643]
  归一化：q = [0.7809, 0.2191]
  H = -0.7809 ln 0.7809 - 0.2191 ln 0.2191 = 0.5259 nats

w=1:
  q = π_X = [0.8448, 0.1552]
  H = -0.8448 ln 0.8448 - 0.1552 ln 0.1552 = 0.4316 nats

结论：H(w=0) > H(w=0.5) > H(w=1)
```

这符合 Theorem 1 的单调递减结论：在 π_X 是 π_E 的 sharpened 版本的假设下，增大 w 会使几何组合分布更集中，熵下降。实际 RL 训练中 π_X 是否持续满足该 sharpening 关系，论文没有验证。

## 4. 工程视角 (Engineering View)

### 4.1 计算开销

论文 Table 7 在 HalfCheetah-v4 上报告：

| 算法 | 策略 | 训练时间 (s) | GPU 内存 (MiB) |
|------|------|--------------|----------------|
| QSM | Original | 2.42 | 507 |
| QSM | CFG-Policy | 2.94 | 577 |
| QVPO | Original | 8.87 | 831 |
| QVPO | CFG-Policy | 9.37 | 867 |

训练时间 measured over 1,000 environment steps。论文称“只带来边际计算开销”。从数字看，QSM 增加约 0.52 s / 70 MiB，QVPO 增加约 0.50 s / 36 MiB。

### 4.2 采样步数与控制频率

| 设置 | Online RL QSM | Online RL QVPO | RoboTwin post-training |
|------|---------------|----------------|------------------------|
| 扩散/flow 反向步数 | 5 | 20 | 5 |
| 噪声调度 | VP (b_min=0.1, b_max=10) | Cosine | Flow-SDE noise level π0=0.5, π0.5=0.3 |
| 每步随机性 | DDPM noise prediction | DDPM noise prediction | 1 SDE + 4 ODE 训练；评估 5 ODE |
| 动作 chunk | 不适用 | 不适用 | 预测 50 步，执行前 25 步，再观察 |

工程含义：在线 RL 中，QSM 每步只跑 5 次反向，适合高频环境交互；QVPO 跑 20 次反向，计算更重。RoboTwin 每次预测 50 步动作但只执行 25 步，控制频率与重新观测频率需要按任务调整。组合两个分支意味着每步去噪要做两次网络前向，实际延迟近似“单策略前向 × 2 − 共享 backbone 部分”。

### 4.3 额外分支的预训练成本

Online RL 的探索分支由 RND 预训练，论文 Table 5 写 RND pre-training steps = 100k。论文明确说“其 environment steps 不计入正式 training budget”。工程上这意味着：与基线比较训练预算时，探索分支的预训练成本被排除在外；若把预训练成本计入，样本效率结论会受影响。

### 4.4 内存与部署约束

额外分支需要额外参数和优化器状态。论文 Table 7 的 GPU 内存增加不大（QSM +70 MiB，QVPO +36 MiB），但这是 MuJoCo 小网络。RoboTwin 上冻结的是从 SFT checkpoint 加载的 π0/π0.5 base policy；可训练的 steering 分支是复制其 action decoder 与 action projections 得到的独立 action expert。两者共享冻结的视觉语言 backbone 与 observation features，但 action-expert 参数各自独立。因此部署时会有两份 action-expert 参数（其中一份冻结），不能简单理解为“只多一个小网络”。

## 5. 数据与评测 (Data & Eval)

### 5.1 Online RL

- 环境：5 个 MuJoCo 任务，HalfCheetah-v4、Walker2d-v4、Ant-v4、Humanoid-v4、Hopper-v4（论文 §4.1）。
- 基线：QSM (Psenka et al., 2023) 和 QVPO (Ding et al., 2024)，原 actor/critic 目标不变。
- 探索分支：单独预训练的 RND exploration policy，训练中冻结。
- 指标：mean episodic return 和 sample standard deviation，5 个独立 run（论文 Table 2）。
- 论文实测数字（Table 2）：

| 方法 | HalfCheetah-v4 | Walker2d-v4 | Ant-v4 | Humanoid-v4 | Hopper-v4 |
|------|----------------|-------------|--------|-------------|-----------|
| QSM Original | 5020±455 | 2041±477 | -24±3 | 5922±262 | 2753±307 |
| QSM CFG-Policy | 6218±533 | 2349±583 | 1425±308 | 6094±169 | 3101±461 |
| QSM 提升 | +23.9% | +15.1% | +6141.3% | +2.90% | +12.7% |
| QVPO Original | 11166±5076 | 5220±689 | 7129±61 | 6612±214 | 3727±176 |
| QVPO CFG-Policy | 12990±1993 | 5692±413 | 7374±45 | 7041±385 | 3871±192 |
| QVPO 提升 | +16.3% | +9.0% | +3.4% | +6.49% | +3.8% |

注意：QSM 在 Ant-v4 上从 -24 到 1425，论文报告 +6141.3%；由于基线接近 0 且为负值，百分比解释需要谨慎。

### 5.2 RL Post-Training

- 环境：4 个 RoboTwin 任务，Adjust Bottle、Pick Dual Bottles、Rotate QR Code、Grab Roller（论文 §4.1）。
- 预训练策略：π0 和 π0.5，冻结为 exploitation branch；可训练分支由同一 SFT checkpoint 的 action decoder/action projections 复制初始化。
- 基线：直接 Flow-SDE + PPO fine-tuning。
- 指标：task success rate；训练 50 rounds，每 10 rounds 在 64 fixed scenes 上评估（论文 B.4.2）。
- 论文实测数字（Table 3）：

| 模型 | 方法 | Adjust Bottle | Pick Dual Bottles | Rotate QR Code | Grab Roller |
|------|------|---------------|-------------------|----------------|-------------|
| π0 | FlowSDE+PPO | 82.81% | 43.75% | 34.38% | 92.19% |
| π0 | CFG-Policy | 98.44% | 57.81% | 42.19% | 93.75% |
| π0.5 | FlowSDE+PPO | 89.06% | 12.50% | 7.81% | 100.00% |
| π0.5 | CFG-Policy | 92.19% | 15.62% | 10.94% | 100.00% |

论文未在 Table 3 报告方差或种子数；评估场景为 64 个固定 scene，存在场景过拟合风险。

### 5.3 消融

- Adaptive vs fixed guidance：论文 Figure 4 在 HalfCheetah-v4 上比较 adaptive guidance 与 fixed w ∈ {0.8, 0.98}，adaptive 的最终训练回报更高。该消融只在 HalfCheetah-v4 上展示。
- Generative composition vs action-space blending：论文 Table 4 在 HalfCheetah-v4 上比较直接对两个策略采样动作做插值。action-space blending 一致差于原始 QVPO，即使 blend coefficient 很大。

| Blend coefficient a | Return | Change vs. original QVPO |
|---------------------|--------|--------------------------|
| 0 | -599±171 | -105.44% |
| 0.2 | -362±127 | -103.29% |
| 0.5 | 883±608 | -91.98% |
| 0.8 | 4078±1132 | -62.96% |
| 0.98 | 10642±4787 | -3.33% |
| 1.0 | 11166±5076 | – |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：

- 在 online diffusion RL 中，用一个冻结探索分支持续提供多样性，同时让利用分支学习奖励。
- 在 RL post-training 中，冻结预训练策略，只训练一个 steering 分支，避免直接破坏预训练能力。
- 作为采样接口插到 QSM、QVPO、Flow-SDE+PPO 等兼容 backbone 上，论文报告这些基线都有提升或保持。
- 通过 adaptive w 在 HalfCheetah-v4 上优于固定 w=0.8/0.98。

**不能做什么 / 失败模式**：

- 任务间最优 w 变化大；论文结论明确说“task-specific tuning can still find weights that outperform” adaptive strategy。
- 如果探索分支本身质量差，冻结它可能提供无效探索，组合策略可能被拖累。
- 如果预训练策略已饱和（Grab Roller 上 π0.5 两种方法都是 100%），CFG-Policy 无法展示提升，只能保持。
- 论文只在仿真环境评估：MuJoCo locomotion 和 RoboTwin manipulation；没有真实机器人实验。
- 高维 Humanoid 上提升较小：QSM +2.90%，QVPO +6.49%。
- post-training 的 w 语义在论文 Eq.9 与 Algorithm 2/4、B.4.2 之间不一致，直接照公式实现可能得到相反方向。

### 6.1 隐含假设 (Hidden Assumptions)

1. **两个分支共享同一扩散/flow 参数化与噪声调度**，才能在同一 x_t 上线性组合 field。论文在 post-training 中共享视觉语言 backbone，只复制 action expert。
2. **固定噪声级的 score 组合精确，但跨时间路径近似**。端点乘积形式依赖 path-consistency error ε_pc 足够小，论文没有给出 ε_pc 的定量上界。
3. **Theorem 1 假设 π_X 是 π_E 的 β>1 sharpened 版本**。在线 RL 中 π_X 由 QSM/QVPO 更新，是否持续满足该关系未验证。
4. **组合采样器对可训练分支的梯度足以优化原始 RL 目标**。post-training 提到“when required, for likelihood recomputation”，但未详细说明梯度偏差。
5. **adaptive w 的 D_path 约束能防止坍缩**。D_path 是辅助路径差异，不是执行动作分布的 KL；δ 校准后固定，不随策略演化。
6. **RND 探索分支的 100k 预训练步数不计入正式训练预算**。这影响与基线的公平比较。
7. **论文称 offline-to-online RL post-training，但实验从 SFT checkpoint 开始 online PPO**，未展示从离线数据集到 online 的完整协议。
8. **Table 3 未报告方差/种子数**；64 个固定评估 scene 可能无法代表泛化性。

> 以上第 2、3、4、6、7、8 条为我的条件性分析，论文没有明确验证或充分说明。

## 7. 与相关工作对比 (Comparison)

论文 Table 1 给出代表性生成策略 RL 方法的对比。以下表格整理其要点，✓/△/✗ 沿用论文记号。

| 设置 | 家族 | 方法 | Exploration | Adaptive E-E | E-E Decoupling | Capability Preserv. |
|------|------|------|-------------|--------------|----------------|---------------------|
| Generative RL | Q-weighted | QVPO | ✓ | △ | ✗ | ✗ |
| Generative RL | Q-weighted | SDAC | ✓ | ✓ | ✗ | ✗ |
| Generative RL | Critic-Guided | DIPO | △ | ✗ | ✗ | ✗ |
| Generative RL | Critic-Guided | QSM | △ | ✗ | ✗ | ✗ |
| Generative RL | Critic-Guided | DACER | ✓ | ✓ | ✗ | ✗ |
| Generative RL | Critic-Guided | DIME | ✓ | ✓ | ✗ | ✗ |
| RL Post-Training | Policy-Gradient Tuning | DPPO | △ | ✗ | ✗ | ✗ |
| RL Post-Training | Policy-Gradient Tuning | ReinFlow | ✓ | △ | ✗ | ✗ |
| RL Post-Training | Policy-Gradient Tuning | Flow-SDE | ✓ | ✓ | ✗ | ✗ |
| RL Post-Training | Policy-Gradient Tuning | Flow-Noise | ✓ | ✓ | ✗ | ✗ |
| RL Post-Training | Residual Adaptation | Residual RL | ✗ | ✗ | ✗ | ✓ |
| 任意兼容 backbone | 本文 | Any compatible backbone + CFG-Policy | ✓ | ✓ | ✓ | ✓ |

**与 CFGRL 的区别**：论文强调 CFGRL 把 CFG 当作从固定离线数据学习到的 policy improvement operator；CFG-Policy 组合两个行为不同的策略，并通过 RL interaction 优化其中一个。CFGRL 不解决“冻结一支、训练另一支”的双分支问题。

**面试 Tip**：被问到时先一句话定位——“CFG-Policy 不是新 RL 算法，而是把 CFG 搬到策略组合的采样接口”。然后主动说两个批判点：固定噪声级精确但端点近似；post-training 的 w 加权方向在论文内部不一致。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 diffusion/flow policy RL 的研究者，尤其是已经在用 QSM、QVPO、Flow-SDE+PPO 的团队。
  2. 做 VLA post-training 的工程师，关心“冻结预训练策略 + 训练 steering 分支”的落地方式。
  3. 需要探索-利用解耦机制、且愿意接受额外一条生成分支成本的研究者。

- **建议章节路径**：
  先读 §1 Introduction + Figure 1/2，理解两个 regime 的痛点；再读 §3.1 组合机制与 Proposition 1；然后读 §3.2/§3.3 看两种 regime 的分支冻结方式；接着读 §4.2/§4.3 和 Table 2/3/4；最后读 Appendix B.1 路径一致性 gap 和 Appendix B.4 实现细节。Appendix B.3 是 Theorem 1 的证明，可在需要复现理论时再看。

- **不值得精读的理由**：
  如果你只关心 VLA 的视觉语言模型结构、或只做纯离线 RL、或已经熟悉 CFG 条件生成机制但不关心 RL 策略组合，读摘要和 Table 1/2/3 即可。论文的核心创新在采样接口，不在网络架构。

---
[← Back to Theory](./README.md)

**关键引用链接**：  
- 论文：https://arxiv.org/abs/2610.14936