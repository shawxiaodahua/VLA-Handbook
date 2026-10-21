# Vector-Quantized World Model：用控制器可达性做离散 latent 规划的机器人操作框架 (Vector-Quantized World Model: Latent Feedback Motion Planning for Robot Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-21
>
> **论文**: Vector-Quantized World Model: Latent Feedback Motion Planning for Robot Manipulation
> **链接**: https://arxiv.org/abs/2610.25600
> **核心定位**: 把 vector quantization 的码本从“Euclidean 最近”改为“controller 到达目标所需 first-passage time”，让离散规划目标天然对齐低层控制器的可达性；规划器在离散 latent 图上做 BFS，控制器用视觉反馈逐段执行。

> **证据版本说明**：本文依据 arXiv:2610.25600v1 的 HTML 正文（2026-10-19）写作；未取得官方 PDF，因此不声称“PDF 正文”版本。候选摘要片段是截断文本，未用于补写论文主张。正文摘要把模拟提升汇总为 13.3–85.3 个百分点（跨 manipulation 与 locomotion），而实验节把 manipulation 的 13.3–34.6 与 Walker 的 22.0→97.3 分开报告；本文以表 I、表 II 和实验节可核对数字为准，并保留这一口径差异说明。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 用 first-passage time（FPT）构造 controller-aware 的状态码本，再把长程规划变成离散图搜索、把执行交给连续视觉反馈控制器；真实四盘 Tower of Hanoi 的 H15 在 expert 数据下达到 99.6% task progress，non-expert short-play 数据下达到 100% task progress |
| 适合精读 | 如果你在做 long-horizon manipulation、world model planning、latent 离散化、或“规划目标低层控制器追不上”的问题，重点看 §IV-B、§IV-C3、§V-A、§V-B |
| 可以跳过 | 如果你只关心 VLA 的 policy 架构、动作 tokenization 或大规模多任务预训练，这篇的距离中等 |
| 落地可行性 | 中：模块边界清晰，但依赖离线数据、FPT 闭环 rollout 估计、码本规模调参；论文未声明代码/数据公开 |
| 主要风险 | FPT 是在 learned latent world model 里估计的，和真实物理可达性有偏差；真实实验只覆盖单机械臂、单任务域；replanning 的收益未系统验证 |

💡 **X-Ray 开场**

它要解决的是：长程操作里，world model 在连续 latent 空间做 autoregressive rollout，预测误差会累积；规划出的 latent 目标又未必是低层 controller 能稳定到达的。论文的答案是：先把状态按“controller 从当前状态追到目标需要多少次调用”来离散化，再在离散码本上做最短路径搜索，最后让连续控制器用视觉反馈逐段跟踪。对 VLA 研究者，这意味着 planning abstraction 不该只看 latent 距离，而要看 controller reachability。

📍 **研究全景时间线**

```
2009  LQR-Trees：反馈运动规划，把轨迹与吸引域/控制器联系起来（论文 ref [11]）
  ↓
2017  VQ-VAE：vector quantization 目标函数（论文 ref [28]）
  ↓
2024  DINO-WM：预训练视觉特征上的 zero-shot planning（论文 ref [1]）
  ↓
2025  V-JEPA 2：自监督视频模型用于理解、预测与规划（论文 ref [20]）
  ↓
2026  H-JEPA / Temporal Straightening：分层 world model 与 latent 直线化（论文 ref [3][8]）
  ↓
2026  VQ-WM（本文）：FPT controller-aware 码本 + 离散搜索 + 连续反馈执行
```

> 时间线只列论文参考文献中有明确年份、且与本文机制直接相关的少量节点；不罗列完整参考文献表。

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

VQ-WM 由 `{E_Z, E_U, P_Z, P_Q, φ_Z, φ_U, π}` 组成。论文明确：`E_Z` 使用 pretrained、frozen 的 V-JEPA 2 编码器；其余模块在离线数据上训练或构造。

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| `E_Z`（frozen V-JEPA 2） | 观测 `o_t` | 连续 latent `z_t` | 规划和执行时都要编码 | 预训练后冻结，不参与训练 |
| `π`（goal-conditioned controller） | 当前 `z_t`、目标 `q_i` 或 `z_goal` | 物理动作 `a_t` | 比 planner 更快，带视觉反馈 | 离线训练，最小化动作 L1 |
| `P_Z`（continuous latent predictor） | `z_t, a_t` | 下一 latent `z_hat_{t+1}` | 用于 FPT 闭环 rollout | 离线训练，监督闭环 rollout |
| `φ_Z` / `Q`（state quantizer/codebook） | `z` | 离散状态 `q` | 离线构造；推理时量化 init/goal | 用 FPT + greedy set cover 构造 |
| `E_U` / `φ_U` / `U`（action encoder/codebook） | 离散状态对 `(q_t, q_t')` | 离散动作 `u_t` | 离线训练 | 标准 VQ 目标 |
| `P_Q`（discrete latent predictor） | `q_t, u_t` | 下一离散状态分布 | planner 展开图时使用 | 交叉熵训练 |
| Planner（BFS） | `q_init, q_goal, P_Q` | 路径 `(q_0,...,q_L)` | 比 controller 慢；可 open-loop 或 receding-horizon | 推理时在线展开图 |
| Controller execution | `z_t, q_i` | `a_t` | 高频控制循环 | 按 FPT 分配每个目标的 controller calls |

论文强调两个时间尺度：planner 较慢，在离散图上搜索；controller 较快，用视觉反馈跟踪计划目标。实验采用“每个 episode 只规划一次”的 open-loop planning，但执行阶段仍是连续视觉反馈控制。

### 1.2 关键机制 (Key Mechanism)

- **离散化不是为了压缩，而是为了对齐 controller**：普通 VQ 用 Euclidean 最近邻分配码本，但 latent 近不代表 controller 容易到达。VQ-WM 用 FPT 作为 directed、controller-dependent 的 quasi-metric。
- **FPT 的定义**：从状态 `z` 出发，用 controller `π` 追目标 `z'`，在 learned latent world model `P_Z` 里闭环 rollout，直到首次进入 `z'` 的 `ε`-邻域所需的 controller calls 数。
- **码本构造**：对每个候选码 `e`，用 `d(z,e) ≤ R` 估计其 `R`-步吸引域 `Rhat(e)`；再用 greedy set cover 选最少码本覆盖离线 latent 状态集。
- **状态分配**：`φ_Z(z)=argmin_{q∈Q} d(z,q)`；goal 特殊处理为 `q_goal=argmin_{q∈Q} d(q,z_goal)`，因为 `d` 是有向的。
- **离散规划**：对每个 `q` 和每个 `u∈U`，取 `P_Q` 的最可能后继，构建图，用 BFS 找最少图转移的最短路径。
- **连续执行**：按计划顺序跟踪 `q_i`；每个 transition 运行由 FPT 估计出的 controller calls 数，然后切换到下一个目标；最终尝试 `z_goal`。
- **避免不可达目标**：这是与“先规划 latent、再让 controller 尽量追”的关键区别；码本本身由 controller 可达性塑造。
- **拒绝类**：真实实验中，作者给离散 predictor 增加一个 rejection class 表示未见过的 transition，并在图搜索中排除该类预测，以减少不支持的外推。

⚡ **Eureka Moment**：THE 关键洞见是——**vector quantization 的距离不该由 latent 几何定义，而该由“controller 从当前状态到达目标需要多少次闭环调用”定义**；这样离散规划目标天然是控制器能追到的状态。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
离线训练
  D = reward-free offline trajectories
        │
        ▼
  E_Z（V-JEPA 2，frozen）──▶ Z_D（连续 latent 状态集）
        │
        ├── 训练 π（goal-conditioned controller）
        └── 训练 P_Z（continuous latent predictor）
                    │
                    ▼
        FPT 估计 d(z,z')：在 P_Z 中闭环 rollout π
                    │
                    ▼
        Rhat(e) = {z : d(z,e) ≤ R}
                    │
                    ▼
        greedy set cover ──▶ Q, φ_Z
                    │
                    ▼
        固定 Q 后，联合训练 E_U、U、P_Q
```

```text
推理/执行
  o_init, o_goal
        │
        ▼
  E_Z ──▶ z_init, z_goal ──▶ φ_Z ──▶ q_init, q_goal
        │
        ▼
  P_Q + BFS 在离散图上找路径 (q_0,...,q_L)
        │
        ▼
  for q_i in path:
      π(z_t, q_i) ──▶ a_t
      运行 d(q_{i-1}, q_i) 次 controller calls
      视觉反馈纠正偏差
        │
        ▼
  最终追 z_goal；任务完成则成功
```

> 若 `d(z_init,z_goal) ≤ R`，论文的实现会绕过图搜索，直接用 `π` 以 `z_goal` 为目标。

---

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```text
q = argmin_{q∈Q} d_FPT(z,q)，其中 d_FPT = controller 首次进入目标 ε-邻域所需的调用次数
```

先给目标：学习一个 controller-aware 的离散状态空间，使 planner 在离散图上找最短路径，controller 在连续 latent 上用视觉反馈执行。再给公式。

**Controller 训练目标**：

```text
L_control = || a_t - a_hat_t ||_1
a_hat_t = E_{a ~ π(·|z_t, z_{t+r})}[a]
```

论文从同一离线轨迹采样状态–目标对 `(z_t, z_{t+r})`，`r ~ Uniform{0,...,R}`；`r=0` 时用 hold action 替换 `a_t`，让 controller 学会保持当前状态。

**Continuous latent predictor 的闭环 rollout 监督**：

```text
L_rollout = (1/R) · Σ_{k=0}^{R-1} || z_hat_{k+1} - z_{t+min{k+1,r}} ||_2^2
```

其中 `z_hat_0 = z_t`，`z_hat_{k+1}` 由 `P_Z` 在 `π` 的闭环动作下预测；记录轨迹在到达目标后保持在目标。

**First-passage time 估计**：

```text
a_hat_k ~ π(·|z_hat_k, z')
z_hat_{k+1} ~ P_Z(·|z_hat_k, a_hat_k),  z_hat_0 = z
d(z,z') = min{ k ∈ {0,...,R} : || z_hat_k - z' ||_2 ≤ ε }
若 R 次内未进入 ε-邻域，则 d(z,z') = ∞
```

**吸引域与码本构造**：

```text
Rhat(e) = { z ∈ Z_D : d(z,e) ≤ R }
目标：选最小 Q，使 Z_D = ∪_{q∈Q} Rhat(q)
实现：greedy set cover 近似
```

**状态量化与 goal 分配**：

```text
φ_Z(z) = argmin_{q∈Q} d(z,q)
q_goal = argmin_{q∈Q} d(q,z_goal)
```

注意 `d` 是有向的，因此 goal 的分配方向与普通状态分配不同。

**Action codebook 的 VQ 目标**：

```text
u_tilde_t = E_U(q_t, q_t')
u_t = φ_U(u_tilde_t) = argmin_{u∈U} || u_tilde_t - u ||_2^2
L_VQ = || sg(u_tilde_t) - u_t ||_2^2 + β · || u_tilde_t - sg(u_t) ||_2^2
```

**离散 transition predictor**：

```text
L_transition = -log P_Q(q_t' | q_t, u_t)
L_joint = L_transition + L_VQ
```

| 符号 | 含义 |
|------|------|
| `z_t` | 连续 latent 状态 |
| `q_t` | 离散化后的状态码 |
| `u_t` | 离散 action code，表示离散状态之间的 transition |
| `a_t` | 物理动作 |
| `R` | controller-call budget，不是所有实验共有的动作 chunk 长度 |
| `ε` | 到达目标的 latent 邻域阈值 |
| `d(z,z')` | 从 `z` 到 `z'` 的 first-passage time 估计 |
| `Q` | 状态码本 |
| `U` | 动作码本 |
| `P_Q` | 离散状态转移的 categorical predictor |
| `β` | commitment loss 权重 |

**直觉**：Euclidean 距离问“latent 离哪个码最近”；FPT 问“controller 从当前状态追哪个码最省调用”。前者是几何，后者是控制能力。VQ-WM 把 planning abstraction 建在后者上，因此离散图上的每一条边更接近“controller 真能走完的一段”。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

> **以下为假设数值，仅用于验算 FPT 码本构造与分配流程，不是论文实验数据。**

设 `R=3`，`ε=0.1`，有三个离线 latent 状态 `z0,z1,z2`，两个候选码 `eA,eB`。假设 FPT 矩阵为：

```text
d(·, ·)        eA     eB
z0             1      2
z1             2      1
z2             4      1     # 4 > R，视为 R 次内不可达
```

**第一步：估计吸引域**

```text
Rhat(eA) = {z0, z1}          # z2 因 d=4 > R 被排除
Rhat(eB) = {z1, z2}          # z0 的 d=2 ≤ R，但 eB 的集合按矩阵为 {z1,z2}
```

这里为了演示 greedy set cover，假设 `Rhat(eB)` 只覆盖 `{z1,z2}`；两个候选各覆盖 2 个状态。

**第二步：greedy set cover**

- 初始未覆盖 `V={z0,z1,z2}`。
- `eA` 和 `eB` 覆盖数相同，假设选 `eA`；覆盖后 `V={z2}`。
- 再选能覆盖 `z2` 的 `eB`；`Q={eA,eB}`。

**第三步：状态分配**

```text
φ(z0) = argmin{ d(z0,eA)=1, d(z0,eB)=2 } = eA
φ(z1) = argmin{ d(z1,eA)=2, d(z1,eB)=1 } = eB
φ(z2) = argmin{ d(z2,eA)=4, d(z2,eB)=1 } = eB
```

**第四步：与 Euclidean 对比**

假设一维坐标：

```text
eA = 0.0
eB = 1.0
z0 = 0.1
z1 = 0.4
z2 = 0.9
```

Euclidean 分配：

```text
z0：|0.1-0.0|=0.1 < |0.1-1.0|=0.9  → eA
z1：|0.4-0.0|=0.4 < |0.4-1.0|=0.6  → eA
z2：|0.9-0.0|=0.9 > |0.9-1.0|=0.1  → eB
```

对比：

| 状态 | Euclidean 分配 | FPT 分配 | 差异 |
|------|----------------|----------|------|
| `z0` | `eA` | `eA` | 相同 |
| `z1` | `eA` | `eB` | FPT 认为 `eA` 虽近但 controller 要 2 次调用，`eB` 只要 1 次 |
| `z2` | `eB` | `eB` | 相同 |

这个玩具例子验证了核心论点：**latent 近不等于 controller 容易到**。若 planner 用 Euclidean 码本，可能把 `z1` 规划到 `eA`，但 controller 实际更快到达 `eB`；FPT 码本把规划目标和执行能力对齐。

**第五步：controller 调用预算**

假设 BFS 在 `P_Q` 上找到路径 `eA → eB`，且 `d(eA,eB)=2`。执行时 controller 先朝 `eB` 运行 2 次调用，然后切换/继续后续目标；这不是固定长度的 action chunk，而是按每个 transition 的 FPT 分配预算。若路径更长，每个 transition 的调用数可以不同。

---

## 4. 工程视角 (Engineering View)

| 工程点 | 论文给出的信息 | 工程含义 |
|--------|----------------|----------|
| 规划延迟 | 作者报告：完整规划约 5 分钟机器人执行，在 NVIDIA RTX 5080 上不到 1 秒 | 离散图搜索比连续 action sequence 优化便宜；但这是作者设置下的数字，不是跨硬件保证 |
| 规划/控制频率 | planner 较慢，controller 较快；论文未给精确 Hz | 不能把数据采集 30 Hz 当成控制频率；部署时要分别测 planner 与 controller 的周期 |
| 码本规模 | 消融覆盖约 250–2000，约 8 倍范围；过小会合并不同配置，过大则每个 transition 的观测变稀疏 | 码本大小是核心超参；不是越大越好 |
| FPT 估计成本 | 需要在 `P_Z` 里做闭环 controller rollout；实现上用 KNN 和记录轨迹近似 `Rhat(e)` | 离线构码本有计算成本；在线只做量化与图搜索 |
| 内存/图规模 | 离散状态图由 `P_Q` 在线展开；实验用 open-loop 每 episode 规划一次 | 没有 K 帧观测缓存式的“记忆窗口”；图搜索的开放列表是主要运行时结构 |
| 动作 chunk | 论文没有 action chunk 机制；controller 每次输出一个物理动作 | 不能把 `R` 或路径长度写成所有实验共有的 action chunk 长度 |
| 记忆窗口 | 论文不以 `K` 帧 × `Δt` 定义记忆；时间跨度由 controller calls 和 replanning interval 决定 | 若工程实现改用视频帧，历史跨度必须按 `(K-1)×Δt` 计算；本文没有统一 `K/Δt`，不能把单一设置的步长推广到所有实验 |
| 失败保护 | 真实实验给 `P_Q` 增加 rejection class，排除未见 transition | 这是对离散 predictor 外推的工程补丁；不是理论保证 |
| 部署约束 | 真实实验用 WidowX AI、单臂、桌面 Tower of Hanoi；推理用单张 RTX 5080 | 不能据此声称移动、双臂、人形或开放世界任务有效 |

一个重要的工程边界是：VQ-WM 的“记忆”不是 Transformer 的上下文窗口，而是**离散状态图 + 每个目标的 FPT 调用预算 + controller 的视觉反馈**。因此，讨论它的长程能力时，应该问“图能否连到 goal、FPT 估计是否准、controller 能否在预算内到达”，而不是问“K 帧历史有多长”。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练/离线训练设置

- **编码器**：`E_Z` 使用 pretrained、frozen 的 V-JEPA 2；论文没有说在本任务上微调该编码器。
- **离线数据**：`D` 是 reward-free offline trajectories，形式为观测–动作序列；目标是测试时从 `o_init` 到 `o_goal`。
- **训练顺序**：先训练 `π` 和 `P_Z`；再用闭环 rollout 估计 FPT；再构造 `Q` 和 `φ_Z`；最后固定 `Q`，联合训练 `E_U`、`U`、`P_Q`。
- **Controller 训练**：从同一轨迹采样 `(z_t, z_{t+r})`，`r ~ Uniform{0,...,R}`；`r=0` 用 hold action。
- **真实数据采集**：Tower of Hanoi play data 由 WidowX AI 以 30 Hz 自主采集；符号 planner 生成随机合法移动序列；episode 边界用视频目标分割模型 Cutie 检测棋盘状态；每次磁盘转移用预定义 Cartesian pick-and-place 执行。
- **仿真消融数据**：MuJoCo 版 Tower of Hanoi 含 104 条 walk trajectories，覆盖所有合法磁盘移动 transition。
- **开放范围**：候选记录 `repo_url` 为空，论文正文未声明代码或数据公开；本文不推断可复现资源。

### 5.2 仿真评测设置

**Long-horizon manipulation**：使用 Stable-WorldModel 的 Cube 和 Scene 环境。

| 任务 | 评测协议 | 执行预算 | 成功条件 |
|------|----------|----------|----------|
| Cube-Easy | 跟随 LeWM 协议；goal 在 25 个 environment steps 后 | 50 steps | 按任务协议 |
| Cube-Hard | 跟随 H-JEPA 协议；50 个 held-out、20-step windows，中心在 grasp | 30 steps | 完整 pick-and-place |
| Scene | 作者定义：从 expert 轨迹采初始/目标图像，相隔 50 environment steps | 100 steps | cube、buttons、drawer、window 等所有对象同时匹配 goal |

**表 I：Long-horizon manipulation success rates（论文 Table I）**

| Method | Cube-Easy | Cube-Hard | Scene |
|--------|-----------|-----------|-------|
| LeWM | 74.0±2.8 | 32.0±6.1 | 42.7±1.9 |
| TS | 68.0±2.8 | 10.7±0.9 | 47.3±2.5 |
| PLDM | 62.0±4.3 | 11.3±3.8 | 44.7±4.1 |
| H-JEPA | 76.7±2.5 | 60.0±5.0 | 40.7±2.5 |
| Ours | **84.7±2.5** | **73.3±0.9** | **75.3±5.0** |

论文的解读：相对当题最强基线，Cube-Easy 高 8.0 个百分点，Cube-Hard 高 13.3 个百分点，Scene 高 28.0 个百分点；相对 H-JEPA，Scene 高 34.6 个百分点。注意“最强基线”在不同任务上不是同一个方法。

**Locomotion**：DeepMind Control Walker；goal 图像在 expert walking 轨迹上向前 50 steps 采样；成功要求 torso height、torso pitch 和六个 leg joint angles 都在 goal 值 0.3 以内。

**表 II：Walker success rate（论文 Table II）**

| Method | Images | Images + Proprio. |
|--------|--------|-------------------|
| DINO-WM | 12.7±3.4 | 12.0±4.9 |
| Ours | 22.0±5.9 | **97.3±2.5** |

论文解释：纯图像下两种方法都差，因为精确关节协调难以从图像中读出；加入 proprioception 后，VQ-WM 的 controller 变强，controller-aware 的 vector quantization 也随之适配，planner 不需要改。DINO-WM 仍停在 12.0%。

**Tower of Hanoi 消融**：

- 环境：MuJoCo 模拟，相同机械臂和谜题设置。
- 数据：104 条 walk trajectories，覆盖所有合法 disk-move transitions。
- 对比：FPT vs squared Euclidean，用于 codebook construction 和 state assignment；encoder、controller、训练流程不变。
- 规模：每个点 3 个 seeds，每个 seed 50 条 trajectories，共 150 条；初始/目标随机采样，最短路径 1–15 个 disk moves。
- 结果：FPT 峰值约 90% success，Euclidean 约 40%，提升约 50 个百分点；FPT 在约 250–2000 的码本规模上保持较好表现，约 8 倍范围。两端下降：太小会合并不同配置，太大会让每个 transition 的观测变稀疏。

### 5.3 真实机器人设置

- **平台**：WidowX AI 机械臂；四盘 Tower of Hanoi。
- **任务**：把整堆磁盘移到另一根杆；一次只能移动一个盘；不能大盘压小盘。最优解 H15 需要 15 个 puzzle moves。
- **对照任务**：H1 单次移动，用于比较短程与长程。
- **数据设置**：
  - Expert 数据：6 个初始–目标配置对的完整解法演示；VLA/WAM baseline 在这 6 个任务上联合训练。
  - Non-expert 数据：play-data 子任务，最多覆盖 5 个 puzzle moves；在完整 H15 上评测，测试能否组合短子任务。
- **评测协议**：每个方法在相同的 6 个测试案例上评测，每个案例 3 次 trials；推理用单张 NVIDIA RTX 5080。
- **Baseline**：Cosmos Policy（WAM）、`π_0.5`（VLA）、V-JEPA 2-AC（world foundation model）；V-JEPA 2-AC* 额外给 6 个手工指定的 subgoal images per puzzle move。
- **VQ-WM 设置**：给离散 latent predictor 增加 rejection class，排除未见 transition；规划采用每 episode 一次（open-loop planning），执行时连续视觉反馈；作者报告完整解可在一次 planning call 中算出。

**真实结果（论文 §V-B / Fig. 4，数值为 task progress）**

| 设置 | 方法 | H1 | H15 |
|------|------|----|-----|
| Expert | Cosmos Policy | 100% | 76% |
| Expert | `π_0.5` | 100% | 23% |
| Expert | V-JEPA 2-AC | 0% | 0% |
| Expert | V-JEPA 2-AC*（手工 subgoal） | — | 94±6% |
| Expert | VQ-WM | — | 99.6% |
| Non-expert | Cosmos Policy | — | 5% |
| Non-expert | `π_0.5` | — | 3% |
| Non-expert | V-JEPA 2-AC* | — | 27% |
| Non-expert | VQ-WM | — | 100% |

> 术语注意：论文实验节主要写 **task progress**，摘要写 **success rate**；本文在真实实验处保留 task progress，避免把“进度”误写成所有案例的完全成功率。Non-expert short-play 设置为 100% task progress；Expert H15 为 99.6%。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**

- **组合短子任务成长程解**：non-expert 数据最多 5 个 puzzle moves，VQ-WM 在完整 H15 上达到 100% task progress；这是论文最直接支持“组合性”的证据。
- **离散搜索降低长程 action 优化难度**：planner 不需要优化完整低层动作序列，而是在 `Q` 上找最短路径。
- **连续反馈纠正执行偏差**：controller 以视觉反馈跟踪每个计划目标，缓解 compounding execution errors。
- **controller-aware 码本比 Euclidean 更有效**：消融中 FPT 峰值约 90%，Euclidean 约 40%；差异主要来自“latent 近但 controller 不可达/切换过早”。
- **模块化扩展传感模态**：Walker 加 proprioception 后，controller 变强，VQ-WM 从 22.0% 升到 97.3%，planner 不变；说明规划空间可以随 controller 能力重量化。

**不能做什么 / 失败模式**

- **FPT 估计依赖 learned world model**：`P_Z` 的闭环 rollout 如果偏，码本和调用预算都会偏；论文没有给出真实物理 FPT 与估计 FPT 的系统误差分析。
- **真实实验域窄**：只有单臂、桌面、四盘 Tower of Hanoi；不能据此声称移动、双臂、人形或开放世界操作有效。
- **码本规模敏感**：太小合并配置，太大会让每个 transition 的观测稀疏，`P_Q` 变差；论文只给了一个消融范围，没有给出自动选码本的方法。
- **open-loop planning 的局限**：实验每个 episode 只规划一次；虽然执行有反馈，但计划序列本身不更新。若偏差需要改变子目标顺序，单次计划可能不够。作者也明确说 replanning 的收益“未探索”。
- **离散 predictor 的外推风险**：真实实验用 rejection class 排除未见 transition；这说明未见 transition 是实际风险，不是理论细节。
- **视觉-only Walker 很差**：22.0% 说明仅靠图像难以精确判断关节协调；VQ-WM 的优势需要 proprioception 才充分体现。
- **未声明代码/数据开放**：复现门槛较高；候选记录没有 repo URL，论文正文也没有开放声明。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设 1**：frozen V-JEPA 2 latent 保留了足够任务相关信息；论文没有系统比较其他 encoder。
- **假设 2**：在 `P_Z` 中估计的 FPT 能代表真实机器人 controller 的可达性；这跨了“learned latent rollout”和“物理执行”两层。
- **假设 3**：离线数据覆盖的 `Z_D` 能通过有限码本覆盖；对分布外初始/目标，`φ_Z` 和 `P_Q` 可能失效。
- **假设 4**：greedy set cover 的局部近似足以构造有效码本；论文用 KNN 和记录轨迹近似 `Rhat(e)`，没有给近似误差界。
- **假设 5**：`P_Q` 的最可能后继足以做 BFS；虽然论文说 categorical 分布可配 stochastic shortest-path，但实现取 `argmax` 后继。
- **假设 6**：controller 能在 FPT 估计的调用数内到达目标；执行时按该预算切换，若实际更慢，可能提前切换。
- **假设 7**：单次 open-loop 计划在视觉反馈下足够可靠；这需要计划的目标序列本身正确。

---

## 7. 与相关工作对比 (Comparison)

| 方法 | 规划抽象 | 执行/控制 | 训练数据与开放范围（论文所述） | 记忆/时间跨度机制 |
|------|----------|-----------|-------------------------------|-------------------|
| VQ-WM（本文） | FPT controller-aware 状态码本 + 离散 `P_Q` + BFS | 连续 `π` 视觉反馈，逐目标按 FPT 调用预算执行 | reward-free offline trajectories；真实 Tower of Hanoi play data 30 Hz；未声明代码/数据公开 | 离散状态图 + 每目标 FPT 预算；不是 K 帧缓存；实验每 episode 规划一次 |
| H-JEPA | 多层级/多预测 horizon 的 hierarchical world model | 预测 subgoal，controller 执行 | 论文未述其原始数据开放范围；在本文中作为 baseline | 层级 latent/horizon 提供长程抽象；不是 VQ-WM 的 FPT 码本 |
| LeWM | 端到端 joint-embedding predictive architecture | 连续 latent planning | 论文未述；作为 baseline | 连续 latent rollout；Cube-Easy 协议 |
| TS（Temporal Straightening） | 重塑 latent 空间，使 Euclidean 距离更适合规划 | 连续规划/控制 | 论文未述；作为 baseline | latent 直线化；没有 controller-aware 离散码本 |
| PLDM | 从 reward-free offline data 学 latent dynamics planning | 连续 latent planning | 论文未述；作为 baseline | 连续 latent dynamics rollout |
| DINO-WM | 预训练视觉特征上的 world model，zero-shot planning | 连续规划 | 论文未述；作为 baseline | 视觉 feature latent planning；Walker 中纯图像 12.7% |
| V-JEPA 2-AC | action-conditioned V-JEPA 2；CEM 优化连续动作序列 | receding-horizon，执行第一个动作后重规划 | 论文未述；作为 baseline | 固定 lookahead 的 CEM；长 lookahead 会累积预测误差；V-JEPA 2-AC* 加手工 subgoal |
| Cosmos Policy | 视频模型 fine-tune 为 WAM | 无 runtime search 的 policy 执行 | 论文未述；作为 baseline | 行为克隆式执行；H15 从 H1 的 100% 掉到 76% |
| `π_0.5` | VLA policy | 无 runtime search | 论文未述；作为 baseline | 端到端 policy；H15 从 H1 的 100% 掉到 23% |

**对比要点**

- VQ-WM 与 H-JEPA 的差别不只是“离散 vs 连续”：VQ-WM 的码本由 controller reachability 构造，H-JEPA 的 subgoal 是层级预测，可能仍出现“controller 追不上”的问题。
- VQ-WM 与 TS 的差别在于：TS 通过重塑 latent 让 Euclidean 距离有用；VQ-WM 保持 frozen encoder，把规划几何单独学成 FPT。
- VQ-WM 与 V-JEPA 2-AC 的差别在于：后者在固定 horizon 上优化连续动作序列，长 lookahead 的预测误差会累积；VQ-WM 用离散图搜索绕开完整动作序列优化。
- 不能把“离散码本”当成所有 baseline 的共享属性：H-JEPA 是层级 world model，TS 是 latent straightening，V-JEPA 2-AC 是 CEM + receding horizon，Cosmos Policy 是 WAM，`π_0.5` 是 VLA；它们的训练数据、开放范围和记忆机制应按模型分别看。

**面试 Tip**：被问到 VQ-WM 的关键，不要只说“用了 vector quantization”。要答：**码本由 first-passage time/controller reachability 构造，规划在离散图上做 BFS，执行由连续视觉反馈 controller 逐段完成；它解决的是 compounding prediction error 和 latent plan 不可达，而不是单纯压缩 latent。**

---

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 long-horizon manipulation、world model planning 的研究者：重点看 §IV-B 的 planner/controller 双速率结构、§IV-C3 的 FPT 码本构造、§V-C 的消融。
  2. 要评估“离散 latent + 连续控制”落地可行性的工程师：重点看 §V-A 的评测预算、§V-B 的真实设置、§VI 的局限。
  3. 研究 VLA 与 world model 结合的人：重点看 §VII 的 modular 扩展讨论，以及 Walker 加 proprioception 的实验。

- **建议章节路径**：
  先读 §IV-B（推理流程）→ 再看 §IV-C3（FPT 与码本）→ 然后看 §V-A 表 I/表 II 和 §V-B 真实结果 → 最后看 §V-C 消融与 §VI 局限。若只关心机制，§II 相关工作可快速扫过；若只关心真实机器人，§V-A 可略读。

- **不值得精读的理由**：
  如果你不做机器人学习、不关心 world model planning，或者已经非常熟悉 vector quantization 与 BFS 规划，读摘要和 §1.2 即可。论文的贡献集中在“controller-aware 离散化 + 长程执行”，不是新的 VLA 架构或大规模多任务预训练。

---

**关键引用**：[arXiv:2610.25600](https://arxiv.org/abs/2610.25600)（本文依据 v1 HTML 正文；未取得官方 PDF；候选摘要片段未用于补写；论文未声明代码/数据公开）

[← Back to Theory](./README.md)