# Vision-Language-Delta-Action：把紧凑 Delta 作为可操作的状态转移接口 (Vision-Language-Delta-Action: Learning Compact Deltas as Actionable Transition Interfaces for Robotic Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-20
>
> **论文**: Vision-Language-Delta-Action: Learning Compact Deltas as Actionable Transition Interfaces for Robotic Manipulation
> **链接**: https://arxiv.org/abs/2610.25874
> **核心定位**: 用低维 Delta 中间变量替代“完整未来预测”或“直接 observation-to-action”，把 VLM 理解与动作生成通过可操作状态转移接口连接起来。

> **证据说明**：来源记录显示 metadata abstract 与 PDF 不一致，且候选摘要为空；本文只依据已验证身份的 arXiv:2610.25874v1 PDF 正文，不用候选摘要补写或替代 PDF 主张。引用只使用本次论文链接。论文正文提到代码/项目链接，但按本次引用约束不展开，也未验证实际可复现性。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文实测 VLDA 在 LIBERO 平均 98.5%、LIBERO-plus 84.5%、RoboTwin 2.0 选定 hard 任务 62.9%、真实 R1 Lite 73.0% 与 ALOHA 72.0%；作者主张紧凑 Delta 比直接动作预测和完整未来预测更适合作为动作接口。 |
| 适合精读 | 如果你在做 VLA 中间表示、world-action model、动作 chunk、长程操作或 flow matching 动作头，重点看 §4.3–§4.5 与 §5。 |
| 可以跳过 | 如果你只关心底层运动控制、跨本体迁移或真实高频控制，这篇距离中等；跨本体只是 future work。 |
| 落地可行性 | 中：推理只需 World Expert 3 步 + Action Expert 10 步，且训练后丢弃 V-JEPA2 编码器；但依赖两阶段训练、额外 World Expert 与 V-JEPA2 特征。 |
| 主要风险 | 固定 horizon H、无显式记忆、Top-K 对背景运动敏感、真实评测仅 4 任务×2 平台×25 trials（共 200 次），且未报告方差/置信区间。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？VLA 直接把 observation 映射到 action，语义理解与底层执行之间的鸿沟全压给一个策略；WAM 预测完整未来，又把容量浪费在静态外观上。

它发现了什么？机器人操作真正需要的是“当前到未来之间，与任务相关且能被动作消费的那一小块变化”，作者把它压成低维 Delta，并注入动作去噪。

对 VLA 研究者意味着什么？动作生成前可以有一个显式、可学习、可微的 transition interface；“预测未来”不必等于“重建完整未来”。

📍 **研究全景时间线**

```
2016 动作作为状态变换的研究线索
  → 2023 RT-2：VLA 用动作 token 连接视觉语言与机器人控制
  → 2024 π0：连续动作 flow matching 动作专家
  → 2025 V-JEPA 2 / π0.5：自监督视频表征与开放世界 VLA
  → 2026 VLDA（本文）：Delta 作为可操作转移接口
```

节点年份取自论文参考文献；这不是完整谱系。本文位置在“连续动作 VLA”与“world-action model”之间，主张用紧凑 Delta 替代完整未来状态，并用 DDAR 把 Delta 变成动作流的结构化种子。局限是固定 H、无显式记忆、未验证跨本体迁移。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块/对象 | 输入 | 输出 | 训练/推理差异 | 归属与备注 |
|---|---|---|---|---|
| VLM backbone | 多视角 observation、proprioception、language instruction | 多模态上下文 h_t | 基于 π0.5，PaliGemma-initialized 2B；Stage II 解冻 | 论文明确 VLDA 实现在 π0.5 之上；不是所有基线都同架构 |
| World Expert | h_t、噪声 Delta 状态、flow time τ | Delta 预测 Δ̂_t | 与 Action Expert 同构同容量，从零独立初始化；推理 3 步 Euler | 只属于 VLDA；不是 π0.5 原有模块 |
| Action Expert | h_t、动作流状态、Delta prefix token | 动作 chunk A_t | Gemma-300M；推理 10 步 Euler；Stage II 解冻 | 继承 π0.5 动作专家 |
| V-JEPA2 online/EMA encoders | 主视角当前/未来图像 | 当前/未来 dense features | 仅用于 transition supervision；训练后丢弃 | 主视角 256×256，P=256，d_v=1408；不等于部署模块 |
| LT-Filter | dense transition D_t | Top-K Delta target Δ* | Stage I/II 训练目标 | K=64，d_Δ=256；K 是 token 数，不是历史帧数 |
| GT-Anchor | pooled future/current features | 全局 transition anchor G_t | Stage I/II 正则 | 不要求重建完整未来 |
| DSAF | Δ̂_t | Delta-induced noise bias、Delta prefix token | 训练与推理均使用 | DDAR 的一部分 |
| DAMG | Delta tokens、action tokens | InfoNCE 对齐损失 | 仅训练使用，无推理开销 | 负样本来自同一轨迹非重叠时间段 |
| 显式记忆机制 | — | — | — | VLDA 正文没有显式 recurrent memory；长程依赖 action chunk 与当前 Delta，不应把 H 写成所有实验共有的记忆窗口 |

> 数据开放范围：论文说明 VLDA 使用 LIBERO、LIBERO-plus、RoboTwin 2.0 与真实 teleoperated demonstrations；对基线模型的数据开放范围未逐一说明，本文不推断。架构与模块归属按论文逐项记录，不把 V-JEPA2 编码器误写成部署模块，也不把 π0.5 的属性合并给所有基线。

### 1.2 关键机制 (Key Mechanism)

- **分解式动作生成**：不直接学 p(A_t|s_t,l)，而是引入 Δ_t，写成 ∫ pθ(Δ_t|h_t)pφ(A_t|h_t,Δ_t)dΔ_t。World Expert 负责“该发生什么变化”，Action Expert 负责“如何执行变化”。
- **LT-Filter 做选择性**：用 online encoder 编码当前主视角，用 EMA teacher 编码未来主视角，得到 dense transition D_t；按 L2 幅度选 Top-K patch，再降维到紧凑 Delta 空间。目的是压掉静态背景与动作无关变化。
- **GT-Anchor 做全局一致性**：对当前/未来特征做 adaptive pooling 后相减，得到粗粒度 transition anchor；把预测 Delta 上投影回视觉特征空间，与 anchor 做集合匹配。它不重建完整未来，只约束整体变化方向。
- **DDAR 让 Delta 可操作**：DSAF 把 Delta 均值池化后投成动作空间 bias，偏移动作流初始噪声，同时把 Delta 投成 prefix token 前置到动作 token；DAMG 用对比学习把 Delta 拉到 action manifold 附近。
- **两阶段优化**：Stage I 先学 transition interface，冻结 VLM 与 Action Expert；Stage II 再联合动作生成、transition 建模与 Delta-action 对齐。

⚡ **Eureka Moment**：不要预测完整未来；只学未来中“会改变动作”的那一小块 Δ，并把它变成动作去噪的结构化种子。

### 1.3 信息流/架构图 (Flow / Diagram)

```
训练时：
  I_t(主视角/多视角), p_t, l
        │
        ▼
  VLM Φ ──► h_t
        │
        ├──► World Expert fθ(Δτ, τ, h_t) ──► Δ̂_t
        │           ▲
        │           │ 监督：LT-Filter Δ* = g↓(TopK(D_t))
        │           │       GT-Anchor G_t = Pool(V̄_{t+H}) - Pool(V_t)
        │
        └──► DDAR
               ├─ Δ̄_t ─► b_t = q_b(Δ̄_t) ─► ε'_A = ε_A + γ b_t
               ├─ Δ̄_t ─► prefix token
               └─ DAMG：Delta tokens ↔ action tokens（仅训练）

推理时：
  I_t, p_t, l ─► VLM h_t ─► World Expert 3 步 ─► Δ̂_t
       ─► mean-pool + bias/prefix ─► Action Expert 10 步 ─► A_t
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
p(A_t | s_t, l) = ∫ pθ(Δ_t | h_t) pφ(A_t | h_t, Δ_t) dΔ_t
```

目标不是重建未来，而是让 Δ_t 成为“理解→转移→执行”的中间接口。

**Delta 目标与两个约束**：

```
V_t = E_on(I_t^pri),  V̄_{t+H} = E_ema(I_{t+H}^pri)
D_t = V̄_{t+H} - V_t
K_t = TopK_K{ ∥D_{t,i}∥₂ }
Δ*_t = g↓(D_{t,K_t}) ∈ R^{K×d_Δ}

L_LTF = E[ D_set(û_Δτ, u_Δτ; d_Δ) ]
G_t = Pool_K(V̄_{t+H}) - Pool_K(V_t)
L_GTA = E[ D_set(g↑(Δ̂_t), G_t; d_v) ]
```

**流匹配路径**：

```
Δτ = τ ε_Δ + (1-τ) Δ*
u_Δτ = ε_Δ - Δ*

ε'_A = ε_A + γ b_t
Aτ = τ ε'_A + (1-τ) A_t
u_Aτ = ε'_A - A_t
```

**集合匹配**：

```
C_ij = ∥x_i - y_j∥₂² / d
T*_η = Sinkhorn_η(C̄)
D_set(X, Y; d) = <sg(T*_η), C>
```

**动作损失与对齐损失**：

```
L_act = E[ ∥û_Aτ - u_Aτ∥₂² ]
L_DDAR = InfoNCE( Delta-action 正负样本 )
L^(1) = L_LTF + L_GTA
L^(2) = L_act + λ^(1)L^(1) + λ_DDAR L_DDAR
```

**变量说明**：

| 符号 | 含义 | 论文设置/备注 |
|---|---|---|
| h_t | VLM 编码的多模态上下文 | Φ_VLM(s_t,l) |
| Δ_t / Δ̂_t | 可操作 Delta / World Expert 预测 | 低维 transition variable |
| Δ* | LT-Filter 构造的 Delta target | 由 Top-K transition 降维得到 |
| K | Delta token 数 | 论文默认 64；不是历史帧数 |
| d_Δ | Delta 维度 | 256 |
| P | 主视角 spatial tokens | 256 |
| d_v | V-JEPA2 特征维度 | 1408 |
| H | action horizon / action chunk 长度 | LIBERO 为 10；RoboTwin 为 50；真实平台在提供正文中未明确，不能推定 |
| τ | flow time | 训练采样与推理 Euler 步使用 |
| γ | Delta bias 强度 | 可学习，初始 0.01 |
| b_t | Delta 诱导的动作空间 bias | q_b(Δ̄_t) ∈ R^{H×d_a} |

> 符号与时序：K=64 是 Delta token 数，不是帧数；未来监督是单帧 t+H，不是 K 帧历史。若按时间步理解，历史跨度由 action chunk 的 H 决定，而不是 (K-1)Δt。论文没有报告所有平台的 Δt 或控制频率，因此不能把 LIBERO 的 H=10 或 RoboTwin 的 H=50 写成所有实验共有的记忆窗口。VLDA 正文也没有显式记忆模块。

**直觉**：LT-Filter 把“哪里变化大”选出来，GT-Anchor 防止 Delta 只看局部，Sinkhorn 解决 Top-K token 无序匹配，DDAR 把 Delta 变成动作流的初始化偏置与条件 token。作者主张这套约束让 Delta 既选择性强又与未来全局方向一致；这是设计主张，真实收益需看消融。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，只用于验算公式与形状，不代表论文实验。

假设主视角有 P=4 个 patch，当前到未来的 transition 范数为：

```
∥D_1∥₂ = 0.2
∥D_2∥₂ = 1.5
∥D_3∥₂ = 0.4
∥D_4∥₂ = 2.0
```

取 K=2，Top-K 选中 token 2 与 token 4，Δ* 的形状为 2×d_Δ，而不是 2 帧历史。若 d_Δ=2，假设降维后：

```
Δ* = [[1.0, 2.0],
      [0.0, 1.0]]
```

再假设 ε_Δ=[[0.2,-0.1],[0.1,0.3]]，τ=0.3，则逐元素：

```
Δτ = 0.3 ε_Δ + 0.7 Δ*
   = [[0.76, 1.37],
      [0.03, 0.79]]
u_Δτ = ε_Δ - Δ*
     = [[-0.8, -2.1],
        [0.1, -0.7]]
```

若 World Expert 完美预测 Δ̂=Δ*，则恢复速度：

```
û_Δτ = (Δτ - Δ̂) / τ
     = (Δτ - Δ*) / 0.3
     = [[-0.8, -2.1],
        [0.1, -0.7]]
```

与目标速度一致，说明 x-prediction 恢复速度的代数闭环成立。

动作侧再假设 H=2、d_a=2，γ=0.01，b_t=[[0.1,-0.1],[0.2,0.0]]，ε_A=[[0.5,0.5],[0.4,0.6]]：

```
ε'_A = ε_A + 0.01 b_t
     = [[0.501, 0.499],
        [0.402, 0.600]]
```

若 A_t=[[1.0,0.0],[0.0,1.0]]，τ=0.5：

```
Aτ = 0.5 ε'_A + 0.5 A_t
   = [[0.7505, 0.2495],
      [0.2010, 0.8000]]
u_Aτ = ε'_A - A_t
     = [[-0.499, 0.499],
        [0.402, -0.400]]
```

Action Expert 预测 û_Aτ 后与 u_Aτ 做 MSE。这个例子只验证了流路径、形状与 bias 的量级；Sinkhorn 的完整迭代不在此手算。

## 4. 工程视角 (Engineering View)

| 项目 | 论文设置 | 工程含义 |
|---|---|---|
| Stage I | 10K optimization steps，global batch 48；冻结 VLM 与 Action Expert，优化 online V-JEPA2、World Expert、投影模块 | 先建立 transition space；不是 epoch 数 |
| Stage II | 额外 60K steps，global batch 128；解冻 VLM 与 Action Expert | 联合动作生成与对齐；总训练步数与 Stage I 分开 |
| 硬件 | 8×NVIDIA A800-SXM4 80GB；真实推理单张 32GB consumer GPU | 训练成本不低；部署可丢弃 V-JEPA2 编码器 |
| 推理步数 | World Expert 3 步 Euler；Action Expert 10 步 Euler | 共 13 个显式积分步；论文未报告延迟/吞吐 |
| Delta 规模 | K=64，d_Δ=256；prefix 1024 维 | 比完整未来特征图紧凑；但仍有额外 World Expert 前向 |
| Action horizon | LIBERO H=10；RoboTwin H=50；真实 H 未明确 | 不能用单一 H 代表所有实验；控制频率未报告 |
| 记忆 | 无显式 memory | 长程一致性来自 action chunk 与当前 Delta，不应声称 recurrent memory |
| 部署保留 | VLM backbone、World Expert、Delta-to-action conditioning、Action Expert | g↓、g↑、q_Δ、q_act 与两个 V-JEPA2 编码器训练后丢弃 |

**条件性分析**：13 步去噪对高频控制是否可接受，取决于 action chunk 执行方式、控制频率与硬件；论文没有给端到端延迟。若用 H=10 的 chunk 分摊 13 步，单位动作摊销可能可接受；若要求逐时间步闭环，额外 World Expert 前向会成为瓶颈。这是工程判断，不是论文实测。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练/训练设置

- VLDA 以 π0.5 为底座：PaliGemma-initialized 2B VLM backbone + Gemma-300M Action Expert。
- 额外 World Expert 与 Action Expert 同构同容量，但从零独立初始化，不从预训练 Action Expert 初始化。
- V-JEPA2 ViT-g/16 编码器：主视角 256×256，P=256，d_v=1408；online 有梯度，teacher 仅 EMA，m=0.99925。
- 两阶段训练：Stage I 10K steps，batch 48；Stage II 60K steps，batch 128；λ^(1)=1，λ_DDAR=0.1；Sinkhorn 50 iterations，η=0.05；τ_Δ=0.1，τ_c=0.07。
- 动作 horizon：LIBERO H=10，RoboTwin H=50；未来监督取 t_future=min(t+H,T_end)。真实平台 H 在提供正文中未明确。

### 5.2 仿真评测

- **LIBERO**：四个 suite（Spatial/Object/Goal/Long），每 suite 10 tasks；VLDA 平均 98.5%，π0.5 96.9%，+1.6；Long suite 95.2% 对 92.4%，+2.8。（论文 Table 4）
- **LIBERO-plus**：在标准 LIBERO 训练后 zero-shot 评测，10,030 rollouts，七个扰动维度；VLDA 平均 84.5%，π0.5 73.6%，+10.9；最强基线 GuidedVLA 75.4%，+9.1；Long suite 78.3% 对 π0.5 64.9%，+13.4。（论文 Table 1）
- **RoboTwin 2.0**：50 个双臂任务；clean 训练、randomized hard 评测，zero-shot。选定 10 个 hard 任务平均 62.9%，π0.5 42.7%，+20.2；全 50 任务 randomized 平均 50.86%，π0.5 43.84%，+7.02。（论文 Table 2、Table 8）

### 5.3 真实机器人设置

- 平台：GALAXEA R1 Lite（23-DoF 移动双臂）与 AgileX Cobot Magic；四个长程任务：Ring Stacking、Conveyor Sorting、Table Sweeping、Basket Relocation。
- 数据：100、125、125、150 条 human-teleoperated demonstrations；每个任务/平台 25 个物理 trials；按 progressive checkpoints 记录成功。
- R1 Lite 平均完整任务成功率：VLDA 73.0%，π0.5 56.0%，+17.0；ALOHA 平均：VLDA 72.0%，π0.5 54.0%，+18.0；OpenVLA-OFT 17.0%。（论文 Table 3、Table 9）
- 论文未报告真实实验的方差/置信区间；25 trials 是每个任务/平台的 trial 数，不是训练轮次。控制频率与评测频率未在提供正文中报告。

### 5.4 关键消融（论文实测）

| 消融 | 结果 | 说明 |
|---|---|---|
| Full-Future Prediction 替代 Delta | LIBERO 97.8% vs 98.5%；LIBERO-plus 81.1% vs 84.5% | 作者解释完整未来保留更多外观/动作无关信息 |
| 去掉 Transition Pre-Training | LIBERO 97.3%；LIBERO-plus 79.4% | 两阶段先建立 transition space 有增益 |
| 去掉 DSAF | LIBERO 97.2% vs 98.5% | Delta 作为动作流种子有效 |
| 去掉 DAMG | LIBERO 98.2% vs 98.5%；LIBERO-plus 82.0% vs 84.5% | 光照 +6.3、背景 +3.7 增益更明显 |
| Top-K vs Random-K / Spatial pooling | 98.5% vs 97.4% / 97.9% | 选显著变化区域优于随机或空间压缩 |
| OT 匹配 vs Point-wise MSE / Chamfer | 98.5% vs 97.2% / 97.8% | 无序 token 的软匹配更有效 |
| K=16/36/64/100 | 97.7/98.1/98.5/98.5 | K=64 为默认，K=100 无进一步平均增益 |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：

- 在 LIBERO/LIBERO-plus/RoboTwin 与两个真实平台上提升长程任务完成率与扰动鲁棒性；论文实测在光照、背景、相机、噪声等扰动下有增益。
- 用 Delta 作为显式中间变量，让“理解”和“执行”之间多了一个可检查、可注入动作流的接口。
- 推理不需要未来 observation；未来图像只在训练时用于构造 transition supervision。

**不能做什么/容易失败**：

- **背景运动污染 Top-K**：作者在 Limitations 明确指出，强烈背景运动（动态屏幕、移动路人）可能进入 Top-K，因为 LT-Filter 主要按特征差幅度选 token。
- **固定 horizon**：当前 Delta 固定由 action chunk 的 H 决定；变速度、多速率、非均匀时间跨度任务未解决。不要把 H=10 或 H=50 写成通用记忆窗口。
- **无显式记忆**：VLDA 没有 recurrent memory；长程一致性依赖 chunk 与当前 Delta，不是长期历史缓存。
- **主视角限制**：transition supervision 只用 primary camera；多视角只进入策略 backbone，不能声称多视角 transition 建模。
- **真实评测规模有限**：4 个任务、每任务/平台 25 trials，无方差/置信区间；不能据此声称广泛真实场景泛化。
- **跨本体未验证**：作者把 cross-embodiment transition transfer 列为 future work；不能声称已跨机器人迁移。
- **训练与部署复杂度**：两阶段训练、额外 World Expert、V-JEPA2 特征依赖；训练后虽丢弃编码器，但训练管线不轻。

### 6.1 隐含假设 (Hidden Assumptions)

- 特征差幅度大的区域就是任务相关、可操作的 transition；背景运动或光照突变可能违反。
- 训练时未来主视角图像在 t+H 可用，且与动作 chunk 末端对齐；轨迹末端 clipping 合理。
- EMA teacher 提供稳定目标，m=0.99925 的慢更新足以避免 collapse；这是作者设计，不是外部验证。
- Top-K token 是无序集合，Sinkhorn 软匹配比索引回归更合理；消融支持，但仍依赖 cost normalization 与 η。
- 低维 Delta 位于结构化 manifold，x-prediction 参数化合适；这是受 JiT 启发的假设。
- zero-shot 协议与基线设置公平；真实基线用相同 demonstrations/observations/control frequency 复现，但论文未给统计检验。
- 25 trials 足以区分方法差异；本文不把它当作统计保证。

## 7. 与相关工作对比 (Comparison)

| 方法类/代表 | 控制接口 | 未来预测 | 训练/推理特征 | 本文证据与边界 |
|---|---|---|---|---|
| 直接 VLA：OpenVLA、OpenVLA-OFT、π0、π0.5 等 | observation-to-action 或 action chunk | 通常不显式预测未来 | 论文把它们作为基线；具体数据开放范围未逐一说明 | VLDA 在 LIBERO-plus 84.5% 对 π0.5 73.6%；不能把 π0.5 的架构属性合并给所有基线 |
| WAM：WorldVLA 等（DreamVLA 在本文仅作为 VLA 基线） | 未来建模 + 动作 | 预测完整未来/视觉 latent/动作条件 dynamics | 未来预测作为上游上下文、联合目标或正则 | 论文主张完整未来含静态冗余；其 full-future 消融 98.5→97.8、84.5→81.1 |
| Latent action：UniVLA、CLAP、LARA 等 | 离散/连续 latent action | 主要作为伪动作标签或预训练目标 | 论文称其未同时做局部/全局 transition 约束与动作流注入 | VLDA 的区别是 Delta 直接作为动作去噪种子；跨方法结论需外部复现 |
| VLDA（本文） | 紧凑 Delta 中间变量 + DDAR | 只学 current-to-future 的任务相关变化 | 两阶段；World Expert 3 步 + Action Expert 10 步；无显式记忆 | 作者主张更可操作；实测见 §5；固定 H 与真实规模是边界 |

> 数据开放范围与记忆机制按模型归属：论文没有逐一交代所有基线的数据开放范围，本文不推断；显式记忆方面，VLDA 正文可判定为“无显式 memory”，其他模型未在本文展开，不合并归因。

**面试 Tip**：被问“VLDA 和 world-action model 的区别”，一句话答：它不是预测完整未来再动作，而是学一个低维、任务相关、可被动作流消费的 Delta；关键证据是 full-future 消融在 LIBERO 98.5→97.8、LIBERO-plus 84.5→81.1，但要补一句“固定 H、无显式记忆、真实评测规模有限”。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA 中间表示、world model、action chunk 与 flow matching 动作头的研究者；
  - 要评估“Delta 接口”能否迁移到新机器人平台的工程师；
  - 需要理解 Top-K transition 监督、Sinkhorn 集合匹配与动作流注入细节的读者。
- **建议章节路径**：先读 §4.1–§4.5（分解式生成、LT-Filter、GT-Anchor、DDAR），再看 §5（两阶段训练与推理步数），最后看 §6 与附录 B 的消融和真实设置；附录 D.1–D.4 适合核对 EMA、Sinkhorn 与 InfoNCE 细节。参考文献表可跳读。
- **不值得精读的理由**：如果你不做机器人学习、只关心底层控制频率，或已经非常熟悉 flow matching VLA 且只需要结论，读摘要与 §5 的评测表即可；跨本体迁移在本文只是 future work，不能从这篇直接得到答案。

---

**关键引用**：论文链接 https://arxiv.org/abs/2610.25874（本文依据 arXiv:2610.25874v1 PDF 正文；metadata abstract 与 PDF 不一致，未用候选摘要补写）。

[← Back to Theory](./README.md)