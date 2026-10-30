# 视觉-语言-动作还是世界-动作？其实只需动作模型 (Vision-Language-Action or World-Action? Just Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-30
>
> **论文**: Vision-Language-Action or World-Action? Just Action Models
> **链接**: https://arxiv.org/abs/2610.37219
> **核心定位**: 把 VLA/WAM 之争拆成 backbone 初始化、未来观测目标、推理保留模块三个可分离选择；JAM 用冻结未来表征的 foresight loss 替代视觉生成损失，在仿真 benchmark 上匹配或超过匹配 WAM，并降低训练成本。

> **来源版本说明**：本文依据来源提供的论文正文（标注 arXiv:2610.37219v1 [cs.CV] 27 Oct 2026）。候选 `abstract_snippet` 是截断摘要，止于“cannot simply be at”，不用于补写；候选元数据标 `cs.RO`，正文标 `cs.CV`，记录差异。以下数字与主张均以论文正文表格为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 未来观测可以作为冻结表征目标来监督当前特征，不必做视觉生成；JAM 在 LIBERO、LIBERO-Plus 上超过匹配的 FastWAM/ImageWAM，RoboTwin 相近或略高。 |
| 适合精读 | 如果你在做 VLA/WAM 架构消融、未来监督、训练成本优化，重点看 §3.2、§4.2、§4.3、§4.5。 |
| 可以跳过 | 如果你只关心真实机器人部署或视频生成模型细节，这篇距离中等：它只做仿真评测。 |
| 落地可行性 | 中：训练配方简单、可缓存 targets、部署路径不变；但真实机器人未验证，cache 依赖固定 anchor 与预处理。 |
| 主要风险 | 跨 backbone 比较不能隔离预训练效应；Ovis-U1 与 Wan 的最佳 target 不同；部分扰动类别不如 action-only。 |

💡 **X-Ray 开场**

这篇论文问：VLA 与 WAM 的差距，究竟来自 backbone 预训练、未来观测上的损失，还是推理时保留的生成模块？作者发现，把未来观测只当作冻结 anchor 的预测目标，就能替代视觉生成损失；跨生成式、自监督、视觉-语言 backbone 都能用同一配方。对 VLA 研究者意味着：不要先问“该继承 VLM 还是生成器”，先把三个选择拆开做对照。

📍 **研究全景时间线**

```
2025  OpenVLA / π0 / π0.5：VLA 以 action loss 为主
  → 2026  FastWAM / ImageWAM：WAM 去掉测试时生成，保留训练时生成/图像目标
  → 2026  JAM（本文）：用 foresight loss 替代生成损失，丢弃预测分支
  ← 当前位置：仿真证据成立；真实机器人、跨 backbone 归因仍待验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 配方 | backbone 初始化 | 未来观测目标 | 推理保留 | 损失 |
|------|----------------|--------------|----------|------|
| VLA reference | 预训练视觉-语言模型 + action expert | 不使用记录的未来 | backbone + action expert | `L_VLA = L_act` |
| WAM reference | 预训练图像/视频生成器 | 对未来观测做生成损失 | 部分 WAM 保留生成路径 | `L_WAM = L_act + λ_G·L_G` |
| JAM | 任意预训练视觉 backbone + action expert | 冻结未来 anchor 的 cosine foresight | 仅 backbone + action expert | `L_JAM = L_act + λ_F·L_F` |

**逐模型归属**（不要把一组模型的属性合并给每个模型）：

| JAM 实例 | backbone | 动作接口/上下文 | 未来 anchor 与 readout | 动作 chunk / 未来位置 |
|----------|----------|------------------|--------------------------|------------------------|
| JAM (Wan) | Wan2.2-5B，30 个视觉 block | 同深度 action expert；directional mixed attention | 冻结 Wan 或冻结 VAE；block 18/30；可选 offsets `[1,2]` | 32 步 action chunk；帧 0,4,…,32 |
| JAM (Ovis-U1) | Yak18 图像编辑 backbone，6 double + 12 single，宽 1536 | 18-block action expert，匹配宽度；attend text、condition-image、action keys | 冻结 Yak layer 11/18，或 VAE；offset `[1]` | 16 步 action chunk；未来单帧 16 步 ahead |
| JAM (FLUX.2) | FLUX.2 Klein 4B，5 double + 20 single，宽 1024 | 匹配 action expert；文本来自冻结 Qwen3-4B，512 cached tokens | 冻结 FLUX.2 layer 15/25；offset `[1]` | 16 步 action chunk；未来单帧 16 步 ahead |
| JAM (Qwen3.5-2B) | 24 层 VLM，图文联合输入 | action expert 复制 18 个 GatedDeltaNet + 6 个 full-attention block；causal token mixing | 冻结 Qwen anchor layer 13/24；未来图像特征监督同层当前图像特征 | 16 步 action chunk；未来 views 16 步 ahead；未来 views 不进 policy prefix |
| JAM (DINOv2-B/L) | DINOv2-B 86.6M / DINOv2-L 304M，自监督图像编码器 | action expert 镜像深度/宽度；单独文本编码器供 action expert，不供视觉 backbone | 冻结 DINOv2 最终 block；offset `[1]` | 16 步 action chunk；未来单帧 16 步 ahead |

> 数据开放范围：论文称实现基于公开 FastWAM/ImageWAM 代码与公开 benchmark，代码和部分模型权重公开；正文未逐模型列许可证。本文不提供额外 URL。

**记忆机制**：JAM 没有外部记忆或 recurrent state。部署时的上下文是当前观测 `f0`（可多视角）与指令 `c`；动作 chunk 长度是 32 步（Wan）或 16 步（Ovis-U1/FLUX.2/Qwen3.5-2B/DINOv2），不是历史记忆窗口。未来 chunk 的预测区间也不能当作已观测历史上下文。

### 1.2 关键机制 (Key Mechanism)

- **三分离**：backbone 初始化、未来观测上的目标、推理是否保留生成机制，是三个独立选择。只比较 VLA 与 WAM 的成功率，无法判断差距来自哪一项。
- **未来只做 target**：记录的未来帧 `f1:T` 只进入冻结 future anchor，产生 `g_k`；当前特征 `h^(ℓ)` 经训练专用 prediction head 预测 `ĝ_k`。未来帧从不进入 action path。
- **训练后丢弃**：anchor 分支与 prediction head 在训练后删除；部署只留 backbone 与 action expert，不做 target extraction，也不做未来预测。
- **可缓存**：固定 anchor、预处理、时间索引与条件后，targets 可离线算一次并缓存；在线训练只处理当前帧。
- **offset 很关键**：Wan 从 offset `[0]` 到 `[1,2]`，LIBERO-Plus 从 44.62% 升到 55.58%；Ovis-U1 从 `[0]` 到 `[1]`，从 69.65% 升到 79.95%。作者解释：Wan 的 VAE latents 已含场景变化，Yak 的 endpoint 更依赖预训练层与当前观测/指令的关系。

⚡ **Eureka Moment**：控制策略不必生成未来；让当前观测特征去预测冻结未来表征，就足以把未来监督注入 action expert 读到的特征。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
训练:
  f0, s0, c ──► visual backbone ──► h^(ℓ) ──► action expert ──► a_1:H
                         │
                         └──► prediction head p ──► ĝ_k
  f1:T ──► frozen future anchor E_φ ──► g_k（stop-gradient）
  loss = L_act + λ_F · mean_k mean_n [1 - cos(ĝ_k,n, sg(g_k,n))]

部署:
  f0, s0, c ──► backbone ──► action expert ──► a_1:H
  （anchor 分支与 prediction head 已丢弃；未来帧不进入 action path）
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```text
L_JAM = L_act + λ_F · mean_{k∈K} E_n[1 - cos(ĝ_k,n, sg(g_k,n))]
```

先给目标：在固定策略架构下，用未来表征预测替代视觉生成损失，同时保持部署接口不变。

```text
L_VLA = L_act
L_WAM = L_act + λ_G · L_G
L_JAM = L_act + λ_F · L_F
L_F = (1/|K|) · Σ_{k∈K} E_n[1 - cos(ĝ_k,n, sg(g_k,n))]
```

| 符号 | 含义 |
|------|------|
| `f0, s0, c` | 当前观测、机器人状态、指令 |
| `a_1:H` | 动作 chunk |
| `f1:T` | 记录的未来观测，只作 target |
| `h^(ℓ)` | backbone 第 ℓ 层当前观测 token |
| `E_φ` | 冻结 future anchor |
| `g_k` | anchor 在 readout offset `k` 的特征 |
| `ĝ_k` | prediction head 对 offset `k` 的预测 |
| `K` | 被监督的 readout offsets 集合 |
| `n` | 空间 token 索引 |
| `sg` | stop-gradient |
| `λ_F` | foresight 总权重；每个 offset 分 `λ_F/|K|` |

直觉：`L_F` 是 cosine distance，不是像素重建；梯度到达 prediction head 与 policy backbone，但不穿过 anchor。`k=0` 是当前观测位置，`k≥1` 是记录的未来位置；offset 是 anchor 时间序列中的位置，不等于环境步数。Wan 每 4 个环境步采样帧，offset `[1]` 覆盖 4–16 步，`[2]` 覆盖 20–32 步；其他 backbone 的未来是 16 步 ahead 的单帧。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，只用于验算损失聚合方式，不是论文实验。

假设 `λ_F = 0.6`，`|K| = 2`，每个 offset 权重 0.3。offset 1 有两个 token，cosine similarity 为 0.8 与 0.6，平均 distance = `[(1-0.8)+(1-0.6)]/2 = 0.3`；offset 2 两个 token 都是 0.5，平均 distance = 0.5。则：

```text
L_F = (0.3 + 0.5) / 2 = 0.4
λ_F · L_F = 0.6 × 0.4 = 0.24
若假设 L_act = 0.50，则 L_JAM = 0.50 + 0.24 = 0.74
```

验算：若错误地把每个 offset 都乘 0.6，会得到 `0.6×0.3 + 0.6×0.5 = 0.48`，与论文“总权重固定、按 offset 均分”不一致。时间记账也需分开：若均匀采样 `K=5` 帧、间隔 `Δt=1` 秒，首末跨度是 `(5-1)×1 = 4` 秒；若从当前参考 `t` 预测 `t+1` 至 `t+H`，`H=5`，终点距当前是 `5×1 = 5` 秒。Wan 的帧位置为 0,4,…,32：当前帧单独编码为一个 latent position，后续每组 4 帧合并为一个 latent position；offset `[1]` 覆盖 4–16 步，`[2]` 覆盖 20–32 步，因此不能把 3 个 temporal VAE positions 的首末差直接当作 8 个环境步的逐帧预测。

## 4. 工程视角 (Engineering View)

| 对比 | 估计 online FLOPs / sample / update | 实测 GPU time / step | 降幅 |
|------|--------------------------------------|----------------------|------|
| FastWAM | 15.52 TFLOPs | 0.789 s | — |
| JAM (Wan), cached | 4.19 TFLOPs | 0.258 s | 73.0% FLOPs；67.3% GPU time |
| ImageWAM (Ovis-U1) | 5.81 TFLOPs | 0.473 s | — |
| JAM (Ovis-U1), cached | 3.67 TFLOPs | 0.294 s | 36.8% FLOPs；37.8% GPU time |

工程含义：节省来自“在线只编码当前帧 + 缓存固定 targets”，不是来自减少 action expert 采样。FLOPs 是估计值；GPU time 是在一块 GH200、batch 16 上对 profiled operations 的测量，不是端到端训练吞吐。Cache 复用要求 future anchor、预处理、时间索引与条件固定；换 anchor 或改输入分辨率就要重算。部署时 anchor 与 head 不运行，action expert 复用一次编码的上下文做 flow-matching 采样；论文没有给 JAM 自身真实控制频率，不能把训练 GPU time 当作推理延迟。我的条件性判断：这套配方最适合“未来帧可离线预处理、anchor 固定”的模仿学习流水线；在线 RL 或频繁换目标的场景，缓存收益会下降。

## 5. 数据与评测 (Data & Eval)

**预训练与初始化**：Wan 用 Wan2.2-5B 视频 backbone；Ovis-U1 用 Yak18 图像编辑 backbone；FLUX.2 用 Klein 4B；Qwen3.5-2B 是视觉-语言模型；DINOv2-B/L 是自监督图像编码器。JAM 不要求 backbone 来自生成器，但生成式 backbone 仍可使用。

**仿真评测**：

| Benchmark | 任务/评测频率 | 说明 |
|-----------|----------------|------|
| LIBERO | 40 tasks，每 task 50 rollouts | Spatial/Object/Goal/Long 四 suite，报告 suite 均值 |
| LIBERO-Plus | 固定 1,506 perturbed tasks，每 task 一次 rollout | 15% 子集：361 Spatial、378 Object、389 Goal、378 Long；本地策略共享 manifest |
| RoboTwin 2.0 | 50 bimanual tasks，每 task 每 setting 100 trials | Clean 与 Randomized；unseen instructions，SAPIEN 3.0.0b1 |

**训练预算**（区分 updates、epochs、batch）：

- LIBERO：FastWAM recipe 20k updates，global batch 128；ImageWAM recipe（其他 backbone 使用）21.7k updates，global batch 128。
- RoboTwin 2.0：FastWAM recipe 5 epochs，29,355 updates，global batch 1,024；ImageWAM recipe 5 epochs，69,900 updates，global batch 384。
- 独立训练：选定配置跑 3 次；LIBERO-Plus 均值 Wan `64.61 ± 0.37%`，Ovis-U1 `79.92 ± 0.43%`。统计用 100,000 次 task-bootstrap resamples。

**真实机器人设置**：论文明确只做仿真，无人类参与者或真实机器人验证；伦理声明把“仿真训练策略未物理验证”列为风险。因此所有“成功 rates”都是仿真数字，不应外推为真实部署成功率。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 证据（论文实测） | 条件 |
|------|------------------|------|
| 替代生成损失 | FastWAM LIBERO 97.00→JAM Wan 97.45；LIBERO-Plus 55.31→55.58；ImageWAM 92.90→JAM Ovis 96.75；65.27→79.95 | 固定架构、数据、优化器、预算、评测器 |
| 未来 offset 优于当前 offset | Wan `[0]` 44.62→`[1,2]` 55.58；Ovis `[0]` 69.65→`[1]` 79.95 | LIBERO-Plus；不是所有架构同一最佳 target |
| 跨 backbone 可用 | Ovis-U1 96.75/79.95；DINOv2-B 95.50/68.19；DINOv2-L 96.30/70.39；Qwen3.5-2B 96.55/81.14 | LIBERO/LIBERO-Plus；架构与 action 接口不同，不能隔离预训练效应 |
| 训练成本下降 | Wan cached 15.52→4.19 TFLOPs；Ovis cached 5.81→3.67 TFLOPs | 估计 online per sample per update；cache 需固定 targets |

**失败/边界**：

- 仅仿真：LIBERO、LIBERO-Plus、RoboTwin 2.0；无真实机器人。
- 不全面领先：Ovis-U1 的 robot-state 49.2% 低于 ImageWAM 54.8%；Wan 的 background 46.2% 低于 FastWAM 49.5%；Qwen3.5-2B 的 robot-state 61.4% 低于 action-only 68.0%，language 71.7% 低于 72.5%；DINOv2-L 的 language/light/background 也低于 action-only（80.7 vs 86.5；85.7 vs 87.0；77.4 vs 85.5）。
- 随机 backbone 会崩：random backbone + random anchor 84.40%，random backbone + pretrained anchor 71.15%，远低于 pretrained backbone 的 97.45% 左右；作者主张 pretrained backbone 仍重要。
- 最佳 target 依架构而变：Wan 最佳是冻结 VAE 64.54%；Ovis-U1 最佳是预训练 Yak 79.95%；不能把单一 target 写成通用最优。
- 跨 backbone 比较不是受控预训练实验：作者自己注明“architecture and action-interface differences prevent isolating pretraining effects”。

### 6.1 隐含假设 (Hidden Assumptions)

- 冻结 anchor 的表征空间对控制有用，且 cosine target 可跨架构比较。
- 记录的未来帧可离线获得，预处理与时间索引在训练中固定；否则 cache 失效。
- 仿真扰动类别能代表真实分布移变；论文没有真实机器人验证。
- action expert 的初始化、容量与匹配基线可比；跨 backbone 的差异不能只归因于预训练。
- 训练预算与单/三次 run 足以支撑结论；部分配置是单次比较，统计功效有限。

## 7. 与相关工作对比 (Comparison)

| 方法/系列 | 关注点 | 未来监督 | 推理保留 | 证据 |
|-----------|--------|----------|----------|------|
| OpenVLA / π0 / π0.5 | VLA action-only | 无未来损失 | backbone + action expert | LIBERO 76.5 / 94.1 / 96.9（published） |
| FastWAM | 视频 WAM，去测试时生成 | 视频 denoising 或 action-only | action 路径 | LIBERO 97.0；LIBERO-Plus 55.3；RoboTwin 91.2 |
| ImageWAM | 图像编辑 WAM | 未来图像生成或 action-only | action 路径 | LIBERO 92.9；LIBERO-Plus 65.3；RoboTwin 89.0 |
| JAM (Wan) | 用 foresight 替代生成 | 冻结未来表征 cosine | 仅 action 路径 | LIBERO 98.2；LIBERO-Plus 64.5；RoboTwin 91.4 |
| JAM (Ovis-U1) | 同上，图像编辑 backbone | 冻结 Yak 表征 | 仅 action 路径 | LIBERO 96.8；LIBERO-Plus 79.9；RoboTwin 89.9 |

与 FLARE、JEPA-WAM、VLA-JEPA 等的区别（作者主张）：JAM 用独立训练专用 head 监督已有视觉 backbone tokens，保持匹配基线的 action 计算；不使用预测未来特征作为 action 输入，也不在部署时做 latent rollout。我的条件性分析：这篇最强的证据是“固定架构内替换损失”的对照；最弱的环节是跨 backbone 归因与真实部署。

**面试 Tip**：被问“VLA 还是 WAM”，先答“先拆三件事：backbone 怎么初始化、未来观测上放什么损失、推理保不保留生成机制”；再说 JAM 的证据是 foresight 可替代生成损失，但仅仿真且跨 backbone 不隔离预训练。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：做 VLA/WAM 架构消融的研究者；想去掉训练时视觉生成、保留未来监督的工程师；评估训练成本与 cache 设计的工程团队。
- **建议章节路径**：先读 §3.1–§3.2（三选择与 foresight loss）→ 再看 §4.2 Table 1（固定架构替换）→ §4.3 Table 3（跨 backbone）→ §4.5（成本）。Related Work 可最后扫。
- **不值得精读的理由**：如果你只关心真实机器人部署、或已熟悉 JEPA/未来表征监督，读摘要与 Table 1/3 即可；论文没有真实机器人实验，也没有把跨 backbone 的预训练效应单独隔离。

---
[← Back to Theory](./README.md)

**关键引用**：本文数字与主张均来自 [论文](https://arxiv.org/abs/2610.37219) 正文；未使用候选截断摘要补写。