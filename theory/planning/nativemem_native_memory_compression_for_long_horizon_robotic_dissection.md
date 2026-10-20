# NativeMEM：把长程视觉历史压进 VLA 原生 Token 空间 (Native Memory Compression for Long-Horizon Robotic Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-20
>
> **论文**: NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation
> **链接**: https://arxiv.org/abs/2607.06678
> **核心定位**: 不再外挂记忆模块或 VLM  planner，而是复用 VLA 自己的 vision encoder，把每个历史「帧-视角」压成 1 个 native token，让预训练 VLA 用原有 attention 直接读长程历史。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在 π0.5 骨架上，用「每帧-视角 1 token」的原生记忆压缩 + 两阶段训练，仿真平均成功率 84.0%、真实机器人 98.7%（论文 Table 1/2） |
| 适合精读 | 如果你在做长程操作、VLA 记忆增强、或想把历史帧塞进有限 context，重点看 §3.2 的压缩公式与 §4.4 的消融 |
| 可以跳过 | 如果你只关心从头预训练 VLA 或非机器人视频理解，这篇距离中等 |
| 落地可行性 | 中：架构改动小、推理可解耦，但依赖 π0.5 骨干，且所给文本未声明代码/权重开放 |
| 主要风险 | 单 token 压缩可能丢失细粒度信息；Press Button 任务需读数字并按指定次数执行，Table 8 的 5% 使用外部数字识别，作者将失败解释为记忆增强无法补足底层 VLA 缺失的能力，识别与可靠执行请求次数仍是挑战 |

💡 **X-Ray 开场**

这篇论文问的是：预训练 VLA 大多只看当前帧，但长程操作需要「之前发生过什么」。现有做法要么外挂 VLM 把记忆写成文本，要么新加记忆模块，前者贵且丢细节，后者与预训练 token 空间不对齐。NativeMEM 的答案是：别加新模块，直接把历史帧压进 VLA 自己的视觉 token 空间，每帧-视角只留 1 个 token。对 VLA 研究者来说，这意味着「记忆」可以不是新架构，而是一种 token 级的压缩与对齐问题。

📍 **研究全景时间线**

```
2022 RT-1 ── 2023 RT-2 ── 2024 OpenVLA / π0 ── 2025 π0.5 / HAMLET / MemoryVLA / MemER ── 2026 RMBench / MEM / NativeMEM(本文)
```

只列来源可确认的节点；本文位置在「VLA 记忆增强」分支的末端，直接对标 HAMLET、MEM-short、MemER、Mem-0。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块/方法 | 记忆载体 | 记忆注入方式 | 是否改 VLA 架构 | 训练方式 | 本文是否实测 |
|---|---|---|---|---|---|
| **NativeMEM（本文）** | 每帧-视角 1 个 native memory token | 追加到原 token 序列，走原 attention | 否（复用 vision encoder） | Stage 1 冻结 VLA 训 tokenizer；Stage 2 全量微调 | 是 |
| HAMLET* | moment tokens + memory module | 新增历史模块 | 是 | 在 π0.5 上复现 | 是（复现） |
| MEM-short* | video encoder 压缩多帧 | 时序 attention | 是 | 在 π0.5 上复现 | 是（复现） |
| MemER | VLM 检索 keyframes → 文本 | 文本指令 | 外挂 VLM | 原论文 | 是 |
| Mem-0 | VLM planner + subtask classifier + keyframe memory | 分层规划 | 外挂模块 | 原论文 | 是 |
| MemoryVLA | perceptual-cognitive memory bank | 外部记忆库 | 是 | 原论文 | 否（仅 related work） |
| ReMem-VLA | recurrent memory queries | 新增 recurrent 模块 | 是 | 原论文 | 否（仅 related work） |
| MAP-VLA | demonstration memory library | 任务阶段 soft prompt | 是 | 原论文 | 否（仅 related work） |

> 注意：HAMLET* 与 MEM-short* 是作者在相同 π0.5 骨干上复现的版本，不能把它们的属性合并给 NativeMEM；MemoryVLA、ReMem-VLA、MAP-VLA 只在 related work 出现，未参与本文实验。

### 1.2 关键机制

- **M=1 的极限压缩**：每个历史「帧-视角」只产 1 个 token，把记忆长度从「帧数 × 每帧 patch 数」压到「帧-视角对数量」。
- **原生 token 空间对齐**：memory token 由 VLA 自己的 vision encoder 派生，经线性投影映射回 VLA token 维度，追加到原输入序列，不引入外部 planner 或新初始化记忆模块。
- **动作监督而非重建监督**：Stage 1 冻结 VLA，只用原 action loss 训记忆分支，逼 token 保留「对动作有用」的信息，而不是「看起来像原图」的信息。
- **训练/推理解耦**：Stage 2 离线缓存 memory token；部署时 tokenizer 可按指定频率更新记忆队列，VLA 只在需要动作时消费最新队列。

⚡ **Eureka Moment**：记忆不是「另一个模块」，而是「VLA 自己视觉 token 空间里的一种压缩 token」——压到每帧 1 个，再用原 attention 读。

### 1.3 信息流/架构图

```
历史帧-视角 o_τ^v
      │
      ▼
E_mem（从 VLA vision encoder 初始化）
  [q_mem, P_τ^v] → q̂_τ^v
      │
      ▼
W_mem 线性投影 → m_τ^v（1 个 token）
      │
      ▼
M_t = [b_mem, {m_τ^v}]  ← 记忆序列
      │
      ▼
x_t = [x_t^obs, x^prompt(ℓ,s_t), M_t]
      │
      ▼
冻结/微调的 VLA → action
```

## 2. 数学核心

📌 **Napkin Formula**：

```
|M_t| = V · T · N · M    （M=1 时，每帧-视角只留 1 token）
```

**目标**：在部分可观测环境中，把单帧 VLA 变成带历史条件策略：

```
a_t ~ π_θ'(· | o_t, s_t, ℓ, M_t)
```

**记忆长度公式（论文式 4）**：

```
|M_t| = V · T · N · M
```

| 符号 | 含义 | 本文设置 |
|---|---|---|
| V | 相机视角数 | 实验用 3 视角（Table 6 的 5×3×256 可反推） |
| T | 历史时间跨度（秒） | 论文称支持 minute-level |
| N | 每秒更新帧数 | 推理每 10 控制步更新；30 Hz 下等效 3 Hz |
| M | 每帧-视角 token 数 | 本文核心：M=1 |

**记忆 token 生成（式 5–6）**：

```
[q̂_τ^v, P̂_τ^v] = E_mem([q_mem, P_τ^v])
m_τ^v = W_mem · q̂_τ^v
```

**记忆序列与输入拼接（式 7–8）**：

```
M_t = [b_mem, {m_τ^v | τ ∈ I_t, τ ≤ t, v=1..V}]
x_t = [x_t^obs, x^prompt(ℓ,s_t), M_t]
```

**Stage 1 目标（式 9）**：θ 冻结，只优化 φ_mem、q_mem、W_mem、b_mem：

```
min  E[ L_act( π_θ(· | o_t, s_t, ℓ, M_t), a_t ) ]
```

**Stage 2 目标（式 10）**：tokenizer 冻结，优化 θ'、W_mem、b_mem：

```
min  E[ L_act( π_θ'(· | o_t, s_t, ℓ, M_t), a_t ) ]
```

**直觉**：式 4 说明记忆成本随 T、N、V 线性涨；M 是唯一能被压到 1 的乘数。压 M 而不压 T、N，才能同时保「长跨度」和「高频细粒度」。但 M=1 是信息瓶颈，所以必须用 action loss 监督，让这 1 个 token 尽量只装动作相关信息。

> TODO：论文未给出 E_mem 的具体层数、q_mem 维度、W_mem 维度；所给文本不足以复现这些超参。

## 3. 带数字走一遍：玩具例子

> 以下为**假设性验算**，用于说明式 4 的量级，不是论文实测。

假设 V=3 个相机、T=60 秒历史、N=30 Hz（每个控制步都更新）、M=1：

```
|M_t| = 3 × 60 × 30 × 1 = 5,400 tokens
```

若不做压缩，每帧每视角保留 256 个 visual tokens（M=256）：

```
|M_t| = 3 × 60 × 30 × 256 = 1,382,400 tokens
```

压缩比 = 256×。这就是「分钟级历史」能塞进 context 的算术来源。

再把更新频率降到论文推理设置「每 10 控制步更新一次」：30 Hz 下等效 N=3 Hz。

```
|M_t| = 3 × 60 × 3 × 1 = 540 tokens
```

**历史跨度验算**：若记忆队列保留 K 个采样，采样间隔 Δt=10 控制步 = 1/3 秒（30 Hz），则跨度 = (K−1)×Δt。

- K=160 → (159)×(1/3) ≈ 53 秒
- K=200 → (199)×(1/3) ≈ 66.3 秒

这与论文「160+ 帧、minute-level」的定性描述自洽，但**K 的具体值、以及是否所有实验都用同一 K，论文没有统一交代**，不能把某一设置的 Δt 写成所有实验共有的记忆窗口。

**Stage 1 采样方式**：论文说每个训练样本取回合首帧 + 8 个近期帧，近期帧按 8~16 步的间隔采样（对应轨迹中的时间/控制步），用于保留初始状态线索并限制图像 I/O 与内存开销。首帧与近期窗口不是等间隔序列，不能据此把 9 个采样点换算成统一的时间跨度。

## 4. 工程视角

| 维度 | 论文给出的数字/设置 | 工程含义 |
|---|---|---|
| 推理延迟 | 100 ms 约束下约 200 帧；32 GB 预算下最多 5,000 帧 | 记忆队列长度是延迟与内存的联合约束，不是固定值 |
| 控制频率 | 30 Hz | 每个控制步 33.3 ms；每 10 控制步更新记忆 ≈ 333 ms 一次 |
| 动作 horizon | 50-step action horizon | 这是 action chunk 长度，不是记忆窗口；不要与 Δt 混淆 |
| 对比 Native History | K=4 时 5×3×256=3,840 visual tokens；K=8 时 6,912 tokens | 全分辨率历史很快撞训练容量，NativeMEM 的压缩比是工程前提 |
| Stage 1 成本 | 50k steps，batch 16，lr 5e-5，约 16 小时 4×H100 | 只训记忆分支，不训 VLA |
| Stage 2 成本 | 20k steps，batch 32，约 5 小时 2×H100 | 缓存 token 后微调，避免重复跑 tokenizer |
| 存储 | 每帧-视角 1 个 token-level 特征 | 论文称预处理存储/I/O 开销可忽略 |

**部署边界**：tokenizer 可与 VLA 推理解耦，记忆更新不阻塞动作查询；但单 token 压缩意味着「读不回原图」，对需要精细空间/文字信息的任务，工程上要预期信息损失。

## 5. 数据与评测

**预训练 / Stage 1（不是从头预训练 VLA）**

- 在冻结的 π0.5 上训练 memory tokenizer。
- 数据：约 4,000 条轨迹，混合仿真与真实、标准操作与 memory-demanding 任务。
- 50k steps，batch 16，lr 5e-5，4×H100 约 16 小时。
- 监督信号：VLA 原 action prediction loss，无重建目标。

**仿真评测**

- 任务：3 个 RMBench 任务（Swap Blocks、Put Back Block、Observe and Pickup）+ 2 个按钮任务（Click Buttons、Click Buttons Hard）。
- Stage 2 微调：50 episodes；20k steps，batch 32。
- 评测：每任务 100 trials（论文 §4.1）。
- 结果（Table 1）：NativeMEM 平均 84.0%；Mem-0 32.4%；π0.5 8.8%；HAMLET* 9.0%；MEM-short* 12.8%。

**真实机器人**

- 任务：Click Buttons、Put Back Block、Grocery Checkout Scanning（unseen）。
- Stage 2 微调：100 episodes；20k steps，batch 32。
- 控制：30 Hz，50-step action horizon，每 10 控制步更新记忆。
- 评测：每任务 50 trials。
- 结果（Table 2）：NativeMEM 平均 98.7%；MEM-short* 30.7%；π0.5+RTC 34.7%；π0.5 14.7%；Mem-0 0.0%。

**其他评测**

- LIBERO（Table 3）：NativeMEM128 平均 93.9，NativeMEM512 平均 94.0，π0.5 93.4；复用已训 tokenizer，不重跑 Stage 1，batch 32、20k steps。
- 鲁棒性（Table 5）：三个 RMBench 任务，clean 79.3% → unseen 背景/光照 67.7%，无额外微调。
- 数据效率（Fig. 4b）：10 demos 时平均 60%，作者称约为 Mem-0 的 3.5×；25/50 demos 也优于 Mem-0。
- 更难 grocery（Table 7）：3/5/7 件，Once 100/88/68%，Retry 100/90/74%。

> 注意：π0.5 基线在同协议下使用相同目标任务示教，是「无记忆的任务微调基线」，不是零样本「预训练即用」。

## 6. 能力与失败模式

**能做**

- 长程召回：Put Back Block 需要记住方块原来的垫子位置，仿真 100%、真实两阶段均 100%。
- 连续进度跟踪：Click Buttons 需要不重复按已按过的按钮，仿真 94%、真实三阶段 100/98/96%。
- 未见任务迁移：Grocery Checkout Scanning 未参与 Stage 1，真实三阶段 100/100/100%。
- 动态人类插入新物体：定性实验显示能识别未扫描物体并只扫新物体（Appendix C.4）。
- 失败恢复的定性行为：视频中抓取失败后重试，仍能区分「尝试次数」与「成功扫描数」。

**不能做 / 明显短板**

- Press Button：需要读数字并按指定次数按按钮，带外部数字识别也只有 5%，easy 版 32%（Table 8）。作者归因于底层 VLA 本身缺乏数字识别/可靠计数能力，记忆增强不能凭空补能力。
- 细粒度视觉细节：单 token 压缩在需要精确空间/文字信息的任务上风险高；论文用 action loss 缓解，但没有给出信息论下界。
- 长程组合任务：7 件 grocery Once 68%，说明进度跟踪随任务长度下降。
- 泛化边界：只在桌面操作、π0.5 骨干、固定相机配置下评测；未验证移动、双臂、人形或跨 embodiment。

### 6.1 隐含假设

1. **假设 action loss 足以让单 token 保留动作关键信息**：论文用 Stage 1 消融支持这一点（w/o Stage1 平均 58.0%），但没有与其他压缩目标做同等预算的对比。
2. **假设冻结 VLA 时 memory token 能被原 attention 读懂**：依赖 tokenizer 从同一 vision encoder 初始化；若换骨干，这个对齐假设需重新验证。
3. **假设首帧 + 近期窗口足以覆盖训练时的记忆需求**：Stage 1 只采样首帧和 8 个近期帧，长程中间段是否被充分监督，论文未消融。
4. **假设仿真与真实数据混合训练不会互相拖累**：论文未给分开训练的对照。
5. **假设 30 Hz / 50-step action horizon / 每 10 控制步更新这一套设置可外推**：这些是实验设置，不是所有机器人平台的通用常数。

## 7. 与相关工作对比

| 方法 | 记忆位置 | 压缩粒度 | 是否需新模块 | 与预训练 VLA 对齐方式 | 主要局限 |
|---|---|---|---|---|---|
| **NativeMEM** | VLA 内部 token 序列 | 1 token / 帧-视角 | 否（复用 vision encoder） | action loss 监督 + 原生 token 空间 | 单 token 瓶颈；依赖 π0.5 |
| HAMLET* | 新增 moment tokens + memory module | 每时间步 token | 是 | time-contrastive + memory module | 架构改动，复现版平均 9.0% |
| MEM-short* | video encoder + 时序 attention | 多帧压缩表示 | 是 | 在 π0.5 上复现 | 真实数据少时不稳定（30.7%） |
| MemER | 外部 VLM 检索 keyframes | 文本/语义 | 是（VLM） | 文本指令 | 需标注，细节易丢 |
| Mem-0 | VLM planner + keyframe memory | 分层子任务 | 是（planner） | 子任务分类器 | 仿真 32.4%，真实 0.0% |
| Native History | 直接拼全分辨率历史帧 | 每帧 256 tokens | 否 | 全量微调 | K=4 已 3,840 tokens，K=8 超容量 |

**面试 Tip**：被问「NativeMEM 和 MemoryVLA/HAMLET 有什么区别」时，一句话答——「它不引入外部记忆库或新模块，而是把记忆压成 VLA 自己 vision encoder 产出的 1 个 native token，用原 attention 读；代价是单 token 信息瓶颈；但 Press Button 的低成功不应归因于该瓶颈：Table 8 使用外部数字识别，C.5 将 5% 归因于底层 VLA 本身缺乏数字识别与可靠执行按压次数的能力，记忆增强不能补足。」

## 8. 精读建议

- **值得精读原文的人**：
  - 做长程机器人操作、正在设计 VLA 记忆机制的研究者：重点看 §3.2 的式 4–8 和 §4.4 的消融。
  - 要把历史帧塞进有限 context 的工程师：重点看 §4 的延迟/内存数字与 Stage 2 缓存流程。
  - 评估「原生 token 对齐 vs 外挂模块」路线的人：重点看 Table 1/2 与 Appendix C.2 的 Native History 对比。
- **建议章节路径**：先读 §3.2（压缩与对齐）→ 再看 §4.1 的设置与 Table 1/2 → 然后 §4.4 消融 → 可跳 §2 related work 的引用表（除非要查基线归属）。
- **不值得精读的理由**：如果你不做机器人学习、或已经熟悉「压缩历史 token + 冻结骨干对齐」这一类方法，读摘要和 Table 1/2 即可；论文的数学细节没有超出式 4–10 的范围。

---

[← Back to Theory](./README.md)

**关键引用**：论文全文 https://arxiv.org/abs/2607.06678