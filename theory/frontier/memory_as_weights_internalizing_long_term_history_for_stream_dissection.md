# 把历史写进权重：流式视频的长期记忆内化 (Memory as Weights: Internalizing Long-Term History for Streaming Videos)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-18
>
> **论文**: Memory as Weights: Internalizing Long-Term History for Streaming Videos
> **链接**: https://arxiv.org/abs/2610.22070
> **核心定位**: 它把流式 VLM 的长期历史从“压缩/检索/缓存后塞回上下文”改成“生成一组随时间演化的 LoRA 权重”，让历史进入参数空间、当前观测继续作为输入，从而缓解历史记忆与当前感知争抢上下文的问题。

> 证据标记：`[实测]` 指论文表格或正文给出的数值；`[主张]` 指作者的解释、摘要 claim 或设计判断；`[分析]` 指本文编辑基于论文条件的推演，不是论文实验结论。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 历史不必以 token 形式留在上下文里；可以用 recurrent HyperNetwork 把历史逐步写成固定大小 LoRA，让基座 VLM 只处理当前观测。`[主张]` |
| 适合精读 | 如果你在做 streaming VLM、长视频记忆、参数高效适配，重点看 §3.1–§3.2 和 §4.1；如果你做 VLA 记忆，重点看 §4.2 与 Appendix C.2。 |
| 可以跳过 | 如果你只关心纯视觉 backbone 预训练，或只做离线长视频理解，这篇距离中等。 |
| 落地可行性 | 中：LoRA 部署友好，但需要额外 HyperNetwork、状态缓存、流式推理管线和数据构造流程；VLA 只在仿真 WidowX 上验证。 |
| 主要风险 | 固定低秩记忆可能丢失细粒度历史；评测用 parallel-forward 加速而非全程 cached-forward；VLA 仅单一仿真平台、96 个 episode，未报告方差。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？流式 VLM 必须持续记住过去，但把历史压缩、检索或缓存后重新放回输入，会让历史与当前观测争夺有限上下文。它发现了什么？可以把历史直接内化成轻量 LoRA 权重，当前帧仍然作为输入，历史则变成模型参数的偏移。对 VLA 研究者意味着什么？记忆不一定要放在 token 或 KV cache 里，也可以放在参数空间；但“固定大小低秩记忆”是否足以支撑高频控制、长程细节和失败恢复，仍需外部复现与真实机器人验证。`[分析]`

📍 **研究全景时间线**

年份取自论文参考文献；未额外考证。只列少量与本文机制直接相关的节点。

```
2017  HyperNetworks（Ha et al.）
  → 2022  LoRA（Hu et al.）
  → 2024  DeltaNet（Yang et al.）
  → 2025  OVO-Bench（Niu et al.）
  → 2026  StreamingBench / StreamArena
  → 2026  MaLoW（本文）← 当前位置
```

本文的位置：把 HyperNetwork、LoRA、DeltaNet 组合成“历史→权重”的流式记忆框架。局限：默认以 2 秒 clip 为更新单位，未解决 agentic 回访、工具调用和细粒度计数等场景。`[分析]`

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| 当前观测编码 | 当前 2 秒视频 clip + 时间戳文本 | 视觉 token V_t、文本 token T_t | 每个时间步/每个 query | 训练和推理都只把当前观测送入基座 VLM；Qwen3-VL-4B 例外使用最近 8 秒窗口。 |
| 历史压缩 q | 前一 clip 的 V_t、T_t、可学习 query Q | 紧凑表示 q_{t-1} | 每 2 秒 clip 一次 | 用 4 层 self-attention aggregator，隐藏维 512、8 头。 |
| SSM 状态 m | q_{t-1}、m_{t-1} | m_t、o_t | 每 2 秒更新，默认 | 4 层 DeltaNet，隐藏维 512；训练用并行 causal 形式，推理可用 cached-forward。 |
| 全局语义 s | q_{t-1}、s_{t-1}、余弦门 β | s_t | 每 2 秒更新，默认 | 门控求和；β=0.05，来自训练集 64,426 个相邻块转变的中位余弦距离 0.0542。 |
| LoRA 解码 | o_t、s_t | Δθ_t（LoRA 权重） | 每个时间步生成，供下一步使用 | 线性读出 + RMSNorm + 可学习门控 α；每个目标模块、每个 rank 单独输出头。 |
| 基座 VLM | 当前观测 C_t、用户查询 I_t、Δθ_t | 回答 R_t 或动作 | 每个 query/控制步 | 基座 θ 冻结；只训练 HyperNetwork 与 LoRA 生成器。 |
| 损失 | 下一个 token 预测、LoRA 幅度正则 | 训练信号 | 每个优化 step | L = L_ntp + γ·L_LoRA；主实验 γ=0.01。 |

### 1.2 关键机制 (Key Mechanism)

- **记忆与输入分离**：历史不进入当前上下文，而是生成 Δθ_t；当前观测仍是输入。这样当前帧的注意力容量不被历史 token 挤占。`[主张]`
- **递归 HyperNetwork**：不是每步重算全历史，而是用单步更新 f 递归累积：h_t, Δθ_t = f(h_{t-1}, C_{t-1})。`[主张]`
- **双状态设计**：m_t 用 DeltaNet 建模局部时序依赖；s_t 用门控求和保留更长程全局语义。消融显示两者组合最好：w/ SSM + sum 平均 68.5，gated sum 平均 68.8。`[实测]`
- **固定大小 LoRA 记忆**：主实验 LoRA rank 8；rank 16 在消融中降到平均 65.8，作者判断为过拟合。`[实测]`
- **冻结基座**：基座架构和参数不变，只注入生成的 LoRA；主实验可训练参数占 1.4%–4.0%。`[实测]`

⚡ **Eureka Moment**：不要把历史塞回上下文，而是让历史生成一组随时间演化的低秩权重；当前帧仍是输入，历史成为参数偏移。`[主张]`

### 1.3 信息流/架构图 (Flow / Diagram)

```
C_{t-1}: 2 秒视频 clip + 时间戳文本
        │
        ▼
[V_{t-1}; T_{t-1}; Q] ──► aggregator g ──► q_{t-1}
        │
        ▼
h_t = [m_t; s_t]
  m_t, o_t = SSM(q_{t-1}, m_{t-1})        # DeltaNet，局部时序
  s_t = s_{t-1} + β_{t-1}·q_{t-1}        # 余弦门控，全局语义
        │
        ▼
readout h(o_t, s_t) ──► Δθ_t = LoRA 权重
        │
        ▼
冻结基座 M_{θ+Δθ_t}(I_t, C_t) ──► R_t
```

注意时序：Δθ_t 由截至 t-1 的历史生成，在 t 时刻应用；当前观测 C_t 仍显式输入。`[实测/论文公式]`

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
历史 → Δθ_t；当前观测 → 输入；R_t = M_{θ+Δθ_t}(I_t, C_t)
```

**目标**：在不保留无限增长上下文的情况下，让 R_t 依赖 C_{0:t-1}。

```
R_t = M_θ(I_t, C_{0:t-1}, C_t)            # 原始流式形式
Δθ_t = Φ(C_{0:t-1}; h_0)
R_t = M_{θ+Δθ_t}(I_t, C_t)                # 内化历史
```

**递归单步更新**：

```
h_t, Δθ_t = f(h_{t-1}, C_{t-1})
```

**历史压缩与双状态**：

```
q_{t-1} = g([V_{t-1}; T_{t-1}; Q])
m_t, o_t = SSM(q_{t-1}, m_{t-1})
s_t = s_{t-1} + β_{t-1}·q_{t-1}
β_{t-1} = I[ sim(V̄_{t-1}, V̄_{t-2}) ≤ β ]
Δθ_t = h(o_t, s_t)
```

**训练损失**：

```
L = L_ntp + γ·L_LoRA
L_ntp = -E[ Σ_k log p_{θ+Δθ_t}(y_{t,k} | I_t, C_t, y_{t,<k}) ]
L_LoRA = Σ_{s∈S} ( mean|A_s| + mean|B_s| ) / |S|
```

> 符号与本文保持一致：C_t 当前观测，C_{0:t-1} 历史，I_t 用户查询，R_t 回答/决策，θ 冻结基座参数，Δθ_t 生成的 LoRA，Φ 递归 HyperNetwork，h_t 隐藏状态，m_t 局部 SSM 状态，s_t 全局语义，q_t 压缩表示，β 余弦门阈值，γ 正则权重，S LoRA 槽位集合。

**直觉**：m_t 像“短期时序缓存”，s_t 像“长程语义摘要”，两者读出一个 LoRA 偏移；基座 VLM 仍用当前观测做推理。`[分析]`

**Appendix A 的理论解释**：作者把历史上下文视为线性读操作，说明历史诱导的特征变换可以用低秩更新近似。`[主张]`

```
O_H(q) = M_H·q,  M_H = Σ_i v_i·k_iᵀ
P_H = W_O·M_H·W_Q
M_H ≈ B_H·A_H,  rank ≤ r
ΔW_g = W_g·P_H,  ΔW_u = W_u·P_H
误差 ≤ L_F·||W_O||_2·σ_{r+1}(M_H)·||W_Q·x||_2
```

作者明确说明：这不是说 MaLoW 精确重建 M_H，也不等价于 full-context attention；它只证明“历史信息在参数空间中用低秩更新表示”在假设下可行。`[主张]`

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设数值，仅用于验算机制，不是论文实验。

**例 A：余弦门控求和。** 设 β=0.05，s_{t-1}=(1,1)，q_{t-1}=(2,3)。

```
情形 1：V_{t-1}=(1,0)，V_{t-2}=(1,0)
sim = 1 > 0.05 → β_{t-1}=0
s_t = (1,1) + 0·(2,3) = (1,1)

情形 2：V_{t-1}=(0,1)，V_{t-2}=(1,0)
sim = 0 ≤ 0.05 → β_{t-1}=1
s_t = (1,1) + 1·(2,3) = (3,4)
```

验算：相似度高表示视觉变化小，门控关闭；相似度低表示变化大，门控打开并写入 q。`[分析]`

**例 B：秩 1 LoRA 精确实现一个历史变换。** 设 W_Q=W_O=I，历史矩阵：

```
M_H = [[2, 0],
       [0, 0]]
```

取 B_H=(1,0)ᵀ，A_H=(2,0)，则 B_H·A_H = M_H，秩为 1。令 W_g=I，P_H=W_O·M_H·W_Q=M_H，则：

```
ΔW_g = W_g·P_H = M_H
x = (1,1)
(I+P_H)x = (3,1)
(W_g+ΔW_g)x = (1,1) + (2,0) = (3,1)
```

验算：若 MLP 的 gate 预激活从 x 变为 (I+P_H)x，那么把 ΔW_g 加到 W_g 上可精确复现同一预激活。这个例子只验证“低秩更新可表示历史诱导的输入变化”，不验证 MaLoW 实际学到了什么。`[分析]`

## 4. 工程视角 (Engineering View)

**训练与部署条件**：

| 项目 | 主实验 | VLA 实验 |
|------|--------|----------|
| 优化步数 | 2,800 steps | 25,000 steps |
| 总 batch | 64 | 64 |
| 学习率 | 峰值 8×10⁻⁵，cosine，warm-up 0.1 | 1×10⁻⁵，cosine，1,500 warm-up |
| 硬件 | 4×H200 | 4×H200 |
| 输入频率 | 1 fps；每个 C_t 为 2 秒 clip | Bridge 5 Hz，控制周期 0.2 秒；两帧一块 |
| 可训练参数 | 主实验 1.4%–4.0% | MaLoW + action decoder；VLM backbone 冻结 |
| LoRA | rank 8；MLP + v_proj，部分模型加 qkv_proj | rank 8；Qwen3.5 混合注意力层同样注入 |

**延迟实测**：论文在 100 个确定性采样的 OVO-Bench 问题上测 streaming latency；视频时长 8–1,117 秒，中位 153.5 秒。`[实测]`

| 延迟项 | 数值 | 说明 |
|--------|------|------|
| 总延迟 | 152.89 ms | MaLoW (Qwen3.5-9B) |
| 基座 prefill + decode 首 token | 130.07 ms | 占 85.07% |
| 编码新帧并更新 h_t | 12.11 ms | 可与基座推理并行 |
| 生成 LoRA 张量 A_{t+1}、B_{t+1} | 3.27 ms | 可并行 |
| 合并 LoRA 生成 Δθ_{t+1} | 5.69 ms | 可并行 |
| 应用 Δθ_t | 1.13 ms | 关键路径 |
| wrapper 开销 | 0.62 ms | 关键路径 |
| 记忆相关总开销 | 约 22.82 ms | 小于总延迟 15% |
| 流水线后关键路径新增 | 1.75 ms | 约基座 130.07 ms 的 1.3% |

**工程含义**：固定大小 LoRA 让记忆占用不随历史长度线性增长；但每 2 秒要跑一次 aggregator、SSM 和 LoRA 生成器，控制频率高时要关注这些模块的吞吐。论文的 1.75 ms 是“可并行 + 流水线”后的关键路径，不是所有部署都能自动获得。`[分析]`

**内存与参数权衡**：主实验 197.6M–321.1M 可训练参数；Qwen3.5-0.8B 的 MaLoW 可训练参数占比 10.8%，明显高于大模型。小模型上相对开销更大。`[实测]`

## 5. 数据与评测 (Data & Eval)

**训练数据 MaLoW-Instruct**：381K QA、56K 视频、55,867 个唯一视频、14 个底层数据集。`[实测]`

| 维度 | 数值/条件 |
|------|-----------|
| 问题时间类型 | real-time 46.1%、past 36.4%、future 17.4% |
| 问题类型 | multiple-choice 35.0%、open-ended 34.7%、yes/no 30.0%、counting 14.0% |
| 过去监督 | 约 36% QA 的 query timestamp 在事件结束后，直接监督历史保留 |
| 视频来源 | CosMo-Cap、YouCook2、Charades、ActivityNet 等 14 个数据集 |
| 生成/验证 | Qwen3.5-122B-A10B 生成问题与答案；Qwen3.5-397B-A17B 验证 |
| 防泄漏 | 过滤 OVO-Bench 重叠视频；StreamingBench/StreamArena 与训练源不重叠 |

> 注意：Table 7 的四类问题占比之和超过 100%，counting 很可能与前三类存在重叠或统计口径不同；引用时保留原表数字，不自行归一化。`[分析]`

**评测设置**：

| 基准/任务 | 规模/条件 | 指标 | 注意 |
|-----------|-----------|------|------|
| OVO-Bench Backward | 631 题；EPM、ASI、HLD | accuracy，宏观平均 | EPM 最适合考察记忆；HLD 的 ground truth 全是 “Unable to answer”，区分度较低。 |
| OVO-Bench Real-Time | 837 题 | accuracy，宏观平均 | 考察当前感知；历史不应拖累实时性。 |
| StreamingBench Real-Time | 2,500 道多选题 | accuracy | 论文 Table 1 的 StreamingBench (R)。 |
| StreamArena HR | 877 题、237 个视频；L1–L4 | 平均分 | 证据-查询间隔中位 12.1 分钟；L4 超过 30 分钟。 |
| SimplerEnv-Bridge WidowX | 4 个任务 × 24 episodes = 96 | 成功率 | 仿真；Stack Green Block、Put Carrot、Put Spoon、Put Eggplant。 |

**关键实测结果**：

- OVO-Bench 总平均：MaLoW (MOSS-VL-Real-11B) 74.5，为表中最高；MaLoW (Qwen3.5-9B) 71.3；MaLoW (Qwen3-VL-4B) 69.3；MaLoW (VideoChat3-4B) 66.6。`[实测]`
- StreamingBench Real-Time：MaLoW (Qwen3-VL-4B) 84.8，为表中最高；MaLoW (Qwen3.5-9B) 82.3；MaLoW (MOSS) 81.0。`[实测]`
- Backward split：MaLoW (Qwen3-VL-4B) 63.8，比 StreamTTT-4B 的 59.9 高 3.9 个百分点；MaLoW (Qwen3.5-9B) 66.4，比 OASIS-8B 的 57.2 高 9.2 个百分点、比 AURA-8B 的 60.4 高 6.0 个百分点。`[实测]`
- EPM：MaLoW (Qwen3.5-9B) 63.6，为表中最高；其基座 Qwen3.5-9B 为 61.6。`[实测]`
- StreamArena HR：MaLoW 22.8，VST 20.6，AURA 21.4，MOSS 18.3，VideoChat3 18.0；相对 VST +2.2、相对 MOSS +4.5。`[实测]`
- VLA：Q3.5-2B-MLP 平均 59.4，MaLoW (Q3.5-2B-MLP) 68.8；按表内数值为 +9.4 个百分点。论文摘要写 9.3%，正文 Table 2 数值差为 9.4，存在 0.1 的口径差异。`[实测/口径差异]`

**评测条件与 caveat**：

- 论文对 MOSS-VL、VideoChat3、Qwen3.5-9B 做了复评；Table 1 灰色数字是原论文报告，与复评可能不同。例如 VideoChat3 的 StreamingBench 原报告 83.0，复评为 79.0，MaLoW (VideoChat3) 为 81.2。`[实测]`
- MaLoW (Qwen3-VL-4B) 使用最近 8 秒作为当前观测，其他 MaLoW 多为 2 秒；这可能影响横向比较。`[实测/条件]`
- MaLoW 训练时可用并行 causal DeltaNet；评测为加速使用 parallel-forward，把 [0,T-2] 当历史、当前窗口取 [T-2,T]。论文称其与 cached-forward 因果等价，但主表不是全程在线逐块推理。`[主张/条件]`
- 在 OVO-Bench real-time split 上，MaLoW (Qwen3-VL-4B) 和 MaLoW (Qwen3.5-9B) 分别低于 StreamTTT-4B 和 AURA-8B；作者推测基座可能经过额外微调，而 MaLoW 保持基座完整。`[主张]`
- 对基座 MOSS 的 backward 平均，MaLoW 为 71.7，基座 MOSS 为 72.5，略低；但 EPM 62.0 vs 60.6、real-time 77.4 vs 75.2 有提升。`[实测]`

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**：

- 长程回看：StreamArena HR 中，MaLoW 22.8 平均分，优于 VST 和 MOSS；定性样例显示可回忆 16、19、29 分钟前的细节。`[实测/主张]`
- 当前感知不被历史明显拖累：论文按历史长度分组，MaLoW 在 OVO-Bench real-time 和加长版 StreamingBench 上保持稳定；作者归因于 LoRA 幅度正则。`[实测/主张]`
- 跨基座迁移：0.8B–11B、Qwen3.5、Gemma4、VideoChat3、MOSS 上均有整体增益。`[实测]`
- VLA 初步迁移：仿真 WidowX 上平均成功率 59.4→68.8，Stack Green Block 20.8→29.2、Put Spoon 70.8→87.5、Put Eggplant 70.8→83.3；Put Carrot 75.0→75.0。`[实测]`

**不能做什么/失败模式**：

- 记忆可能变“模糊”：定性样例中，32 分钟前的 Coca-Cola 被记成 “beverage”，说明固定低秩记忆保留的是粗粒度语义，不是精确细节。`[实测/主张]`
- 需要 agentic 回访的任务仍失败：样例需要回到对应帧并定位画面中的时钟；MaLoW 未用 agentic tool-use 数据训练。`[实测/主张]`
- 细粒度计数困难：样例需要先识别桌上有 4 张卡、再推理新增 1 张，作者认为依赖更强视觉感知。`[主张]`
- VLA 只在 SimplerEnv 仿真 WidowX 上测 4 个任务、96 个 episode；没有真实机器人、没有多机器人平台、没有方差/置信区间。`[实测/缺失]`
- 高频控制下的稳定性未验证：Bridge 是 5 Hz、0.2 秒控制周期；论文没有报告更低延迟或更高频率下的表现。`[分析]`
- 低秩容量有上限：rank 16 消融反而下降，说明容量与过拟合/干扰之间存在权衡；更大规模、更长历史的 scaling 行为未给出。`[实测/分析]`

### 6.1 隐含假设 (Hidden Assumptions)

1. **低秩足够**：作者假设对未来有用的历史效应可以压进固定 rank 的 LoRA；Appendix A 只证明可行性，不证明实际学习到的 Δθ_t 覆盖了所有必要历史。`[分析]`
2. **2 秒窗口足够实时**：默认当前观测是 2 秒 clip；Qwen3-VL-4B 用 8 秒更好，说明窗口大小与基座有关，不是普适常数。`[实测/分析]`
3. **冻结基座不干扰实时能力**：LoRA 正则限制幅度，但论文没有给出不同 γ 对实时感知的完整 trade-off。`[分析]`
4. **parallel-forward 与 streaming 等价**：论文称训练/评测的并行形式与 cached-forward 因果等价；但主表为加速使用并行前向，真实在线部署的抖动、状态复位、并发 query 未充分展示。`[主张/分析]`
5. **数据验证足够**：MaLoW-Instruct 由 LLM 生成与验证，论文未说明人工复核比例；计数题占比之和超过 100% 也提示统计口径需谨慎。`[分析]`
6. **VLA 迁移是初步的**：只在 Qwen3.5-2B-MLP 一个基座、一个仿真环境、四个任务上验证；不能外推到移动、双臂、人形或真实接触-rich 操作。`[分析]`

## 7. 与相关工作对比 (Comparison)

| 方法/类别 | 记忆放在哪 | 更新方式 | 当前观测 | 训练/推理特点 | 适用场景 |
|-----------|------------|----------|----------|----------------|----------|
| Full-context streaming VLM（VideoChat3、MOSS 等） | 上下文/KV 状态 | 追加历史 | 与历史一起输入 | 历史越长，上下文/显存开销越大 | 短中视频、历史全保留 |
| Local-window（AURA、SimpleStream 等） | 最近窗口 | 滑窗丢弃旧帧 | 只留最近观测 | 实时性强，但长程记忆弱 | 实时感知优先 |
| 文本/视觉压缩记忆（VST、Ours 对比中的 memory 方法） | 压缩 token/文本摘要 | 摘要、聚类、检索 | 当前观测 + 压缩记忆 | 记忆有损，依赖压缩策略 | 长视频问答 |
| StreamTTT | 单独 TTT 分支的 fast weights | 在线测试时训练 | 最近视觉 token | 记忆在独立分支，不在 VLM 本体 | 流式长程记忆 |
| **MaLoW（本文）** | **VLM 本体的 LoRA 权重** | **recurrent HyperNetwork 逐步生成 Δθ_t** | **仅当前观测 + query** | **基座冻结；固定大小 LoRA；可并行生成** | **流式视频 QA、VLA 记忆增强** |

**面试 Tip**：被问到时先讲范式差异——“history as tokens” vs “history as weights”；再讲代价——固定大小 LoRA、每 2 秒更新、生成可并行；最后补条件——主表用 parallel-forward 加速、VLA 只在仿真 WidowX 上验证。这样既抓住核心，也不夸大泛化。`[分析]`

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 streaming VLM / 长视频记忆的研究者：重点看 §3.1–§3.2、§4.1、Appendix A。
  - 做 VLA 记忆设计的工程师：重点看 §4.2、Appendix C.2，关注 5 Hz 控制、两帧块、动作历史编码和 episode 复位。
  - 做参数高效适配/ HyperNetwork 的人：重点看 LoRA 解码、rank 消融、LoRA 注入位置消融。
- **建议章节路径**：先读 §3.1–§3.2 搞清“历史→权重”的信息流 → 再看 §4.1 的 Table 1 和 §4.2 的 VLA 表 → 有余力再看 Appendix A 的理论解释和 Appendix D 的消融。可跳 §2 的相关工作长列表，除非你要写综述。
- **不值得精读的理由**：如果你不做流式视频、不做 VLA 记忆、也不关心 LoRA/HyperNetwork，读摘要和 §1 即可；如果你已经熟悉“参数即记忆”的思路，重点只需看它的工程条件和失败模式。

---

关键引用：[arXiv:2610.22070](https://arxiv.org/abs/2610.22070)

[← Back to Theory](./README.md)