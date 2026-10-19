# REKI：把记忆注册进 VLA（REKI: Registering Knowledge in VLAs）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-19
>
> **论文**: REKI: Registering Knowledge in VLAs
> **链接**: https://arxiv.org/abs/2610.22178
> **核心定位**: REKI 用 2.4B 纯公开数据 VLA，把 action expert 里的 register token 变成可缓存的短时记忆，在 LIBERO-Plus、RoboTwin Clean2Rand、RoboCasa365 上冲到公开数据组的性能-规模 Pareto 前沿，并把编译后单 chunk 推理压到 21.8 ms。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文主张：2.4B、纯公开数据预训练的 REKI 在三项基准上达到性能与规模的 Pareto 前沿；LIBERO-Plus 得分最高，RoboTwin Clean2Rand 和 RoboCasa365 仍有更大模型领先。关键机制是 action expert 内的 register 短时记忆。 |
| 适合精读 | 做 VLA memory、公开数据预训练、低延迟部署的读者，重点看 §4.1–4.2、§6.2、§A.1。 |
| 可以跳过 | 只关心通用 VLM 或不碰机器人控制，读摘要与 §1.2 即可。 |
| 落地可行性 | 中高：论文声称开放全流程与权重，编译后 21.8 ms/chunk；但真实机器人只覆盖 5 个 UR5 任务，长时记忆未验证。 |
| 主要风险 | benchmark 成功率与真实部署仍有差距；baseline 数字来自已发表结果，未做等算力重跑。 |

💡 **X-Ray 开场**

REKI 要解决的是：VLA 只看当前帧，遇到两个看起来都像“该往对方走”的位姿时，会在 A、B 之间来回摆。它把每帧 VLM 上下文压成 32 个 register token，只把 K/V 留在 action expert 的 self-attention cache 里，让策略用极低开销看到最近约 2 秒。对 VLA 研究者，这意味着“短时记忆”可以不改 VLM、不串行训练、不引入额外模块，仍然带来 OOD 成功率和稳定性收益。

📍 **研究全景时间线**

```
2024  π0：VLM + action expert 成为主线；OpenVLA 为 VLM 直接输出动作，Octo 为开源通用策略（非 VLM + action expert 架构）
2025  π0.5 / SmolVLA / RDT-1B：开放权重继续推进；强 VLA 仍常依赖 in-house data（论文 Related work 主张）
2026  Qwen-RobotManip（in-house data）/ InternVLA-A1.5（视频生成监督、更大 expert）/ REKI（公开数据 + register 记忆 + 效率）
                         ↑ 本文：register memory + 公开数据 Pareto
```

> 仅列论文引用可确认的少量节点，不覆盖完整参考文献表。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 训练 vs 推理 |
|------|------|------|-----------|--------------|
| VLM encoder | 相机 views + text prompt | 最终层 hidden states E_t，长度 L_t × d | 每帧一次 | 训练：K 帧打包成 batch，一次并行前向；推理：当前帧一次 prefill |
| State pooling | 机器人状态 s_t | 状态嵌入 g_t | 每帧一次 | 与 flow time τ 无关，所有 Euler steps 共用 |
| Action expert | 噪声 chunk A_t^τ + registers M_t | 速度场 vθ | 每 chunk 10 Euler steps | 训练：8 个 flow draw 共享 encoder/register；推理：10 步 |
| Register memory | M_t 初始化 + E_t + 过去 M | M_t 的 K/V cache | 每帧；滚动保留 K-1 帧 | 训练：每个窗口独立；推理：cache 可跨窗口延续 |
| Action chunk | vθ 积分 | H × d_a 的动作块 | 1 s chunk；真实机器人 15 Hz、30 steps = 2 s | 训练监督窗口内每帧；推理每 chunk 一次 |

论文的模型拆分为：Qwen3.5-2B VLM + 10 个 block、宽度 768、12 heads、3072 MLP 的 action expert；每帧 32 个 register；预训练/主基准窗口为 K=3 帧、1 s stride，真机实验（Table 11）为 K=3 帧、2 s apart。表 4 把规模拆为 VLM 2.2B + expert 0.2B，摘要统称 2.4B。

### 1.2 关键机制 (Key Mechanism)

- **VLM 与 expert 解耦**：VLM 最终层 hidden states 作为 encoded context，由 expert 每个 block 的 cross-attention 读取；不做 VLM 与 noisy action 的联合 self-attention。
- **register 作为记忆载体**：每帧有 32 个可学习 register token，和 noisy action 一起进入 expert 流；register 先 cross-attend 当前帧 E_t，再 self-attend 过去帧的 registers；action token 读取当前与过去 registers，以及自身。
- **register 不读 action，也跳过 adaptive normalisation**：因此 register 只依赖观测，不依赖 flow time τ 或机器人状态，可在 10 个 Euler steps 中复用，并在帧间缓存 K/V。
- **训练仍并行**：一个训练样本是 K=3 帧窗口，按 1 s stride 取帧；每帧有自己的 action chunk 和损失，窗口一次 batched pass，无需 recurrent unroll。
- **推理只多读 cache**：第一个 Euler step 计算当前帧 registers，后续 9 步复用其 K/V；cache 是滚动窗口，最老一帧的 registers 被丢弃。
- **8 次 flow draw 降方差**：encoder 成本占主导，额外 draw 只多一次 expert pass；8 个噪声 chunk 共享 encoded context 和 registers。

⚡ **Eureka Moment**：记忆不必进 VLM，也不必做 recurrent state；把每帧上下文压成 action expert 里的一组 register，并只缓存它们的 K/V，就能让 self-attention 自带短时记忆，训练仍并行、推理几乎免费。

### 1.3 信息流/架构图 (Flow / Diagram)

```
camera views + prompt
        │
        ▼
┌──────────────────────────┐
│ VLM: Qwen3.5-2B          │
│ final hidden -> E_t      │
└──────────┬───────────────┘
           │ cross-attention (keys/values)
           ▼
┌──────────────────────────────────────────────┐
│ Action expert: 10 blocks, width 768          │
│                                              │
│  M_t (32 registers) ──┐                      │
│  A_t^τ (noisy chunk) ─┼─ self-attention ──► vθ│
│                       │  reads:              │
│                       │  [M_1;...;M_t; A_t^τ]│
│  registers read: E_t + past M_1:t-1          │
│  registers never read A_t^τ                  │
└──────────┬───────────────────────────────────┘
           │ 10 Euler steps
           ▼
      action chunk A_t
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**

```
记住过去 ≈ 在 action expert 的 self-attention 里多带 K-1 帧的 32 个 register 的 K/V；VLM 每帧只前向一次。
```

**目标**：在 flow matching 框架下，让速度场 vθ 依赖截至当前帧的上下文 c_1:t，而不是只看当前 c_t；但不改变 flow matching 的监督形式。

```text
A_t^τ = (1 - τ)·A_t + τ·ε,    ε ~ N(0, I)

ℓ_fm(θ; A_t, c_1:t) = E_{τ,ε}[ || vθ(A_t^τ, τ | c_1:t) - (ε - A_t) ||_2^2 ]

L_act(θ) = E_{(c_1:K, A_1:K) ~ D}[ (1/K) Σ_{t=1..K} ℓ_fm(θ; A_t, c_1:t) ]

M_t ← Attn(M_t; [M_1; ...; M_t])
A_t^τ ← Attn(A_t^τ; [M_1; ...; M_t; A_t^τ])
```

> 符号与本文/相关文档保持一致：t 为决策步/帧索引；K 为记忆窗口；n_r 为每帧 register 数；d_e 为 expert 宽度；H 为 action chunk 长度；τ 为 flow time；ε 为噪声；M_t 为第 t 帧 registers；Attn(Q; V) 表示以 Q 为 query、以 V 为 keys/values 的 attention。

| 符号 | 含义 | 本文取值/说明 |
|------|------|----------------|
| t | 决策步 / 帧索引 | 窗口内从 1 到 K |
| K | 记忆窗口 | 预训练/主基准：3 帧、1 s stride；真机：3 帧、2 s apart；推理保留 K-1 个过去帧 |
| n_r | 每帧 register 数 | 32 |
| d_e | action expert 宽度 | 768 |
| H | action chunk 长度 | 1 s × control freq；真实机器人 30 steps |
| τ | flow time | τ=0 干净动作，τ=1 噪声 |
| ε | 高斯噪声 | N(0, I) |
| M_t | 第 t 帧 registers | 由 E_t 与过去 registers 计算 |
| c_1:t | 截至 t 的上下文 | prompt、views、state；论文注明窗口内不含过去 states |

**直觉**：

1. flow matching 学的是从噪声到动作的常速度场；τ=0 是干净 chunk，τ=1 是噪声，推理从 τ=1 积分到 0，共 10 个 Euler steps。
2. 记忆不新增损失项，只改变 vθ 的条件：历史通过 attention support 进入速度场。
3. 因为 registers 不读 action、跳过 adaptive normalisation，它们与 τ 无关；所以当前帧 registers 只在第一个 Euler step 算一次，后 9 步复用。
4. 式 (6) 对窗口内每帧的 chunk 都监督；后续帧的损失会把梯度反传到前面帧的 registers，因此寄存器学会“为未来保留信息”。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设性验算，不是论文实验。

**先走一步 flow matching**：假设一个 1D chunk，干净动作 A=0，噪声 ε=1，τ=0.25。

```text
A^τ = (1-0.25)·0 + 0.25·1 = 0.25
目标速度 = ε - A = 1
若模型输出 vθ = 0.8
ℓ_fm = (0.8 - 1)^2 = 0.04
```

梯度会把 vθ 从 0.8 推向 1。记忆不改变这个损失，只把条件从 c_t 扩成 c_1:t；所以收益来自“条件更好”，不是“多了一个损失”。

**再算 register cache 的量级**：假设 K=3、n_r=32、d_e=768、10 个 expert block，K/V 以 bf16 存储。

```text
每帧每 block 的 register K/V 元素数 = 32 × 768 × 2 = 49,152
字节数 = 49,152 × 2 B = 98,304 B = 96 KiB
10 个 block = 960 KiB ≈ 0.94 MiB / 帧
K=3 保留 2 个过去帧 ≈ 1.88 MiB
```

论文 §A.1 报告“约 2 MiB/过去帧，K=3 约 4 MiB”，与上述 bf16 理论值不一致，可能含 fp32、对齐或额外缓存；工程估算应以论文实测为准，不要把两者混用。

**延迟验算**：论文 §A.1 给出 22.1 ms（无 registers）、22.6 ms（K=1）、22.8 ms（K=3）、24.6 ms（K=32）。

```text
K=3 相对 K=1 的增量 = (22.8 - 22.6) / 22.6 ≈ 0.885% < 1%
K=3 相对无 registers 的增量 = (22.8 - 22.1) / 22.1 ≈ 3.17%
```

这验证了论文“memory 相对 K=1 增加不到 1%”的说法；注意 §6.3 的 21.8 ms 是另一套编译协议，不要和 §A.1 的 22.x ms 混用。

## 4. 工程视角 (Engineering View)

| 项目 | 论文数据 | 工程含义 |
|------|----------|----------|
| 单 chunk 延迟 | 21.8 ms（H100，编译后，2×256² 图像，10 actions） | 远小于 1 s chunk；但论文未给实际重规划频率 |
| 延迟拆分 | VLM prefill 9.5 ms，Euler steps 8.1 ms，读 cache 0.1 ms | 记忆本身不是瓶颈，VLM 与 Euler 积分才是 |
| Cache 复用 | encoded context、其 K/V、当前 register K/V、过去 K-1 帧 register K/V | 第一个 Euler step 算 register，其余 9 步复用 |
| 显存 | policy 4.7 GiB；K=3 cache 约 4 MiB | 短时记忆的显存代价很小 |
| 训练成本 | 200k optimizer steps，32×H100，176 h，5,632 GPU-hours | 论文 Table 3；global batch 2,880 samples，Table 9 注明以 K=3 帧×窗口计是上界 |
| 微调成本 | LIBERO 41、RoboTwin 87/341、RoboCasa365 401、RoboCasa-GR1 199 GPU-hours | 真实机器人 8 GPU-hours；适合作为可复现基线 |
| 控制频率 | 真实机器人 15 Hz；chunk 30 steps = 2 s | π0.5 的 chunk 为 50 steps = 3.3 s，由预训练固定 |

**训练/推理差异**：

- 训练时每个窗口 K=3 帧独立处理，每帧一个损失；8 个 flow draw 共享 encoder 与 registers，encoder 只跑一次。
- 推理时当前帧只 encode 一次，10 个 Euler steps 更新 action tokens；register K/V 跨帧滚动缓存。
- 论文明确指出训练/推理不完全一致：训练时窗口内第一帧 registers 没有更早记忆，推理时可能带着更早帧的 cache；信息可超出 K 帧窗口延续。
- VLM prefill 与 expert 各编译成一个 CUDA graph；cache 读取也在图内。

**Trade-off**：K=3 在质量与延迟间取平衡；K=32 仍只 24.6 ms，但论文没有训练/评测 K=32 的质量，不能据此声称长记忆有效。

## 5. 数据与评测 (Data & Eval)

### 5.1 数据组成

| 类别 | 内容 | 关键处理 |
|------|------|----------|
| 机器人数据 | 公开 teleoperation：RoboSet、Galaxea、AgiBot World 2026、RoboCOIN、RoboMIND、RDT-1B、DROID、BridgeData V2、12 个 OXE 数据集（含 RT-1、BC-Z）、InternData-A1 仿真、GR00T-X-Embodiment-Sim、ABC 双臂 YAM | 保留各数据集原生控制约定；按 per-dataset 1%/99% 分位归一化到 [-1,1]；zero-pad + loss mask；旋转统一到 Euler |
| Egocentric human | EgoDex、EgoVerse、Xperience | 约 350k episodes、170M frames；32D 双臂目标（wrist 6D + 10 eigengrasp/手）；无 proprio，使用 learned null state；按 0.6/0.25/1 的 pace 重采样 |
| Vision-language | FineVision、LLaVA-OneVision-1.5 子集 + RoboPoint、RefSpatial、PixMo | 占训练样本 10%；预训练时 LM head 文本损失权重 0.1 |
| 采样权重 | ∝ F^0.5 · L^0.5 · V^0.35 | F 为帧数，L/V 为语言/视觉多样性；RoboCOIN 与 InternData-A1 再乘 0.5；单数据集最多见 4 次 |
| 数据清洗 | 通道选择、退化片段、突变帧、动作表示、符号修正 | 突变阈值取 0.05 raw units 与 4× pooled 95th percentile 的较大者；InternData-A1 因碰撞突变删除 6.3% episodes |

**增强与鲁棒性**：camera dropout 预训练 0.15、微调 0.2；state dropout 0.2；颜色抖动、随机裁剪、相机顺序打乱；RoboTwin/RoboCasa365/RoboCasa-GR1 微调时加 mirroring（概率 0.5，prompt 加 [MIRROR WORLD]）。论文实测：关掉 camera dropout 或图像增强，LIBERO-Plus 的 camera 维度分别掉 21.5 / 23.6 点；关掉 state dropout，robot 维度掉 8.1 点；backbone LR 从 0.3× 降到 0.1×，Overall 掉 3.3 点。mirroring 在 RoboCasa365 平均 +4.9，但在 LIBERO-Plus 因 camera 维度 -6.5 导致 Overall -0.6。

### 5.2 评测设置

| Benchmark | 任务/扰动 | 训练/评测协议 | 评测次数 |
|-----------|-----------|----------------|----------|
| LIBERO-Plus | LIBERO 40 任务，7 个扰动维度（camera、robot state、language、light、background、noise、layout） | 在 LIBERO 四套件微调，再测 LIBERO 与 LIBERO-Plus | Overall 覆盖 10,030 个扰动 episodes，按各维度 episode 数加权 |
| RoboTwin Clean2Rand | 50 个双臂 ALOHA 任务 | 只在 clean 场景微调，测 Easy、Backgd、Light、Clutter、Height、Hard 六条件 | 每任务 20 rollouts；Clean2Rand 为六条件平均 |
| RoboCasa365 | 50 个目标厨房任务：18 atomic、16 seen composite、16 unseen composite | 在 300 个预训练任务的人类演示上训练，测 50 个目标任务 | 每任务 50 rollouts |
| RoboCasa-GR1 | 24 个 Fourier GR-1 桌面任务，每任务 1000 个演示 | 微调后评测 | 每任务 50 rollouts |
| 真实机器人 | UR5 + WSG50，5 个任务（pick/place/stack/sweep） | 每任务 50 个训练 episodes（Meta Quest controller 采集）；随机初始物体位置与 distractors 施加于每任务 10 个测试 episodes | 每任务 10 个测试 episodes；15 Hz，chunk 30 steps |

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 论文实测/作者主张 | 条件与边界 |
|------|------------------|------------|
| 基准成功率 | LIBERO-Plus 90.4；RoboTwin Clean2Rand 65.8（RoboCasa365 38.4 是包含 16 个训练中未见 composite 任务在内的 Overall Avg；其 unseen composite 子分数为 14.1，不能概括为 in-distribution 总平均或非 OOD） | 论文称公开数据组第一；Clean2Rand 65.8 由“领先 ABot-M0 55.7 达 10.1 点”反推，并与表 5 六条件均值一致 |
| 记忆收益 | 论文 §6.2 消融口径：3→3 为 90.4 / 73.4 / 38.4；1→1 为 88.9 / 67.9 / 33.0；3→2 为 90.1 / 73.6 / 39.3 | 大部分增益来自第一个过去帧；这组数字不等同于表 5 的 in-distribution RoboTwin 85.9/84.6 |
| 抑制振荡 | 去掉过去帧，振荡 episode 从 4.1% 升到 14.8%；一个过去帧降到 4.6%；无记忆训练为 18.6%；记忆在无记忆振荡的 seeds 上救回 45% | RoboTwin Clean，50 任务、20 matched seeds；振荡判据在 16/50 任务上设定，另 34 任务确认效果 |
| 真实机器人 | 长时程任务比无记忆版高 12–17.5 点；比 π0.5 高 7–37.5 点；短时程任务无记忆不降（65 vs 65；57.5 vs 57.5） | 5 个任务、每任务 10 次测试；样本有限 |
| 推理速度 | 编译后 21.8 ms/chunk；比 X-VLA 25.5、π0.5 33.8、ABot-M0 46.1、InternVLA-A1.5 51.3 更快 | H100、2×256²、10 actions；ABot-M0.5 例外：用其 RoboCasa365 权重与自有 LIBERO 设置（输入 two 128×128 images），编译后 1.4 s/chunk |

**失败模式与限制**：

- **RoboTwin in-distribution 不强**：REKI 为 85.9/84.6，低于同规模的 InternVLA-A1.5 93.3/93.0；后者额外用视频生成模型监督、动作专家 24 层（REKI 10 层）且多阶段预训练。
- **长时记忆未验证**：主基准/预训练窗口为 K=3 帧、1 s stride，覆盖约 2 秒过去；真机实验（Table 11）为 K=3 帧、2 s apart，覆盖约 4 秒；论文明确说“依赖很久以前事件的任务”仍待测试。
- **记忆不是万灵药**：无记忆会振荡的 seeds 中，记忆只救回 45%；短时程真实任务没有增益。
- **训练/推理 cache 不一致**：训练窗口独立，推理时第一帧 registers 可能带更早记忆；论文承认这一差异，但未量化其影响。
- **真实机器人覆盖窄**：只有 UR5、5 个桌面任务、每任务 10 次测试；未验证移动、双臂、人形真实平台。
- **baseline 非等算力**：表 9 说明对比方法未报告 wall-clock 或 accelerator-hours，成功率不能按训练算力匹配。

### 6.1 隐含假设 (Hidden Assumptions)

1. **32 个 register 足以概括一帧**：论文没有测试更少/更多 register 对长时程的影响。
2. **VLM 最终层 hidden states 足够**：cross-attention 解耦依赖“最终层已含任务所需信息”的假设。
3. **register 不需要读 action 或 adaLN**：因此可跨 τ 复用；若某些任务需要“动作历史”而非“观测历史”，该假设可能不成立。
4. **推理 cache 跨窗口延续是安全的**：训练时窗口隔离，推理时信息可超出 K 帧；论文指出但未做受控消融。
5. **仿真 OOD 增益可迁移到真实**：真实实验只有 5 个任务，且长时程增益的样本量小。
6. **per-dataset 归一化 + prompt 字段足以处理异构 embodiment**：论文保留原生控制约定，依赖 prompt 与 state layout 投影告诉策略约定；跨 embodiment 的负迁移未系统量化。

## 7. 与相关工作对比 (Comparison)

### 7.1 记忆路线对比

| 路线 | 代表 | 记忆放在哪 | 主要代价 | REKI 的差异 |
|------|------|------------|----------|-------------|
| 过去帧进 VLM | MEM、π0.7、ContextVLA | VLM token 序列 | 随窗口增长，至少 vision encoder 变贵 | 不改 VLM，每帧只 encode 一次 |
| Recurrent state / fast weights | μVLA、RoboTTT | backbone 或 head 的状态 | 训练需逐步 unroll | 窗口一次 batched pass |
| 额外记忆模块 | MemoryVLA、HAMLET | 额外 bank/module | 额外模块与监督 | 复用 expert 已有 self-attention |
| Register cache | REKI | action expert 的 register K/V | 推理读 cache | K=3 滚动窗口，训练并行 |

### 7.2 数字对比

| 模型 | 参数/预训练 | 记忆 | LIBERO-Plus | Clean2Rand | RoboCasa365 | 编译后 chunk 延迟 |
|------|-------------|------|-------------|------------|-------------|-------------------|
| REKI | 2.2B VLM + 0.2B expert（2.4B），public | register K/V，K=3 | 90.4 | 65.8* | 38.4 | 21.8 ms |
| InternVLA-A1.5 | 2.2B VLM + 0.5B expert，public | 未列 | 84.8 | – | – | 51.3 ms |
| ABot-M0 | 4.4B + 0.2B，public | – | 80.5 | 55.7 | – | 46.1 ms |
| π0.5 | 2.9B + 0.4B，in-house | 无 | 85.7 | 63.8* | 16.9 | 33.8 ms |
| Qwen-RobotManip | 4.5B + 0.2B（表 6 记 4.7B），in-house | register 但 forward 后丢弃 | 89.0 | 68.5* | 35.9 | – |
| X-VLA | 0.9B，public | – | – | – | – | 25.5 ms |

> * 为按论文表 5 六条件算术平均的验算值；REKI 的 65.8 也可由正文“领先 ABot-M0 55.7 达 10.1 点”反推。– 表示论文未给该设置或未比较。Qwen-RobotManip 的 register 在每次 forward 后丢弃，REKI 保留其 K/V 作为记忆（§4.1）。

**关键读法**：

- REKI 在公开数据组内是 LIBERO-Plus 与 Clean2Rand 第一；但 in-house 的 Qwen-RobotManip 的 Clean2Rand 行（joint-space variant）在其六条件均值上更高（68.5 vs 65.8），说明“公开数据 Pareto”不等于“所有模型第一”。
- RoboCasa365 上 REKI 38.4，高于 RLDX-1 36.0 与 Qwen-RobotManip 35.9，但低于 Xiaomi-Robotics-1 57.4；后者是更大规模 in-house 数据。
- RoboCasa-GR1 上 REKI 58.6，与 RLDX-1 58.7、ABot-M0 58.3 基本持平；LIBERO 97.5，处于多数模型 97.6–99.4 的附近。
- 预训练收益：从 scratch 到预训练，Clean2Rand +13.8，LIBERO-Plus +4.3；在 clutter 与 Hard 条件下增益更大（30.3 / 22.4）。作者主张这体现公开数据预训练主要改善 OOD。

**面试 Tip**：被问 REKI 的关键，不要只答“2.4B 纯公开数据”；要答“register tokens in action expert + 只缓存 K/V + 训练窗口并行、每帧有监督 + 推理读 cache 不到 1% 开销”，再说它用这个机制换到 OOD 成功率和 21.8 ms 延迟。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA memory / 长时程操作的研究者：重点看 §4.1–4.2 与 §6.2，理解 register 如何进入 self-attention、如何缓存、如何影响振荡。
  - 要复现公开数据预训练 pipeline 的工程师：看 §5.1、§B.1–B.5，尤其是异构动作空间、egocentric 重采样、采样权重与数据清洗。
  - 做低延迟部署的人：看 §4.2、§6.3、§A.1，区分训练窗口、Euler steps、cache 与 CUDA graph 的边界。
- **建议章节路径**：先读 §4.1–4.2（机制）→ §6.2（记忆消融与振荡）→ §5.1/§6.1（数据与 benchmark 条件）→ §A.1（cache 成本）。若不做复现，可跳 Related work 与 Appendix 的大表。
- **不值得精读的理由**：如果你不做机器人学习，或只关心通用 VLM 架构，读摘要与 §1.2 即可；如果你已经熟悉 flow matching VLA 且只想知道“记忆有没有用”，看 §6.2 的 3→3 / 1→1 对比就够。

---

[← Back to Theory](./README.md)

关键引用：[REKI 论文](https://arxiv.org/abs/2610.22178)