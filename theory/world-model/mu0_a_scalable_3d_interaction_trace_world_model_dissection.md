# μ0：可扩展的 3D 交互痕迹世界模型（$\mu_0$: A Scalable 3D Interaction-Trace World Model）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-22
>
> **论文**：μ0: A Scalable 3D Interaction-Trace World Model
> **链接**: https://arxiv.org/abs/2606.13769
> **核心定位**：把“世界模型该预测什么”从稠密像素或本体相关动作，换成**一组语义交互关键点的 3D 轨迹（trace）**，从而用无动作标签视频做预训练，再冻结复用给下游 action expert。

> **来源版本说明**：本篇笔记的**唯一证据**是 arXiv:2606.13769v3 的 HTML 全文正文（含附录与表格，页面标注 arXiv:2606.13769v3 [cs.RO] 20 Oct 2026，CC BY 4.0）。本次候选记录中的 `abstract_verification` 为空，即**没有独立核验的摘要记录**；因此文中一切数字、主张均取自该版本正文/表格/附录，不用候选摘要补写或替代。文末不引用论文之外的代码仓库或页面。

---

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 仅用无动作视频预训练的 3D trace 世界模型，冻结后接 action expert，在 RoboCasa365 八任务上平均成功率 30.25%，高于 π0 的 25.25%，低于 π0.5 的 42%（论文 Table 2） |
| 適合精讀 | 如果你在做「无动作标签视频 → 机器人策略」的迁移、或想给 VLA 加一个可复用的 motion prior，重点看 §2（TraceExtract）、§3.3（flow matching）、附录 B.4（action expert 接口） |
| 可以跳過 | 如果你只关心端到端 VLA 架构或大规模动作标签预训练，这篇的增益主要体现在“预训练信号换掉”，架构层改动有限 |
| 落地可行性 | 中：数据引擎需 VGGT + TAPIP3D + VLM 字幕三段流水线，论文自报 ~2.4–3.3 frames/s 单卡吞吐、13M 帧语料约 1.2K 单卡小时 |
| 主要風險 | trace 质量完全继承感知栈误差（作者自述局限）；不含力/触觉/接触模式；只在桌面操作 + 少量本体上验证 |

💡 **X-Ray 开场**

这篇论文问的是：如果我们能大规模拿到人类视频，但拿不到与机器人本体绑定的动作标签，世界模型应该预测什么才有用？作者的回答是：不要预测像素，也不要预测动作，而是预测**物体、工具、手、接触区域这些语义关键点在 3D 空间里的未来轨迹**——这是一个与本体无关、但保留度量和几何的运动接口。对 VLA 研究者的意义在于：它把“预训练信号”和“动作标签”解耦，并给出了一个可验证的下游接口（冻结 trace 模型 + 可训练 action expert）。

📍 **研究全景时间线**

```
2024  Track2Act（任意点轨迹做模仿）→ 2025 3DFlowAction（3D flow world model，arXiv 2506.06199）
      → 2026 TraceGen（CVPR 2026，固定网格 3D trace + episode 级字幕，推理需 depth）
      → 2026 本文 μ0（语义关键点 + 全局对齐 3D + 事件级字幕 + 查询式 B-spline 预测）
      ← 当前位置：把 trace 从“辅助线索”提升为“可冻结复用的世界模型输出”
```

（时间线只保留来源中明确给出年份/会议的点；更早的 Savitzky–Golay 等工具性引用不列入。）

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 训练/推理差异 | 归属说明 |
|------|------|------|----------------|----------|
| TraceExtract（数据引擎） | 异构人类/机器人操作视频 | `{I_t, l_c, Q_t, T_ref}` 四元组 | 仅离线运行；论文报 ~2.4–3.3 frames/s 单卡、全语料 ~1.2K 单卡小时 | 仅服务 μ0 预训练，不是策略的一部分 |
| VLM backbone | RGB（512×512）、语言、（可选）depth 渲染图 | 场景/语言 KV cache | 冻结；取 SmolVLM2-2.2B 前 20 层 text decoder | 属 μ0；不要与 π0/π0.5 的 3.45B backbone 混为一谈 |
| Trace Expert | 每个关键点的 history token + noisy control-point token + 局部 DINO 特征 | B-spline 控制点速度场 | 20 层、隐宽 0.5×VLM，每 2 层与 VLM KV 做 cross-attention | 属 μ0，预训练后冻结 |
| Depth 通路 | Turbo colormap 渲染的 metric depth | 与 RGB 共享深层 SigLIP 的 token | 独立可训练 patch stem；训练时 0.7 概率丢 depth | 属 μ0 可选输入，非必需 |
| Action Expert | 单步部分去噪的 trace 特征 + 夹爪相机 + proprioception + 语言 | 连续 action chunk | 仅此部分在目标域训练 | 属下游策略；μ0 本身不输出动作 |

### 1.2 关键机制 (Key Mechanism)

- **测哪里比测什么更重要**：论文认为固定网格浪费 query 在背景，均匀采样漏掉工具尖端/接触面。TraceExtract 先用 DINOv2 patch 特征聚成 entity 簇，按可见 patch 覆盖率分配每个 entity 的关键点配额（小实体有保底），再在 mask 内做 farthest-point sampling。
- **全局对齐解决相机运动**：先用 VGGT 跑一遍稀疏 anchor 帧建立全局坐标系，再分 chunk 稠密重建，每个 chunk 用 SE(3) 对齐到**同一组全局 anchor**（而不是对齐前一个 chunk），因此误差不累积。
- **跨 chunk 身份延续**：用上一 chunk 最后已知的 3D 世界坐标，作为下一 chunk 首帧的 3D query 交给 TAPIP3D，因此同一个物理点在长视频里不会“每段换一个 ID”。
- **事件级语言**：用平滑后的逐帧 trace 加速度找动作锚点（峰值）与边界（谷值 `b_i = argmin_t ã_t`），把长演示切成 reach/grasp/move/release 级别的片段再让 VLM 打字幕，解决 episode 级字幕对不上局部子目标的问题。
- **查询等变 + 平滑先验**：每个关键点是一个可交换 query，共享同一套参数；未来被参数化为三次 B-spline 的 10 个控制点，天然抑制 tracker 抖动、压缩输出维度。
- **不确定性的处理**：同一指令下未来多解，确定性回归会“平均掉”接触丰富运动，因此改用 conditional flow matching；再加 validity head（轨迹何时终止）与 rigidity loss（同一 DINO 簇内关键点间距应保持）。

⚡ **Eureka Moment**：世界模型的输出不必是“可看的画面”或“可执行的动作”，而可以是**一小撮语义点在 3D 中的未来轨迹**——它足够紧凑以省容量，足够度量以支撑操作，又足够抽象以跨本体复用。

### 1.3 信息流/架构图 (Flow / Diagram)

```
[异构视频]
   │  DINOv2 聚类选点 → VGGT 全局稀疏 + 分块稠密 + SE(3) 对齐 → TAPIP3D 递进跟踪
   ▼
{I_t, l_c, Q_t, T_ref^{t-h:t+H}}     h=8, H=32（附录 B.2）
   │
   ├─► VLM prefix（SmolVLM2-2.2B，冻结，20 层）
   │        └─ KV cache ──cross-attn(每2层)──┐
   ├─► 每关键点 token（history / noisy controls + Fourier(uv) + DINO 局部特征）
   │        └─► Trace Expert（20 层）─► 速度场 v_θ
   ▼
4-step Euler 去噪 → B-spline 解码 → 3D 预测轨迹
   │  （推理时只跑 1 步，取中间隐状态 z_trace）
   ▼
gated cross-attention 注入最后一层 VLM 特征 → Action Expert（flow matching）→ action chunk
```

---

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
预测目标 = 语义关键点集合的 B-spline 控制点；训练目标 = 把噪声沿直线路径推到控制点
```

**目标**：给定观测、语言、可选历史，预测每个 query 关键点未来 H 步的 3D 轨迹；再用该预测的中间特征驱动动作。

**(a) 全局对齐（附录 A.3，式 1）**

```
A^(c) = argmin_{A ∈ SE(3)}  Σ_{t ∈ S∩c} || A·E_t^(c) − E_t^sparse ||²
```

- `E_t^(c)`：第 c 个 chunk 内第 t 帧的 chunk-local 外参
- `E_t^sparse`：全局稀疏 pass 给出的 anchor 帧外参
- `S`：anchor 帧集合；`A^(c)`：把 chunk 局部位姿搬到全局的刚体变换
- 直觉：每个 chunk 都直接对齐到**同一组全局锚点**，所以各 chunk 误差独立有界，不会像链式对齐那样逐段累积。

**(b) B-spline 目标拟合（附录 B.2，式 4）**

```
P*_n = argmin_P  || M_n ⊙ (B·P − [0; T̃¹_n]) ||_F²  +  λ_bsp² || Γ·P ||_F²
```

- `B ∈ R^{(H+1)×D}`：均匀网格上采样的三次 B-spline 基，曲线在 t=0 处钉在 anchor
- `D = 10` 控制点，`H = 32` 未来步；`M_n` 把无效步置零（不产生梯度）
- `Γ`：相邻控制点一阶差分算子，`λ_bsp = 0.2`；拟合后逐元素 clip 到 |P*| ≤ 1.5
- 有效性门槛：某关键点至少要 `D` 个有效未来步才进入 flow loss

**(c) Flow matching 与总损失（附录 B.3，式 6/7/9/10）**

```
P^τ = τ·ε + (1−τ)·P*,   ε ~ N(0, I),  τ ∈ [0,1]
L_flow = E[ || v_θ(P^τ, τ, F_cond) − (ε − P*) ||² ]
L_done = (1/N) Σ_n Σ_t BCE(d̂_{n,t}, y_{n,t})
L_rig  = E[ (1/|R|) Σ_{(n,n')∈R} Var_d( ||P̂_{n,d} − P̂_{n',d}||² ) ]
L = L_flow + λ_done·L_done + λ_rig·L_rig
```

- `τ` 经 adaLN-Zero 注入每层（shift/scale/gate），末层线性层零初始化，因此初始时每个 block 是恒等映射
- `P̂_n = P_n^τ − τ·v_θ`：在训练途中重建的干净控制点，用于 rigidity
- `R`：共享同一 DINO 簇身份的关键点对集合；`d` 索引控制点
- 推理：`τ: 1 → 0`，4 步 Euler；`T̂¹ = B·P̂`（去掉 anchor 行）

**(d) 动作侧（附录 B.4，式 11/12）**

```
z_guided = z + σ(g) · CA(Q=LN(z), K=V=LN(W_proj·z_trace))
L_action = E || v_φ(a^τ, τ, z_guided, c) − (a − ε_a) ||²
```

- `z`：最后一层 VLM 特征；`g`：所有 head 与空间位置共享的标量门，初始化为 0 经 sigmoid 限制在 (0,1)
- `z_trace`：只跑**一步** 4-step Euler 后 Trace Expert 的中间隐状态（不是完整 rollout）
- `c`：proprioception、夹爪相机（DINOv2 编码）、语言
- 直觉：g 初始化为 0，经 sigmoid 后初始 σ(g)=0.5，即约一半强度注入，作者称其为较弱的运动注入状态；论文实测学到的门控收敛到 σ(g) ≈ 0.48–0.50

> 符号与本文一致：`h`/`H` 为历史/未来步数，`D` 为控制点数，`N` 为每样本关键点数（训练时从 [1,256] 均匀采样）。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，仅用于验算流程自洽性，不是论文实验数据。

**(1) 时序跨度验算。** 设 Δt = 0.1 s（假设 10 Hz，论文未对 trace 预测明确给出 Δt）：

```
历史跨度 = (h − 1)·Δt = (8 − 1)·0.1 = 0.7 s
未来跨度 = (H − 1)·Δt = (32 − 1)·0.1 = 3.1 s
总覆盖   = 0.7 + 3.1 = 3.8 s = 38·Δt
```

注意：这是**单次 trace 预测设置**的跨度，不能把它写成“所有实验共有的记忆窗口”——下游 action chunk 完全不同（仿真 action horizon 16 / execution horizon 8；真机 50 / 25，见附录 D.5、D.8）。

**(2) Flow matching 4 步 Euler 验算。** 假设训练完美，`v_θ ≡ ε − P*` 为常数速度，从 `P^1 = ε` 出发，步长 0.25 向 τ=0 积分：

```
P^0.75 = ε − 0.25(ε − P*) = 0.75ε + 0.25P*
P^0.50 = 0.50ε + 0.50P*
P^0.25 = 0.25ε + 0.75P*
P^0.00 = 0·ε + 1·P* = P*   ✓ 精确回收干净控制点
```

这说明 4 步在“速度恒定”的理想情形下无损；实际误差来自 `v_θ` 不是常速度场。

**(3) B-spline 压缩率。** 每关键点原本要回归 32×3 = 96 个数，拟合后只需 10×3 = 30 个控制点，压缩到 31.25%；代价是拟合误差，论文报 B-spline 拟合与原始 trace 的 DTW 差约 0.02（附录 D.2）。

**(4) 门控数值。** 取 σ(g) = 0.49，则 `z_guided = z + 0.49·CA(...)`，即 motion 特征大约以一半强度注入；论文报告固定为 0.5 时成功率 26.0%、固定 0.25 时 25.0%，学出来的 0.48–0.50 对应 30.25%（附录 E.2、Table 10）。

---

## 4. 工程视角 (Engineering View)

| 指标 | 数值（论文） | 工程含义 |
|------|--------------|----------|
| 单图 trace 预测延迟 | 0.29 s | 比次快的 2D 基线 Track2Act（0.85 s）快 2.9×；比 TraceGen（1.20 s）快约 4× |
| 端到端 action chunk 延迟 | 195.9–211.2 ms（均值 200.8 ms） | 对比 π0.5 的 171.1–178.9 ms（均值 173.0 ms），约 +16%，增量来自额外的 trace 去噪阶段 |
| 推理期参数量 | μ0 + AE 3.04B（2.59B 冻结 + 457.0M 可训练：436.4M AE + 20.6M mixer） | 少于 π0 的 4.03B 与 π0.5 的 4.14B（Table 5） |
| 预训练成本 | 2×10⁵ steps；有效 batch 24；`N` 从 [1,256] 均匀采样 | 论文写“two GPUs (6 per GPU)”，与有效 batch 24 存在口径歧义（可能含梯度累积），复现时需按实测对齐 |
| 数据引擎成本 | ~2.4–3.3 frames/s 单卡；13M 帧约 1.2K 单卡小时 | 3D tracking 占总耗时 45–58%，是主要瓶颈；逐视频独立，可并行摊薄 |
| 去噪步数 | 4 步 Euler；策略侧只用 1 步取中间特征 | 单步是刻意的成本-信息折中：完整解码 trace 反而降到 29.5%（Table 10） |

部署上要注意两点：一是 trace 模型是**冻结**的，换机器人只需重训 action expert，这是它相对端到端 VLA 的主要工程卖点；二是 +16% 延迟对高控制频率任务是否可接受，论文没有给控制频率，只给了 chunk 级延迟，实际部署需自己按 execution horizon 折算。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练数据（论文 Table 3）

| 数据集 | 模态 | Episodes | Frames | 存储 | License |
|--------|------|----------|--------|------|---------|
| EgoDex | Human | 51.8K | 4.1M | 4 TB | CC-BY-NC-ND-4.0 |
| EgoVerse | Human | 3.4K | 2.7M | 2.7 TB | CC BY-SA 4.0 |
| Agibot | Robot | 123K | 3.7M | 8 TB | CC BY-NC-SA 4.0 |
| DROID | Robot | 25.4K | 2.5M | 1 TB | CC-BY-4.0 |
| 合计 | — | 203.6K | 13.0M | 15.7 TB | — |

约 52% 帧来自人类视频、48% 来自机器人视频；按跟踪采样率（人类 5 fps、机器人 10 fps）折算约 16 天人类视频 + 7 天机器人视频 ≈ 23 天。**这是跟踪采样率，不是模型的时间步长 Δt**，两者不可混用。作者自报 trace 数据规模约为 TraceGen 的 8×。

### 5.2 三类评测严格分开

- **Trace 预测（离线）**：129 个 slot、每 slot 16 个关键点，来源为 Human(30)/Mobile ALOHA(30)/Franka(30)/UR3(30)/VIMA-Bench(9)；按 source/environment 级别从预训练语料留出。指标 top-1/top-5 ADE、FDE、DTW（附录另报 FD），horizon T ∈ {8,16,32}，只在 moving points 上算，且用原始 GT trace 而非 B-spline 拟合目标。
- **仿真（RoboCasa365）**：8 个原子任务，每任务 100 条演示（共 800），PandaOmron 移动操作臂，两路 256×256 RGB + 16 维 proprio + 语言，12 维动作空间；action horizon 16 / execution horizon 8，batch 32，lr 1e-4，warmup 1000，训练 5×10⁴ steps；**每任务 50 rollouts** 评测。
- **真机（UR3）**：固定基座 UR3 + 两指夹爪，两路 224×224 RGB，7 维状态（6D 末端位姿 + 夹爪）；三个任务分别采集 90/80/50 条演示，action horizon 50 / execution horizon 25，lr 5e-5，训练步数 8000/6000/6000；**每任务 20 rollouts**。

注意单位口径：50 rollouts/task 与 20 rollouts/task 是**评测重复次数**；5×10⁴ 与 6–8×10³ 是**训练步数**；2×10⁵ 是**预训练步数**。三者不可互相替代。

### 5.3 关键结果

| 场景 | μ0 + AE | 对照 | 差值 |
|------|---------|------|------|
| RoboCasa365 平均（8 任务） | 30.25% | Diffusion Policy 22.75% / π0 25.25% / π0.5 42% / TraceGen+AE 23% | +7.5 / +5.0 / −11.75 / +7.25 个百分点 |
| 真机三任务平均 | 91.7% | π0 ≈ 71.7% / π0.5 ≈ 80.0%（由“+20.0 / +11.7 个百分点”反推） | +20.0 / +11.7 个百分点 |
| 真机 vs 同架构去 trace expert | 91.7% | VLM + action expert | +18.4 个百分点 |
| 长时程（3 任务 7 子任务，50 rollouts/task） | 7 子任务中 5 个 ≥ π0 | 多数子任务低于 π0.5 | 三个方法在 Organize Condiments 末子任务均为 0% |

Trace 预测侧（Table 1，2D 与 3D 两栏）：μ0 在 **top-5** ADE/FDE/DTW 的全部 horizon 上最优（2D top5-ADE T=8 为 0.124）；但在 **top-1** 上并非全面领先——2D 的 top1-ADE 在 T=8/16/32 分别为 0.202/0.279/0.315，弱于 Gemini 3.1 Pro 的 0.190/0.274/0.305。3D 栏中 μ0 在全部 ADE/FDE/DTW 上最优（T=8 top1-ADE 0.209 vs TraceGen 0.327）。表中 μ0 的 2D 行与 3D 行数值并不相同（例如 top1-ADE：2D 为 0.202/0.279/0.315，3D 为 0.209/0.288/0.325；top5-ADE：2D 为 0.124/0.188/0.227，3D 为 0.132/0.199/0.239），源文也未说明是否另训 2D 变体。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**：
- 从无动作标签视频学到可冻结、可复用的 3D 运动先验，换本体只重训 action expert。
- 在训练时随机丢 depth（p=0.7）、丢历史（整体 p=0.5 / 逐点 p=0.3）的情况下仍能推理，因此部署不强制要求 depth 传感器（与 TraceGen 需推理期 depth 形成对比）。
- 对上游噪声有一定容忍：字幕 41% 词被破坏时指标上升 2.9–3.3%；30% 关键点被丢弃+抖动时上升 1.2–4.1%；向预测 trace 注入 σ=0.05 高斯噪声后 RoboCasa 成功率从 30.25% 降到 27.5%。

**不能做 / 明确失败**：
- 作者自述继承感知栈误差：语义聚类、3D 重建、跟踪、字幕任一环节出错都会变成噪声监督。
- 不建模力、触觉、接触模式，细操作可能受限。
- 只在桌面操作、有限本体与任务族上评测；移动操作臂、灵巧手、更长时程留作 future work。
- 长时程分解任务上明显吃力：三个方法在 Organize Condiments 的第三个子任务全部 0%。
- 仿真里弱于 π0.5（30.25% vs 42%），论文承认该比较**数据不匹配**（π0/π0.5 backbone 用大规模动作标签预训练），但这也说明“无动作预训练”在当前规模还没追平动作标签预训练。

### 6.1 隐含假设 (Hidden Assumptions)

1. **trace 是充分统计量**：假设“物体/工具/手/接触点的 3D 运动”足以决定动作；对需要力控或形变的任务未验证。
2. **VGGT + TAPIP3D 的 3D 几何在野外视频上足够准**：论文给了运动过滤阈值 `τ_m = sqrt((H/20)² + (W/20)²)`、深度权重 `λ_z = 0.1`，但未给重建精度与下游成功率的定量关系。
3. **事件切分即子目标切分**：以加速度谷值作边界，隐含“低加速度 = 阶段转换”，对连续平滑的倾倒类动作是否成立未单独验证。
4. **冻结即复用**：假设不同本体的 action expert 都能从同一套 trace 特征里取到所需信息；只在 UR3 与 PandaOmron 上验证。
5. **对比公平性**：所有方法共享演示、训练步数、batch size，且 π0/π0.5 从公开 checkpoint 微调；但基线预训练数据规模差异巨大（π0 backbone 903M 动作标签帧 vs 本文 13M 无动作帧），"competitive" 的表述应在此条件下读。

---

## 7. 与相关工作对比 (Comparison)

| 方法 | 预测目标 | 是否需动作标签 | 是否需推理期 depth | 参数（trace 推理时） | 备注 |
|------|----------|----------------|--------------------|----------------------|------|
| Diffusion Policy | 动作 | 是（目标域从零训） | 否 | — | 无预训练基线 |
| π0 | 动作 chunk | 是（大规模） | 否 | 4.03B（推理期） | backbone 3.45B，动作标签 903M 帧 |
| π0.5 | 动作 chunk | 是 | 否 | 4.14B（推理期） | backbone 规模同上，预训练规模未公开 |
| Track2Act | 2D 点轨迹 | 否 | 否 | 0.47B | 无度量深度 |
| 3DFlowAction | 稠密 3D flow | 否 | 视设置 | 2.04B | 固定网格 |
| Dream2Flow | 生成视频后提 3D flow | 否 | 是 | 11.3B | 两段式，延迟 106.8 s |
| TraceGen | 固定网格 3D trace | 否 | 是 | 0.67B | 最近的前作；episode 级字幕、短片段 |
| **μ0（本文）** | 查询式语义关键点 3D B-spline | 否 | 否（可选） | 2.59B | 冻结后可配任意 action expert |

架构归属要分清：`SmolVLM2-2.2B` 前缀 + Trace Expert 是 **μ0 的**结构；action expert 采用 π0.5 的 self-attention 架构但**从零训练**；TraceGen 是独立模型（0.67B、需 depth、配自己的 action expert），不要把 μ0 的“无 depth 依赖”或“事件级字幕”算到 TraceGen 头上。

**面试 Tip**：被问到“这篇和 TraceGen 差在哪”，一句话答——**选点方式（语义实体 vs 固定网格）、坐标（全局对齐 3D vs 局部/2D）、监督粒度（事件级字幕 vs episode 级）、接口（冻结查询式世界模型 vs trace replay）**，四点任选两点讲清楚即可。

---

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 正在做“无动作标签视频 → 机器人策略”迁移，需要判断 trace/flow 类中间表示是否值得投入的研究者；
  2. 要把某个视频预训练模型接到自有机器人上、关心冻结复用与重训成本的工程团队；
  3. 研究世界模型输出表示（像素 / latent / flow / trace）取舍的人。
- **建議章節路徑**：先读 §3.3 与附录 B.3（flow + B-spline + rigidity 的完整目标）→ 再读 §2 与附录 A（数据引擎，决定这套方法能不能在你的数据上跑起来）→ 然后附录 B.4 与 D.4（action expert 接口与延迟）→ 最后看附录 E（消融与 scaling，判断增益来自哪里）。§5 Related Work 与附录 F 可快速扫过。
- **不值得精讀的理由**：如果你不做机器人学习、或已熟悉 flow matching / 点跟踪类世界模型，读摘要与 Table 1、Table 2 即可；本文的架构创新主要在“表示与数据管线”，而非新的生成模型或新的策略骨干。

**读表提醒（易踩坑）**：附录 Table 9 里 “w/ Depth & Trace history” 的 top5-DTW（0.107/0.160/0.203）**优于** “Full μ0”（0.127/0.187/0.223），这不是矛盾——Full μ0 训练时对 depth 与 history 做了随机 dropout（附录 C.1），因此常态推理下略低于“输入永远齐全”的变体，换来的是缺输入时的鲁棒性。同理，数据 scaling 中 5% 数据在 T=8 上（0.134）略优于 20%（0.138），说明该点并非严格单调，论文只主张“长 horizon 上增益更一致”。

---

[← Back to Theory](./README.md)

**关键引用**
- 论文（本笔记唯一来源）：https://arxiv.org/abs/2606.13769 （arXiv:2606.13769v3，HTML 全文，2026-10-20）
- 主要数据位置：Table 1 / Table 2 / Table 3 / Table 5 / Table 9 / Table 10 / Table 11 / Table 12；附录 A.3、A.6、B.2–B.4、C.1、D.2、D.4–D.8、E.1–E.4