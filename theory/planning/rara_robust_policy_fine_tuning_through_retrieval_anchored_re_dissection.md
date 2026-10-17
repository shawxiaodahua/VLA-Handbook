> 历史补录：补回 2026-10-12 的深度阅读任务；论文原始收录日期 2026-10-10；实际补写日期 2026-10-17。本文依据当前读取的指定论文版本，不冒充当时已完成的报告。

# 检索锚定的鲁棒策略微调：RARA 把“相关预训练经验”变成表征锚点 (RARA: Robust Policy Fine-Tuning through Retrieval-Anchored Representation Alignment for Robotic Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: RARA: Robust Policy Fine-Tuning through Retrieval-Anchored Representation Alignment for Robotic Manipulation
> **链接**: https://arxiv.org/abs/2610.13578v1
> **核心定位**: 少样本微调 VLA 时，用检索到的同类预训练 demo 做“表征锚点”，先对齐再约束，缓解表征漂移，提升 OOD 成功率。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：在 13 个仿真/真实任务上，RARA 相对标准 fine-tuning 平均提升 ID 9.5 点、OOD 13.6 点（论文摘要）；分设置增益见 §5.3 |
| 适合精读 | 如果你在做 VLA 少样本微调、OOD 泛化、表征正则，重点看 §4.2、§5.2、§5.5 |
| 可以跳过 | 如果你只关心新 VLA 架构或大规模预训练配方，这篇距离中等 |
| 落地可行性 | 中高：推理零额外成本，但训练要加 Stage 1、frozen reference 前向和检索/DTW 预处理 |
| 主要风险 | 预训练数据没有同类 skill 时，anchor 可能负迁移（附录 G 实测）；实验主要覆盖视觉 OOD，未跨 embodiment/action space |

💡 **X-Ray 开场**

这篇论文问：少样本微调 VLA 时，为什么模型在演示分布内表现不错，一换相机位姿、背景或物体就崩？作者的回答是：标准 fine-tuning 会把预训练学到的视觉表征“拉歪”。RARA 的做法不是冻权重，也不是把检索到的 demo 当额外动作监督，而是从预训练数据里找“同类任务、相近动作阶段”的帧，先让目标帧对齐到这些帧的预训练表征，再在微调时锚住这些相关帧的表征。对 VLA 研究者意味着：**“选对相关经验，并把它变成表征正则”** 可能比“多喂数据”更关键。

📍 **研究全景时间线**

```
2022  Kumar et al.：full fine-tuning 会扭曲预训练特征，OOD 变差
  → 2023  Diffusion Policy；Behavior Retrieval：visuomotor 强基线 + 检索做少样本
  → 2024  DROID；FlowRetrieval：大规模真实数据 + 流/轨迹检索
  → 2025  CLASS；STRAP；GR00T N1：动作序列对比学习、子轨迹检索 co-train、VLA 基座
  → 2026  BlindVLA；RETAIN；RARA：视觉教师对齐、权重插值、检索锚定表征对齐
                         ↑ 本文位置：2026，检索 + 两阶段表征锚定
局限：需要预训练数据含同类 skill；只验证视觉/布局 OOD，未跨机器人或动作空间
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 训练/推理差异 | 关键设置 |
|------|------|------|----------------|----------|
| 指令检索 | 目标任务指令集；预训练 demo 指令 | 检索集 D_ret | 适配前按任务构建；可离线 | CLIP 文本编码器；Top K=100；指令嵌入取平均后算 cosine |
| 动作 chunk 配对 | D_ret 与 D_tgt 的动作序列 | 正样本 P(i)、负样本 N(i) | 适配前/Stage 1 前构建 | chunk 长度 H=16；DTW 距离；DP 设置 m+=4；负样本从大于中位数的 chunk 采样 |
| Stage 1 | 目标帧 + 检索帧 | 对齐后的 encoder + projector | 只训练产生 anchored representation 的层和 projector | 3000 steps；batch 32 target + 32 retrieval；τ=0.07；λ1=1（DP） |
| Stage 2 | 目标 demo + 检索帧 | 微调后的策略 | DP：全策略 fine-tuning；GR00T：冻结 LLM，仅训视觉编码器和动作头；anchor loss 继续 | DP 在 RoboTwin 训练 300 epochs；batch 128，目标/检索 1:1；λ2=1（DP） |
| 推理 | 当前观测 | 动作 chunk | 只用策略；projector 丢弃；无需检索/anchor | DP/GR00T 预测 16-step chunk；真实机器人执行前 8 步再 replan |

### 1.2 关键机制 (Key Mechanism)

- **两层相关性，缺一不可**：先用指令相似度筛“同类任务”，再用动作 chunk 的 DTW 距离配“同一动作阶段”。论文 Fig. 3 的例子很关键：掀锅盖和拆纸巾卷都向上抬，动作相似但任务不同；只按动作检索会配错任务。
- **Stage 1 是“借结构”，不是“保结构”**：目标帧对齐到“动作阶段相同”的检索帧，让目标帧复用预训练见过的表征；anchor 只限制检索帧自身别被 Stage 1 改太多。论文 Table 1 显示，Stage 1 后目标帧的 feature distortion 仍有 0.46–0.56，并不接近预训练策略 P；真正降低到 0.23–0.33 的是 Stage 2。
- **Stage 2 的 anchor 只作用于检索帧，不拉回目标帧**：这是与普通 drift regularizer 的关键区别。目标表征可以自由适应任务；被约束的是“相关预训练经验”的表征，防止小目标数据把这块结构冲掉。
- **单层 anchor，不限制权重更新**：RARA 不冻 encoder、不做权重插值，也不要求特定架构；它只选一个 anchored layer。层选择有影响，但论文 Fig. 6 显示 Diffusion Policy 的 8 个位置都优于 FT，GR00T 5 个位置中 4 个提升。
- **projector 只服务于 Stage 1**：对比损失在投影空间算，projector 是 1024→512→256 的两层 MLP，Stage 1 后丢弃；推理不带额外模块。

⚡ **Eureka Moment**：RARA 的核心不是“再训练预训练数据”，而是**把检索到的相关预训练帧变成表征锚点**——先让目标帧对齐到这些帧，再在微调时只保护这些帧的表征。

### 1.3 信息流/架构图 (Flow / Diagram)

```
D_pre ──CLIP 指令相似度──> D_ret (Top K=100)
D_tgt ──动作 chunk──┐
D_ret ──动作 chunk──┴──DTW 配对──> P(i) 正样本 / N(i) 负样本

Stage 1:
  target frame ──> h_θ^ℓ ──> g_ψ ──> z_tgt
  D_ret frame  ──> h̄^ℓ（frozen pre-trained）──> g_ψ ──> z̄_p / z̄_n
  loss = L_contrastive + λ1 · L_anchor

Stage 2:
  D_tgt ──> π_θ ──> L_BC
  D_ret ──> h_θ^ℓ vs h̄^ℓ ──> L_anchor
  loss = L_BC + λ2 · L_anchor

推理:
  π_θ only；g_ψ 丢弃；不需要 D_ret、DTW 或 anchor
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
Stage 1: L_contrastive + λ1·L_anchor；Stage 2: L_BC + λ2·L_anchor
```

**目标**：在目标 demo 上学会动作，同时不破坏与目标任务相关的预训练表征，从而在 ID 和 OOD 评测中都保持成功率。

**Stage 1 / Stage 2 目标**：

```
Stage 1: min_{θ, ψ}  L_contrastive + λ1 · L_anchor
Stage 2: min_θ       L_BC + λ2 · L_anchor
```

**行为克隆损失**：

```
L_BC = E_{(o,a)~D_tgt}[ L_BC(π_θ(o), a) ]
```

**多正样本对比损失**：

```
L_contrastive = -(1/B) Σ_i log(
  (Σ_{p∈P(i)} exp(sim(z_i^tgt, z̄_p)/τ))
  /
  (Σ_{p∈P(i)} exp(sim(z_i^tgt, z̄_p)/τ) + Σ_{n∈N(i)} exp(sim(z_i^tgt, z̄_n)/τ))
)
```

**Anchor 损失**：

```
L_anchor(θ) = E_{o~D_ret}[ d(h_θ^ℓ(o), h̄^ℓ(o)) ]
```

**DTW 动作 chunk 距离**：

```
δ(A,B) = min_{ω∈Ω(A,B)} (1/|ω|) Σ_{(u,v)∈ω} ||A_u - B_v||_2
```

> 符号与本文保持一致：`θ` 是策略参数；`ψ` 是 projector 参数；`ℓ` 是 anchored layer；`h_θ^ℓ` 是适配中策略的表示；`h̄^ℓ` 是 frozen 预训练策略的表示；`z_i^tgt` 是目标帧 embedding；`z̄_p`、`z̄_n` 是检索帧经当前 projector 映射的 embedding（其 anchored representation 冻结）；`P(i)`、`N(i)` 是正/负样本；`B` 是目标 batch size；`τ` 是温度；`d` 是表示距离。

| 变量/超参 | 含义 | 论文设置 |
|-----------|------|----------|
| `K` | 检索 demo 数 | 100 |
| `H` | 动作 chunk 长度 | 16 |
| `m+` | 正样本数 | DP 设置中为 4 |
| `τ` | 对比温度 | 0.07（DP） |
| `λ1`, `λ2` | anchor 权重 | DP 中均为 1；GR00T 层研究中为 0.1 |
| `d` | anchor 距离 | DP 用 MSE；GR00T 用除以 reference batch variance 的平方误差 |
| Stage 1 步数 | 对齐训练 | 3000 steps（DP） |
| Stage 2 轮次 | 目标微调 | DP 在 RoboTwin 为 300 epochs |

**直觉**：对比损失只把目标帧拉到“动作阶段相同”的检索帧，不要求整条轨迹对齐；anchor 只让检索帧的表示别被目标微调带跑。两者合起来是“先借相关结构，再保相关结构”。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下数值均为**假设**，只用于验算机制，不是论文实验结果。论文实际用 `τ=0.07`；这里为便于手算取 `τ=0.5`。

**假设 1D 动作 chunk，H=3：**

```
目标 chunk A = [0.0, 1.0, 2.0]
正样本 B   = [0.0, 1.0, 2.0]
负样本 C   = [0.0, 0.0, 2.0]
```

**DTW 验算：**

```
δ(A,B) = (|0-0| + |1-1| + |2-2|) / 3 = 0
δ(A,C)：最优对齐 (A0,C0), (A1,C0), (A2,C2)
      = (|0-0| + |1-0| + |2-2|) / 3
      = (0 + 1 + 0) / 3 = 0.333
```

假设负样本阈值（中位数）为 0.2，则 `B` 是正样本（0 < 0.2），`C` 是负样本（0.333 > 0.2）。这演示了论文“用 DTW 距离定义正负样本、负样本从大于中位数的 chunk 采样”的逻辑。

**对比损失验算：**

假设 target embedding `z=(1,0)`，正样本 `p=(0.8,0.6)`，负样本 `n=(0,1)`：

```
sim(z,p) = 0.8
sim(z,n) = 0
τ = 0.5

正样本项 = exp(0.8/0.5) = exp(1.6) = 4.953
负样本项 = exp(0/0.5) = 1

L = -ln(4.953 / (4.953 + 1))
  = -ln(0.8320)
  = 0.1839
```

若用论文的 `τ=0.07`，同样相似度下：

```
L ≈ -ln(exp(11.4286) / (exp(11.4286) + 1)) ≈ 1.07e-5
```

温度越低，对齐好时损失越接近 0，但对难负样本也更敏感。

**Anchor 验算：**

假设检索帧的适配表示为 `h_θ=(0.90, 0.20)`，frozen 预训练表示为 `h̄=(1.00, 0.10)`：

```
MSE = ((0.90-1.00)^2 + (0.20-0.10)^2) / 2
    = (0.01 + 0.01) / 2
    = 0.01
```

Stage 1 总损失（假设 `λ1=1`）：

```
L_stage1 = 0.1839 + 1 × 0.01 = 0.1939
```

Stage 2 若假设 `L_BC=0.05`、anchor 仍为 0.01、`λ2=1`：

```
L_stage2 = 0.05 + 0.01 = 0.06
```

这个玩具例子只说明闭环：对比损失负责“对齐到匹配动作阶段”，anchor 负责“别把检索帧的表示改坏”。

## 4. 工程视角 (Engineering View)

> 论文未报告吞吐、延迟、抖动、量化误差等部署指标；以下是基于方法结构的**条件性分析**，不是论文实测。

| 维度 | 论文给定/可确认 | 工程含义 |
|------|------------------|----------|
| 训练步数 | Stage 1 为 3000 steps；DP 在 RoboTwin 的 Stage 2 为 300 epochs | 比标准 FT 多一段对齐训练；Stage 2 每 batch 还要算 anchor |
| Stage 2 batch | 128，目标 chunk 与检索帧 1:1 | 一半 batch 用于 anchor，不参与 BC；显存和吞吐换 OOD |
| Frozen reference | 需要 frozen 预训练策略的 anchored layer 前向 | 可用 `no_grad`；但要多一份激活/计算，GR00T 视觉塔更明显 |
| 推理成本 | projector 丢弃；不需要检索/anchor | **推理零额外模块**；Co-FT/STRAP 的额外检索/共训数据也只在训练时使用 |
| 动作时序 | DP/GR00T 预测 16-step chunk；真实机器人只执行前 8 步再 replan | 控制频率论文未给，部署时要按实际控制周期测；不要用 chunk 长度直接推 Hz |
| 数据预处理 | CLIP 指令检索、动作归一化、DTW 配对 | 可离线做；但依赖预训练数据有指令、动作和可检索结构 |
| 超参敏感性 | `λ1/λ2`、`τ`、`m+`、anchored layer | DP 用 MSE，GR00T 用 variance-normalized MSE，说明尺度不同要调 |
| 量化/尺度 | anchor 距离对表示尺度敏感 | 低精度部署时，anchor 只在训练用；推理不受影响，但训练稳定性要验证 |

**工程结论**：RARA 的卖点是“训练时多花算力，推理时不加模块”。如果目标场景有可访问的预训练数据、且目标与预训练主要是视觉域差异，它比“推理时检索”或“额外大模型”更容易落地；如果预训练数据不可检索，或目标需要新动作空间，收益不确定。

## 5. 数据与评测 (Data & Eval)

| 设置 | 预训练 | 目标 demo | 策略 | ID/OOD 定义 | 评测次数与单位 |
|------|--------|-----------|------|-------------|----------------|
| RoboTwin 2.0 | 自采约 2k demos，覆盖多种技能，留出 4 个目标任务 | 每任务 50 demos | Diffusion Policy | ID：目标 demo 的窄布局；OOD：拓宽物体布局、背景、干扰物 | 每任务每 setting 150 rollouts；每个训练 seed 用最后 3 个 checkpoint（250/275/300），各 50 rollouts；重复 3 个训练 seed |
| RoboCasa | DROID 76k demos | 每任务 100 demos | GR00T N1.7 | ID：演示厨房，随机物体位姿；OOD：相机位姿、背景、干扰物、物体实例 | 每任务 50 episodes；重复 3 个训练 seed |
| 真实 Franka | DROID | 每任务 50–80 demos，VR 遥操作 | GR00T N1.7 | ID/OOD 同 RoboCasa | 每任务 10 trials；整套评测重复 3 次；所有方法在同一 recorded scene randomizations 上测 |

**关键结果（论文实测）**

| 设置 | 方法 | ID | OOD | 来源 |
|------|------|----|-----|------|
| RoboTwin 任务平均 | FT | 67.4 | 47.3 | 论文 Table 2 |
| RoboTwin 任务平均 | RARA | 76.0 | 58.3 | 论文 Table 2 |
| RoboCasa 任务平均 | FT | 69.2 | 44.3 | 论文 Table 2 |
| RoboCasa 任务平均 | RARA | 74.0 | 55.5 | 论文 Table 2 |
| 真实机器人 | RARA vs FT | +16.4 | +19.2 | 论文 §5.3 / Fig. 5 |
| 13 任务汇总 | RARA vs FT | +9.5 | +13.6 | 论文摘要（作者主张） |

代表性任务（论文 Table 2）：

- RoboTwin `Move Pill Bottle`：FT 32.9 / 9.6 → RARA 51.6 / 23.8。
- RoboTwin `Place Container`：FT 82.0 / 60.7 → RARA 86.7 / 63.3。
- RoboTwin `Click Alarm Clock`：FT 80.4 / 52.4 → RARA 80.4 / 64.2；ID 持平，OOD 提升明显。
- RoboTwin `Shake Bottle`：FT 74.2 / 66.4 → RARA 85.3 / 82.0。
- RoboCasa `Open Blender Lid`：FT 76.0 / 44.7 → RARA 80.7 / 56.7。

**表征指标（论文 Table 1，RoboTwin）**

| 指标 | FT | RARA | 说明 |
|------|----|------|------|
| 目标帧 feature distortion（Move Pill Bottle，ID/OOD） | 0.50 / 0.49 | 0.23 / 0.23 | 越低表示越接近预训练策略 P |
| 目标帧 feature distortion（Place Container，ID/OOD） | 0.47 / 0.48 | 0.31 / 0.33 | 同上 |
| RARA Stage 1 后 distortion（Move Pill Bottle） | — | 0.48 / 0.46 | Stage 1 并不保持接近 P；Stage 2 才降下来 |
| 预训练帧 CKA（RARA） | — | 0.91–0.92 | 预训练结构保留较好 |
| 预训练帧位置探针 R²（RARA） | — | 0.70–0.71 | 物体位置仍可读 |

**消融（论文 Table 3，RoboTwin 四任务平均）**

| 变体 | ID | OOD | 结论 |
|------|----|-----|------|
| FT | 67.4 | 47.3 | 基线 |
| RARA | 76.0 | 58.3 | 完整方法 |
| 随机预训练数据 | 50.3 | 34.9 | 检索数据关键；随机数据比 FT 更差 |
| 随机 Stage 1 配对 | 65.3 | 40.6 | 动作感知配对关键 |
| Stage 2 加 Co-FT | 65.8 | 48.1 | 把检索 demo 当动作监督反而降 |
| 只用 Stage 1 | 66.6 | 43.2 | 没有 anchor，Stage 1 的对齐保不住 |
| 只用 Stage 2 | 73.0 | 46.2 | 没有 Stage 1 对齐，OOD 增益基本消失 |

**层选择（论文 Fig. 6）**：Diffusion Policy 的 8 个 anchored layer 位置全部优于 FT，成功率从 21.2% 提升到 32.0–39.7%（+11 到 +18 点）；GR00T 的 5 个位置中 4 个提升 4–10 点，早期 ViT layer 6 与 FT 持平。我的判断：RARA 不绑定某一层，但“随便选一层”仍可能不赚，层选择是调参项。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**

- 少样本视觉 OOD：在相机位姿、背景、干扰物、物体实例变化下，RARA 在 RoboTwin/RoboCasa/真实机器人的任务平均上优于包括 FT 在内的五个基线（论文 Table 2、Fig. 5）。
- 跨策略：Diffusion Policy 和 GR00T N1.7 都验证；真实机器人用 DROID 预训练的 GR00T。
- 推理零额外模块：projector 丢弃，不需要检索或 anchor；适合部署时保持策略接口不变。
- 单层可插：不限制架构或可训练权重；对 DP 的多个层位置都有效。

**不能做什么 / 失败条件**

- **预训练数据没有同类 skill 时会负迁移**。附录 G 表 5 是强证据：用 Click Bell 数据做预训练时，RARA 在 `Move Pill Bottle` 上 -6.7 点、`Place Container` 上 -20.0 点；换成 pick-and-place 数据，这两个任务分别 +18.8 / +4.9，但 `Click Alarm Clock` 只 +0.2。作者也明确说 RARA 依赖预训练数据含同类任务。
- **未验证跨 embodiment / 新动作空间**。论文结论明确限定在“同一机器人、同一动作空间，主要是视觉变化”；不能外推到新机器人、双臂、人形或新动作维度。
- **真实机器人统计量有限**：每任务 10 trials，重复 3 次；虽然用了 recorded scene randomizations 保证公平，但置信区间仍可能宽。
- **Stage 1 的收敛不是所有任务都保证**：论文只报告 `Place Container` 对比损失 2.73→0.22（92%）和 `Move Pill Bottle` 2.62→0.87（67%）；其他任务未给同样细节。
- **检索错误会放大**：指令过滤或 DTW 配对错了，Stage 1 会把目标帧拉到不相关区域；附录 G 的负迁移就是这类风险。

### 6.1 隐含假设 (Hidden Assumptions)

1. 预训练数据中存在与目标“同类任务”的 demo，且指令文本能区分任务意图；否则检索集 D_ret 不可靠。
2. 目标与预训练的主要差异是视觉外观，动作空间、机器人本体和控制接口基本一致；论文没有验证跨本体。
3. 单层 anchored representation 足以承载“可复用的鲁棒结构”；层选择有影响，但论文认为多数层可用。
4. 动作归一化后，DTW 能衡量“动作阶段相似”；夹爪维度要单独缩放，否则二值开合会压过平移/旋转。
5. Stage 1 的对比对齐不会把目标帧拉到不相关区域；这依赖检索和配对质量。
6. Anchor 的权重和距离度量可跨策略迁移；DP 用 MSE、GR00T 用 variance-normalized MSE，说明这不是完全免调的。
7. 评测 OOD 主要是视觉/布局/物体变化，不是新任务语义或新动作空间；因此结论不能直接外推到“开放世界新技能”。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 干预方式 | 是否需要预训练数据检索 | 适用场景 | 论文结果（RoboTwin / RoboCasa 平均 ID/OOD） |
|------|--------|----------|--------------------------|----------|---------------------------------------------|
| FT | 目标 only | 只算 BC | 否 | 目标分布内 | 67.4/47.3；69.2/44.3 |
| Co-FT | 加预训练 demo 动作监督 | BC on D_ret + D_tgt | 是 | 检索数据与目标接近 | 64.4/47.6；67.1/43.7 |
| STRAP | 检索子轨迹 co-train | DINOv2 + subsequence DTW，检索后 co-train | 是 | 有相似子轨迹 | 72.6/52.4；66.1/48.1 |
| RETAIN | 权重插值 | θ(α)=(1−α)θ_pre+αθ_ft | 否 | 预训练-目标 gap 小 | 47.0/34.2；RoboCasa 附录 H 为 31.1/14.8 |
| BlindVLA | 视觉表征对齐 | 对齐冻结视觉教师 | 否 | 视觉域偏移 | 67.9/47.2；68.5/46.8 |
| RARA | 检索帧表征锚定 | Stage 1 对比对齐 + Stage 2 anchor | 是 | 预训练含同类 skill、视觉 OOD | 76.0/58.3；74.0/55.5；真实 +16.4/+19.2 |

**关键差异**：

- Co-FT 把检索 demo 当**动作监督**；RARA 只把检索帧当**表征锚点**。消融中“Stage 2 加 Co-FT”反而降到 65.8/48.1，说明动作监督不是关键。
- BlindVLA 对齐**冻结视觉教师**；RARA 对齐的是**检索帧的预训练表征**，并且 anchor 只作用于检索帧。
- RETAIN 约束**权重**；RARA 不限制权重，只约束相关表征。附录 H 显示 RETAIN 在 DROID→RoboCasa 的大 gap 下崩得厉害（31.1/14.8），而 RARA 仍能提升。
- STRAP 也做检索，但用于 co-train；RARA 的检索用于“配对 + anchor”，不是复制动作。

**面试 Tip**：如果被问“RARA 和 Co-FT/BlindVLA/RETAIN 的区别”，一句话答：**RARA 不把检索 demo 当动作监督，也不冻权重/编码器；它把检索帧当“只针对相关预训练表征”的 anchor，Stage 1 先对齐动作阶段，Stage 2 再防止这些表征被冲掉。** 补一句条件：预训练数据必须有同类 skill，否则可能负迁移。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 VLA 少样本微调、OOD 泛化、表征正则的研究者：重点看 §4.2 的对比/anchor 设计和 §5.2 的 feature distortion 分析。
  2. 要评估“检索 + 额外训练”是否值得的工程团队：重点看 §5.5 消融和附录 G 的负迁移条件。
  3. 做机器人 OOD 评测的人：看 §5.1 的 ID/OOD 定义和评测次数，不要只看任务平均。

- **建议章节路径**：先读 §4.1 + §4.2（检索和两个损失）→ 再看 §5.2（Stage 1 不保表征、Stage 2 才降 distortion 的反直觉结果）→ 然后 §5.5（两个 stage 都必要、随机数据/随机配对会崩）→ 最后附录 G（skill 不匹配会负迁移）。§2 相关工作可快读，§5.3 表格按需查数字。

- **不值得精读的理由**：如果你不做机器人学习、只关心新 VLA 架构，或者你的场景需要跨 embodiment/新动作空间，这篇的结论不直接适用；读摘要和 Table 2 即可。若你已经熟悉 Co-FT/STRAP/BlindVLA 这一类方法，重点看 RARA 的 anchor 作用域和消融即可。

---

**关键引用**：论文链接 https://arxiv.org/abs/2610.13578v1

[← Back to Theory](./README.md)