# StaMo：从紧凑状态表示无监督学习可泛化机器人运动 / StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation

> ⚙️ 本文由 Moltbot 自动生成 | 2026-04-15  
> 修订日期：2026-10-18（合并重建；保留原生成日期）
> - 论文版本：arXiv:2510.05057v2 [cs.RO]，12 Apr 2026；固定 v2 HTML/正文与表为准。  
> - 核心定位：用冻结 DINOv2 加可训练 Transformer 压缩器，把单帧图像压成 2 个 token、每个 1024 维 的紧凑状态；用预训练 DiT 解码重建；状态差被当作潜运动，用于世界建模、共训、线性探针与目标规划。  
> - 口径说明：同一 v2 的 [arXiv 摘要页](https://arxiv.org/abs/2510.05057v2) 写 +14.3% LIBERO / +30% 真机，v2 HTML 摘要与正文写 +11.6% / +31%，两种界面不一致。本文结果按 v2 HTML 表逐项转录并保留表内疑点；不将 +31% 当作已独立核验的 31 个百分点。

## X-Ray
StaMo 的核心不是只提取视频中的运动，而是先从静态图像学习紧凑状态，再检验状态差是否含有可读出的运动信息。它把图像压到 2 个 token、每个 1024 维，再用预训练扩散 Transformer 重建，从而在瓶颈处保留机器人姿态、物体交互等任务关键信息；起止状态 token 的差被直接当作潜运动。证据主要来自 LIBERO 世界建模、RDT 共训、线性探针与 ManiSkill 规划；真机 Table 7 的原表汇总存在内部不一致，需作者确认。

## ASCII 研究时间线（论文引用脉络，非完整领域史）
```text
方法脉络（不把各基线的发表年份混成一条时间点）：
  视频潜动作、轨迹与世界模型路线：LAPA、ATM、UniVLA、WorldVLA 等
  数据与评测背景：DROID、OXE、LIBERO、OpenVLA/OFT 等
2025-10-06  StaMo v1
2026-04-12  StaMo v2：单帧状态学习、状态差表征、下游共训与规划（本文核对版本）
```

## 方法/组件对比
| 组件/路线 | 作用 | 训练状态/条件 | 论文位置 |
|---|---|---|---|
| DINOv2 编码器 | 冻结特征提取 | 全程冻结 | §3.1 |
| Transformer 压缩器（编码器 E 的一部分） | 将 DINOv2 特征压成状态 s | 可训练 | §3.1 |
| DiT 解码器 D | 以 s 为条件重建图像 | 基于 Stable Diffusion 3，可训练；Flow Matching | §3.1 |
| 状态 s | 2 个 token、每个 1024 维 的紧凑表示 | 单帧图像编码 | §3.1 |
| 运动 m | 状态差 m(t)=s(t+1)−s(t) | 无需显式运动监督 | §3.2、§3.4 |
| OpenVLA 世界模型头 | 预测未来状态 s_pred | 轻量 MLP 头；与动作联合训练 | §3.3、式(2) |
| RDT 共训 | 视频帧差生成伪动作 | 每任务 10 条机器人轨迹 + 40 条视频轨迹 | §4.5 |
| 线性探针 | 从 Δz 回归动作序列 | 冻结 StaMo 编码器，训练 MLP | §4.6、式(3)(4) |

## ASCII 信息流
```text
图像 x0 ──DINOv2(冻结)──> 特征 ──Transformer 压缩器(训练)──> s: 2 tokens × 1024维
         └──────────────── 完整编码器 E ──────────────────┘
                                                   │
                                                   └──DiT D(预训练+训练)──> 重建 x0

s(t), s(t+1) ──差──> m(t)=s(t+1)−s(t)
                         ├──> 视频伪动作，供 RDT 共训
                         ├──> Δz，供训练后的 MLP 探针回归动作
                         └──> 线性插值示意：解码出连续运动

OpenVLA 主干 + MLP 头 ──> 动作 与 s_pred；推理只需预测 token，不必解码全图
```

## Eureka Moment
论文的关键观察是：在高度压缩的状态 token 空间里，对起止状态做线性插值，解码图像呈现连续、合理的运动；状态差还能在 sim-to-sim、sim-to-real、real-to-sim 的定性展示中迁移。作者据此主张“运动从静态状态表示中自然涌现”，无需复杂时序模型或视频动作监督。需要注意：这是表示空间的经验性质与实验主张，不是把 Transformer/attention 模块泛化成动力学保证；论文用线性探针、共训和规划任务来评估其可用性。

## 数学核心
### 目标
学习完整编码器 E 中的压缩器及解码器 D，使 s=E(x0) 能重建图像；DINOv2 部分冻结。随后检验 s(t+1)−s(t) 能否成为有用的运动表示。

### 公式
```text
z0 = τ(x0)
z_t = (1−σ_t) z0 + σ_t ε

L_DAE = E[ || D(z_t, E(x0), t) − u(z_t) ||² ]
```
变量：x0 为输入图像；τ 为预训练扩散模型的 VAE 编码器；z0 为图像潜变量；ε 为噪声；σ_t 为插值系数；u(z_t) 为目标速度；E 为完整图像编码器（冻结 DINOv2 + 可训练压缩器）；D 为 DiT 解码器。

```text
m(t) = s(t+1) − s(t) = E(o(t+1)) − E(o(t))
```

```text
L_total = λ_action L_action + λ_future( MSE(s_pred, s_gt) + L1(s_pred, s_gt) )
```
变量：L_action 为下一 token 交叉熵；s_pred 为预测未来状态；s_gt 为真实未来状态；实验中令 λ_action=λ_future。

```text
Δz = E(I(n+k)) − E(I(n))
MSE(A_n, MLP(Δz))
```
变量：I(n) 为起始图像；I(n+k) 为 k 步后图像；A_n 为真实动作序列；Δz 为状态差；MLP 为需要监督训练的轻量探针；作者将此协议称为 linear probing，但正文未给出足以将整个 MLP 等同于严格线性映射的结构细节。

### 直觉
压缩瓶颈迫使表示保留任务关键信息；预训练 DiT 的生成先验降低低维状态的重建难度；若状态空间把“世界长什么样”和“世界怎么变”线性化，那么简单差分就近似潜动作。

**Napkin Formula**：状态瓶颈 + 预训练生成先验 → 可重建状态；状态差 + 训练后的 MLP → 动作预测。计算潜在差分本身不增加可训练参数，但将差分解码成可执行动作仍需要训练动作头。

## 示意性可计算玩具例子（不是作者实验）
假设一个二维示意状态：
```text
s0=(1.0, 2.0), s1=(3.0, 4.0)
m = s1−s0 = (2.0, 2.0)
```
线性插值 α=0.5：
```text
s_mid = 0.5×s0 + 0.5×s1 = (2.0, 3.0)
```
若线性头 W=diag(0.1, 0.1)，则预测动作：
```text
a_pred = W×m = (0.2, 0.2)
```
若目标动作是 (0.2, 0.2)，MSE=0；若 W=diag(0.05, 0.1)，则 a_pred=(0.1, 0.2)，对目标 (0.2, 0.2) 的 MSE 为：
```text
((0.1−0.2)² + (0.2−0.2)²) / 2 = 0.005
```
这个例子只说明“状态差可经线性映射读动作”的玩具逻辑；真实 StaMo 用 2 个 1024 维 token 表示状态，并用训练后的 MLP 预测动作序列，不能把该数值当论文结果。

## 作者报告结果（固定 v2 正文/表）
### 重建
Table 1 报告 PSNR dB / SSIM：libero_10 25.5194/0.8909；libero_90 27.2470/0.8962；libero_goal 24.6467/0.8926；libero_object 27.0011/0.9105；libero_spatial 25.9683/0.8984；Droid 20.2492/0.7346；ManiSkill（OOD）22.1673/0.8824。作者据此说明 2 token 仍能保持较高重建保真度，但 DROID 与 OOD 指标明显低于部分 LIBERO 子集。

### Table 2：LIBERO 世界建模（成功率%，每任务 1000 次 rollout；列为 Spatial/Object/Goal/Long/Average）
- OpenVLA：84.7 / 88.4 / 79.2 / 53.7 / 76.5  
- OpenVLA* + DINOv2 Feature：88.6 / 90.4 / 83.5 / 61.4 / 80.9  
- OpenVLA* + StaMo state：92.3 / 94.8 / 88.1 / 75.2 / 87.6  
- OpenVLA* + StaMo motion：93.1 / 95.1 / 87.4 / 76.9 / 88.1  
- OpenVLA-OFT：93.7 / 94.2 / 89.7 / 91.3 / 92.2  
- OpenVLA-OFT* + DINOv2 Feature：94.1 / 95.0 / 91.2 / 93.0 / 93.3  
- OpenVLA-OFT* + StaMo state：96.8 / 98.9 / 95.0 / 96.3 / 96.8  
- OpenVLA-OFT* + StaMo motion：95.3 / 97.1 / 94.1 / 92.9 / 94.9  
- 强基线 UniVLA：95.4 / 98.8 / 93.6 / 94.0 / 96.5  

表内一致性提示：以上数字照录原 Table 2。UniVLA 四个分项的等权算术平均为 95.45%，但原表 Average 为 96.5%；论文此处未解释另一种加权方式。因此不能把该 Average 当成已独立复核的均值或据此精算微小排名差。
作者解释：短动作跨度的 OpenVLA 更适合 motion；长动作跨度的 OpenVLA-OFT 更适合 state 作为稳定目标条件信号。该结论是特定架构与设置下的比较，不是通用动力学保证。

### Table 3：推理频率（单张 RTX 4090，Hz↑）
WorldVLA 2.27；UniVLA 2.65；OpenVLA 4.16；OpenVLA + DINOv2 Feature 2.86；OpenVLA + StaMo 4.02；OpenVLA-OFT 18.24；OpenVLA-OFT + DINOv2 Feature 11.42；OpenVLA-OFT + StaMo 17.82。论文强调推理不必解码全图、开销较小；这里只报告频率，不外推延迟、其他硬件或统计显著性。

### Table 5：RDT 共训（成功率%，列同上；每任务 10 条机器人轨迹 + 40 条视频轨迹；20 次试验、3 次完整运行平均）
- RDT（all Real）：91.6 / 93.3 / 86.7 / 73.3 / 86.2  
- RDT（1 Real）：71.7 / 70.0 / 66.7 / 43.3 / 62.9  
- RDT（1Real + 4ATM）：83.3 / 81.7 / 71.7 / 56.7 / 73.4  
- RDT（1Real + 4LAPA）：80.0 / 76.7 / 75.0 / 65.0 / 74.2  
- RDT（1Real + 4StaMo）：90.0 / 91.6 / 86.7 / 70.0 / 84.6  

### 其他结果
Table 4 的 ManiSkill 零样本迁移（LIBERO 训练后直接测试，每任务 100 次）：OpenVLA 28/4/18/16；UniVLA 32/7/22/20；WorldVLA 30/10/17/22；OpenVLA + StaMo state 41/32/48/36（Pick Cube/Push Cube/Stack Cube/Pull Cube Tool）。Table 6 目标规划（ManiSkill，编码器冻结、只训练线性头，每任务 100 次）：DINO-WM 12/8/4/18；StaMo 22/32/16/38。

## Table 7 真机结果与汇总不一致（必须呈现）
Table 7 原表列为：短任务 Pick up the [toyname]、put the toy into the basket、Open the drawer、短 Average；长任务 put all cups into the basket、put the toy into the drawer、Stack all cups、长 Average；最后 All Average。原表数值如下。

| 方法 | 短1 | 短2 | 短3 | 短Avg | 长1 | 长2 | 长3 | 长Avg | All Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial-VLA | 0.45 | 0.50 | 0.20 | 0.38 | 0.35 | 0.20 | 0.20 | 0.25 | 0.32 |
| WorldVLA | 0.55 | 0.20 | 0.15 | 0.30 | 0.25 | 0.40 | 0.15 | 0.26 | 0.28 |
| UniVLA | 0.55 | 0.30 | 0.35 | 0.40 | 0.35 | 0.30 | 0.10 | 0.25 | 0.33 |
| OpenVLA | 0.35 | 0.30 | 0.25 | 0.30 | 0.20 | 0.30 | 0.15 | 0.20 | 0.25 |
| OpenVLA + StaMo state | 0.60 | 0.55 | 0.50 | 0.60 | 0.55 | 0.55 | 0.45 | 0.52 | 0.56 |
| OpenVLA-OFT + StaMo state | 0.70 | 0.65 | 0.60 | 0.65 | 0.70 | 0.65 | 0.55 | 0.63 | 0.64 |

审查要点：OpenVLA + StaMo state 的短任务三项为 0.60/0.55/0.50，但原表短 Average 写 0.60；按三项算术平均应为 0.55。六项算术平均为 3.20/6=0.5333，但原表 All Average 写 0.56。上表照录原表，不静默重算替换；汇总方式需作者确认。摘要宣称 real-world +31%，但由于 Table 7 汇总内部不一致，本文不把 +31% 解释为已独立核验的 31 个百分点。 同表 OpenVLA 的三个长任务值为 0.20/0.30/0.15，等权平均为约 0.2167，表列 0.20；六项等权平均约 0.2583，表列 0.25。应向作者确认所有汇总规则，不只更正其中一行。

## 工程实现边界与 Hidden Assumptions
- DINOv2 冻结，只有 Transformer 压缩器与 DiT 解码器训练；DiT 基于 Stable Diffusion 3，使用 Flow Matching。2 token、1024 维是主设置，Table 8 对 256/512/1024 维做重建消融。
- OpenVLA* 是替换原视觉编码器并预测 StaMo 状态；OpenVLA-OFT* 还做了小改动：只用第三人称相机、去掉腕部相机。因此 Table 2 的增益绑定这些架构条件。
- RDT 共训依赖视频帧差生成伪动作，伪动作质量会受编码器与视频分布影响；论文报告的是特定每任务数据配置下的结果，不能直接外推为任意数据规模或任意策略的保证。
- Table 3 仅单张 RTX 4090 的频率对比；没有给出延迟分布、其他硬件、统计显著性等结论。
- 线性插值与状态差的可解释性主要是定性展示与线性探针证据；不能把 Transformer/attention 模块本身说成动力学保证。
- 可观测性限制（研究者推论）：相似单帧外观可能对应不同速度、接触状态或摩擦条件；只靠图像重建不能证明这些隐变量已被识别。
- 插值安全与容量限制（研究者推论）：视觉上连续的解码序列不等于无碰撞、可执行轨迹；固定 2-token 瓶颈也不保证保留每类任务所需的全部信息。这些需要单独评估。
- Table 7 原表汇总待确认；不据此新增“典型跌幅”等无源数字，也不宣称跨传感器或更广真机泛化。

## 基线比较与面试 Tip
基线可分成四类：直接 VLA（OpenVLA、OpenVLA-OFT）、世界模型 VLA（WorldVLA、UniVLA）、潜动作/轨迹（LAPA、ATM）、规划世界模型（DINO-WM）。Table 2 中，OpenVLA* + StaMo motion 平均 88.1，略高于 state 的 87.6；OpenVLA-OFT* + StaMo state 平均 96.8，高于 motion 的 94.9，说明“state 还是 motion”取决于动作跨度与架构。Table 5 中，1Real+4StaMo 平均 84.6，高于 1Real+4ATM 的 73.4、1Real+4LAPA 的 74.2，但仍低于 all Real 的 86.2。

面试 Tip：先讲清“紧凑状态”与“潜动作”的双重角色；再解释为什么预训练 DiT 先验让 2 token 可重建；最后用 Table 2/3/5 说明世界建模、速度与共训证据。被问真机时，主动指出 Table 7 汇总不一致，不把 +31% 说成已核验的百分点优势；被问机制时，不要把 attention 说成动力学保证，而应说这是状态差在实验中的经验可读性。

## 参考链接（含章节/表号）
- arXiv:2510.05057v2，12 Apr 2026：https://arxiv.org/html/2510.05057v2 — §3.1 式(1)，§3.2，§3.3 式(2)，§3.4；Table 2/3/5/7；Table 4/6；Table 8/9；Figure 4/6。
- 项目页：https://aim-uofa.github.io/StaMo/ — 仅作来源链接；不据此推断代码、数据已开放或未来计划已完成。

[← Back to Theory](./README.md)
