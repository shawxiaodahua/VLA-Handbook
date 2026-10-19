# FastVLA：任务条件化与时间自适应的 VLA 视觉 Token 压缩（FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression for Efficient Vision-Language-Action Models）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-19
>
> **论文**：FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression for Efficient Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2610.22146
> **核心定位**：不改模型、不加缓存，用“任务→视觉”注意力给 token 打分，再用 inverse-Simpson 集中度在线决定每一步保留多少 visual token；在 OpenVLA、OpenVLA-OFT、CogACT 和真实 Franka 上换取 1.09×–1.46× 的 action latency 加速。

> 证据标注约定：下文用【论文实测】表示论文表格/正文给出的数字，用【作者主张】表示论文的定性判断，用【条件性分析】表示我基于论文信息的推断或工程判断。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 每步压缩率不应固定；用 task-to-vision attention 的 inverse-Simpson 集中度做滑动窗口校准，可在成功率接近基线的前提下，把 LLM backbone FLOPs 降 35–59%、action latency 降 9–32%【论文实测】 |
| 适合精读 | 做 VLA 推理加速、token pruning、闭环控制频率优化的读者；重点看 §3.2–3.3、§4.3、§4.4 |
| 可以跳过 | 只关心 VLA 新任务泛化、表征学习，或不打算做部署效率的人 |
| 落地可行性 | 中高：training-free、可插拔、无缓存；但 ρ 需按 suite/backbone 调，attention 打分开销未单独报告 |
| 主要风险 | 真实机器人成功率差异的统计显著性未报告；只在桌面操作和有限 backbone 上验证；跨域迁移时 ρ 与窗口校准的稳定性待测 |

💡 **X-Ray 开场**

这篇论文解决的是 VLA 闭环部署中的“每步视觉 token 太多”问题：高分辨率、多视角图像被编码成数百个 visual tokens，主导 LLM backbone 的计算和延迟。它发现两件事：第一，对控制真正重要的 token 由任务决定，而不是通用视觉显著性；第二，不同控制步需要保留的信息量差别很大，固定压缩率会要么浪费计算、要么丢掉关键 token。对 VLA 研究者来说，这意味着 token 压缩不只是“剪掉冗余”，而是“按步分配预算”的在线决策问题。

📍 **研究全景时间线**

```
2022  RT-1 / SayCan：语言条件机器人策略与技能组合
  → 2023  RT-2 / PaLM-E / LIBERO：VLA 预训练与长程操作基准
  → 2024  OpenVLA / CogACT / π0 / FastV / SparseVLM：开源 VLA 与 VLM token 压缩
  → 2025  OpenVLA-OFT / VLA-Cache / LightVLA / SpecPrune-VLA / ADP：VLA 专用加速
  → 2026  FastVLA（本文，arXiv 2026-10-15）：training-free、任务条件化、时间自适应压缩
```

时间线只列论文引用中可确认的节点；本文的位置是“把 VLA token 压缩从静态策略推进到 per-step 自适应预算”。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| task-to-vision scorer | task tokens T（M×d）、visual tokens V（N×d） | 每个 visual token 的重要性 s_i | 每个 action step | 推理时计算，无训练 |
| inverse-Simpson estimator | top-K 的 s_i | 有效 token 数 K_eff、均匀度 U_t | 每个 step | 推理时计算 |
| sliding-window calibrator | U_t、窗口 W（长度 10） | 相对位置 α_t、压缩率 ρ_t | 每个 step | 推理时计算，维护轻量队列 |
| token selector | V、ρ_t | 保留的 visual tokens | 每个 step | 硬剪枝，无缓存 |
| LLM backbone | 保留 tokens + task tokens | action（或 action chunk） | 每个 step | 预训练/微调后冻结，不因 FastVLA 重训 |

FastVLA 属于 token pruning 家族：每个 action step 做一次 task-conditioned top-K 选择，但 K 由自适应压缩率决定，而不是固定值【作者主张】。它与 token merging、reuse/caching 正交，论文也明确说没有 cache、没有跨步状态【作者主张】。

### 1.2 关键机制 (Key Mechanism)

**（1）为什么用 task-to-vision attention 打分**

论文比较了三种信号：all-to-vision（融合表示对所有视觉 token 的注意力）、task-to-vision（任务 token 对视觉 token 的注意力）、vision-to-vision（视觉自注意力）。在 LIBERO + OpenVLA-OFT 上，task-to-vision 在 10%–50% 压缩下成功率接近饱和，70% 以上仍保留非平凡性能；vision-to-vision 和 all-to-vision 退化更快，90% 压缩时接近失败【论文实测，Fig. 2】。作者据此选择 task-to-vision【作者主张】。

**（2）为什么用 inverse Simpson，而不是 Shannon 熵**

论文的观察是：数百个 visual tokens 中有大量接近零的长尾分数，归一化 Shannon 熵的变化范围很窄，难以映射到不同的压缩率；inverse Simpson 强调平方权重，报告“有效主导 token 数”，动态范围更大【作者主张，Fig. 4】。消融中，在相同打分、预算和校准下，inverse-Simpson 在 LIBERO-Goal 和 LIBERO-Long（表中记作 10）上比 Shannon 高 2.3–3.0 点，比 1−Gini 高 1.7–2.0 点【论文实测，Table 6】。

**（3）为什么需要滑动窗口校准**

直接用 U_t 的绝对阈值不可迁移：不同 suite、不同 backbone 的 U_t 范围会漂移，在一个设置上拟合的锚点会在另一个设置上错误分配预算【作者主张】。FastVLA 用最近 |W|=10 步的 U 极值做 min–max 归一化，得到 α_t，再映射到 [ρ−τ, ρ+τ] 的压缩率。这样，当前步比近期更集中（α 小）就压缩更狠，当前步更需要广上下文（α 大）就压缩更保守【作者主张】。

**（4）为什么是有界线性映射**

映射 `ρ_t = clip(ρ + τ(1 − 2α_t), ρ−τ, ρ+τ)` 保证压缩率不会偏离目标 ρ 超过 τ。ρ 控制平均压缩率，τ 控制允许的步间波动。论文固定 τ=0.1、|W|=10，ρ 按 suite/backbone 设定【论文实测，Table 1/2 注释】。这个设计把“自适应”限制在一个可控带内，避免某一步过度剪枝或过度保留【条件性分析】。

⚡ **Eureka Moment**：每步该保留多少 visual token，不应由固定压缩率决定，而应由“当前任务-视觉注意力分布的集中度相对于近期历史的偏移”决定。

### 1.3 信息流/架构图 (Flow / Diagram)

```
task tokens T ──┐
                ├─► task-to-vision attention ─► s_i ─► TopK(K) ─► q_j ─► U_t
visual tokens V ┘                                                      │
                                                                       ▼
                             sliding window W（|W|=10）─► U_min, U_max ─► α_t
                                                                       │
                                                                       ▼
                                                         ρ_t = clip(ρ + τ(1−2α_t), ρ−τ, ρ+τ)
                                                                       │
                                                                       ▼
                                                    保留 ceil((1−ρ_t)·N) 个 visual tokens
                                                                       │
                                                                       ▼
                                                        LLM backbone ─► action
```

注意图中 TopK(K) 的作用是估计 U_t，K 是压缩带上限对应的最小保留预算；最终保留数由 ρ_t 决定，可能大于 K。论文的 Algorithm 1 也明确：先算 TopK 的 U，再映射到 ρ_t。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
每步保留多少 token，取决于任务→视觉注意力分布的 inverse-Simpson 集中度在近期窗口中的相对位置
```

**目标**：在不训练、不加缓存的前提下，每个 action step 选择一个 visual token 子集，降低 LLM backbone 的计算量，同时尽量保持 action 成功率。

**（1）任务条件化的 token 重要性**

```
S = (T_norm · V_norm^T) / sqrt(d)
A = softmax_over_visual(S)
s_i = (1/|Q|) * Σ_{q∈Q} A_{q,i}
```

其中 T_norm、V_norm 是 RMSNorm 后的 task/visual 表示；A 的 softmax 在 visual token 维度上做；Q 是参与平均的 task token 子集。

**（2）有效 token 数与均匀度**

```
v = TopK({s_i}, K)
q_j = v_j / Σ_{m=1..K} v_m
K_eff = 1 / Σ_{j=1..K} q_j^2
U = K_eff / K
K = ceil((1 - (ρ + τ)) * N)
```

U∈(0,1]：U 小表示少数 token 主导，U 大表示分布更均匀。K 是压缩带上限（ρ+τ）对应的最小保留预算；论文指出 K 在一次运行内固定，U 只通过校准进入决策，而校准对常数缩放不变，因此不会把“集中度”和“预算”混淆【作者主张】。

**（3）滑动窗口校准与每步压缩率**

```
α_t = clip((U_t - U_min) / (U_max - U_min), 0, 1)
ρ_t = clip(ρ + τ * (1 - 2α_t), ρ - τ, ρ + τ)
保留 token 数 = ceil((1 - ρ_t) * N)
```

U_min、U_max 是窗口 W 内的极值。α_t 小（比近期更集中）→ ρ_t 更大 → 保留更少；α_t 大（更均匀）→ ρ_t 更小 → 保留更多。初始化阶段 t<2 时直接用 ρ−τ【论文实测，Algorithm 1】。

**变量说明**

| 符号 | 含义 |
|------|------|
| T, V | task tokens、visual tokens |
| M, N, d | task token 数、visual token 数、维度 |
| A | task-to-vision attention 权重 |
| s_i | 第 i 个 visual token 的重要性 |
| K | TopK 预算，取 ceil((1−(ρ+τ))N) |
| q_j | TopK 分数的归一化权重 |
| K_eff | 有效主导 token 数 |
| U | 均匀度，K_eff/K |
| W | U 的滑动窗口，论文固定长度 10 |
| α | 当前 U 在窗口中的相对位置 |
| ρ | 目标压缩率 |
| τ | 允许的压缩率偏差，论文固定 0.1 |
| ρ_t | 第 t 步实际压缩率 |

**直觉**：如果当前步只有少数 token 被任务强烈关注，U 小，α 小，ρ_t 靠近 ρ+τ，剪得更狠；如果当前步需要广上下文，U 大，α 大，ρ_t 靠近 ρ−τ，保留更多。ρ 控制长期平均压缩率，τ 控制步间波动范围。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，仅用于验算公式，不代表论文实验。设定 N=100，ρ=0.6，τ=0.1，|W|=10，K=ceil((1−0.7)×100)=30。假设已过 t<2 初始化，窗口极值为 U_min=0.10、U_max=0.90。

**Step A：高度集中**

假设 TopK 的 30 个分数为：3 个 0.30，27 个 0.0037037，总和 1.0。

```
q_high = 0.30 / 1.0 = 0.30
q_low  = 0.0037037 / 1.0 = 0.0037037
Σq² = 3×0.30² + 27×0.0037037²
    = 3×0.09 + 27×0.00001372
    = 0.27 + 0.0003704 = 0.27037
K_eff = 1 / 0.27037 = 3.699
U = 3.699 / 30 = 0.1233
α = (0.1233 − 0.10) / (0.90 − 0.10) = 0.0233 / 0.80 = 0.0291
ρ_t = 0.6 + 0.1×(1 − 2×0.0291) = 0.6942
保留 = ceil((1 − 0.6942)×100) = ceil(30.58) = 31
```

**Step B：中等集中**

假设 10 个 0.10，20 个 0.01，总和 1.2。

```
q_high = 0.10 / 1.2 = 0.08333
q_low  = 0.01 / 1.2 = 0.008333
Σq² = 10×0.08333² + 20×0.008333²
    = 10×0.006944 + 20×0.00006944
    = 0.06944 + 0.0013888 = 0.07083
K_eff = 1 / 0.07083 = 14.12
U = 14.12 / 30 = 0.4707
α = (0.4707 − 0.10) / 0.80 = 0.4634
ρ_t = 0.6 + 0.1×(1 − 0.9268) = 0.6073
保留 = ceil(39.27) = 40
```

**Step C：接近均匀**

假设 30 个 0.033333，总和 1.0。

```
q = 0.033333
Σq² = 30×0.033333² = 30×0.001111 = 0.03333
K_eff = 1 / 0.03333 = 30
U = 30 / 30 = 1.0
α = (1.0 − 0.10) / 0.80 = 1.125 → clip 到 1
ρ_t = 0.6 + 0.1×(1 − 2) = 0.5
保留 = ceil(50) = 50
```

**验算结论**：集中步保留 31 个 token，中等步 40 个，均匀步 50 个。压缩率始终在 [0.5, 0.7] 内，符合 ρ±τ 的约束。这说明自适应率确实把预算从“均匀步”挪到“集中步”，而不是全局固定 40 个。

## 4. 工程视角 (Engineering View)

| 指标 | 论文实测 | 工程含义 |
|------|----------|----------|
| LLM FLOPs | OpenVLA：1.86→1.20 T（−35.5%）；OpenVLA-OFT：3.98→1.65 T（−58.5%）；CogACT Visual Matching：1.85→1.16 T（−37.3%）；Variant Aggregation：1.81→1.13 T（−37.6%） | 理论计算下降明显，但 FLOPs 不等比例转化为延迟 |
| Action latency | OpenVLA：186.64→161.53 ms（1.16×）；OFT：118.52→81.17 ms（1.46×）；SIMPLER：174.48→159.59 ms（1.09×）、174.94→157.06 ms（1.11×） | 每步 action prediction 时间，包含 LLM backbone 与相关开销 |
| Control frequency | OFT：67.50→98.56 Hz；OpenVLA：5.36→6.19 Hz；CogACT：5.73→6.27 Hz、5.72→6.37 Hz | 定义为 H/latency；OFT 的 H=8（action chunk），OpenVLA/CogACT 的 H=1 |
| CUDA latency | OFT：65.99→40.39 ms；SIMPLER：82.53→66.10 ms、82.74→65.34 ms | GPU 上 LLM backbone 的端到端推理时间 |
| 真实机器人 | Pick：121.13→96.90 ms（1.25×），66.04→82.56 Hz；Pick and place：120.01→97.77 ms（1.23×），66.67→81.82 Hz | 每任务每条件 50 次 rollouts，交替 baseline/FastVLA |

**工程含义**

1. **FLOPs 与延迟不是一回事**：OpenVLA 的 FLOPs 降 35.5%，但延迟只快 1.16×；OFT 的 FLOPs 降 58.5%，延迟快 1.46×。这说明还有视觉编码、attention 打分、数据传输、kernel 利用率等非 LLM 开销【条件性分析】。论文没有分解这部分开销，部署前应单独 profile。

2. **task-to-vision 打分本身有成本**：每个 step 要算 M×N 的 attention。短任务文本下 M 很小，但 N 数百时仍不可忽略；论文没有报告这部分开销的单独数字【条件性分析】。如果任务描述很长或多视角 token 更多，打分开销占比会上升。

3. **无缓存是优点也是限制**：没有跨步状态和 cache 管理开销，所以 wall-clock 收益更接近 FLOPs 收益；但也没有利用时间冗余。论文把 reuse/caching 视为正交方向，结论里提到可与 adaptive pruning 组合【作者主张】。

4. **ρ 需要按 suite/backbone 调**：OFT 上 ρ=0.8/0.7/0.75/0.6，OpenVLA 上 ρ=0.35/0.5/0.4/0.3，SIMPLER 上 ρ=0.7/0.5/0.3/0.3【论文实测，Table 1/2 注释】。跨域迁移时，ρ 可能需要重新选；τ=0.1、|W|=10 固定，但未必对所有场景最优【条件性分析】。

5. **真实机器人成功率的统计显著性**：Pick 84.0%→80.0%，Pick and place 68.0%→66.0%，每任务每条件 50 次 rollouts。按二项分布粗算，80% 附近 50 次的标准误约 5.7 个百分点，4 个百分点的差异不能仅凭 50 次就断定“无退化”【条件性分析】。论文说“comparable success”，但没有给置信区间或配对检验。

6. **抖动与 P99 延迟**：每步 ρ_t 在 [ρ−τ, ρ+τ] 内变化，保留 token 数会步间波动。论文没有报告 token 数抖动或延迟 P99【条件性分析】。对控制频率敏感的部署，建议监控每步 token 数和延迟分布，而不只看平均值。

## 5. 数据与评测 (Data & Eval)

**模型与基准**

| 模型 | 基准 | 任务/设置 | 备注 |
|------|------|-----------|------|
| OpenVLA | LIBERO | Spatial、Object、Goal、Long（表中记作 10） | 单步 H=1 |
| OpenVLA-OFT | LIBERO | 同上 | action chunk H=8 |
| CogACT-base | SIMPLER | Visual Matching、Variant Aggregation；Pick Coke Can、Move Near、Open/Close Drawer、Put Apple in Drawer | 单步 H=1 |
| OpenVLA-OFT + LoRA | 真实 Franka Research 3 | Pick、Pick and place | 每任务每条件 50 次 rollouts，交替 baseline/FastVLA |

**评测指标**

- Success Rate（SR）：任务成功率。
- FLOPs：LLM backbone 的理论计算量，硬件无关。
- Latency：每步 action prediction 的总时间。
- Control frequency：H/latency，H 为一次预测产生的 action 数（OFT 为 8，OpenVLA/CogACT 为 1）。
- CUDA latency：GPU 上 LLM backbone 的端到端推理时间。
- Speedup：per-backbone baseline latency 与各方法 latency 的比值。

**统计与超参**

- 仿真 SR 为三次随机种子的 mean ± standard error【论文实测，Table 1/2 注释】；Table 4/5/6 同样三次种子。
- 真实机器人没有报告 standard error；每任务每条件 50 次 rollouts。
- 固定超参：τ=0.1，|W|=10；ρ 按 suite/backbone 设定。
- 硬件：单卡 NVIDIA RTX 4090。
- 基线中 VLA-Cache 和 LightVLA 是论文可复现的；FastV、DivPrune、SparseVLM、FlashVLA、EfficientVLA、SpecPrune-VLA、ADP 中部分无法在作者环境中复现，表中标 * 并引用原文结果【论文实测，Table 1 注释】。

**关键结果**

- LIBERO OpenVLA：76.4%→75.8% 平均 SR，FLOPs 1.86→1.20 T，latency 186.64→161.53 ms，1.16×【论文实测，Table 1】。
- LIBERO OpenVLA-OFT：97.0%→96.5%，FLOPs 3.98→1.65 T，latency 118.52→81.17 ms，1.46×；control frequency 67.50→98.56 Hz【论文实测，Table 1】。
- SIMPLER Visual Matching：77.5%→77.1%，FLOPs 1.85→1.16 T，latency 174.48→159.59 ms，1.09×；Variant Aggregation：60.9%→61.9%，1.81→1.13 T，174.94→157.06 ms，1.11×【论文实测，Table 2】。
- 真实机器人 Pick：84.0%→80.0%，121.13→96.90 ms，1.25×；Pick and place：68.0%→66.0%，120.01→97.77 ms，1.23×【论文实测，Table 3】。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么**

- 在三种 backbone（OpenVLA、OpenVLA-OFT、CogACT）上即插即用，不需要重训【作者主张，论文实测支持】。
- 每个 action step 独立剪枝，无缓存、无跨步状态，工程上更容易接入现有推理栈【作者主张】。
- 在任务相关区域集中时更激进压缩，在需要广上下文时保留更多 token；消融显示自适应比同预算固定策略高 0.9–1.3 点【论文实测，Table 4】。
- 真实机器人上降低 20% 端到端延迟、提高最多 25% 控制频率【论文实测，Table 3】。

**不能做什么 / 失败模式**

1. **任务描述模糊或多目标**：task-to-vision attention 可能分散，U 变大，压缩变保守，加速幅度下降；如果 attention 误聚焦，可能保留错 token【条件性分析】。
2. **需要全局场景理解的步骤**：定位、验证场景等步骤如果被误判为“集中”，可能过度压缩，丢掉必要的上下文【条件性分析】。
3. **快速运动、遮挡、多物体**：注意力图可能噪声大，窗口 min–max 校准可能被异常值影响。论文说 U(t) 漂移平滑，所以 min–max 的离群敏感性未被触发；如果观测噪声大，rank 校准是可替代方案【作者主张，Table 5 只显示两者差异 ≤0.6 点】。
4. **ρ 的跨域迁移**：ρ 按 suite/backbone 设定，换机器人、换相机、换任务分布时可能需要重新选；τ=0.1、|W|=10 固定，未必最优【条件性分析】。
5. **验证范围有限**：只在桌面操作、LIBERO/SIMPLER 和 Franka Research 3 上验证；没有移动、双臂、人形、多机器人协作【论文实测范围】。
6. **真实机器人统计显著性**：50 次 rollouts 下，4 个百分点的成功率差异不能直接断言无退化；需要更多 rollouts 或配对检验【条件性分析】。
7. **attention 打分开销未分解**：论文没有单独报告 task-to-vision attention 的耗时；在任务 token 多、视觉 token 更多的场景，打分开销占比可能上升【条件性分析】。

### 6.1 隐含假设 (Hidden Assumptions)

- 任务文本包含决策所需的目标信息；如果任务描述缺失或歧义，task-to-vision 信号会变弱。
- 选定层/头上的 task-to-vision attention 能可靠反映决策重要性；论文没有系统搜索层/头，只比较了三种信号类型。
- 每步重要性分布相对近期历史平滑变化，滑动窗口的 min–max 校准才有意义；如果 U(t) 剧烈跳变，窗口极值会滞后。
- 压缩后的 token 子集足以支持 action prediction；论文用成功率间接验证，但没有分析被丢弃 token 的信息内容。
- τ=0.1、|W|=10 在不同 suite/backbone 上可迁移；论文固定了这两个值，但没有做 τ/|W| 的敏感性扫描。
- 成功率是主要指标；延迟和频率的收益不改变闭环稳定性。真实机器人只报告了均值和 50 次 rollouts，没有报告方差。

## 7. 与相关工作对比 (Comparison)

| 方法 | 类型 | 是否训练 | 是否用缓存 | 预算是否自适应 | 代表结果（论文 Table 1/2） | 备注 |
|------|------|----------|------------|----------------|----------------------------|------|
| OpenVLA baseline | — | 预训练 | 否 | — | 76.4% avg，1.86 T，186.64 ms，1.00× | 单步 H=1 |
| + VLA-Cache | reuse/caching | 否 | 是 | 否 | 76.0%，1.36 T，203.37 ms，0.92× | OpenVLA 上反而更慢 |
| + FastVLA | pruning | 否 | 否 | 是 | 75.8%，1.20 T，161.53 ms，1.16× | 本文 |
| OpenVLA-OFT baseline | — | 微调 | 否 | — | 97.0% avg，3.98 T，118.52 ms，1.00× | action chunk H=8 |
| + VLA-Cache | reuse/caching | 否 | 是 | 否 | 97.1%，3.05 T，99.83 ms，1.19× | 成功率高，但 FLOPs/延迟高于 FastVLA |
| + FastVLA | pruning | 否 | 否 | 是 | 96.5%，1.65 T，81.17 ms，1.46× | 本文 |
| + LightVLA | learned pruning | 是 | 否 | 否 | 96.5%，1.21 T，72.41 ms，1.64× | 效率前沿更好，但需 per-backbone 重训 |
| + FastV* | VLM pruning | 否 | 否 | 否 | 93.3%，2.27 T，1.44× | * 引用原文，未复现 |
| + DivPrune* | VLM pruning | 否 | 否 | 否 | 89.4%，2.15 T，1.46× | * 引用原文 |
| + SparseVLM* | VLM pruning | 否 | 否 | 否 | 95.6%，3.06 T，1.28× | * 引用原文 |
| + EfficientVLA* | VLA 加速 | 否 | 部分 | 否 | 88.9%，1.39 T，1.52× | * 引用原文 |
| + SpecPrune-VLA* | VLA pruning | 否 | 否 | 否 | 96.1%，1.71 T，1.46× | * 引用原文 |
| + ADP* | VLA pruning | 否 | 否 | 否 | 96.3%，3.24 T，1.23× | * 引用原文 |
| CogACT-base（SIMPLER VM） | — | 微调 | 否 | — | 77.5%，1.85 T，174.48 ms，1.00× | 单步 H=1 |
| + FastVLA（SIMPLER VM） | pruning | 否 | 否 | 是 | 77.1%，1.16 T，159.59 ms，1.09× | 本文 |
| CogACT-base（SIMPLER VA） | — | 微调 | 否 | — | 60.9%，1.81 T，174.94 ms，1.00× | 单步 H=1 |
| + FastVLA（SIMPLER VA） | pruning | 否 | 否 | 是 | 61.9%，1.13 T，157.06 ms，1.11× | 本文 |

**对比结论**

- 与 VLA-Cache 相比，FastVLA 在 OpenVLA 上更快（1.16× vs 0.92×），在 OpenVLA-OFT 上 FLOPs 更低、延迟更小；VLA-Cache 的成功率略高，但 wall-clock 收益受缓存管理影响【论文实测】。
- 与 LightVLA 相比，FastVLA 的效率前沿略差（1.65 T/81.17 ms vs 1.21 T/72.41 ms），但 LightVLA 需要训练且绑定 backbone；FastVLA training-free、可跨三个 backbone 和真实机器人【论文实测 + 作者主张】。
- 与 VLM 通用剪枝（FastV、DivPrune、SparseVLM）相比，FastVLA 在 OpenVLA-OFT 上成功率最高（96.5% vs 89–96%），FLOPs 更低【论文实测，Table 1】；但其中部分基线是引用原文，不是同一环境复现，比较需谨慎【条件性分析】。

**面试 Tip**：被问到 FastVLA 和普通 attention pruning 的区别，一句话答：“普通 pruning 是固定预算选 token；FastVLA 先用 task-to-vision attention 打分，再用 inverse-Simpson 集中度加滑动窗口校准，在线决定每步预算，所以它是 training-free 的 per-step adaptive pruning，不是静态剪枝。”

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 VLA 推理加速、token pruning、adaptive computation 的研究者或工程师；
  2. 要在现有 OpenVLA/OpenVLA-OFT/CogACT 推理栈上做低侵入加速的部署工程师；
  3. 需要复现 VLA 加速 baseline、比较 caching/pruning/learned pruning 的人。

- **建议章节路径**：
  先读 §3.3 和 Algorithm 1（核心自适应率与初始化）→ 再看 §4.3 的 Table 1/2/3（效果与单位）→ 再看 §4.4 的 Table 4（证明增益来自“何时花预算”而非“花更多预算”）→ 最后看 §3.2 和 Fig. 2（为什么 task-to-vision 比另外两种信号稳）。§2 相关工作可在熟悉 VLA 加速后跳过。

- **不值得精读的理由**：
  如果你不做机器人学习、不关心推理效率，或者已经熟悉 VLM token pruning 且只想知道结论，读摘要和 Table 1/2 即可；如果你需要移动/双臂/人形平台的验证，本文没有覆盖，精读也不会给出答案。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2610.22146