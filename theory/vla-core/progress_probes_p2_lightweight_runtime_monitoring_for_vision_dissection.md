# Progress Probes P²：用 VLA 内部激活做轻量运行时监控 (Progress Probes P²: Lightweight Runtime Monitoring for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-24
>
> **论文**: Progress Probes P²: Lightweight Runtime Monitoring for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2610.31613
> **核心定位**: 用成功示范训练线性探针，从 π0.5 的内部激活解码“任务进度”，再用进度偏差做 runtime monitor；不训练 failure classifier，也不调用外部 VLM。

> **来源与版本说明**：本文依据 arXiv:2610.31613v1（2026-10-22）的 HTML 全文（对应 PDF 正文）撰写。候选元数据摘要为截断片段，且与正文摘要措辞不完全一致；本文不采用候选摘要补写或替代正文中的主张。HTML 排版中表号与正文引用存在错位，本文按表题和数值定位。论文链接仅使用：https://arxiv.org/abs/2610.31613

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文报告：在 π0.5 上，线性 progress probe 可用成功示范训练，并在仿真与真机上取得有竞争力的 failure detection；P² 无需 failure labels，推理延迟约 2.6×10⁻² ms。 |
| 适合精读 | 如果你在做 VLA 部署监控、runtime failure detection、VLA 可解释性，重点看 §III、§V、§VI-A、§VII。 |
| 可以跳过 | 如果你只关心 VLA 预训练配方或新策略架构，这篇不是策略学习论文，距离中等。 |
| 落地可行性 | 中高：线性探针极轻、可复用主前向激活；但需要任务成功示范、阈值校准，且论文只验证 π0.5。 |
| 主要风险 | “进度=剩余时间”的标签假设较强；只探 VLM backbone；困难任务如 Coffee Bean 上信号弱；跨 VLA 家族未验证。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？VLA 部署后仍会失败，但现有 runtime monitor 往往依赖外部 VLM 或 failure data，成本和复杂度高。它发现了什么？π0.5 的内部激活里存在可线性解码的“任务进度”信号；失败时，解码进度会落后于成功示范的 nominal 进度。对 VLA 研究者意味着什么？失败监控不一定要再训一个大模型；可以把“内部进度读数”做成很薄的运行时模块，但它的可靠性高度依赖任务、阈值和成功示范分布。

📍 **研究全景时间线**

```
2024  DROID 数据集 / Sentinel 等 runtime monitoring 基础
2025  VLABench；π0.5；SAFE；FIPER
2026  Robometer；Progress Probes P²（arXiv:2610.31613v1）← 本文位置
```

本文处在“VLA 内部表征可解释性”与“运行时失败监控”的交叉点：前有 SAFE 用 failure labels 训练内部特征分类器，FIPER 用异常/不确定性信号，Robometer 用大规模 VLM 做 reward/progress；本文则把 progress 作为可从成功示范自动标注的线性目标。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

论文的核心不是改 π0.5 的策略网络，而是在冻结的 VLA 前向激活上挂一个线性探针。下表按论文表 I 的仿真结果整理，延迟为监控器推理延迟，硬件为 NVIDIA RTX A6000。

| 方法 | 输入/信号 | 是否需要 failure labels | 仿真全任务 Episode AUROC | 仿真全任务 State AUROC | 仿真未见任务 Episode AUROC | 仿真未见任务 State AUROC | 延迟 |
|------|-----------|--------------------------|---------------------------|-------------------------|-----------------------------|---------------------------|------|
| Progress Probe P² | VLA 内部激活 | 否，仅成功示范 | 0.909 | 0.788 | 0.900 | 0.794 | 2.6×10⁻² ms |
| SAFE-MLP | VLA 内部激活 | 是 | 0.930 | 0.894 | 0.896 | 0.866 | 1.2×10⁻¹ ms |
| SAFE-LSTM | VLA 内部激活 | 是 | 0.923 | 0.851 | 0.859 | 0.803 | 1.8×10⁻¹ ms |
| FAIL-Detect (logpZ0) | VLA 内部激活 | 否 | 0.904 | 0.888 | 0.634 | 0.571 | 1.7×10⁻¹ ms |
| FIPER (ACE) | 动作分布 | 否 | 0.793 | 0.790 | 0.843 | 0.800 | 1.1×10⁰ ms |
| Robometer FT (progress) | 外部 VLM，10-frame context | 是，使用成功+失败数据 | 0.922 | 0.830 | — | — | 5.5×10² ms |

需要明确：**P² 并非在所有仿真指标上都领先**。按上表，SAFE-MLP 的全任务 Episode/State AUROC 更高；Robometer-FT(progress) 也高于 P²。P² 的差异化优势是“成功示范即可训练”“不需要外部 VLM”“延迟极低”。作者在摘要中的措辞是 competitive with or outperforms；按表值应理解为“有条件地有竞争力”，而不是全面碾压。

**逐对象归属，避免把一组模型的属性混在一起：**

| 对象 | 在论文中的角色 | 架构/输入 | 训练数据/监督 | 时序/记忆机制 |
|------|----------------|-----------|----------------|----------------|
| π0.5 | 被监控的 VLA 策略 | 基于 PaliGemma 的 VLA；输入 o_t=(I_t,q_t,ℓ_t)，输出 h-step action chunk | 仿真在 VLABench 上 fine-tune 5000 条轨迹；真机用 π0.5-DROID checkpoint 再 fine-tune 360 条轨迹 | 策略输出未来 action chunk；但 P² 监控打分不使用历史 recurrent 状态 |
| PaliGemma 3B | π0.5 的 VLM backbone；也是 P²_Pali baseline 的激活来源 | 图像/文本 VLM | 预训练细节未在本文展开 | 无监控记忆设定 |
| Robometer 4B | 外部 VLM progress/success baseline | 外部 VLM；表 I 注明使用 10-frame context | 大规模机器人数据；FT 版本用与 SAFE 相同的成功+失败数据 | 10-frame context 是 Robometer 的设置，不是 P² 的 |
| SAFE-MLP / SAFE-LSTM | 内部特征 failure classifier baseline | VLA 激活；LSTM 有显式时序 | 300 条成功 + 300 条失败轨迹用于监督/校准 | SAFE-LSTM 显式建模时序；SAFE-MLP 无 |
| FIPER (RND-OE) / FIPER (ACE) | 输入级/输出级异常 baseline | 视觉输入 novelty；动作分布 | 无 failure labels | 不是 P² 的记忆机制 |
| P² 线性探针 | 轻量 progress decoder | mean-pooled VLA 激活 z_t^i → scalar ζ_hat | 仅成功示范；线性回归；论文称只调 2K 参数 | 仅用当前 o_t 的激活；无 history window |

### 1.2 关键机制 (Key Mechanism)

P² 的流程可以拆成四步：

1. **给成功轨迹打 progress 标签**：对一条成功轨迹，长度 T，时刻 t 的标签为剩余归一化时间 `ζ_t = 1 - t/T`，从 1 降到 0。
2. **抽取 VLA 激活**：把观测 o_t 输入 π0.5，取第 i 层激活 z_t^i；每个激活有 d×k 形状，论文对 token 做 mean-pool 得到 d 维向量。
3. **训练线性探针**：用成功示范构造 `(z_t^i, ζ_t)` 数据集，线性回归拟合 `ζ_hat = wᵀz_t^i + b`。
4. **运行时打分**：用成功轨迹平均长度 μ_T 定义 nominal 进度 `ζ_nom(t)=1-t/μ_T`；监控分数为 `s = ζ_hat - ζ_nom`。分数为正表示“估计进度落后于 nominal”，超过阈值 δ 就报警。

关键设计是：**不训练 failure classifier，不学习“失败是什么样”，而是学习“成功应该怎么推进”，再把偏离当作异常。** 这使训练标签可以完全来自成功示范，不需要人工标注失败。

⚡ **Eureka Moment**：不要把失败检测做成另一个大模型；把成功示范中的时间进度变成 VLA 激活空间里的线性方向，用“进度落后”作为异常分数。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
离线阶段（仅成功示范）
success trajectories
        │
        ▼
π0.5 forward ──► 第 i 层激活 z_t^i ──► mean-pool tokens ──► (z_t^i, ζ_t)
        │                                                   │
        │                                                   ▼
        └──────────────────────────────────────────► 线性回归 w,b
                                                            │
运行时                                                      ▼
o_t ──► π0.5 forward ──► z_t^i ──► probe ──► ζ_hat(o_t)
                                                │
成功轨迹平均长度 μ_T ──► ζ_nom(t)=1-t/μ_T       │
                                                ▼
                         s(τ_t)=ζ_hat(o_t)-ζ_nom(t)
                                                │
                                      s > δ ? 报警 : 继续
```

注意图中的“未来 action chunk”没有进入监控分数。π0.5 输出 `a_t...a_{t+h}` 是动作预测 horizon；P² 打分只用当前观测 o_t 的激活。不能把动作 chunk 长度 h 直接当作历史上下文或记忆窗口。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```text
s(τ_t) = ζ_hat(z_t^i) - [1 - t/μ_T]；若 s > δ 则报警
```

先给目标：设计一个 runtime monitor `s(τ_t)`，既降低误报，又尽早发现失败。论文把监控目标写成多目标优化：

```text
min_s  E_τ[ FPR(δ,τ;s) + λ·DetectTime(δ,τ;s) ]
```

其中：

| 符号 | 含义 |
|------|------|
| τ_t | 截至时刻 t 的观测-动作轨迹 `(o_0,a_0,...,o_t)` |
| H | 最大任务 horizon |
| Y | rollout 结果，Y=1 成功，Y=0 失败 |
| s(τ_t) | 监控分数，越大表示越偏离成功/名义数据 |
| δ | 报警阈值 |
| T_δ | 首次报警时刻 `inf{t | s(τ_t)>δ}`，从未报警则为 ∞ |
| FPR | 成功轨迹在 H 内被误报的概率 |
| DetectTime | 失败轨迹首次报警时刻的期望 |
| λ | 误报与检测时间的权衡系数 |

Progress 标签：

```text
ζ_t = 1 - t/T
```

探针数据集：

```text
D_probe = {(z_t^0,n, z_t^1,n, ..., z_t^L,n, ζ_t^n)}
```

线性探针：

```text
φ(z^i) = wᵀz^i + b
ζ_hat(o) = φ(z^i)
```

运行时分数：

```text
ζ_nom(t) = 1 - t/μ_T
s(τ_t) = ζ_hat(o_t) - ζ_nom(t)
```

直觉：成功执行时，ζ_hat 应沿着 nominal 曲线下降；如果策略卡住、重复动作或走错分支，ζ_hat 下降变慢，分数变正。负分数表示“比预期更快”，论文明确不惩罚更快执行。

**时序口径必须说清楚。** 论文的监控输入是截至 t 的轨迹 τ_t，但 P² 的分数只读取当前 o_t 的激活；它不是 recurrent monitor，也没有把过去若干帧拼成上下文。均匀间隔采样时：

```text
K 个采样帧的首末跨度 = (K-1)·Δt
从当前参考 t 预测 t+1 至 t+H，终点距当前参考 = H·Δt
```

这两条是时间轴换算，不是 P² 的记忆窗口。尤其注意：连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口；论文中 π0.5 的 h-step action chunk 是动作预测 horizon，不等于 P² 的监控历史长度。Robometer 表 I 注明的 10-frame context 也只属于 Robometer 的设置。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设例子，用于验算分数方向与阈值行为，不是论文实际轨迹。

假设某任务成功轨迹平均长度 μ_T=100 步，阈值 δ=0.15。当前时刻 t=50，则 nominal 进度：

```text
ζ_nom(50) = 1 - 50/100 = 0.50
```

情形 A：成功执行，探针输出 ζ_hat=0.51：

```text
s = 0.51 - 0.50 = +0.01 < 0.15  → 不报警
```

情形 B：策略卡住，探针输出 ζ_hat=0.70：

```text
s = 0.70 - 0.50 = +0.20 > 0.15  → 报警
```

报警发生在 episode 的 50% 处；论文的 detection time 就是用“首次报警时 episode 已过比例”来统计的。情形 C：策略比 nominal 更快，ζ_hat=0.40：

```text
s = 0.40 - 0.50 = -0.10 < 0.15  → 不报警
```

这与论文“不惩罚更快执行”的设计一致。再检查早期时刻 t=25：ζ_nom=0.75；若 ζ_hat=0.80，则 s=0.05，仍不报警。这个例子说明：P² 的报警不是“看到某个失败类别”，而是“当前解码进度比成功示范的 nominal 进度落后足够多”。

## 4. 工程视角 (Engineering View)

| 工程维度 | 论文事实 | 工程含义 |
|----------|----------|----------|
| 额外计算 | 线性探针，论文称只调 2K 参数；P² 延迟 2.6×10⁻² ms | 若已有 VLA 前向激活，监控器本身接近“免费”；主要工程成本在激活 hook、mean-pool 和阈值管理 |
| 与外部 VLM 对比 | Robometer 延迟 5.5×10² ms；作者引言称快 5× 到 20,000× | 按表值 P² 对 Robometer 约 2.1×10⁴×，但这是特定硬件/实现下的监控器延迟，不是端到端机器人控制周期 |
| 策略是否重训 | 不重训策略；只训练线性探针 | 部署时风险低，适合已有 π0.5 checkpoint 上加挂 |
| 监督条件 | 仅成功示范 | 不需要失败数据；但需要与部署任务分布匹配的成功轨迹 |
| 阈值 | 用 20% 验证 split 校准，遵循 FIPER 做法 | 阈值是部署参数；任务、相机、物体分布变化后需重新校准 |
| 时序记忆 | P² 只用当前激活；SAFE-LSTM 显式时序；Robometer 用 10-frame context | 不能把 P² 说成“有时序记忆”；也不能把 Robometer 的 10-frame 设置移植给 P² |
| 硬件 | Franka Research 3 + DROID camera setup；RTX A6000 测延迟 | 论文验证了真机，但只限桌面操作任务与特定相机 setup |

工程上最需要注意的是 **μ_T 的定义与任务边界**。论文公式写 `μ_T = 1/N Σ T^n`，没有在正文中明确说明是否按任务分别估计。若不同任务时长差异很大，全局平均 nominal 曲线会把“任务间差异”误当成“进度偏差”。我的条件性分析是：落地时应按任务或任务簇分别校准 μ_T 与 δ，并在任务切换时重置监控状态；这不是论文已做的实验结论，而是工程化建议。

另外，P² 的低延迟来自“复用已计算激活”。如果部署框架不暴露中间层激活，或者需要把激活从 GPU 拷回 CPU，实际开销会高于表内线性层延迟。论文的 latency 应理解为监控器实现本身的测量，不应直接等同于整条机器人控制回路的延迟。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与模型范围

论文没有展开 π0.5 的预训练数据配比；可确认的是：π0.5 是论文称为 open-source 的 VLA，其 VLM backbone 是 3B PaliGemma。仿真中 π0.5 在 VLABench 上 fine-tune；真机使用 π0.5-DROID checkpoint，再在 6 个硬件任务上 fine-tune。作者在 limitation 中明确：实验聚焦 π0.5，未来才考虑 GR00T、MolmoAct 等不同 VLA 家族；也主要探 VLM backbone，尚未系统研究 diffusion/flow-matching action expert 的内部激活。因此，不能把本文结论外推到所有 VLA 架构。

数据开放范围也要逐项归属：VLABench 是仿真 benchmark；DROID 是硬件相机 setup 与数据集来源；StressDream 是硬件任务轨迹来源；Robometer 是外部 VLM baseline；SAFE 是 failure classifier baseline。它们不是同一个“开放数据包”，不能混写。

### 5.2 仿真设置

| 项目 | 论文设置 |
|------|----------|
| 环境 | VLABench |
| 策略 | π0.5 |
| 策略 fine-tune | 5000 条轨迹，10 个任务 × 500 条/任务 |
| 监控评测任务 | 6 个 VLA 有非平凡成功率的任务 |
| 探针训练 | N=300 条轨迹，6 个任务 × 50 条/任务；80/20 train-validation |
| 阈值校准 | 20% validation split，遵循 FIPER |
| 评测 rollout | 600 条，6 个任务 × 100 条/任务；成功/失败各半 |
| 失败标注 | 人工标失败发生时刻，之后所有 timestep 标为失败，用于 state-level 评测 |
| 未见任务 | 3 个 seen + 3 个 unseen，遍历 C(6,3)=20 个 split；Robometer 因主要零样本使用被排除 |

指标包括 Episode AUROC、State AUROC、TPR@5%FPR、DetectTime（episode 已过比例）。论文报告 P² 在全任务 Episode AUROC 0.909、State AUROC 0.788；未见任务 Episode 0.900、State 0.794。检测时间上，P² 在 75% episode 时 TPR@5%FPR 为 0.54，100% 时为 0.64，平均 detection time 0.60 episode；SAFE-LSTM 平均 0.53，100% 时 0.72。也就是说，P² 的检测时间有竞争力，但不是最快/最强。

### 5.3 真实机器人设置

| 项目 | 论文设置 |
|------|----------|
| 机器人 | Franka Research 3 |
| 相机 | DROID camera setup |
| 策略 | π0.5-DROID checkpoint |
| 策略 fine-tune | 6 个任务 × 60 条/任务 = 360 条轨迹，来自 StressDream |
| 探针训练 | 360 条硬件轨迹；80/20 split；标签同 Eq.2 |
| 评测 rollout | 120 条，6 个任务 × 20 条/任务；成功/失败各半 |
| 失败标注 | 人工标失败发生时刻，之后标为失败，用于 state-level 评测 |
| 未见任务 | 3 seen + 3 unseen，遍历 C(6,3)=20 个 split；Robometer 排除 |

硬件任务包括 Candy Bag、Coffee Bag、Coffee Bean、Knife Put、Block Stack、Stacked Utensil。论文表 IV 报告 P² 全任务 state-level 平均 AUROC 0.78；表 V 报告未见任务平均 0.81。正文特别指出 P² 在 Coffee Bean 这类失败上较弱：失败可能由分布内动作下的随机因素造成，且视觉区域小，VLA 激活变化可能不足以提供强信号。表 IV 中 P² 的 Coffee Bean 为 0.26，是明显的困难案例；表 V 的 unseen 平均 0.81 是另一组 split 结果，不能与表 IV 单任务数值直接混读。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么：**

- 在 π0.5 上，用成功示范训练线性探针，从当前激活解码 progress；
- 在仿真与真机上做 episode-level 和 state-level failure detection；
- 在无需 failure labels 的条件下，取得与 SAFE、Robometer 等基线有竞争力的结果；
- 监控器极轻，适合作为部署时的附加模块，而不是重新训练策略；
- 在论文的 20 个 seen/unseen split 中，硬件未见任务平均 state AUROC 0.81，显示一定的跨任务泛化。

**不能做什么：**

- 不能保证跨 VLA 家族有效；论文只测 π0.5，作者自己把 GR00T、MolmoAct 列为未来工作；
- 不能保证跨 embodiment、跨相机、跨任务分布有效；阈值和 μ_T 都需要重新校准；
- 不能处理“失败但进度不落后”的模式，例如更快但错误的执行、进度标签不单调的任务、需要分支/回退的任务；
- 不能替代动作级安全约束；它是监控器，不是控制器；
- 在 Coffee Bean 这类视觉细微、激活变化弱的失败上，P² 可能失效；
- 不能把“仿真 0.909/0.788”直接外推到所有真实机器人任务。

### 6.1 隐含假设 (Hidden Assumptions)

以下是编辑视角的批判性分析，不是论文已证明的结论：

1. **成功示范定义 nominal**：探针只从成功轨迹学习，默认“成功轨迹的平均时间进程”就是部署时的正常进程。若部署环境更慢、更拥挤，或任务策略改变，nominal 曲线可能失配。
2. **时间剩余是 progress 的有效代理**：`ζ_t=1-t/T` 假设任务沿时间单调推进。对多阶段、可回退、并行操作的任务，这个标签可能过于粗糙。
3. **失败主要表现为“进度落后”**：监控分数只惩罚正偏差，不惩罚“过快”。若失败表现为错误但快速地完成某个子目标，P² 可能不敏感。
4. **当前激活足够**：P² 不使用历史窗口，默认当前 o_t 的激活已包含判断进度所需信息。对需要长时序上下文的卡死/重复动作，这可能不够。
5. **线性可读**：线性探针假设 progress 在激活空间中近似线性可解码。论文在 π0.5 的 VLM backbone 上验证了这一点，但未证明所有层、所有 VLA 都成立。
6. **阈值可迁移**：阈值在 20% 验证集上校准，默认验证分布与部署分布接近。真实部署中的分布漂移会影响 FPR。
7. **μ_T 的任务边界清晰**：论文公式未明确 μ_T 是否按任务分别估计；工程落地需要确认，否则任务间时长差异会污染分数。

## 7. 与相关工作对比 (Comparison)

| 方法线 | 代表 | 监控信号 | 是否需要 failure labels | 是否需要外部模型 | 时序机制 | 主要优势 | 主要局限 |
|--------|------|----------|--------------------------|------------------|----------|----------|----------|
| 内部特征分类 | SAFE-MLP / SAFE-LSTM | VLA 激活 | 是 | 否 | LSTM 有显式时序 | 监督强时 AUROC 高 | 需要失败数据；LSTM 更重 |
| 外部 VLM 奖励/进度 | Robometer | 外部 VLM 的 progress/success | FT 版本需要成功+失败数据 | 是，4B VLM | 10-frame context | 通用性强，可迁移 | 延迟高；依赖外部模型 |
| 输入/输出异常 | FIPER (RND-OE/ACE) | 视觉 novelty 或动作分布 | 否 | 否 | 视具体实现 | 无需失败标签 | 信号间接，AUROC 波动 |
| 内部激活异常 | FAIL-Detect (logpZ0) | VLA 激活 | 否 | 否 | 无显式时序 | 快，无需失败数据 | 未见任务泛化较弱 |
| 线性 progress probe | P² | VLA 激活的 progress 读数 | 否，仅成功示范 | 否 | 仅当前激活 | 极快、轻量、可解释 | 依赖时间进度假设；只验证 π0.5 |

**面试 Tip**：如果被问“P² 和 SAFE 有什么区别”，一句话答：SAFE 学的是“失败分类器”，需要失败标签；P² 学的是“成功进度解码器”，只用成功示范，把进度落后当异常，因此更轻但更依赖任务进度假设。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 VLA 部署与 runtime monitoring 的研究者/工程师：重点看 §IV 的问题形式化、§V 的监控器设计、§VI-A 的仿真对比、§VII 的真机结果。
  - 做 VLA 可解释性的人：重点看 §III 的线性 probing 设定，以及 §VI-C 对“动作数据是否让表征更适合 progress”的讨论。
  - 想把监控器挂到已有 π0.5 系统上的工程师：重点看 §V 的公式、§VI 的训练/校准细节、§VII 的硬件设置与 Coffee Bean 失败案例。

- **建议章节路径**：先读 §V（P² 到底怎么打分）→ 再看 §III（探针怎么训练）→ 然后 §VI-A/§VII（数字与条件）→ 最后 §VIII limitation。Related Work 可在熟悉 SAFE/FIPER/Robometer 后快速扫过。

- **不值得精读的理由**：如果你不做机器人学习、不关心部署监控，或者你已经在做更强的时序 failure detection，这篇的边际信息主要是“线性 progress probe 可作为轻量 baseline”；读摘要和表 I 即可。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2610.31613