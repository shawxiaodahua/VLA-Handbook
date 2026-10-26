# FocusVLA：未来意图引导的多视角动态路由 (FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-25
>
> **论文**: FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2610.31423
> **核心定位**: 针对多视角 VLA 在操作阶段切换时出现的“视角依赖惯性”，用未来动作意图生成逐动作步的 global/local 软路由权重；在 π0.5 上做非侵入式改造，并在 RoboTwin 2.0 与真机上报告增益。

**来源与版本说明**：本文依据 arXiv:2610.31423v1（2026-10-22）的 HTML 全文及其正文/表格写作；该版本标题、作者、摘要与正文一致。候选元数据摘要为截断片段（止于 “do not fully exploit suc”），未发现与全文摘要的实质冲突，因此未用候选摘要补写任何主张。另需标注一处单位不一致：摘要写 “13.3% gain”，正文 §4.2 写 “13.3 and 25.6 percentage points”；按 71.7→85.0 复算为 13.3 个百分点，本文统一用 pp。所有数字均取自该 v1 全文，不引入额外来源。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：用未来意图驱动 action-side 的 global/local 软路由，可缓解静态多视角融合的阶段-视角错配；仿真平均 76.92%，比最强基线 π0.5 高 10.10 pp；真机 85.0% 对 71.7%。 |
| 适合精读 | 如果你在做多视角 VLA、主动感知、action chunk 内的视觉条件调制，重点看 §3.1 诊断、§3.2 架构、§4.5 消融与 Appendix C-G。 |
| 可以跳过 | 如果你只关心大规模预训练数据配方或通用 VLA  backbone  Scaling，这篇不提供预训练语料细节，距离中等。 |
| 落地可行性 | 中：模块非侵入、可挂在 π0.5 上；但增加 70.13M 参数与约 16.6% 每 chunk 延迟，且真机只验证单臂 Piper + 两相机。 |
| 主要风险 | 诊断与真机覆盖有限；BFA 对比是论文式复现的 BFA-style 对照，不是官方 BFA 结果；跨 embodiment 与未见相机布局未验证。 |

💡 **X-Ray 开场**：这篇论文解决什么问题？多视角 VLA 虽然同时看 global 与 wrist/local 视角，但策略容易在早期形成对某一视角的偏好，之后任务阶段变了也不调整。它发现了什么？在 π0.5 上，失败 rollout 的 local-view 主导更强、随时间变化更小；把 local 视角遮掉比遮 global 掉得更狠。对 VLA 研究者意味着什么？多视角融合不只是“把特征拼起来”，还要让 action generation 在 chunk 内按未来意图逐步骤决定看哪边。

📍 **研究全景时间线**

```
2024  π0：VLA flow model 代表作（论文引用）
2025  RDT-1B / RoboTwin 2.0 / π0.5：双臂 benchmark、多视 VLA 基线成熟（论文引用）
2026  FocusVLA：在 π0.5 上加入 future-intent bridge + action-side view routing ← 本文
局限：真机仅 Piper 单臂 + 两相机、6 任务；跨 embodiment/相机布局未验证
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| π0.5 多视角 VLA backbone | 多视角观测 o_t={I_t^g, I_t^l}、语言 L、机器人状态 s_t | 视觉语言特征、动作生成条件 | 每个 action chunk | 论文称 FocusVLA 不修改其视觉编码器、语言模型与动作生成 backbone；是否冻结未明确 |
| Temporal State-History Encoder | 当前状态 s_t、前 30 个动作及其时间差分 | 2048-D state token、2048-D history token | 每个 chunk | 历史动作编码；不是未来预测窗口 |
| 跨模态注意力 | 语言 token 作 query；视觉/状态历史 token 作 key/value | 上下文表示 | 每个 chunk | 4-head、2048-D |
| Intent head | 上下文表示 | I_t=[i_t^(1)…i_t^(H)]，H=30 | 每个未来 action step 一个概率 | 用轨迹弱标签监督；5K step intent-only warm-up |
| Chunk-level intent token | Pool(I_t) 经 MLP | z_intent | 每个 chunk | 作为动作生成的粗粒度条件 |
| Action-side router | 动作特征作 query；global/local 视觉特征 | c_g^(k)、c_l^(k) | 每个未来 action step | 4-head、1024-D cross-attention |
| 路由融合 | c_g、c_l、α^(k) | c^(k)=α_g c_g+α_l c_l | 每个未来 step | 推理时无 teacher forcing |
| 动作细化 | 融合上下文 + 残差 | 最终动作预测 | 每个 chunk/step | 轻量 refinement |

**逐模型归属边界**（避免把不同模型属性混在一起）：

| 模型/方法 | 本文中的角色 | 架构归属 | 数据/开放范围 | 记忆/时序归属 |
|-----------|--------------|----------|----------------|----------------|
| FocusVLA | 本文方法 | 在 π0.5 上增加 Future Intent Bridge 与 Action-Side Dynamic Routing | 论文未声明代码或数据发布；候选来源 repo_url 为空 | 已观测历史：前 30 个动作；未来：H=30 个未来意图/动作步 |
| π0.5 | 最强基线，同时是 FocusVLA backbone | 多视角 VLA backbone；论文将其作为静态多视角基线分析 | 论文只说明同协议训练与评测，未展开其预训练语料 | 不把 FocusVLA 的 30 步历史/未来窗口归给 π0.5 |
| π0 | 基线 | VLA flow model 基线 | 同协议训练/评测 | 论文未展开其时序记忆 |
| RDT | 基线；其 random view dropout 被用作正则对照 | diffusion foundation model 基线 | 同协议训练/评测 | 论文未展开其多视路由机制 |
| BFA | 相关工作，不是官方直接评测 | 预测当前视角权重并在 policy 前融合 | 论文称 BFA 无可用代码与完整标注规则，故实现 paper-based BFA-style 对照 | 不把 FocusVLA 的逐 step 路由归给 BFA |

### 1.2 关键机制 (Key Mechanism)

- **诊断先行**：作者在 π0.5 上计算 global/local attention ratio，发现失败 rollout 的 Dominance Margin 更大、local-view reliance 的时间变化更小；再用 stage-conditioned view masking 做因果性补强。该诊断是后续设计的动机，不是对所有 VLA 的普适结论。
- **未来意图不是未来视频预测**：Future Intent Bridge 预测的是每个未来动作步“更需要 local interaction 视觉证据”的概率 i_t^(k)，再用 1−i 表示 workspace-level motion 的概率。它是动作意图层面的弱监督，不是像素级未来帧生成。
- **两条意图通路**：一条是逐 step routing weights α^(k)，控制每个未来动作步的 global/local 融合；另一条是 chunk-level intent token z_intent，给动作生成器一个整体任务进度条件。消融显示两者互补。
- **路由发生在 action side**：不在视觉编码器里改多视特征，而是让动作特征分别 attend global/local 视觉特征，再按 α 融合。这样保留预训练视觉 backbone 的稳定性，但增加参数与延迟。
- **有界软路由**：α 被线性映射到 (0.25, 0.75) 附近，两个视角权重和为 1，最多约 3:1 偏向，保留至少约 25% 的互补视角。它不是硬切换相机，而是软调制。
- **弱标签自动构造**：用末端执行器轨迹的平移速度、旋转、路径长度、轴主导性、夹爪闭合事件等启发式规则，生成二值 intent label；不依赖人工阶段标注或离线 VLM 标注。

⚡ **Eureka Moment**：把“未来动作需要哪种视觉证据”变成一个可微的逐 step 路由信号，而不是在视觉编码器里静态拼接多视角；并且用 chunk-level intent token 把任务进度条件注入动作生成。

### 1.3 信息流/架构图 (Flow / Diagram)

```
已观测历史（不是未来预测窗口）
  o_t = {I_t^g, I_t^l}, L, s_t, h_t = 前 30 个动作
        │
        ├─ π0.5 VLA prefix → 视觉语言特征
        └─ Temporal State-History Encoder → state token / history token
                    │
                    ▼
        Future Intent Bridge：语言 query × 视觉/状态历史 key/value
                    │
                    ├─ Intent head → I_t = [i^(1) … i^(H)]，H=30 个未来步
                    │        ├─ Pool + MLP → z_intent（chunk 级条件）
                    │        └─ f_route → α^(k)=[α_g^(k), α_l^(k)]
                    ▼
        Action backbone：noisy action tokens + z_intent + diffusion step
                    │
                    ▼
        action feature q^(k) 分别 attend global/local 视觉特征
                    │
                    ├─ c_g^(k)
                    └─ c_l^(k)
                    ▼
        c^(k) = α_g^(k)c_g^(k) + α_l^(k)c_l^(k)
                    │
                    ▼
        残差注入 + 轻量 refinement → 动作 chunk 预测
```

> 关键边界：图中的 H=30 是未来意图/动作 chunk 的预测长度；前 30 个动作是已观测历史。两者数值相同只是实现巧合，不能把未来 chunk 当作历史上下文或记忆窗口。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
每个未来动作步的视觉上下文 = α_g(i)·c_g + α_l(i)·c_l，α 由未来意图概率 i 经有界线性映射得到
```

**目标**：先诊断静态多视融合的阶段-视角错配，再把未来意图概率转成逐 step 路由权重，最后用联合损失训练动作生成与意图预测。

**诊断指标**：

```
D_t = |a_t^global − a_t^local|,  D = (1/T) Σ_t D_t
L_t = a_t^local,  ΔL_t = L_t − L_{t−1},  V_ΔL = Var({ΔL_t})
```

**意图与路由**：

```
I_t = [i_t^(1), …, i_t^(H)],  i_t^(k) ∈ [0,1]
z_intent = MLP_token(Pool(I_t))
α^(k) = f_route(i_t^(k)) = [α_g^(k), α_l^(k)]
α_g^(k) + α_l^(k) = 1,  0.25 < α < 0.75（实现中的有界范围）
c^(k) = α_g^(k)c_g^(k) + α_l^(k)c_l^(k)
```

**联合训练目标**：

```
L_total = L_act + λ_intent · L_intent
L_act = ||V_t − U_t||_2^2
L_intent = −(1/T) Σ_t [λ_+ y_t log p_t + λ_− (1−y_t) log(1−p_t)]
λ_intent = 1, 正类权重 w_+ = 15
```

> 符号说明：原文在 L_intent 中用 T 表示求和长度，而意图序列长度记为 H；本文保留原文记法，不强行统一，避免把动作步与意图步混为一谈。

| 符号 | 含义 | 备注 |
|------|------|------|
| a_t^global, a_t^local | 第 t 步 global/local 归一化注意力比例 | 两者和为 1 |
| D | Dominance Margin 均值 | 越大表示单视角主导越强 |
| V_ΔL | local-view reliance 差分方差 | 越小表示越不随时间调整 |
| H | 未来意图/动作步数 | 实现中 H=30 |
| i_t^(k) | 第 k 个未来步需要 local interaction 视觉证据的概率 | 弱监督二值标签训练 |
| α_g, α_l | global/local 路由权重 | 非负、和为 1、有界 |
| c_g, c_l | 动作特征 cross-attend 得到的 global/local 上下文 | 逐未来 step |
| λ_+ | 正类损失权重 | 实现中 15 |

**直觉**：作者不是简单追求“local 注意力越高越好”。论文明确指出，成功与失败 rollout 都可能偏向 local；真正有区分度的是失败 rollout 的 local 主导更强且随时间更不调整。因此路由目标不是消灭 local，而是让 local/global 的相对贡献随未来动作需求变化。有界软路由则避免 0/1 硬切换导致互补证据丢失。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设例子，用于验算机制，不是论文实验设置；路由映射也是示意，不声称与作者标量参数完全相同。

**先验算历史与未来窗口**：假设动作步间隔 Δt=0.1 s，已观测历史为 K=30 个均匀间隔动作样本，则首末跨度为：

```
(K−1)×Δt = 29×0.1 = 2.9 s
```

若从当前参考 t 预测 t+1 至 t+H，H=30，则终点距当前参考为：

```
H×Δt = 30×0.1 = 3.0 s
```

两者相差 0.1 s，说明“30 步历史”和“30 步未来”不是同一个时间区间。本文实现中，历史是前 30 个动作，未来意图是 H=30 个未来动作步；不能把未来 chunk 长度直接当作记忆窗口长度。

**路由验算**：设某 chunk 的三个未来步意图概率为 i=[0.2, 0.5, 0.9]。用一个满足有界、和为 1 的示意线性映射：

```
α_g = 0.75 − 0.5·i
α_l = 0.25 + 0.5·i
```

| 未来步 k | i^(k) | α_g | α_l | 验证 |
|----------|-------|------|------|------|
| 1 | 0.2 | 0.65 | 0.35 | 0.65+0.35=1，均在 (0.25,0.75) |
| 2 | 0.5 | 0.50 | 0.50 | 均衡 |
| 3 | 0.9 | 0.30 | 0.70 | 偏 local，但仍保留 30% global |

再假设 c_g=[1,0]、c_l=[0,1] 作为二维示意上下文，则：

```
k=1: c = 0.65×[1,0] + 0.35×[0,1] = [0.65, 0.35]
k=2: c = [0.50, 0.50]
k=3: c = [0.30, 0.70]
```

验算成立：每个未来步的融合向量分量和为 1，α 随 i 单调变化，且有界映射保留互补视角。这个例子说明路由是“逐未来 step 的软加权”，不是在整个 chunk 上共享一个固定视角比例。

## 4. 工程视角 (Engineering View)

| 项目 | 论文给出/可确认 | 工程含义 |
|------|------------------|----------|
| 额外参数 | FocusVLA 比 π0.5 增加 70.13M | 非侵入不等于零成本；显存与加载时间需计入 |
| 每 chunk 延迟 | 92.78 ms 对 79.59 ms，增加 13.19 ms，约 16.6% | batch size=1、单张 A100 的条件；不是端到端控制频率 |
| 意图分支结构 | state token 2048-D；history token 2048-D；4-head 2048-D cross-attention；2048→4096→30 MLP | 参数量主要集中在 bridge 与 cross-attention |
| 路由结构 | 4-head、1024-D cross-attention；逐 future step 融合 | 逐 step 计算增加 action-side 开销 |
| 路由范围 | α 有界，约 (0.25,0.75)，最多约 3:1 | 软切换，保留至少约 25% 互补视角；避免硬 0/1 |
| 端点比例 | 报告精度下 64.19% 的 held-out α_g 严格位于内部，其余四舍五入到 0.25 或 0.75 | 多数样本是软路由，少数是接近端点的自信预测 |
| 训练 | 主训练 100K steps，8×A100 80GB；消融变体 20K steps | 不要把消融的 20K 预算与主结果 100K 混用 |
| 稳定化 | 5K step intent-only warm-up；5K–15K teacher forcing 从 0.77 退火到 0.3，之后保持 0.3；推理关闭 | 训练技巧影响复现；论文未给所有随机种子方差 |

**工程含义**：每 chunk 92.78 ms 不能直接换算成控制频率。实际 replan 频率取决于 action chunk 的执行方式（是否 receding horizon、是否一次执行 30 步、控制器插值等），论文未给出这部分协议。若部署在更低算力平台，16.6% 的 chunk 延迟增量可能被放大；若 batch 大于 1，吞吐与延迟关系也需重测。另一个工程边界是：路由是 action-side 软调制，不是相机选择器，因此不会减少相机采集与视觉编码成本，省掉的是“动作侧对两路视觉上下文的相对权重”需要手工设计的部分。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与训练数据

- **预训练**：论文以 π0.5 多视角 VLA 为 backbone，但没有展开 π0.5 的预训练语料、数据配比与开放范围。因此不能把 π0.5 的预训练数据属性归给 FocusVLA，也不能假设本文发布了新的预训练数据。
- **训练数据**：Appendix B 说明，仿真与真机两个 benchmark 分开训练；每个 benchmark 内每个任务收集 50 条 demonstration trajectories，并在该 benchmark 内联合训练一个多任务策略；两个域之间不混数据。
- **优化设置**：主训练 8 张 NVIDIA A100 80GB，100,000 optimization steps；消融变体统一 20K steps。5K step intent-only warm-up，之后联合优化动作损失与意图损失。
- **开放范围**：候选来源 repo_url 为空；论文正文未声明代码、数据或模型权重发布。因此本文不假设存在官方仓库。

### 5.2 仿真评测

论文在 RoboTwin 2.0 全量 50 个任务上评测，每个任务 100 trials，报告 success rate；基线为 RDT、π0、π0.5。作者称所有模型使用相同 demonstration 数据、相同优化步数、相同测试种子与评测协议，无额外数据或调参。

| 模型 | Overall Avg. | 相对 FocusVLA |
|------|--------------|----------------|
| RDT | 34.76% | 低 42.16 pp |
| π0 | 47.06% | 低 29.86 pp |
| π0.5 | 66.82% | 低 10.10 pp |
| FocusVLA | 76.92% | — |

按任务 horizon 分组（论文 Table 2）：

| 分组 | π0.5 | FocusVLA | 差值 |
|------|------|----------|------|
| Short | 63.10% | 74.00% | +10.90 pp |
| Medium | 54.90% | 67.80% | +12.90 pp |
| Medium-Long | 72.00% | 80.70% | +8.70 pp |
| Long | 77.10% | 85.60% | +8.50 pp |
| Extra-Long / Dexterous | 67.00% | 76.50% | +9.50 pp |

**不能只看平均值**：Table 2 中并非逐任务单调提升，例如 Press Stapler 从 85.0% 升到 91.0%，但 Rotate Qrcode 从 82.0% 降到 66.0%，Open Microwave 从 64.0% 降到 61.0%。这说明方法并非所有任务都稳定提升；论文的“consistent improvements”应理解为分组平均与总体平均，而不是逐任务单调。

### 5.3 真机评测

真机使用 Piper 机械臂，两个 Intel RealSense D435i 相机分别提供 global 与 local 观测；6 个代表性操作任务，每个任务 30 trials。论文报告 FocusVLA 平均 85.0%，π0.5 为 71.7%，提升 13.3 pp；并称比 π0 高 25.6 pp（正文未列 π0 绝对值）。摘要中的 “13.3% gain” 与正文的 “13.3 percentage points” 单位不一致，本文按复算采用 pp。

> 条件性提醒：真机结果只覆盖单臂 Piper、固定两相机、6 任务、每任务 30 trials。作者在 Limitations 中明确把跨 embodiment、不同相机配置与真实环境扩展列为未来工作。因此不能据此声称对移动机器人、人形、多臂或未见相机布局普遍有效。

### 5.4 消融、BFA 对照与意图标签评测

**消融（Table 3）**：9 个任务，2 个训练种子，每任务每种子 100 trials，所有变体训练 20K steps；Full 为 73.7%。

| 变体 | 控制 | SR | 与 Full 差值 |
|------|------|----|--------------|
| Full | 多模态未来意图 + 逐 step 路由 + chunk token | 73.7% | — |
| Learned Static | 全局共享学习比例，逐 step 应用 | 67.5% | −6.2 pp |
| Reactive-Vision | 仅当前视觉路由 | 57.4% | −16.3 pp |
| Time + State | 仅时间/状态路由 | 58.5% | −15.2 pp |
| E2E Future | 仅动作损失，无显式意图监督 | 55.7% | −18.0 pp |
| Routing Only | 去掉 chunk-level intent token | 63.3% | −10.4 pp |
| Random Masking | RDT-style 训练期随机视角 dropout | 54.6% | −19.1 pp |

**BFA 相关对照（Table 4）**：4 个任务，每任务 100 trials；Full 为 79.5%。注意这是不同任务子集，不能与 Table 3 的 73.7% 直接横比。

| 变体 | SR | 与 Full 差值 |
|------|----|--------------|
| Full | 79.5% | — |
| Reactive-Vision | 61.3% | −18.2 pp |
| Reactive (H_r=1) | 55.5% | −24.0 pp |
| Feature-side | 67.0% | −12.5 pp |
| BFA-style Labels | 72.0% | −7.5 pp |
| Reactive w/o Token | 49.8% | −29.7 pp |
| Routing Only | 71.5% | −8.0 pp |

论文明确说明：BFA 不提供可用代码与完整标注规则，因此 BFA-style 是用 GPT-4o 做臂状态标注并加任务特定映射的 paper-based 复现，不是官方 BFA 结果。这一点在引用时必须保留。

**意图标签评测（Table 5）**：轨迹弱标签相对人工标注 F1 为 82.9%，BFA-style 为 65.1%，高 17.8 点；held-out intent prediction 为 F1 78.4%、AUROC 91.2%。标签定义放宽/收紧分别改变 45.2% 与 27.6% 的标签，但 SR 为 75.5%/74.0%，相对默认 78.0% 最大变化 4.0 pp；推理时独立 corrupt 10%/20% intent，SR 为 73.5%/75.5%，相对未 corrupt 的 78.0% 最大变化 4.5 pp。上述标签定义与预测误差鲁棒性实验沿用 BFA controls 的 4 个任务、每任务 50 trials；Human F1 与 held-out intent F1/AUROC 的具体评测样本条件，源文未在此处说明。作者据此主张对标签阈值与中等预测误差不敏感。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 隐含假设 (Hidden Assumptions)

- **阶段-视角需求可由未来动作意图近似**：作者假设“未来动作更需要 local interaction 证据”这一二值意图，足以决定 global/local 路由。若任务需要更细的语义子目标、物体状态或语言中间计划，单一 intent 概率可能不够。
- **轨迹弱标签能对齐真实视觉需求**：标签由末端速度、路径、夹爪闭合等启发式构造，默认“低速度局部运动/夹爪闭合前后”对应 local 视角需求。对没有明显夹爪闭合、长程遮挡、工具使用或多物体交互任务，这个映射未必成立。
- **有界软路由总是安全**：α 保留至少约 25% 互补视角，避免硬切换；但某些任务可能需要几乎完全忽略某一视角，或有界范围会限制极端路由。论文未做“去掉下界”的消融。
- **历史 30 步足够**：bridge 只编码前 30 个动作；没有长时记忆或事件记忆。若任务阶段跨度超过该历史窗口，模型可能无法恢复早期上下文。
- **同分布相机与场景**：真机训练与测试使用相同 Piper 与两台 D435i；没有验证相机位置变化、视角遮挡、不同机器人 embodiment。作者也把跨 embodiment 列为未来工作。
- **BFA-style 可代表 BFA**：论文用 paper-based 复现做对照，但明确说 BFA 无可用代码与完整规则；因此 BFA-style 的差距不能直接等同于官方 BFA 与 FocusVLA 的差距。

### 6.2 能力边界与失败模式

- **能做**：在多视角、阶段切换明显的桌面操作中，按未来 step 调整 global/local 软权重；在 action side 做非侵入调制；用自动弱标签减少人工阶段标注；对中等 intent 预测误差与标签阈值变化保持一定鲁棒性。
- **不能做/未验证**：跨 embodiment、跨相机布局、移动操作、人形、多臂协同、长时记忆任务；官方 BFA 直接对比；大规模真机统计（仅 6 任务×30 trials）；极端视角缺失或相机故障下的路由行为。
- **诊断证据的边界**：attention ratio 不是因果依赖，作者自己引用 “attention is not explanation” 并用 masking 补强；但 masking 只在 Stack Blocks Three 上做，每设置 20 rollouts。我的条件性分析：这足以支撑“π0.5 在该任务上存在 local 依赖惯性”的案例证据，但不足以证明所有多视角 VLA 都同样失败。
- **性能回退点**：Table 2 的 Rotate Qrcode 从 82.0% 降到 66.0%，说明路由并非对所有任务都正向；可能与任务视觉需求、标签分布或路由有界性有关，论文未逐任务分析。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 架构/路由 | 训练/监督 | 适用场景 | 本文可确认证据 |
|------|--------|-----------|------------|----------|----------------|
| RDT | 双臂扩散 foundation model 基线 | 本文未展开其多视路由；其 random view dropout 被用作正则对照 | 同协议训练/评测 | 双臂操作基线 | RoboTwin 平均 34.76% |
| π0 | VLA flow model 基线 | 静态多视融合基线之一 | 同协议训练/评测 | 通用操作基线 | RoboTwin 平均 47.06% |
| π0.5 | 最强基线，同时是 FocusVLA backbone | 多视角 VLA；论文将其作为静态融合基线诊断 | 同协议训练/评测 | 多视角桌面操作 | RoboTwin 66.82%；真机 71.7% |
| BFA | 多视特征融合相关工作 | 预测当前视角权重，在 policy 前融合 | 离线 VLM 标注 + 任务特定映射 | 多视细粒度操作 | 本文未直接评测官方 BFA；只做 BFA-style 对照 |
| FocusVLA | 未来意图引导的逐 step 路由 | π0.5 + Future Intent Bridge + action-side router | 轨迹弱标签 + 联合动作/意图损失 | 多视角桌面操作，阶段变化明显 | RoboTwin 76.92%；真机 85.0%；消融 Full 73.7%/79.5%（不同子集） |

**面试 Tip**：被问到时先讲“诊断—机制—证据”三段：π0.5 在多视下存在 local-view 惯性；FocusVLA 用未来意图生成逐 step 有界软路由；增益主要来自显式意图监督与 chunk token，而不是单纯多一个路由器。最后补一句边界：真机只有 Piper/6 任务/30 trials，BFA 是 paper-based 对照，不是官方结果。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做多视角 VLA、主动感知、action chunk 内视觉条件调制的研究者：重点看 §3.1 的诊断设计与 §3.2 的 action-side routing。
  - 要评估“非侵入式模块挂到现有 VLA”可行性的工程师：重点看 Appendix G 的参数、延迟、路由边界与 teacher forcing 细节。
  - 想做弱监督阶段标签的人：重点看 Appendix A 的启发式规则与 Appendix F 的标签鲁棒性。
- **建议章节路径**：先读 §3.1 理解“为什么需要动态路由” → 再看 §3.2 与 Fig.4 理解信息流 → 然后读 §4.5/Table 3 与 Appendix E/Table 4 区分“方法增益”与“BFA-style 对照” → 最后看 Appendix G 的工程代价。Related Works 可在熟悉多视 VLA 后跳过。
- **不值得精读的理由**：如果你不做机器人学习、不关心多视角时序路由，或只想要大规模预训练数据配方，读摘要与 Table 2 即可；论文没有展开预训练语料，也没有跨 embodiment 验证。

---
[← Back to Theory](./README.md)

**关键引用**：[FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models](https://arxiv.org/abs/2610.31423)