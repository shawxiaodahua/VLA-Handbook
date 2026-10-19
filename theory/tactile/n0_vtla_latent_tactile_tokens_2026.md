# N₀-VTLA：把触觉变成“未来接触”的 latent 条件
## N₀-VTLA: Scaling Vision–Tactile–Language–Action Model with Latent Tactile Tokens

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-18  
> 论文版本：arXiv:2607.23782v1，2026-07-26；官方仓库快照 commit 03a0ce4（README）。  
> 核心定位：在 π0.5/PaliGemma 风格 VLA 上外接触觉预测通路，把“未来接触变化”的 latent 作为 action expert 条件，并配套 ALTER 离线策略改进。

## X-Ray

这篇论文的核心不是给 VLA 加一路“触觉相机”，而是把触觉从观测上下文移到动作专家的预测条件上：冻结 DINOv2 编码基线差分触觉图，轻量 predictor 估计未来 50 步的接触变化 latent z，z 不进入视觉语言 prefix，只条件 action expert。三阶段训练先 grounding z、再教 expert 消费 z、最后端到端联合；ALTER 再把部署数据转成 stage-relative 优势标签做离线改进。作者报告九项真机 NeoReal 任务全部胜出、20 项仿真平均 63.8% 对 44.0%，但真机每任务 trial budget 未完整披露，且论文与 README 的 latent token 数存在 10 vs 5 的版本冲突。

## 研究时间线（按论文引用脉络，非完整领域史）

```text
2023  JEPA / flow matching 等基础工作
      → 视觉表征预测、连续动作生成

2024-2025  VLA 扩展：OpenVLA、π0、π0.5、跨 embodiment 动作空间
      → 相机 + 语言 + 状态成为主观测

2025-2026  触觉 VLA 两条常见路线
      → 触觉拼进 prefix；或把当前触觉注入动作通路

2026-07-26  N₀-VTLA v1
      → 不把触觉当额外观测，而预测未来接触变化 latent
      → 三阶段接入 action expert；ALTER 做离线策略改进
```

## 路线选择：为什么不是“把触觉拼进 prefix”

作者在引言中批评了两类做法。第一类把触觉 token 拼进视觉语言上下文，等于把触觉当“另一路相机”；但触觉帧稀疏、远离接触时近乎空白，在信息密集的视觉 prefix 里买不到多少有效条件。第二类把当前触觉读数注入动作通路，问题是当前触觉记录的是“动作已经造成的接触”，本身不直接告诉模型“下一步动作将造成什么接触”。只条件在当前触觉上，策略容易慢于自己的接触事件。

N₀-VTLA 走第三条路：触觉不进入 vision-language prefix；用一个 predictor 读取当前触觉 token 与视觉语言上下文，输出未来动作 chunk 内的接触变化 latent z；action expert 直接条件在 z 上。作者把这称为“把触觉当预测目标，而不是当观测上下文”。

| 路线 | 触觉进入点 | 直觉 | 主要风险 |
|---|---|---|---|
| 触觉拼进 prefix | 视觉语言上下文 | 触觉作为额外模态 | 稀疏信号被密集视觉 token 淹没 |
| 当前触觉注入动作通路 | action expert / denoising | 用当前接触纠偏 | 只反映已发生的接触，反应滞后 |
| N₀-VTLA | 预测未来接触变化 z，再条件 action expert | 让动作针对“即将发生的接触” | z 必须被监督好，否则可能退化成视觉捷径 |
| free-latent 简化版 | 省略 Stage 1 监督，z 只由动作梯度塑形 | 不需未来触觉帧 | 缺少 grounding，可能塌缩或绕过 |

## 方法/组件对比表

| 组件 | 论文机制／README 实现 | 作用与训练状态 |
|---|---|---|
| 基础策略 | PaliGemma 视觉语言 backbone + Gemma 300m flow-matching action expert | 继承 π0.5 风格；动作 chunk 50 步；32 维 canonical 动作容器 |
| 触觉编码 | 触觉帧减 episode 起始基线帧，冻结 DINOv2 编码，后接可训练线性投影 | 每视图 10 个 token：1 个 class token + 3×3 池化得到的 9 个空间 token；编码器主干冻结 |
| 触觉 predictor | 轻量模块，读取当前触觉 token g 与 contextualized vision-language prefix，输出 latent z | 官方 README 称 cross-attention predictor；Stage 1 用未来触觉目标监督 |
| 预测目标 z* | 同一触觉编码器作用于未来 H=50 步的触觉变化，再跨视图平均 | 论文 Eq.2 与方法写 z 为 10 个 token；当前 README 写 5 个 latent token，存在版本冲突 |
| action expert 条件 | z 投影到 expert 宽度，前置到 action suffix；prefix 不反向 attend z | Stage 2 遮掉 prefix 的 key/value，迫使 expert 通过 z 读触觉；Stage 3 恢复 prefix 直连 |
| ALTER | pairwise progress model + stage-relative 二值优势标签 | 用干净示范、物体掉落事件、人类纠正数据训练；标签追加到任务 prompt |

## ASCII 信息流

```text
RGB×3 + instruction + state
        │
        ▼
PaliGemma vision-language prefix
        │
        │        tactile_t − tactile_0（每 episode 零接触基线）
        │                 │
        │                 ▼
        │          frozen DINOv2 + trainable projection
        │                 │
        │                 ▼
        │          g：论文每视图 10 个当前触觉 token
        │                 │
        └────────────┬────┘
                     ▼
              tactile predictor
                     │
                     ▼
        z：未来 50 步接触变化的 latent（论文 10；README 5 tokens）
                     │
                     ▼
        projection → action expert suffix
                     │
                     ▼
        flow matching → 50-step action chunk
```

关键结构约束：z 不进入 vision-language prefix；action tokens 可以 attend z 和 prefix；prefix 不 attend z；z 对应位置在动作输出头前被切掉，不直接产生动作。Stage 2 会遮掉 prefix 对 action query 的 key/value，让动作预测只能通过 z 读取触觉；Stage 3 再恢复 prefix 直连，使 expert 同时看到场景、指令和 z。

## 三阶段训练：先定义 z，再教接口，最后联合

作者把新初始化的触觉通路接到预训练多平台 checkpoint 上，分三步上线：

1. **Stage 1：grounding predictor。** 冻结整个基础策略，只训练 predictor、触觉投影和轻量重构头。z 被拉向未来触觉目标 z*，损失由对称 InfoNCE 与辅助 L1 重构组成。此时 z 的含义被钉在“未来接触变化”上，而不是变成 prefix 的第二份拷贝。
2. **Stage 2：对齐 action expert。** 冻结触觉感知栈，只训练 latent-to-expert 投影和 action expert。通过遮掉 prefix 的 key/value，动作查询只能 attend z 与噪声动作 token，于是 expert 必须学会通过 z 读触觉。
3. **Stage 3：端到端联合。** 除始终冻结的触觉编码器主干外全部解冻；移除 Stage 2 的 mask，恢复 prefix 到 expert 的直连；只保留动作目标，不再施加 Stage 1 的对比与重构损失。作者报告 z 在联合训练后仍偏触觉，而不是被视觉捷径改道。

free-latent 简化版把三阶段压成单阶段联合训练，省略 Stage 1 监督，z 完全由动作梯度塑形。作者讨论该变体的风险：没有未来触觉监督，瓶颈可能退化成 prefix 拷贝或被绕过；本文未提供该变体与三阶段方案的任务成功率对照表。

## 数学核心

### 目标

让 action expert 条件在一个紧凑 latent z 上，而 z 估计“未来动作 chunk 将造成的净接触变化”，而不是描述当前触觉帧。

### 公式

触觉 token：

```text
g_k = f_enc(tac_τ^k − tac_0^k) ∈ R^(10×d)
g   = [g_1; …; g_n] ∈ R^(10n×d)
```

未来触觉目标：

```text
z* = (1/n) Σ_{k=1}^{n} f_enc(tac_{τ+H}^k − tac_τ^k) ∈ R^(10×d),  H=50
```

Stage 1 监督：

```text
s_ij = <h(z_i), h(z*_j)>

L_NCE = −(1/(2B)) Σ_i [
  log(exp(s_ii) / Σ_j exp(s_ij))
  + log(exp(s_ii) / Σ_j exp(s_ji))
]

L_rec = || r_ψ(z) − D̄_{τ→τ+H} ||_1

L_1 = L_NCE + λ_rec L_rec
```

ALTER 的 stage progress 与优势标签：

```text
w_k = d̄_k / Σ_j d̄_j

φ_t = Σ_{j<k} w_j + w_k u_t

A_θ(x_a, x_b) ≈ φ(x_a) − φ(x_b)  （干净示范对的回归目标）

φ̂_t = δ_e + A_θ(x_t, x_0^e)
s_t = stage(φ̂_t; w_1, …, w_K)

r̂_t = A_θ(x_{min(t+H, T_e−1)}, x_t)  （论文 Eq.8；尾段处理见下）

q_k = Quantile_{1−ρ}{r̂_i : s_i = k}

c_t = 1[r̂_t ≥ q_{s_t}]
```

### 变量

- g_k：第 k 个触觉视图的当前差分 token；f_enc 是冻结编码器加可训练线性投影。
- tac_0：每 episode 起始的零接触基线帧；tac_τ 是当前帧；H=50 是动作 chunk horizon。
- z*：未来触觉变化目标；论文 Eq.2 对 n 个视图取平均。
- B：batch 大小；h 是 10 个 latent token 均值池化后做 ℓ2 归一化；s_ij 是余弦相似度。
- D̄：下采样后的未来接触变化场；r_ψ 是重构头；λ_rec 平衡对比项与重构项。
- ALTER 中 K 为阶段总数，d̄_k 是干净示范中阶段 k 的平均时长；u_t 是当前阶段完成比例；A_θ 是成对进度模型。φ̂_t 是估计全局进度，决定阶段 s_t；正常起点 δ_e=0，staged-recovery 的 δ_e 用首帧与三个干净参考示范的起点比较估计。ρ=0.3 是阶段内标为正样本的目标比例，其他样本仍以负标签参与训练。
- 干净示范对用进度差的平方误差；事件对用 squared hinge loss `max(0, m − y A_θ(x_a,x_b))²`，m=0.02；y=+1 表示 x_a 优于 x_b，反向为 −1，采样时随机交换输入顺序。掉落前优于掉落后，人工纠正末端优于起点。
- 附录 F 在分配阶段前对 φ̂_t 做五帧中值滤波，再对阶段索引取截至当前帧的最大值；这不改动阶段内排序用的局部变化。若剩余未来帧数小于 H，局部分数乘 H/可用未来帧数；终端自比较为零。
- ALTER 从对应的 pretrained checkpoint 开始，保持 flow-matching 目标；训练 prompt 追加 `Advantage: positive` 或 `Advantage: negative`，并以 0.3 概率省略标签。部署时固定使用 `Advantage: positive`。来源：[论文 §4.4、Appendix F](https://arxiv.org/html/2607.23782v1#S4.SS4)。

### 直觉

InfoNCE 让 z 在 batch 内检索到自己的未来触觉目标，以同批其他样本的未来目标为负样本，提供判别性；L1 重构把 z 锚定到接触发生的空间布局，避免模型学到一个“能分开 batch 但不编码接触位置”的捷径。Stage 2 的 mask 则是对接口的强制对齐：如果 expert 还能直接看 prefix，它就没有动力通过 z 读触觉。

### Napkin Formula

```text
z = predictor(g, prefix)        # g 提供当前触觉，z 预测未来接触变化
action = expert(prefix, z)      # 部署／Stage 3；g 不直接进入 expert
```

作者用检索实验佐证“接下来”确实存在：Stage 1 后，在 378 个 held-out queries 上，z 在约 32 个候选的未来触觉目标中 top-1 达 92.3%，随机为 3.2%；只用当前触觉 g 检索为 57%，在 128 候选池中 g 为 40%、predictor 为 81%。这些是 within-pool 检索率，不是全库检索，也不是任务成功率。

## 示意玩具例子（非作者实验）

在一个二维、仅 3 个候选的玩具里，当前触觉编码为 g=[1.0, 0.1]，未来候选为 A=[0.8, 0.2]、B=[0.2, 0.8]、C=[0.1, −0.1]；假设 predictor 输出 z=[0.3, 0.7]。用余弦相似度排序：

```text
cos(g, A) ≈ 0.989
cos(g, B) ≈ 0.338
cos(g, C) ≈ 0.633
当前触觉检索 → A

cos(z, A) ≈ 0.605
cos(z, B) ≈ 0.987
cos(z, C) ≈ −0.371
预测 latent 检索 → B
```

这个示意只说明“z 可以携带当前 g 之外的未来接触信息”，不 replicate 论文的 32/128 候选实验，也不能推出任何成功率。它的作用是帮助理解：当前触觉更像 A，但动作真正要应对的可能是即将发生的 B。

## 数据与评测条件

作者报告的预训练语料是 NeoData：每 episode 3 路 RGB（一路第三人称、两路腕部）、语言指令、机器人本体状态；单臂平台 2 路触觉流，双臂 4 路；统一 30 fps；模型消费 224×224；动作与状态统一到 32 维容器，前 20 维为两条 10 维手臂槽位，每条含 3 维末端位置、6 维 rot6d 旋转、1 维夹爪通道，剩余 12 维恒零。每 episode 开头要求至少 0.5 秒开爪静止零接触基线，后续触觉帧都相对该基线做差分。

预训练在集群规模进行，但附录只给出“多节点、现代加速器”的粒度，省略硬件型号、卡数、显存和吞吐；因此不能从本文推断训练成本或实时性。下游任务用数百条示范做 SFT；NeoSim 每个策略用 100 条示范训练。NeoReal 每任务有各自 trial budget，但本次固定论文全文未列出完整逐任务分母（companion data report 未核对）；不能补造样本数，也不能把 47.2% 当作统一大样本估计。

评测同时报告二值成功率和 100 分阶段 rubric 进度分。NeoSim 使用固定每任务语言 prompt；对照 no-tactile base 时，关闭触觉 flag 后模型结构上退化为 base policy，因此作者把差异归因于触觉通路。但这个“结构隔离”依赖实现等价性，仍需核对 checkpoint 与开关行为。

## 作者报告的结果与基线

以下均为作者报告，未在本机重跑机器人。

| 评测 | N₀-VTLA | 对照 | 说明 |
|---|---:|---:|---|
| NeoReal 九任务成功率均值 | 47.2% | π0.5 29.4% | ACT 未完成任务，进度均分 10.2 |
| NeoReal 九任务进度分均值 | 56.8 | π0.5 42.3 | 作者称八项领先、均值领先 |
| UniVTAC 八任务成功率均值 | 83.1% | InternVLA-A1 67.1% | 同一对齐协议 |
| NeoSim 十二任务成功率均值 | 50.8% | π0.5 45.8% | 单臂 73.8%，双臂 39.4% |
| 二十任务合并均值 | 63.8% | π0.5 44.0% | 作者称 π0.5 为最强总体基线 |
| ALTER 三任务成功率 | 95/80/75% | π0.5+ALTER 90/75/60% | 依次为 Towel Folding、Bag Packing、Cardboard Box Folding；与九任务基础模型结果分开 |

真机细节包括：Socket Plugging 85% 对 60%；Board Insertion 25% 对 0%；Bottle Standing 30% 对 0%。作者用这些案例说明触觉把插入从“一次性视觉承诺”变成“接触感知的重试循环”，并通过夹爪开度微调控制夹持力。但这些都是作者定性解释，不是独立的因果实验。

Figure 7 的图题写“Simulation success rate”，而 §5.2 标题与正文明确是 NeoReal 真机；这是原文编辑歧义。本文保留该冲突，不替作者消歧。引用该图数字时应同时标注图题措辞与正文口径。

## 失败模式与 Hidden Assumptions

- **零接触基线假设。** 触觉差分依赖每 episode 开头有稳定、无接触的基线帧。若基线期间已经接触、传感器被遮挡，或 episode 中途传感器移位，差分输入会失真。作者称基线差分对安装差异更稳健，但不是严格不变；本批没有跨传感器真机泛化实验。
- **未来接触可预测假设。** z 的监督假设未来 50 步的接触变化能从当前触觉与视觉语言上下文预测。若接触事件稀疏、候选负样本太容易，检索率高不一定等于闭环控制收益；论文也把闭环任务表现作为更严格检验。
- **Stage 2 的可学性假设。** 遮掉 prefix 后，expert 只能通过 z 读条件。若某些自由空间动作主要依赖视觉，Stage 2 可能很难；Stage 3 恢复 prefix 直连，但接口已经先被 z 对齐。
- **token 数版本冲突。** 论文 Eq.2 与方法写 z 为 10 个 token；当前 README 的 Model summary 写 5 个 latent token。本批未读取 checkpoint config，不能判定发布权重对应哪一个；工程复现必须以实际配置为准，不能把论文公式硬编码成唯一事实。
- **ALTER 标注依赖。** pairwise progress model 以任务 prompt 为条件，观测输入是两个时点的同步多相机 RGB；触觉、运动学和事件信号用于构造离线目标，不是模型输入。任务模板由 Gemini-3.5-Flash 生成并经人工审阅，事件对约 5% 采样且损失乘 0.1；标注噪声会传导到优势标签。
- **评测分母缺失。** NeoReal 各任务有各自 trial budget，本次固定论文全文未列出完整逐任务分母，companion data report 未核对；不能计算置信区间、显著性，也不能把“胜出”扩展成跨硬件、跨传感器结论。
- **“首个大规模”是作者主张。** 论文称据其所知是首个大规模触觉预训练 VTLA；本批不把“首个”当作独立证明，NeoData 的规模细节在 companion data report 中，本次固定论文全文未列出其小时数或 episode 总量；本批未核对 companion data report。

## 工程实现边界

官方 README 当前开放的是预训练 checkpoint、post-training toolkit 和 inference server；明确说明**不提供大规模预训练 pipeline**。base checkpoint 是“供 post-training 的预训练基座”，不是可直接部署的策略，仍需任务微调。仓库提供 canonical 数据转换、归一化统计、websocket/ZMQ 服务，以及检查触觉通路是否生效、关闭触觉时是否与 base policy 字节等价的脚本。

实现上有几个硬边界：

- 按作者部署协议，每次请求做一次 prefix 编码与十步 Euler 去噪，并必须执行完整 50 步 action chunk，再请求下一次预测。作者指出夹爪闭合命令集中在 chunk 尾部；若提前重规划、截断执行，机械臂可能到达目标位置但夹爪从未闭合。这是执行调度问题，不是模型不会闭爪。
- serve 时每个 episode reset 会清空并重设触觉基线；第一个观测作为新的零接触参考。某一路触觉不可用时，控制器把该视图当作自身基线，而不是让请求失败。
- 任务适配要重新计算该任务自己的归一化统计，不能复用预训练统计；动作在训练时转成 chunk-relative，部署时再映回绝对位姿。
- README 提到零初始化 gate，使训练从 base policy 行为开始、逐步打开触觉通道；论文正文没有展开这个 gate。论文则描述无触觉时用 learned null token 回退。两者分别属于实现与机制描述，不能合并成同一个实验结论。
- 附录 B 省略硬件型号、设备数、显存和训练吞吐，本文也没有给出可用于确定实际延迟或控制频率的测量。另有任务成功吞吐：N₀-VTLA+ALTER 在 Towel Folding、Bag Packing、Cardboard Box Folding 分别为 43、15、30 次成功完成/小时（§5.4、Figure 12）；这不是推理 FPS。

## Eureka Moment（解读）

作者的设计假设是：**让未来接触变化成为预测目标，再用 latent 条件 action expert。** N₀-VTLA 仍以当前触觉 g 为输入，但不把它拼进视觉语言 prefix；z 结合当前触觉和 prefix，预测未来 50 步的净接触变化。这是本文针对接触任务采用的归纳偏置，不能推出其他 VTLA 都不该使用当前触觉或 prefix 融合。

第二个洞察是训练课程：先让 z 学会“什么是接触”，再在 mask 下逼 expert 学会“怎么用 z”，最后联合微调。这样新通路不是和整个策略一起从随机状态硬碰硬，而是在已有语义和接口上逐步上线。

第三个洞察在 ALTER：部署数据里的物体掉落、人工纠正和阶段进度，可以被转成相对优势标签，而不需要额外环境交互。但这条线的关键不是“有 RL 标签”，而是标签是否忠实于接触事件与阶段语义。

## 面试 Tip

如果被问 N₀-VTLA，先答三件事：触觉没有进入 vision-language prefix，而是被预测成未来接触 latent z；z 由 Stage 1 的未来触觉目标 grounding，Stage 2 用 mask 强制 expert 通过 z 读触觉，Stage 3 再联合；ALTER 是另一条离线策略改进线，不要把 95/80/75% 与九任务基础模型结果混在一起。追问时再补：NeoReal 每任务 trial budget 未完整披露、论文与 README 的 token 数冲突、以及 full-chunk 执行边界。

## 参考链接

- 论文 v1：https://arxiv.org/html/2607.23782v1  
  Abstract；§2.2 Eq.1–2；§4.2 Stage 1–3；§5.2 与 Figure 7；Table 1–2；§5.4；§5.5；Appendix D/F。
- 官方仓库 README（commit 03a0ce4）：https://github.com/neoteai/N0-VTLA/tree/03a0ce4d7091ca2354864796770715aa212601b7  
  Model summary；Model Download；Serving；Highlights。
- 官方项目页：https://research.neoteai.com/n0-vtla/

[← Back to Theory](./README.md)
