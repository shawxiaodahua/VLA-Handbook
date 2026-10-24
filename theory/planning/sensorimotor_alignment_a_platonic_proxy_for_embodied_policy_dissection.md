# 感知运动对齐：具身策略成功的柏拉图式代理 (Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-23
>
> **论文**: Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success
> **链接**: https://arxiv.org/abs/2610.31670
> **核心定位**: 把机器人末端轨迹当作 PRH 的“第三种模态”，用视觉-语言表征的近邻集合与未来轨迹的近邻集合之重叠度，构造一个 training-free 的离线指标 S²，在闭环评测前给 VLA / raw VLM / WAM / 真机 policy 排序。

## 来源与版本说明

本文只依据 arXiv:2610.31670v1 的 HTML 全文（对应 PDF 正文）写作。该版本标注为 **[cs.CV] 22 Oct 2026**，作者为 Yunfei Xie、Howard H. Qian、Shiyi Lan、Kaiyu Hang、Chen Wei。

来源记录中存在两处元数据不一致，需要先说明：

1. 候选元数据把来源标为 **cs.RO**，而 PDF 正文标为 **cs.CV**；这是分类元数据差异。
2. 候选摘要 snippet 在 “should th” 处截断，不是完整摘要；PDF 摘要是完整的，并明确写出 S² 是 “training-free offline metric”，以及 12 个 VLA 上 Pearson r=+0.717。

因此，本文不使用候选 snippet 补写或替代 PDF 中的主张；凡涉及论文结论，均以 PDF 正文为准。论文中出现的项目网站与代码链接，按本次编辑约束不展开、不作为证据引用。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | S² 与下游任务成功率正相关：12 个 VLA 上 r=+0.717，8 个 raw VLM 上 r=+0.681，6 个 WAM 上 r=+0.718（p=0.054），5 个真机 policy 上 r=+0.831；作者主张它可作为闭环前的低成本离线代理。 |
| 适合精读 | 做 VLA 模型选型、离线评测、表征诊断的人；重点看 §2 的时序定义、§3.1 的四组相关性、§4 的表征解释。 |
| 可以跳过 | 只关心在线控制律、运动规划或 RL 训练信号的人；这篇不提供控制方法，也不证明训练时优化 S² 能提升成功率。 |
| 落地可行性 | 中。离线可算、motor 邻居可跨模型复用；但需要参考数据集、可访问的模型特征，且相关性不是因果。 |
| 主要风险 | 相关性证据为主；WAM 的 p=0.054 边缘；3 个 WAM 的视频质量对比只有 3 个点；真机只有一个统一设置、5 个 policy；horizon 选择会改变结论。 |

💡 **X-Ray 开场**

这篇论文解决的是“闭环评测太贵”的问题：真机 rollout 慢、噪声大、依赖 benchmark，因此在 fine-tune 或上真机之前，需要一个便宜的离线信号来判断哪个模型更值得测。

它发现的不是某个新策略，而是一个**表征层面的相关性**：如果一个模型的图像-文本表征把“视觉上相近的 episode”排在一起，而这些 episode 的未来运动轨迹（末端位姿或双臂关节轨迹）也相近，那么这个模型的下游成功率往往更高。作者把这种一致性称为 Sensorimotor Alignment（S²）。

对 VLA 研究者意味着：S² 可以在 VLA fine-tuning 之前筛 VLM backbone，也可以在闭环之前给一组 VLA / WAM 排序；但它不能替代闭环，因为论文自己明确说这是相关性研究，尚未证明可直接用于训练。

📍 **研究全景时间线**

```
2023  BridgeDataV2 / ALOHA 公开
  → 2024  DROID / PRH / SimplerEnv
  → 2025  RoboTwin
  → 2026  本文 S²（arXiv:2610.31670v1, 2026-10-22）
```

这里只列来源中有明确年份的节点。PRH 指 Platonic Representation Hypothesis；本文的位置是把 PRH 从“图像-文本”扩展到“图像-文本-机器人轨迹”，并用 k-NN 邻域重叠把对齐度变成可计算指标。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

先要把“已观测历史帧”和“未来预测帧”分开。S² 的输入不是一段历史视频，而是一张**当前帧**加上一段**未来运动轨迹**；对 DROID/BridgeDataV2 是 7 维末端位姿+夹爪，对 ALOHA/WAM 是 14 维双臂关节位置动作（每臂 6 关节+1 夹爪）：

| 组件 | 输入 | 输出 | 时间语义 | 训练/推理 | 归属 |
|------|------|------|----------|-----------|------|
| Sensory encoder | 采样帧图像 + 任务指令（WAM 为用于动作解码的 video-generator latent） | z_i ∈ R^d | 当前已观测帧 t_i | 不训练；VLA/raw VLM 用被评模型的 vision/text encoder，WAM 用 video-generator latent（B.4 取 video-backbone Block 7） | 每个被评模型各自持有 |
| Motor representation | 数据集中的运动状态（DROID/BridgeDataV2：末端位姿+夹爪，7 维；ALOHA/WAM：双臂关节位置+夹爪，14 维） | DROID/BridgeDataV2：τ_i ∈ R^{h×7}；ALOHA/WAM：τ_i ∈ R^{h×14} | 未来帧 t_i+1 到 t_i+h | 不训练；来自参考数据集 | DROID / BridgeDataV2 / ALOHA |
| Sensory distance | z_i, z_j | cosine distance | 离线 | 推理时计算 | 全局共用 |
| Motor distance | τ_i, τ_j | DTW 距离 | 离线 | 推理时计算 | 全局共用 |
| 对齐分数 | 两个 k-NN 集合 | m_NN 或 S² | 离线 | training-free | 每个模型一个分数 |
| 下游成功率 | 闭环 rollout | 成功率标量 | 仿真/真机 | 单独评测 | SimplerEnv / RoboTwin / MolmoB0T 设置 |

关键点：**S² 不读取 policy 的 action head 输出，也不把 h 当作历史上下文或记忆窗口。** 它读的是模型内部表征（LLM 输出 token 或 video backbone latent），以及数据集中记录的未来 h 步运动轨迹（DROID/BridgeDataV2 为末端位姿+夹爪，ALOHA/WAM 为双臂关节位置动作）。这个区分是全文的时序基础。

### 1.2 关键机制

**为什么用邻域重叠，而不是全局 CKA？** 全局 CKA 会把所有样本对都纳入比较；CKNNA 则把比较限制到互近邻，因而更聚焦局部对齐。S² 只问一个局部问题：对每个 episode，sensory 空间的 k 个最近邻，是否也是 motor 空间的 k 个最近邻？这更贴近“视觉上像的 episode 是否该做相似动作”。

**为什么 motor 距离用 DTW？** 同一动作在不同速度下执行，逐帧欧氏距离会很大；DTW 允许时间轴对齐，把“形状相似但速度不同”的轨迹判为近邻。论文用 symmetric2 递归、无 warping window、不做路径长度归一化。

**为什么还要 CKNNA 归一化？** 原始的邻域重叠计数 m_NN 只统计共享邻居比例，未做中心化和归一化。论文最终报告的 S² 沿用 Huh et al. 的 centered kernel nearest-neighbor alignment（CKNNA）变体：用 cosine 核 K 和二值邻居矩阵 L，做 masked HSIC，再除以自对齐项的几何平均。这样分数不只是一个 [0,1] 的计数，而是带居中和归一化的对齐度。

**为什么 training-free？** 它不需要改模型、不需要额外训练、不需要闭环数据。给定参考数据集和模型特征，S² 是一次离线计算。这也是它能作为“模型选择工具”的前提。

⚡ **Eureka Moment**：把机器人轨迹当成 PRH 的另一种“现实投影”——如果表征真的在逼近世界结构，那么“视觉-语言近邻”与“未来运动近邻”应该重合；重合度就是具身能力的离线代理。

### 1.3 信息流/架构图

```
参考数据集（DROID / BridgeDataV2 / ALOHA）
        │
        ├── 采样 episode i，锚点 t_i
        │
        ├── 已观测侧：image(t_i) + instruction ──► 模型 encoder ──► z_i
        │
        └── 未来侧：state(t_i+1 ... t_i+h) ──► τ_i ∈ R^{h×d_m}（DROID/Bridge d_m=7；ALOHA/WAM d_m=14）
                                                      │
z_i ──► cosine kNN ──► N_sensory(i)                   │
τ_i ──► DTW kNN   ──► N_motor(i) ◄────────────────────┘
        │
        └── 交集大小 / k ──► m_NN
                              │
                              └── CKNNA 归一化 ──► S²
                                                     │
                                     与闭环成功率做相关（Pearson r）
```

注意图中的箭头方向：**未来轨迹只用于定义 motor 邻居，不进入模型输入**。模型看到的是 t_i 时刻的图像和指令；S² 衡量的是“模型当前表征”与“数据集中未来轨迹结构”的一致性。

## 2. 数学核心

📌 **Napkin Formula**：

```
S² ≈ 模型表征的 k 近邻集合 与 未来运动轨迹的 k 近邻集合 的重叠度
```

先给目标：对每个 episode i，分别构造 sensory 表示 z_i 和 motor 表示 τ_i；在两种空间中各取 k 个最近邻；重合越多，说明“视觉-语言上相近的样本”也越可能在“未来运动”上相近。

**Sensory 表示**：

```
z_i = f(image_i, instruction_i) ∈ R^d
```

f 分别调用被评模型的 vision encoder 和 text encoder，输出 token 特征后做 average pooling，得到一个 d 维向量。

**Motor 表示**：

```
DROID/BridgeDataV2（单臂末端位姿+夹爪，7 维）:
τ_i = [ (x, y, z, φ, θ, ψ, g)_{t_i+u} ]_{u=1..h} ∈ R^{h×7}

ALOHA/WAM（双臂关节位置动作，14 维）:
τ_i = [ (q_1, ..., q_6, g, q_7, ..., q_12, g')_{t_i+u} ]_{u=1..h} ∈ R^{h×14}
```

行 u 是采样帧之后第 u 步的运动状态；对 DROID/BridgeDataV2，x,y,z 是位置，φ,θ,ψ 是 roll/pitch/yaw，g 是夹爪坐标，位置按 R^3 向量处理，姿态按 SO(3) 旋转处理；对 ALOHA/WAM，每行是 14 维双臂关节位置动作向量（每臂 6 关节 + 1 夹爪），DTW 局部代价用 14 维关节向量的欧氏距离。

**距离**：

```
d_sensory(i,j) = 1 - cos(z_i, z_j)

d_motor(i,j) = DTW(τ_i, τ_j)
```

**DTW 局部代价**：

```
d_pos  = || p_i,u - p_j,v ||₂
d_rot  = 2 · arccos( min{1, | q_i,u^T q_j,v |} )
d_grip = | g_i,u - g_j,v |
c_uv   = w_pos·d_pos + w_rot·d_rot + w_grip·d_grip
```

默认 w_pos = w_rot = w_grip = 1。q 是由 roll/pitch/yaw 转成的单位四元数；q 与 -q 表示同一旋转，所以取绝对值。

**symmetric2 DTW 递归**：

```
D_11 = c_11
D_u1 = Σ_{a=1..u} c_a1
D_1v = Σ_{b=1..v} c_1b
D_uv = min( D_{u-1,v} + c_uv,
            D_{u,v-1} + c_uv,
            D_{u-1,v-1} + 2c_uv )
```

两条 h 步轨迹的距离取 D_hh，不做路径长度归一化；h=1 时退化为单状态代价 c_11。

**邻域重叠**：

```
N_sensory(i) = 用 d_sensory 取 z_i 的 k 个最近邻
N_motor(i)   = 用 d_motor   取 τ_i 的 k 个最近邻

m_NN = (1/(N·k)) · Σ_i | N_sensory(i) ∩ N_motor(i) |
```

**最终报告的 S²（CKNNA 归一化）**：

```
K_ij = z_i^T z_j / (||z_i||·||z_j||)      （cosine 核）
L_ij = 1{ j ∈ N_motor(i) }                 （二值邻居矩阵）

α_ij(A,B) = 1{ j ∈ N_A(i) ∩ N_B(i) }

Align_knn(A,B) = HSIC_u( α(A,B) ⊙ A, α(A,B) ⊙ B )

S² = Align_knn(K,L) / sqrt( max( Align_knn(K,K)·Align_knn(L,L), ε_A ) )
```

其中 HSIC_u 是去偏的有限样本估计；α 是“共享邻居”掩码；⊙ 是逐元素乘。论文说明，如果把共享邻居都视为等距，sensory 权重退化为 1，则 Σα = N·k·m_NN，于是 m_NN 是 S² 的一个简化计数版本；实际报告的 S² 保留了 sensory 权重，且不保证落在 [0,1] 内。

**时间语义（容易读错的地方）**：

- t_i 是**已观测**的采样帧；τ_i 从 t_i+1 到 t_i+h，是**未来** h 步。
- 若记录频率为 f，则 Δt = 1/f。连续 h 步的终点距 t_i 是 **h×Δt**，不是 (h-1)×Δt，因为第一步是 t_i+1。
- 若只在均匀间隔上取 K 个采样帧（不是连续步），首末跨度是 **(K-1)×Δt**。本文的 h 是连续步，所以用 h×Δt；不要把它和“K 帧采样跨度”混用。
- DROID 记录频率 15 Hz，Δt ≈ 66.7 ms；h=7 约 0.47 s，h=110 约 7.33 s，h=220 约 14.67 s（按记录频率换算，不是论文报告的指标）。
- BridgeDataV2 记录频率 5 Hz，Δt = 200 ms；h=40 约 8 s。

**直觉**：如果模型的 sensory 表征真的编码了“该做什么动作”的结构，那么视觉-语言上相近的 episode 应该在未来轨迹上也相近；S² 高说明这种一致性高。反过来，两个视觉上相似但需要不同动作的场景，会拉低 S²。

## 3. 带数字走一遍：玩具例子

> 以下数值是**假设的玩具例子**，只用于验算公式，不是论文结果。

假设 N=5 个 episode：A、B、C、D、E；k=2。假设 sensory 和 motor 的 2-NN 集合如下：

| episode | N_sensory | N_motor | 交集 | 交集大小 / k |
|---------|-----------|---------|------|--------------|
| A | {B, C} | {B, D} | {B} | 0.5 |
| B | {A, C} | {A, E} | {A} | 0.5 |
| C | {A, B} | {D, E} | 空 | 0.0 |
| D | {E, A} | {C, A} | {A} | 0.5 |
| E | {D, B} | {B, C} | {B} | 0.5 |

按公式：

```
m_NN = (1/(5×2)) × (1 + 1 + 0 + 1 + 1)
     = 4 / 10
     = 0.4
```

验算：每个 episode 的 |交集| 分别是 1、1、0、1、1，总和 4；N·k = 10；所以 m_NN=0.4。这个例子说明 m_NN 是“平均每个 episode 有多少比例的 k 个邻居是共享的”。

再走一遍 CKNNA 归一化。假设某模型上：

```
Align_knn(K,L) = 0.30
Align_knn(K,K) = 0.50
Align_knn(L,L) = 0.40
```

则分母为 sqrt(0.50×0.40)=sqrt(0.20)≈0.4472，S²≈0.30/0.4472≈0.671。注意这只是归一化后的对齐分数，不是成功率；论文用它与成功率做相关，而不是把它当成功率。

再验算时间跨度。假设某数据集 10 Hz，h=5：

```
Δt = 1/10 = 0.1 s
连续 5 步终点距 t_i = 5×0.1 = 0.5 s
```

如果换成“均匀取 K=5 个采样帧”，首末跨度是 (5-1)×0.1=0.4 s。两者差一步，原因是连续 h 步从 t_i+1 开始，而 K 帧采样的首帧就是参考帧。这个区别在把 h 与“历史窗口/记忆窗口”混读时特别容易出错：**h 是未来轨迹长度，不是历史上下文长度，也不是 policy 的 action chunk 长度。**

## 4. 工程视角

**计算分解。** S² 的主要成本不在模型前向，而在 motor 侧的成对 DTW：

- sensory embedding：对 N 个 episode 各算一次 z_i；论文在 horizon sweep 时明确说 z_i 只算一次，改变 h 只改变轨迹长度，不改变 episode、锚点或 sensory 输入。这是重要的复用点。
- motor 距离：朴素实现要对 N(N-1)/2 对轨迹算 DTW；每对 DTW 的朴素复杂度是 O(h²)。DROID 的 N=6000、h 最大 220，成对数量约 1.8×10^7，DTW 单对约 4.8×10^4 个单元，量级很大。论文没有报告 wall-clock，因此这里只能做条件性分析：工程落地需要下采样、并行、或用近似 DTW；不能假设“离线”就等于“便宜”。
- **可复用性**：论文明确说 motor-neighbor indices are reused across models。也就是说，DTW 邻居只依赖参考数据集和 h，不依赖被评模型；换模型时只需重算 sensory kNN 和 overlap。这是 S² 相对于“每个模型都跑闭环”的最大工程优势。
- kNN 查询：k=10 默认；k 在 {2,5,10,15,20} 变化时，Pearson r 在 +0.644 到 +0.717 之间，默认 k=10 略高。k 小意味着可以用近似最近邻，但论文没有给近似实现的误差分析。

**特征位置必须固定。** 论文对不同模型族用了不同特征位置：

- 12 个 fine-tuned VLA：LLM 输出的 joint vision-text 特征（默认）。
- 8 个 raw VLM：LLM 输出的 joint 特征，h=3。
- 6 个 WAM：video backbone 的 Block 7 特征，h=15。
- 5 个真机 policy：joint LLM-output 特征，h=110。
- 14 个冻结 vision backbone：ViT-out 特征。

这些不是“同一个 S² 设置”在所有模型上通用；跨模型族比较时，必须按论文的设置分别计算。把 VLA 的 LLM-out 设置套到 WAM 的 video latent 上，或者把 WAM 的 Block 7 套到 VLA 上，都不是论文验证过的做法。

**horizon 不是超参可以随便调。** 论文在 VLA 上报告 h∈{1,3,7,15,25,40,75,110,150,220} 的完整 sweep：joint 特征的 r 从 +0.610 到 +0.717，vision 特征从 +0.554 到 +0.703，text 特征从 +0.227 到 +0.274 且不显著。真机 policy 的 h=110 是 exploratory horizon；prespecified h=7 只给 r=+0.429、p=0.236。这说明**真机结论对 horizon 选择敏感**，不能把 h=110 写成所有实验共有的默认值，也不能把某个 h 的显著结果当作先验假设。

**S² 不是控制器。** 它不输出动作，不参与闭环，不替代 policy。工程上它适合放在“模型选择/回归测试”环节：先离线排序，再决定哪些模型进入昂贵的闭环评测。论文的作者 limitations 也明确说，S² 是闭环前的低成本的模型选择与诊断工具，闭环仍是最后一步。

## 5. 数据与评测

### 5.1 预训练 / 离线表征

**Raw VLM 队列。** 论文用 VLM4VLA 的 8 个 raw VLM backbone：KosMos-2、PaliGemma、PaliGemma 2、Qwen2.5-VL-3B、Qwen2.5-VL-7B、Qwen3-VL-2B、Qwen3-VL-4B、Qwen3-VL-8B。S² 在**fine-tune 之前**的 raw backbone 上计算，参考数据为 DROID，h=3，pool image 和 instruction tokens。然后与 VLM4VLA 用同一 recipe fine-tune 出的 VLA 在 SimplerEnv 上的成功率做相关：r=+0.681，one-tailed p=0.031。同一队列的 VQA 分数与 VLA 成功率几乎不相关，r=-0.154。

**冻结 vision backbone。** 论文复用 Hu et al. 的 14 个预训练 backbone（9 个 ResNet-50、5 个 ViT-B/16），在 21 个任务上评测。S² 用这些 released checkpoints 在同一个 6000-episode DROID 子集上计算，特征为 ViT-out。低数据 behavior cloning（每任务 1 个 demonstration，Robosuite 为 5 个）下 r=+0.677，p=0.004；visual reward function（VRF）下 r=+0.774，p<0.001。BC budget 增大后，相关性降到 r=+0.236（最大 budget），作者解释为更多 demonstration 让 policy 能补偿较弱表征。RL 在两个 Meta-World seed set 和 Robosuite 上没有正相关，r 在 -0.193 到 -0.002 之间。

这里要注意归属：这 14 个 backbone 不是 VLA，也不是 WAM；它们只有 ViT-out 特征，没有 LLM 输出。S² 对它们有效，说明指标不依赖语言模型；但这也意味着不能把 VLA 的 LLM-out 结论直接套到 frozen backbone 上。

### 5.2 仿真评测

**12 个 VLA。** 包括 5 个 StarVLA 变体（Qwen2.5-GR00T-Bridge、Qwen2.5-GR00T-RT-1-Bridge、Qwen2.5-OFT-RT-1-Bridge、Qwen3-GR00T-RT-1-Bridge、Qwen3-OFT-RT-1-Bridge）、CogACT Small/Base/Large、GR00T-N1.5-Bridge、GR00T-N1.6-Bridge、Pi0-Bridge、SpatialVLA-Bridge。覆盖 4 种 action head（flow matching、diffusion、autoregressive、deterministic MLP）和 3 种 VLM 训练制度（从 frozen 到 end-to-end）。S² 用 DROID 6000 episodes、h=7、k=10、LLM-out joint 特征；成功率来自 SimplerEnv-WidowX。Pearson r=+0.717，model-bootstrap 95% CI [+0.436, +0.912]。

**6 个 WAM。** Vidar、LingBot-VA、Motus、Fast-WAM、FlowWAM、Kairos。S² 在双臂 ALOHA 参考数据上计算，所有 6 个模型共享同一批 ALOHA 轨迹和 S² 设置；报告比较用 video-backbone Block 7、h=15、k=10。成功率来自 RoboTwin 2.0（论文正文称 RoboTwin 2.0，引用 [32]）。r=+0.718，one-tailed p=0.054。这个 p 值是边缘的，不能写成“显著”。

**视频质量对比。** 只取 3 个 WAM（Motus、LingBot-VA、Vidar），因为另外 3 个在闭环评测中不生成未来视频。SSIM r=+0.558，PSNR r=+0.468，FID r=+0.680（方向与“质量越高越好”相反，论文说低视频质量反而对应高成功率）；S² 在这 3 个模型上 r=+0.999。**3 个点的 r=+0.999 统计上非常脆弱，不能当作强证据**；它只能说明“在这 3 个模型上，S² 恢复了成功率排序”，不能外推到所有 WAM。

**鲁棒性。** 论文变了 8 种设置：参考数据集、轨迹 horizon、5 个实现选择、DTW warping。DROID 和 BridgeDataV2 两个单臂参考数据都给出接近 r=+0.7 的相关；DROID horizon 到 h=220 仍 r>0.5，BridgeDataV2 从 h=3 到 40。负控制：把 3 个双臂 WAM 用单臂 DROID 参考评分，r 从 +0.999 降到 +0.246（p=0.421），说明 S² 对 arm configuration mismatch 敏感。

### 5.3 真实机器人

真机部分只用一个统一设置，来自 MolmoB0T 的 physical-robot evaluation：

- 3 台 Franka DROID 平台，4 个环境（kitchen、workroom、bedroom、office），2 个机构。
- 每个环境 10 个 pick-and-place 任务，每任务 3 trials，因此每个 policy 120 trials。
- 成功定义：在 900 steps 内把指定物体稳定放入/放到 receptacle。
- 5 个 policy：MolmoB0T、MolmoB0T-Img、MolmoB0T-Pi0、π0-DROID、π0.5-DROID。
- S² 用 public checkpoints 的 joint LLM-output 特征，在 6000-episode DROID 子集上算，h=110。
- 结果：r=+0.831，one-tailed p=0.041。

关键限定：**h=110 是 exploratory horizon；prespecified h=7 只给 r=+0.429、p=0.236。** 所以真机结论是“在这个统一设置、这个 horizon 下观察到正相关”，不是“S² 在所有真机设置下都稳定”。论文作者也把“单一真实机器人设置、少量模型”列为 limitation。

### 5.4 模型族汇总表

| 模型族 | 数量 | 参考数据 | 特征位置 | h | 评测基准 | 结果 | 统计限定 |
|--------|------|----------|----------|---|----------|------|----------|
| Fine-tuned VLA | 12 | DROID 6000 episodes | LLM-out joint | 7 | SimplerEnv-WidowX | r=+0.717 | bootstrap 95% CI [+0.436, +0.912] |
| Raw VLM | 8 | DROID | LLM-out joint | 3 | fine-tune 后 SimplerEnv | r=+0.681 | one-tailed p=0.031 |
| WAM | 6 | ALOHA 839 windows | Block 7 | 15 | RoboTwin 2.0 | r=+0.718 | one-tailed p=0.054 |
| Real-robot policy | 5 | DROID 6000 episodes | LLM-out joint | 110 | MolmoB0T 真机，120 trials/policy | r=+0.831 | one-tailed p=0.041；h=7 时 r=+0.429, p=0.236 |
| Frozen vision backbone | 14 | DROID 6000 episodes | ViT-out | 沿用主实验默认 | BC / VRF，21 tasks | BC r=+0.677；VRF r=+0.774 | BC p=0.004；VRF p<0.001；RL 无正相关 |

**数据开放范围要逐模型族说清楚**：DROID、BridgeDataV2、ALOHA 是公开数据集；12 个 VLA 的成功率来自模型论文和公开评测仓库；6 个 WAM 仅见 B.4 列名并按相同 ALOHA 轨迹计算，未说明模型或 checkpoint 是否公开；5 个真机 policy 用 public checkpoints；14 个 frozen backbone 是 Hu et al. 的 released checkpoints。论文没有逐一说明每个 checkpoint 的许可证；本 HTML 也未展开统一的 S² 计算脚本的完整复现实验说明（页面提供 Code 链接），但说明其 CKNNA 实现沿用 Huh et al. 的公开实现。因此“可复现性”是中等：数据公开、指标定义完整，但具体模型特征提取和 checkpoint 版本仍需按各模型仓库对齐。

### 5.5 表征层面的解释实验

论文用 S² 解释了四个此前只在闭环中观察到的现象。这些是**作者主张的解释**，不是因果证明：

| 现象 | 做法 | 关键数字 | 解释 |
|------|------|----------|------|
| 模态主导 | 12 个 fine-tuned VLA 分别算 vision / text / joint 的 S² | vision r=+0.694；joint r=+0.699；text r=+0.237（不显著） | fine-tune 后 VLA 主要靠视觉表征产生动作 |
| Raw VLM 反转 | 8 个 raw VLM 同样分解 | text r=+0.501；vision r=+0.425；joint r=+0.681 | 未 fine-tune 时 text-only 和 vision-only 单独都不显著，只有 joint vision-text 特征与 VLA fine-tune 后成功率显著相关 |
| 冻结 vision encoder | Qwen2.5-VL-3B + FCDecoder，BridgeDataV2，比较 frozen vs unfrozen | h=7 时 S² 降 41%，成功率降 40%；Table 10 各 horizon 降 38.1%–49.4% | 端到端 fine-tune 通过把 sensory 表征拉近 motor 表征来提升性能 |
| 迭代去噪 | 3 个 WAM，5 个 denoising 位置（0%–100%） | S² 早期就高，整体变化最多约 6%；Block 14 在 13/15 个 model-schedule 点最高 | 早期去噪特征已足够对齐，解释为何可跳过视频生成 |
| 中层 block | 3 个 WAM，30-block backbone，取 Block 0/7/14/21/29 | Block 14 在所有 3 个模型最高 | 语义推理重要的中层，也与运动最对齐 |

行为验证：LingBot-VA 在 RoboTwin 的 turn_switch 任务上跑 100 episodes，把 denoising steps 从默认 25 降到 5，成功率从 56% 降到 48%，S² 从 0.288 降到 0.241。这是**同一模型内的干预**，比跨模型相关更强，但仍然只有一个任务、100 episodes，不能外推成普遍规律。

## 6. 能力与失败模式

### 6.1 能做什么

- **模型排序**：在闭环前给一组 VLA、raw VLM backbone、WAM 排序，优先测 S² 高的模型。
- **backbone 筛选**：在昂贵的 VLA fine-tune 之前，用 raw VLM 的 S² 判断哪个 backbone 更可能成功。
- **表征诊断**：比较 vision / text / joint 特征、不同层深、不同 denoising step 的对齐度，解释“为什么这个模型成功/失败”。
- **跨参考数据迁移（单臂内）**：DROID 和 BridgeDataV2 虽然机器人、场景、任务不同，但对 12 个 VLA 给出接近的相关；说明参考数据不必与目标机器人完全一致，但 arm configuration 要匹配。

### 6.2 不能做什么

- **不能替代闭环**：作者明确说这是 correlational study，闭环仍是模型选择的最后一步。
- **不能保证同 S² 同成功率**：作者 limitation 2 明确说，S² 相近的模型仍可能成功率不同。
- **不能证明训练时优化 S² 会提升成功率**：作者 limitation 1 明确说尚未证明。
- **不能处理 arm configuration mismatch**：双臂 WAM 用单臂 DROID 参考，r 从 +0.999 掉到 +0.246。
- **不能外推到所有 RL 设置**：在 Meta-World 两个 seed set 和 Robosuite 上，RL 没有正相关，r 在 -0.193 到 -0.002。
- **不能把 3 点 r=+0.999 当强证据**：视频质量对比只有 3 个 WAM，统计上非常脆弱。
- **不能把 h 当记忆窗口**：S² 不读历史帧，也不把未来 h 步当作 policy 的历史上下文或记忆；它只是用未来轨迹定义 motor 邻居。

### 6.3 隐含假设（Hidden Assumptions）

1. **单帧图像 + 指令足以代表 sensory representation。** 论文没有用历史帧或视频上下文；如果任务需要记忆或多帧因果，S² 可能测不到。
2. **未来 h 步运动轨迹足以代表 motor representation。** 对 DROID/BridgeDataV2 是末端位姿+夹爪，对 ALOHA/WAM 是双臂关节位置动作；它不区分接触、物体状态、任务阶段；两条运动轨迹相同但接触力不同的 episode 会被视为 motor 近邻。
3. **参考数据集的 episode 分布与目标评测分布可比。** 论文用 DROID/BridgeDataV2/ALOHA 做参考，但没有证明这些分布覆盖目标任务的视觉-运动结构。
4. **模型特征可访问。** 闭源模型或只给 API 的模型无法算 S²；论文的所有模型都提供了内部特征。
5. **DTW 的绝对 pose 在各自 base frame 中比较。** 论文明确说不把轨迹变换到共享世界/相机系；跨 embodiment 时这可能引入 offset，负控制已经显示 arm mismatch 会破坏相关性。
6. **kNN 的局部结构比全局 CKA 更接近控制相关结构。** 这是设计假设，论文用 CKNNA 的局部化来支持，但没有做“全局 CKA vs 局部 CKNNA”的消融对比。
7. **成功率是足够的评测标量。** S² 只与成功率相关，不涉及安全性、平滑性、恢复能力、指令遵循度。

## 7. 与相关工作对比

| 方法 | 测什么 | 是否需要闭环 | 是否用动作/轨迹 | 优点 | 局限 |
|------|--------|--------------|------------------|------|------|
| VQA score | 语言/视觉理解 | 否 | 否 | 便宜、通用 | 与 VLA 成功率弱或负相关（r=-0.154） |
| Video quality（FID/SSIM/PSNR） | 像素生成质量 | 否 | 否 | 通用、成熟 | 与 WAM 成功率弱相关，FID 方向甚至相反（r=+0.680） |
| CKA / HSIC | 全局表征相似 | 否 | 可用 | 理论成熟 | 全局配对，不聚焦局部邻域 |
| Theia | 特征范数熵 | 否 | 否 | 简单 | 只在 encoder 组内报告相关 |
| Hu et al. | 冻结 vision encoder 与策略学习 | 部分 | 否 | 系统评测 14 个 backbone | 只覆盖 vision encoder，相关性随 policy 方法不稳 |
| **S²（本文）** | sensory kNN 与 motor kNN 的邻域对齐 | 否 | 是（未来运动轨迹；DROID/BridgeDataV2 为末端位姿+夹爪，ALOHA/WAM 为双臂关节动作） | 跨 VLA/VLM/WAM/真机；training-free；motor 邻居可复用 | 相关性非因果；需参考数据；对 arm configuration 和 horizon 敏感；计算成本未报告 |

**面试 Tip**：如果被问“S² 和 VQA/视频质量指标有什么区别”，一句话答：**“VQA 测语言理解，FID/SSIM 测像素生成，S² 测的是模型表征的近邻结构是否与未来运动轨迹（DROID/BridgeDataV2 的末端位姿+夹爪，或 ALOHA/WAM 的双臂关节动作）的近邻结构一致；它在 12 个 VLA 上 r=+0.717，但只是离线相关，不能替代闭环。”**

## 8. 精读建议

**值得精读原文的人**：

- 正在做 VLA 模型选型或 benchmark 设计的研究者：你需要知道“离线代理”能到什么程度、哪些条件下会失效。
- 要低成本筛 VLM backbone 的工程师：§3.1 的 raw VLM 实验和 §4.1 的 modality 分解直接相关。
- 做表征诊断的人：§4.2–§4.4 展示了如何用 S² 解释冻结 encoder、去噪步数、层深选择。

**建议章节路径**：

先读 §2（尤其是 t_i 已观测帧与 t_i+1..t_i+h 未来帧的区分）→ 再看 §3.1 和 §3.3（四组相关性 + 鲁棒性/负控制）→ 然后 §4（表征解释）→ 最后 Appendix A.1–A.3（采样规则、DTW、CKNNA 细节）。如果已经熟悉 PRH，可以跳过 §5 的 related work 前半部分；如果只关心工程落地，优先看 §3.3 和 §5.3 的限定条件。

**不值得精读的理由**：

如果你不做机器人学习、不需要模型选择、或者只关心在线控制律/RL 训练信号，这篇距离中等；读摘要和 §6.2 的失败模式即可。论文没有给出训练时优化 S² 的方法，也没有证明因果，因此对“如何提升 policy 性能”的直接帮助有限。

---

[← Back to Theory](./README.md)

**关键引用**：本文论文链接 https://arxiv.org/abs/2610.31670