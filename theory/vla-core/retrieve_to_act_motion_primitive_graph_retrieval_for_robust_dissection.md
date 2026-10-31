# 检索即行动：运动基元图检索提升 VLA 鲁棒执行 (Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-31
>
> **论文**: Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution
> **链接**: https://arxiv.org/abs/2610.37215
> **核心定位**: R2A 把 OOD 视觉扰动下的动作纠偏转成“成功演示的运动学图检索 + 逐维门控融合”，在不更新基座 VLA 的前提下修正空间动作轨迹；相比只抑制视觉噪声的 test-time 方法，它显式利用训练演示中的运动学先验。

**证据边界**：本文只依据 arXiv:2610.37215v1 的 HTML 正文（标注 2026-10-27，cs.RO）写作。候选摘要片段在 “formulates OO” 处截断，未用于补写论文主张；来源记录未提供 `abstract_verification`，因此不把候选摘要当作额外证据。论文内部存在若干口径不一致，本文在相应位置分别标注，例如：正文 temporal edges 用时间步 Δ∈{2,4,6,8}，附录 C 又用“每 5 帧一个节点、连向 1/2/4/8/16 个节点之后”；附录 B 的定性例子说 query 用最近 16 步运动，而附录 C 表 7 说 retriever 读取最近 8 个 pose。二者不能合并成一个“记忆窗口”。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：把成功演示组织成纯运动学图，测试时检索未来运动基元并用 learned gate 修正 frozen VLA，可提升多种扰动下的成功率；论文实测在 LIBERO-Plus 上平均提升最高 +11.9 点，RoboTwin2.0-Plus 上 π0.5 从 56.8% 到 66.2%。 |
| 适合精读 | 如果你在做 VLA 的 test-time robustness、retrieval-augmented policy、action chunk correction，重点看 §3.3–§3.5、§4 和 §5 的消融/真实机器人设置。 |
| 可以跳过 | 如果你只关心 VLA 预训练架构、大规模数据配比或新模型 backbone，这篇是外挂式方法，距离中等。 |
| 落地可行性 | 中：推理延迟从 138ms/step 到 159ms/step，额外参数约 0.37M；但需要离线建图、按基座策略训练 locator/retriever/harmonizer，且检索图没有显式场景几何。 |
| 主要风险 | 成功演示覆盖不足时检索无解；真实 OOD 若超出 harmonization 训练时的模拟噪声分布，门控可能不修正；真实实验只有两个任务、一个 Aloha 平台。 |

💡 **X-Ray 开场**

这篇论文问的是：VLA 在语义上知道“做什么”，为什么一遇到相机角度、背景、初始位姿变化就“不会做”？作者的观察是：策略自己的成功训练演示里已经包含干净的运动学执行先验，只是 OOD 视觉把“当前观测”和“历史成功运动”的对应关系打乱了。于是 R2A 不再去修视觉特征，而是把演示做成运动基元图，测试时用当前运动 query 定位并检索未来 chunk，再用逐维门控把检索结果与 VLA 预测融合。对 VLA 研究者意味着：鲁棒性可以作为 frozen policy 的 test-time wrapper 来做，但代价是要维护图、训练纠正器，并接受“图中没有的运动”无法被恢复。

📍 **研究全景时间线**

```
2024  π0 / OpenVLA 等 VLA 基础模型出现（论文引用）
      ↓
2025  LIBERO-Plus 系统评测 VLA 鲁棒性；π0.5、GR00T N1 等基座出现（论文引用）
      ↓
2026  R2A：测试时运动基元图检索（arXiv:2610.37215v1，2026-10-27）
      ↓
局限：检索图无显式场景几何；不支持图中不存在的新恢复序列
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

R2A 是挂在 frozen VLA 外面的 test-time 模块。它不改基座参数，但会增加离线图构建、初始状态纠正、每步检索和门控融合。

| 模块 | 输入 | 输出 | 时序/频率 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| Graph-Index | 成功演示 D | 运动基元图 G=(V,E) | 离线，每个 evaluation unit 建一次 | 图本身无参数；节点为运动学段，边为时序/空间关系 |
| Rectify | OOD 初始关节 q0_OOD、任务指令 ℓ | 校正后的 q0_rect | 仅 episode 开始 t=0 | diffusion 模型离线训练；推理时从高斯噪声去噪 |
| R2A-Retriever | 当前 query Q_t、图 G | 未来参考动作 A_t_ref | 每次 VLA 查询时 | query-dependent GNN + top-K 加权；离线训练 |
| Harmonize | A_t_ref、VLA 预测 A_t_π | 执行动作 A_t_exec | 每次查询时 | learned dimension-wise gate；离线训练，基座 frozen |
| Frozen VLA | 观测、指令 ℓ | 动作 chunk A_t_π | 按各基座设置查询 | 不更新；真实机器人实验前会先 fine-tune 基座作为 baseline |

**逐模型归属（不能把不同模型的属性合并）**

| 模型/对象 | 仿真设置 | 每次查询执行 | 训练/开放范围 | 备注 |
|-----------|----------|--------------|----------------|------|
| GR00T N1.7 | LIBERO-Plus，released checkpoint，per suite | 8 actions/query | 不 fine-tune 基座 | 真实机器人实验另用 3B checkpoint fine-tune：150 demos/task、batch 128、lr 1e-4、6000 training steps、4 denoising steps；R2A 加在同一 fine-tuned policy 上 |
| π0.5 | LIBERO-Plus 用 `pi05_libero`（openpi） | 5 actions/query | 不 fine-tune 基座 | RoboTwin2.0-Plus 用作者发布的 co-train joint checkpoint；整个 50-step joint chunk 执行完再查询 |
| StableVLA | LIBERO-Plus，per suite | 8 actions/query | 不 fine-tune 基座 | 论文未报告其真实机器人或 RoboTwin 实验 |
| R2A 模块 | LIBERO-Plus / RoboTwin2.0-Plus | 每次查询输出 C 步参考 | 图与 start-state alignment 只依赖演示，可跨基座共享；locator/retriever/harmonization 读基座输出，按基座训练 | 总计约 0.37M trainable params；graph 无参数 |

**已观测历史帧 vs 未来预测帧**

| 对象 | 已观测历史 | 未来预测/动作 chunk | 证据与注意 |
|------|------------|----------------------|------------|
| Query Q_t | x_{t−H:t}、A_{t−H:t}_exec | A_t_π_pose（VLA 当前预测的未来 C 步） | 正文只给符号 H，未统一数值；附录 B 定性例子说最近 16 步运动；表 7 说 retriever 读取最近 8 个 pose。不能把 16 或 8 写成所有实验共有的记忆窗口 |
| 图节点 v | q_v、δq_v、γ_v、PE(s̄_v) | 节点关联的 A_v_demo 是记录的未来 C 步 pose | 节点每 5 帧一个；segment 16 frames；temporal edges 正文写 Δ∈{2,4,6,8} 时间步，附录 C 写 1/2/4/8/16 个节点 ahead，二者索引不同 |
| 检索参考 A_t_ref | 来自 top-K 节点的未来 chunk | K=8；先 shortlist M=24 | A_t_ref 是未来预测，不是历史上下文 |
| 动作 chunk C | — | LIBERO：GR00T/StableVLA 为 8，π0.5 为 5；RoboTwin：π0.5 为 50-step joint chunk | 动作 chunk 长度是执行接口，不是记忆窗口 |

> 均匀间隔下，K 个采样帧的首末跨度是 `(K−1)×Δt`；若从当前参考 t 预测 t+1 至 t+H，终点距当前参考是 `H×Δt`。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。论文没有把 C 或 K 统一成“所有实验共有的记忆窗口”，本文也不这样写。

### 1.2 关键机制

- **纯运动学图**：节点编码关节 q_v、段内关节位移 δq_v、夹爪状态 γ_v，并加相对时间位置 PE(s̄_v)；视觉特征被明确丢弃。这样做的原因是 OOD 主要破坏视觉观测，而运动学执行仍可能落在训练分布内。
- **Query-dependent GNN**：消息传递直接以 Q_t 为条件，让 GNN 在图上传播“当前执行与哪些历史段相关”，而不是做静态视觉 embedding 最近邻。之后对节点 relevance 做 marginalization，得到任务分布 ŷ_t 和进度 p̂_t，再用它们重排。
- **Diffusion 初始状态纠正**：当 q0 明显偏离训练初始分布时，先从扩散模型采样一个 in-distribution q0_rect，再用刚性纠正轨迹把机器人带过去。它只在 t=0 发生，不负责运行中的错误恢复。
- **逐维门控融合**：S_t 是 C×6 的门控矩阵，只修正 pose 维，夹爪命令保持 VLA 原样。门控由 Q_t、A_t_π_pose、A_t_ref 共同决定，目的是“只在 VLA 空间动作偏离几何先验时介入”，保留局部反应性。
- **训练目标**：GNN 用 BCE + ranking loss 学习节点相关性；harmonization MLP 用模拟噪声注入的 query 训练，让门控在 VLA 预测受扰时更依赖检索参考。

⚡ **Eureka Moment**：把 OOD 鲁棒性从“抑制坏视觉特征”转成“在成功演示的运动学图中定位当前执行，并检索未来 chunk”；视觉被绕开，纠正发生在动作空间。

### 1.3 信息流/架构图

```
成功演示 D
   │
   ▼
Graph-Index(D) ──► G = (V, E)
                     │
t=0: q0_OOD ──► Rectify(diffusion) ──► q0_rect
                     │
每次查询 t:
  观测 + ℓ ──► frozen VLA π ──► A_t_π = [A_t_π_pose | A_t_π_grip]
                     │
  A_t_π_pose + x_{t−H:t} + A_{t−H:t}_exec
                     │
                     ▼
                 Encoder ──► Q_t
                     │
                     ▼
        query-dependent GNN over G
                     │
                     ▼
        w_v ──► marginalize ŷ_t, p̂_t ──► re-rank
                     │
                     ▼
        top-K 节点 ──► A_t_ref = Σ α_v A_v_demo
                     │
                     ▼
  S_t = σ(MLP([Q_t ‖ A_t_π_pose ‖ A_t_ref]))
                     │
                     ▼
  A_t_exec_pose = A_t_π_pose + S_t ⊙ (A_t_ref − A_t_π_pose)
  A_t_exec_grip = A_t_π_grip
                     │
                     ▼
                  机器人执行
```

## 2. 数学核心

📌 **Napkin Formula**：

```
A_exec = A_π + Gate(Q_t, A_π, Retrieve(G, Q_t)) ⊙ (A_ref − A_π)
```

目标：在 frozen VLA 预测 A_π 可能被 OOD 视觉带偏时，用运动学图检索出的 A_ref 做残差修正；门控决定每个 pose 维修多少。

**图节点初始化**

```
h_v(0) = [ norm(q_v, δq_v, γ_v) ‖ PE(s̄_v) ]
```

- q_v：节点对应段的关节配置，正文写 7 维。
- δq_v：段内关节位移。
- γ_v：二值夹爪状态。
- PE(s̄_v)：节点在演示中的相对时间位置编码。
- y_v：任务标签；p_v∈[0,1]：任务完成进度。

**初始状态扩散纠正**

```
L_Diff(φ) = E_{q0~D, ε~N(0,I), k} [ ‖ε − ε_φ(q^k, k, ℓ)‖² ]
```

推理时从 q^K~N(0,I) 迭代去噪，得到 q0_rect；再计算从 q0_OOD 到 q0_rect 的刚性纠正轨迹。直觉：先把“起手姿势”拉回训练分布，避免后续 query 一开始就落在图外。

**Query 构造**

```
Q_t = Encoder( [ A_t_π_pose ‖ x_{t−H:t} ‖ A_{t−H:t}_exec ] )
```

- A_t_π_pose：VLA 当前预测的未来 C 步 pose，是未来预测帧。
- x_{t−H:t}、A_{t−H:t}_exec：最近 H 步已观测状态与已执行动作，是历史帧。
- H 未在正文统一给出；不能把附录 B 的 16 步或表 7 的 8 poses 直接当成所有实验共享的记忆窗口。

**Query-dependent message passing**

```
m_{u→v}(l) = Message(l)( h_u(l−1), h_v(l−1), Q_t )
h_v(l) = Update(l)( h_v(l−1), Aggregate_{u∈N(v)} m_{u→v}(l), Q_t )
w_v = Softmax( MLP( h_v(L), Q_t ) )
```

直觉：同样的图，在不同 query 下消息传递被调制，最后得到每个节点与当前执行的相关性 w_v。

**Marginalization 与重排**

```
ŷ_t = Σ_v w_v y_v
p̂_t = Σ_v w_v p_v
Q̃_t = MLP( [ Q_t ‖ ŷ_t ‖ p̂_t ] )
s(v) = cosine( Q̃_t, h_v(L) ) + Bias(y_v, p_v, ŷ_t, p̂_t)
```

ŷ_t 估计“当前在做哪个任务”，p̂_t 估计“做到哪一步”；Bias 把任务/进度一致性加入检索分数。

**Top-K 参考动作融合**

```
A_t_ref = Σ_{v∈V_K} α_v A_v_demo
α_v = exp(s(v)/τ) / Σ_{u∈V_K} exp(s(u)/τ)
```

- V_K：top-K 节点集，K=8。
- A_v_demo：节点 v 对应演示段记录的未来 C 步 pose。
- τ：温度，控制融合尖锐程度。
- 注意：这里的 A_t_ref 是未来 chunk，不是历史记忆。

**门控融合**

```
S_t = σ( MLP( [ Q_t ‖ A_t_π_pose ‖ A_t_ref ] ) )
A_t_exec_pose = A_t_π_pose + S_t ⊙ ( A_t_ref − A_t_π_pose )
A_t_exec_grip = A_t_π_grip
```

- S_t∈[0,1]^{C×6}，逐维门控。
- 当 S_t→0，完全信任 VLA；当 S_t→1，完全采用检索参考。
- 夹爪维不修正，保持离散命令。

**检索训练损失**

```
L_BCE = −(1/|P_u|) Σ_{v∈P_u} log w_v − (1/|N_u|) Σ_{v∈N_u} log(1−w_v)
L_RANK = −(1/|P_u|) Σ_{v∈P_u} w_v / Σ_{v'∈N_u} w_v'
L = α L_BCE + (1−α) L_RANK
```

直觉：BCE 把正/负节点分开；ranking loss 在稀疏正样本下显式拉大正负 margin。

**Harmonization 训练**

```
L_Harm(θ) = E_{t~D, ε} [ ‖ A_t_π_pose + S_t(Q̃_t;θ) ⊙ (A_t_ref − A_t_π_pose) − A_t_gt_pose ‖² ]
```

训练时给 query 注入模拟噪声 ε，使门控学会在 VLA 预测受扰时更依赖 A_t_ref。这是“模拟 OOD”而不是真实 OOD 的监督。

## 3. 带数字走一遍：玩具例子

> 以下为假设的 2D 玩具例子，只用于验算门控融合，不代表论文实验。

假设 C=1，把 6 维 pose 简化为 2 维；VLA 预测 `A_π = (1.0, 0.0)`，检索参考 `A_ref = (0.2, 0.8)`，门控 `S = (0.75, 0.25)`。

```
A_exec = A_π + S ⊙ (A_ref − A_π)
       = (1.0, 0.0) + (0.75, 0.25) ⊙ ((0.2, 0.8) − (1.0, 0.0))
       = (1.0, 0.0) + (0.75×(−0.8), 0.25×0.8)
       = (1.0 − 0.6, 0.2)
       = (0.4, 0.2)
```

验算边界：若 S=(0,0)，A_exec=A_π=(1.0,0.0)；若 S=(1,1)，A_exec=A_ref=(0.2,0.8)。本例中第一维施加了所需修正的 75%（0.6/0.8），第二维施加了 25%（0.2/0.8），因为门控判断第二维 VLA 预测更可信。这个例子说明 R2A 不是“用检索替换策略”，而是逐维修正。

再看 top-K 加权：假设 K=2，s(1)=2，s(2)=0，τ=1，则 α1=e²/(e²+1)≈0.881，α2≈0.119。若 A1=(0.2,0.8)，A2=(0.25,0.75)，则：

```
A_ref ≈ 0.881×(0.2,0.8) + 0.119×(0.25,0.75)
      ≈ (0.20595, 0.79405)
```

若仍用 S=(0.75,0.25)，A_exec≈(0.4045,0.1985)。这说明检索加权和门控是两级“软选择”：先选历史段，再选每个动作维的信任度。

## 4. 工程视角

| 指标 | 论文实测/设置 | 工程含义 |
|------|----------------|----------|
| 推理延迟 | R2A 159ms/step vs base 138ms/step；VLS >800ms；SDN 需外部检测/分割 | 额外约 21ms/step；但论文称 per-step，基座实际按 chunk 查询，部署时要区分“每次查询延迟”和“控制周期” |
| 参数量 | R2A 约 0.37M trainable params；graph 无参数 | 模型很小，主要成本在图存储、检索和离线训练 |
| 图规模 | LIBERO-Plus 13k–28k nodes/evaluation unit；RoboTwin 8k–10k nodes/arm | 需要维护每套演示的图；跨任务检索依赖 task prior 和 align relation |
| 检索预算 | shortlist M=24，top-K=8，segment 16 frames | 推理时只融合 8 段，计算可控；但 K 和 τ 是超参 |
| 训练成本 | GNN 30 epochs；retriever 30 epochs、lr 1e-3、weight decay 1e-5、4,096 queries/epoch、batch 256、8 hard negatives/query；start-state 6,000 steps each | 训练不需要机器人 rollout，只用缓存演示和 frozen policy 回放动作；但按基座策略分别训练 locator/retriever/harmonizer |
| 真实机器人 | Mobile ALOHA，双 6-DoF 臂，top-down + wrist RGB 640×480；150 demos/task；GR00T N1.7 3B fine-tune 6,000 steps；4 denoising steps | 真实部署需要先 fine-tune 基座作为 baseline，再加 R2A；不是“零数据”方案 |

工程上的关键 trade-off 是：R2A 用离线图换 test-time 鲁棒性，用逐维门控保留局部反应性；但它没有显式场景几何，遇到新障碍物时只能通过 VLA 输出间接体现，检索图本身不会“看到”障碍。另一个约束是动作空间：LIBERO 用 pose 动作，RoboTwin 用 joint targets，且每臂单独建图、双臂联合检索；不能把一个设置的动作维度和 chunk 长度直接搬到另一个设置。

## 5. 数据与评测

### 5.1 预训练与基座策略

- 论文使用三个 released VLA：GR00T N1.7、π0.5、StableVLA。LIBERO-Plus 上均按 released checkpoint 使用，不 fine-tune 基座。
- 真实机器人实验不同：先用 150 条/task 的 teleoperated demonstrations fine-tune GR00T N1.7 3B，得到 baseline；R2A 加在这个 fine-tuned policy 上，且不更新它。因此“frozen VLA”指 R2A 训练时不更新该策略，不表示基座从未微调。
- R2A 的训练只读取演示和 frozen policy 在演示上的回放动作，不需要评测 episode，也不需要机器人 rollout；推理/真实评测仍需要观测与部署。

### 5.2 仿真评测：LIBERO-Plus 与 RoboTwin2.0-Plus

LIBERO-Plus 的 evaluation unit 是一个 LIBERO suite；每个 suite 用 500 条演示（50/task）建一张图，图含 13k–28k 节点。每个扰动 episode 在七类扰动下各跑一次；对比 OOD 方法和消融使用 LIBERO-Long。RoboTwin2.0-Plus 以十个双臂任务为一组，每任务每类扰动 10 trials；π0.5 用 co-train joint checkpoint，整段 50-step joint chunk 执行完再查询，夹爪闭合后每 10 action steps 查询未来 80 steps。

**论文实测的平均成功率（表 1、表 2）**

| 基准 | 基座 | 基座平均 | +R2A 平均 | 绝对提升 |
|------|------|----------|-----------|----------|
| LIBERO-Plus | GR00T N1.7 | 77.1% | 89.0% | +11.9 点 |
| LIBERO-Plus | StableVLA | 78.1% | 89.5% | +11.4 点 |
| LIBERO-Plus | π0.5 | 85.9% | 90.4% | +4.5 点 |
| RoboTwin2.0-Plus | π0.5 | 56.8% | 66.2% | +9.4 点 |

条件性读法：最大提升集中在 Robot Initial States。LIBERO-Plus 上 GR00T +49.1、StableVLA +48.0、π0.5 +21.9；RoboTwin 上 π0.5 从 34.2% 到 75.9%。但 π0.5 在 Lighting 上出现 -0.8 点的小幅下降（表 1：97.1→96.3；Background 实为 +1.5，95.6→97.1），GR00T 在 Background 上 -0.1 点；StableVLA 七类均正增长。来源正文 Q1 将 π0.5 的下降误归到 Background，表 1 与正文在此处矛盾，本文按表 1 归属并保留该负面结果。不能说“所有模型所有扰动都稳定提升”。

论文表 3 还在 LIBERO-10 上与 VLS、SDN、ICL 比较：π0.5 平均 80.9→86.3，SDN 83.8，ICL 77.7；GR00T N1.7 平均 67.0→82.1，SDN 70.5，ICL 74.5。作者解释 SDN 依赖 object-centric negative references，在相机/初始状态这类 execution-context shift 下不可靠；这是作者主张，不是本文独立复现。

### 5.3 真实机器人设置

Mobile ALOHA，双 6-DoF 臂，top-down RGB + wrist RGB，640×480；每任务 150 条 teleoperated demonstrations。任务为 Banana2Pot 与 Cube2Drawer；Robot Initial States 扰动把机械臂放到演示初始位姿范围之外，物体不动。

**论文实测（表 4，成功率 %）**

| 任务 | 方法 | Base | Robot Initial States |
|------|------|------|----------------------|
| Banana2Pot | GR00T N1.7 | 60.0 | 30.0 |
| Banana2Pot | + SDN | 75.0 | 55.0 |
| Banana2Pot | + R2A | 80.0（+20.0） | 80.0（+50.0） |
| Cube2Drawer | GR00T N1.7 | 55.0 | 10.0 |
| Cube2Drawer | + SDN | 70.0 | 30.0 |
| Cube2Drawer | + R2A | 85.0（+30.0） | 45.0（+35.0） |

这里有一个口径问题：摘要称真实世界最高 +40%，但表 4 的 Robot Initial States 列显示 Banana2Pot +50 点、Cube2Drawer +35 点；Base 列是 +20/+30 点。二者不是同一个统计口径，本文只按表 4 陈述，不把“+40%”当作表 4 的最大值。表 4 未在摘录中给出每任务试次数，因此无法核验置信区间，暂标 TODO。

### 5.4 消融

GR00T N1.7 在 LIBERO-Plus 上的组件消融（论文表 5）：

| 设置 | 平均成功率 |
|------|------------|
| GR00T N1.7 | 77.1% |
| R2A w/o GNN Retrieval | 63.8% |
| R2A w/o Harmonization | 1.2% |
| R2A w/o Ranking Loss | 80.7% |
| R2A | 89.0% |

Robot Initial States 下，w/o Initial State Rectification 为 49.9%，R2A 为 87.5%。读法：直接回放检索动作（w/o Harmonization）几乎崩溃，说明“检索到”不等于“能用”；GNN 检索也比直接 top-k 选择更稳。w/o Ranking Loss 仍有 80.7%，高于 base 但低于完整 R2A，说明 ranking loss 是增益项而非唯一关键。

## 6. 能力与失败模式

| 能力 | 具体场景 | 条件 |
|------|----------|------|
| 视觉 OOD 下的空间轨迹纠偏 | 相机视角、光照、背景、传感器噪声 | 演示图覆盖相似运动；门控学会在预测偏离时介入 |
| 初始状态偏移恢复 | Robot Initial States | 扩散纠正把 q0 拉回训练流形；否则后续 query 可能落在图外 |
| 跨任务运动共享 | LIBERO-Plus 一个 suite 内跨任务检索 | 依赖 task prior、progress 和 align relation；不是任意任务泛化 |
| 双臂操作 | RoboTwin2.0-Plus 的 π0.5 | 每臂单独建图，双臂联合检索；50-step joint chunk 设置专用 |
| 真实 Aloha 部署 | Banana2Pot、Cube2Drawer | 仅两个任务；基座先 fine-tune；真实扰动只测初始状态 |

| 失败模式 | 原因 | 证据/边界 |
|----------|------|-----------|
| 新障碍物/场景几何变化 | 检索图没有显式 scene geometry，只能依赖 VLA 输出传递变化 | 作者在 Limitations 明确承认 |
| 图中不存在的新恢复序列 | 图只由成功演示构成；显式 error recovery 需要新纠正序列 | 作者承认尚未解决 |
| 真实 OOD 超出模拟噪声 | harmonization 用注入噪声训练，真实视觉偏移可能产生分布外预测误差 | 作者承认可能失效 |
| 演示覆盖不足 | 检索不到结构兼容的未来段；top-K 也可能是错段 | 方法前提是成功演示足够丰富 |
| 动作空间/embodiment 限制 | 依赖 7 维关节、6D pose、二值夹爪；RoboTwin 需 joint-space 适配 | 不能直接外推到其他机器人 |
| 真实实验规模小 | 两个任务、一个 Aloha 平台、初始状态扰动 | 不能声称广泛真实泛化 |
| 评测方差 | 表 4 未给试次数；仿真每类扰动 episode 数不一 | 需要作者补充置信区间/配对检验细节 |

### 6.1 隐含假设

- **成功演示覆盖假设**：测试时需要的运动基元已在训练演示图中；如果 OOD 需要全新运动，R2A 无解。
- **运动学可定位假设**：当前执行在图中存在结构兼容的节点；视觉 OOD 不一定破坏运动学，但严重动力学偏差可能破坏。
- **初始状态可纠正假设**：扩散模型能采样出有效 q0_rect，且刚性纠正轨迹物理可行、安全。
- **门控可泛化假设**：harmonization 在模拟噪声上训练，能迁移到真实视觉 OOD；作者自己指出这可能不成立。
- **任务语义不变假设**：扰动主要是视觉/初始状态，语言指令和任务目标不变；语言扰动下的提升较小。
- **夹爪不修正假设**：只修正 pose，夹爪保持 VLA 原命令；如果 OOD 导致夹爪时序错误，R2A 不直接处理。

## 7. 与相关工作对比

| 方法 | 关注点 | 检索/纠正对象 | 是否依赖视觉 | 训练方式 | 主要局限 |
|------|--------|----------------|--------------|----------|----------|
| BYOVLA | 抑制无关视觉区域 | 图像 inpainting | 是 | test-time | 依赖外部视觉干预 |
| VLS | VLM 奖励引导动作 | 外部 VLM reward | 是 | test-time | 延迟 >800ms；依赖 VLM 质量 |
| SDN | 负参考引导去噪 | diffusion noise steering | 是 | test-time | object-centric negative refs 在 context shift 下不可靠 |
| ProbeAct | 3D 跟踪失败恢复 | 3D positional tracking | 是 | test-time | 聚焦抓取失败 |
| RAM | affordance memory | 可复用 affordance | 是 | 外部记忆 | 静态匹配，非时序执行 |
| Retrieve, Don’t Retrain | 检索轨迹条件化 frozen policy | 轨迹 | 部分 | test-time | 避免 per-task finetuning；非运动学图 |
| FlowRetrieval | 光流引导 few-shot 检索 | 视觉光流 latent | 是 | 检索 | 视觉 OOD 下脆弱 |
| CLASS | 动作序列对比学习 | 运动相似度 | 是 | 预训练 | 不是 test-time 检索 |
| ReGIL | 运动相似度辅助奖励 | 局部轨迹对齐 | 是 | 辅助奖励 | 仍依赖视觉检索管线 |
| R2A | 纯运动学图 + query-dependent GNN | 未来运动基元 chunk | 检索阶段否 | 离线训练 test-time 模块 | 无场景几何；需成功演示图 |

**面试 Tip**：被问到时先一句话定位——“R2A 不是新 VLA，而是 frozen VLA 的 test-time geometric retrieval wrapper”；然后讲三件事：图节点是运动学段、query 是当前预测加历史、gate 是逐维残差；最后主动说消融里 harmonization 最关键（去掉后 1.2%），以及它没有场景几何、真实实验只有两个任务。

## 8. 精读建议

- **值得精读原文的人**：做 VLA test-time robustness 的研究者；想把 retrieval 做成工程外挂的工程师；需要评估“不重训基座”方案落地成本的人。
- **建议章节路径**：先读 §3.3–§3.5 和 §4，理解图、query、gate 的数据流；再看 §5 表 1/2/4/5，核对提升与消融；最后读附录 C.1/C.2，确认每个基座的动作 chunk、图规模、训练轮次和评测频率；附录 A 的 paired outcomes 适合看“修好多少、弄坏多少”。
- **不值得精读的理由**：如果你不做机器人学习、只关心 VLA 预训练架构或大规模数据配方，读摘要和 §1 即可；如果你已经熟悉 retrieval-augmented policy，重点只看 §3.5 的 query-dependent GNN 和 §4 的门控即可。

---

[← Back to Theory](./README.md)

**关键引用链接**：https://arxiv.org/abs/2610.37215