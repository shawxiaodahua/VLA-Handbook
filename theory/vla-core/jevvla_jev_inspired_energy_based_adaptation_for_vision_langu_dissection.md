# JevVLA：用同一个能量标量为离散候选与连续动作块做轻量适配 (JevVLA: Jev-Inspired Energy-Based Adaptation for Vision-Language-Action Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-28
>
> **论文**：JevVLA: Jev-Inspired Energy-Based Adaptation for Vision-Language-Action Policies
> **链接**: https://arxiv.org/abs/2610.35010
> **核心定位**：在不更新 frozen VLA 的前提下，用一个约 8 万参数的条件能量适配器同时做两件事：能量值给离散候选排序，能量对动作的梯度给连续动作做有界修正或生成引导。

**来源说明**：本文依据 arXiv:2610.35010v1（26 Oct 2026）正文与表格；来源记录中的候选摘要只到 “framewor” 截断，未包含量化结果，不能替代正文。正文摘要与表 1 给出 1.28、2.31、1.60、1.52、1.28 pp 的增益，本文数字均取自正文表 1 至表 8。未发现需要以候选摘要覆盖正文的实质冲突；若后续 PDF 版本变化，以对应 PDF 正文为准。

## ⚡ 快速判断（30 秒读完这段就够）

| 维度 | 判断 |
|------|------|
| 核心结论 | frozen base policy 外挂一个条件标量能量：值排序候选，梯度修正连续动作；五个仿真 benchmark 的 aggregate success 提升 1.28 至 2.31 pp |
| 适合精读 | 做 frozen VLA post-hoc adaptation、candidate reranking、energy-based guidance 的研究者与工程师 |
| 可以跳过 | 只关心真实机器人部署、全参数微调、跨回合记忆或大规模预训练配方的人 |
| 落地可行性 | 中：参数少、接口清楚；但每个 benchmark 独立权重与固定编解码，且依赖候选/专家/演示监督 |
| 主要风险 | 增益不大且任务间有负迁移；无真实机器人实验；未报告延迟、吞吐、置信区间；没有显式记忆机制 |

💡 **X-Ray 开场**  
这篇论文解决的是“冻结的 VLA 已经能提出动作，但动作在具体场景里不一定成功”的适配问题。它的发现是：不必更新 base policy，只学一个条件能量函数，就能把“候选比较”和“连续修正”接到同一个标量上。对 VLA 研究者而言，这意味着后训练适配可以拆成“proposal 由 frozen policy 给，evaluation 由轻量 adapter 学”，但共享的是核心结构，不是共享一组跨 benchmark 的权重。

📍 **研究全景时间线**

```
2020 MetaWorld → 2022 CLIPort / Implicit BC → 2023 VIMA / Diffusion Policy
→ 2024 π0 → 2025 π0.5 / OpenVLA → 2026 LIBERO Plus / JevVLA ← 当前位置
```

这条线只列论文正文可确认的节点。JevVLA 的位置是 2026 年的“frozen VLA 后置适配器”分支：不训练新 VLA，而是在已有策略外面学习动作偏好。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

JevVLA 的共享核心是一个条件标量能量 `Eθ(c,z)`。`c` 是固定 context 编码，`z` 是固定 action 编码。核心网络 width 128、rank 8、79,817 个可训练参数；每个 benchmark 使用相同网络结构，但独立训练权重，输入/输出编解码也按 benchmark 固定。

| 组件 | 输入 | 输出 | 训练/推理差异 |
|------|------|------|----------------|
| context encoder `χ_b` | 当前观测 `o`、任务指令/多模态提示 `ℓ`、frozen policy 特征 | `c ∈ R^128` | 参数固定；可用 generation time、control progress、proposal summary 等上下文 |
| action encoder `φ_b` | base policy 的 native action `a` | `z ∈ R^32` | 固定；动作可为 pick-and-place 命令、位移或 action chunk |
| energy core `Eθ` | `c, z` | 标量能量 | 只训练 adapter；base policy 冻结 |
| 离散选择 | 候选集 `A={a_k}`、可选 native score `r_k` | logits `s_k`、softmax 概率、`argmax` | 值用于比较候选；可加 Direct margin `m` |
| 连续修正 | `c, a_0`、能量梯度 | 修正动作 `a'` | 梯度用于有界修正；或插入 Euler 生成步 |
| 训练信号 | 执行结果、专家修正、演示偏好 | ranking loss 或 gradient loss | 每个 benchmark 按可用监督选择损失 |

**逐 benchmark 归属**（不要把某一模型的属性合并给全部模型）：

| Benchmark | Frozen base policy | 动作接口与 adapter 用法 | 监督/数据范围 | 记忆与上下文 | 评测规模 |
|-----------|--------------------|--------------------------|----------------|----------------|----------|
| MetaWorld MT50 | `π0.5–MT50` | 连续动作；用能量梯度修正 | physically executed branch returns + scripted expert corrections；训练/评测共享 50 个任务身份，但 test episodes/seeds 留出 | 未报告显式记忆；固定 context 编码 | 2,500（50 任务 × 50） |
| VIMA L1 | VIMA-200M | 离散候选；用能量值排序 | executed candidate outcomes + expert alignment；5 个训练候选 | 未报告显式记忆；候选来自当前 proposal | 1,300（13 族 × 100） |
| LanguageTable | language-conditioned BC | 连续动作；用能量梯度修正 | scripted oracle 在 observed 与 later states 提供监督；每步 16 个初始 anchor、最多 128 个 future states | 未报告跨回合记忆；future states 只是训练监督 | 250（5 族 × 50） |
| LIBERO Plus | `π0.5–LIBERO`（原 LIBERO 预训练） | 离散候选；用能量值排序 | future-action proximity 到官方 demonstration trajectories；2 个候选 | 未报告显式记忆；future proximity 只是训练目标 | 10,030 个变体，variant-weighted aggregate |
| CLIPort | CLIPort | 离散候选；用能量值排序 | physically executed candidates 的 terminal outcomes；5 个执行候选 | 未报告显式记忆；闭环由 frozen policy 继续 | 1,032 fit / 258 val / 1,800 test |

共享的是核心网络结构与参数预算，不是权重；五个 benchmark 各自训练 adapter。论文也没有发布新的数据集或数据包；“实现将开源”是作者声明，本次来源未包含仓库内容，无法核验代码或数据许可。

### 1.2 关键机制

- **冻结 base policy，只学 evaluation**：base policy 继续提出 native action，adapter 不改变其参数。这样保留了预训练策略的动作生成能力，适配成本集中在一个小标量网络上。
- **条件二次势 + 非线性残差**：二次项提供上下文相关的曲率与坐标耦合，线性项把偏好区域按 context 移动，残差网络补足二次势表达不了的非线性偏好。
- **同一个标量的两种用法**：函数值 `Eθ` 用于离散候选比较；对动作的负梯度 `−∇aEθ` 用于连续修正。排序与修正共享参数，但训练损失按可用监督选择。
- **保留 Direct 的 margin**：在 native score 为零时，挑战者只有在能量优势超过 margin 时才替换 Direct；一般选择 logit 还包含 native score 差项 (r_k−r_d)/τ0，因此该阈值不是普遍规则。CLIPort 的 margin 在 258 个验证样本上选，不在测试集上调。
- **Jev 只是接口层灵感**：论文借用了 Jev 的 Choice/Score“显式候选与相对偏好”思想；但 robot action 编码、二次能量、训练目标和连续引导都是作者自己的实现。论文明确说 Jev 公共材料没有提供完整模型与训练配方，复现 Jev 内部模型不在范围内。

⚡ **Eureka Moment**：把“动作好不好”压缩成一个条件标量——它的值负责“选”，它的梯度负责“改”；因此离散候选选择和连续动作修正不再是两套网络，而是同一个能量面的两种读取方式。

### 1.3 信息流/架构图

```
观测 o, 指令 ℓ ──► χ_b（固定）──► c ∈ R^128
                                      │
frozen π0 ──► native action a ──► φ_b（固定）──► z ∈ R^32
                                      │
                                      ▼
                           Eθ(c,z) = 1/2 zᵀQθ(c)z − bθ(h)ᵀz − Rθ(h,v)
                             │                               │
                             │ 值                             │ 对动作求梯度
                             ▼                               ▼
                  s_k = (r_k−r_d)/τ0 − β(E_k−E_d)     gθ = −∇a Eθ(c,φ_b(a))
                  + m·I[k=d]                          │
                             │                         ├─► a' = Π_B(a0 + ηgθ)
                             ▼                         └─► Euler 生成步引导
                  argmax s_k / softmax p(k|c,A)
                             │
                             ▼
                  执行候选或修正动作；frozen policy 继续闭环
```

训练时，执行结果、专家修正或演示偏好被转成 ranking target 或 local gradient target；只更新能量核心，不更新 base policy。

## 2. 数学核心

📌 **Napkin Formula**：

```
Eθ(c,z) 越小越偏好；值排序候选，−∇aEθ 修正动作
```

先给目标：学一个条件标量 `Eθ(c,z)`，让它在候选之间可比，并在连续动作空间提供局部下降方向。核心公式如下。

```
c = χ_b(o, ℓ; π0) ∈ R^128
z = φ_b(a) ∈ R^32
Eθ(c,z) ∈ R
```

```
h = fθ^c(c),  v = fθ^a(z)
Qθ(c) = diag(dθ(h)) + Lᵀ diag(qθ(h)) L
Eθ(c,z) = 1/2 zᵀQθ(c)z − bθ(h)ᵀz − Rθ(h,v)
```

```
−∇z Eθ(c,z) = bθ(h) − Qθ(c)z + ∇z Rθ(h, fθ^a(z))
gθ(c,a) = −∇a Eθ(c, φ_b(a)) = −Jφ_b(a)ᵀ ∇z Eθ(c,z)
```

离散选择的 logits 与概率：

```
s_k = (r_k − r_d)/τ0 − β(E_k − E_d) + m·I[k=d]
pθ(k|c,A) = exp(s_k) / Σ_j exp(s_j)
k* = argmax_k s_k
```

其中 `r_k` 是可选 native proposal score，`r_d` 是 Direct 的分数；没有 native score 时取 0。`τ0>0` 缩放 native score，`β≥0` 控制能量偏好强度，`m≥0` 是保留 Direct 的 margin。无效候选概率为 0。

偏好学习损失：

```
L_pref = −E[ Σ_k y_k log pθ(k|c,A) ]
```

有二元执行结果时，成对比较成功候选 `i` 与失败候选 `j`：

```
L_pair = E_x [ 1/|P_x| Σ_(i,j)∈P_x softplus(−(s_i − s_j)) ]
P_x = {(i,j): y_i^term > y_j^term}
```

连续修正的梯度目标：

```
g*(ã; μ, σ) = (μ − ã) / σ²
L_grad = E_w[ || M(gθ(c,ã) − g*(ã)) ||₂² ]
```

其中 `M` 选择被监督的动作坐标，`μ` 可来自专家动作、按偏好采样的分支动作，或朝专家移动后的执行动作：`μ = a + α(a* − a)`。当 `ã = μ + σϵ, ϵ~N(0,I)` 时，对应的 denoising score matching 目标为：

```
L_DSM = E[ (1/d_a) || σ gθ(c, μ+σϵ) + ϵ ||₂² ]
```

推理时连续修正：

```
a' = Π_Bb( a0 + η gθ(c,a0) )
```

一阶能量变化：

```
Eθ(c, φ_b(a0 + ηgθ)) = Eθ(c, φ_b(a0)) − η||gθ(c,a0)||₂² + O(η²)
```

生成引导的 Euler 步：

```
x_(n+1) = x_n + Δt_n v0(x_n,t_n) + |Δt_n| C_b[ η gθ(c_n,x_n) ]
```

`C_b` 是固定坐标 mask 与分量修正上界；`|Δt_n|` 让修正方向在时间递减的采样器里仍指向能量下降。

| 符号 | 含义 |
|------|------|
| `o, ℓ` | 当前观测与任务指令/多模态提示 |
| `π0` | frozen base policy |
| `c, z` | 固定 context 编码与 action 编码 |
| `Qθ` | 条件二次矩阵；`diag(dθ)` 给坐标曲率，低秩项耦合动作坐标 |
| `bθ` | 上下文相关的线性偏好项 |
| `Rθ` | 非线性标量残差 |
| `s_k, pθ` | 候选 logit 与 softmax 概率 |
| `gθ` | 能量对 native action 的负梯度 |
| `η` | 连续修正的全局增益 |
| `m` | 保留 Direct 的离散 margin |
| `τ0, τ_y` | native score 缩放 / 目标分布温度 |

直觉：二次项像一个随 context 变形的“碗”，把动作往偏好区域拉；低秩项让不同动作坐标互相耦合；残差修正碗形表达不了的偏好。值读取“碗底在哪”，梯度读取“往哪走”。

## 3. 带数字走一遍：玩具例子

> 以下数值是假设，仅用于验算公式，不是论文实验。

**A. 连续修正的一阶下降**

假设一维动作 `z`，能量 `E(z)=z²−z`，当前 `z0=1`，增益 `η=0.1`。梯度 `∇E=2z−1`，所以 `g=−∇E=1−2z`。在 `z0=1` 时 `g=−1`。

```
z1 = z0 + ηg = 1 + 0.1×(−1) = 0.9
E(z0) = 1 − 1 = 0
E(z1) = 0.81 − 0.9 = −0.09
实际下降 = 0 − (−0.09) = 0.09
一阶预测下降 = η||g||² = 0.1×1 = 0.1
O(η²) 修正 = 0.1 − 0.09 = 0.01
```

验算吻合：`E(z0+ηg)=E(z0)−η||g||²+O(η²)`。

**B. 离散候选的 margin**

假设 Direct 能量 `E_d=0`，挑战者能量 `E_1=−0.5`，`β=1`，native score 都为 0。若 `m=0`：

```
s_d = 0
s_1 = −1×(−0.5−0) = 0.5
p_1 = exp(0.5)/(exp(0)+exp(0.5)) ≈ 0.6225
```

挑战者胜出。若 `m=0.7`：

```
s_d = 0.7
s_1 = 0.5
log(p_1/p_d) = 0.5 − 0.7 = −0.2
p_1 = 1/(1+exp(0.2)) ≈ 0.4502
```

Direct 保留。阈值是 `m>0.5`；这对应论文的“能量优势要超过 margin 才替换 Direct”。CLIPort 的 `m` 是在 258 个验证样本上选的，玩具数值不代表论文设置。

## 4. 工程视角

**参数与训练成本**：核心 79,817 参数，width 128、rank 8。论文说 VIMA、LanguageTable、LIBERO Plus、CLIPort 的 adapter 可在 CPU 上 fit；MetaWorld 的 energy continuation 用一张 A100。frozen policy 推理与 simulator rollouts 用 GPU 服务。没有多 GPU adapter 优化。论文未报告训练总时长、推理延迟、吞吐或显存占用，因此不能从参数少直接推断端到端低延迟。

**候选选择的成本**：离散接口要评估 `K` 个候选。CLIPort 是 5 个执行候选，LIBERO Plus 是 2 个候选，VIMA 训练时是 5 个候选。候选越多，能量前向与闭环执行成本越高；但论文没有给统一的 `K` 延迟曲线。注意：候选数不是记忆窗口。

**连续修正的成本**：每个控制步一次梯度评估；MetaWorld 的 `η=0.08`、component clip 0.12，LanguageTable 的 `η=0.0004`。增益差两个数量级，说明不同动作尺度下不能直接复用同一个 `η`。Euler 引导会在每个中间动作重新评估能量与梯度，但只维护一条生成轨迹。

**接口约束**：连续修正要求 action encoder 可微；若 base policy 的动作接口不可微，只能使用能量值做候选排序。离散选择不需要 `η、σ、α`；连续修正不需要候选 margin `m`。论文明确区分了这两套推理参数。

**时序约定（阅读约定，不是论文新增实验）**：必须先区分已观测历史帧与未来预测帧。均匀间隔 `Δt` 下，`K` 个采样帧的首末跨度是 `(K−1)×Δt`；若从当前参考 `t` 预测 `t+1` 至 `t+H`，终点距当前参考是 `H×Δt`。

```
历史上下文：K 帧均匀采样，间隔 Δt → 首末跨度 = (K−1)·Δt
未来预测：从 t 预测 t+1..t+H → 终点距 t = H·Δt
```

连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。论文没有统一给出 `K` 或 `H`；LanguageTable 的“最多 128 个 future states”是训练监督，不是推理记忆；CLIPort 的 5 个候选、LIBERO 的 2 个候选也不是记忆窗口。不能把单一设置的步长或动作 chunk 长度写成所有实验共有的记忆窗口。

## 5. 数据与评测

### 5.1 预训练设置

论文不训练新的 VLA，也不更新 base policy。使用的 frozen 策略分别是：MetaWorld 的 `π0.5–MT50`、VIMA-200M、LanguageTable 的 language-conditioned BC、LIBERO Plus 的原 LIBERO 预训练 `π0.5`、CLIPort。JevVLA 只训练每个 benchmark 的能量 adapter；base policy 权重与 feature encoder 冻结。作者说没有加入 task-specific decision gate 或 teacher-policy distillation，也省略了 LoRA 微调比较，理由是这些 base policy 已在各自任务族上训练得较好，进一步低秩更新会变成另一个 policy-training 阶段。

### 5.2 仿真评测

评测指标是 task success `100S/N`，变化用 percentage points。每个成功计数是五个评测种子的平均后取整；报告率用取整计数和表中分母。论文没有给置信区间或显著性检验，因此 1.28 pp 等应视为该设置下的点估计。

| Benchmark | N | Direct (%) | JevVLA (%) | Δ (pp) |
|-----------|---|------------|------------|--------|
| MetaWorld MT50 | 2,500 | 75.84 | 77.12 | +1.28 |
| VIMA L1 | 1,300 | 84.77 | 87.08 | +2.31 |
| LanguageTable | 250 | 78.00 | 79.60 | +1.60 |
| LIBERO Plus | 10,030 | 78.63 | 80.15 | +1.52 |
| CLIPort | 1,800 | 34.22 | 35.50 | +1.28 |

CLIPort 使用严格 complete-task success，不是 partial cumulative reward：选中候选后执行，frozen policy 继续闭环到终局。CLIPort 的 fit 是 1,032 例，validation 是 258 例，test 是 1,800 例（18 变体 × 100），fit/validation 与 test 不重叠。LIBERO Plus 用 variant-weighted aggregate；perturbation、suite、original-task 是同一批结果的三种视图。

**任务依赖**：VIMA 在 13 个 task family 中 6 个提升、4 个不变、3 个下降；MetaWorld 50 个任务中 18 个提升、24 个持平、8 个下降；CLIPort 18 个变体中 11 个提升、6 个下降；LanguageTable 5 个族中 3 个提升、2 个不变。VIMA 的 `twist` 从 24% 到 56%，但 `rearrange_then_restore` 从 54% 到 47%。MetaWorld 的 `basketball` 从 28% 到 70%，`faucet open` 从 100% 到 84%，`coffee pull` 与 `stick pull` 各下降 14 pp。

**LIBERO Plus 扰动**：Camera Viewpoints +3.56 pp、Light Conditions +2.36 pp、Objects Layout +2.10 pp、Language Instructions +1.63 pp、Sensor Noise +1.50 pp、Robot Initial States +0.52 pp；Background Textures 反而 −1.95 pp。按 suite 看，LIBERO-10 +2.78 pp、Goal +3.59 pp、Spatial +0.21 pp、Object −0.64 pp。说明“aggregate 提升”不等于每个扰动都提升。

### 5.3 真实机器人设置

论文没有真实机器人实验。Figure 1 明确说机器人场景是本地模拟器渲染，用作示意，不是 JevVLA rollout frames。因此不能把五个仿真 benchmark 的结果外推到真实机械臂、移动平台或人形平台。

## 6. 能力与失败模式

**能做什么**：

- 在 frozen policy 外做轻量后置适配，参数量约 8 万，适合“不想全参数微调”的场景。
- 同一核心结构覆盖离散候选排序与连续动作修正；CLIPort 的消融显示，学习到的 contextual preference 比 uniform random、minimum norm、nearest alternative 更好。
- 小步修正可以改变闭环结果。LanguageTable 的例子中，JevVLA 在 41 步成功，Direct 到 200 步限制仍未成功；第 23 步 adapter 对 native action 的改变是 0.109 mm。这是论文报告的单例，不是统计结论。
- CLIPort 的 rescue/harm 分析提供工程可解释性：Full 替换候选 1,316/1,800（73.11%），但只有 141 例改变二元结果；82 个 rescue、59 个 harm，净增 23。随机初始化能量有 86 个 rescue 但也有 96 个 harm；shuffled labels 是 52/63。说明“多改动作”不等于“改得更好”。

**不能做什么/失败模式**：

- **任务间负迁移**：VIMA 的 `rearrange_then_restore` −7 pp，MetaWorld 的 `faucet open` −16 pp、`coffee pull` −14 pp、`stick pull` −14 pp，CLIPort 的 `separating piles unseen colors` −12 pp。aggregate 提升可能掩盖局部退化。
- **候选集限制**：离散选择只能在 base policy 给的候选里选；如果正确动作不在候选集，能量排序无法救回。论文自己也指出 finite-candidate selection 受 proposal distribution 与 candidate set 限制。
- **排序可靠不等于梯度可靠**：能正确排序观测动作，不保证候选之间有可靠的局部梯度。JevVLA 用 ranking 与 gradient 两种监督分别训练，但这不是理论保证。
- **监督依赖**：MetaWorld 需要执行分支回报与 scripted expert corrections；VIMA 需要执行候选结果与 expert alignment；LanguageTable 需要 scripted oracle；LIBERO Plus 需要 future-action proximity；CLIPort 需要 terminal outcomes。没有这些信号时，方法不能直接套用。
- **每个 benchmark 独立权重与固定编解码**：论文的“统一”是核心架构统一，不是跨 benchmark 共享 adapter。换机器人或换动作接口需要重新训练与重新设计编码。
- **无显式记忆**：论文没有描述跨回合历史缓存或 recurrent memory。不要把 future states 监督、候选数或 action chunk 当作记忆窗口。
- **无真实机器人、无延迟数据、无显著性检验**：仿真结果不能直接外推；1.28 pp 这类小增益需要更多 seed、置信区间与真实部署验证。

### 6.1 隐含假设

- **候选里存在更好的动作**：离散接口默认 base policy 的候选集包含可提升的替代项；否则排序无意义。
- **局部能量下降会带来任务成功**：连续修正依赖一阶近似与合适的 `η`；局部下降不保证全局任务完成。
- **context 编码足够**：固定 `χ_b(o,ℓ)` 能捕获任务相关上下文；论文没有证明不需要历史记忆。
- **监督信号与动作质量一致**：terminal success、expert correction、demonstration proximity 被视为偏好证据；但不同时间尺度的信用分配可能不一致。
- **“共享架构”可跨接口复用**：共享的是 79,817 参数的核心结构；每个 benchmark 仍有独立权重、固定编码与独立超参。不能把它说成一个 adapter 零样本迁移到五个 benchmark。
- **Jev 灵感停留在接口层**：论文没有复现 Jev 内部模型；公共 Jev 材料也不提供完整训练配方。把 JevVLA 称为“Jev 模型”是不准确的。

## 7. 与相关工作对比

| 方法/基线 | 设置 | 关键结果 | 与 JevVLA 的差异 |
|-----------|------|----------|------------------|
| Direct | 同一 frozen policy，不做适配 | MetaWorld 75.84%；CLIPort 34.22% | JevVLA 的局部对照 |
| Shared linear value probe | MetaWorld，同一 `π0.5` | 1,839/2,500 = 73.56% | 线性 probe 低于 Direct；JevVLA 1,928/2,500 |
| RSS RAS decoding rule | MetaWorld，同一 `π0.5` | 1,168/2,500 = 46.72% | 该规则在该设置下大幅低于 Direct |
| UA-CLIPort | CLIPort，在线 11×11 grasp rule | 607/1,800 = 33.72% | 论文说没有 matched online Direct run，不能把与 Direct 的差当配对效应 |
| Logistic regression | CLIPort 候选分类 | 623/1,800 = 34.61% | 表格模型基线 |
| Shared MLP | CLIPort 候选分类 | 617/1,800 = 34.28% | 表格模型基线 |
| TabM | CLIPort 候选分类 | 627/1,800 = 34.83% | 该组最强替代；JevVLA 639/1,800 |
| FT-Transformer | CLIPort 候选分类 | 624/1,800 = 34.67% | 表格模型基线 |
| TabNet | CLIPort 候选分类 | 614/1,800 = 34.11% | 表格模型基线 |
| JevVLA | frozen policy + energy adapter | MetaWorld 77.12%；CLIPort 35.50% | 值排序 + 梯度修正共用核心 |

**面试 Tip**：被问到 JevVLA 时，一句话答——“它不是训练一个新 VLA，而是在冻结策略外面加一个约 8 万参数的条件能量适配器；能量值做候选 rerank，能量梯度做连续 correction，但每个 benchmark 单独训练权重，而且只在仿真里验证。”

## 8. 精读建议

- **值得精读原文的人**：做 frozen VLA post-hoc adaptation 的研究者；需要 candidate reranking 或 energy-based guidance 的工程师；想评估“轻量 adapter 能否跨动作接口复用”的系统设计者。
- **建议章节路径**：先读 §3.1 至 §3.4 与 Figure 2，抓住“值排序、梯度修正”的一条主线；再看 §4.2 表 1 与 §4.4 CLIPort 消融，理解增益来源与失败案例；最后看 §4.1 与 Appendix A 的超参、数据划分和训练成本。Related Work 可在熟悉 energy-based policy 后跳读。
- **不值得精读的理由**：如果你只关心真实机器人部署、跨回合记忆、大规模预训练配方，或需要显著性检验与延迟数据，这篇目前不够；读摘要与表 1 即可。

---

[← Back to Theory](./README.md)

论文链接：https://arxiv.org/html/2610.35010