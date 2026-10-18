> 历史补录：补回 2026-10-11 的深度阅读任务；论文原始收录日期 2026-10-10；实际补写日期 2026-10-17。本文依据当前读取的指定论文版本，不冒充当时已完成的报告。

# Evo-World：用光流监督未来运动预测的轻量 VLA (Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**：Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision
> **链接**: https://arxiv.org/abs/2610.12936v1
> **核心定位**：把“未来预测”从未来 RGB 或光流重建，改成“光流渲染图的潜表示回归”，再用梯度阻断的 FiLM 注入轻量 VLA；推理时不需要未来图或光流估计。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 光流潜表示监督 + 当前/预测表示 FiLM 条件 + SIGReg/梯度阻断，0.87B 模型在 Meta-World MT50 达 91.1% tier 平均成功率，真实 AgileX 三任务平均 63.3% |
| 适合精读 | 做轻量 VLA 部署、预测辅助表示学习、光流监督、FiLM 条件注入、单/双臂操作的研究与工程读者 |
| 可以跳过 | 只关心 3D 几何、长时序规划、跨机器人泛化；这篇的监督目标是 2D 图像平面位移 |
| 落地可行性 | 中高：推理 2.54 GB GPU 内存、6.16 Hz、无需光流估计；但真实任务成功率 63.3%，每任务仅 20 trials |
| 主要风险 | 2D 光流无显式 3D 几何；真实评测样本小；训练依赖离线 RAFT 缓存；附录显示 z_pred 的线性可解码性低于 z_t |

💡 **X-Ray 开场**

这篇论文解决什么问题？动作监督只要求“看到当前画面会输出动作”，并不明确要求策略预判场景将怎么变；而预测未来 RGB 又会把任务相关运动和大面积静态外观纠缠在一起。它发现了什么？用光流渲染图的 DINOv2 嵌入作为未来目标，可以让预测分支聚焦“哪里会动”，再通过有界 FiLM 把当前表示和预测表示注入策略；SIGReg 防止表示坍缩，stop-gradient 把动作优化与预测表示学习分开。对 VLA 研究者意味着什么？轻量策略也可以带“运动预判”信号，且推理时不必生成未来图像或光流；但 2D 光流不等于 3D 几何，真实泛化仍待外部复现。

📍 **研究全景时间线**

仅列论文参考文献中可确认年份的节点，不表示完整演进：

```
2020 Meta-World 基准
   → 2023 RT-2 / DINOv2
   → 2024 OpenVLA / GR-1
   → 2025 SIGReg (LeJEPA)
   → 2026 Evo-World（本文）← 当前位置
```

本文的位置：在“未来 RGB 预测”和“光流辅助策略”之间，选择“光流渲染的潜表示”作为预测目标，并把预测信号作为条件而非生成目标。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 训练/推理差异 | 频率/时序 |
|------|------|------|----------------|-----------|
| Motion Representation Module | 指定视图 O_t^w、指令 ℓ | z_t、z_pred | 训练与推理都前向；目标分支仅训练 | 每决策一次 |
| 目标编码分支 | 光流渲染图 Y_t | z_h | 仅训练；与 z_t 共享 E_I、P_I | 每训练样本一次 |
| VLM backbone | 多视角 O_t、指令 ℓ | H_VL | Stage 1 冻结；Stage 2 解冻 | 每决策一次 |
| FiLM 条件 | sg(z_t)、sg(z_pred) | γ_t、β_t | 训练与推理都前向；动作梯度被阻断 | 每决策一次 |
| Action expert | x_τ、τ、s_t、H'_VL | 速度 v̂_θ | 训练预测速度；推理 N 步 Euler | 推理 N 次前向 |
| RAFT + renderer | O_t^w、O_{t+Δ}^w | Y_t | 仅训练数据预处理；离线缓存 | Δ=15 帧 |

论文报告的总参数为 0.87B（含预测分支），backbone 为 InternVL3-1B；具体参数统计口径正文未展开，工程比较时应以同一测量口径复核。

### 1.2 关键机制 (Key Mechanism)

1. **预测目标选光流，而不是未来 RGB。** 未来 RGB 同时包含运动与静态纹理；光流渲染图强调图像平面位移，静态区域接近白色。论文用 RAFT 计算 O_t^w 到 O_{t+Δ}^w 的光流，Δ=15 帧，再渲染成三通道图 Y_t，编码为 z_h。预测分支不重建像素，也不解码光流场。
2. **共享图像编码器，预测“同一空间里的未来”。** 当前图与光流渲染图共享图像编码器 E_I（由 DINOv2 初始化）与投影头 P_I，得到 z_t 与 z_h；预测头 P_M 以 [z_t; e_ℓ] 为输入输出 z_pred。这样预测损失比较的是同一表示空间中的当前与未来运动。
3. **SIGReg 防止表示坍缩。** 对 z_t 和 z_h 施加各向同性高斯正则，避免预测分支把所有输入映射到接近常数。论文附录 C 报告：去掉 SIGReg 后，z_t 方差收缩 2.95×10^5 倍，z_pred 收缩 2.47×10^9 倍（作者图注，PCA 分析）。
4. **梯度分离。** FiLM 输入使用 sg(z_t) 和 sg(z_pred)，动作损失只训练 FiLM 与策略，不反传进 motion representation module；motion branch 继续由 world loss 和 SIGReg 优化。作者用消融支持这一设计：去掉梯度阻断，tier 平均从 91.1% 降到 82.2%，Hard 从 88.3% 降到 68.3%。
5. **有界 FiLM 注入。** γ、β 由 α·tanh 产生，α=0.1，生成器最后一层零初始化，初始为恒等调制。预测信号被限制在小幅调制内，避免压过 VLM 的视觉语言特征。

⚡ **Eureka Moment**：不是让模型“画出未来”，而是让模型“编码未来会往哪里动”，再把这份运动潜表示作为策略的条件信号；动作梯度被阻断，预测分支只对自己的运动目标负责。

### 1.3 信息流/架构图 (Flow / Diagram)

```
O_t, ℓ, s_t
  │
  ├─ 指定视图 O_t^w ── DINOv2 E_I ── P_I ── z_t ─┐
  │        ℓ ── CLIP E_T ── P_T ── e_ℓ ── P_M ── z_pred ─┤
  │                                                      ├─ sg ── G_φ ── (γ_t, β_t)
  └─ InternViT + InternVL3 LLM ── H_VL ──────────────────┘
                                      │
                                      ▼
                     H'_VL = H_VL ⊙ (1 + γ_t) + β_t
                                      │
                                      ▼
                     Flow-matching action expert A_θ
                                      │
                                      ▼
                              action chunk â

训练时额外目标通路：
O_t^w, O_{t+Δ}^w ── RAFT ── 光流 F ── renderer R ── Y_t
Y_t ── DINOv2 E_I ── P_I ── z_h
L_pred = ||z_pred - z_h||²；SIGReg 作用于 z_t、z_h
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
L = L_action + λ_world (L_pred + λ_sig L_SIGReg)
```

目标：动作损失学控制；world loss 让预测分支回归未来运动潜表示；SIGReg 保持表示分布；stop-gradient 分离两条优化路径。

```text
z_t = P_I(E_I(O_t^w))
e_ℓ = P_T(E_T(ℓ))
z_pred = P_M([z_t; e_ℓ])
z_h = P_I(E_I(R(RAFT(O_t^w, O_{t+Δ}^w))))

L_pred = (1/(B D)) Σ_b ||z_pred,b - z_h,b||²
L_SIGReg = 1/2 [S(z_t) + S(z_h)]
L_action = Σ_{b,j} M_{b,j} (v̂_{θ,b,j} - v*_{b,j})² / (Σ_{b,j} M_{b,j} + 1e-8)
L = L_action + λ_world (L_pred + λ_sig L_SIGReg)
```

FiLM 条件与动作专家：

```text
(γ_t, β_t) = α tanh(G_φ([sg(z_t); sg(z_pred)]))
H'_VL = H_VL ⊙ (1 + γ_t) + β_t
α = 0.1

ũ = M ⊙ u,  u ~ U(-1,1)
τ ~ Beta(2,2), clip 到 [0.02, 0.98]
x_τ = (1 - τ) ũ + τ a
v* = a - ũ
v̂_θ = A_θ(x_τ, τ, s_t, H'_VL)

推理：
x^{k+1} = M ⊙ [ x^k + (1/N) A_θ(x^k, k/N, s_t, H'_VL) ]
k = 0,...,N-1;  â = x^N
```

SIGReg 的离散形式（附录 A.2）：

```text
S(Z) = (B/P) Σ_p Σ_k q_k exp(-ξ_k²/2)
       × [ (avg_b cos(ξ_k u_pᵀ z_b) - exp(-ξ_k²/2))²
           + (avg_b sin(ξ_k u_pᵀ z_b))² ]

P = 1024 个单位方向；K = 17 个节点
ξ_k = 3(k-1)/(K-1), δ = 3/(K-1)
q_1 = q_K = δ；内部 q_k = 2δ
```

| 符号 | 含义 | 本文取值/备注 |
|------|------|----------------|
| O_t | 当前多视角观测 | 策略图像 448×448 |
| O_t^w | 指定视图 | 运动分支 224×224 |
| ℓ | 语言指令 | CLIP 编码后与 z_t 拼接 |
| s_t | 机器人状态 | 经 MLP 成 state token |
| a | 归一化动作 chunk | chunk length 50 |
| M | 有效动作坐标 mask | 排除 padding 坐标 |
| z_t | 当前运动表示 | 与 z_h 共享图像编码器 |
| z_pred | 预测运动表示 | P_M 输出 |
| z_h | 光流渲染目标表示 | 仅训练使用 |
| Δ | 预测帧间隔 | 15 帧 |
| B | SIGReg 的局部 batch（每 GPU） | 16（每 GPU；world loss 不跨 worker gather） |
| D | 表示维度 | 论文未给具体值 |
| λ_world | world loss 权重 | 0.1 |
| λ_sig | SIGReg 权重 | 0.09 |
| α | FiLM 调制幅度 | 0.1 |
| N | 推理 Euler 步数 | 论文正文未报告 |

直觉：动作损失只告诉模型“该做什么”，world loss 额外告诉模型“场景会怎么动”。但预测表示若直接接收动作梯度，可能被动作目标塑造成对控制有用、却不再预测运动的特征；因此用 sg 把动作优化挡在 motion branch 外，让预测分支保持自己的运动目标。FiLM 的 tanh 与零初始化则让这份条件信号“先不打扰”backbone，再逐步学会调制。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下数值为**假设**，仅用于验算 flow-matching 的 mask、插值与 Euler 推理闭环，不是论文实验。

设动作 chunk 长度 H=2、每个动作 2 维：

```text
a   = [[1.0, 0.5], [-0.5, 0.0]]
u   = [[0.2, -0.1], [0.4, 0.3]]
M   = [[1, 1], [1, 0]]        # 最后一个坐标是 padding
ũ   = M ⊙ u = [[0.2, -0.1], [0.4, 0]]
τ   = 0.25
```

训练插值：

```text
x_τ = (1-τ) ũ + τ a = 0.75 ũ + 0.25 a
    = [[0.40, 0.05], [0.175, 0]]
v*  = a - ũ = [[0.8, 0.6], [-0.9, 0]]
```

若模型预测：

```text
v̂ = [[0.7, 0.5], [-0.8, 0.2]]
```

masked MSE 只算 3 个有效坐标：

```text
L_action = [(0.7-0.8)² + (0.5-0.6)² + (-0.8+0.9)²] / 3
         = [0.01 + 0.01 + 0.01] / 3
         = 0.01
```

最后一个坐标的 0.2 被 mask 掉，不贡献损失。

推理：设 N=4，x^0=ũ，若速度恒为 v*，则

```text
x^1 = x^0 + (1/4) v*
    = [[0.2+0.2, -0.1+0.15], [0.4-0.225, 0]]
    = [[0.40, 0.05], [0.175, 0]]
```

恰好等于 τ=0.25 时的 x_τ。继续 4 步：

```text
x^4 = ũ + v* = [[1.0, 0.5], [-0.5, 0]]
```

在 mask 有效坐标上到达目标动作 a。真实 A_θ 每步重估速度，不是常数；N 的具体取值论文正文未报告，所以实际延迟需以代码/附录为准。

## 4. 工程视角 (Engineering View)

| 项目 | 论文报告 | 工程含义 |
|------|----------|----------|
| 训练阶段 | Stage 1：10,000 steps，冻结 VLM；Stage 2：100,000 steps，解冻 VLM，新 optimizer/LR schedule | 先对齐轻量头，再联合微调；避免随机预测头早期破坏 VLM |
| 优化配置 | 4×RTX 4090，global batch 64（16/GPU×4），AdamW，peak LR 1e-5，warmup 1,000/stage，cosine，weight decay 1e-3，clip 1.0，BF16，DeepSpeed ZeRO stage 2 | 不是 epoch 口径；Stage 2 的 100,000 steps 是额外阶段，不是总步数 |
| 输入/动作 | 策略图 448×448；运动分支 224×224；chunk length 50 | 动作块一次生成 50 步；执行频率未报告 |
| 正则/条件 | λ_world=0.1，λ_sig=0.09，α=0.1，action expert dropout=0.2；SIGReg P=1024、K=17 | 预测信号被有界注入；超参是 Meta-World 代表配置，其他设置未展开 |
| 光流管线 | RAFT，Δ=15 帧；renderer 固定 magnitude scale；边界目标 clip 到最后一帧；scale 用前 5 个 episode 的 99 分位校准 | 训练数据需离线缓存；换 Δ 需重建 cache；推理不需要 RAFT |
| 真实部署 | 0.87B，2.54 GB GPU 内存，6.16 Hz，H20 BF16 | 由 1/6.16 换算约 162 ms/次决策（未计传感器/执行器）；内存口径未说明是否峰值 |
| 对比 | π0：3.50B、6.69 GB、5.42 Hz；OpenVLA-OFT：7.73B、15.56 GB、2.41 Hz | 相对 π0 内存少 62.0%，频率高 13.7%（论文报告） |

工程 trade-off：

- **训练重、推理轻。** 光流只在训练时算，推理只前向 motion branch 输出 latent；但训练数据准备要跑 RAFT、渲染、缓存，且 Δ 固定会限制预测间隔的灵活性。
- **两阶段训练的关键是“先别弄坏 backbone”。** Stage 1 冻结 VLM，让 motion module、FiLM、action expert 先对齐；Stage 2 才解冻。这个顺序对轻量 VLA 的稳定性有直接工程意义。
- **FiLM 的保守注入。** α=0.1 与零初始化让条件信号初始为恒等；消融中 concat 只有 53.2%，cross-attention 86.4%，FiLM 91.1%。这说明“怎么注入”比“有没有预测”同样重要，但该排序只在本文测试配置下成立。
- **延迟瓶颈需实测。** 论文给的是推理频率，不是端到端控制频率；N 步 Euler 的 N 未报告，实际延迟会随 N、batch、实现变化。部署时应分别测 motion branch、VLM、action expert 的耗时。

## 5. 数据与评测 (Data & Eval)

| 基准/设置 | 数据与评测条件 | 关键结果 | 读法 |
|-----------|----------------|----------|------|
| Meta-World MT50 | 四个难度 tier，等权 tier 平均；Evo-World 无大规模机器人数据预训练；4×4090 | Easy 93.2、Medium 89.1、Hard 88.3、Very Hard 94.0，平均 91.1% | 比 non-predictive baseline 80.6% 高 10.5 个百分点；相对提升 13.0%（10.5/80.6，验算） |
| Meta-World 对比 | 表 1 列出参数量与是否机器人预训练 | 比 Evo-1 80.6% 高 10.5 点；比 FabriVLA 90.0% 高 1.1 点 | 作者称 Hard 与 tier 平均最佳，其余三个 tier 第二 |
| RoboTwin | Clean2Clean 与 Clean2Random；两设置平均 | Clean 67.1%、Random 16.5%、平均 41.8% | 平均最高；Random 低于 Xiaomi Robotics-0 的 18.2%（表 1） |
| 真实 AgileX | 3 任务；每任务 100 条遥操作演示，共 300；每模型每任务 20 trials；H20 BF16 | Floral 55.0、Mug 70.0、Microwave 65.0，平均 63.3% | π0 平均 51.7%，OpenVLA-OFT 23.3%；三任务等权平均验算：(55+70+65)/3=63.33 |
| 真实效率 | 同 H20、BF16 | 0.87B、2.54 GB、6.16 Hz | 对比 π0：3.50B、6.69 GB、5.42 Hz；内存少 62.0%、频率高 13.7%（验算） |
| 消融 | 同一训练预算；checkpoint 取 second-stage 100,000 steps | A：None 80.6、Future RGB 84.3、Optical flow 91.1；B：z_t 83.7、z_pred 85.5、noise 81.0、joint 91.1；C：concat 53.2、cross-attn 86.4、FiLM 91.1；D：w/o SIGReg 85.8、w/o Grad Detach 82.2 | 作者报告；差值分别为 A2 +3.7、A3 +6.8；B joint +7.4/+5.6/+10.1；C +37.9/+4.7；D -5.3/-8.9 |

评测频率与统计口径要分清：

- **训练**：Stage 1 10,000 steps + Stage 2 100,000 steps；global batch 64；不是 epoch。
- **消融**：统一取 second-stage 100,000 steps 的 checkpoint；同一训练预算。
- **真实评测**：每任务 20 trials，共 3 任务；成功率粒度为 5%。
- **未报告**：Meta-World/RoboTwin 的每任务 trials 数、随机种子数、置信区间；真实实验是否多轮重复。
- **条件性分析**：若把单任务 20 trials 粗略视为独立二项样本，p≈0.633 时标准误约 sqrt(0.633×0.367/20)≈10.8 个百分点；论文未给置信区间，这只是读法，不是论文结论。20 trials 下 55% 与 70% 的差距也可能受抽样波动影响，需外部复现。

表示分析（附录 C/D，作者报告）：

- 去掉 SIGReg 后，z_t 方差收缩 2.95×10^5 倍，z_pred 收缩 2.47×10^9 倍；PCA 图显示表示集中到均值附近。
- 线性解码未来末端执行器位移：光流监督下 z_t 的 test R²=0.661、中位角误差 20.9°；z_pred 为 0.508、29.4°。未来帧监督下 z_t 为 0.497、29.6°；z_pred 为 0.421、35.2°。
- 作者明确写：光流监督下 z_t 的解码优于 z_pred；该比较**不能**证明 predictor 增加了超出当前表示的线性可解码运动信息。这是论文自己承认的限制，读的时候不要把它当成“预测分支一定带来额外信息”的证据。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 证据 | 边界 |
|------|------|------|
| 语言条件单臂/双臂操作 | 真实三任务：Floral 单臂；Mug/Microwave 双臂 | 仅 AgileX PiPER 平台；未测其他机器人/相机配置 |
| 轻量部署 | 0.87B、2.54 GB、6.16 Hz | 参数口径与 InternVL3-1B backbone 的关系未展开；内存是否峰值未说明 |
| 推理无需未来图/光流 | 方法设计；推理只前向 latent 分支 | 训练仍需 RAFT 离线光流；换 Δ 要重建 cache |
| 预测信号改善控制 | 消融 A/B/C/D 均支持 | 同一训练预算、同一评测协议；差异不能直接外推到其他数据规模 |
| 运动表示可分析 | 附录 C/D 的 PCA 与线性解码 | z_pred 的线性可解码性低于 z_t；作者承认不能证明额外信息 |

失败模式与风险（含条件性分析）：

- **2D 光流无显式 3D 几何。** 论文结论也写“current motion targets describe 2D image-plane displacement without explicit 3D geometry”。遮挡、深度变化、透明/反光物体上的光流可靠性，论文未测；这是方法外推风险，不是本文实测结论。
- **快速运动与固定 Δ。** Δ=15 帧；快速运动可能造成光流混叠或边界 clip 误差；更长预测间隔需要新 cache，且预测难度上升。
- **长时序任务未验证。** 论文任务是桌面操作与双臂协作，未测长时序规划；固定 Δ 的预测不等于多步未来推演。
- **真实评测样本小。** 每任务 20 trials，成功率 5% 粒度；没有置信区间/种子数。63.3% 是论文实测，但不足以声称高可靠部署。
- **跨平台泛化未证。** 只在 AgileX 双臂 + 三任务上测；不能声称对移动机器人、人形、其他相机布局有效。
- **FiLM 的排序可能是配置相关。** concat 消融只有 53.2%，cross-attn 86.4%，FiLM 91.1%；这个差距很大，说明融合实现/训练敏感，不能直接说“FiLM 普遍最优”。
- **N 未报告。** 推理 Euler 步数影响延迟；6.16 Hz 是特定实现测量，不能直接换算成其他硬件上的控制频率。

### 6.1 隐含假设 (Hidden Assumptions)

1. **光流是任务相关运动的充分代理。** 论文用光流抑制静态外观，但抓取/放置也依赖纹理、几何、物体身份；这些信息是否被光流目标削弱，未做单独验证。
2. **单视图 O_t^w 足以预测未来 Δ 帧。** 方法只选一个 designated view；多视图融合、遮挡下的预测收益未消融。
3. **真实数据也按同样光流管线构造。** 正文与附录 F 说明了真实演示采集与 SFT，但没有展开真实视频的 RAFT 参数、Δ、renderer scale 是否与仿真一致；这是信息缺口，TODO 待补。
4. **动作梯度阻断不会损害策略利用预测特征。** 消融支持，但只在此训练配置下；更长的 Stage 2 或不同 λ 可能改变结论。
5. **有界 FiLM 足以表达运动条件。** α=0.1 且零初始化；如果任务需要更强运动调制，边界可能限制表达。
6. **0.87B 参数口径可直接与基线比较。** 论文表 1/2 用参数量比较，但 backbone 是 InternVL3-1B；具体统计口径未展开，工程选型时应同口径复核。

## 7. 与相关工作对比 (Comparison)

| 方法族 | 代表 | 预测目标 | 是否重建未来 | 注入策略方式 | 推理需预测器？ | 与 Evo-World 的差异 |
|--------|------|----------|----------------|----------------|------------------|----------------------|
| 动作监督 only | π0、OpenVLA、Evo-1 | 无显式未来目标 | 否 | 无 | 否 | 不要求预判场景变化 |
| 未来 RGB | GR-1、GR-2 | 未来 RGB | 是 | 生成/表示 | 可能需要 | 外观与运动纠缠 |
| 视频嵌入 | V-JEPA 2 | 掩码视频嵌入 | 否 | 规划 | 是 | 非直接 VLA 策略 |
| 运动 token | Moto | 离散运动 token | 条件帧重建 | token | 可能 | 两阶段运动先验 |
| 光流辅助 | Fang et al. 2025 | motion-image diffusion | VAE latent | 辅助头 | 推理丢弃 | 共享 VLM，辅助头监督 |
| 联合流/世界模型 | FlowWAM | RGB + flow latent | 生成 flow | 动作预测/世界模型 | 是 | 联合建模 RGB 与 flow |
| 3D 运动先验 | LaMP | 3D motion prior | 冻结专家一步去噪 | 先验 | 否 | 需 3D 运动专家 |
| 执行历史流 | TemporalFlow-VLA | robot-surface flow | 执行历史查询 | 监督 | 否 | 需运动学/相机标定 |
| **Evo-World** | 本文 | 2D 光流渲染的潜嵌入 | 否 | 有界 FiLM | 否（仅 latent 分支） | 0.87B，sg 阻断动作梯度，不重建 flow |

面试 Tip：被问“它和 GR-1/FlowWAM 有什么区别”，先答：预测目标不是未来 RGB，也不是生成光流，而是光流渲染图的 DINOv2 潜嵌入；再用 sg + 有界 FiLM 把当前/预测表示注入策略，推理不需要光流估计。代价是 2D 无几何、训练依赖离线 RAFT 缓存、真实评测样本有限。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做轻量 VLA 部署的工程师：关心 0.87B、2.54 GB、6.16 Hz、两阶段训练和 FiLM 注入。
  2. 研究预测辅助表示学习的读者：关心光流目标、SIGReg、stop-gradient、表示坍缩分析。
  3. 做单/双臂操作与光流监督的研究者：关心 Meta-World/RoboTwin/AgileX 的设置与失败边界。
- **建议章节路径**：
  先读 §3.1–§3.3（方法主链）→ 再看 §5 与表 3（消融）→ 然后 §4.2/§4.3 与表 1/2（评测）→ 附录 A.2/A.3（SIGReg 与光流缓存细节）→ 附录 C/D（表示收缩与线性解码）。§2 相关工作可在需要定位时再读；§6 结论与摘要重复度高，可后置。
- **不值得精读的理由**：
  如果你只关心 3D 几何、长时序规划、跨机器人泛化，这篇距离中等：它的监督是 2D 图像平面位移，实验平台也有限。若你已熟悉 flow matching、FiLM、DINOv2 和光流监督，读摘要 + 表 1/2/3 即可掌握大部分信息。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2610.12936v1（本文数字均来自该论文正文、表 1–3、附录 A–F；未使用其他来源）。