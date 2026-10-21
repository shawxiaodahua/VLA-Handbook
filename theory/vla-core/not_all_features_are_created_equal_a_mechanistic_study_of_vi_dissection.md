# 不是所有特征都生而平等：VLA 的机制可解释性研究 (Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-21
>
> **论文**: Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2603.19233
> **核心定位**: 用跨任务激活注入和 SAE 把 VLA 的失败拆成“VLM 选目标、expert 执行坐标绑定运动程序”两层，给出比“换物体就崩”更机制化的解释。

> 版本说明：本文依据 arXiv:2603.19233v4 [cs.RO]（2026-10-20）的 HTML/PDF 正文与表格；来源记录中的候选摘要只到 “execute t”，是截断片段，未用于补写或替代 PDF 摘要中的主张。以下数字均以论文正文/表格为准，不采用候选摘要扩写。

## ⚡ 快速判断（30 秒读完这段就够）

| 维度 | 判断 |
|------|------|
| 核心结论 | 微调后的 VLA 往往执行“视觉选中的、绑定绝对工作空间坐标的运动程序”；语言被编码，但在五个语言能力模型中有四个在场景本身可识别目标时把指令权重压低；SmolVLA 的 LIBERO checkpoint 是反例（libero_object 空 prompt 下 0/500）。 |
| 适合精读 | 做 VLA 可解释性、鲁棒性评测、SAE、部署失败诊断的人，重点看 §3.2、§4.3、§4.5、§4.6、§4.7。 |
| 可以跳过 | 只关心性能 SOTA、或只做移动/人形长程规划的读者，距离中等。 |
| 落地可行性 | 中：注入协议可做诊断工具，但硬件证据仅 π0.5 + UR10e 单任务对；跨架构结论来自仿真。 |
| 主要风险 | 绝对坐标与场景相对坐标尚未用刚性变换区分；跨架构结果以仿真为主；SAE pooling 选择强依赖架构。 |

💡 **X-Ray 开场**  
这篇论文问：VLA 到底是“听懂指令去找物体”，还是“看见场景后回放一条训练时学到的坐标轨迹”？作者把任务 A 的 expert 激活注入到任务 B 的前向过程，发现 π0.5 在 99.6% 的 1,968 个 episode 中执行 A 的轨迹，哪怕眼前是 B 的场景。对 VLA 研究者意味着：语言接口存在，但行为可能被视觉上下文和坐标绑定程序劫持；评测不能只看“指令换词”，还要看物体位置扰动和通路干预。

📍 **研究全景时间线**  
只列论文明确引用且可确认的节点：

```text
2023  RT-2：VLM 生成 tokenized robot actions，确立离散 token 范式
2025  π0.5：flow matching + 专用 action expert；OpenVLA-OFT：连续 L1 回归，LIBERO 97.1%
2026  本文：跨任务激活注入 + SAE，在 6 个架构、4 个仿真基准和 UR10e 上做机制归因
```

本文位置：把“扰动脆弱性”从行为现象推进到通路/层/特征级因果探针；局限是跨架构主要在仿真，硬件只覆盖一个模型和一个任务对。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

论文研究 5 个 VLA + 1 个无语言控制 ACT，参数从 80M 到 7B。下表只归入论文明确写出的属性；“记忆/动作 chunk”列只写论文明确说明的序列或 chunk，不把 50 个 action-token 注入位置写成所有模型的通用记忆窗口。

| 模型 | 参数/层/维度 | 动作生成与通路 | 本文评测域 | 记忆/动作 chunk（论文明确） | 数据开放范围（本文未逐模型声明） |
|---|---|---|---|---|---|
| π0.5 | 3B；18 层；1024 | Flow matching 10 steps；双通路 PaliGemma + Expert | LIBERO；UR10e | 每次前向生成 50 个 action；硬件 15 Hz | 论文未逐模型说明训练数据开放范围；只说明使用预训练 checkpoint |
| OpenVLA-OFT | 7B；32 层；4096 | 连续 L1 回归；单通路 Llama-2 | LIBERO | 8-step action chunk，共 56 个 action token | 未逐模型说明 |
| X-VLA | ~1B；24 层；1024 | Flow matching；单通路 Florence-2 | LIBERO；SimplerEnv | 未明确 chunk 长度 | 未逐模型说明 |
| SmolVLA | 450M；32 层；960/480 | Flow matching；双通路 VLM + Expert，interleaved fusion | LIBERO；MetaWorld | 未明确 chunk 长度；SAE 挂在 MLP 输出 | 未逐模型说明 |
| GR00T N1.5 | 3B；32 层；维度 varies | Diffusion/flow；三通路 DiT + Eagle + VL-SA | LIBERO | DiT 16 层、Eagle 12 层、VL-SA 4 层；未给统一 chunk | 未逐模型说明；使用公开 checkpoint |
| ACT（控制） | 80M；Encoder-Decoder | CVAE；无语言 | ALOHA | 100 timesteps action chunk，14-DOF | 未逐模型说明 |

> 关键提醒：论文的“50 个 action-token 位置”是注入时覆盖的位置，不是所有架构共有的记忆窗口。若要把帧数 K 与帧间隔 Δt 换算成历史跨度，只能对给定设置使用 `(K-1)×Δt`；论文没有把任一 chunk 长度声明为通用记忆窗口。

### 1.2 关键机制 (Key Mechanism)

- **跨任务激活注入**：记录任务 A 成功 rollout 的 residual stream，在任务 B 推理的每个时间步，把 expert 通路 L16/L17 的 50 个 action-token 位置替换为 A 的激活；图像和 prompt token 不变。若目标比源长，注入在 t=T_A 停止，之后目标策略自由运行（<5% 跨任务 episode；限制 T_B≤T_A 时 π0.5 的 override 变化 ≤1pp）。
- **反事实提示**：空 prompt、否定 prompt、动作命令、物体替换、时序切换六类条件，用来测语言敏感度。
- **SAE 读出**：TopK SAE，k=64，扩展因子 4d 或 8d；用 Cohen’s d × 频率选概念特征；特征必须通过同一注入-and-rollout 机制复现才算机制性。
- **线性探针**：ridge regression 逐动作维度探针，并用投影掉探针方向做因果检验。
- **通路分离**：多通路模型分别注入 VLM 通路和 expert 通路；expert 注入导致“主动误导”，VLM 注入导致“被动停滞”。

⚡ **Eureka Moment**：VLA 的“语言泛化”可能不是一条从指令到物体的语义通路，而是“视觉上下文选出一个坐标绑定运动程序，expert 执行它”；语言更像在视觉歧义时解锁程序的条件，而不是每步重新定位目标。

### 1.3 信息流/架构图 (Flow / Diagram)

```text
场景 + 指令
   │
   ├─ VLM 通路：读场景、编码指令、选目标/解锁任务
   │        │
   │        └─ 目标选择信号（在 π0.5/SmolVLA/GR00T 可分离）
   │
   └─ Expert/动作通路：生成动作 chunk，绑定工作空间坐标
            │
            └─ 跨任务注入：把 A 的 expert 激活写入 B 的前向
                     └─ 结果：B 场景下执行 A 的轨迹，而非重定位到 B 的物体

单通路模型（OFT）：VLM 与动作在同一个 Llama 栈中纠缠，通路编辑不响应
SmolVLA：interleaved fusion 衰减跨任务 override（52.1%）
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：  
```text
注入 A 的 expert 激活后，若动作更像 A 而不像 B，则说明被注入的是可移植的坐标程序，不是场景重定位。
```

**目标**：用行为变化判断注入的激活编码“任务概念”还是“坐标绑定运动程序”。

**注入方程**：
```text
H_B,t(ℓ) ← H_A,t(ℓ),  ℓ ∈ L_inject
```

**Override 定义**：
```text
Δ_t = cos(a_t(B|A-inj), a_t(A)) - cos(a_t(B|A-inj), a_t(B))
OR = Pr[Δ_t > τ | |Δ_t| > τ],  τ = 0.05
```

**SAE 读出**：
```text
z = TopK_k(W_e (h - b_d))
h_hat = W_e^T z + b_d
L(h) = ||h - h_hat||_2^2 + α_aux * ||r - W_e^T TopK_kaux(z_dead)||_2^2
```

| 符号 | 含义 |
|---|---|
| H_B,t(ℓ) | 目标 episode B 在时间步 t、层 ℓ 的 residual stream |
| L_inject | 注入层集合；生产条件为 expert L16/L17 |
| a_t(A) | 源任务 A 在 t 的动作向量 |
| Δ_t | 注入后动作相对源/目标的方向优势 |
| τ | 余弦 tie band，论文取 0.05 |
| z | TopK 稀疏 latent，k=64 |
| α_aux | AuxK 死特征回收权重，论文取 1/32 |

**直觉**：如果注入后动作方向仍贴近 A，说明 A 的激活携带了“怎么动”的程序；如果贴近 B，说明它被 B 的场景重新解释。论文还报告 tie-inclusive override（Pr[Δ_t>0]），因为不同架构 tie band 比例差异很大（π0.5 0.3%，OFT 20.9%，X-VLA 60.0% 动作余弦），跨模型比较必须限定 metric。

> 论文实测：π0.5 的 tie-conditional override 为 99.6%，n=1,968（Table 3）；同一模型 tie-inclusive 也是 99.6% [99.2, 99.8]（Table 10）。OFT tie-conditional 96.4%，tie-inclusive 77.9%；X-VLA 在 EEF-velocity 下 tie-inclusive 52.8%；GR00T 在 EEF-velocity 下仅 6.7%。所以“57.0% on GR00T”是动作余弦/tie-conditional 口径，不能与 6.7% 混用。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设例子**，只用于验算公式，不是论文实验设置。

假设 2D 动作：源任务 A 的动作 `a_A=(1,0)`，目标任务 B 的动作 `a_B=(0,1)`；注入后动作 `a=(0.8,0.2)`。算余弦：

```text
||a|| = sqrt(0.8^2 + 0.2^2) = sqrt(0.68) = 0.8246
cos(a, a_A) = 0.8 / 0.8246 = 0.9701
cos(a, a_B) = 0.2 / 0.8246 = 0.2425
Δ = 0.9701 - 0.2425 = 0.7276 > τ=0.05
```

按 Eq. 2，这个 step 会被判为 override；如果整段 episode 多数 step 如此，就得到“执行源轨迹”的行为证据。再算一个临界例子：`a=(0.6,0.6)`，`||a||=0.8485`，两个余弦都是 0.7071，Δ=0，落在 tie band 内，不计入条件 override。

若假设某序列 K=50 帧、帧间隔 Δt=0.1 s，则历史跨度为：

```text
(K-1) × Δt = 49 × 0.1 = 4.9 s
```

这只是一次假设演算。论文没有把 50 个 action token 或某个 chunk 长度宣布为所有模型的记忆窗口；换模型、换控制频率必须重新按 `(K-1)×Δt` 算。

## 4. 工程视角 (Engineering View)

- **注入 hook 位置**：多数模型 hook 整层 residual 输出；SmolVLA 因为 interleaved loop 不跑标准 DecoderLayer.forward，hook 的是 `layer.mlp.forward`，即 MLP 写入 residual 的位置。读结果时要注意：整层替换的 0% 或 14–15% 恢复是“层重要性上界”，SmolVLA 只改 MLP 写入，保留 residual passthrough。
- **Flow matching 的噪声匹配**：真实 UR10e 上，整段替换 expert residual 会固定速度场，去噪器无法抵消自身初始噪声，返回 `x_final = a_source + (ε_dest - ε_source)`。作者在硬件注入时复用源端的 per-call 初始噪声；仿真 Tabs. 3/26 未控制这一项，但用整段动作方向余弦仍得到 54/54 的 directed override。工程上做 activation transfer 到 flow/diffusion head 时，这一步必须显式控制。
- **SAE pooling 是架构级选择**：π0.5 和 OFT 在 replacement write-back 下需要 per-token；mean-pool 把 π0.5 的 per-token 81.3% 拉到 3.3%，OFT 的 per-token 80.1% 拉到 7.3%。X-VLA 和 GR00T-DiT 反而 mean-pool 更好（51%→80%，80%→83%），SmolVLA 基本无差（67.2% vs 66.4%）。residual write-back 可在 OFT/GR00T/X-VLA 的单点恢复 mean-pool，但论文没有 π0.5 的 mean-pool residual write-back 结果。
- **时间与频率**：仿真 episode 最大 300 steps；主结果每任务 20–50 episodes，早期 sweep 5，OFT goal-steering 筛选用 3。真实 UR10e 数据 60,416 帧、15 Hz、20k fine-tune steps。论文未报告注入带来的推理延迟；工程落地需自行测 hook 开销，不能从论文数字外推吞吐。
- **条件性分析**：若部署场景中物体位置变化大，坐标绑定会直接变成脆弱性；若只用 mean-pool SAE 读 π0.5/OFT，行为可能被 SAE 重建破坏。前者是论文机制解释的工程含义，后者是论文实测的 SAE 约束。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练

论文不预训练新 VLA，使用现成预训练 checkpoint；模型覆盖 80M–7B、四种动作生成范式：flow matching、连续回归、diffusion、CVAE。SAE 训练在每个模型的 action-relevant activations 上进行：500,000 个 activation 样本（约 10,000 次前向 × 50 action tokens），batch 4096，100 epochs，学习率 3e-4 余弦衰减，k=64。总共 520+ 个 SAE、79 个唯一概念。

### 5.2 仿真评测

- **LIBERO**：4 个 suite、40 个任务（Goal/Object/Spatial/LIBERO-10）。
- **MetaWorld**：50 个任务、4 个难度（easy/medium/hard/very hard）。
- **SimplerEnv**：10 个任务、2 个 embodiment。
- **ALOHA**：2 个 bimanual 任务。
- 总 rollout 超过 420,000 episodes，12 类实验；每任务主结果 20–50 episodes，最大 300 steps。Table 7 给出逐模型 episode 数：π0.5 71,500+；OFT 70,700+；X-VLA 74,500+；SmolVLA 58,000+；GR00T 164,700+；ACT 1,870。

### 5.3 真实机器人设置

π0.5 在 UR10e 上全量微调 20k steps，7 个桌面任务，335 个 teleoperated episodes，60,416 帧、15 Hz，DROID 观察格式，两个外置相机 + 腕部相机 320×180。基线 76.2%（160/210，每任务 30 trials，三名观察者）。跨任务注入对 plushie-in-bowl 与 gray-on-blue stacking 双向测试：26/26 和 18/26；no-injection 控制 0/24、0/25；norm-matched random 0/25、1/23。位移测试把源物体移开 10–15 cm，50/50 次手臂仍去记录坐标，未注入策略则去移动后的物体。

> 数据开放范围：论文声称发布 520+ SAE、79 个概念、rollout 视频和交互平台；本文不展开其 URL，也未核验仓库内容。论文没有逐模型说明训练数据开放范围，因此上表统一标注“未逐模型说明”，不把某个模型的公开权重状态外推给全部模型。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**论文实测能做什么**：
- 跨任务注入可让 π0.5 在 B 场景执行 A 轨迹：99.6%，n=1,968；OFT 96.4%；X-VLA 在 LIBERO 为 99.8%、WidowX 为 54.2%；SmolVLA 52.1%，n=720；GR00T 57.0%，n=270（动作余弦/tie-conditional 口径）。
- 语言并非完全无用：π0.5 在 libero_object 空 prompt 下保留 31.6%，在 libero_goal 只剩 5.8%（baseline 91.0%/92.6%，n=500/cell）；线性探针仍能以 >99% 读回 prompt 类别。说明“编码了”不等于“行为上咨询了”。
- 通路分离在 SmolVLA 上量化：expert 注入 override 78.6%，VLM 注入 25.6%；expert 的 oracle state ratio 0.58，VLM 0.13。π0.5 的 PaliGemma goal 分类 76.4%，expert 62.6%，是较小的 14pp 差距；GR00T DiT 层对 ablation 更敏感（40–80% 下降），VL-SA 更稳。
- SAE pooling 偏好按架构分裂：π0.5/OFT 要 per-token；X-VLA/GR00T-DiT 可 mean-pool；SmolVLA 不敏感。temporal-contrastive 臂让相邻 latent code 余弦从 0.86 升到 0.997，并在所有跑过的架构上降低 rollout 保真度。
- 硬件上坐标锚定成立：源物体移开 10–15 cm 后，注入程序仍去记录坐标，50/50 次；未注入策略跟随物体。

**不能做什么/失败模式**：
- 跨架构主结果全部来自仿真；硬件只覆盖 π0.5、一个任务对。不能声称对移动、人形、双臂通用。
- 绝对工作空间坐标与“与源任务位置重合的场景相对坐标”尚未用整体刚性变换区分；作者在局限中明确说没跑这个控制。
- 注入激活是 off-manifold 的；论文用 norm-matched random direction 和位移分析反驳“只是扰动导致漂移”，但不能排除所有时序错位混淆。
- 单特征 ablation 对 28–92% 的（概念，任务）对无效，概念在特征间冗余；SAE 特征不保证解耦。
- OFT 没有独立 VLM 通路，VLM 编辑不响应；ACT 无语言，跨任务注入是 no-op（cos=1.0）。
- SmolVLA 的 LIBERO checkpoint 在空 prompt 下 0/500，呈全有全无；MetaWorld easy  checkpoint 空 prompt 仍 82.0%。这是 checkpoint 属性，不能外推为 SmolVLA 全系列属性。
- GR00T 公开 spatial checkpoint 低于 NVIDIA reference（论文 Table 2 标注 †），跨模型比较要保留这一条件。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设 1**：跨场景注入的 residual stream 仍能被目标模型“读成”运动程序。论文用 random-direction 控制和硬件随机臂反驳“任意大扰动都会导致源行为”，但注入仍是分布外。
- **假设 2**：动作余弦/EEF 速度余弦能代表“执行了哪个任务”。论文同时报 DTW 和位移分析，libero_10 上 DTW 与余弦分歧明显（45.8% vs 95.8%，n=24），作者归因于轨迹截断；说明 metric 选择会改变结论强度。
- **假设 3**：SAE 的 per-token/mean-pool 结论在 replacement write-back 下成立。residual write-back 改变 OFT/GR00T/X-VLA 的 mean-pool 结果，π0.5 缺数据；工程部署必须写明 write-back 方式。
- **条件性分析**：如果目标是做“部署前诊断”，跨任务注入可以作为压力测试，但需要为每个架构单独选 hook 和 pooling；如果目标是“提升鲁棒性”，论文只给出机制解释，没有训练新策略来消除坐标绑定。

## 7. 与相关工作对比 (Comparison)

| 工作 | 干预架构数 | 干预类型 | 跨任务注入 | Pooling/Pathway | 硬件 |
|---|---:|---|---|---|---|
| Häon et al. 2025 | 2 | FFN、运动学 | 否 | 否/否 | UR5 |
| Swann et al. 2026 | 2 | SAE feature | 否 | 否/否 | DROID |
| Buurmeijer et al. 2026 | 1 | 线性、运动学+物体 | 否 | 否/否 | DROID |
| Miao et al. 2026 | 3 | 成功子空间 | 否 | 否/否 | DROID |
| Jin et al. 2026 | 2 | SAE 抑制 | 否 | 否/否 | ALOHA |
| Khan et al. 2025 | 1 | SAE 方向 | 否 | 否/否 | 仿真 |
| 本文 | 6 | 注入 + SAE | 是 | 是/是 | 仿真 + UR10e |

**面试 Tip**：被问“这篇和已有 VLA 可解释性有什么区别”时，答：已有工作多在单架构里编辑一个方向、任务固定；本文用跨任务整段激活注入做因果分离，并在五个 VLA 架构上比较 SAE pooling；VLM/expert 通路分工只在多通路模型上可分离（SmolVLA 量化，π0.5/GR00T 方向性复现，OFT/X-VLA 单通路，ACT 无语言）。但硬件只验证 π0.5 单任务对，跨架构结论仍在仿真。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做 VLA 可解释性/机制研究，想复用跨任务注入协议的人；
  2. 做机器人部署诊断，需要判断“换物体就崩”是视觉、语言还是坐标绑定问题的人；
  3. 做 SAE on VLA，关心 per-token vs mean-pool vs temporal-contrastive 的工程选择的人。
- **建议章节路径**：先读 §3.2 注入协议与 §4.3 跨任务结果 → 再看 §4.5 通路分离和 §4.6 SAE pooling → 最后读 §4.7 硬件与 App. O 的噪声匹配细节。§2 相关工作可快速扫，App. A 可跳。
- **不值得精读的理由**：如果你不做机器人学习、只关心 VLA 的 benchmark SOTA，或只关心移动/人形长程任务，读摘要和 §4.3 结论即可；论文没有给出训练新策略来消除坐标绑定的方法。

---

[← Back to Theory](./README.md)

关键引用：论文链接 https://arxiv.org/abs/2603.19233 （本文所有数字与表格均来自该论文正文；未使用其他外部链接。）