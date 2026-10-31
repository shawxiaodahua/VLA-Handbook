# DRIFT：用残差流适配器解码 VLM 的连续输出 (DRIFT: A Residual Flow Adapter for Decoding Continuous Outputs in Vision-Language Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-30
>
> **论文**: DRIFT: A Residual Flow Adapter for Decoding Continuous Outputs in Vision-Language Models
> **链接**: https://arxiv.org/abs/2606.05758
> **核心定位**: 给自回归离散 token 的 VLM 加一个“粗预测 + 残差 flow”连续解码 adapter，在不改 backbone 的前提下输出连续动作/时间边界，并用 flow matching 在粗预测附近学习局部残差。

> **来源核验**：本文依据 arXiv:2606.05758v2 [cs.CV] 27 Oct 2026 的 HTML 正文（v2）。来源记录中的 candidate 摘要只到 “general framew”，是截断片段，未用于补写或替代正文主张；正文摘要完整，与截断片段未见实质冲突。candidate 元数据标 cs.AI，正文标 cs.CV，本文以正文版本为准。另：原文存在表号引用不一致，正文 “Inference Latency” 段落写 Table 6，但表题为 Table 5；空间 grounding 正文写 Table 4，但表题为 Table 3。本文按表题数字记录，并保留这一来源注记。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 离散 token 不适合精确连续输出；DRIFT 用 base predictor 给粗预测，再用 flow matching 学残差，作者报告在 TVG、空间 grounding、VLA/WAM 上优于 MLP、tokenizer、diffusion/flow 基线。 |
| 适合精读 | 如果你在做 VLA 连续动作头、离散 token 替换、flow matching adapter，重点看 §3.2、§4.1、§4.4 和 Appendix B.2；做 TVG 看 §4.2 与 B.3。 |
| 可以跳过 | 如果你只关心真实机器人跨本体泛化、长期记忆、移动/双臂/人形部署，这篇距离中等：评测主要是仿真和固定 benchmark。 |
| 落地可行性 | 中高：adapter 可插拔、延迟增加不大；但需要 backbone hidden state、额外 flow refiner 和 K 步 ODE，真实机器人闭环未验证。 |
| 主要风险 | 证据集中在 Libero、Simpler WidowX、Charades-STA、ActivityNet Captions、RefCOCO；没有真实机器人实验，不能把仿真成功率外推到真实部署。 |

💡 **X-Ray 开场**

这篇论文解决什么问题？自回归 VLM 输出离散 token，做文本很自然，但做连续动作和时间边界会引入量化误差、丢掉序结构。它发现了什么？先用一个便宜的 base predictor 给出粗连续预测，再让 flow matching 只学“粗预测到真值”的残差，比从噪声直接生成全目标更好优化。对 VLA 研究者意味着什么？如果你已经有一个 tokenizer 或 MLP 动作头，可以把它保留为 prior，再叠一个 residual flow adapter；但论文的验证仍以仿真 benchmark 为主。

📍 **研究全景时间线**

```
2023  Flow Matching / Rectified Flow 成为连续生成的可训练框架
2024  OpenVLA 用离散 token 解码机器人动作
2025  π0 / π0.5 把 flow-based 动作头推进到 VLA
2026  FastWAM / VLM4VLA / DRIFT ← 本文：在已有连续/离散头之上加残差 flow adapter
```

这里的节点只列论文正文明确引用且年份可确认的工作；不把整张参考文献表搬成时间线。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

DRIFT 不是一个新的 VLM backbone，而是一个“连续解码 adapter”。不同实验的 base predictor、refiner、条件表示和预测长度必须逐模型归属，不能合并成一套统一架构。

| 设置 | Backbone | Base predictor | Flow refiner | 条件 z | 输出/未来长度 | 适配数据 | 评测 |
|------|----------|----------------|--------------|--------|----------------|----------|------|
| VLA-Libero | Qwen3-VL-2B-Instruct | 可学习 MLP，MSE 辅助损失 Lp | 4 层 self-attention，8 heads，hidden 768 | `<think>` 最后一层 hidden state | 未来 8 个连续动作 | LIBERO Spatial/Object/Goal/Long 各自训练 | LIBERO 仿真 |
| VLA-Simpler | Qwen3-VL-2B-Instruct | 可学习 MLP | 同上 | `<think>` hidden state | 未来 4 个连续动作 | OXE Bridge + Fractal | Simpler WidowX 仿真 |
| VLA-OpenVLA | OpenVLA-7B | tokenizer，256 bins，NLL 训练 | VLA flow refiner | 新增 special token 的最后一层 hidden state | 未来 1 个动作 | 对应 LIBERO 训练集；Simpler 为 OXE Bridge+Fractal | LIBERO / Simpler |
| TVG | ET-Chat-3.8B（Phi-3-Mini-4K） | `<VALUE>` hidden 上的 MLP，center-width + 1.2× width 扩展 | 2 层 self-attention，16 heads，hidden = LLM hidden/2 | `<VALUE>` hidden state | 连续 start/end 时间边界 | TimeLens-100K（约 20K 视频、100K 标注） | Charades-STA / ActivityNet Captions zero-shot |
| 空间 grounding | Qwen3-VL-2B-Instruct | 可学习 MLP | 8 层 self-attention，8 heads，hidden 768 | `<think>` hidden state | 4 维 box `[cx,cy,w,h]` | RefCOCO / RefCOCO+ / RefCOCOg 训练集 | RefCOCO 系列 val |
| FastWAM | VideoDiT + ActionDiT | 2 层 QFormer + mean + MLP | FastWAM 的 ActionDiT | text embeddings + VAE vision tokens | 连续动作；chunk 长度论文未明确 | LIBERO 全 splits，272K 样本 | LIBERO 仿真 |

**开放范围要逐模型说**：OpenVLA 是论文所称的 open-source VLA，使用 OXE 预训练和 Libero 微调 checkpoint；Qwen3-VL-2B-Instruct 是 MLLM backbone；ET-Chat 使用 released checkpoint 复现基线；FastWAM 按其原实现初始化。论文没有逐项列出许可证与数据再分发条款，本文不推断“所有模型都开放”或“所有数据都同许可”。

**记忆机制也要逐模型说**：论文没有描述任何显式记忆库/历史缓存模块。DRIFT 本身是解码 adapter，不引入记忆。VLA 设置中的 8/4/1 是未来动作 chunk 长度，不是历史上下文窗口；TVG 只说视频按 1 FPS 采样，未给上下文帧数。不能把 action chunk 或 1 FPS 写成所有实验共有的记忆窗口。

### 1.2 关键机制

- **先粗后细**：base predictor `g(z)` 先给粗连续估计 `ŷ`；flow refiner 不从头生成 `y`，而是学 `r = y - g(z)` 的残差迁移。
- **桥接状态以粗预测为中心**：训练时构造 `y_t = (1-t)(g(z)+σε)+ty`，起点是“粗预测 + 噪声”，终点是真值。这样生成模型只需在 prior 附近做局部修正。
- **x-pred + skip connection**：网络直接预测目标 `y_θ`，再换算成速度 `v_θ=(y_θ-y_t)/(1-t)`；`y_θ = α⊙g(z)+(1-α)⊙h_θ(y_t,t,z)`，`α` 是可学习门控，把粗预测和残差修正组合起来。
- **联合训练与梯度停止**：当 `g` 是可学习 MLP 时，用 `L_FM + L_p` 联合训练；构造桥接状态时对 `g` 做 gradient stopping，避免模型退化为只依赖 flow、忽略粗预测。OpenVLA 的 tokenizer base 用 NLL 训练；ET-Chat 的 MLP base 用 MSE。
- **推理是 ODE 积分**：从 `y(0)=g(z)+σε` 出发，用 `dy/dt=v_θ` 做 forward-Euler；论文主要设置用 `K=10` 步、每个 query 一个噪声样本。

⚡ **Eureka Moment**：DRIFT 的关键不是“用 flow 生成连续值”，而是“让 flow 只学 base predictor 解释不了的残差”；这把全局生成问题变成局部残差迁移，作者的理论分析说 Bayes 最优风险不变，但待学目标和桥接半径更小。

### 1.3 信息流/架构图

```text
图像/视频 X_v + 文本 query X_q
        │
        ▼
VLM backbone f(·)  ──►  special token 的 hidden state z
        │
        ├──► base predictor g(z) ──► 粗预测 ŷ = g(z)
        │                              │
        │                              ├──► skip connection α⊙g(z)
        │                              │
        ▼                              ▼
采样 ε ~ N(0,I), t ~ U(0,1)     桥接状态 y_t = (1-t)(g(z)+σε)+t y
        │                              │
        └──────────────► flow refiner h_θ(y_t,t,z)
                                       │
                                       ▼
                          y_θ = α⊙g(z)+(1-α)⊙h_θ
                                       │
                                       ▼
                          v_θ = (y_θ - y_t)/(1-t)
                                       │
                        训练：MSE(v_θ, v*) + L_p
                        推理：K 步 Euler 积分 → y(1)
```

## 2. 数学核心

📌 **Napkin Formula**：

```text
粗预测 g(z) + 残差流 vθ：从 g(z)+σε 出发，学 y-g(z)-σε 的速度场
```

**目标**：在给定 VLM 条件表示 `z` 时，解码连续输出 `y ∈ R^c`，同时保留预训练 VLM 的 backbone，不把连续值强行离散成 token。

**核心公式**：

```text
桥接状态：
y_t = (1 - t) · (g(z) + σ ⊙ ε) + t · y
t ~ U(0,1),  ε ~ N(0,I)

目标速度：
v* = dy_t/dt = y - g(z) - σ ⊙ ε

Flow matching 损失：
L_FM = E[ || v_θ(y_t, t, z) - v* ||₂² ]

x-pred 参数化：
v_θ = (y_θ(y_t, t, z) - y_t) / (1 - t)

带 skip connection 的预测：
y_θ = α ⊙ g(z) + (1 - α) ⊙ h_θ(y_t, t, z)

MLP base predictor 的辅助损失：
L = L_FM + L_p
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| `z` | VLM 在 special token 位置输出的条件 hidden state |
| `y` | 连续目标，如动作向量或 `(t_start,t_end)` |
| `g(z)` | base predictor 的粗预测 `ŷ` |
| `r = y - g(z)` | 残差目标 |
| `σ` | 可学习的噪声尺度向量，控制初始高斯方差 |
| `ε` | 标准高斯噪声 |
| `t` | flow 时间，训练从 `U(0,1)` 采样 |
| `v_θ` | 学到的速度场 |
| `h_θ` | 残差 refinement 模块 |
| `α` | 可学习门控向量，平衡粗预测与修正 |
| `K` | 推理 Euler 步数；主要实验 `K=10` |

**直觉**：标准 flow matching 从 `N(0,I)` 运到 `p(y|z)`，网络要学全局结构；DRIFT 从 `N(g(z),σI)` 出发，网络只需学 `g(z)` 附近的残差。作者在 Appendix A 给出总体和有限样本分析：固定 `g` 时，只要桥接状态 `y_t` 携带超出 `g(z)` 的残差信息，残差修正就能严格降低风险。作者也明确说明：在无限制函数类下，DRIFT 与 direct flow matching 是 Bayes 等价的；优势来自残差速度场二阶矩更小、桥接更局部，而不是“Bayes 风险更低”。这些是作者主张，不是外部复现结论。

## 3. 带数字走一遍：玩具例子

> **以下为假设的 2D 玩具例子，不是论文实测。** 目的只是验算桥接状态、目标速度和 x-pred 参数化是否闭合。

假设：

```text
y       = [1.0, -0.5]        真值
g(z)    = [0.8, -0.2]        粗预测
r       = y - g(z) = [0.2, -0.3]
σ       = [0.2, 0.2]
ε       = [0.5, -1.0]
σ ⊙ ε   = [0.1, -0.2]
t       = 0.25
```

计算桥接状态：

```text
g(z) + σ⊙ε = [0.9, -0.4]
y_t = 0.75 × [0.9, -0.4] + 0.25 × [1.0, -0.5]
    = [0.675, -0.3] + [0.25, -0.125]
    = [0.925, -0.425]
```

计算目标速度：

```text
v* = y - g(z) - σ⊙ε
   = [1.0, -0.5] - [0.8, -0.2] - [0.1, -0.2]
   = [0.1, -0.1]
```

验算 x-pred 参数化：若网络理想地输出 `y_θ = y`，则

```text
v_θ = (y - y_t) / (1 - t)
    = ([1.0, -0.5] - [0.925, -0.425]) / 0.75
    = [0.075, -0.075] / 0.75
    = [0.1, -0.1]
```

与 `v*` 一致，MSE 为 0。再做一步 Euler 推理：从 `y(0)=g(z)+σ⊙ε=[0.9,-0.4]` 出发，若用目标速度走一步：

```text
y(1) ≈ y(0) + v* = [0.9, -0.4] + [0.1, -0.1] = [1.0, -0.5]
```

恢复真值。真实网络不会完美，MSE 非零；这里只验证“桥接 + 速度参数化 + 一步 Euler”的闭环。论文实际用 `K=10` 步，且每个 query 采一个噪声样本，不是这个玩具的一步设置。

## 4. 工程视角

论文给出的延迟/解码器对比集中在 Table 5（原文段落误写为 Table 6）。VLA 行是 Libero-Long 成功率，TVG 行是 Charades-STA mIoU；延迟在 NVIDIA H200 上测，VLA 平均 1000 次 pass、10 个任务，TVG 平均 200 个视频的 510 个 query，均用 10 个 forward-Euler steps。

| 任务 | Decoder | 性能（%） | 延迟（ms） |
|------|---------|-----------|------------|
| VLA | MLP | 88.20 ± 0.00 | 28.4 |
| VLA | Diffusion | 92.23 ± 0.64 | 33.5 |
| VLA | Flow | 93.67 ± 0.35 | 31.1 |
| VLA | DRIFT | 96.33 ± 0.23 | 31.4 |
| TVG | ET-Chat | 42.3 | 603.3 |
| TVG | DRIFT (ET-Chat) | 43.8 | 605.8 |

**工程含义**：

- 在论文的 VLA 设置中，DRIFT 比 MLP 高 8.1 个百分点，延迟从 28.4 ms 到 31.4 ms，增加约 10.5%；相对 direct flow 高 2.7 点，额外 0.3 ms。TVG 上 mIoU 提高 1.5 点，延迟从 603.3 ms 到 605.8 ms，增加约 0.4%。
- 这些数字是特定 backbone、refiner 大小、`K=10`、单噪声样本、H200 上的均值；论文没有给延迟方差/吞吐，不能外推到其他硬件或实时控制频率。
- 训练侧：VLA 用 8×H100 训练 8–16 小时；TVG 用 8×A100 约 10 小时；空间 grounding 用 8×A100 约 2 小时；FastWAM 用 8×H200 24 小时。训练步数/epoch 要分开看：Libero Spatial/Object/Goal 30K steps，Long 60K steps；Simpler 50K steps；TVG 4 epochs；空间 grounding 1 epoch（约 5K steps）。不能把“30K steps”写成所有实验共有设置。
- **动作 chunk 不是记忆窗口**：Qwen3-VL Libero 预测未来 8 个动作，Simpler 预测 4 个，OpenVLA 预测 1 个；FastWAM 的 chunk 长度论文未明确。若动作步长为 `Δt_a`，预测 `t+1` 到 `t+H` 的终点距当前参考是 `H×Δt_a`；这不是历史上下文长度。论文没有系统报告 VLA 历史帧窗口，不能把 8/4/1 当 memory。
- **视频采样也不是记忆窗口**：TVG 按 1 FPS 采样；若取 `K` 帧，首末跨度是 `(K-1)×1s`；但论文未报告 TVG 的上下文帧数 `K`，所以不能给出具体历史窗口。连续时间边界预测更不是“未来帧 chunk”。

## 5. 数据与评测

**预训练/初始化（分开写）**：

- Qwen3-VL-2B-Instruct：论文称其在大规模多模态语料上预训练；DRIFT 在其上加 LoRA，并联合训练 MLP base predictor 与 flow refiner。
- OpenVLA：论文使用 OXE 预训练的 OpenVLA checkpoint；在 Libero 各 suite 上用对应 Libero 微调 checkpoint，在 Simpler 上用 OXE checkpoint。DRIFT 适配时对 vision encoder 和 LLM 加 LoRA。
- ET-Chat：以 ET-Chat 为 TVG backbone；论文说 ET-Chat 在 E.T. Instruct 164K 上指令微调；DRIFT 的 TVG 变体在 TimeLens-100K 上训练，冻结 visual encoder/frame compressor，对 LLM attention 加 LoRA，并训练 MLP base 与 flow refiner。
- FastWAM：按 FastWAM 原实现初始化；DRIFT 额外加 QFormer+MLP base predictor，并与 FastWAM 原 flow decoder 联合优化。

**适配训练数据（逐设置）**：

| 设置 | 数据 | 规模/采样 |
|------|------|-----------|
| Libero | LIBERO Spatial/Object/Goal/Long | 分别 432/454/428/379 条训练轨迹，53K/67K/52K/101K 样本；动作按 max-min 归一化 |
| Simpler WidowX | OXE Bridge + Fractal | 60K + 87K 轨迹，1.28M + 3.4M 样本；动作按 99th percentile 归一化 |
| TVG | TimeLens-100K | 约 20K 源视频、100K query–segment 标注；MLLM 生成的时间标注 |
| 空间 grounding | RefCOCO / RefCOCO+ / RefCOCOg | 120K / 120K / 80K 训练样本；box 归一化到 `[0,1]`，格式 `[cx,cy,w,h]` |
| FastWAM | LIBERO 全 splits | 272K 训练样本 |

**仿真评测（分开写）**：

- Libero：每个 suite 10 个子任务，每个子任务 50 trials；报告 action success rate。Qwen3-VL-2B + DRIFT 的 Libero AVG 为 97.9%；OpenVLA + DRIFT 为 77.7%。
- Simpler WidowX：4 个子任务，每个 24 trials；Qwen3-VL-2B + DRIFT 为 61.5%；OpenVLA + DRIFT 为 13.5%。论文明确说 Libero 和 Simpler 都是 simulation-based benchmark。
- TVG：在 Charades-STA 和 ActivityNet Captions 的 test split 上做 cross-dataset zero-shot；Charades-STA/ActivityNet 训练 split 未出现在 TimeLens-100K 中。报告 R@1@{0.3,0.5,0.7} 和 mIoU。DRIFT 在 7/8 个 IoU/mIoU 列上优于 ED-VTG；唯一例外是 ActivityNet R@0.7，DRIFT 14.1% 对 ED-VTG 16.0%，低 1.9 点。
- 空间 grounding：RefCOCO 系列 val，Accuracy@1。论文 Table 3 表内标签是 “Qwen-VL”，正文称 Qwen3-VL-2B，存在命名不一致；按表内数字，AVG 从 85.6% 到 88.5%。
- FastWAM：Libero 四个 suite，Table 4 报告 AVG 从 96.9% 到 98.2%。

**真实机器人设置**：论文没有真实机器人实验。Simpler WidowX 也是仿真评测，不是真实世界部署。因此不能声称“在真实机器人上验证了 DRIFT”。

**评测来源要小心**：Table 1 中 OpenVLA 结果是作者用 checkpoint 复现并标 `*`；π0/π0.5 结果来自前作/openpi；VLM4VLA、InternVLA 等来自对应论文。Table 2 的 `# Train` 列对 ET-Chat 是 164K 指令微调语料，对 DRIFT 是 TimeLens-100K 的 100K 适配语料，不是同预算对照。Charades-STA 的 R@0.5：DRIFT 44.8 对 ET-Chat 45.9，论文文字称“在 ±1% 内”，按表内数字是低 1.1 点；本文保留表内数字。

## 6. 能力与失败模式

**能做什么（论文实测范围内）**：

- 把离散 token 动作头升级为连续输出：OpenVLA tokenizer 在 Libero AVG 从 75.5% 到 77.7%，Simpler 从 7.7% 到 13.5%。这说明在 OpenVLA 这个基座上，残差 flow 能修正 tokenizer 的量化/离散化误差。
- 在强 MLLM backbone 上做连续动作：Qwen3-VL-2B + DRIFT 在 Libero AVG 97.9%、Simpler 61.5%，作者称达到当时 SOTA；这是论文表 1 的实测，不是外部复现。
- TVG 连续时间边界：Charades-STA R@0.3 65.7→67.2、mIoU 42.3→43.8；ActivityNet R@0.3 24.1→57.9、mIoU 18.9→37.3。ActivityNet 上 ET-Chat 基线较弱，DRIFT 提升很大；但 R@0.7 仍低于 ED-VTG。
- 空间 grounding：Table 3 报告 AVG 85.6→88.5；但表内标签与正文模型名不一致，需以表内数字为准。
- 与 FastWAM 组合：Libero AVG 96.9→98.2；说明 residual adapter 可以叠在已有 flow decoder 上，而不是只能替代 MLP/tokenizer。

**不能做什么/失败模式**：

- **没有真实机器人证据**：所有 VLA 结果都是 Libero/Simpler 仿真；真实接触、延迟抖动、感知噪声、执行器误差未验证。
- **跨本体泛化未建立**：只在特定桌面操作和 WidowX 仿真设置上测；不能外推到移动、双臂、人形或未见过机器人。
- **base predictor 质量很重要**：Table 8 中随机/未训练 base predictor 的 DRIFT 只有 94.0%，接近 direct flow 的 94.2%，远低于正常 DRIFT 的 96.4%。这说明“残差”不是免费午餐；prior 没用时会退化成 flow。
- **低数据 regime 未系统验证**：作者说 flow/diffusion 在低数据下可能高方差，DRIFT 试图缓解；但论文没有给出低数据 scaling 曲线，不能断言普遍解决。
- **TVG 标注/查询歧义**：Appendix C 的失败样例包括“吃三明治只有约 2 秒，但 GT 标成 5 秒”和“坐在床上”歧义；这类错误所有 decoder 都共享，换 decoder 不一定解决。
- **小动作误差导致任务失败**：Figure 3 展示抓取角度/释放偏移会让任务失败；DRIFT 改善稳定性，但不代表闭环控制容错问题消失。
- **延迟/步数开销**：`K=10` 的 ODE 积分、额外 self-attention refiner 会增加延迟；论文只给均值，未给最坏情况/吞吐。
- **没有记忆机制**：DRIFT 不提供跨时间记忆；论文未报告历史窗口、长期依赖或记忆缓存。不能把 action chunk 8/4/1 当 memory window。

### 6.1 隐含假设

- 假设 `g(z)` 是一个有用的 prior，且 `y_t` 携带超出 `g(z)` 的残差信息；若 `m(X)=E[r|X]` 接近 0，残差修正的理论收益就消失。
- 假设仿真 success rate 与真实机器人能力有足够相关性；论文没有真实机器人验证这一点。
- 假设时间边界标注是明确的；Appendix C 显示短事件和歧义查询会破坏这个假设。
- 假设动作归一化（max-min 或 99th percentile）不会改变残差分布的关键结构；论文未系统比较其他归一化。
- 假设 gradient stopping 足以避免联合训练塌缩；论文报告了塌缩风险，但没有给出更大规模/更多数据下的稳定性边界。
- 假设 `K=10` 个 Euler 步、一个噪声样本足以代表条件分布；论文没有给 `K` 的完整消融曲线或多样本推理收益。

## 7. 与相关工作对比

| 方法/类型 | 连续输出机制 | 训练方式 | 代表结果（论文表内） | 适用场景/限制 |
|-----------|--------------|----------|----------------------|----------------|
| OpenVLA + Tokenizer | 256 bins 离散 token | NLL | Libero AVG 75.5；Simpler 7.7 | 离散量化误差；简单但精度受限 |
| VLM4VLA + MLP | 直接回归连续值 | MSE | Qwen3-VL-2B：Libero-Long 55.8；Simpler 49.0 | 确定性回归，难建模多模态/高方差 |
| InternVLA + Diffusion | diffusion head | diffusion loss | Libero AVG 95.8；Simpler 54.2 | 迭代生成，计算更重；全局生成 |
| π0 / π0.5 + Flow | flow head | flow matching | π0：94.1 / 27.1；π0.5：96.9 / 57.1 | 大规模 VLA 预训练；不是即插即用 adapter |
| DRIFT（OpenVLA） | tokenizer base + residual flow | NLL + flow MSE | Libero AVG 77.7；Simpler 13.5 | 在已有 tokenizer 上修正；提升有限但一致 |
| DRIFT（Qwen3-VL-2B） | MLP base + residual flow | MSE + flow MSE | Libero AVG 97.9；Simpler 61.5 | 强 MLLM backbone；仿真 SOTA 主张 |
| DRIFT（FastWAM） | QFormer+MLP base + ActionDiT refiner | 联合优化 | Libero AVG 98.2 | 可叠在已有 flow decoder 上；chunk 长度未明确 |

**面试 Tip**：被问“DRIFT 和 direct flow 的本质区别”时，不要答“Bayes 风险更低”。正确答法是：两者在无限制函数类下 Bayes 等价；DRIFT 把可预测的均值结构交给 `g(z)`，flow 只学残差，残差速度场二阶矩更小、桥接更局部，所以优化/低数据行为可能更好；证据是 Appendix A 的理论和 Table 6/7/8 的消融，但真实机器人尚未验证。

## 8. 精读建议

- **值得精读原文的人**：做 VLA 连续动作头/离散 token 替换的研究者；想把 flow matching adapter 插到已有 VLM/VLA 上的工程师；做 TVG 连续时间定位、需要 zero-shot 跨数据集评测的人。
- **建议章节路径**：先读 §3.2 看桥接/残差/x-pred/skip 的设计 → 再看 §4.1 和 §4.4 看 VLA 数字与消融 → 做 TVG 的读 §4.2 + Appendix B.3 → 关心理论边界读 Appendix A。§2 Related Work 可以最后扫。
- **不值得精读的理由**：如果你不做连续输出、不关心 flow matching adapter，或者只关心真实机器人跨本体部署，读摘要和 §4.1 的仿真结论即可；论文没有真实机器人实验，也没有记忆机制研究。

---
[← Back to Theory](./README.md)

**关键引用**：本文所有论文主张与数字均来自 [DRIFT 论文](https://arxiv.org/html/2606.05758)（arXiv:2606.05758v2，27 Oct 2026）。