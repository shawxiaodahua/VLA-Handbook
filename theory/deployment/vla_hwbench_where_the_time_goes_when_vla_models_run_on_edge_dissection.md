# 边缘 VLA 的时间去哪了：VLA-HWBench 七阶段延迟解剖 (VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-28
>
> **论文**: VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware
> **链接**: https://arxiv.org/abs/2610.34717
> **核心定位**: 把 18 个开放 VLA、4 种 action generation、2 块边缘平台、7 个推理阶段和 backend/低比特优化放在同一套测量协议下，回答“边缘 VLA 的时间到底花在哪、优化把时间挤到哪、剩余瓶颈是什么”。

> **版本说明**：本次可用全文是 arXiv:2610.34717v2 的 HTML 正文（页面标注 `arXiv:2610.34717v2 [cs.RO] 27 Oct 2026`，对应 PDF v2）；候选元数据里的摘要片段是 v1 公告且被截断。v2 正文与 v1 摘要的主干一致，但本文所有数字、模型清单、实验条件均以 v2 正文的 §/Table 为准，不用候选摘要补写正文未给出的主张。
>
> **标注约定**：`【实测】` 指论文表格/正文中的测量结果；`【作者主张】` 指作者 insight 或建议；`【条件性分析】` 指本文基于论文数据做的推演，不是论文结论。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 【作者主张】边缘 VLA 的延迟主要由 action generation 的形式与迭代次数决定，而不是参数量；优化会把时间转移到缩短最少的阶段（如 CPU 预处理）。 |
| 适合精读 | 做边缘部署、推理 runtime、量化、机器人控制频率的人，重点看 §5.1–§5.3、Table 7/10/12/14/15。 |
| 可以跳过 | 只关心 VLA 训练方法、任务成功率或新模型架构的人；读摘要和 §5.1 即可。 |
| 落地可行性 | 中高：七阶段拆解和优化排序可直接复用；但 fused 4-bit 只覆盖两个模型且只在 Jetson，真实机器人闭环未验证。 |
| 主要风险 | 单次观测延迟不等于闭环成功率；论文的 per-action 指标用 chunk 长度 H 除，而实际每次调用执行的动作数 H_e 可能更小；编译/量化会改变动作。 |

💡 **X-Ray 开场**

这篇论文不问“VLA 能不能完成任务”，而问“一次 observation-to-action 调用在边缘板上到底把时间花在哪”。它把调用拆成 7 段，测 18 个开放模型、19 个 reference configuration、两块边缘板，发现 action generation 的形式和迭代次数比参数量更能解释延迟。对 VLA 研究者，这意味着“缩小模型”不是低延迟的默认答案；减少 denoising step、token 迭代，以及优化 host 预处理，可能更直接。

📍 **研究全景时间线**

```
2023  RT-2 / LIBERO
2024  OpenVLA / Octo / LeRobot
2025  π0 / π0.5 / GR00T N1 / SmolVLA / FAST / OpenVLA-OFT
2026  VLA-HWBench（本文）：18 模型 × 2 边缘平台 × 7 阶段
        ↑ 当前位置：从“任务成功率”转向“边缘推理时间解剖”
局限：只测推理延迟，不测任务成功率；只覆盖两块板
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

VLA-HWBench 本身不是新模型，而是一套测量协议。它的“系统”由四层组成：模型层、runtime backend 层、硬件层、测量层。

| 模块 | 输入/输出 | 频率/时序 | 训练/推理差异 |
|------|-----------|-----------|----------------|
| 模型层 | 2 张 256×256 图像 + robot state + language instruction → H 个未来动作 | 每次调用生成一个 chunk；LIBERO controller 20 Hz | 只做 inference，固定已发布 LIBERO checkpoint，不训练、不微调 |
| Runtime backend | LeRobot 0.6.1、authors’ code、FlashRT、TensorRT | 每次调用一次；可选 torch.compile、CUDA graphs、FP8/NVFP4 | 优化逐项叠加，固定权重和输入 |
| 硬件层 | Jetson AGX Thor T5000、Ryzen AI Max+ PRO 395 | 每次调用；host 与 GPU 共享内存 | 只测推理，不测训练 |
| 测量层 | host-side observation → host-side action chunk，拆成 S0–S6 | 每配置独立进程；20 次 warm-up，最多 100 次 timed calls | 中位数延迟；p5–p95；24 帧内容变化；11 帧动作一致性 |

两块平台的差异很关键：Jetson 的 FP16 峰值约是 AMD 的 4.4 倍，但内存带宽只差约 1.1 倍。这让论文能把“host CPU”“GPU 算力”“内存带宽”三件事分开看。

| 平台 | GPU | CPU | 内存/带宽 | FP16/FP32 (TFLOP/s) | FP8/FP4 (TFLOP/s) | 软件 |
|------|-----|-----|-----------|----------------------|-------------------|------|
| Jetson AGX Thor T5000 | Blackwell, 2560 CUDA cores, 1.57 GHz | 14 Neoverse-V3AE, 2.6 GHz | 128 GB LPDDR5X, 273 GB/s | 258 / 8.1 | 517 / 1035 | CUDA 13.2, PyTorch 2.11 |
| Ryzen AI Max+ PRO 395 | RDNA 3.5, 40 CUs, 2.9 GHz | 16 Zen 5, 5.1 GHz | 128 GB LPDDR5X-8000, 256 GB/s | 59.4 / 29.7（由 CU 与 clock 推导） | – | ROCm 7.2, PyTorch 2.14 |

### 1.2 关键机制

**机制一：七阶段拆解。** 论文把一次调用拆成 S0 CPU preprocessing、S1 host-to-device copy、S2 vision encoding、S3 LLM prefill、S4 action generation、S5 postprocessing、S6 residual（Python dispatch 等）。这样做的理由很直接：CPU 预处理、数据传输、vision、prefill、action generation 压的是不同资源；只分三段看不到 CPU 预处理，也看不到 prefill 和 action generation 的相对变化。

**机制二：按 action generation 形式建模。** 论文把模型分成四类：flow matching、parallel decoding、autoregressive tokens、reasoning then actions。核心变量不是参数量，而是 action generation 的串行迭代次数 N：

```
flow matching:       N = denoising steps
parallel decoding:   N = 1
autoregressive:      N = generated tokens
reasoning:           N = reasoning tokens + action iterations
```

**机制三：优化逐项叠加。** 论文在 π0.5 和 GR00T N1.7 上按“compile → specialized backend → lower precision”逐级加优化，每次只改一件事，然后把增益归因到阶段。这样能回答“哪个优化最大、缩短了哪段、剩余时间移到哪”。

**机制四：roofline 判断 hardware-bound 还是 software-bound。** 每个 GPU 阶段用 `max(F/P, W/B)` 作为下界；若 `F/W < P/B` 则是 memory-bound。Jetson 上 FP8 的 ridge point 约 1900，NVFP4 约 3800；action generation 每字节只有 20–150 次运算，远低于 ridge，所以是 memory-bound。

⚡ **Eureka Moment**：边缘 VLA 的延迟主变量不是“模型多大”，而是“action generation 要串行跑多少次”；优化不会均匀缩短所有阶段，而是把时间挤到它最不擅长的阶段。因此必须逐阶段测量，不能只看端到端一个数。

### 1.3 信息流/架构图

```
[2×256×256 camera images + robot state + instruction]
        │
   S0 CPU preprocessing: resize / normalize / tokenize
        │
   S1 host-to-device copy
        │
   S2 vision encoding ──────────► image tokens
        │
   S3 LLM prefill ──────────────► KV cache
        │
   S4 action generation: N iterations
        │   flow: denoising steps
        │   parallel: 1 pass
        │   autoregressive: token by token
        │   reasoning: reasoning tokens + action iterations
        │
   S5 CPU postprocessing: unnormalize / copy back
        │
   S6 residual: Python dispatch between stages
        ▼
   H future actions (action chunk)
        │
   controller executes H_e actions at 20 Hz
```

这里要区分两件事：`H` 是模型一次调用生成的未来动作数，`H_e` 是 release 的 LIBERO 评测每次调用实际执行的动作数。两者不一定相等；本文的参考配置和主实验没有历史帧堆叠或跨帧记忆机制，`H` 不是历史上下文窗口（Appendix F 的 MolmoAct2-Think warm mode 会复用上一帧 depth tokens，但那是该快速模式的跨帧复用，不是 `H` 的历史窗口）。

## 2. 数学核心

📌 **Napkin Formula**：

```
T ≈ T0 + T1 + T2 + T3 + N·t + T5 + T6
keep-up if T/H < 50 ms
```

先给目标：把一次 observation-to-action 调用的总延迟 T 拆成可归因的阶段，并判断它能否跟上 LIBERO 的 20 Hz 控制。再给公式：

```
T = Σ_{k=0}^{6} T_k

T4 ≈ N · t

T/H < 50 ms
```

变量说明：

| 符号 | 含义 | 备注 |
|------|------|------|
| T | 一次调用总延迟 | host-side observation 到 host-side action chunk |
| T_k | 第 k 阶段时间 | S0–S6 |
| N | action generation 的串行迭代次数 | flow step / 1 / token 数 / reasoning+action |
| t | 一次迭代成本 | 由 `T=a+bN` 拟合得到 |
| H | action chunk 长度 | 一次调用生成的动作数 |
| H_e | execution horizon | 每次调用实际执行的动作数，见 Table 4 |
| F | 阶段浮点运算数 | 来自 eager run 的 GEMM 形状 |
| W | 阶段权重字节数 | 来自 eager run 的算子形状与 number format |
| P | 峰值矩阵单元吞吐 | 按 number format |
| B | 峰值内存带宽 | 273 GB/s 或 256 GB/s |

Roofline 部分：

```
T_stage ≥ max(F/P, W/B)

memory-bound if F/W < P/B
```

直觉：flow matching 的 T4 随 denoising step 线性增长；parallel decoding 的 N=1，所以 action generation 很轻；autoregressive 每个 token 都要一次 LM pass，N 可能上百；reasoning 先产生推理 token，再把 action 迭代加上去。roofline 则告诉我们：少 token 的阶段（如 denoising step）容易被权重读取带宽限制；长 prefill 和 vision 更容易被 kernel 效率、number format 支持限制。

**帧与 chunk 的时间边界**（【条件性分析】，用于防止概念混淆）：

```
K 个均匀采样帧，间隔 Δt：首末跨度 = (K-1)·Δt
从当前参考 t 预测 t+1 ... t+H：终点距 t = H·Δt
```

本文的 `H` 是未来动作数，不是历史帧数。不能把 `H=50` 当成 50 帧历史窗口，也不能把连续未来 chunk 的预测区间当成历史上下文或记忆窗口。论文的主实验没有做记忆机制实验；参考配置按单次 LIBERO-10 观测计时。Appendix F 的 MolmoAct2-Think warm mode 会复用上一帧 depth tokens，但这是该快速模式的跨帧复用，不是 `H` 的历史窗口，也不改变主实验的单次观测口径。

## 3. 带数字走一遍：玩具例子

### 3.1 假设例子（仅用于验算公式，不是论文实验）

假设一个 flow policy：`a=200 ms`，`b=20 ms/step`，`H=20`，控制周期 50 ms。

```
N=5:   T = 200 + 20×5  = 300 ms;  T/H = 15 ms < 50 ms → 跟得上
N=20:  T = 200 + 20×20 = 600 ms;  T/H = 30 ms < 50 ms → 仍跟得上
H=10, N=20: T/H = 60 ms > 50 ms → 掉出阈值
```

这个例子说明：长 chunk 能“藏”慢调用；但 chunk 变短后，同样的 N 可能立刻掉出 20 Hz。

### 3.2 用论文拟合做验算（π0.5，Jetson）

论文 Table 10 给出 π0.5 在 Jetson 上的线性拟合：`a=259 ms`，`b=14.7 ms/step`，实际 `H=50`。

```
N=10: T = 259 + 14.7×10 = 406 ms;  T/H = 8.12 ms
N=3:  T = 259 + 44.1     = 303.1 ms; T/H = 6.06 ms
N=20: T = 259 + 294      = 553 ms;  T/H = 11.06 ms
```

如果条件性地把 H 改成 10（不是论文设置）：

```
N=10: T/H = 40.6 ms < 50 ms
N=17: T/H = (259 + 249.9)/10 = 50.89 ms > 50 ms
阈值: N < (50×10 - 259)/14.7 = 16.39
```

所以“H=10 时 N 超过约 20 才掉出”是不对的；按这组拟合，N=17 就已经失败。上述是【条件性分析】，不是论文实验；π0.5 的实际 H 是 50。

### 3.3 H_e 的影响（【条件性分析】）

论文的 per-action 指标用 `T/H`，但 Table 4 的 `H_e` 是每次调用实际执行的动作数。若控制器每次只执行 `H_e` 个动作，有效的每次执行动作延迟应按 `T/H_e` 估算：

```
VLA-0:    T=6293 ms, H=8, H_e=1
          T/H=787 ms, T/H_e=6293 ms（都 >50，仍失败）

GR00T N1.7: T=119 ms, H=16, H_e=8
          T/H=7.4 ms, T/H_e=14.9 ms（都 <50，但余量从 6.8× 降到 3.4×）
```

这不是说论文的 keep-up 结论全错，而是说“用 H 除”会高估 `H_e < H` 时的控制余量。部署时要按实际执行频率算，不能只看 chunk 长度。

## 4. 工程视角

### 4.1 延迟与 action generation 形式

【实测】Jetson 上 action generation 在 19 个配置中的 14 个是最大阶段，占 42–99%。延迟范围从 X-VLA 的 77 ms 到 VLA-0 的 6.3 s（82×）；AMD 上从 114 ms 到 17.9 s（157×）。只有 Jetson 的 10/19、AMD 的 9/19 满足 `T/H < 50 ms`。

【实测】参数量对延迟的解释力很弱：log latency 对 log parameter count 的最小二乘拟合，Jetson 上 R² 约 9%，AMD 上约 15%。GR00T N1.7 3.1B 参数，VLA-0 3.8B 参数，但 VLA-0 在 Jetson 上慢 53×；OpenVLA-OFT 参数量是 VLA-Adapter 的 5.2×，每次调用只慢 2.7×。

【实测】一次 flow step 成本 4–107 ms；parallel decoding 只需 1 次迭代；autoregressive 每个 token 要 20–52 ms 的 LM pass。VLA-0 的 208 个 decimal token 占其调用 99%；DeepThinkVLA 与 MolmoAct2-Think 的 reasoning token 约占 91%。相同权重下，MolmoAct2 的 discrete head 比 flow head 慢 2.3×。

### 4.2 优化收益与“时间转移”

| 优化 | 【实测】Jetson 结果 | 工程含义 | 边界 |
|------|---------------------|----------|------|
| torch.compile | π0.5 2.5×；GR00T N1.7 仅 1.02× | 整图捕获有效；只编译 action head 会被未编译的 vision/prefill 拖累 | π0.5 的 CUDA graph 把 kernel launch 从 13k 降到 15；compile 会改动作 |
| FlashRT backend | π0.5 1.8×、GR00T 1.4×（相对上一级） | fused kernel + CUDA graphs 是主要增益来源 | 只在 Jetson；只覆盖少数模型 |
| FP8 / NVFP4 | 再快 1.3–1.8×，主要在 action generation | 低比特对 memory-bound 阶段有效 | π0.5 FP8 min cosine 0.998，NVFP4 min 0.934；任务成功率未测 |
| 总计 | π0.5 318→44 ms（7.2×）；GR00T 119→46 ms（2.6×） | 端到端收益显著 | 只对这两个模型做三级叠加；Appendix F 还有 SmolVLA、pi0-FAST、X-VLA、MolmoAct2、DeepThinkVLA 的单点优化 |

【实测】“kernel fusion + 4-bit weights”相对 `torch.compile` 的增益是 2.6–2.9×，不是“compile + fusion + 低比特”的合计。π0.5 的分级增益是 compile 2.5×、backend 1.8×、precision 1.6×；GR00T 是 1.02×、1.4×、1.8×。不能把 2.6–2.9× 说成总优化增益。

【实测】优化会改变阶段占比：π0.5 的 vision encoding 从 33% 降到 13%，prefill 从 14% 升到 34%；GR00T N1.7 的 CPU 预处理从 12% 升到 39%。注意 12%→39% 只属于 GR00T N1.7，不是所有模型。

【实测】AMD 上只有 compile 可用，优化后平台差距从 2.0–2.5× 扩大到 5.7–7.8×。AMD 并非所有模型都更慢：Table 7 的 AMD/Jetson 比值从 0.67（InternVLA-A1.5）到 3.46（SimpleVLA-RL），说明平台与模型结构强相关。

### 4.3 剩余瓶颈：memory、kernel、CPU

【实测】优化后没有单一资源限制所有阶段：

- **Action generation（Jetson）**：每个 denoising step 只处理 10–41 个 token，算术强度 20–150 ops/byte，远低于 FP8 的 1900 和 NVFP4 的 3800，属于 memory-bound。FlashRT 的 step 时间是下界的 1.6–1.8×；GEMM 读权重达到峰值带宽的约 3/4，短 normalization/attention/quantization kernel 又给 GEMM 时间加三分之一以上。
- **Prefill**：π0.5 的 574–712 个 prefix token 在 AMD 上是 compute-bound，但只达到 BF16 峰值的 42%，且五分之一 token 是 LeRobot 的 prompt padding；Jetson 上仍在 ridge 左侧，但比 roof 低 3.7×，unfused attention 和 re-quantization 占一半时间。GR00T N1.7 只 prefill 156 个 token，Jetson 上是 memory-bound，约为 FP8 权重读取时间的 1.7×；AMD 上 GEMM 只用到峰值带宽的五分之一。
- **Vision encoding**：AMD 上 vision tower 是 compute-bound，但大量运算没走矩阵单元。LeRobot 用 FP32 跑 π0.5 的 tower，RDNA 3.5 没有对应矩阵指令；ROCm 的 MIOpen 把 GR00T 的 3D patch embedding 执行成 512 个 matrix-vector product，导致两个 tower 分别低于 roof 8× 和 17×。Jetson 上 FlashRT 的 FP8 tower 是 memory-bound，但低于 roof 2.9–3.7×。
- **CPU preprocessing**：GR00T N1.7 最快配置里 CPU 预处理占 39%，是最大阶段。约三分之一时间用于把 PyTorch 的 14 个 worker thread 从最深 idle state 唤醒，每次小并行操作都要唤醒；单线程可避免这部分。其余是在一个 Arm core 上的串行工作，AMD host 快 5×。

【作者主张】因此边缘硬件应优先高内存带宽和快 host CPU，而不是只追峰值 GPU 算力；推理软件应把 kernel fusion 和低比特扩展到更多模型/平台，并优化 host 预处理。

### 4.4 抖动、量化误差与动作一致性

【实测】p5–p95 范围可很宽：VLA-Adapter 在 Jetson 上 72.8–102 ms，OpenVLA-OFT 243–360 ms。24 帧内容变化下，四个 scene-dependent token count 配置的波动为 12–122%；MolmoAct2 discrete 在 Jetson 上帧间中位数波动达 122%。

【实测】Table 14 的动作一致性基线随优化类型不同：低比特行相对 FlashRT FP16；X-VLA 是 FP32→BF16；compile 行相对 eager。不能把整张表概括成“相对 FP16”。

- π0.5 FP16→FP8：min cosine 0.99839，max |Δ| 0.124；NVFP4：min 0.93429，max 0.677。
- GR00T N1.7 FP8：min 0.99986，max 0.033；NVFP4：min 0.99974，max 0.047。
- π0.5 两个 compiled 进程之间 max |Δ| 可达 2.0，论文明确说这可能翻转 gripper command；这不是 NVFP4 的结论。
- SmolVLA eager→compile max |Δ| 1.918，重复调用本身也可差 1.96。

【条件性分析】cosine 相似度高不代表任务成功；论文没有做任务成功率评测。把 0.93 的 cosine 直接关联到 gripper 翻转，是过度概括；论文只把 compile 的最大差异 2.0 与 gripper 翻转关联。

## 5. 数据与评测

### 5.1 预训练：本文不涉及

论文不训练、不微调、不评测预训练数据组成。所有模型加载已发布的 LIBERO checkpoint，固定权重和输入；因此不能从本文推断预训练数据规模、数据配比或预训练对延迟的影响。

### 5.2 仿真评测：LIBERO-10 上的推理延迟

【实测】模型选择标准：2025 或 2026 年发布、LIBERO 成功率强、公开 LIBERO checkpoint 与评测代码、使用广泛，并覆盖四种 action generation 形式。论文口径是 18 个开放模型、19 个 reference configuration；MolmoAct2 的 flow/discrete 两个 head 与 Think 变体使行数多于模型数。

| 模型 | Action generation | 参数 (B) | 权重 | H | H_e | 运行入口 |
|------|-------------------|----------|------|---|-----|----------|
| X-VLA | flow matching | 0.88 | BF16 | 30 | 30 | authors’ code |
| GR00T N1.7 | flow matching | 3.14 | BF16 | 16 | 8 | authors’ code |
| SmolVLA | flow matching | 0.45 | BF16+FP32 | 50 | 50 | LeRobot 0.6.1 |
| InstructVLA | flow matching | 2.60 | BF16+FP32 | 8 | 8 | authors’ code |
| π0.5 | flow matching | 4.14 | BF16+FP32 | 50 | 50 | LeRobot 0.6.1 |
| NORA-1.5 | flow matching | 3.99 | BF16 | 5 | 5 | authors’ code |
| Evo-1 | flow matching | 0.78 | BF16+FP32 | 50 | 50 | authors’ code |
| FastWAM | flow matching | 6.02 | BF16 | 32 | 32 | LeRobot 0.6.1 |
| MolmoAct2 (flow) | flow matching | 5.44 | BF16 | 10 | 10 | LeRobot 0.6.1 |
| InternVLA-A1.5 | flow matching | 2.68 | BF16+FP32 | 8 | 8 | authors’ code |
| VLA-Adapter | parallel decoding | 1.47 | BF16 | 8 | 8 | authors’ code |
| SimpleVLA-RL | parallel decoding | 7.54 | BF16 | 8 | 8 | authors’ code |
| OpenVLA-OFT | parallel decoding | 7.71 | BF16 | 8 | 8 | authors’ code |
| RIPT-VLA | parallel decoding | 7.99 | BF16+FP32 | 8 | 8 | authors’ code |
| pi0-FAST | autoregressive tokens | 3.45 | BF16+FP32 | 10 | 10 | LeRobot 0.6.1 |
| MolmoAct2 (discrete) | autoregressive tokens | 5.44 | BF16 | 10 | 10 | LeRobot 0.6.1 |
| VLA-0 | autoregressive tokens | 3.75 | BF16 | 8 | 1 | authors’ code |
| DeepThinkVLA (CoT) | reasoning then actions | 2.92 | BF16 | 10 | 10 | authors’ code |
| MolmoAct2-Think (cold) | reasoning then actions | 5.45 | BF16 | 10 | 10 | authors’ code |

数据开放范围：所有条目都发布 LIBERO checkpoint 与 LIBERO 评测代码；Table 5 记录 GitHub stars 与 HuggingFace 下载量，GR00T N1.7 和 RIPT-VLA 的下载量以“–”标记是 HuggingFace 不计数，不代表未发布。论文的主实验没有研究记忆机制；参考配置只处理当前观测，输出未来动作 chunk，不能把 H 当历史窗口。Appendix F 的 MolmoAct2-Think warm mode 在 24 个连续帧上逐帧计时并复用上一帧 depth tokens，但这是该快速模式的跨帧复用，不是 H 的历史窗口。

测量协议：

- 同一张真实 LIBERO-10 观测：两张 256×256 相机图、robot state、task instruction；仅在 release 的输入契约要求时适配。
- 每配置独立进程；20 次 warm-up calls；最多 100 次 timed calls；总时长截到 90 s，但不少于 20 次；每次调用前后 device synchronization；报告中位数。
- 另外在同一个 episode 的 24 帧上跑内容变化；动作一致性用 11 帧；重复 sweep 取平均，R² 取最低。
- Reference configuration 关闭可选优化、不用 CUDA graphs；stage timer 放在各 release 自己的阶段边界；CUDA graph 内按 kernel 顺序和重复次数拆阶段。
- Q2 的系统优化只对 π0.5 和 GR00T N1.7 做三级逐项叠加；Appendix F 还报告 SmolVLA、pi0-FAST、X-VLA、MolmoAct2、DeepThinkVLA 的 compile/CUDA graph/FP32→BF16/快速模式等单点结果。准确说法是“fused kernel + 4-bit 的 FlashRT 路径只覆盖这两个模型且只在 Jetson”，不是“所有优化只在这两个模型上可用”。

### 5.3 真实机器人设置：没有

论文的动机是 on-robot deployment，但实验只在两块边缘平台的 host + GPU 上测推理延迟；没有真实机器人闭环、没有控制器执行、没有通信/安全层、没有任务成功率。Table 1 的 “Real input” 指使用真实 LIBERO-10 观测图像，不是真实机器人。不能把本文结果直接外推为真实机器人能跑多快、成功率多高。

## 6. 能力与失败模式

### 6.1 能力

| 能力 | 证据 | 边界 |
|------|------|------|
| 定位时间花在哪 | 7 阶段拆解；14/19 配置 action generation 最大，占 42–99% | 只在两块板、LIBERO-10、固定观测 |
| 比较 action generation 形式 | flow/parallel/AR/reasoning 四类；N 与 T4 线性 | 不是所有模型都覆盖；模型清单有限 |
| 排序优化 | π0.5 7.2×、GR00T 2.6×；fused 4-bit 相对 compile 2.6–2.9× | 只在两个模型、Jetson；AMD 只有 compile |
| 诊断剩余瓶颈 | roofline + kernel 分类 + CPU 子步骤 | F/W 来自 eager run；峰值规格不等于实际热/功耗状态 |
| 暴露“时间转移” | π0.5 vision 33→13%、prefill 14→34%；GR00T CPU 12→39% | 占比是特定配置，不是普遍规律 |

### 6.2 失败模式

- **不能预测任务成功率。** 论文只测延迟和动作相似度，没有 LIBERO 成功率；低比特/compile 的动作差异可能改变任务结果，但未被系统评测。
- **不能替代真实机器人验证。** 没有闭环执行、没有网络抖动、没有安全层；真实部署还要考虑控制器、时钟同步、热稳定性和功耗。
- **Backend 覆盖有限。** FlashRT/TensorRT 只在 Jetson；AMD 没有 FP8/FP4；fused 4-bit 只覆盖两个模型。不能假设其他模型能获得同样增益。
- **单观测/24 帧的局限。** scene-dependent token count 的配置帧间波动可达 12–122%；单帧中位数不能代表所有场景。
- **compile 会改动作。** π0.5 两个 compiled 进程之间 max |Δ| 2.0，可能出现 gripper 翻转；SmolVLA 重复调用本身也可差 1.96。部署前要做动作级回归。
- **H 与 H_e 不等。** 论文的 `T/H` 用 chunk 长度，不是 execution horizon；`H_e < H` 时实际控制余量更小。VLA-0 的 `H=8, H_e=1` 是典型例子。
- **CPU 预处理可能反超。** GPU 优化后，GR00T N1.7 的 CPU 预处理占 39%；线程唤醒和串行 Python 不是 GPU 带宽能解决的。
- **AMD 的软件限制明显。** FP32 vision tower 没有矩阵指令、MIOpen 的 patch embedding 退化成 matrix-vector、GEMM 带宽利用低；这些是软件/后端问题，不是硬件峰值问题。

### 6.3 隐含假设

1. **单次调用延迟可代表闭环控制。** 论文没有建模队列、异步推理、动作执行时间重叠；实际闭环可能隐藏或放大延迟。
2. **中位数与 warm-up 后的稳态可代表部署。** 20 次 warm-up、最多 100 次 timed calls；没有长时间热稳定性/功耗 sweep。
3. **`T/H < 50 ms` 足以跟上 20 Hz。** 这忽略了 `H_e`、jitter、通信、安全余量和控制器开销。
4. **同一观测的内容变化有限。** 24 帧来自一个 episode；不同场景、不同 token 长度可能改变结果。
5. **Roofline 的 F/W 可代表 kernel 实际工作。** F/W 来自 eager run 的 GEMM 形状，实际 fused kernel 的算子 mix 可能不同；峰值吞吐/带宽也未考虑热降频。
6. **优化增益可迁移。** 论文只对两个模型做三级优化；其他模型的 compile/CUDA graph 结果在 Appendix F，不能直接外推到所有模型。
7. **动作相似度可作代理。** cosine 和 max |Δ| 不是任务成功率；compile 的 2.0 差异已经提示这一点。
8. **“Real input”不是真实机器人。** 它指真实 LIBERO-10 观测，不是真实部署。

## 7. 与相关工作对比

论文 Table 1 的对比维度：模型数、action generation 形式数、是否有边缘实测、是否覆盖非 NVIDIA、阶段数、是否计 CPU 预处理、是否有 backend、是否测低比特、是否用真实输入。

| 研究 | 模型数 | 形式数 | 边缘实测 | 非 NVIDIA | 阶段数 | CPU 预处理 | Backend | Low-bit | 真实输入 |
|------|--------|--------|----------|-----------|--------|------------|---------|---------|----------|
| Taherin et al. | 5 | 2 | ✓ | ✗ | – | ✗ | ✗ | ✗ | ✗ |
| Zhou et al. | 5 | 2 | ✓ | ✓ | 3 | ✗ | ✗ | ✓ | ✗ |
| Vishwanathan et al. | 1 | 1 | ✓ | ✗ | 3 | ✗ | ✗ | ✗ | ✗ |
| VLA-Perf | – | 3 | ✗ | ✗ | 3 | ✗ | ✗ | ✗ | ✗ |
| VLA-HWBench（本文） | 18 | 4 | ✓ | ✓ | 7 | ✓ | ✓ | ✓ | ✓ |

读表要注意：本文的 “Real input” 是真实 LIBERO-10 观测输入，不是真实机器人闭环；本文的 “18 模型” 对应 19 个 reference configuration。与 prior work 相比，本文的主要增量不是新模型，而是“七阶段 + 四类 action generation + 两平台 + backend/低比特同模型对比”的组合。

**面试 Tip**：被问“边缘 VLA 为什么慢”，先问 action generation 形式和 N，再问 backend/量化，最后问 CPU 预处理；不要一上来报参数量。

## 8. 精读建议

- **值得精读原文的人**：
  1. 做边缘 VLA 部署的工程师：需要知道延迟花在 vision、prefill、action generation 还是 CPU 预处理。
  2. 做推理 runtime/量化的研究者：需要看 fused kernel、CUDA graph、FP8/NVFP4 的逐阶段增益和动作一致性。
  3. 做 VLA benchmark 的研究者：需要一套可复用的七阶段测量协议和失败模式清单。

- **建议章节路径**：
  先读 §5.1 + Figure 3/4（时间花在哪）→ 再读 §5.2 + Figure 5/Table 12（优化收益与时间转移）→ 再读 §5.3 + Figure 6/Table 15（剩余瓶颈）→ 最后看 Appendix D（step/view sweep）、Appendix F（优化细节与动作一致性）、Appendix G（roofline 数字）。如果只关心部署，可跳过 §2 的 related work 细节和大部分参考文献。

- **不值得精读的理由**：
  如果你不做机器人学习、不关心推理延迟，或者只关心任务成功率/新模型架构，读摘要和 §5.1 即可；本文没有训练方法、没有新策略网络、没有真实机器人实验。

---

[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2610.34717