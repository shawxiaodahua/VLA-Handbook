# FluxVLA Engine：面向具身智能的一站式 VLA 工程平台 (FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-17
>
> **论文**: FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence
> **链接**: https://arxiv.org/abs/2609.17210
> **代码**: https://github.com/FluxVLA/FluxVLA （LimX Dynamics 等多家机构）
> **核心定位**: 不提出新策略模型，而是把「数据格式 / 训练栈 / 评测协议 / 推理运行时 / 机器人接口」这五处碎片化的工程债，收敛到一套 config 驱动的数据→训练→仿真→真机的闭环契约里。它要回答的不是「谁的策略更强」，而是「凭什么你的 VLA 能可复现地跑上一台真机」。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 异构 VLA / WAM / 离线 RL 策略可以在统一契约下共存：14 个集成策略在 LIBERO 上 11 个 >95%（DiT4DiT 98.65% 最高），模型侧推理经 Triton/CUDA Graph 加速 2.31×–9.64×，真机 ALOHA 上 π0.5 达 63.64% |
| 適合精讀 | 正在把某个 VLA 从「论文代码」搬上真实机器人的人；想评估「换 backbone / 换 action head / 换机器人」迁移成本的工程师 |
| 可以跳過 | 如果你只关心新算法（新 loss、新架构），这篇明确不提供——它是工程平台，不是方法论突破 |
| 落地可行性 | 中—高：开源、契约清晰、自带 LeRobot 兼容数据层 + RTC + 轨迹后处理；但真机安全边界要自己补 |
| 主要風險 | 报告自认「不构成受控对比」——各行 pretraining / 数据预算 / 评测预算不一致，数字只能当「能跑通」的存在性证据，不是排行榜 |

💡 **X-Ray 开场**
这篇论文解决的是一个很「脏」但很真实的问题：一个 VLA 算法能跑通仿真，不等于它能稳定驱动一台真机——中间隔着数据 schema、相机顺序、归一化统计、checkpoint 格式、推理延迟、chunk 边界抖动、机器人 SDK 一大串「非模型」决策。作者的发现是：与其再造一个策略，不如把「组件选择」和「组件实现」彻底解耦，用 registry + 分层 config 把整条链路写成一份可复现的执行规范。对 VLA 研究者而言，它意味着「你论文里的模型」和「你能演示的机器人系统」之间的那段路，被工程化成了一组明确契约。

📍 **研究全景时间线**

```
2022 RT-1      → 2023 OpenVLA/ACT    → 2024 Octo/π0/GR00T-N1   → 2025 π0.5/SmolVLA
 离散 token        扩散/动作分块          开源通用策略 + 流匹配          高效化 + 流匹配动作专家
                                                                          │
                    2025 Realtime-VLA (30Hz 单卡实时推理)                 │
                    2025 RTC (Real-Time Chunking 异步 chunk 衔接)         │
                                                                          ▼
        → 2026 WAM 三连 (DreamZero/DiT4DiT/FastWAM) + Cosmos3  ← 本文 ← 当前位置
        世界-动作模型成为独立分支                              FluxVLA：不做模型，做把上面全部收拢的「工程底盘」
        局限：WAM 引入视频 VAE / 多流 DiT，与既有 VLA 语义完全不兼容，集成成本爆炸
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

FluxVLA 把一个「实验」定义为一次完整解析（resolve）后的配置：数据管线、模型构图、分布式训练、仿真评测、推理加速、轨迹后处理、远程服务、机器人操作器，全部由同一份分层 Python config 描述。它的单位不是「一个注册了名字的模型」，而是「整条配置解析后的执行路径」。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 (Layer) | 输入 | 输出 | 时序/频率 | 训练 vs 推理差异 |
|------|------|------|------|------|
| Data Layer | 原始 episode (HDF5/LeRobot Parquet) + 在线观测 | canonical sample dict（图像/本体状态/语言/动作/mask） | 采样窗口含未来 action chunk，不跨 episode | 训练走 Parquet；推理走 online adapter，但复用同一组 transform |
| Model Layer | canonical batch | 训练时 → 标量 loss + 诊断量；推理时 → action chunk | 视策略族而定（自回归逐 token / 流匹配迭代去噪） | `forward` 出 loss，`predict_action` 出动作；共用同一 `BaseVLA` |
| Execution Layer | model + optimizer + dataset | checkpoint（自包含）+ 评测结果 | 分布式 rank 打散；eval-after-train | 训练后释放计算态，checkpoint 交给独立评测进程 |
| Serving Layer | 观测（MessagePack/Protobuf，可 JPEG 压缩） | 连续动作 | ZeroMQ request-reply，RTT 敏感 | 本地推理 = 机器人机载；远程推理 = 加速 GPU 服务器 |
| Operator Layer | framework 级 state/action 契约 | 机器人 SDK 命令 | 时间戳同步、有界缓冲、限速发布 | 仿真/真机共用同一 policy-facing 接口 |

### 1.2 关键机制 (Key Mechanism)

- **Registry-based construction**：数据/模型/训练/评测/部署按类型注册；launcher 只问「给我建一个 dataset/runner」，不写「if 模型族 == X」的分支。注册只保证「构造接口」，不保证可互换——张量形状、动作维度、时限、生命周期语义仍需一致。
- **Self-contained inference artifact**：把「完整训练权重 + 解析后 config + 归一化统计」打包成一个 run artifact。这是针对一个经典事故的补丁：checkpoint 只含任务微调，却隐式依赖原始预训练权重 / config / 预处理资产，导致训练与推理语义漂移。
- **Runner / Operator 边界**：runner 管模型侧决策（观测组装、chunk 调度、本地 vs 远程、RTC 状态）；operator 管硬件 I/O（传感器采集、命令传输）。这条边界是整篇设计里最关键的一条——它让机器人通信代码不再变成模型的隐藏依赖。
- **RTC（Real-Time Chunking）**：两条完整路线。Training-time RTC = 训练时模拟推理延迟 + 推理时 prefix conditioning；Test-time RTC = 不改训练，只在推理时用 VJP 引导（另有低成本的直接近似）。单次查询二选一，不叠加。
- **Trajectory post-processing**：在反归一化之后、执行之前插入的可选 stage，两个后端——关节空间 MPC（OSQP 求解）与 Ruckig jerk-limited 滤波，外加 cross-chunk stitching。

⚡ **Eureka Moment**：**「注册一个组件」不等于「它能跑」——真正该被复用的是那条配置解析后的完整执行路径，而不是某个共享基类。** 作者把「集成单位」从 model 提升到 (dataset + ordered transforms + runner + checkpoint + evaluator + operator) 整体，从而让 π0、GR00T、DreamZero、DiT4DiT 这些内部耦合度天差地别的策略，能共存于同一 registry / checkpoint / runner / inference 契约之下。

### 1.3 信息流/架构图 (Flow / Diagram)

```
              ┌──────────────────────── 训练侧 (training) ─────────────────────────┐
 原始数据 ──► ParquetDataset ──► [ordered transforms] ──► collator ──► BaseVLA
 (HDF5/        (LeRobot 兼容)      prompt/image/state/       pad+mask     ├ vision backbone
  LeRobot)                          动作归一化/padding      构建 attn    ├ language backbone
                                                                        ├ projector
                                                                        └ action head
                                        │                                    │
                                        ▼                                    ▼
                                  loss 报告 (标量)                    checkpoint (自包含)
                                        │                          = 权重 + config + 统计
                                        ▼                                    │
  ┌──────────────────────── 执行侧 (execution) ─────────────────────────────┐│
  │  仿真评测 (LIBERO / RoboCasa)  ◄──── 同一 canonical batch 契约 ────────┘│
  │              │  预测 H 步 chunk，执行前 K 步 (1 ≤ K ≤ H)                   │
  │              ▼                                                            │
  │  推理加速 (Triton 融合 / CUDA Graph) → RTC (prefix conditioning / VJP)   │
  │              ▼                                                            │
  │  轨迹后处理 (关节 MPC / Ruckig + 跨 chunk stitching)                      │
  │              ▼                                                            │
  │  Operator (时间戳同步采集 + 限速发布) ──► 真机                              │
  └──────────────────────────────────────────────────────────────────────────┘
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
run_artifact = θ_full ⊕ config_resolved ⊕ normalization_stats      # 自包含 → 训练/评测/部署共用语义
```

**目标**：让三种策略族（离散自回归 VLA、连续流匹配 VLA、世界-动作模型）在同一 loss 报告与 `predict_action` 接口下训练与推理。以下三块公式正是「同一 runner 只消费 total loss 与 action 输出」的底气。

**(a) 离散自回归动作 (OpenVLA 路径)** —— 把连续动作用 action tokenizer 量化、追加到语言目标后做掩码 next-token 交叉熵：

```
L_tok = -(1 / Σ_k m_k) · Σ_k m_k · log₂ p_θ(z_k | o, l, z_<k)

o    : 视觉观测
l    : 语言指令
z_k  : 第 k 个动作分量被量化后的 token
m_k  : 掩码，排除非动作位置与 padding
```

**(b) 连续流匹配策略 (π0 约定)** —— 在 clean 动作与噪声之间学一个速度场，训练用掩码 MSE：

```
x_t = (1 - t)·a + t·ε          # 插值输入
u_t = ε - a                    # 目标速度
L_fm = E_w[ || v_θ(x_t, t | o, l, s) - u_t ||² ]   # w 为有效时刻/维度的掩码

a    : clean action chunk（真实动作块）
ε    : 采样噪声
t    : 采样的时间，t ∈ [0, 1]
s    : 本体状态 (proprioception)
```

**(c) RTC 的时间对齐与 Training-time 条件化** —— 让新 chunk 与「已承诺执行」的旧 chunk 平滑衔接：

```
δ = (t_query - t_ref) / Δt          # 旧 chunk 起点到新查询的相对偏移（可为小数）
```
```
# 训练时模拟延迟 d：前 d 个位置视为「已知前缀」，用 clean-time 覆盖其时间、并掩掉其重建损失
t_j' = t_clean   (j < d)
t_j' = t        (j ≥ d)

m̃_j = m_j · 1{ j ≥ d }              # 只对未知后缀算 loss
```

> 符号与本文一致：`o` 观测、`l` 语言、`s` 状态、`a` 动作块、`ε` 噪声、`t` 流时间、`d` 模拟延迟步数、`Δ` 控制周期。
> 约定差异提醒：clean 端点在 GR00T flow head 是 `1`，在 π0 / π0.5 约定是 `0`——跨模型迁移时这里最容易踩坑。

**(d) 轨迹后处理的关节空间 MPC**（三阶积分器 + box 约束，OSQP 求解）：

```
min_{q,v,a,j}   w_trk · Σ_{t=0}^{H-1} || q_t - q̂_t ||²      # 跟踪策略参考轨迹
              + λ · || z ||²                                 # 高阶正则（抖动惩罚）
              + w_term · || q_{H-1} - q̂_{H-1} ||²             # 终端位置（settle 模式）
              + w_stop · || v_{H-1} ||²                       # 终端速度归零（settle 模式）

s.t.  q_{t+1} = q_t + Δt·v_t
      v_{t+1} = v_t + Δt·a_t
      a_{t+1} = a_t + Δt·j_t          # 三阶积分 = 位置/速度/加速度/加加速度链
      以及 v, a, j 的 box 约束
```

**直觉**：加速侧赌「同一份权重、不同执行图」（checkpoint 不变，只换推理实现）；RTC 赌「把已执行的动作当作 inpainting 目标，固定必须执行的、对后续重叠区施加衰减一致性引导」；后处理赌「预测连贯 ≠ 可执行，反归一化后还要过一道运动学约束」。三者互补而非竞争：RTC 塑造「下一次预测」，后处理塑造「最终发给机器人的命令」。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

**场景**：2 维动作块，控制周期 `Δt = 0.05 s`（20 Hz），策略一次预测 `H = 5` 步、执行前 `K = 2` 步。

**① 流匹配前向（单步）**
假设真实动作块某一分量 `a = 0.8`，采样噪声 `ε = -0.2`，采样时间 `t = 0.5`：
```
x_t = (1 - 0.5)·0.8 + 0.5·(-0.2) = 0.4 - 0.1 = 0.30
u_t = ε - a = -0.2 - 0.8 = -1.0
```
若模型预测 `v_θ = -0.9`，则该分量单点损失 `(v_θ - u_t)² = (-0.9 + 1.0)² = 0.01`。推理时从 `x_1 ~ 噪声`出发，沿学到的场数值积分若干步收敛到一个 clean chunk；`H` 步、`K=2` 前缀立刻下发。

**② RTC 时间对齐（异步衔接）**
旧 chunk 起点 `t_ref = 0.00 s`，模型推理耗时导致新查询在 `t_query = 0.60 s` 发出：
```
δ = (0.60 - 0.00) / 0.05 = 12
```
即旧 chunk 已执行掉 12 个位置，剩余轨迹从索引 12 线性重采样；请求的前缀长度 `L` 被 clamp 到「还剩多少」。若 `L` 超出可用后缀，就被截断——避免把陈旧数组位置当成有效前缀。

**③ Training-time RTC 条件化**
设最大延迟 `D = 8`，本 batch 采样到 `d = 3`：前 3 个位置标记为已知前缀（时间置为 clean 端点），loss 掩码 `m̃_j = 0`（j<3）；只对第 3..H-1 位算损失。这样模型学会「带着干净前缀去补未知后缀」。

**④ 轨迹后处理 MPC（H=3，单关节）**
策略参考 `q̂ = [0.0, 1.0, 2.0]`，取 `w_trk = 1`、只开跟踪项：
```
min  (q0-0)² + (q1-1)² + (q2-2)²
s.t. 三阶积分约束 + 关节 v/a/j 上下限
```
若速度上限把 `q1` 限制在 `0.8`（原本想跳 1.0），求解器会给出「最接近参考且满足运动学边界」的平滑轨迹——这就是「预测连贯 ≠ 可执行」的落点。这一层是 host 侧 CPU 开销来源（见 §4）。

## 4. 工程视角 (Engineering View)

| 关注点 | 论文给出的机制 / 数字 | 工程含义 |
|------|------|------|
| 模型推理频率 | 加速后提升 2.31×–9.64×（8 个 device/model 组合） | 加速是「部署需求」不是「锦上添花」 |
| A100 | GR00T 5.96→32.6 Hz；π0.5 2.2→21.2 Hz | 数据中心卡上把之前「跟不控制环」的模型拉到可用 |
| RTX 5090 | GR00T 42.6 Hz；GR00T+RTC 47.6 Hz；π0.5 31.6 Hz | 桌面卡也能跑到接近 50 Hz 级 |
| AGX Orin 64GB（边缘） | GR00T 3.2→7.4 Hz；π0.5 1.4→4.4 Hz | 机载算力下仍是瓶颈，远程 GPU 服务因此必要 |
| 数值一致性 | cosine similarity > 0.99999，最大绝对差 ~0.02（bf16 融合 + CUDA Graph，随机权重下） | bf16 kernel/融合/重排会引入小误差；须比较动作空间误差而非 bit 对齐 |
| 轨迹后处理延迟 | 12-DoF × 50 步合成轨迹，Ruckig 比关节 MPC 快 ~4.19×（Intel Xeon Platinum 8336C） | host 侧 ms 级开销；高控制频率下不可忽略，后端选择影响实时性 |
| CUDA Graph 约束 | 固定相机数 / 最大 prompt 长度 / chunk 长度 / 精度 / 去噪步数 | 改这些要 rematerialize + recapture；超界算子落在静态图之外而非被静默近似 |
| 远程服务 | ZeroMQ（明文 TCP，无内置鉴权/加密） | **必须**限制在可信网络或走 SSH 隧道 |
| 端到端 vs 模型频率 | 明确分离：模型 40 Hz ≠ 机器人 40 Hz 控制环 | 图像采集/预处理/传输/反归一化都在捕获区之外 |

一句话：这篇的工程价值在于把「延迟 / 抖动 / chunk 边界 / 量化误差 / 内存」当成一等公民。尤其 RTC + 后处理的顺序——RTC 在「策略查询内部」保证新 chunk 与旧原始预测的归一化前缀一致；后处理在「反归一化之后」施加关节空间边界与跨 chunk stitching；operator 只执行最终 robot-space 轨迹。

## 5. 數據與評測 (Data & Eval)

**仿真（Table 7，LIBERO）**：14 个集成（13 个 checkpoint 策略 + 1 个 inference-only GPT-6 API）。平均成功率 37.50%（GPT-6）→ 98.65%（DiT4DiT），其中 **11 个集成 >95%**；Long（libero_10）套件最高 98.0%（Cosmos3-Nano）。GPT-6 因资源受限每任务仅评 5 个 episode（每套件 50、共 200）：Spatial 16/50、Object 33/50、Goal 19/50、Long 7/50。
⚠️ 作者明确声明：各行 pretraining / 优化 / checkpoint 选择 / 控制接口 / 评测预算**均不匹配**，不是受控排行榜。

**仿真（Table 8，RoboCasa GR-1）**：8 个集成，平均成功率 8.75%（SmolVLA）→ 57.25%（DiT4DiT）。GR00T N1.5（每任务 30 条演示）44.30%、GR00T N1.7（全量数据）46.42%、FastWAM 49.92%（泛化组 50.00%）、π0 51.00%、π0.5 51.42%、DiT4DiT 57.25%。每任务 50 trials，但数据/配置不一致，非受控对比。

**真机（Table 12，ALOHA，5 任务）**：GR00T N1.5 成功 36/110（32.73%）；π0.5 成功 70/110（**63.64%**），且每个任务上都更高。

**真机（Table 13，Oli）**：
- 静止糖果抓取：GR00T N1.5 40.5% / π0.5 62.5%
- 箱体搬运（移动操作 1）：阶段 1–3 累计完成 60.0% / 60.7%；**全 4 阶段完成仅 6.7% / 32.8%**（作者特意区分「中间里程碑」与「完整成功」）
- 篮筐玩具抓取（移动操作 2，≥2 个玩具）：36.3% / 29.7%

> 评测严谨性亮点：作者按「来源等级」分层——源码树（可审计的 config/测试/评测器）> 项目自报的仿真与推理测量（标注 project-reported）> 伴随研究给出的真机结果。「存在一个 robot operator」只算集成覆盖，不算任务成功证据。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 让内部结构迥异的策略（自回归 / 流匹配 / 扩散-transformer / 世界-动作）共用同一数据、训练、评测、推理契约。
- 把「从 checkpoint 到真机」的路径标准化：自包含 artifact + 时间戳同步采集 + 限速发布 + 远程 GPU 服务。
- 在同一套 runners 里比较不同 closed-loop 调度（K 步前缀）与不同后处理后端。

**不能做 / 有明显边界**：
- **不是算法贡献**：作者反复强调「不主张上游架构/权重/数据集/benchmark 的算法贡献」。想找新 loss / 新架构的人应直接跳过。
- **不是受控对比**：所有成功率来自不一致的数据预算 / 初始化 / 优化过程，**不能**据此说「A 比 B 强」。
- **不做安全**：FluxVLA 本身**不**实现碰撞避免、工作空间约束、力矩限制或认证急停；这些必须由机器人 SDK / 底层控制器承担。`Tron2` operator 会转发急停请求、`Franka` 可在夹爪 action server 不可用时回退到 publisher——但这是集成层的点缀而非保证。
- **轨迹后处理只做关节可分（joint-separable）的关节空间优化**：不建模耦合动力学、力矩、碰撞、工作空间约束。Ruckig 在 settle 模式下延伸出策略时限后重采样会改变导数，必须重新验证限制。
- **远程服务无内置鉴权/加密**（明文 TCP），网络暴露即风险。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设「共享契约」足以承载语义**：作者自己承认「同一 registry 能构造」不等于「语义兼容」——张量形状、时限、动作表示仍需人工对齐。配置作者要负责挑选兼容组件，框架只把不兼容「显式化 + 局部化」。
- **假设 registry 条目 ≠ 端到端支持**：训练需可复现 checkpoint、评测需固定协议 + 原始 rollout、部署需可运行 operator + 实测。空配置 stub 不算支撑。
- **假设嵌入式部署算力可被远程服务绕过**：AGX Orin 上 GR00T 仅 3.2→7.4 Hz，论文把「轻量远程 GPU 服务」当作解法——但这引入网络/超时/恢复的新失败面。
- **假设 bf16 融合的小误差可接受**：~0.02 的最大绝对差只在随机权重、head 级验证，**未**证明 checkpoint 级等价或任务级一致。
- **评测预算假设**：GPT-6 每任务仅 5 episode，作者诚实标注为 resource-limited，但读者若忽略脚注会误读其 37.50%。

## 7. 與相關工作對比 (Comparison)

| 系统 | 关注点 | 架构/机制 | 训练方式 | 适用场景 | 与 FluxVLA 的关系 |
|------|------|------|------|------|------|
| MMDetection / MMDet3D | 检测器工具箱 | registry + 声明式 config | 通用训练/评测工具 | 视觉检测 | 方法论先例：把「软件架构」当系统贡献 |
| robomimic | 离线示教模仿 | 标准化实现 + 经验对比 | 离线 | 操作策略复现 | 降低复现成本，但无跨预处理一致性保证 |
| LeRobot | 硬件接口→数据→策略→异步推理 | 端到端学习栈 | 离线 + 异步推理 | 真机学习全栈 | FluxVLA **直接复用**其 Parquet 数据布局，并补上契约与部署层 |
| StarVLA | Lego 式 VLA 开发 | 模块化模型抽象 + 统一 benchmark 接口 | 通用 | VLA 快速拼装 | 相近野心，但 FluxVLA 把「集成单位」升级为完整执行路径 |
| OpenVLA / FAST | 离散自回归动作 | 动作 tokenizer + next-token CE | 微调预训练 VLM | 离散动作 VLA | FluxVLA 提供其 contract 与 autoreg 训练/推理路径 |
| π0 / π0.5 / GR00T N1 / SmolVLA | 连续生成动作 | 流匹配 / 扩散-transformer 动作专家 | 预训练 + 微调 | 通用操作 | FluxVLA 保留其原生融合与参数化，只统一 loss / `predict_action` 接口 |
| DreamZero / DiT4DiT / FastWAM / Cosmos3 | 世界-动作模型 | 视频 VAE + 多流 DiT / MoT 模态打包 | 联合流匹配 / 扩散 | 预测视觉动力学 | FluxVLA 分离 backbone / 世界模块 / 动作头 / loss，允许视频+动作 loss 分开报告 |
| Realtime-VLA | 单卡 30Hz 流式推理 | 流式推理框架 | — | 高频控制 | FluxVLA 的 Triton/CUDA 算子部分**构建于其开源实现之上** |

🎤 **面试 Tip**：如果被问「FluxVLA 和 LeRobot / StarVLA 有什么本质区别」——一句话答：**「LeRobot 给你一条端到端栈，StarVLA 给你可拼的模型模块，FluxVLA 把『一个受支持的策略』重新定义为配置解析后的完整执行路径（数据+变换+runner+自包含 checkpoint+评测器+operator），并保证这条路从离线数据一直走到真机命令。」**

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 正在把某个 VLA 从论文代码搬到真实机器人上的工程师——§3.2 数据管线、§3.5 真机推理、§3.7 RTC、§3.8 轨迹后处理是你的施工图。
  2. 要评估「换 backbone / 换 action head / 换机器人平台」迁移成本的研究者——§4 的 contract 与三种 extension recipe（加数据 / 加模型 / 加机器人）直接给出清单。
  3. 做多策略横向评测的人——§5 的「来源分级」与「不受控对比」声明是很好的评测自律范本。
- **建議章節路徑**：先讀 §1.1 + §1.2（动机与设计原则）→ 再看 §3.3 模型契约 + §3.7 RTC（最核心的机制）→ 最后挑 §3.8 后处理里你机器人相关的后端。§2.2 相关工作可跳读（除非你要写 related work）。
- **不值得精讀的理由**：如果你不做机器人学习、或已熟悉 LeRobot/MMDetection 式工具箱、或你只想要新算法——读 §1.1 + ⚡ 快速判斷 即可，正文的工程细节对你没有增益。

---

[← Back to Theory](./README.md)

**关键引用**
- 论文: [arXiv:2609.17210](https://arxiv.org/abs/2609.17210) | [HTML](https://arxiv.org/html/2609.17210v1)
- 代码: [github.com/FluxVLA/FluxVLA](https://github.com/FluxVLA/FluxVLA)
- 数字来源: 论文 Table 7（LIBERO）、Table 8（RoboCasa GR-1）、Table 9（推理加速）、Table 10（数值一致性）、Table 11（后处理延迟）、Table 12–13（ALOHA / Oli 真机）
