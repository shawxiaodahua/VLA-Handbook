# 紧凑 VLA 的持久循环记忆：SmoLSTM (SmoLSTM: A Compact Vision-Language-Action Model with Recurrent Memory that Persists)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-23
>
> **论文**: SmoLSTM: A Compact Vision-Language-Action Model with Recurrent Memory that Persists (Habekost et al., University of Hamburg, 提交 ICRA 2027)
> **链接**: https://arxiv.org/abs/2609.22854
> **核心定位**: 用「贯穿整个 episode 从不重置」的 mLSTM 循环状态替代「越开越大的观测窗口」，让记忆成本与 episode 长度解耦，同时把可训练参数压到 0.04B。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 循环状态本身就是任务记忆：把观测 token 与动作 query 放进同一条从不重置的因果流，LIBERO-Mem 全任务成功率 77.5%，每步成本 O(1) |
| 適合精讀 | 如果你在做**长时序/遮挡/物体身份追踪**的操作策略，重点看 §III-B（统一循环流）与 §III-D（逼模型用记忆的两项损失） |
| 可以跳過 | 如果你只关心**大规模机器人预训练 + SOTA 绝对分数**，这篇没做 robot pretraining，绝对分不如 OpenVLA-OFT（97.1%） |
| 落地可行性 | 中高：0.04B 可训参数 + 单卡可训；但依赖腕部相机（去掉后成功率 12%），且只在 LIBERO 仿真验证 |
| 主要風險 | 全部结论都在 LIBERO/LIBERO-Mem 仿真内，跨真机、跨基准（MIKASA-Robo）尚未验证 |

💡 **X-Ray 开场**
这篇论文解决一个问题：只看当前帧的策略，在「三个一模一样的碗要按左右顺序摆」或「先装满篮子再挪篮子」这类任务上会失忆。
它的发现是：不该把观测窗口拉长（窗口一长，单步成本跟着涨、窗口长度还变成超参），而应该把整段历史压进一个**固定大小**的循环状态里。
对 VLA 研究者意味着：记忆不需要额外模块——让感知 token 和动作 query 共享同一条循环流、且永不重置，记忆就自然产生了。

📍 **研究全景时间线**

```
2023  RoboFlamingo：冻结 VLM + LSTM policy head（分离式循环头）
  |
2024  OpenVLA：VLM 直接 token 化输出动作
  |
2024  LRAM / X-IL：xLSTM/Mamba 状态-动作序列模型（但未测「记忆依赖」任务）
  |
2025  π0 / SmolVLA：flow-matching 动作专家；MemoryVLA / μVLA：memory token / 记忆库
  |
2026  【本文 SmoLSTM】← 当前位置
      观测 token 与动作 query 共用一条 mLSTM 流，全 episode 不重置，
      记忆 O(1)、动作头与记忆解耦，在 LIBERO-Mem 上明确测「记忆依赖」
  |
局限  仅仿真（LIBERO/LIBERO-Mem）、无 robot pretraining、强依赖腕部相机、
       OR 类任务即便不重置也只有 8% 成功率
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 类型 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|------|-----------|----------------|
| 视觉-语言骨干 | SmolVLM-256M-Instruct，**冻结** | 指令 + 第三人称帧 | 64 个 image placeholder（8×8 网格）+ 指令 token | 每控制步一次 | 全程不更新 |
| 指令读出 | 冻结 LM 的两层读out（layer 1 & 30），K=8 分段池化 + 去均值 | 指令文本 | 16 个有序分段 token | 每 episode 常量→缓存 | 推理复用缓存 |
| 附加模态编码器 | ResNet-18（腕部相机 / 单目深度，各 36 token）；DINOv2-S 冻结（64 token）；MLP（本体 1 token） | 腕部图 / 深度 / 本体 | 137 token | 每控制步 | 训练更新 ResNet/MLP |
| 循环控制层 | 6 层 mLSTM，宽 512，4 头，矩阵记忆 | 217 观测 token + T=10 动作 query | 记忆状态 C_t, n_t；query 位置隐状态 | 每控制步推进一次，**永不重置** | 单次前向；推理每步推进一次 |
| 动作头 | rectified flow-matching，约 4.87M 参数 | query 隐状态 z_t + 噪声块 | 10 步 EE 位姿增量 + 夹爪 | 每控制步生成一个 chunk | 训练 1 次前向；推理 8 步 Euler 积分 |

可训练参数 40.4M / 总 0.32B（仅 12.7% 可训，VLM 与 DINOv2-S 冻结）。

### 1.2 关键机制 (Key Mechanism)

- **统一因果流**：把 `[o_1..217][q_1..10][o_1..217][q_1..10]...` 排成一条序列，观测和动作共享同一个 recurrence。
- **永不重置**：常见的「慢记忆模块 + 每 chunk 重置的快解码器」在 chunk 边界丢失循环状态，记忆负担全压在一个 bridge 上；本文把两者合成一条流，消除接缝。
- **query 是学出来的位置嵌入**，不是回灌的上一动作 → chunk 内 T 个预测可并行。
- **两项「逼记忆」损失**：观测 dropout（p=0.15）+ 模态 dropout（p=0.10）；辅助 DINOv2 未来特征预测（5 与 10 步后），权重 0.5，推理时丢弃。

⚡ **Eureka Moment**：**别再扩窗口——让循环状态本身就是记忆。** 观测 token 与动作 query 共享同一条从不重置的 mLSTM 流后，"已摆好几个物体"这类计数信息天然存在于状态中，因为那些动作**曾经流过**它，而不需要一个专门的追踪模块。

### 1.3 信息流/架构图 (Flow / Diagram)

```
每个控制步 t（= 10 个环境帧）:
 指令(缓存) ┐
 第三人称帧 ┼→ [冻结 SmolVLM-256M] → 64 grid token
 腕部图 ────→ ResNet-18 ────────────→ 36
 深度图 ────→ ResNet-18 ────────────→ 36     ⇒ 拼接 217 观测 token
 DINOv2-S ──→ 冻结 ─────────────────→ 64 (亦为辅助损失目标)
 本体 ──────→ MLP ──────────────────→ 1
        │
        ▼
 [ o(217) , q(10) ] → 6× mLSTM(512,4head, 状态不重置) → z_t (10 个 query 隐状态)
        │                                   │
        │                                   └→ 辅助头 → 预测 5/10 步后 DINOv2 特征 (仅训练)
        ▼
 flow-matching 头 (4.87M) → 8 步 Euler → 10× (Δpose + gripper) 动作 chunk
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：
```
新记忆 = 衰减旧记忆 + 写入当前键值外积   (mLSTM 矩阵记忆)
```

**目标**：构造一个固定大小的记忆，使每个控制步的读写成本与 episode 已流逝长度无关。

**mLSTM 记忆更新与读取**（论文 Eq. 2–3）：

```
C_t = f_t · C_{t-1} + i_t · (v_t · k_t^T)          # 矩阵记忆，C_t ∈ R^{d×d}
n_t = f_t · n_{t-1} + i_t · k_t                     # 归一化向量
h_t = C_t · q_t / max( | n_t^T · q_t | , 1 )        # 读出（带下界防爆）
```

**流匹配动作头**（论文 Eq. 4–6）：

```
a_τ = τ·a + (1-τ)·ε ,  u = a - ε                    # 直线路径上的点与其常速度
v_θ(z_t, a_τ, τ) = MLP( SA( z_t + W_a·a_τ + φ(τ) ) ) # 预测速度场
L_flow = E_{τ,ε}[ || v_θ(z_t, a_τ, τ) - u ||_2^2 ]
```

**总损失**：

```
L = L_flow + 0.5 · L_aux        # L_aux = 对 5 步、10 步后 DINOv2 特征的预测误差
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| i_t, f_t | 输入门、遗忘门（由 token 计算） |
| v_t, k_t, q_t | 值、键、查询向量 |
| C_t, n_t | 矩阵记忆与归一化器，尺寸固定 → 记忆 O(1) |
| a | 真实动作 chunk；ε 噪声；τ~U(0,1) 流时间 |
| z_t | 第 t 个控制步 10 个 query 位置的 mLSTM 隐状态 |
| SA | chunk 局部自注意力（2 层）；W_a 噪声块线性映射；φ 正弦时间嵌入 |

**直觉**：`C_t` 是一个 d×d 的"联想记忆矩阵"。遗忘门 `f_t` 给旧记忆乘衰减，外积 `v_t k_t^T` 把当前信息写进去。因为状态尺寸固定，episode 跑到第 5 步和第 500 步，单步成本完全一样（对比：注意力解码器对历史是 O(t) 计算 O(t) 存储）。分母的 `max(...,1)` 是数值稳定阀。

> 符号与本文/相关文档保持一致：mLSTM 出自 xLSTM（论文 ref [6]）；流匹配沿用 π0（ref [2]）的做法，但只把 flow 限制在动作头内。

## 3. 带数字走一遍：玩具例子 (Worked Example)

以「把 3 个一模一样的碗从左到右依次摆到空盘上」为例，走一遍一个控制步的闭环（数值为说明用，结构遵循论文）。

```
第 t 步输入：
  指令 token(16) + 第三人称 8×8 grid(64) + 腕部(36) + 深度(36)
  + DINOv2(64) + 本体(1)  = 217 个观测 token
拼接 T=10 个动作 query：
  流 = [o1..o217][q1..q10]  (与之前所有步的流连续，状态不重置)

mLSTM 推进（每层）:
  用旧 C_{t-1} 并写入本步信息 → 新的 C_t（尺寸不变，与 t 无关）
  "已摆放 2 个碗"这一计数，因前两次放置动作流过状态而隐含其中

读取 query 隐状态 z_t（10 个）

flow-matching 生成动作 chunk:
  a ← 噪声(10×7)
  重复 8 次:  a ← a + (1/8)·v_θ(z_t, a, τ)
  → 10 步 EE 位姿增量 + 夹爪命令

若把状态在每步重置（N=1）:
  → 全任务成功率从 77.5% 掉到 7.0%（论文 Table III）
```

可计算闭环的关键点：**重置间隔 N 就是可调旋钮**。N=1 → 7.0%；N=5 → 43.5%；N=10 → 56.0%；N=20 → 64.0%；N=30 → 74.5%；不重置 → 77.5%。这条曲线本身就是"记忆被真正使用"的证据（论文 Table III）。

## 4. 工程视角 (Engineering View)

| 工程维度 | 数字 / 含义 |
|----------|-------------|
| 单步成本 | 与 episode 长度解耦：第 5 步与第 500 步开销相同（记忆 O(1)） |
| 记忆存储 | O(1)（对比注意力解码器 O(t)） |
| 推理步数 | 循环流每控制步推进 1 次；动作头独立做 8 步 Euler 积分 |
| 解耦好处 | 只有 4.87M 的动作头参与 ODE 迭代，mLSTM 主干每步只前向一次，不被反复重入 |
| 控制频率 | 1 控制步 = 10 个环境帧（与 10 步动作 chunk 对齐） |
| batch 组织 | 按 token 预算（≤37,500 padded tokens ≈ 165 控制步）而非固定 batch size → 变长演示下显存恒定 |
| 训练成本 | 30k 步，AdamW lr 3e-4、wd 0.05、warmup 100、clip 0.5、accum 4、bfloat16，单张 GPU；报告 checkpoint 为 25k 步 |
| 部署约束 | 强依赖腕部相机（去掉后成功率 12%）；去掉深度仅 −3.0pp、去掉 DINOv2 仅 −1.5pp |

工程含义：这是一个**边界清晰**的架构——记忆在控制层、动作生成在外挂头，二者通过 query 隐状态通信。对做实时控制的团队友好：想换动作头（flow ↔ 回归）或换记忆后端，改动面很小（论文 Table II 里 flow 79.6% vs 回归 65.9% 就是换头的效果，但作者强调这不构成 flow 普遍的优越性）。

## 5. 数据与评测 (Data & Eval)

- **训练数据**：7,461 条演示、140 个任务，联合训练一个策略；无 robot pretraining。
- **数据混合**：LIBERO-{90, Spatial, Object, Goal, Long, Mem}，权重未在正文给出具体配比 → `> TODO: 训练集权重配比未列出`。
- **评价基准**：
  - **LIBERO-Mem**（10 任务，20 unseen 初始状态/任务）：任务分四类——Object Motion (OM)、Object Sequence (OS)、Object Relations (OR)、Object Occlusion (OO)。指标为 subgoal coverage（平均子目标完成率）与 full-task success。
  - **标准 LIBERO**（Spatial/Object/Goal/Long，40 任务×50 rollouts）。
- **结果（论文 Table I / II）**：

| 设置 | SmoLSTM (flow) | 关键对比 |
|------|----------------|----------|
| LIBERO-Mem coverage | **85.1%** | 超 2AM 8.8pp；超 E.-S.SSM 70.3pp |
| LIBERO-Mem full-task | **77.5%** | MemoryVAM 42.5%；2AM 63.0% |
| 标准 LIBERO 平均 | 79.6% | SmolVLA-0.24B 82.8%；SmolVLA-0.45B 87.3%；π0 86.0%；OpenVLA-OFT 97.1% |
| LIBERO-Long | 69.4% | 高 SmolVLA-0.24B 6.4pp，低 SmolVLA-0.45B 1.6pp |

- **消融（论文 Table III）**：状态重置 N∈{1,5,10,20,30}；跨 rollout 状态替换；删模态。**注意**：作者自己指出 OS 任务的成功指标无法区分"按指令停在正确次数"和"一直重复"——用单一 pick-and-place 指令替换后，成功率仍有 0.0%–75.0%，说明**基准成功本身不能证明指令条件下的计数能力**。这是难得的诚实。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做**
- 涉及物体身份追踪、遮挡、顺序依赖的长时序操作（LIBERO-Mem 全 10 任务至少 1 次成功）。
- 在 0.04B 可训参数下，标准 LIBERO 平均 79.6%，超过 VLM-initialized 的 π0（71.8%）7.8pp。

**不能做 / 会失败**
- **Object Relations (OR)**：即便不重置状态，成功率也只有 8%——这是架构性失败，非记忆问题。
- **无腕部相机 → 崩**：成功率跌到 12%，尽管训练时做过模态 dropout。说明腕部视角是隐性的强依赖。
- **计数指令不可靠**：OS 类任务成功指标无法证明数字条件下的停止行为（见 §5）。
- **环境单一**：仅在 LIBERO/LIBERO-Mem 仿真；编译环境差异（相机角度、分辨率）使其难以并入 MIKASA-Robo。
- **绝对 SOTA 不是**：在 per-suite fine-tune 的 OpenVLA-OFT（97.1%）面前有明显差距。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设 mLSTM 状态能无损压缩 episode**：论文只证明"重置会掉分"，未隔离出 mLSTM 相对其它记忆机制（memory token / 记忆库）的优势。作者自己写明"do not isolate an advantage of mLSTM over every memory mechanism"。
- **假设仿真内的成功 = 真实记忆能力**：OS 任务的指标缺陷被作者点出，说明部分成功可能来自"继续重复"而非"记住次数"。
- **假设腕部相机会一直有**：模态 dropout 未能让策略对腕部视角缺失免疫（12% 成功率）。
- **假设监督信号够干净**：演示中含 re-grasp 尝试，破坏了"抓取次数 = 完成重复次数"的对应。

## 7. 与相关工作对比 (Comparison)

| 方法 | 记忆机制 | 架构 | 训练方式 | 适用场景 | 记忆成本 |
|------|----------|------|----------|----------|----------|
| RoboFlamingo | LSTM policy head | 冻结 VLM + 分离 LSTM 头 | 训 policy head + 部分 VLM | 通用操作 | O(1) |
| LRAM | xLSTM 状态-动作 | 序列模型 | 轨迹上下文训练 | 仿真控制 | O(1) |
| X-IL | Transformer/Mamba/xLSTM | 语言条件 + rectified flow | — | LIBERO/RoboCasa | 混合 |
| MemoryVLA | 记忆库检索 | VLA + memory bank | — | 通用 | O(t) 检索 |
| μVLA | memory token | OpenVLA-OFT 骨干 | 干预测试 | 通用 | O(1)~ |
| 2AM | 任务记忆 steer 无状态动作模型 | 分离 agent | — | LIBERO-Mem | — |
| **SmoLSTM** | **mLSTM 矩阵记忆，全 episode 不重置** | **冻结 SmolVLM + 统一循环流 + flow 头** | **联合训练 0.04B** | **长时序/遮挡操作** | **O(1)** |

🎯 **面试 Tip**：被问到"这篇和 μVLA/memory token 类方法区别在哪"，一句话答：**"它不额外加记忆模块，而是让观测 token 和动作 query 共享同一条从不重置的循环流——记忆是架构的副产品，不是外挂；代价是它只证明了'重置会掉分'，没证明 mLSTM 优于其它记忆机制。"**

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做**长时序灵巧操作**、需要物体身份/顺序记忆的研究者；
  2. 想在**小模型 + 单卡**预算内复现记忆型 VLA 的工程同学；
  3. 关心"如何用损失设计逼模型真正使用记忆"（观测 dropout + 未来预测）的方法论读者。
- **建議章節路徑**：先讀 §III-B（统一循环流，含 Eq. 1–3）→ 再看 §III-D（逼记忆的损失）→ 最后 §IV-C（Table III 的状态干预，这是全文最有说服力、也最诚实的部分）。§III-A 的 217 token 细节可速覽。
- **不值得精讀的理由**：如果你不做机器人学习、或已熟悉 xLSTM/Mamba 一类循环策略，读摘要 + 本文 §6.1 即可；若你只追绝对 SOTA 分数，这篇的 OpenVLA-OFT 差距说明它不在那条赛道上。

---
[← Back to Theory](./README.md)

**关键引用**
- 论文: https://arxiv.org/abs/2609.22854
- LIBERO: Liu et al., LIBERO benchmark
- LIBERO-Mem + Embodied-SlotSSM 基线: 论文 ref [8]
- 2AM: 论文 ref [9] · MemoryVAM: 论文 ref [10]
- xLSTM / mLSTM: 论文 ref [6]
- π0（flow-matching 动作专家）: 论文 ref [2] · SmolVLA: 论文 ref [12]
