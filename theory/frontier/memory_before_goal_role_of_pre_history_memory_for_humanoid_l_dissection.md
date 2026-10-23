# 前史记忆：人形 Loco-Manipulation 的对象事件关键帧记忆 (Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-23
>
> **论文**: Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation
> **链接**: https://arxiv.org/abs/2610.31166
> **核心定位**: 它把 humanoid whole-body VLA 的盲区从“当前帧看不到”推进到“用对象事件记住动作前发生的关键帧”；不是更大的记忆，而是更对的记忆。

> 版本说明：本文依据 arXiv:2610.31166v1 [cs.RO] 2026-10-22 的 HTML 全文。候选摘要与正文摘要的关键数字一致（29.1→72.6、36.7→88.3、70.2→86.9）；来源记录未给出 `abstract_verification` 结果，未发现官方 PDF 与元数据摘要不一致。若后续 PDF 覆盖，以 PDF 正文为准；本文不采用候选摘要补写正文未出现的主张。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 在人形 loco-manipulation 中，用 object events 选择关键历史帧，比均匀采样或保留更多帧更有效；仿真记忆任务 29.1%→72.6%，真机 36.7%→88.3%，SIMPLE 70.2%→86.9%（论文表 I/III/V）。 |
| 适合精读 | 做 whole-body VLA、机器人记忆机制、人形 benchmark 的人；重点看 §IV-B 选帧、§IV-C 记忆进入 action expert、§V-C/§V-D/§V-E 评测。 |
| 可以跳过 | 只关心桌面机械臂记忆或纯 VLM 预训练的人，距离中等；但“对象事件选帧”可迁移。 |
| 落地可行性 | 中：需要在线 SAM3 跟踪、帧缓存、改造 action expert；真机约 1.25 Hz 跟踪，论文未给端到端延迟。 |
| 主要风险 | SAM3 漏检即失效；在线跟踪有成本；sim-to-real gap；仿真演示是脚本生成；数据/代码开放范围未说明。 |

💡 **X-Ray 开场**

这篇论文问的是：人形机器人转身、走路、物体被遮挡之后，VLA 还能不能根据“动作前发生过什么”做决定？它发现当前 whole-body VLA 基本只条件化当前观测，于是提出 WholeBodyMem：用 SAM3 跟踪任务对象，把“出现、消失、重现、移动、形变”等对象事件变成选帧信号，只把少量关键历史帧送进 action expert。对 VLA 研究者而言，这意味着记忆的瓶颈可能不是“存多少”，而是“选哪几帧、以什么接口读”。

📍 **研究全景时间线**

```
2023  Diffusion Policy / ACT / Flow Matching：动作生成与模仿学习基础
2024  HumanoidBench：人形全身控制 benchmark
2025  GR00T N1 / π0.5 / AMO / SAM2Act：人形 VLA、全身控制、记忆架构并行推进
2026  Ψ0 / SIMPLE / RoboMME / SAM3：全身 VLA 与记忆评测进一步成熟
2026-10  本文 WholeBodyMem + HuMeGo ← 当前位置
```

---

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 | 归属/备注 |
|------|------|------|-----------|----------------|-----------|
| Frozen SAM3 tracker | head-camera 帧、每任务 1–2 个短语 prompt | 每个对象每帧的 mask、box、confidence | 仿真每个 50 Hz 控制步跟踪；真机每个执行 chunk 跟踪一帧，约 1.25 Hz | 冻结，不训练 | 论文用 SAM3 只做 tracker，不用其记忆机制 |
| Camera-motion removal | 相邻帧背景特征 | similarity transform | 每跟踪帧 | 拟合，非学习 | 把对象运动换算到场景相对坐标 |
| Object event detector | 跟踪状态 | appear / disappear / reappear / move / shape change | 每跟踪帧 | 阈值固定，全任务共享 | 消失/重现优先，confidence<0.35 忽略 |
| Causal selector | event store、已选帧 S、预算 B_max=7、gap g=5 | 3–7 个历史帧 | 每次 policy call 的 t | 无训练，算法 1 | 只允许 ≤t 的帧，因果 |
| Memory token encoder | 选中帧的 77 个 native tokens/帧 | M_t，最多 539 个 token | 每次 policy call 算一次，缓存 | W_m、W_p 训练 | 7×11 grid，2048-D→1536-D |
| Object token encoder | SAM3 mask、box、area、visibility、帧号、质心 | O_t，最多 14 个 token | 每次 policy call | 训练 | 每帧最多 2 个对象 |
| Action expert memory attention | H=30 个 action tokens、M_t、O_t | cross-attention read-out、FiLM 参数 | 6 个 DiT block，每个 flow step 共享 M_t | memory attention 从零训练；joint attention 保持预训练 | 记忆只进 action expert，不进冻结 VLM |
| Decision head | M_t、O_t、null token | 每个答案的 logit，softmax 为 w | 每次 policy call | 交叉熵训练，λ=0.5 | 离散答案 2–3 个 |
| Pad tokens | 当前帧可见 pad 的 SAM3 box、几何、pad index | P_t | 每次 policy call | 零初始化输出投影 | w 给 pad token 加分，分支阈值 0.6 |
| Base policy/backbone | I_t、s_t、ℓ | 仿真 36-D whole-body action；真机为 SONIC 64-D motion token + 14-D hand targets | 50 Hz | Qwen3-VL-2B 冻结；action expert 微调；真机 action space 不同 | 每任务单独 fine-tune |

### 1.2 关键机制 (Key Mechanism)

- **非马尔可夫痛点**：动态场景里 cue 可能被遮挡、熄灭、被拿走，或者因为机器人转身走路而离开 head camera。当前帧无法决定正确动作， chance 只有 `1/|Y|`。
- **对象事件而不是均匀采样**：SAM3 跟踪任务对象；对象“出现/消失/重现/移动超过图像对角线 3%/mask 重叠低于 65%”才成为候选帧。事件按类型和 confidence 排序，消失/重现优先。
- **自适应预算**：`B_max=7`，始终保留 first、middle、current 三帧；最多再加 4 个事件帧；若某事件帧的对象在 S 中 5 帧内已有帧，则跳过。实际记忆 3–7 帧，平均 5.4 帧。
- **Native tokens 而不是 pooled summary**：每帧保留 77 个 native vision tokens（7×11 grid），因为 cue 可能只占图像不到 1%；pooled 会丢小区域。
- **记忆只通过 action expert 进入**：M_t 和 O_t 通过 cross-attention 被 H=30 个 action tokens 读取，再用 FiLM 调制 action tokens；冻结的 VLM 不被改写。论文的 “memory as context” 消融反而低于 memoryless 基线。
- **Decision head + pad tokens**：decision head 把记忆映射到离散答案，softmax 权重 w 既调制 action expert 的 adaptive layer norm，也给当前可见 pad token 加分，形成“记住 cue → 指向 pad → 执行动作”的闭环。
- **因果与缓存**：只选 `t` 及以前的帧；M_t 每次 policy call 算一次，所有 block 和 flow-matching step 共享；重复选帧因缓存而免费。

⚡ **Eureka Moment**：THE 关键洞见是——记忆的瓶颈不是容量，而是“选对帧”；对象事件是几乎免费的选帧监督信号，而 action expert 的 cross-attention 是把这些帧变成动作条件的合适接口。

### 1.3 信息流/架构图 (Flow / Diagram)

```
Head camera I_t ──► SAM3 tracker ──► object events ──► causal selector (B_max=7, g=5)
      │                                      │
      │                                      ▼
      │                             selected frames (3–7)
      │                                      │
      │                     ┌────────────────┴────────────────┐
      │                     ▼                                 ▼
      │              native memory tokens M_t          object tokens O_t
      │                     │                                 │
      └──► VLM backbone (frozen) ──► action expert (6 blocks) ◄┘
              ▲   ▲                        │  cross-attention + FiLM
              │   │                        ▼
           I_t   ℓ                 H=30 future actions
                                           │
                                           ▼
                                  decision head w ──► pad tokens P_t
```

注意：图中的 `H=30` 是未来动作 chunk，不是历史记忆窗口；记忆只包含 `t` 及以前的帧。

---

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
πθ(a_t | I_t, s_t, ℓ, M_t, O_t) = 用对象事件选出的历史帧来条件化未来动作
```

先给目标：让未来动作依赖已经消失的 cue，而不是只依赖当前帧。基础策略是 `πθ(a_t | I_t, s_t, ℓ)`，输出 `H` 个未来动作；WholeBodyMem 增加 `M_t`（memory tokens）和 `O_t`（object tokens）。

**Memory token 构造**：

```
m_i = W_m [ f_i ; SiLU(W_p φ(τ_i, p_i)) ]
```

- `f_i ∈ R^2048`：第 i 个 native vision token。
- `τ_i`：帧索引；`p_i`：7×11 grid 的 cell 位置。
- `φ(τ_i, p_i)`：帧号和 grid cell 的 sinusoidal code。
- `W_m`、`W_p`：投影到 action expert 的 1536-D 宽度。
- 空 slot 被 mask；`M_t = [m_1, ..., m_539]`。

**Memory retrieval**（每个 DiT block j，joint attention 之后）：

```
c_j = Attn( Ā_j W_Q, M_t W_K, M_t W_V ) W_O
```

- `Ā_j`：RMS-normalized 的 action tokens。
- 多查询注意力；6 个 query heads，每个 256 维，一个共享 key/value head。
- rotary code 把 action steps 放在 packed memory tokens 之后。
- 预训练 joint attention 不变，记忆读取是旁路。

**FiLM 调制**：

```
[γ_j, β_j] = W_F c_j
Ā~_j = Ā_j ⊙ (1 + γ_j) + β_j
```

- `W_F` 初始化接近零，所以训练开始时策略接近预训练策略。
- 记忆不进入 residual stream，而是调制 block 的 adaptive layer norm 和 FFN 前的 action tokens。

**Object token 分支**：

```
c_j ← c_j + g · Attn(Ā_j, O_t)
```

- `g` 可学习，初始为 1；输出投影零初始化，所以分支初始“静音”但可训练。
- 每帧最多 2 个对象，7 帧最多 14 个 object tokens。

**Decision head（概念式）**：

```
w = softmax( MLP( Attn(q, [M_t ; O_t ; null]) ) )
```

- 一个 learned query 对 memory、object 和 null token 做 attention，MLP 输出每个答案一个 logit。
- 从 decision step 起用 demonstrated answer 做 cross-entropy。
- softmax 权重 `w` 进入每个 block 的 adaptive layer norm；也给 pad token 加分：

```
score_k ← score_k + log w_k
```

- pad 分支只在最大权重和可见 pad 权重达到 0.6 时激活。

**训练损失**：

```
L = E || v_θ(a_t^σ, σ) − (ε − a_t) ||² + λ · L_CE
a_t^σ = (1 − σ) · a_t + σ · ε,  σ ~ U(0,1),  λ = 0.5
```

- 第一项是 flow-matching loss；第二项是 decision head 的交叉熵。
- Qwen3-VL-2B backbone 冻结；memory encoder、memory attention、W_F 从零训练。
- AdamW，β=(0.95,0.999)，lr=1e-4，cosine decay，1k warmup，bf16，dropout 0.1，global batch 128；仿真 40k steps，真机 5k steps。

> 符号说明：`I_t` 当前 head-camera 图像，`s_t` proprioceptive state，`ℓ` 指令，`M_t` memory tokens，`O_t` object tokens，`H=30` 未来动作数，`B_max=7` 记忆帧预算，`K=77` 每帧 native tokens。

**直觉**：memory tokens 保留“哪里发生了什么”的原始视觉细节；object tokens 提供“哪个对象、何时、多大、可见性”的对象级摘要；decision head 把连续记忆压成离散答案；pad tokens 把答案落到当前可见的 pad 上。四者分工，不是把一组属性混在一个模型里。

---

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设的 10 帧例子，仅用于验算选帧逻辑，不是论文实验。

设 `B_max=7`，`g=5`，初始 `S={0,5,9}`（first、middle、current）。跟踪到 5 个事件：

1. blue cup 在帧 2 出现；blue cup 不在 S 中 → 加入 2，`S={0,2,5,9}`。
2. green can 在帧 1 出现；green can 不在 S 中 → 加入 1，`S={0,1,2,5,9}`。
3. blue cup 在帧 4 移动；blue cup 在 S 中有帧 2，距离 2 ≤ 5 → 跳过。
4. red box 在帧 7 消失；red box 不在 S 中 → 加入 7，`S={0,1,2,5,7,9}`。
5. red box 在帧 8 重现；red box 在 S 中有帧 7，距离 1 ≤ 5 → 跳过。

最终 6 帧，满足 3–7 帧预算。按每帧 77 个 native tokens 算：

```
6 × 77 = 462 memory tokens
满预算 7 × 77 = 539 memory tokens
少 77 / 539 = 14.3%（这个假设例子）
论文平均 5.4 帧，少 (7−5.4)/7 = 22.9%
```

若每帧最多 2 个 object tokens，6 帧最多 12 个；满预算最多 14 个。token 总量约 `462 + 12 = 474`，对比 `539 + 14 = 553`。这说明“少帧”确实省 token，但论文的核心不是省 token，而是在同样预算下选对帧。

**时序换算（假设/换算，不是论文实测）**：

```
K 个均匀采样帧的首末跨度 = (K−1) × Δt
仿真 50 Hz，7 帧连续：6 × 0.02 s = 0.12 s
真机跟踪约 1.25 Hz，7 帧连续：6 × 0.8 s = 4.8 s
```

但实际 selector 不是固定间隔，可能从 episode 起点跨到当前 `t`，所以 0.12 s / 4.8 s 只是“连续选帧”的下界，不能当作所有任务的记忆窗口。

**未来动作 chunk 不要混用**：

```
论文记法 a_t ... a_{t+H−1}：最后动作距 t = (H−1) × Δt = 29 × 0.02 = 0.58 s
若记法 t+1 ... t+H：终点距 t = H × Δt = 30 × 0.02 = 0.60 s
真机执行前 24 个动作：24 × 0.02 = 0.48 s（按 50 Hz 控制步）
```

这些是未来动作 horizon，不是历史记忆窗口。连续未来 chunk 的预测区间不能直接当作历史上下文或记忆窗口。

---

## 4. 工程视角 (Engineering View)

| 项目 | 论文设置 | 工程含义 |
|------|----------|----------|
| 仿真跟踪 | SAM3 每个 50 Hz 控制步跟踪 | 事件检测粒度细，但计算重；论文未给 SAM3 延迟 |
| 真机跟踪 | 每个执行 chunk 跟踪一帧，每 24 帧，约 1.25 Hz | 若 cue 很短且落在两次跟踪之间，可能漏检；这是条件性风险，不是论文实测 |
| 动作 chunk | H=30，真机先执行 24 个再取下一 chunk | 减少 policy call 频率；但记忆更新也按 chunk 异步 |
| 低层控制/动作空间 | 仿真：SIMPLE/HuMeGo 导出 36-D whole-body action；真机：动作标签为 SONIC 64-D motion token + 14-D hand targets，由 SONIC 以 50 Hz 执行 | 36-D 是仿真训练/导出接口，不等于真机策略动作接口；真机 action space 不同，需按 64+14-D 训练/部署；策略输出与低层控制器解耦 |
| 记忆读取 | M_t 每次 policy call 算一次，6 个 block 和所有 flow steps 共享 | 避免每个 block/step 重算；缓存选中的帧 |
| 记忆容量 | 539 memory tokens + ≤14 object tokens | 按 2048-D、539 token 粗算约 1.10M 标量；bf16 约 2.2 MB，fp32 约 4.4 MB（论文未给 dtype，这是估算） |
| 训练成本 | 仿真 40k steps，batch 128；真机每任务 15 个演示、5k steps | 每任务单独 fine-tune；不是一次性通用记忆模型 |
| 部署硬件 | 真机 policy 在一张 RTX 5090 上 | 论文未给端到端 Hz；不能仅凭 50 Hz 控制频率推断 policy 实时性 |

**工程边界**：

- 记忆是因果的：只选 `≤t` 的帧，不会偷看未来。
- 记忆接口只在 action expert：冻结 VLM 不变；如果改成把记忆塞进 VLM context，论文消融显示反而降到 25.9%，低于 memoryless 的 29.1%。
- `B_max=7` 是论文的设置，不是所有实验共有的“记忆窗口”；SIMPLE 上用的是固定 7 帧、无 adaptive memory 的变体。
- 真机 `24 帧 ≈ 1.25 Hz` 与 `H=30、执行 24 个动作` 的计时基准论文没有完全展开；按 30 Hz 相机算是 0.8 s，按 50 Hz 控制步算是 0.48 s，不能把两者混为同一时间尺度。

---

## 5. 数据与评测 (Data & Eval)

### 5.1 数据组成

| 数据 | 内容 | 规模/格式 |
|------|------|-----------|
| HuMeGo 仿真 | 6 个记忆任务，场景诱发 cue 消失 | 每任务 100 训练演示、90 测试场景；MuJoCo 2 ms physics、50 Hz control；Isaac Sim 渲染 640×360、50 Hz；LeRobot v2.1，32-D state、36-D action |
| HuMeGo 真机 | 6 个 G1 任务，含运动诱发 cue 消失 | 每任务 15 个遥操作演示，共 90；PICO headset + controllers + ankle trackers 驱动 SONIC 50 Hz |
| SIMPLE | 14 个标准 whole-body loco-manipulation 任务 | 每任务 10 episodes/level；L0/L1/L2 三级随机化 |
| 记忆任务评测 | 6 个仿真任务 | 每任务 30 episodes/level，答案平衡 |

仿真任务：FlashPad（T，计数）、HideMug（S，永久性）、ShellGame（S，永久性）、LitLamp（O，参照）、BlinkPad（O，参照）、TouchPad（O，参照）。真机任务：Trash bag（S）、Remember-chip（O）、Café A（T）、Café B（O+S）、Café C（O+S+T）、Chip-cart（P，进度）。

任务定义要求：cue 在 decision time `t_d` 已不可观测；指令和 `t_d` 前的机器人运动不依赖答案 `y`；`y` 在答案集上平衡。这样 memoryless 策略只能接近 chance，论文把 memory-decision accuracy 与 manipulation success 分开评分。

### 5.2 主要结果

| 设置 | 指标 | Ψ0 / memoryless | Uniform memory | WholeBodyMem | 来源 |
|------|------|------------------|----------------|--------------|------|
| SIMPLE 14 tasks | success / 420 episodes | 295/420 = 70.2% | 323/420 = 76.9% | 365/420 = 86.9% | 表 I |
| HuMeGo 6 sim memory tasks | success / 540 episodes | 29.1% | 44.4% | 72.6% | 表 III/IV |
| HuMeGo 6 real tasks | success / 60 trials | 22/60 = 36.7% | 未测 | 53/60 = 88.3% | 表 V |

注意：SIMPLE 上的 WholeBodyMem 使用固定 7 帧记忆（无 adaptive memory）；记忆任务上才是完整自适应版本。Uniform memory 是 WholeBodyMem 去掉 object tokens 和 decision head、改用均匀间隔帧的对照，预算同为 539 tokens。

**记忆任务分项（表 III，30 episodes/level）**：

| 方法 | FlashPad (T) | HideMug (S) | ShellGame (S) | LitLamp (O) | BlinkPad (O) | TouchPad (O) | 平均 |
|------|--------------|-------------|---------------|-------------|--------------|--------------|------|
| Ψ0 | 7/7/5 | 11/14/12 | 7/11/7 | 11/6/4 | 11/11/11 | 3/11/8 | 8.3/10.0/7.8 |
| Uniform | 22/23/22 | 12/20/15 | 10/9/9 | 12/12/11 | 10/9/13 | 6/13/12 | 12.0/14.3/13.7 |
| WholeBodyMem | 26/27/23 | 21/20/20 | 21/20/19 | 18/15/12 | 26/22/25 | 26/25/26 | 23.0/21.5/20.8 |

Memory-decision accuracy：chance 36%，Ψ0 39%，Uniform 50%，WholeBodyMem 94%（正文称 93.7%）。

### 5.3 消融（表 IV，六任务平均）

| 变体 | L0/L1/L2 成功数（每级 30） | 平均成功率 | 说明 |
|------|----------------------------|------------|------|
| Uniform memory | 12.0/14.3/13.7 | 44.4% | 基线 |
| w/o object tokens & head | 18.0/19.0/17.3 | 60.4% | 只保留事件选帧 |
| w/o decision head | 20.7/18.7/19.8 | 65.7% | 加 object tokens |
| w/o adaptive memory | 21.7/21.3/20.3 | 70.4% | 加 decision head |
| Full WholeBodyMem | 23.0/21.5/20.8 | 72.6% | 加 adaptive memory |
| Memory as context | 8.3/7.0/8.0 | 25.9% | 记忆喂给冻结 VLM，而非 action expert |

论文把 28.2 点增益拆为：key frames 16.0、object tokens 5.3、decision head 4.7、adaptive memory 2.2。adaptive memory 用 22.9% 更少的 token，基本不掉点。

### 5.4 真机结果（表 V，每任务 10 trials）

| 任务 | 记忆类型 | Ψ0 | WholeBodyMem |
|------|----------|-----|--------------|
| Trash bag | S | 4/10 | 9/10 |
| Remember-chip | O | 3/10 | 10/10 |
| Café A | T | 4/10 | 9/10 |
| Café B | O+S | 3/10 | 9/10 |
| Café C | O+S+T | 3/10 | 8/10 |
| Chip-cart | P | 5/10 | 8/10 |
| Total | | 22/60 = 36.7% | 53/60 = 88.3% |

真机只对比了 Ψ0，没有对比 Uniform memory；因此“真机上 Uniform 是否也差”论文没有回答。

---

## 6. 能力与失败模式 (Capabilities & Failure Modes)

### 6.1 能做什么

| 能力 | 证据 | 条件 |
|------|------|------|
| 记住场景诱发消失的 cue | 仿真 6 任务 72.6%，decision accuracy 94% | cue 对象能被 SAM3 跟踪到 |
| 记住运动诱发消失的 cue | 真机 Trash bag、Café A/B/C 8–10/10 | 机器人转身/走路后 cue 离开视野；Remember-chip 不在此列 |
| 记住无痕对象触碰 cue（O） | 真机 Remember-chip 10/10 | 人物触碰罐头后不留痕迹，cue 不是运动诱发消失 |
| 组合 O+S+T | Café C 8/10 | 答案在 2–3 个离散目标中 |
| 不牺牲反应式技能 | SIMPLE 86.9% vs Ψ0 70.2% | SIMPLE 上用固定 7 帧变体 |
| 真机迁移 | 88.3% vs 36.7% | 每任务 15 个演示、5k steps、单张 RTX 5090 |

### 6.2 不能做什么 / 失败模式

- **SAM3 漏检**：如果 cue 对象从未被检测到，记忆里就没有它；论文明确列为 limitation。
- **在线跟踪成本**：SAM3 在线跟踪增加推理成本；论文没有给延迟数字，不能声称实时。
- **短 cue 落在跟踪间隙**：真机约 1.25 Hz 跟踪，若 cue 极短且不在跟踪帧上，可能漏掉；这是本文的条件性分析，不是论文实测。
- **sim-to-real gap**：仿真用 Isaac Sim 渲染，真机有真实光照、材质、运动模糊；论文承认渲染 gap。
- **脚本化仿真演示**：仿真 demonstrations 由 scripted whole-body expert 生成，不是人类演示；真实感与多样性有限。
- **离散答案与每任务训练**：decision head 假设 2–3 个答案，且每任务单独 fine-tune；不是开放语言目标或零样本记忆。
- **对象数量与记忆长度**：每帧最多 2 个对象、最多 7 帧；更多对象、更长程序性记忆未验证。
- **数据/代码开放范围未说明**：论文正文未给出代码仓库或 HuMeGo 数据许可；不要假定可直接复现。

### 6.3 隐含假设 (Hidden Assumptions)

1. **对象可用 1–2 个短语命名**，且 SAM3 能稳定跟踪；prompt 由 Claude Opus 5.5 每任务写一次并固定。
2. **cue 是对象事件式的**：出现、消失、重现、移动、形变；不是长程 procedural memory。
3. **相机运动可由背景相似变换补偿**；动态背景或纯旋转场景未充分验证。
4. **答案集小且离散**，decision head 有 demonstrated answer 监督；开放-ended 指令未验证。
5. **冻结 VLM + action expert 记忆接口**是正确接口；memory as context 消融更差，但这只说明该实现下如此。
6. **每任务 15 个真机演示**足够；更大规模、多任务联合训练未验证。
7. **论文未说明 HuMeGo 数据与代码是否开放**；复现前需要确认。

---

## 7. 与相关工作对比 (Comparison)

| 方法/基线 | 记忆机制 | 架构/训练归属 | 本文评测 | 关键差异/局限 |
|-----------|----------|----------------|----------|----------------|
| Ψ0 | memoryless，只看当前观测 + 短 proprio history | 论文称 open foundation model；本文作为 memoryless base policy，使用 fine-tuned checkpoint | SIMPLE 70.2%；记忆任务 29.1% | 无记忆，cue 离开视野后只能猜 |
| Uniform memory | RoboMME 风格：均匀采样帧 + memory modulator | 作者在本文任务上适配；同 539-token 预算；native tokens 替代 pooled | SIMPLE 76.9%；记忆任务 44.4% | 均匀帧会错过短 cue；无 object tokens/decision head |
| WholeBodyMem | 对象事件选关键帧 + adaptive budget + native tokens + object tokens + cross-attention/FiLM + decision head/pad tokens | 在预训练 base policy 上每任务 fine-tune；Qwen3-VL-2B 冻结；memory 模块从零训练 | SIMPLE 86.9%（固定 7 帧变体）；记忆任务 72.6%；真机 88.3% | 依赖 SAM3；在线跟踪成本；离散答案；每任务训练 |
| GR00T | 本文未描述其记忆机制 | 论文称 open foundation model for humanoids，训练于 real/synthetic/video data；本文作为基线重训 | SIMPLE 37.9% | 不是本文的记忆方法；不能把 WholeBodyMem 的属性归给它 |
| π0.5 | 本文未描述其记忆机制 | 论文称 generalist VLA，适配到 whole-body action space；本文作为基线重训 | SIMPLE 49.0% | 同上；无本文的 object-event memory |

其余 SIMPLE 基线：DP 136/420=32.4%，ACT 176/420=41.9%，EgoVLA 169/420=40.2%，InternVLA-M1 89/420=21.2%。论文没有把它们描述为记忆方法，本文不把记忆属性归给它们。

与 tabletop memory 工作的差异：长历史保留成本高且稀释短 cue；压缩会丢细节；均匀采样会漏短 cue；事件选帧已有 KEMO/EventVLA 等，但 WholeBodyMem 针对 humanoid 的“自身转身走路导致 cue 离开视野”，并用 action-expert cross-attention 读取。SAM2Act+ 用 SAM2 的记忆机制；WholeBodyMem 只用 SAM3 做 tracker，让对象事件选帧。并发工作 POT-VLA 保留“对象现在在哪”的 3D 记录，但不保留“之前发生了什么”。

**面试 Tip**：被问到这篇的核心，不要答“它加了记忆”；要答“它把记忆选择变成对象事件驱动的关键帧选择，并且只让 action expert 通过 cross-attention 读记忆，保持 VLM 冻结；关键证据是同样 7 帧预算下，事件选帧比均匀选帧高 16 个点”。

---

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 whole-body VLA / humanoid loco-manipulation 的研究者：重点看 §III-A 任务定义、§IV-B 选帧、§IV-C 记忆接口。
  - 做机器人记忆机制的工程师：重点看 §IV-B 的事件规则、§V-D 消融、§VI 限制。
  - 做人形 benchmark 的人：重点看 §III 的 HuMeGo 设计、表 II 的对比、§V-A 评测协议。

- **建议章节路径**：先读 §III-A（什么算 memory-dependent）→ §IV-B（对象事件如何选帧）→ §IV-C（记忆如何进入 action expert）→ §V-C/§V-D（记忆任务与消融）→ §V-E（真机）。Related Work 可快速扫。

- **不值得精读的理由**：如果你只做桌面机械臂、或只关心 VLM 预训练、或已经熟悉 RoboMME 类记忆评测，读摘要和表 I/III/V 即可；这篇的增量主要在“人形 + 对象事件选帧 + action-expert 接口”。

- **待补/待证**：
  - 论文未给 SAM3 在线跟踪延迟与端到端 policy Hz；真机 1.25 Hz 与 24 帧的计时基准需确认。
  - HuMeGo 数据/代码是否开放，正文未说明。
  - 更多对象、更长历史、开放语言目标未验证。

---

[← Back to Theory](./README.md)

**关键引用**：[论文链接](https://arxiv.org/abs/2610.31166)