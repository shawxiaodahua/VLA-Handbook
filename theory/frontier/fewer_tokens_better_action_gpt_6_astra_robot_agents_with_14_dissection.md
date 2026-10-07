# 更少 Token，更好动作：PyRUA-Lean 用代码执行接口让机器人智能体成功率 +14%、输入 Token −65% (Fewer Tokens, Better Action: GPT-6 Astra Robot Agents with 14% Higher Success Rate but 65% Fewer Tokens)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-06
>
> **论文**: Fewer Tokens, Better Action: GPT-6 Astra Robot Agents with 14% Higher Success Rate but 65% Fewer Tokens
> **链接**: https://arxiv.org/abs/2610.01939
> **核心定位**: 把 VLM 机器人智能体的"工具调用"接口换成"可执行 Python 代码"接口 —— 在同一 planner、同一底层原语下，成功率从 63.1% 提到 71.7%，共同解出任务的输入 token 减 65%。它踩中的是接口设计（interface）这一层，而非新的 VLA 模型。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 让 VLM 写 Python 代码去组合机器人原语（含 VLA policy），比逐次 tool call 解出更多任务、花更少 token |
| 适合精读 | 如果你在做 VLM/VLA Agent 的 **接口层设计**（tool-calling vs code-execution）、上下文治理、推理成本优化 |
| 可以跳过 | 如果你只关心新 VLA 架构、新感知/动作表示、真机泛化 —— 本文不动模型本身 |
| 落地可行性 | 中偏高：接口改造不依赖新模型，可直接套在现有原语栈上（论文用 Codex CLI + Python 即可复现） |
| 主要风险 | 全部在仿真、单一 planner（GPT-6 Astra）、每个实例只跑一次；收益强依赖任务结构（VLA 主导的任务收益最小） |

💡 **X-Ray 开场**（非专家也能读懂）
机器人智能体现在流行"让 VLM 调工具"：看一眼图、动一下、再看一眼图，每一步都要再问一次大模型。这篇发现，**如果把动作写成一段 Python 小程序**（先移动、到达后再抓、失败就重试），中间过程留在程序里、只把"你真正需要的那张图"传回模型，那么同一套机器人原语能解出更多任务、还更省 token。对 VLA 研究者的意义：**Agent 的瓶颈可能不在 VLA policy 本身，而在动作原语"怎么暴露给上层规划器"**。

📍 **研究全景时间线**

```
2022  SayCan / Inner Monologue        语言模型选技能 + 环境反馈闭环
      │
2023  Code as Policies / ProgPrompt   用 LLM 生成机器人程序
      VoxPoser                        生成代码构造 3D 价值图做规划
      │
2024  CodeAct / SWE-agent             通用 Agent：代码作接口更高效
      π0 / OpenVLA                    可学习 visuomotor policy 兴起
      │
2026Q3 Harness VLA / RPent            冻结 VLA + 分析原语 + 记忆引导（工具调用接口）
      CaP-X / ASPIRE / VLCP           代码 Agent 评测 / 技能发现 / 闭环代码重规划
      │
2026-10 [本文 PyRUA-Lean]  ← 当前位置：把 RPent 的 tool-calling 换成 interactive code execution
      │                              并量化"调用次数 vs 每次 prompt 大小"的 token 分解
      ▼
局限：单 planner + 纯仿真 + 单次运行；接口收益未与"跨 episode 技能复用"结合
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

论文对比的是**同一套机器人栈上的两种接口**，这决定了它的科学价值 —— 变量被尽量控住了。

| 模块 | Tool-Calling 基线 (RPent) | PyRUA-Lean (本文) |
|------|--------------------------|-------------------|
| 规划器 | GPT-6 Astra（high reasoning，Codex CLI） | 同左（完全相同） |
| 动作接口 | 每个原语一次 tool call | 一个 `python(code)` 工具，模型写代码 cell |
| 原语实现 | RPent primitives（move/perception/grasp + VLA policy） | 同一个 `robo` 对象上的同名方法，实现一致 |
| 控制流 | 依赖前一步结果的操作用**下一次 VLM 调用**串起来 | cell 内用 `if` / `for` / 局部重试串起来 |
| 观测回传 | 每次 move 自动返回多张相机图像 + 状态 | 只有 cell 里显式 `print` / `robo.show` 的内容回传 |
| 中间状态 | 累积进对话上下文 | 留在 runtime 的 persistent namespace，不进上下文 |
| 作用域 | 无跨 episode 记忆 | 无跨 episode 记忆（同样关闭，保证公平） |
| VLA policy | π0.5 / LingBot-VLA / RLDX-1（各自基准，冻结） | 同左（冻结，完全共用） |
| 分割 | SAM 3 | SAM 3 |

**一句话**：这不是"新模型 vs 旧模型"，是"接口 A vs 接口 B"。论文标题里的 GPT-6 Astra 只是被当作一个固定强 planner。

### 1.2 关键机制 (Key Mechanism)

- **反馈驱动的原语组合**：一个 cell 可以连跑多个原语，并让后一步**条件依赖**前一步结果（"到达了才抓、没解完才看"）。这些判断在 cell 内完成，**不需要额外 LLM 调用**（论文 §3.2）。
- **持久程序状态**：变量与 helper 函数跨 cell 保留在 namespace 里，中间结果不必塞回对话。
- **选择性观测**：原语结果默认不进入 LLM 上下文；只有代码里显式要求的图像/状态在 cell 末尾统一回传（论文 §3.3）。
- **异常也是反馈**：未捕获异常终止当前 cell 并把 traceback 返回 VLM，异常前已赋值的变量保留，模型可在下一个 cell 里修补（论文 §3.1）。
- **任务成功即熔断**：每个作用于机器人的原语执行前，runtime 会检查任务是否已完成，完成则终止 cell。

⚡ **Eureka Moment**：把"规划器与机器人的对话"从**多轮问答**降级成**一次代码提交** —— 依赖链、重试、几何计算全部下沉到 runtime，于是"减少调用次数"而非"缩小每次 prompt"，才是省 token 的主因。

### 1.3 信息流/架构图 (Flow / Diagram)

```
        ┌──────────────────── 一个交互回合 ────────────────────┐
        │                                                       │
   VLM (GPT-6 Astra) ──python(code)──►  运行时 persistent namespace
        ▲                                │  robo.move_to(...) / robo.segment(...)
        │                                │  robo.pi0_pick(...) / np.linalg.eigh(...)
        │                                │  if 到达: 抓 ; if not done: 再看
        │                                ▼
        └──◄── cell 末尾只回传: 显式 print 行 + robo.show 请求的图像 ──┘
                    （其余中间结果留在 runtime，不进上下文）

   对比 Tool-Calling：每一步操作 = 一次 LLM 往返，且 move 后自动带回多张图
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
总输入 token  ≈  (LLM 调用次数) × (每次调用平均输入 token)
本文的省法：压前者，而非后者
```

先给目标，再给公式，再给变量说明。

目标：在**相同 LLM 调用预算**下最大化成功率，同时最小化累计输入 token。

核心分解式（论文 §4.3，Figure 5）：

```
token_reduction_factor = call_reduction × per_call_token_reduction

LIBERO-PRO:
  token 比 4.48×  =  2.55× (更少调用)  ×  1.76× (每次更小)

即：4.48 ≈ 2.55 × 1.76
```

变量说明：

| 符号 | 含义 | 来源 |
|------|------|------|
| call_reduction | tool-calling 调用数 / code 调用数 | 论文 Figure 5 |
| per_call_token_reduction | 每次调用平均输入 token 之比 | 论文 Figure 5 |
| token_reduction_factor | 二者乘积，即总 token 缩减倍数 | 论文 Figure 5 |

直觉：**如果每次 prompt 都一样大，唯一能省的就是"少问几次"。** PyRUA-Lean 让一次代码提交替掉好几次往返，所以主导项是 call_reduction（LIBERO-PRO 上 2.55× 来自调用数，只有 1.76× 来自每次变小）。反过来，在 RoboCasa365 atomic 上，code 的**单次调用反而更大**（API 描述更长），但调用次数少仍让总量下降。

> 符号与本文保持一致：全文用 `×` 表示"倍数"。论文用 `Δ`（百分点）与 `×`（倍数）两种口径，正文会分开标注。

> TODO: 论文未给出统一的显式优化目标函数（没有 loss），本文的"总 token"口径是作者在 §4 使用的记账式分析，非训练目标。

## 3. 带数字走一遍：玩具例子 (Worked Example)

看论文附录的一个真实 episode（LIBERO-PRO, spatial swap, task 7, seed 2，见 §3.2 的 bowl-placement 例子）：

**Tool-calling agent**：读指南 → 定位 → 每次调用只走一步 → 每次 move 回 3 张相机图。
- 解出任务用了 **第 17 次 LLM 调用**；
- 全程**返回 33 张图像**；
- 累计 **816k token** 才解出。

**PyRUA-Lean code agent**（同一实例、同一底层原语）：
1. cell 1：一次 `segment` 定位 bowl 与 plate（依赖检查在 cell 内）；
2. cell 2：approach + 条件抓取；
3. cell 3：合爪并抬起；
4. cell 4：检查抓取；
5. cell 5：搬运 —— 用 `grip_offset` 算出放置目标，waypoint 循环，只在失败时 `break` 并 `robo.show`；
6. cell 6：算 bowl 底部高度（`np.quantile` 取 2% 分位）决定放置 z，条件释放。
- 解出任务用了 **第 6 次 LLM 调用**；
- 全程**只请求 5 张图像**；
- 累计 **156k token** 解出。

一个 cell 就替掉了基线的第 14–17 步，把 **4 次 LLM 往返压成 1 次**，而且成功放置时**一张图都不回传**（只在"还没解完"时才 `robo.show`）。

这就是 §2 公式的闭环：调用次数 ↓ ≈ 17→6，图像与中间状态不再进上下文 → 总 token 从 816k 掉到 156k。

## 4. 工程视角 (Engineering View)

| 工程维度 | 本文的设定/证据 | 工程含义 |
|----------|----------------|----------|
| 决策频率 | 以 **LLM 调用次数** 而非时间计量（论文 §Appendix E 明确"用调用不用秒"） | 共享 model gateway 延迟抖动大，跨 agent 比延迟不公平；但真机部署仍需补延迟数据 |
| 调用预算 | 每 episode **40 次** LLM 调用（RoboCasa365 复合任务 100 次）+ **2 小时**墙钟上限 | 预算是"调用数"，不是"原语执行数"；一个 cell 内可跑多原语 |
| 单 cell 限制 | **无**单独 per-cell 上限；Codex CLI 300s 超时**不取消** cell，后续 cell 排队 | 长 cell 不会被打断，但可能拖长 tail latency |
| 上下文管理 | Codex CLI 保留完整对话，**接近 272K token 才压缩**（摘要早期轮次） | "选择性观测"控的是**新增**上下文，不是删历史 —— 这是它相对朴素"裁剪"更稳的边界 |
| 成本口径 | 输入 $10/M、缓存输入 $1/M、缓存写 $12.50/M、输出 $50/M（2026-09 牌价） | 货币收益小于 token 收益，因基线有**缓存折扣**：$1.63→$0.74（非等比例） |
| 首 prompt 规模 | tool 20k / code 22k（LIBERO-PRO）等，code 的 API 描述更大 | 说明"每次变小"不是必然；大 API 会让 code 在短任务上吃亏 |
| 内存/状态 | persistent namespace 持有变量、helper、局部世界坐标 map | 相当于把"工作记忆"放在进程内而非对话里，减少上下文带宽压力 |

一句话工程结论：**这套改造的收益来自"降低模型交互频率"，代价是"把更多逻辑写进代码、依赖 Python 运行时正确性"**。对延迟敏感的真机闭环，还缺论文未提供的 wall-clock 数据。

## 5. 数据与评测 (Data & Eval)

| 项目 | 内容 |
|------|------|
| 基准 | LIBERO-PRO（4 个扰动 suite 取 40 任务）、RoboTwin 2.0（50 双臂任务）、RoboCasa365（Target50 取 50 任务） |
| 规模 | 每任务 5 个环境 seed，每任务-seed 各跑一次两 agent → **700 个配对实例 / 1400 episodes** |
| VLA policy | LIBERO-PRO: π0.5；RoboTwin 2.0: LingBot-VLA；RoboCasa365: RLDX-1（均冻结） |
| 成功判定 | 各基准自带的 success check |
| 预算 | 40 LLM calls（RC365 复合 100）+ 2h |
| 主要指标 | 全实例成功率；**双方都解出**的子集上的 mean LLM calls / 输入 token / 估计成本 |
| 附加 | RoboDojo pilot：3 个任务，30 次尝试 |

关键数字（论文 Table 2）：

| 指标 | Tool-Calling | PyRUA-Lean | 变化 |
|------|--------------|------------|------|
| 整体成功率（全实例） | 63.1% | 71.7% | +8.6 pt（约 +14% 相对） |
| LIBERO-PRO | — | — | +11.0 pt |
| RoboTwin 2.0 | — | — | +9.2 pt |
| RoboCasa365 atomic | — | — | +7.8 pt |
| RoboCasa365 composite | — | — | +5.0 pt |
| 仅本 agent 解出实例 | 36 | 96 | — |
| 共同解出：mean LLM calls | 17.0 | 8.7 | −49% |
| 共同解出：mean 输入 token | 788k | 276k | −65% |
| 共同解出：成本/EP | $1.63 | $0.74 | 降（受缓存折扣影响） |

补充口径（论文 §4.2 与 Appendix E）：按 token 预算回溯，LIBERO-PRO 上 PyRUA-Lean 用 **564k** token/EP 就达到基线的最终成功率，而基线要 **3.18M**。RoboDojo pilot 中 26.7%（8/30）vs 3.4%（1/29），每成功一次的总输入+输出 token 从 11.56M 降到 1.07M（−90.7%）。

> 注意评测条件：**这些数字都建立在仿真 + GPT-6 Astra + 单次运行之上**；"共同解出子集"的 token 比不等于"全部尝试"的成本比（见 §6）。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做（有证据）**：
- 在同一调用预算下解出更多任务（+8.6 pt），尤其 LIBERO-PRO（+11.0 pt）；
- 在双方都解出时显著省钱省 token（−49% 调用 / −65% 输入 token）；
- **移除 VLA policy 后仍保留优势**：RoboTwin 2.0 上无 VLA 时 68.8% vs 42.8%（论文 Table 3）；
- 能自行用 NumPy 做感知/几何：30% 的 cell 用 NumPy（颜色掩码、主成分轴求 yaw、平面拟合找朝向）。

**不能做 / 代价（论文自述）**：
- **VLA 主导的任务收益最小**：RoboCasa365 atomic 上，VLA 已能完成大部分，初始 prompt 又占大头；code 的更大 API 描述抵消了一部分省下的调用。**含失败尝试**时，基线在该子集上的估计成本仅约为 code 的 **1.1×**。
- **"按需图像"单独用没用**：把基线改成"只在请求时返回图像"后，LIBERO-PRO 成功率反从 **83.0% 掉到 68.5%**（论文 Table 8）—— 说明选择性观测必须和代码执行、重规划**一起**才有用。
- **指南（operating guides）不是一律有益**：RoboTwin 2.0 无 VLA 时，去掉指南反而把 tool-calling 成功率从 42.8% 抬到 60.4%（论文 §4.4）。
- **96 个只有 code 解出的实例中，52 个是基线耗尽调用预算、44 个是基线自己终止**；后者里也有"同一 VLA 换个运行结果就不同"的情况 —— 增益**不能全部归因**于更好的恢复逻辑。
- 只有一个 planner、只有仿真、每实例只跑一次 → 泛化性、真机、重复运行方差**均未验证**。

### 6.1 隐含假设 (Hidden Assumptions)

作者默认成立但未明确验证的前提：

1. **原语"可被代码安全组合"**：代码 cell 直接操作真机/仿真原语，隐含假设 `robo` 各方法**幂等性/失败可检测性足够**，且"任务完成即熔断"的检查足够可靠。真机上若原语失败静默，cell 内的条件分支可能误判。
2. **对话只需在 272K 时才压缩**：假设 Codex CLI 的长上下文 + 自动摘要不会丢失关键早期信息；如果关键约束在早期轮次被压缩掉，收益可能不成立。
3. **单次运行代表能力**：每实例每 agent 只跑一次，隐含假设 seed 方差小到可忽略 —— 但作者自己承认"VLA 在不同运行里会成功/失败互换"。
4. **token = 成本的有效代理**：以 2026-09 牌价折算成本，假设缓存折扣稳定、且调用延迟不进入目标函数；对实时机器人闭环，这一假设可能失效。
5. **冻结的 VLA 足够可靠**：结论部分依赖"原语多数情况下可靠"这一前提；论文实际上在无 VLA 消融里补充了这点，但主实验的 VLA 版本没有系统性刻画原语失败率。

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 接口/架构 | 训练方式 | 适用场景 |
|------|--------|-----------|----------|----------|
| SayCan | 技能选择接地 | 语言模型 + affordance | 无需训练 LLM | 长程家务技能挑选 |
| Inner Monologue | 规划中融入反馈 | 文本反馈闭环 | 无需训练 | 反馈驱动的重规划 |
| Code as Policies / ProgPrompt | 语言→机器人程序 | 代码生成 | 无需训练 | 结构化任务，静态程序 |
| VoxPoser | 空间价值图规划 | 代码构造 3D 价值图 | 无需训练 | 运动规划 |
| CodeAct / SWE-agent | 通用 Agent 效率 | 代码作行动接口 | 无需训练 | 软件工程等通用任务 |
| Harness VLA / RPent | 冻结 VLA + 原语 | **tool-calling** 记忆引导 | 冻结 VLA | 机器人操作（本文基线） |
| **PyRUA-Lean（本文）** | **接口效率** | **interactive code execution + 选择性观测** | **冻结 VLA** | **同一机器人栈上的成功率与推理成本对比** |

**面试 Tip**：被问到"这篇新在哪"，一句话答 —— **"它没改模型，而是把 VLM↔机器人的接口从 tool-calling 换成 Python 代码执行，并用 token 分解证明省 token 主要来自'少调用'而不是'更小的 prompt'；同时用'按需图像单独用会掉点'的实验说明选择性观测必须和代码执行绑定。"**

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 在做 **VLM/VLA Agent 接口层**（tool-calling vs code-execution、context 治理）的研究者；
  - 要评估"把现有原语栈改造成 code-execution"可行性的工程团队；
  - 对 Agent 推理成本/调用预算敏感、想找省 token 杠杆的人。
- **建议章节路径**：先读 §3（机制：组合 + 选择性观测）→ 再看 §4.2/§4.3（主结果 + token 分解）→ 之后读 §4.4 消融与 §5 局限；Appendix A/C（真实 cell 代码与 cell 行为统计）是精华，值得对照看。可跳 §2 相关工作（除非你要写综述）。
- **不值得精读的理由**：如果你不做机器人学习、且已熟悉 CodeAct 一类"代码作接口"的思想，读摘要 + §4.3 的分解式即可。

## Pulsar 系统洞察 (System Insight)

- **跨域交叉信号**：本文与 AI 应用线的 `Code Mode`（Cloudflare）/`Code execution with MCP`（Anthropic）是同一套直觉 —— **把 Agent 的工具调用协议从"逐次 JSON"改成"一次代码提交"**，用 runtime 承接中间态。VLA 侧这次给出了**机器人场景下的量化证据**（调用数 −49%、token −65%），对 Ken 的 Agent+UI 方向是可迁移的接口设计原则。
- **对 VLA 线的含义**：论文强化了一个判断 —— **在强 planner + 可用原语的设定里，瓶颈常常在"接口与上下文治理"，而非再加一个大 VLA**。对"RL 训练 VLA / VLA 后训练"路线是互补视角：policy 更强时，接口收益会缩小（RoboCasa365 就是例子）。
- **可复用实验范式**：`token_reduction = call_reduction × per_call_reduction` 这个分解式，可直接搬到任何"多轮工具调用 vs 代码执行"的对比里，作为买单前先做的成本体检。

---
[← Back to Theory](./README.md)

**关键引用**
- 论文: https://arxiv.org/abs/2610.01939
- 基线系统 RPent / Harness VLA: https://arxiv.org/abs/2607.08448
- 相关接口工作 CodeAct: https://arxiv.org/abs/2402.01030 （Executable Code Actions Elicit Better LLM Agents）
