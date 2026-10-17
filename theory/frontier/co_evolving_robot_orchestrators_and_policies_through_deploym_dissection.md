> 历史补录：补回 2026-10-07 的深度阅读任务；论文原始收录日期 2026-10-07；实际补写日期 2026-10-17。本文依据当前读取的指定论文版本，不冒充当时已完成的报告。

# 部署中协同演化机器人编排器与策略 (Co-Evolving Robot Orchestrators and Policies through Deployment)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: Co-Evolving Robot Orchestrators and Policies through Deployment
> **链接**: https://arxiv.org/abs/2610.09228v1
> **核心定位**: 把部署期学习从「冻结策略 + 自适应 harness」推进到「orchestrator 与 policy 共同演化」——用机器人自己的执行轨迹策展技能级数据、按需微调、候选策略通过验证后才替换，并在换模型后主动修订记忆。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 论文报告：在 10 个 RoboLab 模拟任务上，held-out 平均成功率 64.8%（冻结策略 harness）→ 73.8%（Robo-COP）；固定日程训练但不做验证只到 65.8%。3 个真机任务 38.3%→50.0% |
| 适合精读 | 做 agentic robot harness、VLA 部署后持续学习、真实环境数据闭环的人，重点看 §3.2 数据策展、§3.3 候选验证、§4.2 失败分析 |
| 可以跳过 | 只关心单一 VLA 的模型结构或预训练配方、不做部署更新的读者，距离中等 |
| 落地可行性 | 中：闭环组件都可替换，但依赖 VLM judge 稳定切分、每任务多轮 episode 预算、真机人工复位 |
| 主要风险 | 技能级验证依赖 judge 质量与足够候选调用；训练数据全部来自自身执行，学不到从未被部分演示过的技能 |

💡 **X-Ray 开场**

这篇论文要解决的问题是：VLA 策略在训练域里很强，一进真实部署就退化；已有的 agentic 方案让一个 VLM orchestrator 学会「何时调 VLA、怎么下指令、何时改用脚本化技能」，但因为策略权重被冻住，orchestrator 只能绕开失败、不能消除失败。

论文给出的答案是 Robo-COP：让 orchestrator 在部署中把自己的执行切分成技能片段、把成功的片段（哪怕整条 episode 失败）变成 VLA 的训练数据，按需微调，并且把每个新策略当作「候选」来验证——过了才上线，没过就回退，同时修订围绕旧策略形成的记忆。

对 VLA 研究者意味着什么：部署不再只是「推理 + 记忆」，而是一个带版本发布纪律的策略演化过程；评测要从 episode 级下沉到技能级，否则 fallback 会掩盖候选策略的真实退步。

📍 **研究全景时间线**

```
2022  Do As I Can / 语言模型零样本规划器：LLM 写程序调 primitive
  │
2024  OpenVLA / DROID：开源 VLA 与大规模真机数据
  │
2025  π0.5：开放世界泛化的 VLA
  │
2026  RoboLab / CaP-X / ASPIRE：高保真评测、coding-agent harness、技能发现
  │
2026-10  Robo-COP ← 当前位置：把 policy 也拉进部署期更新闭环
  │
局限：脚本化技能、judge、orchestrator 自身推理仍冻结；跨任务迁移未验证
```

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块 | 输入 | 输出 | 是否更新 | 频率/预算（论文 Table 3） |
|------|------|------|----------|--------------------------|
| Orchestrator（VLM coding agent，基于 CaP-X） | 任务目标、当前观测、episode 历史、task memory M | 一段程序，调用脚本化 primitive、感知函数、`execute_vla` | 不更新权重，通过改写 M 适应 | 每 episode 最多 20 turns |
| Scripted primitives | 分割/抓取提议/放置几何 | 关节目标与夹爪命令 | 冻结 | 与 VLA 调用交替 |
| VLA policy π | 语言指令 + 观测 | 动作块，horizon H=15 | LoRA + 部分模块全量微调 | 单次 policy call 有步数上限 |
| VLM monitor | 执行流 | 中断某个 call 或结束 episode | 冻结 | 在线 |
| Text judge | 执行代码、console 输出、turn 结果评估 | 技能指令 + OK/FAILED/UNCLEAR | 冻结，verdict 缓存 | 每 turn |
| Video judge | 6 帧原分辨率画面 + Stage A 指令 | SUCCESS / USEFUL_PARTIAL / NO_PROGRESS / FAILED / UNCLEAR | 冻结 | 每段 |
| Controller | curated 数据、近期表现、M、失败、历史尝试 | TRAIN 或 WAIT；TRAIN 时给最多 4 个目标技能指令 E | 冻结（prompt 调用） | 最多 3 次训练尝试/任务 |
| Verifier | 候选与 incumbent 的技能级调用统计 | adopt / revert / starved | 冻结 | 评估窗口 W 最多 25 episodes |

> 说明：表中「冻结/不冻结」指论文在部署循环中是否更新其参数；controller 与 judge 都是模型调用，但论文不训练它们。

### 1.2 关键机制

1. **动作统一（action unification）**：orchestrator 在一个 episode 里可能先用脚本化 primitive 接近，再调 VLA 下压夹爪。系统按执行顺序记录 commanded joint targets、gripper commands、观测和机器人状态，拼成 VLA 训练格式的连续动作流。这样「脚本动作 + 策略动作」可以一起变成一条带语言标签的演示。
2. **技能级策展，而不是 episode 级过滤**：每个 turn 由 text judge 提出技能指令并判定，再由 video judge 用采样帧确认。只有 text judge 判 OK 且 video judge 判 success 或 useful partial progress 的片段才进入 D。论文 Figure 3 明确展示了一个被 text judge 接受、被 video judge 否决的 grasp——说明单靠文本判定的风险是作者自己先承认的。
3. **训练时机交给 controller**：固定日程的两个问题是「在已经能解的任务上训练只会增加风险」和「数据不够就训练」。controller 看 curated 数据量、近期统计、失败代码与 M，返回 TRAIN/WAIT；TRAIN 时给最多 4 个目标技能指令，用于后续验证。
4. **候选策略验证（policy verification）**：每个候选在 W 个 trial 的窗口里与 incumbent 对照。因为 fallback 可能在候选失败后把任务救回来，所以不能只看 episode 成功率，必须看技能级 finish rate。候选还要检查此前已接纳技能 S 是否退步。
5. **记忆修订**：换策略后，M 中「因为以前 VLA 抓取失败，所以改用脚本抓」这类建议会失效。系统把这类条目标记为「对当前 checkpoint 未验证」，而不是删除；同时把提到候选的引用改成在役策略名。这样历史保留，但计数不再决定 orchestrator 的置信度。

⚡ **Eureka Moment**：真正稀缺的不是「再训一个策略」，而是**把每次微调当成一次带技能级回归检查的候选发布**——训练数据来自部署、上线决定也来自部署，失败可以回退，记忆随 checkpoint 失效。

### 1.3 信息流/架构图

```
                 ┌────────────────────────────────────────────┐
                 │              Orchestrator (VLM)             │
                 │  program: primitives + execute_vla(...)     │
                 └───────┬───────────────────────┬────────────┘
                         │                       │
                 scripted primitives        VLA policy π
                         │                       │
                         └──────────┬────────────┘
                                    ▼
                     unified action stream（逐步记录）
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
             Text judge (Stage A)            VLM monitor（中断）
                    │
             Video judge (Stage B)
                    │
              pass → 片段进 D（技能级演示）
                    │
             Controller: TRAIN / WAIT
                    │
        Fine-tune π' from base checkpoint (LoRA)
                    │
        Verify over W trials: Eq.(1) + 已接纳技能检查
                    │
          adopt → π ← π', S ← S ∪ E
          revert → 保留 π
                    │
          ReviseMemory(M)：标记旧建议、重写引用
```

## 2. 数学核心

📌 **Napkin Formula**（一行抓住本质）：

```
候选策略上线 = 任务成功率不显著掉  AND  每个已接纳技能不显著掉
```

论文的目标不是最大化某个可导损失，而是在部署约束下做「候选策略接纳」的决策。下面两个公式分别对应训练目标与接纳条件。

**训练目标（论文 Eq. 2）**：候选 π' 从预训练 checkpoint 初始化，在 curated 数据集 D 上用策略原本的 flow-matching 目标优化：

```
L(θ) = E_{(X,ℓ)~D} E_t [ L_FM( π_θ(o_t, s_t, ℓ), a_{t:t+H-1} ) ]
H = 15
```

**接纳条件（论文 Eq. 1）**：

```
succ(W) ≥ b − δ_task
r_c(s) ≥ r_inc(s) − δ_skill    ∀ s ∈ S_obs
δ_task = δ_skill = 0.1
```

| 符号 | 含义 |
|------|------|
| W | 候选评估窗口，模拟中最多 25 个 episode，真机实验中缩短为 10 |
| succ(W) | 候选窗口内整个系统的任务成功率（含 fallback 后的结果） |
| b | incumbent 近期任务成功率参照，最多取最近 30 个由 incumbent 服务的 episode |
| δ_task | 任务级容忍下降幅度，论文取 0.1 |
| s | 已接纳技能（admitted skill），S 为已接纳技能集合 |
| r_inc(s), r_c(s) | incumbent 与候选在技能 s 上的每次调用完成率（completion check 触发即算完成） |
| S_obs | 有足够候选调用且有 incumbent 参照的已接纳技能子集 |

**直觉**：任务级指标会被 fallback 污染，所以必须同时看技能级完成率；`δ` 给了容错，但方向是「不能明显更差」。如果候选调用太少无法评估，返回 starved，丢弃候选且不改记忆——这是防止「数据不足就上线」的闸门。

> 符号说明：`succ(W)` 与 `b` 都是系统级统计，不是单个策略的孤立指标；论文明确说 b 度量的是「运行该 checkpoint 的整个系统」。

## 3. 带数字走一遍：玩具例子

> 以下数值为**假设示例**，用于演示 Eq.(1) 的计算方式，不是论文实验数据。论文中的真实数字见 §5。

假设某任务当前 incumbent 的近期任务成功率 `b = 0.68`，δ_task = 0.1，评估窗口 W = 25 episodes。系统已接纳技能集合 S = {「pick up the red block」}，且该技能在候选窗口内有 18 次候选调用、incumbent 参照为 0.80。

**情况 A：候选通过**

```
任务级：succ(W) = 0.60 ≥ 0.68 − 0.10 = 0.58   ✓
技能级：r_c = 0.72 ≥ 0.80 − 0.10 = 0.70        ✓
结论：adopt，π ← π'，并把本次目标技能加入 S
```

**情况 B：候选被回退**

```
任务级：succ(W) = 0.62 ≥ 0.58                  ✓
技能级：r_c = 0.68 < 0.70                      ✗
结论：revert，保留 incumbent，不改策略权重
```

**情况 C：暴露不足**

```
含候选调用且该调用提交了动作的 episode 只有 4 个 < 暴露下限 5 个 episode（门槛按 episode 计，不按候选调用次数计）
结论：starved，丢弃候选，记忆保持原样
```

这个例子说明论文验证逻辑的两个关键点：任务成功率可以通过，但技能级仍可能否决；反过来，任务成功率短暂下降 0.08 也在 δ 容忍范围内。后一点是作者的设计选择——容忍小幅波动以换取更新机会，代价是对真实退步的敏感度降低。

## 4. 工程视角

| 工程问题 | 论文给出的约束 | 含义 |
|----------|----------------|------|
| 单 episode 决策预算 | 20 turns/episode | orchestrator 的 deliberation 被硬上限约束；turn 越多，同样 trial 预算下能跑的 episode 越少 |
| 策略调用粒度 | 动作 horizon H=15 | 每次 policy call 输出 15 步动作块；更长承诺意味反应更粗，更短则调用次数上升 |
| 静止退出 | 16 steps | 手臂不动且仍抓物不算完成放置；这影响技能完成率的判定 |
| 记忆预算 | 8000 字符，固定分区 | 控制 prompt 长度；作者引用 long-context 相关文献说明这是有意的取舍 |
| 训练成本 | LoRA rank 16（语言主干）/ 32（action expert），部分模块全量微调；batch 32；步数 500–10000 | 训练期间任务部署暂停；每个任务最多 3 次训练尝试 |
| 验证窗口 | W ≤ 25 episodes（模拟），W = 10（真机） | 真机 trial 更贵，窗口更短，统计噪声更大 |
| 数据门槛 | 至少 30 个含 kept segment 的 episode，且距上次尝试新增至少 10 个 | 防止过早训练；代价是首次更新被推迟 |

**部署约束**：真机每个任务只有 50 个 learning trials + 20 个 held-out trials，且需要人工复位。这意味着「训练一次 + 验证一个窗口」本身就消耗相当比例的可用 episode；论文把 W 从 25 降到 10 是对这一约束的直接回应。

**吞吐与延迟的取舍**：论文没有报告 wall-clock 数字，但结构上训练会暂停部署；如果要在真实产线用，需要把「暂停」改成「影子候选 + 异步评估」，否则每次更新都有机会成本。

**抖动与判定稳定性**：video judge 只采 6 帧，且明确把「决定性时刻落在帧间」判为 UNCLEAR 而非 SUCCESS。这是保守设计，减少假阳性训练数据，但会增加丢弃率。

## 5. 数据与评测

### 5.1 数据组成

| 数据 | 来源 | 过滤/标注 |
|------|------|-----------|
| 训练集 D | 部署执行的动作流 | 按指令变化、被拒行介入、关节跳变 > 0.3 rad 切段；少于 2 步丢弃；每段补 15 个 held frames |
| 技能标签 ℓ | text judge 生成；VLA 调用保留原指令字符串 | 规范动词 + 参数；函数名不得出现在指令中 |
| 视频确认 | video judge 采 6 帧 | SUCCESS / USEFUL_PARTIAL 才保留 |
| 存储字段 | 每步双视角画面、末端位姿、测得关节状态、命令关节目标（第 8 通道为夹爪）、turn 索引 | 动作转成相对当前关节状态的偏移，夹爪保持绝对 |

### 5.2 评测设置（区分论文实测与条件）

| 项目 | 模拟 | 真机 |
|------|------|------|
| 任务数 | 10 个 RoboLab 任务，覆盖「VLA 可解 / primitive 可解 / 两者都不够」三类 | 3 个 DROID 任务，时限 40s / 60s / 90s |
| 部署 trial | 每任务 100 episodes，三种方法共享前 30 个 | 每任务 50 learning trials |
| Held-out | 每任务 50 个共享初始化 | 每任务 20 个 held-out trials |
| 基线 | π0.5 only、Frozen-policy harness、Fixed schedule（每 30 trials 训练一次、不验证）、ASPIRE（作者按其算法适配到 RoboLab） | Frozen-policy harness（同 harness、同记忆、不更新策略） |
| 主要指标 | held-out success；deployment success；技能级完成率 | 同左 |

**论文实测的关键数字**（均来自论文表格/正文）：

- 模拟 held-out 十任务均值：Frozen 64.8%，Fixed 65.8%，Robo-COP 73.8%。
- 模拟 deployment：Frozen 68.7%，Fixed 70.3%，Robo-COP 73.2%；trial 31–100 分别为 66.4%、68.7%、72.9%。
- 六个发生策略更新的任务上，Robo-COP 相对 Frozen 平均 +11.3 点；四个保留 base policy 的任务上 +5.5 点。
- 12 个候选里 5 个被验证拒绝；7 个任务请求训练，3 个任务 controller 选择等待。
- 500 个 held-out trial 中，Robo-COP 失败 131 次，Frozen 失败 176 次；「grasp or drop」失败为 40 与 68，解释了 45 次差距中的 28 次。
- 真机 held-out 均值：Frozen 38.3%，Robo-COP 50.0%；learning 阶段 36.7% vs 38.0%。
- 真机 60s 任务：验证接纳候选后 held-out 从 35% 到 70%；40s 任务：候选被回退，40% vs 40%；90s 任务：controller 选择等待，两者均 40%。

**必须注意的评测条件**：模拟结果是「单次学习运行 + 共享 held-out 初始化」协议，不是多次随机种子；置信区间是固定任务集上的 beta 后验可信区间。真机每任务只有 50 个 learning trials，W=10，统计功效有限。论文没有报告跨任务迁移或跨机器人平台的结果。

## 6. 能力与失败模式

**能做**：

- 在「VLA 单独可解」的任务上保持接近 π0.5 的水平（论文 Table 1：80.0% vs π0.5 的 82.0%），同时不牺牲其他两类任务。
- 在「两者都不够」的任务上通过微调补技能，论文报告该类 70.0%，比 Frozen 高 16.7 点。
- 在「脚本 primitive 可解」的任务上接近 ASPIRE（72.0% vs 73.5%），说明记忆适配本身有效。
- 拒绝坏候选：作者报告 5/12 候选被拒，且在四个首个候选被回退的任务上，固定日程（部署每个 checkpoint）在其中三个低于 Frozen。

**不能做 / 做不好**：

- 学不到自身从未部分演示过的技能——所有训练样本都来自自身执行，这是作者在 Limitations 中明确写的。
- 在脚本 primitive 已覆盖的任务上不如 ASPIRE 的程序搜索（论文点名 Fill the empty bin 与 Sort blocks by initial stack position）。
- 真机仍需人工复位；每次策略更新都带来策展、训练、评估的额外成本。
- 可扩展性未验证：作者明确说需要在更大实验上确认结论是否成立。

### 6.1 隐含假设

1. **judge 的切分与判定足够可靠**。但论文自己展示了 text judge 接受、video judge 否决的案例，说明单 judge 会引入噪声；两个 judge 都可能把「部分进展」误判为可用数据。
2. **技能级完成率能代表策略能力**。完成率由 completion check 是否触发决定；论文把「手臂停住但仍抓物」排除在完成之外，但更细的失败模式（抓歪、放置偏移）不在公式里。
3. **「已接纳技能」的回归检查覆盖足够**。检查只对 S_obs 生效——有足够候选调用且有 incumbent 参照的技能；调用少的技能会被报告但不能阻止接纳。
4. **记忆修订不会过度保守**。系统把劝阻使用 VLA 的建议标为「未验证」而非删除，依赖后续 episode 重新试探；如果 orchestrator 不再尝试，这些建议可能长期主导行为。
5. **每个任务独立学习**。论文的闭环是单任务部署；跨任务技能池化被列为 future work，不是已证结论。

## 7. 与相关工作对比

| 方法 | 更新什么 | 何时训练 | 候选验证 | 利用失败 episode 中的正确片段 | 能调 VLA |
|------|----------|----------|----------|------------------------------|----------|
| π0.5 only | 无 | 无 | 无 | — | 是（自身） |
| Frozen-policy harness | 仅记忆 M | 无 | 无 | 不用于训练 | 是 |
| Fixed schedule | M + policy | 每 30 trials | 无，直接部署 | 是 | 是 |
| ASPIRE（作者适配版） | 技能库/程序 | 搜索预算内 | 保留已验证程序 | 用于技能发现 | 否（论文称其程序不能调用 VLA） |
| Robo-COP | M + policy | controller 决定 | 技能级 + 任务级 Eq.(1) | 是，技能级保留 | 是 |

**面试 Tip**：如果被问「这篇和普通部署期微调的区别是什么」，一句话答：它把微调当成候选发布——训练前有数据门槛，训练后有技能级回归检查与回退，换模型后还会把围绕旧模型形成的记忆标为失效；这三点合起来才是「co-evolve」，而不是「隔一段时间 fine-tune 一次」。

## 8. 精读建议

- **值得精读原文的人**：
  1. 正在搭 agentic robot harness、需要决定「何时把 VLA 换掉/升级」的系统工程师；
  2. 研究 VLA 部署后持续学习、技能级数据策展与策略验证的研究者；
  3. 要在真实机器人上做长期自治、需要评估「更新成本 vs 收益」的团队。
- **建议章节路径**：先读 §3.2（数据策展）→ §3.3 与 Eq.(1)（验证逻辑）→ §4.2 与 Table 1/6（分 regime 结果与失败分析）→ 最后看 Appendix C/E（切分规则、接纳算术）。Appendix G 的 prompt 原文适合要复现的人。
- **不值得精读的理由**：如果你不做机器人学习、或只关心单一 VLA 的模型结构与预训练数据配比，读摘要与 §1 即可；论文的贡献在部署闭环，不在策略网络本身。

---
[← Back to Theory](./README.md)

**关键引用**：论文链接 https://arxiv.org/abs/2610.09228v1（本次唯一引用来源；论文正文提到的项目页未在本文中作为引用使用）。