# MobiAgent：长时程移动操作的双环递归策略自改进 (MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-06
>
> **论文**: MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation (CoRL 2026)
> **链接**: https://arxiv.org/abs/2610.03476
> **项目页**: https://kaiknower.github.io/mobiagent · **代码**: https://github.com/kaiknower/MobiAgent
> **核心定位**: 用「原子技能 + 共享 VLM 主干 + 技能专属 action expert」的 Inner Loop 解决长时程移动操作的误差累积与容量干扰，再用「VLM 自动切分回放 + 聚类 + 微调」的 Outer Loop 把部署数据零标注地回收成新策略——把单体 VLA 从「反应式短时程」推进到「可自进化的分级长时程」。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 把「子任务→专属模型」的僵化分层，换成「原子技能→可复用技能专家」，再叠一个 critic 驱动的重规划 + 部署数据自动回收闭环，长时程成功率大幅提升（BEHAVIOR-1K 65.0% vs π0.5-TA 42.5%）|
| 適合精讀 | 如果你在做多模态具身 Agent、长时程移动操作，或任何需要「部署数据自动变训练数据」的闭环系统，重点看 §3.1（Inner Loop）与 §3.2（Outer Loop）|
| 可以跳過 | 如果你只关心单臂桌面级短时程 VLA 的架构细节，这篇距离中等——本文的核心价值在系统编排而非单点模型 |
| 落地可行性 | 中（模块多数是现成 LLM/VLM + π0.5 微调，工程拼装可行；但依赖强 VLM、多轮真机 rollout 与算力）|
| 主要風險 | 新技能加入共享主干会干扰旧行为（作者自己承认）；critic 在遮挡下会漂移；真机绝对成功率仍偏低（57.5%）|

💡 **X-Ray 开场**
这篇论文解决什么问题？移动机器人做「跨房间、多阶段」的家务（如把垃圾捡起来、走到垃圾桶、丢进去）时，单体 VLA 会误差累积、且「走路」和「抓取」两种截然不同的运动映射挤在一个模型里互相干扰。
它发现了什么？把任务拆成可复用的「原子技能」（move_to / pick_up / place_in …），让每个技能共享同一个 VLM 主干、只挂各自独立的 flow-matching action head，可以既复用又隔离；再让一个 VLM critic 每执行若干动作块就做视觉验收并触发重规划，能救回「物体掉落」这类失败。
对 VLA 研究者意味着什么？长时程可靠性可能不来自更大的端到端模型，而来自「分级编排 + 技能隔离 + 自动数据回收」这套系统级设计。

📍 **研究全景时间线**

```
2023  RT-2 / VoxPoser 语言驱动分层雏形
  │         （plan-execute-verify，但技能库固定、无法学连续动作）
2024  OpenVLA / Octo 开源短时程 VLA
  │
2025  π0 / π0.5 flow-matching 连续动作；Mobi-π、N2M 把 VLA 搬到移动平台
  │         （仍偏反应式，缺长时程上下文）
2025  π0.5-TA：BEHAVIOR Challenge 冠军，但每任务单独训练 → 容量干扰
  │
2026  RoboClaw 自动数据采集（但需成对 forward/reset 策略）
  │
[本文] MobiAgent 双环：原子技能可复用 + critic 重规划 + 零标注自改进   ← 当前位置
  │
局限  新技能干扰旧行为 / 遮挡下 critic 漂移 / 泛化到新本体待验证
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

MobiAgent 用「双环」把**部署时执行**与**离线训练**解耦：

- **Inner Loop（部署，在线）**：Task Planner → Skill Executor → Reflection Critic → Orchestrator，反复「规划下一子任务 → 执行一段技能动作块 → 视觉验收」。
- **Outer Loop（离线，训练）**：Data Curator → Skill Generator → Skill Trainer，把回放轨迹自动切成技能片段、聚成技能词表、微调策略库。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 载体/模型 | 输入 | 输出 | 频率/时序 | 训练 or 推理 |
|------|-----------|------|------|-----------|--------------|
| Task Planner (Π_Plan) | VLM（默认 GPT-5.4） | 全局指令 ℓ、当前观测 o_k、episode 记忆 H_k、技能目录 S | 子任务指令 ℓ_k、技能提示 s_k | 事件触发（技能完成或失败）在决策步 k | 推理（zero-shot prompt） |
| Skill Executor (Π_Exec) | π0.5 VLA：共享 trunk π_vlm + 每技能 flow-matching head π_action^(s_k) | 子任务指令 ℓ_k、多模态观测 o_t | 长度 L 的动作块 a_{t:t+L} | 低层控制步 t，块式输出 | 训练（BC 微调） + 推理 |
| Reflection Critic (Π_Critic) | VLM（默认 GPT-5.4） | ℓ_k、抽帧观测窗 x_t | 判决 v_t、建议 φ_t、证据 c_t | 每执行 C 个动作块后（共 C·L 个动作） | 推理 |
| Orchestrator (Π_Orch) | 确定性状态机 | v_t、φ_t、计数 n_t、上限 n_max | advance / retry / replan | 每次 critic 输出后 | 无训练（规则） |
| Data Curator | VLM（Gemini-3.1-Pro，原生视频输入） | 原始回放视频（拼接头视+腕视） | 带指令描述的技能片段 + 验收 | 离线，逐 episode | 推理 |
| Skill Generator | LLM（默认 GPT-5.4） | 片段指令集合 | 规范化的原子技能词表 | 离线，一次性聚类 + 后续归类 | 推理 |
| Skill Trainer | 训练管线 | 按技能聚好的片段数据 | 新/更新的技能策略 | 离线，逐轮 | 训练（BC） |

> 注：默认基座模型信息来自论文 Appendix A / Table S1（GPT-5.4 用于 Planner、Critic、Skill Generator；Gemini-3.1-Pro 用于 Data Curator 切分，因为需要原生连续视频输入）。

### 1.2 关键机制 (Key Mechanism)

- **可组合原子技能 + 递减视界（receding-horizon）规划**：Planner 每次只吐「下一个」原子子任务，而不是一次性生成长计划，规避了「预定义子任务清单」的可扩展性瓶颈，也天然支持「完成或失败后重新规划」。
- **技能专属 action expert + 共享 VLM 主干**：把 locomotion 与 arm control 分到不同 expert，缓解「一个预测头同时建模迥异感知-运动映射」的容量干扰；同时共享 trunk 保证可复用性与优化效率。
- **critic 驱动的最细粒度验收**：在**原子技能**级别做视觉验收（而非任务级），才能显式处理空间推理（导航到位、距离/视角调整），这是移动操作里恢复失败的关键。
- **确定性编排 + 合法 schema**：critic 输出被严格约束到合法组合（如 complete ⇒ next；error ⇒ replan），任何违规被拒绝并把 v_t 设为 error，保证状态机可控。
- **零标注数据回收**：Outer Loop 用 VLM 自动「切分 + 验收」回放，聚类成技能，再微调；部署中获得的失败/成功轨迹都变成训练燃料。

⚡ **Eureka Moment**：THE 关键洞见是——**长时程可靠性不靠一个更大的端到端策略，而靠「把任务切到原子技能粒度、让技能共享主干但动作头隔离、并在技能粒度上闭环验收与重规划」**；再配一个自动数据回收闭环，让系统边部署边变强。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                     ┌──────────────────── Inner Loop (部署, 在线) ────────────────────┐
  全局指令 ℓ ──────► │  Task Planner Π_Plan                                            │
                     │   输入: (ℓ, o_k, H_k, S)  →  输出: (ℓ_k, s_k)                    │
                     │        │                                                        │
                     │        ▼   s_k 选中对应技能专家                                   │
                     │  Skill Executor Π_Exec                                          │
                     │   a_{t:t+L} = π_action^(s_k)( π_vlm(ℓ_k, o_t) )                 │
                     │        │  执行 C 个动作块 (C·L 步)                                 │
                     │        ▼                                                        │
                     │  Reflection Critic Π_Critic                                     │
                     │   (v_t, φ_t, c_t) = Π_Critic(ℓ_k, x_t)                          │
                     │        │   c_t 追加进 episode 记忆 H_k ──┐                        │
                     │        ▼                                  │                        │
                     │  Orchestrator Π_Orch (确定性状态机)  ◄────┘                        │
                     │   advance → 交给 Planner 规划下一步                              │
                     │   retry   → 继续执行当前技能                                     │
                     │   replan  → 停低层执行，回到 Planner（用 c_t 做空间感知重规划）    │
                     └───────────────┬───────────────────────────────────────────────┘
                                     │ 回放轨迹 (rollouts)
                                     ▼
                     ┌──────────────── Outer Loop (离线, 训练) ───────────────────────┐
                     │  Data Curator   → VLM 切分 + 验收 → 带描述的原子技能片段         │
                     │  Skill Generator→ LLM 语义聚类 + 命名 → 规范原子技能词表          │
                     │  Skill Trainer  → 按技能簇 BC 训练 / 持续微调 → 更新技能库        │
                     └───────────────────────────────────────────────────────────────┘
                                     │ 更新后的技能专家 (回到 Inner Loop)
                                     └──────────────────────────────►
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
技能专家 = 共享躯干 ∘ 专属动作头：a = π_action^(s_k)( π_vlm(ℓ_k, o_t) )
```

**目标**：给定全局语言指令 ℓ，逐步输出连续动作，使机器人在长时程多阶段任务中成功；核心是让「高层语义推理」与「低层感知-运动控制」解耦，且技能可复用、失败可恢复、数据可自改进。

**1) 规划器动态组合原子技能**（论文 §3.1.1）：

```
ℓ_k, s_k = Π_Plan( ℓ, o_k, H_k, S )
```

变量说明：

| 符号 | 含义 |
|------|------|
| ℓ | 全局任务指令（natural language） |
| o_k | 当前多相机观测 |
| H_k | episode 记忆（历史子任务、验收判决、critic 给出的空间线索） |
| S | 原子技能目录（open-vocabulary 注册表） |
| ℓ_k | 立即子任务的紧凑自然语言指令（如 "pick the red can"） |
| s_k ∈ S | 技能提示（如 pick），作为 Executor 选择策略的接口 |

**2) 技能执行器 = 共享 VLM 主干 ∘ 技能专属动作头**（论文 §3.1.2）：

```
a_{t:t+L} = Π_Exec^(s_k)( ℓ_k, o_t ) = π_action^(s_k)( π_vlm( ℓ_k, o_t ) )
```

直觉：π_vlm 是所有技能共享的「理解层」，π_action^(s_k) 是按技能选择的「动作生成层」。语义切分让移动与手臂控制各用各的 head，避免容量干扰；共享 trunk 保证复用。

**3) 反思 critic 的结构化判决**（论文 §3.1.3）：

```
(v_t, φ_t, c_t) = Π_Critic( ℓ_k, x_t ),   x_t = ( o_{t-2C·L}, o_{t-C·L}, o_t )
```

其中判决 v_t ∈ {complete, incomplete, error}，建议 φ_t ∈ {next, retry, replan}，c_t 是可观测线索（供离线审计）。合法组合被硬约束：

```
v_t = complete   ⇒  φ_t = next
v_t = incomplete ⇒  φ_t ∈ { retry, replan }
v_t = error      ⇒  φ_t = replan
（任何违规 → 拒绝，并把 v_t 置为 error）
```

**4) 确定性编排状态机**（论文 §3.1.3）：

```
Π_Orch( v_t, φ_t, n_t ) :

  advance   若 v_t = complete
  replan    若 v_t = error  或  φ_t = replan
  retry     若 v_t = incomplete 且 φ_t = retry 且 n_t <  n_max
  replan    若 v_t = incomplete 且 φ_t = retry 且 n_t ≥  n_max
```

直觉：把「重试预算」和「任务终止」从 LLM 手里收回，交给确定性规则——LLM 只负责「判断」，规则负责「控制」。这是长时程系统里抑制抖动的重要工程手法。

> 符号与本文/相关文档保持一致：π_vlm=共享视觉语言主干；π_action^(s_k)=技能 s_k 专属的 flow-matching 动作专家；L=动作块长度；C=每执行多少个动作块做一次 critic 验收。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

设想一个极简 2 阶段任务：**把桌上的罐子丢进垃圾桶**，设 L=2（每块 2 个动作）、C=1（每块后都验收）。

```
初始: ℓ = "dispose of the can",  o_0 = 场景含罐子 + 垃圾桶,  H_0 = []

步 k=0: Π_Plan(ℓ, o_0, H_0, S) → (ℓ_0="pick up the can", s_0=pick)
        Π_Exec^(pick)(ℓ_0, o_0) → a_{0:2}   (抓取动作块)
        执行后 Π_Critic(ℓ_0, x) → v=complete, φ=next, c=["can moves with gripper"]
        Π_Orch → advance

步 k=1: Π_Plan → (ℓ_1="move to the trash bin", s_1=move_to)
        Π_Exec^(move_to) → a_{2:4}  (底座导航)
        Π_Critic → v=complete, φ=next → advance

步 k=2: Π_Plan → (ℓ_2="place the can in the trash bin", s_2=place_in)
        Π_Exec^(place_in) → a_{4:6}  (放置动作块)
        —— 放置失败：罐子掉在桶外 ——
        Π_Critic → v=incomplete, φ=replan, c=["can outside trash boundary"]
        H_3 += c    (空间线索写入记忆)
        Π_Orch → replan

步 k=3: Π_Plan(ℓ, o_3, H_3, S) 依据「罐子现在在桶外」→ (ℓ_3="move to the displaced can", s=move_to)
        重新靠近 → 重抓 → 再放置 … 直到 v=complete
```

这个闭环说明三点：(a) critic 在**技能粒度**发现失败；(b) 失败被写成**可读线索**传给 planner；(c) 重规划不是「重放同一技能」，而是根据失败后的真实场景生成**新步骤序列**。论文也印证这一点：在 BEHAVIOR-1K 上 39% 的成功 episode 至少需要一次 replan，成功 episode 平均 0.95 次 replan（重规划预算上限为 3）。

## 4. 工程視角 (Engineering View)

| 维度 | 设计与权衡 |
|------|-----------|
| 控制频率 | 真机 Astribot S1 多视角相机 + 本体感知以 30 Hz 运行；动作以动作块（chunk，长度 L）输出，降低高层推理频率 |
| 推理分层 | Planner / Critic 是 VLM 调用（昂贵、事件触发）；Executor 是 VLA 前向（高频）。二者解耦避免每步都调用大模型 |
| 验收频率 | Critic 每 C 个动作块（C·L 个动作）才跑一次——C 越大越省算力但误差累积风险越高，是核心超参 |
| 状态机确定性 | Orchestrator 用硬规则而非 LLM 决定 advance/retry/replan，且 schema 违规强制置 error，避免不确定行为累积 |
| 训练成本 | 共享 VLM trunk（π0.5 初始化）+ 每技能独立 action expert；技能数量由 LLM 聚类自适应，不预设 |
| 数据成本 | 启动阶段：仿真用合成轨迹 / 真机用人类遥操作示范（真机每任务 100 段示范）；后续每轮自动回收约 20 段/任务，**零人工标注** |
| 部署约束 | 依赖强 VLM（GPT-5.4 / Gemini-3.1-Pro）做规划、验收、切分——离线成本与 API 依赖是现实约束 |
| 抖动问题 | 核心稳定性来自「技能粒度验收 + 重试预算上限 + 确定性编排」三件套；但论文仍报告 pose-correction churn（姿态校正抖动） |

工程含义：这套设计的本质是**把不确定性集中到「可验收、可回退」的技能边界上**，而不是指望底层策略零失误。对部署的启示是——模块边界（planner/executor/critic）与验收频率（C、L）比单点模型精度更影响长时程鲁棒性。

## 5. 數據與評測 (Data & Eval)

**评测设置**（论文 §4.1）：

| 平台 | 环境 | 机器人 | 设置 |
|------|------|--------|------|
| RoboCasa365 Composite-Seen | 仿真 | Franka 臂 + Omron 移动底座 | 16 个家庭任务，每个含 2–15 个子任务；每场景 5 次试验；评估「部署→训练」多轮改进 |
| BEHAVIOR-1K | 仿真 (OmniGibson) | Galaxea R1 Pro | 4 个长时程任务（push-radio / dispose-trash / stack-storage / fetch-beer），每任务 10 次试验 |
| 真机 | Astribot S1 | 双臂移动操作机器人，30 Hz | 4 个任务（trash-garbage / trash-bottle / trash-can / pour-blue），每任务 100 段示范 |

**主要结果**：

| 基准 | 基线 | MobiAgent | 差距 |
|------|------|-----------|------|
| RoboCasa Composite-Seen | CaP-X 5.00% / π0.5 7.50% | R1 18.75% → R3 27.50%（R4–6 稳定 25–27.5%） | +约 20 pp vs π0.5 |
| BEHAVIOR-1K（均值） | π0.5-TA 42.5% | 65.0% | **+22.5 pp**；fetch-beer（最长任务）增益达 50 点 |
| 真机 Astribot S1（均值） | π0.5 10.0% | bootstrap 32.5% → 40.0% → 57.5% | +47.5 pp vs π0.5 |

**消融（BEHAVIOR-1K，论文 Table 4）**：

| 变体 | 均值成功率 | 说明 |
|------|-----------|------|
| 完整 MobiAgent | 65.0% | — |
| 换成固定子任务 + 每任务单独模型 | 20.0% | 证明「任务无关原子技能可复用」的价值 |
| 用单个 action head 替换技能专属 expert | 42.5% | 证明技能隔离缓解容量干扰 |
| 移除 Reflection Critic | 10.0% | **降幅最大**：没有视觉验收，错误会向后传播 |
| 保留验收但禁用全局重规划 | 40.0% | 只剩本技能 retry；dispose-trash 上从 60% 掉到 40% |

> 一句话读法：critic 与全局重规划是这套系统的命门；技能隔离与技能复用是性能的另一半。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 长时程多阶段移动操作（跨房间运输、堆叠、反复取物），在仿真与真机上都能规划-执行-验收闭环。
- 从执行失败中恢复：物体掉落 → 生成新的取回步骤；抓取失败 → 先校正底座姿态再重试（论文 Figure 4）。
- 部署数据自动回收：每轮约 20 段/任务回放就能持续提升成功率（真机 32.5%→57.5%），无需人工标注。

**不能 / 受限**（论文 §5 Limitations + Appendix E）：
- **技能扩展会干扰旧行为**：向共享主干加入新技能可能破坏已学行为（未用 LoRA/持续学习缓解，作者仅建议）。
- **姿态校正抖动（pose-correction churn）**、**放置恢复不准**、**遮挡下 critic 漂移**（transient visual occlusion）仍存在。
- 失败模式（Appendix E）：FM1 critic 假阳性；FM2 双臂协调失败；FM3 技能间过渡/姿态不匹配；FM4 低层感知极限。
- 真机绝对成功率仍偏低（最高 57.5%），离「可用级」还有距离。
- 泛化未充分验证：仅在 RoboCasa / BEHAVIOR-1K / Astribot S1 上评测，声称更广任务族/环境/本体需要进一步验证。

### 6.1 隐含假设 (Hidden Assumptions)

- **假设强 VLM 可用且稳定**：Planner/Critic/Curator 全靠 GPT-5.4 或 Gemini-3.1-Pro；论文未讨论这些模型失效或 API 不可用时的降级。
- **假设技能可被清晰切分且原子化**：Data Curator 的「segment-and-verify」依赖 VLM 能稳定识别 TRAVEL / STATE CHANGE 边界，边界错分会污染训练数据。
- **假设 critic 的视觉判断可靠**：critic 是系统命门（移除后掉到 10%），但其假阳性/遮挡漂移已被列为失败模式——即系统对 critic 的信任本身是个未经充分验证的假设。
- **假设部署回放中的成功轨迹能代表分布**：自改进主要回收「成功 rollout」，可能放大已会行为的偏差，对长尾失败覆盖有限。
- **假设原子技能词表动态稳定**：技能聚类/命名由 LLM 决定，「最优簇数」无先验，稳定性未量化。

## 7. 與相關工作對比 (Comparison)

| 工作 | 关注点 | 架构 | 训练方式 | 适用场景 |
|------|--------|------|----------|----------|
| RT-2 / OpenVLA | 短时程桌面 | 单体 VLM → 离散动作 token | 大规模模仿 | 短时程、单臂 |
| π0 / π0.5 | 短时程通用操作 | flow-matching 连续动作 | 预训练 + 微调 | 桌面级 |
| π0.5-TA（2025 BEHAVIOR 冠军） | 长时程但每任务单独训练 | 单体 + 时间索引阶段共享一个预测头 | 每任务单独训练 | 特定任务，容量干扰 |
| Mobi-π / N2M | 移动操作交接 | VLA 扩展到移动平台 | 模仿 | 导航-操作 hand-off |
| CaP-X | 组合预定义原语 | 编码化原语 | 固定技能库 | 无法学连续感知-运动 |
| RoboClaw | 自动数据采集 + 分层 | agentic | 自动采集（但需成对 forward/reset 策略） | 长时程 |
| **MobiAgent（本文）** | 长时程移动操作 + 自改进 | 双环：原子技能 + 共享 VLM 主干 + 技能专属 expert + critic 重规划 + 自动数据回收 | 启动 BC + 部署数据持续微调 | 长时程移动操作，跨任务复用技能 |

差异一句话：**别人在用「更大的端到端模型」或「固定技能库」，MobiAgent 在用「可组合原子技能 + 技能粒度闭环验收 + 自动数据回收」**。

🎯 **面试 Tip**：被问到时这样答——「MobiAgent 的关键不是某个模型，而是把长时程拆到原子技能粒度，用共享 VLM 主干 + 独立 action head 同时拿到复用与隔离，再用一个 critic 做技能级验收和重规划；它的 Outer Loop 让部署数据零标注地回流成新策略。最硬的证据是消融：去掉 critic 从 65% 掉到 10%，说明闭环验收才是长时程的命门。」

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做多模态具身 Agent、长时程 mobile manipulation 的研究者——§3.1/§3.2 是最有价值的设计参考。
  2. 要评估「部署数据自动回收 / 自改进闭环」落地可行性的工程师——§3.2 与 Appendix D（wall-clock 成本）值得逐节看。
  3. 想借鉴「技能隔离 vs 容量干扰」张力的人——§4.2 消融表与 §5 局限性必读。
- **建議章節路徑**：先讀 §1 与 Figure 1/2（建立双环直觉）→ 再看 §3.1 的 Inner Loop 三件套与 §3.1.3 的确定性编排 → 然后 §3.2 Outer Loop 三阶段 → 接着 §4.2 消融（看系统的命门在哪）→ 可跳 §4.1 的重复性数字罗列（看表即可）→ 最后读 §5 Limitations 与 Appendix E 失败模式。
- **不值得精讀的理由**：如果你不做机器人学习，或已熟悉「分层 plan-execute-verify + 技能库复用」这套范式，读摘要 + 消融表即可——本文的方法论增量更多在系统编排而非全新数学。

---
[← Back to Theory](./README.md)

**关键引用**：
- 论文 (arXiv): https://arxiv.org/abs/2610.03476
- 项目页: https://kaiknower.github.io/mobiagent
- 代码: https://github.com/kaiknower/MobiAgent
- π0.5（基座 VLA）: Physical Intelligence, CoRL 2025
- π0.5-TA（对比基线，2025 BEHAVIOR Challenge 冠军）: arXiv:2512.06951
- BEHAVIOR-1K: arXiv / CoRL 2022；RoboCasa365: ICLR 2026
