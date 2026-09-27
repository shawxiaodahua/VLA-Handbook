# 世界模型真的让机器人更强吗？——预测式具身智能评测基准综述 (Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-27
>
> **论文**: Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence
> **链接**: https://arxiv.org/abs/2609.29669
> **作者**: Gaytri Jena (UC Berkeley), Kapil Wanaskar (SJSU), Vinija Jain (Meta), Aman Chadha (Apple), Vasu Sharma (PocketFM), Amitava Das (BITS Pilani Goa)
> **规模**: 34 页 · 11 图 · 13 表 | 2026-08-30 提交 arXiv (cs.RO)
> **核心定位**: 它不训练任何模型，而是**测绘 160 个机器人评测基准**，指出「直接 VLA 策略（看闭环成功率）」与「世界模型（看开环预测质量）」两条评测轨道从不交汇——因此「世界模型是否让机器人更强」这个问题**当前无法回答**，缺的是测量方法，不是模型本身。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 160 个基准里 138 个 model-agnostic，只有 11 个（7%）真正跑了「VLA vs 世界模型」对照；counterfactual 能力几乎无人测量；只有 4 个基准把预测接到真实执行动作 |
| 適合精讀 | 如果你在设计机器人评测、要写 VLA/世界模型的 evaluation section、或需要一个「benchmark 选型地图」，重点看 §3（taxonomy 四车道）和 §4–§5（对照缺口 + 四指标协议）|
| 可以跳過 | 如果你只关心「哪个世界模型/哪个 VLA 最强」——本文**刻意不给模型排名**（§6 局限一）|
| 落地可行性 | 中（协议可直接借用，但需要自己搭 harness；论文只给 target shape，不给测量结果）|
| 主要風險 | 「对照」定义严格（两分支必须同一协议下跑），放宽阈值会多算一部分为 partial，定性结论稳健但边界计数有弹性 |

💡 **X-Ray 开场**（2-3 句，非专家也能读懂）

机器人学现在有两条几乎不交叉的技术路线：一条是**直接策略**（VLA，看一眼就出手，用闭环任务成功率打分），另一条是**世界模型**（先预测未来画面，再规划动作，用预测/生成质量打分）。问题很自然：**先预测再动作，到底比直接出手强多少？强在哪些能力上？** 这篇综述翻遍 2017–2026 的 160 个基准后发现：**没人真的测过这个问题**——世界模型基准只评预测质量、从不执行，任务成功基准只跑单一策略、从不做家族对照。对 VLA 研究者的意义：下一步真正稀缺的不是新模型，而是**能把两条轨道接到同一个闭环里的评测协议**。

📍 **研究全景时间线**

```
2017-2019         2020-2022              2023              2024                2025                2026
 起步期            多任务/导航扩张        VLA 与WM 浪潮      双轨并行            开环反超闭环        桥接前沿
 2 个基准/年        Meta-World/Habitat     31 个/年          36 个/年(峰值)      开环13 vs 闭环6      ◄── 本文
 ManiSkill 前身     ALFRED/RLBench        LIBERO/VBench     RoboCasa/WorldScore  4 个 bridge 出现     (综述测绘)
                    物理推理(Physion)                        ← 双轨开始分化 →     counterfactual 空缺
                                                                                  ↑ 本文指出：对照缺口(silent cell)
局限：早期基准几乎全 model-agnostic；世界模型评测车道 2023 后才成形；迄今无人做 capability×family 交叉对照
```

## 1. 核心架构/方法总览 (Overview / Architecture)

这不是一篇方法论文，而是一篇**测绘型综述（survey）**。它提出一套**操作性分类法（operational taxonomy）**作为「方法」，把 160 个基准放到四个评测车道 + 十二个能力区里，然后论证一个空缺。

### 1.1 系统对比概览 (System Component Comparison)

作者把整个机器人评测landscape拆成四个「车道（lane）」，按「评分距离真实执行动作有多近」排序：

| 车道 (Lane) | 数量 | 评什么 | 闭环? | 是否建 VLA-vs-WM 对照 | 代表基准 |
|------|------|--------|-------|----------------------|----------|
| §3.1 Policy / manipulation suites | 37 | 闭环任务成功率 | ✅ 是 | 基本无（model-agnostic）| LIBERO, CALVIN, ManiSkill2 |
| §3.2 Embodied agents | 85（最大）| 闭环任务成功率（导航/社交/长程/驾驶/运动）| ✅ 是 | 仅隐藏状态/ToM/counterfactual 子类零星有 | ALFRED, BEHAVIOR-1K, Habitat 3.0 |
| §3.3 World-model evaluation | 34 | 开环预测/生成质量 | ❌ 否（从不执行）| 无 | VBench, EWMBench, Physics-IQ |
| §3.4 Prediction-to-action bridges | 4（前沿）| 世界模型的闭环成功率 | ✅ 是 | 部分（"~"）| WorldSimBench, World-in-World, RoboWM-Bench, WorldArena |

按评测模式分：闭环任务成功 113 个 / 开环预测·生成 43 个 / 桥接 4 个。
按对照轴分：**model-agnostic 138 · partial 11 · explicit 11（7%）**。

### 1.2 关键机制 (Key Mechanism)

为什么这个缺口存在？作者给出两条机制解释：

- **世界模型基准「不闭环」**：它给 rollout 打「真实感/物理合理性」的分就停手，从不执行预测，所以它无法回答「这个预测到底帮没帮机器人做事」。
- **任务成功基准「只跑一条分支」**：它闭环，但只托管**单一策略**，且刻意保持 model-agnostic（不偏袒任何架构）——正是这种中立性使它**天然无法做对照**。
- **「世界模型」是个被滥用的词**（论文 Table 7 的核心洞见）：它同时横跨两个正交轴——
  - **表征轴**：latent-dynamics（LWM，如 DreamerV3/TD-MPC2）/ action-conditioned video（WAM，如 iVideoGPT/GR-2）/ generative video（WFM，如 Sora/Cosmos/Genie）；
  - **角色轴**：作为「神经模拟器（neural simulator）」训练·评测策略，还是作为「被基准评分的对象」。
  - 关键事实：**没有任何一种「世界模型」同时满足「动作条件化 + 闭环 + 可与 VLA 同协议比较」**。latent / action-conditioned 会 act 但很少用标准生成指标评分；generative 有指标但极少闭环。要对照，必须把没有任何单一语义提供的属性拼装起来。

⚡ **Eureka Moment**：**「问题答不出来，不是因为模型不行，而是因为没有基准是为问这个问题而建的」**——评测方法上的空缺（capability × model-family 交叉格），而不是模型能力上的空缺。所以结论不是「世界模型有用/没用」，而是「要回答它，必须先造一个能问它的基准」。

### 1.3 信息流/架构图 (Flow / Diagram)

论文 Figure 10 给出的「隔离预测优势」评测环——这是全文的中心图：

```
                       ┌──────────────────────── 同一个闭环 ────────────────────────┐
   task/scene ──┬──► [ direct VLA policy ] ──────────────────────────► ┌──────────┐
                │          (直接出手)                                    │ 环境执行 │
                │                                                       │ 状态反馈 │
                └──► [ World Model ] ──► predict ──► plan ──────────►  └────┬─────┘
                        (先预测再规划)                                       │
                                                                            ▼
                                                              task success (统一度量)
                                                                            │
                                                                            ▼
                                            Δ_adv sliced per capability β  ← 对照在此读出

  今天两条轨道各自断裂：
  · model-agnostic 策略/具身套件 → 闭环 ✅ 但只跑一条分支 → 无对照 ❌
  · 世界模型评测套件          → 评预测 ✅ 但从不执行     → 无闭环 ❌
  · bridges(4个)             → 世界模型闭环 ✅ 但没做 per-capability 的 VLA 对照 ❌
  ⇒ 没有任何车道完成这张图
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）——本文的「核心方程」其实是**优势的定义式**：

```
Δ_adv(β) = Success_closed(WM-policy | capability β) − Success_closed(matched VLA policy | capability β)

其中：两个分支必须在同一闭环、同一成功度量下运行；结果按能力 β 切片，而非报告单一总数
```

- **目标**：把「预测是否有用」从一句口号变成一个**可测量的量**。
- **变量说明**：
  - `β`（capability）：能力维度，如 hidden-state / counterfactual / long-horizon / social（论文 Table 5 逐能力审计）。
  - `Success_closed(...)`：闭环任务成功率——唯一「正确」的度量，因为只有它反映执行。
  - `matched`：匹配基线。所谓「matched VLA policy」即与 WM 分支同数据同协议的直接策略。
- **直觉**：绝对值没有意义（「某基准 78%」不是优势），**差值才有意义**，而且差值必须**按能力切片**——因为预测在「隐藏状态 / 遮挡 / 长程」处应更值钱，在别处可能毫无收益甚至更差。

> 符号与本文保持一致：论文用 `δ` 表示 advantage-of-prediction，`β` 表示 capability cut，`γ` 表示 capability×family 交叉对照，`ε` 表示基准目录覆盖。本文用 `Δ_adv` 代替 `δ` 以免与其它记号混淆。

作者还给出**四项可操作指标**（§5 / Table 9），本质是把上式拆成四条可落地的测量：

| 指标 | 定义 | 命名 testbed | 针对的轴 |
|------|------|-------------|----------|
| ① Advantage-of-prediction curve | 世界模型策略对 matched VLA 基线的闭环增益，按能力画成**曲线**而非单点 | LIBERO / CALVIN（双家族同 harness）| δ |
| ② Counterfactual accuracy | 在需推理未见/假设动态的 episode 上的成功率 | WorldPrediction + LIBERO-Mem 式遮挡/记忆 split | 最被忽视的能力 |
| ③ Prediction-to-action fidelity gap | 世界模型「开环生成质量」与「闭环成功率」之差 | WorldSimBench / World-in-World | 量化"视觉质量 ≠ 任务成功" |
| ④ Per-capability contrast coverage | 真正跑了 VLA-vs-WM head-to-head 的能力占比 | landscape audit（对照 Table 10 声明 schema）| γ |

**关键纪律**：四项**都必须以分布/曲线报告，绝不报告单一 headline 数字**——把「prediction helps」改写成「prediction helps *here*, by *this much*」。

## 3. 带数字走一遍：玩具例子 (Worked Example)

假设你按指标 ① 搭了一个双分支 harness（下面数字为**假设的合理值**，用于演示计算闭环，非论文实测）：

```
场景：LIBERO 上三类能力，各跑 100 episode
                     direct VLA    WM policy     Δ_adv = WM − VLA
  hidden-state 类       42%           61%          +19 pt  ← 预测在遮挡下最值钱
  long-horizon 类       55%           58%           +3 pt  ← 收益微弱
  short-horizon 类       78%           77%           −1 pt  ← 预测反而略拖累

  若直接报告「WM 平均 65% vs VLA 58%」→ +7 pt，会掩盖「一类大赢、一类打平、一类小输」
  ✅ 正确报告：按能力切片的三点曲线 + 每点配 matched 基线（本例基线已列出）
```

再看指标 ③（prediction-to-action gap，同样是演示值）：

```
某世界模型：开环生成质量 = 0.85（VBench 风格高分）
            闭环任务成功率 = 0.34
            fidelity gap = 0.85 − 0.34 = 0.51   ← 巨大鸿沟
解读：生成得越漂亮，越要警惕——高分只证明"画得像"，不证明"能干活"。
      这个 gap 越大，说明开环指标对闭环能力的代理性越差。
```

指标 ④ 则是**审计**而非实验：把 160 个基准对齐一张「能力 × 家族」表，数一数有多少格真的跑了对照。论文审计结果：**explicit contrast 只有 11/160 = 7%**，counterfactual 能力那几格几乎全空。

## 4. 工程视角 (Engineering View)

- **闭环采样成本**：闭环任务成功评测天然昂贵——每个 episode 都要真实执行。开环评测便宜（只生成不执行），这正是「世界模型评测车道」扎堆开环指标的一个**隐性工程动机**：闭环对照贵，所以没人做。
- **matched baseline 的工程含义**：要「匹配」，两个分支必须共享数据、观测接口、动作空间与评测 harness。这意味着工程上需要**一个能插拔 policy 的 harness**（论文反复强调 "under one protocol / one harness"）——这不是加个 flag，而是接口标准化工作。
- **能力切片 vs 单一数字**：从系统设计看，按能力切片要求 harness 能对任务做**可控扰动**（遮挡、延迟、反事实替换），即需要可控 episode 生成器，而不只是固定任务集。
- **报告纪律的工程含义**：指标必须输出**曲线/分布**，意味着日志与统计层要保留 per-episode、per-capability 的原始记录，不能只存一个 success rate。对控制频率/延迟的直接影响：预测-规划分支比直接策略多一步 inference+planning，闭环时延更高，收益必须能抵消这部分开销才有意义（论文未量化，属待补）。

## 5. 数据与评测 (Data & Eval)

综述本身的「数据集」是**基准语料库**，采样本着 PRISMA 2020 风格：

- **语料规模**：160 个 web-verified 基准（155 个有 arXiv ID，5 个没有）+ 8 篇最接近的综述 = 164 篇参考文献。
- **筛选口径**：单位是「benchmark」（可单独引用、含评分协议的评测物）；**刻意排除**三类易混物——单个模型/策略、无评分协议的裸数据集、无任务分布的纯模拟器。凡是「既发模型又发基准」的工作，只取其基准入榜。
- **验证强度**：每个候选都 web-verify 过 arXiv/venue 页面的标题、一作、年份、编号；验证不过的直接丢弃（**不凭记忆记录**）；按 citation key 与 arXiv ID 双向去重。Figure 3 记录了 PRISMA 漏斗，未记录计数的阶段明确标注「未记录」而非估算。
- **核心分布**：
  - 车道：policy 37 / embodied 85 / wm-eval 34 / bridge 4
  - 能力：long-horizon planning 26、manipulation & dexterity 22、generation quality 19、navigation 16、physical reasoning 13、social & multi-agent 12、hidden-state & ToM 9、counterfactual 9、autonomous driving 9、generalization 7、locomotion 6、other 12
  - 年份：2017 起 2 个/年 → 2024 峰值 36 个 → 2025 年**开环评测首次反超闭环（13 vs 6）**，标志世界模型生成文献的涌入。
- **能力深挖（Table 5）**：走对照路线最可能的是 hidden-state（3/4）与 ToM（3/4）；而应该最需要预测的 **counterfactual 只有 3/6、long-horizon 仅 1/6、social 是 0/5**——「最该用预测的地方，恰恰最少被测」。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**这份综述能做什么**：
- 给出一张**可导航的基准地图**（四车道 × 十二能力），并配 placements 的书面规则（Table 2），边界案例靠规则而非品味判定。
- 提供**唯一一个跨 capability × model-family 的 coverage 对照**（对比 8 篇近邻综述，Table 1）；其余综述要么只铺目录（ε），要么只沾 α/β。
- 给出**可借用的评测协议**（四指标）与**隔离预测优势的闭环图**。

**它不能做什么 / 失败模式**：
- ❌ **不给模型排名**：单位是基准不是模型，想知道「谁最强」的读者找不到答案（作者明说 by design）。
- ❌ **不给实测结果**：四指标图（Figure 11）是 **target shape（目标形态）而非测量结果**——它是一个协议，不是 leaderboard。
- ⚠️ **对照计数对定义敏感**：严格定义（两分支须同协议）会把一些基准判为 model-agnostic；放宽则会多判 partial。作者承认定性图景稳健，但**具体计数有弹性**。
- ⚠️ **覆盖偏向**：偏近期、偏 arXiv 可索引、偏英文、偏 web search 可达的工作；两本图册只收「能找到且可验证来源」的图，**缺席不等于不存在**。
- ⚠️ **优势度量未验证**：所讨论的 advantage-of-prediction 工具（如 ACTION-ATLAS 的 World Advantage Score）仅出自 position paper，本文**只当作候选评估、不采纳也不据此排名**。

### 6.1 隐含假设 (Hidden Assumptions)

作者默认成立但**未明确验证**的前提：

1. **「闭环成功率是唯一正确的度量」**——但闭环评测成本极高、复现难（sim-to-real、硬件差异），隐含假设是其收益 > 成本；论文未量化这一取舍。
2. **「两个分支可以做到 matched」**——假设 WM 与 VLA 能在同数据同 harness 下公平对齐；但实际上两者输入（观测 vs 观测+未来预测）、算力预算、推理步数都不同，如何定义「匹配基线」本身是开放问题。
3. **「能力 β 可被干净地切片」**——假设 hidden-state/counterfactual 等能力可被隔离成独立 episode 类型；但这些能力在真实任务中常常耦合。
4. **「对照缺口是主要瓶颈」**——假设只要造出对照基准，问题就能答；但如果世界模型的闭环收益本身就高度场景依赖，对照也可能给出「因场景而异」的碎片化结论，而非一个可推广的答案。
5. **benchmark 作为分析单位的完备性**——假设 160 个基准已覆盖 landscape 全貌；但搜索方法论偏 web-search 可达，闭源/工业界内部评测很可能缺席。

## 7. 与相关工作对比 (Comparison)

论文 Table 1 把 8 篇最接近的综述/position paper 放在五个轴上比较（α 评测模式 / β 能力切 / γ 能力×家族交叉 / δ 优势度量 / ε 基准目录）：

| 工作 | 年份 | 类型 | α 模式 | β 能力 | γ 能力×家族 | δ 优势度量 | ε 目录 |
|------|------|------|--------|--------|-------------|-----------|--------|
| RL Reproducibility (Lynnerup 2020) | 2020 | 综述 (CoRL) | 部分 | ✘ | ✘ | ✘ | 部分 |
| Embodied AI Simulators (Duan 2022) | 2022 | 综述 (IEEE TETCI) | 部分 | 部分 | ✘ | ✘ | ✅ |
| Eval. of Embodied AI (Hou 2026a) | 2026 | preprint | ✅ | 部分 | ✘ | ✘ | ✅ |
| VLA Datasets (Wang 2026) | 2026 | preprint | 部分 | ✘ | ✘ | ✘ | ✅ |
| WM for Robots (Hou 2026b) | 2026 | preprint | ✅ | 部分 | ✘ | ✘ | ✅ |
| Embodied WM (Shang 2026b) | 2026 | preprint | ✅ | 部分 | ✘ | ✘ | ✅ |
| Bench. Constr. (Lai 2026) | 2026 | preprint | 部分 | ✘ | ✘ | ✘ | 部分 |
| WM Eval. (pos.) (Yu 2026) | 2026 | **position** | ✅ | 部分 | ✘ | 部分（唯一）| 部分 |
| **Ours** | 2026 | **survey** | ✅ | ✅ | **✅（唯一）** | 评估不采纳 | ✅ |

**与基线对比的要点**：其它工作把 ε（目录）填得很满，α/β 顺带沾一点；但 **γ（能力×家族交叉）在所有前作中都是空的**，δ（优势度量）只有 Yu 2026 那篇 position paper 部分触及，**从未被任何综述达到**。本文是该格（"empty cell"）的填补者。

**面试 Tip**：被问到「世界模型值不值得用」时，标准答法不是站队，而是：「当前没有基准能回答它——评测缺口是 capability×model-family 交叉。要回答得先建一个在同一闭环里跑 VLA 分支和世界模型分支、并按能力报 Δ_adv 的协议。」这句话同时展示你懂方法、懂评测、且不轻信高开环分数。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. **做机器人评测/benchmark 的研究者**——§3 taxonomy 与 Table 2 的 placement 规则可直接当设计 checklist。
  2. **要在论文里写 evaluation 或 related work 的 VLA/世界模型作者**——Table 1 与 Table 8 帮你精确定位自己的贡献落在哪个"格"。
  3. **评估「世界模型迁移到新平台可行性」的工程師**——§4/§5 的闭环对照逻辑与四指标是评估方案模板。
- **建議章節路徑**：先读 §1 Introduction + Figure 1/2（拿到全图）→ 再看 §4（对照缺口的形式化）与 §5（四指标协议）→ 需要选型时查 §3 各车道表格与附录 Table 10（160 基准全表）→ §2（PRISMA 方法论）和 §6 FAQ/图册可跳。
- **不值得精讀的理由**：如果你不做机器人学习、或只想知道「哪个世界模型最强」、或已熟悉 VLA/世界模型评测现状——读摘要 + §4 即可，全文 34 页的目录细节对你不产生增量。作者本人也说：本文**刻意不回答模型强弱**。

---
[← Back to Theory](./README.md)

**关键引用**
- 论文主页：https://arxiv.org/abs/2609.29669 · HTML: https://arxiv.org/html/2609.29669v1
- 核心工作引用：LIBERO (Liu 2023)、CALVIN (Mees 2021)、WorldPrediction (Chen 2025)、WorldSimBench (Qin 2024)、World-in-World (Zhang 2025)、RoboWM-Bench (Jiang 2026)、WorldArena (Shang 2026a)
- 唯一触及优势度量的前作（position paper）：Yu et al. 2026, World Advantage Score (ACTION-ATLAS)
