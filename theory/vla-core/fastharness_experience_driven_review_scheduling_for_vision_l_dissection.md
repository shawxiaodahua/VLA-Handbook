# FastHarness：用经验调度 GPT-6 复核，降低 VLA 执行监控成本
(FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-25
>
> **论文**: FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies
> **链接**: https://arxiv.org/abs/2610.31427
> **核心定位**: 不训练新 VLA，而是在冻结的 `π0.5` 与 `GPT-6 Astra` 之间加一个外部记忆/调度器，决定“何时需要复核、何时进入恢复”；在 4 个 RoboDojo 仿真任务上降低 reviewer token 与墙钟时间，均值成功率为 97.19%，对比 GPT-as-Policy 报告的 80%。

> **版本与证据说明**：本文依据 arXiv:2610.31427v1（2026-10-22）的 HTML/PDF 正文与 Appendix A–F。候选元数据摘要只到 “as experience acc” 截断，未发现可用其补写正文的实质内容；以下机制与数字均取自正文、表格和附录。全文用“论文实测”“作者主张”“条件性分析”区分证据强度。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 冻结 `π0.5`、保留 `GPT-6 Astra` reviewer，用 stage-level reviewed outcomes 调度复核频率；4 个仿真任务上 token 降 57.2%，平均 episode 时间 85.5→52.0 分钟，均值成功率 97.19% vs 80%。 |
| 适合精读 | 做 agentic robot execution、VLA 部署成本、selective supervision 的读者；重点看 §4、Appendix A 的公式，以及 §5.4–5.5 的运行时与恢复分析。 |
| 可以跳过 | 只关心 VLA 预训练、网络结构或新策略学习的读者；本文不训练新 VLA，也不改 `π0.5` 权重。 |
| 落地可行性 | 中：调度器和记忆机制不复杂，但依赖 reviewer API 延迟、stage memory 阈值和任务资产；证据仅在仿真。 |
| 主要风险 | 无真实机器人实验；基线来自外部报告；逐任务成功率并非全升，Fold clothes 从 100% 降到 95%；失败检测仍有漏检。 |

💡 **X-Ray 开场**

这篇论文问的是一个部署问题：VLA 执行时，是否每个 action chunk 都必须让大模型复核？FastHarness 的答案是“不必”，它把复核当成可调度资源：熟悉的 stage 少复核，出现错误或接近历史错误时密集复核。对 VLA 研究者而言，这意味着 agentic robot 的成本瓶颈可能不在策略推理，而在 reviewer 调用；但当前证据只覆盖 4 个 RoboDojo 仿真任务。

📍 **研究全景时间线**

```
2019 HG-DAgger（交互式干预/纠错）
  → 2022 SayCan / Inner Monologue（语言规划 + 执行反馈）
  → 2025 π0.5（预训练 VLA）
  → 2026 GPT-as-Policy（π0.5 + GPT-6 Astra，每 action chunk 复核）
  → 2026 FastHarness（经验驱动复核调度）← 本文位置
```

> 时间线只列来源可确认的少量节点，不展开完整参考文献表。

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块（归属） | 输入 | 输出 | 频率/时序 | 训练与记忆归属 |
|---|---|---|---|---|
| `π0.5` policy（预训练 VLA，冻结） | 当前观测：3 路 RGB、joint/EE state；语言指令 | 预测 50 个 actions；实际执行 proposal 的 1–15 control steps | 每个新观测生成新 proposal；control step 0.04 s（25 Hz） | 仅推理；权重固定；**不持有** FastHarness 的 stage memory |
| `GPT-6 Astra` reviewer（外部 reviewer） | 观测、proposal、历史、memory、recovery context | approve / corrective feedback；stage label；success/failure/intent judgment | 按调度触发，不是每 chunk | 本文不做训练；不更新权重；持久记忆在 harness |
| FastHarness memory + scheduler（harness） | 590 维特征、reviewed outcomes、stage 统计、error records、waypoints | review flag、`I_s`、skip/execute、recovery state | 每个 action chunk 决策；review 时更新 | 规则 + 统计调度；无梯度训练；**持久 memory 属 harness** |
| Recovery path（harness + reviewer） | error / intent mismatch、local attempts、waypoints | local repair、waypoint correction、verified resume | recovery active 时每 chunk review | 不训练；error record 保留 |
| Baseline GPT-as-Policy（`π0.5` + `GPT-6 Astra`） | 同 policy/reviewer | 每 action chunk 都复核 | 每 chunk | 无自适应 scheduling；是本文外部基线 |

**逐模型归属要分清**：`π0.5` 是冻结的 action generator，不是“带 stage memory 的策略”；`GPT-6 Astra` 是 reviewer，不因复核而学习；FastHarness 的 adaptation 发生在外部 memory 和调度规则里。Baseline 也不是本文的同轮消融，而是来自 GPT-as-Policy 报告的外部对比（Su et al., 2026）。

**时序口径：先区分已观测历史帧与未来预测帧**

```
已观测历史：当前 obs_t；memory 中保存的 past reviewed examples / outcomes
未来预测：π0.5 一次输出 50 actions（未来动作序列），实际执行 proposal 的 1–15 control steps
自动 span：两次 review 之间已执行的连续 chunks，是过去执行段，不是预测 horizon
review interval I_s：以 chunk 计的调度间隔，不是历史窗口或记忆窗口

通用核算：K 帧均匀采样、间隔 Δt → 首末跨度 = (K-1)×Δt
从当前参考 t 预测 t+1 至 t+H → 终点距 t = H×Δt
本文配置换算（条件性计算）：50 actions ≈ 2.0 s 预测；1–15 steps ≈ 0.04–0.6 s 执行；120 steps allowance ≈ 4.8 s
```

不能把 `π0.5` 预测的 50 个未来 actions 当成历史上下文，也不能把 120 control steps 的 allowance 写成所有实验共有的“记忆窗口”。论文的记忆匹配用的是过去 review 保存的 590 维特征，而不是未来 chunk。

### 1.2 关键机制 (Key Mechanism)

- **Stage recognition + outcome memory**：reviewer 在复核时给 stage 标签（如 approaching、grasping、placing）。Harness 用降采样 RGB 与 proprioceptive state 组成 590 维特征，按 stage 保存最近 128 个不同示例；新观测匹配最近示例，距离过远或两个 stage 距离接近会触发 review。
- **只给“紧邻的前一个 transition”记 outcome**：每次 review 只对 immediately preceding ordinary policy transition 记 success/failure，并归到其 pre-action stage。两次 review 之间的更早 chunks 不单独打标签，corrective/recovery actions 也不计入普通 outcome 统计。这是避免“把恢复动作算进策略成功率”的关键。
- **Conservative score → review interval**：对每个 stage 用 reviewed success/failure 计算保守成功率下界；样本少或刚出错后连续成功不足时，间隔固定为 1 chunk；样本多且分数高时，间隔变长。
- **Observation-conditioned execution**：跳过 review 时，不重放旧轨迹；执行当前 proposal 的一个 prefix，但受“自上次 review 以来剩余 control-step allowance”限制，然后取新观测、新 proposal。也就是说，跳过的是 review，不是环境闭环。
- **Recovery path**：review 发现执行错误或 intent mismatch 后，进入每 chunk 都 review 的恢复态；先 local repair，若两次 local attempt 失败且有 eligible waypoint，则转向此前 episode 内验证过的 waypoint。恢复完成需要 reviewer 验证，之后才恢复普通调度；error record 保留，靠近该错误的观测仍会触发 review。
- **触发优先级**：always-review、episode start、active recovery、unknown/ambiguous context、可选 contact gating（本次关闭）、stage change、previous uncertainty、monitor/proposal warning、error proximity、control-step allowance、scheduled interval、random audit。调度不是只看 `I_s`，而是多个触发器取最先满足者。

⚡ **Eureka Moment**：把 LLM 复核从“每个 chunk 的固定税”变成“按 stage 经验分配的资源”，同时用 error proximity 和 recovery state 保住安全网。

### 1.3 信息流/架构图 (Flow / Diagram)

```
instruction + current obs_t
        │
        ▼
  π0.5 (frozen) ──► proposal A_t
        │            （预测 50 actions；执行 1–15 control steps）
        ▼
  FastHarness scheduler
   ├─ match stage memory (590-d feature)
   ├─ compute r_s, I_s, ordered triggers
   ├─ if review:
   │     GPT-6 Astra review
   │       ├─ approve → execute
   │       ├─ error / intent mismatch → recovery
   │       └─ stage label + outcome → memory
   └─ if skip:
         execute proposal prefix within remaining 120-step allowance
        │
        ▼
  new obs_{t+1} → repeat
```

```
error / intent mismatch
  → recovery active：每 chunk review
  → local repair（每次 attempt 最多 8 个 executed chunks）
  → 2 次 local attempts 失败且存在 eligible waypoint
  → waypoint-directed correction
  → 验证 return goal + 单独评估 original fault
  → 恢复普通调度；保留 error record 与 recovery evidence
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
review_t = Trigger( d_s(x_t), r_s, n_s, c_s, f_s, allowance_t, audit_t )
I_s = 1 + floor((L-1) · r_s^2 · min(C, T_s/m))   （证据足够且 cooldown 清除时）
```

> 论文没有给出单一显式优化目标；下面是从 §4 与 Appendix A 提炼的调度规则，不是作者写出的统一损失函数。

**Stage 距离与 unknown/ambiguous 判定**

```
d_s(x) = min_{e in E_s} sqrt( (1/590) · Σ_{j=1}^{590} (x_j - e_j)^2 )

unknown：memory 为空，或 min_s d_s(x) > 0.06
ambiguous：最小两个 stage 距离之差 < 0.01
nearby error：与历史 error 的特征距离 ≤ 0.04
```

**Conservative score（Wilson 下界形式）**

```
r_s = max(0,
      ( p_hat_s + z^2/(2 n_s)
        - z · sqrt( p_hat_s(1-p_hat_s)/n_s + z^2/(4 n_s^2) ) )
      / (1 + z^2/n_s)
    )

z = 1.96
r_s = 0，若 n_s = 0
```

**Review interval**

```
I_s = 1，若 T_s < m 或 (f_s 且 c_s < c_min)

I_s = 1 + floor( (L-1) · r_s^2 · min(C, T_s/m) )，否则

m = 2
c_min = 3
L = 20
C = 6
```

| 符号 | 含义 | 本文配置/说明 |
|---|---|---|
| `s` | stage 标签 | reviewer 给出，如 approaching/grasping/placing |
| `S_s`, `F_s` | reviewed success/failure 数 | 只统计普通 policy transitions |
| `n_s` | `S_s + F_s` | 样本量 |
| `p_hat_s` | `S_s / n_s` | 经验成功率 |
| `r_s` | 保守成功率分数 | 样本少时被压低 |
| `T_s` | labeled ordinary policy transitions 数 | count decay = 1.0，故本文 `T_s = n_s` |
| `c_s` | 上次错误后的连续成功数 | 用于 cooldown |
| `f_s` | 是否有历史错误 | 有错误且 clean streak 不足时强制 review |
| `L` | interval scale | 论文明确说不是 hard cap |
| `C` | experience multiplier cap | 限制样本量乘子 |
| `I_s` | 以 chunk 计的调度间隔 | 不是记忆窗口 |

**直觉**：`r_s` 是“保守的成功率下界”。样本少时，即使 `p_hat_s=1`，下界也不高；样本多且失败少时，下界升高，`I_s` 变长。`I_s=k` 只允许最多 `k-1` 个后续 automatic chunks，且随时可能被 unknown、error proximity、stage change、120-step allowance 或 audit 截断。因此“计划间隔”和“实际自动执行 span”不是同一个量。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为**假设数值**，只用于验算公式与工程口径，不表示论文实际 stage 统计。

**假设 stage A：早期证据少**

```
S=2, F=0, n=2, p_hat=1, T=2, f=0, c=2
z=1.96

r ≈ 0.342
min(C, T/m) = min(6, 2/2) = 1
I = 1 + floor(19 × 0.342^2 × 1) = 1 + 2 = 3 chunks
```

含义：2 次全成功后，计划间隔只有 3 个 chunks。若每个 chunk 执行 10 steps，约 30 steps，未触及 120-step allowance。

**假设 stage B：证据多，但历史出过错**

```
S=18, F=2, n=20, p_hat=0.9, T=20, f=1, c=5
r ≈ 0.699
min(C, T/m) = min(6, 20/2) = 6
I = 1 + floor(19 × 0.699^2 × 6) = 1 + 55 = 56 chunks
```

含义：计划间隔为 56 chunks。但若平均 chunk 为 10 steps，56 chunks 约 560 steps，远大于 120-step allowance；实际自动执行会先被 allowance 截断到约 12 chunks。若 `c=2` 而不是 5，则 `(f_s 且 c_s < 3)` 成立，`I_s` 被强制回到 1，说明 cooldown 会压过经验分数。

这个例子说明：**调度间隔不是实际 span，更不是记忆窗口**；公式给出 chunk 级计划，control-step allowance 和触发器决定现实执行长度。

## 4. 工程视角 (Engineering View)

- **reviewer 时间是主成本**：论文实测 reviewer interaction 占 Fold clothes 总墙钟 82.0%、Bottle disposal 89.7%；memory operations 均小于 0.1%。因此优化 reviewer 调用频率比优化 590 维记忆匹配更划算。
- **控制频率与复核延迟分离**：control step 为 0.04 s（25 Hz），但平均 episode 时间仍是 85.5→52.0 分钟。按配置，120 steps allowance 约 4.8 s 机器人执行；reviewer 等待/重试占大头。论文未报告异步流水线，因此不能假设复核不阻塞执行。
- **预测 horizon ≠ 执行 chunk**：`π0.5` 预测 50 actions，约 2.0 s；实际执行 1–15 steps，约 0.04–0.6 s。工程上要区分“模型未来预测长度”和“本次实际执行长度”，否则会把预测窗口误当记忆窗口。
- **记忆开销小但语义敏感**：590 维特征、每 stage 最多 128 个示例、checkpoint 容量 64，存储与匹配成本低；但阈值 0.06/0.01/0.04 是任务相关超参。阈值过松会漏掉错误，过紧会退回高频 review。
- **无额外训练成本**：FastHarness 不改 `π0.5` 权重，也不训练 scheduler；它依赖 reviewer 的结构化输出和外部 memory。部署门槛主要是 reviewer API 稳定性、stage label 质量和任务资产。
- **条件性分析**：若 reviewer latency 降低，绝对墙钟收益会缩小，但“减少调用”仍有效；若 stage 识别在真实传感器上漂移，调度可能把不熟悉状态误判为熟悉，风险高于仿真。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练与模型来源

- `π0.5` 是预训练 VLA（Physical Intelligence et al., 2025），本文使用固定 checkpoint 目录 59999、配置 `pi05_base_aloha_full_sim_arx-x5_seed_0`，权重冻结。论文没有描述其预训练数据配比或训练轮次。
- `GPT-6 Astra` 是外部 reviewer（Su et al., 2026），配置 reasoning effort `xhigh`，通过 Responses protocol 输出结构化 review。本文不训练它。
- FastHarness 本身**不做额外模型训练**；论文没有 epoch、训练步数或 fine-tuning 轮次。实验次数单位是 episode、control step、action chunk、review call，不是训练轮次。

### 5.2 仿真评测设置

RoboDojo on Isaac Sim；NVIDIA H20，CPU-based physics；Python 3.11.15、Isaac Sim 5.1.0、JAX/JAXlib 0.5.3、Flax 0.10.2、NumPy 1.26.4。控制 25 Hz；三路 RGB（high、left wrist、right wrist）+ joint/EE state，无 depth；14 维动作接口；每 task chain 共享 memory。

| Task | Episodes | Native step limit | FastHarness success | GPT-as-Policy 基线（外部报告） |
|---|---:|---:|---:|---:|
| Build tower | 16 | 1,050 | 15/16 = 93.75% | 60% |
| Fold clothes | 20 | 500 | 19/20 = 95% | 100% |
| Bottle disposal | 20 | 700 | 20/20 = 100% | 100% |
| Classify objects | 9 | 1,100 | 9/9 = 100% | 60% |
| 未加权任务均值 | — | — | **97.19%** | **80%** |

> 基线来自 GPT-as-Policy 报告，不是本文同轮配对实验；其本地 hybrid 与 GPT-only 每任务 5 个 case（Fold clothes 为 2 standard + 3 random，其余 5 standard）。因此 97.19% vs 80% 是跨来源、未加权任务均值比较，不能解读为逐任务全面提升。

**效率实测（论文 §5.2、§5.4、Appendix D）**

| Task | 首 episode token | 最后五 episode 均值 | 降幅 |
|---|---:|---:|---:|
| Build tower | 1.47 M | 0.47 M | 68% |
| Fold clothes | 0.72 M | 0.25 M | 65% |
| Bottle disposal | 1.64 M | 0.58 M | 64% |
| Classify objects | 3.50 M | 2.42 M | 31% |

- 四个 per-task 降幅的未加权均值为 57.2%；equal-task mean token 从 1.83 M 降到 0.93 M。
- equal-task mean episode time 从 85.5 分钟降到 52.0 分钟，约低 39%。
- Build tower reviewer calls 40→13.4（最后五均值），约 75% chunks 自动执行；Classify objects 94→68.2，约 24% 自动执行。
- Fold clothes / Bottle disposal 的 automatic span 中位数均从 30→60 control steps；未知上下文 trigger 在五 episode 窗口中分别从 35→16、142→64。
- Recovery：Classify objects episodes 1–8 共 16 个 recovery events，全部回到 normal scheduling；但 Fold clothes 有一例 20 次 review 未触发 recovery，native score 仍为 0.2，说明“review 了”不等于“检测到任务级失败”。

### 5.3 真实机器人设置

论文**没有真实机器人实验**。所有结果来自 RoboDojo 仿真；Limitations 还指出四个仿真任务的覆盖、不同传感器接口可能需适配记忆表示，以及绝对运行时节省取决于 reviewer 延迟和执行硬件。不能把结果外推到真实桌面、移动或人形平台。

### 5.4 评测口径与条件

- 效率比较是“首个 evaluated episode”对“最后五个 episode 均值”；Classify objects 是 episode 1 对 episodes 5–9。
- 复核分配与 automatic span 比较 episodes 1–5 与 16–20（Fold clothes、Bottle disposal）。
- Build tower 的首个 evaluated episode 跟随更早任务经验，对应 source episode 2；不是从零开始。
- Classify objects 在 episode 开始时强制 review；其 recovery 分析覆盖 episodes 1–8。
- Episodes 共享 task memory，不是独立试验；论文明确报告 descriptive chain-level results，不把 episode 或 recovery event 当独立样本。
- Success 由 native evaluator 的 `native_success` 决定；reviewer approval 与 partial score 不改变 native outcome。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

| 能力 | 证据 |
|---|---|
| 降低 reviewer 调用与 token | 四任务 token 降 57.2%；Build tower calls 40→13.4 |
| 降低墙钟时间 | 85.5→52.0 分钟；reviewer 占 82–89.7% |
| 保留闭环执行 | 跳过 review 时仍取新观测、新 proposal，不重放旧轨迹 |
| 支持恢复 | local repair、waypoint correction、verified resume；Classify 16/16 recovery events 回到 normal scheduling |
| 保留失败经验 | error record 与 recovery evidence 保留，近错误观测仍触发 review |

**失败模式与限制**

- **逐任务成功率并非全升**：Fold clothes 从 100% 降到 95%（19/20）；均值提升主要来自 Build tower 60→93.75% 与 Classify objects 60→100%。
- **失败检测会漏**：Fold clothes 一例 20 次 review 未触发 recovery，最终 native score 0.2；review 频率高不等于任务级失败被识别。
- **无真实机器人**：仅 4 个 RoboDojo 仿真任务；传感器、接触、延迟分布变化都可能改变记忆匹配与调度收益。
- **外部基线**：GPT-as-Policy 基线来自报告，case mix 与本文 episodes 不完全相同；80% 是未加权任务均值，不是配对比较。
- **stage label 依赖 reviewer**：stage 标签与 success/failure 判断若有噪声，会污染 outcome counts 和 `r_s`。
- **记忆表示较窄**：590 维来自 3 个 8×8 RGB 缩略图 + 14 维状态；无 depth，contact gating 关闭。迁移到不同传感器接口需重设阈值。
- **运行时依赖 reviewer**：绝对节省取决于 reviewer latency、重试与硬件；论文的 52 分钟/ episode 也不是实时系统指标。

### 6.1 隐含假设 (Hidden Assumptions)

1. reviewer 的 stage label 与 outcome judgment 足够可靠，错误不会系统性累积。
2. 590 维降采样特征 + 距离阈值能跨 scene layout 识别“同一 stage”。
3. 冻结 `π0.5` 在任务域内足够强，harness 不需要 policy adaptation。
4. reviewer 延迟是主要成本，且调用频率与墙钟时间近似同向变化。
5. 近错误 review 能提供足够安全网；但论文已展示任务级失败漏检。
6. episodes 共享 memory 的 chain-level 结果可描述趋势，但不能当独立同分布试验。
7. 外部基线虽用于均值对比，但其任务 case mix 与本文不完全一致。

## 7. 与相关工作对比 (Comparison)

| 方法 | 关注点 | 架构/适应方式 | 与 FastHarness 的差异 |
|---|---|---|---|
| SayCan (2022) | 语言技能选择与 affordance grounding | LLM + robot skills | 不做固定 VLA 的复核调度 |
| Inner Monologue (2022) | 执行反馈引导语言规划 | LLM + feedback | 不是 chunk 级 reviewer 预算分配 |
| HG-DAgger (2019) | 交互式模仿学习 | 人类干预 + policy learning | 需要学习策略；FastHarness 冻结策略 |
| ThriftyDAgger (2021) | budget-aware intervention gating | novelty/risk gating + interactive IL | 关注人类干预预算；本文关注 LLM review 调度 |
| Reflexion / REFLECT (2023) | 经验总结、失败解释与纠正 | verbal feedback / experience summary | 不做在线 review interval 控制 |
| GPT-as-Policy (2026) | `π0.5` + `GPT-6 Astra` 每 chunk 复核 | 固定高频 review | FastHarness 的直接基线；本文把 review 变成可调度资源 |
| **FastHarness（本文）** | 经验驱动 review scheduling + recovery | 冻结 VLA + 外部 memory/scheduler + reviewer | 不改策略权重；用 stage 经验与 error proximity 分配复核 |

**面试 Tip**：如果被问“FastHarness 和 GPT-as-Policy 的本质区别”，答：不是换 VLA 或换 reviewer，而是在同一对模型之间加 stage-level experience scheduler 和 recovery state；核心变量是 reviewer 调用频率，不是 policy 精度。但要注意 Fold clothes 从 100% 降到 95%，均值提升来自 Build tower 与 Classify objects。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  - 做 agentic robot execution / LLM-in-the-loop 监控的研究者：重点看 §4 与 Appendix A 的触发优先级、outcome attribution、recovery state。
  - 要评估 VLA 部署监控成本的工程师：重点看 §5.4 的 reviewer 时间占比、token 口径、120-step allowance 与 automatic span 的区别。
  - 做 selective supervision / intervention scheduling 的读者：重点看与 HG-DAgger、ThriftyDAgger 的差异，以及“冻结策略 + 外部调度”的设定。

- **建议章节路径**：先读 §4.1–4.2 建立机制 → 再看 Appendix A 的公式与触发顺序 → 最后读 §5.4–5.5 和 Appendix D 的数字与恢复案例。若只关心结论，读 Abstract + §5.3 + Limitations 即可。

- **不值得精读的理由**：如果你不做机器人执行监控、不关心 reviewer 成本，或已经熟悉 selective supervision，本文的机制增量主要是“把 stage 经验用于 chunk 级复核调度”；不需要逐行读附录。

---

[← Back to Theory](./README.md)

论文链接：https://arxiv.org/html/2610.31427