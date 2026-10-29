# RePO-VLA：把失败 rollout 拆成恢复监督的 VLA 策略优化 (Recovery-Driven Policy Optimization for Vision-Language-Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-29
>
> **论文**: RePO-VLA: Recovery-Driven Policy Optimization for Vision-Language-Action Models
> **链接**: https://arxiv.org/abs/2605.09410
> **核心定位**: 不是新 VLA backbone，而是在 π0.5 微调阶段把成功、失败、恢复轨迹分配不同监督：RAI 切恢复后缀、PAS-VF 打 progress 价值标签、VCR 用固定高价值 token 做 value-conditioned refinement。

> 证据版本说明：本笔记依据 arXiv:2605.09410v2 HTML 正文（2026-10-26，CC BY 4.0，对应 PDF v2）。候选元数据摘要只保留到 “slicing each corrective” 的截断片段，未用于补写正文主张。正文摘要中的 “20%→75% / 80%” 必须按 Table 3/4 拆开：75% 是 Table 4 的 Full∗∗(4x) 两个代表性任务 adversarial 平均，80% 是其中 Pour Water 单任务；Table 3 的 Full(1x) 四任务 adversarial 平均是 20%→30%，Phase I 为 37.5%。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 失败 rollout 不必丢弃；切出 corrective suffix 做 RAI，混合质量数据用 dense progress label 做 VCR；4x 真实恢复数据下 adversarial 平均 20%→75% |
| 适合精读 | 做 bimanual contact-rich VLA、失败数据再利用、value-conditioned fine-tuning 的读者；重点看 §3、§5、Appendix A.2/A.4 |
| 可以跳过 | 只关心新 VLA backbone、预训练阶段机制、实时控制部署的读者；这篇是 fine-tuning framework |
| 落地可行性 | 中高：不改 pretrained backbone，推理无 online detector；但需要 recovery 数据引擎与 dense labels，真实 1x 下 VCR 并不稳定 |
| 主要风险 | 依赖成功参考集与规划器/人工恢复数据；未覆盖开放世界失败；真实每任务仅 10 trials，统计样本小 |

💡 **X-Ray 开场**

这篇论文解决的是：success-only imitation 对 execution drift 几乎没有监督，失败 rollout 往往被丢弃。它发现失败轨迹不是一整条负样本：早期可能有可用的 approach，末期是漂移，恢复后缀是可学习的纠正行为。对 VLA 研究者意味着：失败数据可以被“组织”成 value 分层监督，而不是把 raw failure-recovery 直接 append 进训练集；但本文只证明 fine-tuning 阶段的收益，不证明预训练阶段会自然出现 recovery 表征。

📍 **研究全景时间线**

```
2025  OpenVLA / π0 / π0.5 / RDT / GO-1：成功示范 SFT 与 flow-matching/diffusion VLA 基线
      ↓
2025  FailSafe 等：合成 failure-recovery 数据，但仍以 SFT 为主
      ↓
2026  RePO-VLA v2：RAI 切片 + PAS-VF 价值 + VCR；提出 FRBench
      ↑ 本文位置
局限：fine-tuning 框架；未进入预训练；真实 1x 下 VCR 不一定赢 Phase I，4x 恢复数据才明显超过
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块/归属 | 输入 | 输出 | 训练/推理 | 时序/频率 |
|-----------|------|------|-----------|-----------|
| RAI / TSER | raw recovery episode、expert demos | D_rec^reset；Phase I checkpoint | 训练阶段 I；离线数据切片 | 从 t_rec 切到 T，后缀重编号为 t=0；不改观测历史缓存 |
| PAS-VF | 成功轨迹、语言指令、失败轨迹 | dense v_t ∈ [0,1] | 离线训练 frozen V-JEPA + CLIP 上的 adapters | 成功轨迹用前缀 τ_{0:t} 做 monotonic alignment；失败轨迹用 V(τ_fail)=max CosSim(z_{τ_fail}^v,z) 算 self-referential value；部署不运行 |
| VCR / Full | D_total = D_succ ∪ D_rec^raw ∪ D_fail | value-conditioned policy | 训练阶段 II；flow-matching π0.5 + value token | 每个时间步输入 o_t、l、e_val(v_t)；推理固定 v=1.0 |
| FRBench data engine | 仿真任务、规划器、真实遥操作 | 注入错误与恢复轨迹 | 仿真自动注入 + 真实 HITL | 仿真生成 23,453 episodes；真实每任务 50 teleoperated recovery episodes（50 rollouts/10 trials 属评测协议） |
| π0.5 backbone | 单时间戳观测 o_t、指令 l | 动作 a_t | 预训练 backbone；本文只微调 | 论文未报告控制频率/延迟 |

### 1.2 关键机制 (Key Mechanism)

- **失败利用盲点**：纯失败轨迹被丢弃，但失败早期常含有效 approach、contact preparation 或 partial manipulation；末期才应标记为低价值。作者主张失败应被分解，而不是整条丢弃。
- **raw recovery 的因果混淆**：一条 recovery episode 同时含“导致失败的 prefix”和“纠正 suffix”。若整条模仿，策略可能把“前面犯错”当成“触发纠正”的上下文。TSER 丢弃 failure prefix，把 corrective suffix 序列化成新 episode，让纠正动作只条件于当前 adverse state。
- **PAS-VF 的 dense label**：成功轨迹用于训练 monotonic progress alignment；失败轨迹用成功参考集的最大 cosine similarity 得到 self-referential progress；纯失败再用 reliability decay 保留早期有用动作、压低末期失败动作。
- **VCR 的 value conditioning**：把 v_t 经 MLP 变成 value token，与视觉、语言 token 一起进 transformer；同一观测在低/高 value token 下可映射到不同动作模式。部署时固定 v=1.0，把高 progress 作为默认目标。
- **无 online detector**：推理不估计 V(τ)、不切换策略、不跑 retry controller；nominal 与 corrective action 来自同一接口。

⚡ **Eureka Moment**：THE 关键洞见是——把失败轨迹从“一条负样本”改写成“带时间价值密度的混合质量数据”，并用 episode reset 切断失败前缀与纠正动作之间的虚假因果。

### 1.3 信息流/架构图 (Flow / Diagram)

```
训练:
成功 demos ─┐
失败 rollouts ─┼→ 数据引擎 → D_succ / D_fail / D_rec^raw
恢复 rollouts ─┘                     │
                                     ├─ TSER → D_rec^reset → RAI (Phase I)
                                     └─ PAS-VF → v_t → VCR (Full)

部署:
o_t, l → π0.5 + e_val(1.0) → a_t
无 online V、无 failure detector、无 heuristic retry
```

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
失败轨迹 = 可用前缀 + 低价值漂移 + 高价值修正；训练用 v_t 分层，推理固定 v=1.0 拉回成功流形
```

先给目标：让策略在 drift 后回到 success manifold，同时不奖励 terminal failure。RAI 先切断虚假因果，PAS-VF 再给混合质量数据打 dense label，VCR 最后学 value-conditioned action。

**RAI 的 SFT 目标**：

```
L_SFT = -E_{τ~D_expert} Σ_t log π(a_t | o_t)
        - λ E_{τ'~D_rec^reset} Σ_t log π(a_t | o_t)
```

**PAS-VF 的 monotonic progress alignment**：

```
L_align = E_{τ∈T_succ} [ (1/T_τ) Σ_{t=1}^{T_τ} ( CosSim(z_t^v, z^l) - t/T_τ )^2 ]
```

其中 z_t^v = f_θ(E_v(τ_{0:t}))，z^l = g_φ(E_t(l))；E_v 是 frozen V-JEPA，E_t 是 frozen CLIP，只训练轻量 adapters。

**失败轨迹的 self-referential value**：

```
V(τ_fail) = max_{z∈C_ref} CosSim(z_{τ_fail}^v, z)
```

**纯失败的 reliability decay**：

```
v_t = V(τ_fail) · (1 - t/T)^α,  α=3.0
```

**VCR 的 value-conditioned 模仿目标**：

```
L_VCR = -E_{τ~D_total} [ Σ_{t=1}^{T_τ} log π_θ(a_t | o_t, l, e_val(v_t)) ]
```

**仿真数据引擎的自动失败判定**（只用于数据生成，不是部署检测器）：

```
F_t = I(t > T_max),  T_max = max_i T_succ^i
```

| 符号 | 含义 | 备注 |
|------|------|------|
| τ, o_t, a_t, l | 轨迹、观测、动作、语言指令 | 策略输入是单时间戳 o_t |
| D_expert, D_rec^reset, D_rec^raw, D_fail | 专家数据、切片恢复数据、原始恢复数据、纯失败数据 | RAI 用 reset；VCR 用 raw |
| λ | RAI 中 nominal 与 recovery 的平衡系数 | 论文未给具体值 |
| α | reliability decay 指数 | 主实验 α=3.0 |
| v_t, e_val | 每帧价值标签、value token | 推理固定 v=1.0 |
| C_ref | 成功轨迹 embedding 参考集 | 用于失败轨迹的 progress 估计 |

直觉：RAI 让纠正动作依赖当前 adverse state，而不是前面那段错误历史；PAS-VF 把“离成功流形多近”变成 dense label；VCR 让同一观测在 value token 条件下选择 nominal 或 corrective 动作模式；推理固定高 value，使高 progress 行为成为默认目标。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下为假设例子，不是论文实验；只验算公式行为。

假设一条纯失败轨迹 T=10，V(τ_fail)=0.64，α=3.0。按 reliability decay：

| t | (1 - t/10)^3 | v_t = 0.64 × 该项 |
|---|--------------|-------------------|
| 0 | 1 | 0.64 |
| 5 | 0.125 | 0.08 |
| 9 | 0.001 | 0.00064 |
| 10 | 0 | 0 |

可以看到：早期仍保留 0.64 的可用先验，中段快速降到 0.08，接近 terminal failure 时几乎为 0。这正对应作者主张的“保留 useful pre-failure behavior，压低 terminal breakdown”。

再验算时间跨度，避免把 frame 数当成时间：

```
K=7 个均匀采样帧，Δt=0.1s → 首末跨度 = (K-1)×Δt = 0.6s
若从当前参考 t 预测 t+1 至 t+H，H=10，Δt=0.1s → 终点距当前参考 = H×Δt = 1.0s
```

本文仿真恢复试验中“gripper open 30 frames（约 1s）”是扰动保持窗口，不是策略记忆窗口，也不是 action chunk 长度。论文未报告策略的 action chunk 长度或未来预测 horizon，不能把 30 frames 直接当作历史上下文。

## 4. 工程视角 (Engineering View)

- **训练资源**：论文报告 2×NVIDIA A100 80GB、128 CPU 核、512GB host RAM；RAI + VCR 一次完整训练约 30 小时，包含 value-label preparation 与 PAS-VF adapter 训练。FRBench-Sim 数据生成最多用 128 个 CPU workers。
- **部署边界**：真实机器人用两条 Dobot X-Trainer arms；部署只需单 GPU 做策略推理。没有 online value estimation、failure detector、retry controller，模块边界比“检测器 + 重规划器”简单。
- **数据成本是主矛盾**：真实每任务 200 expert demos + 50 teleoperated recovery episodes；Table 3 的 Full(1x) 在 adversarial 平均只到 30%，低于 Phase I 的 37.5%；Table 4 把恢复数据加到 4x 后，Full∗∗ 才到 75%。这说明 VCR 需要足够密的 failure/recovery 区域，1x 下可能太稀疏。
- **时序与记忆**：策略输入是单时间戳观测，不堆叠历史图像；TSER 是 dataset-level episode bookkeeping，不是 observation-history buffer reset。PAS-VF 对 τ_{0:t} 做前缀编码，但只用于训练价值模型，不是部署记忆。若未来实现加入 action chunk 或未来帧预测，必须另测；本文没有给出可复用的 chunk/horizon 数值。
- **未报告项**：控制频率、端到端推理延迟、显存峰值、量化误差、动作 chunk 长度均未在正文给出。TODO：这些是工程落地前必须补测的指标，不能从 30 小时训练时间或 30 frames 扰动窗口推断实时性。

## 5. 数据与评测 (Data & Eval)

### 5.1 预训练、微调与数据开放范围

| 模型/组件 | 预训练/初始化 | 数据开放范围（论文可确认） | 记忆/历史机制 |
|-----------|---------------|----------------------------|----------------|
| RePO-VLA Full | π0.5 pretrained backbone + value token；不修改 backbone | FRBench-Sim 23,453 episodes；真实每任务 200 demos + 50 recovery；摘要称 code/video，但本次来源未给可验证仓库 | 单时间戳 o_t；无堆叠历史；无 online V |
| Phase I (RAI) | 同 π0.5 + RAI | D_rec^reset 切片数据 | 单时间戳；TSER 只改 episode 边界 |
| π0.5 baseline | flow-matching VLA | 仅作为成功示范 SFT 基线；论文未给其数据开放细节 | 论文未描述记忆机制，不能推断 |
| PAS-VF | frozen V-JEPA + CLIP，只训 adapters | 成功轨迹参考集 C_ref | 编码 τ_{0:t} 用于价值训练，不是部署记忆 |
| FRBench data engine | 规划器/人工遥操作 | 仿真 17,061 nominal + 6,392 verified recovery；46 tasks | phase 变量 φ_t 是评测状态，不是记忆 |

作者明确说 RePO-VLA 是 fine-tuning-time 的 recovery-data 与 supervision framework，不修改 pretrained VLA backbone，也不声称 recovery representation 在大规模预训练中涌现。这一点必须和“新 backbone”区分开。

### 5.2 仿真评测（FRBench-Sim）

FRBench-Sim 覆盖 46 个 bimanual tasks，共 23,453 episodes：17,061 nominal-success、6,392 verified failure-recovery。E1–E4 的 pre-filter injection attempts 合计 16,912（E1 premature close 8,022；E2 grasp slip 3,516；E3 position offset 4,686；E4 orientation mismatch 688），过滤后保留 6,392 条恢复轨迹。E2 的实现是 lift 阶段强制打开 gripper 30 frames（约 1s）。

评测协议：每个任务 50 rollouts，分 Clean 与 Random；Table 2 报十个 RoboTwin tasks 的平均成功率。恢复试验在 grasp initiation 注入 grasp disturbance，迫使策略重新建立 object control。

| 设置（Table 2，论文实测） | π0.5 Clean | π0.5 Rand | RePO Full Clean | RePO Full Rand |
|---------------------------|------------|-----------|-----------------|----------------|
| Injected failure avg success | 15.0 | 15.4 | 37.0 (+22.0) | 43.0 (+27.6) |

同一张表还显示：Phase I 在 nominal 下 Clean/Rand 为 44.6/44.0，而 π0.5 为 27.4/33.6；作者主张 RAI 没有牺牲基础任务能力。注意这是 nominal 设置，不是恢复成功率。

### 5.3 真实机器人设置与数据缩放

真实评测用两条 Dobot X-Trainer arms，四个任务：Pour Water、Cook Vegetable、Tidy Desk、Fold Towel。每任务 200 expert demonstrations + 50 teleoperated recovery episodes；标准与 adversarial 各每任务 10 trials；checkpoint 盲评，操作者只知道 ip:port，不知道方法身份。adversarial 每次只注入一个物理扰动：dynamic displacement、kinematic interference 或 forced slippage。

| 条件（Table 3，论文实测） | π0 | π0.5 | Phase I | Full (1x) |
|---------------------------|----|------|---------|-----------|
| Standard avg | 27.5 | 25.0 | 42.5 | 40.0 (+15.0 vs π0.5) |
| Adversarial avg | 12.5 | 20.0 | 37.5 | 30.0 (+10.0 vs π0.5) |

关键细节：Full(1x) 在 adversarial 平均低于 Phase I（30.0 vs 37.5）。作者没有声称 VCR 在所有数据规模都赢；相反，作者把 1x 的失败/恢复区域稀疏列为瓶颈。

| 条件（Table 4，两个代表性任务平均） | π0.5 | Phase I | Full 1x | Full 2x | Full 4x | Δ4x-0.5 | Δ4x-I |
|--------------------------------------|------|---------|---------|---------|---------|---------|-------|
| Standard avg | 35 | 50 | 40 | 70 | 80 | +45 | +30 |
| Adversarial avg | 20 | 40 | 30 | 65 | 75 | +55 | +35 |

Table 4 只含 Pour Water 与 Fold Towel 两个任务，不能把它当成四任务平均。正文摘要的“20%→75%”对应 Full∗∗(4x) 的 adversarial 两任务平均；“80%”对应 Pour Water adversarial 单任务。作者主张恢复数据密度增加会缩小 1x 瓶颈，但这不是“value conditioning 在所有规模都占优”的证据。

### 5.4 消融与对照

- **Raw DAgger-style**：Fold Towel 标准 0%（-40 vs success-only BC），说明把 raw failure-recovery 整条 append 会伤害策略。Phase-I DAgger-style 标准 40%、adversarial 20%，与 success-only BC 持平。
- **Full-4x, no value**：标准 20%、adversarial 20%；RePO-VLA Full-4x 为 70%/70%。作者据此主张“imperfect data 必须被组织，而不是均匀追加”。
- **SARM+RA-BC, Full-4x**：标准 50%、adversarial 20%；低于 RePO-VLA Full-4x 的 70%/70%。这是 external value-model baseline，不是 PAS-VF 的消融。
- **Offline RECAP 对照**：Table 6 中 Full-2x 对 RECAP-2x 的标准平均为 70 vs 75（-5），adversarial 为 65 vs 60（+5）。作者明确说只比较 offline advantage-conditioned 组件，且没有官方 RECAP code/checkpoints，不能宣称全面优于 RECAP。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做什么（论文实测范围）**

- 在 FRBench-Sim 的 Dynamic Grasp Failure 代理上恢复：Table 2 的仿真评测在 grasp initiation 用 holding gripper open 30 frames（约 1s）的 grasp disturbance，而非覆盖完整 E1–E4；其 injected failure 平均从 π0.5 的 15.0/15.4 提升到 Full 的 37.0/43.0。
- 在真实 bimanual 任务中应对 dynamic displacement、kinematic interference、forced slippage；Table 3 的 Full(1x) adversarial 平均 30.0，高于 π0.5 的 20.0，但低于 Phase I 的 37.5。
- 在恢复数据加到 4x 时显著提升：Table 4 的 adversarial 两任务平均 20→75（+55 vs π0.5），Pour Water 单任务到 80%。
- 部署时不需要 online failure detector、value estimator 或 retry controller；同一策略接口处理 nominal 与 corrective action。

**不能做什么 / 未验证**

- 不是 open-world zero-shot recovery；作者明确说 severe occlusion、tool misuse、subgoal mismatch、major perception errors 不在评测范围。
- 只在 observed failure families 上验证；不能推断对全新物理故障有效。
- 不修改 pretrained backbone，也不证明 recovery representation 在预训练阶段出现；把 failure modeling 放进预训练是 future work。
- 真实 1x 下 VCR 并不稳定优于 Phase I；需要更密 recovery data。
- 真实每任务仅 10 trials，统计不确定性未报告；盲评协议降低了偏倚，但不能替代大样本。
- 无记忆机制：策略输入单时间戳；若任务需要历史帧或长程记忆，本文没有验证。

### 6.1 隐含假设 (Hidden Assumptions)

> 以下为本笔记的条件性分析，不是论文实验结论。

- **单帧足以识别 adverse state**：TSER 让纠正动作只看当前观测；若某些失败必须靠历史才能区分，单时间戳设定会限制恢复。
- **成功参考集覆盖任务 progress**：PAS-VF 用 C_ref 的最大 cosine 作为 progress；若成功流形多模态或任务需要 backtracking，最大相似度不一定等于真实 progress。
- **error/recovery 边界可得**：训练标签依赖数据引擎注入点或人工恢复片段；真实场景中边界标注成本高，且可能不精确。
- **固定 v=1.0 可泛化到新失败**：部署时始终条件于高价值 token；这假设策略已学到“高 value = 回到成功流形”的动作模式。对未见失败类型，作者没有给出保证。
- **真实扰动代表部署分布**：三类物理扰动覆盖有限；更复杂的接触、工具使用、感知错误未测。

## 7. 与相关工作对比 (Comparison)

| 方法 | 定位 | 训练/恢复机制 | 与本文关系/证据 |
|------|------|----------------|------------------|
| RePO-VLA | recovery-driven fine-tuning framework | RAI + PAS-VF + VCR；固定 v=1.0 推理 | 本文；π0.5 backbone；不改 pretrained backbone |
| Phase I (RAI) | 消融/中间 checkpoint | 只做 TSER + RAI，不做 VCR | Table 3 真实 adversarial 37.5，高于 Full(1x) 30.0 |
| π0.5 | flow-matching VLA baseline | success-only imitation | 本文主要 baseline；Table 2/3/4 对照 |
| π0 | flow-model VLA baseline | success-only imitation | Table 2/3/4 对照 |
| RDT | diffusion bimanual baseline | success-only imitation | Table 2 中 Clean/Rand 较低；论文列为 baseline |
| GO-1 | VLA baseline | success-only imitation | Table 2 中 Random 条件下较低 |
| FailSafe | 合成 failure-recovery 数据 | 仍用 SFT | 作者称最接近，但不区分 error state 与 corrective action |
| RACER / RoboFAC | failure reasoning / replanning | 外部检测或重规划 | 与低层策略内化恢复不同 |
| ThriftyDAgger / TRANSIC | 交互纠正 / residual correction | intervention timing / gated residual | 外部或附加模块，不是同一 low-level policy 的 value conditioning |
| RECAP / RLinf 复现 | offline advantage-conditioned | 固定 offline budget 对照 | Table 6 只比较组件；Full-2x 标准 -5、adversarial +5，不能宣称全面胜出 |

**面试 Tip**：被问到时先答——“它不是新 VLA backbone，而是在 π0.5 上用 RAI 切恢复后缀、PAS-VF 给混合质量数据打 progress 标签、VCR 用固定高价值 token 做 value-conditioned fine-tuning；真实 1x 下 VCR 不一定赢 Phase I，4x 恢复数据才明显超过。”

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：做 bimanual contact-rich manipulation 的研究者；要把失败 rollout 转成训练数据的工程师；评估 value-conditioned fine-tuning 是否适合现有 VLA pipeline 的团队。
- **建议章节路径**：先读 §3.1–§3.3 抓 RAI/PAS-VF/VCR 三个机制 → 再看 §4 与 §5.1–§5.3 分清仿真、真实、1x/4x 口径 → 最后看 Appendix A.2.1–A.2.4 与 A.4.1–A.4.4 确认 episode reset、reliability decay、E1–E4 注入细节。相关工作可略读，除非要做定位对比。
- **不值得精读的理由**：如果你不做机器人学习、只关心新 backbone 或预训练阶段的 recovery 表征，读摘要与 Table 3/4 即可；本文的贡献集中在 fine-tuning 数据组织与评测协议，不在实时控制或开放世界泛化。

---

[← Back to Theory](./README.md)

关键引用：[论文链接](https://arxiv.org/abs/2605.09410)