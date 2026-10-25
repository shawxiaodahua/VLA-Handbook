# DRILL：训练世界模型以改进策略，而非预测像素 (DRILL: Training World Models to Improve Policies, Not Predict Pixels)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-25
>
> **论文**: DRILL: Training World Models to Improve Policies, Not Predict Pixels
> **链接**: https://arxiv.org/abs/2610.30814
> **核心定位**: 把视频 world model 的训练目标，从“单步像素预测误差”改成“在其内部做一次 GRPO 后，策略在目标仿真器中的下游回报”；用 VWI 做低成本代理，用 IMG 做截断元梯度。

**版本与证据说明**：本次来源记录未标记官方 PDF 与元数据摘要冲突。以下证据取自 `arXiv:2610.30814v1 [cs.CV] 22 Oct 2026` 的 HTML 正文（对应 PDF v1）。候选摘要片段覆盖了核心主张，但省略了“13.1/7.9 是五个 benchmark 平均”“17% higher one-step pixel MSE”“B_real=30、K=2”“simulation-only”等条件；本文不以候选摘要替代正文，数字以正文 Table 1/3/7/9 与附录为准。若后续官方 PDF 与摘要出现实质差异，应只以已验证身份的 PDF 正文为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：world model 的像素 MSE 与它诱导的策略收益可解耦；DRILL-IMG 在五个仿真 benchmark 上平均比 WMPO 高 13.1 个百分点，但 one-step pixel MSE 高约 17%。 |
| 适合精读 | 做 world-model-based RL 后训练 VLA 的团队；重点看 §3.2–3.4、§5.4、Appendix D.2/D.6。 |
| 可以跳过 | 只关心真实机器人部署或纯视频生成的人；本文没有 real-robot 结果。 |
| 落地可行性 | 中：DRILL-VWI 可近似 drop-in，不额外增加仿真器 rollout；DRILL-IMG 需要 B_real=30 条 simulator rollouts/outer step 与截断 HVP，工程成本更高。 |
| 主要风险 | 全部结果在仿真器；K=2 是经验选择；对 WM 规模、reward model、horizon 有条件依赖；human evaluation 未报告 IRB 批准/豁免。 |

💡 **X-Ray 开场**  
这篇论文问的是：既然 world model 的价值在于“让策略在里面学得更好”，为什么还按“下一帧像素像不像”来训练它？作者发现，像素 MSE 更低的 world model 不一定产生更好的 policy gradient；于是把 world model 的更新目标改成“策略在它内部做一次 GRPO 后，到目标仿真器里能拿多少回报”。对 VLA 研究者意味着：world model 不只是视频预测器，而是策略改进环境；训练目标应直接对准下游策略效用。

📍 **研究全景时间线**

```
2017 value-aware model learning
  → 2020 MuZero
  → 2022 IRIS
  → 2023 DreamerV3
  → 2024 GRPO
  → 2025 WMPO / RLVR-World
  → 2026 DRILL（本文，arXiv v1 2026-10-22）
```

> 只列论文正文可确认的节点；不把整张参考文献表搬成时间线。

---

## 1. 核心架构/方法总览

### 1.1 系统对比概览

先逐模型归属，避免把一组模型的不同属性合并给每个模型。

| 组件/模型 | 在论文中的角色 | 输入→输出 | 训练/使用边界 | 数据开放范围 | 时序/记忆归属 |
|---|---|---|---|---|---|
| π0-3B | headline 主 VLA policy backbone | 观测历史 o≤t、语言 ℓ → 动作 chunk a_{t:t+H_a} | 五个 benchmark 主实验；H_a=16（Table 3） | 使用公开 checkpoint，不重新分发 | 论文未给固定历史窗口；H_a 是动作 chunk 长度，不是 memory window |
| OpenVLA-7B | 跨 backbone 复现 | 同上 | 只在 LIBERO-Long、CALVIN、RoboCasa 复现；H_a=16 | 论文称 MIT；消费公开模型，不重新分发 | 同上；不能把 π0 的属性直接套到 OpenVLA |
| Ctrl-World 1.2B WM | 默认视频 world model | o≤t、a_t → 预测 ô_{t+1} | 主实验与 OpenVLA 复现共用；H=50、K=2 是默认设置 | 按原始 release 使用；论文称将释放 DRILL checkpoints，但本次来源无 repo URL | next-frame predictor，条件于 o≤t；正文未给固定历史帧数；H 是未来 rollout horizon，K 是 BPTT 深度，二者都不是记忆窗口 |
| 7B WM | 规模消融 | 同上 | 只在 LIBERO-Long scaling 中使用；不是所有实验默认 | 同上 | 只归属 scaling 实验，不外推到五个 benchmark |
| Cosmos-Predict2-style WM | 跨 WM backbone 检查 | 同上 | 只在 LIBERO-Long cross-WM 实验；一个 seed 因 outer loss spike 重启并 halve η_φ | 同上 | 只归属该 cross-WM 实验；不能写成默认 WM |
| Qwen3-VL-4B reward | 默认 frozen reward model r_ψ | o_t、ℓ → [0,1] | 主实验默认；reward swap 只在 LIBERO-Long | 论文称使用公开模型资产 | 不是策略或 WM 的记忆机制 |
| LLaVA-CoT-7B / InternVL-reward-4B | reward-model swap | 同上 | 只用于 Table 10 的 LIBERO-Long 检查 | 同上 | 只归属 reward swap，不外推到全部实验 |
| DRILL-VWI | 低成本 signed surrogate | demo anchors + imagined GRPO 统计 → WM 更新 | 不额外增加 simulator rollouts over WMPO；使用 FAISS visitation 估计 | 论文称代码/checkpoints 将开源；本次来源无 repo URL | 使用同一批 imagined rollouts；不把 H 或 H_a 写成 memory |
| DRILL-IMG | 截断元梯度 | imagined GRPO + B_real 条 simulator rollouts → WM 更新 | 每个 outer step 用 B_real=30 条 simulator rollouts；K=2 truncated HVP | 同上 | K 是 BPTT 截断深度，不是历史上下文或记忆窗口 |

### 1.2 关键机制

- **目标错位**：WMPO 类流程先用像素 MSE 拟合 world model，再在模型内做 GRPO。像素 MSE 对所有像素、所有时间均匀惩罚；policy gradient 只关心当前策略访问的状态与 advantage 大的动作。两者权重不同。
- **双层目标**：内层在 Wφ 内做一次 GRPO，得到 θ⁺(φ)；外层用目标仿真器 P_env 的回报 J(π_{θ⁺(φ)}) 评价这次 WM 更新。关键是外层期望在 P_env，不在 Wφ——world model 不自己评价自己。
- **VWI 代理**：把像素 MSE 乘以三个 stop-gradient 系数：imagined visitation、signed group-relative advantage、WM confidence。正 advantage 的转移被锐化，负 advantage 的转移可被有意降权。
- **IMG 元梯度**：直接对 θ⁺(φ) 求导，第一因子是目标仿真器中的 policy gradient，第二因子是穿过 inner GRPO 的 mixed derivative，用 K-step truncated BPTT/HVP 估计。
- **PUR 诊断**：用像素 MSE 梯度与下游回报梯度的夹角，衡量“降像素误差”是否真的在改进策略。

⚡ **Eureka Moment**：world model 的“好”不应由它预测像素多准来定义，而应由“在它里面学一次之后，策略回到真实目标环境能变多好”来定义。

### 1.3 信息流/架构图

```text
已观测历史帧 o≤t = (o_0, ..., o_t)
        │
        ├── VLA policy πθ: 输出动作 chunk a_{t:t+H_a}
        │
        └── video WM Wφ: 预测未来帧 ô_{t+1}, ..., ô_{t+H}
                （H 是未来 rollout 长度，不是历史窗口）

DRILL outer loop:
  demo anchors ──► 在 Wφ 内 rollout H 步 ──► imagined τ
        │
        ├── GRPO advantage Â ──► θ⁺ = θ + ηθ ∇θ Ĵ_GRPO(πθ; Wφ)
        │
        ├── VWI 分支:
        │     w = d̂_φ^{πθ}(o_t) · sgn(Â)|Â|^α · s(o_t)
        │     L_VWI = E_demo [ w · ||ô_{t+1}(φ) − o_{t+1}||₂² ]
        │     φ ← φ − ηφ ∇φ L_VWI
        │
        └── IMG 分支:
              B_real 条 simulator rollouts ──► ∇θ⁺ J(π_{θ⁺})
              K-step truncated HVP ──► ∇φ∇θ Ĵ_GRPO
              φ ← φ + ηφ · ηθ · ∇θ⁺J · ∇φ∇θĴ_GRPO
        │
        └── π ← θ⁺
```

---

## 2. 数学核心

📌 **Napkin Formula**：

```text
φ* = argmax_φ  J( π_{θ + ηθ ∇θ Ĵ_GRPO(πθ; Wφ)} )
```

目标不是最小化像素 MSE，而是最大化“内层 GRPO 更新后策略”在目标环境中的回报。

### 2.1 基线 WMPO：像素拟合 + 模型内 GRPO

```text
L_MSE(φ) = E_{(o≤t, a_t, o_{t+1}) ~ D_demo} || ô_{t+1}(φ) − o_{t+1} ||₂²

θ ← θ + ηθ ∇θ Ĵ_GRPO(πθ; Wφ, rψ)
```

| 符号 | 含义 |
|---|---|
| o≤t | 已观测历史帧，不是未来 |
| o_{t+1} | 下一帧预测目标；H 步未来从 t+1 到 t+H |
| Wφ | 参数为 φ 的视频 world model |
| rψ | frozen vision-language reward model |
| Ĵ_GRPO | 在 imagined rollouts 上的 GRPO surrogate |

**直觉**：第一式让 WM 对所有 demo 像素一视同仁；第二式却只通过 policy gradient 使用 WM。两者优化的“误差位置”不一致。

### 2.2 DRILL 双层目标

```text
θ⁺(φ) = θ + ηθ ∇θ Ĵ_GRPO(πθ; Wφ)

φ* = argmax_φ J(π_{θ⁺(φ)})

J(π) = E_{τ ~ P_env(π)} [ Σ_t r_t ]
```

| 符号 | 含义 |
|---|---|
| θ | VLA policy 参数 |
| φ | world model 参数 |
| ηθ | inner policy step size |
| P_env | 目标仿真器轨迹分布；本文所有实验都是仿真器 |
| J | 目标环境中的期望回报，不是 WM 自评回报 |

**条件性分析**：这个目标把 WM 当成“策略改进环境”，而不是“视频生成器”。若 P_env 与真实机器人差异大，外层信号是否仍对准真实任务，本文没有回答。

### 2.3 DRILL-VWI：signed visitation-weighted surrogate

```text
L_VWI(φ) = E_{(o_t, a_t, o_{t+1}) ~ D_demo} [
    w(o_t, a_t) · || ô_{t+1}(φ) − o_{t+1} ||₂²
]

w(o_t, a_t) = d̂_φ^{πθ}(o_t) · sgn(Â(o_t, a_t)) · |Â(o_t, a_t)|^α · s(o_t)
```

| 符号 | 含义 |
|---|---|
| d̂_φ^{πθ}(o_t) | model-rollout visitation 估计；实现用 FAISS k-NN |
| Â(o_t, a_t) | group-relative GRPO advantage，stop-gradient |
| α | signed advantage 指数；默认 α=1 |
| s(o_t) | WM confidence/stability 正系数 |
| w | 实现中会 clip、normalize、stop-gradient；正文核心理论用未 clip 版本 |

**直觉**：不是“所有像素误差都重要”，而是“当前策略常去、advantage 大、WM 有把握的转移更重要”。负 advantage 的转移可被降权，甚至允许局部把 WM score 推离被判定有害的转移。

### 2.4 DRILL-IMG：截断元梯度

```text
∇φ J(π_{θ⁺(φ)}) =
    ηθ · ∇θ⁺ J(π_{θ⁺}) · ∇φ∇θ Ĵ_GRPO(πθ; Wφ)
```

| 符号 | 含义 |
|---|---|
| ∇θ⁺ J | 目标仿真器中的 policy-gradient 估计；用 B_real 条 simulator rollouts |
| ∇φ∇θ Ĵ_GRPO | 穿过 inner GRPO 的 mixed derivative；用 K-step truncated HVP |
| B_real | 每个 outer step 的 simulator-rollout 批次；主实验 30 |
| K | BPTT 截断深度；主实验 K=2 |

**直觉**：IMG 更“忠实”地估计外层元梯度，但要多付 simulator interaction 与 HVP memory。VWI 是局部代理；IMG 是多步轨迹效应的更直接估计。

### 2.5 理论边界：VWI 不是精确等式

Proposition 1 的核心界：

```text
|| ∇φ J(π_{θ⁺(φ)}) + (κ ηθ / (2 σ_W²)) ∇φ L_VWI_core(φ) ||
≤ ηθ M_W ε_comp + ηθ R_H + ηθ R_demo + O(ηθ²)
```

| 符号 | 含义 |
|---|---|
| κ | 局部 policy-score contraction 的 alignment scalar，要求 κ>0 |
| σ_W | Gaussian WM head 的尺度 |
| M_W | per-sample WM score 的范数上界 |
| ε_comp | score-compatibility residual |
| R_H | 省略多步 trajectory-score 的 locality residual |
| R_demo | demo-anchor 与 on-policy imagined transition 的 mismatch |

**读法**：VWI 在 score-compatibility 成立、trajectory-locality 与 demo-anchor residual 小的时候，恢复 IMG 元梯度的局部一步分量；它不是全局精确恒等式。Gaussian head 是证明里的关键假设；diffusion/flow WM 只能写“analogous score objective”，不能直接照搬像素 MSE 恒等式。

Theorem 1 的收敛界：

```text
min_{0≤t<T} E || ∇φ J̄θ(φ_t) ||²
≤ O( L(J* − J0) / T )
  + O( sqrt( L(J* − J0) σg² / T ) )
  + O( B² ρ^{2K} )
```

| 符号 | 含义 |
|---|---|
| J̄θ | 固定当前 policy 后的 conditional outer objective |
| L | J̄θ 对 φ 的 smoothness 常数 |
| σg | 截断元梯度估计的 conditional variance 上界 |
| B, ρ | 截断偏差满足 ||bias_K|| ≤ B ρ^K |
| T | outer iterations |
| K | BPTT 截断深度 |

**注意**：这是“固定策略子问题”的收敛到邻域保证，不是联合 (θ, φ) 过程的全局收敛证明。论文也明确说 K=2 是经验 return-per-compute 选择，不是从最坏界读出来的；在 Appendix F.5 的保守常数下，K=2 的截断项甚至高于 stochastic floor，界是松的。

### 2.6 PUR：像素 MSE 与策略改进的夹角

```text
PUR(φ) = 1 − < ∇φ L_MSE(φ), −∇φ J(π_{θ⁺(φ)}) >
          / ( ||∇φ L_MSE|| · ||∇φ J|| )
```

等价地，PUR = 1 + cos angle(∇φ L_MSE, ∇φ J)。

| 值 | 含义 |
|---|---|
| PUR ≈ 0 | 降像素 MSE 与改策略同向 |
| PUR ≈ 1 | 两方向近似正交 |
| PUR ≈ 2 | 降像素 MSE 与策略改进反 aligned |

**论文实测**：在报告 PUR 的 headline rows 上，PUR 与 downstream SR 的 Spearman 相关为 ρ_S = −0.89，95% bootstrap CI 为 [−0.97, −0.71]；两个 held-out variants 用 step 5k 的 checkpoint-level PUR 预测 final SR，平均绝对误差 1.5 个百分点。PUR 是诊断，不是训练目标。

### 2.7 时序算术与记忆边界

```text
已观测历史：o≤t = (o_0, ..., o_t)
未来预测：o_{t+1}, ..., o_{t+H}
从当前参考 t 到未来终点：H · Δt

均匀间隔 Δt 下，K 个采样帧的首末跨度：
(K − 1) · Δt

动作 chunk：a_{t:t+H_a}，长度 H_a
BPTT 截断深度：K
```

**假设性算术示例（非论文实验）**：若 K=5、Δt=2，则首末跨度 = (5−1)×2 = 8；若 H=50、Δt=1，则终点距当前参考 = 50×1 = 50。验算仅说明“采样帧跨度”和“未来预测终点”不是同一个量。

**关键边界**：H 是 imagined rollout 的未来 horizon；K 是 BPTT 截断深度；H_a 是 policy 输出的 action chunk 长度。三者都不能直接当作历史上下文或 memory window。论文没有给出统一的固定历史帧数；H=50、H_a=16、K=2 是默认设置，不是所有实验共有的记忆窗口。连续未来 chunk 的预测区间也不能倒推成历史上下文。

---

## 3. 带数字走一遍：玩具例子

> **假设性玩具例子（非论文实验数据）**。只用于说明 signed VWI 的算术，不代表论文结果。

取三个 demo transitions，理想化核心 VWI：α=1、s=1、d̂=1、不 clip、不 normalize。

| 转移 | 平方误差 E=||ô−o||₂² | Â | w=sgn(Â)|Â| | w·E |
|---|---:|---:|---:|---:|
| A | 1.00 | +1.0 | +1.0 | +1.00 |
| B | 0.25 | −1.0 | −1.0 | −0.25 |
| C | 0.04 | 0.0 | 0.0 | 0.00 |

验算：

```text
MSE  = (1.00 + 0.25 + 0.04) / 3 = 1.29 / 3 = 0.43
VWI  = (1.00 − 0.25 + 0.00) / 3 = 0.75 / 3 = 0.25
```

再假设模型 M2 的误差账本变为 A=0.80、B=0.50、C=0.04：

```text
M2 MSE = (0.80 + 0.50 + 0.04) / 3 = 1.34 / 3 ≈ 0.447
M2 VWI = (0.80 − 0.50 + 0.00) / 3 = 0.30 / 3 = 0.10
```

M2 的 MSE 略高（0.447 > 0.43），但 VWI 更低（0.10 < 0.25）。在这个假设账本里，两个目标给出不同排序。它说明 VWI 不是 MSE 的简单重排；但**不能**由此推断 M2 的策略一定更好——那需要目标环境的 J，正是 DRILL 外层要估计的量。

---

## 4. 工程视角

| 维度 | DRILL-VWI | DRILL-IMG |
|---|---|---|
| 额外 simulator rollout | 相对 WMPO 不增加 | B_real=30 条/outer step |
| 额外计算 | 主要是 FAISS visitation 估计；论文称 k-NN 开销 <2% 相对 inner-loop forward | truncated HVP + simulator rollouts；论文称每 outer step 约 1.4× 成本 |
| 内存 | 无 BPTT 大图 | K=2、H=50、1.2B WM 时，峰值 activation 约 K·|W| 而非 H·|W|；论文称相对 full BPTT 节省约 25× memory |
| 适用 horizon | 短 horizon 更划算；H=5 时 IMG−VWI 为 −0.2±1.1 | 中长 horizon 更划算；H=100 时 IMG 83.4 vs VWI 75.9 |
| 实现注意 | signed weight 要 stop-gradient、clip、normalize；负系数允许但受 clip 控制 | outer gradient clip 10；HVP 用 float32 累加；WM normalization stats 冻结 |
| 评测频率 | 可用同一 imagined rollouts | PUR 每 M=500 outer iterations 记录一次；每次 2r·B_real=7680 条 simulator rollouts，成本高 |

**论文实测的计算账**：headline 实验总计约 17,040 H20-hours，24 张 H20 约 4 周；其中 WM pretrain 7,200、inner loop 4,800、outer loop 3,840、evaluation 1,200。DRILL-IMG 相对已有 WMPO pipeline 的边际成本主要是 outer-loop 的约 3.8k H20-hours。不要把 17k 全部归给 DRILL-IMG 的 outer update；pretrain 与 inner loop 和基线共享。

**工程含义**：如果团队已有 WMPO 风格 pipeline，VWI 是最低摩擦替换；如果 horizon 中等偏长且有 simulator budget，IMG 的额外 outer-loop 成本才可能值得。若只有真实机器人、没有高保真仿真器，本文没有给出可外推的证据。

---

## 5. 数据与评测

### 5.1 预训练与资产

- **离线数据**：训练使用 offline demonstrations；WMPO 类基线先在 demo 像素上拟合 WM。DRILL 替换的是 WM 的更新目标，不是 policy 的预训练数据。
- **WM 预训练与 DRILL 微调分开**：Table 4 把 WM pretrain 与 inner/outer loop 分开计费；DRILL-IMG 可以叠加在 RLVR-World 预训练的 WM 上，论文称这样做与 prediction-correctness pretraining 互补，而不是替代。
- **数据开放范围**：五个 benchmark 公开；π0-3B、OpenVLA-7B 按公开 release 使用，不重新分发；论文称 DRILL 代码与 headline checkpoints 将开源，但本次来源未给 repo URL，本文不编造仓库地址。
- **真实机器人**：无。Appendix E.4 明确所有定量结果在仿真器；没有 real-robot 结果。

### 5.2 仿真评测

**设置**：五个 manipulation simulators——LIBERO-Long、CALVIN ABC→D、RoboCasa、Meta-World MT50、SimplerEnv；两个 VLA backbones——π0-3B 主实验、OpenVLA-7B 复现；headline 用 5 seeds，ablations 用 3 seeds；报告 Welch t-test p 值与 95% CI。

| 方法 | LIBERO | CALVIN | RoboCasa | MT50 | Simpler | 说明 |
|---|---:|---:|---:|---:|---:|---|
| WMPO | 64.2±1.6 | 71.8±1.3 | 38.5±2.1 | 67.4±1.5 | 55.1±2.0 | 像素 MSE 基线 |
| RLVR-World | 70.8±1.4 | 75.3±1.6 | 42.7±1.8 | 72.9±1.5 | 61.6±1.9 | prediction-correctness 预训练 WM |
| DRILL-VWI | 76.1±1.2 | 80.7±1.1 | 51.8±1.7 | 75.6±1.4 | 64.0±1.9 | 无额外 simulator rollout |
| DRILL-IMG | 81.3±1.3 | 83.1±1.4 | 55.7±2.0 | 79.4±1.2 | 63.1±2.6 | B_real=30、K=2 |
| DRILL-IMG+RLVR-WM | 83.9±1.1 | 85.4±1.2 | 59.1±1.6 | 81.8±1.3 | 67.3±2.1 | 叠加 RLVR-World WM |

**口径提醒（重要）**：DRILL-IMG 相对 WMPO 的 “13.1 个百分点”是五个 benchmark 的平均增益，不是 LIBERO-Long 的 1.2B 增益。LIBERO-Long 上 DRILL-IMG 81.3 vs WMPO 64.2，差值是 17.1 个百分点；CALVIN 11.3、RoboCasa 17.2、MT50 12.0、SimplerEnv 8.0，五者平均约 13.1。相对 RLVR-World 的平均增益约 7.9 个百分点。one-step pixel MSE 方面，DRILL-IMG 为 9.8×10⁻³，WMPO 为 8.4×10⁻³，约高 17%。

**compute-matched control（论文实测）**：LIBERO-Long 上把 WMPO 从 10k outer steps 延长到 14k（1.4× compute）只到 65.1±1.9，延长到 20k（2.0×）为 64.4±2.2；DRILL-IMG 10k 为 81.3±1.3。作者主张这排除了“纯算力解释”。

**horizon sweep（论文实测）**：

| H | WMPO | DRILL-VWI | DRILL-IMG | IMG−VWI |
|---:|---:|---:|---:|---:|
| 20 | 67.4±1.5 | 72.6±1.3 | 74.4±1.5 | 1.8 |
| 50 | 64.2±1.6 | 76.1±1.2 | 81.3±1.3 | 5.2 |
| 100 | 58.0±2.2 | 75.9±1.5 | 83.4±1.7 | 7.5 |
| 200 | 49.8±2.8 | 70.5±2.1 | 78.6±2.5 | 8.1 |

短 horizon 检查：H=10 时 IMG−VWI 约 0.6；H=5 时为 −0.2±1.1；SimplerEnv 的 effective horizon 较短，VWI 64.0 略高于 IMG 63.1。作者据此建议：很短 horizon 或没有 simulator rollouts 时用 VWI；中长 horizon 且可接受 B_real=30 时用 IMG。

**OOD 与诊断（论文实测）**：LIBERO zero-shot perturbations 下，DRILL-IMG 平均相对 SR drop 为 18.6%，WMPO 30.2%，RLVR-World 25.6%；唯一例外是 pose perturbation，DRILL-VWI 18.0 略优于 IMG 18.9。TRPS 上，WMPO 0.61±0.02，DRILL-IMG 0.81±0.03，尽管后者 pixel MSE 更高。human forced-choice：120 trials、6 名 blinded raters；视觉质量偏好不显著，但展示任务完成情况后 62% 偏好 DRILL-IMG over WMPO，p<0.001。

**跨 backbone（论文实测）**：OpenVLA-7B 上 DRILL-IMG 平均 SR 68.9，WMPO 53.8，差 15.1；LIBERO-Long 77.1、CALVIN 78.9、RoboCasa 50.8。RoboCasa 上 DRILL-IMG 50.8±2.3 与 VWI 48.7±2.1 统计打平，p=0.12；作者归因于该 benchmark episode 较短、object-centric。

### 5.3 真实机器人设置

**无**。论文明确 simulation-only；没有 real-robot 结果，也没有 sim-to-real 实验。所有关于真实部署的推断都超出本文证据。

---

## 6. 能力与失败模式

### 6.1 能力边界

| 场景 | 论文证据 | 条件性建议 |
|---|---|---|
| 中长 horizon（H=100） | IMG 83.4 vs VWI 75.9 vs WMPO 58.0 | 有 simulator budget 时优先 IMG |
| 很短 horizon（H=5/10） | IMG−VWI 为 −0.2±1.1 / 约 0.6 | 优先 VWI，或至少不要假设 IMG 一定更好 |
| 短 effective horizon（SimplerEnv） | VWI 64.0 vs IMG 63.1 | VWI 可作为默认 |
| RoboCasa + OpenVLA | IMG 50.8 vs VWI 48.7，p=0.12 | 统计打平，不应宣称 IMG 全面胜出 |
| 小 WM（300M） | IMG 相对 WMPO +7.9，但 RoboCasa 5 seeds 下不显著 | 小模型不要直接套用 1.2B 结论 |
| OOD pose perturbation | VWI 18.0 vs IMG 18.9 relative drop | IMG 不是每个扰动条件都最好 |
| reward model 换用 | LIBERO-Long 上 IMG−WMPO 在 15.6–18.4 之间 | 增益方向可保持，幅度不保证恒定 |

### 6.2 隐含假设（Hidden Assumptions）

1. **Gaussian WM head**：Proposition 1 的闭式 score 恒等式依赖 Gaussian observation head；diffusion/flow WM 只有“analogous”版本，不能把像素 MSE 恒等式直接当定理。
2. **Advantage stop-gradient、clip inactive、visitation 一步平稳**：理论把 GRPO advantage 当常数，把 clip 激活与 occupancy drift 收进 residual。真实训练中这些 residual 不一定小。
3. **目标仿真器就是 J 的 ground truth**：外层 J 在 P_env 中估计；若 P_env 与真实机器人差距大，DRILL 的“下游回报”只是仿真器回报。
4. **H、K、H_a 都不是记忆窗口**：论文没有给统一的固定历史长度；H=50 是默认未来 rollout horizon，K=2 是 BPTT 深度，H_a=16 是动作 chunk。不能把任一设置写成所有实验共有的 memory window。
5. **IMG 不是把 inner problem 解到收敛**：它用短 unroll 的 truncated BPTT；Theorem 1 是固定策略子问题的邻域保证，不是全局收敛。
6. **Human evaluation 的治理缺口**：论文 checklist 对 IRB 回答 No，正文未报告批准/豁免；这是待补项，不影响技术结论，但影响发布合规判断。

---

## 7. 与相关工作对比

| 方法 | WM 训练目标 | 是否用 simulator rollout 更新 WM | 策略更新位置 | 关键备注 |
|---|---|---|---|---|
| WMPO | 单步像素 MSE | 否（按论文描述） | 在 Wφ 内 GRPO | 像素保真与策略效用错位的基线 |
| RLVR-World | prediction correctness reward | 否（按论文描述） | 不直接为 policy update 优化 WM | 提升 predictor；与 DRILL 互补 |
| VLAW | 论文列为 policy/WM co-improvement | 正文未展开 | 迭代共同改进 | 不应把 DRILL 的属性归给它 |
| Policy-Aware WM / ESNR | 诊断指标 | 不提供训练目标 | 诊断 | 说明像素保真不跟踪策略能力 |
| DRILL-VWI | signed visitation/advantage/confidence 加权 MSE | 相对 WMPO 不增加 | 同一批 imagined GRPO | 低成本局部代理 |
| DRILL-IMG | 下游回报元梯度 | 是，B_real=30/outer step | imagined GRPO + simulator outer gradient | 更忠实但更贵；K=2 HVP |

**面试 Tip**：被问 DRILL 的核心，不要先背 pixel MSE。先说“WM 的优劣应由它诱导的策略更新在目标环境中的回报来评价”；再补“VWI 是局部 signed 代理，IMG 是截断元梯度；短 horizon 先上 VWI，中长 horizon 再考虑 IMG”。最后提醒“13.1 是五 benchmark 平均，不是 LIBERO-Long 的 1.2B 增益；后者是 17.1”。

---

## 8. 精读建议

- **值得精读原文的人**：
  - 正在做 world-model-based RL 后训练 VLA 的研究者：重点看 §3.2–3.4 与 Appendix D.2，理解 VWI/IMG 的接口。
  - 要评估“能否在现有 WMPO pipeline 上低成本替换”的工程师：先看 §4 与 §5.4，判断 VWI 是否够用。
  - 做 policy-aware world model 理论的人：看 §4.1–4.2 与 Appendix B/C，注意 Gaussian head 与固定策略子问题的边界。

- **建议章节路径**：先读 §1.2–1.3 建立信息流 → 再看 §3.2–3.4 分清 VWI/IMG → 然后 §4.1–4.2 看假设与收敛边界 → 最后 §5.4 + Appendix F.3 看 horizon 条件。若只关心落地，可跳过 Appendix B/C 的证明细节。

- **不值得精读的理由**：如果你只做真实机器人部署、没有高保真仿真器，或只关心视频生成质量，本文的证据距离较远；读摘要与 §5.3 的 simulation-only 声明即可。

---

[← Back to Theory](./README.md)

**关键引用**：[DRILL 论文 arXiv HTML v1](https://arxiv.org/html/2610.30814)