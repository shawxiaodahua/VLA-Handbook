# 保留未来，丢掉推演：RIFT 让世界动作模型一步生成未来缓存 (Keep the Future, Drop the Rollout: RIFT for World Action Models)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-29
>
> **论文**: Keep the Future, Drop the Rollout: RIFT for World Action Models
> **链接**: https://arxiv.org/abs/2608.11521
> **作者机构**: Australian National University · FreiNexus · Beijing Normal University
> **核心定位**: 回答一个被长期混淆的问题——世界动作模型（WAM）生成动作时，真正需要的到底是"未来视频"本身，还是仅仅是它的内部表征？答案指向后者，于是提出用一个前向 pass 直接写出未来 K/V 缓存，砍掉部署期的迭代视频推演。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | WAM 的动作专家消费的是"未来 K/V 缓存"，而非迭代推演的轨迹；用一组可学习的 anticipation token 单次前向就能写出这个缓存，成功率高过 rollout 基线，延迟降 3.1–9.2× |
| 適合精讀 | 做 WAM / 世界模型 + VLA 推理加速的研究者；要把迭代视频生成从部署路径里拿掉的系统工程师 |
| 可以跳過 | 只关心纯 VLA（无未来预测分支）或非 WAM 骨架的读者，距离中等 |
| 落地可行性 | 中高——保留 Wan2.2-5B 骨架与训练配方，只加 anticipation token 与两个辅助损失；但真实机器人验证仅限单一平台 |
| 主要風險 | 真机评测仅 Galaxea R1 Lite 且绝对成功率偏低（45.3%）；干预结论只在"已评估的 co-denoising 配置"内成立 |

💡 **X-Ray 開場**
先问三个问题。**它解决什么问题？** 世界动作模型靠"先预测未来、再据此出动作"提升成功率，但代价是部署时要迭代生成视频，延迟是纯当前观测策略的 3.3–9.6×。**它发现了什么？** 作者用因果干预证明：动作专家真正读取的只是未来 token 的 K/V 缓存；只要缓存内容对、位置对，即使整个缓存被换成一份"冻结的最终干净缓存"，执行轨迹几乎不变（EE-ADE 仅 1.9 cm，成功率 97.9% vs Oracle 98.4%）。**对 VLA 研究者意味着什么？** "未来表征"和"产生它的迭代过程"是可分离的——迭代推演只是构造缓存的工具，不是动作质量的来源。任何聪明的非专家读完这三点，就能复述核心。

📍 **研究全景时间線**

```
2023 Diffusion Policy ——> 2024-25 世界模型 × VLA 耦合 ——> 2026 WAM 显式未来条件
(动作扩散)              (UniPi / GR-2 / UWM 等)          (Joint 共去噪 vs IDM 先出视频再出动作)
                                                              ↓
2026.03 Fast-WAM 砍掉未来分支 / PFD 蒸馏未来修正 ——> [本文 RIFT 2026.09] ← 当前位置
(降延迟但丢掉显式测试期未来条件)                     (单 pass 写出完整未来 K/V，无 rollout)
                                                              ↓ 局限
                                              真机仅单平台；非 WAM 融合骨架未验证
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

### 1.1 系統對比概覽 (System Component Comparison)

| 模块 | 输入 | 输出 | 训练/推理差异 |
|------|------|------|----------------|
| 视频专家 (video expert, 共享) | 首帧 token f0(o) + anticipation token E + 语言指令 l | 每层 anticipation token 的 K/V | 训练用原生 video-flow 损失；部署只做一次 cache prefill |
| Anticipation token E ∈ R^{m×d} | 可学习参数，继承对应未来时空索引 | 未来 token 的 K/V 缓存 C_φ | 全对齐时 m = n·(T_lat − 1)，LIBERO 取 m=196 |
| 动作专家 (action expert) | 观测 o、指令 l、缓存 C_φ(o,l) | 动作块 â_{1:H} | H=32；10 步 action flow-matching；缓存无 action-flow 索引 |
| FM 辅助头 v_ψ | (X_σ, σ; S_φ) | 速度场 | 仅训练存在，不进入动作/策略部署 |
| L2 线性探针 g_ω | RMS(stopgrad(S_φ)) | 确定性未来 latent 读出 | 仅训练 + 可选监控，梯度不回流 |

### 1.2 關鍵機制 (Key Mechanism)

- **把"未来"降级为缓存，而非视频。** WAM 动作专家在每一层通过注意力读取未来 token 的 K/V。缓存是"给定观测 o、指令 l、视频生成随机性"下**与动作无关**的干预位点（视频 token 从不注意动作 token），因此可以"录制—重放"来做因果实验。
- **一次前向填满缓存。** 用与未来时空位置对齐的可学习 anticipation token 替代被 rollout 出的未来 token，单次经过视频骨干即可得到每层完整 K/V，部署时不做视频扩散、不做 VAE 解码。
- **用分布型目标而非点回归塑造 anticipation 状态。** 对多模态未来直接做 L2 回归会收敛到条件均值（把多个合理未来"平均掉"），因此引入 conditional flow matching 作为辅助分布目标；同时保留一个 stopped-gradient L2 探针，仅作确定性读出与可选监控。
- **训练/部署对齐。** 第二个前向与部署路径同构（输入 [f0;E] 与相同注意力掩码），并施加首帧扰动，使策略对首帧噪声鲁棒。

⚡ **Eureka Moment**：WAM 的动作专家读的是未来 K/V 缓存的内容与空间/时间组织——而不是缓存必须逐去噪步演化的那条轨迹；因此未来条件可以在**一次前向**里"写"出来，rollout 从部署路径中彻底移除。

### 1.3 信息流/架構圖 (Flow / Diagram)

```
部署 (policy-only, 单 pass 缓存):
   o ──► f0(o) ┐
               ├─► CachePrefill_φ ──► C_φ(o,l) = {(K_E^(ℓ), V_E^(ℓ))}_{ℓ=1..L}
   E (可学习) ─┘         (1 次骨干 pass, 无 rollout / 无 VAE 解码)
                                           │
   [o, l] ────────────────────────────────┤
                                           ▼
                              ActionDenoise(o, l; C_φ)  ──►  â_{1:H}
                              (10 步 action flow, 每步复用同一缓存)

训练 (每步 2 个前向, 共享视频专家):
  Forward A: [干净首帧, 加噪未来 latent] ──► L_vid        (原生 video-flow)
  Forward B: [f0;E] + 部署同款掩码 ──► 缓存 ──► L_act      (动作流匹配, 仅未扰动样本)
                                      S_φ ──► L_FM       (conditional flow matching)
                                      S_φ ──► L_probe    (stopgrad L2 探针, 不回传骨干)
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
C_φ(o,l) = CachePrefill_φ([f0(o); E], o, l)   ⇒   â_{1:H} = ActionDenoise(o, l; C_φ(o,l))
```

一句话：**用一组学习到的未来 token 单次前向写出缓存，动作专家对这份固定缓存做去噪。**

**目标**：让动作条件化于一个"完整的未来表征"，但不产生迭代视频 rollout 的成本。

**核心方程**：

```
缓存构造（一次前向）:
  C_φ(o,l) = { (K_E^(ℓ), V_E^(ℓ)) }_{ℓ=1..L} = CachePrefill_φ([f0(o); E], o, l)

动作生成:
  â_{1:H} = ActionDenoise(o, l; C_φ(o,l))

conditional flow matching 辅助损失:
  X_σ = (1−σ)·Y + σ·ε ,    v*_σ = ε − Y
  L_FM = E_{Y,ε,σ}[ w_vid(σ) · || v_ψ(X_σ, σ; S_φ) − v*_σ ||²_2 ]

stopped-gradient L2 探针:
  Ŷ_L2 = g_ω( RMS( stopgrad(S_φ) ) )
  L_probe = mean( (Ŷ_L2 − Y)² )

总目标:
  L = L_vid + L_act + λ_FM·L_FM + λ_probe·L_probe
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| o, l | 观测（多相机图像）与语言指令 |
| f0(o) | 从观测 o 提取的首帧 token |
| E ∈ R^{m×d} | m 个可学习 anticipation token，宽度 d |
| m | token 数；全对齐 m = n·(T_lat − 1)，LIBERO m=196，RoboTwin 2.0 m=240 |
| L | 视频骨干层数 |
| C_φ | 每层未来 K/V 缓存 |
| S_φ ∈ R^{m×d} | 部署对齐前向产出的最终 anticipation 状态 |
| Y | 对齐后的真值未来 latent patch |
| v_ψ | 仅训练的 FM 头；g_ω 为 L2 探针 |
| λ_FM, λ_probe | 前 70% 课程为 1，后 30% 余弦衰减到 0.2 |
| H | 动作块视界，H=32 |

**直觉**：K/V 缓存是自注意力机制的"记忆"。视频 rollout 花了 3–9× 的时间去逐步雕刻这份记忆，而实验表明动作只需要最终那份记忆。RIFT 于是把"雕刻过程"换成"一次写满"——用一组会自己学出未来语义的 token，直接生成记忆本身。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

**场景**：LIBERO 某任务，动作专家在去噪动作时读取未来缓存。

**第 1 步 —— 干预实验，判断动作"读了什么"**（论文 §3，2,000 配对试验 × 40 任务）：

```
Oracle（未修改）:        成功率 98.4%
掩蔽对未来 token 的注意力:  EE-ADE 18.7 cm,  成功率 98.4% → 9.7%
空间置换未来 value:         EE-ADE 14.3 cm,  成功率 65.2%
时间交换未来 value:         EE-ADE 15.6 cm,  成功率 0.7%
替换为冻结的最终干净缓存:    EE-ADE  1.9 cm,  成功率 97.9%
```

解读：内容重要（掩蔽/加噪会崩），**位置也重要**（空间置换 14.3 cm 但成功率仍有 65.2%；时间交换同为 ~16 cm 却只剩 0.7%——说明动作对"哪个未来帧"很敏感）。但把整个演化缓存换成一份固定的最终缓存，轨迹几乎不动。→ 结论：**迭代轨迹不是必需，最终表征才是。**

**第 2 步 —— 用一次前向写出这份缓存**：

假设 LIBERO 上每帧 latent 有 n 个 token、clip 有 T_lat 帧，全对齐取 m = 196 个 anticipation token。一次 CachePrefill 之后，动作专家读到与 rollout 近乎等价的缓存，用 10 步 action flow 去噪出 H=32 的动作块。

**第 3 步 —— 效果**：

```
LIBERO 整体成功率:   RIFT 98.8%   vs  rollout 基线 98.4%–98.6%
LIBERO 每块延迟:     RIFT 247.9 ms vs  rollout 基线 (降 68.2%–89.1%)，Fast-WAM 235.7 ms
加速比:              3.1×–9.2×（相对已评估的 rollout 方法）
```

注意一个反直觉点：RIFT **成功率更高**，同时延迟逼近纯当前观测的 Fast-WAM（235.7 ms）——又快又准，不是"用精度换速度"。

## 4. 工程視角 (Engineering View)

| 维度 | 数值 / 含义 | 工程含义 |
|------|-------------|----------|
| 每块延迟 (LIBERO, A800) | 247.9 ms（含 cache prefill + action 去噪） | 与 current-only Fast-WAM 235.7 ms 接近；延迟瓶颈从"视频去噪"移到"动作去噪" |
| 真机延迟 (RTX 4090) | RIFT 420 ms vs Fast-WAM-Joint 1069 ms | 比 rollout 版低 60.7%；与 Fast-WAM 404 ms 相当 |
| 缓存复用 | 同一 C_φ 供每个动作去噪步复用，无 action-flow 索引 | 一次 prefill 摊销到 10 步去噪，省下逐步重算 |
| 部署路径 | 无视频扩散、无 VAE 解码 | GPU 上省掉整条视频生成子系统；显存与算子图显著简化 |
| 掩码约束 | 视频 token 从不注意 action token → 缓存与动作无关 | 缓存可预计算/缓存化，利于流水线并行与批处理 |
| 步数 | 10 步 action flow-matching；CFG=1.0 | 推理步数固定，抖动小 |
| 可选监控 | L2-FM 分歧 + CUSUM 每 10 个环境步一次 | 监控开销不计入报告延迟，可关 |

**trade-off 要点**：RIFT 把"是否做视频生成"从一个每次推理都要付的成本，变成一个**训练期的构造任务**。代价是需要两个前向/步训练、并新增 anticipation token、FM 头、探针三处参数与损失；收益是部署路径不再包含迭代视频生成。对 2GB RAM 这类边缘场景，省掉的是整条视频解码与多步扩散的显存/时间开销。

**不确定性监控（可选）**：用 L2 探针点估计与 FM 采样的归一化分歧驱动 CUSUM：

```
d_ratio = (D⁻¹·|| μ_L2 − x̄_FM ||²_F) / ((KD)⁻¹·Σ_i || x_i^FM − x̄_FM ||²_F + ε)
S_n = max(0, S_{n−1} + d_n − μ_d − 0.25·σ_d)
告警条件: S_n > η
```

校准（1,967 个成功回合，排除 33 个失败）：μ_d=0.5408, σ_d=0.1114, η=6.74。平均 CUSUM 在 t=210 越过阈值，而回合终点在 t=420——即提前约一半回合长度告警。注意这是"告警统计量"，不是失败概率：两个头可能同时同意同一个错误未来。

## 5. 數據與評測 (Data & Eval)

| 基准 | 设定 | 结果 | 对比 |
|------|------|------|------|
| LIBERO | 40 任务（Spatial/Object/Goal/Long），3 次运行 × 2,000 试验 | 98.8% 整体 | rollout 基线 98.4%–98.6%；Fast-WAM 2.0pp↑、PFD 1.5pp↑ |
| LIBERO-Plus (OOD) | 10,030 变体，不额外训练，每变体 1 rollout | 81.1% | +9.7pp vs Fast-WAM-IDM；7 个扰动类、5 难度级、4 源套件全面领先 |
| RoboTwin 2.0 | 50 双臂任务，clean / randomized | 92.9 / 92.6 | PFD 92.5/92.1；LingBot-VA 92.4/91.4；Fast-WAM 91.9/91.6 |
| 真机 | Galaxea R1 Lite，3 任务，每任务 50 试验 | 45.3% 平均 | Fast-WAM-Joint 39.3%、Fast-WAM 26.7%（+6.0pp） |

**训练配比与骨干**：全部内部模型共享 Wan2.2-5B 的预训练视频 DiT、文本编码器、视频 VAE；动作专家复用视频分支，d_a=1024（1B 动作 / 6B 总参），H=32，视频时间下采样 4× 到每块 9 帧。LIBERO 训 20k 步，RoboTwin 2.0 训 30k 步（2,500 clean + 25,000 randomized 示范），AdamW 1e-4。

**消融**：
- **监督方式**：Rift-L2（直接 L2 回归）98.37±0.12 vs Rift（conditional FM）98.8±0.17，同图同 247.9 ms——差异纯来自监督目标，+0.4pp。
- **token 数 m**：从 m=2 扫到全对齐 m=196；无缓存的 Fast-WAM（m=0）为 96.75%。Rift-L2 从 97.08% 升到 98.37%；conditional FM 从 m=4 起反超，峰值 98.78%。全对齐对两种监督都最优。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 在 LIBERO（98.8%）、LIBERO-Plus OOD（81.1%）、RoboTwin 2.0（92.9/92.6）三个仿真基准上取得所评估方法中的最高成功率，且部署路径无 rollout。
- 真机三任务（堆篮、物体分类、试管混合）全部取得最高成功数；延迟 420 ms/块（对比 Fast-WAM-Joint 1069 ms）。
- 提供 L2-FM 分歧作为可选"早期失败预警"，平均提前约 210 环境步。

**不能做 / 失败模式**：
- **真机绝对成功率仍低**：45.3% 平均，其中 Tube Mixing 仅 10/50。桌面操作可行，但远未到"通用可用"。
- **真机平台单一**：仅 Galaxea R1 Lite，三相机视角；作者在附录 G 明确承认迁移到其他实体平台与"非 WAM 融合骨架"是未来工作。
- **干预结论有边界**：单份冻结缓存"几乎无损"只在**已评估的 co-denoising 配置**（Joint、Cosmos-2）成立；IDM / LingBot-VA 因结构上本就使用固定缓存，该干预是 N/A。跨模型的量级只是描述性，不能直接比较。
- **缓存仍依赖视频骨干的质量**：RIFT 保留原生 video-flow 监督，若上游视频世界模型本身不准，未来表征也会受限（论文未做此压力测试）。
- **监控非失败概率**：d_ratio 大只说明两个估计器分歧，二者可能同时错。

### 6.1 隱含假設 (Hidden Assumptions)

作者默认成立、但未直接证伪的前提：
1. **"未来表征可分离于其构造过程"具有普适性** —— 只在少数 co-denoising WAM 上验证；对 generate-then-act IDM 类结构，该问题本身被构造性地绕过。
2. **单份冻结缓存足够** —— 假设动作对缓存的时间演化不敏感；但时间交换干预（成功率骤降到 0.7%）说明"哪些帧"高度敏感，这两者的张力论文未完全厘清。
3. **视频 token 不注意力于动作 token** 是架构不变式 —— 一旦未来融合骨架改变掩码，缓存的 action-independence 与"录制—重放"干预协议都会失效。
4. **OOD 泛化可由单一 benchmark 代表** —— LIBERO-Plus 的 10,030 变体被视为 OOD 强度的充分代理。
5. **L2-FM 分歧是可校准的失败信号** —— 用成功回合做 conformal 校准，本身不给出独立的误报率估计。

## 7. 與相關工作對比 (Comparison)

| 方法 | 测试期未来条件 | 部署是否 rollout | 缓存构造 | 关键点/结果 |
|------|----------------|------------------|----------|-------------|
| Fast-WAM | 无（训练期共训后丢弃未来分支） | 否 | 无 | 235.7 ms；LIBERO ~96.75% |
| Fast-WAM-Joint | 有（演化缓存） | 是（共去噪） | 逐步演化 | rollout 基线，RoboTwin 91.0/91.1 |
| Fast-WAM-IDM | 有（固定最终缓存） | 是（先出视频再出动作） | 迭代生成后取最终 | LIBERO-Plus 71.4%，本文 +9.7pp 指向的对象 |
| PFD | 无（蒸馏未来修正） | 否 | 无 | LIBERO 少 RIFT 1.5pp |
| LingBot-VA | 有 | 是 | 迭代生成 | RoboTwin 92.4/91.4 |
| **RIFT (本文)** | **有（固定单 pass 缓存）** | **否** | **一次前向写满** | **LIBERO 98.8%，3.1–9.2× 加速** |

**面试 Tip**：被问"RIFT 和 Fast-WAM 差在哪？"一句话答——**Fast-WAM 把未来分支整个扔掉，RIFT 是"保留未来、只扔掉生成未来的迭代过程"**：用一个前向 pass 直接写出未来 K/V 缓存，所以既拿到未来条件带来的成功率优势，又拿到接近 current-only 的部署延迟。

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做 WAM / 世界模型 + VLA 的研究者，尤其是关心"未来条件在测试期到底有没有用"这一争论的；
  2. 需要评估把迭代视频生成从部署路径移除可行性的系统/工程团队；
  3. 对"用因果干预（而非仅探针/注意力权重）理解具身策略内部表征"感兴趣的方法论读者。
- **建議章節路徑**：先讀 §3（干预协议与发现，是全文论证地基）→ 再看 §4（RIFT 设计与训练目标）→ 然後 §5 表格（LIBERO / LIBERO-Plus / RoboTwin / 真机）→ 附錄 A/E（训练细节与监控）可按需查證，§2 相关工作可跳读。
- **不值得精讀的理由**：若你不做机器人学习、或已熟悉 Fast-WAM / PFD 一类"未来条件部署化"路线且只关心骨架细节，读摘要与 §1/§4 概览即可；论文的核心增量是"缓存构造接口"而非全新骨干或全新训练范式。

---
[← Back to Theory](./README.md)

**關鍵引用**
- 论文: https://arxiv.org/abs/2608.11521 （v3, 2026-09-25）
- 项目页: https://github.com/ChushanZhang/RIFT
- 骨干 / 基线: Fast-WAM (arXiv:2603.16666)、PFD (arXiv:2604.25859)、LingBot-VA (arXiv:2601.21998)、Cosmos Policy (arXiv:2601.16163)
- 基准: LIBERO (NeurIPS 2023)、LIBERO-Plus (CVPR 2026)、RoboTwin 2.0 (arXiv:2506.18088)
