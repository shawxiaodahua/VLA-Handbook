# 运动基元 Token：从人类视频到人形技能迁移（Tokens of Motion: Primitive Motion Learning for Human-Humanoid Skill Transfer）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-31
>
> **论文**: Tokens of Motion: Primitive Motion Learning for Human-Humanoid Skill Transfer
> **链接**: https://arxiv.org/abs/2610.37330
> **核心定位**: 把人类手部轨迹切成局部运动段，学习可复用的因子化运动 token，再用轻量 embodiment-specific decoder 迁到人形；相比端到端模仿完整轨迹，它把“共享运动结构”和“机器人具体执行”拆开。

**证据版本与摘要差异**：以下依据来源提供的 arXiv:2610.37330v1 论文正文 HTML（页面标注 27 Oct 2026，对应论文正文）。候选摘要 snippet 描述的是 AFIL（Adaptive Failure-Informed Learning），与正文标题/摘要的 M-PACT 不一致；来源记录未给出摘要核验结果，因此本文不采用候选摘要，也不把 AFIL 的主张混入正文。若引用 PDF，应以同一 v1 正文为准。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 作者主张：从人类手部轨迹学到的局部运动 token 可跨数据集/跨 embodiment 复用，迁移到新人形时只需适配轻量 decoder。 |
| 适合精读 | 如果你在做 human-to-humanoid transfer、动作 tokenization、轻量 decoder adaptation，重点看 §3.3–§3.6、Table 1、Table 3。 |
| 可以跳过 | 如果你只关心大规模端到端 VLA 策略训练，或只接受真实闭环成功率证据，这篇距离中等。 |
| 落地可行性 | 中低：离线重建与 decoder 适配有量化结果，但真实人形只有定性演示，缺成功率/试次/控制频率。 |
| 主要风险 | CC-EPMT 的 rotation 预测误差 20.28°，远高于 recorded EPMT 的 0.93°；aperture 在 HumanoidEveryday 的归一化误差 1.388，不优于零变化。 |

💡 **X-Ray 开场**  
这篇论文问的是：人类视频没有机器人可执行动作，且人-机 embodiment 差异大，能否只学“可复用的局部运动结构”，把“机器人怎么执行”留给一个小 decoder？作者的答案是：用 SkillSeg 自监督切段，用 EPMT 把每段压成 translation/aperture 离散 token 加 rotation 连续 latent，冻结词表，只为目标机器人训练 decoder。对 VLA 研究者的含义是：动作 token 不一定只能是机器人动作序列，也可以是 embodiment-agnostic 的运动效果表示；但预测器和真实闭环仍是缺口。

📍 **研究全景时间线**  
仅列来源参考文献中有明确年份的节点，不代表完整谱系：

```
2022  Ego4D / RT-1
2023  RT-2 / MimicPlay / HoloAssist
2024  π0 / DexCap
2025  EgoMimic / FAST / LAPA / Being-H0 / EgoDex / HumanoidEveryday
2026  M-PACT（arXiv:2610.37330v1，27 Oct 2026）← 本文
```

本文位置：在“人类视频预训练 + 动作 tokenization + 人形迁移”之间，选择“局部运动基元 + 冻结词表 + 轻量 decoder”的路线。局限：真实人形只做定性演示，未给出闭环成功率。

## 1. 核心架构/方法总览

### 1.1 系统对比概览

| 模块/归属 | 输入 | 输出 | 训练/冻结 | 时序注意 |
|---|---|---|---|---|
| SkillSeg（M-PACT 的分段前端） | 平滑运动学特征 f_t：平移速度、速率、aperture、aperture rate、旋转表示导数 | 软片段索引 c_t、片段边界 b_i | 自监督训练；推理时平均 M=7 个独立训练网络 | 片段级；边界连续整数交叉，距上一边界 K=8 帧内的交叉跳过。K=8 是边界跳过阈值，不是采样点数。 |
| EPMT（Embodied Primitive Motion Tokenization） | 片段内已观测轨迹，重采样 L=8 点 | translation code 1–16、aperture code 1–4、连续 rotation latent；decoder 输入 Z̃ 为 67 维 | 与 source decoder 联合训练；迁移时冻结 | L=8 是段内均匀重采样点数；不是未来预测 horizon，也不是记忆窗口。 |
| Source decoder D_θ | h_i（10 维初始状态）+ Z̃_i（67 维） | y_i（7 维净运动效果） | 与 EPMT 联合训练 | 每个 decoder 输入 77 维、输出 7 维、27,399 参数。 |
| Target decoder D_θe | 目标机器人对齐后的 h_i^e + 冻结 Z̃_i^e | 目标机器人 y_i^e | 只训练新 decoder；SkillSeg/EPMT/codebook 冻结 | 状态/效果 normalizer 用目标训练数据拟合；运动特征 normalizer 固定。 |
| CC-EPMT（Context-Conditioned EPMT） | 片段起点可用的 state-only，或 image/text/state 特征 | 预测的 translation/aperture 分布与 rotation latent | 只训练 context predictor；EPMT 与 decoder 冻结 | 从片段起点预测该片段表示；论文没有定义固定未来 H 或动作 chunk 长度。 |
| M-PACT 闭环 | 当前观测、任务上下文、当前手/臂状态 | 下一 primitive 表示 → decoder → 机器人运动目标 | 组合上述模块 | predict–realize–observe 循环；论文未给控制频率。 |

### 1.2 关键机制

- **先分段，再 token 化**：SkillSeg 用可微时间分解发现局部运动段，避免把整条 demonstration 当作一个单体监督信号。分段由运动学线索驱动，不需要人工段标签。
- **因子化表示**：EPMT 把每段拆成 translation、rotation、aperture 三个通道。translation 和 aperture 离散化，rotation 保持连续。作者明确说这是结构性选择，不要求统计独立，也不保证最小重建误差。
- **只监督净效果**：EPMT 的 encoder 看段内轨迹，但 decoder 只重建段末相对段初的净效果 y_i。因此 token 表示“运动结果”，不要求保留完整路径。
- **状态条件化**：decoder 输入包含初始状态 h_i，同一个 primitive 在不同初始构型下可以产生不同效果。
- **迁移只改 realization**：目标 embodiment 适配时冻结 SkillSeg、EPMT、codebook，只训练新 decoder。人类到人形还可选继续训练 robot decoder，即 human-to-humanoid grounding。
- **CC-EPMT 是预测器，不是记忆模块**：它从片段起点上下文预测该片段表示；论文没有提出跨片段记忆窗口。

⚡ **Eureka Moment**：把“人类运动”压成局部、因子化、与 embodiment 无关的 primitive token；迁移时不动词表，只学一个很小的 decoder 把 token 映射成目标机器人运动效果。

### 1.3 信息流/架构图

```
人类 egocentric 视频
  └─ 3D hand trajectories τ={(p_t,R_t,α_t)}
       └─ SkillSeg: 自监督分段 → 片段 [a_i,e_i]
            └─ EPMT: resample L=8 → tr/rot/ap 编码
                 ├─ tr: VQ16
                 ├─ ap: VQ4
                 └─ rot: 连续 3D latent
                      └─ Z̃_i ∈ R^67
                           ├─ 记录因子 → D_θ 重建 y_i
                           ├─ 目标 embodiment → 冻结 Z，只训 D_θe
                           └─ CC-EPMT: 从片段起点上下文预测 Z_i
                                └─ 闭环: predict → decode → execute → re-observe
```

注意：图中 `[a_i,e_i]` 是来源定义的**首末包含帧**；`a_i=b_{i-1}` 是首个包含帧，`e_i=b_i−1` 是最后一个包含帧。不要写成半开区间 `[a_i,e_i)`，否则会排除来源定义的最后一帧。

## 2. 数学核心

📌 **Napkin Formula**：

```
Z_i = (VQ16(tr_i), rot_i, VQ4(ap_i));  y_i = D_θ(h_i, Z̃_i)
迁移：冻结 SkillSeg/EPMT/codebook，只解 θ_e = argmin E[ℓ_bal(D_θ(h_i^e,Z̃_i^e), y_i^e)]
```

**目标**：从人类手部轨迹学一个可复用的运动表示，并让目标 embodiment 只学“表示→运动效果”的映射。

**SkillSeg 分段**：

```
c_t = Σ_{s=0}^{t} m_s q_s
w_{t,k} = max(0, 1 - |c_t - (k+1/2)| / ρ),  ρ=0.6
w̃_{t,k} = w_{t,k} / (Σ_s w_{s,k} + ε)
L_seg = L_motion + λ_c L_VQ + λ_r (c_{T-1}-N*)^2 / max((N*)^2,1) - λ_e H(q)
```

- q_t：时间网络预测的增量，取值在 (0,1)。
- m_t：二值 motion gate，抑制非活跃帧的累计。
- c_t：软片段索引。
- w_{t,k}：第 k 个片段槽的三角成员权重。
- N*：duration-dependent 的目标片段数；论文说它控制 segment count。
- H(q)：平均 Bernoulli 熵；熵项抑制饱和增量。

**EPMT 因子化 token**：

```
resample_L({(p_t-p_a, π6(R_a^T R_t), α_t)}) ∈ R^{3L}×R^{6L}×R^L, L=8
z_tr = VQ16(E_tr(x̄_tr))
z_rot = E_rot(x̄_rot)
z_ap = VQ4(E_ap(x̄_ap))
Z̃_i = [e_tr(z_tr), z_rot, e_ap(z_ap)] ∈ R^67
h_i = [p_a, π6(R_a), α_a] ∈ R^10
y_i = [p_e-p_a, Log_v(R_a^T R_e), α_e-α_a] ∈ R^7
```

- translation 是相对段初的位移，rotation 是相对初始 wrist orientation，aperture 保留绝对对齐值。
- codebook 大小：K_tr=16，K_ap=4；embedding 32 维。两个离散索引合计 6 bit；rotation latent 另有精度相关存储成本。
- 论文说 factorization 是结构性选择；它不保证统计独立，也不保证最小重建误差。

**平衡重建损失**：

```
ℓ_bal = (1/3)(
  ||ŷ_tr - y_tr||² / 3
  + ||ŷ_rot - y_rot||² / 3
  + |ŷ_ap - y_ap|²
)
L_EPMT = E_i[ℓ_bal + β Σ_{f∈{tr,ap}} L_commit^f]
```

直觉：每个通道先在自己的维度内平均，再三通道平均，避免 3D translation/rotation 因维度多而压过 1D aperture。监督只重建净效果，不要求保留整条轨迹。

**目标 embodiment decoder 适配**：

```
θ_e = argmin_θ E_i[ℓ_bal(D_θ(N_h^e(h_i^e), Z̃_i^e), N_y^e(y_i^e))]
```

每个 decoder 使用相同 77 维输入、7 维输出、27,399 参数架构；状态/效果 normalizer 用目标训练数据拟合，运动特征 normalizer、segmentation、encoders、codebooks 固定。源与目标 demonstration 不需要配对。

**Human-to-humanoid grounding**：

```
L_gnd(θ) = E_{i~D_hum}[MSE(D_θ(N_h^rob(h_i^hum), Z̃_i^hum), N_y^rob(y_i^hum))]
```

只更新 embodiment-specific decoder；所有 continuation 输入和目标都来自人类片段，不需要配对 human–robot 轨迹或 robot-effect 标签。

**CC-EPMT 预测目标**：

```
u_i^state = N_h^ctx(h_i)
u_i^full = [f_img(I_i), f_text(ℓ_i), N_h^ctx(h_i)]
L_CC = E[-log P_tr(z_tr|u) - log P_ap(z_ap|u)
        + λ_rot MSE(ẑ_rot,z_rot)
        + λ_pose MSE(ŷ_soft, N_y(y))]
```

- 图像/文本特征来自冻结 CLIP encoders。
- 离散因子的期望 embedding μ_i^f = Σ_k P_φ^f(k|u_i)c_k^f，使 decoded-effect 项可微。
- 推理时离散预测使用 codebook entry，不用 expected embedding。
- λ_pose=0 的消融用来检查 decoded-effect supervision 的贡献；论文报告变化最多 0.03°/0.13 accuracy points，单 seed 不能 establish auxiliary-loss benefit。

## 3. 带数字走一遍：玩具例子

> 以下为**假设验算**，不是论文实验设置；论文未给帧率。

**假设**采样间隔 Δt=1/30 s，某个已观测片段用 L=8 个均匀采样点重采样。则首末采样点跨度是：

```
(L-1)×Δt = 7/30 ≈ 0.2333 s
```

如果从当前参考 t 预测 t+1 至 t+H，H=8，则终点距当前参考是：

```
H×Δt = 8/30 ≈ 0.2667 s
```

两者不同：前者是“已观测段内 L 个采样点的首末跨度”，后者是“未来预测区间终点距当前参考”。本文的 L=8 属于前者；论文的 K=8 是 SkillSeg 边界跳过帧数，属于另一件事，不能套用采样跨度公式。

**损失验算**：假设某片段的 normalized 误差为 translation 每维 0.2、rotation 每维 0.3、aperture 0.1，则：

```
ℓ_bal = (1/3)(0.2² + 0.3² + 0.1²)
      = (1/3)(0.04 + 0.09 + 0.01)
      = 0.14/3 ≈ 0.0467
```

这说明三通道被显式平衡；若只看 translation，会低估 aperture/rotation 的失败。

## 4. 工程视角

- **压缩与接口**：translation 16 codes、aperture 4 codes，合计 6 bit 离散索引；rotation 连续 3D latent。decoder 输入 77 维，输出 7 维，27,399 参数。工程含义：词表很小，适合作为低带宽 primitive 接口；但 rotation 仍是连续量，不能按纯离散 token 处理。
- **时序边界**：SkillSeg 推理平均 M=7 个独立训练网络，边界在累计分数连续整数交叉处提取，距上一边界 K=8 帧内的交叉跳过。M=7 是网络数，不是 epoch 数；K=8 是帧阈值，不是未来 chunk 长度。
- **记忆机制归属**：论文没有显式记忆模块。SkillSeg 是时间分段；EPMT 的 L=8 是段内重采样；CC-EPMT 从片段起点预测该片段表示。三者都不是跨片段历史记忆窗口。不能把 5–90 帧过滤范围写成所有实验共有的记忆窗口或动作 chunk 长度。
- **部署约束**：目标机器人需要 forward kinematics，并用坐标变换、arm-length scaling、wrist-orientation alignment、affine aperture calibration 对齐到共享约定。论文明确说共享约定不保证对齐后的人类运动机器人可达。
- **延迟/频率**：论文未给推理延迟、控制频率、每次决策耗时。CC-EPMT 需要图像/文本特征（full 版本）加 predictor 加 decoder；工程上应先测端到端延迟，再决定是否用于闭环。
- **量化误差**：HumanoidEveryday 的 aperture 绝对误差仅 0.229 cm，但归一化误差 1.388，不优于零变化，因为该数据集 aperture 变化有限。工程含义：aperture 通道要按任务标定，不能只看绝对厘米数。

## 5. 数据与评测

### 5.1 表示训练/预训练

| 数据集 | 用途 | episodes | retained segments | 备注 |
|---|---|---|---|---|
| EgoDex | 表示训练 | 27,411 | 2.92M | 90/10 episode split；292,806 test segments |
| HoloAssist | 冻结迁移 | 35,087 | 0.80M | 用 frozen source decoder |
| HumanoidEveryday | G1 adaptation | 4,064 | 96,241 | robot decoder 85/15 → 14,754 test；robot prediction 70/30 → 28,871 test |

论文过滤掉 5–90 帧以外或所有因子变化可忽略的片段。统计只用训练数据，source tokenizer 保持冻结。候选集结果平均三个 episode-level calibration/test partitions；task generalization 留出 37/124 个 task names，跑三个 seeds。注意：这些是离线数据划分，不是闭环仿真成功率。

### 5.2 离线评测

指标包括 median endpoint translation、rotation、aperture error；括号内是误差除以 median motion magnitude，小于 1 表示优于零变化。另有 code use、top-1 accuracy、90%-coverage translation-set size。Replay 用 interpolated-path 和 geodesic orientation errors；multi-run 值平均 runs。

**Table 1 关键结果（论文实测）**：

| 指标 | EgoDex source | HoloAssist frozen | HumanoidEveryday refitted |
|---|---:|---:|---:|
| Translation (cm) | 2.658 (0.532) | 3.123 (0.811) | 3.442 (0.454) |
| Rotation (°) | 0.931 (0.041) | 2.732 (0.152) | 0.901 (0.062) |
| Aperture (cm) | 0.439 (0.493) | 0.454 (0.525) | 0.229 (1.388) |
| Translation codes used | 16/16 | 16/16 | 16/16 |
| Aperture codes used | 4/4 | 4/4 | 4/4 |
| Effective translation vocabulary | 10.23 | 8.63 | 13.19 |

解释：HoloAssist 冻结迁移时所有 code 仍活跃，但 translation/rotation 误差上升、有效词表下降，说明分布漂移。HumanoidEveryday 只重训 target decoder，translation/rotation 归一化误差较低，code 全用；但 aperture 归一化误差 1.388，不优于零变化。作者摘要声称“as few as ~100 robot demonstrations”可做 decoder adaptation；正文 Table 1 未列出该数字对应的逐项实验次数，因此该 100 条应视为作者主张/摘要级证据，而不是表格中可逐项核对的实验次数。

**Table 3(a) CC-EPMT on held-out EgoDex（论文实测）**：

| 表示 | Trans. (cm) | Rot. (°) | Ap. (mm) | Trans. top-1 | Ap. top-1 |
|---|---:|---:|---:|---:|---:|
| Recorded EPMT | 2.658 | 0.93 | 4.39 | — | — |
| State-only prediction | 5.307 | 21.65 | 9.19 | 36.37 | 57.54 |
| Context prediction | 4.952 | 20.28 | 8.46 | 40.42 | 59.07 |
| Context, no effect loss | 4.960 | 20.25 | 8.47 | 40.29 | 59.08 |
| Training-majority label | — | — | — | 35.01 | 54.67 |

解释：context 把 translation top-1 从 36.37% 提到 40.42%，误差从 5.307 降到 4.952 cm，但 rotation 仍 20.28°，远高于 recorded 的 0.93°。作者结论是 prediction 是主要下游误差源；这是论文实测支撑的判断。单 seed 不能 establish auxiliary-loss benefit。

**Table 3(b) robot translation prediction（论文实测，candidate set 平均三个 partitions）**：

| Predictor | Top-1 (%) | Candidate size | Coverage (%) |
|---|---:|---:|---:|
| Frozen human (T+S) | 7.48 | 13.25 | 89.6 |
| Robot (S) | 30.64 | 9.12 | 89.9 |
| Robot (T) | 22.83 | 11.36 | 90.0 |
| Robot (T+S) | 33.98 | 8.35 | 89.9 |
| Majority | 22.56 | — | — |

解释：语言+状态把 top-1 提到 33.98%，高于 majority 22.56%；在约 90% coverage 下，candidate set 从 13.25 降到 8.35/16。但 language 让 held-out-task accuracy 降低 0.8 个百分点，作者说 task generalization 证据有限。Coverage 只表示包含记录 code，不表示 feasibility 或 task success。

### 5.3 真实机器人设置

论文在真实人形上描述四个递进设置：a) EPMT 运动重建；b) Human2Robot 跨 embodiment 迁移，只适配 decoder；c) CC-EPMT state-only 闭环 token 预测；d) CC-EPMT full 用相机观测和任务指令做 context-conditioned manipulation。作者主张这些设置从重建、迁移到自主闭环逐步验证框架。但正文**没有给出真实机器人成功率、试次次数、控制频率或统计置信区间**；只有定性演示描述。因此真实闭环能力应标为“作者主张/定性证据”，不能与 Table 1/3 的离线数字等同。

## 6. 能力与失败模式

| 能力/失败 | 证据 | 边界 |
|---|---|---|
| 跨数据集冻结迁移 | HoloAssist 冻结 source decoder，所有 code 活跃，归一化误差仍小于 1 | translation/rotation 误差上升，有效词表从 10.23 降到 8.63，说明分布漂移 |
| 跨 embodiment decoder 适配 | HumanoidEveryday 只重训 decoder，translation 3.442 cm (0.454)、rotation 0.901° (0.062)，code 全用 | aperture 0.229 cm 但归一化 1.388，不优于零变化 |
| 轻量 decoder | 每个 decoder 77 维输入、7 维输出、27,399 参数 | 论文未给训练时长/算力；~100 demos 是摘要主张 |
| 上下文 primitive 预测 | context 把 translation top-1 提到 40.42%，误差降到 4.952 cm | rotation 20.28°，预测仍是主要误差源 |
| 机器人域候选集 | T+S top-1 33.98%，candidate size 8.35/16 at ~90% coverage | coverage 不是 feasibility/success；language 对 held-out task 反而 -0.8 pp |
| 真实人形闭环 | 四个设置定性演示 | 无成功率、试次、频率；不能声称已稳健闭环 |

### 6.1 隐含假设

- **共享 kinematic convention 足够**：作者用坐标变换、臂长缩放、wrist orientation 对齐、affine aperture calibration 把机器人映射到同一约定，但明确说这不保证人类对齐运动机器人可达。
- **分段边界可跨数据集复用**：SkillSeg 在 EgoDex 训练后用于 HoloAssist/HumanoidEveryday；论文显示 code 仍活跃，但 HoloAssist 的有效 translation vocabulary 从 10.23 降到 8.63，而 HumanoidEveryday 为 13.19，高于 EgoDex 的 10.23，说明跨数据集边界/表示并非完全不变，且不同目标域表现不一致。
- **净效果监督足够**：EPMT 只重建段末净效果，不要求保留完整轨迹；这对“运动结果”有效，但可能丢失段内避障、姿态约束等信息。
- **小 decoder 足以吸收 embodiment gap**：HumanoidEveryday 结果支持这一点，但真实人形 grounding 的分布漂移和可达性未定量。
- **上下文特征足够选 primitive**：CC-EPMT 用 CLIP image/text + state；rotation 预测误差很大，说明当前上下文不足以精确选旋转。
- **候选集 coverage 可转化为控制**：论文明确 coverage 不是 feasibility 或 success；实际部署还需要验证器/重排序。

## 7. 与相关工作对比

| 方法 | 关注点 | 架构/训练方式 | 适用场景 |
|---|---|---|---|
| EgoMimic | 用对齐的 egocentric human 与 robot 数据联合训练策略 | 联合训练 imitation policy | 人类视频辅助机器人策略 |
| Point Policy | 用 keypoint 轨迹连接观察与动作 | 预测 keypoint trajectories，经几何约束转机器人动作 | 几何可对齐的 manipulation |
| DexCap | 灵巧操作手部运动迁移 | 捕获手运动，经 IK 与 imitation learning | 灵巧手/动捕 |
| DexImit | 从单目人类视频学生物双手操作 | 重建 hand–object interaction，子任务分解生成机器人 demo | 双手灵巧操作 |
| MimicPlay | 长时程模仿 | 人类手轨迹预测做 planner，指导 teleop 训练的 robot policy | 长时程任务 |
| VQ-BeT / FAST / LAPA | 离散动作表示/动作 tokenization | VQ、action sequence tokenization、latent action | VLA 动作建模 |
| **M-PACT（本文）** | 人类手轨迹的局部 primitive token + 轻量 decoder 迁移 | SkillSeg 分段；EPMT 因子化 token；冻结词表只训 decoder；CC-EPMT 预测 | human-to-humanoid skill transfer |

**面试 Tip**：被问到时一句话答——“M-PACT 的关键不是把整条人类轨迹端到端模仿，而是把局部运动段压成 translation/aperture 离散、rotation 连续的 primitive token，冻结词表只适配小 decoder；它的强项是跨数据集/跨 embodiment 的重建复用，弱项是 primitive 预测尤其是 rotation，以及真实闭环缺定量成功率。”

## 8. 精读建议

- **值得精读原文的人**：做 human-to-humanoid transfer 的研究者；做动作 tokenization/离散动作表示的人；需要评估“冻结共享表示 + 轻量 decoder 适配”工程可行性的工程师。
- **建议章节路径**：先读 §3.3–§3.6 搞清 SkillSeg/EPMT/decoder adaptation/CC-EPMT 的边界；再看 Table 1 和 Table 3 核对数字；最后看 §4.3 的真实机器人设置，但把它当定性证据。Related Work 可快读。
- **不值得精读的理由**：如果你只关心大规模端到端 VLA 策略、或要求真实闭环成功率/试次统计，这篇目前不够；读摘要和 Table 1/3 即可。论文未给代码仓库信息，来源记录 repo_url 为空，本文不推断代码或数据开放承诺。

---

[← Back to Theory](./README.md)

**关键引用**：Tokens of Motion: Primitive Motion Learning for Human-Humanoid Skill Transfer，arXiv:2610.37330v1，https://arxiv.org/abs/2610.37330