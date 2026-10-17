> 历史补录：补回 2026-10-12 的深度阅读任务；论文原始收录日期 2026-10-10；实际补写日期 2026-10-17。本文依据当前读取的指定论文版本，不冒充当时已完成的报告。

# SmolWAM：面向边缘设备的隐空间世界-动作模型（SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices）

> ⚙️ 本文由 Moltbot 自动生成 | 2026-10-17
>
> **论文**: SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices
> **链接**: https://arxiv.org/abs/2610.13740v1
> **核心定位**: 把 2B 参数的 Cosmos Policy 视频 WAM 压缩成 0.88B 的隐空间 WAM，保留预训练 trunk 的边界块与单步 readout，在 LIBERO 上保住 95.5% 成功率的同时把每 16-action chunk 的延迟从 624 ms 降到 57 ms。

## ⚡ 快速判断（30 秒读完这段就够了）

| 维度 | 判断 |
|------|------|
| 核心结论 | 预训练视频 WAM 的成本大头在推理接口（33 帧 VAE、T5-XXL、5 步去噪、1,764 token），不在控制所需的世界表征；按诊断逐项重建接口并蒸馏，能在 0.88B 参数下保住 97% 教师成功率、57 ms/chunk |
| 适合精读 | 如果你在做边缘端 VLA/WAM 部署、动作 chunking 延迟优化，或想把视频生成模型改造成控制器，重点看 §3.2 与 §4 的 Q2 |
| 可以跳过 | 如果你只关心新的大规模 VLA 架构或新 benchmark，这篇是压缩与蒸馏工程，不是新能力 |
| 落地可行性 | 中：A100 上 57 ms/chunk、2.36 GiB，但只在 LIBERO 仿真验证，未上真机、未测其他 benchmark |
| 主要风险 | 长时任务退化最明显（LIBERO-Long 89% vs 97%）；减少深度对长程控制的伤害作者自己也没解决 |

💡 **X-Ray 开场**
这篇论文问三个问题：最贵的世界-动作模型（WAM）到底贵在哪？把视频扩散 transformer 的推理接口换掉、深度砍掉，还能不能保住控制成功率？砍完之后它还是不是世界模型？答案是：贵在“为视频生成设计的接口”而不是控制所需；用单帧 latent + 缓存文本 + 12/28 blocks + 单步 readout，能保住 95.5% 的成功率；世界头还能预测 16 步后的场景运动，但纹理糊了。

📍 **研究全景时间线**

```
2024  GR-1 / OpenVLA / PAD：视频预训练与联合去噪进入机器人控制
  │
2025  SmolVLA / WorldVLA / Cosmos-Predict2：紧凑 VLA 与视频世界模型并行
  │
2026  Cosmos Policy (ICLR'26)：2B 视频 DiT 微调成 WAM，LIBERO 98.5%
  │   TurboVLA / Fast-WAM：实时 VLA 与“测试时是否需要想象未来”
  │
2026-10  SmolWAM（本文）← 当前位置：保留 trunk、重接口、12 块、单步
  │
局限  只在 LIBERO 仿真；长时任务退化；未上真机
```

## 1. 核心架构/方法总览 (Overview / Architecture)

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 教师 Cosmos Policy | SmolWAM | 差异含义 |
|------|-------------------|---------|----------|
| DiT blocks | 28 块，1.96B | 12 块（0–2, 19–27），0.85B | 保留边界高影响块，去掉中间近恒等块 |
| 视觉编码 | Wan2.1 VAE，33 帧 clip/query | 单帧 VAE + 0.5M adapter | 每帧只编码一次，adapter 把单帧 latent 映到 clip-mode |
| 文本编码 | T5-XXL 4.8B，512 embeddings | 缓存 BERT-base，零填充到 512 | 保留 attention sink 的零 logits 质量 |
| 序列布局 | 9 frame × 196 = 1,764 tokens | 5 frame × 196 + 28 queries = 1,008 | 去掉 state/value frame，动作 frame 换成 28 个状态条件 query |
| 去噪步数 | 5 步 | 1 步（τ_max，固定噪声） | 动作与未来 latent 从同一次 x̂_0 读出 |
| 每 chunk 成本 | 624 ms / 42.6 TFLOP / 7.55 GiB | 57 ms / 2.30 TFLOP / 2.36 GiB | 延迟 11×、FLOPs 18.5×、内存 3.2× 改善 |
| 训练 | 教师微调 | 3.6 GPU 小时蒸馏（5,000 步，batch 32） | 68,830 观测，教师一步目标缓存 |

> 成本数字来自论文 Table 1 与 §4；参数不含 VAE 与文本编码器。教师 5 步行的 Full 98.2% 是早期管线在 L4 上的结果，论文正文报告 98.5%。

### 1.2 关键机制 (Key Mechanism)

论文的核心论证不是“蒸馏能压缩”，而是“教师的成本主要来自为视频生成设计的接口，而非控制所需”。四个改动分别对应一类成本：

1. **单帧 latent**：教师每次 query 要把 33 帧 clip 送进 VAE。改成单帧编码后，教师自己的动作误差从 0.0089 升到 0.038；加一个 0.5M 参数的 adapter 把单帧 latent 映到 clip-mode，误差回到 0.0092。也就是说，单帧本身损失的信息可以用一个很小的映射补回来。
2. **缓存文本 + attention sink 保留**：T5-XXL 的约 500 个零填充位置在无 bias 的 key/value 投影下贡献零 logits 和零 value，充当 attention sink。直接截断到 32 或 128 位置会让教师闭环成功率从 5/5 掉到 0/5；解析恢复被删的 softmax 质量能回到 5/5，且与零填充注意力匹配到 6×10^-8。所以学生保留 512 位置的零填充 BERT 特征，再用 6 个零初始化 vision-to-language 层让文本 attend 图像。
3. **边界保留剪枝**：用 block influence `1 − cos(h_in, h_out)` 排序，block 0、1、25–27 影响最大，3–22 单独接近恒等。零样本筛选（每项 5 episodes）显示移除不组合：单删 block 1、26、27 全失败，单删 7、14、20 不失败，但一起删 12–15 全失败。最终保留 0–2 与 19–27。
4. **状态条件动作 query + 单步 readout**：28 个零初始化 query 替换 196-token 动作 frame，零初始化 MLP 把机器人状态加上去。trunk 在 τ_max 只评估一次，动作与未来 latent 都从 x̂_0 读出。消融显示：去掉 future-state 与 value frame 影响很小（0.009 → ≤0.019），但去掉 blank（0.30）、state（0.058）或把 vector frame 压到 28 token（0.21）会显著变差——说明布局是学出来的，不能随手删。

⚡ **Eureka Moment**：预训练 WAM 的“贵”主要在推理接口（33 帧 VAE、4.8B 文本编码器、5 步去噪、1,764 token），而不在控制所需的世界表征；只要逐项替换接口并用教师一步预测蒸馏，trunk 的大部分深度可以不要。

### 1.3 信息流/架构图 (Flow / Diagram)

```
观测 o^w, o^p (224 px)          指令文本
      │                            │
      ▼                            ▼
 Wan2.1 VAE 单帧编码          缓存 BERT-base
      │                            │
      ▼                            ▼
 0.5M adapter → clip-mode    零填充到 512 位置
      │                            │
      └──────────┬─────────────────┘
                 ▼
    5 个 14×14 latent frame (980 tokens)
    blank, o^w, o^p, o'^w, o'^p
                 +
    28 个状态条件 action query
                 │
                 ▼
   12 个 DiT block（继承教师 0–2, 19–27 权重）
                 │
                 ▼
   τ_max 单次前向（固定噪声）
                 │
                 ▼
   x̂_0 = x_τ − τ·v_θ(x_τ, τ)
        │                    │
        ▼                    ▼
  16-action chunk      未来相机 latent（world head，训练监督，不解码）
```

> token 预算：1,008 = 5 × 196 + 28。论文没有逐项列出 5 个 frame 的构成，这是从 1,008 与 196 反推的分解，标记为本文推算。

## 2. 数学核心 (Math Core)

📌 **Napkin Formula**：

```
x̂_0 = x_τ − τ · v_θ(x_τ, τ)     在 τ = τ_max 只算一次，动作和未来 latent 同源读出
```

**目标**：把教师的单步 rectified-flow 预测蒸馏到学生的紧凑布局里，同时匹配演示与教师目标。

**蒸馏损失**（论文 §3.3）：

```
L = ‖â − a‖₁ + ‖â − a^T‖₁ + ‖ẑ − z‖₁ + ‖ẑ − z^T‖₁
    + 0.1 · (state/value 项，仅在这些 frame 存在时)
```

**变量说明**：

| 符号 | 含义 |
|------|------|
| a | 演示中的 16-action chunk |
| a^T | 教师在固定噪声下的一步动作预测（缓存） |
| z | 演示对应的未来 latent |
| z^T | 教师一步未来 latent 预测（缓存） |
| â, ẑ | 学生的一步估计 |
| τ_max ≈ 0.988 | 生成起点，对应 σ_max = 80 |
| v_θ | trunk 预测的速度场，目标为 ε − x_0 |

**直觉**：rectified flow 的闭式 clean 估计允许“一步到位”——不需要迭代去噪，只要在 τ_max 处前向一次，用速度场把含噪 latent 拉回 clean 估计。学生的动作和世界头共享这一次前向，因此动作不会条件于一个已经去噪的未来。论文的 Q3 与 Fast-WAM 的结论一致：测试时不需要显式想象未来也能行动。

**训练课程**：剪枝与紧凑化是两个不同的分布偏移。学生在前 1,000 / 5,000 步用教师的 1,764-token 布局训练，之后切到 1,008-token 紧凑布局；继承权重学习率 2×10^-5，新权重 10^-4；AdamW、batch 32、bf16，单卡 A100 共 3.6 h。

## 3. 带数字走一遍：玩具例子 (Worked Example)

> 以下数值为假设，用于验算闭式 readout，不是论文报告的数字。

设某个 latent 维度的真实 clean 值 x_0 = 0.96，噪声 ε = 1.0，τ = 0.988。

1. 构造含噪输入：
   ```
   x_τ = (1 − τ)·x_0 + τ·ε
       = 0.012 × 0.96 + 0.988 × 1.0
       = 0.01152 + 0.988
       = 0.99952
   ```
2. 真实速度场：
   ```
   v = ε − x_0 = 1.0 − 0.96 = 0.04
   ```
3. 闭式 clean 估计：
   ```
   x̂_0 = x_τ − τ·v
        = 0.99952 − 0.988 × 0.04
        = 0.99952 − 0.03952
        = 0.96   ✓
   ```
4. 如果模型预测的速度有误差，比如 v̂ = 0.05（偏 0.01），则：
   ```
   x̂_0 = 0.99952 − 0.988 × 0.05 = 0.95012
   误差 = 0.96 − 0.95012 = 0.00988
   ```
   即速度误差 δ 造成的 clean 估计误差为 τδ；τ≈0.988 时是被略微缩小（约 0.988 倍），而非放大；这也是为什么单步 readout 对 v_θ 的精度敏感，蒸馏要把一步误差压到教师附近。

> 第二个可验算点：学生每 chunk 57 ms，16 个动作开环执行，chunk 速率 = 1000/57 ≈ 17.5 chunks/s，对应 17.5 × 16 = 280 actions/s。论文报告 17.7 chunks/s 与 282 actions/s，差异来自它用的是约 56.5 ms 的有效值。

## 4. 工程视角 (Engineering View)

| 指标 | 教师 5 步 | 教师 1 步 + 单帧 | SmolWAM | 工程含义 |
|------|-----------|------------------|---------|----------|
| 延迟 / 16-action chunk | 624 ms | 126 ms | 57 ms | 57 ms 中 transformer 37、VAE 13、预处理 7 |
| FLOPs / chunk | 42.6 TFLOP | 7.52 TFLOP | 2.30 TFLOP | 含 VAE encoder 与 attention |
| 峰值显存 | 7.55 GiB | 4.39 GiB | 2.36 GiB | 单卡 A100-SXM4-40GB，batch 1，bf16 |
| 参数（不含 VAE/文本） | 1.96B | 1.96B | 0.88B | 12/28 块 + 28 query + adapter |
| chunk 速率 | 1.6 Hz | 7.9 Hz | 17.5 Hz | 开环执行，chunk 率决定反应频率 |

**工程含义**：

- **延迟分解决定优化顺序**：VAE 编码 13 ms 与预处理 7 ms 合计 20 ms，已经占 57 ms 的三分之一；如果换更小的 VAE 或缓存观测，收益可能比继续砍 transformer 更直接。
- **chunk 率是反应频率的上界**：因为 16 个动作开环执行，策略每 57 ms 才能看一次新观测；论文的 Q2 明确试过“8 步后重新查询教师”，一步或五步去噪都解不了筛选集，说明更快的反应需要训练时改动，而不是推理时截短。
- **深度与长时任务强相关**：8 块版本在筛选集掉到 83.3%，比 12 块的 93.3% 低 10 个点；作者在局限里也承认减深度对 LIBERO-Long 伤害最大。
- **INT8 的现状**：对 172 个最大线性层做 per-channel INT8，存储从 1.64 降到 0.85 GiB，筛选集掉 1 个 episode；但用 bf16 kernel 模拟，延迟没降（65 ms，且在共享 GPU 上测）。这是“省内存不省时间”的典型中间状态。
- **“边缘设备”的说法要打折**：所有延迟数字都在 A100 上测，唯一非 A100 的数字是附录 A.3 在 L4 上的零训练阶梯（1.83 → 0.23 s），以及 A.6 在 Apple M2 Pro 上录的 rollout 视频。论文没有报告 Jetson 或手机端帧率。

## 5. 数据与评测 (Data & Eval)

**训练数据**：68,830 个观测，每 4 步采样一次，来自 Cosmos Policy 发布的 1,700 条成功 LIBERO 演示。教师的一步动作与未来 latent 目标在固定噪声下缓存，耗时 66 分钟（单 A100）。学生训练 5,000 步，batch 32，bf16，3.6 h。

**评测协议**：
- LIBERO 四个 suite（Spatial / Object / Goal / Long），每 suite 10 个任务。
- seed 195，每任务 10 trials，共 400 episodes。
- 另有一个 60-episode 筛选集（每 suite 3 任务 × 5 trials），故意包含早期剪枝学生失败最多的任务，所以分数低于完整协议。
- 延迟：A100-SXM4-40GB，batch 1，bf16，含预处理与 VAE 编码，所有模型文本特征缓存。
- FLOPs：PyTorch FlopCounterMode，含 attention。

**关键结果（论文 Table 1）**：

| 模型 | 参数 | TFLOP | ms | GiB | Screen | Spatial | Object | Goal | Long | Full |
|------|------|-------|----|-----|--------|---------|--------|------|------|------|
| Cosmos Policy 5 步 | 1.96B | 42.6 | 624 | 7.55 | 96.7 | 97.0 | 100.0 | 99.0 | 97.0 | 98.2† |
| 教师 1 步 + 单帧 | 1.96B | 7.52 | 126 | 4.39 | 100.0 | – | – | – | – | – |
| Pruned + LoRA | 1.40B | – | – | – | – | 83.0 | 100.0 | 82.0 | 85.0 | 87.5† |
| SmolWAM | 0.88B | 2.30 | 57 | 2.36 | 93.3 | 99.0 | 100.0 | 94.0 | 89.0 | 95.5 |
| + 7.5k 步 | 0.88B | 2.30 | 60 | 2.36 | 93.3 | 94.0 | 100.0 | 94.0 | 89.0 | 94.2 |
| INT8§ | 0.88B | 2.30 | 65‡ | 2.36 | 91.7 | – | – | – | – | – |
| 8 blocks | 0.60B | 1.76 | 134‡ | 1.84 | 83.3 | – | – | – | – | – |
| SmolWAM-LA 4 步 | 0.92B | 4.28 | 239 | 2.43 | 53.3 | – | – | – | – | – |

> † 来自早期评估管线，在 L4 上；‡ 共享 GPU；§ INT8 为模拟（bf16 kernel）。SmolWAM 的 95.5% 对应 382/400，95% Wilson 区间 93.0–97.1；教师 98.2% 对应 393/400，区间 96.4–99.1。

**Q1 的细节**：Spatial 与 Object 上没有检测到差异；差距集中在 Goal（94 vs 99）与 Long（89 vs 97），失败 rollout 通常完成第一个子目标后卡在第二个。更早的 LoRA 学生只去 8 块、19.7M 可训练参数、不到一小时 L4、414 条教师 rollout，达到 87.5%；SmolWAM 与它的差异（蒸馏覆盖、全量微调、学习布局）没有分离。更长训练（+7.5k 步）不提升成功率。

**Q2 的细节**：教师未改权重、单步、单帧输入就能把延迟降 5×、计算降 5.7×，筛选集无损失；在 L4 上，无 adapter + 精确 sink 修正的版本从 1.83 s 降到 0.23 s，11 episodes 无失败（附录 A.3）。这说明接口改动本身贡献了大部分收益，trunk 重设计只是在此之上继续压。

**Q3 的细节**：96 个 held-out context，预测 16 步后的视角，latent L1 比 copy-current-frame 低 24%（0.101 vs 0.133）；RAFT 光流的端点误差从 1.63 降到 1.43 像素。解码后的预测保持场景、手臂接近未来姿态，但外观模糊；LPIPS 0.128 vs copy 的 0.115，基本持平。世界头捕获运动而非纹理。

**Q4 的细节**：把 ABot-M0.5 的 latent-action token、mixture-of-transformer、四步 flow matching、dream forcing 移植过来，得到 SmolWAM-LA（+39M 参数），筛选集 53.3%、239 ms；一步动作离线 L1 与 SmolWAM 相当（0.089 vs 0.091），但四步采样把误差推到 0.155。世界头 reward fine-tuning 在 300 步内把成功率打到 5.0–16.7%，LPIPS 从 0.130 恶化到 0.167–0.177。作者的解释是：该目标没有动作项，却训练共享 trunk。

## 6. 能力与失败模式 (Capabilities & Failure Modes)

**能做的**：
- 在 LIBERO 四个 suite 上以 382/400（95.5%）完成 16-action chunk 的开环控制；Spatial 99%、Object 100%。
- 以 57 ms/chunk 输出动作（论文报告 17.7 chunks/s、282 actions/s；约 56.5 ms 有效值），在 A100 上。
- 世界头能预测 16 步后的相机 latent，运动方向与手臂位置基本正确（RAFT 端点误差 1.43 px）。
- 单步 readout 下动作不条件于去噪后的未来，仍然可用——与 Fast-WAM 的结论一致。

**不能做 / 会失败的**：
- **长时任务**：LIBERO-Long 89% vs 教师 97%；Goal 94% vs 99%。失败模式是完成第一个子目标后停滞。
- **更深剪枝**：8 块掉到 83.3%（筛选集），说明 12 块接近这个 recipe 的下限。
- **更丰富的 WAM 组件**：latent-action cascade、mixture-of-transformer、四步采样在这个规模下都变差（SmolWAM-LA 53.3%）；world head 的 reward fine-tuning 直接崩溃。
- **截短 chunk 换反应速度**：8/16 后重新查询教师，一步或五步都解不了筛选集；更快反应需要训练时改动。
- **纹理级预测**：LPIPS 与 copy-frame 持平，解码预测模糊；世界头不是视频生成器。
- **跨 benchmark / 真机**：论文只评 LIBERO，作者自己在局限里说需要扩展到 RoboCasa 与真机。

### 6.1 隐含假设 (Hidden Assumptions)

1. **“控制只需单步 readout”**：这个结论在 LIBERO 的 60 个筛选 episode 上成立：教师单步（单帧）解完全部筛选（Table 1 中 5 步行为 96.7%，单步单帧行为 100.0%）；但它依赖任务本身不需要多步想象。长时任务退化可能部分来自这里，论文没有把“单步”与“少块”的贡献分开。
2. **“attention sink 可以用零填充复现”**：论文用解析 softmax 质量匹配到 6×10^-8 来支撑，但这是在教师权重上的零样本诊断；学生用 BERT 特征 + 6 层 vision-to-language，是否等价没有单独 ablation（附录 A.5 的六项计划消融里包含移除蒸馏目标与布局课程，但没有跑）。
3. **“LIBERO 成功率可迁移到边缘设备”**：所有延迟数字在 A100 或 L4 上，唯一端侧证据是 M2 Pro 上的 rollout 录制；没有 Jetson、手机或实时控制循环的测量。
4. **“教师一步目标是可靠监督”**：蒸馏目标来自教师固定噪声下的一步预测，若教师在长时任务上的单步误差本身较大，学生会继承。论文报告学生验证 L1 稳定在 0.09 左右，是教师一步误差的 10 倍，但成功率仍有 95.5%——作者自己说离线 L1 是闭环成功的差代理。

## 7. 与相关工作对比 (Comparison)

| 工作 | 关注点 | 架构/训练 | 与 SmolWAM 的关系 |
|------|--------|-----------|-------------------|
| Cosmos Policy (ICLR 2026) | 视频 DiT 微调成 WAM | 28 块 1.96B，5 步，9 frame | 本文的教师与初始化来源 |
| SmolVLA (2025) | 紧凑 VLA | 只预测动作，无世界模型 | 同类“小”路线，但不保留未来预测 |
| TurboVLA (2026) | 实时 VLA | 32 Hz on RTX 4090，<1 GB VRAM | SmolWAM 借用其 vision-to-language 层与部分原则 |
| Fast-WAM (2026) | 测试时是否需要想象未来 | 跳过测试时未来预测 | 与本文 Q2/Q3 的“单步足够”结论一致 |
| Light-WAM (2026) | 更小视频 backbone 解动作 | 从更小视频骨干解码动作 | SmolWAM 从动作 pass 读未来 latent，路径不同 |
| ABot-M0.5 (2026) | latent-action + 多模态 | latent-action token、mixture-of-transformer | 被移植成 SmolWAM-LA，在本规模下变差 |

**面试 Tip**：如果被问“SmolWAM 和 SmolVLA/TurboVLA 有什么区别”，一句话答：SmolVLA/TurboVLA 只预测动作，SmolWAM 保留了世界头（未来相机 latent）并从同一次单步前向读出，同时用“保留边界块 + 缓存文本 sink + 单帧 adapter”把 2B 教师压到 0.88B，代价是长时任务退化。

## 8. 精读建议 (Reading Guide)

- **值得精读原文的人**：
  1. 做边缘端 VLA/WAM 部署的工程师：§3.2 的四个接口改动与 §4 的 Q2 成本分解最有用。
  2. 想把视频生成模型改造成控制器的研究者：§3.2 的 attention sink 诊断与剪枝不组合现象值得复现。
  3. 做动作 chunking 与实时推理的人：§4 的 chunk 率、截短实验与 INT8 现状是直接的工程约束。

- **建议章节路径**：先读 §3.2（四个改动与诊断）→ 再看 §4 的 Q1/Q2（成功率与成本来源）→ 可跳 §2 的相关工作（除非要写 survey）。

- **不值得精读的理由**：如果你不做机器人学习、或已经熟悉“剪枝 + 蒸馏 + 单步 flow”这套组合，读摘要与 Table 1 即可；这篇的贡献主要在工程 recipe 与三个具体陷阱（attention-sink padding、联合致命的 block、不能截短的 chunk），不在新算法。

---

[← Back to Theory](./README.md)

**关键引用**：
- 论文：https://arxiv.org/abs/2610.13740v1