# 从噪声引导到双隐空间：给冻结生成式策略加一条「内部表征」控制通道 (Beyond Noise Steering: Dual-Latent Space Reinforcement Learning for Generative Robot Policy)

> ⚙️ 本文由 Moltbot 自动生成 | 2026-09-13
>
> **论文**: Beyond Noise Steering: Dual-Latent Space Reinforcement Learning for Generative Robot Policy
> **链接**: [arXiv:2609.11270v1](https://arxiv.org/abs/2609.11270) (cs.RO, 2026-09-10) · [HTML 全文](https://arxiv.org/html/2609.11270v1) · [代码仓库](https://github.com/xianchaoxiu/DLSRL)
> **作者**: Teng Sun, Xianchao Xiu（上海大学机电工程与自动化学院；代码仓库署名 P. Zhang, T. Sun, X. Xiu；基金 NSFC 12371306）
> **核心定位**: 相比 DSRL 只调「初始噪声」这一条外部接口，DLSRL 在冻结生成器**内部的动作 token 隐状态**上再开第二条 RL 控制通道——不改基座参数、不引入可训练 Transformer 层，却让在线适应更快。

## ⚡ 快速判斷（30 秒讀完這段就夠了）

| 維度 | 判斷 |
|------|------|
| 核心結論 | 把「只扰动初始噪声」扩展为「初始噪声 + 中间动作表征」双隐空间 RL 控制；在 RoboMimic 与 LIBERO 上加速在线适应，Can 任务后期约 99% vs DSRL 约 90%（论文 Fig. 3） |
| 適合精讀 | 如果你在做**生成式 VLA 的在线 RL 精调 / 部署后适应**，重点看 §III-B、§III-C（双隐空间 actor + 隐状态注入）与 §IV-C（消融与 λ_inj 扫描） |
| 可以跳過 | 如果你只关心**离线模仿学习 / 从零训练 VLA**，或者不做 diffusion / flow-matching 动作头，这篇距离中等 |
| 落地可行性 | **中**：适配器与 actor 极轻量、基座可完全冻结，工程上易插拔；但需在线环境与奖励，且 λ_inj 目前靠人工调 |
| 主要風險 | 实验**全在仿真**（RoboMimic / LIBERO），作者自承真实世界留作 future work；收益主要是「更快」而非「更高上限」 |

💡 **X-Ray 开场**（2-3 句，非专家也能读懂）
生成式机器人策略（Diffusion Policy、π0 这类 flow-matching VLA）像一个「从噪声里雕出动作」的雕塑家。已有的强化学习只在**开头**改噪声（换个起点），或者等**雕完了**再修最终动作——中间那段雕刻过程没人管。这篇的做法是：在雕塑家下刀的中途，往动作 token 的内部隐状态里注入一点点可学习的「修正方向」，同时仍然调起点；两条通道一起用 RL 优化，而雕塑家本人（基座策略）全程冻结、一个参数都不动。

📍 **研究全景時間線**

```
2023  Diffusion Policy 确立"动作即条件去噪"范式
        │
2025  DPPO：直接对 diffusion policy 做策略梯度微调（要更新基座）
        │
2025  DSRL ★：冻结基座，只学"观测条件下的初始噪声"去 steering  ← 本文的直接对手
        │
2026  DynaGuide / UniSteer / LPS：改采样轨迹、改 latent、把人类纠正转成噪声监督
        │     （都是在"边界"上起作用：起点、轨迹、或最终输出）
        │
2026  [本文 DLSRL] ← 当前位置
        │   第一次把 RL 控制伸进冻结生成器内部的 action-token 隐状态
        │
局限  ⚠ 仅仿真验证；真实世界、注入层选择、生成步数影响均列为 future work
```

## 1. 核心架構/方法總覽 (Overview / Architecture)

DLSRL 的本质是**在冻结的生成式基座上，外挂两个轻量模块 + 两个 critic**，用在线 RL 训练它们。

### 1.1 系统对比概览 (System Component Comparison)

| 模块 | 输入 | 输出 | 形状 / 时序 | 训练 or 推理 | 是否更新 |
|------|------|------|------------|-------------|---------|
| 基座生成策略 G_φ | 观测 o_t、初始噪声 z_t、适配特征 f_t | 动作块 a_t | a_t ∈ R^{H×d_a}，H 为预测视野 | 两者皆用 | ❌ 全程冻结 |
| 双隐空间 Actor π_θ | 观测 o_t | (z_t, u_t) 两路 latent | 每个控制步一次 | 仅训练更新，部署保留 | ✅ |
| 适配特征映射器 A_ω | u_t | f_t | f_t ∈ R^{N_a×d_h}，与 action token 同形 | 两者皆用 | ✅ |
| 动作空间 critic Q_ψ^act | (o_t, a_t) | 动作块价值 | off-policy TD | 仅训练，部署丢弃 | ✅ |
| 隐空间 critic Q_ν^lat | (o_t, z_t, f_t) | 隐空间价值 | 从 Q_ψ^act 蒸馏 | 仅训练，部署丢弃 | ✅ |
| 残差注入 δ_t = λ_inj · f_t | f_t | 加到隐状态 H̃ | 仅注入 action token 位置 | 两者皆用 | 标量超参 λ_inj |

关键点：**部署时只剩 Actor + Mapper**，两个 critic 全部丢弃（论文 §III-D 末）。

### 1.2 关键机制 (Key Mechanism)

- **两条控制通道职责分离**：z_t 控制**生成起点**，主要影响基座会选出哪种「行为模式」（behavioral mode）；u_t 通过 f_t 直接调制**生成过程中的中间动作表征**，管的是接触位置、动作幅度、局部轨迹这类需要在生成中被纠正的细节。
- **注入方式极简**：f_t 与 action-token 隐状态形状**天然对齐**（都是 R^{N_a×d_h}），因此不需要任何额外投影层，也不往基座里加可训练参数。做法就是 `H̃ = H + λ_inj · f_t`。
- **注入只针对 action token**：视觉、语言、本体感等 context token 完全不动——把「能改什么」约束在动作上，避免污染感知表征。
- **同一份 f_t 在所有注入块与所有生成步共享**：实现简单、参数极少，但同时意味着这是一种**粗粒度、全局一致**的调制（见 §6.1 隐含假设）。
- **价值蒸馏绕开不可导的采样过程**：动作 critic 用在线交互数据做标准 off-policy TD；隐 critic 用 stop-gradient 把动作空间价值蒸馏到 (z_t, f_t) 空间，从而在不反传穿过冻结生成过程的前提下，给两路 latent 学到信用分配。

⚡ **Eureka Moment**：existing RL 只控制生成的**边界**（起点噪声或最终动作），而生成式策略真正决定成败的是**中间过程**——所以把 RL 的价值梯度落到「生成器内部 action token 隐状态」这一层，是此前没人占用的一条控制接口。

### 1.3 信息流/架构图 (Flow / Diagram)

```
                o_t = (I_t, q_t, ℓ)
                      │
        ┌─────────────┴──────────────┐
        │                            │
   双隐空间 Actor π_θ            (观测条件表征 e_t)
        │                            │
   ┌────┴─────┐                      │
   │          │                      │
  z_t        u_t                     │
(初始噪声)  (动作表征)                │
   │          │                      │
   │     适配映射 A_ω                 │
   │          │                      │
   │        f_t ──────┐              │
   │                  │              │
   │            λ_inj · f_t = δ_t    │
   │                  │              │
   │                  ▼              ▼
   │        ┌────────────────────────────────┐
   │        │  冻结生成器 G_φ                 │
   └───────▶│  for k = 1..K (去噪 / 积分步):   │
            │    for l = 1..L (Transformer块): │
            │      H^(l)  = T_φ^(l)(H̃^(l-1), e_t, k) │
            │      H̃^(l)  = H^(l) + δ_t   ← 注入   │
            │ 输出 a_t ∈ R^{H×d_a}             │
            └────────────────────────────────┘
                            │
                     环境执行 → r_t, o_{t+1}
                            │
                     replay buffer D → 更新 critics / actor / mapper
```

## 2. 數學核心 (Math Core)

📌 **Napkin Formula**（一行抓住本质）：

```
max_θ,ω  E[ Σ_t γ^t r_t ]   s.t.   a_t = G_φ(o_t, z_t, f_t),  f_t = λ_inj · A_ω(u_t),  (z_t,u_t) ~ π_θ(·|o_t)
```

**目标**：冻结 φ，只学 θ（双隐空间 actor）与 ω（适配映射），最大化期望折扣回报。

**逐步公式（论文编号对应）**：

基座生成（式 1，纯噪声驱动）：

```
a_t = G_φ(o_t, z_t)                     # 原范式：起点即全部
```

加入表征调制后的生成（式 6）：

```
a_t = G_φ(o_t, z_t, f_t)
```

冻结 Transformer 块的前向与残差注入（式 7、8）：

```
H^(l)_{t,k}  = T_φ^(l)( H̃^(l-1)_{t,k} , e_t , k )     # 冻结块，不更新
H̃^(l)_{t,k} = H^(l)_{t,k} + δ_t                        # 注入发生在块输出
δ_t          = λ_inj · f_t                              # 全局注入强度 × 适配特征
```

符号说明：

| 符号 | 含义 | 备注 |
|------|------|------|
| φ | 基座生成策略参数 | 全程冻结 |
| θ | 双隐空间 actor 参数 | 只学这个 + ω |
| ω | 适配特征映射器参数 | 只学这个 + θ |
| z_t | 初始噪声 latent | 控制生成起点 / 行为模式 |
| u_t | 动作表征 latent | 经 A_ω 变成 f_t |
| f_t | 适配特征 | ∈ R^{N_a×d_h}，与 action token 同形 |
| λ_inj | 注入强度 | 人工超参，扫 {0, 0.03, 0.06, 0.09} |
| k | 生成更新步 | diffusion：反向去噪步；flow matching：离散积分步 |
| N_a, d_h | action token 数、Transformer 隐维 | RoboMimic 上 d_h = 128 |

**隐空间 critic 的价值蒸馏（式 9）**：

```
L_distill = E[ ( Q_ν^lat(o_t, ẑ_t, f̂_t) − sg[ Q_ψ^act(o_t, â_t) ] )^2 ]
```

`sg[·]` 为 stop-gradient；该步**只更新 ν**（隐空间 critic 参数），把动作空间价值「翻译」到隐空间。

**Actor 目标（式 10，熵正则只加在噪声分支）**：

```
L_actor = E[ α · log π_θ^z(z_t | o_t) − Q_ν^lat(o_t, z_t, f_t) ]
```

注意一个细节：熵正则只作用在 z_t 分支，但 Q_ν^lat 对 z_t 和 f_t 的梯度会**同时回传到两条分支和适配映射器**——这是「两条通道共享同一个价值信号」的实现方式。

**直觉**：把「在动作空间里改结果」换成「在表征空间里改过程」。动作空间的价值梯度无法穿过冻结的采样过程，于是作者训练一个隐空间 critic 去近似它，再让 actor 通过 f_t 这条可导的旁路去影响生成结果。

> 符号与本文/相关文档保持一致：z（噪声 latent）、u（表征 latent）、f 或 δ（注入量）。DSRL = DSRL 的 z 分支；DLSRL = z + u 双分支。

## 3. 帶數字走一遍：玩具例子 (Worked Example)

取一个极度简化的 2D 场景：**1 个 action token、隐维 d_h = 2**，这样每步都是手算可验证的闭环。

假设（合理假设值，用于演示计算链，非论文数据）：

```
初始隐状态      H = [ 0.50, −0.30 ]          # 冻结块 l 的输出
actor 输出      u_t → A_ω(u_t) = f_t = [ 0.20, 0.40 ]
注入强度        λ_inj = 0.06                  # 论文扫描中的最优值
```

第 1 步——算注入量：

```
δ_t = λ_inj · f_t = 0.06 × [0.20, 0.40] = [0.012, 0.024]
```

第 2 步——残差注入后的隐状态：

```
H̃ = H + δ_t = [0.50 + 0.012, −0.30 + 0.024] = [0.512, −0.276]
```

第 3 步——把它喂给下一个冻结块（式 7）：

```
H_next = T_φ^(next)( H̃ , e_t , k )
```

第 4 步——感受一下 λ_inj 的量级效应（论文 Fig. 6 的 Can 任务扫描，定性一致）：

| λ_inj | δ_t（本玩具例） | 论文观察到的行为 |
|-------|----------------|-----------------|
| 0.00 | [0, 0]（等于纯 DSRL，无表征通道） | 无表征调制 |
| 0.03 | [0.006, 0.012] | 提升平缓但稳定 |
| 0.06 | [0.012, 0.024] | **速度与稳定性最佳平衡，Can 最终达 100%** |
| 0.09 | [0.018, 0.036] | 早期最快，但后期波动明显变大 |

**闭环含义**：扰动只有隐状态数值的 ~1%–4% 量级，却足以改变后续块的输出轨迹；而扰动太大（0.09）会让冻结基座**已有的动作先验被冲掉**，导致后期不稳。这解释了作者那句「有效的表征控制需要足够强、但不能过度扰动预训练动作表征」。

另一个可算的关键数字：**消融 DLSRL-Rep**（关掉可学习的初始噪声引导，只留表征调制）在 Can 上从约 **27% 提升到约 52%**（论文 §IV-C）——说明这条新通道**自己就能带来增益**，不是搭 z 分支的便车。

## 4. 工程视角 (Engineering View)

| 维度 | DLSRL 的表现 | 工程含义 |
|------|-------------|---------|
| 基座改动 | **零**：不加可训练层、不加投影层，只做一次残差加法 | 基座可以是任意已部署 checkpoint（diffusion 或 flow matching），改造成本极低 |
| 推理开销 | 每控制步多一次 Actor 前向 + 一次 Mapper 前向 + 每注入块一次 `+δ_t` | 加法是 O(N_a·d_h)，可忽略；真正的开销是 actor/mapper 的额外前向（小网络） |
| 部署产物 | 只保留 Actor + Mapper，两个 critic 丢弃 | 上线体积小；critic 训练开销不进入推理路径 |
| 训练稳定性 | 隐 critic 用 stop-gradient 蒸馏；mapper 的 hidden 分支用 warmup + 渐进 ramp-up（代码 README：`adapter.ramp_steps` 扫 0 / 25k / 125k / 250k） | 直接开满注入容易崩，官方默认是「先学噪声、再逐步放表征」 |
| 在线预算 | Can 配置：400,000 在线 action-chunk 转移 + 75,050 初始采集（1,501 次向量化调用 × 50 环境）；50 训练 / 25 评测环境；每 chunk 最多执行 4 步 | 需要**真在线交互**——这是落地最大门槛，仿真便宜、真机贵 |
| 控制频率耦合 | 注入发生在生成步 k 与块 l 的组合上，f_t 在所有 k、l 共享 | 不需要重算注入；但若想做「按步变化」的注入，需改架构（论文列为 future work） |
| 超参敏感 | λ_inj 需要人工扫；0.06 最优、0.09 不稳 | 换模型/换任务大概率要重扫，目前无自适应机制 |

**一句话工程结论**：这是一个**「低侵入、需要在线、收益在速度」**的方案。如果你的基座已经冻结部署、又能拿到在线奖励（仿真大规模平行采样，或真机带安全约束的少量交互），它的插拔成本几乎为零；反之若只能离线微调，它不适用。

## 5. 數據與評測 (Data & Eval)

| 平台 | 任务 | 基座策略 | 基线 | 评测方式 |
|------|------|---------|------|---------|
| RoboMimic | Lift, Can, Square | 预训练 **Transformer Diffusion Policy**（4 个 Transformer 块，hidden dim 128，动作块长度 4），**冻结** | Base Policy, JSRL, DPPO, DSRL | 训练过程中 100 episodes 成功率（论文 Fig. 3、5、6） |
| LIBERO | Stove-On, CreamCheese-to-Tray, Bowl-Drawer-to-Plate, WineBottle-to-Rack, Plate-to-StoveFront, Bowl-to-TopDrawer（6 个） | **π0**（flow matching；视觉-语言主干与动作生成模块均冻结） | DSRL（主对照） | 成功率曲线（Fig. 4）+ **平均 episode 长度**（Table I） |

数据观测说明：RoboMimic 使用**低维本体感观测**（代码 README：「robotic experiments use low-dimensional Robomimic observations」）；离线数据可选 `load_offline_data=True`，默认先用基座交互采集初始数据。

主要数字（来自论文正文与图）：
- Can 任务后期：DLSRL 约 **99%** vs DSRL 约 **90%**（§IV-A）。
- DLSRL-Rep（仅表征调制）Can 上约 **27% → 52%**（§IV-C）。
- λ_inj = 0.06 在 Can 上最终达 **100%**；0.09 早期最快但后期波动大（§IV-C / Fig. 6）。
- Table I：LIBERO 六个任务上 DLSRL 平均 episode 长度**全部更短**（原文未在 HTML 中给出具体数值，`> TODO: 待从 PDF 的 Table I 补具体数值`）。

> 注意评测口径：作者反复强调优势是「**更快达到饱和**」，而非「最终成功率一致更高」——Lift 这类简单任务上 DSRL 后期会追平。

## 6. 能力與失敗模式 (Capabilities & Failure Modes)

**能做**：
- 在**冻结**基座（diffusion 或 flow matching）上做在线适应，基座参数一个不动。
- 在需要**精细动作精度**的任务（Can、Square）上更快逼近高成功率。
- 加速跨架构迁移：同一套框架在 RoboMimic 的 Transformer diffusion policy 与 LIBERO 的 π0 上都观察到收益。

**失败模式 / 已知局限**：
- **仅在仿真验证**：RoboMimic + LIBERO，**没有真机实验**。作者明确把「real-world environments」列为 future work——不要外推到真机部署。
- **收益主要在速度**：不是普遍更高的最终性能；简单任务（Lift）上 DSRL 后期追平。
- **依赖在线奖励与交互预算**：400k 级别的在线转移在仿真可行，真机成本高；离线场景不适用。
- **λ_inj 手工调参**：跨任务/跨模型需重扫；无自适应机制。
- **注入位置固定**：「注入哪些块、哪些生成步」未被系统研究（future work）。
- **f_t 全步共享**：无法做逐步细粒度控制。
- **两 critic 训练开销**：训练期需同时维护动作 critic、隐 critic 与 actor，流程比单 critic 复杂（部署时无此负担）。

### 6.1 隐含假设 (Hidden Assumptions)

1. **生成过程的表征可分性**：假设存在一个「加一点、就按预期方向改善」的中间表征方向。但 f_t 是所有生成步 k 与所有注入块 l **共享的常数**——这隐含「动作表征在去噪轨迹上近似平稳」的强假设；对 K 大、轨迹曲率高的 flow matching，这一近似未验证。
2. **标量 λ_inj 足够**：用**一个全局标量**平衡注入强度，等于假设不同块/不同步对扰动的敏感度相近。Fig. 6 显示 0.09 会不稳，恰恰说明敏感度并非平坦。
3. **价值蒸馏的保真性**：假设隐空间 critic 能忠实学到「(z, f) → 动作空间回报」的映射。但梯度不穿过冻结生成过程，蒸馏误差会直接偏置 actor 的梯度方向；论文未给出蒸馏误差的量化诊断。
4. **奖励可编码且可在线获得**：任务奖励在仿真里免费，真机上「什么算成功」本身是难题。
5. **仿真成功率可代表真实能力**：所有结论建立在仿真成功率与 episode 长度上。
6. **基座先验必须被保留**：方法的卖点之一是「不破坏预训练能力」，但「不破坏」并未用独立指标度量（如基座行为保持度、OOD 任务退化），只有成功率曲线间接支撑。

## 7. 與相關工作對比 (Comparison)

| 方法 | 介入位置 | 是否更新基座 | 控制信号来源 | 适用场景 |
|------|---------|-------------|-------------|---------|
| **DPPO** (ICLR 2025) | 整个生成网络 | ✅ 更新 | 策略梯度 | 允许在线微调基座、算力充足 |
| **DSRL** (CoRL 2025) | 初始噪声 z_t | ❌ 冻结 | 隐空间 RL | 冻结基座 + 只需行为模式选择 |
| **UniSteer** (2026) | 噪声空间（由人类纠正反演） | ❌ 冻结 | 人类修正动作 → 噪声监督 | 有人工纠正、VLA 适配 |
| **LPS** (2026) | latent steering + 动作空间 critic | ❌ 冻结 | 一步 flow 策略 | 单步/少步 flow 策略 |
| **DynaGuide** (NeurIPS 2025) | 去噪轨迹（外部动力学引导） | ❌ 冻结 | 外部动力学模型 | 有可靠动力学模型 |
| **Residual RL / Policy Decorator** | 最终输出动作空间 | ❌ 冻结 | RL / 在线精修 | 只需要输出级修正 |
| **DLSRL（本文）** | **初始噪声 + 中间 action-token 隐状态** | ❌ 冻结 | 双隐空间 RL（价值蒸馏） | 冻结生成式基座 + 在线奖励 + 需精细动作纠正 |

差异一句话总结：**别人管「起点」或「终点」，DLSRL 管「过程」——而且过程控制不需要更新基座。**

**面试 Tip**：被问到「DLSRL 相比 DSRL 的关键创新」时，答：*「DSRL 只在噪声空间 steering，等于只选行为模式；DLSRL 用第二个 latent 映射成与 action token 同形的适配特征，以标量强度残差注入冻结 Transformer 块的 action-token 隐状态，把 RL 控制推进到生成过程内部；靠 stop-gradient 的价值蒸馏绕开不可导的采样过程——而且只在仿真验证，优势是适应更快。」*

## 8. 精讀建議 (Reading Guide)

- **值得精讀原文的人**：
  1. 做**生成式 VLA 在线适应 / 部署后 RL 精调**的研究者（想找一条「不碰基座参数」的控制接口）。
  2. 评估**把 DSRL 类方法迁移到自己基座**（π0、RDT、自有 diffusion policy）可行性的工程师。
  3. 对「parameter-efficient / adapter 注入」在机器人领域的迁移感兴趣的人（ControlNet / T2I-Adapter 思路 → 在线机器人反馈）。
- **建議章節路徑**：先讀 §III-B（雙隱空間 actor）→ 再看 §III-C（隱狀態注入的式 7/8）→ 然後 §III-D（價值蒸餾，理解梯度為何不穿過生成器）→ 實驗優先看 Fig. 3 與 §IV-C 消融 → §II-C 了解與 ControlNet 系的淵源 → 可跳 §II-A（生成式策略背景，多數讀者已熟）。
- **不值得精讀的理由**：如果你不做机器人学习、或已经熟悉 DSRL/DPPO 这类 latent steering 工作，且不关心适配器注入细节，**读摘要 + §IV-C 消融**就够了——本文的核心信息量集中在「双通道 + 同形注入 + 价值蒸馏」这三点，其余是标准 RL 组件与仿真实验。

---

## 关键引用

- 论文：[arXiv:2609.11270](https://arxiv.org/abs/2609.11270)（v1, 2026-09-10, cs.RO）
- 全文：[arxiv.org/html/2609.11270v1](https://arxiv.org/html/2609.11270v1)
- 代码：[github.com/xianchaoxiu/DLSRL](https://github.com/xianchaoxiu/DLSRL)（含 RoboMimic Lift/Can/Square/Transport 配置、DSRL 基线、ramp_steps 消融脚本）
- 核心基线 DSRL：Wagenmaker et al., *Steering your diffusion policy with latent space reinforcement learning*, CoRL 2025
- 相关方法：DPPO (ICLR 2025)、UniSteer (arXiv:2605.10821)、LPS (arXiv:2603.05296)、DynaGuide (NeurIPS 2025)、Policy Decorator (ICLR 2025)

---
[← Back to Theory](./README.md)
