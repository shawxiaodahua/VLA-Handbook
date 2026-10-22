🤔 *双周反思* | 2026-10-08 – 2026-10-21

_读完没立场 = 这两周在消费而不在研究_

本期先说清楚：14 天里只有 6 天有完整采集收据。下面问题锚定的是这 6 天的样本，不是领域全貌。能在缺口里下判断，才是能力。

━━━ 趋势与判断 ━━━

1️⃣ 70 篇理论精读里 *记忆* 占了 8 篇（NativeMEM、LTM、Mem4Act、REKI、Double-Stream、Memory as Weights…），而 WORLD MODEL 家族在本期 method trends 里是 accelerating=[] 且列在 declining。但上一期（10-02–10-15）world_model 才刚被标记 accelerating。你打算把哪条读成趋势？还是这两条其实在说同一件事，只是命名不同？

2️⃣ 假设你只能给团队配一个人，方向二选一：*长程潜记忆*（8 篇，RSS 样本内）还是 *触觉/力*（7 篇：Spark-0、TouchScale、VISTA、DEFT-VLA、OpenViTac、TACROSS、ForeTac）。给出你的选项和一条不可反驳的理由，不许说"两个都要"。

3️⃣ 基准侧在样本里同时冒出 *LIBERO-Pro2.0*、*LIBERO-Dynamic*、*BiDex-100*、*LADDER*、*LIBERO-Industry*、*AutoBio+*、*JitterVLA*、*ContactBench 1.0*，而 CALVIN/ABCD-D、LIBERO/standard-opensource 在存储快照里已顶到阈值（4.8 对 4.7、99.8 对 98.0）。这些新基准里，哪一个是真在测量现有限制，哪一个只是给已经饱和的旧榜换壳？说清楚你的判据。

4️⃣ LoopWAM 被报道在 *RoboChallenge Table30 V2* 真机榜成功率与过程分双第一，三分之二的证据是媒体转述；另一边 *Phoenix-MWA* 在 LIBERO Plus 94.6（+7.80）、RoboCasa-GR1-Tabletop 82.7 刷出大跨度提升。这两个"第一"，你会先复现哪一个？为什么不是另一个？

5️⃣ 本期 12 条社交信号里，有 5 条是融资（星尘智能 10 亿元 / 150 亿估值、格物通智近亿元天使、四川具身超亿元 Pre-A、伊塔动力种子轮、FF Nova 2,490 美元人形），技术讨论几乎为零，一级信号只有 *GroundingPI* 和 *REKI* 两条。资金在扩张而技术讨论在收缩——这是产业在成熟，还是研究社区已经跑到了产业不敢跟的地方？不要两边都说。

━━━ 技术追问 ━━━

🔬 本期至少 5 篇（*VeriVLA*、*Verify Act Distill*、*Flow-Support-Weighted Proposals*、*Policy-Aligned Predictive Verification*、*SWAP*）都在做"测试时验证"：生成候选动作、再挑一个。你能说清它跟 *CFG-Policy* 那种把采样、探索、RL 微调塞进同一套 classifier-free guidance 的做法，差在哪一层吗？差不清的话，先读 CFG-Policy（arXiv 2610.14936）。

🔬 *NativeMEM*（arXiv 2607.06678）和 *LTM*（arXiv 2610.21362）都在压历史，一个压进原生 token 空间，一个用潜转移。它们对"记忆里到底该留什么"的答案是不同的。读完两篇告诉我：如果你的任务有 20 分钟跨度且含一次人为中断，哪个假设会先崩？不要复述摘要。

🔬 本期有 *Beyond Reconstruction: What Matters in Action Tokenization*、*Vision-Language-Delta-Action*、*What Should Robot Policies Learn to Predict?* 三篇在质疑"动作该被怎么表示"。同时 calibration 里 V-001（diffusion/flow 类动作生成仍是主导）被标为 strengthened 0.76→0.78。你觉得这两件事矛盾吗？动作 tokenization 的变化会先动摇 *pi0* 式的 flow 头部，还是先动摇 ACT 式 chunking？

🔬 触觉这条线本期有 *TouchScale*（500 小时人类视触数据）和 *TACROSS*（跨异构触觉传感器采集），一份走人类数据，一份走传感器多样性。*VISTA* 和 *OpenViTac* 都在接触丰富任务上做评测。说清一件事：跨传感器泛化与跨本体泛化，哪个是触觉 VLA 更硬的瓶颈？你的依据必须来自本期这几篇，不是直觉。
