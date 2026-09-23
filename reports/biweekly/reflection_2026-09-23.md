🤔 *双周反思* | 2026-09-10 – 2026-09-23

_读完没立场 = 这两周在消费而不在研究_

━━━ 趋势与判断 ━━━

1️⃣ *language_grounding* 7天 75 篇、14天 114 篇、加速 1.92×，是 15 个方法族里唯一的 SURGE，其余 11 个全部 declining；而本期 347 篇评级里 ⚡ 为 0、74% 只能当背景读。一个零突破、单极坍缩的领域——你判断这 1.92× 是能力前沿的引力，还是引用与话题驱动的拥挤？给出你据此调仓或不调仓的理由。

2️⃣ 本期你只有一笔算力预算。A：挤进 *language_grounding* 红海；B：在 *tactile* × *rl_finetuning* 上开新线（tactile 学术加速 0.31、rl 独占 0.88，资本刚在樞途近 1 億 Pre-A、帕西尼逾 40 億累计下注）；C：把 *world model* 重构成可部署规划器（宇樹 *UnifoLM-X2-1.0* 已拿它做全自动格斗）。选哪个，为什么另外两个是错的？

3️⃣ CALVIN 三个 split 加 LIBERO 三个 split 已全部 saturated，却仍贡献 14/40 次 SOTA 变动；真正有换手的 *LIBERO Plus* / *MetaWorld* / *RoboCasa-GR1-Tabletop* / *RoboChallenge* 各自只有 4–6 次。这是"进步"还是"噪声"？如果是噪声，为什么还有这么多人往上投？给一个能证伪你判断的观测。

4️⃣ *tactile* 学术 14天 28 篇掉到 7天 13 篇、持续走低，同期却拿到樞途近 1 億元 Pre-A、帕西尼累计逾 40 億元融资；而東京養老院实测机器人「能理解语言、能递物，却提不稳水壶、倒水手抖」。学术退潮 + 产业下注 + 落地翻车同时发生——这是窗口打开还是泡沫前兆？未来 6 个月你会盯哪个信号来确认？

5️⃣ 上期你押过：*flow matching* 会在高动态对抗/真实车队基准上取代 *diffusion* 成新 SOTA。本期数据是 diffusion_policy（21、0.64）仍压 flow_matching（19、0.49），且全部 SOTA 变动还在 CALVIN/LIBERO 饱和 split 上。这是 ❌ 落空还是 ⏳ 时间未到？基于当前数据给出你的判断（不允许回答"两方面都有道理"）。

━━━ 技术追问 ━━━

🔬 *flow matching* 本期在理论清单里高频出现——*AdaVLA* 免训练自适应步数、*IMLE-VLA* 单步 cIMLE、*DIDO* 单步去噪蒸馏、*TraceFlow*、*Q-VGM*——但方法族里 diffusion 仍领先。你能说清 flow matching 与 diffusion policy 在策略蒸餾上的本质区别，以及"少步数"这个卖点为何还没换成社群选票吗？说不清，这就是你这两周最该补的课。

🔬 两篇攻击论文同时出现：*Bit-Flip Attacks on VLA*（动作解码架构决定脆弱性）与 *DropVLA*（action-level backdoor）。你知道 bit-flip 与 drop 攻击落在 action chunk 还是 token 级别，会导出完全不同的防御层吗？去读这两篇的 threat model 段落，再回答你的策略该在哪一层做鲁棒化。
