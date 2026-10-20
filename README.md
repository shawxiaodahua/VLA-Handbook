<p align="center"><img src="docs/banner.svg" width="100%" alt="VLA Handbook"></p>

# VLA Handbook

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Auto-updated](https://img.shields.io/badge/内容-每日自动更新-blue)
[![PULSAR 照见](https://img.shields.io/badge/PULSAR_照见-每日精选-FF6B35.svg)](https://sou350121.github.io/pulsar-web/)
[![Agent-Playbook](https://img.shields.io/badge/Agent_Playbook-配套-8B5CF6.svg?logo=github&logoColor=white)](https://github.com/sou350121/Agent-Playbook)
[![VLA Expert Skill](https://img.shields.io/badge/VLA_Expert_Skill-Claude_×_Cursor-00C7B7.svg?logo=github&logoColor=white)](https://github.com/sou350121/VLA-expert-skill)
[![RSS 订阅](https://img.shields.io/badge/RSS-订阅-FFA500.svg?logo=rss&logoColor=white)](https://sou350121.github.io/pulsar-web/subscribe)
[![Daily Pulse](https://img.shields.io/badge/📊_Daily_Pulse-15方法族趋势-D4910A.svg)](PULSE.md)
[![Spatial-Intelligence-Handbook](https://img.shields.io/badge/姊妹仓库-Spatial_Intelligence_Handbook-6B8AFE.svg?logo=github&logoColor=white)](https://github.com/sou350121/Spatial-Intelligence-Handbook)

📊 **[PULSAR 照见](https://sou350121.github.io/pulsar-web/)** · 汇集各主题更新；网站固定同步时间为北京时间 18:00，内容以实际来源日期为准 — `sou350121.github.io/pulsar-web`

📈 **[Daily Pulse](PULSE.md)** · 15 个 VLA 方法族的论文流量 + 加速度 + 30 日趋势图 · 每日自动生成

📡 **[RSS 订阅](https://sou350121.github.io/pulsar-web/subscribe)** · 4 个主题 feed + 一键 OPML 导入 — 新文章/⚡🔧 论文/SOTA/周报直达你的阅读器 · [完整使用说明](docs/SUBSCRIBE.md)

🤖 **[VLA Expert Skill](https://github.com/sou350121/VLA-expert-skill)** · 一键让 AI 编程助手变成 VLA 专家 — 支持 Claude Code / Cursor / Codex / OpenCode

🛰️ **[Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook)** · 姊妹仓库 — VLA 管 action policy，Spatial 管 world representation（3DGS / VGGT / depth foundation + 跨 embodiment 横向对比）。两者交集是 3D-aware VLA。

🏭 **[行业情报雷达](companies/industry_mainline.md)** · 🆕 每日扫描机器人/具身公司动态（融资 · 产品 · IPO · 合作）→ 现有 DeepSeek 原生搜索、来源审计与官方页面核验后追加到公司档案；周五出[行业判断地图](companies/industry_mainline.md)，原始记录见 [`memory/blog/archives/industry-radar/`](memory/blog/archives/industry-radar)。

## 2026-10-20 内容与检索更新

- **新增深度解析**：[HexaShell](theory/frontier/hexashell_a_thin_harness_for_exploring_the_limits_of_vlm_age_dissection.md) 探查 VLM 编码 agent 控制机器人的能力边界；[VLDA](theory/frontier/vision_language_delta_action_learning_compact_deltas_as_acti_dissection.md) 讨论紧凑状态增量如何连接观测与动作。两篇均按已核查原文解读，实验设置与适用边界见正文。
- **历史稿件补录**：[CFG-Policy](theory/diffusion-flow/cfg_policy_unified_diffusion_policy_learning_framework_via_c_dissection.md) 为 10 月 15 日生成、10 月 20 日重新核查后发布的稿件；[NativeMEM](theory/planning/nativemem_native_memory_compression_for_long_horizon_robotic_dissection.md) 保留已有发布版。补录日期不当作论文首发日期。
- **永久正文检索上线**：维护服务器每 15 分钟核对 GitHub 已发布版本，更新 VLA 与 Spatial 的本地全文索引；原近 60 日情报索引继续保留。生成与核验沿用现有 DeepSeek，索引不调用模型 API、不使用 OpenRouter。

### 2026-10-18 内容更新

- **触觉 VLA**：[VLA-Touch](theory/tactile/vla_touch_dual_level_tactile_feedback_2025.md) 与 [N0-VTLA](theory/tactile/n0_vtla_latent_tactile_tokens_2026.md) 补入深度解读；训练范围、实验口径及论文／仓库版本差异分别说明。
- **旧文校正与去重**：[StaMo](theory/frontier/stamo_unsupervised_learning_of_generalizable_robot_motion_fr_dissection.md)、[Embodied-R1.5](theory/foundation/embodied_r15_evolving_physical_intelligence_via_embodied_fou_dissection.md) 合并同论文重复稿；[GR00T-N1.6](theory/vla-core/gr00t_n1_6.md) 按固定官方模型卡重建，未获支持的频率和参数不再作为事实。
- **行业补录**：[近期四项官方事件](memory/blog/archives/industry-radar/2026-10-18.md) 与 [三项历史事件](memory/blog/archives/industry-radar/backfill-2026-10-18.md) 分开归档；历史补录不冒充本周新增。全部入口见 [理论目录](theory/README.md) 与 [行业判断地图](companies/industry_mainline.md)。

> **历史更新 · 2026-06-11**
> - **理论总纲两个月来首次全面同步**：10 个主题 `*_mainline.md` 综合了 4-6 月新进的 ~109 篇深度解析（保留旧判断并显式记录张力），71 篇散落文章归位，200+ 失效内链修复，全库 254 → **380 篇**。
> - **行业情报层上线**：`companies/*.md` 不再静态——新增自动雷达每日补充融资/产品/IPO 动态（已回填过去 2 个月积压），周度生成行业判断地图。

### 全中文 · 工程实战导向的 VLA 知识库

VLA 论文每天几十篇，真正能用的工程细节零散在 GitHub Issue 和论文附录里。
这个 Handbook 做一件事：**整理从论文到复现的入口、参数口径与部署经验，并标出来源、未验证项和适用边界。**

> **754 篇理论文档**（10 个主题目录） · 中英文社区实战笔记 · GitHub Issues 工程经验 · **18 份双周前沿报告 + 15 份反思** · 行业情报雷达 · 固定排程采集、评分与深度解析
>
> 理论文档计数口径：10 个主题目录内的 Markdown 文件，含综述与索引，不含各目录 README；统计日期：2026-10-20；双周报告与反思按各自文件计数，不表示每期均已配对。

---

## 三句话说清楚这个 Handbook 的价值

1. **不只是摘要**：重点整理入口脚本、关键超参与 shape 口径的 sanity-check；复现状态和未验证项以具体文章标注为准。
2. **Robotics 的脏活写清楚**：多模态同步、Sim2Real 断点、动作空间对齐、触觉/灵巧手硬件选型——这些在其他地方要么没有，要么藏在论文附录里。
3. **活的知识库**：自动 pipeline 每天抓最新 VLA 论文（⚡/🔧/📖 评级），精选后生成深度解析写入仓库，不是六个月没人维护的静态文档。

### 🔥 社区实战笔记 — 论文不会告诉你的事

这三份笔记是 Handbook 最独特的部分：**中英文社区与 GitHub Issues 的部署经验，按条目保留来源和时间。**

| | 来源 | 内容 | 更新 |
|---|---|---|---|
| **[📕 中文社区（小红书）](deployment/community_field_notes_xiaohongshu.md)** | 中文社区经验、可追溯索引与黑话辞典 | LingBot-VLA / UnifoLM / π0.6 真机 RL / Figure Helix / GR00T N2 + ACT 训练成本案例 + 国标发布 + 薪资讨论 + 社区黑话辞典。 | 更新时间见条目 |
| **[📘 英文社区（HF Blog / Discord）](deployment/community_field_notes_english.md)** | HF Blog · 厂商博客 · LeRobot Discord 可追溯条目 | SmolVLA / DiffusionVLA / π0-FAST / GR00T N1.5 / OpenVLA-OFT / Figure Helix / Cosmos Policy / ACTSmooth 等前沿论文 + 真实部署经验。 | 更新时间见条目 |
| **[🔧 GitHub Issues](deployment/community_field_notes_github.md)** | 核心仓库的高互动 Issues 蒸馏 | GPU 兼容矩阵（RTX 50 系列 / Jetson）、Pi0 微调陷阱、GR00T 显存优化、训练收敛失败根因、跨仓库收敛信号。 | 更新时间见条目 |

## 📡 订阅（让新内容主动找到你）

不想每天来刷网站？订阅 RSS feed，新内容自动推送到你的阅读器：

| Feed | 内容 | 链接 |
|------|------|------|
| 🧠 **VLA 新文章** | theory/ 每日新增深度解读 | [vla-theory.xml](https://sou350121.github.io/pulsar-web/rss/vla-theory.xml) |
| ⚡ **VLA 每日信号** | ⚡🔧 级论文 + SOTA 榜（过滤 ❌📖） | [vla-daily.xml](https://sou350121.github.io/pulsar-web/rss/vla-daily.xml) |
| 📘 **AI 每日** | AI Agent 生态精选 + 深度解读 | [ai-daily.xml](https://sou350121.github.io/pulsar-web/rss/ai-daily.xml) |
| 📚 **周/双周报告** | 前瞻侦察 + 回顾分析 | [weekly.xml](https://sou350121.github.io/pulsar-web/rss/weekly.xml) |

**🎁 一键全订阅**：[OPML 导入文件](https://sou350121.github.io/pulsar-web/rss/opml.xml)（支持 Feedly / Inoreader / NetNewsWire 等）

**📖 完整使用说明**：[docs/SUBSCRIBE.md](docs/SUBSCRIBE.md) — 含各阅读器教学、CC BY 4.0 说明、FAQ

---

## 和这些方式相比

读 VLA 领域信息，大多数人用的是这四种方式——先说各自真正好在哪：

**看公众号**（机器之心 / 量子位 / PaperWeekly）：中文团队写的可读综述，编辑质量有保证，适合碎片时间、移动端阅读。
**查 GitHub Awesome 列表 / 公开综述**：整理好的资源书签，方便快速找到经典论文和开源项目。
**刷 X/Twitter 跟踪 VLA 作者**：第一时间看到作者反应和社区讨论，实时感强。
**刷小红书**：一线从业者的真实踩坑、复现参数、失败复盘——论文和公众号里绝对看不到的一手经验，评论区比正文更有料。

**选 VLA Handbook**：需要工程级深度——论文怎么跑通、部署怎么踩坑、Sim2Real 断在哪，而且每天自动更新，永久可查。中文社区笔记提供已整理的经验与出处。

| 维度 | 公众号 ML 文章 | Awesome 列表 / 综述 | X/Twitter 速报 | 小红书社区 | **VLA Handbook** |
|------|-------------|-------------------|--------------|-----------|-----------------|
| **最擅长** | 可读中文综述，移动端友好 | 资源书签，快速入手 | 实时讨论，作者第一反应 | 一线踩坑、真实参数、失败复盘 | 工程实战 + 每日自动深度解析 |
| **工程细节** | ❌ 媒体视角 | ❌ 链接汇总 | ❌ 碎片化 | ⚠️ 有但散落在评论区 | ✅ 入口脚本 · 关键超参 · shape 校验 |
| **一手踩坑** | ❌ | ❌ | ⚠️ 偶尔 | ✅ 评论区大量真实经验 | ✅ [中文蒸馏](deployment/community_field_notes_xiaohongshu.md) · [英文蒸馏](deployment/community_field_notes_english.md) |
| **更新频率** | 不定期 | 月 / 季度 | 实时 | 实时（但搜索困难） | 固定排程采集；社区更新见条目日期 |
| **历史可查** | ❌ 90 天后限流失效 | ✅ 静态存档 | ❌ 算法埋没 | ❌ 搜索质量差，帖子易沉 | ✅ Git 永久记录，全文 grep |
| **生产踩坑** | ❌ | ❌ | ❌ | ✅ 但需要自己挖掘整理 | ✅ Sim2Real · 多模态同步 · 硬件选型 |
| **趋势预测验证** | ❌ 无追踪 | ❌ | ❌ | ❌ | ✅ 双周报告与反思留档 |


---

## 先看这几篇（30 分钟内建立正确框架）

按依赖顺序排列——每一篇回答上一篇读完后自然产生的问题。

**第一层：VLA 是什么、核心设计选择是什么（~15 min）**

**① [VLA 架构总览](theory/vla-core/vla_arch.md)** `5 min`
先建立全局图：输入（视觉 + 语言）→ Backbone → Action Head → 机器人动作。RT-1 → RT-2 → OpenVLA → π0 的演化逻辑，读完你知道每个模块的作用和它们怎么拼在一起。

**② [动作生成三范式](theory/diffusion-flow/action_representations.md)** `10 min`
比较离散 Token、Diffusion 与 Flow Matching 的建模和执行取舍；速度、精度与控制频率需要结合具体实现和实验条件判断。

**第二层：动作生成的取舍与研究演化（~15 min）**

**③ [Flow Matching 原理拆解](theory/diffusion-flow/pi0_flow_matching.md)** `10 min`
从概率路径、速度场与采样过程理解 Flow Matching，再对照 π0 的实现；不把某个模型的步数或频率推广为整类方法的保证。

**④ [研究主线梳理](theory/vla-core/vla_research_mainline.md)** `5 min`
拉远一步看全局：为什么 ACT/DP 仍是 baseline、数据规模化 → 感知增强 → RL 后训练三条改进主线怎么交汇。读完你有一张完整的领域地图。

**第三层：对接现实（按需深入）**

**⑤ [社区实战笔记（中文）](deployment/community_field_notes_xiaohongshu.md)** 🔥
论文不会告诉你的事——中文社区整理的参数、失败和部署讨论。可用来定位训练数据规模、标定与部署故障的具体案例；参数需结合原实验条件阅读。

**⑥ [社区实战笔记（英文）](deployment/community_field_notes_english.md)** 🔥
英文社区精选（HF Blog + 厂商博客 + LeRobot Discord）。前沿论文深度拆解 + 真实部署经验、训练配方、推理延迟实测。

**⑦ [真机部署总入口](deployment/README.md)**
硬件选型 · 多模态同步 · Sim-to-Real · 调参 checklist。准备上真机时再看。

**想深入特定模型？**

| 方向 | 入口 | 适合谁 |
|------|------|--------|
| 开源训练栈复现 | [UnifolM-VLA-0](theory/vla-core/unifolm_vla_0_unitree_2026.md) | 想在宇树机器人上跑通 |
| 端到端代码级理解 | [Spirit-v1.5 解析](theory/vla-core/spirit_v1_5_dissection.md) | 想读懂每行 shape 变换 |
| 最新零样本迁移 | [World Action Model](theory/world-model/world_action_models_are_zero_shot_policies_dissection.md) | 想了解 2026 前沿方向 |
| 触觉 + 力控对齐 | [TAF-VLA](theory/tactile/taf_vla_tactile_force_alignment_2026.md) | 想做触觉融合 |
| VLA 知识蒸馏 | [Shallow-Pi](theory/foundation/shallow_pi_knowledge_distillation_flow_vla_2026.md) | 想做模型压缩 / 边缘部署 |
| 完整学习路线 | [学习路线图](theory/README.md) | 想系统性地从头学 |

---

## 自动更新时刻表（北京时间；计划触发时间）

| 内容 | 计划时间 | 去哪看 |
|------|---------|--------|
| 🛰️ VLA 社交情报 | 每日 09:30 | [社交归档](memory/blog/archives/vla-social-intel/) |
| 📈 SOTA / 🔧 Release | 每日 09:50 / 10:00 | [Benchmark Tracker](theory/benchmark_tracker.md) · [Release Tracker](deployment/release_tracker.md) |
| 📥 论文收集 / ⚡ 评分 | 每日 14:30 / 14:45 | [论文索引](theory/foundation/paper_index.md) |
| 🔬 理论深度解析 | 非周五：12:30、13:15、14:00、15:30、16:30；周五：19:20、19:40、20:20、21:00、21:40 | [理论目录](theory/README.md) |
| 📋 周度深度分析 | 每周五 16:30 | [周度归档](reports/weekly/README.md) |
| 📊 双周报告 / 反思 | 每 14 天，独立排程 | [双周归档](reports/biweekly/README.md) |
| 🧭 近况索引 / 网站同步 | 每日 17:15 / 18:00 | [GitHub 理论目录](theory/README.md) · [PULSAR](https://sou350121.github.io/pulsar-web/) |
| 🔎 已发布版本与全文索引 | 每小时 07、22、37、52 分 | 维护服务器本地检索；覆盖两本 Handbook 的已发布正文 |
| 🛠️ 服务器巡检 / 运维快照 | 巡检每小时 07、22、37、52 分及每日 18:45；快照每周日 06:05 | GitHub Issues 故障通知；快照后立即隔离还原核验 |
| 🏭 行业雷达 / 主线 | 每日 22:30 / 每周五 23:00 | [公司档案](companies/README.md) · [行业日档](memory/blog/archives/industry-radar/) |

时间表示触发计划，不保证该时刻已完成或有新文章。来源不足、核查失败与部分覆盖会保留状态，不能当作完整成功；计划核对日期为 2026-10-20。

---

## 项目结构

| 目录 | 内容 |
|------|------|
| [`theory/`](theory/) | 754 篇理论文档：vla-core 177 · foundation 88 · world-model 100 · planning 74 · tactile 66 · frontier 88 · diffusion-flow 49 · deployment 47 · rl 41 · perception 24 |
| [`deployment/`](deployment/) | 真机部署：硬件选型 · 多模态同步 · Sim-to-Real + 三份社区实战笔记 |
| [`reports/biweekly/`](reports/biweekly/) | 18 份双周报告 + 15 份反思（按文件分别计数）|
| [`reports/weekly/`](reports/weekly/) | 周度分析 + 每日 digest + SOTA + 风向洞察 |
| [`memory/blog/archives/`](memory/blog/archives/) | VLA 社交情报 + 已归档社区数据 |
| [`scripts/`](scripts/) | 自动化 pipeline（SCRIPTS.md 含完整 DAG）|
| [`question-bank/`](question-bank/) | 14 份面试题库与代码实战 |
| [`companies/`](companies/) | 11 份公司与行业文档 |
| [`cheat-sheet/`](cheat-sheet/) | 速查表（时间线 · 核心公式）|
| [`book/`](book/) | 电子书版本 |

---

## 快速导航

| 目标 | 入口 | 说明 |
|---|---|---|
| 补理论 / 刷面试 | [`theory/README.md`](theory/README.md) | 路线图 + 核心概念索引 |
| 找论文 / 做综述 | [`theory/foundation/paper_index.md`](theory/foundation/paper_index.md) | 多维索引 + 发展史全景图 |
| 真机落地 | [`deployment/README.md`](deployment/README.md) | 硬件选型 · 多模态同步 · Sim-to-Real |
| 社区踩坑（中文） | [`deployment/community_field_notes_xiaohongshu.md`](deployment/community_field_notes_xiaohongshu.md) | 中文社区经验 · 来源索引 · 黑话辞典 |
| 社区踩坑（英文） | [`deployment/community_field_notes_english.md`](deployment/community_field_notes_english.md) | HF Blog + Discord + 厂商博客可追溯条目 |
| GitHub 工程经验 | [`deployment/community_field_notes_github.md`](deployment/community_field_notes_github.md) | 高互动 Issues · GPU 兼容矩阵 |
| 公司 / 求职 | [`companies/README.md`](companies/README.md) | 公司档案 + 行业判断地图与日档 |
| 双周前沿报告 | [`reports/biweekly/README.md`](reports/biweekly/README.md) | 18 份报告 / 15 份反思 · VLA / 触觉 / 人形 |
| 周报 + 风向洞察 | [`reports/weekly/README.md`](reports/weekly/README.md) | 周度归档 · 论文精选 + SOTA + 趋势分析 |
| 变更记录 | [`CHANGELOG.md`](CHANGELOG.md) | 从 git 历史提炼 |

---

## 2026 前沿深度解析（专题选读）

> ⚡ = 重要进展  🔧 = 工程可用  每日持续更新中

| 论文 | 方向 |
|------|------|
| **架构与范式** | |
| [Figure Helix 0.2: Full-Body Autonomy](theory/vla-core/figure_helix_02_full_body_autonomy_2026.md) | S1+S2 双系统人形 VLA |
| [SimVLA: Simple VLA Baseline](theory/vla-core/simvla_simple_vla_baseline_robotic_manipulation_2026.md) | 极简 VLA 基线 |
| [ABOT-M0: Action Manifold Learning](theory/vla-core/abot_m0_action_manifold_learning_vla_foundation_2026.md) | 动作流形学习 |
| [StarVLA: LEGO-like VLA Codebase](theory/vla-core/starvla_lego_like_vla_codebase_2026.md) | 模块化架构 |
| [LingBot: Pragmatic VLA Foundation Model](theory/vla-core/lingbot_vla_pragmatic_vla_foundation_model_2026.md) | 语言引导 VLA |
| [ViTRA: Scalable VLA Pretraining from Human Videos](theory/vla-core/vitra_scalable_vla_pretraining_human_activity_videos_2026.md) | 人类视频预训练 |
| **世界模型** | |
| [DreamZero: World Action Models Zero-shot](theory/world-model/dreamzero_world_action_models_zero_shot_policies_2026.md) | 零样本迁移 |
| [AtomVLA: Offline Post-Training + Predictive World Models](theory/world-model/atomvla_offline_post_training_predictive_latent_world_models_2026.md) | 离线后训练 |
| [VLAW: Iterative Co-Improvement VLA + World Model](theory/world-model/vlaw_iterative_co_improvement_vla_world_model_2026.md) | VLA-WM 协同 |
| [WAM: Three Routes to Video Pretraining](theory/world-model/wam_three_routes_video_pretraining_vs_vla_2026.md) | 视频预训练路线 |
| [Video Generation Models in Robotics Survey](theory/world-model/video_generation_models_in_robotics_survey_2026.md) | 视频生成综述 |
| **强化学习后训练** | |
| [GigaBrain: World Model RL Ramp](theory/world-model/gigabrain_0_5m_star_world_model_based_rl_ramp_2026.md) | 世界模型 RL |
| [Evo-RL: Open Real-World RL](theory/rl/evo_rl_open_real_world_rl_recap_pistar06_so101_2026.md) | 真机 RL 闭环 |
| **触觉与感知** | |
| [TAF-VLA: Tactile-Force Alignment](theory/tactile/taf_vla_tactile_force_alignment_2026.md) | 触觉力控对齐 |
| [TacRefineNet: Tactile-Only Grasp Refinement](theory/tactile/tacrefinenet_tactile_only_grasp_refinement_2026.md) | 纯触觉抓取 |
| [TouchGuide: Inference-Time Touch Steering](theory/tactile/touchguide_inference_time_steering_touch_guidance_2026.md) | 推理时触觉引导 |
| [Visual-Tactile Pretraining + Online Multitask](theory/tactile/visual_tactile_pretraining_online_multitask_learning_2026.md) | 视觉-触觉预训练 |
| **效率与部署** | |
| [Shallow-Pi: Knowledge Distillation for Flow VLA](theory/foundation/shallow_pi_knowledge_distillation_flow_vla_2026.md) | VLA 知识蒸馏 |
| [QVLA: Action-Centric Quantization](theory/foundation/qvla_action_centric_quantization_2026.md) | 动作量化 |
| [RDT2-UMI: Zero-Shot Cross-Embodiment](theory/foundation/rdt2_umi_zero_shot_cross_embodiment_2026.md) | 跨具身零样本 |
| [RoboPocket: Robot-Free Policy Iteration via Phone](theory/deployment/robopocket_robot_free_instant_policy_iteration_phone_2026.md) | 手机端策略迭代 |
| **训练栈与基础设施** | |
| [UnifolM: Open-source VLA Training Stack](theory/vla-core/unifolm_vla_0_unitree_2026.md) | 开源训练栈 |
| [RynnBrain: Open Embodied Foundation Models](theory/vla-core/rynnbrain_open_embodied_foundation_models_2026.md) | 开源基础模型 |
| [Physical Intelligence Layer: Robot API](theory/deployment/physical_intelligence_layer_robot_api_2026.md) | PI Robot API |
| [NVIDIA Physical AI + Autonomous Driving](theory/deployment/nvidia_physical_ai_autonomous_driving_2026.md) | NVIDIA 物理 AI |
| [NVIDIA AI 5-Layer Cake Infrastructure](theory/deployment/nvidia_ai_5_layer_cake_infrastructure_2026.md) | NVIDIA 基建 |
| **视觉与推理** | |
| [Thinker-VLM: Embodied Intelligence](theory/planning/thinker_vlm_embodied_intelligence_2026.md) | VLM 具身推理 |
| [WaveFormer: Wave Equation Vision](theory/perception/waveformer_wave_equation_vision_2026.md) | 波动方程视觉 |
| [FAST Foundation Stereo](theory/perception/fast_foundation_stereo_real_time_zero_shot_stereo_matching_2026.md) | 零样本立体匹配 |
| [GeOPT: Geometric Pretraining for Physics Sim](theory/world-model/geopt_lifted_geometric_pretraining_physics_simulation_2026.md) | 几何预训练 |

---

## 背后的系统：照见 Pulsar

VLA Handbook 的自动采集、评分、深度解读与报告由 [照见 Pulsar](https://github.com/sou350121/Pulsar-KenVersion) 的固定管线运行，生成与搜索沿用现有 DeepSeek。先核来源、版本及主张，再决定是否发布；不能把自动生成等同于实验复现。

- **固定排程与验收**：服务器检查实际收据与产物，漏跑沿用现有有界补救；故障通过 GitHub Issues 通知，不需要电脑或 Codex 常开，不使用 Telegram。
- **三种阅读入口**：GitHub 提供目录与仓库搜索；PULSAR 网站每日 18:00 同步；维护服务器提供独立永久正文检索和原有近 60 日情报检索。晚间新稿先随定期刷新进入本地全文索引，网站在下一次同步更新。
- **已发布版本才入库**：永久全文索引跟踪已发布 Git 版本，保留文件路径和版本引用，不把草稿当成已发布资料。网站验收同时核对来源清单、逐篇 Git blob 与公开 RSS。
- **来源缺口保持可见**：截至 2026-10-20 最近完整行业实跑覆盖 28/33 家、RSS 7/7；Robot Era、Tesla Optimus、Astribot、Sanctuary AI、Covariant 仍保留来源核验告警，不降低核验标准。
- **备份范围**：每周保存运维程序与排程快照并做隔离还原；备份保留在同一台服务器，尚不属于异地备份。
- **可追溯内容**：论文与代码版本、来源日期、历史补录和未验证项分开标记；双周判断可回查报告及反思文件。
- **保护既有正文**：受管目录由固定程序维护；人工正文与旧文替换遵循 [AGENTS.md](AGENTS.md) 及维护者的具体授权。

## 🤖 让 AI 编程助手变成 VLA 专家

想让你的 AI 助手使用本 Handbook 的理论目录与可追溯资料？

**[VLA Expert Skill](https://github.com/sou350121/VLA-expert-skill)** 把 Handbook 的知识压缩成一个即插即用的 AI Skill，支持多个平台：

| 平台 | 安装方式 |
|------|---------|
| **Claude Code / Cowork** | 复制到 `.claude/skills/vla-expert/` |
| **Cursor** | 复制到 `.cursor/rules/` |
| **Codex / OpenCode** | 作为 system prompt 加载 |

该 Skill 提供论文评估、研究方向判断、产业分析与部署指南等工作流；功能和同步范围以其说明为准，结论需要回到引用原文核对。

👉 **[github.com/sou350121/VLA-expert-skill](https://github.com/sou350121/VLA-expert-skill)**

---

## 贡献

欢迎提 Issue 和 PR：补论文解读 · 真机经验 · 面试题。见 `CONTRIBUTING.md`。

## 许可证

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh) · 由 [照见 Pulsar](https://github.com/sou350121/Pulsar-KenVersion) 系统自动驱动
