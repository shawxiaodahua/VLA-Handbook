# 机器人数据采集研究目录 (Robot Data Collection Research Catalog)

> ⚙️ 研究整理：2026-10-25。观察窗口：2026-04-25 至 2026-10-25。
> 核验公开论文、作者项目、代码入口及数据卡；未下载大型数据、接受访问条款或复现训练。

本目录将 **86 个研究／工具项目**按八个工程问题组织。它不是“86 篇半年新论文”：按本次记录，57 项关联窗口内的论文首发／官方页面日期，16 项为窗口前基础，13 项首发日期未核定。普通页面的日期不代替论文首发；不同版本、同一数据的格式与公开子集也不另算独立项目。

先确定你缺的是动作、接触、场景覆盖、全身协调还是质量证据，再挑采集路线。小时数、帧数和机器人可执行动作的数量不能直接换算。

```text
你想补什么监督？
├─ 工具运动/动作接口 → 手持与无机器人采集
├─ 人类行为/视觉覆盖 → 人类视频与姿态
├─ 接触/压力/振动 → 触觉与力觉
├─ 平衡/移动/手部协调 → 全身采集
├─ 失败/恢复与结果 → 自动真机采集
├─ 可控扩展与重定向 → 合成数据引擎
└─ 来源接入与数据选择 → 数据集矩阵 / 质检工具
```

## 八个专题入口

| 工程问题 | 项目数 | 阅读入口 |
|---|---:|---|
| 手持接口与无机器人采集 | 11 | [专题与接入研究卡](./robot_free_collection_interfaces_2026.md) |
| 人类视频、姿态与动作监督 | 11 | [专题与接入研究卡](./human_video_robot_data_2026.md) |
| 触觉、力觉与接触数据采集 | 15 | [专题与接入研究卡](./contact_rich_data_collection_2026.md) |
| 人形全身与灵巧操作采集 | 14 | [专题与接入研究卡](./humanoid_teleoperation_data_2026.md) |
| 自动真机采集、接管与恢复 | 6 | [专题与接入研究卡](./autonomous_robot_data_collection_2026.md) |
| 合成、重定向与仿真数据引擎 | 12 | [专题与接入研究卡](./synthetic_robot_data_engines_2026.md) |
| 开放数据集与访问状态 | 7 | [专题与接入研究卡](./robot_dataset_access_matrix_2026.md) |
| 数据筛选、质检与工具链 | 10 | [专题与接入研究卡](./robot_data_curation_tools_2026.md) |

## 先读的两条接触路线

- [PolyUMI / VisTA 深读](../theory/tactile/polyumi_vista_collection_2026.md)：同一传感指连接手持采集与机器人执行；保留论文与 README 的规格差异。
- [已有 TouchScale 深读](../theory/tactile/touchscale_500_hours_of_human_vision_and_touch_for_visual_ta_dissection.md)：无动作标签的人类视触觉中间训练，仍有真机后训练。复用既有文章，不另造重复稿。
- [采集验收与有效成本](./robot_data_collection_acceptance.md)：记录字段、时间／坐标、独立样本、访问条件与留出评测。

## 项目逐项导航

下表“时间”对论文使用 arXiv 首发；其他条目保守标为页面／资料日期，无法核定则明确未知。卡片中的机制和数字均为来源报告，工程检查是本文建议。

### 手持接口与无机器人采集

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [UMI](./robot_free_collection_interfaces_2026.md#p-umi) | 首发 2024-02-15；窗口前基础 | 提出手持夹爪采集接口与可直接部署到机器人的策略框架，是 UMI 类数采的基础一手来源。 |
| [UMI-FT](./robot_free_collection_interfaces_2026.md#p-umi-ft) | 首发 2026-01-15；窗口前基础 | 手持式UMI改装件，每指加装六轴力/力矩传感器，直接面向力敏感操作的真机数据采集。 |
| [UMI-3D](./robot_free_collection_interfaces_2026.md#p-umi-3d) | 首发 2026-04-15；窗口前基础 | 在UMI腕持夹爪上加入LiDAR做LiDAR-惯性SLAM，直接提升演示采集的位姿可靠性与数据质量。 |
| [MV-UMI](./robot_free_collection_interfaces_2026.md#p-mv-umi) | 首发 2025-09-23；窗口前基础 | 直接提出手持夹爪多视角数采接口并开源采集/训练流水线，属一手采集方法。 |
| [RoboPocket](./robot_free_collection_interfaces_2026.md#p-robopocket) | 首发 2026-03-05；窗口前基础 | 手机化机器人无关手持采集接口，带实机位姿精度与数据有效性量化，是UMI型数采一手来源。 |
| [EgoGuide](./robot_free_collection_interfaces_2026.md#p-egoguide) | 首发 2026-06-12 | 为UMI式无机器人示教采集增加腕+头戴双眼同步观测与在线覆盖度引导，直接改进入数采质量。 |
| [BRIDGE](./robot_free_collection_interfaces_2026.md#p-bridge) | 首发 2026-06-25 | 直接涉及UMI手持采集与遥操作数据混合使用的一手采集/监督口径研究。 |
| [KIW](./robot_free_collection_interfaces_2026.md#p-kiw) | 首发 2026-09-19 | 面向UMI式手持采集的硬件与离线重建方案，直接关系双臂移动操作演示的采集接口与位姿恢复质量。 |
| [Grabette](./robot_free_collection_interfaces_2026.md#p-grabette) | 页面资料 2026-07-21；非首发认定 | 这是UMI思路的开源手持夹爪采集系统发布说明，直接涉及一手数据的采集硬件、同步与格式转换。 |
| [Orbbec robot-free hardware](./robot_free_collection_interfaces_2026.md#p-orbbec-robot-free-hardware) | 页面资料 2026-08-05；非首发认定 | 提供面向UMI/EGO/WristCam的无机器人数据采集硬件平台，属采集工具而非数据集本体。 |
| [HiFi-UMI](./robot_free_collection_interfaces_2026.md#p-hifi-umi) | 首发 2026-07-28 | 便携UMI采集系统并公开发布2000小时实采示范子集，直接涉及机器人数据生产与质量口径。 |

### 人类视频、姿态与动作监督

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [EgoLive](./human_video_robot_data_2026.md#p-egolive) | 首发 2026-04-26 | 面向机器人操作学习的大规模真人自我中心实采数据集，直接提供人类示范采集硬件与多模态标注口径。 |
| [Open-AoE](./human_video_robot_data_2026.md#p-open-aoe) | 首发 2026-07-15 | 它是公开的自中心人类操作数据集与工具链，直接提供人类演示数据与结构化标注。 |
| [HumanNet](./human_video_robot_data_2026.md#p-humannet) | 首发 2026-05-07 | 百万小时人类视频语料，直接为人到机器人迁移提供采集与策展口径。 |
| [Do as I Do](./human_video_robot_data_2026.md#p-do-as-i-do) | 首发 2026-06-17 | 直接提出从单目 RGB 人类视频重建并重定向到多指灵巧手的采集管线，属人类视频转机器人数据的一手来源。 |
| [EgoSuite-Open100K](./human_video_robot_data_2026.md#p-egosuite-open100k) | 首发待核 | 公开10,000小时（计划10万小时）第一视角人类活动数据，属具身数采一手来源。 |
| [RekaDaily-10k](./human_video_robot_data_2026.md#p-rekadaily-10k) | 页面资料 2026-09-16；非首发认定 | 直接公布人类自我中心家庭操作数据集的规模、分层、采集与质量口径。 |
| [EgoVerse](./human_video_robot_data_2026.md#p-egoverse) | 首发待核 | 跨机构自我中心人类演示数据集及其处理/训练代码的官方仓库。 |
| [EgoScale](./human_video_robot_data_2026.md#p-egoscale) | 页面资料 2026-02-19；非首发认定；窗口前基础 | 直接给出大规模自我中心人类视频->灵巧操作迁移的数据规模、动作标注与训练接口。 |
| [Gen-HumanEgo](./human_video_robot_data_2026.md#p-gen-humanego) | 首发待核 | 这是带手部重建与层级标注的第一人称人类操作数据卡，可直接用于人→机器人转化。 |
| [World In Your Hands](./human_video_robot_data_2026.md#p-world-in-your-hands) | 首发 2025-12-30；窗口前基础 | 大规模野外第一人称人类操作数据集与可穿戴采集硬件，直接服务机器人策略学习。 |
| [HumanVerse / λ0](./human_video_robot_data_2026.md#p-humanverse-0) | 首发 2026-10-13 | 500小时开放世界第一视角人体全身+手部运动数据集及人-人形对齐管线，直接用于人形移动操作训练。 |

### 触觉、力觉与接触数据采集

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [TouchScale](./contact_rich_data_collection_2026.md#p-touchscale) | 首发 2026-10-07 | 500小时统一穿戴采集的人手视觉-触觉数据，并验证其对机器人策略mid-training的增益。 |
| [PolyUMI / VisTA](./contact_rich_data_collection_2026.md#p-polyumi-vista) | 首发 2026-09-24 | 提供开源无线手持UMI式夹爪，同步采集视觉-触觉-音频-本体感知示教并复用于机器人执行。 |
| [HapTile](./contact_rich_data_collection_2026.md#p-haptile) | 首发 2026-06-03 | 含指尖视触觉与遥操作触觉反馈的接触丰富操作数据集，直接针对VLA数据采集接口。 |
| [TouchAnything / EgoTouch](./contact_rich_data_collection_2026.md#p-touchanything-egotouch) | 首发 2026-05-13 | 双手触觉压力与多视角第一人称同步实采数据集，直接面向触觉监督采集。 |
| [RCT](./contact_rich_data_collection_2026.md#p-rct) | 首发 2026-06-30 | 机器人实采的触觉-视觉-语言数据集，直接对应触觉数据采集与评测口径。 |
| [DexJoCo](./contact_rich_data_collection_2026.md#p-dexjoco) | 首发 2026-05-15 | 提供灵巧手任务基准、低成本手套遥操作采集系统与1.1K轨迹数据口径，直接关系数据采集与评测。 |
| [ForceBand](./contact_rich_data_collection_2026.md#p-forceband) | 首发 2026-06-24 | 直接涉及人类演示的力觉监督采集硬件与 10 小时多模态数据集，是人力数据+力信号采集的一手来源。 |
| [FACTR 2](./contact_rich_data_collection_2026.md#p-factr-2) | 首发 2026-06-10 | 提供无专用力传感器下的外力矩估计与力反馈遥操作，涉及力觉数据采集与再采样训练口径。 |
| [Ego2Contact](./contact_rich_data_collection_2026.md#p-ego2contact) | 首发 2026-10-16 | 从人类第一视角视频预测触觉并迁移到机器人手，直接涉及触觉监督与数据接口。 |
| [OSMO](./contact_rich_data_collection_2026.md#p-osmo) | 首发待核 | 触觉手套开源硬件与人到机器人技能迁移数据流水线的直接一手说明。 |
| [FreeTacMan](./contact_rich_data_collection_2026.md#p-freetacman) | 首发待核 | 免机器人可穿戴视触觉采数系统及配套大规模数据集，直接关系接触丰富操作采数接口。 |
| [DexViTac](./contact_rich_data_collection_2026.md#p-dexvitac) | 首发 2026-03-18；窗口前基础 | 直接提供人类视觉-触觉-运动学多模态采集硬件与2400+演示数据集。 |
| [OmniUMI](./contact_rich_data_collection_2026.md#p-omniumi) | 首发 2026-04-12；窗口前基础 | 扩展UMI式手持采集到触觉、内外力多模态，属直接数据采集接口。 |
| [ViTaMIn](./contact_rich_data_collection_2026.md#p-vitamin) | 首发 2025-04-08；窗口前基础 | 在手持夹爪采集流程上加入全向触觉传感，直接涉及视触觉演示数据采集与策略学习。 |
| [Tactile Genesis](./contact_rich_data_collection_2026.md#p-tactile-genesis) | 首发 2026-06-21 | 提供可配置触觉传感抽象的大规模仿真平台，是触觉数采/表征设计的关键工具而非直接实采数据集。 |

### 人形全身与灵巧操作采集

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [BifrostUMI](./humanoid_teleoperation_data_2026.md#p-bifrostumi) | 首发 2026-05-05 | 直接提出机器人无关的人形全身示教采集系统并验证数据可训练部署策略。 |
| [MATE](./humanoid_teleoperation_data_2026.md#p-mate) | 首发 2026-09-22 | 多操作员VR虚拟遥操作采集多人形协作全身数据，属直接采集平台与数据集。 |
| [IMU Whole-Body Teleoperation](./humanoid_teleoperation_data_2026.md#p-imu-whole-body-teleoperation) | 首发 2026-05-12 | 提供IMU动捕到人形全身实时遥操作与零修改sim2real的采集/重定向管线，可作数据采集工具参考。 |
| [Whole-Body UMI](./humanoid_teleoperation_data_2026.md#p-whole-body-umi) | 首发 2026-09-19 | 用原生UMI末端轨迹示教经运动先验生成人形全身参考，直接涉及全身示教采集与迁移。 |
| [EgoExplorer](./humanoid_teleoperation_data_2026.md#p-egoexplorer) | 首发 2026-10-13 | 100小时可穿戴人类全身运动+手指关节+触觉的自采数据集，覆盖全身协调与接触，属于一手采集资源。 |
| [EgoHTR](./humanoid_teleoperation_data_2026.md#p-egohtr) | 首发 2026-07-15 | 提供崎岖地形上场景对齐的人体全身 4D 运动数据，直接用于全身/运动技能学习。 |
| [HiPHI](./humanoid_teleoperation_data_2026.md#p-hiphi) | 首发 2026-08-17 | 大规模光学动捕全身运动与物体交互数据集，直接面向人形全身策略学习。 |
| [HumanoidArena](./humanoid_teleoperation_data_2026.md#p-humanoidarena) | 首发 2026-06-16 | 面向拟人全身分层策略的仿真基准，含闭环遥操作示教数据管线与多GMT数据集规范。 |
| [ModPack](./humanoid_teleoperation_data_2026.md#p-modpack) | 首发 2026-07-21 | 可穿戴背包式模块化遥操作接口，直接用于双臂移动操作示教采集与策略部署。 |
| [HoMMI](./humanoid_teleoperation_data_2026.md#p-hommi) | 首发 2026-03-03；窗口前基础 | 机器人无关人体演示直接采集全身移动操作数据，是UMI扩展的一手数采框架。 |
| [TWIST2](./humanoid_teleoperation_data_2026.md#p-twist2) | 首发 2025-11-04；窗口前基础 | 提出无动捕、单操作员的全身人形遥操作与数据采集系统，并开源数据，直接关系数采口径。 |
| [EgoHumanoid-V2](./humanoid_teleoperation_data_2026.md#p-egohumanoid-v2) | 首发 2026-09-29 | 用机器人自由的自我中心人体示教（GoPro+PICO）直接监督人形全身loco-manipulation，属一手人因数据转化采集。 |
| [Ropedia W0](./humanoid_teleoperation_data_2026.md#p-ropedia-w0) | 首发 2026-10-19 | 构建跨人/机器人本体的全身移动操作数据集与世界模型，含自采人形遥操作基准。 |
| [Universal Manipulation Exoskeleton](./humanoid_teleoperation_data_2026.md#p-universal-manipulation-exoskeleton) | 首发 2026-06-12 | 上肢外骨骼可同时记录整臂构型与关节力矩，直接产出全身/双臂柔性操作示教数据。 |

### 自动真机采集、接管与恢复

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [AutoDex](./autonomous_robot_data_collection_2026.md#p-autodex) | 首发 2026-06-22 | 端到端无人接管的真实灵巧抓取数据采集系统，直接产出带物理成功/失败标签的实采数据。 |
| [Zero2Skill](./autonomous_robot_data_collection_2026.md#p-zero2skill) | 首发 2026-07-15 | 它是自主采集真实机器人操作数据的系统，直接涉及数据采集流程与数据质量。 |
| [RECALL](./autonomous_robot_data_collection_2026.md#p-recall) | 首发 2026-06-22 | 提供不确定性触发的恢复数据采集与回放混合的定量口径，是主动数据筛选与灾难性遗忘权衡的关键方法来源。 |
| [ENPIRE](./autonomous_robot_data_collection_2026.md#p-enpire) | 首发 2026-06-18 | 展示编码智能体在真机上自动完成重置、验证与策略迭代的采集-评测闭环，是机器人数据/评测基础设施的一手工程来源。 |
| [RoboCade](./autonomous_robot_data_collection_2026.md#p-robocade) | 首发 2025-12-24；窗口前基础 | 用游戏化远程遥操作扩大机器人演示数据采集的实采系统与数据。 |
| [PlayWorld](./autonomous_robot_data_collection_2026.md#p-playworld) | 首发 2026-03-09；窗口前基础 | 提出全自动机器人自博弈数采流水线并用于训练视频世界模型，直接涉及数采接口与数据口径。 |

### 合成、重定向与仿真数据引擎

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [Ego2Robot](./synthetic_robot_data_engines_2026.md#p-ego2robot) | 首发 2026-08-03 | 直接给出从第一视角人类视频合成机器人训练数据的管线、数据口径与规模，是机器人数采的一手合成数据来源。 |
| [HuRo](./synthetic_robot_data_engines_2026.md#p-huro) | 首发 2026-09-09 | 将异构人类自我中心视频机器化为带观测-动作-语言对齐的伪机器人episode，直接构成VLA预训练数据来源。 |
| [RoboEdit](./synthetic_robot_data_engines_2026.md#p-roboedit) | 首发 2026-08-19 | 直接把人类操作视频编辑为机器人视频并生成3D机器人状态，提供成对视觉与运动监督，属人转机器人数据的核心一手来源。 |
| [Mirror Learning](./synthetic_robot_data_engines_2026.md#p-mirror-learning) | 首发 2026-07-30 | 该工作用视频扩散模型把第三人称示范转成伪第一人称视频，再用逆动力学模型恢复动作，是合成示范数据的采集接口参考。 |
| [AdvDex](./synthetic_robot_data_engines_2026.md#p-advdex) | 首发 2026-08-14 | 提出大规模人类操作数据集OmniShare与跨本体关节对齐动作空间，服务灵巧操作策略预训练。 |
| [Dex-X](./synthetic_robot_data_engines_2026.md#p-dex-x) | 首发 2026-09-07 | 以人类单目视频经仿真补全触觉监督，训练可部署视觉-触觉灵巧操作策略，直接关系合成数据与触觉信号生成。 |
| [GRAIL](./synthetic_robot_data_engines_2026.md#p-grail) | 首发 2026-06-03 | 用世界模型与三维资产合成人形移动操作数据，并只凭合成数据完成真机部署的生成式数据采集流程。 |
| [OSCAR](./synthetic_robot_data_engines_2026.md#p-oscar) | 首发 2026-06-03 | 面向策略评测的合成世界模型，含跨机器人/人类数据的过滤去重筛选口径。 |
| [EmbodiedSmith](./synthetic_robot_data_engines_2026.md#p-embodiedsmith) | 首发 2026-10-06 | 以仿真资产-场景-任务联合生成与执行反馈闭环产生机器人示教数据的一手系统。 |
| [DREAM](./synthetic_robot_data_engines_2026.md#p-dream) | 首发 2026-08-29 | 提供部署现场实景重建+TAMP自动生成图像-动作数据用于VLA微调的一手方案。 |
| [OASIS](./synthetic_robot_data_engines_2026.md#p-oasis) | 首发 2026-06-07 | 提出全仿真采集的人形全身移动操作数据管线，是纯合成数据与实机遥操作对比的一手证据。 |
| [NVIDIA V2D](./synthetic_robot_data_engines_2026.md#p-nvidia-v2d) | 页面资料 2026-08-01；非首发认定 | 公开的合成机器人灵巧操作数据集，含人示教重定向与机器人执行片段，属一手数据来源。 |

### 开放数据集与访问状态

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [ABC-130K](./robot_dataset_access_matrix_2026.md#p-abc-130k) | 首发待核 | 这是双臂 YAM 遥操作实采轨迹数据集卡，直接提供机器人数采的接口与口径信息。 |
| [AGIBOT WORLD 2026](./robot_dataset_access_matrix_2026.md#p-agibot-world-2026) | 页面资料 2026-04-07；非首发认定；窗口前基础 | 官方发布真实场景多模态灵巧/全身采集数据集，直接构成VLA数据来源。 |
| [HIW-500](./robot_dataset_access_matrix_2026.md#p-hiw-500) | 首发待核 | 这是全身人形机器人在真实家庭遥操作采集的数据集页面，含规模和模态口径，是全身数据一手来源。 |
| [BONES-SEED](./robot_dataset_access_matrix_2026.md#p-bones-seed) | 首发待核 | 它提供带语言标注的人体全身动作捕捉并重定向到 Unitree G1 关节轨迹，是人形全身控制与模仿学习的直接数据来源候选。 |
| [RoboCOIN](./robot_dataset_access_matrix_2026.md#p-robocoin) | 首发待核 | 提供多机器人双手机器人操作数据集及标准加载、遥操作与采集脚本，是直接数据来源。 |
| [Daimon-Infinity / TacClaw](./robot_dataset_access_matrix_2026.md#p-daimon-infinity-tacclaw) | 首发待核 | 直接采集的视触觉-语言-动作数据集，含手持UMI类设备与穿戴外骨骼两条采集路径。 |
| [N0-Foundation / NeoData](./robot_dataset_access_matrix_2026.md#p-n0-foundation-neodata) | 首发 2026-08-30 | 同时覆盖触觉传感硬件、手持TacUMI采集、大规模视触觉数据集与评测，直接服务于触觉数采一手核对。 |

### 数据筛选、质检与工具链

| 项目 | 时间口径 | 阅读问题 |
|---|---|---|
| [HFlow](./robot_data_curation_tools_2026.md#p-hflow) | 首发待核 | 面向机器人多模态数据管线与质检的开源SDK，直接涉及数据质量证据与筛选。 |
| [Flora](./robot_data_curation_tools_2026.md#p-flora) | 页面资料 2026-04-25；非首发认定 | 大规模数据引擎，为人-机统一标注流程与奖励/描述标注提供可复用工具。 |
| [LeRobotDataset v3](./robot_data_curation_tools_2026.md#p-lerobotdataset-v3) | 首发待核 | 机器人数采的标准化数据格式规范，决定存储、元数据与流式加载口径。 |
| [DataMIL](./robot_data_curation_tools_2026.md#p-datamil) | 首发待核 | 提供面向模仿学习的数据选择方法与OXE实机评测，属数据筛选关键工具。 |
| [CUPID](./robot_data_curation_tools_2026.md#p-cupid) | 首发 2025-06-23；窗口前基础 | 直接面向机器人示范数据的筛选与子选择，属于数据质量管理与采集后处理核心方法。 |
| [HELP](./robot_data_curation_tools_2026.md#p-help) | 首发 2026-07-08 | 直接讨论人机接管、自动rollout分段与混合质量数据筛选的采集/后训练口径。 |
| [Phase-Localized Curation](./robot_data_curation_tools_2026.md#p-phase-localized-curation) | 首发 2026-06-13 | 直接针对机器人演示数据筛选/质量度量策略的受控负面结果，属数据口径与筛选方法一手证据。 |
| [SCRIM](./robot_data_curation_tools_2026.md#p-scrim) | 首发 2026-10-13 | 提出以逐状态关键性打分的模仿学习数据筛选与加权方法，直接关系数据质量口径。 |
| [Ego4WAM](./robot_data_curation_tools_2026.md#p-ego4wam) | 首发 2026-09-30 | 系统研究第一人称人类数据对齐、时长/多样性、监督与使用方式对机器人学习的影响，直接关系数据采集与质量口径。 |
| [REAL-I Lessons](./robot_data_curation_tools_2026.md#p-real-i-lessons) | 首发 2026-09-12 | 以固定数据预算的真实机器人挑战赛为背景，给出数据筛选、checkpoint选择与部署接口的经验，属于数据质量与评测口径的关键参考。 |

## 本轮排除与不确定项

- [HumanoidUMI: arXiv:2606.27239](https://arxiv.org/abs/2606.27239) 的摘要页明确标为撤稿，本轮不将其列为有效新成果。BifrostUMI 按自己的 arXiv 身份及正文阅读，不自行断言作者确认的替代关系。
- 镜像、论文笔记和聚合站用于发现线索，不作为数据规模、许可或发布完成的最终依据。
- 代码／项目页可读不等于完整数据已下载。HiFi-UMI、EgoSuite、NeoData、HiPHI 等需特别区分内部／公开、原始／派生和版本口径。
- HTML／Markdown 抽取可能打乱内嵌数学和尺寸；关键配置应与官方 PDF、表格或实际记录交叉核对。

## 后续值得补收的证据

优先记录新公开子集及访问条款、版本化硬件与标定协议、真实多模态缺失测试、目标机器人留出结果，以及按独立合格 episode 计算的有效成本。只增加宣传规模而没有交付状态或任务收益时，先留作线索。

本目录提供研究入口与接入问题，不代表这些外部数据、软件或采集设备已安装、下载或接入任何自动流程。

---

[← 部署目录](./README.md) · [近半年地图](./robot_data_collection_landscape_2026.md) · [论文索引](../theory/foundation/paper_index.md#數據)
