# VLA Handbook 工程专题：公开机器人数据集的访问、许可与验收 / Engineering Brief: Access, Licensing, and Acceptance for Public Robot Datasets

> 整理日期：2026-10-25。观察窗口：2026-04-25 至 2026-10-25；窗口外或日期未知的旧基础另标注。本专题基于公开材料核验条目与措辞，未下载大数据、未复现实验，也未逐篇复现论文。所有条目是阅读/接入研究卡，不是完整复现笔记。

## 核心结论

本专题的核心不是“哪个数据集更大”，而是“公开交付物、当前卡片所列访问路径与可验收证据是否一致”。可公开下载的数据卡、需申请的门控数据、代码仓库和厂商发布规划必须分开记录，不能把代码许可当作数据许可，也不能把“已在某平台开源”的公告当成“本次已下载并通过验收”。ABC-130K 的卡片自述 Apache 2.0，但只是数据集卡证据；AGIBOT WORLD 2026 三个主题分别发布，访问与许可均未在本次核验中确认，且不得把三个主题的数量相加。BONES-SEED 的 G1 轨迹是重定向结果而非机器人实采，HIW-500 是公开子集且页面称有额外数据需联系获取。Daimon-Infinity 的“将公开”为规划表述，OpenNeoData 是 gated 非商业研究访问。NeoData 论文所述全量规模与 OpenNeoData 数据卡自述规模口径不同，须并列记录、分别标注来源，不能预设数据卡一定正确。

ASCII 链路（读卡 → 判断交付物 → 申请/下载 → 验收）：

```
数据卡/README ──> 交付物类型判定 ──> 访问路径 (开放/gated/规划)
      │                  │                       │
      │                  ├─ 代码公开 ≠ 数据许可   ├─ 未下载: 标"未核实"
      │                  └─ 派生字段 vs 实采      └─ 获批后: schema/时限/许可核验
      ▼
  证据强度分级: 卡片自述 < 公告 < 实下文件核验
```

## 路线比较（按工程问题）

| 工程问题 | 项目 | 当前卡片访问状态 | 许可/门控要点 | 接入首要检查 |
|---|---|---|---|---|
| 双机械臂开放数据快速起步 | ABC-130K | HF 公开仓库，未声明 gated；本次未下载 | 卡片自述 Apache 2.0 | episode.mcap 时间戳与标注覆盖率 |
| 实采多模态+长尾交互 | AGIBOT WORLD 2026（三主题分别看） | 公告称已开源并给 HF 地址，未验证可下载；许可 unknown | 三主题不可相加；许可未明 | HF 子集齐全性与触觉/力控字段 |
| 人形家庭场景遥操作 | HIW-500 | HF 公开，页面称有额外数据需联系；查看器报列名不匹配 | CC BY 4.0（卡片声明） | 两种格式 schema 差异与动作是否为原始指令 |
| 动捕+重定向人形轨迹 | BONES-SEED | 门控表单，学术/低营收条件 | bones-seed-license | v003 主表、v002 时间标签与 G1 接触信息的对应关系 |
| 多本体 LeRobot 兼容聚合 | RoboCOIN | 网页浏览+HF/ModelScope 下载；逐数据集许可 unknown | 派生 action/EEF 需分辨 | 目标数据集版本与 action 可用性 |
| 触觉中心 VTLA（规划+样例） | Daimon-Infinity / TacClaw | 页面称将公开；有 Lite 样例；完整数据未验证 | CC BY-NC-SA 4.0 声明 | 占位维度比例与触觉解码 |
| 大规模视触觉示范（论文+子集） | N0-Foundation / NeoData | 论文称发布 OpenNeoData 子集；OpenNeoData 为 gated，未验证 | 论文 CC BY-NC-SA ≠ 数据许可 | 论文与数据卡口径并列核验 |
| 视触觉+UMI 异构本体 | OpenNeoData | gated，需提交用途并同意不重分发 | CC BY-NC-SA 4.0（README 声明） | 获批后 umi/ur 子集 schema 与触觉流 |

---

### ABC-130K

<a id="p-abc-130k"></a>

**页面日期/窗口状态**：未知（undated），不冒充首发。**机制与监督**：双臂 YAM 站台遥操作采集的双臂操作轨迹，episode.mcap 含每臂关节状态与动作、夹爪开合度、压缩视频流与相机标定；卡片自述约 3,590.7 小时、130,703 条轨迹，其中 42,980 条带标注。监督信号为关节/夹爪指令与视频，未说明是否存在人工接管或全自动评测流程。**边界**：“largest open-source”为卡片自述，未经独立核实。**接入检查**：抽查若干 episode.mcap 的时间戳是否均为纳秒绝对时间、视频与关节流是否需最近邻匹配、annotation.mcap 覆盖率，以及卡面与仓库许可证是否均为 Apache 2.0。来源：[ABC-130K 数据集卡](https://huggingface.co/datasets/XDOF/ABC-130k/blob/main/README.md?code=true)。

### AGIBOT WORLD 2026（三主题：模仿学习 / 多样交互 / 强化学习）

<a id="p-agibot-world-2026"></a>

**页面日期/窗口状态**：三主题页面日期分别为 2026-04-07（模仿学习，窗口外）、2026-06-08（多样交互，窗口内）、2026-09-01（强化学习，窗口内）；均按厂商页面日期标注，非数据首发或修订界定。**机制与监督**：模仿学习主题称数据采自真实环境、基于 G2 采集，同步采集多视角 RGB(D)、触觉、LiDAR、IMU、全身关节与力传感器，分五阶段开源；多样交互主题称显式记录抓取失败、碰撞、掉落、液体飞溅等长尾，采用遥操作范式；强化学习主题首批称含 11,430 条真实轨迹、14 类任务，含成功与失败 policy rollout 及人工在环纠正，并给出子任务、扰动、错误状态与干预分段标注。**边界**：三主题是分别发布的不同主题，不得把各主题的轨迹数相加；三个公告页面均未提供本次可核验的独立许可证文本，因此许可证状态记为未核实；公告称已开源不等于许可证文件或本次下载验收。**接入检查**：核对 HF 仓库各阶段子集是否齐全、三类轨迹与接管段的时间戳对齐、触觉/力控字段采样率与单位、失败 rollout 标注口径、许可文件是否存在。来源：[模仿学习主题](https://www.agibot.com.cn/article/315/detail/148.html)；[多样交互主题](https://www.agibot.com.cn/article/315/detail/167.html)；[强化学习主题](https://www.agibot.com/article/231/detail/95.html)。

### HIW-500

<a id="p-hiw-500"></a>

**页面日期/窗口状态**：未知（undated）。**机制与监督**：Unitree G1 在东南亚真实家庭中的人类全身遥操作演示，每 episode 含 29-DoF 关节状态、末端状态、IMU、里程计与遥操作动作轨迹；卡片自述 500+ 小时、23K+ episodes、约 10 TB、12 个真实家庭、161 个子任务标签。**边界**：遥操作由人接管，不是全自动策略执行；数据为公开子集，原文提示更大/额外数据需联系获取；页面数据集查看器加载失败并报列名不匹配，实际文件下载未验证。**接入检查**：核对两种发布格式路径与列 schema 差异，确认 CC BY 4.0 是否覆盖全部下载文件，以及遥操作动作为原始关节指令还是后处理重放。来源：[HIW-500 数据集卡](https://huggingface.co/datasets/BitRobot/HIW-500)。

### BONES-SEED

<a id="p-bones-seed"></a>

**页面日期/窗口状态**：未知（undated）。**机制与监督**：动捕人体动作数据，卡片自述 142,220 条动作（71,132 原始+71,088 镜像）、约 288 小时、120fps，输出 SOMA Uniform、SOMA Proportional 与 Unitree G1 MuJoCo 兼容三种格式，每条动捕最多 6 条自然语言描述加时序分段与技术描述。**边界**：数据为动捕动画而非机器人实采，G1 轨迹是重定向结果，不能当实机轨迹使用；访问需门控表单，且要求学术用户或营收门槛条件；主 metadata 为 seed_metadata_v003.parquet；另有 seed_metadata_v002_temporal_labels.jsonl 时间分段标签，二者文件角色不同，不能仅凭版本号不同认定冲突。**接入检查**：核对 tar 包与 v003 主表、v002 时间分段标签的对应关系、G1 重定向轨迹是否保留接触/物体信息，以及商业许可边界。来源：[BONES-SEED 数据集卡](https://huggingface.co/datasets/bones-studio/seed/blob/main/README.md)。

### RoboCOIN

<a id="p-robocoin"></a>

**页面日期/窗口状态**：未知（undated）。**机制与监督**：维护 LeRobot 数据格式兼容并额外支持子任务、场景描述、运动描述等元数据；标准化观测/动作含关节角弧度、夹爪开合[0,1]，以及经仿真统一坐标系获得的末端位姿；缺少动作数据时由 observation.state 序列派生 action。**边界**：仿真派生 EEF 位姿与派生 action 并非全部字段都是实采原始信号；数据许可随各数据集页面，未在此页面明确；页面 News 中的规模数字未独立核实。**接入检查**：确认目标数据集实际版本（v2.1/v3.0）、逐数据集许可，以及派生 action 的数据在训练中的可用性，检查遥操作录制是否含人工接管。来源：[RoboCOIN 仓库](https://github.com/FlagOpen/RoboCOIN)。

### Daimon-Infinity / TacClaw

<a id="p-daimon-infinity-tacclaw"></a>

**页面日期/窗口状态**：未知（undated）。**机制与监督**：页面称计划开源 10,000+ 小时 VTLA 多模态数据，首批 1000 小时，多数由 DM-DataClaw（触觉 UMI 设备，广角鱼眼、双目+IMU 轨迹跟踪、指尖高分辨触觉与夹爪位置编码器）采集，少量由基于 TacExoGaze 的 DM-DataDex 采集；动作为遥操作命令。**边界**：“将公开”为规划表述，仅 Lite 样例有明确访问指向，完整数据公开状态未知；统计基于约 1209 小时已采集数据，与首批 1000 小时或总量 10,000+ 口径不一致，以实际发布为准；页面称实际编码与格式可能随批次变化。**接入检查**：核实 Lite 样例与实际仓库中 DataClaw/DataDex 子集的轨迹数、时长与触觉解码脚本，检查占位维度比例、左右流同步，并确认完整数据是否在 ModelScope 上线及是否有独立许可文件。来源：[Daimon-Infinity 数据集卡](https://huggingface.co/datasets/daimonrobotics/Daimon-Infinity)。

### N0-Foundation / NeoData

<a id="p-n0-foundation-neodata"></a>

**页面日期/窗口状态**：arXiv 提交记录 2026-08-30（v1，窗口内），2026-09-20 修订为 v2（修订版本不是新发布）。**机制与监督**：论文称 NeoData 含同步视觉与触觉示范，覆盖多种本体与任务，N0-TacUMI 贡献大部分轨迹，其余本体提供可执行机器人行为；每回合经完整性、运动质量、回合质量、视频质量与 IK 可执行性等质量检查；并声明发布 OpenNeoData 开源子集。**边界**：直接核对 arXiv v2 摘要：NeoData 总库为 30,000+ 小时、六种本体、450 项任务；OpenNeoData 为 5,000 小时公开子集。当前数据卡自述 200k+ 轨迹、5,000+ 小时、257 项任务；总库与公开子集的任务数不能视为同一分母，也不相加。本文不采用 HTML 抽取中语义未厘清的“200 小时”作为公开规模，不把论文许可当作数据许可。论文许可 CC BY-NC-SA 4.0 不能推断为数据许可；触觉表示暂限于平行夹爪视触觉手指。**接入检查**：打开 OpenNeoData 数据卡确认实际开放小时数、任务数与许可文本，核对是否含派生力场、触觉原始图与清洗元数据，并确认全量 NeoData 申请流程。来源：[N0-Foundation 论文](https://arxiv.org/abs/2608.29601)；[OpenNeoData 数据集卡](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData/blob/main/README.md)。

### OpenNeoData

<a id="p-neodata-open"></a>

**页面日期/窗口状态**：未知（undated）。**机制与监督**：README 自述大规模实采操作数据，每个夹爪装两个凝胶视触觉相机、30fps 与 RGB 时间对齐；UMI 无关节编码器，state/action 直接为 raw tracked 末端轨迹（位置+rot6d+夹爪）；视频流为 HEVC、30fps。**边界**：全部数字与口径来自 README 自述，未从实际文件核验轨迹数、时长或模态完整性；README 未说明采集是遥操作/有人接管还是全自动，也未区分首发与修订；为 gated 表单访问，要求不得重分发、仅限非商业研究并引用。**接入检查**：获批后抽查 umi 与 ur 子集的 info.json 与 meta/episodes，核对轨迹数、时长、tactile 视频流是否齐全、时间戳对齐与 rot6d 约定，并确认仓库内是否有独立 LICENSE 文件。来源：[OpenNeoData 数据集卡](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData/blob/main/README.md)。

---

## 最小对照试验设计（本文建议，尚未运行）

目的：用最小成本验证“公开数据卡所述交付物”与“实际可下载并通过接入检查的内容”之间的一致性，并区分代码许可与数据许可。

- **输入**：每个项目仅取 1 个公开可访问子集或样例（例如 ABC-130K 的少量 episode、HIW-500 的公开格式之一、BONES-SEED 的门控批准样本、OpenNeoData 获批后的 1–2 个本体子集）；读取其数据卡/README、独立 LICENSE 文件与版本/修订记录。
- **对照**：组 A“仅读卡与许可文本”；组 B“实际下载文件名、目录结构、列 schema、时间戳与视频解码抽查”。对 gated 项目再对照“批准前可读信息”与“批准后实际字段”。
- **输出**：逐项记录：交付物类型（代码/数据卡/样例/全量）、访问路径（开放/gated/规划/未验证）、数据许可文本是否与卡片一致、实采 vs 派生字段标注、时间戳与流对齐抽查结果、发现的不一致清单。
- **停止/复核条件**：任一项出现“卡片称开放但实际不可下载”“代码许可与数据许可冲突或缺失”“派生信号被表述为原始实采”“门控条件与卡片许可文本不一致”时，暂停该项并标记为需复核；对论文所述全量 NeoData 规模与 OpenNeoData 数据卡自述规模口径不同的情况，并列记录并分别标注来源，不预设任一方数值替代另一方。

## 下一步值得跟踪的问题

1. AGIBOT WORLD 2026 各主题在 HF 仓库中的实际子集完整性与许可文件是否补齐，三主题标注 schema 是否一致。
2. BONES-SEED 的 v003 主表、v002 时间分段标签与 tar 包对应关系，以及 G1 重定向轨迹是否保留接触与物体信息。
3. RoboCOIN 逐数据集许可与 v3.0 支持状态，以及派生 action/EEF 在训练中的既定用法。
4. Daimon-Infinity 完整数据是否按规划在 ModelScope 上线，Lite 样例与实际批次格式差异，触觉占位维度比例。
5. OpenNeoData 获批后实际字段与 README 自述是否一致，是否存在独立 LICENSE 文件。
6. HIW-500 查看器列名不匹配的具体原因与两种发布格式的 schema 差异。

---

[← 数据采集研究目录](./robot_data_collection_research_catalog_2026.md)、[采集验收](./robot_data_collection_acceptance.md)。
