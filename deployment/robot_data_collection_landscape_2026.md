# 机器人数据采集：近半年进展与选型地图 (Robot Data Collection Landscape)

> ⚙️ 本文由 Codex 辅助整理 | 2026-10-25
> **观察窗口**：2026-04-25 至 2026-10-25；窗口前的工作另列为基础。
> **核验范围**：官方发布页、论文摘要／版本记录、公开数据卡与代码入口；未下载大型数据、逐文件验收或复现训练。
> **核心定位**：把“采更多数据”拆成采集接口、监督信号、质量验收与下游收益四个问题。

近期变化不只是采集规模变大。无机器人手持采集、人类第一视角、多模态接触记录和自动试验，各自提供不同的监督，不能把它们的“小时数”直接相加比较。

**X-Ray**：一段视频可以说明人做了什么，却未必告诉机器人该发什么控制指令。真正有用的数据链路，需要把观测、动作、接触和结果对齐，并在目标机器人上验证收益。

## 1. 从哪里开始读

| 读者问题 | 入口 |
|---|---|
| 想扩展为更完整的资料库？ | [86 项研究目录与八个专题](./robot_data_collection_research_catalog_2026.md) |
| 最近有什么值得跟进？ | 本页 §3；[数据论文索引](../theory/foundation/paper_index.md#數據) |
| 我要自己采，或者验收供应商数据 | [采集验收与有效成本](./robot_data_collection_acceptance.md) |
| 坐标、动作、格式、归一化怎么接训练 | [VLA 数据工程指南](../theory/foundation/vla_data_engineering_guide.md) |
| 灵巧手与手部重定向 | [灵巧手数据采集](./dexterous_hand_data_collection.md) |
| 相机、触觉和控制怎么对时 | [多模态同步](./multimodal_data_synchronization.md) |
| 找现成数据 | [数据集速查的本次补充](../cheat-sheet/datasets.md#2026-10-25-补充公开规模与访问状态) |

既有[采集概览](./embodied_data_collection_overview.md)和[采集方案](./data_collection_solutions.md)适合认识路线。工程决策时，应重新核对其固定比例、价格和“质量最高”等概括的适用条件，不把旧经验当统一门槛。

## 2. 先分清你买到或采到的是什么

```text
任务与失效清单
      ↓
采集：机器人遥操 / 手持接口 / 人类视频 / 自动试验
      ↓
可追溯原始记录 → 同步与标定 → 动作/姿态/接触/结果标注
      ↓
质量筛选 + 去重 + 分组划分
      ↓
训练配方 → 独立真机评测 → 下一轮定向补采
```

| 路线 | 直接记录的信号 | 还缺什么 | 适合先验证的用途 |
|---|---|---|---|
| 机器人遥操作 | 机器人观测、指令和状态，具体依记录器而定 | 操作者偏差、时延、复位开销、失败覆盖 | 目标形态微调、精细控制基线 |
| UMI 类手持接口 | 接口位姿、夹爪状态和相机；扩展接口可含接触传感 | 可达性、动力学、观测与执行一致性 | 搬开机器人采集瓶颈，增加场景变化 |
| 人类第一视角 | 视频与部分手／身体姿态、语言标注 | 机器人动作、不可见接触、形态映射 | 表征、预训练、人到机器人迁移 |
| 自动真机试验 | 真实执行结果和状态，取决于闭环设计 | 复位、结果标签可靠性、候选分布 | 失败／成功样本、抓取与闭环改进 |
| 仿真／视频转机器人 | 合成观测、动作或接触监督 | 物理真实性、原始来源重叠与真实部署验证 | 补覆盖、控制变量实验、预训练 |

以上是本文的工程划分，不是各路线的固定优劣排名。真机记录同样可能错时或漏动作；无机器人采集也必须经过目标平台验证。不存在通用的“加固定比例真机数据就够了”。

## 3. 近半年优先补看的工作

日期按首次发布；修订单独注明。“论文报告”不等于本手册已复现。

| 日期 | 工作／发布 | 对采集的新增价值 | 本次核验边界与局限 |
|---|---|---|---|
| 04-26 | [EgoLive](https://arxiv.org/abs/2604.23570v1) | 面向实际人类工作场景的第一视角采集与多模态标注 | 已核摘要和首发日期；未据此认定含目标机器人 action，未核下载清单 |
| 06-22 | [AutoDex](https://arxiv.org/abs/2606.23689v1) | 把候选抓取、执行、结果标记和复位连成自动真机采集闭环 | 论文摘要报告实际抓取试验；其中代码／数据为“将公开”，本页不标成已验收开放数据 |
| 06-25 | [ABC / ABC-130K](https://arxiv.org/abs/2606.27375v1) | 真机双臂示教与训练、仿真及评测配套发布 | [官方数据卡](https://huggingface.co/datasets/XDOF/ABC-130k)显示访问条件；论文约数与当前卡片统计应分别记录 |
| 07-07 修订 | [EgoVerse v2](https://arxiv.org/abs/2604.07607v2) | 协作采集、共同处理框架和跨团队迁移研究 | **首发 04-08，在窗口外**；本行是 v2 更新，不冒充七月新论文。另见[官方代码](https://github.com/GaTech-RL2/EgoVerse) |
| 07-21 | [Grabette](https://huggingface.co/blog/grabette) | 开放手持采集、共享时钟及 LeRobot 导出链路 | 官方示例是部署参考；跨机器人执行仍需匹配接口及验证，不等于即插即用 |
| 07-28 | [HiFi-UMI](https://arxiv.org/abs/2607.25895v1) | 将位姿精度、双手相对位姿、同步和视场共同设计 | 作者报告在其测试设置中，后训练阶段无需真机遥操示范；结论限于该阶段的示范来源，不是通用零真机方案。[数据卡](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)区分公开 2,000 小时子集与 20,000+ 小时源总库 |
| 08-03 | [Ego2Robot](https://arxiv.org/abs/2608.02580v1) | 人类视频经动作重定向、机器人外观合成和质量筛选形成训练数据 | 论文报告的 18,561 小时是转化／合成数据口径，不能算作同量真实机器人采集 |
| 08-26 | [EgoSuite-Open100K](https://huggingface.co/blog/LightwheelAI/egosuite-open100k) | 分布式人类采集与分层姿态／语义标注 | 公告称首批 10,000 小时上线、总量 100,000 小时分批发布；当前 [EgoStandard](https://huggingface.co/datasets/LightwheelAI/EgoStandard)／[EgoPro](https://huggingface.co/datasets/LightwheelAI/EgoPro)仍列 planned 时长，全量可下载未核。[EgoDemo 卡片](https://huggingface.co/datasets/LightwheelAI/EgoDemo)标 commercial-training-no-resale-v1.0，需访问登记 |
| 09-19 | [Whole-Body UMI](https://arxiv.org/abs/2609.22829v2) | 把任务示范与全身协调学习解耦；v2 为 09-22 | 运动生成器仍需要独立训练；不表示手持末端轨迹本身就记录了完整全身动作 |
| 09-24 | [PolyUMI / VisTA](https://arxiv.org/abs/2609.29760v2) | 手持同步记录视觉、光学触觉、接触音频与本体信息；v2 为 10-20 | 记录器与执行端共享感知几何是亮点；[官方硬件／代码](https://github.com/PolyUMI-VisTA/PolyUMI-system)不代替任务效果验证 |
| 10-06 | [Pantheon 采集运营案例](https://pantheon.inc/research/diverse-umi) | 自由操作采集、后续标注及运营吞吐的案例 | **厂商自报**成本约 10 美元／数据小时，并宣布 100 小时样本；未独立核验全成本、样本文件或泛化收益 |

### 3.1 触觉＋VLA 的优先级

建议优先读 **PolyUMI → 既有 OmniUMI／TAMEn → 同步与验收指南**。PolyUMI 的多传感流设计值得作为接触操作选题的阅读起点；这是一条选题建议，不是已证明的普遍性能排序。

- [OmniUMI](../theory/tactile/omniumi_towards_physically_grounded_robot_learning_via_human_dissection.md)：首发 04-12，属于窗口前基础，重点看握持力与外部交互 wrench 的区分。
- [TAMEn](../theory/tactile/tamen_tactile_aware_manipulation_engine_for_closed_loop_data_dissection.md)：连接接触信号与采集闭环；沿用既有解读，不重复造稿。
- [MobileEgo Anywhere](../theory/deployment/mobileego_anywhere_open_infrastructure_for_long_horizon_egoc_dissection.md)：已收录的长时序便携采集路线，和新的人类视频项目对照阅读。

## 4. 按研究问题选路线

| 你的瓶颈 | 优先做的最小试验 | 主要看什么 |
|---|---|---|
| 不同场景就失败 | 固定任务，改变场景／物体，补采留出环境 | 环境覆盖和真实留出收益，而非总帧数 |
| 双手相对运动不准 | 先检验双手位姿、共同坐标及相机同步 | 相对位姿误差、时钟漂移、重建丢失 |
| 接触开始后失败 | 在相同视觉条件比较有／无触觉或力信号 | 接触可观测性、标定与执行端传感一致性 |
| 操作员和复位成本太高 | 分开记录示范、等待、复位、返工时间 | 每合格独立 episode 的全成本 |
| 人类视频很多但迁移差 | 固定真机配方，只改变筛选／映射策略 | 可执行性、动作定义、数据与目标任务的对齐 |
| 人形移动操作难采 | 分开评估手部技能与全身协调模块 | 基座／手／头的状态、延迟与控制边界 |

## 5. 后续采集与选题的来源表

这是可核验的来源和检索口径，**不表示已为这些站点新增自动订阅或排程**。

| 主题 | 一手入口／检索词 | 值得更新手册的触发条件 |
|---|---|---|
| 便携采集 | Grabette、HiFi-UMI；`robot-free data collection`, `bimanual relative pose` | 新硬件／标定协议公开，有明确版本与执行验证 |
| 第一视角 | EgoLive、EgoVerse、EgoSuite；`egocentric human data`, `human-to-robot` | 新公开子集、标注字段或迁移实验；区分首发与修订 |
| 触觉／音频／力 | PolyUMI、OmniUMI、TAMEn；`visuo-tactile`, `contact audio`, `wrench` | 明确传感配置、同步和数据可访问性 |
| 自动采集 | AutoDex；`autonomous data collection`, `reset`, `intervention` | 复位／结果判定闭环与失败日志可检查 |
| 数据运营 | ABC/XDOF、Pantheon、Lightwheel 官方发布 | 原始量、去重量、标注量、访问条件或成本口径有实质变化 |
| 格式与工具 | [LeRobotDataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) 与各发布者的数据卡 | 格式版本／loader／episode 边界变化；先验证再迁移 |

## 6. 读规模和宣传时的核对顺序

1. **单位**：原始视频小时、合格示范小时、episodes、tasks、对象和场景不是同一个数。
2. **状态**：规划、内部总库、公开子集和本地验收完成的子集分别记录。
3. **访问**：数据卡可读、代码可读、数据有条件访问和已成功下载分别记录。
4. **模态**：手部姿态不是机器人控制指令；触觉图像不是已标定力值；音频不是天然接触标签。
5. **收益**：先看目标机器人留出任务的结果，再讨论每单位成本的边际收益。

下一步的工程落地见[采集验收与有效成本](./robot_data_collection_acceptance.md)。

---

[← 返回部署目录](./README.md) · [数据集速查](../cheat-sheet/datasets.md) · [论文索引](../theory/foundation/paper_index.md)
