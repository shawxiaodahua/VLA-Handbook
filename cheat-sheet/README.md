# 速查表 (Cheat Sheet)

涵盖 VLA 领域核心概念、公式、模型对比、常见陷阱与工程误区的快速参考。

> 用于快速定位概念与专题；版本、指标和来源以链接文章为准。需要完整推导时回到[理论目录](../theory/README.md)。

---

## 📚 基础篇

1. **[关键论文时间线](./timeline.md)** — 从 CLIP 到 π0.7、EgoScale、GR00T-N1.7 · 34 篇关键工作
2. **[核心公式速查](./formulas.md)** — Attention · Diffusion · Flow Matching · LoRA · 6D 旋转 · RL 基础 · 评估指标
3. **[模型对比总表](./model_comparison.md)** — 16+ 主流 VLA 模型架构 × 能力 × 真机成绩

## 🔍 专项篇

4. **[Benchmark 地图 + 可信度警告](./benchmarks.md)** — **30+ benchmark · 9 大类 + 智能/推理专节** + BEHAVIOR-1K 深度拆解
5. **[数据集速查](./datasets.md)** — 25+ 数据集 · 许可证继承规则 · 商用合规组合推荐
6. **[数据失效模式 F1-F6](./failure-modes.md)** — 虚假相关 / OOD / 时序错位 / 坐标漂移 / 语言解耦 / 动作偏斜
7. **[RL 后训练三流派](./rl-post-training.md)** — 传统 RL / ACP / RL Token · World Model 辅助

## 使用建议

- **建立领域全景**：扫一遍 [timeline](./timeline.md)，按"epoch → 代表模型 → 唯一重要贡献"组织记忆
- **复习核心算法**：翻 [formulas](./formulas.md)——Attention / LoRA / Diffusion vs Flow Matching / 6D 旋转 / RL Advantage
- **需要比较多个模型**：看 [model_comparison](./model_comparison.md) 架构 × 能力 × 真机成绩矩阵
- **报 benchmark 分数前**：先过 [benchmarks 警告](./benchmarks.md)——PRO/Plus/X 扰动结果 + 真机验证**一起报**
- **选数据集**：看 [datasets](./datasets.md)——**先看许可证再看规模**，商用合规数据集组合
- **Debug 真机出问题**：用 [failure-modes F1-F6](./failure-modes.md) 分类排查
- **规划后训练路线**：看 [rl-post-training](./rl-post-training.md) 三流派选型

## 🚨 常见工程/学术误区

1. **脱离设置给模型排总榜**：先对齐任务、数据、硬件与评测条件，再比较结果。
2. **只报单一 benchmark**：同时核对扰动评测、失败情况与真机证据，见[评测速查](./benchmarks.md)。
3. **混用方法名称**：比较 Diffusion 与 Flow Matching 的训练目标、采样过程和部署条件，见[动作表示](../theory/diffusion-flow/action_representations.md)。
4. **只看数据量**：结合采集质量、覆盖范围与迁移设置，见[数据集](./datasets.md)和[失效模式](./failure-modes.md)。
5. **把代码许可当作全部许可**：分别查看原项目对代码、权重和数据的说明。

---

[← Back to VLA Handbook](../README.md)
