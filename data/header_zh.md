# Awesome GFlowNet x 最优传输

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

围绕 **生成流网络（GFlowNet）与最优传输（OT）交叉** 的精选文献清单：一个被训练去匹配奖励诱导目标分布的 GFlowNet，若在（允许有环的）状态图上最小化内部总边流，其最优边流恰好编码了该图最短路代价下的 **Kantorovich 最优传输计划**。这条主线把「终止分布相同时内部流不唯一」这个 GFlowNet 的老问题，变成了一个有明确答案的最优传输问题。

每篇论文配套：

- 中文**深度解读报告**（`reports/`），逐定理、逐表格标注原文出处；
- **原文 PDF**（`papers/`）与 [SuperTranslate](https://github.com/asimfish/super_translate) 生成的**保版式中译 PDF**（`papers_zh/`，附对象级 QA 结果）；
- 机器可读的元数据卡（`data/meta/`），本 README 由 `src/generator.py` 从中生成。

发表状态纪律：主会 / 期刊 / Workshop / 预印本分开标注，Workshop 绝不写成主会。

*维护者：[asimfish](https://github.com/asimfish)。结构参考 [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)。*
