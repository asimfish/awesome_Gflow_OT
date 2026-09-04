<a id="insights"></a>
### Insight 与开放问题

跨论文综合见 [reports/INSIGHTS.md](reports/INSIGHTS.md)，竞品矩阵见 [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md)。核心结论：

1. **内部流非唯一性是入口。** 匹配终止分布只固定了边界，状态图的环空间留下一大族合法内部流；最小化总流从中选出一个，而这一个就是最优传输计划。
2. **等价本身是经典结果，贡献在翻译。** 图上最短路代价的 Kantorovich OT 等于最小费用流。GFlowNet 增加的是：在无法显式构造 cost 矩阵的组合图上，输出一个可执行的局部路由策略。
3. **最不拥挤的后续课题是误差界。** Balance 残差已能给终止分布的全变差误差作证书；把它推到 OT cost gap 与边缘违反量（以对偶势作证书），目前没有直接竞品。
4. **两个后续课题已经拥挤。** 条件 GFlowNet 摊销一族图上的 OT 与 ULOT（NeurIPS 2025）、UNOT（ICML 2025）撞车；熵正则 GFN–OT 与离散/图上 Schrödinger 桥一系（DDSBM，ICLR 2025；GSBoG，ICML 2026 主会，与 O08 同象限但无误差界）撞车。
5. **发表状态的现实。** 两篇 GFN × OT 论文（O07、O08）都是 ICML 2026 SPIGM *Workshop*，三篇竞品全是主会。GFlowNet 剩下的结构性护城河只有一处：无法构造代价矩阵的隐式组合图。

<a id="deliverables"></a>
### 汇总报告与幻灯

| 交付物 | 路径 |
|---|---|
| 汇总报告 · 中文（98 页） | [PDF](reports/pdf/awesome_gflow_ot_report_zh.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_zh.md) |
| 汇总报告 · 英文（129 页） | [PDF](reports/pdf/awesome_gflow_ot_report_en.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_en.md) |
| 汇总幻灯（单文件 HTML，24 页；键盘 / 滚轮 / 触摸） | [slides/awesome_gflow_ot_slides.html](slides/awesome_gflow_ot_slides.html) |
| 汇总幻灯（Beamer PDF，29 页含备份页） | [slides/awesome_gflow_ot_slides.pdf](slides/awesome_gflow_ot_slides.pdf) |
| 跨论文综合 | [reports/INSIGHTS.md](reports/INSIGHTS.md) |
| 竞品矩阵 | [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md) |
| 2026 趋势扫描 | `reports/TRENDS_GFN_2026.md`、`reports/TRENDS_OT_2026.md` |

<a id="contributing"></a>
### 贡献与引用

新增论文：在 `data/meta/` 放一张 JSON 卡（字段见任一现有卡），可选地在 `reports/` 放解读，然后运行 `python3 src/generator.py`。请保持发表状态纪律（主会 / 期刊 / Workshop / 预印本），概述时标注定理或表格编号。

```bibtex
@misc{awesome_gflow_ot_2026,
  title  = {Awesome GFlowNet x Optimal Transport: a curated, evidence-first reading list},
  author = {asimfish},
  year   = {2026},
  url    = {https://github.com/asimfish/awesome_Gflow_OT}
}
```

翻译引擎：[SuperTranslate](https://github.com/asimfish/super_translate)。幻灯工具：[beamer-skill](https://github.com/Noi1r/beamer-skill)。写作纪律：[anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing)、[shuorenhua](https://github.com/MrGeDiao/shuorenhua)。清单规范：[awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)。

代码与文本以 [MIT](LICENSE) 许可；论文 PDF 保留其原始许可（arXiv）。
