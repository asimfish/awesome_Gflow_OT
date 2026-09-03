# 第 10 章 仓库导览与方法论

## 10.1 目录结构

| 路径 | 内容 |
|---|---|
| `papers/` | 18 篇原文 PDF（arXiv 版），文件名为 arXiv 号 |
| `papers_zh/` | SuperTranslate 保版式中译 PDF（`<arXiv>.zh.pdf`）与对象级 QA 结果（`<arXiv>.inspect.json`） |
| `reports/` | 18 篇深度解读（`<ID>_<arXiv>.md`）、竞品矩阵、两份趋势扫描、INSIGHTS、本汇总报告及其 PDF |
| `data/meta/` | 每篇一张 JSON 元数据卡（标题、作者、venue、venue_type、代码、一句话中英） |
| `data/papers.yaml`、`data/candidates_*.csv`、`data/scan_*.json` | 种子清单、趋势候选、arXiv 原始扫描 |
| `src/generator.py` | 从 `data/meta` 与候选 CSV 生成 `README.md` / `README_zh.md`（awesome-ml4co 的数据驱动模式） |
| `scripts/` | 下载与 ID 解析、翻译批处理、趋势扫描、UTF-8 校验、报告拼装、PDF 构建 |
| `slides/` | 汇总幻灯（单文件 HTML 与 Beamer PDF） |

## 10.2 解读报告模板

每篇解读严格按 8 节：一句话定位与元数据表 → 问题设定与记号 → 核心贡献（按原文编号）→ 方法与推导要点 → 实验与证据 → 前提假设与适用边界 → 在主线中的位置 → 可复用 insight 与开放问题 → 引用；末尾附「编者注」记录自主决定的歧义处。所有数字与定理编号带原文出处，原文没给的写「原文未给出」。

## 10.3 翻译流水线与 QA

翻译引擎为 SuperTranslate（`pdf_zh_translator`）：不重排页面，公式、图表、引用先冻结，正文翻译后按原坐标回填；参数 `--preserve-graphics-text --skip-overflow`，DeepSeek 后端。每篇译后运行 `inspect` 做逐页对象级比对，issue 数记入附录 A。已知限制：`--skip-overflow` 会让放不下的译文保留英文原句，附录证明页的数学密集段因此出现 `untranslated_block`；字号缩放会触发 `font_size_drift`。18 篇全部完成翻译。O01（Peyré 讲义，480 页）体量最大，用 OpenRouter 上的 Gemini 2.5 Flash 后端单独翻译（`scripts/translate_o01_openrouter.sh`），耗时约 4 小时，QA 报 69 个 issue（主要为字号漂移与附录数学密集段的英文残留）。

## 10.4 发表状态纪律

主会 / 期刊 / Workshop / 预印本分开标注；arXiv comment 常滞后，凡能核到 OpenReview、dblp 或官方 proceedings 的以后者为准（竞品三篇即如此核出 GSBoG 为 ICML 2026 主会）；趋势扫描部分只依据 comment 字段并逐条标「未核实」。

## 10.5 如何贡献

在 `data/meta/` 增加一张 JSON 卡，可选地在 `reports/` 增加解读，运行 `python3 src/generator.py` 重新生成 README；翻译新论文用 `scripts/translate_batch.sh <arXiv>`；重建本报告用 `python3 scripts/build_report.py zh && bash scripts/build_pdf.sh zh`。

