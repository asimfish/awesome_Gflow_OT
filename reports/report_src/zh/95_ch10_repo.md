# 第 10 章 仓库导览与方法论

## 10.1 目录结构

| 路径 | 内容 |
|---|---|
| `papers/` | 24 篇原文 PDF（18 核心 + 6 篇 2026 增补）（arXiv 版），文件名为 arXiv 号 |
| `papers_zh/` | SuperTranslate 保版式中译 PDF（`<arXiv>.zh.pdf`）与对象级 QA 结果（`<arXiv>.inspect.json`） |
| `reports/` | 24 篇深度解读（`<ID>_<arXiv>.md`）、竞品矩阵、两份趋势扫描、INSIGHTS、本汇总报告及其 PDF |
| `data/meta/` | 每篇一张 JSON 元数据卡（标题、作者、venue、venue_type、代码、一句话中英） |
| `data/papers.yaml`、`data/candidates_*.csv`、`data/scan_*.json` | 种子清单、趋势候选、arXiv 原始扫描 |
| `src/generator.py` | 从 `data/meta` 与候选 CSV 生成 `README.md` / `README_zh.md`（awesome-ml4co 的数据驱动模式） |
| `scripts/` | 下载与 ID 解析、翻译批处理、趋势扫描、UTF-8 校验、报告拼装、PDF 构建 |
| `slides/` | 汇总幻灯（单文件 HTML 与 Beamer PDF） |

## 10.2 解读报告模板

每篇解读严格按 8 节：一句话定位与元数据表 → 问题设定与记号 → 核心贡献（按原文编号）→ 方法与推导要点 → 实验与证据 → 前提假设与适用边界 → 在主线中的位置 → 可复用 insight 与开放问题 → 引用；末尾附「编者注」记录自主决定的歧义处。所有数字与定理编号带原文出处，原文没给的写「原文未给出」。

## 10.3 翻译流水线与 QA

翻译引擎为 SuperTranslate（`pdf_zh_translator`）：不重排页面，公式、图表、引用先冻结，正文翻译后按原坐标回填；参数 `--preserve-graphics-text --skip-overflow`，DeepSeek 后端。每篇译后运行 `inspect` 做逐页对象级比对，issue 数记入附录 A。已知限制：`--skip-overflow` 会让放不下的译文保留英文原句，附录证明页的数学密集段因此出现 `untranslated_block`；字号缩放会触发 `font_size_drift`。24 篇全部完成翻译。O01（Peyré 讲义，480 页）体量最大，用 OpenRouter 上的 Gemini 2.5 Flash 后端单独翻译（`scripts/translate_o01_openrouter.sh`），耗时约 4 小时，QA 报 69 个 issue（主要为字号漂移与附录数学密集段的英文残留）。

## 10.4 发表状态纪律

主会 / 期刊 / Workshop / 预印本分开标注；arXiv comment 常滞后，凡能核到 OpenReview、dblp 或官方 proceedings 的以后者为准（竞品三篇即如此核出 GSBoG 为 ICML 2026 主会）；趋势扫描部分只依据 comment 字段并逐条标「未核实」。

## 10.5 如何贡献

在 `data/meta/` 增加一张 JSON 卡，可选地在 `reports/` 增加解读，运行 `python3 src/generator.py` 重新生成 README；翻译新论文用 `scripts/translate_batch.sh <arXiv>`；重建本报告用 `python3 scripts/build_report.py zh && bash scripts/build_pdf.sh zh`。

## 10.6 外部审稿与修订记录

本报告的综合部分（INSIGHTS、摘要与第 1 章、六篇 2026 增补解读）经本地 Codex CLI（模型 `gpt-6-astra`，推理档位 max，只读沙箱）以「ICML 领域主席级读者」提示词逐份审稿（脚本 `scripts/codex_review.sh`，结构化 JSON 输出存于 `reviews/codex/`）。七份审稿（另含核心三篇 O08/O07/T36 的解读）共提出 135 条事实风险、58 条逻辑缺口。已按审稿修正的实质性问题包括：O08 Eq. (20) 的训练目标是带源分布项的正则化 TB（原误写为 DB + 流正则）；T36 恒等式的编号是 Prop. 3.12 / Eq. (10)（原误写 Prop. 3.6）；N03 Thm. 3.6 中 \(c\) 是最大损失的平方根；N02 Eq. (19) 第二项用目标分布 \(\pi\) 而非核的平稳分布；O08 Thm. 3.3 只针对去掉首步约束的单源问题，固定双边缘须用 Appendix A.4；N01 原文确与 network simplex 比过且组均值快 12 倍；N02/N05/N06 的代码链接原文均有给出；INSIGHTS §6b 命题 D「机械组合即成立」的判断撤回（零残差不蕴含 OT 最优、端点代价不等于路径代价两个反例）；O07 的 Workshop 状态降为待核实。第二轮（核心三篇）又修正：O08 解读 §7.2 的「残差 → OT gap」草稿界漏掉耦合次优项（交叉配对反例，已在 `experiments/` 数值证实）、「有环是必需的」与「源集终集隔离」两条前提写错、Theorem 3.3 的 BFS 势被误用于固定 \(L\) 的问题；O07 解读的 reward matching 漏 \(Z\)、\(\lambda\) 倍数与 Figure 3 读法；T36 解读把状态流平方和误等同于 O02 的边流二次正则。凡审稿指出而本报告无法独立核验的条目，均在正文改为限定表述或标注「待核实」。

## 10.7 可复现实验

`experiments/day1_reference_lp.py` 在 hypergrid（\(H\in\{10,15,20\}\)、ball/moon 源、corner 目标）上用精确 LP 验证：GFlow* = OT*（差 \(\le4\times10^{-14}\)）、对偶势逐边可行、互补松弛为零；并数值证实 INSIGHTS §6b 的两个反例——零残差错配流代价高 60–78%，最优耦合加绕路后执行代价为 OT* 的 3.2–3.5 倍而残差仍为零。结果与说明见 `experiments/README.md`。
