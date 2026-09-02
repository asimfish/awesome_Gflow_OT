# 第 8 章 2025–2026 趋势

两侧各做一次 arXiv 系统扫描（GFlowNet 侧 106 篇、OT 侧 93 篇，2025-01 → 2026-09），检索口径与逐篇卡片见 `reports/TRENDS_GFN_2026.md`、`reports/TRENDS_OT_2026.md`，候选收录清单见 `data/candidates_gfn.csv`（38 条）与 `data/candidates_ot.csv`（49 条）。本章只收趋势判断与对主线的含义。发表状态只依据 arXiv `comment` 字段；为空者一律视为预印本。

## 8.1 GFlowNet 侧

@@INCLUDE reports/TRENDS_GFN_2026.md SECTIONS=3,4 DEMOTE=2 SKIP_META

## 8.2 最优传输侧

@@INCLUDE reports/TRENDS_OT_2026.md SECTIONS=3,4,5 DEMOTE=2 SKIP_META

## 8.3 本报告判断：两侧合看

- GFN 侧把「残差当度量」（Stable GFlowNets、Evaluation Balance）与 OT 侧把「误差理论放在对偶势上」（对偶预测加速最小费用流、Sinkhorn 势统计率、QOT 的 PL 不等式）在 2026 年同时成熟，却没有任何一篇把两端接起来。这是课题①的窗口。
- 非无环 GFlowNet 的三个应用出口（最短路、OT、MCMC 终止）全部停在 Workshop；同象限的 GSBoG 在主会。叙事占位的差距比技术差距大。
- 摊销 OT 与离散 SB 两个方向都已进入「有基准、有收敛率」阶段，新进入者没有定义问题的红利。

