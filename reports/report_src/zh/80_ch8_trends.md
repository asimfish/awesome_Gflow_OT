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

## 8.4 六篇 2026 增补的深度解读（N01–N06）

趋势扫描中 relevance=5 的六篇已补下载、翻译并配深度解读。下面收录各篇解读的核心贡献、前提假设、主线位置与 insight 四节（记号与推导细节见 `reports/N0*.md`）。

### 8.4.1 N01 · Minimum-Cost Network Flow with Dual Predictions（AAAI 2026）

@@INCLUDE reports/N01_2601.20203.md SECTIONS=2,5,6,7 DEMOTE=3

### 8.4.2 N02 · Stop the Sampler!（ICML 2026 SPIGM Workshop）

@@INCLUDE reports/N02_2606.16073.md SECTIONS=2,5,6,7 DEMOTE=3

### 8.4.3 N03 · Stable GFlowNets with TV Monitoring and Probabilistic Guarantees（预印本）

@@INCLUDE reports/N03_2605.01729.md SECTIONS=2,3,5,6,7 DEMOTE=3

### 8.4.4 N04 · Generative Modeling on Metric Graphs via Neural OT（预印本）

@@INCLUDE reports/N04_2606.16273.md SECTIONS=2,5,6,7 DEMOTE=3

### 8.4.5 N05 · Orlicz-Sobolev Transport for Unbalanced Measures on a Graph（NeurIPS 2025 Spotlight）

@@INCLUDE reports/N05_2502.00739.md SECTIONS=2,5,6,7 DEMOTE=3

### 8.4.6 N06 · A Benchmark for Discrete Schrödinger Bridges and EOT（ICLR 2026）

@@INCLUDE reports/N06_2509.23348.md SECTIONS=2,5,6,7 DEMOTE=3

### 8.4.7 六篇合看：对四个候选课题评级的影响

| 课题 | 六篇带来的变化 |
|---|---|
| ① 残差 → OT 误差界 + 对偶势证书 | 从「配件在别处」变成「配件在手」：N03 Thm. 3.5/3.6 是残差 → 边缘误差的现成证书，N01 Thm. 2 是对偶误差 → 运行时间的界，N06 Thm. 3.1 是构造解析基准的配方。评级维持**做**，且可开工的程度提高 |
| ② 条件 GFN 摊销图上 OT | 无新证据改变「不做」 |
| ③ 熵正则 GFN–OT / 图上 SB | N06 以 ICLR 2026 主会身份定下离散 SB 的评测口径，窗口关闭的证据更强；N02 说明连续非无环 GFN 的团队自己也在往采样器方向走，而不是 SB |
| ④ GFN proposal + 经典 OT 修正 | N01 把它具体化为「对偶预测 → ε-relaxation」，但只在显式图上成立；N04 说明显式连续图上另有神经 OT 路线。评级维持「降为①的应用」 |
