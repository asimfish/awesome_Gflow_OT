# 2025–2026 趋势扫描 · 最优传输侧

> 视角固定：图上 OT、最小费用流学习、离散/图上 Schrödinger 桥、神经·摊销 OT、不平衡 OT、OT 误差界——每一篇都回答同一个问题：对「GFlowNet 学最优传输」是竞争、互补，还是可借用的技术。发表状态只依据 arXiv `comment` / `journal_ref`；为空的写「arXiv 预印本（未核实）」。数据截止 2026-09-02。

## 1. 检索口径与覆盖

| 项目 | 内容 |
|---|---|
| 数据源 | arXiv API，查询串见 `scripts/scan_trends.py`，原始命中存 `data/scan_ot.json` |
| 时间窗 | `submittedDate` 2025-01-01 → 2026-09-30（最新命中 2026-08-30） |
| 查询 | 10 条：OT × graph × neural/learning；Schrödinger bridge × discrete/graph；minimum cost flow × learning；unbalanced OT × graph/neural；amortized OT；OT × RL × graph/discrete；entropic OT × discrete；Beckmann；OT × error bound × dual/certificate |
| 去重后 | **93 篇**；其中 92 篇不在旧调研里（旧调研的 OT 板块只有 8 篇，本次是第一次系统扫 OT 侧） |
| 按月分布 | 2025 全年 42 篇，2026-01 → 08 共 51 篇；2026-02 与 2026-05 各 10 篇为峰 |
| 有 venue 线索 | 38 篇 comment 非空 |

比 GFN 侧更需要说明的是**噪声**：OT 是通用工具，93 篇里约一半是把 OT 当损失函数用在 GNN、推荐、EEG、人脸识别上的应用工作，与主线无关；下面只解读与「图上传输 / 离散 SB / 摊销 / 误差界」直接相关的约 25 篇，其余在 `data/candidates_ot.csv` 里以 relevance 3 记录或干脆不收。

## 2. 主题分组卡片

### 2.1 图上 OT 与最小费用流

**Minimum-Cost Network Flow with Dual Predictions** · Chen, Yao, Yin · [2601.20203](https://arxiv.org/abs/2601.20203) · AAAI 2026（comment）
第一个用「学习到的对偶预测」增强的最小费用流算法，基于经典 \(\varepsilon\)-relaxation；给出以预测误差无穷范数刻画的时间复杂度界（consistent 且 robust），并证明 PAC 学习该预测的样本复杂度界；交通网络与芯片逃逸布线上分别加速 12.74× 与 1.64×（摘要数字）。
关系：**可借用 + 潜在竞争**。它证明「预测对偶势 → warm-start 经典求解器」在最小费用流上成立且有理论界。对 O08 的启示有两层：(i) O08 Thm 3.3 的对偶势 \(\pi\) 正是这里的「dual prediction」，GFlowNet 学出的状态流可以直接喂给 \(\varepsilon\)-relaxation 做带保证的精化——这是「GFN proposal + 经典修正」课题唯一还站得住的形态；(ii) 在显式图上，它比 GFN 更快更有界，GFN 只剩隐式图。

**Generative Modeling on Metric Graphs via Neural Optimal Transport** · Micheli, Cao, Monod, Bhatt · [2606.16273](https://arxiv.org/abs/2606.16273) · arXiv 预印本（未核实）
面向连续支撑在紧度量图上的分布：把图嵌入光滑环境空间，用神经半对偶参数化解熵 Kantorovich 问题，再把样本投影回图；证明神经表达力增大时生成器弱收敛到合法耦合；实验称匹配或优于基于离散图 OT 的启发式基线且扩展性更好。
关系：**相邻竞争**。同样是「图上、两端边缘、输出可采样的传输」，但对象是连续度量图上的分布，用的是熵正则 + 半对偶，不给逐边策略。它和 O08 在「图」这个词上重合，在对象与输出上不重合。

**An Efficient Orlicz-Sobolev Approach for Transporting Unbalanced Measures on a Graph** · Le, Nguyen, Hino, Fukumizu · [2502.00739](https://arxiv.org/abs/2502.00739) · NeurIPS 2025 Spotlight（comment）
把 Orlicz–Wasserstein / 广义 Sobolev 传输推到图上不等总质量的测度，避开两层优化。
关系：**互补**。给「不平衡的图上 OT」提供了一个非 KL 罚的几何；O08 的 Assumption 3.1 要求 \(\sum L=\sum R\)，若要放开，这是候选的目标函数之一。

**Regularity of Solutions to Beckmann's Parametric Optimal Transport** · Gottschalk, Riedlinger · [2603.19755](https://arxiv.org/abs/2603.19755) · arXiv 预印本（未核实）
连续 Beckmann 问题（最小化总平方通量）的正则性理论：拉格朗日乘子满足 Poisson 方程、通量是势的梯度，并给出目标分布依赖参数时（可提示的条件生成）的 Hölder 连续性条件。
关系：**可借用**。O08 是离散 Beckmann（O02 报告第 6 节），这篇是连续版的正则性；若做「条件 GFN 学一族 OT」，「解对参数 Hölder 连续」正是泛化保证需要的那类命题。

**Statistical Mechanics of the Sub-Optimal Transport** · Piombo, Buffa, Mazzilli, Patelli · [2602.04308](https://arxiv.org/abs/2602.04308) · arXiv 预印本（未核实）
用平均场理论刻画「熵主导的稠密配置 ↔ 代价主导的稀疏结构」之间的过渡，对象是加权二分图集合上的次优传输。
关系：**可借用的直觉**。O08 的 LP 解是零温顶点解，GSBoG/DDSBM 在有限温度；这篇给出两者之间过渡的解析框架。

其余题录：GRAND 多智能体路径调度（[2512.03194](https://arxiv.org/abs/2512.03194)，预印本）；OTAP 结构感知 OT 评估 agent 规划（[2607.17082](https://arxiv.org/abs/2607.17082)，预印本）；OT for Network Comparison 综述（[2608.27500](https://arxiv.org/abs/2608.27500)，预印本）。

### 2.2 离散 / 图上 Schrödinger 桥与熵正则

**Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport** · Carrasco, Ksenofontov, Leonov, Koshelev et al. · [2509.23348](https://arxiv.org/abs/2509.23348) · arXiv 预印本（未核实）
构造离散空间上有解析 SB 解的分布对，作为基准；附带得到两个新算法 DLightSB / DLightSB-M，并扩展出 \(\alpha\)-CSBM。
关系：**必用基线**。任何「熵正则 GFN–OT」或「图上 SB」的实验都应在这个有解析解的基准上报数，否则无法说明自己解的是 SB 而不是别的东西。

**Adjoint Schrödinger Bridge Sampler** · Liu, Choi, Chen, Miller et al. · [2506.22565](https://arxiv.org/abs/2506.22565) · NeurIPS 2025 Oral（comment）
从 Boltzmann 分布采样的扩散采样器，用 SB 的动能最优传输提升效率，以随机最优控制视角给出可扩展的 matching 目标，训练中无需估计目标样本。
关系：**相邻竞争**。它与 GFlowNet 争的是同一件事——从未归一化能量采样；差别是连续 SDE 与 SB 语言。GFN 侧的对应物是 Berner 等的离散↔连续等价（GFN 侧报告 2.3）。

**Uniform Statistical Convergence of Empirical Sinkhorn Potentials with Exponential and Polynomial Dependence on the Regularization Parameter** · Belomestny · [2608.29152](https://arxiv.org/abs/2608.29152) · arXiv 预印本（未核实）
经验 Sinkhorn 势在商上确界范数下的 \(n^{-1/2}\) 统计率；常数一般随 \(1/\varepsilon\) 指数增长，在多项式残差稳定性条件下降为多项式。
关系：**可借用，且作者重合**。Belomestny 是 O08 的合作者。这篇给的是「势的统计误差界」，正是「对偶势作证书」课题需要的统计一侧；把它与 O08 Thm 3.3 的确定性对偶合起来，就是一个完整的误差分解。

**Exponential Convergence of the Sinkhorn Algorithm for the Schrödinger Bridge with Regime Switching** · Eichinger, Kazeykina, Ren, Wang · [2607.19176](https://arxiv.org/abs/2607.19176) · arXiv 预印本（未核实）
混合状态空间 \(\mathbb R^d\times\{1,\dots,m\}\) 上 regime-switching SB 的 Sinkhorn 迭代在相对熵意义下指数收敛。
关系：**可借用**。离散分量 \(\{1,\dots,m\}\) 上的 Sinkhorn 收敛证明技术，是图上熵正则流的 IPF 收敛证明可以模仿的模板。

**Polyak-Łojasiewicz Inequality for Quadratically Regularized Optimal Transport** · González-Sanz, Nutz, Riveros Valdevenito · [2605.27175](https://arxiv.org/abs/2605.27175) · SIAM J. Optim.（comment："To appear"）；同组 **Stability of Quadratically Regularized OT** · [2605.27883](https://arxiv.org/abs/2605.27883) · 预印本
QOT 对偶目标凹但含正部函数、不强凹；仍证明局部误差界与 PL 不等式，常数只依赖问题原语，由此得到算法线性收敛。
关系：**可借用**。O02（Essid & Solomon）用的正是二次正则；这两篇给了二次正则图 OT 缺失的优化理论。若想给 O08 加正则又不想撞 SB 一系，二次正则是唯一没被占满的方向。

**Entropic selection for OT on the line with distance cost** · Ley · [2512.05282](https://arxiv.org/abs/2512.05282) · 预印本；**Entropic Selection Principle for Monge's OT** · Aryan, Ghosal · [2502.16370](https://arxiv.org/abs/2502.16370) · 预印本
两篇都研究距离代价下 OT 计划不唯一时，小正则极限选出哪一个（弱可乘性、传输射线上的相对熵极小）。
关系：**概念上最接近 O08 的问题**。距离代价 + 计划不唯一 + 用一个外生原则选出一个——这与「图最短路代价 + 内部流不唯一 + 最小总流选出一个」是同一类问题的连续版本。GFN 侧从未引用这条「选择原理」文献。

其余题录：cuRegOT GPU 熵正则求解器（[2605.08793](https://arxiv.org/abs/2605.08793)，预印本）；SinkSLOT 稀疏提升 Sinkhorn（[2608.28262](https://arxiv.org/abs/2608.28262)，预印本）；Sinkhorn-Drifting Generative Models（[2603.12366](https://arxiv.org/abs/2603.12366)，预印本）；SB for Gaussian Mixtures in discrete time（[2604.01144](https://arxiv.org/abs/2604.01144)，预印本）；Mean-Field SB 操控大规模 agent 群（[2503.23705](https://arxiv.org/abs/2503.23705)，L-CSS）；Markov 链聚合数据估计的逆 OT 凸方法（[2511.16458](https://arxiv.org/abs/2511.16458)，预印本，投 ECC 2026）。

### 2.3 神经 · 摊销 · 条件 OT

**Amortized Optimal Transport from Sliced Potentials** · Truong, Nguyen · [2604.15114](https://arxiv.org/abs/2604.15114) · arXiv 预印本（未核实）
用切片 OT 得到的 Kantorovich 势作预测子，回归或直接优化对偶目标来预测多对测度间的 OT 计划。
关系：**竞争**（摊销轴）。与 UNOT（O05）、ULOT（C01）同一象限；再次说明「条件 GFN 学一族 OT」的窗口在关闭。

**Efficient Transferable OT via Min-Sliced Transport Plans** · Liu et al. · [2511.19741](https://arxiv.org/abs/2511.19741) · 预印本 · 题录，同一象限。

**Riemannian Neural OT** · Micheli et al. · [2602.03566](https://arxiv.org/abs/2602.03566) · 预印本；**Entropic Riemannian Neural OT** · [2605.04255](https://arxiv.org/abs/2605.04255) · 预印本 · 题录：同一组把神经 OT 推到流形，是 2.1 中度量图工作的前身。

**HyperTransport: Amortized Conditioning of T2I Generative Models** · [2605.08254](https://arxiv.org/abs/2605.08254) · 预印本 · 题录。

### 2.4 不平衡 OT

除 2.1 的 Orlicz–Sobolev 图上不平衡传输（2502.00739，NeurIPS 2025 Spotlight）外，命中的多是应用：Generative Molecular Morphing via Unbalanced OT（[2606.07239](https://arxiv.org/abs/2606.07239)，预印本）、OptiMAG 结构–语义对齐（[2601.22856](https://arxiv.org/abs/2601.22856)，预印本）、GCL-OT 异配文本属性图对比学习（[2511.16778](https://arxiv.org/abs/2511.16778)，AAAI 2026）、Gaussian KL-UOT 闭式解（[2605.02497](https://arxiv.org/abs/2605.02497)，预印本）。
关系：O08 报告 §7.6 已指出「unbalanced 与 unknown-\(Z\)」是 GFlowNet 本可发挥却被 Assumption 3.1 关掉的方向；这里的应用工作说明需求真实存在，但理论侧只有 Orlicz–Sobolev 一篇在图上做，空位仍在。

### 2.5 OT 误差界与对偶证书

直接命中的只有两篇（**Statistical Estimation of Monge Transport Maps via Brenier Potentials** · Cazelles et al. · [2604.22366](https://arxiv.org/abs/2604.22366) · 预印本；以及 2.2 的 Sinkhorn 势统计收敛 2608.29152），加上 2.1 的最小费用流对偶预测（2601.20203，AAAI 2026）与 2.2 的 QOT 误差界（2605.27175）。四篇合起来的图景：**OT 侧的「误差界」全是关于势（对偶变量）的**——统计率、PL 不等式、预测误差到运行时间的映射。没有一篇把「一个近似可行流的 balance 残差」映射到「cost gap」。这正是 GFN 侧能补的位置。

### 2.6 OT 用于离散 / 组合生成与 RL

**Reinforcement Learning via Value Gradient Flow** · Xu et al. · [2604.14265](https://arxiv.org/abs/2604.14265) · ICLR 2026（comment）· 题录：把策略优化写成概率空间上的梯度流，与 GFN–RL 等价一线相邻。

**Global Convergence of Wasserstein Policy Gradient for Entropy-Regularized RL** · Zhu et al. · [2605.26078](https://arxiv.org/abs/2605.26078) · 预印本 · 题录：熵正则 RL 的 Wasserstein 策略梯度全局收敛；GFN 与熵正则 RL 等价意味着这类结果可平移。

**Transport Novelty Distance** · Hagemann et al. · [2512.09514](https://arxiv.org/abs/2512.09514) · 预印本 · 题录：用传输距离评估材料生成模型的新颖性——OT 作评估指标，可作 GFN 多样性评估的候选口径。

其余为 OT 作损失的应用（GNN 融合、超图对齐、图重连、图凝聚、EEG、人脸、交通预测等），与主线无关，未收录。

## 3. 趋势判断

1. **图上 OT 的理论重心在「势」而不在「计划」。** 对偶预测加速最小费用流（2601.20203，AAAI 2026）、Sinkhorn 势的统计率（2608.29152）、QOT 对偶的 PL 不等式（2605.27175，SIAM J. Optim.）、Brenier 势估计（2604.22366）——四篇的共同对象都是对偶变量。含义：GFlowNet 学到的状态流 \(F(s)\) 在 O08 里就是对偶势的原始变量对应物，接上这套「势理论」比接上「计划预测」更顺。

2. **离散 SB 完成了从方法到基准的闭环。** DDSBM（C03，ICLR 2025）、GSBoG（C02，ICML 2026）之后出现了有解析解的离散 SB 基准（2509.23348）和收敛率结果（2607.19176）。一个方向有了基准和收敛率，就意味着新进入者必须在这套口径下报数——熵正则 GFN–OT 已没有「先定义问题」的红利。

3. **摊销 OT 拥挤度继续上升。** UNOT（O05）、ULOT（C01）之后，切片势摊销（2604.15114）、min-sliced 可迁移计划（2511.19741）、HyperTransport（2605.08254）在 2025-11 → 2026-05 连续出现。条件 GFN–OT 若要立足，只剩「隐式图、无法实例化代价矩阵」这一条护城河。

4. **二次正则是唯一没被占满的正则化方向。** 熵正则有 SB 一整套生态，二次正则（O02 的选择）只有 González-Sanz–Nutz 组在做优化理论（2605.27175 / 2605.27883），且都在连续/半离散设定。图上二次正则流 + GFlowNet 参数化，目前无人做。

5. **「非唯一时选哪一个」有独立的数学文献。** 距离代价下 OT 计划不唯一，熵选择原理（2512.05282、2502.16370）研究小正则极限选出哪个。O08 用最小总流作选择原则，与这条文献平行却互不引用；把两者接起来是一篇短文的体量。

6. **连续图（metric graph）上的神经 OT 刚出现。** 2606.16273 是第一篇，用嵌入 + 半对偶 + 投影；它不给逐边策略，与 O08 的离散、可执行策略互补而非替代。

7. **主会占比。** 93 篇里可据 comment 判定主会/期刊的 12 篇：NeurIPS 2025 ×2（含 Spotlight、Oral）、AAAI 2026 ×2、ICML 2026 ×1、ICLR 2026 ×1、ICDM 2026、WACV 2026、SIAM J. Optim.、L-CSS、VLDBJ、Globecom 2026。图上 OT 的理论工作主要以预印本形态存在。

## 4. 对 GFlowNet × OT 方向的含义

**可以直接借的技术**

| 技术 | 来源 | 用在哪 |
|---|---|---|
| 对偶预测 → \(\varepsilon\)-relaxation 精化，含预测误差到运行时间的界 | 2601.20203（AAAI 2026） | 把 GFlowNet 学出的状态流当对偶预测，做带保证的后处理；替代「GFN proposal + 经典修正」的模糊表述 |
| Sinkhorn 势的统计率与残差稳定性条件 | 2608.29152 | 「对偶势作证书」的统计一侧；与 O08 Thm 3.3 的确定性对偶合成完整误差分解 |
| QOT 对偶的局部误差界与 PL 不等式 | 2605.27175 | 图上二次正则流（O02 路线）的收敛率证明模板 |
| 有解析解的离散 SB 基准 | 2509.23348 | 任何熵正则 GFN–OT 实验的必报基准 |
| Regime-switching SB 的 Sinkhorn 指数收敛证明 | 2607.19176 | 离散分量上的 IPF 收敛技术 |
| Beckmann 解对参数的 Hölder 连续性 | 2603.19755 | 条件 GFN–OT 泛化保证需要的那类命题（连续版） |
| 熵选择原理 | 2512.05282、2502.16370 | 把「最小总流」与「小正则极限」两种选择原则对照 |

**已被占住的卖点**

- 「一个模型服务一族源—目标分布」：UNOT、ULOT、切片势摊销、min-sliced 计划，四篇以上。
- 「图上 Schrödinger 桥 / 熵正则传输，输出可执行策略」：GSBoG（ICML 2026 主会）已用几乎相同的句子。
- 「离散空间 SB 的基准与算法」：2509.23348 定了口径。

**结论**：OT 侧的扫描把 `COMPETITOR_MATRIX.md` 的评级又推了一步——课题 1（残差 → 误差界 + 对偶证书）不仅撞车风险最低，OT 侧还刚好提供了它需要的全部配件；课题 2、3 的窗口比 2026-08 的旧调研判断时更窄。

## 5. 做 GFN–OT 实验必须对比的强基线

| 方法 | 类型 | arXiv / 来源 | 代码（据 comment/摘要） |
|---|---|---|---|
| Network simplex / \(\varepsilon\)-relaxation 精确最小费用流 | 精确求解器 | 经典；对偶预测加速版 2601.20203 | 未知 |
| Sinkhorn（熵正则 OT，含 GPU 版 cuRegOT） | 正则化求解器 | 2605.08793 | 未知 |
| 二次正则图 OT（Essid & Solomon） | 正则化求解器 | 本仓库 O02 | 未知 |
| UNOT | 摊销 neural OT | 本仓库 O05（ICML 2025） | 有（见 O05 报告） |
| ULOT | 图间摊销 OT 计划 | 本仓库 C01（NeurIPS 2025） | 有 |
| GSBoG | 图上广义 SB，逐边策略 | 本仓库 C02（ICML 2026） | 未公开 |
| DDSBM / DLightSB / \(\alpha\)-CSBM | 离散 SB | C03；2509.23348 | DDSBM 有；其余未知 |
| ASBS | SB 采样器（连续） | 2506.22565（NeurIPS 2025 Oral） | 未知 |
| min-flow GFlowNet（T36 代码） | 本方 | 本仓库 T36 / O07 | 有 |

选基线的原则：显式小图上必须有精确求解器（否则「学到了 OT」无从验证）；隐式大图上必须有 GSBoG 或其复现（同象限的唯一竞品）；只要加了熵正则，就必须报 2509.23348 的解析解基准。

## 6. 方法论备注

- 全部判断基于 arXiv 元数据与摘要，未读正文；摘要没给的数字不写。
- 发表状态只信 `comment` / `journal_ref`；候选表 `data/candidates_ot.csv`（49 条）里 44 条标为预印本，其中相当一部分可能已被接收但 comment 未更新——引用前请自行核对 OpenReview / dblp。
- 检索噪声高于 GFN 侧：OT 是通用工具，约一半命中是应用；本报告只解读与图上传输、离散 SB、摊销、误差界直接相关的工作。

