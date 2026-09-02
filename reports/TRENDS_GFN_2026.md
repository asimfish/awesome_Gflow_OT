# 2025–2026 趋势扫描 · GFlowNet 侧

> 视角固定：这些新工作对「GFlowNet 学最优传输」这条主线意味着什么。发表状态只依据 arXiv 元数据的 `comment` / `journal_ref` 字段判断；字段为空的一律写「arXiv 预印本（未核实）」，Workshop 绝不写成主会。数据截止 2026-09-02。

## 1. 检索口径与覆盖

| 项目 | 内容 |
|---|---|
| 数据源 | arXiv API（`export.arxiv.org/api/query`），查询串见 `scripts/scan_trends.py`，原始命中存 `data/scan_gfn.json` |
| 时间窗 | `submittedDate` 2025-01-01 → 2026-09-30（实际最新命中 2026-08-24） |
| 查询 | 8 条：`all:GFlowNet`（95 命中）、`all:"generative flow network"`（75 命中，新增 11）、GFlowNet × {optimal transport / shortest path / minimum flow / non-acyclic / cyclic / Schrodinger bridge}、`"flow network" AND sampler AND reward` |
| 去重后 | **106 篇**；其中 **48 篇不在 2026-08-26 截止的旧调研（206 篇）里** |
| 按月分布 | 2025：01 月 6 · 02 月 5 · 03 月 7 · 04 月 3 · 05 月 8 · 06 月 6 · 07 月 1 · 08 月 5 · 09 月 6 · 10 月 9 · 11 月 6；2026：01 月 1 · 02 月 12 · 03 月 7 · 04 月 4 · 05 月 4 · 06 月 5 · 07 月 3 · 08 月 8 |
| 有 venue 线索的 | 64 篇 comment 非空；能据此判定主会/期刊接收的约 30 篇 |

两个覆盖上的事实要先说清：`all:GFlowNet` 与 `all:GFlowNets` 命中集合完全相同（arXiv 做了词干合并），所以 106 不是漏掉复数形式的结果；单条查询上限 100，`all:GFlowNet` 返回 95 < 100，说明该查询没有被截断。标题不含 "GFlowNet" 也不含 "generative flow network" 的论文（例如只在正文用 "GFN"）会漏，这是 arXiv 全文不可检索造成的固有盲区。

## 2. 主题分组卡片

每条格式：英文标题 · 第一作者 et al. · [arXiv] · 发表状态 · 解读 · 与主线的关系。已收录进本仓库 18 篇的只标注、不重复解读。

### 2.1 非无环 · 最短路 · 最小流 · OT（主线）

**Your GFlowNet Secretly Learns an Optimal Transport Plan** · Maksimov et al. · [2606.06272](https://arxiv.org/abs/2606.06272) · ICML 2026 SPIGM Workshop（comment）· 本仓库 O08，见 `reports/O08_2606.06272.md`。

**Learning Shortest Paths with Generative Flow Networks** · Morozov et al. · [2603.01786](https://arxiv.org/abs/2603.01786) · comment 为空；本仓库依旧调研记为 ICML 2026 SPIGM Workshop · 本仓库 O07。

**Revisiting Non-Acyclic GFlowNets in Discrete Environments** · Morozov et al. · [2502.07735](https://arxiv.org/abs/2502.07735) · ICML 2025（comment）· 本仓库 T36。

**Ergodic Generative Flows** · Brunswic et al. · [2505.03561](https://arxiv.org/abs/2505.03561) · ICML 2025（comment："accepted at ICML 2025"）
用遍历性（ergodicity）构造只含有限个全局定义微分同胚的生成流，给出普适性保证和可计算的 flow-matching 损失；另提出 KL-weakFM 损失做无需独立奖励模型的模仿学习。摘要点名「非无环训练的测试仍然有限」是它要解决的问题之一。
与主线的关系：T19 作者的续作，把非无环流理论推进到连续/微分同胚设定。它不谈 OT，但把「有环流的可计算 FM 损失」这个 T19 留下的缺口补上了一部分；若要把 O08 的最小流目标推到连续状态空间，这是最近的现成框架。

**Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels** · Korolev, Morozov et al. · [2606.16073](https://arxiv.org/abs/2606.16073) · ICML 2026 SPIGM Workshop（comment）
把 MCMC 放进非无环 GFlowNet 理论里，训练与状态相关的神经分类器决定轨迹何时终止；用 detailed balance 条件把最优分类器与目标密度联系起来，并用多层级训练方案处理复杂几何。实验称平均轨迹长度显著缩短、模式覆盖与混合改善。
与主线的关系：与 O07/O08 同一 HSE 团队。它把「轨迹长度」当成可学习的量，而 O07/O08 把「期望轨迹长度 = 总流」当成目标函数——两者是同一个量的两种用法。它也说明非无环 GFlowNet 理论正在成为该团队的统一语言，OT 只是其中一个出口。

**A Theory of Multi-Agent Generative Flow Networks** · Brunswic et al. · [2509.20408](https://arxiv.org/abs/2509.20408) · NeurIPS 2025 SPIGM Workshop（comment）
提出多智能体 GFlowNet 框架与四种算法（集中式、独立式、联合式及其条件版本），核心是「局部—全局原则」：一组局部 GFN 可以当作一个全局 GFN 来训练，从而复用单智能体理论保证。
与主线的关系：弱。值得记的是「局部—全局」这个证明技巧，与 O08 把局部路由策略拼成全局耦合的思路同构。

**Exploration through Generation: Applying GFlowNets to Structured Search** · Matovic et al. · [2510.21886](https://arxiv.org/abs/2510.21886) · arXiv 预印本（未核实）· 仅题录：本报告未读其摘要以外的内容，不作解读。

### 2.2 训练目标 · 稳定性 · 误差证书

**Stable GFlowNets with TV Monitoring and Probabilistic Guarantees** · Lei et al. · [2605.01729](https://arxiv.org/abs/2605.01729) · arXiv 预印本（comment 为空，未核实）
先证明学到的分布与目标分布之间的 TV 距离很小并不排除训练损失无界；再反过来给出 loss→TV 界：有界的 trajectory balance 损失可以认证全局保真度；最后用自适应参考流做稳定训练。
与主线的关系：**这是 GFN 侧与「Balance 残差 → OT 误差界」课题最近的一篇**。它已经把 TB 残差和 TV 距离连起来；OT 课题要做的是把右端换成 OT cost gap 与边缘违反量。做该课题必须引用并区分。

**\(f\)-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data** · Fawkes, Hartford · [2605.15417](https://arxiv.org/abs/2605.15417) · ICML 2026（comment："Published at ICML 2026"）
证明目标与模型对数概率之间的平移不变损失与 \(f\)-散度一一对应：on-policy 时梯度等于对应 \(f\)-散度的梯度，off-policy 时仍是同一全局最小点的合法损失。
与主线的关系：损失族不改变最小点，所以不改变 O08 的解；但它改变到达最小点的梯度几何，对「加了最小流正则后训练是否稳定」有直接影响，是做实验时应扫的一个轴。

**Proximal Policy Optimization for Amortized Discrete Sampling** · Zykova-Myzina, Gritsaev, Tiapkin, Morozov · [2606.15793](https://arxiv.org/abs/2606.15793) · arXiv 预印本（未核实）
基于 GFlowNet 与熵正则 RL 的等价，推导标准策略梯度算法的 GFlowNet 版本（含 baseline 与 advantage 估计），首次把 PPO 用于 GFlowNet 训练，报告收敛更快、数据效率更高。
与主线的关系：同一团队。O08 的实验基于 T36 的公开代码；PPO 版本是他们下一轮实验最可能用的优化器。做对照实验时应把 PPO-GFN 列为训练器之一。

**Information-Geometric Forward Policy Training in GFlowNets** · Raykov, Veiga · [2608.03967](https://arxiv.org/abs/2608.03967) · arXiv 预印本（未核实）
把前向策略看成诱导的轨迹采样器，证明其一阶几何是轨迹族的 Fisher–Rao 度量，自然梯度是典范局部更新；给出轨迹 Fisher 到逐步条件二阶矩的精确分解，并划分三种可计算性 regime。
与主线的关系：O08 只谈最优解，不谈优化路径；Fisher 几何决定训练在「同终止分布的一族内部流」里往哪边走，这正是隐式正则化问题。

**Controlling Exploration-Exploitation in GFlowNets via Markov Chain Perspectives** · Chen, Drapeau et al. · [2602.01749](https://arxiv.org/abs/2602.01749) · arXiv 预印本（未核实）
建立 GFlowNet 目标与 Markov 链可逆性之间的等价，指出现有目标隐含地固定了前向/后向策略的等权混合；提出 \(\alpha\)-GFN 用可调参数控制混合，保持收敛到唯一流，模式发现数最多提升 10 倍（摘要数字）。
与主线的关系：「可逆性」是 DB 条件的 Markov 链读法，与 T19/T36 把 GFlowNet 看成吸收 Markov 链的视角同源。

**Evaluating GFlowNet from partial episodes for stable and flexible policy-based training** · Niu, Wu, Qian · [2603.01047](https://arxiv.org/abs/2603.01047) · ICLR 2026（comment）
证明 flow balance 同时给出一个有原则的策略评估器（度量策略散度），提出在部分片段上学习该评估器的 evaluation balance 目标，连接价值型与策略型两条训练路线。
与主线的关系：把 balance 残差当「度量」而不只是「损失」，与 Stable GFlowNets 的方向一致。

**Relative Trajectory Balance is equivalent to Trust-PCL** · Deleu, Nouri, Bengio, Precup · [2509.01632](https://arxiv.org/abs/2509.01632) · arXiv 预印本（未核实）
证明 RTB 与带 KL 正则的 off-policy RL 方法 Trust-PCL 等价，并重做 RTB 论文的示例，显示 KL 正则 RL 能达到可比性能。
与主线的关系：又一条 GFN ⇔ KL 正则 RL 的等价；对 OT 主线的启示是「熵正则 GFN–OT」若成立，其 RL 侧对应物大概率已经存在。

**Secrets of GFlowNets' Learning Behavior: A Theoretical Study** · Yu · [2505.02035](https://arxiv.org/abs/2505.02035) · arXiv 预印本（未核实）
从收敛性、样本复杂度、隐式正则化、鲁棒性四个维度做理论分析。摘要未给具体定理内容。
与主线的关系：「隐式正则化」直接对应「终止分布相同时训练会选到哪个内部流」；是否与最小流原则一致，是可检验的问题。

**Symmetry-Aware GFlowNets** · Kim, Lee, Oh · [2506.02685](https://arxiv.org/abs/2506.02685) · ICML 2025（comment）
指出图生成中状态转移概率计算因图对称性产生系统偏差，通过奖励缩放把修正并入学习过程，避免显式计算转移概率。
与主线的关系：对称性意味着「同一对象有多条等价构造路径」——这正是内部流自由度的组合来源之一。

其余训练类工作（只列题录）：Avoid What You Know: Divergent Trajectory Balance（Dall'Antonia et al., [2602.17827](https://arxiv.org/abs/2602.17827)，预印本 "under review"）；Boosted GFlowNets（同组，[2511.09677](https://arxiv.org/abs/2511.09677)，预印本；顺序训练残差奖励的集成）；Rooted Absorbed Prefix Trajectory Balance（[2603.00454](https://arxiv.org/abs/2603.00454)，预印本；LLM 微调中的前缀塌缩）；Loss-Guided Auxiliary Agents（[2505.15251](https://arxiv.org/abs/2505.15251)，AAAI 2026）；Partial GFlowNet（[2602.11498](https://arxiv.org/abs/2602.11498)，预印本）；Signal from Structure: Submodular Upper Bounds（[2601.21061](https://arxiv.org/abs/2601.21061)，预印本）；GFlowState 训练可视化（[2604.21830](https://arxiv.org/abs/2604.21830)，预印本）。

### 2.3 扩散采样器 · 路径空间控制 · 连续时间

**Sampling Decisions: Exact Path-Space Control for Physics-Informed Generative Sampling** · Chertkov, Behjoo, Ahn · [2503.14549](https://arxiv.org/abs/2503.14549) · arXiv 预印本（未核实）
在增长的状态图上逐步装配对象，再用精确的路径空间控制律做全局修正：对给定 Gibbs 目标，修正律是序列先验的相对熵投影，由 Doob \(h\)-变换加一个线性的后向「desirability」递推实现。摘要明说同一对象可以等价地读成 KL 最优控制器、单侧 Schrödinger 传输、以及自回归/GFlowNet 型生成的理想价值或流函数。
与主线的关系：**它把 GFlowNet 的流函数与单侧 Schrödinger 传输写进同一句话**。做「熵正则 GFN–OT」的人会发现地基已经被打好一半——这是该课题撞车风险高的又一证据。

**From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training** · Berner, Richter, Sendera, Rector-Brooks et al. · [2501.06148](https://arxiv.org/abs/2501.06148) · TMLR（comment："TMLR final version"；代码 GFNOrg/gfn-diffusion）
在离散步长趋零的极限下证明若干目标族的等价，把熵正则 RL 方法（GFlowNets）与连续时间对象（PDE、路径空间测度）连起来；并证明合适的粗时间离散化能大幅提高样本效率。
与主线的关系：给「O08 的图上结论能否推到连续状态空间」提供极限工具；连续版的最小总流对应什么泛函，目前没有人写。

**PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching** · Chen et al. · [2603.18363](https://arxiv.org/abs/2603.18363) · ICML 2026（comment："Camera-ready version accepted at ICML 2026"）
把无监督 LLM 微调改写成分布匹配问题，将 GFlowNet 当作未归一化密度的摊销变分采样器，提出长度感知的 TB 目标。
与主线的关系：弱；记录它是因为「长度感知」的 TB 目标与 O07 的轨迹长度惩罚形式相近，但动机完全不同。

### 2.4 LLM 中的 GFlowNet

这一组数量最多，但与 OT 主线只有一个交点：配分函数 \(Z\) 的命运。

**GFlowRL: Scaling Distribution-Matching RL to Large Language Models** · Liu et al. · [2607.13394](https://arxiv.org/abs/2607.13394) · arXiv 预印本（未核实）
摘要直接指出：当模型规模、rollout 长度、奖励噪声与分布式系统复杂度一起增长时，学习的 prompt 条件配分函数会变成梯度不稳定与工程负担的来源而不是有用的归一化器。

**Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance** · Kwon et al. · [2605.00553](https://arxiv.org/abs/2605.00553) · ICML 2026 Spotlight（comment）
用成对比较消掉 \(Z\) 的估计，降低训练不稳定性；面向红队攻击的多样性与鲁棒性。

**Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR** · Kim et al. · [2602.12642](https://arxiv.org/abs/2602.12642) · arXiv 预印本（未核实）
反其道而行：把 \(Z\) 重新解释为每个 prompt 的期望奖励（在线准确率）信号，用它做难度调度。

**Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets** · Xue et al. · [2602.10583](https://arxiv.org/abs/2602.10583) · ICLR 2026（comment）
同一句子可由不同长度的 span 组成，形成 DAG 状态空间而非树；用 GFlowNet 在 DAG 上做探索与泛化。
与主线的关系：「同一对象、多条构造路径」是内部流自由度的语言学版本，天然适合作 O08 类分析的应用场景。

其余（题录）：Generating Attacks for LLMs with GFlowNets（[2608.10171](https://arxiv.org/abs/2608.10171)，预印本）；Active Attacks: Red-teaming LLMs via Adaptive Environments（[2509.21947](https://arxiv.org/abs/2509.21947)，预印本）；GFlowPO 提示优化（[2602.03358](https://arxiv.org/abs/2602.03358)，预印本）；PRM-Guided GFlowNets for LLM Mathematical Reasoning（[2504.19981](https://arxiv.org/abs/2504.19981)，预印本）；Do GFlowNets Transfer? Case Study on the Game of 24/42（[2503.01819](https://arxiv.org/abs/2503.01819)，预印本）；Latent Thought Flow（[2606.16222](https://arxiv.org/abs/2606.16222)，预印本）；Rooted Absorbed Prefix TB（见 2.2）。

主线读法：\(Z\) 在 LLM 场景被两派分别「消灭」与「重用」；而在 O08 的设定里，\(Z\) 根本不出现——固定源分布 \(L\) 与归一化目标 \(R\) 之后问题变成 LP，GFlowNet 放弃了自己「只需未归一化奖励」这一最强能力。这一点在 `reports/O08_2606.06272.md` §7.6 有专门讨论。

### 2.5 组合优化 · 结构学习 · 工具

**Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems** · Zhang, Cao · [2510.04792](https://arxiv.org/abs/2510.04792) · NeurIPS 2025（comment）
把 TB（全局）与 DB（局部）按互补优势自适应整合，并给 VRP 设计专门的推断策略。
与主线的关系：VRP 的状态图是巨大的隐式组合图，正是 O08 声称的生态位；但 HBG 优化的是解质量，不是分布或传输代价。

**Adversarial Generative Flow Network for Solving Vehicle Routing Problems** · Zhang et al. · [2503.01931](https://arxiv.org/abs/2503.01931) · ICLR 2025（comment）· 题录。

**Unrealized Expectations: Comparing AI Methods vs Classical Algorithms for Maximum Independent Set** · Wu, Zhao, Arora · [2502.03669](https://arxiv.org/abs/2502.03669) · TMLR 2026-04（comment）
即使在同分布随机图上，主流 AI 方法（含生成模型与 RL）也稳定输给单 CPU 上的经典求解器 KaMIS，部分方法连基于度的贪心都赢不了。
与主线的关系：给 GFN×OT 一个方法论警告——若拿 GFlowNet 去解显式图上的 OT，network simplex 一类经典求解器会是同样残酷的基线；GFN 的价值只能建立在经典求解器进不去的隐式图上。

**Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation** · Yoon et al. · [2602.21565](https://arxiv.org/abs/2602.21565) · ICML 2026（comment："Appears in the 43rd International Conference on Machine Learning"）
推断时组合预训练 GFlowNet，无需为每种目标组合重训。
与主线的关系：「组合多个流」在 OT 语言里对应多边缘/多目标传输，是一个尚未有人写下的对应。

**gfnx: Fast and Scalable Library for Generative Flow Networks in JAX** · Tiapkin, Agarkov, Morozov et al. · [2511.16592](https://arxiv.org/abs/2511.16592) · arXiv 预印本（comment 给出 GitHub `d-tiapkin/gfnx`）
JAX 实现，含 hypergrid、多种序列环境、分子生成、系统发育树、贝叶斯结构学习、Ising 能量采样；宣称显著的 wall-clock 加速。
与主线的关系：O07/O08 团队的库。做复现实验时，它与 torchgfn 是两套候选基础设施；hypergrid 环境与 O08 Table 1 的实验直接对应。

**Generative Flow Networks: Theory and Applications to Structure Learning** · Deleu · [2501.05498](https://arxiv.org/abs/2501.05498) · 博士论文（预印本）· 题录：系统整理到 2024 年的 GFlowNet 理论，可作 T02 之后的第二本教材。

其余（题录）：Learning Decision Trees as Amortized Structure Inference（[2503.06985](https://arxiv.org/abs/2503.06985)，预印本，GFNOrg 代码）；Consistent Amortized Clustering via GFlowNets（[2502.19337](https://arxiv.org/abs/2502.19337)，AISTATS 2025）；FlowQ-Net 量子电路设计（[2510.26688](https://arxiv.org/abs/2510.26688)，预印本）。

### 2.6 应用（与主线无直接关系，只记发表状态以校准「主会占比」）

EraseFlow 概念擦除（[2511.00804](https://arxiv.org/abs/2511.00804)，NeurIPS 2025 Spotlight）；Compositional Flows for 3D Molecule and Synthesis Pathway Co-design（[2504.08051](https://arxiv.org/abs/2504.08051)，ICML 2025）；Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training（[2505.20110](https://arxiv.org/abs/2505.20110)，ICML 2026）；FlowPipe 数据准备流水线（[2606.24679](https://arxiv.org/abs/2606.24679)，SIGMOD 2027）；LM-based TTS 幻觉缓解（[2508.15442](https://arxiv.org/abs/2508.15442)，EMNLP 2025 Main Oral）；Torsional-GFN 分子构象（[2507.11759](https://arxiv.org/abs/2507.11759)，预印本）；IFlowNets 不完全信息博弈（[2608.05422](https://arxiv.org/abs/2608.05422)，NeurIPS 2025 Workshop）；FlowNeg 知识图谱负采样（[2608.23849](https://arxiv.org/abs/2608.23849)，预印本）；Curriculum-Augmented GFlowNets for mRNA（[2510.03811](https://arxiv.org/abs/2510.03811)，预印本）；Interpreting GFlowNets for Drug Discovery（[2511.19264](https://arxiv.org/abs/2511.19264)，预印本）。

## 3. 趋势判断

每条给出支撑论文的 arXiv 号；只有一篇支撑的不算趋势。

1. **非无环 GFlowNet 从「理论修补」变成「生成工具」，但停在 Workshop。** T36（2502.07735，ICML 2025 主会）之后一年内，同一 HSE 团队把非无环理论推向最短路（2603.01786）、最优传输（2606.06272）、MCMC 终止（2606.16073），Brunswic 组推向连续/遍历设定（2505.03561，ICML 2025 主会）与多智能体（2509.20408）。三篇应用出口全部落在 SPIGM Workshop，主会只有两篇理论。判断：这条线的理论已被主会接受，应用叙事还没有。

2. **误差证书成为新的竞争点。** Stable GFlowNets（2605.01729）证明低 TV 不排除无界损失并给出 loss→TV 反向界；Evaluation Balance（2603.01047，ICLR 2026）把 flow balance 当策略评估器；Secrets（2505.02035）谈样本复杂度与隐式正则。「低 loss 不等于低分布误差」这个老问题正被系统处理。对本仓库最重要的含义：把残差界从 TV 推到 OT cost gap 的空位还在，但 GFN 侧的工具已经成熟。

3. **训练目标进入「族」的时代。** \(f\)-TB（2605.15417，ICML 2026）、\(\alpha\)-GFN（2602.01749）、Divergent TB（2602.17827）、Hybrid-Balance（2510.04792，NeurIPS 2025）、RapTB（2603.00454）、Evaluation Balance（2603.01047）：目标不再是 FM/DB/TB 三选一，而是同一最小点下可调的梯度几何。O08 的最小流目标可以叠加在其中任一族之上，这是做实验时该扫的轴。

4. **策略梯度回流 GFlowNet。** PPO for amortized discrete sampling（2606.15793）、信息几何自然梯度（2608.03967）、RTB ≡ Trust-PCL（2509.01632）、PowerFlow（2603.18363，ICML 2026）、GFlowRL（2607.13394）：GFN 与 KL 正则 RL 的等价被反复用来借 RL 的优化器。含义：熵正则 GFN–OT 的 RL 对应物大概率已存在于 KL 正则 RL 文献里。

5. **\(Z\) 的两种命运。** LLM 场景里 \(Z\) 被消灭（Stable-GFN 成对比较，2605.00553，ICML 2026 Spotlight；GFlowRL 指出 prompt 条件 \(Z\) 是不稳定源，2607.13394）或被重用（\(Z\) 作难度调度器，2602.12642）。而 O08 的 LP 设定里 \(Z\) 不出现——这条主线放弃了 GFlowNet「只需未归一化奖励」的招牌能力，见 O08 报告 §7.6。

6. **路径空间控制与 Schrödinger 语言正在统一 GFN 与采样器。** Sampling Decisions（2503.14549）把 GFlowNet 流函数、Doob \(h\)-变换、单侧 Schrödinger 传输写成同一对象；Berner 等（2501.06148，TMLR）给出离散↔连续时间的渐近等价。熵正则 GFN–OT 的理论地基已被别人打好一半，撞车风险因此高。

7. **主会占比与类型。** 可据 comment 判定的主会/期刊接收约 30 篇：ICML 2025 ×4、ICML 2026 ×5、ICLR 2025/2026 ×3、NeurIPS 2025 ×2、AAAI 2026 ×1、AISTATS 2025 ×1、TMLR ×2、EMNLP 2025、KDD 2026、SIGMOD 2027、ACM MM 2025。应用类多于理论类；与 OT 直接相关的主会论文为零（O07/O08 均为 Workshop）。

8. **工具层出现第二套基础设施。** gfnx（JAX，2511.16592）与 torchgfn 并列，GFlowState（2604.21830）补可视化。复现实验有了两条独立路径。

## 4. 对 GFlowNet × OT 方向的含义

对照 `reports/COMPETITOR_MATRIX.md` 的四个候选课题：

| 课题 | 窗口状态 | 依据 |
|---|---|---|
| Balance 残差 → OT 误差界（+ 对偶势证书） | **打开，且工具已备齐** | GFN 侧有 loss→TV 界（2605.01729）与 balance-as-evaluator（2603.01047）可直接引用；OT 侧的 LP 对偶与互补松弛在 O08 Thm 3.3 现成；尚无人把两端接起来 |
| 条件 GFN 学一族图上 OT | 关闭中 | ULOT（本仓库 C01，NeurIPS 2025）与 UNOT（O05，ICML 2025）已占据摊销轴；GFN 侧没有出现新的条件化工具 |
| 熵正则 GFN–OT / Schrödinger 桥 | 关闭中 | Sampling Decisions（2503.14549）已把 GFN 流函数与单侧 Schrödinger 传输统一；GSBoG（C02，ICML 2026）与 DDSBM（C03，ICLR 2025）占据图上 SB；α-DSBM（O04）给出同构的投影算法 |
| GFN proposal + 经典 OT 修正 | 降为基线 | Unrealized Expectations（2502.03669，TMLR）警示：显式图上的经典求解器是残酷基线；GFN 只能在隐式图上立足，而隐式图上没有经典 OT 求解器可「修正」 |

两个新打开的小窗口：(a) **最小流目标 + 策略梯度/PPO 训练器**（2606.15793 是同一团队的下一步，先做就先占）；(b) **对称性与内部流自由度**（2506.02685 的对称修正与 T02 的 \(P_B\) 自由度是同一现象的两面，尚无人从 OT 角度写）。

## 5. 与旧调研（206 篇，截止 2026-08-26）的差异

按 arXiv 号与标题双重比对，106 篇中 **48 篇不在旧调研里**。其中 2026-06 之后的全新论文：

| arXiv | 标题 | 状态 |
|---|---|---|
| 2608.23849 | FlowNeg: GFlowNet-Guided Diverse Hard Negative Sampling for KG Embedding | 预印本 |
| 2608.11396 | Generative Learning for Quantum Measurement Design | 预印本 |
| 2608.10171 | Generating Attacks for LLMs with GFlowNets | 预印本 |
| 2608.05422 | IFlowNets: Incomplete Information Games | NeurIPS 2025 Workshop |
| 2608.05314 | ML for sample-based quantum diagonalization（综述） | 预印本 |
| 2608.03967 | Information-Geometric Forward Policy Training in GFlowNets | 预印本 |
| 2608.01789 | Autonomous Formulaic Alpha Discovery | 预印本 |
| 2608.01303 | AlphaG-OPD | 预印本 |
| 2607.13394 | GFlowRL | 预印本 |
| 2607.06432 | TILDE concept unlearning | 预印本 |
| 2607.05266 | Distributional Framework for Molecular Crystals | 预印本 |
| 2606.24679 | FlowPipe | SIGMOD 2027 |
| 2606.16222 | Latent Thought Flow | 预印本 |
| 2606.16073 | Stop the Sampler! | ICML 2026 SPIGM Workshop |
| 2606.15793 | PPO for Amortized Discrete Sampling | 预印本 |
| 2606.06272 | Your GFlowNet Secretly Learns an OT Plan（O08） | ICML 2026 SPIGM Workshop |

旧调研遗漏但 2026-06 之前就存在、且与主线相关的：Ergodic Generative Flows（2505.03561，ICML 2025）、Sampling Decisions（2503.14549）、Symmetry-Aware GFlowNets（2506.02685，ICML 2025）、Multi-Agent GFlowNets 理论（2509.20408）、Evaluation Balance（2603.01047，ICLR 2026）、Routing by Reaching（2602.21565，ICML 2026）、Flow of Spans（2602.10583，ICLR 2026）、Deleu 博士论文（2501.05498）。完整候选清单（38 条，含 relevance 评分与理由）见 `data/candidates_gfn.csv`。

## 6. 方法论备注

- 本报告全部判断基于 arXiv 元数据与摘要，未读正文；凡摘要未给的数字一律不写。
- 发表状态只信 `comment` / `journal_ref`。已知 comment 会滞后：O07（2603.01786）comment 为空，但本仓库依旧调研记为 ICML 2026 SPIGM Workshop；类似情况在候选表里都标了「未核实」。
- 检索固有盲区：标题与摘要都不含 "GFlowNet" / "generative flow network" 的论文查不到。

