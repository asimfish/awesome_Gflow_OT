# 竞品矩阵 · GFlowNet–OT 主线的撞车分析

> 覆盖 O08 / O07（本方）与 ULOT / GSBoG / DDSBM（竞品）五篇。所有字段来自各论文原文，出处标在括号里。
> 数据截止 **2026-09-02**。发表状态已通过 arXiv API、dblp、OpenReview、会议官网四路核实。

## 0. 五篇的身份牌

| 编号 | 简称 | 标题 | 发表状态（已核实） | 代码 |
|---|---|---|---|---|
| **O08** | GFN-OT | Your GFlowNet Secretly Learns an Optimal Transport Plan | **ICML 2026 SPIGM Workshop**（arXiv comment 明写；PDF 首页 "Preprint. June 5, 2026"） | 未见公开 |
| **O07** | GFN-SP | Learning Shortest Paths with Generative Flow Networks | **ICML 2026 SPIGM Workshop**（本仓库 `data/papers.yaml`；arXiv 2603.01786 v1 2026-03-02） | `github.com/GreatDrake/gfn-pathfinding`（原文 §1 给出） |
| **C01** | ULOT | Unsupervised Learning for OT plan prediction between unbalanced graphs | **NeurIPS 2025 主会**（proceedings 页 "Main Conference Track"，DOI 10.52202/085713-3146；dblp NeurIPS 2025） | `github.com/smazelet/ULOT` |
| **C02** | GSBoG | Generalized Schrödinger Bridge on Graphs | **ICML 2026 主会**（PDF 页脚 PMLR 306 camera-ready；作者主页 2026-04 接收公告；GaTech Spotlight 2026）。**dblp 截至 2026-09 仅 CoRR 条目** | 未公开 |
| **C03** | DDSBM | Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation | **ICLR 2025 Poster**（OpenReview `tQyh0gnfqW`；dblp ICLR 2025；arXiv comment "Accepted to ICLR 2025"） | `github.com/junhkim1226/DDSBM` |

**第一个要认清的事实**：本方两篇是 **workshop**，三篇竞品分别是 NeurIPS 主会、ICML 主会、ICLR 主会。在"叙事占位"这件事上，本方目前处于劣势。

## 1. 主矩阵

| 维度 | **O08** GFN-OT | **O07** GFN-SP | **C01** ULOT | **C02** GSBoG | **C03** DDSBM |
|---|---|---|---|---|---|
| **对象空间** | **隐式组合图**：一张有向状态图 \(G=(\mathcal S,E)\) 的内部；permutation \(n=20\) 时状态数 \(20!\approx2.4\times10^{18}\)（§4.2） | **隐式组合图**：Cayley 图、Rubik's Cube（3×3×3 状态数远超可枚举） | **显式图**：两张给定图的节点集之间；需实例化 \(D_1,D_2\) 与 \(n_1\times n_2\) plan；硬上限 \(n\le10^4\)（§4） | **显式图**：固定已知拓扑，需枚举 \(N(x)\)；实测到 \(10^6\) 节点（§4.4）；但 \(\mu,\nu\) 必须是显式 \(n\) 维向量 | **图空间**：一个样本 = 一整张分子图；传输在图与图之间，不在图内部 |
| **传输的是什么** | 图上的概率质量 | 单源到目标的路径（OT 的退化情形） | 两图节点之间的对应质量 | 图上的概率质量（粒子） | 整张图（分子 → 分子） |
| **cost 结构** | 线性 Kantorovich，\(d(u,x)=|\tau_{u,x}|\) 图诱导最短路（Eq. (7)(8)） | 最短路长度（\(R\equiv1\) 的特例） | **二次非凸** FUGW：W 项 + GW 项（耦合 \(P_{i,j}P_{k,l}\)）+ \(\rho\)-KL（Eq. (1)(2)(3)） | \(\mathrm{KL}(p^u\|p^r)+\mathbb E\!\int f_t(x,p_t)\)；\(f_t\) 可依赖**当前边缘**（mean-field，Eq. (13)） | \(-\log q_{\tau|0}(y|x)\)，**正比于 GED**（附录 D.6，原文明说"不等于、成比例"） |
| **是否熵正则** | **否**。纯线性 LP，最优解是顶点解、支撑在最短路子图上（Thm 3.3 互补松弛） | 否 | **否**（是二次正则，不是熵正则）；对照实验里有 Sinkhorn 熵正则 FUGW 基线 | **是**。KL 到参考 CTMC \(r_t\)，解为指数族 \(u^\star=r\,e^{\Delta V}\)（Eq. (16)） | **是**。KL 到参考 CTMC；温度由噪声调度 \(\bar\alpha\) 隐式给定 |
| **参考动力学** | 无（等价于所有内部边代价 = 1） | 无 | 不适用 | **有且必须给定** \(r_t\)，编码拓扑 + 名义路由偏好 | **有**：节点/边独立跳变（Eq. (8)(9)） |
| **是否可条件化 / 摊销** | **否**。每对 \((L,R)\) 训一个模型 | 否 | **是，且是核心卖点**：一模型服务多图对 + 多 \((\alpha,\rho)\)；\(\alpha\) 用 Fourier 编码（Eq. (5)） | 否。每个 \((\mu,\nu,r,f)\) 训一次 | 否。每对分布训一次 |
| **输出：plan 还是 policy** | **policy** \(P_F(s'|s)\)；coupling \(\Pi_{u,x}=\sum_{\tau:u\leadsto x}P^\star(\tau)\) 是采样副产品（Thm 3.2） | **policy**（+ 可选 beam search） | **plan** 矩阵 \(P\in\mathbb R^{n_1\times n_2}\)。不回答"怎么走" | **policy** \(u_t(y,x)\)，逐边可执行；软 plan 可从通量解码（Eq. (87)） | **生成模型**：给 \(x_0\) 采 \(x_\tau\)。既非静态 plan 也非路由 policy |
| **误差 / 收敛保证** | **Thm 3.2** GFlow\(^\star=\)OT\(^\star\)（精确）；**Thm 3.3** LP 对偶 + 互补松弛 = primal-dual gap。**神经训练与 LP 之间无界** | **Thm 3.4**：最小总流 ⟹ 策略只走最短路 | **无。全文零定理** | **无误差界**。Thm 3.1 依赖未具体化的 "mild regularity assumptions"；IPF/TD 无离散收敛证明 | **Thm 3.3**：IMF 单调下降 + 依分布收敛到 \(\mathbb P^{\mathrm{SB}}\)。**无速率** |
| **时间结构** | 无时间轴，吸收态 \(s_f\)，\(\mathbb E[n_\tau]\) 自由 | 同左 | 不适用（静态） | **固定有限时域** \(T\)（供应链 100、MD 200） | 固定 \([0,\tau]\)，100 扩散步 |
| **边缘约束** | 硬：\(\sum L=\sum R=1\)（Assumption 3.1） | 单源 + 目标集 | **软**：\(\rho\)-KL 罚，unbalanced | 硬：\(p_0=\mu,p_1=\nu\) | 硬（数据集层面）：\(\mathbb P_0=\Gamma,\mathbb P_\tau=\Xi\) |
| **代码可用性** | 未见公开 | **已开源** | **已开源** | **未公开** | **已开源** |
| **最大实验规模** | hypergrid \(H=20\)；permutation \(n=20\)（Table 1/2） | 3×3×3 Rubik's Cube | 图 \(n=1000\)（fMRI），标度测到 \(10^4\) | 供应链 \(n=9559\)；标度测到 \(10^6\) 节点（Fig 6） | ZINC250K / Polymer 7603 对 |
| **算力披露** | MLP 2 层 ×128，CPU 训练（附录 B） | 原文未在本次精读范围内提取 | 单 V100 **100 小时**（附录 A） | AdamW \(5\times10^{-5}\)；\(5\times10^4\) 节点 1.4 GB / ~1 h（Fig 6） | **4× RTX A4000**（Table 4） |

## 2. 三条不能混淆的分界线

**分界线 A：图是"被传输的对象"还是"传输发生的场所"。**
ULOT 与 DDSBM 在前者（图是对象），O08 / O07 / GSBoG 在后者（图是场所）。跨线比较数字毫无意义——ULOT 的 "100× 加速" 与 O08 的 "\(\mathbb E|\tau|=3.990\) vs OT\(^\star=3.997\)" 不在同一个问题上。

**分界线 B：显式枚举 vs 隐式展开。**
GSBoG 虽然跑到 \(10^6\) 节点，但它要求 \(\mu,\nu\) 是显式 \(n\) 维向量、要求能枚举 \(N(x)\)、要求拓扑完全已知（§4.5 Limitations 首句）。O08 的 permutation 环境里 \(20!\) 个状态**没有任何显式表示**，只能通过"当前排列 + 合法相邻对换"局部展开。这是 GFN 侧唯一无法被复制的结构性优势。

**分界线 C：温度。**
O08/O07 在 \(\varepsilon=0\)（纯 LP，顶点解，有对偶证书）；GSBoG/DDSBM 在 \(\varepsilon>0\)（指数族解，光滑，有 IPF/IMF 训练机制）。ULOT 既不在这条线上（它的正则是二次的、罚在边缘上）。**"给 O08 加 KL 项"这个动作，就是把自己从 \(\varepsilon=0\) 挪到 \(\varepsilon>0\)——那里已经站着 GSBoG（ICML 2026 主会）和 DDSBM（ICLR 2025）。**

## 3. 各竞品对本方的具体威胁

| 竞品 | 威胁的是什么 | 强度 | 反制点 |
|---|---|---|---|
| **C02 GSBoG** | O08 的**整个定位叙事**。两篇的卖点句几乎逐字相同（"不是静态 coupling，而是可执行策略"）。GSBoG 在 ICML 2026 主会、\(10^6\) 节点、三个真实应用域 | **最高** | (i) 隐式组合图；(ii) 不定时域（GSBoG 必须固定 \(T\)）；(iii) O08 有 LP 对偶证书，GSBoG 无任何误差界；(iv) GSBoG 未开源 |
| **C01 ULOT** | 「条件 GFN 学一族图 OT」这个**具体课题** | 高 | (i) ULOT 硬上限 \(n\le10^4\)；(ii) 输出 plan 不输出 policy；(iii) 全文零定理；(iv) cost 是二次 FUGW，与线性 Kantorovich 不是同一问题 |
| **C03 DDSBM** | 「熵正则 GFN-OT 用在分子编辑上」这个**应用位** | 中 | (i) DDSBM 需解 QAP 图匹配，GFN 沿路径走天然不需要；(ii) DDSBM 不条件化；(iii) 它的 Thm 3.3 是"迭代收敛"不是"残差→cost gap" |
| **O07（本方）** | 不是威胁，是 O08 的前驱。O08 Thm 3.3 明说"This recovers the corresponding claim of Morozov et al. (2026)" | — | 两篇要合并叙事，不要各自为战 |

## 4. 四个候选课题的撞车风险评级

评级依据：**设定重合度**（竞品是否已在同一设定下工作）× **卖点重合度**（核心 claim 是否已被说过）× **发表场次**（对方在主会还是 workshop）。

### 课题 1 · Balance 残差 → OT 误差界
> 目标：证明形如 \(\big|\mathbb E[|\tau|]-\mathrm{OT}^\star\big|\le C(\varepsilon_{\mathrm{TB}},\lambda,|\mathcal S|)\) 的定量界，把 TB 损失残差翻译成传输代价 gap 与边缘误差。

**撞车风险：低（当前四个课题中最值得做）**

依据：
- 五篇竞品**没有一篇给出误差界**。ULOT 零定理；GSBoG 无界且 Thm 3.1 的正则性条件未具体化；DDSBM 的 Thm 3.3 只给单调性与依分布收敛、无速率、且假设 \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\) 对神经近似未验证。
- O08 已经把接口铺好：Thm 3.3 给出 LP 对偶 \(\max_\pi\sum_x R(x)\pi_x+\sum_u L(u)(-\pi_u)\) s.t. \(\pi_{s'}-\pi_s\le1\)（附录 A.4 Eq. (24)(25)），以及互补松弛 \(F^\star(s\to s')(\pi^\star_{s'}-(1+\pi^\star_s))=0\)。**primal-dual gap 是现成的误差证书载体**。
- 本仓库背景文档 §5 已把 primal-dual 列为最值得做的方向，与本次撞车扫描独立得出同一结论。
- 需要预防的质疑：审稿人会拿 DDSBM Thm 3.3 来问"IMF 都有收敛定理了"。回答必须准备好：那是"迭代序列收敛到 SB"，本课题是"单个模型的残差 → cost gap"，前者不蕴含后者，也不能给出可计算的证书。

### 课题 2 · 条件 GFN 学一族图 OT
> 目标：训练 \(P_F(a|s,L,R,c)\)，对未见的 \((L,R)\)、cost 泛化。

**撞车风险：高**

依据：
- ULOT（NeurIPS 2025 主会）已经把摊销 + 条件化 + 图 + 无监督 + unbalanced + warm-start 六件事一次做完，并在 fMRI 上给出 14400 图对的泛化实验。
- O05 Universal Neural OT（ICML 2025）在通用神经 OT 侧也占了摊销位。
- **仍可做的形态（唯一）**：把条件变量取为**隐式组合图上无法显式化的东西**——例如条件于 cost 权重 \(\lambda\)（O08 Table 2 显示 \(\lambda=10^{-1}\) vs \(10^{-2}\) 存在强 trade-off，一次训练给出整条正则路径是干净的增量），或条件于 Cayley 图的生成元集合。**在显式图上做条件化 = 直接撞 ULOT。**
- 若坚持做，必须在实验里把 ULOT 当基线跑一遍（在它能跑的规模上），否则审稿人会问"为什么不用 ULOT"。

### 课题 3 · 熵正则 GFN–OT / Schrödinger 桥
> 目标：\(\min_P\mathbb E_P[c(\tau)]+\varepsilon\mathrm{KL}(P\|P_0)\)，固定两端边缘。

**撞车风险：高**

依据：
- GSBoG（ICML 2026 主会）已完整占据「图 + 参考动力学 + 熵正则 + 可执行局部策略 + 中间态成本」。它甚至给了完整技术栈：对偶（Thm 3.1）→ Hopf–Cole（Prop 3.4）→ 生成元恒等式（Prop 3.5）→ gIPF（Prop 3.6）→ TD 修正（Eq. (28)(29)）。
- DDSBM（ICLR 2025）占据离散 SB + 图 + GED cost。
- 加上 Ksenofontov & Korotin (2025) categorical SB、Guo et al. (2026) 离散 adjoint SB、Yang (2025) topological SBM——**这个生态已经拥挤**。
- **仍可做的形态**：(i) **不定时域**——GSBoG/DDSBM 都必须固定 \(T\)，GFN 是吸收型、轨迹长度自适应，把 KL 打在轨迹分布上而非固定时间网格上，这条缝是真的；(ii) **隐式组合图上的熵正则 SB**；(iii) **mean-field / 拥塞感知 GFlowNet**（GSBoG 的 \(f_t(x,p_t)\) 在 GFN 文献里无对应物，但要先排查多智能体 GFlowNet 是否已沾边）。
- 建议：**不要把它当独立课题，把它当课题 1 的 \(\varepsilon>0\) 推广**。误差界在 \(\varepsilon>0\) 下往往更容易证（强凸性），这样两条线合并后既有理论增量又不与 GSBoG 正面撞。

### 课题 4 · GFN proposal + 经典 OT 修正
> 目标：GFN 输出粗解 → 喂给 network simplex / Sinkhorn 做精修。

**撞车风险：中（建议降级为对照基线，不作主线）**

依据：
- ULOT §3.2 + Figure 8(right) 已经把"神经预测 → 经典 solver warm start"这个 idea 演示过一遍，并且明确写进 abstract。novelty 已被占。
- 但它作为**评测协议**仍然有价值：把 GFN 学到的策略采样成 coupling，喂给精确 solver，报告迭代数下降比例——这是一个几乎零成本、能直接放进消融表的实验，也是回应"GFN 到底有没有用"这类质疑的最简答案。
- 独立发表价值低，作为课题 1 的一节则很合适。

## 4.5 可以立刻开跑的四个 red-team 实验

这些实验的目的不是发论文，是**在投稿前先把审稿人会问的问题自己答一遍**。

1. **ULOT 反打实验**：把 O08 的 permutation 环境压缩到 \(n_1,n_2\le10^4\) 个代表状态，构造显式 cost matrix，直接跑 ULOT / POT。如果 ULOT 在这个规模上打得过 GFN，那"隐式图"这条护城河必须收窄到"连 \(10^4\) 个代表状态都取不出来"的场景。
2. **warm-start 协议**：把 GFN 策略采样成 coupling → 喂给 network simplex，报告迭代数下降比例。对标 ULOT Figure 8(right)。零成本，直接进消融表。
3. **primal-dual gap 监控**：训练过程中同时跟踪 O08 Thm 3.3 的对偶目标与互补松弛违反量 \(\sum_{s\to s'}F(s\to s')\big|\pi_{s'}-(1+\pi_s)\big|\)。这既是课题 1 的原型，也是唯一能证明"GFN 输出可信"的在线证书。
4. **四轴 scaling 对照**：复现 GSBoG §4.4 的（节点数 / 边密度 / 时间步 / rollout 预算）四轴内存与耗时图，把 GFN 放进去。注意 GSBoG **未开源**，需自行实现——这也意味着如果本方开源，在可复现性上反而占优。

## 5. 一句话结论

**对 O08 威胁最大的是 C02 GSBoG**：它与 O08 在同一象限（图 + 两端边缘 + 可执行局部策略）、用几乎相同的卖点句、发在 ICML 2026 主会、实验规模高出四个数量级。但它有三个 O08 没有的短板——需要完全已知的显式拓扑、必须固定有限时域、**没有任何误差界**。

因此推荐的主线是：**把「Balance 残差 → OT 误差界」（低撞车）与 primal-dual/对偶势（O08 Thm 3.3 现成）合并为一条主线，实验场景钉死在隐式组合图（permutation / 分子编辑 / Rubik's Cube），并把 ULOT 与 GSBoG 作为"在它们能跑的规模上"的对照基线。** 熵正则版本作为该主线的 \(\varepsilon>0\) 推广，而不是独立课题。

---

**编者注**（歧义处的自行决定）：
1. O07 的发表状态取自本仓库 `data/papers.yaml`（ICML 2026 SPIGM Workshop），本次未对其单独走 arXiv/dblp 核实——它是本方论文，不在"三篇竞品核实"的任务范围内。O08 的 workshop 状态则由 arXiv API 的 comment 字段直接确认。
2. C02 标为"ICML 2026 主会"而非"预印本"，依据是三条独立证据（PMLR 306 camera-ready 页脚、作者主页接收公告、GaTech Spotlight），但 dblp 截至 2026-09 仅有 CoRR 条目，此差异已在 C02 报告与本表中双重标注。
3. 表中"最大实验规模"的可比性有限：五篇的规模指标含义不同（状态数 / 节点数 / 分子对数），不宜横向排序，仅供判断"各自在什么量级上工作"。
4. §4 的风险评级（低/中/高）为本报告判断，不是任何原文的表述。评级方法（设定重合 × 卖点重合 × 发表场次）已在该节开头写明。
5. 建议在 NeurIPS 2026 放榜（2026-09-24）后复查 O07/O08 是否升级为主会，并复查 C02 在 PMLR 306 卷的正式页码。
