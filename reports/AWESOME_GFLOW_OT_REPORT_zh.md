---
title: "GFlowNet × 最优传输：从流守恒到 Kantorovich 计划"
subtitle: "awesome_Gflow_OT 汇总报告"
author: "awesome_Gflow_OT 项目组"
date: "2026-09"
lang: zh-CN
---

# 摘要与阅读指南

**摘要。** 生成流网络（GFlowNet）把奖励匹配写成图上的流守恒：终止边流等于奖励，内部状态流入等于流出。奖励只钉住边界，内部流有一族合法解。本报告围绕一条主线整理 18 篇论文：在允许有环的有限图上，内部状态总流与期望轨迹长度成正比（T36，Prop. 3.12，内部状态计数）；单源、固定终止分布时，期望长度取到下界 \(\sum_xR(x)d_G(s_0,x)/Z\) 当且仅当每个终止对象都沿各自的最短路到达（O07，Thm. 3.4，须允许非最短轨迹概率为零）；再把第一步边流固定为一个源分布 \(L\)，扣除一个常数后的内部边总流目标就等于以图最短路为代价的 Kantorovich 最优传输，最优策略下「第一步进入的源状态 \(u\)、停止前的终止对象 \(x\)」的联合分布是一个最优耦合（O08，Thm. 3.2）。这条等价链的每一环在经典最小费用流与 Beckmann 问题里都有对应（O01、O02）；本报告判断：新增之处不是「输出策略」本身（经典边流归一化后也是局部策略，GSBoG 同样学可执行策略），而是「非无环 GFlowNet 最小流目标 + 固定两端边缘 + 神经策略参数化 + 隐式图上的训练与执行」这一组合——O08 的 \(20!\) 排列实验是可执行性示范，该规模下没有精确 OT 参照值。报告同时给出竞争格局（三篇主会论文 ULOT、GSBoG、DDSBM 与两篇 Workshop 论文的比较，比较维度是图是被匹配的对象还是运输发生的场所、是否熵正则、是否固定时域、是否需要显式图）、2025–2026 两侧趋势扫描（arXiv 命中 199 篇，其中精读 6 篇、收录候选 87 条）、四个候选课题的评级与一个可立即开工的实验设计。本报告判断：在所比较的四个候选中，竞争最少、且所需基础已具备的是「以对偶势为证书的 OT 最优性认证」；其核心难点——balance 残差为零只保证流合法而不保证总流最小，故证书必须同时处理原始可行性修复、对偶可行性与最优性缺口，且在隐式图上尚无可计算的全局证书——见第 9 章 §6b。

**阅读指南。**

| 读者 | 从哪里进 |
|---|---|
| 熟悉 GFlowNet、不熟 OT | 第 1 章 → 第 6 章（O01 导读与对照表）→ 第 5 章 |
| 熟悉 OT、不熟 GFlowNet | 第 2 章 → 第 3 章 → 第 5 章 |
| 只想知道该不该做这个方向 | 第 7 章 → 第 8 章 → 第 9 章 |
| 想直接开工 | 第 9 章的决定性实验 + 第 10 章的仓库导览 |

**证据纪律。** 每个定理、数字、实验结果都标注来源（仓库编号 + 原文定理/表号）；凡本报告的推断而非论文结论，以「本报告判断：」开头。发表状态区分主会 / 期刊 / Workshop / 预印本：O08 是 ICML 2026 SPIGM Workshop（arXiv 页面标注，非主会）；O07 的 Workshop 状态只有本仓库沿用的旧调研标签、无一手证据，标为**待核实**；GSBoG 是 ICML 2026 主会；ULOT 是 NeurIPS 2025 主会；DDSBM 是 ICLR 2025 主会。

**本报告的生成方式。** 第 2–7 章的论文级内容取自 `reports/` 下 18 篇独立解读报告的对应章节（核心贡献、前提假设、主线位置、insight），由 `scripts/build_report.py` 拼装；第 1、8、9、10 章与附录为综合撰写。逐篇的记号、推导与实验细节请读原解读报告。


# 第 1 章 问题：内部流为什么不唯一，为什么这是一个最优传输问题

## 1.1 奖励只约束终止边流

GFlowNet 在有限有向图 \(G\)（T02 的设定是 pointed DAG：单源 \(s_0\)、单汇 \(s_f\)，T19/T36 之后放开为允许有环）上定义非负边流 \(F(s\to s')\)，要求两件事：内部状态流守恒 \(\sum_{u}F(u\to s)=\sum_{v}F(s\to v)\)，终止边流等于奖励 \(F(x\to s_f)=R(x)\)。只要两者成立，前向策略 \(P_F(s'\mid s)=F(s\to s')/F(s)\) 的终止分布就是 \(R(x)/Z\)，\(Z=F(s_0)=\sum_xR(x)\)（T02 核心正确性定理；见 `reports/T02_2111.09266.md` §2）。

这两条约束都是线性的（流守恒：T02 Prop. 19 / Eq. (22)；终止边流 = 奖励：T02 v5 Eq. (32)），可行集是一个多面体。奖励只出现在终止边上，所以它固定的是多面体的「边界」；内部边流还有多大自由，取决于图的结构。T02 把这件事写成 Prop. 18 第 3 条：在有限 pointed DAG 上，Markovian flow 由终止流和非终止边上（与图兼容、在父集合上归一化的）后向策略 \(P_B\) 唯一确定。换句话说，**选一个 \(P_B\) 就是选一个内部流**。T02 §2.6 列举了几种选法——对父节点均匀分配、偏好更短路径、或学一个让前向策略更易学的 \(P_B\)——但没有为这个自由度给出明确的优化准则。本报告关心的正是：给这个自由度装上什么目标函数。

## 1.2 有环之后自由度更大，也更危险

DAG 上轨迹长度有上界，流的自由度只来自 \(P_B\)。一旦允许环，T19 证明非负 R-流集合是「非负无环 R-流集合」与「非负环流锥 \(H^1_+(G)\)」的集合和（Prop. 5；分解不唯一，无环分量随所分解的流而变），环上的流不改变终止分布却可以任意放大。T19 Thm. 3 说明对 FM/DB/TB 这类比值型损失，添加 0-flow 能降低损失（结构性质，不是任意优化过程都会无限爆流）；Thm. 4 为一族满足条件的差值型 FM/DB 损失给出稳定性的**充分条件**（不是「只有差值型才稳定」）。总流与期望轨迹长度之间只有不等式 \(\mathbb E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\)（Thm. 2）——这里的 \(F\) 是给定边流，T19 允许采样过程实际诱导的 sampler flow 小于它。

T36 在有限离散图上把这套理论重建为可计算形式：从全边严格正的 \(P_B\) 出发构造流，流定义为期望访问次数乘汇点总流 \(F(s_f)\)，在「满足流匹配的严格正边流」与「全边严格正的 \(P_B\) + 正的 \(F(s_f)\)」之间是双射（Prop. 3.7；O07 随后允许零概率边以容纳最短路解，零流状态上的策略不能由流唯一恢复）。在 T36 的诱导流框架内，不等式成为等式

\[\sum_{s\notin\{s_0,s_f\}}F(s)=Z\cdot\mathbb E[n_\tau]\qquad\text{（T36 Prop. 3.12 / Eq. (10)；}n_\tau\text{ 计内部状态访问次数，重复访问重复计）}.\]

这条等式是全报告的转折点：它将总流这一线性泛函与期望轨迹长度联系起来。「学一个期望轨迹最短的 GFlowNet」由此可以写成 T36 Eq. (11) 的约束优化——目标对流线性，但原变量还含前、后向策略，约束含乘积与对数；要变成多面体上的 LP，须先消去策略、改用边流为变量（O08 Appendix A.1 做的正是这一步），并把严格正域放宽到非负边界。神经参数化加有限权重的 balance + 流正则训练，不等于精确求解这个 LP。

## 1.3 为什么这是最优传输

以边流为变量后，线性目标、线性约束和非负变量构成一个具有图流结构的线性规划：约束是顶点守恒，目标是总流。O01（Peyré 讲义）Prop. 6.23 与 O02（Essid & Solomon）Eq. (1)⇔(3) 给出的是同一件事的两种说法：图上以最短路为代价的 Kantorovich 问题，等价于以边流为变量、以顶点守恒为约束的最小费用流（离散 Beckmann 问题）。缺的是一个**非退化的源侧边缘约束**：单源 \(\delta_{s_0}\) 下耦合集 \(\Gamma(\delta_{s_0},R)\) 是单点集（O01 Remark 3.2），端点耦合没有选择余地——但实现该耦合的**路径**仍有自由度，O07 处理的就是后者。

O08 补的正是这一条：把第一步边流固定为源分布 \(L\)。此时内部状态总流 = 1 + 内部边总流（归一化边缘下第一步贡献常数 1，O08 §3.1 明确扣除），扣除常数后的内部边总流目标等于 \(\min_{\Pi\in\Gamma(L,R)}\sum_{u,x}d_G(u,x)\Pi(u,x)\)（Thm. 3.2），其中耦合定义为 \(\Pi(u,x)=\Pr(s_1=u,\ s_{n_\tau}=x)\)。等价的两个方向：任一可行流的代价不小于其端点耦合的最短路代价（路径长度 ≥ 端点距离）；任一耦合沿最短路分配质量得到同代价的可行流。最优流诱导某个最优耦合，但聚合边流后不保留任意指定耦合的源标签。第 3–5 章按 T19 → T36 → O07 → O08 的顺序把这条链的每一环写清。

## 1.4 本报告判断：什么是新的、什么不是

- **不是新的**：图上最短路代价的 OT ≡ 最小费用流（经典，Beckmann 1952；O02 2018 给出正则化版本）。注意约定差异：O01 Prop. 6.23 用无向图的定向表示、带符号边流与绝对值费用，O02 用有向非负边流、费用可不对称；本报告沿用 O08 的有向非负流与单侧势差约束。
- **新的（本报告判断）**：不是「输出局部策略」本身——经典最小费用边流在正流支撑上归一化后也是局部策略，GSBoG 同样学可执行策略——而是「非无环 GFlowNet 最小流目标 + 固定两端边缘 + 神经策略参数化 + 隐式图上的训练与执行」这一组合。O08 §4.2 的 \(S_{20}\) 排列实验是可执行性示范；该规模没有精确 OT 参照，不能作为「已获得可验证最优计划」的证据。
- **代价**：LP 结构来自边流重参数化，与归一化无关；但 O08 采用已归一化的两端边缘（Assumption 3.1，\(\sum L=\sum R=1\)）以避免处理未知归一化常数。在该论文设定下不再需要学习 \(Z\)；GFlowNet「只需未归一化奖励」这一特点在这条线上未被使用。若推广到未归一化奖励，需另行处理质量匹配与归一化常数。

## 1.5 适用条件对照表与计数约定

| 论文 | 图 | 流的定义 | 正性 / 吸收条件 | 边缘约束 | 结论类型 |
|---|---|---|---|---|---|
| T02 | 有限 pointed DAG | 满足 FM 的边流；Markovian flow | 流非负；总奖励非零 | 终止边流 = \(R\) | 充分条件：FM + 奖励匹配 ⇒ \(P_T=R/Z\)；\(P_B\) 自由度（Prop. 18） |
| T19 | 一般可测空间，允许环 | 有限非负测度；sampler flow ≤ 给定流 | 0-flow 刻画环；吸收性由有限测度推出 | 终止流 = \(R\)（测度等式） | 结构性质：比值损失不稳定（Thm. 3）；差值损失稳定的充分条件（Thm. 4）；\(\mathbb E(\tau)\) 上界（Thm. 2） |
| T36 | 有限离散图，允许环 | 期望访问次数 × \(F(s_f)\)，由 \(P_B\) 诱导 | \(P_B\) 全边严格正；\(\mathbb E[n_\tau]<\infty\) | \(P_B(x\mid s_f)=R(x)/Z\) | 等式 \(\sum_{\text{内部}}F=Z\,\mathbb E[n_\tau]\)（Prop. 3.12）；最小流优化 Eq. (11) |
| O07 | 有限有向图，允许环，单源 | 同 T36 | 放宽为 \(\mathbb E[n_\tau]<\infty\)（Assumption 3.1），允许零概率边 | 奖励匹配 | 充要：期望长度最小 ⇔ 只走最短路（Thm. 3.4） |
| O08 | 有限有向图，允许环，多源 | 边流（消去策略后） | 全对可达；\(\sum L=\sum R=1\) | 第一步边流 = \(L\)；终止边流 = \(R\) | 精确等式：约化目标 = Kantorovich OT（Thm. 3.2）；对偶（Thm. 3.3 单源；App. A.4 多源） |

计数约定（全报告统一）：\(n_\tau\) 计**内部状态**访问次数（不含 \(s_0,s_f\)），重复访问重复计；O08 的 \(\mathbb E|\tau|\) 是运输段长度，等于内部边数；内部状态总流与内部边总流在归一化边缘下相差常数 1。\(R\) 在 T02/T19/T36/O07 中是未归一化奖励，在 O08 中已归一化为概率分布。当前主线针对**无正则、单位边费用**诱导的图最短路 OT；一般加权费用需对应的加权边费用目标，耦合熵、路径空间 KL、边流二次正则各是不同的目标，不能不加区分地并入同一等价链。


# 第 2 章 GFlowNet 基础与训练目标

只保留通向主线所需的内容：流的定义与正确性定理、内部流自由度的原文出处、训练目标族及其在流空间中的含义。每篇的完整解读见 `reports/`。

## 2.1 训练目标族一览

| 目标 | 约束粒度 | 关键量 | 与主线的关系 |
|---|---|---|---|
| FM（T00） | 单状态入流 = 出流 | 边流 \(F(s\to s')\) | 直接约束流守恒；有环时比值型不稳定（T19 Thm. 3） |
| DB（T02） | 单条边 \(F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\) | 状态流 \(F(s)\) | 需显式状态流；T36 公开代码用 DB + 状态流正则训练最小流 GFN（O08 自身的 Eq. (20) 则是带源分布项的正则化 TB） |
| TB（T03） | 整条轨迹 \(Z\prod P_F=R\prod P_B\) | \(\log Z\) | \(\log Z\) 是学习出来的 baseline；O08 的 LP 设定里 \(Z\) 已知 |
| SubTB(\(\lambda\))（T05） | 任意子轨迹，\(\lambda^{n-m}\) 加权 | 插值参数 \(\lambda\) | DB 与 TB 的连续插值；O08 的 \(\lambda\) 是流正则系数，不是这个 \(\lambda\) |

零残差时四者指向同一 reward-matching 解族；差别只在梯度的空间尺度与信用传播距离。**它们都不回答「选哪个内部流」。**

## 2.2 T00 · 原始 GFlowNet

> **一句话**：这篇论文把「按奖励比例采样组合对象」这个问题重写成有向无环图上的流守恒问题，并证明了满足流守恒的策略必然给出 \(\pi(x)=R(x)/Z\)。它解决的是自回归/MaxEnt RL 在「一个对象有多条生成路径」时被路径数 \(n(x)\) 偏置的硬伤。在 GFlowNet × OT 的地图上，它是原点：终止分布被奖励唯一钉死，内部流却留下了一整族解——论文在附录里明确写出了这一族解，但没有给出任何挑选原则，OT 主线正是要给这个自由度装上目标函数。

| 字段 | 内容 |
|---|---|
| arXiv | [2106.04399](https://arxiv.org/abs/2106.04399) |
| 发表 | **NeurIPS 2021 主会**（论文首页脚注：35th Conference on Neural Information Processing Systems (NeurIPS 2021)）。本地 PDF 为 arXiv v2（2021-11-19） |
| 作者 | Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio |
| 代码 | <https://github.com/bengioe/gflownet>（原文 §1 末与附录 A 首段给出） |
| 本仓库 PDF | `papers/2106.04399.pdf` · 中译 `papers_zh/2106.04399.zh.pdf`（尚未生成） |
| 阅读优先级 | **P0** · 全部后续 GFlowNet 理论（含 OT 结果）都在改写它的流守恒方程；不读它就无法判断哪些约束是本质的、哪些是后来加的 |

#### 2. 核心贡献（按原文编号）

原文 §1 自列了四条贡献：提出基于流网络与局部 flow-matching 条件的 GFlowNet；
证明 flow-matching 条件与生成策略匹配奖励之间的联系，以及离线性质与渐近收敛；
在合成数据上证明「建模整个分布而非单个模式」的价值；
在大规模分子合成域上与 PPO、MCMC 做对比实验。下面按定理编号展开。

**Proposition 1（§2.1，证明在 A.1）——树视角的路径数偏置。** 设 \(\tilde V(s)=\sum_{\vec b\in A^*(s)}R(s+\vec b)\) 是 \(s\) 可达终点奖励之和，策略取 \(\pi(a|s)=\tilde V(s+a)/\sum_b \tilde V(s+b)\)。则 (a) \(\pi(s)=\sum_{\vec a_i:C(\vec a_i)=s}\pi(\vec a_i)\)；(b) \(C\) 双射时 \(\pi(x)=R(x)/\sum_{x}R(x)\)；(c) \(C\) 非单射且有 \(n(x)\) 条不同动作序列到达 \(x\) 时
\[\pi(x)=\frac{n(x)R(x)}{\sum_{x'}n(x')R(x')}.\]
原文对 (c) 的评价是「in combinatorial spaces ... this can become exponentially bad as trajectory lengths increase」，具体后果是大分子仅仅因为路径多就被指数级地过采样（§2.1）。这条命题是整篇论文的动机，也是判断「某个方法是不是真 GFlowNet」的试金石。

**Proposition 2（§2.1）——流守恒推出正比采样。** 在 \(\pi(a|s)=F(s,a)/F(s)\)（Eq. (5)）、\(F(s)=R(s)+\sum_{a\in A(s)}F(s,a)\)（内部节点 \(R(s)=0\)、终点 \(F(x)=R(x)>0\)）、且流守恒 Eq. (4) 成立时：(a) \(\pi(s)=F(s)/F(s_0)\)；(b) \(F(s_0)=\sum_{x\in\mathcal X}R(x)\)；(c) \(\pi(x)=R(x)/\sum_{x'}R(x')\)。

**Eq. (11) → Eq. (12)——对数域 flow matching 目标。** 朴素的平方残差（Eq. (11)）在高维空间里会因为根附近流量指数级大于叶子附近而失效，改成对数域并加平滑常数 \(\epsilon\)：
\[\mathcal L_{\theta,\epsilon}(\tau)=\sum_{s'\in\tau\neq s_0}\Big(\log\big[\epsilon+\!\!\sum_{s,a:T(s,a)=s'}\!\!\exp F^{\log}_\theta(s,a)\big]-\log\big[\epsilon+R(s')+\!\!\sum_{a'\in A(s')}\!\!\exp F^{\log}_\theta(s',a')\big]\Big)^2 .\]
原文对 \(\epsilon\) 的定位说得很直白：它不改变全局最小点，只避免对极小流取对数，并「trades-off how much pressure we put on matching large versus small flows」，实验中取到接近 \(R\) 的最小可能值（§2.2）。

**Proposition 3（§2.2，证明在 A.1）——离线/off-policy 性质。** 若训练轨迹来自与最优 \(\pi\) 同支撑的探索策略 \(P\)，模型族足够丰富（\(\exists\theta:F_\theta=F^*\)），且 loss 在流匹配时取到最小值，则期望损失的全局最优满足 \(F_{\theta^*}=F^*\)、对所有 \(\tau\sim P\) 有 \(\mathcal L_{\theta^*}(\tau)=0\)，且 \(\pi_{\theta^*}(x)=R(x)/Z\)。原文自比为异步动态规划：「converges provided every state is visited infinitely many times asymptotically」（§2.2）。

**附录 A.1 命题 3 证明之后的一段——内部流的非唯一性。** 这是本报告最关心的一句：原文写 "Note that in the general case, an infinite number of solutions exist"，并给出显式反例：两条轨迹 \(s_0\!\to\!s_A\!\to\!s_T\) 与 \(s_0\!\to\!s_B\!\to\!s_T\) 通向同一终点、奖励 \(r\)，则解集是一维族
\[F(s_A)=u,\qquad F(s_B)=r-u,\qquad u\in[0,r].\]
论文只是把它作为「解不唯一，所以证明只能保证终止分布」的技术说明，没有给出任何挑选 \(u\) 的原则。

**Proposition 4（A.2）——与 Q 函数的对应（仅双射情形）。** 取均匀策略 \(\mu(a|s)=1/|A(s)|\)、\(f(x)=\prod_{t=0}^{n}|A(s_t)|\)、\(\hat R(x)=R(x)f(s_{n-1})\)，则 \(Q^\mu(s,a;\hat R)=F(s,a;R)f(s)\)。非单射情形原文只给了一个猜想（A.2 末尾 "Conjecture"），并明确说 "since an infinite number of valid flows exists, it's not clear that such a simple equivalence always exists"。

**与 Soft Q-Learning 的区别（§3）。** 两者目标形式相似，差别在于 GFlowNet 的入流包含**所有父节点**，而 soft Q-learning 只用轨迹上的那个父节点；后果由 Proposition 1 给出：soft Q-learning 得到 \(P(\tau)\propto R(\tau)\)，GFlowNet 得到 \(P(x)\propto R(x)\)。

**Algorithm 1（A.5）——多轮主动学习循环。** 输入初始数据集 \(D_0=\{x_i,y_i\}\)、轮数 \(N\)、逆温度 \(\beta\)；
每轮先在 \(D_{i-1}\) 上拟合 proxy \(M\)，再用 \(r(x)=M(x)^\beta\) 作为奖励训练 \(\pi_\theta\)，
从 \(\pi_\theta\) 采一个 batch \(B\) 交给 oracle \(O\) 评测，把新标注并入数据集。
这个外循环解释了为什么论文如此看重多样性：proxy 的覆盖范围由生成器决定，生成器塌到单模式，proxy 就永远学不到别处。

#### 5. 前提假设与适用边界

结论成立的范围，正向陈述：

1. **有限、无环、确定性**的状态图：动作序列长度有界，环境转移 \(T(s,a)\) 确定（§2）。
2. **奖励只在终止状态**，且严格为正 \(R(x)>0\)；内部节点 \(R(s)=0\)（§2.1）。
3. **所有边流严格为正**，零流边通过限制父状态的可用动作集来排除（§2.1）。
4. **训练策略支撑覆盖最优策略的支撑**（Prop. 3 的前提），实践中用 \(0.95\pi+0.05\text{uniform}\) 近似。
5. **模型族可实现真流**，且优化达到期望损失的全局最优（Prop. 3 的两个前提）。
6. 结论只钉住**终止分布**。内部流、路径分布、轨迹熵都不被 Prop. 2 约束（见 §2 引用的 A.1 一维解族）。

论文自己列的限制（§5 Discussion & Limitations）只有一条：与 TD 方法一样依赖 bootstrapping，可能带来优化困难，并建议未来把这种生成方法与局部优化结合。

假设失效时具体会坏在哪里，逐条对应：

- **有环**：Prop. 2 的归纳证明依赖拓扑序（从 \(s_0\) 出发对父节点归纳），有环则归纳无起点，且轨迹可能不终止。
  T02 §3.3.1 用时间戳增广绕过，T19/T36 才真正重建理论。
- **随机环境**：Prop. 2 假定 \(T(s,a)\) 确定；随机转移下策略无法自由分配边流，T02 附录 C 的 Counterexample 50 给出了显式反例。
- **\(R(x)=0\)**：Eq. (12) 要取 \(\log\)，零奖励终点会破坏正流假设；工程上用 \(R_{\min}\) 裁剪（A.4 取 0.01），代价是改变了目标分布。
- **训练分布支撑不全**：Prop. 3 的结论对未覆盖的边不成立。Fig. 12 的均匀随机策略数据集就是一个部分失效的实例——
  论文说明其中「many points are left out intentionally as a generalization test」，拟合质量明显差于 Fig. 13。

#### 6. 在 GFlowNet × OT 主线中的位置

**前驱。** Buesing et al. (2019) 的 MCTS+value 方法（原文 §3 把它当成主要对照，Prop. 1 就是为反驳它写的）、MaxEnt RL / soft Q-learning（Haarnoja et al. 2017）、离散 MCMC（Grathwohl et al. 2021、Xie et al. 2021 的 MARS）。

**后继。** T02（GFlowNet Foundations）把 §2 的流网络重铸为轨迹测度并引入 \(P_B\) 与 DB；T03（Trajectory Balance）用完整轨迹约束替换局部约束。本文的 Eq. (12) 就是后来所说的 FM loss。

**对「内部流选择 = 最优传输」这条主线的贡献，有三处，全部在原文里。**

1. **它写出了自由度本身。** 附录 A.1 的 \(u\in[0,r]\) 一维解族是 GFlowNet 文献里第一次显式给出「同一终止分布、不同内部流」的构造。OT 主线要做的事，就是在这个解集上加一个目标函数。

2. **它明确宣布这个自由度是开放问题。** A.6 原话：「Note that our loss does not enforce any kind of distribution on flows, and a uniform flow is not necessarily desirable (investigating this could be interesting future work, perhaps some distributions of flows have better generalization properties).」这句话是本仓库整条 OT 主线的合法性来源——不是后人硬造的问题，是原作者留的口子。

3. **它给出了自由度被"随便选中"的经验图像。** Fig. 10 中间图画的是所有通向 \((6,6)\) 的路径上的访问分布，右图是同一组路径上的均匀分布，两者明显不同；原文解释为「some preference towards other corners, presumably due to early bias during learning as well as the position of the other modes」。也就是说：**没有正则时，内部流由优化动力学的偶然性决定。** 这正是 min-flow / OT 原则想要取代的东西。

三套设定的差异摆在一起看最清楚（OT 一列依据 O08 的前提，本报告整理）：

| 维度 | T00（2021） | T02（Foundations） | O08 的 OT 定理设定 |
|---|---|---|---|
| 图 | 有限 DAG，单源，每个终点自成汇 | 有限 pointed DAG，单源 \(s_0\) 单汇 \(s_f\) | 有限有向图，**允许环** |
| 源侧约束 | \(F(s_0)=Z\)，无源分布 | 同左 | 固定源分布 \(L\) over 首步状态集 \(U\) |
| 终端约束 | \(F(x)=R(x)>0\) | \(F(s\to s_f)=R(s)\ge0\) | 固定终止分布 \(R\)，\(\sum L=\sum R=1\) |
| 反向策略 | 无此对象 | \(P_B\) 可在非终止边上自由指定 | 由最优流诱导 |
| 内部流 | 一族解，无选择原则 | 一族解，由 \(P_B\) 参数化 | 由**最小总流**唯一化（一般情形下） |
| 流的语义 | 轨迹质量（每条轨迹至多过一条边一次） | 轨迹测度 | expected visit counts |

**一个必须说清的反向限制。** T00 的设定不满足 2606.06272（O08）的 OT 定理前提：那里要求固定的**源分布** \(L\) over 首步状态集合 \(U\)、允许有环、以及最小总流目标；T00 是单源 \(s_0\)、严格无环、无任何流量正则。所以「T00 的 GFlowNet 在做 OT」是错的。T00 提供的是**问题的存在性**（自由度存在且未被约束），不是答案。

#### 7. 可复用的 insight 与开放问题

1. **T00 的双路径反例恰好是 min-flow 的退化例。** 在 A.1 的例子里，两条路径等长，内部总流 \(2u+2(r-u)=2r\) 与 \(u\) 无关（本报告推导，原文只给出解族没有算总流）。这说明：**最小总流原则在等长路径上完全不做选择**，需要额外的 tie-breaking（熵正则、\(P_B\) 先验或二阶目标）。这可以直接做成一个最小实验：在 diamond DAG 上验证 min-flow LP 的解集维数。

2. **路径数偏置 \(n(x)\) 在 OT 语境下会以另一种形式回来。** Prop. 1(c) 说的是「按轨迹均匀 ⟹ 按 \(n(x)R(x)\) 采样」。在最小流问题里，最优解通常把质量集中在少数最短路上，等价于人为把 \(n(x)\) 压到很小。值得写成一条命题草稿：min-flow 解的支撑集大小与 \(n(x)\) 的关系。

3. **\(\lambda_T=10\) 的终端损失加权是个可迁移的工程结论。** 在 OT 版本里，边缘约束（源分布 + 终止分布）就是终端约束，加权它相当于优先保证 coupling 的边缘正确、其次才优化传输代价。可直接用作 min-flow 正则 TB 的调参起点。

4. **反向生成的数据比前向随机的数据更有效（Fig. 12 vs Fig. 13）。** 这条经验和 OT 里「从目标边缘出发反推 coupling」的直觉一致，也预示了 T02 引入 \(P_B\) 后的 backward sampling 训练法。可复现的检查：在同一 hypergrid 上比较前向均匀采样、后向采样、以及按最短路后向采样三种数据集下的 \(L_1\) 与平均轨迹长度。

5. **对数域 + \(\epsilon\) 平滑对 min-flow 正则是有害还是有利，尚不清楚。** \(\epsilon\) 的作用是压低小流量的梯度权重；而最小总流目标恰恰需要精确控制小流量边（把它们压到 0）。这两者的张力是一个具体的、可实验的开放问题。

6. **Prop. 4 的 Q 函数对应只在树上成立，非单射情形只有猜想。** 如果 OT 主线想借用 RL 的对偶理论（例如把最小流写成某个 Q 的最优化），必须先补上这个缺口，或者绕道 T02 的 \(P_B\) 参数化。

## 2.3 T02 · GFlowNet Foundations

> **一句话**：这篇论文把 GFlowNet 从「一组流守恒方程」重铸为「完整轨迹集合上的测度」，由此得到状态流/边流是可测集的测度而非公设、Markovian flow 的三种等价参数化、以及 detailed balance 条件。对 OT 主线来说它是唯一必读的先修：**Proposition 18 精确刻画了「固定终止奖励后还剩多少自由度」——剩下的恰好是非终止边上的反向策略 \(P_B\)**，而 §2.6 原文就写着「我们可能偏好更短的路径」。OT 论文做的事，就是给这个自由度换一个可优化的目标。

| 字段 | 内容 |
|---|---|
| arXiv | [2111.09266](https://arxiv.org/abs/2111.09266) |
| 发表 | **JMLR 24(210):1−55, 2023（期刊）**，见 <https://jmlr.org/papers/v24/22-0364.html>。**本地 PDF 是 arXiv v5（2026-01-24），共 76 页，页眉自印「24 (2023) 1-76」，是比 JMLR 已发表版更长的更新版**——两版内容不同，见编者注 |
| 作者 | Yoshua Bengio, Salem Lahlou, Tristan Deleu（三人共同一作）, Edward J. Hu, Mo Tiwari, Emmanuel Bengio |
| 代码 | 未公开（纯理论论文，原文未给出代码链接） |
| 本仓库 PDF | `papers/2111.09266.pdf` · 中译 `papers_zh/2111.09266.zh.pdf`（尚未生成） |
| 阅读优先级 | **P0** · 内部流自由度的精确表述在这里，OT 主线的问题本身由这篇论文定义 |

**编号警告（先说，后面反复用到）**：这篇论文至少有三套编号在流通——早期 arXiv 版（T03 引用的 Prop. 3 / Corollary 1 / Prop. 6 / Prop. 10）、JMLR 2023 已发表版（55 页）、以及本地 arXiv v5（76 页）。**本报告全部按本地 v5 PDF 的编号引用**（Prop. 8/10/14/16/18/19/21/23 等），引用到别处时务必同时写出命题内容而不只写编号。对照关系见编者注。

#### 2. 核心贡献（按原文编号）

**Proposition 16（§2.4）——Markovian flow 的三个等价刻画。** 以下三者等价：(1) \(F\) 是 Markovian flow（Def. 15：\(P(s\to s'|\tau)=P(s\to s'|s)=P_F(s'|s)\)）；(2) 存在**唯一**的相容 \(\hat P_F\) 使得所有完整轨迹满足 \(P(\tau)=\prod_{t=1}^{n+1}\hat P_F(s_t|s_{t-1})\)，且 \(\hat P_F=P_F\)；(3) 存在**唯一**的相容 \(\hat P_B\) 使得 \(P(\tau)=\prod_{t=1}^{n+1}\hat P_B(s_{t-1}|s_t)\)，且 \(\hat P_B=P_B\)。

**Corollary 17（§2.4）——采样定理。** 从 \(s_0\) 出发按 \(P_F(\cdot|s)\) 迭代采样直到 \(s_f\)，终止在 \(s\) 的概率就是 \(P_T(s)\)。证明第一句是「the procedure terminates with probability 1, given that \(G\) is acyclic」——**无环性在这里被用掉了**，这是后来非无环理论必须重建的地方。

**Proposition 18（§2.4）——三种参数化，也是本报告的主角。** 给定 pointed DAG \(G\)，一个 Markovian flow 被下列任一组合**完整且唯一**地确定：

1. 总流 \(\hat Z\) + 所有边上的前向转移 \(\hat P_F(s'|s)\)；
2. 总流 \(\hat Z\) + 所有边上的后向转移 \(\hat P_B(s|s')\)；
3. **所有终止边上的终止流 \(\hat F(s\to s_f)\) + 所有非终止边上的后向转移 \(\hat P_B(s|s')\)**。

第 3 条是 OT 主线的入口：奖励只钉住第一项，第二项完全自由。证明是构造性的——由终止流定义 \(\hat Z:=\sum_{s\in Par(s_f)}\hat F(s\to s_f)\)，把 \(\hat P_B\) 延拓到终止边 \(\hat P_B(s|s_f):=\hat F(s\to s_f)/\hat Z\)，就归约到第 2 种情形。

**Proposition 19（§2.5）——flow matching 条件的充要性。** 非负函数 \(\hat F\)（定义在状态与边上）对应一个流，当且仅当 flow matching 条件 Eq. (22) 成立；此时它**唯一**确定一个 Markovian flow
\[F(\tau)=\frac{\prod_{t=1}^{n+1}\hat F(s_{t-1}\to s_t)}{\prod_{t=1}^{n}\hat F(s_t)}\quad\text{(Eq. (23))}.\]
原文在证明后特别点出：Eq. (22) 可以用来**递归地**定义所有状态上的流——给定 \(Z\) 与前向（或后向）转移之一，从 \(s_0\)（或 \(s_f\)）出发沿 DAG 分配即可。「A setting of particular interest, that will be central in Sec. 3, is when we are given all the terminal flows \(F(s\to s_f)\)」。

**Definition 20 + Proposition 21（§2.5）——detailed balance。** \(\hat F\)（状态上）、\(\hat P_F\)、\(\hat P_B\) 联合对应一个流，当且仅当
\[\forall s\to s'\in A,\quad \hat F(s)\hat P_F(s'|s)=\hat F(s')\hat P_B(s|s')\quad\text{(Eq. (26))},\]
且此时 \(\hat P_F,\hat P_B\) 是相容的（Def. 20）。DB 相对 FM 的卖点原文写得很清楚：**不含对后继或前驱的求和**，因此适用于后继数极多或状态空间连续的情形（§2.5 引言段）。

**Proposition 23（§2.7）——等价类与唯一 Markovian 代表元。** 定义 \(F_1\sim F_2\) 当且仅当二者在**所有边流**上相同（Def. 22）。则：等价的两个 Markovian flow 必然相等；且任一流 \(F'\) 的等价类中存在**唯一**的 Markovian flow。Fig. 4 给了显式数值：\(F_1,F_2\) 等价，\(F_3,F_4\) 等价，\(F_2,F_4\) 是 Markovian 而 \(F_1,F_3\) 不是，四者在终止流上完全一致。

这条命题的用途是**降维**：学一个流本来要指定 \(|\mathcal T|\) 个数（关于边数指数级），限制到 Markovian 后只要指定 \(|A|\) 个边流（还需满足 Eq. (22)）。它同时告诉我们：「选内部流」= 「选边流」，因为边流唯一决定 Markovian flow。

**Definition 24–26 + Examples 1–6（§3.2）——GFlowNet 的形式定义与损失分类。** 一个 flow parametrization 是三元组 \((\mathcal O,\Pi,H)\)：\(\Pi\) 把配置映到 \(\Delta(\mathcal T)\)，\(H\) 是从 \(\mathcal F_{Markov}(G,R)\) 到 \(\mathcal O\) 的**单射**，且 \(\Pi(H(F))\) 恰为 \(F\) 诱导的测度。GFlowNet = \((G,R,\mathcal O,\Pi,H)\)（Def. 25）。flow-matching loss 定义为 \(L(o)=0\iff o\in H(\mathcal F_{Markov}(G,R))\)（Def. 26），并按 edge/state/trajectory 可分解性分类。三个具体例子：Example 4 = FM loss（含平滑常数 \(\delta\)，state-decomposable）、Example 5 = DB loss（edge-decomposable）、Example 6 = TB loss（trajectory-decomposable，归功于 Malkin et al. 2022）。

**Eq. (38)–(39)——off-policy 正确性的形式化。** 对 edge-decomposable 的 loss，
\[\min_{o\in\mathcal O}L(o)\iff\min_{o\in\mathcal O}\ \mathbb E_{(s\to s')\sim\pi_T}\big[L(o,s\to s')\big],\]
其中 \(\pi_T\) 是 \(A\) 上**任意全支撑**分布。state/trajectory 可分解情形同理。这就是「GFlowNet 可以任意 off-policy 训练」这句话在原文里的准确形态：全支撑 + 可分解 + 全局最小，缺一不可。

**§4 条件流与自由能。** Def. 27 自由能 \(e^{-\mathcal F(s)}=\sum_{s'\ge s}e^{-E(s')}\)；Def. 28 条件流网络；Def. 29 reward-conditional；Def. 30 state-conditional（锚在 \(s\) 的子图 \(G_s\)，要求 \(F_s(s'\to s_f)=F(s'\to s_f)\)）；Prop. 31 存在性（构造解 \(F_s(\tau):=F(C_\tau)+\frac1n F(U_{s'|s})\)）；**Prop. 32**：\(F_s(s_0|s)=F_s(s)=\sum_{s'\ge s}F(s'\to s_f)=\exp(-\mathcal F(s))\)；Cor. 33：\(P_T(s'|s)=\mathbb 1_{s'\ge s}e^{-E(s')+\mathcal F(s)}\)。

**Fig. 5 的反例值得单独记**：原始流里 \(F(s_2)=4\)，但从 \(s_2\) 可达的终止流之和是 6，差额来自经过 \((s_0,s_1,s_5)\) 的那部分流——\(s_1\) 与 \(s_2\) 之间没有序关系。**结论：普通状态流 \(F(s)\) 不是「\(s\) 下游奖励和」**，要得到边缘化必须换成 state-conditional flow。这是实践中最容易误用的一条。

**Proposition 35 / 36（§4.7）——熵与互信息。** 训练第二个 GFlowNet 匹配熵化奖励 \(R'(s)=-R(s)\log R(s)\)（Def. 34，要求 \(R(s)<1\)），则 \(H[S]=F'(s_0)/F(s_0)+\log F(s_0)\)；条件版本把 \(s_0\) 换成 \(s_0|x\)；互信息由两者相减得到（Eq. (57)）。

**§5 集合与图上的 GFlowNet。** Def. 37 把状态空间取为 \(2^{\mathcal U}\cup\{s_f\}\)，每步加一个元素，要求 \(Z=\sum_{s\in2^{\mathcal U}}R(s)<\infty\)（Eq. (58)）；**Prop. 38**：任一集合 \(s\) 的全部超集的概率质量 \(P_T(\mathcal S(s))=e^{-\mathcal F(s)}/Z=F(s|s)/F(s_0)\)。图被当作「两类元素的集合」处理（§5.2）；§5.3 用「\((i,x_i)\) 对」的集合来做联合分布的边缘化；§5.4 讨论模块化能量分解。

**§6 连续/混合空间。** 把求和换积分即可，真正的困难在于输出端要能同时算密度和采样；给出的路子有：可积归一化常数（高斯）、cluster-ID 混合、自回归/normalizing flow、扩散式多步重采样，以及直接参数化边流 \(F((s^i,s^x)\to(s'^i,s'^x))\)。原文注明 Lahlou et al. (2023) 在评审期间完成了完整的连续理论。

**附录 A（Prop. 39–41）——直接信用分配。** \(\frac{d\log F(s')}{d\log F(s)}=P(s|s')\)（Prop. 39）、\(\frac{d\log F(s')}{d\log F(s\to s')}=P_B(s|s')\)（Prop. 40）；由此构造两个在「流已匹配」极限下无偏的梯度估计器 \(G_1,G_2\) 及其凸组合 \(G=\lambda G_1+(1-\lambda)G_2\)（Prop. 41）。原文自评：「something very close to policy gradient actually provides an asymptotically unbiased gradient」，但只在 on-policy 且流已匹配时成立，否则有偏。

**附录 C（Prop. 47–49 + Counterexample 50）——随机环境。** 把转移拆成「偶状态 \(s\) →奇状态 \((s,a)\) →偶状态 \(s'\)」（Def. 46），\(P_F(s_{t+1}|s_t)=\sum_{a_t}P(s_t\to s_{t+1}|s_t,a_t)\pi(a_t|s_t)\)（Eq. (79)）。**Prop. 49**：随机环境中任何策略都能给出一个 Markovian flow，但**未必**能达到 \(\hat F(s\to s_f)=R(s)\)（反例：环境到某个正奖励状态的转移概率为 0）。**Counterexample 50**：随机环境下 \(P_B\) 不能自由选择。也就是说 §2.6 的「\(P_B\) 自由」是**确定性环境专属**的性质。

**附录 D（Prop. 51–55）。** \(V_{P_T}(s)=\frac{\sum_{s'\ge s}R(s')^2}{\sum_{s'\ge s}R(s')}\)（Prop. 52）；策略改进定理（Prop. 53）与最优策略存在性（Cor. 54）；用匹配 \(R^2\) 的第二个流可得 \(V_{P_T}(s)=F'(s|s)/F(s|s)\)（Prop. 55）。D.1 指出可以通过设计 \(P_B\) 让 GFlowNet 偏好「先构造高期望奖励的部分」。

**附录 E–F。** Def. 56–59 引入中间奖励与 return-augmented 状态；Def. 60–61 outcome-conditioned / distributional GFlowNet；**Prop. 62**：训练完成的 outcome-conditioned GFlowNet 可以在**事后**合成任意 \(R=r\circ f\) 的流：\(F_{r\circ f}(A)=\sum_y r(y)F(A|y)\)，代价是运行时要对 outcome 空间求和；Def. 63–64 Pareto GFlowNet。

#### 5. 前提假设与适用边界

正向陈述适用范围：

1. **有限状态空间**（Def. 1 明写 \(S\) 有限）与 **pointed DAG**（Def. 3）。单源单汇可由一般 DAG 通过加点归约得到，这一步是无损的。
2. **无环**。用到无环性的地方是可点查的：偏序 \(<\) 的定义（Def. 1/3）、Cor. 17 的终止性、Prop. 19 的递归赋值、Prop. 23 的 Markovian 代表元构造。§3.3.1 给了唯一的放宽手段：把状态换成 \((s_t,t)\) 的时间戳增广，从而自动无环。
3. **确定性环境**（正文默认）。附录 C 才处理随机环境，并给出两条负面结果：Prop. 49（未必能达到目标终止流）与 Counterexample 50（\(P_B\) 不能自由选）。
4. **终端奖励非负** \(R:S^f\to\mathbb R^+\)（Eq. (32)）；集合 GFlowNet 额外要求 \(Z<\infty\)（Eq. (58)）；熵估计额外要求 \(R(s)<1\)（Def. 34 / Prop. 35）。
5. **正确性只保证终止分布**：\(o\in H(\mathcal F_{Markov}(G,R))\Rightarrow P_T\propto R\)（Def. 25 之后）。内部流由 Prop. 18(3) 的第二个自由项决定，奖励对它没有任何约束。
6. **全支撑训练分布 + 全局最小 + 可实现性**（Eq. (39) 与 Def. 26 的组合）。Remark 44 是 v5 里说得最直白的一条：在共享参数的摊销设定下，有限容量时 \(L=0\) 可能根本达不到，实际最优是「各条件之间的折中」。
7. **状态流不等于下游奖励和**（Fig. 5 反例）。要做边缘化必须用 state-conditional flow（Def. 30 + Prop. 32）。这条在实现里最容易踩。

#### 6. 在 GFlowNet × OT 主线中的位置

##### 6.1 前驱与后继

**前驱**：T00（2106.04399）。本文 §1 开篇即说是在 Bengio et al. (2021) 之上「provide an in-depth formal foundation and expansion」。FM loss 成了 Example 4，T00 的 Prop. 1（MaxEnt RL 得到 \(P_T\propto n(s)R(s)\)）被写进 §7.2 作为与正则化 RL 的分界线。

**后继/对照**：T03（TB）在本文的参数化框架里是 Example 6；SubTB（T05）是 T03 附录 A.2 的 hub 版本；T19/T36 的非无环理论要重建的正是本文 §2 里被无环性支撑的部分；O07/O08 的最小流与 OT 结果，是给本文 Prop. 18(3) 留下的自由度加目标函数。

##### 6.2 内部流非唯一性：原文命题清单

这是本报告的核心检索任务，按重要性排序，全部标出处。

**(i) Proposition 18（§2.4）第 3 条 —— 自由度的精确参数化。**
> a Markovian flow on \(G\) is completely and uniquely specified by ... 3. the combination of the terminating flows \(\hat F(s\to s_f)\) for all terminating edges and the backwards transition probabilities \(\hat P_B(s|s')\) for all non-terminating edges \(s\to s'\in A^{-f}\)

奖励只指定第一项，第二项完全自由。**「内部流有多少自由度」这个问题在这里被回答为：等于 \(A^{-f}\) 上的 \(P_B\) 的自由度。**

**(ii) §2.6 整节「Backwards Transitions can be Chosen Freely」——原文的自然语言陈述。**
> What this means is that the terminating flows do not specify the flow completely, e.g., because many different paths can land in the same terminating state. The preference over such different ways to achieve the same final outcome is specified by the backwards transition probability \(P_B\) ... For example, we may want to give equal weight to all parents of a node \(s\), **or we may prefer shorter paths, which can be achieved if we keep track in the state \(s\) of the length of the shortest path to the node \(s\)**, or we may let a learner discover a \(P_B\) that makes learning \(P_F\) or \(F\) easier.

加粗部分是 Foundations 里唯一一次出现 "shortest"。**最短路作为内部流选择原则，原作者已经点名了**，只是当作 \(P_B\) 设计的一个例子，没有形式化、没有给目标函数、没有与 OT 建立联系。这是 O07/O08 的直接出发点。

**(iii) Definition 22 + Proposition 23（§2.7）—— 等价类在边流层面。** 两个流等价当且仅当边流处处相同；每个等价类里恰有一个 Markovian flow。推论：**「挑内部流」等价于「挑边流」**，而边流的可行集由 Prop. 19 的线性方程组 Eq. (22) 加终端条件给出——这是一个多面体，正是线性规划能作用的对象。Fig. 4 给出的 \(F_1,\dots,F_4\) 四个流在终止流上完全一致，是同一现象的最小数值例子。

**(iv) Proposition 19（§2.5）—— 可行集是线性的。** flow matching 条件 Eq. (22) 是边流上的线性等式；加上终端约束 \(F(x\to s_f)=R(x)\) 后，可行集是一个多面体（本报告的措辞，原文只说「flow matching conditions」）。这是 OT 论文能把内部流选择写成 LP 的结构原因。

**(v) Eq. (31) 及其讨论（§2.5 末）—— 自由度不是任意的。** \(P_B\) 只能在「每个状态上对父集合归一化」这张流形上自由，一旦要求 Eq. (31) 那种显式解，就等价于要求 flow matching 本身。这条防止把「\(P_B\) 自由」误读成「\(P_B\) 是任意函数」。

**(vi) Counterexample 50（附录 C.3）—— 自由度在随机环境下消失。** 「Whereas with a deterministic environment for the GFlowNet, one can freely choose \(P_B\) for non-terminal edges, it is not so for stochastic environments」。所以整条 OT 主线默认在确定性环境里。

**(vii) Proposition 39 / 40（附录 A）—— 自由度的微分刻画。**
\[\frac{d\log F(s')}{d\log F(s)}=P(s|s'),\qquad \frac{d\log F(s')}{d\log F(s\to s')}=P_B(s|s').\]
在流已匹配的邻域内，这两式说明「在某处扰动流，会沿 \(P_B\) 传播到别处」。如果要给最小流正则算梯度，这是原文里现成的工具。

**(viii) §3.3.1「Introducing Time Stamps to Allow Cycles」—— 有环情形与最短路偏好同时出现的唯一地方。**
> Define the augmented state space \(S'=S\times\mathbb N\) ... With this augmented state space, we automatically avoid cycles. Furthermore, we may design or train the backwards transition probabilities \(P_B(s'_t\mid s'_{t+1}=(s_{t+1},t+1))\) to create a preference for shorter paths towards \(s_{t+1}\), as discussed in Sec. 2.6.

**(ix) Appendix D.1「Preference for High-Reward Early Trajectory」—— 另一个内部流选择原则。** 用 \(V((s_t,a_t))\) 加权 \(P_B\)，让 GFlowNet 偏好「先构造高期望奖励的部分」。这说明 Foundations 已经意识到「\(P_B\) 是一个可以承载偏好的设计接口」，只是给的例子是奖励导向而非几何导向。

##### 6.3 自由度到底有多大（本报告推导）

由 Prop. 18(3)，固定 \(R\) 后剩余自由参数就是每个 \(s'\in S\setminus\{s_0,s_f\}\) 上的 \(P_B(\cdot|s')\)，各贡献 \(|Par(s')|-1\) 个自由度。注意非终止边恰好是那些头结点 \(s'\neq s_f\) 的边，而 \(s_0\) 没有父节点，于是
\[\dim=\sum_{s'\in S\setminus\{s_0,s_f\}}\big(|Par(s')|-1\big)=|A^{-f}|-\big(|S|-2\big).\]
用 T00 附录 A.1 的 diamond 例子检验：\(S=\{s_0,s_A,s_B,s_T,s_f\}\)，\(|S|=5\)，非终止边 4 条，\(\dim=4-5+2=1\)——正好是那一族 \(u\in[0,r]\)。**这个公式给了「内部流自由度」一个可计算的规模指标**，可以直接用来判断某个环境值不值得上 OT 正则：\(\dim=0\)（树）时 OT 无事可做。

##### 6.4 把最小流目标翻译回 Foundations 的语言（本报告推导）

在 DAG 上，每条完整轨迹经过任一条边至多一次，因此
\[\sum_{e\in A}F(e)=\sum_{e\in A}\sum_{\tau\ni e}F(\tau)=\sum_{\tau}F(\tau)\,|\tau|=Z\cdot\mathbb E_{P}\big[|\tau|\big],\]
其中 \(|\tau|\) 是边数。每条完整轨迹恰有一条终止边，所以内部（非终止）边上的总流是
\[\sum_{e\in A^{-f}}F(e)=Z\cdot\big(\mathbb E_P[|\tau|]-1\big)=\sum_{s\in S\setminus\{s_0,s_f\}}F(s),\]
最后一个等号是因为 \((s_0,s_1,\dots,s_n,s_f)\) 恰访问 \(n\) 个内部状态、恰用 \(n\) 条非终止边。

三条推论：

1. **「最小总内部流」= 「最小期望轨迹长度」**（\(Z\) 由 \(R\) 固定，不参与优化）。O08 用的目标在 DAG 特例下就是这个。
2. **按内部边计和按内部状态访问数计，在 DAG 上完全相等**（不只是相差常数）。有环时二者才会分叉，因为流要改成 expected visit counts。
3. 由 Prop. 18(3)，这个目标可以直接写成 \(P_B\) 的函数：先由 \(P_B\) 递推出所有边流（§3.2 的递推），再求和。**于是 OT 的最小流问题在 Foundations 框架里是「在 \(P_B\) 流形上最小化期望轨迹长度」**，而不需要引入任何新对象。这是本报告认为最值得写成命题草稿的一条。

##### 6.5 与 O08 之间还差什么

Foundations **不能**直接推出 O08 的 OT 定理，缺口有三处，必须说清楚：

| 缺口 | Foundations 的设定 | O08 需要的设定 |
|---|---|---|
| 源侧 | 单一 \(s_0\)，\(F(s_0)=Z\)，没有「源分布」这个概念 | 首步状态集 \(U\) 上固定的 \(L(u)\)，\(\sum_u L(u)=\sum_x R(x)=1\) |
| 环 | 严格无环；只提供 §3.3.1 的时间戳增广 | 允许环，流解释为 expected visit counts |
| 目标 | 无任何关于流大小的偏好（自由度完全开放） | 最小化内部总流，取到全局最优 |

第二条尤其要小心：时间戳增广 \((s,t)\) 虽然把有环图变成 DAG，但它把「同一状态的不同访问时刻」拆成不同节点，图上的最短路代价 \(d(u,x)\) 在增广图里要重新定义（本报告判断，原文未讨论）。也就是说 §3.3.1 是一个**可行的但语义会变的**桥，不是 O08 结论的证明路径。

##### 6.6 为什么它仍是 OT 主线的必要先修

一句话：**OT 主线的问题陈述（「在所有 reward-matching 的流里挑一个」）只有在 Prop. 18 + Prop. 23 之后才是良定义的。** 没有 Prop. 23，「内部流」这个对象在轨迹层面就不唯一（Fig. 4 的 \(F_1\) 与 \(F_2\) 边流相同但轨迹流不同）；没有 Prop. 18(3)，就不知道自由度到底寄存在哪个参数上；没有 §2.6，就无法判断「偏好短路径」是后人强加的还是原框架里就有的接口。

#### 7. 可复用的 insight 与开放问题

1. **把 min-flow 写成 \(P_B\) 上的优化问题**（§6.4 第 3 条）。命题草稿：在有限 pointed DAG 与固定 \(R\) 下，\(\min_{P_B}\sum_{e\in A^{-f}}F_{P_B}(e)\) 与 O08 的边流 LP 有相同的最优值；\(F_{P_B}\) 由 §3.2 的逆拓扑序递推给出。可先在 \(4\times4\) hypergrid 上用 `scipy.optimize.linprog` 与直接对 \(P_B\) 做投影梯度两条路径求解并比对。
2. **自由度维数 \(|A^{-f}|-|S|+2\) 作为环境筛选指标**（§6.3）。实验草稿：在若干标准环境（hypergrid、bag、分子片段图）上算这个数，检验它是否预测「不同训练目标之间内部流差异」的大小。树上应当恒为 0，可以作为 sanity check。
3. **用 Prop. 39/40 给最小流正则设计低方差梯度。** Prop. 41 的 \(G_2\) 估计器已经在用 \(P_B(s|s_t)\) 采样父节点；把 \(L\) 换成流量惩罚项就得到一个现成的估计器，但要注意其无偏性只在「流已匹配」邻域成立——加了正则之后最优点不再是 reward matching 的流，这个前提是否还成立是一个真开放问题。
4. **用 Prop. 35/36 的熵估计当作内部流的诊断量。** 训练第二个匹配 \(R'=-R\log R\) 的流即可得到 \(H[S]\)；把同样的技巧用在轨迹层面（对 \(P_B\) 诱导的轨迹分布求熵）可以定量刻画「最小流解有多集中」。原文没有做这个推广，但 Prop. 32 的自由能机制是现成的。
5. **Remark 44（摊销参数化的容量折中）是 conditional OT 的直接警告。** 如果要学一族随源分布 \(L\) 变化的 OT plan，用一个共享网络时 \(L=0\) 在有限容量下不可达，最优解是各条件之间的折中。任何「conditional GFlowNet 学 OT」的实验都必须报告每个条件上的误差分布，而不只是平均值。
6. **Fig. 5 反例在 OT 语境下会变成一个陷阱。** 若有人想用状态流 \(F(s)\) 来读「从 \(s\) 出发的剩余传输代价」，那是错的——\(F(s)\) 不是下游量的边缘化，必须用 state-conditional flow（Prop. 32）。这可以直接写成一条「不要这样做」的实现规范。

## 2.4 T03 · Trajectory Balance

> **一句话**：这篇论文把 GFlowNet 的训练约束从「每个状态」「每条边」提到「整条完整轨迹」，用 \(Z_\theta\prod P_F=R(x)\prod P_B\) 一个等式取代逐层自举，并证明零残差解仍然给出 \(P_T\propto R\)。它换掉的不是正确性而是**信用传播的尺度**：终端奖励一步作用到起点，代价是随机梯度方差和必须跑完整条轨迹。对 OT 主线来说它是工作母机——O08 的神经实验就是在 TB 上加流量正则；同时它给出了内部流自由度被优化器「自行选中」的最清晰经验图像（Fig. 1 右）。

| 字段 | 内容 |
|---|---|
| arXiv | [2201.13259](https://arxiv.org/abs/2201.13259) |
| 发表 | **NeurIPS 2022 主会**（论文首页脚注：36th Conference on Neural Information Processing Systems (NeurIPS 2022)）。本地 PDF 为 arXiv v3（2023-10-04） |
| 作者 | Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio |
| 代码 | 分子域 <https://github.com/GFNOrg/gflownet/tree/trajectory_balance>；hypergrid 与 bit-sequence 各有一个 gist（原文脚注 4、5、6） |
| 本仓库 PDF | `papers/2201.13259.pdf` · 中译 `papers_zh/2201.13259.zh.pdf`（尚未生成） |
| 阅读优先级 | **P0** · 它是 GFlowNet 的默认训练目标，也是所有带正则的 GFlowNet-OT 实现的基线 |

#### 2. 核心贡献（按原文编号）

先记下本文对两个既有目标的复述（§2.2），后面的对比全部基于它们：

- **Flow matching（Eq. (9)–(10)，归于 T00）**：模型 \(F_\theta(s,s')\) 逼近边流，\(P_F(s'|s;\theta)\propto F_\theta(s,s')\)，损失
  \(\mathcal L_{FM}(s)=\big(\log\frac{\sum_{(s''\to s)\in A}F_\theta(s'',s)}{\sum_{(s\to s')\in A}F_\theta(s,s')}\big)^2\)，终止节点另有一项 \(\mathcal L'_{FM}\) 把入流推向 \(R(x)\)；沿 \(\pi_\theta\) 采样的轨迹对 \(t=1..n-1\) 的状态与终点求和做随机梯度（Eq. (10)）。原文注明其正确性依据是 Foundations 的 "Proposition 10"。
- **Detailed balance（Eq. (11)–(12)，归于 T02）**：模型对每个状态输出 \(F_\theta(s)\)、\(P_F(\cdot|s;\theta)\)、\(P_B(\cdot|s;\theta)\) 三类量，损失 \(\mathcal L_{DB}(s,s')=\big(\log\frac{F_\theta(s)P_F(s'|s;\theta)}{F_\theta(s')P_B(s|s';\theta)}\big)^2\)，终止边另有 \(\mathcal L'_{DB}\)；正确性依据是 Foundations 的 "Proposition 6"。原文特别点出：由 \(P_F\) 与 \(F_\theta(s_0)\) 唯一确定的那个 Markovian flow **未必与模型输出的 \(P_B\) 相容**——这正是 DB 损失要消除的不一致。

**Eq. (13)——trajectory balance 约束。** 对完整轨迹 \(\tau=(s_0\to\cdots\to s_n=x)\)，由 Eq. (3)(4)(5) 直接代数变形得
\[Z\prod_{t=1}^{n}P_F(s_t|s_{t-1})=F(x)\prod_{t=1}^{n}P_B(s_{t-1}|s_t),\]
用到的唯一事实是 \(P(s_n=x)=F(x)/Z\)。

**Eq. (14)——TB 损失。** 模型输出 \(P_F(\cdot|s;\theta)\)、\(P_B(\cdot|s;\theta)\) 与**一个全局标量** \(Z_\theta\)：
\[\mathcal L_{TB}(\tau)=\Big(\log\frac{Z_\theta\prod_{t=1}^{n}P_F(s_t|s_{t-1};\theta)}{R(x)\prod_{t=1}^{n}P_B(s_{t-1}|s_t;\theta)}\Big)^2 .\]
沿训练策略 \(\pi_\theta\)（通常是 \(P_F\) 或其升温版本）采样轨迹做随机梯度（Eq. (15)），完整算法见 Algorithm 1。

**Proposition 1（§3，证明在 A.1）——正确性。** 设 \(R\) 在 \(\mathcal X\) 上严格为正。
(a) 若 \(P_F,P_B,Z_\theta\) 来自某个满足 reward matching（Eq. (8)）的 Markovian flow，则对所有完整轨迹 \(\mathcal L_{TB}(\tau)=0\)。
(b) 反之，若对所有完整轨迹 \(\mathcal L_{TB}(\tau)=0\)，则对应的 Markovian flow \(F_\theta\) 满足 Eq. (8)，且 \(P_F(\cdot|\cdot;\theta)\) 按奖励比例采样。
原文随即补充：若 \(\pi_\theta\) 全支撑且 \(\mathbb E_{\tau\sim\pi_\theta}\mathcal L_{TB}(\tau)\) 在所有 \((P_F,P_B)\) 与 \(Z\) 上全局最小，则结论同样成立；\(R>0\) 只是为了避免除零，可用平滑常数放宽（与 T00/T02 的做法一致）。

**§3.1——reward-matching 流的典范选择。** 原文原话：「The constraint (8), in general, does not have a unique solution: if the underlying undirected graph of \(G\) has cycles, there may be multiple Markovian flows whose corresponding action policies sample proportionally to the reward. However, by the uniqueness properties, **for any choice of backward policy \(P_B\), there is a unique flow satisfying (8)**, and thus a unique corresponding forward policy \(P_F\) for states with nonzero flow.（见 Fig. 1）」自然的默认选择是 \(P_B(\cdot|s)\) 在 \(s\) 的所有父节点上均匀，即 \(1/\#\{s'\mid(s'\to s)\in A\}\)。原文给的固定 \(P_B\) 的动机是工程性的：分子域里很难构造一个对分子同构保持不变的父分布模型。

**次要贡献：DB 的首次实证。** §1 明写「As a secondary contribution, we perform the first empirical validation of the detailed balance training objective」。也就是说 T02 提出的 DB 损失在此之前没有实验。

**附录 A.2——两个推广。**
- **Subtrajectory balance（Eq. (19））**：对任意部分轨迹 \(\tau=(s_m\to\cdots\to s_n)\)，\(F(s_m)\prod_{t=m}^{n-1}P_F(s_{t+1}|s_t)=F(s_n)\prod_{t=m}^{n-1}P_B(s_t|s_{t+1})\)。两端都等于「经过这段子轨迹的完整轨迹的流之和」（Eq. (20)）。**DB 是单边特例，TB 是完整轨迹特例**；模型只在部分状态（原文称 "hubs"）上输出状态流即可，DB 对应「所有节点都是 hub」，TB 对应「只有 \(s_0\) 是 hub」。
- **Non-forward trajectories（Eq. (21））**：从终点 \(s_n\) 先后退到分岔点 \(s_1=s_1'\) 再前进到另一终点 \(s'_{n'}\)，则
\[R(s'_{n'})\!\!\prod_{t=1}^{n'-1}\!\!P_B(s'_t|s'_{t+1})\prod_{t=1}^{n-1}P_F(s_{t+1}|s_t)=R(s_n)\!\!\prod_{t=1}^{n-1}\!\!P_B(s_t|s_{t+1})\prod_{t=1}^{n'-1}P_F(s'_{t+1}|s'_t).\]
原文强调 **\(F(s_1)\) 不出现在里面**，所以这个约束可以变成「不需要模型输出任何状态流、连 \(Z\) 都不需要」的训练目标。推导方式是把两条从 \(s_0\) 出发、在 \(s_1\) 之前完全相同的轨迹的 TB 约束相除。

**附录 A.3——与变分方法的关系。** 固定 \(P_B\)、并设 \(\sum_x R(x)=1\)，则 on-policy TB 的梯度（Eq. (22)）与 \(D_{KL}(P_F(\tau)\Vert R(x)P_B(\tau|x))\) 的 Reinforce 梯度（Eq. (24)）**在期望意义下相差一个常数**，理由是 Eq. (25) 的 \(\mathbb E[\nabla_\theta\log P_F(\tau;\theta)]=0\)。进一步给出方差比较：两者方差之差等于
\[-\mathbb E_{(\tau,x)\sim P_F}\Big[\big(\nabla_\theta\log P_F\,\nabla_\theta\log P_F^\top\big)\Big(1+2\log\frac{R(x)P_B(\tau|x)}{P_F(\tau;\theta)}\Big)\Big],\]
当括号项恒正（特别是在 \(P_F(\tau)=R(x)P_B(\tau|x)\) 的解附近）时，TB 估计器方差更低。**注意这是「最优点邻域」的局部结论，不是全局的。**

#### 5. 前提假设与适用边界

1. **有限 DAG**，唯一初始状态 \(s_0\)，终止状态无出边（脚注 1 给出与 T00 约定的转换方法）。
2. **奖励严格为正**（Prop. 1 的前提，用于 Eq. (14) 的除法），可用平滑常数放宽。
3. **表达能力充分 + 全局最小 + 训练策略全支撑**：Prop. 1 之后的那句话把三者一起列出。缺任何一条，结论都只覆盖被访问到的轨迹。
4. **必须跑完整条轨迹**才能计算一次损失（Algorithm 1 第 3–4 行），不能从中间状态起步训练；这是相对 FM/DB 的实打实的限制。
5. **\(Z_\theta\) 是全局标量**（非条件情形）。条件 GFlowNet 需要 \(Z_\theta(x)\)，本文未涉及。
6. **方差论证是局部的**：A.3 的方差比较要求括号项 \(1+2\log\frac{R(x)P_B(\tau|x)}{P_F(\tau;\theta)}\) 为正，原文明说这在解的邻域成立。远离最优点时 TB 未必方差更低。
7. **\(P_B\) 的处理是一个设计选择而非定理**：§3.1 只保证「每个 \(P_B\) 对应唯一流」，不保证均匀 \(P_B\) 是好的选择。Fig. 2 的实验恰恰显示固定均匀 \(P_B\) 在大网格上收敛更慢。

#### 6. 在 GFlowNet × OT 主线中的位置

**前驱**：T00（FM loss、hypergrid 与分子环境、PPO/MCMC 基线全部沿用）、T02（记号、Markovian flow、唯一性性质、DB loss）。

**后继/对照**：SubTB（T05，本文 A.2 的 hub 版本，原文在评审期间已知并引用为 [18]）；GFlowNets and Variational Inference（Malkin et al. 2022，[19]）与 A Variational Perspective（Zimmermann et al.，[31]）把 A.3 的联系做完整；O08 的神经实验以**正则化 TB** 为训练目标。

**对「内部流选择 = 最优传输」这条主线，本文贡献三样东西。**

**(1) 它把自由度显式地交给了 \(P_B\)（§3.1）。** 「for any choice of backward policy \(P_B\), there is a unique flow satisfying (8)」这句话，加上 Fig. 1 的三张图（奖励、固定均匀 \(P_B\) 得到的 \(P_F\)、学习 \(P_B\) 得到的 \(P_F\)），是「同一个终止分布可以由形态完全不同的内部流实现」的最直观图像。OT 主线要做的，就是在这张图上再加一张「最小流 \(P_B\)」。

**(2) 它给出了「不加正则时优化器会挑哪个内部流」的经验答案（§5.1）。** 原文观察：当 \(P_F,P_B\) 联合学习时，模型偏向「先把一个坐标走完再走另一个」的 L 形路径，并给了解释——「a constant distribution over two actions ('continue to the right' and 'terminate') can be modeled with higher precision over a large portion of the grid than the complex position-dependent distribution」。**也就是说，内部流的选择由模型的可表示性与数值精度决定，而不是由任何几何原则决定。** 这是 OT 正则化最直接的动机：不加约束时选出来的流是「网络好拟合」的流，不是「传输代价小」的流。顺带一提，在 hypergrid 上所有单调路径等长，最小流原则在这里恰好是退化的（本报告判断，原文未讨论），所以 L 形偏好并不违反最短路。

**(3) 它提供了不依赖状态流、甚至不依赖 \(Z\) 的约束（A.2 Eq. (21)）。** 终点—终点路径的「后退再前进」不变式只涉及 \(R\)、\(P_F\)、\(P_B\)。在 OT 设定下，源侧与终止侧的边缘都被固定、总质量归一（\(Z=1\)），此时 Eq. (21) 是一族**不引入任何额外网络头**的可训练约束。这一点在本仓库里值得作为实现选项记录。

**与 O08 的接口与差距。** TB 是 O08 神经实验的 base loss，但 TB 本身对内部流没有任何偏好；O08 加的是流量正则项，并报告「更强的 flow regularization 会缩短路径，却可能增加终止分布偏差」。用 TB 的语言看，这个折中是可以预期的：正则项把最优点从「TB 零残差流形」上推开，而 Prop. 1 只对零残差点成立。**TB 的定理不覆盖带正则的目标**，这是必须写清楚的边界。

#### 7. 可复用的 insight 与开放问题

1. **把 min-flow 正则加进 TB 后，Prop. 1 失效的程度可以量化。** 命题草稿：设正则化目标为 \(\mathbb E_\tau[\mathcal L_{TB}(\tau)]+\lambda\cdot\mathbb E_{P_F}[|\tau|]\)（用 T02 报告 §6.4 的恒等式，第二项就是内部总流除以 \(Z\)），给出 \(P_T^\lambda\) 与 \(R/Z\) 的偏差随 \(\lambda\) 的上界。可在小 hypergrid 上先做数值扫描，与 O08 报告的「路径变短、偏差变大」定性一致性对照。
2. **\(\log Z_\theta\) 是 OT 设定里现成的守恒诊断量。** 当源与目标边缘都归一化时理论上 \(Z=1\)，于是 \(|\log Z_\theta|\) 直接给出「总质量是否守恒」的读数。TB 训练里 \(\log Z\) 本来就用更高学习率单独优化，把它当监控指标几乎零成本。
3. **A.3 的方差优势在加正则后是否保留，是一个真开放问题。** 原论文的方差比较依赖 \(1+2\log\frac{R P_B}{P_F}>0\)，而在最小流解处 \(P_F\neq R P_B/Z\)，括号项的符号不再有保证。这可以直接写成一条待验证的命题。
4. **固定均匀 \(P_B\) 与 min-flow \(P_B\) 的收敛速度对比是一个便宜的实验。** Fig. 2 已经显示「固定 \(P_B\) 收敛慢于学习 \(P_B\)」；min-flow 正则本质上也是在给 \(P_B\) 加约束，因此可以预期它同样会拖慢收敛（本报告推论）。在 \(64\times64\) hypergrid 上用三种 \(P_B\)（均匀固定、自由学习、流量正则）跑同一套指标，就能验证这个推论。
5. **比特序列实验的设计范式可以直接搬到 OT 评测。** 通过改变 \(k\) 在**保持目标分布不变**的前提下改变轨迹长度与动作空间——这正是隔离「传输代价」与「分布拟合」两个因素所需要的实验设计。OT 版本可以固定源/目标边缘，只改图的粒度。
6. **Eq. (21) 的终点—终点约束可以与 MCMC 式局部搜索结合。** 原文已经指出这一点（A.2 末尾，引 [32]；并提到 Bayesian structure learning 里用过「一步后退两步前进」的特例）。在 OT 语境里，这相当于在两个目标点之间做质量重分配，是一个天然的 coupling 修正算子。

## 2.5 T05 · SubTB(λ)

> **一句话**：SubTB(λ) 把 GFlowNet 的一致性约束从「单条边」和「整条轨迹」推广到「任意长度的子轨迹」，
> 用一个只依赖长度的几何权重 λ 在 DB 与 TB 之间连续插值，在梯度偏差与方差之间取一个更好的折中。
> 它是 GFlowNet 训练目标谱系的收官之作，也是本仓库里唯一把「状态流 \(F(s)\) 到底有什么用」讲成可测量命题的训练论文。
> 在 GFlowNet × OT 地图上，它站在「内部流的自由度如何被训练目标隐式选择」这一侧，是通往「内部流选择 = 最优传输」的必要台阶。

| 字段 | 内容 |
|---|---|
| arXiv | [2209.12782](https://arxiv.org/abs/2209.12782)（v3, 2023-06-03） |
| 发表 | ICML 2023 **主会**（Proceedings of the 40th ICML, PMLR 202, 2023） |
| 作者 | Kanika Madan, Jarrid Rector-Brooks\*, Maksym Korablyov\*, Emmanuel Bengio, Moksh Jain, Andrei Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin（\* 同等贡献，首页脚注） |
| 代码 | 原文未给出仓库链接；附录 §B 只说「基于 Malkin et al. (2022) 的公开代码」 |
| 本仓库 PDF | `papers/2209.12782.pdf` · 中译 `papers_zh/2209.12782.zh.pdf` |
| 阅读优先级 | P1 —— 结论本身偏工程，但它给出的「状态流 = TB 损失中随机项的学习式期望估计」这一读法，是后面把 \(F(s)\) 当对偶势／价值函数的直接前身 |

#### 2. 核心贡献（按原文编号）

**(C1) SubTB 约束及其充分性（§2.3, Eq. (8)）。**
结论：DB 条件 Eq. (6) 对所有动作成立，**当且仅当**子轨迹平衡条件 Eq. (8) 对所有（不必完整的）轨迹成立。
原文把这条等价性归给 Malkin et al. (2022) 的 §A.2；本文的贡献是把它当训练目标用。
前提：需要状态流函数 \(F(\cdot;\theta)\)，且在终止状态处强制 \(F(x;\theta)=R(x)\)。

**(C2) SubTB(λ) 目标（§2.3, Eq. (9) + Eq. (11)）。**
结论：对采到的完整轨迹的全部 \(\binom{n+1}{2}=O(n^2)\) 条子轨迹（Eq. (10) 定义 \(\tau_{i:j}\)）
取以 \(\lambda^{\,j-i}\) 为权的凸组合作为损失。
前提：\(\lambda>0\) 是超参；\(\lambda=1\) 对应均匀加权。
原文明说「其他加权方案也是可能的，应在未来工作中探索」——这是一个被作者本人标出的开口。

**(C3) DB 与 TB 是两个极限（§2.3, Eq. (11) 后一段）。**
结论：\(\lambda\to 0^+\) 时 Eq. (11) **精确**退化为轨迹上所有转移的平均 DB 损失 \(\mathcal L_{DB}(s_i\to s_{i+1})\)；
\(\lambda\to+\infty\) 时退化为 TB 损失 \(\mathcal L_{TB}(\tau)\)，此时的对应关系是 \(Z_\theta=F(s_0;\theta)\)。
这是「插值」一词的全部严格含义——它是**损失权重**层面的插值，不是估计量层面的凸组合。

**(C4) 计算代价不是 \(O(n^2)\)（§2.3「Computational considerations」）。**
结论：Eq. (11) 关于 \(\log F(s_i;\theta)\)、\(\log P_F(-\mid s_i;\theta)\)、\(\log P_B(-\mid s_i;\theta)\) 的梯度
只需要**一次前向、一次反向**通过神经网络；\(O(n^2)\) 的开销只发生在对已算出的 log-flow 与 logits 做线性运算上。
因此 SubTB 相对 DB/TB 几乎没有额外计算成本——这是它能被直接采用的现实原因。

**(C5) 偏差–方差假说及其经验证据（§2.3「Hypothesized benefits」+ §4.1.1）。**
结论：SubTB(λ) 的小批量梯度方差介于 DB 与 TB 之间，
而它对「大批量 TB 梯度」的估计精度在训练中期**优于小批量 TB 自身**。
这是本文最有信息量的一条，见第 4 节。

**(C6) 状态流泛化带来的加速（§2.3「Faster learning due to generalization of state flows」）。**
结论：标量的 \(\log F(s;\theta)\) 比高维的策略 logits 更容易被高精度拟合并在状态之间泛化，
在「图远离起点后变宽」的环境里尤其重要。
原文补充了一个统计事实：除 §4.1 的 hypergrid（以及最大的那几个 hypergrid）之外，
所有实验域的终止状态数都比训练中见到的状态总数**大好几个数量级**。但 C6 原文用的是 "may come from"，是**假说**，只有间接证据。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：T00（Flow Network based Generative Models，FM 目标与 hypergrid 环境）；
  T02（GFlowNet Foundations，DB 条件与「固定 \(P_B\) 则全局最优唯一」）；
  T03（Trajectory Balance，TB 目标，以及「TB 梯度的期望 = 轨迹分布 KL 的梯度」这一事实，本文 §4.1.1 引其 §A.3）。
- **同期/竞争**：T10（Shen et al., ICML 2023）从另一侧攻同一问题。
  T05 认为「流的自由度是训练动力学问题，用更好的目标解决」；
  T10 认为「流的自由度是归纳偏置问题，用引导分布解决」。
  T10 §4 明确点评 T05：它参数化状态流并假设状态流泛化有益，但**不用学到的状态流做决策**。
  这句点评就是两篇论文的分界线。
  本文脚注 4 还提到同期的 Pan et al. (2023)（local credit / incomplete trajectories），
  以及 Hu et al. (2023) 把不完整轨迹设定用到组合对象的贝叶斯后验推断上。
- **对主线的贡献**：SubTB 把「内部流 \(F(s)\) 是什么」从一个记账变量变成了有操作含义的对象——
  它是 TB 损失中随机项的**学习式条件期望估计**（Eq. (12)(13) 那一段）。
  在 OT 语言里这正是 Kantorovich 对偶势的角色：
  势函数本身不是运输计划，它是把全局配平问题拆成局部可验证条件的**证书**；
  O01 的 Prop. 5.3（离散互补松弛）说的是同一件事的静态版本。
  要理解 O08「固定初始流分布的最小流 GFlowNet，其最优解就是一个 Kantorovich 传输计划」，
  必须先接受「\(F(s)\) 是势/价值函数」这个读法，而 T05 是把这个读法写清楚的第一篇训练论文。
- **另一条隐线**：Eq. (8) 说的是「局部（逐边）一致性 ⟺ 全局（任意子段）一致性」，这与 O01 Prop. 6.23 的机制同构——
  在图测地距离下，所有点对的 Lipschitz 约束可压缩成**只在边上**的约束，因为沿路径求和能恢复任意点对的约束。
  两处都是「局部条件因图的路径结构而自动全局化」。

#### 7. 可复用的 insight 与开放问题

1. **λ 的最优值系统性地落在 1 附近而非端点**（Fig. A.2 + 六个任务的调参结果）。
   这暗示存在一个由「轨迹长度分布 × 图宽度」决定的最优插值点。
   可做的实验：hypergrid 上固定 \(H\) 扫维度 \(d\)，看最优 λ 是否随平均轨迹长度单调变化。
   若是，就能拟合一条 λ 的先验规则，省掉一整维网格搜索。
2. **「小批量 SubTB 梯度比小批量 TB 梯度更像大批量 TB 梯度」是一个可形式化的命题**（Fig. 4 右）。
   定理草稿：在表格化设定下给出
   \(\mathbb E\|g_{\mathrm{SubTB}}^{(k)}(\lambda)-g_{\mathrm{TB}}^{(\infty)}\|^2\) 关于 λ 的偏差-方差分解，
   并证明存在有限的 \(\lambda^{*}>0\) 使其严格小于 \(\lambda\to\infty\) 处的值。原文只给了余弦相似度曲线。
3. **短子轨迹截断几乎不损失性能**（Fig. A.4，\(j-i\le 4\)）。
   这把 SubTB 的适用面从「完整回合」扩展到「只观测到片段」。
   在 OT 侧的对应物是 O01 Prop. 6.23 里「把所有点对的 Lipschitz 约束压缩到边约束」的机制。
   值得写一个严格的对应：SubTB 的截断长度 ↔ 图上局部约束能生成的最大测地范围。
4. **状态流的泛化优势没有被直接测量**（C6 是假说）。
   可做实验：hypergrid 上人为把 \(\log F(s;\theta)\) 冻结为解析真值 \(F_F\)（§A.2 已给出算法），
   对比学习流版本的收敛曲线。若差距小，C6 被证伪，SubTB 的全部收益归于方差。
5. **λ 加权是长度的几何函数，完全没用图结构信息**。
   开放问题：把 \(\lambda^{j-i}\) 换成依赖端点流值差 \(|\log F(s_i)-\log F(s_j)|\) 的权重，
   是否等价于某种自适应的 Bregman 投影步长？
   原文在结论里把「可学习的子轨迹选择与加权策略」列为最有意思的未来方向。
6. **与变分方法的联系未被实证**。原文结论明写「未来工作应当实证评估 SubTB 与
   Malkin et al. (2023)（GFlowNets and variational inference）中方法之间的联系」。
   这是作者本人标记为未完成的线索，也是把 GFlowNet 目标翻译成 KL 投影语言、进而接到 Sinkhorn 交替投影（O01 §9.1）的最短路径。

## 2.6 T10 · 训练诊断

> **一句话**：这篇论文的真正贡献不是三个训练技巧，而是一个诊断——在可枚举的生化基准上，
> GFlowNet 会**系统性地欠拟合目标分布**，长期给低奖励对象分配过高概率，而以往常用的 Spearman 相关性根本看不出这件事。
> 在此基础上它提出「流分布的好坏 = 泛化能力」这一评判标准，并用 guided trajectory balance 让用户可以**指定**流该往哪里走。
> 在 GFlowNet × OT 地图上，它是把「内部流的自由度」从副作用升格为**可设计对象**的那篇论文——
> 而「用一个外生准则去选多解中的哪一个」正是 OT 的题设。

| 字段 | 内容 |
|---|---|
| arXiv | [2305.07170](https://arxiv.org/abs/2305.07170)（v1, 2023-05-11） |
| 发表 | ICML 2023 **主会**（Proceedings of the 40th ICML, PMLR 202, 2023） |
| 作者 | Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani |
| 代码 | <https://github.com/maxwshen/gflownet>（原文附录 §B 明确给出） |
| 本仓库 PDF | `papers/2305.07170.pdf` · 中译 `papers_zh/2305.07170.zh.pdf` |
| 阅读优先级 | P1 —— 方法部分可选读，但 §3（评估口径）与 §5（子结构信用分配）是本仓库主线的必读前提 |

#### 2. 核心贡献（按原文编号）

**(C1) 一套可枚举的评估方案与「欠拟合」诊断（§3, Remark 1）。**
Remark 1 原文——「GFlowNet 训练期间的一个首要实践挑战，是降低采样到低奖励 \(x\) 的概率」。做法是设计
\(|\mathcal X|\) 可枚举的基准，用采样得到的**奖励样本**（而不是计算 \(p_\theta(x)\)）与目标奖励分布做拟合优度检验。
前提：\(\mathcal X\) 必须可枚举，因此不能用于真实规模任务。

**(C2) 指出既有评估指标的口径漏洞（§3）。** 多篇工作用留出集上 \(\log p_\theta(x)\) 与 \(R(x)\) 的 Spearman 相关性
（原文点名 Madan et al., 2022；Nica et al., 2022）；但「当 \(\log p_\theta(x)=cR(x)\) 时，对**任意** \(c>0\) 相关性都是 1.0，
而只有 \(c=1\) 才真正匹配目标分布」。这是纯逻辑论证，不依赖实验。

**(C3) Remark 2：流分布决定泛化，泛化决定分布匹配（§4）。** 原文——「流分布对泛化重要，而泛化对在
\(\mathcal X\) 上匹配目标分布重要。此外，给定数据 \(\{x,R(x)\}\) 后流一般是欠定的」。这把「哪个流更好」从空问题变成可评判的问题。

**(C4) Remark 3 + Prop. 5.2/5.4：TB 与 MaxEnt 都会低估重要子结构（§5）。** 当每个 \(x\) 有多条轨迹时，
TB 与最大熵 GFlowNet 分配流的方式会**不充分地**给「对 \(R(x)\) 最负责的子结构」记功；
这类子结构在奖励函数是组合式（compositional）时存在。两个可量化的命题见第 3 节。

**(C5) Guided trajectory balance（§6, Theorem 6.1）。** 给定任意（可非马尔可夫、可随训练变化的）引导分布 \(p(\tau_{\to x})\)，
若约束 (3) 对所有 \(x\) 与所有 \(\tau_{\to x}\) 成立，则 \(P_F\) 以 \(R(x)/Z\) 采样 \(x\)——与既有 GFlowNet 目标有**相同的渐近保证**。
前提：一般引导分布非马尔可夫，标准 GFlowNet 无法处处满足，需两阶段优化（见第 3 节）。

**(C6) 三个可组合的改动 + 一个 MDP 设计结论（§3, §4, §6, §7）。** prioritized replay training (PRT)、
relative edge flow parametrization (SSR)、substructure guidance (Sub)；以及一个反直觉的经验结论：
**自回归 MDP 未必最好**——SIX6 上只有「prepend/append MDP + Sub + PRT + SSR」能匹配目标均值。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：T00（把 MDP 看作流网络）；T02（DB 条件，以及被本文当定理引用的 Corollary 1
  「固定马尔可夫 \(P_B\) 下 TB 全局最优唯一」——这是两阶段 GTB 成立的支点）；T03（TB 目标，Eq. (1)）；
  Zhang et al. (2022)（MaxEnt GFlowNet：\(P_B\) 均匀时 TB 的唯一全局最优是流熵
  \(H[F]=\mathbb E_{\tau\sim P_F}\sum_{t=0}^{n-1}H[P_F(\cdot\mid s_t)]\) 最大的马尔可夫流）。
- **同期/对照**：T05（Madan et al., 2023）。本文 §4 直接点评它：「参数化状态流以从部分回合学习，
  并假设状态流泛化有益，但其方法**不用学到的状态流做决策**」——SSR 正是补上这一环。
  两篇都承认「流是欠定的」，分歧在于欠定性是**害**（T05：造成方差/偏差问题）还是**资源**（T10：可被设计）。
- **对主线的贡献，三条**：
  1. **它把「选哪个流」变成显式的变分问题**。TB 的解集是一个多面体（所有与 \(R\) 相容的马尔可夫流），
     MaxEnt 用熵最大化在其中选一点，Sub 用引导似然选另一点。这在结构上就是 OT 的题设：
     可行集是耦合多面体 \(\mathcal U(a,b)\)（O01 Def. 3.1），用一个外生的代价/正则项从中挑一个点。
     MaxEnt ↔ 熵正则 OT 的 \(\varepsilon\to\infty\) 极限（O01 Prop. 8.10：\(P_\varepsilon\to a\otimes b\)，最扩散的耦合）；
     Sub ↔ 换一个参考测度做 KL 投影（O01 Prop. 8.9 说明参考测度在余量固定时只差常数，
     但在**不固定**的情形下会真正改变解——这正是 T10 想要的效果）。
  2. **两阶段 GTB 在算法形态上就是交替投影**：先把 \(P_B\) 投到引导分布上（Eq. (4) 是对数比平方，即经验 KL 的代理），
     再把 \(P_F\) 投到「固定 \(P_B\) 的 TB 约束」上。和 Sinkhorn 的两个 KL 投影（O01 §9.1 Eq. (9.2)、Prop. 9.5）是同一骨架，
     区别是 T10 的两个约束集不对称、且第一个随 \(X\) 变化。
  3. **子结构信用分配 ↔ 代价函数设计**。Prop. 5.2 的 \(\Theta(1/(n-k))\) 说的是：
     均匀先验下流按「路径计数」而非「语义重要性」分配。在 OT 里这对应「用图的路径数当代价」。
     O08 把代价固定为**图诱导的最短路代价**（几何而非计数），T10 的引导分布则是**学出来的**代价——同一空位的两种填法。
- **后继**：O08（Your GFlowNet Secretly Learns an Optimal Transport Plan）与 O07（Learning Shortest Paths with GFlowNets）
  把「选哪个流」推到底：加上「最小化总流量」这一外生准则后，最优解不再任意，而正好是 Kantorovich 传输计划。
  T10 是这条线上「先意识到有得选」的那一步。

#### 7. 可复用的 insight 与开放问题

1. **「Spearman 对 \(\log p_\theta=cR\) 的任意 \(c\) 都是 1」这条批评可以直接量化**。
   可做实验：在可枚举基准上同时报告 Spearman、AD、均值误差与真正的 \(\mathrm{KL}(p_\theta\|p^*)\) 或 TV 距离，看四者何时分歧。
   本文只做了 AD-vs-均值误差的相关性（\(R^2=0.87\)），没有把 \(p_\theta\) 本身算出来——
   而在 \(|\mathcal X|=65{,}536\) 的 SIX6 上这**是可算的**（虽然要对 \(2^{n-1}\) 条轨迹做动态规划）。这是本文留下的最大且最容易补的空白。
2. **Pólya 罐子模型（Thm. C.7）给出 TB 训练的一个可证伪预测**：\(s^*\) 吸收的流比例服从参数已知的 Beta-二项分布。
   可做实验：小规模表格化环境里直接量这条分布，检验神经参数化下偏离多大。
   若偏离很大，说明「富者愈富」在函数逼近下会被泛化抵消——那 Sub 的动机就弱化了。
3. **\(\alpha=1\) 普遍最优（§B）是一个被埋没的发现**：最好的做法是**完全丢掉** \(P_B\)、只回归引导似然，
   即与其学一个马尔可夫反向策略，不如直接指定一个非马尔可夫先验。
   对照 O01 §14.3：Schrödinger 桥的端点约简（Prop. 14.11）恰恰说明，一旦固定端点耦合，
   最优路径律就是**参考桥的混合**——参考动力学可以是任意先验，不必马尔可夫可表示。
   值得写一个严格对应：GTB 的引导分布 ↔ Schrödinger 问题的参考路径律 \(\mathcal R^\varepsilon\)。
4. **PRT 在 Bag 上反而变慢（13820 → 20575）没有被解释**。Bag 的奖励是离散三值（0.01 / 10 / 30）且有随机性（75%/25%）。
   猜想：奖励分位数在离散奖励下退化，top-\(\beta\) 分位采样等价于均匀采样某个大集合。
   可做实验：把 PRT 的分位阈值换成基于奖励值的阈值，看 Bag 是否恢复（原文取 top 10% 占 batch 的 50%，§B）。
5. **SSR「表达力不更强、效率更低」却有效，说明这是纯归纳偏置收益**。
   开放问题：SSR 的归纳偏置能否写成一个显式正则项加到 SA 参数化上？原文把它与「最优传输正则化」并置但没给形式。
   若能写成「相似状态的动作策略相似」的显式惩罚，就直接落在 O01 §12.5（度量学习与逆 OT）的框架里。
6. **本文的开放问题原话（§1 末、§8 末）**：「如何诱导 GFlowNet 学到更优的流，从而提升其求解未归一化密度估计问题的能力」，
   以及「如何最好地学到有利的流分布仍是开放问题」。
   这在 OT 语言里有现成答案的雏形：给流空间加一个代价泛函，让「最优」变成良定的。
   O08 选了最短路代价，O07 选了期望轨迹长度。T10 没走到这一步，但它把问题问对了。


# 第 3 章 从 DAG 到有环：非无环 GFlowNet 理论

两篇论文构成 O07/O08 的全部理论前置。T19 在一般可测空间上建立有环流理论并诊断 flow explosion；T36 在有限离散图上重建为可计算形式，给出 \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) 与「最小总流」约束优化。本章包含两篇解读报告的完整正文（记号、贡献、推导、实验、假设、位置、insight）。

## 3.1 T19 · A Theory of Non-Acyclic Generative Flow Networks（AAAI 2024）

> **一句话**：这篇论文把 GFlowNet 的流理论从 DAG 搬到一般可测空间（含连续、含环），用「0-flow」统一刻画环和一切「永不终止的质量」，证明 FM/DB/TB 这类比值型损失在有环图上会把流无限堆进环里（Theorem 3），并给出一族差值型稳定损失（Theorem 4）。在 GFlowNet × OT 地图上，它第一次把「R-流集合 = 一个无环流 + 环空间」（Theorem 5 / Prop. 5）和「期望轨迹长度 ≤ 总流 / 总奖励」（Theorem 2、Corollary 1）写成定理——这两条正是 O07/O08 把「最小总流」当目标函数的前提。

| 字段 | 内容 |
|---|---|
| arXiv | [2312.15246](https://arxiv.org/abs/2312.15246)（v1，2023-12-23；本仓库 PDF 即此版，含附录 A–C） |
| 发表 | AAAI 2024 主会（Proceedings of the AAAI Conference on Artificial Intelligence, 38(10): 11124–11131）；8 页会议版不含附录，附录级结果只在 arXiv 版 |
| 作者 | Leo Maxime Brunswic, Yinchuan Li（通讯）, Yushun Xu, Shangling Jui, Lizhuang Ma（arXiv 版署名；AAAI 版另列 Yiheng Feng，见编者注） |
| 代码 | 未公开（原文未给链接；附录 C.1 只说明用 TensorFlow 2.4.4 实现） |
| 本仓库 PDF | `papers/2312.15246.pdf` · 中译 `papers_zh/2312.15246.zh.pdf` |
| 阅读优先级 | P0：非无环 GFlowNet 的第一篇理论，T36 的全部修正和 O07/O08 的「最小流」目标都以它的记号与定理为起点 |

#### 1. 问题设定与记号

**图上的记号（Sec. 2）。** 有向图 \(G\)，状态集 \(S=S^*\cup\{s_0,s_f\}\)，\(s_0\) 无入边、\(s_f\) 无出边，非多重图。edgeflow 是边上的非负赋值 \(F(s\to s')\ge 0\)。Definition 1 分三层：

- flow：满足流匹配约束 Eq. (1)：\(\forall s\in S^*,\ \sum_{s'\to s}F(s'\to s)=\sum_{s\to s'}F(s\to s')\)；
- R-edgeflow：满足奖励约束 Eq. (2)：\(\forall s\in S^*,\ F(s\to s_f)=R(s)\)；
- R-flow：两者同时满足。

前向策略由流诱导，Eq. (3)：\(P(s_{t+1}=s\mid s_t)=F(s_t\to s)/\sum_{s_t\to s'}F(s_t\to s')\)。采样时间是到达 \(s_f\) 前的最后一步，Eq. (8)：\(\tau=\max\{t\mid s_t\ne s_f\}\)，\(s_\tau\) 即被采出的对象。Eq. (4)–(6) 是 FM、DB、TB 的平方对数比损失，期望取在当前策略采出的路径分布上。

**与标准 DAG-GFlowNet 的差异。**

- 允许有向环：轨迹集合无限、\(\tau\) 无上界，\(E(\tau)\) 可以是任意大甚至无穷。
- 第一性对象是流（一个测度）：\(\pi_f,\pi_b\) 由流经 Radon–Nikodym 导数导出（Prop. 1），不存在「先固定 \(P_B\)」这一步，FM 损失里根本没有 \(P_B\)。
- 边缘约束只有奖励约束 Eq. (2)：源端 \(F(s_0\to\cdot)\) 不固定（只有 Cayley 实验人为固定为均匀，Sec. 5.2）。

**可测空间上的推广（Sec. 3.2，Table 1）。**

- edgeflow 是 \(S\times S\) 上的有限非负测度 \(F\)；\(F_{\mathrm{out}}(A):=F(A\to S)\)、\(F_{\mathrm{in}}(A):=F(S\to A)\)；奖励是 \(S\) 上的测度 \(R\)。
- 两条约束变成测度等式：\(F(\cdot\to s_f)=R\) 与 \(\mathbf 1_{S^*}F_{\mathrm{in}}=\mathbf 1_{S^*}F_{\mathrm{out}}\)。
- 「边」推广为支配测度 \(\mu\)：\(F\ll\mu\)，\(\mu(s_f\to S^*)=\mu(S\to s_0)=0\)、\(\mu(s_f\to s_f)=1\)。
- 前向核 \(\pi_f^F(\cdot\to A)=dF(\cdot\to A)/dF(\cdot\to S)\)（Eq. (23)），\(F=F_{\mathrm{out}}\otimes\pi_f=F_{\mathrm{in}}\otimes\pi_b\)（Prop. 1）。
- 轨迹流 \(F^\otimes\) 是 \(F_{\mathrm{out}}(s_0)\) 乘以从 \(s_0\) 反复施加 \(\pi_f\) 直到 \(s_f\) 的路径分布。
- Table 1 把 Bengio et al.（DAG）、Lahlou et al.（拓扑空间）、本文（可测空间）三套记号逐行对照，最后一行是路径长度：DAG 上 \(\tau\le\#S\)，Lahlou 框架下 \(\tau\le\tau_{\max}\)，本文只有 \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\)。

**0-flow、子流、稳定性（Sec. 3.1）。** 对环 \(\gamma=(s_1\to\cdots\to s_\ell\to s_1)\)，指示流 \(\mathbf 1_\gamma\) 守恒且终止流为零。Definition 2：0-flow 是 \(R=0\) 的 R-flow。\(F_0\) 是 \(F\) 的子流指 \(F_0\le F\)（逐边）。Definition 3：损失 \(L\) 稳定，指对任意 0-flow \(F_0\) 有 \(L(F_1+F_0,\dots,F_p+F_0)\ge L(F_1,\dots,F_p)\)。本文的 \(E(\tau)\) 与 T36 的 \(\mathbb E[n_\tau]\) 是同一量。

#### 2. 核心贡献（按原文编号）

**C1. 流困环机制与稳定性定义（Sec. 3.1，Definition 3，Lemma 1）。**
结论：在有环图上，FM/DB/TB 的梯度下降收敛到沿环无穷大的 edgeflow，\(E(\tau)\to+\infty\)。
前提：损失只看入流/出流的比值。Lemma 1 给稳定的充分条件：对每个同时是各 \(F_i\) 子流的 0-flow \(F_0\)，方向导数 \(\partial_{F_0}L\ge 0\)。

**C2. 稳定正则化极限无环（Theorem 1，Eq. (7)）。**
结论：\(L_\alpha=L+\alpha\mathcal R\)，取 \(\alpha_n\to0^+\)，\(L_{\alpha_n}\) 的极小化 R-edgeflow 序列若收敛到一个 flow，则极限是无环 R-flow。
前提：\(L\) 稳定、\(\alpha>0\)、\(\partial_{F_0}\mathcal R>0\) 对一切 0-子流成立。附录版（Appendix B）把条件改写为「\(S\) 至多可数 + \(L_{\mathrm{FM}}\) 是 strong FM loss + \(\partial_{F_0}L_{\mathrm{FM}}\ge0\)」并去掉了收敛假设。

**C3. 可测空间框架与采样定理（Sec. 3.2，Theorem 2，Eq. (9)）。**
结论：对 \(R\ne0\) 的 R-flow，\(\tau\) 几乎必然有限，\(s_\tau\sim R/R(S^*)\)，且 \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\)。
前提：\(F\) 是有限测度。这是 Bengio 采样定理去掉无环性后的测度版本，并附带总流对期望长度的上界。

**C4. 无环的 0-flow 存在（Sec. 3.2「Acyclic 0-Flows」）。**
单位圆上无理角旋转 \(\pi_f(z)=e^{2i\pi\theta}z\) 永不闭合却永不终止，是一个不含任何环的 0-flow。所以在连续空间「控制环」不够，必须控制全部 0-flow；这是本文用 0-flow 而不是环作为稳定性基元的理由。

**C5. 三类损失的统一表示与不稳定定理（Sec. 4.1，Eq. (10)–(16)，Theorem 3）。**
结论：\(\mathrm{div}_{g,\nu}(\alpha,\beta)=\int g(d\alpha/d\beta)\,d\nu\)；FM = \(\mathrm{div}_{g,\nu_{\mathrm{state}}}(F_{\mathrm{in}},F_{\mathrm{out}})\)、DB = \(\mathrm{div}_{g,\nu_{\mathrm{edge}}}(F_f,F_b)\)、TB = \(\mathrm{div}_{g,\nu_{\mathrm{path}}}(F_f^\otimes,F_b^\otimes)\)，\(g=\log^2\) 还原 Eq. (4)–(6)。Theorem 3：若 \(\mathrm{div}_f\) 是 proper f-divergence，则 \(L_{\mathrm{FM},f},L_{\mathrm{DB},f}\) 不稳定，唯一例外是 total variation；\(L_{\mathrm{FM},g,\nu},L_{\mathrm{DB},g,\nu},L_{\mathrm{TB},g,\nu}\) 均不稳定。
前提：\(f(x)\ge0\) 且仅在 \(x=1\) 取零（这排除了 KL），加 Appendix B.1 的正则性条件。

**C6. 稳定损失族（Sec. 4.2，Eq. (17)–(18)，Theorem 4，Example 1–3）。**
结论：用差值型 \(\Delta_{f,g,\nu}(\alpha,\beta)=\int f(\alpha-\beta)\,g(\alpha,\beta)\,d\nu\) 替换比值型，Theorem 4 的五个条件下 \(L_{\mathrm{FM},\Delta},L_{\mathrm{DB},\Delta}\) 对 R-edgeflow 稳定。Example 1 = 稳定 FM（Eq. (19)），Example 2 = 稳定 DB（Eq. (20)），Example 3 = 稳定 CFlowNet（Eq. (21)）。
TB 没有稳定版本，原文明言原因是 \(F\mapsto F^\otimes\) 非线性。

**C7. sampler flow 与总流恒等式（Appendix A.3，Definition 5，Prop. 2，Corollary 1）。**
采样实际实现的流 \(\bar F\) 是期望访问次数流；\(\bar F\le F\)，且 \(E(\tau)=\bar F(S^*\to S^*)/R(S^*)\le F(S^*\to S^*)/R(S^*)\)。

**C8. 图上流空间的结构（Appendix A.4–A.5，Prop. 3–6，Theorem 5，Corollary 2–3，Lemma 2）。**
连通图上 \(\mathcal F_R=F+H^1(G)\)（R-流的仿射空间由环空间张成），\(\mathcal F_R^+=\mathcal F^+_{R,\mathrm{acyclic}}+H^1_+(G)\)（每个非负 R-流 = 无环 R-流 + 有向环的非负组合），所有 0-flow 都是环型。任何 edgeflow 可分解为 \(F=F_0+F_{\min}\)（极大 0-子流 + 极小流），分解不唯一（Example 7.1）。

**C9. 实验（Sec. 5）。** Hypergrid（±步长转移）、Cayley 图 \(S_{20}\)、连续 Point-Robot-Sparse，验证不稳定损失下流量爆炸、稳定损失下有界。

#### 3. 方法与理论推导要点

##### 3.1 expected visit flow：sampler flow \(\bar F\) 与「僵尸流」

Definition 5（Eq. (24)）：对 \(E(\tau)<+\infty\) 的 edgeflow \(F\)，令
\[
\bar F_{\mathrm{out}}(A):=F_{\mathrm{out}}(s_0)\times E\big(\mathrm{card}\{t\in\mathbb N\mid s_f\ne s_t\in A\}\big),\qquad \bar F:=\bar F_{\mathrm{out}}\otimes\pi_f .
\]
即：把前向链从 \(s_0\) 跑到吸收为止，数它落在 \(A\) 里的期望次数，再乘源流。\(\bar F=F\) 时称 \(F\) 被 exactly sampled。Prop. 2：\(\bar F\) 是 flow 当且仅当 \(E(\tau)<+\infty\)；若 \(F\) 是 R-flow，则 \(\bar F\) 也是 R-flow 且 \(\bar F\le F\)。

为什么 \(\bar F\le F\)、差在哪里。Lemma 3 给出 R-edgeflow 是 flow 的等价刻画
\[
F_{\mathrm{out}}\pi_f-F_{\mathrm{out}}=R(S^*)\,\delta_{s_f}-F_{\mathrm{out}}(s_0)\,\delta_{s_0}.
\]
反复用它做望远镜和（Appendix B，Prop. 2 的证明）：
\[
\mathbf 1_{S^*}\bar F_{\mathrm{out}}=\mathbf 1_{S^*}\sum_{k\ge0}\big(F_{\mathrm{out}}\pi_f^k-F_{\mathrm{out}}\pi_f^{k+1}\big)=\mathbf 1_{S^*}\Big(F_{\mathrm{out}}-\lim_{k}F_{\mathrm{out}}\pi_f^k\Big).
\]
差项 \(\lim_kF_{\mathrm{out}}\pi_f^k\) 在 Prop. 8 的证明里被写成 \(\nu+\beta\delta_{s_f}\)，其中 \(\nu\pi_f=\nu\)、\(\nu\otimes\pi_f(S^*,s_f)=0\)：一块被 \(\pi_f\) 保持不变、永远漏不到 \(s_f\) 的质量。\(F-\bar F\) 就是这块「守恒但采不到」的 0-flow。

什么时候 \(\bar F<F\)（编者推断，依据上式）：

- 有限图上，\(F\) 在所有边严格正且每个状态能到 \(s_f\) ⇒ 前向链是吸收链 ⇒ \(\nu=0\) ⇒ \(\bar F=F\)。
- 某个环上有流、但从 \(s_0\) 可达集进入它的边流为零 ⇒ 前向链永不进入 ⇒ 该环流全部是僵尸流。这违反 T36 Prop. 3.7 的「\(F>0\) 于全部边」。
- 连续空间的无理旋转（C4）：质量永远游荡，\(\nu\ne0\)。

这正是 T36 用「正性 + Assumption 3.1」换取「流 = 期望访问次数」唯一性的原因：T36 的世界里不存在僵尸流。

##### 3.2 吸收条件与总流上界（Theorem 2 的证明链：Lemma 3 → Prop. 7 → Prop. 8 → Corollary 1）

- Prop. 7（\(\tau\) 几乎必然有限）：由 Lemma 3，\(P(\tau>k)=\delta_{s_0}\pi_f^k(S^*)=\big[F_{\mathrm{out}}\pi_f^k(S^*)-F_{\mathrm{out}}\pi_f^{k+1}(S^*)\big]/F(s_0)\)。序列 \(F_{\mathrm{out}}\pi_f^k(S^*)\) 非增非负故收敛，相邻差趋零，所以 \(P(\tau>k)\to0\)。吸收性不是假设出来的，而是从「\(F\) 是有限测度且 \(R\ne0\)」推出来的：有限总流不允许质量在 \(S^*\) 里以正概率永远绕下去。
- Prop. 8（\(s_\tau\sim R/R(S^*)\)）：把 \(P(s_\tau\in A)=\big(\sum_k\delta_{s_0}\pi_f^k\big)\otimes\pi_f(A\times\{s_f\})\) 望远镜求和，得 \(\frac{1}{R(S^*)}\big[F(A\times\{s_f\})-\nu\otimes\pi_f(A\times\{s_f\})\big]=R(A)/R(S^*)\)，第二项因 \(\nu\) 不漏到 \(s_f\) 而为零。
- Corollary 1：每一步恰好落在 \(S^*\) 的某个状态，所以 \(E(\tau)=\bar F(S^*\to S^*)/R(S^*)\)，再由 \(\bar F\le F\) 得 Theorem 2 的不等式。
- T36 Prop. 3.12 后来指出：在其正性假设下 \(\bar F=F\)，这条不等式本身就是等式。

与 T36 的出发点对照：T19 假设流是有限测度、推出吸收；T36 假设 \(P_B>0\) 与连通、推出流有限。两条路在有限正图上交汇。

##### 3.3 flow explosion：为什么比值型损失一定不稳定

Sec. 3.1 的四状态例子：链 \(s_0\xrightarrow{f_1}A\xrightarrow{f_2}B\xrightarrow{f_3+c}C\xrightarrow{1}s_f\) 加一条回边 \(C\to B\)（流量 \(c\)），参数 \(\theta=(f_1,f_2,f_3,c)\)。满足奖励约束的 flow 恰是 \(c\ge0\)、\(f_1=f_2=f_3=1\)；但 \(L_{\mathrm{FM}}\) 可以让 \(c\to+\infty\)、\(f_i\to0\) 来最小化，\(\partial_cL_{\mathrm{FM}}(F^\theta)<0\)，原文明言 DB、TB 同样如此。

编者验算（数值不出自原文，只用来把机制看清）：在 \(C\) 处 FM 残差是 \(\log^2\frac{1+c}{f_3+c}\)。固定一个错误的 \(f_3=1/2\)，只沿 \(c\) 下降：

| \(c\) | 0 | 1 | 4 | \(\to\infty\) |
|---|---|---|---|---|
| \(\log^2\frac{1+c}{1/2+c}\) | 0.480 | 0.083 | 0.011 | 0 |

损失单调降到 0，\(f_3\ne1\) 的失配从未被修复，而诱导策略的终止概率 \(P(s_f\mid C)=\frac{1}{1+c}\to0\)。

一般机制（Sec. 4.2 开头）：\(t\mapsto\frac{d(F_1+tF_0)}{d(F_2+tF_0)}\) 在 \(t\to\infty\) 时趋于 1。Theorem 3 的证明（Appendix B.1，Lemma 4）把它算成方向导数：令 \(F_2=\alpha F_1\)、\(F_0=\beta F_1\)，
\[
\varphi(t)=\int g\Big(\tfrac{\alpha+t\beta}{1+t\beta}\Big)(1+t\beta)\,dF_1,\qquad \varphi'(0)=\int\psi(\alpha-1)\,dF_0,\quad \psi(x)=g(1+x)-x\,g'(1+x).
\]
\(g\) 凸且在 1 取零 ⇒ \(\psi\le\psi(0)=0\) ⇒ \(\varphi'(0)\le0\)：沿任何 0-子流方向损失不增。等号对所有 \((F_0,F_1,F_2)\) 成立当且仅当 \(g\) 解 \(y+(x-1)y'=0\)，即 \(g\propto|1-x|\)——这就是 total variation 例外的来源：TV 看的是差值，加公共质量不改变差值。对 \(g=\log^2\)（Lemma 5 的情形，\(\psi(x)=-x\,g'(1+x)\)）存在 \(F_2\) 使导数严格负，所以 Eq. (4)–(6) 的三个经典损失都不稳定。

TB 弱一些（Lemma 6）：在 \(\mathbf 1_{S^*\times S^*}\mu\ll F_0\)、\(dF_f/dF_b\) 有界、\(\int g(e^{\xi\,\mathrm{len}(s)})d\nu<\infty\) 等条件下，
\[
\lim_{c\to\infty}L_{\mathrm{TB},g,\nu}\big((F_f+cF_0)^\otimes,(F_b+cF_0)^\otimes\big)=\int g\Big[\tfrac{dF_f}{dF_b}(s_0\to s)\,\tfrac{dF_f}{dF_b}(s'\to s_f)\Big]d\omega(s,s'),
\]
只剩首尾两条边的比值；若实现强制首尾流一致（本文 hypergrid 的 TB 实现即如此），极限为 0。沿环无限加流可以把 TB 压到 0，但这个极限可能比 \(c=0\) 时更大也可能更小，所以只能说「某些情形不稳定」（Sec. 6.2：TB「可能对小流稳定」）。

第二重恶化来自 on-policy：\(\nu_{\mathrm{state}},\nu_{\mathrm{edge}},\nu_{\mathrm{path}}\) 由当前策略采样，流越困在环里，环上「比值趋 1」的伪零残差在损失里权重越大。Sec. 5.1 的描述：\(F(s_t\to s_f)\ll F(s_t\to s')\)，即便奖励很高，终止概率也被环流压得极低，链持续游荡（Figure 1、3）。

##### 3.4 训练损失族：比值型 vs 差值型（Theorem 3、Theorem 4、Theorem 1）

差值型 \(\Delta_{f,g,\nu}\) 的稳定性证明只有两行：加 \(t\gamma\) 后 \(f(\alpha+t\gamma-\beta-t\gamma)=f(\alpha-\beta)\) 与 \(t\) 无关，只剩 \(g(\alpha+t\gamma,\beta+t\gamma)\)，条件 \(\partial_{(1,1)}g\ge0\) 保证它非减（Lemma 8）；\(f\ge0\)、\(f(x)=0\Leftrightarrow x=0\)、\(g\ge1\) 保证 \(\Delta\) 是 distance-like（Lemma 7）。Theorem 4 的五个条件：\(f\ge0,\ g\ge1\)；\(f(x)=0\Leftrightarrow x=0\)；\(f,g\) 连续且分段 \(C^1\)；\(f\) 在 \(\mathbb R^-\) 递减、\(\mathbb R^+\) 递增；\(\partial_{(1,1)}g\ge0\)。附录 B.2 还要求 \(R\ll\nu\) 且 \(dR/d\nu\) 有界。

| 损失 | 形式（出处） | 比较对象 | 尺度 | 稳定性 |
|---|---|---|---|---|
| FM | \(\mathrm{div}_{g,\nu_{\mathrm{state}}}(F_{\mathrm{in}},F_{\mathrm{out}})\)，Eq. (11)；\(g=\log^2\) 即 Eq. (4) | 状态入流 vs 出流 | 比值 | 不稳定（Theorem 3、Prop. 9） |
| DB | \(\mathrm{div}_{g,\nu_{\mathrm{edge}}}(F_f,F_b)\)，Eq. (12)；Eq. (5) | 边的前向流 vs 反向流 | 比值 | 不稳定（Theorem 3、Prop. 10） |
| TB | \(\mathrm{div}_{g,\nu_{\mathrm{path}}}(F_f^\otimes,F_b^\otimes)\)，Eq. (13)；Eq. (6) | 轨迹前向流 vs 反向流 | 比值 | 不稳定（Theorem 3；Lemma 6 只给部分结论） |
| f-divergence FM/DB/TB | Eq. (14)–(16) | 同上 | 比值 | 不稳定，唯一例外 TV（Theorem 3） |
| 稳定 FM | \(\Delta_{f,g,\nu_{\mathrm{state}}}(F_{\mathrm{in}},F_{\mathrm{out}})\)，Eq. (17)；Example 1 = Eq. (19) | 状态入流 vs 出流 | 差值 | 稳定（Theorem 4） |
| 稳定 DB（SDB） | \(\Delta_{f,g,\nu_{\mathrm{edge}}}(F_f,F_b)\)，Eq. (18)；Example 2 = Eq. (20) | 边流差 | 差值 | 稳定（Theorem 4） |
| 稳定 CFlowNet | Eq. (21)，用密度 \(f_i,f_o\) 代替 \(F_{\mathrm{in}},F_{\mathrm{out}}\) | 密度差 | 差值 | 稳定（Example 3，按 Theorem 4） |
| 稳定 TB | — | — | — | 原文未给出 |

Example 1（Eq. (19)）：\(f(x)=\log(1+\epsilon|x|^\alpha)\)、\(g(x,y)=(1+\eta(x+y))^\beta\)，
\[
L=E\sum_{t=1}^{\tau}\log\!\big[1+\varepsilon\,|F_{\mathrm{in}}(s_t)-F_{\mathrm{out}}(s_t)|^\alpha\big]\cdot\big(1+\eta(F_{\mathrm{in}}(s_t)+F_{\mathrm{out}}(s_t))\big)^\beta ,
\]
推荐 \((\alpha,\beta,\epsilon,\eta)=(2,1,0.001,1)\)，理由是凸性有利于这个欠定线性问题的收敛。Example 2（Eq. (20)）把差值换成 \(|F^f(s_t\to s_{t+1})-F^b(s_t\to s_{t+1})|\)、权重换成 \((1+\eta F_{\mathrm{out}}(s_t))^\beta\)，其中 \(F^f=F_{\mathrm{out}}(s_t)\pi_f\)、\(F^b=F_{\mathrm{out}}(s_{t+1})\pi_b\)——这就是 T36 记作 SDB 的损失。\(\log(1+\cdot)\) 的作用是压制远未满足守恒的节点对训练的主导，延续 Bengio 原始 FM 的设计意图。

Theorem 1 的证明（Appendix B）：若某个 \(L_\alpha\) 的极小点 \(F\) 不是极小流，用 Prop. 3 分解 \(F=F_0+F_{\min}\)，\(F_0\ne0\)；极小性要求 \(0=\partial_{F_0}L_\alpha=\partial_{F_0}L+\alpha\,\partial_{F_0}\mathcal R\)，但右边 \(\ge0+\alpha\cdot(\text{正数})>0\)，矛盾。于是每个 \(F_n\) 都是极小流，Lemma 2 说至多可数空间上极小流集合对窄收敛封闭，极限仍极小，而极小 ⇒ 无环。Theorem 1 之后的评论直接点名「控制总 edgeflow 大小的正则（如边流矩阵的范数）可能有助于杀死环、缩短期望采样时间」——这是「最小流原理」在文献里的第一次出现。

##### 3.5 与 DAG 理论的对应：图上流空间的几何（Appendix A.5）

把图看成可数离散空间 + 计数测度 \(\mu_E\) 的支配，A.5 得到：

- Prop. 6：广义 R-流构成闭仿射子空间，方向空间是广义 0-流；R-流（非负）是闭凸域。
- Theorem 5（Prop. 4）：连通图、可和 \(R\) 时 \(\mathcal F_R=F+H^1(G)\)，\(\mathcal F_0=H^1(G)\)，其中 \(H^1(G)\) 是拓扑简单环（边可正向或反向穿越）的指示流张成空间在 \(\ell^1(E)\) 中的闭包。原文指出这可由 Kalpazidou (2007) Theorem 3.3.1 推出；非空性的构造是给每条终止边 \(e=s\to s_f\) 选一条路径 \(\gamma_e\)，取 \(F=\sum_eR(e)\mathbf 1_{\gamma_e}\)。
- Prop. 5：\(\mathcal F_R^+=\mathcal F^+_{R,\mathrm{acyclic}}+H^1_+(G)\)，\(H^1_+(G)\) 是有向简单环的非负组合的闭包。Corollary 3：所有 0-flow 都是环型。
- Prop. 3：任何 edgeflow 有极大 0-子流（Zorn 引理），\(F=F_0+F_{\min}\)；极大 0-子流不唯一（Example 7.1 给出有红、蓝两个不同极大 0-子流的例子）。
- Lemma 2：至多可数空间上，无环 edgeflow 与极小 edgeflow 的集合对窄收敛封闭；对 edgeflow「极小 ⇒ 无环」，对 flow「极小 ⇒ exactly sampled」；不可数空间上存在无环、exactly sampled 但不极小的流。

翻成 DAG 语言：DAG 上没有有向环，\(H^1_+(G)=\{0\}\)，所以 \(\mathcal F_R^+=\mathcal F^+_{R,\mathrm{acyclic}}\)，每个 R-流都自动是「无环 / 极小 / exactly sampled」的——但 \(\mathcal F_R^+\) 仍然不是单点（菱形图 \(s_0\to A\to C\)、\(s_0\to B\to C\)、\(C\to s_f\) 上 \(F(s_0\to A)\in[0,R(C)]\) 任取），这份自由度就是 GFlowNet Foundations 里「选 \(P_B\)」的自由度。有环图多出来的是 \(H^1_+(G)\) 这一份可以无限放大的方向；Theorem 1 杀的只是这一份，它不会把 \(\mathcal F_R^+\) 收缩成一个点。Prop. 5 的分解与最小费用流文献里的「流分解定理」（O02 Theorem 1 引 Ahuja–Magnanti–Orlin）是同一件事：路径流 + 环流。

##### 3.6 三套语言的词典（编者整理，逐项标注出处）

| 概念 | DAG-GFlowNet（T00/T02） | T19（本文） | T36 |
|---|---|---|---|
| 第一性对象 | 边流或 \((P_F,Z)\) | 测度 \(F\)（Sec. 3.2） | \(P_B\)，Eq. (7) |
| 流的语义 | 轨迹经过边的未归一化概率 | 抽象测度；采样实现的是 \(\bar F\)（Def. 5） | 期望访问次数 × \(F(s_f)\)（Def. 3.5） |
| 吸收性 | 有限 DAG 自动成立 | 有限测度 + \(R\ne0\) ⇒ Theorem 2 | \(P_B>0\) + Assumption 3.1 ⇒ Lemma 3.4 |
| 期望长度 | \(\le\#S\)（Table 1） | \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\)（Eq. (9)）；\(=\bar F(S^*\to S^*)/R(S^*)\)（Cor. 1） | \(=\sum_sF(s)/F(s_f)\)（Prop. 3.12） |
| 内部流的自由度 | 选 \(P_B\) | \(H^1_+(G)\) 加上无环部分的凸组合（Prop. 5） | 选 \(P_B\)（Prop. 3.7 的双射） |
| 环 | 不存在 | 0-flow（Def. 2）；图上皆环型（Cor. 3） | 沿环加流 = 换一个 \(P_B\) |
| 损失失效模式 | — | 比值型不稳定（Theorem 3） | \(\Delta\log F\) 尺度 + 学 \(P_B\) 时长度发散（Sec. 4） |
| 修正 | — | 差值型稳定损失 + 流正则（Theorem 4、1） | 固定 \(P_B\) 或状态流正则 \(\lambda F_\theta(s)\)（Eq. (12)） |

#### 4. 实验与证据

| 任务 | 环境设定（出处） | 对比对象 | 指标 | 结果 |
|---|---|---|---|---|
| Hypergrid，2D，\(W=20\) | \(S^*=\{1,\dots,W\}^D\)，转移 \(s\to s\pm e_i\)（双向，故有环），初始转移 \(s_0\to a\)，终止转移 \(\forall s:\ s\to s_f\)；流用邻接矩阵 + 边流数组实现（Sec. 5.1，Appendix C.1） | \(L_{\mathrm{FM}}\)、\(L_{\mathrm{TB}}\) vs \(L_{\mathrm{FM},\Delta,f,g}\)；Figure 3 另比较 \(L_{\mathrm{FM},\chi^2,\nu}\)（\(f=(1-x)^2\)）与 \(L_{\mathrm{FM},TV,\nu}\)（\(f=\lvert1-x\rvert\)），稳定版取 \(f(x)=x^2\) | 平均奖励、平均采样路径长度 | 各损失都收敛，但 \(L_{\mathrm{FM}}\) 的期望路径长度爆炸、路径游荡；稳定 FM 的路径「相对直」地奔向奖励区。TB 路径游荡但长度不爆炸，稳定在明显高于稳定损失的值，部分情形长时间训练后追平（Sec. 5.1；Figure 1、3。曲线数值原文未给） |
| Cayley 图，\(S_{20}\) | 状态为置换，边 \(g\to g\sigma_i\)；生成元为一个对换、20-循环及其逆；\(R_1(\sigma)=c\,\mathbf 1_{\sigma\in S_1}\)，\(S_1=\{\sigma\mid\sigma(i)=i,\,i\le k\}\)，\(c=20,\ k=1\)；\(\pi_f(s_0\to\cdot)\) 固定均匀；路径截断 80（Sec. 5.2，Figure 4） | Stable-GFlowNets vs GFlowNets vs Metropolis–Hastings | 平均奖励、平均长度 | 最优策略期望奖励 20、期望长度 5（Figure 4 caption）；不稳定损失下流量不受控增长，稳定损失下流量有界（Sec. 5.2）。曲线数值原文未给 |
| Point-Robot-Sparse（连续，CFlowNets 框架） | 两个目标 \((10,10)\)、\((0,0)\)，起点 \((5,5)\)，最大回合长 12；动作角度范围由 \((0,90^\circ)\) 改为 \((0,360^\circ)\) 以制造环；其余设定同 Li et al. 2023d（Sec. 5.3） | Stable-CFlowNets（Example 3 的损失）vs CFlowNets、DDPG、TD3、SAC、PPO | 5000 次探索中有效且互异的轨迹数；平均奖励 | 稳定损失不损失探索能力，奖励「提高很多」且方差更小（Figure 5）；训练初期轨迹多环，末期收敛到目标（Figure 6）。具体数值原文未给 |

**实现细节（Appendix C）。**

- Hypergrid：TensorFlow 2.4.4，Adam，学习率 0.01；「自训练」每隔一段把训练分布 \(\nu\) 更新为 \(\mu=F_{\mathrm{out}}+\delta\)，\(\delta=0.001\)，每 epoch 200 步梯度；多种奖励分布「对最终结果影响很小」。
- 评估：\(\hat R=\max(F(S\to\cdot)-F(\cdot\to S^*),0)\) 估奖励；\(F_{\mathrm{out}}\) 用幂法近似、截断 \(\lambda W\)，\(\lambda\in\{4,10\}\)；指标 \(E_F\)（采样分布误差）、\(E(\tau)=F_{\mathrm{out}}(S^*)/R(S^*)\)、\(E_R\)（奖励误差）、\(E_I\)（初始流误差）。
- Cayley：\(\rho(\sigma)=(\sigma(1),\dots,\sigma(p))\) 嵌入，MLP 深 3 宽 32、LeakyReLU，学习率 1e-2，路径批 64；初始流 \(F(s_0\to\cdot)=F_{\mathrm{init}}\,U(S^*)\)，\(F_{\mathrm{init}}\) 可训练；采路径时忽略到 \(s_f\) 的转移、跑到截断长度，再按「\(t\le\tau\)」的条件概率加权。
- Appendix C.2 对稳定损失变体的观察：超参难调，「高稳定化正则有用」；\(g\) 取幂 \(<1\) 时小流会「内爆」到 0 但采样性质仍好；\(g(x)=1+|x+y|^\beta\)、\(\beta=0\) 或 \(\beta>1\) 的不内爆版本不爆炸但期望奖励次优。

**证据支撑到哪一步。**

- 实验直接支持的只有定性结论：不稳定损失 ⇒ 流量/路径长度增长；稳定损失 ⇒ 有界。三个环境都没有给出可复述的数字。
- 主指标是平均奖励，它不衡量对 \(R/Z\) 的拟合精度（T36 Sec. 4.1 的批评：学到最高模态就能拿高奖励）。
- 稳定损失在 \(\Delta F\) 尺度上计算误差，T36 Table 1、Figure 2 后来显示这类损失在 \(20^4\) hypergrid 和 \(S_{20}\) 上对奖励分布的拟合明显偏差。
- 原文自己在 Sec. 6.2 列的限制：稳定/不稳定定理的假设「不是最优的」；TB 只被部分研究、没有稳定变体；实验规模相对理论覆盖范围偏小，连续设定下没比较稳定 DB；探索的遍历性完全开放；Cayley 图上初始流未能快速收敛到理论值；只用了最简单的前馈架构。

#### 5. 前提假设与适用边界

- **测度有限性**：所有 \(F,F_{\mathrm{in}},F_{\mathrm{out}},R\) 都是有限非负测度（Sec. 3.2）。Theorem 2 的吸收性正是从有限性推出的；「流量无穷大的环」不在框架内，它只作为训练动力学的极限出现。
- **\(R\ne0\)**：Theorem 2、Prop. 7–8 要求。\(R=0\) 的 0-flow 可以永不终止（C4）。
- **转移约束**：\(F\ll\mu\)，\(\mu(s_f\to S^*)=\mu(S\to s_0)=0\)，\(\mu(s_f\to s_f)=1\)；图的情形取 \(\mu_E\) 为边集计数测度。
- **Theorem 3 的技术条件（Appendix B.1）**：\(f\) 连续分段 \(C^1\)；\(dF_{\mathrm{in}}/dF_{\mathrm{out}}\) 有界且 \(F_{\mathrm{out}}\ll\nu\ll F_{\mathrm{out}}\)（Prop. 9，FM）；\(F_b\ll F_f\) 且 \(dF_b/dF_f\) 有界、\(\pi_f,\pi_b\) 独立参数化（Prop. 10，DB）；TB 还需 \(\mathbf 1_{S^*\times S^*}\mu\ll F_0\) 与 \(\int g(e^{\xi\,\mathrm{len}})d\nu<\infty\)（Lemma 6）。原文称这些是「非最小的简单假设」。
- **Theorem 4 的适用对象**：只对 R-edgeflow 成立，即奖励约束由实现强制满足；另需 \(R\ll\nu\)、\(dR/d\nu\) 有界（Appendix B.2）。稳定性是「加 0-flow 不降损失」，不是「梯度下降一定收敛」，也不是「解唯一」。
- **Theorem 1**：主文版需要极小化序列收敛这一外加假设；附录版换成 \(S\) 至多可数 + strong FM loss（后者原文未给出定义）。结论是「无环」而不是「唯一」（见 3.5）。
- **流空间结构（A.5）**：图连通（每个状态都在某条 \(s_0\to s_f\) 路径上）、\(R\) 可和、图至多可数；Corollary 2–3「一切 0-flow 皆环型」只在可数图上成立，不可数空间有反例（C4）。
- **sampler flow**：Definition 5 要求 \(E(\tau)<\infty\)。
- **适用范围正向表述**：任何有限总流、\(R\ne0\)、允许环的有限或可数图，以及带支配测度的 Polish 空间（脚注 4、6）；稳定 FM/DB 可直接用于这些空间，TB 需另想办法。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：T00（FM 损失、采样定理，本文 Eq. (1)–(4) 的来源）；T02（DB、轨迹流、\(F_{\mathrm{out}}\otimes\pi_f\) 的分解，Table 1 第一列）；T03（TB，Eq. (6)）；Lahlou et al. 2023 连续 GFlowNet（Table 1 第二列；本文自称比它「less involved」，且不假设 \(s_f\) 有限吸收）；Li et al. 2023d CFlowNets（Example 3 与 Sec. 5.3 的基座）；Kalpazidou 2007（环表示理论，Theorem 5 的数学来源）。
- **后继 / 对照**：T36 在有限图上重建并修正本文——把 Theorem 2 的不等式收紧为等式（T36 Prop. 3.12）、指出固定 \(P_B\) 时稳定性与结果无关（T36 Corollary 3.11）、把稳定性重新解释为误差尺度 \(\Delta F\) vs \(\Delta\log F\)、并给出「最小期望长度 ⇔ 最小总流」的约束优化（T36 Eq. (11)）。O07 把「最小总流」推到「最短路」，O08 再推到 Kantorovich 耦合。竞争路线是 Lahlou et al. 2023 的「有限吸收 ⇒ 无环」框架：它直接排除环，本文选择容纳环再用损失控制。
- **对「内部流选择 = 最优传输」主线的贡献**：三块砖。
  - (a) Theorem 5 / Prop. 5 说清了可选的内部流空间是什么：一个由环空间张成的仿射族，非负部分 = 无环流 + 有向环的非负组合。「选内部流」在这里第一次成为一个有明确定义域的优化问题。
  - (b) Theorem 2 / Corollary 1 把「期望轨迹长度」和「总流」绑在一起，O07/O08 的目标函数 \(\sum_eF(e)\) 因此有了行为学意义。
  - (c) Theorem 1 是「最小流原理」的原型：对稳定损失加一个沿 0-flow 严格增的正则并令其系数趋零，极限流无环——O07「最短路被选中」的理论种子就是这里的「剪掉一切绕行」。
- **本文没有做的事**：没有多源设定、没有把总流最小化写成线性规划、没有讨论极限流的唯一性、没有把「学 \(P_B\)」显式化。这些分别由 O02、T36、O07/O08 补上。

#### 7. 可复用的 insight 与开放问题

1. **僵尸流检测**。\(F-\bar F=\nu\otimes\pi_f\) 是被 \(\pi_f\) 保持的 0-flow（3.1）。实验草案：在 T36 的非无环 hypergrid 上，用 T36 Appendix B.4 的基本矩阵 \(N=(I-Q)^{-1}\) 算出真实期望访问次数流，与网络输出的 \(F_\theta\) 逐边比较；\(F_\theta-\bar F\) 的正部集中在哪些环上，就是训练把多余质量堆去的地方。O08 把学到的流解读为传输方案之前，这个差必须为零。
2. **Theorem 1 取 \(\mathcal R=\lVert F\rVert_{\ell^1}\) 时的极限是什么**。\(\partial_{F_0}\lVert F\rVert_1=\lVert F_0\rVert_1>0\) 满足条件，极限无环。定理草稿：有限图上，若 \(L\) 在 \(\mathcal F_R^+\) 上恰好为零，则 \(L_{\alpha}\) 极小点在 \(\alpha\to0^+\) 时趋于 \(\arg\min_{\mathcal F_R^+}\lVert F\rVert_1\) 的某个点，即 O02 意义的最小费用流（单位边费用）。缺的一步是 \(L\) 的非零梯度如何与 \(\alpha\mathcal R\) 竞争——这正是 T36 Sec. 4 用 \(\lambda\) 扫描（Figure 5）在实验上看到的偏差。
3. **稳定 ≠ 无偏**。差值型损失对 \(\log F_F\ll\log F_B\) 的区域梯度趋零（T36 Figure 3），系统性低估流。定理草稿：在 Eq. (19) 的 \(\Delta\)-损失下，固定 \(P_B\) 的唯一解仍是全局极小点（损失为零），但有限步梯度下降的稳态偏差随 \(\eta\) 单调增。实验：固定 \(P_B\)，比较 SDB 与 DB 的 \(L_1\) 误差随 \(\eta\) 的曲线。
4. **TB 的稳定化仍开放**。Lemma 6 说明沿 0-flow 无限加流时 TB 的极限只看首尾边比值。O07/O08 的神经实验用「TB + 总流正则」，它在 Definition 3 意义下是否稳定原文未给出。可证的方向：把 \(\alpha\sum_sF(s)\) 加到 TB 上，沿 0-子流的方向导数下界为 \(\alpha\lVert F_0\rVert_1-\lvert\partial_{F_0}L_{\mathrm{TB}}\rvert\)，后者在 Lemma 6 的条件下有界吗？
5. **连续空间的 0-flow 不是环**（C4）。把 O08 的图上 OT 推到连续状态空间时，「无环」不再是正确的目标，应换成「无 0-子流」（Definition 7 的极小性）。O02 Sec. 7 提到的 Beckmann 连续模型是对接点。
6. **极大 0-子流不唯一**（Prop. 3、Example 7.1）。「去掉环」没有典范做法，\(F=F_0+F_{\min}\) 的 \(F_{\min}\) 依赖于选哪个极大 0-子流。最小费用（O02 的 \(\alpha\to0\) 稀疏极限，其 Prop. 4、Corollary 1）给出的是一个典范选择；把「T19 的极小流」与「O02 的最小费用流」的关系写清楚是一条可做的定理。

#### 8. 引用

```bibtex
@inproceedings{brunswic2024theory,
  title     = {A Theory of Non-Acyclic Generative Flow Networks},
  author    = {Brunswic, Leo Maxime and Li, Yinchuan and Xu, Yushun and Feng, Yiheng and Jui, Shangling and Ma, Lizhuang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {38},
  number    = {10},
  pages     = {11124--11131},
  year      = {2024},
  note      = {arXiv:2312.15246}
}
```

---

**编者注。**

- 作者列表：本仓库 PDF（arXiv v1）署名五人，不含 Yiheng Feng；T36 参考文献与 AAAI proceedings 引用为「Brunswic, Li, Xu, Feng, Jui, Ma」。表格按 PDF 写，bibtex 按 AAAI 版写。
- Sec. 3.1 的四状态例子中回边的端点在 PDF 文本抽取里不可读；按「\(f_1=f_2=f_3=1\) 且 \(c\ge0\) 任意」这一原文结论反推，回边只能是 \(C\to B\)（若是 \(C\to A\) 则守恒会迫使 \(f_2=1+c\)）。3.3 的数值表是编者按该例子验算的，不出自原文。
- Figure 3 caption 把稳定 FM 记为 \(L_{\mathrm{FM},\Delta,f,0,\nu}\)，第三个下标「0」按上下文理解为 \(g\equiv1\)（\(\eta=0\)）。
- Theorem 1 主文版与附录版的假设不一致（主文要求收敛、附录改为 strong FM loss），报告两处都写出，不替原文取舍。
- 3.5 把 Prop. 5 的分解与 O02 Theorem 1 的流分解定理等同，是编者判断：两者都是「非负守恒流 = 路径流 + 环流」，前者用 Zorn 引理与 Kalpazidou 表示，后者用 Ahuja–Magnanti–Orlin 的组合构造。3.6 的词典表是编者整理，每格都注了出处。

## 3.2 T36 · Revisiting Non-Acyclic GFlowNets in Discrete Environments（ICML 2025）

> **一句话**：这篇论文在有限离散图上把非无环 GFlowNet 理论从零重建：以 \(P_B\) 为第一性对象，把流定义为「期望访问次数 × 终止流」，证明它与 \((P_B,F(s_f))\) 一一对应（Prop. 3.7），从而把 T19 的「流可以存在但采不出来」彻底关掉。它的两条主线结论——固定 \(P_B\) 时解唯一、稳定性无关（Corollary 3.11）；学 \(P_B\) 时「最小期望长度 ⇔ 最小总流」（Prop. 3.12、Eq. (11)）——就是 O07/O08 把内部流选择写成最小费用流 / 最优传输的直接前置。

| 字段 | 内容 |
|---|---|
| arXiv | [2502.07735](https://arxiv.org/abs/2502.07735)（本仓库 PDF 为 v3，2025-09-11） |
| 发表 | ICML 2025 主会（Proceedings of the 42nd ICML, Vancouver；PMLR 267, 44887–44910） |
| 作者 | Nikita Morozov\*, Ian Maksimov\*（HSE University）, Daniil Tiapkin（CMAP, École polytechnique / Université Paris-Saclay）, Sergey Samsonov（HSE University）；\* 同等贡献，通讯 Morozov |
| 代码 | [github.com/GreatDrake/non-acyclic-gfn](https://github.com/GreatDrake/non-acyclic-gfn)（Sec. 1 末） |
| 本仓库 PDF | `papers/2502.07735.pdf` · 中译 `papers_zh/2502.07735.zh.pdf` |
| 阅读优先级 | P0：O07/O08 的「最小总流」目标、\(\sum_sF(s)=Z\cdot\mathbb E[n_\tau]\) 恒等式、以及「\(P_B\) 自由度 = 内部流自由度」全部出自这里 |

#### 1. 问题设定与记号

**环境（Sec. 3.1，Assumption 3.1）。** \(G=(S,E)\) 有限有向图，允许环。三条假设：0) \(G\) 有限；1) 初始状态 \(s_0\) 无入边、汇 \(s_f\) 无出边；2) 任何 \(s\in S\) 都从 \(s_0\) 可达且能到达 \(s_f\)。\(\mathrm{in}(s)\)、\(\mathrm{out}(s)\) 为父、子集合。终止状态 \(\mathcal X\) = 有边指向 \(s_f\) 的状态，奖励 \(R(x)>0\)，\(Z=\sum_{x\in\mathcal X}R(x)\)。Definition 3.2：轨迹 \(\tau=(s_0\to s_1\to\cdots\to s_{n_\tau}\to s_{n_\tau+1}=s_f)\)，长度 \(n_\tau\) 只数内部状态；\(\mathcal T\) 是全部有限长轨迹的集合，有环时可以是无限集、且含任意长的轨迹。

**策略与轨迹分布（Sec. 3.2）。** Definition 3.3：\(P_F(s'\mid s)\) 是 \(\mathrm{out}(s)\) 上的分布（\(s\ne s_f\)），\(P_B(s\mid s')\) 是 \(\mathrm{in}(s')\) 上的分布（\(s'\ne s_0\)）。本文从 \(P_B\) 出发定义轨迹分布 Eq. (7)：
\[
P(\tau)\triangleq\prod_{t=0}^{n_\tau}P_B(s_t\mid s_{t+1}).
\]
图关于「交换 \(s_0,s_f\) 并反转所有边」对称，所以从 \(P_F\) 出发是等价的；选 \(P_B\) 是因为奖励约束落在 \(P_B(x\mid s_f)\) 上更自然。

**流（Sec. 3.3，Definition 3.5，Eq. (8)）。** 给定全边 \(P_B>0\) 与终止流 \(F(s_f)>0\)，
\[
F(s\to s')\triangleq F(s_f)\,\mathbb E_{\tau\sim P}\Big[\sum_{t=0}^{n_\tau}\mathbb I\{s_t=s,s_{t+1}=s'\}\Big],\qquad
F(s)\triangleq F(s_f)\,\mathbb E_{\tau\sim P}\Big[\sum_{t=0}^{n_\tau+1}\mathbb I\{s_t=s\}\Big].
\]
无环时期望访问次数 = 访问概率，与经典定义一致；有环时两者不同，只有前者守恒（3.2 节）。

**与标准 DAG-GFlowNet 及 T19 的差异。**

- 与 DAG 理论（Sec. 2.1）：DB 条件 Eq. (2)、FM 条件 Eq. (3)、TB 条件 Eq. (1) 形式不变，但轨迹集合无限，\(\mathbb E[n_\tau]\) 不再有界；「固定 \(P_B\) 则 \(F,P_F\) 唯一」这条 DAG 事实被 Prop. 3.8 推广。
- 与 T19（Sec. 2.2）：T19 用测度论、流为原语；本文只做有限图、策略为原语。本文明确纠正 T19 的一句话——「无环情形的所有定义都能原样搬到有环情形」不成立，因为访问概率流不守恒。
- 边缘约束：奖励匹配 Eq. (9) \(F(x\to s_f)=R(x)\)；源端 \(P_F(\cdot\mid s_0)\) 在「可训练 \(P_B\)」实验里固定为均匀（Appendix B.3）。

#### 2. 核心贡献（按原文 Sec. 1 的编号）

**C1. 有限离散非无环 GFlowNet 的从零构建（Sec. 3，Lemma 3.4，Def. 3.5，Prop. 3.6–3.10）。**
结论：从全边正的 \(P_B\) 出发，轨迹分布良定义、期望长度有限（Lemma 3.4）；期望访问次数流满足 DB 与 FM（Prop. 3.6）；正的守恒边流与 \((P_B,F(s_f))\) 一一对应（Prop. 3.7）；\(P_F\) 存在唯一（Prop. 3.8），反向也成立（Prop. 3.9）；奖励匹配等价于 \(F(x\to s_f)=R(x)\)（Prop. 3.10）。
前提：Assumption 3.1 与 \(P_B>0\)。本文自评：比 T19 简单，并澄清了 T19 未触及的「流的本性」与「\(P_B\) 的重要性」。

**C2. 固定 \(P_B\) 时稳定性无关（Corollary 3.11）。**
结论：\(P_B>0\) 固定后，无环文献里任何损失都可直接用来学 \(P_F\)，解唯一、\(\mathbb E[n_\tau]\) 有限，T19 的损失稳定性「不起作用」。
前提：同上。这是对 T19 的直接修正：稳定性只在 \(P_B\) 也被训练时才是问题。

**C3. 学 \(P_B\) 时：最小期望长度 ⇔ 最小总流（Prop. 3.12，Eq. (10)–(12)）。**
结论：\(\mathbb E_{\tau\sim P}[n_\tau]=\frac{1}{F(s_f)}\sum_{s\notin\{s_0,s_f\}}F(s)\)，把 T19 Theorem 2 的「≤」收紧为「=」；因此「学一个期望长度最小的非无环 GFlowNet」等价于「学一个总流最小的非无环 GFlowNet」，写成约束优化 Eq. (11)；近似解法是 DB + 状态流正则 \(\lambda F_\theta(s)\)（Eq. (12)）。
原文把「指出这一等价」列为自己的关键贡献之一，并称利用它是未来研究的关键方向。

**C4. 尺度假说与实验（Sec. 4，Appendix B.2）。**
结论：实践中决定稳定性的是误差在哪个尺度上算：\(\Delta\log F\)（标准 DB）不加正则时长度可以任意大；\(\Delta F\)（SDB）偏向小流从而不发散，但拟合奖励分布更差。不稳定损失 + 状态流正则反而给出最好的采样质量。
这是一条假说加实验，不是定理。

**C5. 熵正则 RL 等价的非无环推广（Theorem 3.13，Appendix A.7，Lemma A.1）。**
结论：在图诱导的确定性 MDP 上取 \(r(s,s')=\log P_B(s\mid s')\)、\(r(x,s_f)=\log R(x)\)、\(\gamma=1\)、\(\lambda=1\)，熵正则最优策略就是 \(P_F\)，\(V^\star=\log F(s)\)、\(Q^\star=\log F(s\to s')\)。
前提：\(P_B>0\) 固定且满足奖励匹配。证明绕开了 Tiapkin et al. 2024 依赖的拓扑序归纳，改用 occupancy measure 上的严格凸优化。

#### 3. 方法与理论推导要点

##### 3.1 吸收链与有限期望长度（Lemma 3.4，Appendix A.1）

把图的边全部反向，从 \(s_f\) 出发按 \(P_B\) 随机游走，令 \(s_0\) 为吸收态。对任一中间状态 \(s\)，返回概率 \(p_s:=\mathbb P[\exists t>0:Y_t=s\mid Y_0=s]<1\)：Assumption 3.1 保证存在一条从 \(s_0\) 到 \(s\) 的无环路径，\(P_B>0\) 保证沿它走回 \(s_0\) 的概率严格正，而一旦到 \(s_0\) 就不可能再回 \(s\)。于是 \(\mathbb P[N'_s>k]=p_s^k\)，
\[
\mathbb E[N'_s]=\sum_{k\ge0}p_s^k=\frac1{1-p_s}<\infty,\qquad
\mathbb E[n_\tau]=\sum_{s\notin\{s_0,s_f\}}\mathbb E[N_s]\le\sum_s\mathbb E[N'_s]<\infty .
\]
期望长度有限反过来给出 \(\sum_\tau P(\tau)=1\)：否则以正概率出现无限长轨迹，与期望有限矛盾。对照 T19：T19 从「流是有限测度」推出吸收（其 Theorem 2），本文从「\(P_B>0\) + 连通」推出流有限，两条路在有限正图上汇合。

##### 3.2 访问概率不守恒、期望访问次数守恒（Sec. 3.3 的反例）

原文取自 T19 的例子：\(s_0\to a\to b\to c\to s_f\) 加回边 \(c\to b\)，\(P_B(a\mid b)=P_B(c\mid b)=\tfrac12\)，其余 \(P_B=1\)。绕 \(k\) 圈的轨迹 \(s_0\to a\to b\to(c\to b)^k\to c\to s_f\) 概率 \(2^{-(k+1)}\)。

- 边访问概率：\(a\to b\)、\(b\to c\)、\(c\to s_f\) 都是 1，\(c\to b\) 是 \(\tfrac12\)。在 \(b\)：入 \(1+\tfrac12\ne\) 出 \(1\)；在 \(c\)：入 \(1\ne\) 出 \(1+\tfrac12\)。这就是原文的「\(1\ne1+0.5\)」。
- 期望访问次数：\(\mathbb E[k]=\sum_kk\,2^{-(k+1)}=1\)，于是 \(F(b\to c)=2\)、\(F(c\to b)=1\)、\(F(a\to b)=F(c\to s_f)=1\)（取 \(F(s_f)=1\)）。在 \(b\)：入 \(1+1=\) 出 \(2\)；在 \(c\)：入 \(2=\) 出 \(1+1\)。守恒恢复。

Prop. 3.6 的一般证明就是把这个观察写成恒等式：对每条轨迹逐点有 \(\mathbb I\{s_t=s\}=\sum_{s''\in\mathrm{in}(s)}\mathbb I\{s_{t-1}=s'',s_t=s\}=\sum_{s'\in\mathrm{out}(s)}\mathbb I\{s_t=s,s_{t+1}=s'\}\)，取期望即得 FM；DB 条件 \(F(s\to s')=F(s')P_B(s\mid s')\) 用反向链的 Markov 性加 Fubini（Lemma 3.4 保证可交换）。\(F(s_0)=F(s_f)\) 因为每条轨迹恰访问 \(s_0\) 一次。

##### 3.3 存在性与唯一性：正守恒流 ⇔ \((P_B,F(s_f))\)（Prop. 3.7，Appendix A.3）

给定全边正、满足 FM 的 \(F\)，定义 \(P_B(s\mid s')=F(s\to s')/\sum_{s''}F(s''\to s')\)，令 \(\hat F\) 是它诱导的期望访问次数流。由 Prop. 3.6，\(\hat F(s\to s')=\hat F(s')P_B(s\mid s')=C(s')F(s\to s')\)，\(C:=\hat F/F\)。两者都守恒，所以
\[
\forall s\ne s_f:\quad\sum_{s'\in\mathrm{out}(s)}C(s')F(s\to s')-C(s)F(s)=0,\qquad C(s_f)=1 .
\]
\(|S|\) 个未知数、\(|S|\) 个方程，\(C\equiv1\) 是解。若有非常数解 \(C'\)，取 \(S_{\max}=\arg\max C'\)：任一穿过 \(S_{\max}\) 的轨迹必有一步 \(s_t\in S_{\max}\)、\(s_{t+1}\notin S_{\max}\)，在 \(s_t\) 处 \(1=\sum_{s'}C'(s')F(s_t\to s')/(C'(s_t)F(s_t))<1\)，矛盾（若 \(s_f\in S_{\max}\) 则对 \(S_{\min}\) 做同样论证）。这是一个极大值原理，用到了 \(F>0\)（严格不等号）与 Assumption 3.1（轨迹能穿出 \(S_{\max}\)）。

含义：T19 的「僵尸流」\(F-\bar F\)（T19 Def. 5、Prop. 2）在这里恒为零——每个正守恒流都被它自己的 \(P_B\) 精确采样。Appendix B.4 把同一件事写成线性方程组 Eq. (17) \(\hat F(s)=\sum_{s'\in\mathrm{out}(s)}P_B(s\mid s')\hat F(s')\)、\(\hat F(s_f)=F(s_f)\)，其解是吸收链基本矩阵 \(N=(I-Q)^{-1}\)（Kemeny–Snell Theorem 3.2.1）对应 \(s_f\) 的那一行乘 \(F(s_f)\)。

Prop. 3.8（\(P_F\) 唯一）：用前向链重复 Prop. 3.6 的计算得 \(F(s\to s')=F(s)P_F(s'\mid s)\)，与 DB 联立得 \(P_F(s'\mid s)=F(s')P_B(s\mid s')/F(s)\)；存在性靠 \(F(s_0)=F(s_f)\) 让 \(\prod P_B\) 与 \(\prod P_F\) 的望远镜积相等。Prop. 3.9 是逆命题：任何满足 DB 的正三元组 \((F,P_F,P_B)\) 都诱导同一轨迹分布，且 \(F\) 就是 \(P_B\) 的期望访问次数流。Prop. 3.10：终止边至多访问一次，\(F(x\to s_f)=F(s_f)\,\mathbb P[s_{n_\tau}=x]\)，所以奖励匹配 ⇔ Eq. (9)，且 \(F(s_0)=F(s_f)=Z\)、\(P_B(x\mid s_f)=R(x)/Z\)。

##### 3.4 为什么终止分布相同时内部流不唯一

把 3.3 的结论合起来读：

- 奖励匹配只约束 \(P_B(\cdot\mid s_f)=R/Z\)（Prop. 3.10）。其余每个 \(s'\ne s_0,s_f\) 上的 \(P_B(\cdot\mid s')\) 是 \(\mathrm{in}(s')\) 上任意一个正分布，共 \(\sum_{s'}(|\mathrm{in}(s')|-1)\) 个自由参数。
- 不同的 \(P_B\) 给出不同的边流（Prop. 3.7 是双射），但终止分布都是 \(R/Z\)（Prop. 3.10）。所以「内部流不唯一」不是病态，它就是 \(P_B\) 的参数空间；T19 的「加一个 0-flow」翻译过来是「换一个 \(P_B\)」。
- 原文 Sec. 3.4 指出 T19 忽略了一件无环文献里的常识：手工选 \(P_B\)（例如对非终止状态取 \(1/|\mathrm{in}(s')|\)、对 \(s_f\) 取 \(R/Z\)）就能得到一个合法解，\(Z\) 未知的问题由学习未归一化流或把 \(Z\) 设为可学参数绕开。

编者延伸计算（在 3.2 的例子上把 \(P_B(c\mid b)\) 换成自由参数 \(q\)，数值不出自原文）：绕 \(k\) 圈概率 \((1-q)q^k\)，\(\mathbb E[k]=q/(1-q)\)，取 \(F(s_f)=Z\)：
\[
F(a)=Z,\qquad F(b)=F(c)=\frac{Z}{1-q},\qquad
\mathbb E[n_\tau]=\frac{F(a)+F(b)+F(c)}{Z}=1+\frac{2}{1-q}.
\]
唯一终止状态是 \(c\)，任何 \(q\in(0,1)\) 都给出同一终止分布，但内部流从 \(q\to0\) 时的 \((1,1,1)\) 一路涨到 \(q\to1\) 时的无穷；\(q=\tfrac12\) 就是原文的 \((1,2,2)\)、\(\mathbb E[n_\tau]=5\)。这张图上「终止分布相同的内部流」是一条以 \(q\) 参数化的曲线，而不是一个点。有环与无环的区别只在于：DAG 上这条曲线是紧的（长度有上界），有环图上它跑到无穷远。

##### 3.5 minimum flow 如何选出一个（Prop. 3.12，Eq. (10)–(12)，Appendix B.1）

Prop. 3.12 的证明只有一行：轨迹长度是各状态访问次数之和，取期望再除以 \(F(s_f)\)，
\[
\mathbb E_{\tau\sim P}[n_\tau]=\sum_{s\notin\{s_0,s_f\}}\mathbb E_\tau\Big[\sum_t\mathbb I\{s_t=s\}\Big]=\frac{1}{F(s_f)}\sum_{s\notin\{s_0,s_f\}}F(s).
\]
于是「最短期望轨迹」这个行为学目标变成了流空间上的线性泛函。Eq. (11)：
\[
\min_{F,P_F,P_B}\ \sum_{s\notin\{s_0,s_f\}}F(s)\quad\text{s.t.}\quad\log^2\frac{F(s)P_F(s'\mid s)}{F(s')P_B(s\mid s')}=0\ \ \forall s\to s',\qquad F(s_f)P_B(x\mid s_f)=R(x)\ \ \forall x\to s_f .
\]
可行集是 3.4 描述的「所有合法 \(P_B\) 对应的正守恒流」；目标线性；在 3.4 的例子里它是 \(1+2/(1-q)\)，在 \(q\to0\) 处取下确界 3，即最短路 \(s_0\to a\to b\to c\to s_f\)。这就是 minimum flow 的选择方式：线性目标在闭凸多面体上的极小值落在边界上，而边界恰是「某些 \(P_B\) 取 0」——回边不再被使用、绕行被剪掉。两点要说清：

- 极小点在 Definition 3.5 的正性假设之外（\(P_B(c\mid b)=0\)）。本文不直接求 Eq. (11)，而用 Eq. (12) 的 \(\lambda F_\theta(s)\) 正则从内部逼近；\(\lambda\) 权衡期望长度与奖励分布精度（Figure 5）。闭包的理论由 T19 Prop. 5 提供：非负 R-流 = 无环 R-流 + 有向环的非负组合，去掉任何一个环分量都严格减小 \(\sum_sF(s)\)，所以极小点必然无环。
- Appendix B.1：on-policy 训练时正则项的期望梯度是 \(\mathbb E_{\tau\sim P_F}\sum_t\nabla_\theta F_\theta(s_t)=\frac{1}{2F_\theta(s_f)}\nabla_\theta\sum_sF_\theta(s)^2\)——实际最小化的是状态流的平方和，因为采样分布给每个状态的权重正好是它的期望访问次数。若 \(P_F(\cdot\mid s_0)\) 固定为均匀并只在每条轨迹的第一个状态加正则，权重才回到均匀。作者说实验里两者差别不显著。这个平方和恰好是 O02 的二次正则（其 Eq. (4)）。

Corollary 3.11 与 Eq. (11) 的关系：固定 \(P_B\) 是在可行集里钉住一个点，稳定性与否只影响优化路径而不影响终点；学 \(P_B\) 才是在可行集里挑点，这时没有目标函数（标准损失在整个可行集上都是零）就会被优化动力学随意推向无穷远——这是本文对「流困环」的重新诊断：不是损失「不稳定」，而是问题欠定。

##### 3.6 稳定性的重新解释：误差尺度（Sec. 4，Appendix B.2，Figure 3）

Eq. (15) 定义两种残差：
\[
\Delta_{\log F}(s,s',\theta)=\log F_\theta(s)+\log P_F(s'\mid s,\theta)-\log F_\theta(s')-\log P_B(s\mid s',\theta),\qquad
\Delta_F=e^{\log F_\theta(s)+\log P_F}-e^{\log F_\theta(s')+\log P_B}.
\]
标准 DB = \(\Delta_{\log F}^2\)；SDB = \(\log(1+\varepsilon\Delta_F^2)(1+\eta F_\theta(s))\)（Eq. (5)）。两者可以互换尺度，得到四种组合，Sec. 4 全部测了。

- Figure 3 固定反向 \(\log\) 流为 1、扫前向 \(\log\) 流 \(x\)：绿 \(y=(x-1)^2\)，红 \(y=(e^x-e)^2\)，棕 \(y=\log(1+(x-1)^2)(1+0.001e^x)\)，蓝 \(y=\log(1+(e^x-e)^2)(1+0.001e^x)\)。
- \(\Delta F\) 尺度的两条曲线在 \(x\) 减小时迅速平台化、导数趋零：需要把流调大时几乎没有梯度，需要调小时梯度很大。
- 结合 Prop. 3.12（长度 ∝ 总流），这解释了 \(\Delta F\) 损失为什么「稳定」——它系统性低估流、偏向短轨迹——也解释了为什么它拟合更差（Table 1 的塌缩）。
- 原文指出同样的推理适用于 T19 的稳定 FM（Eq. (19)），因为它同样在 \(\Delta F\) 尺度上算差。
- 尺度假说的正式表述（Sec. 4 开头）：\(P_B\) 可训练时，实践中决定「平均轨迹长度是否受控」的主因是误差尺度；\(\Delta\log F\) 不加正则可能长度任意大，\(\Delta F\) 偏向小流而不发散。

##### 3.7 熵正则 RL 等价（Theorem 3.13，Lemma A.1）

MDP \(\mathcal M_G\)：状态 = 顶点，动作 = \(\mathrm{out}(s)\)，转移确定，\(\gamma=1\)，\(r(s,s')=\log P_B(s\mid s')\)、\(r(x,s_f)=\log R(x)\)。Eq. (13) 的熵正则价值在 \(\lambda=1\) 时化为
\[
V^\pi_{\lambda=1}(s_0)=\mathbb E_{\tau\sim P^\pi_T}\Big[\sum_t r(s_t,s_{t+1})-\log\pi(s_{t+1}\mid s_t)\Big]=\log Z-\mathrm{KL}\big(P^\pi_T\,\|\,P\big),
\]
所以最优 \(\pi^\star\) 是 KL 为零的策略，即 Prop. 3.8 的唯一 \(P_F\)。软 Bellman 方程 Eq. (14) 由 \(Q^\star(s,s')=\log F(s\to s')\) 满足（用 Prop. 3.6 的 \(\log F(s\to s')=\log F(s')+\log P_B(s\mid s')\)），\(V^\star=\log\sum_{s'}e^{Q^\star}=\log F(s)\)。Lemma A.1 处理无折扣情形的良定性：把它叫作「正则化最短路问题」，在 \(r\le0\)（\(r=0\) 仅当 \(|\mathrm{in}(s')|=1\)）与最优策略期望长度有限的假设下，最优化在 occupancy measure 多面体
\[
\mathcal K=\Big\{d\ge0:\ \sum_{s'}d(s,s')=\sum_{s''}d(s'',s),\ \sum_{s'}d(s_0,s')=1,\ \sum_{s''}d(s'',s_f)=1\Big\}
\]
上是严格凹的（线性奖励 + 强凹的相对条件熵），解唯一；\(d^\pi(s,s')=\pi(s'\mid s)d^\pi(s)\) 与 Prop. 3.7 在反向图上的双射把 occupancy 和策略绑死。\(\mathcal K\) 就是 3.5 的流多面体归一化后的样子。

##### 3.8 训练循环要点（Sec. 3.5，Appendix B.1、B.3、C.1）

原文没有伪代码，训练流程从 Eq. (12)、Eq. (16) 与 Appendix C 拼出来是：

1. 网络共享主干，三个线性头输出 \(\log F_\theta(s)\)、\(P_F\) 的 logits、\(P_B\) 的 logits（Appendix C.2）；可训练 \(P_B\) 时另有标量 \(\log Z_\theta\)。
2. 用当前 \(P_F\) 采一批轨迹（on-policy；批大小 hypergrid 16、置换 512）。第一步 \(s_0\to s\) 按固定的均匀 \(P_F(\cdot\mid s_0)\) 采。
3. 对每条转移算 DB 残差：中间转移用 Eq. (4)；首转移用 Eq. (16) \(\big[\log Z_\theta-\log|S\setminus\{s_0,s_f\}|-\log P_B(s_0\mid s,\theta)-\log F_\theta(s)\big]^2\)；终止转移把 \(F_\theta(s_f)P_B(x\mid s_f)\) 替换为 \(R(x)\)。
4. 对每个非 \(s_0,s_f\) 的访问状态加 \(\lambda F_\theta(s_t)\)（\(\log Z_\theta\) 不加，Appendix B.3）。
5. Adam 一步；\(\log Z_\theta\) 用 10 倍学习率（沿用 Malkin et al. 2022）。

on-policy 的后果（Appendix B.1）：第 4 步在期望意义下是 \(\frac{\lambda}{2F_\theta(s_f)}\nabla\sum_sF_\theta(s)^2\)。选 FM 以外的损失是有意的：FM 不显式参数化 \(P_B\)，也不能做固定 \(P_B\) 的实验（Appendix C.1）。

##### 3.9 与 T19 的定理级对照（编者整理）

| T19（B24） | T36（本文） | 关系 |
|---|---|---|
| Definition 5 sampler flow \(\bar F\le F\) | Definition 3.5 期望访问次数流 | T36 把 \(\bar F\) 直接当定义；正性下 \(\bar F=F\)（Prop. 3.7） |
| Theorem 2：\(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\) | Prop. 3.12：\(\mathbb E[n_\tau]=\sum_sF(s)/F(s_f)\) | 不等式收紧为等式（原文明说是「refinement」） |
| Theorem 2 吸收性（有限测度 ⇒） | Lemma 3.4 吸收性（\(P_B>0\) + Assumption 3.1 ⇒） | 假设方向相反，结论同 |
| Prop. 5：\(\mathcal F_R^+=\)无环流 \(+H^1_+(G)\) | Prop. 3.7：正守恒流 \(\leftrightarrow(P_B,F(s_f))\) | 同一集合的两种参数化：环坐标 vs \(P_B\) 坐标 |
| Definition 3 稳定性；Theorem 3 不稳定 | Corollary 3.11：固定 \(P_B\) 时无关；尺度假说 | 把「损失性质」改写成「问题是否欠定 + 误差尺度」 |
| Theorem 1：稳定损失 + 流正则 ⇒ 无环极限 | Eq. (11)–(12)：最小总流 + \(\lambda F_\theta(s)\) | 同一思想；T36 去掉了「损失必须稳定」的前提 |
| Example 2 稳定 DB | Eq. (5) SDB | 同一损失；T36 实验显示其在 \(\Delta F\) 尺度下有偏 |
| （无） | Theorem 3.13 熵正则 RL 等价 | T36 新增 |

#### 4. 实验与证据

**共同设定（Sec. 4.1，Appendix C）。** 损失记法：DB / SDB 指损失形式，\(\Delta\log F\) / \(\Delta F\) 指误差尺度，\(\lambda=C\) 指状态流正则强度；(DB, \(\Delta\log F\)) = Eq. (4)，(SDB, \(\Delta F\)) = Eq. (5)，(DB, \(\Delta\log F\), \(\lambda\)) = Eq. (12)。两种 \(P_B\) 设定：固定（\(\mathrm{out}(s_0)=\{s_{\mathrm{init}}\}\)，其余状态 \(P_B\) 在父节点上均匀，\(P_B(s_0\mid s_{\mathrm{init}})=1-\varepsilon\)，\(\varepsilon=10^{-8}\)）与可训练（\(\mathrm{out}(s_0)=S\setminus\{s_0,s_f\}\)，\(P_F(\cdot\mid s_0)\) 固定均匀，\(\log Z_\theta\) 可学，Eq. (16)）。全部 on-policy，Adam 学习率 \(10^{-3}\)，\(\log Z_\theta\) 学习率 \(10^{-2}\)；SDB 取 \(\varepsilon=1.0\)、\(\eta=10^{-3}\)（更大的 \(\eta\) 会缩短轨迹但「显著干扰采样保真度」）。指标不用 T19 的平均奖励，理由是它不衡量对 \(R/Z\) 的拟合。

| 任务 | 设定（出处） | 指标 | 结果 |
|---|---|---|---|
| Hypergrid \(7\times7\) | 状态 \(\{0,\dots,H-1\}^D\)，转移 ±1 且不出界，每个状态可终止；\(R=R_0+R_1\prod_i\mathbb I\{0.25<\lvert s_i/(H-1)-0.5\rvert\}+R_2\prod_i\mathbb I\{0.3<\lvert s_i/(H-1)-0.5\rvert<0.4\}\)，\((R_0,R_1,R_2)=(10^{-3},0.5,2.0)\)（Appendix C.2）；MLP 2×256，批 16，共 \(2\cdot10^6\) 条轨迹 | 最后 \(2\cdot10^5\) 个样本的经验分布与 \(R/Z\) 的 TV 距离 \(=\tfrac12L_1\)（Sec. 4.2，Appendix C.2；图中纵轴标为 \(L_1\)）、平均轨迹长度 | 固定 \(P_B\) 时 (DB, \(\Delta\log F\)) 与 (SDB, \(\Delta F\)) 都收敛到该 \(P_B\) 的真实 \(\mathbb E[n_\tau]\)（Appendix B.4 精确解算），验证 Corollary 3.11；可训练 \(P_B\) 对所有损失都给出更短轨迹；\(\Delta F\) 尺度收敛更慢并带轻微偏差；此环境上不加正则的 \(\Delta\log F\) 也没有发散（Figure 1） |
| Hypergrid \(20^4\) | 同上，只做可训练 \(P_B\)（固定 \(P_B\) 的 \(\mathbb E[n_\tau]\) 比小网格大几个数量级，不可用） | \(L_1\)、平均长度、\(\log Z_\theta\) | \(\Delta F\) 尺度（DB 与 SDB）都学到有偏策略，偏差比 \(7\times7\) 明显更大；\(\Delta\log F\) + \(\lambda=0.001\) 正确拟合并学对 \(\log Z\)；各方法最终长度相近，但 \(\Delta F\) 在训练中段更短（Figure 2）。不加正则的 \(\Delta\log F\) 平均长度「趋于无穷」（Appendix D，Figure 4）；\(\lambda\in\{0.1,0.01,0.001,0.0001\}\) 越大长度越短、策略越偏（Figure 5） |
| 置换 \(S_n\)，\(n=4\) | Cayley 图：\(n-1\) 个相邻对换 + 一个右循环移位；\(R(s)=\exp(\tfrac12\sum_k\mathbb I\{s(k)=k\})\)（比 T19 的 \(\mathbb I[s(1)=1]\) 更难，后者有平凡策略）；MLP 2×128，批 512，\(10^5\) 步 | \(C(k)\) 的 \(L_1\)（\(k\) 个不动点的概率向量）、平均长度 | 固定 vs 可训练 \(P_B\) 的结论与 hypergrid 相同，差别只是 (SDB, \(\Delta F\)) 在可训练 \(P_B\) 下收敛更快（Figure 6） |
| 置换 \(S_n\)，\(n=8,20\) | 同上，全部可训练 \(P_B\)；\(n=20\) 有 \(\approx2.4\cdot10^{18}\) 个状态；真值 \(\log Z\approx11.2533\)（\(n=8\)）、\(42.9843\)（\(n=20\)）由 rencontres 数解析算出（Appendix C.3.1） | \(C(k)\) \(L_1\)、\(\Delta R\)（Shen et al. 2023 的相对均值奖励误差）、\(\Delta\log Z\)、\(\mathbb E[n_\tau]\)；3 个种子 | 见下表（Table 1） |

**Table 1（原文数字，均值 ± 标准差，3 种子）。**

| 损失 | \(n=8\)：\(C(k)\,L_1\) | \(\Delta R\) | \(\Delta\log Z\) | \(\mathbb E[n_\tau]\) | \(n=20\)：\(C(k)\,L_1\) | \(\Delta R\) | \(\Delta\log Z\) | \(\mathbb E[n_\tau]\) |
|---|---|---|---|---|---|---|---|---|
| DB, \(\Delta F\) | 0.215 ±0.198 | 0.214 ±0.086 | 0.814 ±0.826 | 2.43 ±0.28 | 0.453 ±0.002 | 0.343 ±0.000 | 42.98 ±0.000 | 2.00 ±0.00 |
| SDB, \(\Delta F\) | 0.031 ±0.012 | 0.046 ±0.023 | 0.074 ±0.025 | 3.32 ±0.15 | 0.452 ±0.001 | 0.343 ±0.000 | 42.98 ±0.000 | 2.01 ±0.00 |
| DB, \(\Delta\log F\), \(\lambda=10^{-3}\) | 0.036 ±0.015 | 0.056 ±0.024 | 0.018 ±0.010 | 2.80 ±0.04 | 0.041 ±0.002 | 0.064 ±0.000 | 0.023 ±0.005 | 3.23 ±0.00 |
| SDB, \(\Delta\log F\), \(\lambda=10^{-3}\) | 0.037 ±0.013 | 0.056 ±0.019 | 0.020 ±0.015 | 2.79 ±0.04 | 0.041 ±0.002 | 0.064 ±0.000 | 0.026 ±0.003 | 3.22 ±0.00 |
| DB, \(\Delta\log F\), \(\lambda=10^{-5}\) | 0.005 ±0.001 | 0.001 ±0.000 | 0.005 ±0.004 | 4.31 ±0.05 | 0.017 ±0.002 | 0.035 ±0.002 | 0.003 ±0.003 | 7.55 ±0.50 |
| SDB, \(\Delta\log F\), \(\lambda=10^{-5}\) | 0.005 ±0.001 | 0.002 ±0.000 | 0.006 ±0.006 | 4.36 ±0.09 | 0.014 ±0.001 | 0.025 ±0.001 | 0.005 ±0.005 | 7.31 ±0.07 |

读法：\(n=20\) 时两种 \(\Delta F\) 损失的 \(\Delta\log Z=42.98\) 恰等于真值 \(\log Z\approx42.98\)，即学到的 \(\log Z_\theta\approx0\)，且 \(\mathbb E[n_\tau]\approx2\)——它们完全没学到归一化常数，只是走两步就终止；「稳定」在这里是塌缩。\(\Delta\log F\) + 正则在两个 \(n\) 上都把三个误差压到 \(10^{-2}\) 量级；\(\lambda\) 从 \(10^{-3}\) 降到 \(10^{-5}\)，\(n=20\) 的 \(C(k)\,L_1\) 从 0.041 降到 0.017/0.014，代价是 \(\mathbb E[n_\tau]\) 从 3.2 升到 7.5——这是 Eq. (11) 里长度与精度权衡的直接数字证据。

**证据支撑到哪一步。**

- 直接验证：Corollary 3.11（Figure 1、6 的固定 \(P_B\) 曲线落在解析 \(\mathbb E[n_\tau]\) 上）；Prop. 3.12 隐含在所有长度曲线里（长度由采样估计，与 \(\sum F/Z\) 一致但原文没有单独画）。
- 假说层面：尺度假说由 Figure 2–4 与 Table 1 支持，但原文自己称之为 hypothesis，Figure 3 的解释是启发式。
- 未测：Theorem 3.13 没有任何 RL 算法实验（原文未给出）；Appendix B.1 的「平方和 vs 和」差异「不显著」但没给数字。
- Sec. 4.4 三条结论：固定 \(P_B\) 可不用稳定损失与正则，但手选低 \(\mathbb E[n_\tau]\) 的 \(P_B\) 很难；学 \(P_B\) 时 \(\Delta F\) 尺度稳定但常常拟合失败；\(\Delta\log F\) 尺度拟合好但必须配状态流正则。

#### 5. 前提假设与适用边界

- **有限性与连通性**：Assumption 3.1 全程使用。Lemma 3.4 的 \(p_s<1\)、Prop. 3.7 的极大值原理都靠「每个状态都在某条 \(s_0\to s_f\) 轨迹上」。
- **正性**：\(P_B>0\) 于全部边、\(F(s_f)>0\)、\(R(x)>0\)；Prop. 3.7 要求 \(F:E\to\mathbb R_{>0}\)，Prop. 3.9 要求 \(P_F,P_B>0\)。最小流的极小点（某些 \(P_B=0\)）落在这些假设的边界之外，只能被逼近。
- **Markov 流**：只考虑 Markovian 轨迹分布（Sec. 3.2 明说），非 Markov 轨迹流不在框架内。
- **奖励匹配的形式**：Eq. (9) \(F(x\to s_f)=R(x)\)，等价于 \(P_B(x\mid s_f)=R(x)/Z\)；终止边只能访问一次是这条等价的全部依据。
- **Theorem 3.13**：\(P_B>0\) 固定且满足奖励匹配；\(\gamma=1\)、\(\lambda=1\)；Lemma A.1 另需 \(r\le0\)（\(r=0\) 仅当 \(|\mathrm{in}(s')|=1\)）与最优策略期望长度有限——证明里把 \(R\) 归一化为 \(Z=1\)、\(\log R<0\) 来满足前者。
- **正则的实际目标**：on-policy 下 Eq. (12) 最小化 \(\sum_sF(s)^2\) 而非 \(\sum_sF(s)\)（Appendix B.1）；\(\log F_\theta(s_0)\) 不应加正则，因为最优解处它恒等于 \(\log Z\)（Appendix B.3）。
- **实验设定**：可训练 \(P_B\) 时 \(P_F(\cdot\mid s_0)\) 必须固定（否则 \(\mathrm{out}(s_0)=S\setminus\{s_0,s_f\}\) 时不可学），且 \(P_B(s_0\mid s)\) 必须可训练（Appendix B.3）。
- **适用范围正向表述**：任何有限图上、全边正的 GFlowNet，无论有无环，都可用无环文献的损失学 \(P_F\)（固定 \(P_B\)）或用「标准损失 + \(\lambda F_\theta(s)\)」学 \((P_F,P_B)\)；SubTB 与 Deleu et al. 2022 的隐式流参数化也可加同样的正则（Appendix C.1）。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：T19（被重建与修正的对象：其 Theorem 2 → Prop. 3.12，Theorem 3 → Corollary 3.11 与尺度假说，Theorem 1 的流正则 → Eq. (12)，其环境与 SDB 被沿用）；T02（DAG 上的 DB/FM/TB 条件、「固定 \(P_B\) 解唯一」、Definition 4 的策略定义）；T03（TB、\(\log Z\) 可学、手选 \(P_B\) 的常识）；T05（SubTB，作为正则可迁移的对象被提及）；T10（Shen et al. 2023 的 \(\Delta R\) 指标）；Tiapkin et al. 2024（Theorem 3.13 推广的对象，不在仓库）；Deleu 2025 博士论文（约束优化思路的先声）。
- **后继 / 对照**：O07 把 Eq. (11) 的「最小总流」解释为最短路；O08 把它推到多源设定并识别为 Kantorovich 耦合。竞争者是 T19 的稳定损失路线——本文用 Table 1 说明它在大环境上塌缩。C02（图上 Schrödinger 桥）是「熵正则」而非「最小流」的另一条选内部流的路，Theorem 3.13 的 KL 形式是两者对接点。
- **对「内部流选择 = 最优传输」主线的贡献**：
  - (a) Prop. 3.7 + 3.10 给出主线的第一句话：终止分布固定后，内部流的全部自由度 = 非终止状态上 \(P_B\) 的选择（3.4）。没有这个双射，「选内部流」没有参数化。
  - (b) Prop. 3.12 给出主线的目标函数：\(\sum_sF(s)=Z\cdot\mathbb E[n_\tau]\)，线性、可加、有行为学解释。O07/O08 的 \(\sum_eF(e)=\mathbb E[|\tau|]\) 与它只差一个不影响极小点的常数。
  - (c) Eq. (11) 是主线的第一个显式优化问题；Lemma A.1 的多面体 \(\mathcal K\) 就是最小费用流的可行域（O02 Eq. (3) 的 \(\{J\ge0:D^\top J=f\}\) 在单源单汇、归一化后的形式）。
  - (d) Appendix B.1 意外地给出主线的正则化版本：on-policy 训练实际最小化 \(\sum F^2\)，即 O02 的二次正则最小费用流。
- **本文没有做的事**：没有把 Eq. (11) 认出是线性规划、没有讨论极小点的结构（无环、最短路）、没有多源、没有 OT 语言。这些是 O02（数学）与 O07/O08（识别）的工作。

#### 7. 可复用的 insight 与开放问题

1. **Eq. (11) 是单位费用最小费用流**。定理草稿：在 Assumption 3.1 下，令 \(\overline{\mathcal F}_R\) 为「非负、守恒、\(F(x\to s_f)=R(x)\)、\(F(s_0)=Z\)」的边流集合（Prop. 3.7 双射像的闭包）；则 \(\min_{\overline{\mathcal F}_R}\sum_sF(s)\) 是线性规划，极小点集合的每个顶点是一族从 \(\mathrm{out}(s_0)\) 到 \(\mathcal X\) 的最短路上的路径流；\(Z\cdot\min\mathbb E[n_\tau]=\sum_xR(x)\,d_G(\text{源},x)\)（单源时）。证明工具：T19 Prop. 5（去环减总流）+ O02 Theorem 1（路径分解）。这正是 O07 要证的事。
2. **on-policy 正则 = 二次正则**（Appendix B.1）。O02 Prop. 4 / Corollary 1 说二次正则在 \(\alpha\) 足够小时选出 LP 的一个唯一解且解不随 \(\alpha\) 变。实验草案：在 \(7\times7\) 上用 Appendix B.4 精确算出不同 \(\lambda\) 下 Eq. (12) 的稳态流，检验是否存在 \(\lambda\) 阈值以下流不再变化；若存在，说明 GFlowNet 训练在小 \(\lambda\) 下精确到达了最小流解而非仅逼近。
3. **状态流正则 = RL 里的每步负奖励**。由 Prop. 3.12 与 Theorem 3.13 的 \(V=\log Z-\mathrm{KL}\)：给每条非终止转移加常数奖励 \(-c\)，目标变为 \(\log Z-\mathrm{KL}-c\,\mathbb E[n_\tau]\)，与 Eq. (12) 的期望形式只差权重。定理草稿：软 Q 学习在 \(r(s,s')=\log P_B(s\mid s')-c\) 下的最优策略对应 \(F\) 的一个重加权版本，\(c\to\infty\) 极限给出最短路策略（Lemma A.1 名字里的「正则化最短路」）。这为 O07 提供了 RL 实现路径。
4. **固定 \(P_B\) 的设计问题**。\(20^4\) 上均匀 \(P_B\) 的 \(\mathbb E[n_\tau]\) 大几个量级，作者说手选难。但 Appendix B.4 的基本矩阵 \(N=(I-Q)^{-1}\) 在小图上可精确算，在大图上可用 \(P_B\) 的局部启发式（例如偏向减小到 \(s_{\mathrm{init}}\) 的曼哈顿距离）。实验：比较均匀、距离偏置、学习三种 \(P_B\) 的 \(\mathbb E[n_\tau]\) 与 \(L_1\)。
5. **尺度假说的定量化**。Figure 3 只是一维切片。可证的命题：对 \(\Delta F\) 损失，当 \(\log F_F-\log F_B\to-\infty\) 时梯度按 \(e^{\log F_F}\) 衰减，而 \(\Delta\log F\) 梯度线性增长；由此在固定 \(P_B\) 的凸问题上给出两种损失稳态偏差的显式比较，解释 Table 1 里 \(\Delta F\) 在 \(n=20\) 的塌缩。
6. **闭包理论缺口**。Prop. 3.7 只覆盖 \(F>0\)。最小流解在边界上，需要一条「\(P_B\) 允许取零但保持 Assumption 3.1 意义下的可达性」的存在唯一性定理；T19 Prop. 5–6 的闭凸域给了拓扑，但「边界流 ⇔ 部分退化的 \(P_B\)」的双射尚未写出。

#### 8. 引用

```bibtex
@inproceedings{morozov2025revisiting,
  title     = {Revisiting Non-Acyclic {GFlowNets} in Discrete Environments},
  author    = {Morozov, Nikita and Maksimov, Ian and Tiapkin, Daniil and Samsonov, Sergey},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {44887--44910},
  year      = {2025},
  publisher = {PMLR},
  note      = {arXiv:2502.07735}
}
```

---

**编者注。**

- 页码 44887–44910 取自背景文档核验过的 PMLR 条目；本仓库 PDF（arXiv v3）只写 "PMLR 267, 2025"，未印页码。
- Sec. 3.3 反例的边标签在 PDF 文本抽取中错位，3.2 节按原文的「\(1\ne1+0.5\)」与期望访问次数图中的「2」反推出图结构 \(s_0\to a\to b\to c\to s_f\) 加回边 \(c\to b\)，并逐项验算一致；3.4 节把 \(P_B(c\mid b)\) 推广为参数 \(q\) 的计算是编者延伸，不出自原文。
- 3.5 节「极小点无环」引用了 T19 Prop. 5 而非本文；本文没有对 Eq. (11) 极小点结构下任何断言。
- 6 节把 Lemma A.1 的多面体 \(\mathcal K\) 与 O02 Eq. (3) 的可行域等同，是编者判断：两者都是「非负 + 逐点守恒 + 端点边缘固定」，差别只在多源/单源与归一化。
- Table 1 数值逐格对照 PDF 抄录；原文用蓝/红标注最优指标与最短长度，此处未保留颜色。


# 第 4 章 最小流 ⇔ 最短路（O07）

O07 把 T36 的最小总流目标变成一个充要刻画：期望轨迹长度取最小值，当且仅当策略把全部质量放在最短路上（Thm. 3.4）；并把无权图寻路归约为训练一个带流正则的非无环 GFlowNet。它是 O08 的「\(R\equiv1\)、单源」特例，O08 Thm. 3.3 明说自己「recovers the corresponding claim of Morozov et al. (2026)」。发表状态：ICML 2026 SPIGM Workshop，非主会。本章为解读报告全文。

> **一句话**：证明非无环 GFlowNet 的期望轨迹长度 \(\mathbb E[n_\tau]\) 取到最小值，当且仅当策略把全部概率质量放在 \(s_0\) 到终止状态的最短路上（Theorem 3.4，充要）。基于这个刻画，把任意无权图的寻路问题归约成「训练一个带流正则的非无环 GFlowNet」：反转边、把目标点当 \(s_0\)、每个点都设成终止状态、取 \(R\equiv1\)，训完的后向策略就是最短路求解器。在 3×3×3 魔方上以更小的 beam 预算达到与 CayleyPy Cube 可比的解长（Table 1）。它是 O08 的直接桥梁：O08 只是把这里的单源 \(s_0\) 换成一个源分布 \(L\)。

| 字段 | 内容 |
|---|---|
| arXiv | [2603.01786](https://arxiv.org/abs/2603.01786)（v1 [cs.LG] 2026-03-02） |
| 发表 | **ICML 2026 SPIGM Workshop（Workshop 论文，非主会）** |
| 作者 | Nikita Morozov¹、Ian Maksimov¹、Daniil Tiapkin²˒³、Sergey Samsonov¹（¹HSE University，²CMAP, CNRS, École polytechnique，³LMO, Université Paris-Saclay；通讯 Nikita Morozov） |
| 代码 | [github.com/GreatDrake/gfn-pathfinding](https://github.com/GreatDrake/gfn-pathfinding)（原文 Sec. 1 末给出） |
| 本仓库 PDF | `papers/2603.01786.pdf`（14 页：正文 10 页 + 附录 A/B/C） |
| 阅读优先级 | **P0** —— O08 的必读先修。「最小流为什么偏最短路」这条链的充要证明只在这里，O08 只给了它的 LP 对偶版 |

### 1. 问题设定与记号

沿用原文 Sec. 2.1（记号与理论取自 Morozov et al. 2025，本仓库 T36）。

**环境**：有限有向图 \(\mathcal G=(S,E)\)。**Assumption 2.1**：(1) \(s_0\) 无入边、\(s_f\) 无出边；(2) 对任意 \(s\in S\)，存在 \(s_0\to s\) 的路径与 \(s\to s_f\) 的路径。允许有环。

**轨迹与终止**：\(\tau=(s_0\to s_1\to\cdots\to s_{n_\tau}\to s_f)\)，约定 \(s_{n_\tau+1}=s_f\)。终止状态集 \(\mathcal X\)，目标分布 \(R(x)/Z\)，\(R(x)>0\)，\(Z=\sum_{x\in\mathcal X}R(x)\)。

**策略与一致性**（原文 Eq. (1)(2)）：\(\mathbb P(\tau)=\prod_tP_F(s_{t+1}\mid s_t)=\prod_tP_B(s_t\mid s_{t+1})\)，reward matching 写成 \(P_B(x\mid s_f)=R(x)/Z\)。

**流与期望长度**（原文 Eq. (3)，即 T36 Definition 3.5 / Proposition 3.12）：

\[
\mathcal F(s)=Z\cdot\mathbb E_{\tau\sim\mathbb P}\Big[\sum_{t=0}^{n_\tau+1}\mathbb I\{s_t=s\}\Big],
\qquad
\mathbb E[n_\tau]=\frac1Z\sum_{s\in S\setminus\{s_0,s_f\}}\mathcal F(s).
\]

所以「最小期望长度」= 「最小总流」，这是全篇的出发点。原文 Sec. 2.1 也给出实践做法：损失加 \(\lambda\mathcal F_\theta(s)\)，例如 DB + 状态流正则

\[
\mathcal L(s\to s')=\Big(\log\frac{\mathcal F_\theta(s)P_F(s'\mid s,\theta)}{\mathcal F_\theta(s')P_B(s\mid s',\theta)}\Big)^2+\lambda\mathcal F_\theta(s),
\]

其中 reward matching 通过代入 \(\mathcal F_\theta(s_f)P_B(x\mid s_f,\theta)=R(x)\) 强制。

**与标准 DAG-GFlowNet 设定的三处差异**：

1. 图有环，\(n_\tau\) 无上界，流必须按期望访问次数理解；
2. **\(P_B>0\) 的假设被换掉了**。T36 为了让 \(\mathbb P\) 良定义要求所有边上 \(P_B>0\)，但最小 \(\mathbb E[n_\tau]\) 的策略恰恰必须给非最短路边赋 0（这正是 Theorem 3.4 的内容），两者直接冲突。原文用 **Assumption 3.1：\(\mathbb E[n_\tau]<+\infty\)** 替代，其中 \(n_\tau\) 是从 \(s_f\) 出发、按 \(P_B\) 走的后向随机游走到达 \(s_0\) 的步数。这条更弱、更自然，同时保住了「\(P_B\) 是吸收 Markov 链的转移核、\(\mathbb P\) 是 \(\mathcal T\) 上的合法概率分布」（Appendix A 有推导，引 Kemeny & Snell 1969）；
3. 目标不是采样质量本身，而是**用后向策略做寻路**：\(\ell(s')\) 记 \(s_0\to s'\) 最短路长度，由 Assumption 2.1 它总存在。

### 2. 核心贡献（按原文编号）

**贡献 1（Theorem 3.4，充要刻画）**：Assumption 2.1 + Assumption 3.1 下，满足 reward matching \(P_B(s\mid s_f)=R(s)\) 的后向策略 \(P_B\) 最小化 \(\mathbb E[n_\tau]\)，**当且仅当**对任何终止于 \(x\in\mathcal X\) 的轨迹 \(\tau\)，\(n_\tau\ne\ell(x)\Rightarrow\mathbb P(\tau)=0\)。也就是：最小化期望轨迹长度 ≡ 给所有非最短路轨迹赋零概率。原文 Remark 3.5 补充：由 T36 Proposition 3.8（任意 \(P_B\) 存在唯一等价 \(P_F\)，反之亦然），该定理对前向策略同样成立，因为条件只涉及轨迹分布本身。

**贡献 2（Sec. 3.2，构造性归约）**：任意无权有限图上的「所有点到目标点 \(v_g\) 的最短路」问题，可归约为训练一个最小化 \(\mathbb E[n_\tau]\) 的非无环 GFlowNet。与「学 value function 再喂给搜索」的路线（Agostinelli et al. 2019；Chervov et al. 2025b）不同，这里直接学一个最优解就是精确最短路的策略。

**贡献 3（Sec. 3.3，训练算法）**：正则化 trajectory balance（原文 Eq. (7)(8)，Algorithm 1），加上两个工程决定：训练轨迹长度截断到 \(N_{\max}\)、对每个前缀都算 TB。

**贡献 4（Sec. 4，实验）**：Swap Puzzle（\(n=15,20\)）+ 2×2×2 / 3×3×3 魔方，与 CayleyPy Cube（Chervov et al. 2025b，NeurIPS 2025）对比。

### 3. 方法与理论推导要点

#### 3.1 为什么最小流准则偏向最短路（Lemma 3.2 + Lemma 3.3 → Theorem 3.4）

这是全篇的核心，两条引理夹出一个等式，充要性来自「下界可达 + 严格性」。

**下界（Lemma 3.2）**：Assumption 2.1 下，任何满足 Assumption 3.1 的 \(P_B\) 满足（原文 Eq. (4)）

\[
\mathbb E[n_\tau]\ \ge\ \sum_{x\in\mathcal X}P_B(x\mid s_f)\,\ell(x).
\]

**为什么成立**：塔性质拆条件期望（原文 Eq. (5)）\(\mathbb E[n_\tau]=\sum_{x}P_B(x\mid s_f)\mathbb E[n_\tau\mid s_{n_\tau}=x]\)，而任何从 \(s_0\) 到 \(x\) 的轨迹按定义长度至少 \(\ell(x)\)，所以逐项有 \(\mathbb E[n_\tau\mid s_{n_\tau}=x]\ge\ell(x)\)。注意这里的第一步就用到了「后向游走从 \(s_f\) 起、先按 \(P_B(\cdot\mid s_f)\) 选终止状态」这个结构——终止状态的选择概率被 reward matching 钉死，所以下界是常数，不随策略变。

**下界可达（Lemma 3.3，构造性）**：Assumption 2.1 下总存在满足 reward matching 的 \(P_B\)，其期望长度恰为（原文 Eq. (6)）\(\mathbb E[n_\tau]=\sum_{x\in\mathcal X}\frac{R(x)}{Z}\ell(x)\)。

**构造怎么做**：定义 \(\mathrm{par}:S\setminus\{s_0,s_f\}\to S\setminus\{s_f\}\)，对每个 \(s\) 任取一个父节点 \(\mathrm{par}(s)\) 使 \((\mathrm{par}(s)\to s)\in E\) 且 \(\ell(s)=\ell(\mathrm{par}(s))+1\)。这样的父节点一定存在，因为 \(s_0\to s\) 的最短路上 \(s\) 的前驱就满足它。然后取 \(P_B(x\mid s_f)=R(x)/Z\)、\(P_B(s\mid s')=\mathbb I\{s=\mathrm{par}(s')\}\)。于是后向采样先从 \(s_f\) 跳到某个 \(x\)，之后每一步严格靠近 \(s_0\) 一条边，必然采到一条最短路，故 \(\mathbb E[n_\tau\mid s_{n_\tau}=x]=\ell(x)\)。

**Theorem 3.4 的两半**（证明在 Appendix B）：

- **(⇐) 只走最短路 ⇒ 最优**：若 \(\mathbb P(\tau)>0\) 蕴含 \(n_\tau=\ell(s_{n_\tau})\)，则由 Eq. (5) 直接得 \(\mathbb E[n_\tau]=\sum_x\frac{R(x)}{Z}\ell(x)\)，正是 Lemma 3.2 的下界，故最优。
- **(⇒) 最优 ⇒ 只走最短路**：反证。若存在 \(\tau\) 终止于 \(x'\) 且 \(n_\tau>\ell(x')\) 而 \(\mathbb P(\tau)>0\)，则 \(\mathbb E[n_\tau\mid s_{n_\tau}=x']>\ell(x')\)，其余终止状态仍有 \(\ge\ell(x)\)，于是 \(\mathbb E[n_\tau]>\sum_x\frac{R(x)}{Z}\ell(x)\)。但 Lemma 3.3 表明这个右端可被达到，矛盾。

**关键机制一句话**：终止状态的边缘分布被 reward matching 固定（\(R>0\) 保证每个 \(x\) 的权重严格正），所以 \(\mathbb E[n_\tau]\) 是一组固定权重下的条件期望长度加权和；每一项的下界是 \(\ell(x)\)，且这些下界能被同时取到。因此「最小总流」不是**倾向**最短路——它是**精确等价**于只走最短路。原文 Remark 3.5 还指出最优 \(P_B\) 一般**不唯一**：对任意 \(s\)，它可以在「位于最短路上的父节点」之间任意分配概率，只要给不在最短路上的父节点赋 0。

**Remark 3.6 的副产品**：Lemma 3.3 的构造若取退化的 \(P_B(s\mid s')=\mathbb I\{s=\mathrm{par}(s')\}\)，则 \(\mathrm{par}\) 在排除 \(s_f\) 后定义了一棵以 \(s_0\) 为根的有向树，对应的前向策略有闭式 \(P_F(s'\mid s)=V(s')/V(s)\)，\(V(s)\) 是 \(s\) 可达的树叶上 GFlowNet 奖励之和（引 Bengio et al. 2021）。

#### 3.2 与 T36 的关系：修掉一条假设，补上一个刻画

**T36 缺什么**：T36 建立了非无环框架（期望访问次数流、\(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\)、固定 \(P_B\) 时任意无环损失可用、熵正则 RL 等价），也提出了 \(\lambda\mathcal F_\theta(s)\) 正则来压期望长度。但正如本文 Sec. 1 所说，「the structural implications of minimizing this quantity have not been fully analyzed」——T36 知道要压短，不知道压到底是什么形状。Theorem 3.4 给出了这个形状，并且是充要的。

**修掉的假设（Appendix A，这是最实质的技术关系）**：T36 的理论建立在「所有边上 \(P_B(s\mid s')>0\)」之上（T36 Lemma 3.4 用它证明后向链吸收、\(\mathbb P\) 归一、\(\mathbb E[n_\tau]<\infty\)）。本文指出这条假设**对最优解本身不成立**：最小 \(\mathbb E[n_\tau]\) 的 \(P_B\) 必须给非最短路边赋零概率，所以「T36 的理论不能直接用于学习最小期望长度的非无环 GFlowNet」。修法有两条，原文都给了：

1. 把 \(P_B>0\) 换成 Assumption 3.1（\(\mathbb E[n_\tau]<\infty\)）。原文说明 T36 中唯一真正用到 \(P_B>0\) 的是它的 Lemma 3.4 本身，其余结论的证明只需要「\(\mathbb E[n_\tau]<\infty\)」+「\(\mathbb P\) 是 \(\mathcal T\) 上的合法概率测度」，因此都可以在更弱假设下重写。Appendix A 显式验证了后者：\(\mathbb E[n_\tau]<\infty\) 蕴含后向游走以概率 1 到达 \(s_0\)（否则 \(\{\sum_t\mathbb I\{X_t\ne s_0\}=\infty\}\) 有正概率），进而 \(\sum_{\tau\in\mathcal T}\mathbb P(\tau)=\mathbb P[\exists t:X_t=s_0]=1\)。
2. 或者做子图约简：取 \(\mathcal T'=\{\tau:\mathbb P(\tau)>0\}\)，它诱导的子图 \(\mathcal G'\) 仍满足 Assumption 2.1（含 \(s_0,s_f\)，且每个状态都在某条正概率轨迹上），在 \(\mathcal G'\) 上 \(P_B>0\) 成立，于是 T36 Proposition 3.8 可用，把得到的 \(P_F\) 在 \(\mathcal G'\) 外补零。原文用这条把 Remark 3.5 的唯一性结论合法化。

**继续使用 T36 的部件**：\(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\)（Proposition 3.12）、\(P_B\leftrightarrow P_F\) 唯一对应（Proposition 3.8）、终止边流 \(\mathcal F(s\to s_f)=R(s)\)（Proposition 3.10）、\(\lambda\mathcal F_\theta(s)\) 正则、DB 损失形式。所以本文相对 T36 是「同一框架下补一个结构定理 + 松一条假设 + 换一个应用领域」，不是替代。

#### 3.3 归约构造（Sec. 3.2）

给定任意有限有序图 \(G=(V,E)\) 与目标点 \(v_g\)，要找每个 \(v\) 到 \(v_g\) 的最短路（无权图，且假设每点都能到 \(v_g\)）：

1. 先删掉 \(G\) 中 \(v_g\) 的所有出边——不改变问题，因为这些边不在任何指向 \(v_g\) 的最短路上；
2. 状态 \(S\) 对应 \(V\) 的顶点，外加汇 \(s_f\)；**\(s_0\) 对应 \(v_g\)**；
3. 转移 \(\mathcal E\) 是 \(G\) 中边的**反向**，另外每个状态（除 \(s_f\)）加一条到 \(s_f\) 的边。

**为什么这是合法环境**：\(s_0\) 无入边（因为删了 \(v_g\) 的出边、又反向）；\(s_f\) 无出边；每个状态都有到 \(s_f\) 的边，所以**所有非 \(s_f\) 状态都是终止状态**（这条后面被训练算法反复利用）；\(s_0\) 到每个状态可达，因为原图中每点可达 \(v_g\)、反向后即 \(v_g\) 可达每点。

**为什么解决了原问题**：取任意正奖励 \(R\)（原文推荐任务无关的 \(R\equiv1\)）。由 Theorem 3.4，最小 \(\mathbb E[n_\tau]\) 的 \(P_B\) 只给 \(s_0\)（即 \(v_g\)）到终止状态（即所有顶点）的最短路赋非零概率；而 \(P_B\) 在反向图上走 = 在原图上正向走，所以从任意状态 \(s\) 出发按 \(P_B\) 采样，得到的就是原图中 \(v\to v_g\) 的最短路。取 \(R\equiv1\) 时最优 \(\mathbb E[n_\tau]\) 恰是全图平均最短路长 \(\frac1{|V|}\sum_{v\in V}\ell(v)\)，这给了训练一个直接可读的 ground-truth 参照量。

**两个策略的分工**（原文 Sec. 3.2「Forward policy」）：\(P_B\) 是产品——从任意配置出发解到目标；\(P_F\) 是训练所需的辅助件——从目标出发反向游走，采样服从 \(R\)（均匀）的状态，并决定何时停（\(P_F(s_f\mid s)\)）。魔方语境下：\(P_B\) 用最少步数还原任意配置，\(P_F\) 用最少步数把还原态打乱成均匀分布的配置。

#### 3.4 训练目标（Sec. 3.3，Eq. (7)(8)，Algorithm 1）

两条来自实践的修正，原文都给了理由：

- **截断轨迹长度到 \(N_{\max}\)**：训练早期 \(\mathbb E[n_\tau]\) 极大，采完整轨迹的开销在大环境里不可承受（Brunswic et al. 2024 也这么做）。采样时把 stop 动作 \(P_F(s_f\mid s,\theta)\) mask 掉，得到固定长度的部分轨迹 \(\tau'=(s_0\to\cdots\to s_{N_{\max}})\)。
- **弃 DB 用 TB**：早期实验中 DB 收敛极慢。原文给了两条解释：TB 的信用分配更高效（Malkin et al. 2022）；更关键的是**本构造的训练信号主要来自目标状态 \(s_0\) 本身**，而 DB 定义在单条转移上、大多数转移不含 \(s_0\)，TB 定义在整条轨迹上、按构造每条都含 \(s_0\)。作者把这个现象的深入分析留作 future work。

因为每个状态都是终止状态，每个前缀 \(\tau'_{0:i}\) 加一条终止转移就是一条完整轨迹，于是对所有前缀求和（原文 Eq. (7)）。又因为 \(R\equiv1\)，\(Z=|V|\) 已知、无需学 \(\log Z_\theta\)。流也不用单独学：在「每点都有到 \(s_f\) 的边」的环境里 \(\mathcal F(s)=R(s)/P_F(s_f\mid s)\)——这一步由终止边流 \(\mathcal F(s\to s_f)=R(s)\)（T36 Proposition 3.10）与 DB 的一部分 \(\mathcal F(s\to s_f)=\mathcal F(s)P_F(s_f\mid s)\)（T36 Proposition 3.8）组合得到。最终目标（原文 Eq. (8)）：

\[
\mathcal L_{\mathrm{regTB}}(\theta,\tau)=\sum_{i=0}^{N_{\max}}\Big(\mathcal L_{\mathrm{TB}}(\theta,\tau_{0:i})+\frac{\lambda}{P_F(s_f\mid s_i,\theta)}\Big),
\qquad
\mathcal L_{\mathrm{TB}}(\theta,\tau_{0:i})=\Big(\log\frac{P_F(s_f\mid s_i,\theta)\prod_{t=1}^iP_F(s_t\mid s_{t-1},\theta)}{(1/|V|)\prod_{t=1}^iP_B(s_{t-1}\mid s_t,\theta)}\Big)^2.
\]

Algorithm 1 就是：采 \(B\) 条长度至多 \(N_{\max}\) 的轨迹 → 算 \(\frac1B\sum_i\nabla_\theta\mathcal L_{\mathrm{regTB}}\) → 更新。原文强调**同时优化 \(P_F\) 与 \(P_B\) 是算法的关键组成**，并指出专门的后向策略优化方法（Jang et al. 2024；Gritsaev et al. 2025）可能进一步改善训练。

#### 3.5 测试时 beam search（Sec. 3.4）

理论最优策略精确给出最短路，但大图上学到的只是近似，所以测试时用 beam search（跟随 Chervov et al. 2025b）：宽度 \(W\)，每步扩展所有续接、按轨迹后向转移概率对数之积保留 top \(W\)，到达 \(s_0\) 即停；另加一个去重启发式（每步丢掉重复状态），原文说略有改善。\(W=1\) 退化为对 \(P_B\) 的贪心 \(\arg\max_sP_B(s\mid s',\theta)\)——**如果策略最优，贪心仍产出最短路**，因为非最短路转移的概率必须为 0（Theorem 3.4）。原文也提到 entropy-regularized MCTS（Morozov et al. 2024；Xiao et al. 2019）是可选替代。

### 4. 实验与证据

网络：6 个 `ReLU(Linear(LayerNorm(x)) + x)` 残差块的 MLP，状态 one-hot 输入，\(P_F,P_B\) 共享 backbone 不同线性头（前向 logits 数 = 原图出边数 + 1 个 stop）。与前作的差别是把 BatchNorm 换成 LayerNorm 以稳住训练与评测。JAX 实现，整训练循环 JIT（Appendix C，引 Tiapkin et al. 2025 的 gfnx）。

#### 4.1 Swap Puzzle（Sec. 4.1，Figure 2、Figure 3 左上）

任务：把任意 \(n\) 元排列用最少的相邻对换排序，即 \(S_n\) 以相邻对换为生成集的 Cayley 图上寻路。**ground truth 可算**：最优解长 = 排列的逆序数（原文给的两条理由：只有恒等排列逆序数为 0；一次相邻对换使逆序数 ±1）。

- 规模：\(n=15\) 与 \(n=20\)，Cayley 图分别约 \(1.3\cdot10^{12}\) 与 \(2.4\cdot10^{18}\) 个状态（原文 Sec. 4.1）。
- 测试集：各 500 个均匀采样的排列。
- 三种评测协议：忠实从 \(P_B\) 采样、对 \(P_B\) 贪心（\(W=1\)）、\(W=4\) 的 beam search。
- 结果（Figure 2）：充分训练后**贪心与 beam search 两种协议在测试集每个排列上都给出精确最短路**；忠实采样的策略平均接近最优。
- 泛化：\(n=20\) 的模型训练中只见过约 \(10^9\) 个状态，占 \(2.4\cdot10^{18}\) 的极小一部分（原文 Sec. 4.1）。
- 成本：\(n=20\) 跑 100k 次迭代在单张 NVIDIA H200 上 15 分钟（Figure 2 caption）。
- 超参（Appendix C.1）：100,000 迭代、batch 128、AdamW（weight decay \(10^{-5}\)）、lr \(3\cdot10^{-4}\)、hidden 1024、梯度范数裁剪阈值 100、\(N_{\max}=50\)；\(\lambda=10^{-3}\)（\(n=15\)）、\(\lambda=10^{-4}\)（\(n=20\)）。

#### 4.2 魔方（Sec. 4.2，Table 1）

对手：CayleyPy Cube（Chervov et al. 2025b，NeurIPS 2025），当前该任务的 SOTA ML 方法，已被证明在解长与运行效率上优于 DeepCubeA（Agostinelli et al. 2019）与 EfficientCube（Takano 2023）。公平性处理：对 CayleyPy 同时训练「原文超参」与「与本文同等网络规模」两个模型，取更好者；两法都用 beam search；只允许 90° 面转，故解长按 QTM 计。测试集：2×2×2 用 Chervov et al. 2025b 的 100 例，3×3×3 用 Agostinelli et al. 2019 的 1000 例。Solve rate 指在 100 步内找到合法路径的比例。

| 2×2×2 Beam | Ours 解长 | Ours solve | CayleyPy 解长 | CayleyPy solve |
|---|---|---|---|---|
| \(W=2^0\) | 11.62 | 1.0 | x | 0.00 |
| \(W=2^2\) | 11.24 | 1.0 | x | 0.00 |
| \(W=2^4\) | 10.78 | 1.0 | x | 0.00 |
| \(W=2^6\) | **10.64** | 1.0 | x | 0.06 |
| \(W=2^8\) | **10.64** | 1.0 | x | 0.89 |
| \(W=2^{10}\) | **10.64** | 1.0 | 10.64 | 1.0 |
| \(W=2^{12}\) | **10.64** | 1.0 | 10.64 | 1.0 |

| 3×3×3 Beam | Ours 解长 | Ours solve | CayleyPy 解长 | CayleyPy solve |
|---|---|---|---|---|
| \(W=2^0\) | x | 0.471 | x | 0.000 |
| \(W=2^3\) | x | 0.984 | x | 0.001 |
| \(W=2^6\) | 25.33 | 1.0 | x | 0.687 |
| \(W=2^9\) | 23.49 | 1.0 | 24.34 | 1.0 |
| \(W=2^{12}\) | 22.42 | 1.0 | 22.44 | 1.0 |
| \(W=2^{15}\) | 21.70 | 1.0 | 21.61 | 1.0 |
| \(W=2^{18}\) | 21.24 | 1.0 | 21.15 | 1.0 |

（数字照抄原文 Table 1；"x" 是原文标记，表示该 beam 宽度下未对全测试集给出有效解、故不报解长。10.64 是该 2×2×2 测试集的最优平均长度，原文用 BFS 验证。）

- 2×2×2：本文在 \(W=2^6\) 达到最优 10.64，CayleyPy 要到 \(W=2^{10}\)，即**beam 宽度小 16 倍**；且本文在贪心（\(W=1\)）下就对全测试集给出有效解，CayleyPy 在小 beam 下找不到任何有效路径。
- 3×3×3：\(W\in[2^0,2^9]\) 区间本文更好（\(W=2^9\)：23.49 vs 24.34），\(W\in\{2^{12},2^{15},2^{18}\}\) 两者相当（22.42/22.44、21.70/21.61、21.24/21.15，后两档 CayleyPy 略优）。
- 运行时（\(W=2^{18}\)，单张 H200）：本文 25M 参数模型平均 **1.74 秒**解一个配置，CayleyPy 4M 参数模型 **6.19 秒**。原文给的机制解释：CayleyPy 与 DeepCubeA 类方法要对每个邻居各跑一次前向来估值/距离（3×3×3 有 12 个邻居，故 12 倍前向次数），而本文一次前向就输出所有邻居对应的后向策略 logits。
- 超参（Appendix C.2）：2×2×2 用 500,000 迭代 / batch 128 / hidden 1024 / \(\lambda=10^{-2}\) / \(N_{\max}=12\)；3×3×3 用 1,000,000 迭代 / batch 2048 / hidden 2048 / \(\lambda=5\cdot10^{-7}\) / \(N_{\max}=24\)。原文注明这两个 \(N_{\max}\) 都小于对应环境的最长路径，仍训得好，作为泛化能力的进一步证据。与 CayleyPy 对比时用相同数量的采样配置。

#### 4.3 \(\lambda\) 消融（Sec. 4.3，Figure 3）

观察：**\(\lambda\) 越大越好，但太大会完全失败**——Figure 3 的两个子图（Swap \(n=15\)、2×2×2）中最大的 \(\lambda\) 都导致模型找不到任何有效路径。由此给出的调参 rule of thumb：训很少的迭代后就评测，取「仍能找到到目标的有效路径」的最大 \(\lambda\)。

#### 4.4 证据强度判断

- 实验直接支持：Swap Puzzle 上贪心/beam 协议给出**精确**最短路（有逆序数 ground truth）；2×2×2 上达到 BFS 验证的最优 10.64 且 beam 预算小 16 倍；3×3×3 小 beam 区间优于 SOTA、大 beam 区间持平；单配置求解墙钟时间 1.74s vs 6.19s。
- 作者推断、未直接验证：「TB 优于 DB 是因为信用分配 + 每条轨迹都含目标状态」——原文明说是 hypothesize 并留作 future work；泛化能力由「见过状态数 / 总状态数」与「\(N_{\max}\) 小于最长路径仍有效」两条间接证据支撑，没有分布外测试。
- 原文未给出：Swap Puzzle 的具体数值表（只有 Figure 2 曲线，报告中无法引用精确数字）；与经典 BFS/双向 BFS 在中等图上的对比；3×3×3 的 \(\lambda=5\cdot10^{-7}\) 为何比 2×2×2 的 \(10^{-2}\) 小五个数量级的解释；带权图（作者在 Sec. 5 列为 future work）。

### 5. 前提假设与适用边界

1. **有限有向图 + Assumption 2.1**：\(s_0\) 无入边、\(s_f\) 无出边、每个状态从 \(s_0\) 可达且可达 \(s_f\)。归约构造（Sec. 3.2）验证了这三条在「反向图 + 每点连 \(s_f\)」下自动成立。
2. **Assumption 3.1：\(\mathbb E[n_\tau]<+\infty\)**（后向游走的期望步数有限）。这条替代了 T36 的 \(P_B>0\)，是本文能谈「最优策略给某些边赋 0」的前提。
3. **无权图、跳数距离**：\(\ell(\cdot)\) 数边数。带权/成本敏感设定被原文 Sec. 5 明确列为 future work。
4. **\(R>0\) 于所有终止状态**：Theorem 3.4 的严格性论证要求每个 \(x\) 在 \(\mathbb E[n_\tau]\) 里有正权重。归约用 \(R\equiv1\)，自动满足。
5. **可达性**：归约假设每个顶点都能到 \(v_g\)。若图不连通到目标，需要先做可达性剪枝。
6. **定理是关于精确最优解的**：Theorem 3.4 刻画的是最小 \(\mathbb E[n_\tau]\) 的策略。实际训练是 \(\lambda\) 罚的软版本，且 \(\lambda\) 太大会训崩（Sec. 4.3），所以大图上要靠 beam search 兜住近似误差。这不是理论瑕疵，是把定理转成算法时的真实代价。

**适用范围**：状态空间大到无法存储、但转移规则明确且可逆（Cayley 图、置换谜题、组合构型空间）的无权寻路。特别适合「一次训练、任意起点求解」的摊销模式，以及每步需要对全部邻居打分的场景（单次前向出所有邻居 logits 是它对 value-based 方法的结构优势）。不适合带权图、需要最优性证书的场景、以及小到 BFS 直接可跑的图。

### 6. 在 GFlowNet × OT 主线中的位置

**前驱**：

- **T19**（Brunswic et al. 2024, AAAI）：非无环 GFlowNet 与流正则化思想；训练轨迹长度截断的做法也来自它。
- **T36**（Morozov et al. 2025, ICML）：全部记号、期望访问次数流、\(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\)、\(P_F\leftrightarrow P_B\) 唯一性、终止边流条件、\(\lambda\mathcal F_\theta(s)\) 正则。本文对它做了一次假设级修正（\(P_B>0\to\mathbb E[n_\tau]<\infty\)），这是两篇之间最实质的技术关系。
- **TB**（Malkin et al. 2022）：训练目标本体。
- **寻路侧对手/参照**：DeepCubeA（Agostinelli et al. 2019）、EfficientCube（Takano 2023）、CayleyPy（Chervov et al. 2025a,b）。

**后继**：

- **O08**（arXiv 2606.06272）：直接建立在本文之上。桥梁关系是精确的——本文是**单源**（\(s_0\) 一点）到多终点、代价为 \(\ell(x)=d(s_0,x)\)；O08 加一条约束 \(\mathcal F(s_0\to u)=L(u)\) 把单点源换成**源分布**，目标就从 \(\sum_xR(x)\ell(x)\) 变成 \(\sum_{u,x}d(u,x)\Pi(u,x)\)，即 Kantorovich OT。两个方向的证明也一一对应：本文 Lemma 3.2 的「实际路长 \(\ge\ell(x)\)」对应 O08 Eq. (18) 的放缩；本文 Lemma 3.3 的「沿最短路构造」对应 O08 Part 1 的「沿最短路铺质量」。另外 O08 Theorem 3.3 用 LP 对偶 + 互补松弛重新得到本文的结论（O08 原文说 "recovers the corresponding claim of Morozov et al. (2026)"），并把本文的 TB 目标、每前缀计算技巧、\(\lambda\) 权衡观察整套继承。
- 换个角度看两篇的分工：**本文提供「为什么最小流会选最短路」的概率论证明（充要），O08 提供同一事实的 LP 对偶证明并给出证书（\(\pi^\star_x=d(x)\)、互补松弛）**。想做 primal-dual 版 GFN-OT，两篇都要读：目标函数的语义在这里，对偶变量的身份在 O08。

**对主线的贡献**：它是「内部流选择 = 最优传输」这条链的第一环。链条是三段：T36 把流定义成期望访问次数、于是「最小总流」这个目标写得出来 → 本文证明最小总流 ⟺ 只走最短路 → O08 加源分布约束、把它抬成 Kantorovich OT。缺了本文这一环，O08 的 Theorem 3.2 仍然能证（它的证明自洽），但「为什么这个目标函数值得最小化」「最优解的路径结构长什么样」这两个问题只有本文回答了，而且回答得比 O08 的对偶版更强——**充要**，而非只给支撑集包含关系。

### 7. 可复用的 insight 与开放问题

1. **「reward matching 钉住终止边缘」是最小流准则能奏效的隐藏支点**。Lemma 3.2 的下界 \(\sum_xP_B(x\mid s_f)\ell(x)\) 之所以是常数、能被当作下界比较，全靠 \(P_B(x\mid s_f)=R(x)/Z\) 固定。如果放开终止分布，最小化 \(\mathbb E[n_\tau]\) 的解会退化成「只终止在离 \(s_0\) 最近的那个状态」，最短路结构就没了。任何想改动 reward matching 的 GFN-OT 变体（尤其 unbalanced 版）都必须重做这个论证。
2. **最优 \(P_B\) 不唯一，可以当成免费的正则自由度**（Remark 3.5）。在最短路父节点之间如何分配概率完全自由。这意味着可以在「只走最短路」的约束内再优化第二目标：最大熵（多样最短路）、最小方差、或对未见状态的鲁棒性。定理草稿：在最短路子图上做最大熵 GFlowNet，得到「所有最短路上的均匀分布」，这对需要多样解的应用（谜题多解、分子合成路径）直接可用。
3. **训练信号定位解释了 TB ≫ DB**（Sec. 3.3.1）。这个论证可迁移到任何「信号集中在少数特殊状态」的 GFlowNet 任务：DB 的逐转移损失会被大量不含信号的转移稀释，TB 的逐轨迹损失按构造每条都含信号。可测假设：把 SubTB 的部分轨迹长度作为插值参数扫描，收敛速度应随「片段包含目标状态的概率」单调变化。
4. **单次前向出所有邻居 logits 是策略型方法对 value 型方法的结构优势**（Sec. 4.2）。3×3×3 上这带来 12 倍前向次数差、实测 1.74s vs 6.19s。这条优势在分支因子大的环境里会放大（分子编辑动作空间可达数百），值得作为独立的效率论点在其他任务上复现。
5. **\(\lambda\) 的悬崖式失败需要更好的机制**。Sec. 4.3 的 rule of thumb（取最大的仍能找到路径的 \(\lambda\)）本质是手工试错，且 2×2×2 与 3×3×3 的最佳 \(\lambda\) 差五个数量级（\(10^{-2}\) vs \(5\cdot10^{-7}\)），说明它跟图直径/规模强耦合。可做的替换：把 \(\lambda\) 罚换成对 \(\mathbb E[n_\tau]\) 的显式约束（augmented Lagrangian 或对偶上升自动调 \(\lambda\)），此时 O08 的对偶势 \(\pi_s=d(s)\) 正好提供了乘子的解析参照。
6. **带权图扩展（原文 Sec. 5 列为 future work）技术上是低风险的**：把 \(\ell(\cdot)\) 换成加权最短路、把 \(\lambda\mathcal F_\theta(s)\) 换成 \(\lambda c(s)\mathcal F_\theta(s)\)。Lemma 3.2/3.3 的论证只用到「路径代价 \(\ge\) 最短路代价」与「存在沿最短路的确定性父节点映射」，严格正权重下两者都保持。零权边会造出零代价环，必须排除——这是唯一需要额外条件的地方。
7. **没有最优性证书是当前形态的主要缺口**。Swap Puzzle 有逆序数、2×2×2 有 BFS，所以能验证；3×3×3 无 ground truth，只能靠横向比较。O08 的互补松弛 \(\mathcal F^\star(s\to s')(\pi^\star_{s'}-1-\pi^\star_s)=0\) 给出了逐边可检验的替代方案：只要额外学一个 \(\pi_\phi(s)\approx d(s)\)，就能在没有 ground truth 的图上报告一个 gap 代理。这是把两篇合起来读最直接的可执行收益。

8. **\(N_{\max}\) 截断没有理论代价分析，但实测可以远小于图直径**。2×2×2 用 \(N_{\max}=12\)、3×3×3 用 \(N_{\max}=24\)，原文注明两者都小于对应环境的最长路径，训练仍然有效（Appendix C.2）。截断改变了训练时的轨迹分布：\(\mathcal L_{\mathrm{regTB}}\) 只在长度 \(\le N_{\max}\) 的前缀上评估，等于只对「离目标不超过 \(N_{\max}\) 步」的那部分状态空间施加平衡约束。这留下一个可证的问题：设最短路长 \(>N_{\max}\) 的状态占比为 \(p\)，训练目标的极小解与真实最小流解的差距能否由 \(p\) 控住？如果能给出这样的界，\(N_{\max}\) 就从工程旋钮变成有依据的超参。原文把它当纯粹的成本控制手段，未做分析。

9. **贪心 \(W=1\) 的表现应该被当作最优性的诊断指标，而不只是一个廉价评测协议**。Theorem 3.4 说最优策略给非最短路转移赋零概率，所以贪心也能出最短路——反过来，贪心一旦失手，就说明存在非最短路转移仍带有最大概率，即策略离最优有可定位的差距。原文在 Figure 2 里已经用「贪心失败的 checkpoint 不画点」的方式隐含使用了这个信号（caption 明说这些点被省略），但没有把它做成量化指标。可执行的做法：报告「贪心与 \(W\) 大时解长的差值」随训练的曲线，并按状态到目标的距离分层，定位误差集中在近目标还是远目标区域。这比单一平均解长信息量大得多。

10. **归约构造把「目标点」编码进 \(s_0\)，因此一个模型只服务一个目标点**。魔方里目标就是还原态、这不成问题；但换成多目标（任意一对起终点的最短路、或目标随时间变化）时，需要把目标条件化进网络 \(P_B(s\mid s',v_g)\)，此时 Theorem 3.4 只逐个 \(v_g\) 成立，跨目标的一致性没有理论保证。这是与 O08 的摊销课题正交的另一条摊销轴：O08 摊销的是「源—目标分布对 \((L,R)\)」，这里摊销的是「目标顶点 \(v_g\)」。后者在 Cayley 图上有额外结构可用——群的平移不变性使 \(d(u,x)=d(e,u^{-1}x)\)，所以条件化在原则上可以简化为把状态换成 \(v_g^{-1}s\)。这一点原文完全未提，但对置换谜题以外的一般图恰恰不成立，值得单独验证。

### 8. 引用

```bibtex
@article{morozov2026shortest,
  title  = {Learning Shortest Paths with Generative Flow Networks},
  author = {Morozov, Nikita and Maksimov, Ian and Tiapkin, Daniil and Samsonov, Sergey},
  journal = {arXiv preprint arXiv:2603.01786},
  note   = {ICML 2026 SPIGM Workshop (workshop paper, not main conference).
            Code: https://github.com/GreatDrake/gfn-pathfinding},
  year   = {2026},
  url    = {https://arxiv.org/abs/2603.01786}
}
```

---

### 编者注（自主决定的判断，与原文区分）

1. **发表状态**：PDF 内没有 venue 行（模板看起来是 UAI/AISTATS 风格的双栏 + Supplementary Material 分页）。按仓库 `data/papers.yaml` 登记为 `ICML 2026 SPIGM Workshop`，报告与 bibtex 均标注为 Workshop 论文、非主会。
2. **定理编号**：主结论是 **Theorem 3.4**，两条引理是 **Lemma 3.2 / Lemma 3.3**，弱化后的后向策略假设是 **Assumption 3.1**（环境假设是 **Assumption 2.1**）。证明在 Appendix B，后向策略假设的讨论在 Appendix A。
3. **Swap Puzzle 无数值表**：原文只用 Figure 2 曲线报告，未给表。报告因此只转述定性结论（贪心与 beam 给出精确最短路）与 caption 中的明确数字（500 例测试集、\(\approx1.3\cdot10^{12}\) / \(2.4\cdot10^{18}\) 状态、\(10^9\) 见过状态、15 分钟 / H200），没有从曲线上读数。
4. **Table 1 中的 "x"** 是原文标记，含义按 caption 推断为「该 beam 宽度下未对全测试集找到有效解，故不报平均解长」；原文未逐条解释，此为我的读法。
5. **§7 第 2、5、6、8、9、10 条是我的推断与实验草稿**，原文未给出；第 6 条的带权扩展方向本身由原文 Sec. 5 背书（列为 future work），但「零权边会造零代价环」这一限制是我加的。第 10 条里 Cayley 图的平移不变性 \(d(u,x)=d(e,u^{-1}x)\) 是群论常识，不是论文结论。
6. **与 O08 的桥梁对应关系**（Lemma 3.2 ↔ O08 Eq. (18)、Lemma 3.3 ↔ O08 Part 1）是我做的映射；O08 原文只笼统说「recovers the corresponding claim of Morozov et al. (2026)」，没有逐条对齐。


# 第 5 章 核心定理：GFlowNet 最小总流 ⇔ Kantorovich 最优传输（O08）

本章是全报告的中心，收录 O08 解读报告全文：四步证明骨架（双线性 → LP → divergence 约束 → 两侧夹逼）、对偶与互补松弛证书、八条前提假设、hypergrid 与排列实验的全部数字，以及「等价是经典的、接口是新的」这一判断的依据。发表状态：ICML 2026 SPIGM Workshop，非主会。

> **一句话**：在非无环 GFlowNet 的最小总流问题上再钉一条约束——把第一步边流固定成源分布 \(L\)——最小总流目标就精确变成以图最短路为 ground cost 的 Kantorovich OT，最优前向策略采样出的轨迹端点分布就是最优耦合（Theorem 3.2）。图上 shortest-path OT 等价于 min-cost flow 是经典结果，这篇的价值不在这个等价本身，而在于它把等价翻译成 GFlowNet 语言之后，输出的不再是一张耦合矩阵，而是一个能在巨大隐式图上逐步执行合法局部动作的路由策略 \(P_F(s'\mid s)\)。这是本仓库 GFlowNet × OT 主线的接口定义者，也是唯一给出「内部流该选哪一个」几何原则的工作。

| 字段 | 内容 |
|---|---|
| arXiv | [2606.06272](https://arxiv.org/abs/2606.06272)（v1 [cs.LG] 2026-06-04；PDF 页脚自署 "Preprint. June 5, 2026"） |
| 发表 | **ICML 2026 SPIGM Workshop（Workshop 论文，非主会）** |
| 作者 | Ian Maksimov¹、Nikita Morozov¹、Denis Belomestny¹˒²、Sergey Samsonov¹（¹HSE University，²Duisburg-Essen University；通讯 Ian Maksimov） |
| 代码 | 未公开（原文正文与附录均未给出代码链接；Appendix B 只说实现基于 Morozov et al. 2025 的公开代码） |
| 本仓库 PDF | `papers/2606.06272.pdf`（17 页：正文 8 页 + Appendix A 理论 + Appendix B 实验细节） |
| 阅读优先级 | **P0** —— 全仓库唯一把 GFlowNet 目标函数与 Kantorovich OT 写成等号的工作，后续所有 GFN-OT 课题都得从它的假设集出发 |

### 1. 问题设定与记号

沿用原文 Sec. 2.1 的记号（转述自 Morozov et al. 2025，本仓库 T36）。

**环境**：有限有向图 \(G=(S,E)\)，\(E\subseteq S\times S\)，\(\mathrm{out}(s)\) 为子节点集、\(\mathrm{in}(s)\) 为父节点集。允许有环（non-acyclic），这是全部后续结论的前提，不是背景板。

**轨迹**：\(\tau=(s_0\to s_1\to\cdots\to s_{n_\tau}\to s_f)\)，\(n_\tau\) 为轨迹长度，约定 \(s_{n_\tau+1}=s_f\)。形如 \(s\to s_f\) 的边叫终止转移，有出边指向 \(s_f\) 的状态构成终止状态集 \(\mathcal X\)，目标分布 \(R(x)/Z\) 定义在 \(\mathcal X\) 上，\(R(x)>0\)，\(Z=\sum_{x\in\mathcal X}R(x)\)。

**策略与平衡条件**：前向策略 \(P_F(s'\mid s)\)、后向策略 \(P_B(s\mid s')\)，训练目标是让两者诱导的轨迹分布相同（原文 Eq. (1)）：

\[
\mathbb P(\tau)=\prod_{t=0}^{n_\tau}P_F(s_{t+1}\mid s_t)=\prod_{t=0}^{n_\tau}P_B(s_t\mid s_{t+1}),
\]

加上 reward matching（原文 Eq. (2)）\(P_B(x\mid s_f)=R(x)/Z,\ \forall x\in\mathcal X\)。

**流 = 期望访问次数**（原文 Eq. (3)，即 T36 Definition 3.5）：

\[
\mathcal F(s)=Z\cdot\mathbb E_{\tau\sim\mathbb P}\Big[\sum_{t=0}^{n_\tau+1}\mathbb I\{s_t=s\}\Big],
\qquad
\mathcal F(s\to s')=Z\cdot\mathbb E_{\tau\sim\mathbb P}\Big[\sum_{t=0}^{n_\tau}\mathbb I\{s_t=s,s_{t+1}=s'\}\Big].
\]

有环时「访问概率」不守恒、「期望访问次数」才守恒，所以这个定义是必须的（原文 Sec. 2.1 引 T36 Proposition 3.6 给出流匹配条件 Eq. (4)：\(\sum_{s\to s'}\mathcal F(s\to s')=\sum_{s''\to s}\mathcal F(s''\to s)\)，以及 \(\mathcal F(s_0)=\mathcal F(s_f)=Z\)）。

**最小总流问题**：记内部状态集 \(\mathcal I:=S\setminus\{s_0,s_f\}\)。原文 Sec. 2.1 复述 T36 的关键等式 \(\mathbb E[n_\tau]=\frac1Z\sum_{s\in\mathcal I}\mathcal F(s)\)（T36 Proposition 3.12），于是「最小期望轨迹长度」= 「最小总流」，原文 Eq. (5)：

\[
\min_{\mathcal F,P_F,P_B}\sum_{s\in\mathcal I}\mathcal F(s)
\quad\text{s.t.}\quad
\mathcal F(s)P_F(s'\mid s)=\mathcal F(s')P_B(s\mid s'),\ \forall (s,s')\in E;\quad
\mathcal F(s_f)P_B(x\mid s_f)=R(x),\ x\in\mathcal X.
\]

实践中的常规做法是给损失加 \(\lambda\mathcal F_\theta(s)\)（Brunswic et al. 2024；T36），原文 Sec. 2.1 明确说自己延用这一路线。

**OT 侧**：原文 Definition 2.1 给耦合定义，Eq. (6) 给 Kantorovich LP。

**本文新增的结构假设**（原文 Assumption 3.1，四条，作者说「similarly to Essid & Solomon (2018)」）：

1. 特殊初始状态 \(s_0\) 无入边，特殊汇 \(s_f\) 无出边；
2. 存在状态集 \(U\) 与 \(\mathcal X\)，使 \(s_0\) 的出边只进入 \(U\)、\(s_f\) 的入边只来自 \(\mathcal X\)；
3. 任意 \(u\in U\) 与 \(x\in\mathcal X\) 之间存在有限长度路径；
4. 给定 \(U\) 上的分布 \(L(u)\) 与 \(\mathcal X\) 上的分布 \(R(x)\)，且 \(\sum_{u\in U}L(u)=\sum_{x\in\mathcal X}R(x)=1\)（\(L,R\) 在支撑外补零，视为定义在全部内部状态上）。

**ground cost**（原文 Eq. (7)）：\(d(u,x)=|\tau_{u,x}|\)，\(\tau_{u,x}\) 是 \(u\to x\) 的一条最短路，\(|\cdot|\) 是它的边数。于是原文 Eq. (8) 就是 \(L,R\) 之间以 \(d\) 为代价的 Kantorovich 问题。原文在 Eq. (8) 后自己点明：这种以最短路为 ground cost 的图 OT 有经典 min-cost-flow 表述（Essid & Solomon 2018），也等价于 Beckmann 问题的离散图版本（Beckmann 1952）。

**关键的新约束**：作者把 \(L\) 称为 *leward*（left reward），并仿照 reward matching 钉住第一步边流

\[
\mathcal F(s_0\to u)=L(u).
\]

**与标准 DAG-GFlowNet 设定的差异**，四点，都是硬差异：

- 图允许有环，流必须按期望访问次数解释，否则 \(\sum_s\mathcal F(s)=\mathbb E[n_\tau]\) 这个目标函数根本写不出来；
- 不只固定终止分布，还固定第一步分布 \(\mathcal F(s_0\to u)=L(u)\)——这条是全篇定理的支点；
- 两个边缘都归一化，因此 \(Z=1\) 已知（原文 Sec. 3.1 明说选归一化表述是为了「eliminates the need to handle an unknown normalizing constant」）。GFlowNet 招牌的 unknown-\(Z\) 能力在这个设定下主动放弃；
- 目标不是「找一个满足终止分布的解」，而是「在所有满足两个边缘的解中取内部总流最小者」，即把标准 GFlowNet 严重欠定的内部流自由度用一条几何原则钉死。

### 2. 核心贡献（按原文编号）

**贡献 1（原文 Sec. 1 第 1 条 + Theorem 3.2 + Appendix A.1）**：T36 的最小总流学习问题可等价改写为线性规划；在此 LP 上固定初始边流分布后，最小总流目标精确变成以图距离为代价的 Kantorovich OT，也等价于 Beckmann 问题的离散图形式（Beckmann 1952；Essid & Solomon 2018）。此设定下的最优前向策略在初始分布与终止分布之间采样最优路径，从而诱导出最优耦合。作为特例，这个构造复现 Morozov et al. (2026)（本仓库 O07）的最短路理论。

**贡献 2（原文 Sec. 1 第 2 条 + Sec. 3.3 + Sec. 4）**：GFlowNet 学习框架给出了逼近图 OT 解的可行手段。基于神经参数化，GFlowNet 目标可以学出一个逼近最优耦合的策略；实验在精确解可算时复现精确 OT 解，并在精确解不可算的规模上（排列 \(n=20\)）给出近似。

原文把定理级结果放在三处，报告后文按这个编号引用：

- **Theorem 3.2**（原文 Sec. 3.2）：Assumption 3.1 下，约简原问题 Eq. (11) 与 \(L,R\) 之间的 Kantorovich 问题 Eq. (8) 等价，\(\mathrm{GFlow}^\star=\mathrm{OT}^\star\)（Eq. (12)）；且若 \(\mathbb P^\star\) 是 \(\mathrm{GFlow}^\star\) 解诱导的轨迹分布，则 \(\Pi^\star_{u,x}:=\sum_{\tau:u\rightsquigarrow x}\mathbb P^\star(\tau)\) 是 Eq. (8) 的最优耦合。
- **Theorem 3.3**（原文 Sec. 3.2 末；完整版是 Appendix A.5 的 Theorem A.5）：**不含**第一步约束的原始问题 Eq. (10) 的对偶为 \(\max_\pi\sum_{x}R(x)\pi_x\) s.t. \(\pi_{s_0}=0,\ \pi_{s'}-\pi_s\le 1\)（\(s'\ne s_f\)）。三条结论：任意可行 \(\pi\) 满足 \(\pi_s\le d(s)\)；\(d(\cdot)=|\tau_{s_0,\cdot}|\) 可行且对偶最优；若 \(R(x)>0\ \forall x\)，则任意最优 \(\pi^\star\) 在终止状态上满足 \(\pi^\star_x=d(x)\)。互补松弛给出 \(\mathcal F^\star(s\to s')\big(\pi^\star_{s'}-(1+\pi^\star_s)\big)=0\)，即最优流只能落在 tight 子图上——而 tight 子图就是最短路子图。原文明确说这一条「recovers the corresponding claim of Morozov et al. (2026)」。
- **Proposition A.4 与 Appendix A.4**：提升 LP 的对偶推导（Eq. (22)→Eq. (23)）；扩展问题（含第一步约束）的对偶为 \(\max_\pi\sum_x R(x)\pi_x+\sum_u L(u)(-\pi_u)\)，换记号 \(a_x=\pi_x,\ b_u=-\pi_u\) 后得到 \(a_x+b_u\le d(u,x)\)（Eq. (24)→Eq. (25)），与 Kantorovich 对偶同形。

### 3. 方法与理论推导要点

整条链是四步：**双线性 → 线性（LP）→ 约简成 divergence 约束 → 两侧夹逼得等号**，再加一步对偶给出证书。

#### 3.1 第一步：消去策略，把双线性约束变成线性（Appendix A.1）

Eq. (5) 的 detailed balance 约束在决策变量里是双线性的，因为（原文 Eq. (9)）

\[
\mathcal F(s\to s')=\mathcal F(s)P_F(s'\mid s)=\mathcal F(s')P_B(s\mid s').
\]

把 \(\mathcal F(s)\) 与 \(\mathcal F(s\to s')\) 都当独立变量、消掉 \(P_F,P_B\) 之后问题变线性。**为什么成立**：Proposition A.1 说「存在 \(P_F(\cdot\mid s)\in\Delta(\mathrm{out}(s))\) 使 \(\mathcal F(s\to v)=\mathcal F(s)P_F(v\mid s)\)」与「\(\mathcal F(s\to v)\ge0\) 且 \(\sum_{v\in\mathrm{out}(s)}\mathcal F(s\to v)=\mathcal F(s)\)」等价——正向是把概率求和为 1 代入，反向在 \(\mathcal F(s)>0\) 时直接归一化定义策略，在 \(\mathcal F(s)=0\) 时所有出边流被非负性与零和挤成 0，此时任意策略都满足等式。Proposition A.2 对入边同理。Corollary A.3 汇总成提升 LP（原文标 (P)）。这一步是整篇的技术起点：detailed balance 与流匹配在「策略可自由重构」的意义下是同一个约束。

#### 3.2 第二步：约简掉状态流，得到 divergence 形式（原文 Eq. (10)→(11)）

加上 \(\mathcal F(s_0\to u)=L(u)\) 与 \(\mathcal F(x\to s_f)=R(x)\) 后得到扩展原始问题 Eq. (10)。定义内部边集

\[
E^\circ:=\{s\to s'\in E:\ s\ne s_0,\ s'\ne s_f\},
\]

用入流等式把 \(\mathcal F(s)\) 全部代掉，得约简原始问题（原文 Eq. (11)）：

\[
\min_{\mathcal F(s\to s')\ge0}\sum_{s\to s'\in E^\circ}\mathcal F(s\to s')
\quad\text{s.t.}\quad
\sum_{v:\,s\to v\in E^\circ}\mathcal F(s\to v)-\sum_{u:\,u\to s\in E^\circ}\mathcal F(u\to s)=L(s)-R(s),\ \ s\in\mathcal I.
\]

**这一步有一个必须记住的常数**。原文在 Eq. (11) 后给出

\[
\sum_{s\in\mathcal I}\mathcal F(s)=\sum_{s_0\to u}\mathcal F(s_0\to u)+\sum_{s\to s'\in E^\circ}\mathcal F(s\to s')=1+\sum_{s\to s'\in E^\circ}\mathcal F(s\to s'),
\]

并写明「We will omit this constant, as it does not change minimum of the problem」。所以 **Theorem 3.2 里的 \(\mathrm{GFlow}^\star\) 是约简目标 \(\sum_{E^\circ}\mathcal F\)，不是 \(\sum_{s\in\mathcal I}\mathcal F(s)=\mathbb E[n_\tau]\) 本身**，两者差 1。这个 1 就是 \(s_0\to u\) 那一步：它是「进入源分布」的动作，不属于 \(u\rightsquigarrow x\) 的运输段。Table 1 里 \(\mathbb E|\tau|\) 与 \(\mathrm{OT}^\star\) 数值直接对齐（如 \(H=10\), Moon：4.352 vs 4.351），说明实验报告的 \(\mathbb E|\tau|\) 度量的是运输段长度、与约简目标同尺度。

约束右端 \(L(s)-R(s)\) 就是离散 divergence：这正是 min-cost flow / 离散 Beckmann 的标准形式。原文同时指出，把所有内部状态的流匹配约束求和会得到 \(\sum_x R(x)=\sum_u L(u)\)，即质量平衡是约束的必要推论，所以 Assumption 3.1 第 4 条不是可选装饰。

#### 3.3 第三步：\(\mathrm{GFlow}^\star\le\mathrm{OT}^\star\)（原文 Theorem 3.2 证明 Part 1）

取 Eq. (8) 的最优耦合 \(\Pi^\star\)（存在性引 Villani 2008, Theorem 4.1, p. 43），对每对 \((u,x)\) 任选一条最短路 \(\tau_{u,x}\)，令

\[
\mathcal F(s\to s')=\sum_{u\in U}\sum_{x\in\mathcal X}\Pi^\star_{u,x}\cdot\mathbb I[s\to s'\in\tau_{u,x}].
\]

**可行性为什么成立**：定义 \(\Delta^{u,x}_s=\sum_{v:s\to v\in E}\mathbb I[s\to v\in\tau_{u,x}]-\sum_{v:v\to s\in E}\mathbb I[v\to s\in\tau_{u,x}]\)。因为 \(\tau_{u,x}\) 是一条有向路径，它在起点只有出边、终点只有入边、中间节点进出计数相等，所以 \(\Delta^{u,x}_s=\mathbb I[s=u]-\mathbb I[s=x]\)。代入 \(\Pi^\star\) 的两个边缘约束即得（原文 Eq. (13)）

\[
\Delta\mathcal F(s)=\sum_{u,x}\Pi^\star_{u,x}\big(\mathbb I[s=u]-\mathbb I[s=x]\big)=L(s)-R(s).
\]

**目标值为什么等于 OT 代价**：交换求和次序，\(\sum_{s\to s'\in E^\circ}\mathbb I[s\to s'\in\tau_{u,x}]=|\tau_{u,x}|\)，而 \(\tau_{u,x}\) 是最短路所以 \(|\tau_{u,x}|=d(u,x)\)，于是（原文 Eq. (14)(15)）目标值 \(=\sum_{u,x}d(u,x)\Pi^\star_{u,x}=\mathrm{OT}^\star\)。这个流可行但未必最优，故 \(\mathrm{GFlow}^\star\le\mathrm{OT}^\star\)（原文 Eq. (16)）。

这一半的全部内容就是：**把耦合的每一份质量沿最短路铺成边流**。它同时证明了 Eq. (10)/(11) 这种「带额外等式约束的 GFlowNet 问题」一定可行——原文在 Sec. 3.1 末专门提醒，从 GFlowNet 理论本身看不出这一点。

#### 3.4 第四步：\(\mathrm{GFlow}^\star\ge\mathrm{OT}^\star\)（原文 Theorem 3.2 证明 Part 2）

取 Eq. (11) 的最优解 \(\mathcal F^\star\)（存在性来自可行集非空 + 目标有下界），用 Appendix A.1 的等价性还原前向策略

\[
P^\star_F(s'\mid s)=\mathcal F^\star(s\to s')\Big/\sum_{s\to s''}\mathcal F^\star(s\to s''),
\qquad
\mathbb P^\star(\tau)=\prod_{i=0}^{n_\tau}P^\star_F(v_{i+1}\mid v_i),
\]

由 Assumption 3.1 有 \(Z=1\)。定义 \(\Pi_{u,x}:=\sum_{\tau:u\rightsquigarrow x}\mathbb P^\star(\tau)\)。两个边缘直接由约束读出：\(\sum_x\Pi_{u,x}=P^\star_F(u\mid s_0)=L(u)\)（第一步约束），\(\sum_u\Pi_{u,x}=R(x)\)（终止约束），所以 \(\Pi\) 是合法耦合。再算目标值（原文 Eq. (17)）

\[
\mathrm{GFlow}^\star=\sum_{s\to s'\in E^\circ}\mathcal F^\star(s\to s')=\sum_\tau|\tau|\,\mathbb P^\star(\tau),
\]

其中 \(|\tau|\) 是轨迹 \(\tau\) 落在 \(E^\circ\) 内的边数。最后用 \(|\tau_{u,x}|\ge d(u,x)\)（\(d\) 是最短路长）逐项放缩（原文 Eq. (18)）得 \(\mathrm{GFlow}^\star\ge\sum_{u,x}d(u,x)\Pi_{u,x}\ge\mathrm{OT}^\star\)（Eq. (19)）。与 Eq. (16) 合并即 Eq. (12)。

**这一半的实质**：策略采样出来的轨迹，其端点边缘化后必然是一个合法耦合，而任何实际走过的路都不比最短路短。所以最小化总流 = 迫使每一条被采样的路都退化成最短路。同时这说明最优 GFlowNet 给出的信息比耦合矩阵更多：不仅是「从 \(u\) 运多少质量到 \(x\)」，还有「具体经过哪些合法局部动作」。

#### 3.5 对偶与证书（Theorem 3.3 / Appendix A.2–A.4）

Proposition A.4 的推导干净：对 \(\mathcal F(s)\)（无符号约束）取下确界要求其系数为零，得 \(c=\bar\alpha+\bar\beta\)；对 \(\mathcal F(s\to s')\ge0\) 取下确界要求系数非负，得 \(\alpha_s+\beta_{s'}\ge0\) 与 \(\eta_x\le\alpha_x+\beta_{s_f}\)。由 \(c=\bar\alpha+\bar\beta\) 得 \(\alpha_{s_0}=0,\beta_{s_f}=0,\beta_s=1-\alpha_s\ (s\in\mathcal I)\)，代回边不等式化为 \(\alpha_{s'}-\alpha_s\le1\)，改名 \(\pi\) 即 Eq. (23)。

Theorem A.5 的三步也都是初等但关键的：

1. 沿任一 \(s_0\rightsquigarrow s\) 路径把 \(\pi_{v_{i+1}}-\pi_{v_i}\le1\) 相加，望远镜求和给 \(\pi_s-\pi_{s_0}\le k\)，取最小得 \(\pi_s\le d(s)\)；
2. \(d(\cdot)\) 可行，因为把最短路接上一条边给出 \(d(s')\le d(s)+1\)；
3. \(R>0\) 使目标严格单调，所以任意最优解在终止状态上必须取到上界 \(\pi^\star_x=d(x)\)。

**对偶变量的身份是 BFS 距离**，这一点值得直接说：Eq. (23) 的最优对偶势就是从 \(s_0\) 出发的跳数距离。互补松弛 \(\mathcal F^\star(s\to s')(\pi^\star_{s'}-1-\pi^\star_s)=0\) 于是把「最优流的支撑 \(\subseteq\) 最短路子图」变成一条可逐边检验的等式，这就是 O07 主定理的 LP 对偶版证明。Appendix A.4 进一步把扩展问题的对偶 \(\max_\pi\sum_xR(x)\pi_x-\sum_uL(u)\pi_u\) 通过 \(a_x=\pi_x,b_u=-\pi_u\) 与沿路径望远镜求和 \(a_x+b_u=\pi_x-\pi_u\le d(u,x)\)，映成 Kantorovich 对偶 Eq. (25)。

#### 3.6 训练目标（原文 Sec. 3.3，Eq. (20)）

不用 detailed balance + 流正则，而用带 leward 的正则化 trajectory balance（作者说 Morozov et al. 2026 已证 TB 在非无环寻路里更有效，他们的设定里也是 TB 最合适）：

\[
\mathcal L_{\mathrm{TB}}(\theta,\tau)=\left(\log\frac{L(s_1)\prod_{t=1}^{n_\tau}P_F(s_{t+1}\mid s_t,\theta)}{R(s_{n_\tau})\prod_{t=0}^{n_\tau-1}P_B(s_t\mid s_{t+1},\theta)}\right)^{2}+\lambda\,\frac{R(s_{n_\tau})}{P_F(s_f\mid s_{n_\tau},\theta)}.
\]

三个部件各有出处：分子里的 \(L(s_1)\) 顶替了通常 TB 里的 \(Z\)——因为 \(Z=1\) 且第一步分布被钉在 \(L\) 上，原文把它写成 leward matching \(P_F(s\mid s_0)=L(s)\)；分母的 \(R(s_{n_\tau})\) 承担 reward matching \(P_B(s\mid s_f)=R(s)\)；正则项 \(\lambda R(s)/P_F(s_f\mid s)\) 就是终止状态处的状态流 \(\mathcal F(s)\)（由 Eq. (5) 的 reward matching 条件推出），即最小总流目标的罚项实现。训练是 on-policy：每步用 \(P_F\) 采一批轨迹算损失。当每个内部状态都是终止状态（\(\mathcal X=\mathcal I\)）时，Eq. (20) 可对采样轨迹的每个前缀计算，样本效率更高（引 Morozov et al. 2026），实验采用了这个选项。

### 4. 实验与证据

两个环境，全部结果 3 个随机种子平均（原文 Sec. 4.1、Sec. 4.2 均声明 "averaged over three seeds"）。模型：2 隐层、宽度 128 的 MLP，输入状态 one-hot，\(\mathcal F_\theta(s)\)、\(P_F\)、\(P_B\) 共享 backbone、不同线性头；on-policy，batch 512，AdamW，\(\mathrm{lr}=10^{-3}\)，weight decay \(10^{-4}\)；**全部实验在 CPU 上完成**（Appendix B）。

#### 4.1 Hypergrid（原文 Table 1）

环境：格点 \(\{0,\dots,H-1\}^D\) 加 \(s_0,s_f\)；从 \(U\) 按 \(L\) 起步，转移为单坐标 \(\pm1\) 且不越界，每个状态都有终止转移（因此 \(\mathcal X=\mathcal I\)）。奖励 \(R\) 取角落多模态（corner-shaped，定义见 Appendix B.1，沿用 Bengio et al. 2021 / Madan et al. 2023 / Malkin et al. 2022 的形式），leward \(L\) 取 moon-shaped 与 ball-shaped（公式见 Appendix B.1）。TV 用 \(2\cdot10^5\) 个模型样本估计。\(\mathrm{OT}^\star\) 由 POT solver（Flamary et al. 2021）算出。

| \(L\) | \(H\) | \(\widehat{\mathrm{TV}}\downarrow\) | \(\mathrm{TV}^\star\downarrow\)（完美采样器参考） | \(\mathbb E|\tau|\) | \(\mathrm{OT}^\star\) |
|---|---|---|---|---|---|
| Ball | 10 | 0.024 ±0.0004 | 0.024 | 3.990 ±0.015 | 3.997 |
| Ball | 15 | 0.036 ±0.0009 | 0.033 | 6.325 ±0.011 | 6.303 |
| Ball | 20 | 0.037 ±0.0006 | 0.040 | 8.326 ±0.021 | 8.325 |
| Moon | 10 | 0.022 ±0.018 | 0.024 | 4.352 ±0.015 | 4.351 |
| Moon | 15 | 0.023 ±0.006 | 0.033 | 6.907 ±0.010 | 6.868 |
| Moon | 20 | 0.032 ±0.008 | 0.040 | 9.001 ±0.112 | 9.059 |

（数字全部照抄原文 Table 1。\(\mathrm{TV}^\star\) 非零是因为它是有限样本经验分布与真实分布之间的 TV。）

原文 Figure 1 另给 \(H=10,\ L=\mathrm{Moon}\) 的可视化：左图是 Eq. (11) 的精确 LP 解（`scipy.linprog`，Virtanen et al. 2020）在格子上的边流，右图是 Kantorovich 最优耦合的直连源—目标连线，两者**transport cost 都等于 4.351**（Figure 1 caption）。这张图是全篇最有说服力的一条证据：同一个数值，一边是边流、一边是耦合。

**证据强度判断**：\(H\in\{10,15,20\}\) 三档上 \(\mathbb E|\tau|\) 与 \(\mathrm{OT}^\star\) 的相对差都在 \(10^{-3}\)–\(10^{-2}\) 量级，同时 \(\widehat{\mathrm{TV}}\) 不高于完美采样器参考（\(H=20\) Ball 是 0.037 vs 0.040，Moon 是 0.032 vs 0.040）。所以「学到的策略同时满足目标边缘 + 达到 OT 最优代价」这个结论在小规模上是实验直接支持的。注意 \(H=20\) Moon 的 \(\mathbb E|\tau|=9.001<\mathrm{OT}^\star=9.059\)：这不是击败了 OT 下界，而是采样器没有精确满足边缘约束（TV 非零）时读数会偏低——原文没有讨论这一点，属于我的判断。

#### 4.2 排列（原文 Table 2）

环境：对称群 \(S_n\) 的 Cayley 图（沿用 T36）；内部状态是长度 \(n\) 的排列，转移是相邻对换 \(s(k)\leftrightarrow s(k+1)\)。\(L\) 取全部排列上的均匀分布，\(R(s)=\exp\!\big(\tfrac12\sum_{k=1}^n\mathbb I\{s(k)=k\}\big)/Z\)（T36 给出了该归一化常数的闭式）。诊断指标沿用 T36：\(C(k)\) 为奖励分布下「恰有 \(k\) 个不动点」的概率，报告模型经验估计与真值之间的 \(L^1\) 误差（Appendix B.2，用训练最后 \(10^5\) 个样本估计）。参考值：\(n=4\) 时 \(\mathrm{OT}^\star=0.567\)，\(n=8\) 时 \(\mathrm{OT}^\star=1.008\)（POT solver）；更大的 \(n\) 精确 \(\mathrm{OT}^\star\) 不可算。

| \(\lambda\) | \(n=4\) \(C(k)L^1\downarrow\) | \(n=4\) \(\mathbb E|\tau|\) | \(n=8\) \(C(k)L^1\downarrow\) | \(n=8\) \(\mathbb E|\tau|\) | \(n=20\) \(C(k)L^1\downarrow\) | \(n=20\) \(\mathbb E|\tau|\) |
|---|---|---|---|---|---|---|
| \(10^{-1}\) | 0.012 ±0.001 | 0.445 ±0.002 | 0.011 ±0.005 | 0.645 ±0.013 | 0.016 ±0.000 | 2.313 ±0.018 |
| \(10^{-2}\) | 0.002 ±0.000 | 0.557 ±0.001 | 0.001 ±0.002 | 1.010 ±0.011 | 0.002 ±0.000 | 4.436 ±0.014 |

（数字照抄原文 Table 2。）

**\(\lambda\) 的权衡在这张表里是定量的**：\(\lambda=10^{-2}\) 时 \(n=4\) 的 \(\mathbb E|\tau|=0.557\) 对 \(\mathrm{OT}^\star=0.567\)、\(n=8\) 的 \(1.010\) 对 \(1.008\)，同时 \(C(k)L^1\) 降到 0.002/0.001；\(\lambda=10^{-1}\) 时路径大幅变短（0.445、0.645）但 \(C(k)L^1\) 升到 0.012/0.011，说明短出来的部分是靠牺牲终止边缘换的。原文 Sec. 4.2 的表述与 T36 一致：\(\lambda\) 越大轨迹越短、采样越有偏，越小则采样准确、轨迹更长。

**证据强度判断**：

- 实验直接支持的：小规模（hypergrid \(H\le20\)、排列 \(n\le8\)）上学习法能同时逼近 \(\mathrm{OT}^\star\) 与目标边缘；LP 精确解与 Kantorovich 精确解数值相同（Figure 1）；\(\lambda\) 的方向性权衡。
- 作者推断、实验未直接验证的：\(n=20\) 时「produces a reasonable approximation」——这一档没有 \(\mathrm{OT}^\star\) 参考值，只能看 \(C(k)L^1=0.002\) 说明采样边缘正确，路径最优性无 ground truth；「可扩展性」只由 \(n=20\) 一个点支撑。
- 原文未给出的：与 network simplex / Sinkhorn / 神经 OT 的 wall-clock 与内存对比；耦合矩阵层面的误差（如 \(\|\Pi_\theta-\Pi^\star\|_1\)）；primal-dual gap 的实测；学习率外的超参搜索范围；\(H,D\) 的具体取值（Table 1 只给 \(H\)，正文写状态空间是 \(\{0,\dots,H-1\}^D\) 但**没给 \(D\)**）。

### 5. 前提假设与适用边界

写成正向的适用范围。**Theorem 3.2 在下面这组条件下成立**：

1. **有限有向图**，允许有环。有环是必需的：无环分层图（如单向 hypergrid）上到某个 \(x\) 的所有路径等长，最小流原理在其上退化为常数，OT 目标里没有可优化的自由度（原文 Assumption 3.1 只要求有限 + 可达，但流理论本身需要非无环框架，见 T36）。
2. **端点结构**：\(s_0\) 无入边、\(s_f\) 无出边；\(s_0\) 只出到 \(U\)、\(s_f\) 只从 \(\mathcal X\) 入（Assumption 3.1 前两条）。这条把「源集」与「终集」在图上物理隔离，是 divergence 约束右端能写成 \(L(s)-R(s)\) 的原因。
3. **全对可达**：任意 \(u\in U\) 与 \(x\in\mathcal X\) 之间有有限路径（Assumption 3.1 第三条）。否则 \(d(u,x)=\infty\)，耦合集里的质量无路可走。
4. **两个边缘都是归一化概率分布**：\(\sum_uL(u)=\sum_xR(x)=1\)（Assumption 3.1 第四条），从而 \(Z=1\) 已知。原文自己说这条同时来自约束的必要推论（流匹配约束求和）与实现便利。
5. **单位边长的最短路 cost**：\(d(u,x)=|\tau_{u,x}|\) 数的是边数（Eq. (7)），即无权图跳数距离。带权图不在定理覆盖范围内。由于图有向，\(d\) 未必对称，它是合法的 OT ground cost 但不必是通常意义的度量。
6. **吸收性 / 有限期望长度**：流的期望访问次数语义要求反向链是吸收链（T36 Lemma 3.4；O07 用更弱的 Assumption 3.1 \(\mathbb E[n_\tau]<\infty\) 替代 \(P_B>0\)）。原文正文没有重述这一条，但它是 Eq. (3) 良定义的前提。
7. **精确流守恒 + 全局最优**：Theorem 3.2 是关于 LP 最优解的陈述。神经训练用的是 Eq. (20) 的软罚（系数 \(\lambda\)），不等于精确约束满足，所以「训练完的模型是 OT 最优」这一步在实验层面靠数值接近程度支撑，不由定理保证。
8. **Theorem 3.3 的对偶最优性还额外需要 \(R(x)>0\ \forall x\in\mathcal X\)**（第三条结论）；这是终止状态处对偶势唯一确定为 \(d(x)\) 的条件。

**适用范围一句话**：适合「状态空间巨大到无法枚举、cost matrix 无法显式构造、但局部合法动作明确且允许回退」的组合图上的均衡 OT——排列/群、分子编辑、程序变换、机器人配置空间。不适合小规模显式图（经典 network simplex 更快更准且自带证书）、带权/连续代价、以及质量不等的 unbalanced OT。

### 6. 在 GFlowNet × OT 主线中的位置

**前驱（依赖链，缺一不可）**：

- **T19**（Brunswic et al. 2024, AAAI）：非无环 GFlowNet 的测度论框架、0-flow 与损失稳定性、流正则化思想。本文的 hypergrid 环境直接取自它（原文 Sec. 4.1）。
- **T36**（Morozov et al. 2025, ICML）：本文正文 Sec. 2.1 的记号、流定义、\(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\)（Proposition 3.12）、\(\lambda\mathcal F_\theta(s)\) 正则、排列环境与 \(C(k)\) 诊断协议全部来自它。没有 T36 的期望访问次数流，本文的目标函数写不出来。
- **O07**（Morozov et al. 2026，本仓库 O07，arXiv 2603.01786）：证明最小总流 ⇒ 只走最短路。本文 Theorem 3.3 用 LP 对偶 + 互补松弛重新导出这一结论（原文明说 "recovers the corresponding claim"），并把 TB 目标、每前缀计算的技巧也继承过来。
- **OT 侧**：Beckmann (1952) 的连续运输模型、Essid & Solomon (2018)（本仓库 O02）的图上二次正则 OT——本文的 Assumption 3.1 与 min-cost-flow 表述都自称仿照后者；Villani (2008) Theorem 4.1 提供最优耦合存在性。

**后继与对照**：

- **C01 / ULOT**（arXiv 2506.12025, NeurIPS 2025）：GNN + cross-attention 摊销预测图间 FUGW（fused unbalanced Gromov-Wasserstein）plan，条件于 tradeoff 超参，比经典 solver 快两个数量级，还能给 solver 做 warm start。它与本文的分工是清晰的：ULOT 输出**耦合矩阵**、面向两张显式给定的图做节点对齐，且原生 unbalanced；本文输出**可执行的局部路由策略**、面向一张隐式的巨大转移图，且要求边缘归一化。摊销/条件化这条路 ULOT 已占位，本文若走「条件 GFN 学一族 \((L,R)\)」会直接撞车。
- **C02 / GSBoG**（arXiv 2602.04675）：图上广义 Schrödinger bridge，学 CTMC 控制策略，固定两端边缘、同时优化 state-dependent running cost，用 IPF + TD 目标，声称是首个图上 GSB 表述。它与本文最相邻：都要「端点边缘 + 中间代价 + 可执行策略」。差别在三处——GSBoG 是**连续时间** CTMC 且带 KL 到参考过程的熵正则（entropic OT 家族），本文是**离散步数**且无熵项（纯 LP，最优解在多面体顶点上、可退化为确定性路由）；GSBoG 的代价是 state cost 泛函，本文的代价是图跳数；GSBoG 靠 IPF/TD，本文靠 TB + 流正则。想在 GFN 侧加熵正则化以对接 Schrödinger bridge，必须正面回答「TB 目标与路径空间 KL 的关系」，这不是自动成立的。
- **C03 / DDSBM**（arXiv 2410.01500, ICLR 2025）：离散扩散 SB 做图变换，同属 entropic 阵营。

**对主线的贡献**：它给出了「内部流选哪一个」的第一条几何原则。标准 reward matching 只钉住终止分布，内部流严重欠定（这个欠定会影响信用分配与未见状态泛化）。本文的答案是：**在所有正确终止分布的流中，选运输代价最小的那一个**，而这个选择等价于一个显式的 OT 问题，因此可以借用 OT 的全套工具（对偶势、互补松弛、primal-dual gap）来分析和验证 GFlowNet 训练。反过来，它也把 GFlowNet 送进了计算 OT 的地盘：在无法显式构造 \(C_{ux}\) 的隐式组合图上，GFlowNet 只需访问局部邻居。

### 7. 可复用的 insight 与开放问题

#### 7.1 新意到底在哪：不是等价本身，是接口

「图上以最短路为 ground cost 的 OT ≡ min-cost network flow」是经典结果，图 OT 文献长期这样表述（原文自己在 Eq. (8) 后引 Essid & Solomon 2018 承认这一点，并把它同时挂到 Beckmann 1952）。所以 Theorem 3.2 的数学内容对 OT 研究者不是新闻：Part 1 是「沿最短路铺质量」，Part 2 是「实际路径不短于最短路」，两步都是本科级论证。

新意在三处，都在 GFlowNet 一侧：

1. **翻译**：把 min-cost flow 的 primal 变量换成 GFlowNet 的边流、约束换成 detailed balance + reward/leward matching，于是 OT 问题第一次可以用「采样器训练」的方式求解。Appendix A.1 的 Proposition A.1/A.2 是这层翻译的技术粘合剂——它说明 detailed balance 与流匹配在策略可重构的意义上是同一约束。
2. **输出物升级**：经典 min-cost flow 给出边流，需要额外的流分解才能得到路径级策略；GFlowNet 直接给出 \(P_F(s'\mid s)\)，可以在不枚举 \((u,x)\) 对、不存储 \(|U|\times|\mathcal X|\) 矩阵的情况下逐步执行运输。原文 Sec. 5 的自我定位说得准确：「addressing both where and how transport occurs」。
3. **把最短路结论抬升为 OT 结论**：O07 是单源（\(s_0\) 一点）到多终点的最短路；本文加一条 \(\mathcal F(s_0\to u)=L(u)\) 就变成多源—多终点，OT 是最短路的自然母题。这一步几乎不花技术成本，但把整个问题类换了。

**容易被误读的地方**（也是我认为最重要的一条）：Theorem 3.2 说的是**最优值相等 + 最优解诱导最优耦合**，不是「任意训练好的 GFlowNet 都在做 OT」。标题里的 "secretly" 只在两条额外约束下成立：(i) 第一步边流被钉在 \(L\) 上，(ii) 内部总流被最小化。少了 (i)，第一步分布由 \(P_F(\cdot\mid s_0)\) 自由决定，问题退化成 O07 的单源最短路；少了 (ii)，任意满足 reward matching 的流都合法，可以把质量无限堆在环里（T19 的 flow explosion），跟 OT 毫无关系。普通 GFlowNet 并不「秘密地」学 OT plan。

#### 7.2 Balance 残差 → OT 误差界（我认为最值得做的方向，撞车风险最低）

**现状**：论文只给最优解处的等式，完全没有近似解的定量刻画。训练时你有的是 TB 残差 \(\delta_\tau=\log\frac{L(s_1)\prod P_F}{R(s_{n_\tau})\prod P_B}\)（Eq. (20)）；你想要的是两件事的界：源/目标边缘误差，以及运输代价的次优性 \(\mathrm{cost}(\Pi_\theta)-\mathrm{OT}^\star\)。

**为什么可做**：这个设定比一般 GFlowNet 更有利。第一，\(Z=1\) 已知，TB 残差里没有未知常数（这正是标准 TB→TV 界最难处理的项）。第二，两个边缘都是硬约束，残差同时携带「起点边缘误差」和「终点边缘误差」两类信息。第三，代价函数是 \(\mathbb E_{\mathbb P_\theta}[|\tau|]\)，是轨迹长度这个可直接估计的量，而不是需要额外模型的抽象泛函。

**草稿命题**：设 \(\mathbb P_\theta\) 是模型轨迹分布，\(\Pi_\theta\) 是其端点边缘化耦合，\(\varepsilon_1=\mathrm{TV}(\text{first-step marginal},L)\)、\(\varepsilon_2=\mathrm{TV}(\text{terminal marginal},R)\)、\(D=\max_{u,x}d(u,x)\)（图直径级量）。则

\[
\Big|\mathbb E_{\mathbb P_\theta}[|\tau|]-\mathrm{OT}^\star\Big|
\ \le\
\underbrace{\big(\mathbb E_{\mathbb P_\theta}[|\tau|]-\textstyle\sum_{u,x}d(u,x)\Pi_\theta(u,x)\big)}_{\text{绕路松弛，}\ \ge 0}
\;+\;
\underbrace{C\,D\,(\varepsilon_1+\varepsilon_2)}_{\text{边缘违反的代价}},
\]

第一项由 Eq. (18) 的放缩直接得到（它天然非负，且可以用「实际路长 − 端点最短路长」的样本均值无偏估计）；第二项需要 OT 值关于边缘的稳定性（Lipschitz-in-marginals），在有限图 + 有界代价下是标准结果。然后把 \(\varepsilon_1,\varepsilon_2\) 用 TB 残差的二阶矩控住，就得到一条端到端的「残差 → OT gap」界。**这条界能直接解释 4.1 节里 \(\mathbb E|\tau|<\mathrm{OT}^\star\) 的现象**：读数偏低正是被 \(\varepsilon_2\ne0\) 换来的，界的两项符号相反。撞车扫描的结论也支持这个方向：未见直接竞品，而 ULOT/GSBoG 都不在这条线上。

#### 7.3 primal-dual：对偶势当 critic，互补松弛当逐边证书

Theorem 3.3 / Theorem A.5 已经把材料备齐了，但论文完全没用于算法：

- 对偶势 \(\pi_s\) 的最优值就是 BFS 距离 \(d(s)\)，可以用一个网络 \(\pi_\phi(s)\) 拟合，约束 \(\pi_{s'}-\pi_s\le1\) 是逐边的、可采样惩罚的——这与 DeepCubeA / CayleyPy 那类「学 distance-to-goal」的做法在函数上完全重合，但这里它有明确的 LP 对偶身份，不是启发式。
- **互补松弛给出免费的逐边证书**：\(\mathcal F^\star(s\to s')(\pi^\star_{s'}-1-\pi^\star_s)=0\)。训练中可以监控 \(\sum_{s\to s'}\mathcal F_\theta(s\to s')\cdot|\pi_\phi(s')-1-\pi_\phi(s)|\) 作为 primal-dual gap 的代理，它对每条边都可评估，不需要全局 solver。这是 GFlowNet 训练里罕见的、可计算的最优性指标（TB 残差为零只保证一致性，不保证最优性）。
- 具体实验：actor = \((P_F,P_B)\)，critic = \(\pi_\phi\)，损失 = 正则化 TB + 对偶可行性罚 + gap 罚。可测量的假设是「加 critic 后在排列 \(n=20\) 上 \(\mathbb E|\tau|\) 更低且 \(C(k)L^1\) 不变差」。注意扩展问题的对偶（Appendix A.4）里 \(a_x=\pi_x,\ b_u=-\pi_u\) 由同一个势函数生成，所以只需一个网络，不是两个。

#### 7.4 与 ULOT（arXiv 2506.12025）的差异化：不要去抢摊销预测

ULOT 的卖点是：GNN + cross-attention，条件于 FUGW 的 \((\rho,\alpha)\) 超参，\(O(n_1n_2)\) 复杂度直接输出 plan，比经典 solver 快两个数量级，plan 对输入图可微，还能 warm-start IBPP solver。它天生 unbalanced、天生条件化、天生摊销。

因此「训练一个条件 GFlowNet 覆盖一族 \((L,R)\)」这个看起来最顺的下一步，其价值主张与 ULOT 高度重叠，且 GFlowNet 在**两张显式给定的中小图之间做节点对齐**这件事上没有结构优势——需要 rollout、方差更大、还得处理环。

差异化只有一条护城河，必须守住：**图是隐式的、cost matrix 无法构造、传输只能通过合法局部动作执行**。排列 \(n=20\)（\(\sim2.4\cdot10^{18}\) 状态，见 O07 Sec. 4.1）已经在这个区间；分子编辑、程序/证明变换、机器人配置空间也是。可比实验应该是：在 ULOT 能跑的 SBM 图上承认打不过（或平手），但在 \(|U||\mathcal X|\) 无法枚举的 Cayley 图上，ULOT 根本无法构造输入。把这条边界写清楚比刷点更有说服力。

#### 7.5 与 GSBoG（arXiv 2602.04675）的差异化：熵正则不是自动送的

GSBoG 解的是图上广义 Schrödinger bridge：固定两端边缘，在路径空间上最小化 \(\mathbb E[\int f_t\,\mathrm dt]+\mathrm{KL}(p^u\|p^r)\)，控制对象是 CTMC 的跳率，训练用 IPF（Sinkhorn 的路径版）+ 一个 TD 目标（因为 state cost 项在 IPF 聚合里会正负相消，光靠 IPF 不足以约束它）。

三个结构差异决定了两者不可直接互换：

1. **有无熵项**。本文是无正则 LP，最优解可以是顶点解、可退化为确定性路由（Theorem 3.3 说流只落在最短路子图上）。GSBoG 的 KL 项让解严格随机、支撑铺满可行路径。要把 GFN-OT 变成 entropic 版本，需要在目标里显式加路径 KL 到参考策略 \(P^0\)，而**标准 TB 并不自动等价于路径空间 KL 最小化**——TB 是一致性约束，不是散度目标；这一步必须重新推导（这也是本仓库背景分析里已标注的高撞车但未解决的技术点）。
2. **时间参数化**。GSBoG 有连续时间 \(t\in[0,1]\) 与时间边缘；本文只有离散步数、没有「在 \(t=0.5\) 时质量在哪」这个概念。想比较中间行为，得先给 GFN 侧引入时间条件（或步数条件），这本身是个建模选择。
3. **代价的表达力**。GSBoG 支持 state- 与 distribution-dependent running cost（含 mean-field 项）；本文只有跳数。**GFN 侧最容易吃下的扩展是把 \(\lambda\mathcal F_\theta(s)\) 换成 \(\lambda c(s)\mathcal F_\theta(s)\)**，即状态相关的流罚，此时 LP 目标变成 \(\sum_s c(s)\mathcal F(s)\)、ground cost 变成加权最短路——严格正 \(c\) 时 Theorem 3.2 的两半论证几乎逐字成立（Part 1 沿加权最短路铺质量，Part 2 用 \(\mathrm{cost}(\tau_{u,x})\ge d_c(u,x)\) 放缩）。零权边会引入零代价环、破坏最优流的无环性，必须排除。这是一个低风险、高确定性的定理级增量。

#### 7.6 unbalanced 与 unknown-\(Z\)：GFlowNet 放弃了自己最强的能力

标准 GFlowNet 的招牌是只需未归一化 \(R\)。本文为了让 min-cost flow 的质量平衡成立，要求 \(\sum_uL(u)=\sum_xR(x)=1\)（Assumption 3.1），并在 Sec. 3.1 明确说这是为了避免处理未知归一化常数。代价是：Eq. (20) 里 \(L(s_1)\) 顶替了 \(Z\)，模型再也不学 \(Z\)。

开放问题很具体：把汇 \(s_f\) 与源 \(s_0\) 之间加一条「弃货/造货」边并赋予单位代价 \(\kappa\)，LP 就变成带 penalty 的 unbalanced OT（相当于 \(\ell_1\) 型边缘松弛）。此时质量平衡由虚拟边吸收，\(L,R\) 不必等质量。需要回答的是：(i) Theorem 3.2 的两半论证在虚拟边存在时如何改写（Part 1 要把「没运出去的质量」显式路由到虚拟边）；(ii) 训练目标里 \(Z\) 重新变成未知量后，TB 是否还稳定；(iii) 与 ULOT 的 FUGW-unbalanced 相比，GFN 的优势是否仍只在隐式图上。这是把 GFlowNet 的原生能力接回 OT 的关键一步，目前完全空白。

#### 7.7 短流是否有利于泛化：尚未验证，且可能是负面的

本文把「最短」当作内部流的选择原则，但没有测过它对泛化的影响。最小流解会把质量集中到最短路子图（Theorem 3.3 的互补松弛），支撑集可能非常窄。这有两个可测的后果，方向相反：正面是 rollout 更短、终端信用更直接、推断更便宜；负面是探索坍缩、子结构共享变少、对未见状态的策略外推变差。

可执行实验：在排列环境上按 \(\lambda\) 扫描（沿用 Table 2 的两档并加密），除 \(C(k)L^1\) 与 \(\mathbb E|\tau|\) 外额外报告：被访问状态的覆盖率、终止状态支撑的熵、以及在训练时未见状态上的贪心成功率（O07 Table 1 的评测协议可以直接搬）。假设是「\(\lambda\) 与泛化呈倒 U 形」——如果成立，「最小流」就不能无条件当作正确的内部流原则，而要写成带正则的版本。

#### 7.8 其他开放问题（简）

- **有向不对称的后果**：\(d(u,x)\ne d(x,u)\) 时 Theorem 3.2 仍成立（证明未用对称性），但由此得到的「距离」不是度量，所以不能直接称 Wasserstein。若下游要用它做距离比较，需要单独论证。
- **精确 LP 与神经解的桥**：论文用 `scipy.linprog` 验证小规模等价性（Sec. 4.1、Figure 1）。中间规模（比如 \(H=30,D=3\)）上应该做「LP 解 vs 神经解」的边流层面比较（不只是标量代价），才能判断神经解是找到了另一个同代价最优解，还是在同一个解附近。
- **\(D\) 未报告**：Table 1 只给 \(H\)，正文写状态空间是 \(\{0,\dots,H-1\}^D\)，但**原文未给出 \(D\) 的取值**；Figure 1 的可视化是二维格子，可以推测该图对应 \(D=2\)，但表格三档是否同一 \(D\) 无法从原文确认。复现时必须先定这个数。

### 8. 引用

```bibtex
@article{maksimov2026gflownet_ot,
  title  = {Your GFlowNet Secretly Learns an Optimal Transport Plan},
  author = {Maksimov, Ian and Morozov, Nikita and Belomestny, Denis and Samsonov, Sergey},
  journal = {arXiv preprint arXiv:2606.06272},
  note   = {ICML 2026 SPIGM Workshop (workshop paper, not main conference)},
  year   = {2026},
  url    = {https://arxiv.org/abs/2606.06272}
}
```

相关引用（本报告用到的对照工作）：

```bibtex
@article{morozov2026shortest,
  title  = {Learning Shortest Paths with Generative Flow Networks},
  author = {Morozov, Nikita and Maksimov, Ian and Tiapkin, Daniil and Samsonov, Sergey},
  journal = {arXiv preprint arXiv:2603.01786},
  note   = {ICML 2026 SPIGM Workshop; repository ID O07},
  year   = {2026}
}

@inproceedings{morozov2025revisiting,
  title  = {Revisiting Non-Acyclic GFlowNets in Discrete Environments},
  author = {Morozov, Nikita and Maksimov, Ian and Tiapkin, Daniil and Samsonov, Sergey},
  booktitle = {ICML}, series = {PMLR}, volume = {267}, pages = {44887--44910},
  year   = {2025}, note = {repository ID T36}
}

@article{essid2018quadratically,
  title  = {Quadratically Regularized Optimal Transport on Graphs},
  author = {Essid, Montacer and Solomon, Justin},
  journal = {SIAM Journal on Scientific Computing}, volume = {40}, number = {4},
  pages  = {A1961--A1986}, year = {2018}, note = {repository ID O02}
}
```

---

### 编者注（自主决定的判断，与原文区分）

1. **发表状态**：PDF 页脚只写 "Preprint. June 5, 2026"，正文没有 venue 行。按仓库 `data/papers.yaml` 登记的 `ICML 2026 SPIGM Workshop` 记为 Workshop 论文（非主会），报告表头与 bibtex 的 `note` 都已写明这一区分。
2. **定理编号**：原文正文的主定理编号是 **Theorem 3.2**（不是 3.1，3.1 是 Assumption），对偶结论是 **Theorem 3.3**，其完整版在 Appendix A.3 标为 **Theorem A.5**；原文 Sec. 3.2 末写「Detailed proof of Theorem 3.3 can be found in Appendix A.5」，而 Appendix 里该定理位于 **A.3** 节、编号为 A.5，这是原文的小笔误，本报告按实际内容引用。另有一处：Appendix A.4 开头写「see Appendix A.4 for derivation」自引，同属笔误。
3. **§3.2 关于常数 1 的强调**是我的判断（原文只写了一句 "we will omit this constant"），因为它决定了 \(\mathrm{GFlow}^\star\) 与 \(\mathbb E[n_\tau]\) 的关系，是复现时最容易搞错的地方。
4. **§4.1 关于 \(\mathbb E|\tau|<\mathrm{OT}^\star\) 的解释**（\(H=20\) Moon：9.001 vs 9.059）原文未讨论，是我基于边缘未精确满足给出的推断，已在正文标注。
5. **§7.2 的界是我写的草稿**，不是论文结论；其中第一项来自原文 Eq. (18) 的放缩方向，第二项的 Lipschitz-in-marginals 属于标准 OT 稳定性、需要单独引文补齐。
6. **§7.5 的加权扩展**（\(\sum_sc(s)\mathcal F(s)\)）是我的推断，原文只在 Sec. 5 提到未来工作里未涉及此项；O07 Sec. 5 把带权图列为 future work，可视为同一方向的作者背书。
7. **代码**：原文正文与附录均未给出本文的代码链接（只说实现基于 T36 的公开代码），故元数据 `code_url` 记为 `null`。
8. **超网格维度 \(D\)**：原文未给出，报告中已按「原文未给出」处理，未做任何填补。


# 第 6 章 最优传输侧：先修与相邻工作

## 6.1 O01 · Optimal Transport for Machine Learners（Peyré 讲义）—— 面向 GFlowNet 读者的导读

> **一句话**：Peyré 这份 460 页讲义是本仓库的 OT 侧「词典」。
> 对 GFlowNet 读者而言，它真正不可替代的不是 Wasserstein 距离或 Sinkhorn 本身，
> 而是 **§6.5** 把 \(\mathcal W_1\) 写成「图上的边流 + 顶点守恒」这一形式——
> 那就是 GFlowNet 的对象，逐字逐句。
> 本导读的任务是给出一张「OT 概念 ↔ GFlowNet 概念」对照表，
> 并且**明确标出哪几条是严格等价、哪几条只是类比**，避免把「GFlowNet 在做 OT」当成一句无条件的口号。

| 字段 | 内容 |
|---|---|
| arXiv | [2505.06589](https://arxiv.org/abs/2505.06589)（本仓库副本为 v3, stat.ML, 2026-08-08；扉页署 August 11, 2026） |
| 发表 | **讲义／书稿（lecture notes）**，非会议非期刊。作者自述定位为「以 ML 为组织压力的 OT 教材」 |
| 作者 | Gabriel Peyré（CNRS and ENS, PSL Université） |
| 代码 | 全部图表代码见 `gpeyre/ot4ml`；多数计算图用 Python Optimal Transport (POT) 库生成（摘要页明确致谢） |
| 本仓库 PDF | `papers/2505.06589.pdf`（480 个 PDF 页 / 正文 16 章 + 结论 + 记号表 + 索引，正文页码到 460 附近） |
| 阅读优先级 | P0 —— 但**不要通读**。按第 2 节的章节地图取用；全书只有约 40 页与本仓库主线直接相关 |

#### 2. 章节地图：GFlowNet 读者该读哪几章、每章解决什么

全书 16 章。按与本仓库主线的相关度分四档。**A 档共约 40 页，是全部必读内容。**

| 档 | 章节 | 起始页 | 这一章解决什么 | 为什么 GFlowNet 读者需要 |
|---|---|---|---|---|
| **A** | **§6.5 Wasserstein-1** | p.108 | 把 \(\mathcal W_1\) 从「点对耦合」改写成「边上的流 + 顶点守恒」：Kantorovich–Rubinstein 对偶 Eq. (6.23)、Beckmann 问题 Def. 6.20、**图版本 Def. 6.22 + Prop. 6.23** | 这是全书唯一一处 OT 的变量就是**图上的边流**、约束就是**每个顶点的守恒**的地方。与 GFlowNet 的 flow matching 条件是同一个方程 |
| **A** | **§5.1–5.3 Dual Problem** | p.84 | 对偶势 Def. 5.1、强对偶 Prop. 5.2/5.5、**互补松弛 Prop. 5.3/5.7**、\(c\)-变换 Def. 5.8 与其正则化作用 Prop. 5.9/5.10 | 「状态流 \(\log F(s)\) = 对偶势/价值函数」这个读法的严格出处。互补松弛解释了「最优流只能活在接触集上」= 稀疏支撑 |
| **A** | **§14.3 Path-Space Schrödinger** | p.314 | 路径空间传输 Def. 14.8、**端点约简 Prop. 14.9 / Prop. 14.11**、Schrödinger 桥 Def. 14.10、KL 链式分解 Eq. (14.20) | 唯一严格处理「轨迹分布 vs 端点耦合」关系的地方。直接回答「GFlowNet 的路径熵正则和 OT 的端点熵正则差在哪」 |
| **A** | **§3.1 Discrete Relaxation** | p.42 | 耦合多面体 Def. 3.1、LP 结构 Def. 3.5、**稀疏最优解 Prop. 3.8**、Birkhoff–von Neumann Thm. 3.17、有理权重的整数化 Prop. 3.21 | 「流是欠定的、可行集是多面体、最优解可取稀疏顶点」这一整套语言。Remark 3.2 给出单源退化的精确说明 |
| **A** | **§9.1 Bregman 视角** | p.157 | Sinkhorn = 在两个仿射边缘约束集上做**交替 KL 投影**：Eq. (9.1)、Algorithm 9.1、Prop. 9.4、**Prop. 9.5（投影就是行/列缩放）** | 把「交替强制两个约束」这一算法骨架讲清楚。T10 的两阶段 GTB 与之同构 |
| **B** | §8.1–8.5 熵正则与 Sinkhorn | p.123 | 熵正则 Def. 8.2、缩放形式 Prop. 8.5、KL 重写 Eq. (8.8)、\(\varepsilon\) 的两个极限 Prop. 8.10、对偶 Prop. 8.18、**soft-\(c\)-变换 Def. 8.19** | soft-min = log-sum-exp 与 GFlowNet 的 log 域损失同形；Prop. 8.10 给出「熵越大越趋向独立耦合」的定量版本 |
| **B** | §3.2 LP 算法 | p.53 | 运输单纯形、网络单纯形、Orlin 强多项式最小费用流、内点法 Eq. (3.5) | 说明「图上最小费用流」有成熟精确解法，是评估 GFlowNet 近似解质量的基线 |
| **B** | §14.1–14.2 动态 OT | p.308 | 连续性方程 Def. 14.1、**Benamou–Brenier Thm. 14.5**、动量凸化 Def. 14.6 | 「流 = 守恒律 + 最小动能」的连续版；帮助理解为何 GFlowNet 的离散对应物是 Beckmann 而不是 BB |
| **B** | §2.3 Monge 形式 | p.16 | Monge 问题 Def. 2.14、经验测度上 Monge 映射就是置换 Prop. 2.15、**质量不可分裂的障碍 Example 2.18** | 解释为什么必须用「计划」而不是「映射」——GFlowNet 的随机策略天然是计划侧 |
| **B** | §4.1 Wasserstein 距离 | p.64 | Def. 4.2/4.6、粘合引理 Lemma 4.4、三角不等式 Prop. 4.3/4.7 | 需要用到 \(\mathcal W_p\) 作为评估指标时的最小知识 |
| **C** | §1.1–1.2 指派问题 | p.1 | Def. 1.1、一维排序最优 Prop. 1.2、匈牙利法 Prop. 1.8、对偶证书 Prop. 1.7 | 建立「离散 OT 本质是组合优化」的直觉，可跳读 |
| **C** | §12.5 度量学习与逆 OT | p.258 | 从观测到的耦合反推代价 | 与 T10 的 guided TB「学一个引导分布 = 学一个代价」是同一件事的 OT 版本 |
| **C** | §11.1 Unbalanced OT | p.201 | 放松边缘约束 | 对应 GFlowNet 里奖励不归一、或允许「泄漏流」的变体 |
| **C** | §15.1 Wasserstein 梯度流 | p.335 | JKO 隐式欧拉 Def. 15.1、Wasserstein 梯度 Def. 15.5、Eq. (15.4) | **只需知道存在这个框架**。它描述的是测度空间上的梯度下降，与 GFlowNet 的参数空间训练不是一回事，不要过度类比 |
| **D（跳过）** | §7 散度与对偶范数、§10 统计 OT、§11.2–11.6、§12.1–12.4、§13、§16 | — | MMD/GAN、样本复杂度、切片/低秩/多边缘/Gromov–Wasserstein、流匹配生成模型 | 与「离散图上的流选择」这条主线无直接关系 |

**读法建议。** 先读 §6.5（约 5 页）看懂「OT 在图上长什么样」，
再回头读 §3.1 补齐多面体语言，然后 §5.1–5.3 拿到对偶势，
最后 §14.3 处理轨迹 vs 端点的区别。§9.1 在需要设计训练算法时再看。

#### 5. 前提假设与适用边界

讲义本身是数学教材，前提写得很清楚。对 GFlowNet 读者，以下几条决定了对照表能推到多远。

1. **紧性与连续性**。连续对偶（Prop. 5.5）在 \(\mathcal X,\mathcal Y\) 紧、\(c\) 连续下给出 max 可达；
   一般情形要退到下半连续 + 可积假设，且 max 变 sup。离散有限图上这些全部自动满足。
2. **平衡质量**。第 3–9 章全是 \(\alpha,\beta\) 同为概率测度的**平衡** OT，放松边缘要到 §11.1（Unbalanced OT）。
   GFlowNet 里若允许「流泄漏」或奖励不归一，对应的就是 unbalanced 那一支，不能直接套 Prop. 6.23。
3. **代价的正性/有限性**。Prop. 8.3 与 Prop. 8.5 需要 \(\mathbf C\) 有限、\(\mathbf K>0\)；
   若 \(c\) 可取 \(+\infty\)（即某些边不存在），存在唯一性需要额外的支撑条件（§8.2 明确指出）。
   **有向图上不可达的状态对就是 \(c=+\infty\) 的情形**，这条限制在 GFlowNet 应用中是实打实的。
4. **图 Beckmann 需要连通、边长为正、\(\sum_i r_i=0\)**（Prop. 6.23 证明中用到）。
   有向图上的版本讲义没有给——Def. 6.22 的 \(d_G\) 是无向图测地距离，
   Remark 6.24 是把每条无向边拆成两条有向弧后的转运问题。
   **GFlowNet 的 DAG 是有向且不可逆的，这不是 Prop. 6.23 覆盖的情形**，需要自己补。
5. **\(p=1\) 才有 Beckmann/流形式**。\(\mathcal W_1\) 的特殊性来自 Prop. 6.17：
   \(c=d\) 时 \(c\)-变换的像恰是 1-Lipschitz 函数，且 \(f^c=-f\)，对偶塌缩成单个势。
   \(p=2\) 时对应的是 Benamou–Brenier（Thm. 14.5），那是**时间连续**的流，不是图上的静态边流。
   把 GFlowNet 类比到 \(\mathcal W_2\) 需要额外论证。
6. **Sinkhorn 的收敛结论是「固定 \(\varepsilon\)、固定边缘」的算法收敛**（Ch. 9 开头明确区分）；
   统计收敛（边缘随样本变化）是 Ch. 10 的独立问题。
7. **熵正则改变问题本身，不只是算法**（Fig. 3.8 说明的对比）：
   内点法的 \(\varepsilon\) 是障碍参数、沿中心路径趋于 0；
   Sinkhorn 的 \(\varepsilon\) 通常固定，是**目标函数的一部分**。混淆这两者会得出错误的极限结论。

#### 6. 在 GFlowNet × OT 主线中的位置：概念对照表

**判定口径**：
「**严格**」= 两侧是同一个数学对象，可以逐符号翻译，且讲义中有对应命题；
「**条件严格**」= 在明确附加条件下严格，条件已写出；
「**类比**」= 结构同构但不是同一个对象，不能直接搬定理。

| # | OT 概念 | 讲义出处 | GFlowNet 概念 | 判定 | 说明 |
|---|---|---|---|---|---|
| 1 | 耦合 \(\mathbf P\in\mathcal U(\mathbf a,\mathbf b)\) | Def. 3.1, Eq. (3.1); Def. 3.5, Eq. (3.2) | 轨迹分布 \(P_F(\tau)\)（端点视角） | **类比**，标准设定下退化 | 单源时 \(\mathcal U(\delta_{s_0},\beta)\) 是单点集（Remark 3.2）。必须放开初始流分布才非平凡 |
| 2 | 有向边流 \(m_e\)，目标 \(\sum_e\ell_e\lvert m_e\rvert\) | Def. 6.20; **Prop. 6.23** | 边流 \(F(s\to t)\) | **严格**（无向图上） | 同一个变量、同一个目标形式。有向 DAG 版本讲义未给，见 §5 边界 4 |
| 3 | 边缘约束 \(\pi_1=\alpha,\pi_2=\beta\) | Def. 3.1, Eq. (3.1); Def. 3.25 | 终止状态处 \(F(s\to x)=R(x)\) 的奖励匹配 | **条件严格** | 需先把 \(R\) 归一化为 \(\beta=R/Z\)。源侧对应 \(F(s_0)=Z\) |
| 4 | 守恒律 \(\mathrm{div}_G m=r\) | **Prop. 6.23**；连续版 Def. 14.1, Eq. (14.2) | flow matching：内部状态流入 = 流出 | **严格** | 这是全表最硬的一条。\(r=\mathbf a-\mathbf b\) 是净供给，内部顶点 \(r_i=0\) 即 \(\mathrm{div}\,m=0\) |
| 5 | 代价 \(c(x,y)\) = 图最短路 \(d_G\) | Def. 6.22；路径版 Eq. (14.16) + **Prop. 14.9** | 轨迹长度 / 期望轨迹步数 | **条件严格** | 条件：路径作用取「边长之和」。Prop. 14.9 保证「路径空间问题 = 以 \(c_{\mathcal A}\) 为代价的 Kantorovich 问题」 |
| 6 | Kantorovich 对偶势 \(f_i\)（顶点标量，\(\lvert f_i-f_j\rvert\le\ell_e\)） | Def. 5.1, Eq. (5.1); **Prop. 6.23** 左式 | 状态流 \(\log F(s)\) / 价值函数 | **类比**（有严格内核） | 同为「顶点标量 + 边上关系」。区别：OT 是**不等式**约束，DB 是**等式**。等式 = 已在互补松弛的接触集上 |
| 7 | 互补松弛：支撑 ⊂ 接触集 | **Prop. 5.3, Eq. (5.4)**; Prop. 5.7, Eq. (5.7) | 最优流只在「紧」的边上为正 | **严格**（在 LP 形式下） | 给出稀疏支撑的机制性解释，配合 Prop. 3.8 的 \(n+m-1\) 上界 |
| 8 | 熵正则 \(-\varepsilon H(\mathbf P)\) / \(\varepsilon\mathrm{KL}(\pi\|\alpha\otimes\beta)\) | Def. 8.2, Eq. (8.1); Eq. (8.8); Def. 8.13 | MaxEnt GFlowNet 的路径熵 \(H[F]=\mathbb E_\tau\sum_t H[P_F(\cdot\mid s_t)]\) | **类比**，差一个条件项 | OT 的熵在**端点耦合**上，GFN 的熵在**路径**上。二者经 Eq. (14.20) 相差 \(\int\mathrm{KL}(M_{x,y}\|\mathcal R^{\varepsilon,x,y})\mathrm d\pi\) |
| 9 | Sinkhorn 缩放 \(\mathbf P=\mathrm{diag}(\mathbf u)\mathbf K\mathrm{diag}(\mathbf v)\) | Prop. 8.5, Eq. (8.2); Eq. (8.5) | \(F(s\to t)=F(s)P_F(t\mid s)=F(t)P_B(s\mid t)\) | **类比** | 两侧都是「边上的量 = 两个顶点因子 × 一个核」。GFN 的「核」是图的邻接结构 |
| 10 | Sinkhorn = 交替 KL 投影 | Eq. (9.1)(9.2); **Prop. 9.4/9.5** | 交替强制约束的训练（如 T10 两阶段 GTB） | **类比**（算法同构） | 差别本质：OT 的每个半步有**闭式**投影（行/列缩放），GFN 只能 SGD 近似 |
| 11 | soft-\(c\)-变换 \(-\varepsilon\log\sum_j e^{-h_j/\varepsilon}\mathbf b_j\) | **Def. 8.19, Eq. (8.21)**; Alg. 8.2 | log 域的 DB/TB 残差与 logsumexp 归一化 | **类比** | 同为「硬 min 的软化」。GFN 的 \(\log Z\) 与 \(\log F\) 起 additive gauge 的作用，与 OT 势的规范自由度同构 |
| 12 | Schrödinger 桥参考路径律 \(\mathcal R^\varepsilon\) | Def. 14.10, Eq. (14.17); **Prop. 14.11** | 固定的反向策略 \(P_B\) | **条件严格** | 固定马尔可夫 \(P_B\) 后 TB 全局最优唯一（T02 Cor. 1），对应「给定参考桥、匹配端点边缘」 |
| 13 | Birkhoff–von Neumann 分解 | Thm. 3.17; **Cor. 3.20** | 随机策略 = 确定性策略的凸组合 | **类比** | OT 侧是均匀边缘下的置换分解，GFN 侧的「确定性策略」不构成置换矩阵 |
| 14 | Benamou–Brenier 动能作用 | Def. 14.4; **Thm. 14.5, Eq. (14.8)** | 「流 × 速度」的最小作用读法 | **类比**（不要用作定理） | BB 是连续时间连续状态；图上的正确离散对应物是 Beckmann（Prop. 6.21/6.23），不是 BB |
| 15 | Wasserstein 梯度流 / JKO | Def. 15.1, Eq. (15.1); Def. 15.5 | GFlowNet 的训练动力学 | **仅氛围类比** | JKO 是测度空间上的隐式欧拉，GFN 训练是参数空间上的 SGD。除非另有构造，不要建立对应 |
| 16 | 逆 OT：从观测耦合反推代价 | §12.5（p.258）; Prop. 12.31 附近的重心投影 | T10 的引导分布 \(p(\tau_{\to x}\mid X)\) | **类比** | 两侧都是「代价/偏好是被学出来的」。可作为把 guided TB 形式化的接口 |

**这张表在主线上的作用。**
第 2、4、5、7 行合起来就是 O08 的定理骨架：
固定初始流分布后，最小流 GFlowNet 的目标退化为**以图最短路为代价的 Kantorovich 问题**，
其最优解就是传输计划。
Prop. 6.23 提供了「边流 + 守恒」的静态形式，Prop. 14.9 提供了「路径 → 端点」的约简，
两者拼起来就是从 GFlowNet 到 OT 的完整翻译路径。
O07（用 GFlowNet 学最短路）走的是同一条路的另一端：先把代价定成最短路，再问流会不会自动走最短路。

#### 7. 可复用的 insight 与开放问题

1. **把 Prop. 6.23 推广到有向 DAG，是一个可以直接动手的定理草稿。**
   讲义的图 Beckmann 建立在**无向**图测地距离上（Def. 6.22），
   Remark 6.24 把无向边拆成两条反向弧后得到转运问题。
   GFlowNet 的图是有向且**不可逆**的：\(s\to t\) 存在不蕴含 \(t\to s\) 存在。
   此时 \(d_G\) 不再对称，Kantorovich–Rubinstein 的 1-Lipschitz 刻画（Prop. 6.17）需要换成
   「\(f_j-f_i\le\ell_e\) 单边约束」，流也只能非负 \(m_e\ge0\)。
   命题草稿：在有向连通 DAG 上，\(\min\{\sum_e\ell_e m_e:m\ge0,\ \mathrm{div}_G m=r\}
   =\max\{\sum_i f_ir_i:f_j-f_i\le\ell_e\}\)，且最优解的支撑无环。
   这一条一旦写清，第 6 节表格第 2、4 行就从「无向严格」升级为「DAG 严格」。
2. **稀疏性上界是一个可直接检验的经验预测。**
   Prop. 3.8 给出 \(n+m-1\) 个非零元的上界，Remark 3.19 给出支撑无环。
   在 hypergrid 或 SIX6 这类可枚举环境里，把训练好的 GFlowNet 边流按阈值截断，
   量一下有效支撑大小与是否含圈。若远超上界，说明训练目标没有把解推向 LP 顶点——
   这正好量化 T10 Remark 2 说的「流是欠定的」。
3. **「路径熵 vs 端点熵」的差值 \(\int\mathrm{KL}(M_{x,y}\|\mathcal R^{\varepsilon,x,y})\mathrm d\pi\) 是可计算的。**
   Eq. (14.20) 的分解在有限 DAG 上是有限和。
   实验草稿：在小 DAG 上分别训练 MaxEnt GFlowNet 与熵正则 OT 的解，
   显式算出这个条件项，看它是否随「每个 \(x\) 的轨迹数」增长。
   若增长，则「MaxEnt GFlowNet ≈ 熵正则 OT」这条常见说法在多轨迹环境里是错的——
   而 T10 Prop. 5.2 的 \(\Theta(1/(n-k))\) 恰好暗示了这一点。
4. **闭式投影是 Sinkhorn 的全部优势来源，值得追问 GFlowNet 有没有对应物。**
   Remark 8.4 的判断很干脆：熵放在 \(\mathbf P\) 的**元素**上、约束只有行列边缘，
   这种可分结构使 Bregman 投影退化成对角缩放。
   GFlowNet 的约束是「每个顶点一条守恒律」，在树上其实也是可分的。
   开放问题：在有向树（自回归 MDP）上，DB 的 KL 投影是否有闭式？
   若有，就得到一个不需要 SGD 的 GFlowNet 训练算法，可作为强基线。
5. **网络单纯形是被忽视的评估基线。**
   §3.2 与 Remark 6.24 都指出，稀疏图上的最小费用流有成熟精确解法（网络单纯形、Orlin 强多项式算法）。
   本仓库的 O07/O08 都在图上工作，理应报告「与精确 min-cost-flow 解的差距」。
   O08 摘要确实提到「与精确 OT 求解器一致」，这条基线应当被系统化。
6. **逆 OT（§12.5）是形式化 guided TB 的现成接口。**
   T10 的引导分布 \(p(\tau_{\to x}\mid X)\) 在做的事情，用 OT 语言说就是
   「从观测到的高奖励样本反推一个使这些样本成为最优传输目标的代价」。
   §12.5 提供了这类问题的标准提法与可微分 OT 损失的求导框架。
   把 guided TB 重写成逆 OT，能立刻得到「引导分布唯一吗、可辨识吗」这类以往没问过的问题。
7. **讲义没有覆盖、但主线需要的两块。**
   其一：**有向图/非对称代价上的 OT**（上面第 1 条）。
   其二：**同时优化耦合与内部路由**——讲义把「端点耦合」与「路径填充」分成两步
   （Prop. 14.9 先选耦合再填最优路径），
   而 GFlowNet 是把两者混在一个策略里同时学的。
   这两块是「内部流选择 = 最优传输」这条主线上剩下的真正空白。

## 6.2 O02 · 图上二次正则最优传输（SIAM J. Sci. Comput. 2018）

> **一句话**：这篇论文把图上的 1-Wasserstein 距离写成边流上的最小费用网络流（Beckmann 形式，Eq. (3)），在费用上加一个 \(\frac\alpha2\sum_eJ_e^2\) 的二次正则以换取解的唯一性，并证明当 \(\alpha\) 足够小时正则解就是未正则 LP 的某个解（Prop. 4、Corollary 1）；对偶问题的 Hessian 是活动子图的图 Laplacian（Prop. 2），由此得到一个带闭式线搜索和秩一 Cholesky 更新的 Newton 型算法。它对 O08 的价值不在算法，而在 Sec. 3.1 那一页数学：Kantorovich 耦合（费用 = 图最短路）⇔ 边流上的最小费用流，而流分解定理（Theorem 1）把两边的解互相翻译。

| 字段 | 内容 |
|---|---|
| arXiv | [1704.08200](https://arxiv.org/abs/1704.08200)（v4，2018-03-23，math.OC；本仓库 PDF 即此版） |
| 发表 | 期刊：SIAM Journal on Scientific Computing 40(4): A1961–A1986, 2018，DOI 10.1137/17M1132665（2017-06-01 投稿，2018-03-08 接收，2018-07-03 在线） |
| 作者 | Montacer Essid（NYU Courant）, Justin Solomon（MIT） |
| 代码 | 原文未给链接；Algorithm 1 caption 提到「accompanying Matlab implementation」，Sec. 6 说明用 SuiteSparse/CHOLMOD |
| 本仓库 PDF | `papers/1704.08200.pdf` · 中译 `papers_zh/1704.08200.zh.pdf` |
| 阅读优先级 | P1：O08「GFlowNet 暗地里学的是 OT 方案」所需的「Kantorovich ⇔ 图上最小费用流」这一块数学在这里有最干净的陈述；算法部分与主线无关 |

#### 2. 核心贡献（按原文编号）

**C1. 图上 \(W_1\) 的 Beckmann 表述（Sec. 3.1，Eq. (1) ⇔ Eq. (3)）。** 结论：两点间搬运费用可以分解为沿最短路逐边的费用，因此 Eq. (1) 与 Eq. (3) 给出同一个 \(W_1\)。原文把这一步当作已知（「Formalizing this argument provides an alternative to (1)」），未给出形式证明，指向 Santambrogio 2015 的 Beckmann 问题。

**C2. 二次正则的动机（Sec. 3.2）。** 熵正则在图设定下的三个缺点：写在 \(T\) 上而非 \(J\) 上、\(\alpha\to0\) 时交替投影收敛慢且有数值问题、任何正则强度下 \(T_{vw}>0\) 严格成立（失去稀疏性）。二次正则的两个理由：允许 \(J_e=0\) 恰好成立、\(\alpha\) 以可控方式调节稀疏性；算法适合低正则与 \(|E|\ll|V|^2\) 的稀疏图。

**C3. 对偶与 Laplacian 结构（Sec. 3.3，Prop. 1–2，Eq. (5)–(9)）。** Prop. 1：\(W_{1,\alpha}=\frac1\alpha\sup_p\big[\alpha f^\top p-\frac12|(Dp-c)_+|_2^2\big]\)，且 \((Dp-c)_e\le0\) 的边上 \(J_e=0\)。Definition 1：活动集 \(S(p)=\{e:(Dp-c)_e>0\}\)，互补松弛给出 \(S(J^\alpha)=S(p^\alpha)=:S(\alpha)\)。Eq. (8)–(9)：\(\nabla g=\alpha f-D^\top M(p)(Dp-c)\)，\(\mathrm{Hess}[g]=-D^\top M(p)D\)；Prop. 2：对偶 Hessian 是活动子图 \((V,S(p))\) 的无权 Laplacian（取负），零空间由连通分支的指示向量张成。

**C4. 小正则极限的稀疏性（Sec. 4，Prop. 3–5，Corollary 1，Lemma 1，Conjecture 1）。** Prop. 3：\(0<\alpha<\alpha'\) 且 \(J^\alpha\ne J^{\alpha'}\) ⇒ \(c^\top J^\alpha<c^\top J^{\alpha'}\) 且 \(|J^\alpha|^2>|J^{\alpha'}|^2\)。Prop. 4（Sparsity）：存在只依赖 \(G,f\) 的 \(\tilde\alpha>0\)，\(\forall\alpha\in(0,\tilde\alpha)\)，\(J^\alpha\) 也是 (LP) 的解。Corollary 1：进一步存在唯一的 \(J_0\)，使 \(\forall\alpha\in(0,\tilde\alpha)\) 它都是 (QP) 的唯一解。两个反例：Figure 1（\(J^\alpha\) 可能比某些 LP 解更稀疏）、Figure 2（活动集不随 \(\alpha\) 单调）。Conjecture 1 提出用无散度流 \(R_i\) 表述的单调性猜想。

**C5. 算法（Sec. 5，Eq. (20)–(28)，Algorithm 1）。** 对偶上升，交替使用梯度方向与伪 Newton 方向 \(L_k^+(\alpha f-D^\top M_kv_k)\)；线搜索闭式（抛物线极小 vs 活动集翻转的「击中时间」）；\(L_k^+=(L_k+N_kN_k^\top)^{-1}P_k\) 处理零空间；活动边增删对应 Laplacian 的秩一更新，用 CHOLMOD 维护稀疏 Cholesky 因子。

**C6. 实验（Sec. 6，Figure 6–8）。** 随机图 50–5000 节点上与梯度上升、全图 Laplacian 预条件、单纯形比较；\(\mathbb R^2\) 网格上与 Li–Osher–Gangbo 的 Fast \(L_1\) 比较。

#### 3. 方法与理论推导要点

##### 3.1 Kantorovich ⇔ 最小费用流：为什么 Eq. (1) 与 Eq. (3) 相等（原文陈述 + 编者补全论证）

原文只说「两点间搬运费用可分解为最短路上逐边费用」。把它写完整需要两个方向：

- 耦合 → 流：给定可行 \(T\)，把每份质量 \(T_{vw}\) 沿一条 \(v\to w\) 最短路推送，得到边流 \(J=\sum_{v,w}T_{vw}\,\delta(r_{vw})\)（\(\delta(r)\) 是路径 \(r\) 的边指示向量，Sec. 4.3.1 的记号）。逐边求和即 \(D^\top J=\rho_1-\rho_0\)，费用 \(c^\top J=\sum T_{vw}C_{vw}\)。所以 \(\min\text{(3)}\le\min\text{(1)}\)。
- 流 → 耦合：给定可行 \(J\)，Theorem 1（流分解，引 Ahuja–Magnanti–Orlin Theorem 3.5）把它分解为源到汇的路径流加环流 \(\hat J:\mathcal P\cup\mathcal C\to\mathbb R_+\)，且每条正流路径从某个 \(f_s<0\) 的源出发、到某个 \(f_t>0\) 的汇结束。丢掉环只会降费用（Sec. 4.3.1 原话）；令 \(T_{vw}=\sum_{r:s(r)=v,t(r)=w}\hat J(r)\)，则 \(T\) 是可行耦合，且每条路径费用 \(\ge C_{vw}\)，所以 \(c^\top J\ge\sum T_{vw}C_{vw}\)，\(\min\text{(3)}\ge\min\text{(1)}\)。

两个方向合起来就是 O08 需要的那句话：**图上以最短路为费用的 Kantorovich 问题，等价于边流上的最小费用流；最优边流的每一个路径分解都给出一个最优耦合，反之每个最优耦合沿最短路推送给出一个最优边流。** 分解不唯一（Sec. 4.3.1「a corresponding path flow」），这正是 GFlowNet 里「同一边流对应多种轨迹分布」的对应物。

##### 3.2 对偶（Prop. 1 的证明，Eq. (6)）

拉格朗日化 \(D^\top J=f\)，交换极大极小（凸二次规划 + 仿射约束，强对偶由 affine Slater 条件保证）：
\[
W_{1,\alpha}=\max_p\Big[f^\top p+\min_{J\ge0}\big(J^\top(c-Dp)+\tfrac\alpha2J^\top J\big)\Big].
\]
内层对每条边独立：\(J_e=\max\{(Dp-c)_e,0\}/\alpha\)，即 \(J=(Dp-c)_+/\alpha\)。代回得 Eq. (5)。读法（编者）：\(p\) 是图上的 Kantorovich 势，\((Dp)_e=p_w-p_v\) 是势差；未正则极限要求 \((Dp-c)_+=0\)，即 \(p_w-p_v\le c_e\) 于每条边——这是图上 1-Lipschitz 约束的形式，Kantorovich–Rubinstein 对偶的离散版。正则化把「紧边」\(p_w-p_v=c_e\) 换成「活动边」\(p_w-p_v>c_e\)，流量与超出量成正比。Eq. (7) 用 \(M(p)=\mathrm{diag}(\mathbb I\{e\in S(p)\})\) 把目标写成 \(g(p)=\alpha f^\top p-\frac12(Dp-c)^\top M(p)(Dp-c)\)，在 \(M\) 不变的区域内它是二次的，Hessian \(-D^\top MD\) 就是活动子图的 Laplacian（Prop. 2）。

##### 3.3 小正则极限选出唯一的 LP 解（Prop. 3、4，Corollary 1）

- Prop. 3 的证明是两条变分不等式相减：由 \(J^\alpha\) 极小得 Eq. (11) \(c^\top(J^{\alpha'}-J^\alpha)+\alpha(J^\alpha)^\top(J^{\alpha'}-J^\alpha)<0\)（严格，因严格凸且 \(J^\alpha\ne J^{\alpha'}\)），由 \(J^{\alpha'}\) 极小得反向的 \(\ge0\)；相减得 Eq. (12) \((J^{\alpha'})^\top(J^{\alpha'}-J^\alpha)>0\)，回代即得两个结论。
- Prop. 4 的证明分三步（Sec. 4.4）：(i) 用 Lemma 1 把任一 LP 解写成 \(\hat J_0=\hat J^\alpha+\sum_k\epsilon_kR_k\)，\(R_k\) 是路径空间上的无散度扰动（\(D^\top R_k=0\)），由「多的路径集 \(X_-^k\)」减「少的路径集 \(X_+^k\)」构成，每个源/汇各恰连一条；(ii) 证明 \(\check c^\top\hat R_k\le0\) 对所有 \(k\)（否则可改进 \(J_0\)），且至少一个严格 \(<0\)（否则 \(J^\alpha\) 已是 LP 解）；(iii) 沿这个 \(R_k\) 扰动 \(\epsilon\)，Eq. (17) 给目标变化 \(\epsilon[\check c^\top\hat R_k+\alpha(\hat J^\alpha)^\top S\hat R_k]+\frac{\epsilon^2}{2}\alpha\hat R_k^\top S\hat R_k\)；Prop. 5 的一致有界性（\(\hat J(r)\le-f_{s(r)}\)，\(J_e\le-\sum_{f_v<0}f_v\)）、路径有限性给出常数 \(K_1,K_3\)，Eq. (18) 定义 \(K_{\min}<0\) 为所有「负费用路径组合」中最接近零者，取 \(\tilde\alpha=|K_{\min}|/(2K_1)\) 即得矛盾。
- Corollary 1：若 \(0<\alpha<\alpha'<\tilde\alpha\) 给出不同的 LP 解，两者 \(c^\top J\) 相等，Prop. 3 要求 \(|J^\alpha|^2>|J^{\alpha'}|^2\)，但 \(J^\alpha\) 又是 \(V_\alpha\) 唯一极小点要求相反，矛盾。
- 编者推论（Corollary 1 的直接后果，原文未明说）：对 \(\alpha<\tilde\alpha\)，\(J_0\) 是 \(V_\alpha\) 在全体可行流上的唯一极小点，特别地在 LP 解集上极小；LP 解集上 \(c^\top J\) 为常数，所以 \(J_0=\arg\min\{|J|^2:J\text{ 解 (LP)}\}\)。二次正则是一条明确的 tie-breaking 规则：在所有最小费用流里挑 \(\ell^2\) 范数最小的那一个。
- 两个反例划定边界：Figure 1（\(c\equiv1\)，节点值 \(-1,-99,10,90\)）中 \(J^\alpha\) 只用一条稀疏边而两个 LP 解更稠密；Figure 2（一条总 \(L^1\) 费用低、但由许多连续边串成的长路径 vs 少数 \(L^1\) 费用高的边）中 \(\alpha\) 小时走长路径、\(\alpha\) 大时因为多条连续边的 \(L^2\) 代价累加而改走高费用边，活动集不单调。Figure 4 的 5 点 7 边例子给出 \(\alpha=10\) 时激活 \(R_1\)、系数 \(\epsilon_1=0.07\)，\(\alpha=10^3\) 时 \(\epsilon_1=0.25\) 并激活 \(R_2\)、\(\epsilon_2=0.25\)。

##### 3.4 算法循环（Sec. 5，Algorithm 1）

1. 随机初始化 \(p\)；\(v=Dp-c\)，\(M=\mathrm{diag}(v>0)\)，\(L=D^\top MD\)；用 flood fill 取零空间基 \(N\)（每个连通分支一列）；对 \(W_0=[MD;\,N^\top]\) 做稀疏 QR 得 \(L+NN^\top=R^\top R\)（Eq. (28)）。
2. 方向 Eq. (25)：奇数步梯度 \(s=\alpha f-D^\top Mv\)；偶数步伪 Newton \(s=R^{-1}R^{-\top}(s-NN^\top s)\)（Eq. (27)）；平移使 \(s\) 和为零。
3. 线搜索：抛物线极小 Eq. (21) \(t_{\mathrm{quad}}=(\alpha f^\top s-v^\top MDs)/(s^\top Ls)\)；击中时间 Eq. (22) \(h=-v\oslash(Ds)\)，Eq. (23) \(t_{\mathrm{active}}=\min\{h_e>0\}\)；Eq. (24) \(t=\min(t_{\mathrm{quad}},t_{\mathrm{active}})\)。
4. \(p\leftarrow p+ts\)；对每条新激活/去激活的边做 \(\pm d_ed_e^\top\) 的秩一更新，若连通分支合并/分裂则对 \(N\) 的列做至多四次秩一更新（Figure 5），先更新后降更新以保持满秩。

每步梯度方向 \(O(|V|)\)，伪 Newton 靠因子更新避免 \(O(|V|^3)\)。原文说从未在收敛前遇到 \(t_{\mathrm{active}}=0\)，但没有证明。

##### 3.5 二次正则 vs 熵正则（Sec. 2、Sec. 3.2 的对照，编者整理成表）

| 维度 | 熵正则（Cuturi 2013；Benamou et al. 2015） | 二次正则（本文 Eq. (4)） |
|---|---|---|
| 正则写在哪个变量上 | 耦合矩阵 \(T\)：\(-\sum_{vw}T_{vw}\ln T_{vw}\) | 边流 \(J\)：\(\frac\alpha2\sum_eJ_e^2\) |
| 变量个数 | \(|V|^2\) | \(|E|\)，稀疏图上远小于 \(|V|^2\) |
| 解的支撑 | 任何正则强度下 \(T_{vw}>0\) 严格成立，无稀疏性 | 允许 \(J_e=0\) 恰好成立；\(\alpha<\tilde\alpha\) 时解就是 LP 解（Prop. 4） |
| 小正则极限 | 交替投影收敛变慢、近零值数值不稳（原文引 Schmitzer 2016 的改进） | 解在 \(\alpha<\tilde\alpha\) 后不再变化（Corollary 1），算法适合低正则 |
| 大正则极限 | 独立耦合 \(\rho_0\otimes\rho_1\) | 类电流（electrical flow），Sec. 2 指出与 Christiano et al. 2011、Mądry 2013 的联系 |
| 算法 | Sinkhorn–Knopp 迭代缩放；一般图上无 Sinkhorn 型方法（Sec. 2） | 对偶 Newton 型，Hessian 为活动子图 Laplacian（Prop. 2） |
| 唯一性来源 | 严格凸的熵 | 严格凸的 \(\ell^2\) 项；\(\alpha\to0\) 时 tie-breaking 为 \(\ell^2\) 最小的 LP 解（3.3 编者推论） |
| 在 GFlowNet 里的对应物 | 轨迹分布上的 KL 型损失（TB）、Schrödinger 桥（C02） | T36 Appendix B.1 的 on-policy 状态流正则 \(\sum_sF(s)^2\) |

原文同时提到（Sec. 2）：二分图情形可以用 Benamou et al. 的交替投影框架把熵换成二次项得到「效率较低的 Sinkhorn 型」算法，每步需对浮点数排序（Duchi et al. 2008 的 \(\ell^1\) 球投影）。

#### 5. 前提假设与适用边界

- **图**：连通、有向、无容量约束、边费用 \(c\ge0\)。费用矩阵 \(C\) 必须是图最短路距离——这是 Eq. (1) ⇔ Eq. (3) 成立的全部条件；对一般的 \(C\)，Beckmann 形式不再等价于 Kantorovich。
- **数据**：\(\sum_vf_v=0\)；可行性由连通性保证（实验用双向图确保）。
- **正则**：\(\alpha>0\) 时 (QP) 严格凸、解唯一；\(\alpha\to0\) 的 \(\Gamma\)-收敛给出 LP 的某个解。Prop. 4 的 \(\tilde\alpha=|K_{\min}|/(2K_1)\) 依赖 \(G\) 与 \(f\)，非构造性（\(K_{\min}\) 需枚举路径组合）。
- **Prop. 3–5 的对象**：非负流的路径分解，环流在最优解里为零（Sec. 4.3.1）；Lemma 1 要求两个流都可分解为源—汇路径流。
- **算法**：目标分片二次，线搜索闭式成立的前提是活动集在步内不变（Eq. (24) 保证）；\(t_{\mathrm{active}}>0\) 是观察不是定理；Laplacian 零空间维数 = 活动子图连通分支数，需显式处理。
- **未覆盖**：容量约束、非线性费用、连续 Beckmann（Sec. 7 列为未来工作）、熵正则的图版本（Sec. 2 说无 Sinkhorn 型方法处理一般图）。
- **适用范围正向表述**：任何有限有向连通图上、以最短路为费用的 \(W_1\)（EMD），要求解唯一且稀疏、正则很小、图稀疏时，本文的表述和算法都适用。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：O01（Kantorovich 形式、\(W_1\)、对偶的教科书背景）；仓库外：Beckmann 1952（连续流形式）、Santambrogio 2015（Beckmann 问题）、Ahuja–Magnanti–Orlin 1993（流分解 Theorem 3.5、最小费用流算法）、Cuturi 2013 / Benamou et al. 2015（熵正则与 Sinkhorn，被对照）、Li–Osher–Gangbo 2016（网格上的二次正则 \(L^1\) 算法，被比较）、Yin 2010（\(L^1+L^2\) 稀疏性的更一般结果，原文承认 Prop. 4 可由其导出）。
- **后继 / 对照**：O08 直接依赖 Sec. 3.1；O07 的「最小总流 = 最短路」是 Eq. (3) 在单源、\(c\equiv1\) 下的特例；T36 Appendix B.1 的「on-policy 正则实际最小化 \(\sum F^2\)」恰是 Eq. (4) 的正则项；C01/C02（图上的（不平衡）OT 与 Schrödinger 桥）走的是熵正则路线，是本文 Sec. 3.2 批评的那一边；O03/O06 是连续域上的 OT 求解器，与此处的图离散设定平行。
- **它为 O08 提供了哪一块数学**：O08 的命题是「GFlowNet 学到的内部流是一个 Kantorovich 最优传输方案」。要让这句话成立，需要三件事，全部出自本文 Sec. 3.1 与 Sec. 4.3.1：
  - (a) **Kantorovich ⇔ min-cost flow on graphs with shortest-path cost**（Eq. (1) ⇔ Eq. (3)）。GFlowNet 的边流 \(F\) 满足 \(D^\top F=\)（终止边缘）\(-\)（源边缘），总流 \(\sum_eF(e)\) 是 \(c\equiv1\) 的 Beckmann 费用；T36 Prop. 3.12 说总流 = \(Z\cdot\mathbb E[n_\tau]\)。于是「最小期望长度」= Eq. (3) 的最小费用流 = Eq. (1) 的 \(W_1\)，地面费用是图最短路（在允许双向移动的非无环图上它是真正的度量）。
  - (b) **流分解定理**（Theorem 1）把边流翻译成路径流。GFlowNet 的轨迹分布本身就是一个路径分解，\(T_{vw}=\) 从 \(v\) 出发、在 \(w\) 终止的轨迹质量；分解不唯一这一点解释了为什么 O08 能说「同一边流下不同的 \(P_F\) 给出同一耦合」。
  - (c) **最优流无环**（Sec. 4.3.1「removing a cycle can only decrease the total cost」）。这是 T19 Theorem 1「正则极限无环」与 T36 Eq. (11) 极小点落在边界的组合学根源。
- **本文没有做而主线需要的**：单源单汇以外的 GFlowNet 语义（本文天然多源，反而比 T36 更接近 O08 的设定）；随机策略与期望访问次数（本文没有概率）；对偶变量 \(p\) 与 GFlowNet 的 \(\log F\) 之间的关系（见 7.3）。

#### 7. 可复用的 insight 与开放问题

1. **O08 核心恒等式的定理草稿**。设 \(G\) 满足 T36 Assumption 3.1，源边缘 \(L\) 落在 \(\mathrm{out}(s_0)\)、终止边缘 \(R\) 落在 \(\mathcal X\)，\(\sum L=\sum R=Z\)。令 \(\mathcal F_{L,R}\) 为非负守恒、两端边缘固定的内部边流集合。则 \(\min_{\mathcal F_{L,R}}\sum_eF(e)=Z\cdot W_1(L/Z,R/Z;d_G)\)，极小点无环，其任一路径分解给出 \(W_1\) 的最优耦合。证明：Eq. (1) ⇔ Eq. (3) 取 \(c\equiv1\)、\(f=R-L\)，加 Theorem 1。这是 O08 应当引用的精确出处。
2. **二次正则 = T36 的 on-policy 正则，且它有典范极限**。T36 Appendix B.1 说 on-policy 训练最小化 \(\sum_sF(s)^2\)；按 3.3 的编者推论，\(\alpha<\tilde\alpha\) 时解是「所有最小费用流中 \(\ell^2\) 范数最小者」。实验草案：在多条等长最短路的 hypergrid 上，检查小 \(\lambda\) 下学到的流是否把质量均分到各条最短路（\(\ell^2\) 最小 ⇒ 尽量均匀），并与 O02 的 (QP) 解逐边比较；若吻合，GFlowNet 的「多样性」在 OT 语言里就是二次正则的 tie-breaking。
3. **对偶势 \(p\) 与 \(\log F\)**。Prop. 1 的 \(p\) 是 Kantorovich 势，活动边上 \(J_e=(p_w-p_v-c_e)/\alpha\)；T36 Theorem 3.13 的 \(V^\star(s)=\log F(s)\) 是软 Bellman 势。定理草稿：把 T36 的每步负奖励 \(-c\)（见 T36 报告 7.3）推到 \(c\to\infty\)，\(\log F(s)/c\) 收敛到图上的最短路势，即 \(\alpha\to0\) 的对偶解 \(p\)；这会把「GFlowNet 的状态流」与「OT 对偶变量」接起来。
4. **活动集不单调（Figure 2）对 \(\lambda\) 扫描的含义**。T36 Figure 5 扫 \(\lambda\) 只看长度与误差；按 Conjecture 1 的框架，不同 \(\lambda\) 下 GFlowNet 使用的边集可能非嵌套地切换。实验：记录 T36 \(20^4\) 实验各 \(\lambda\) 下 \(F_\theta>\)阈值的边集，检验是否出现 Figure 2 型的路线切换。
5. **熵正则 vs 二次正则的 GFlowNet 对应**。熵正则写在耦合 \(T\) 上（Sinkhorn，C01/C02 的 Schrödinger 桥），二次正则写在边流 \(J\) 上（本文）。TB 损失是轨迹分布上的 KL 型散度（T19 Eq. (13)），更像前者；状态流正则像后者。开放问题：O08 的「学到的 OT 方案」在有限 \(\lambda\) 下到底是哪一种正则化 OT 的解？
6. **Newton 结构可否用于 O08 的求解器基线**。Prop. 2 的 Laplacian Hessian + 秩一更新是精确求解小图 OT 的快速方法；在 T36 的 \(7\times7\) 与 \(S_4\) 上用它算出 (QP) 精确解，可作为 GFlowNet 学到的流的 ground truth。

## 6.3 O03 · GeONet：学习 Wasserstein 测地线的神经算子（UAI 2024）

> **一句话**：GeONet 把「给定一对边缘分布、求 Wasserstein 测地线」这件事从"每来一对就重解一次 OT"改写成一次性训练的算子学习：用两组 DeepONet 分别拟合 Benamou–Brenier 问题的原始–对偶 KKT 方程（连续性方程 + Hamilton–Jacobi），训练只需要边界分布对、不需要测地线真值，推理是一次前向传播。在 GFlowNet × OT 地图上它站在「摊销 / amortization」这一侧：它证明了动态 OT 的最优性条件可以被一个跨分布族的算子吃下来，而这正是「条件 GFlowNet–OT」想要的能力——只不过 GeONet 的实现机制（PDE 残差）在离散图上不存在对应物。

| 字段 | 内容 |
|---|---|
| arXiv | [2209.14440](https://arxiv.org/abs/2209.14440)（v4, 2024-05-23） |
| 发表 | **UAI 2024 主会**（原文首页脚注：Accepted for the 40th Conference on Uncertainty in Artificial Intelligence） |
| 作者 | Andrew Gracyk（UIUC 统计系）、Xiaohui Chen（USC 数学系） |
| 代码 | https://github.com/agracyk2/GeONet （原文 Sec. 4 给出） |
| 本仓库 PDF | `papers/2209.14440.pdf` · 中译 未生成 |
| 阅读优先级 | **P2** —— 方法本体与 GFlowNet 没有共享结构，但它是「摊销式 OT 求解器」这条支线里最早把*整条动态轨道*而非静态映射作为输出的工作，作为对照系必须读 |

#### 2. 核心贡献（按原文编号）

**(C1) 把测地线求解重写为算子学习问题（Sec. 3, Eq. (12)–(13)）。** 结论：一次训练之后，对新的 \((\mu_0,\mu_1)\) 只需一次前向传播即可给出整条 \(\{\mu_t\}\)。前提：训练分布族与测试分布族同构（原文 OOD 实验显示分布外误差上升 2–3 倍，见 Sec. 4）。

**(C2) 训练不需要测地线真值（Sec. 1「Surprisingly...」段 + Eq. (19)）。** 结论：监督信号只有边界对 \((\mu_0^{(i)},\mu_1^{(i)})\)，中间时刻靠 PDE 残差把 KKT 条件"焊"住。这是它相对纯数据驱动 neural operator 的核心便宜之处，也是它误差在 \(t=0.25/0.5/0.75\) 明显大于 \(t=0,1\) 的原因（原文 Appendix H.1 自己承认）。

**(C3) 原始与对偶两套网络联合训练，用零对偶间隙做正确性锚（Sec. 2.1 + Appendix B, Eq. (32)）。** 结论：只要 \((\mu^\ast,u^\ast)\) 同时满足 CE 与 HJ 并满足边界条件，对偶间隙为零，解就是测地线。这条是 GeONet 全部设计的支点。

**(C4) 输出网格无关，支持 zero-shot 超分辨率（Sec. 1, Fig. 1；Table 2 的 high-res. 行）。** 结论：在 \(24\times24\) 上训练、在 \(75\times75\) 上取值，误差与同分辨率随机测试基本持平（2D high-res. \(6.29\%\text{–}7.88\%\) vs 2D random \(6.33\%\text{–}7.13\%\)，Table 2）。

**(C5) 熵正则版 ER-GeONet（Appendix D, Eq. (56)–(58)）。** 结论：把约束改成 \(\partial_t\mu+\mathrm{div}(\mu v)+\varepsilon\Delta\mu=0\)，KKT 变成 \(\partial_t\mu+\mathrm{div}(\mu\nabla u)=-\varepsilon\Delta\mu\)、\(\partial_t u+\tfrac12\|\nabla u\|_2^2=\varepsilon\Delta u\)。后者是抛物型方程、有唯一光滑解，训练更稳；\(\varepsilon\downarrow0\) 时按消失粘性方法收敛到 Benamou–Brenier 解（引 Mikami 2004、Evans 2010）。**注意这条是 Appendix，正文实验没有报 ER-GeONet 的数字。**

**(C6) 梯度增强（Appendix E, Eq. (61)–(62)）。** 把 PDE 对空间坐标再求一次导、把新残差也加进损失。原文说法是提高样本效率与精度，未给出增强前后的对照数字。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：O01（OT for Machine Learners，Benamou–Brenier 与 Kantorovich 对偶的标准参考）；数值参考实现依赖 POT（Flamary et al. 2021）。方法论前驱是 DeepONet（Lu et al. 2021）、physics-informed DeepONet（Wang et al. 2021b）与 enhanced DeepONet（Tan & Chen 2022）。
- **后继 / 对照**：
  - **O05（UNOT, ICML 2025）** 是它的直接竞争者，且给出了不利于 GeONet 的对照（O05 Sec. 4.3, Fig. 8）。两者摊销的对象不同：GeONet 摊销"整条测地线"，UNOT 摊销"熵 OT 对偶势"，后者再由势导出计划、重心、测地线。
  - **O06（Q-flow, AISTATS 2025）** 与它解同一个 Benamou–Brenier 问题，但走神经 ODE + 双向 KL 松弛，是 per-pair 求解、不摊销。
  - **O04（α-DSBM, NeurIPS 2024）** 解熵正则的动态版本（Schrödinger 桥），同样 per-pair。GeONet 的 ER-GeONet（Appendix D）与 O04 在目标上重叠，但 GeONet 用 PDE 残差、O04 用迭代投影 + bridge matching。
- **对主线（"内部流选择 = 最优传输"）贡献了什么**：GeONet 提供了这条主线的**连续侧对偶模板**。O08 的 Theorem 3.2 说：min-flow 非无环 GFlowNet 在固定初始边流分布后等价于以图最短路为代价的 Kantorovich 问题（也等价于离散 Beckmann）；O08 的 Theorem 3.3 给出其对偶 LP，并说明最优流只能支撑在最短路子图上。把这两条与 GeONet 的 KKT 对照，结构是一一对应的：
  - 连续性方程 \(\partial_t\mu+\mathrm{div}(\mu\nabla u)=0\) ↔ 图上流守恒（O08 Eq. (4)）；
  - HJ 方程 \(\partial_t u+\tfrac12\|\nabla u\|^2=0\) ↔ 图上最短路/Bellman 方程，其解 \(d(u,x)\) 正是 O08 里的 ground cost；
  - 零对偶间隙 ↔ O08 证明中的 \(\mathrm{GFlow}^\star \leqslant \mathrm{OT}^\star\) 与反向不等式夹逼。
  换句话说：**GeONet 学的 \(H_\psi\) 就是 GFlowNet 里"状态流的对数/势函数"的连续版本**。这个对应关系是把 GeONet 的摊销技巧搬到 GFlowNet 上的唯一可靠接口——不是搬 PDE 残差，而是搬"用对偶量做无真值监督"这个想法。

#### 7. 可复用的 insight 与开放问题

1. **「用最优性条件替代真值」可以直接移植成 GFlowNet 的辅助损失。** GeONet 的 \(\mathcal{L}_{\mathrm{HJ}}\) 在图上的对应物是"势函数满足 \(\pi_x - \pi_u \leqslant d(u,x)\) 且在支撑边上取等"（O08 Theorem 3.3 的对偶可行性 + 互补松弛）。可写成一条实验：在 min-flow GFlowNet 训练里加一项对偶互补松弛残差，看是否加速收敛到 OT⋆ 而不需要更长的轨迹采样。
2. **摊销的正确切面是"对偶势"而不是"轨道"。** GeONet 输出整条 \(\{\mu_t\}\) 的代价是中间时刻误差显著大于两端（Table 2 每一行都是中间高、两端低）；UNOT 只摊销势、其余靠一步 Sinkhorn 补，反而更准。对条件 GFlowNet–OT 的启示：条件网络应该预测 \(\log F\) 或 \(P_B\) 这类"势型"量，而不是直接预测整条轨迹分布。
3. **网格无关 ≠ 输入无关**，这个坑值得在图上重演一次。GeONet 的 trunk 无关而 branch 有关；GFlowNet 若要跨图规模泛化，编码器必须对节点数不变（例如图神经网络 + 位置编码），否则会重复 GeONet 的"输入侧被钉死"限制。
4. **中间时刻缺监督是 physics-informed 摊销的结构性弱点。** 一条定理草稿：在只有两端约束 + PDE 残差的训练目标下，中间时刻的误差下界与 collocation 采样密度和 PDE 残差算子的条件数有关；对应到 GFlowNet，就是"只用 TB 损失（两端）而不用 DB/SubTB（中间）时的 credit assignment 误差"（可与 T03/T05 的结论对照）。
5. **熵正则是共享的稳定化旋钮。** ER-GeONet 的 \(\varepsilon\Delta u\) 让 HJ 变抛物、解唯一光滑（Appendix D）；O04 的 \(\varepsilon\) 是布朗运动方差；GFlowNet 里对应的是"策略熵/温度"。开放问题：在 min-flow GFlowNet 里显式加熵项，是否得到图上的熵 OT（即 Sinkhorn 计划），以及其 \(\varepsilon\downarrow0\) 极限是否恢复最短路支撑（O08 Theorem 3.3）。
6. **OOD 退化幅度可以当作条件化能力的量尺。** GeONet OOD 误差从 \(\sim5\%\) 涨到 \(\sim20\%\)（1D，Table 2）。评估"条件 GFlowNet–OT 是否真有跨任务泛化优势"时，应该照抄这个协议：训练分布参数区间 → 测试时外扩区间 → 报告误差涨幅，而不是只报同分布指标。

## 6.4 O04 · Schrödinger Bridge Flow（NeurIPS 2024 Spotlight）

> **一句话**：这篇把 DSBM 的"交替解两个投影子问题"换成"沿一条路径测度流做单步梯度下降"：定义路径测度上的流 \(\partial_s\hat{\mathbb{P}}_s=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}_s))-\hat{\mathbb{P}}_s\)，其唯一不动点就是 Schrödinger 桥；用步长 \(\alpha\in(0,1]\) 离散得到 \(\alpha\)-IMF（\(\alpha=1\) 恰好退化成原 IMF），参数化实现叫 \(\alpha\)-DSBM，一个网络、一套损失、在线微调。对 GFlowNet–OT 主线，它是**唯一一篇把"多条构造路径如何分配质量"写成两个显式投影算子、并给出算子层面收敛证明**的邻居工作：\(\mathrm{proj}_{\mathcal{R}}\) 就是"固定 \(P_B\)"、\(\mathrm{proj}_{\mathcal{M}}\) 就是"从边流反解 \(P_F\)"，两者的交替与 GFlowNet 训练的结构同源。

| 字段 | 内容 |
|---|---|
| arXiv | [2409.09347](https://arxiv.org/abs/2409.09347)（v1, 2024-09-14） |
| 发表 | **NeurIPS 2024 主会（Spotlight）**；本仓库 PDF 是 v1 预印本，页脚仍写 "Submitted to 38th Conference on NeurIPS 2024" |
| 作者 | Valentin De Bortoli\*、Iryna Korshunova\*（并列一作）、Andriy Mnih、Arnaud Doucet（均 Google DeepMind） |
| 代码 | **未公开**（NeurIPS Checklist 第 5 项：因 IP 限制无法公开代码库，计划发布小规模复现 notebook） |
| 本仓库 PDF | `papers/2409.09347.pdf`（47 MB，主要是图）· 中译 未生成 |
| 阅读优先级 | **P1** —— 它给出的两投影算子框架是把"GFlowNet 的 \(P_F/P_B\) 一致性训练"翻译成测度论语言的最好模板，且其失败模式（不是 simulation-free、必须自采样）直接对应 GFlowNet 的 on-policy 采样代价 |

#### 2. 核心贡献（按原文编号）

**(C1) Schrödinger 桥流（Sec. 3.1, Eq. (3)）。** 结论：定义

\[
\hat{\mathbb{P}}_0=(\pi_0\otimes\pi_1)\mathbb{Q}_{|0,1},\qquad \partial_s\hat{\mathbb{P}}_s=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}_s))-\hat{\mathbb{P}}_s,\qquad \mathbb{P}_s=\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}_s),
\]

则该流的**唯一不动点是 SB**。论证只有两行（Sec. 3.1）：不动点 \(\bar{\mathbb{P}}\) 满足 \(\bar{\mathbb{P}}=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\bar{\mathbb{P}}))\)，反复代入即知 \(\bar{\mathbb{P}}\) 是 IMF 序列的极限点，故为 SB（引 Peluchetti 2023, Thm. 2）。前提：流本身 well-defined（原文假设）。

**(C2) \(\alpha\)-IMF 与收敛定理（Eq. (4) + Theorem 3.1）。** \(\hat{\mathbb{P}}^{n+1}=(1-\alpha)\hat{\mathbb{P}}^{n}+\alpha\,\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}^{n}))\)，\(\mathbb{P}^n=\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}^n)\)。对**任意** \(\alpha\in(0,1]\) 有 \(\lim_n\mathbb{P}^n=\mathbb{P}^\star\)；\(\alpha=1\) 恰好是 IMF/DSBM。注意 \(\hat{\mathbb{P}}^n\in\mathcal{R}(\mathbb{Q})\) 对所有 \(n\) 成立——凸组合保 reciprocal 类，这是整个构造能闭合的关键。

**(C3) 非参数更新 = \(\alpha\)-IMF（Prop. 3.2 / Prop. D.1）。** 取 \(\delta_n=\alpha\)、\(\mu_n=(1-\alpha)\hat{\mathbb{P}}^n+\alpha\,\mathrm{proj}_{\mathcal{R}}(\mathbb{P}^n)\)，则泛函梯度下降 \(v_t^{n+1}=v_t^{n}-\delta_n\nabla_{\mu_n}L_t(v_t^n,\mathbb{P}_{v^n})\) 产生的路径测度 \(\mathbb{P}_{v^n}\) 恰等于 \(\mathbb{P}^n\)。**这条是全文最有信息量的技术结论**（详见第 3 节）。

**(C4) 双向过程与单网络参数化（Sec. 4, Eq. (10)–(13), Prop. 4.1）。** 单向在线更新会累积误差（Appendix I 给出高斯情形的精确递推 Prop. I.1/I.2）；改成同时训前向与后向即可（因为 Markov 投影对前后向一致，引 Shi et al. 2023, Prop. 9）。参数化上不用两个网络，而是给网络加一个方向输入 \(s\in\{0,1\}\)，\(v_\theta(1,\cdot)\approx\overrightarrow{v}\)、\(v_\theta(0,\cdot)\approx\overleftarrow{v}\)。

**(C5) 参数更新是非参数更新的预条件版本（Prop. D.4 / D.5）。** \(\theta\leftarrow\theta-\alpha\nabla_\theta L(\theta,\mathbb{P}_{\bar\theta})\)（\(\bar\theta\) 为 stop-gradient）在 \(\alpha\to0\) 时平均意义上是非参数损失的下降方向。前提：Hessian 有界（Eq. (28) 的常数 \(C\)）。

**(C6) 与 Sinkhorn 流 / EM / RL 的三条平行关系（Appendix H）。** \(\gamma\)-Sinkhorn（Karimi et al. 2024）与 \(\gamma\)-IMF 的差别归结为一句话：**前者是隐式更新，后者是显式更新**（Table 2）。按 Brekelmans–Neklyudov 的分类，IPF 是 Expectation–Expectation、IMF 是 Maximisation–Maximisation，本文算法是 MM 的增量版（Appendix H.3）。Appendix H.2 给出 replay-buffer 版本（Algorithm 6），\(n_{\mathrm{refresh}}=1,\ N=B\) 时退化为 Algorithm 1。

#### 5. 前提假设与适用边界

1. **参考过程必须有闭式桥**。\(\mathrm{proj}_{\mathcal{R}}\) 免费的前提是 \(\mathbb{Q}_{|0,1}\) 为布朗桥（Eq. (12)/(30)）；换成一般代价，reciprocal 投影需要学（Sec. 5 引 Neklyudov 等、Liu 等 2022）。
2. **\(\varepsilon>0\) 固定且已知**，且 \(\pi_0,\pi_1\) 有二阶矩与有限微分熵（Lemma D.2 的显式条件：\(\int\|x\|^2\mathrm{d}\pi_i<\infty\)、\(H(\pi_i)<\infty\)）。
3. **必须能从模型自采样**。算法不是 simulation-free：每步微调都要解一次前/后向 SDE（Discussion 里列为主要局限）。
4. **双向训练是稳定性的必要条件**，不是可选优化项（Appendix I 的定量证据）。
5. **凸组合保持 reciprocal 类**是 \(\alpha\)-IMF 合法性的结构前提（Sec. 3.1）；如果 \(\mathcal{R}(\mathbb{Q})\) 不是凸集，整套构造失效。
6. **理论 \(\alpha\) 与实现 \(\alpha\) 不是同一个东西**。实现用 Adam，\(\alpha\) 隐式自适应（Sec. 4 末）；Prop. D.4/D.5 只保证 \(\alpha\to0\) 时的下降方向。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：DSBM/IMF（Shi et al. 2023；Peluchetti 2023）、bridge matching / stochastic interpolants（Peluchetti 2021；Albergo–Vanden-Eijnden 2023；Lipman et al. 2023）、DSB/IPF（De Bortoli et al. 2021）。OT 侧前驱是 Léonard 的 Schrödinger 问题综述与 Cuturi 的 Sinkhorn。本仓库内：O01（OT 基础）、O03（同为动态 OT，但用 PDE 残差）。
- **后继 / 竞争**：C02（Generalized Schrödinger Bridge on Graphs，预印本 2026）与 C03（Discrete Diffusion SB Matching for Graph Transformation，ICLR 2025）是把这套投影框架搬到离散/图状态空间的尝试——**这两篇才是 GFN–OT 的直接竞争者**，O04 是它们的方法论母本。
- **与主线（"内部流选择 = 最优传输"）的公式级对照**。把 GFlowNet 的对象与 O04 的对象逐一对齐：

  | GFlowNet（T02 / O08 记号） | O04（SB 记号） | 是否同构 |
  |---|---|---|
  | 轨迹测度 \(\mathbb{P}(\tau)=\prod_t P_F(s_{t+1}|s_t)\) | Markov 路径测度 \(\mathbb{P}\in\mathcal{M}\)，\(\mathrm{d}X_t=v_t\mathrm{d}t+\sqrt{\varepsilon}\mathrm{d}B_t\) | 同构：都是"由局部转移律生成的全局路径律" |
  | 固定参考 \(P_B(s|s')\) → 给定终点后的轨迹条件律确定 | \(\mathbb{P}=\mathbb{P}_{0,1}\mathbb{Q}_{|0,1}\)，即 \(\mathrm{proj}_{\mathcal{R}}\) | **同构**：\(\mathrm{proj}_{\mathcal{R}}\) 就是"把 \(P_B\) 拍回参考 \(P_B\)"，两者都只改中间、不改两端 |
  | 由边流反解前向策略 \(P_F(s'|s)=F(s\to s')/F(s)\) | \(v_t^\star(x_t)=(\mathbb{E}[X_1|X_t=x_t]-x_t)/(1-t)\)，即 \(\mathrm{proj}_{\mathcal{M}}\) | **同构**：都是"求与给定（非 Markov）轨迹测度同边缘的 Markov 策略"，且都保边缘（\(\mathbb{M}_t=\mathbb{P}_t\)） |
  | 训练目标：\(P_F\) 与 \(P_B\) 诱导的轨迹分布相同（O08 Eq. (1)） | 不动点条件：\(\bar{\mathbb{P}}=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\bar{\mathbb{P}}))\)（Sec. 3.1） | 同构：都是"两个算子的公共不动点" |
  | 代价 = 期望轨迹长度 \(\mathbb{E}[n_\tau]\)（O07/O08） | 代价 = \(\int\tfrac12\|x-y\|^2\mathrm{d}\Pi-\varepsilon H(\Pi)\)（Eq. (1)） | **不同构**：图上代价来自"步数"，SB 的代价来自"位移平方 + 熵"；\(1-t\) 分母在图上无对应 |
  | 轨迹长度 \(n_\tau\) 无界（非无环） | 时间区间固定 \([0,1]\) | **不同构**，且这是移植的第二道墙 |
  | 熵/温度旋钮：策略熵、reward 指数 | \(\varepsilon\)（布朗运动方差） | 结构同源、量纲不同 |

- **"多条构造路径分配"的同与异，落到公式上**：
  - **同**：两边都在解"同一对端点之间有多条路径，质量怎么分"。GFlowNet 的答案是 \(\mathbb{P}(\tau)=\prod P_F=\prod P_B\)（O08 Eq. (1)）配合 reward matching \(P_B(x|s_f)=R(x)/Z\)（O08 Eq. (2)）；O04 的答案是 \(\mathbb{P}=\mathbb{P}_{0,1}\mathbb{Q}_{|0,1}\) 配合边缘约束 \(\mathbb{P}_0=\pi_0,\mathbb{P}_1=\pi_1\)。**两者都是"端点约束 + 给定端点后的路径律被参考测度钉死"这一模式的实例。**
  - **异（第一处，本质）**：GFlowNet 的路径分配是**组合的、离散的**，"多条路径"指图上不同的边序列；\(\mathrm{proj}_{\mathcal{R}}\) 的类比物是"给定 \((s_0\to u,\ x\to s_f)\) 后按参考随机游走重采中间状态"，而这在图上**没有闭式**（布朗桥的闭式插值 Eq. (12) 是欧氏结构的礼物）。C03 之所以要引入离散扩散，就是为了造一个可采样的离散桥。
  - **异（第二处，可利用）**：O04 的 \(\alpha\) 插值发生在**耦合空间**（Eq. (22) 塌缩式：新漂移 = 在 \((1-\alpha)\hat{\mathbb{P}}^n+\alpha\,\mathrm{proj}_{\mathcal{R}}(\mathbb{P}^n)\) 下对终点取条件期望）。GFlowNet 训练里对应的操作是"用旧策略与新目标的混合分布做 off-policy 更新"，也就是 replay buffer——O04 自己已经把这层关系写出来了（Appendix H.2, Algorithm 6）。**这意味着 GFlowNet 的 replay buffer 不只是工程技巧，它在 SB 语言下就是路径测度流的一种离散化。**
  - **异（第三处，方向性）**：GFlowNet 的 \(P_B\) 通常是**可学的**（或按 T02 的自由度被选择），而 O04 的 \(\mathbb{Q}_{|0,1}\) 是**钉死的**（正是"reciprocal class of \(\mathbb{Q}\)"的定义）。因此 O04 的解唯一（Lemma D.2），GFlowNet 在 \(P_B\) 自由时解不唯一——这个不唯一性恰恰是 O08 用"最小总流量"来消解的，消解后得到最短路支撑（O08 Theorem 3.3）。**换句话说：O04 用"固定参考桥 + 熵"选流，O08 用"最小总流量"选流；两者是同一件事的两种正则化。**

#### 7. 可复用的 insight 与开放问题

1. **把 \(\alpha\)-IMF 直接翻译成"图上的 \(\alpha\)-IMF GFlowNet"。** 实验草案：在 O08 的 hypergrid 环境上，固定参考 \(P_B\) 为父节点均匀分布（充当 \(\mathbb{Q}_{|0,1}\)），交替执行 (i) 用当前 \(P_F\) 采样轨迹、只保留端点对 \((u,x)\)、按参考 \(P_B\) 重采中间路径（离散 \(\mathrm{proj}_{\mathcal{R}}\)）；(ii) 对重采后的轨迹做一步 TB/DB 梯度（离散 \(\mathrm{proj}_{\mathcal{M}}\)），步长 \(\alpha\)。度量：与 scipy LP 解的 \(\mathrm{OT}^\star\) 差距 vs 采样预算。假设 \(\alpha<1\) 在大图上更省，因为 (ii) 不需要训到收敛。
2. **一条定理草稿：图上 reciprocal 类的凸性与不动点唯一性。** O04 的合法性依赖两件事：\(\mathcal{R}(\mathbb{Q})\) 凸、且 "Markov + reciprocal + 边缘正确 ⟹ SB"（Lemma D.2）。在有限图上，"给定端点的参考游走条件律"定义的类同样是凸的（凸组合只改 \(\Pi_{0,1}\)）；需要补的是离散版 Léonard Thm. 2.12。若成立，则"最小总流量 GFlowNet + 熵正则"的唯一解就是图上的 Schrödinger 桥，且 \(\varepsilon\downarrow0\) 时应收敛到 O08 Theorem 3.3 的最短路支撑解。**这是本仓库最值得先做的理论题。**
3. **双向训练在 GFlowNet 上的对应物需要被显式检验。** O04 用 Prop. I.1/I.2 证明单向会累积误差、双向不会。GFlowNet 的"双向"天然存在（\(P_F\) 与 \(P_B\) 同时学），但很多实现把 \(P_B\) 固定。可做实验：固定 \(P_B\) vs 联合学 \(P_B\)，在长轨迹环境（如 O07 的 Rubik's Cube）里看 \(\mathbb{E}[n_\tau]\) 的偏差是否随迭代累积。
4. **"不是 simulation-free"是共享的成本瓶颈，值得共享解法。** O04 每步都要解 SDE，GFlowNet 每步都要 rollout 轨迹。O04 给出的两条缓解路线可以照搬：replay buffer（Appendix H.2）与 model stitching（Appendix G：用已有的两个生成模型初始化，取 \(\sigma_{t'}^2=\varepsilon/2\)，等价于在线版 DSBM-IPF）。对 GFlowNet 的翻译分别是"经验回放"与"从两个已训好的采样器拼接初始化"。
5. **\(\varepsilon\) 的分辨率平移规则是可迁移的工程结论。** \(\varepsilon_{256}=\varepsilon_{64}\cdot(256/64)^2\)（Appendix K.4）。图上的类比问题："状态空间规模从 \(n\) 增到 \(n'\) 时，熵正则系数该怎么缩放才能保持同样的 exploration"——这是一个可以直接做出来的标度律实验。
6. **评估协议要防止被噪声指标骗。** O04 主动指出 FID 在 <500 张测试集上不可靠、2-Wasserstein 在 10K 样本下仍高方差（Fig. 11）。GFN–OT 的评估若用"与精确 OT 解的传输代价差"，必须报多 seed 与置信区间，并在可解的小图上用 LP 做锚点（O08 已经这么做了）。
7. **一个尚未被利用的技术点：Radon–Nikodym 权重即"状态依赖步长"（Eq. (23)）。** 它说明理论上正确的更新对访问频率低的状态应该有更大的有效步长。GFlowNet 里存在同构问题（罕见状态的 credit assignment），可以据此设计一个显式的重要性权重项，并与 T05（SubTB）/T10 的方差分析对照。
8. **隐式 vs 显式更新这条区分值得搬到 GFlowNet 侧做消融。** Appendix H.1 的 Table 2 把 \(\gamma\)-Sinkhorn 与 \(\gamma\)-IMF 的差别定位为"隐式更新 vs 显式更新"（\(v^{n+1}_t=v^n_t-\delta\nabla_{\mu_n}L_t(v^{n+1},\bar{\mathbb{P}}^n)\) 对 \(v^{n+1}_t=v^n_t-\delta\nabla_{\mu_n}L_t(v^{n},\mathbb{P}^n)\)，Eq. (51) vs (52)）。GFlowNet 的 target network / stop-gradient 选择正是同一个轴上的选择，但据本报告所知尚无人把它当作"隐式–显式离散化"来分析。
9. **一个负面结论要记住，别浪费算力去复现"在线一定更好"。** AFHQ-64 上在线微调 FID 比迭代微调差（28.75 vs 25.41，Table 1），2D 玩具上 \(\alpha\)-DSBM 在 3/4 任务上不如两个基线（Table 4）。若 GFN–OT 里也做"在线 vs 分阶段"消融，应预期收益只出现在"内层子问题很贵"的大规模离散环境（这与 Appendix B 的成本模型一致），而不是小环境。

## 6.5 O05 · Universal Neural Optimal Transport（ICML 2025）

> **一句话**：UNOT 训练一个 Fourier Neural Operator \(S_\phi\)，输入一对离散测度、输出熵 OT 的对偶势 \(g\)，再由势导出计划、距离、重心、测地线；训练靠一个对抗生成器 \(G_\theta\) 造分布对 + 一个自监督 bootstrapping 损失（用 5 步 Sinkhorn 的结果当目标），因此**不依赖任何真实数据集**。它是本仓库里唯一一篇真正做到"跨数据集、跨分辨率泛化"的摊销 OT 求解器——也因此是评估「条件 GFlowNet–OT 是否真有跨任务泛化优势」时无法回避的强基线：如果一个条件 GFN–OT 打不过"把图最短路矩阵喂给 UNOT 式对偶预测器 + 一步 Sinkhorn"，那它的泛化叙事就不成立。

| 字段 | 内容 |
|---|---|
| arXiv | [2212.00133](https://arxiv.org/abs/2212.00133)（v6, 2025-06-12） |
| 发表 | **ICML 2025 主会**（原文首页：Proceedings of the 42nd ICML, Vancouver, PMLR 267, 2025） |
| 作者 | Jonathan Geuter\*（Harvard SEAS / Kempner Institute）、Gregor Kornhardt\*、Ingimar Tomasson\*（TU Berlin 数学系）、Vaios Laschos（Weierstrass Institute）；前三人并列一作 |
| 代码 | https://github.com/GregorKornhardt/UNOT （含实验用模型权重） |
| 本仓库 PDF | `papers/2212.00133.pdf` · 中译 未生成 |
| 阅读优先级 | **P1** —— 它定义了"摊销 OT 求解器"这条赛道的当前 SOTA 与评估协议；GFN–OT 的泛化主张必须相对它来陈述 |

#### 2. 核心贡献（按原文编号）

**(C1) 第一个能跨数据集、跨输入维度泛化的 neural OT 求解器（Sec. 1 贡献列表第 1 条）。** 结论：固定代价下，一个模型处理 \(10\times10\) 到 \(64\times64\) 的任意分辨率、任意数据集，相对误差 1–3%。对比对象 Meta OT（Amos et al. 2023）只能吃固定维度、且离开训练集就崩（Table 4）。

**(C2) 对偶势的离散→连续收敛性（Proposition 2，正式版在 Appendix B）。** 结论：若 \(c\) 对两个变元都 Lipschitz、\(\mathcal{X}\subset\mathbb{R}^N\) 紧、\((\mu_n),(\nu_n)\) 是绝对连续 \(\mu,\nu\) 的离散化序列，则（做了 \(f_n(x_0)=0\) 规范化后的）延拓势 \(f_n,g_n\) **在整个 \(\mathcal{X}\) 上一致收敛**到连续问题的解 \((f,g)\)。这是用 neural operator（而不是普通网络）的正当性来源：被学的对象本身有连续极限。

**(C3) 生成器的普遍性（Theorem 3 + Corollary 4）。** 设 \(0<\lambda\leqslant1\)、\(G_\theta(z)=\mathrm{ReLU}(NN_\theta(z)+\lambda z)\)、\(\mathrm{Lip}(NN_\theta)=L<\lambda\)，则 \(\tilde G_\theta(z)=NN_\theta(z)+\lambda z\) 在 \(\mathbb{R}^d\) 上可逆（引 Behrmann et al. 2019 的 invertible ResNet 结果），且

\[
\rho_{G_{\theta\#}\rho_z}(x)\ \geqslant\ \frac{1}{(L+\lambda)^d}\,\mathcal{N}\big(\tilde G_\theta^{-1}(x)\,\big|\,0,I\big)\qquad \forall x\in\mathbb{R}^d_{\geqslant0}.
\]

即生成器在**任意**非负 \(x\) 处密度为正：训练过程原则上可以见到任何一对固定维度的离散分布。Corollary 4 推广到 \(\tilde G_\theta\) 的复合（覆盖一大类 ResNet）。**注意 Appendix C 自己承认 Theorem 3 与实际架构有三处不符**（加了常数 \(\delta\)、输入输出维度不同、训练中未控制 Lipschitz 常数），所以这是"架构类的可能性定理"，不是"本文实现的保证"。

**(C4) 自监督 bootstrapping 损失（Proposition 5）。** 令 \(\tau_k\) 表示以当前预测热启动、跑 \(k\) 步 Sinkhorn 得到的 \(\boldsymbol{g}_{\tau_k}\)（并平移到零和）。则

\[
L_2(\boldsymbol{g}_\phi,\boldsymbol{g})\ \leqslant\ c(K,k,n)\,L_2(\boldsymbol{g}_\phi,\boldsymbol{g}_{\tau_k}),\qquad c(K,k,n)>1 .
\]

即最小化"与 \(k\) 步 Sinkhorn 结果的差"必然压住"与真解的差"。证明走 Hilbert 投影度量（Appendix B, Lemma B.5 + Peyré–Cuturi Thm. 4.1 的收缩性）。**这条是全文最可迁移的技术**（见第 7 节）。取 \(k=5\)、\(\epsilon=0.01\)。

**(C5) 对抗训练目标（Eq. (7) + Algorithm 2）。** \(\max_\theta\min_\phi\mathbb{E}_{z\sim\rho_z}\big[L_2\big(\tau_k(G(z),S(G(z))),\,S_\phi(G_\theta(z))\big)\big]\)，其中无下标的 \(S,G\) 表示不追踪梯度。生成器**最大化**求解器的损失（与 GAN 中生成器最小化判别器目标的方向相反，Sec. 5 专门澄清了这点）。

**(C6) 下游能力：不只是距离。** 用 \(\nabla_{\boldsymbol{a}}\mathrm{SD}_\epsilon(\mu,\nu)=\boldsymbol{f}-\boldsymbol{p}\)（Eq. (10)，引 Feydy et al. 2018；\(\boldsymbol{p}\) 是 \((\mu,\mu)\) 的势）做投影梯度下降求 Sinkhorn divergence 重心（Eq. (9)、Algorithm 3，200 步）；用 \(t\)-加权重心或熵计划近似 McCann 插值求测地线（Sec. 4.3）；用 Eq. (11)（引 Li et al. 2024b）做"分布之上的分布"的 Wasserstein-on-Wasserstein 梯度流（Sec. 4.4）。

**(C7) 离散化误差的度量几何刻画（Appendix A.3, Prop. A.13 + Cor. A.14）。** \((\mathcal{P}_2([0,1]^2),W_2)\) 与 \((\mathcal{P}([[n]]^2/n),W_2)\) 是 \((1,\tfrac{1}{\sqrt{2n}})\)-拟等距的：

\[
W_2(\mu,\nu)-\tfrac{1}{\sqrt{2n}}\ \leqslant\ W_2(f(\mu),f(\nu))\ \leqslant\ W_2(\mu,\nu)+\tfrac{1}{\sqrt{2n}} ,
\]

因此连续常速测地线在离散空间里是 strong-\(\epsilon\) 拟测地线。**这条对 GFN–OT 极有用**：它给出"在网格图上算 OT 与在连续空间算 OT"之间误差的显式尺度 \(O(n^{-1/2})\)，是把 hypergrid GFlowNet 的结论翻译回连续 \(W_2\) 语言的桥。原文给出的动机是"离散空间里不存在非常值的常速测地线"（Appendix A.3 的 Dirac 反例），所以必须放松到拟测地线。

#### 4. 实验与证据

测试集：MNIST（28×28）、灰度 CIFAR10（28×28）、Quick, Draw! 的 teddy bear 类（64×64）、LFW（64×64），以及跨数据集组合 CIFAR-MNIST、LFW-BEAR（\(\mu\) 来自一个集、\(\nu\) 来自另一个）。附录另加 CARS 与 Facial Expressions（48×48）。球面设定把图像投到 \(S^2\)。除注明外均在 \(S_\phi\) 输出上跑**一步** Sinkhorn 以得到 \(\boldsymbol{f}\)；误差在 500 个样本上平均。

| 任务 | 基线 | 指标 | 关键数字（出处） |
|---|---|---|---|
| 单步 Sinkhorn 后 OT 距离相对误差，\(c=\|x-y\|_2^2\)，全部缩放到 28×28 | Meta OT、Gauss（Thornton–Cuturi 2022）、Ones（默认 \(\mathbf{1}_n\)） | 相对误差（%） | UNOT **2.7±2.4 / 1.3±1.1 / 2.8±2.6 / 1.5±1.3 / 2.0±1.6 / 1.8±1.3**（MNIST/CIFAR/MNIST-CIFAR/LFW/BEAR/LFW-BEAR）；Meta OT 2.4±1.8 / 23.1±15.7 / 11.4±5.8 / 24.6±15.7 / 11.8±8.3 / 31.0±14.8；Gauss 18.1 / 19.7 / 32.2 / 21.1 / 20.4 / 19.3；Ones 39.5 / 47.4 / 74.5 / 56.9 / 54.2 / 66.4（Table 4） |
| 达到 0.01 相对误差所需 Sinkhorn 迭代数 | Ones、Gauss | 平均迭代数 | UNOT 3±5 / 3±6 / 4±4 / 7±8 / 4±6 / 4±6；Ones 16±9 / 80±22 / 32±15 / 78±20 / 41±16 / 53±18；Gauss 10±7 / 52±19 / 13±9 / 35±14 / 25±13 / 29±13（Table 1） |
| 达到 0.01 相对误差的 wall-clock 加速比（JAX，batch 64，float32，RTX 4090） | Ones | 加速倍数 | MNIST 1.25、CIFAR **7.4**、CIFAR-MNIST 2.07、LFW 5、BEAR 3.8、LFW-BEAR 4.4；28×28 平均 3.57、64×64 平均 4.4；Meta OT 原文报的加速为 1.96（Table 2） |
| MNIST 重心（100 步梯度下降后） | Gauss、Ones 初始化 | 到真重心的 \(W_2\) | UNOT **0.021±0.011**、Gauss 0.033±0.018、Ones 0.057±0.034（Table 6） |
| 测地线（McCann 插值） | 真 OT 计划、**GeONet（本仓库 O03）** | 视觉对比 | 原文 Sec. 4.3 + Fig. 8：UNOT 的两种测地线（由 OT 计划、由重心）都比 GeONet 更接近真值，**尽管 UNOT 既没在测地线上训练也没见过 MNIST**。仅定性，无数值表 |
| 跨分辨率（10×10 至 64×64 上/下采样） | Ones、Gauss | 相对误差曲线 | Fig. 12：全区间稳定；Appendix D.5 的变-\(\epsilon\) 变体在分辨率 \(<15\times15\) 或接近 \(70\times70\) 时误差上升 |
| 变 \(\epsilon\)（0.01–1，作为第三个输入通道） | — | 相对误差热图 | Fig. 13：跨 \(\epsilon\) 表现相对稳定（同一模型） |
| 生成器难度追踪 | — | 生成样本的 OT 距离误差 | 训练 0% 时新生成的样本误差 53.2%，10% 时 3.1%，之后 1.6–2.1%；在训练结束时回测早期样本均降到 1.1–2.0%（Table 5） |

证据强度分级：

- **实验直接支持**：跨数据集泛化（Table 4 里 Meta OT 在非训练集上错 11–31%，UNOT 一律 1.3–2.8%）；跨分辨率泛化（Fig. 12）；作为 Sinkhorn 初始化的 SOTA 地位（Table 1/2）；重心比其他初始化更准（Table 6）。
- **诚实的例外**：MNIST 上 Meta OT（2.4%）略优于 UNOT（2.7%）——但 Meta OT 是**在 MNIST 上训练的**，UNOT 没见过。原文把这解读为"生成器覆盖了 MNIST 类分布"，这是推断而非证明。另外 Appendix D.7 指出**在 MNIST 上默认初始化的 wall-clock 反而更快**（Fig. 17），与 Table 2 的 1.25 倍是同一现象的两面：小问题上网络前向的开销吃掉了迭代数的收益。
- **作者自陈的实现劣势**：FNO 处理复数而 PyTorch 对实数优化更好，若有复数 kernel 支持会更快（Sec. 4.1 脚注 7 与 Appendix D.7 都提到）；Thornton–Cuturi 的初始化在他们手上实现得比默认还慢，故未进入 wall-clock 图。
- **未验证的便宜替代**：Appendix D.3 报告，固定尺寸场景下把 neural operator 换成 MLP，**几分钟**就能训到 <5% 相对误差。这条对"是否真的需要 operator"提出了严肃的性价比问题，原文没有给出 MLP 版的完整对照表。
- **限制（Sec. 6 原文自述）**：不能外推到显著高于训练分辨率的输入；不能泛化到训练代价之外的代价函数。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：Sinkhorn（Cuturi 2013）与其初始化研究（Thornton & Cuturi 2022）；Meta OT（Amos et al. 2023，用对偶目标本身作损失、完全无监督但不能 OOD）；Neural Operator / FNO（Li et al. 2021；Kovachki et al. 2024）；Sinkhorn divergence 的梯度公式（Feydy et al. 2018）。本仓库内前驱是 O01（OT 基础）、O02（图上二次正则 OT，提供"离散/图结构 + 正则化"的先例）。
- **对照 / 竞争**：**O03（GeONet）是它直接击败的对象**（Sec. 4.3, Fig. 8）；C01（Unsupervised Learning for OT plan prediction between unbalanced graphs, NeurIPS 2025）是它在图结构上的对应物，也是 GFN–OT 更贴近的竞争者。
- **对主线（"内部流选择 = 最优传输"）的意义**：UNOT 本身**不**贡献任何流选择的机制，它贡献的是**这条主线的对照基准与评估协议**。具体说三件事：
  1. **它把"摊销"的正确对象定为对偶势**。O08 Theorem 3.3 给出的 min-flow GFlowNet 对偶 LP 里，对偶变量 \(\pi_x\) 就是图上的 Kantorovich 势。**这意味着"条件 GFN–OT"存在一个非常自然的实现路径：不去条件化 \(P_F\)，而去条件化势/状态流 \(\log F\)，再由势按 O08 的互补松弛导出策略。** UNOT 的成功给这条路径提供了经验支持。
  2. **它把评估协议钉死了**：跨数据集（\(\mu,\nu\) 来自不同族）、跨分辨率（含超出训练区间）、跨 \(\epsilon\)、以及"距离准 vs 几何准"的分离（Appendix A.2 反例 + Sec. 4.2/4.3 的重心/测地线实验）。GFN–OT 若只报同分布指标，说服力不足。
  3. **Prop. A.13 的 \((1,\tfrac{1}{\sqrt{2n}})\) 拟等距是把 hypergrid 结论翻回连续 \(W_2\) 的现成工具**：在 \(n\times n\) 网格图上算出的 OT 代价与连续 \(W_2\) 的偏差不超过 \(1/\sqrt{2n}\)。O08 的 hypergrid 实验（\(H=10\)）可以据此换算成连续语言，从而与 UNOT/GeONet 直接可比。

##### 对照实验怎么设计（评估"条件 GFN–OT 是否真有跨任务泛化优势"）

**任务定义。** 一个任务 \(=(G, L, R)\)：环境图 \(G=(\mathcal{S},E)\)、初始流分布 \(L\) 于 \(U=\mathrm{out}(s_0)\)、目标（reward）分布 \(R\) 于终止态 \(\mathcal{X}\)。按 O08 Theorem 3.2，最优解是以图最短路 \(d(u,x)\) 为代价的 Kantorovich 计划 \(\Pi^\star\)。条件 GFN–OT = 一个策略网络 \(P_F(\cdot|\cdot;\,\mathrm{enc}(G,L,R))\)，在任务分布上训练一次、对新任务零样本推理。

**四个基线，缺一不可。**

| 编号 | 基线 | 作用 |
|---|---|---|
| B1 | 精确 LP（`scipy.linprog`，同 O08 Sec. 4.1） | 提供 \(\mathrm{OT}^\star\) 锚点；只在小图可用 |
| B2 | 每任务从头训一个 GFlowNet 到收敛（O08 的 Sec. 3.3 算法） | "不摊销"的质量上界 + 成本上界 |
| B3 | **UNOT 移植版**：预计算最短路矩阵 \(D\)、令 \(K=\exp(-D/\epsilon)\)，用一个算子/GNN 预测 \(\boldsymbol{g}\)，+1 步 Sinkhorn 得 \(\boldsymbol{f}\)（本文 Eq. (5) + Prop. 1） | **核心对手**。它享有与 GFN–OT 完全相同的条件输入 \((L,R)\)，差别只在"用 Sinkhorn 解静态问题"vs"用策略解动态问题" |
| B4 | Meta OT 移植版：固定尺寸 MLP 预测 \(\boldsymbol{g}\)（Amos et al. 2023 的做法，本文 Table 4 的 MetaOT 列） | 检验"离散不变的算子参数化"是否真的必要；对应本文 Appendix D.3 的 MLP-UNOT 疑问 |

**指标（五项都报，理由在括号里）。**
1. 传输代价相对误差 \(|\sum_{u,x}\Pi_{u,x}d(u,x)-\mathrm{OT}^\star|/\mathrm{OT}^\star\)（O08 的主指标）。
2. 耦合本身的误差，例如 \(\|\Pi-\Pi^\star\|_1\) 或到 \(\Pi^\star\) 的 \(W_2\)（**必须单独报**，理由是本文 Appendix A.2 的反例：代价接近不代表耦合接近）。
3. 边缘约束违反度 MCV \(=\tfrac12(\|\mathbf{1}_m^\top\Pi-\nu^\top\|_1+\|\Pi\mathbf{1}_n-\mu\|_1)\)（本文 Eq. (20)；GFN 侧对应 reward matching 与 \(P_F(\cdot|s_0)=L\) 的违反度）。
4. 期望轨迹长度 \(\mathbb{E}[n_\tau]\)（O07/O08 指标；B3 没有轨迹，此项只对 GFN 侧有意义，用于说明"动态解额外提供了什么"）。
5. **达到 1% 相对误差的 wall-clock，含摊销**：横轴取任务数 \(N_{\mathrm{task}}\)，画出"训练 + \(N_{\mathrm{task}}\) 次推理"的总时间曲线，与 B1/B2 的 \(N_{\mathrm{task}}\times\) 单任务时间求交点。**这才是摊销主张的真正检验**，本文 Table 2 与 Appendix D.7（MNIST 上默认初始化反而更快）已经示范了这个坑。

**五条泛化轴（照抄 UNOT 的协议，并加一条 GFN 独有的）。**
- **(a) 同族新任务**：\((L,R)\) 从训练族里重采。对应本文的 MNIST→MNIST。
- **(b) 跨族**：\(L\) 取 moon 形、\(R\) 取角落多模态（O08 用的两类），训练时只见单族、测试时混搭。对应本文 CIFAR-MNIST / LFW-BEAR。**这是 Meta OT 崩掉、UNOT 没崩的那一列**，GFN–OT 必须在这里给出数字。
- **(c) 规模外推**：hypergrid 边长 \(H\) 从 8 扫到 64（含超出训练区间），或排列环境 \(n\) 从 10 扫到 20（O08 的做法，\(n=20\) 时 LP 已不可解，改用 B2 当参考）。对应本文 Fig. 12 与 Appendix D.5 的"超出训练分辨率就退化"。
- **(d) 换代价 = 换图**：训练在网格图上、测试在加了随机捷径/删了边的图上。**这一轴 UNOT 结构性地做不到**（Sec. 6 限制：不泛化到训练代价之外）——因为它的 \(C\) 在训练时被烧进权重，而 GFN–OT 的策略以 \(G\) 为输入。**如果条件 GFN–OT 有真实优势，只会出现在这一轴上。** 若这一轴也打不过"重算 \(D\) 后重跑 B3"，则应放弃泛化叙事，改主张"无需实例化 \(n\times n\) 核矩阵"的内存优势。
- **(e) 换 \(\epsilon\) / 换熵正则强度**：对应本文 Appendix D.5 的变-\(\epsilon\) 变体（把 \(\epsilon\) 作为额外输入通道）。GFN 侧对应"流正则系数 \(\lambda\)"（O08 Sec. 4.1 的消融）。

**两个必须控制的混淆变量。**
1. **训练任务分布的覆盖度**。UNOT 的泛化很大程度来自对抗生成器 \(G_\theta\)（Theorem 3 的密度下界 + Table 5 显示生成器确实先造出难样本）。如果条件 GFN–OT 只在手工设计的任务族上训练，(b)(c) 的对比就是不公平的。**要么给 GFN–OT 配一个同类的对抗任务生成器，要么把两者都在同一固定任务集上训练并同时报告**。
2. **计算预算对齐**。B3 的 35 小时级训练成本（本文 Sec. 4）与 GFN–OT 的采样成本量纲不同，必须在同一横轴（wall-clock 或函数求值次数）上比较，而不是比"迭代数"。

**可证伪的预测（写下来以便被打脸）。** 在 (a)(b)(c) 三轴上，B3（UNOT 移植 + 一步 Sinkhorn）在小到中等规模图上以显著优势胜过条件 GFN–OT，因为它解的是凸问题、且有 Prop. 5 那样的收缩性保证；条件 GFN–OT 的优势应当只出现在 (d)（换图/换代价）与"\(|\mathcal{S}|\) 大到无法存 \(D\) 或 \(K\)"的场景。

#### 7. 可复用的 insight 与开放问题

1. **把 bootstrapping 损失搬到 GFlowNet 上：用 \(k\) 步精确更新的结果当目标。** Prop. 5 的结构是"目标 = 自己的预测 + \(k\) 步收缩算子"，正确性靠收缩性。GFlowNet 里的对应收缩算子候选是"\(k\) 步 flow-matching/DB 精确回代"。**可写成一条定理草稿**：若离散 flow-matching 算子在某度量（例如 \(\log F\) 的 Hilbert 投影度量 / span 半范数）下是 \(\kappa<1\) 压缩，则 \(\|\log F_\phi-\log F^\star\|\leqslant\frac{1}{1-\kappa^k}\|\log F_\phi-\log F_{\tau_k}\|\)。这会给条件 GFN–OT 一个**不需要真值 \(Z\)** 的自监督损失。
2. **对抗任务生成器是被 GFlowNet 社区低估的部件。** Theorem 3 说明只要架构是 \(\mathrm{ReLU}(NN_\theta(z)+\lambda z)\) 且 \(\mathrm{Lip}(NN_\theta)<\lambda\)，生成器就能覆盖全部非负向量。移植到图任务：生成 \((L,R)\) 对的网络用同样的可逆 ResNet 结构 + softmax 归一化，即可获得同样的覆盖性论证。实验：对比"手工任务族训练"与"对抗任务生成训练"在泛化轴 (b)(c) 上的差距。
3. **\((1,\tfrac{1}{\sqrt{2n}})\) 拟等距（Prop. A.13）应当成为 GFN–OT 报数的标准换算。** 现在 hypergrid 上的 OT 代价数字与连续 \(W_2\) 文献不可比；有了这条界，可以直接声明"我们的解与连续 \(W_2\) 最优值相差至多 \(\mathrm{err}_{\mathrm{graph}}+1/\sqrt{2n}\)"。**顺带给出一个开放问题**：对非网格图（例如排列图、Cayley 图），是否存在类似的拟等距界？这决定了 GFN–OT 在这些环境上的结果能否被翻译成几何语言。
4. **"距离准 ≠ 几何准"必须写进 GFN–OT 的评估清单。** Appendix A.2 的两点反例可以逐字搬到图上：构造两个 reward 分布使传输代价任意接近而最优耦合固定相距。**这是一个 10 行的构造题**，做出来可以直接放进 GFN–OT 论文的评估动机段。
5. **一个负面性价比信号值得先做验证：MLP 够不够？** Appendix D.3 说固定尺寸下 MLP 几分钟训到 <5%。GFN–OT 若主张"必须用 GNN 才能跨图泛化"，应当先跑一个固定图规模的 MLP 条件版做对照，否则容易把"架构复杂度"误当成"泛化能力"。
6. **摊销的盈亏平衡点应当被显式画出来。** Table 2 的 1.25×（MNIST）与 7.4×（CIFAR）差 6 倍，Appendix D.7 甚至给出 MNIST 上默认初始化更快的反例。对 GFN–OT：画"任务数 vs 总 wall-clock"的交点图，明确说出"超过 \(N^\ast\) 个任务才划算"。这个数字比任何相对误差表都更能说服 reviewer。
7. **一个尚未有人做的组合：UNOT 式的势预测器 + O08 式的最短路代价 + GFlowNet 的策略导出。** 三步流水线：(i) 用算子网络预测图上的 Kantorovich 势 \(\pi_x\)（O08 Theorem 3.3 的对偶变量）；(ii) 由互补松弛条件确定支撑边集（最短路子图）；(iii) 在该子图上用 GFlowNet 只学"如何在最短路之间分配质量"。这样把凸的部分交给算子、把组合的部分交给策略，可能同时避开 B3 的内存瓶颈与 GFN 的收敛慢。**这是本报告认为最值得优先尝试的方案。**

## 6.6 O06 · 用流神经网络计算高维 OT（AISTATS 2025）

> **一句话**：Q-flow 用一个神经 ODE 直接解 Benamou–Brenier 动态 OT：两端分布 \(P,Q\) 都只有有限样本、都没有解析密度，于是把两个终端约束松弛成 KL（用 logistic 分类器现场估计密度比），再加一个有限差分形式的 \(W_2\) 传输代价项，双向交替训练。训练出的 OT 轨道被直接用于两个下游任务：图像到图像翻译与高维密度比估计（DRE）。**先把最容易的误读挡掉：标题里的 "flow neural network" 指的是连续正规化流 / 神经 ODE 的速度场，与 GFlowNet 无任何关系**（详见第 1 节末）。

| 字段 | 内容 |
|---|---|
| arXiv | [2305.11857](https://arxiv.org/abs/2305.11857)（v5, 2025-03-10） |
| 发表 | **AISTATS 2025 主会**（本仓库 `data/papers.yaml` 登记）；本仓库 PDF 为纯 arXiv v5，**页面上没有会议页脚**，故发表状态未能由 PDF 自身证实 |
| 作者 | Chen Xu（Georgia Tech, ISyE）、Xiuyuan Cheng（Duke 数学系）、Yao Xie（Georgia Tech, ISyE） |
| 代码 | https://github.com/hamrel-cxu/FlowOT （原文 Sec. 5） |
| 本仓库 PDF | `papers/2305.11857.pdf` · 中译 未生成 |
| 阅读优先级 | **P2** —— 与 GFlowNet 无结构共享，但它是"每对分布单独求解动态 OT"这一端的干净参照物（与 O03/O05 的摊销端正好相反），且它的 KL-松弛技巧对 GFN–OT 的边缘约束处理有直接可迁移性 |

#### 2. 核心贡献（按原文编号）

**(C1) Q-flow net：从两组样本直接学连续可逆传输映射，并用传输代价把它推向动态 OT（Sec. 3, Eq. (3)–(8)）。** 结论：端到端训练可以"精修"任何初始流（两个 CNF 拼接、或分布插值网络给出的流），使其从"只匹配终端"变成"匹配终端且传输代价小"。前提：初始流已经近似匹配两端（Sec. 3.3），否则分类器难训。

**(C2) 双向训练（Eq. (3) + Remark 1）。** 结论：Benamou–Brenier 对 \(P,Q\) 是对称的——把 \(P\) 放 \(t=0\) 或放 \(t=1\)，最优解是同一个速度场、只差时间方向。因此同时优化 \(L^{P\to Q}\) 与 \(L^{Q\to P}\) 在最优处指向同一个解，有限样本下可提升精度。**这条有实验支撑**（Table A.3：BPD 从单向 1.08/1.19/1.31 改进到双向 1.05/1.14/1.31）。

**(C3) flow-ratio net：用 OT 轨道做无穷小 DRE（Sec. 4, Eq. (10)–(12)）。** 结论：由 \(\log(q/p)=\int_0^1\partial_t\log p(x,t)\mathrm{d}t\)（Eq. (A.2)），把"时间 score" \(\partial_t\log p(x,t)\) 参数化为 \(r(x,t;\theta_r)\)，在相邻时间格点上做 logistic 分类训练；训好后任意 \(s<t\) 的 \(\log(p(x,t)/p(x,s))\) 都可积分得到。**关键论点是"用 OT 轨道当桥比用线性插值桥更好"**，因为后者（Eq. (A.8) 的 \(X(t_k)=\sqrt{1-\alpha_k^2}X(0)+\alpha_kX(1)\)，TRE/DRE-\(\infty\) 用的方案）与 OT 轨道明显不同（Fig. A.9 vs Fig. 3(a)）。

**(C4) 实验上同时打过静态与动态 OT 基线，以及 SB 基线（Sec. 5）。** 在 Korotin 的 OT benchmark、CelebA64 对齐、图像翻译、高维 MI、MNIST 能量模型上给出数字（见第 4 节）。

#### 6. 在 GFlowNet × OT 主线中的位置

- **前驱**：Benamou–Brenier；CNF / 神经 ODE（Chen et al. 2018；Kobyzev et al. 2020）；带传输正则的 CNF（Finlay et al. 2020；Onken et al. 2021；Xu et al. 2022, 2023 的 JKO-iFlow）；分布插值 / Flow Matching（Albergo–Vanden-Eijnden 2023；Lipman et al. 2023；Liu 2022）；DRE 侧是 TRE（Rhodes et al. 2020）与 DRE-\(\infty\)（Choi et al. 2022）。
- **对照 / 竞争**：与 **O03（GeONet）** 解同一个 Benamou–Brenier 问题但走反方向（O03 摊销 + PDE 残差，O06 单对求解 + KL 松弛 + 采样）；与 **O04（α-DSBM）** 竞争同一批图像翻译任务，且在本文的 Table 2 里 DSBM 落后明显；与 **O05（UNOT）** 完全不同层次——O05 摊销静态熵 OT 的对偶势，O06 不摊销、解动态 OT。
- **对主线（"内部流选择 = 最优传输"）贡献了什么**：它贡献的是**约束处理技术**，而不是流选择机制。具体三点：
  1. **"把硬边缘约束换成可微 KL 读数"这一招在 GFN–OT 上直接可用**。O08 的构造需要固定初始边流分布 \(P_F(\cdot|s_0)=L\)（O08 Sec. 3.3 所说的 leward matching）与 reward matching \(P_B(x|s_f)=R(x)/Z\)。这两个都是硬约束，实践中靠损失项软化。O06 的做法给出一个更强的版本：用一个二分类器估计"当前推过去的分布"与"目标分布"的密度比，从而得到 KL 的无偏读数——在图上就是"用一个判别器区分 GFlowNet 采出的终止态与目标 reward 采样"，可作为 reward matching 违反度的一个可微代理。
  2. **传输代价项的有限差分形式（Eq. (8)）是"期望轨迹长度"的连续对应物**。O07/O08 用 \(\mathbb{E}[n_\tau]\) 或总流量作为代价；O06 用 \(\sum_k\|X(t_k)-X(t_{k-1})\|^2/h_k\)。两者都是"沿轨迹累加局部移动量"，且都**免费**（前者是轨迹长度、后者是已算出的中间点）。这条对应关系说明：GFlowNet 的流正则化项在连续设定下就是动能项，反之亦然。
  3. **双向训练的对称性论证（Remark 1）在图上未必成立**，值得单独查。Benamou–Brenier 的时间对称性来自 \(P,Q\) 地位对等；GFlowNet 的图是有向的、\(s_0\) 与 \(s_f\) 地位不对等（O07 Assumption 2.1），所以"前向训练与后向训练指向同一解"这个结论**不能照搬**。这是一个具体的、可以判真假的问题。
- **它不属于主线的原因（必须说清）**：本文的 "flow neural network" 与 GFlowNet 同名不同物（第 1 节已澄清）。把 O06 归到 GFN–OT 的相关工作里，只能以"动态 OT 的神经解法"的身份，不能以"flow network"的字面相似性为由。

#### 7. 可复用的 insight 与开放问题

1. **用判别器读 KL，替代 GFN–OT 里的硬边缘约束。** 实验草案：在 O08 的 hypergrid 环境上，把 leward/reward matching 的平方损失换成"判别器估计的 KL"（Eq. (6)–(7) 的离散版），比较收敛速度与最终 \(\mathrm{OT}^\star\) 差距。风险点已被原文标出：内层判别器必须频繁重训，否则读数失效。
2. **一条定理草稿：KL-松弛的一致性。** 原文 Sec. 6 把"KL 处理边界条件的一致性分析"列为开放问题。在**有限图**上这个问题反而更容易：状态空间有限、密度比是有限维向量，可以直接给出"判别器估计误差 → KL 估计误差 → 最终耦合的传输代价误差"的传播界。**做出来同时补上 O06 的理论空白与 GFN–OT 的约束松弛理论。**
3. **"OT 轨道 vs 线性插值桥"这个消融必须补，而且在图上是廉价的。** O06 没做（第 4 节已指出）。图上可做：固定 GFlowNet 的下游任务（例如用中间状态分布做课程学习或 DRE），分别用 (i) 最小流量 GFlowNet 的轨道、(ii) 随机游走轨道、(iii) 均匀插值轨道当桥，看下游指标差异。这直接检验"最优轨道是否真的更有用"这个 GFN–OT 的隐含卖点。
4. **可逆性作为诊断量值得引入。** Table A.2 的逆映射误差（\(10^{-7}\sim10^{-5}\)）是一个便宜且信息量大的健康检查。GFlowNet 的对应量是"\(P_F\) 与 \(P_B\) 诱导的轨迹分布之差"（O08 Eq. (1) 的违反度），建议在训练日志里当作一等公民打印。
5. **时间方向对称性在有向图上是否成立，是一个可判定的小问题。** Remark 1 的论证依赖 \(P,Q\) 对等；图上 \(s_0\to s_f\) 有向。可构造反例或给出成立条件（例如图是可逆/双向的、或状态空间上存在使转移核可逆的参考测度）。
6. **摊销 vs 单对求解的成本交点。** O06 单对训练 1.25–2.5 小时（Table A.1），O05 一次性训练约 35 小时。因此"摊销划算"的阈值大约在 15–30 个任务量级。**这是把 O05 与 O06 的数字直接相除得到的粗估**（本报告的计算，非原文结论），但它给 GFN–OT 的摊销主张定了一个必须跨过的门槛。
7. **一个尚未有人做的组合**：用 O06 的判别器 KL 读数处理边缘约束、用 O08 的最小总流量选流、用 O05 的算子网络做跨任务摊销。三者的约束/代价/摊销分工互不冲突，是一条干净的组合路线。
8. **"精修已有流"这个范式本身可以搬到 GFlowNet。** O06 的定位不是从零学 OT，而是把任意初始流（两个 CNF 拼接 / 插值网络）**精修**成 OT 流（Sec. 3.2 的 "refinement"），预训练只占总时间的 9–12%（Appendix B.1）。GFN–OT 的对应做法：先用标准 TB 训一个只满足 reward matching 的 GFlowNet（不管流量大小），再加上流量正则做第二阶段精修，而不是一开始就带着正则项训。这与 O04 的两阶段（bridge matching 预训练 + \(\alpha\)-DSBM 微调）结构一致，**两篇不同路线的论文都收敛到"先匹配端点、再优化代价"这个顺序，值得当作一条经验规律采纳并验证**。
9. **副产品视角：最优耦合可以当采样器修正器。** 见第 3 节 (6)。GFN–OT 里的对应实验：用学到的最优耦合把一个已有的（次优的）终止态采样器修正到目标 reward 分布，报告修正前后的 TV 距离——O08 已经在用 TV 作指标（其 Table 1），所以这个实验的评估管线是现成的。


# 第 7 章 竞争格局与撞车分析

三篇竞品全部是主会（NeurIPS 2025、ICML 2026、ICLR 2025），本方两篇是 Workshop。本章先给三篇的核心贡献与主线位置，再收录完整的竞品矩阵。

## 7.1 C01 · ULOT：不平衡图间 OT 计划的无监督预测（NeurIPS 2025）

> **一句话**：ULOT 用一个条件于 FUGW 超参 \((\alpha,\rho)\) 的 GNN + 交叉注意力网络，直接把 FUGW 损失当训练信号，摊销地预测两张**显式图**之间的不平衡 OT plan，推理复杂度 \(O(n_1n_2)\)，比经典求解器快两个数量级。它是「摊销求解一族图 OT」这个课题目前最强的占位者：O08 想做的「条件 GFN 学一族图 OT」在**摊销、条件化、图、无监督**这四个词上被它占了大半，唯一没被占的是「隐式组合状态图 + 可执行局部动作」。

| 字段 | 内容 |
|---|---|
| arXiv | [2506.12025](https://arxiv.org/abs/2506.12025)（arXiv API：published 2025-05-21，updated 2025-07-08；本地 PDF 页边为 v3, 8 Jul 2025） |
| 发表 | **NeurIPS 2025 主会**（Main Conference Track，已核实：NeurIPS proceedings 页面标注 "Advances in Neural Information Processing Systems 38 Main Conference (NeurIPS 2025) Main Conference Track"，DOI 10.52202/085713-3146；dblp 收录条目 venue=NeurIPS year=2025 type=Conference and Workshop Papers）。注意 PDF 首页仍写 "Preprint. Under review."，那是 arXiv v3 快照，不是最终状态 |
| 作者 | Sonia Mazelet, Rémi Flamary, Bertrand Thirion（CMAP École Polytechnique / Inria-Saclay） |
| 代码 | <https://github.com/smazelet/ULOT>（`github.com/SoniaMaz8/ULOT` 会 301 重定向到该地址；原文附录 A 说 "code is available in the supplementary materials and will be released on github upon publication"，并承诺放出预训练权重） |
| 本仓库 PDF | `papers/2506.12025.pdf` |
| 阅读优先级 | **P1** — 不是理论竞品，是**课题占位竞品**：它决定了「条件/摊销图 OT」这个方向还剩多少空地 |

#### 2. 核心贡献（按原文编号）

原文**没有任何 Theorem / Proposition / Lemma**，全部贡献是架构与实证层面的。这一点要先说清楚：ULOT 的说服力来自实验曲线，不来自定理。

1. **摊销训练目标**（原文 §2.2 Eq. (4)）：
   \(\min_\theta\ \mathbb E_{G_1,G_2\sim\mathcal D^2,\ \alpha,\rho\sim\mathcal P}\big[\mathcal L_{\alpha,\rho}(G_1,G_2,P_\theta^{\rho,\alpha}(G_1,G_2))\big]\)。
   无监督的含义就在这里：**不需要预先用求解器算出 ground-truth plan**，直接把 FUGW 目标本身当损失反传。这是 amortized optimization（Amos 2023）的标准套路，但被搬到了非凸二次 OT 上。

2. **参数条件化**（原文 §2.2「Encoding the parameters」+ Eq. (5)）：\(\rho\) 是正标量直接拼进节点特征；\(\alpha\) 用 Fourier 基做位置编码
   \(\hat\alpha=\big[(\cos(k\pi\alpha))_{k=1..d}\ \big|\ (\sin k\pi(1-\alpha))_{k=1..d}\big]\)，实验取 \(d=10\)。采样分布：\(\mathcal P_\rho\) 是 \([10^{-7},1]\) 上的 log-uniform，\(\mathcal P_\alpha=\mathrm{Beta}(0.5,0.5)\)。原文自评这条"is particularly novel and has not been done before to the best of our knowledge"（§2.4）。

3. **交叉注意力架构**（原文 §2.3 Eq. (6)–(10)，Figure 1）：每层两条路——self path 是各自的 GCN \(F_k^{\text{self}}=\mathrm{GCN}(F_k)\)；cross path 先算 \(F_k^{\text{cross}}=\mathrm{MLP}(F_k,\rho,\hat\alpha)\)，再算余弦相似度矩阵 \(S_{i,j}=s((F_1^{\text{cross}})_i,(F_2^{\text{cross}})_j)\) 与行/列 softmax \(S_1=\mathrm{softmax}_{\text{row}}(a^2S),\ S_2=\mathrm{softmax}_{\text{col}}(a^2S)\)（\(a\) 是温度超参）。

4. **不平衡 plan 输出层**（原文 Eq. (11)(12)）：先预测节点门控 \(v_k=\mathrm{sigmoid}(\mathrm{Linear}(F_k^{\text{final}},\rho,\hat\alpha))\)，再
   \(P_\theta^{\rho,\alpha}(G_1,G_2)=\tfrac12\big(\tfrac1{n_1}S_1\mathrm{diag}(v_1)+\tfrac1{n_2}\mathrm{diag}(v_2)S_2\big)\)。
   设计意图原文写得很直白：把「跨图节点对应」（交叉注意力管）和「每个节点该出多少质量」（\(v_k\) 管）解耦，后者才是 unbalanced OT 特有的自由度。

5. **三个下游用法**（原文 §3.1、§3.2）：(a) plan 对 \((\rho,\alpha)\) 完全可微 → 可以对超参做梯度下降（bi-level）；(b) plan 对图输入可微 → 可以优化 FUGW 泛函的图；(c) plan 总质量 \(m(P)=\sum_{i,j}P_{i,j}\) 当作 \(O(n^2)\) 的图相似度。

#### 5. 前提假设与适用边界

ULOT 的正向适用范围：

1. **两张图都能显式实例化**：需要能构造并放进显存的 \(F_k\) 与 \(D_k\)（邻接或最短路距离矩阵），以及 \(n_1\times n_2\) 的 plan 矩阵。
2. **节点权重均匀**：\(\omega_k^i=1/n_k\)，原文 §2.1 显式假设。非均匀权重需要改动。
3. **规模上限 \(n\le 10^4\)**：原文 §4 明说受 GPU 显存限制，超过需要 lazy tensor 等专门技术。
4. **有同分布的图对训练集**：SBM 靠生成器，fMRI 靠随机 parcellation 数据增强。换一个图族要重训。
5. **cost 结构锁死在 FUGW**：条件化只覆盖 \((\alpha,\rho)\) 两个标量，不覆盖 cost 函数形式本身。
6. **输出是 plan 矩阵，不是策略**：告诉你"多少质量从 \(i\) 到 \(j\)"，不告诉你"怎么从 \(i\) 走到 \(j\)"。对没有中间路径概念的节点匹配任务，这不是缺陷；对需要执行合法局部动作的任务，这是硬伤。

#### 6. 在 GFlowNet × OT 主线中的位置

##### 6.1 与 O08「条件 GFN 学一族图 OT」的重叠面积

**被占掉的部分（重叠约 60–70%）**：

| 卖点 | ULOT 是否已做 | 证据 |
|---|---|---|
| 摊销：一个模型服务一族 OT 问题 | ✅ 完全做到 | Eq. (4)，IBC 上 14400 图对训一个模型 |
| 条件化：策略/网络吃 OT 问题参数 | ✅ 做到 \((\alpha,\rho)\) | Eq. (5)，Fourier 编码 |
| 无监督：不需要 ground-truth plan | ✅ 做到 | Eq. (4) 直接用 OT 目标当 loss |
| 图结构 | ✅ GNN + 交叉注意力 | §2.3 |
| 不平衡（质量可增减） | ✅ \(\rho\)-KL + \(v_k\) 门控 | Eq. (3)(11) |
| 「神经预测 → 经典 solver warm start」范式 | ✅ 已演示 | Figure 8(right) |
| 对未见图对泛化 | ✅ 60/20/20 测试集 | §3.2 |

结论直说：**「条件 GFN 学一族图 OT」如果按字面做，卖点几乎全被 ULOT 占了，而且对方在 NeurIPS 2025 主会。** 特别是"GFN proposal + 经典 OT 修正"这一条，Figure 8(right) 已经把 idea 演示过一遍，只能降级成对照基线，不能当主线卖点。

**没被占的部分（差异化护城河）**：

1. **对象空间。** ULOT 需要显式 \(D_1,D_2\) 和 \(n_1\times n_2\) 的 plan，硬上限 \(n\le10^4\)。O08 的 permutation 环境 \(n=20\) 时状态数是 \(20!\approx2.4\times10^{18}\)，plan 矩阵在物理上不可能实例化。这是**质的差别，不是量的差别**。
2. **输出类型。** ULOT 给 plan，O08 给 \(P_F(s'|s)\)。当"传输"必须通过一串合法局部动作实现（分子编辑、排列的相邻对换、程序变换），plan 本身是无用的——你还得再解一遍"怎么走"。
3. **cost 结构。** ULOT 是二次 FUGW（GW 项耦合两个 plan 元素），O08 是线性 Kantorovich、cost 由图最短路 \(d_G(u,x)\) 诱导。两者不是同一个数学问题，不能直接比数字。
4. **误差保证。** ULOT 一条都没有。O08 有 LP 强对偶（Theorem 3.3）、互补松弛给出 primal-dual gap。**这是 GFN 侧唯一在理论上明确占优的接口**，也是「Balance 残差 → OT 误差界」课题的立身之本。
5. **不平衡方向反转。** ULOT 天然 unbalanced，O08 硬性要求两端归一。所以"unbalanced GFN-OT"这个课题**在设定上与 ULOT 撞车，但在方法上不撞**——ULOT 的 \(v_k\) 门控可以直接搬去做 GFN 的可学习质量汇。

##### 6.2 一张对照表：ULOT 与 O08 的可比与不可比

| 维度 | ULOT（C01） | O08 |
|---|---|---|
| 图的角色 | 被传输的对象（两张显式图的节点集） | 传输发生的状态空间（一张有向图的内部） |
| 规模 | \(n\le10^4\)（原文 §4 显存上限） | permutation \(n=20\)，状态数 \(20!\approx2.4\times10^{18}\)（O08 §4.2） |
| cost | 二次 FUGW（W + GW + KL） | 线性 Kantorovich，cost \(=d_G(u,x)\) 最短路（O08 Eq. (7)(8)） |
| 边缘约束 | 软（KL 罚，\(\rho\)） | 硬（Assumption 3.1：\(\sum L=\sum R=1\)） |
| 输出 | plan 矩阵 \(P\in\mathbb R^{n_1\times n_2}\) | 前向策略 \(P_F(s'|s)\)，coupling 是采样副产品 |
| 摊销 | 是（一模型多图对 + 多超参） | 否（每对 \(L,R\) 训一个模型） |
| 理论保证 | 无（全文无定理） | Theorem 3.2（GFlow\(^\star=\)OT\(^\star\)）、Theorem 3.3（对偶 + 互补松弛） |
| 加速证据 | 100× vs solver（Figure 7 右） | 与 POT 精确解一致：\(H=10\) 时 \(\mathbb E|\tau|=3.990\pm0.015\) vs OT\(^\star=3.997\)（O08 Table 1） |
| 代码 | 已开源 | 未见公开 |

**不可比之处必须讲明**：两者的 OT 问题不同（二次 vs 线性）、cost 不同（FUGW vs 图最短路）、边缘约束强度不同，因此**任何"ULOT 快 100 倍，所以 GFN 没戏"的论断都是错误比较**。真正的竞争在于"课题叙事"层面，而不是数字层面。

##### 6.3 前驱与对照

- **前驱**：FUGW（Thual et al., NeurIPS 2022，原文引 [30]）；GW（Mémoli 2011 / Peyré et al. 2016）；Meta OT（Amos et al. 2022，原文引 [2]，ULOT 自称是它在 unbalanced 图 OT 上的推广）；amortized optimization（Amos 2023）。
- **对照/竞争**：O05 Universal Neural OT（ICML 2025）；Neural GW OT（Nekrashevich et al. 2023）；深度图匹配那一系（SuperGlue 等，但那些是监督的）。
- **对本仓库主线的贡献**：ULOT 是"内部流选择 = 最优传输"这条线的**外部竞品参照系**。它证明了摊销图 OT 这件事在工程上是可行且有价值的（100× 加速是真的），同时也证明了它做不到什么（隐式图、可执行策略、误差证书）。差异化必须钉死在后三点上。

#### 7. 可复用的 insight 与开放问题

1. **把 \(v_k\) 门控搬进 GFN 做 unbalanced OT**。ULOT Eq. (11)(12) 用 sigmoid 门控每个节点的输出质量。GFN 侧的对应物是：把终止边流 \(F(x\to s_f)\) 参数化为 \(g_\theta(x)\cdot R(x)\)，\(g_\theta\in(0,1)\)，并在损失里加 \(\rho\,\mathrm{KL}\) 罚代替硬 reward matching。这直接回应 O08 遗留的"未知 \(Z\) / 质量不等"开放问题，且是一条可以在一周内跑通的实验。
2. **Fourier 参数编码 → 条件 GFN 的正则强度输入**。O08 的 \(\lambda\)（流正则系数）目前是外部超参，实验 Table 2 显示 \(\lambda=10^{-1}\) vs \(10^{-2}\) 在轨迹长度和采样精度之间强烈折中。把 \(\lambda\) 按 ULOT Eq. (5) 编码进 \(P_F(\cdot|s,\lambda)\)，一次训练就能给出整条正则路径，是"条件化"在 GFN 侧最容易做且不与 ULOT 正面撞车的形态。
3. **plan-level 评测是空缺，抢先占位**。ULOT 只报 loss 相关性。任何 GFN-OT 工作只要报告 cost gap、边缘 TV、支撑重叠率、primal-dual gap（O08 Theorem 3.3 现成提供），在评测严谨性上就压过 ULOT 一头。
4. **break-even 分析没人做**。摊销方法的核心问题是"训练成本要摊到多少次查询才划算"。ULOT 附录 A 给了 100 GPU 小时但没给 break-even。这是一条谁都能做、谁做谁得分的实验。
5. **反向问题：ULOT 能不能做隐式图？** 不能，因为交叉注意力需要枚举 \(n_1\times n_2\) 对。但可以问一个更尖锐的问题：**如果把 O08 的 permutation 环境压缩到 \(n_1,n_2\le10^4\) 个"代表状态"，ULOT 是不是就能打过 GFN？** 这是必须自己先跑一遍的 red-team 实验，否则审稿人会问。
6. **待证命题草稿**：ULOT 完全没有"预测 plan 与最优 plan 的距离界"。对应的 GFN 版本是可能的——设 TB 损失残差为 \(\varepsilon\)，能否证明 \(\big|\mathbb E[|\tau|]-\mathrm{OT}^\star\big|\le C(\varepsilon,\lambda,|\mathcal S|)\)？O08 Theorem 3.3 的互补松弛式 \(F^\star(s\to s')(\pi^\star_{s'}-(1+\pi^\star_s))=0\) 是这条界的天然起点。

## 7.2 C02 · GSBoG：图上广义 Schrödinger 桥（ICML 2026）

> **一句话**：GSBoG 把广义 Schrödinger 桥（GSB）从 \(\mathbb R^d\) 搬到**固定稀疏图上的受控 CTMC**，用 Hopf–Cole 势 \(V_t\) 把最优跳转率写成 \(u_t^\star(y,x)=r_t(y,x)e^{V_t(x)-V_t(y)}\)，再用 gIPF + TD 两个损失把它学出来。它与 O08 是**同一象限的两个极点**：都在「有向图 + 两端边缘 + 输出可执行局部策略」的设定下挑一个路径测度，差别是 GSBoG 选「熵正则（KL 到参考动力学）+ 中间态成本」，O08 选「无正则的最小总流」。这是本仓库里对 O08 定位威胁最直接的一篇。

| 字段 | 内容 |
|---|---|
| arXiv | [2602.04675](https://arxiv.org/abs/2602.04675)（v1 2026-02-04，v2 2026-06-10） |
| 发表 | **ICML 2026 主会**。核实链条：(a) PDF 页脚为 camera-ready 格式 "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"；(b) 通讯作者 Jaemoo Choi 个人主页 2026-04 条目 "7 papers are accepted to ICML"，明列 "GSBoG: Generalized Schrödinger Bridge Sampler on graphs"；(c) Georgia Tech Research Conference Spotlight 2026 收录该题目。**但 dblp 截至 2026-09 仅有 CoRR 条目**，PMLR 正式卷尚未可检索，故元数据 `venue_type` 记为 `main` 并附注此差异 |
| 作者 | Panagiotis Theodoropoulos, Juno Nam, Evangelos Theodorou, Jaemoo Choi（Georgia Tech / MIT） |
| 代码 | **未公开**（正文与附录均无仓库链接，致谢/Impact Statement 也未提及） |
| 本仓库 PDF | `papers/2602.04675.pdf`（29 页） |
| 阅读优先级 | **P1**（对「熵正则 GFN–OT/SB」课题而言实为 P0） |

#### 2. 核心贡献（按原文编号）

1. **Theorem 3.1（对偶表示）**。在"mild regularity assumptions"下，存在时变函数 \(V:[0,1]\times\mathcal X\to\mathbb R\)，使最优概率路径 \(p_t^\star\) 满足耦合系统（原文 Eq. (15)）
   \[
   \partial_tp_t^\star(x)=\sum_y r_t(x,y)e^{-V_t(x)+V_t(y)}p_t^\star(y),\qquad
   \partial_tV_t(x)=\sum_y r_t(y,x)e^{-V_t(y)+V_t(x)}-f_t(x,p_t^\star),
   \]
   边界条件 \(p_0^\star=\mu,\ p_1^\star=\nu\)；且最优控制率为（Eq. (16)）
   \[
   \boxed{\,u_t^\star(y,x)=r_t(y,x)\exp\big(-V_t(y)+V_t(x)\big).\,}
   \]
   证明在附录 A.1：对 CE 引入拉格朗日乘子 \(V_t\)、分部积分、调用强对偶、对 \(u\) 求导得 \(\log(u/r)+V_t(y)-V_t(x)=0\)。

2. **Remark 3.3（拓扑自动满足）**。Eq. (16) 里 \(\exp(\cdot)>0\)，所以 \(r_t(y,x)=0\Rightarrow u_t^\star(y,x)=0\)。最优控制**自动继承参考生成元的稀疏支撑**，不需要额外的可行性投影。

3. **Proposition 3.4（图上 Hopf–Cole）**。定义 \(\varphi(t,x):=e^{-V(t,x)},\ \hat\varphi(t,x):=p_t^\star(x)/\varphi(t,x)\)（Eq. (17)），则（Eq. (18)）
   \[
   \partial_t\varphi_t(x)=-\sum_y r_t(y,x)\varphi_t(y)+f_t(x,p_t^\star)\varphi_t(x),\qquad
   \partial_t\hat\varphi_t(x)=\sum_y r_t(x,y)\hat\varphi_t(y)-f_t(x,p_t^\star)\hat\varphi_t(x),
   \]
   且 \(p_t^\star=\varphi_t\hat\varphi_t\)，\(\nu=\varphi_1\hat\varphi_1\)。最优率化为 **局部比值** \(u_t^\star(y,x)=r_t(y,x)\varphi_t(y)/\varphi_t(x)\)（Eq. (19)）。

4. **Proposition 3.5（生成元恒等式）**。令 \(Y_t=\log\varphi_t,\ \hat Y_t=\log\hat\varphi_t\)，\(Z_t(y,x)=Y_t(y)-Y_t(x)\)，则前向/反向生成元作用在 \((Y,\hat Y)\) 上有闭式（Eq. (21)–(24)）。核心是 \(A_t^uY(x)=f_t+\sum_y r_te^{Z_t}(Z_t-1)\)，\(A_t^u\hat Y(x)=\sum_y(r_t(x,y)e^{\hat Z_t}+\hat Z_tr_te^{Z_t})-f_t\)。

5. **Proposition 3.6（离散 gIPF 损失）**。基于 Dynkin 公式（Eq. (20)）把端点似然展开成生成元的积分（Eq. (25)），得到（Eq. (26)(27)）
   \[
   \mathcal L_{\mathrm{IPF}}^{Z}(\hat Z)=\mathbb E_{p^Z}\Big[\int_0^1\sum_y\Big(r_t(X_t,y)e^{\hat Z_t}+r_t(y,X_t)e^{Z_t}\big(-1+Z_t+\hat Z_t\big)\Big)\,dt\Big],
   \]
   反向对称。附录 A.4 说明它是 \(-\log p_0(x_0)\) 的上界，在最优处取等。

6. **TD 损失（Eq. (28)(29)）**——本文最关键的技术点，见 §3。

7. **Algorithm 1**：交替「前向 rollout → 更新 \(\phi\)」与「后向 rollout → 更新 \(\theta\)」，总损失
   \(L=\mathcal L_{\mathrm{IPF}}^{Z}(\hat Z^\phi)+\mathcal L_{\mathrm{IPF}}^{\hat Z}(Z^\theta)+\lambda_{\mathrm{TD}}\big(\mathcal L_{\mathrm{TD}}^{Z}(\hat Y^\phi)+\mathcal L_{\mathrm{TD}}^{\hat Z}(Y^\theta)\big)\)。

8. **附录 B（与随机最优控制的等价）**：从 KL 正则的有限时域控制问题（Eq. (77)）出发做动态规划，得到 HJB（Eq. (83)）
   \(-\partial_tV(t,x)=\sum_{y\ne x}r_t(y,x)\big(1-e^{V(t,x)-V(t,y)}\big)+f(t,x,p_t)\)，与 Theorem 3.1 一致。原文明说唯一概念差别是**终端条件**：SB 用硬边缘约束，SOC 用软终端成本 \(g(X_1)\)。

#### 5. 前提假设与适用边界

正向适用范围：

1. **图拓扑固定且完全已知**——原文 §4.5 Limitations 首句就是这条。参考生成元与学到的率都定义在给定边集上，rollout 与损失求值都要求可枚举 \(N(x)\)。
2. **参考生成元 \(r_t\) 必须给定且稀疏**：它同时承担"哪些动作合法"与"名义偏好"两个角色。整个方法的稀疏性红利全部来自 \(r_t\) 的稀疏支撑。
3. **两端边缘 \(\mu,\nu\) 是显式给定的概率向量**（不是样本集），定义在同一个节点集上。
4. **固定有限时域** \([0,1]\)，实践中固定步数 \(T\)（供应链 100、Chignolin 200）。到不了目标就是到不了——没有"轨迹长度自适应"这一说。
5. **成本可写成 \(f_t(x,p_t)\)**：state cost 或 mean-field cost。pairwise cost 要靠 Eq. (85) 的中间节点技巧转化。
6. **容量等硬约束不被强制**，只被拥塞成本隐式压低（原文 §4.1 明确 remark）。
7. 结论是**关于所选参考过程的相对最优**：换 \(r_t\) 就换问题。这与 O08 "cost 由图拓扑唯一决定"是两种哲学。

#### 6. 在 GFlowNet × OT 主线中的位置

##### 6.1 与 O08 的相邻性：同一象限的两个极点

两者共享的结构性设定（这是"相邻"的实质）：

- 给定一张有向图；
- 两端有边缘约束（O08：\(F(s_0\to u)=L(u)\) 与 \(F(x\to s_f)=R(x)\)；GSBoG：\(p_0=\mu,\ p_1=\nu\)）；
- 在**所有满足边缘约束的流/路径测度**中挑一个"最优"的；
- 输出都不是静态 coupling，而是**逐步可执行的局部策略**（O08：\(P_F(s'|s)\)；GSBoG：\(u_t(y,x)\)）；
- 两者的卖点句几乎逐字相同。O08 结论段："Unlike standard OT approaches that only learn a coupling, our framework learns a stochastic policy that transports samples through feasible local moves"；GSBoG 引言第二段："it does not describe how mass should move over time, nor does it yield an executable control policy on the graph"。**同一个叙事位被两篇论文同时占据。**
- 两者都有对偶势，且都用势差刻画最优支撑：O08 Theorem 3.3 的 \(\pi\)（Kantorovich potential，最优时 \(\pi_x^\star=d(x)\)），互补松弛 \(F^\star(s\to s')(\pi_{s'}^\star-(1+\pi_s^\star))=0\) 说明流只落在 tight 子图（=最短路子图）；GSBoG Theorem 3.1 的 \(V_t\)（HJB 值函数），\(u^\star=r\,e^{\Delta V}\) 说明率按势差指数重加权。

##### 6.2 差别就在「熵正则 vs 最小总流」

| 维度 | O08（最小总流） | GSBoG（熵正则 GSB） |
|---|---|---|
| 目标 | \(\min\sum_{e\in E^\circ}F(e)\)，线性、无正则（O08 Eq. (11)） | \(\min\mathbb E\!\int f_t+\mathrm{KL}(p^u\|p^r)\)（Eq. (13)） |
| 参考动力学 | 无（隐含"所有内部边代价 1"） | \(r_t\) 显式给定，是问题的一部分 |
| 解的性质 | LP 顶点解，支撑在最短路子图上，**可以是稀疏甚至退化的** | 指数族形式 \(u^\star=r\,e^{\Delta V}\)，在 \(r>0\) 的边上**处处严格正、光滑** |
| 时间结构 | 无时间轴，吸收态 \(s_f\)，\(\mathbb E[n_\tau]\) 自由 | 固定时域 \([0,1]\)/\(T\) 步，边缘打在两端 |
| cost | \(d_G(u,x)\)，只依赖起终点对 | \(f_t(x,p_t)\)，可依赖中间态、可依赖当前边缘（mean-field） |
| 是否给误差刻画 | 有：LP 强对偶 + 互补松弛 = primal-dual gap（Theorem 3.3） | 无（原文未给出） |
| 收敛证明 | LP 有精确解；神经训练无保证 | IPF/TD 交替无离散收敛证明 |
| 隐式图 | 支持（permutation \(20!\) 状态） | 不支持（需枚举 \(N(x)\) 且 \(\mu,\nu\) 是显式向量） |

一句话概括这个对比：**GSBoG 把"温度"调到有限值并引入参考过程，换来了光滑解、mean-field 成本和 IPF 训练；O08 把温度调到 0 并丢掉参考过程，换来了与 Kantorovich LP 的精确等价和对偶证书。**

##### 6.3 撞车判定

「熵正则 GFN–OT / Schrödinger bridge」这个课题，如果做成"给 O08 的目标加一个 \(\varepsilon\,\mathrm{KL}(P\|P_0)\)"，**撞车风险高**：GSBoG 已经在 ICML 2026 主会上把「图 + 参考动力学 + 熵正则 + 可执行策略 + 中间态成本」这一整套组合占了，而且实验规模（\(10^6\) 节点）远超 O08 的 hypergrid/permutation。

留下的缝，按可做性排序：

1. **不定时域**。GSBoG 必须固定 \(T\)；GFN 是吸收型、轨迹长度由策略自己决定。"不定时域的熵正则 GFN–SB"（把 KL 打在轨迹分布上而非固定时间网格上）没有被占，且正好落在非无环 GFlowNet 理论（T19/T36）的舒适区。
2. **隐式组合图**。GSBoG 的所有实验图都是显式、可枚举、来自物理网络（供应链、MSM）。Cayley 图、分子编辑图这类"只能局部展开"的空间它没碰，也不容易碰（\(\mu,\nu\) 得是 \(n\) 维向量）。
3. **误差界**。两边都没有。谁先把"平衡损失残差 → OT cost gap"的界写出来，谁就拿到这条线上唯一的理论增量。
4. **mean-field cost 在 GFN 里没有对应物**。\(f_t(x,p_t)\) 依赖当前边缘的设定，在 GFlowNet 文献里目前是空白（"拥塞感知 GFlowNet"）。这是一个真缝，但要小心：多智能体 GFlowNet（Brunswic et al. 2025b，O08 引用）可能已经沾边。

##### 6.4 前驱与对照

- **前驱**：Chow, Li, Mou, Zhou (2022) 图上动力学 SB（本文的直接对照 GrSB，在 9559 节点上内存耗尽）；Liu et al. (2022) DeepGSB（TD 损失的来源）；Liu et al. (2024) GSBM；De Bortoli et al. (2021) DSB / IPF；Essid & Solomon (2018) 图上二次正则 OT（本仓库 O02，也是 O08 Assumption 3.1 的来源）。
- **同期/对照**：Guo et al. (2026) 离散 adjoint SB sampler（原文 §3.1 明说"concurrently considered the same formulation"）；Ksenofontov & Korotin (2025) categorical SB；**C03 DDSBM**（原文 §3 与附录 D 反复强调 DDSBM 做的是"图空间上的生成建模"，与 GSBoG 的"固定拓扑上的路由"是不同问题）；Yang (2025) topological SB matching。
- **对主线的贡献**：GSBoG 给出了「图上熵正则路径测度选择」的完整技术栈（对偶 → Hopf–Cole → 生成元 → IPF+TD）。对本仓库而言，它既是竞品也是**现成的推导模板**：把 CTMC 换成离散时间 GFN 的转移核，Theorem 3.1 的拉格朗日推导几乎可以逐行搬。

#### 7. 可复用的 insight 与开放问题

1. **可直接推的命题草稿（熵正则 GFN–OT 的最优策略形式）**。给定参考前向策略 \(P_F^0\)，在 O08 的约束（\(F(s_0\to u)=L(u),\ F(x\to s_f)=R(x)\)，流守恒）下最小化 \(\mathbb E[|\tau|]+\varepsilon\,\mathrm{KL}(P\|P^0)\)，仿 GSBoG 附录 A.1 引入乘子 \(V(s)\) 并对策略求导，应得
   \(P_F^\star(s'|s)\propto P_F^0(s'|s)\exp\big((V(s)-V(s')-1)/\varepsilon\big)\)。
   \(\varepsilon\to0\) 时应退化到 O08 的 tight-subgraph 支撑（Theorem 3.3）。**这条推导是本次精读最直接的产出，值得先在纸上做完再决定要不要立项。**
2. **中间节点技巧（Eq. (85)）能扩展 O08 的 cost 表达力**。O08 的 cost 被钉死为单位边长的最短路 \(d_G\)。用 GSBoG 的做法——为需要额外代价的转移插入中间节点、把代价写成停留成本——可以在**不改动 Theorem 3.2** 的前提下表达任意正 pairwise cost（把权 \(w\) 的边替换成 \(w\) 段单位边）。这是一条能立刻写进 O08 后续工作的 remark，同时也解释了本仓库背景文档里"零权边会破坏最优流无环性"这个担忧的边界：只要权是正整数，展开成单位边即可。
3. **抵消检查应成为标准流程**。GSBoG §3.3 发现 \(f_t\) 在 IPF 求和中抵消。对应到 GFN：任何加进 TB/DB 损失的成本项，都要先验证它在 \(\log\frac{\prod P_F}{\prod P_B}\) 中不会消失。O08 的 \(\lambda R/P_F(s_f|\cdot)\) 项因为是加性附加项而幸免，但如果有人把成本写成对 \(P_F\) 的乘性重加权，就会踩坑。
4. **GrSB 内存耗尽是可复用的对照点**。想论证"GFN 的局部性带来可扩展性"，最有说服力的实验就是复现 GSBoG §4.4 的四轴 scaling 图，把 GSBoG 也放进去当基线。注意：GSBoG 未开源，复现需要自己实现。
5. **待验证的疑点**：Theorem 3.1 只说 "mild regularity assumptions"。在有限 \(\mathcal X\) 上，如果某些边的 \(r_t(y,x)=0\) 导致图不强连通、或 \(\nu\) 的支撑在 \(T\) 步内从 \(\mu\) 不可达，强对偶还成不成立？原文没讨论。O08 的 Assumption 3.1 第三条（任意 \(u\to x\) 存在有限长路径）恰恰是这条的显式版本——**GFN 侧在这一点上比 GSBoG 更严谨，可以在相关工作里点出来**。
6. **数字勘误备忘**：Table 3 与 Table 5 的 \(n=20\) 行熵不一致（0.17 / 0.22）。任何引用该数字的地方都应写成"原文两处不一致"。

## 7.3 C03 · DDSBM：图变换的离散扩散 Schrödinger 桥匹配（ICLR 2025）

> **一句话**：DDSBM 把 Iterative Markovian Fitting（IMF）从连续扩散搬到有限状态空间的 CTMC 上，证明了离散情形的单调收敛，并指出当参考过程取"节点与边独立跳变"时，对应的熵正则 OT cost **正比于图编辑距离（GED）**。它是「图上熵正则 OT」这条线上离 O08 最远的一篇——传输的对象是整张图而非图上的质量——但正因为 GED 恰好是"分子编辑图上的最短路距离"，它与 O08 定理之间存在一条精确的对应关系，这条关系既是灵感来源也是撞车风险来源。

| 字段 | 内容 |
|---|---|
| arXiv | [2410.01500](https://arxiv.org/abs/2410.01500)（v1 2024-10-02，v2 2025-02-28，arXiv comment 明写 "Accepted to ICLR 2025"） |
| 发表 | **ICLR 2025 Poster**（已核实：OpenReview `tQyh0gnfqW` 标注 "ICLR 2025 Poster"；iclr.cc virtual poster 28054；dblp venue=ICLR year=2025 type=Conference and Workshop Papers）。**主会 poster，不是 workshop** |
| 作者 | Jun Hyeong Kim\*, Seonghwan Kim\*, Seokhyun Moon\*, Hyeongwoo Kim\*, Jeheon Woo\*, Woo Youn Kim†（KAIST，前五位同等贡献） |
| 代码 | <https://github.com/junhkim1226/DDSBM>（官方实现，Python 3.9 / Torch 2.0.1 / cu118 / torch_geometric 2.3.1） |
| 本仓库 PDF | `papers/2410.01500.pdf`（47 页，18 MB，主要体积来自分子结构图） |
| 阅读优先级 | **P2** — 设定差得最远，但提供了「离散 IMF 收敛定理」和「GED = OT cost」两个可直接复用的零件 |

#### 2. 核心贡献（按原文编号）

1. **Definition 3.1（reciprocal projection）**：\(\Pi_{\mathcal R(\mathbb Q)}(\Lambda)(\cdot)=\iint_{(\cdot)}\Lambda(dx_0,dx_\tau)\,\mathbb Q(dx_t|x_0,x_\tau)\)。保留 coupling、把中间路径换成参考桥的混合（每座桥由 Doob 的 \(h\)-变换给出）。混合体一般**不再是马氏的**（原文引 Léonard et al. 2014：马氏测度集合非凸）。

2. **Definition 3.2（Markov projection）**：\(\Pi_{\mathcal M}(\Lambda)=\arg\min_M\{D_{\mathrm{KL}}(\Lambda\|M):M\in\mathcal M\}\)。保留所有时刻的边缘，**不保留 coupling**。投影后测度的生成元与 \(D_{\mathrm{KL}}(\Lambda\|\Pi_{\mathcal M}(\Lambda))\) 的显式形式在 Proposition B.2。

3. **Theorem 3.3（离散 IMF 的收敛）**——本文的理论核心。设 \(D_{\mathrm{KL}}(\Lambda^{(0)}_{0,\tau}\|\mathbb P^{\mathrm{SB}}_{0,\tau})<\infty\) 且 \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\)（\(\forall n\)），迭代 \(\Lambda^{(2n+1)}=\Pi_{\mathcal M}(\Lambda^{(2n)}),\ \Lambda^{(2n+2)}=\Pi_{\mathcal R(\mathbb Q)}(\Lambda^{(2n+1)})\)（Eq. (4)），则
   \[
   D_{\mathrm{KL}}(\Lambda^{(2n)}\|\mathbb P^{\mathrm{SB}})\ \ge\ D_{\mathrm{KL}}(\Lambda^{(2n+1)}\|\mathbb P^{\mathrm{SB}})\ \ge\ D_{\mathrm{KL}}(\Lambda^{(2n+2)}\|\mathbb P^{\mathrm{SB}}),
   \]
   等号成立当且仅当 \(\Lambda^{(2n)}=\Lambda^{(2n+1)}=\mathbb P^{\mathrm{SB}}\)；且 \(\Lambda^{(n)}\) 依分布收敛到 \(\mathbb P^{\mathrm{SB}}\)。

4. **训练损失**（Eq. (5)(6)）：前向的 Markov projection 用
   \(\mathcal L(\theta)=\int_0^\tau\mathbb E_{\Lambda^{(2n)}_{t,\tau}}\big[(A_t^{\mathbb Q_{\cdot|\tau}}-A_t^{M^\theta})(X_t,X_t)+\sum_{y\ne X_t}A_t^{\mathbb Q_{\cdot|\tau}}\log\frac{A_t^{\mathbb Q_{\cdot|\tau}}}{A_t^{M^\theta}}(X_t,y)\big]dt\)，
   反向用时间反演生成元 \(\tilde A_t^{\mathbb Q_{\cdot|0}}\) 对称给出。

5. **§4.3 图置换匹配**：参考过程的转移概率依赖节点编号，所以算似然/构造 reciprocal bridge 前必须先做图匹配。这被形式化为 **QAP（NP-hard）**，用 Cho et al. (2014) 的 max-pooling 连续松弛近似 + Hungarian 算法（Kuhn 1955，经 Pygmtools）离散化。

6. **附录 D.6（与 GED 的关系）**：引 Bougleux et al. (2015)，GED 计算本身等价于一个 QAP。给定 \(\alpha=\bar\alpha(\tau)/\bar\alpha(0)\)，节点/边替换代价为
   \(c^V(v_i,v'_a)=-\log\frac{(d^V-1)\alpha+1}{d^V}\)（相同）或 \(-\log\frac{-\alpha+1}{d^V}\)（不同），边同理，\(d^V,d^E\) 是状态基数。于是 QAP 目标 \(f(x)=\sum-\log P^E_{0:\tau}+\sum-\log P^V_{0:\tau}=p(\sigma G'|G)\)。
   **原文明确澄清 GED \(\ne\) NLL，而是成比例**：差值正比于最优匹配 \(\sigma^\star\) 下相同的节点数与边数（因为恒等操作也被计了一个较小但非零的代价）。结论句："solving the SB problem can be understood as finding the OT plan between graph distributions, where the transport cost is defined by the GED."

#### 6. 在 GFlowNet × OT 主线中的位置

**与 O08 的距离最远，但接口最精确。** 四个维度的差异：

| 维度 | DDSBM | O08 |
|---|---|---|
| 传输对象 | 整张图（一个分子 = 一个点） | 图上的质量（一个节点 = 一个点） |
| cost | \(-\log q_{\tau|0}\)，∝ GED（附录 D.6） | 图诱导最短路 \(d_G(u,x)\)（Eq. (7)） |
| 熵正则 | 是（KL 到参考 CTMC，温度由 \(\bar\alpha\) 隐式给定） | 否（纯线性最小总流） |
| 输出 | 生成模型：给 \(x_0\) 采 \(x_\tau\) | 路由策略 \(P_F(s'|s)\)：给状态选下一步 |
| 收敛保证 | Theorem 3.3（单调 + 依分布收敛，无速率） | Theorem 3.2（LP 等价，精确）；神经训练无保证 |

**精确的对应关系（本报告的主要产出之一）**：若把 O08 的状态图 \(G=(\mathcal S,E)\) 取为**分子编辑图**——节点是分子、边是单步编辑（改一个原子类型或一个键）——那么 O08 定义的最短路距离 \(d_G(u,x)\)（Eq. (7)）**就是 \(u\) 与 \(x\) 之间的图编辑距离**。于是：

> **O08 的 Theorem 3.2 应用在分子编辑图上 = 一个无熵正则、以 GED 为 cost 的 DDSBM。**

反过来说，DDSBM 是它的 \(\varepsilon>0\) 版本。这条对应关系有两面：

- **正面（灵感）**：它给"GFN 做分子优化 OT"提供了一个现成的问题定义、现成的数据集（ZINC250K / Polymer）和现成的指标（NLL / NSPDK / FCD / MAD）。而且 GFN 有一个 DDSBM 没有的优势——**GFN 沿编辑路径行走，路径本身给出节点对应，根本不需要解 QAP 图匹配**。这是一条可以直接写进 motivation 的话。
- **负面（撞车）**：任何"用 GFN 在分子编辑图上做熵正则 OT"的工作，都会被要求解释与 DDSBM 的关系。且 DDSBM 已在 ICLR 2025 主会 poster、代码开源、实验完整。

**撞车评级**：
- 对「熵正则 GFN–OT / SB」：**中等偏高**——占了"离散 SB + 图 + GED cost + 分子应用"这一格，但没占"固定拓扑上的路由"（那是 C02 GSBoG 的格）。
- 对「条件 GFN 学一族图 OT」：**低**——DDSBM 完全不条件化，一个模型对应一对固定分布。
- 对「Balance 残差 → OT 误差界」：**低**，但它的 Theorem 3.3 是**必须引用的先例**——审稿人会问"IMF 都有收敛定理了，你的 TB 残差界新在哪"。答案应当是：Theorem 3.3 给的是"迭代到 SB"的单调性，不是"残差到 cost gap"的定量界；两者回答的问题不同。

**前驱**：Peluchetti (2023a) / Shi et al. (2024) 的 IMF；Léonard (2013) 的 SB–EOT 等价；De Bortoli et al. (2021) DSB；Vignac et al. (2022) DiGress（网络与噪声调度）；Bougleux et al. (2015) GED–QAP 等价。
**对照**：C02 GSBoG（其 §3 与附录 D 反复强调"DDSBM 做的是图空间上的生成建模，我们做的是固定拓扑上的路由"，两篇互相划清界限）；Ksenofontov & Korotin (2025) categorical SB。

#### 7. 可复用的 insight 与开放问题

1. **写一条 remark：GED = 分子编辑图上的 \(d_G\)**。这是 §6 那条对应关系的形式化，一段话可以写完，能同时服务于"相关工作"与"应用动机"两处。注意边界：O08 要求单位边长，所以只有当所有编辑操作代价相同时等式才成立；带权 GED 需要用 C02 的中间节点技巧展开。
2. **GFN 免图匹配是一个真实的比较优势**。DDSBM 每次构造 reciprocal bridge 都要解一次近似 QAP（§4.3）。GFN 沿编辑序列走，源与目标的对应由路径隐式给出。这条可以做成一个实证对比：同样的 ZINC 任务，报告 DDSBM 花在图匹配上的时间占比。
3. **IMF 的"迭代改善 coupling"能否搬到 GFN 的 \(P_B\)？** DDSBM 的核心机制是交替更新 coupling 与马氏过程。GFN 里的对应物是交替更新 \(P_B\)（决定 credit assignment）与 \(P_F\)。O08 结论段自己就提到 backward policy optimization（引 Jang et al. 2024、Gritsaev et al. 2025）。**把 IMF 的 reciprocal/Markov 双投影结构映射到 \((P_F,P_B)\) 交替优化上，是一条有明确出处的技术路线。**
4. **评测协议直接可用**：NLL（joint，衡量传输代价）+ NSPDK/FCD（marginal，衡量分布匹配）+ MAD（衡量副作用）这三层结构，恰好对应 OT 的"cost / 边缘 / 附加约束"。GFN-OT 的实验表可以照抄这个骨架。
5. **Theorem 3.3 缺速率，是可攻击点**。它只给单调下降与依分布收敛，且假设 \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\)（对神经近似未验证）。若能在 GFN 侧给出**带速率**或**带残差**的结论（哪怕更弱的设定），那就是清晰的理论增量。
6. **未回答的问题**：DDSBM 从没测过它学到的 coupling 与真 EOT plan 的距离——因为真解算不出来。GFN 在小规模上可以用 POT/LP 算真解（O08 Table 1/2 已经这么做了）。**"小规模有真解、大规模有代理指标"的双层评测**是 GFN 侧能拿出而 DDSBM 拿不出的东西。

## 7.4 竞品矩阵

#### 0. 五篇的身份牌

| 编号 | 简称 | 标题 | 发表状态（已核实） | 代码 |
|---|---|---|---|---|
| **O08** | GFN-OT | Your GFlowNet Secretly Learns an Optimal Transport Plan | **ICML 2026 SPIGM Workshop**（arXiv comment 明写；PDF 首页 "Preprint. June 5, 2026"） | 未见公开 |
| **O07** | GFN-SP | Learning Shortest Paths with Generative Flow Networks | **ICML 2026 SPIGM Workshop**（本仓库 `data/papers.yaml`；arXiv 2603.01786 v1 2026-03-02） | `github.com/GreatDrake/gfn-pathfinding`（原文 §1 给出） |
| **C01** | ULOT | Unsupervised Learning for OT plan prediction between unbalanced graphs | **NeurIPS 2025 主会**（proceedings 页 "Main Conference Track"，DOI 10.52202/085713-3146；dblp NeurIPS 2025） | `github.com/smazelet/ULOT` |
| **C02** | GSBoG | Generalized Schrödinger Bridge on Graphs | **ICML 2026 主会**（PDF 页脚 PMLR 306 camera-ready；作者主页 2026-04 接收公告；GaTech Spotlight 2026）。**dblp 截至 2026-09 仅 CoRR 条目** | 未公开 |
| **C03** | DDSBM | Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation | **ICLR 2025 Poster**（OpenReview `tQyh0gnfqW`；dblp ICLR 2025；arXiv comment "Accepted to ICLR 2025"） | `github.com/junhkim1226/DDSBM` |

**第一个要认清的事实**：本方两篇是 **workshop**，三篇竞品分别是 NeurIPS 主会、ICML 主会、ICLR 主会。在"叙事占位"这件事上，本方目前处于劣势。

#### 1. 主矩阵

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

#### 2. 三条不能混淆的分界线

**分界线 A：图是"被传输的对象"还是"传输发生的场所"。**
ULOT 与 DDSBM 在前者（图是对象），O08 / O07 / GSBoG 在后者（图是场所）。跨线比较数字毫无意义——ULOT 的 "100× 加速" 与 O08 的 "\(\mathbb E|\tau|=3.990\) vs OT\(^\star=3.997\)" 不在同一个问题上。

**分界线 B：显式枚举 vs 隐式展开。**
GSBoG 虽然跑到 \(10^6\) 节点，但它要求 \(\mu,\nu\) 是显式 \(n\) 维向量、要求能枚举 \(N(x)\)、要求拓扑完全已知（§4.5 Limitations 首句）。O08 的 permutation 环境里 \(20!\) 个状态**没有任何显式表示**，只能通过"当前排列 + 合法相邻对换"局部展开。这是 GFN 侧唯一无法被复制的结构性优势。

**分界线 C：温度。**
O08/O07 在 \(\varepsilon=0\)（纯 LP，顶点解，有对偶证书）；GSBoG/DDSBM 在 \(\varepsilon>0\)（指数族解，光滑，有 IPF/IMF 训练机制）。ULOT 既不在这条线上（它的正则是二次的、罚在边缘上）。**"给 O08 加 KL 项"这个动作，就是把自己从 \(\varepsilon=0\) 挪到 \(\varepsilon>0\)——那里已经站着 GSBoG（ICML 2026 主会）和 DDSBM（ICLR 2025）。**

#### 3. 各竞品对本方的具体威胁

| 竞品 | 威胁的是什么 | 强度 | 反制点 |
|---|---|---|---|
| **C02 GSBoG** | O08 的**整个定位叙事**。两篇的卖点句几乎逐字相同（"不是静态 coupling，而是可执行策略"）。GSBoG 在 ICML 2026 主会、\(10^6\) 节点、三个真实应用域 | **最高** | (i) 隐式组合图；(ii) 不定时域（GSBoG 必须固定 \(T\)）；(iii) O08 有 LP 对偶证书，GSBoG 无任何误差界；(iv) GSBoG 未开源 |
| **C01 ULOT** | 「条件 GFN 学一族图 OT」这个**具体课题** | 高 | (i) ULOT 硬上限 \(n\le10^4\)；(ii) 输出 plan 不输出 policy；(iii) 全文零定理；(iv) cost 是二次 FUGW，与线性 Kantorovich 不是同一问题 |
| **C03 DDSBM** | 「熵正则 GFN-OT 用在分子编辑上」这个**应用位** | 中 | (i) DDSBM 需解 QAP 图匹配，GFN 沿路径走天然不需要；(ii) DDSBM 不条件化；(iii) 它的 Thm 3.3 是"迭代收敛"不是"残差→cost gap" |
| **O07（本方）** | 不是威胁，是 O08 的前驱。O08 Thm 3.3 明说"This recovers the corresponding claim of Morozov et al. (2026)" | — | 两篇要合并叙事，不要各自为战 |

#### 4. 四个候选课题的撞车风险评级

评级依据：**设定重合度**（竞品是否已在同一设定下工作）× **卖点重合度**（核心 claim 是否已被说过）× **发表场次**（对方在主会还是 workshop）。

##### 课题 1 · Balance 残差 → OT 误差界
> 目标：证明形如 \(\big|\mathbb E[|\tau|]-\mathrm{OT}^\star\big|\le C(\varepsilon_{\mathrm{TB}},\lambda,|\mathcal S|)\) 的定量界，把 TB 损失残差翻译成传输代价 gap 与边缘误差。

**撞车风险：低（当前四个课题中最值得做）**

依据：
- 五篇竞品**没有一篇给出误差界**。ULOT 零定理；GSBoG 无界且 Thm 3.1 的正则性条件未具体化；DDSBM 的 Thm 3.3 只给单调性与依分布收敛、无速率、且假设 \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\) 对神经近似未验证。
- O08 已经把接口铺好：Thm 3.3 给出 LP 对偶 \(\max_\pi\sum_x R(x)\pi_x+\sum_u L(u)(-\pi_u)\) s.t. \(\pi_{s'}-\pi_s\le1\)（附录 A.4 Eq. (24)(25)），以及互补松弛 \(F^\star(s\to s')(\pi^\star_{s'}-(1+\pi^\star_s))=0\)。**primal-dual gap 是现成的误差证书载体**。
- 本仓库背景文档 §5 已把 primal-dual 列为最值得做的方向，与本次撞车扫描独立得出同一结论。
- 需要预防的质疑：审稿人会拿 DDSBM Thm 3.3 来问"IMF 都有收敛定理了"。回答必须准备好：那是"迭代序列收敛到 SB"，本课题是"单个模型的残差 → cost gap"，前者不蕴含后者，也不能给出可计算的证书。

##### 课题 2 · 条件 GFN 学一族图 OT
> 目标：训练 \(P_F(a|s,L,R,c)\)，对未见的 \((L,R)\)、cost 泛化。

**撞车风险：高**

依据：
- ULOT（NeurIPS 2025 主会）已经把摊销 + 条件化 + 图 + 无监督 + unbalanced + warm-start 六件事一次做完，并在 fMRI 上给出 14400 图对的泛化实验。
- O05 Universal Neural OT（ICML 2025）在通用神经 OT 侧也占了摊销位。
- **仍可做的形态（唯一）**：把条件变量取为**隐式组合图上无法显式化的东西**——例如条件于 cost 权重 \(\lambda\)（O08 Table 2 显示 \(\lambda=10^{-1}\) vs \(10^{-2}\) 存在强 trade-off，一次训练给出整条正则路径是干净的增量），或条件于 Cayley 图的生成元集合。**在显式图上做条件化 = 直接撞 ULOT。**
- 若坚持做，必须在实验里把 ULOT 当基线跑一遍（在它能跑的规模上），否则审稿人会问"为什么不用 ULOT"。

##### 课题 3 · 熵正则 GFN–OT / Schrödinger 桥
> 目标：\(\min_P\mathbb E_P[c(\tau)]+\varepsilon\mathrm{KL}(P\|P_0)\)，固定两端边缘。

**撞车风险：高**

依据：
- GSBoG（ICML 2026 主会）已完整占据「图 + 参考动力学 + 熵正则 + 可执行局部策略 + 中间态成本」。它甚至给了完整技术栈：对偶（Thm 3.1）→ Hopf–Cole（Prop 3.4）→ 生成元恒等式（Prop 3.5）→ gIPF（Prop 3.6）→ TD 修正（Eq. (28)(29)）。
- DDSBM（ICLR 2025）占据离散 SB + 图 + GED cost。
- 加上 Ksenofontov & Korotin (2025) categorical SB、Guo et al. (2026) 离散 adjoint SB、Yang (2025) topological SBM——**这个生态已经拥挤**。
- **仍可做的形态**：(i) **不定时域**——GSBoG/DDSBM 都必须固定 \(T\)，GFN 是吸收型、轨迹长度自适应，把 KL 打在轨迹分布上而非固定时间网格上，这条缝是真的；(ii) **隐式组合图上的熵正则 SB**；(iii) **mean-field / 拥塞感知 GFlowNet**（GSBoG 的 \(f_t(x,p_t)\) 在 GFN 文献里无对应物，但要先排查多智能体 GFlowNet 是否已沾边）。
- 建议：**不要把它当独立课题，把它当课题 1 的 \(\varepsilon>0\) 推广**。误差界在 \(\varepsilon>0\) 下往往更容易证（强凸性），这样两条线合并后既有理论增量又不与 GSBoG 正面撞。

##### 课题 4 · GFN proposal + 经典 OT 修正
> 目标：GFN 输出粗解 → 喂给 network simplex / Sinkhorn 做精修。

**撞车风险：中（建议降级为对照基线，不作主线）**

依据：
- ULOT §3.2 + Figure 8(right) 已经把"神经预测 → 经典 solver warm start"这个 idea 演示过一遍，并且明确写进 abstract。novelty 已被占。
- 但它作为**评测协议**仍然有价值：把 GFN 学到的策略采样成 coupling，喂给精确 solver，报告迭代数下降比例——这是一个几乎零成本、能直接放进消融表的实验，也是回应"GFN 到底有没有用"这类质疑的最简答案。
- 独立发表价值低，作为课题 1 的一节则很合适。

#### 4.5 可以立刻开跑的四个 red-team 实验

这些实验的目的不是发论文，是**在投稿前先把审稿人会问的问题自己答一遍**。

1. **ULOT 反打实验**：把 O08 的 permutation 环境压缩到 \(n_1,n_2\le10^4\) 个代表状态，构造显式 cost matrix，直接跑 ULOT / POT。如果 ULOT 在这个规模上打得过 GFN，那"隐式图"这条护城河必须收窄到"连 \(10^4\) 个代表状态都取不出来"的场景。
2. **warm-start 协议**：把 GFN 策略采样成 coupling → 喂给 network simplex，报告迭代数下降比例。对标 ULOT Figure 8(right)。零成本，直接进消融表。
3. **primal-dual gap 监控**：训练过程中同时跟踪 O08 Thm 3.3 的对偶目标与互补松弛违反量 \(\sum_{s\to s'}F(s\to s')\big|\pi_{s'}-(1+\pi_s)\big|\)。这既是课题 1 的原型，也是唯一能证明"GFN 输出可信"的在线证书。
4. **四轴 scaling 对照**：复现 GSBoG §4.4 的（节点数 / 边密度 / 时间步 / rollout 预算）四轴内存与耗时图，把 GFN 放进去。注意 GSBoG **未开源**，需自行实现——这也意味着如果本方开源，在可复现性上反而占优。

#### 5. 一句话结论

**对 O08 威胁最大的是 C02 GSBoG**：它与 O08 在同一象限（图 + 两端边缘 + 可执行局部策略）、用几乎相同的卖点句、发在 ICML 2026 主会、实验规模高出四个数量级。但它有三个 O08 没有的短板——需要完全已知的显式拓扑、必须固定有限时域、**没有任何误差界**。

因此推荐的主线是：**把「Balance 残差 → OT 误差界」（低撞车）与 primal-dual/对偶势（O08 Thm 3.3 现成）合并为一条主线，实验场景钉死在隐式组合图（permutation / 分子编辑 / Rubik's Cube），并把 ULOT 与 GSBoG 作为"在它们能跑的规模上"的对照基线。** 熵正则版本作为该主线的 \(\varepsilon>0\) 推广，而不是独立课题。

---

**编者注**（歧义处的自行决定）：
1. O07 的发表状态取自本仓库 `data/papers.yaml`（ICML 2026 SPIGM Workshop），本次未对其单独走 arXiv/dblp 核实——它是本方论文，不在"三篇竞品核实"的任务范围内。O08 的 workshop 状态则由 arXiv API 的 comment 字段直接确认。
2. C02 标为"ICML 2026 主会"而非"预印本"，依据是三条独立证据（PMLR 306 camera-ready 页脚、作者主页接收公告、GaTech Spotlight），但 dblp 截至 2026-09 仅有 CoRR 条目，此差异已在 C02 报告与本表中双重标注。
3. 表中"最大实验规模"的可比性有限：五篇的规模指标含义不同（状态数 / 节点数 / 分子对数），不宜横向排序，仅供判断"各自在什么量级上工作"。
4. §4 的风险评级（低/中/高）为本报告判断，不是任何原文的表述。评级方法（设定重合 × 卖点重合 × 发表场次）已在该节开头写明。
5. 建议在 NeurIPS 2026 放榜（2026-09-24）后复查 O07/O08 是否升级为主会，并复查 C02 在 PMLR 306 卷的正式页码。


# 第 8 章 2025–2026 趋势

两侧各做一次 arXiv 系统扫描（GFlowNet 侧 106 篇、OT 侧 93 篇，2025-01 → 2026-09），检索口径与逐篇卡片见 `reports/TRENDS_GFN_2026.md`、`reports/TRENDS_OT_2026.md`，候选收录清单见 `data/candidates_gfn.csv`（38 条）与 `data/candidates_ot.csv`（49 条）。本章只收趋势判断与对主线的含义。发表状态只依据 arXiv `comment` 字段；为空者一律视为预印本。

## 8.1 GFlowNet 侧

#### 3. 趋势判断

每条给出支撑论文的 arXiv 号；只有一篇支撑的不算趋势。

1. **非无环 GFlowNet 从「理论修补」变成「生成工具」，但停在 Workshop。** T36（2502.07735，ICML 2025 主会）之后一年内，同一 HSE 团队把非无环理论推向最短路（2603.01786）、最优传输（2606.06272）、MCMC 终止（2606.16073），Brunswic 组推向连续/遍历设定（2505.03561，ICML 2025 主会）与多智能体（2509.20408）。三篇应用出口全部落在 SPIGM Workshop，主会只有两篇理论。判断：这条线的理论已被主会接受，应用叙事还没有。

2. **误差证书成为新的竞争点。** Stable GFlowNets（2605.01729）证明低 TV 不排除无界损失并给出 loss→TV 反向界；Evaluation Balance（2603.01047，ICLR 2026）把 flow balance 当策略评估器；Secrets（2505.02035）谈样本复杂度与隐式正则。「低 loss 不等于低分布误差」这个老问题正被系统处理。对本仓库最重要的含义：把残差界从 TV 推到 OT cost gap 的空位还在，但 GFN 侧的工具已经成熟。

3. **训练目标进入「族」的时代。** \(f\)-TB（2605.15417，ICML 2026）、\(\alpha\)-GFN（2602.01749）、Divergent TB（2602.17827）、Hybrid-Balance（2510.04792，NeurIPS 2025）、RapTB（2603.00454）、Evaluation Balance（2603.01047）：目标不再是 FM/DB/TB 三选一，而是同一最小点下可调的梯度几何。O08 的最小流目标可以叠加在其中任一族之上，这是做实验时该扫的轴。

4. **策略梯度回流 GFlowNet。** PPO for amortized discrete sampling（2606.15793）、信息几何自然梯度（2608.03967）、RTB ≡ Trust-PCL（2509.01632）、PowerFlow（2603.18363，ICML 2026）、GFlowRL（2607.13394）：GFN 与 KL 正则 RL 的等价被反复用来借 RL 的优化器。含义：熵正则 GFN–OT 的 RL 对应物大概率已存在于 KL 正则 RL 文献里。

5. **\(Z\) 的两种命运。** LLM 场景里 \(Z\) 被消灭（Stable-GFN 成对比较，2605.00553，ICML 2026 Spotlight；GFlowRL 指出 prompt 条件 \(Z\) 是不稳定源，2607.13394）或被重用（\(Z\) 作难度调度器，2602.12642）。而 O08 的 LP 设定里 \(Z\) 不出现——这条主线放弃了 GFlowNet「只需未归一化奖励」的招牌能力，见 O08 报告 §7.6。

6. **路径空间控制与 Schrödinger 语言正在统一 GFN 与采样器。** Sampling Decisions（2503.14549）把 GFlowNet 流函数、Doob \(h\)-变换、单侧 Schrödinger 传输写成同一对象；Berner 等（2501.06148，TMLR）给出离散↔连续时间的渐近等价。熵正则 GFN–OT 的理论地基已被别人打好一半，撞车风险因此高。

7. **主会占比与类型。** 可据 comment 判定的主会/期刊接收约 30 篇：ICML 2025 ×4、ICML 2026 ×5、ICLR 2025/2026 ×3、NeurIPS 2025 ×2、AAAI 2026 ×1、AISTATS 2025 ×1、TMLR ×2、EMNLP 2025、KDD 2026、SIGMOD 2027、ACM MM 2025。应用类多于理论类；与 OT 直接相关的主会论文为零（O07/O08 均为 Workshop）。

8. **工具层出现第二套基础设施。** gfnx（JAX，2511.16592）与 torchgfn 并列，GFlowState（2604.21830）补可视化。复现实验有了两条独立路径。

#### 4. 对 GFlowNet × OT 方向的含义

对照 `reports/COMPETITOR_MATRIX.md` 的四个候选课题：

| 课题 | 窗口状态 | 依据 |
|---|---|---|
| Balance 残差 → OT 误差界（+ 对偶势证书） | **打开，且工具已备齐** | GFN 侧有 loss→TV 界（2605.01729）与 balance-as-evaluator（2603.01047）可直接引用；OT 侧的 LP 对偶与互补松弛在 O08 Thm 3.3 现成；尚无人把两端接起来 |
| 条件 GFN 学一族图上 OT | 关闭中 | ULOT（本仓库 C01，NeurIPS 2025）与 UNOT（O05，ICML 2025）已占据摊销轴；GFN 侧没有出现新的条件化工具 |
| 熵正则 GFN–OT / Schrödinger 桥 | 关闭中 | Sampling Decisions（2503.14549）已把 GFN 流函数与单侧 Schrödinger 传输统一；GSBoG（C02，ICML 2026）与 DDSBM（C03，ICLR 2025）占据图上 SB；α-DSBM（O04）给出同构的投影算法 |
| GFN proposal + 经典 OT 修正 | 降为基线 | Unrealized Expectations（2502.03669，TMLR）警示：显式图上的经典求解器是残酷基线；GFN 只能在隐式图上立足，而隐式图上没有经典 OT 求解器可「修正」 |

两个新打开的小窗口：(a) **最小流目标 + 策略梯度/PPO 训练器**（2606.15793 是同一团队的下一步，先做就先占）；(b) **对称性与内部流自由度**（2506.02685 的对称修正与 T02 的 \(P_B\) 自由度是同一现象的两面，尚无人从 OT 角度写）。

## 8.2 最优传输侧

#### 3. 趋势判断

1. **图上 OT 的理论重心在「势」而不在「计划」。** 对偶预测加速最小费用流（2601.20203，AAAI 2026）、Sinkhorn 势的统计率（2608.29152）、QOT 对偶的 PL 不等式（2605.27175，SIAM J. Optim.）、Brenier 势估计（2604.22366）——四篇的共同对象都是对偶变量。含义：GFlowNet 学到的状态流 \(F(s)\) 在 O08 里就是对偶势的原始变量对应物，接上这套「势理论」比接上「计划预测」更顺。

2. **离散 SB 完成了从方法到基准的闭环。** DDSBM（C03，ICLR 2025）、GSBoG（C02，ICML 2026）之后出现了有解析解的离散 SB 基准（2509.23348）和收敛率结果（2607.19176）。一个方向有了基准和收敛率，就意味着新进入者必须在这套口径下报数——熵正则 GFN–OT 已没有「先定义问题」的红利。

3. **摊销 OT 拥挤度继续上升。** UNOT（O05）、ULOT（C01）之后，切片势摊销（2604.15114）、min-sliced 可迁移计划（2511.19741）、HyperTransport（2605.08254）在 2025-11 → 2026-05 连续出现。条件 GFN–OT 若要立足，只剩「隐式图、无法实例化代价矩阵」这一条护城河。

4. **二次正则是唯一没被占满的正则化方向。** 熵正则有 SB 一整套生态，二次正则（O02 的选择）只有 González-Sanz–Nutz 组在做优化理论（2605.27175 / 2605.27883），且都在连续/半离散设定。图上二次正则流 + GFlowNet 参数化，目前无人做。

5. **「非唯一时选哪一个」有独立的数学文献。** 距离代价下 OT 计划不唯一，熵选择原理（2512.05282、2502.16370）研究小正则极限选出哪个。O08 用最小总流作选择原则，与这条文献平行却互不引用；把两者接起来是一篇短文的体量。

6. **连续图（metric graph）上的神经 OT 刚出现。** 2606.16273 是第一篇，用嵌入 + 半对偶 + 投影；它不给逐边策略，与 O08 的离散、可执行策略互补而非替代。

7. **主会占比。** 93 篇里可据 comment 判定主会/期刊的 12 篇：NeurIPS 2025 ×2（含 Spotlight、Oral）、AAAI 2026 ×2、ICML 2026 ×1、ICLR 2026 ×1、ICDM 2026、WACV 2026、SIAM J. Optim.、L-CSS、VLDBJ、Globecom 2026。图上 OT 的理论工作主要以预印本形态存在。

#### 4. 对 GFlowNet × OT 方向的含义

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

#### 5. 做 GFN–OT 实验必须对比的强基线

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

## 8.3 本报告判断：两侧合看

- GFN 侧把「残差当度量」（Stable GFlowNets、Evaluation Balance）与 OT 侧把「误差理论放在对偶势上」（对偶预测加速最小费用流、Sinkhorn 势统计率、QOT 的 PL 不等式）在 2026 年同时成熟，却没有任何一篇把两端接起来。这是课题①的窗口。
- 非无环 GFlowNet 的三个应用出口（最短路、OT、MCMC 终止）全部停在 Workshop；同象限的 GSBoG 在主会。叙事占位的差距比技术差距大。
- 摊销 OT 与离散 SB 两个方向都已进入「有基准、有收敛率」阶段，新进入者没有定义问题的红利。

## 8.4 六篇 2026 增补的深度解读（N01–N06）

趋势扫描中 relevance=5 的六篇已补下载、翻译并配深度解读。下面收录各篇解读的核心贡献、前提假设、主线位置与 insight 四节（记号与推导细节见 `reports/N0*.md`）。

### 8.4.1 N01 · Minimum-Cost Network Flow with Dual Predictions（AAAI 2026）

> **一句话**：第一个「带对偶预测」的最小费用流算法。把经典 ε-relaxation 用一个预测的对偶解 \(\hat p\) 热启动，运行时间从最坏情形的 \(O(n^3\log(nC))\) 变成 \(O(\min\{n^3\log\|\hat p-p^\star\|_\infty,\ n^3\log(nC)\})\)（Theorem 2）：预测越准越快，预测全错也不比经典差。对本仓库主线的意义在于：O08 固定双边缘问题的对偶势（Appendix A.4 Eq. (24)，目标 \(\sum_xR(x)\pi_x-\sum_uL(u)\pi_u\)；按本篇同向边约定对应 \(p=-\pi\) 加常数）可以充当这里的 \(\hat p\)，「GFlowNet 学势 → 喂给经典求解器精化」这条路第一次有了带运行时间保证的接口。注意 O08 Thm. 3.3 本身讨论的是去掉首步约束的单源特例，不能直接套用。

| 字段 | 内容 |
|---|---|
| arXiv | [2601.20203](https://arxiv.org/abs/2601.20203)（v1，2026-01-28） |
| 发表 | **AAAI 2026**（arXiv comment "accepted by AAAI 2026"；PDF 版权页为 AAAI 2026） |
| 作者 | Zhiyang Chen（清华）、Hailong Yao（北京科技大学，通讯）、Xia Yin（清华） |
| 代码 | 原文未给出链接 |
| 本仓库 PDF | `papers/2601.20203.pdf` · 中译 `papers_zh/2601.20203.zh.pdf`（QA 7 issues） |
| 阅读优先级 | P1：课题④「GFN proposal + 经典 OT 修正」唯一站得住的具体形态 |

##### 2. 核心贡献（按原文编号）

**理论。** 给 \(\hat p\) 做预处理（平移使 \(\min\hat p=0\)，再裁剪到 \([0,(n-1)C]\)，Algorithm 1 第 1–2 行；Lemma 1 保证存在落在该区间内的最优对偶，所以裁剪不放大误差），然后热启动 ε-relaxation：

- **Theorem 1**：Algorithm 1 复杂度 \(O(\min\{n^3+n^2\varepsilon^{-1}\|\hat p-p^\star\|_\infty,\ n^3\varepsilon^{-1}C\})\)；0/1 流（\(b=0,c=1\)，如二分匹配）改进为 \(O(\min\{mn+m\varepsilon^{-1}\|\hat p-p^\star\|_\infty,\ mn\varepsilon^{-1}C\})\)。
- **Theorem 2**：加 cost-scaling（Algorithm 2）后 \(O(\min\{n^3\log\|\hat p-p^\star\|_\infty,\ n^3\log(nC)\})\)；0/1 流 \(O(\min\{mn\log\|\hat p-p^\star\|_\infty,\ mn\log(nC)\})\)。第一项是 **consistency**（预测准则快），第二项是 **robustness**（预测全错退回经典界）。
- **Theorem 3 / Theorem 4**：PAC 样本复杂度。固定拓扑、边代价随机时学一个固定 \(\hat p\) 需 \(\tilde O(n/\varepsilon^2)\) 样本（Theorem 3）；学一个特征到对偶的神经预测器需 \(k=\Omega\big(\tfrac{H^2}{\varepsilon^2}(n\,d_{NN}\log(nC)+\log\tfrac1\delta)\big)\)（Theorem 4，\(d_{NN}\) 为网络伪维度，Lemma 4）。

**实验。** 交通网络（固定拓扑、随机代价，学固定 \(\hat p\)）加速 6.2–21.4×，平均 12.74×；PCB 逃逸布线（UNet 式 CNN 预测网格对偶）加速 1.1–2.3×，平均 1.64×（Abstract、Contributions、Experiments）。

##### 5. 前提假设与适用边界

适用于：整数容量与供给（Assumption 1）以及**整数或可整数化的费用**（精确最优性判据 \(\varepsilon<1/n\) 与取整缩放都依赖此，原文 Assumption 1 未明示、后文按整数费用处理——这是源文的假设缺口，推广到实值 OT 费用需另行处理精度）；显式图（需要枚举节点做 ε-relaxation 迭代）；有一族相似实例可学。界是关于 ε-relaxation 的，不迁移到 network simplex。

##### 6. 在 GFlowNet × OT 主线中的位置

- 与 O08 的接口：固定双边缘时用 O08 Appendix A.4 的扩展对偶（目标 \(\sum_xR(x)\pi_x-\sum_uL(u)\pi_u\)，内部边约束 \(\pi_{s'}-\pi_s\le1\)）；Thm. 3.3 的「\(\pi^\star_x=d(x)\)」只对去掉首步约束的单源特例成立。一个近似对偶可行势 \(\hat p\)（符号按本篇约定取 \(-\pi\)）喂给 ε-relaxation，Theorem 2 给出带运行时间保证的精确解。**待验证的一环**：GFlowNet 学到的状态流对数与 Kantorovich 势之间没有已证明的接近关系（状态流反映奖励、路径选择与访问次数），把它当 \(\hat p\) 只是一个假设，须与常数势、解析势和直接训练的势预测器比较。
- 竞争面：在**显式图**上，本篇的「学习对偶 + ε-relaxation」比 network simplex 还快（Table 1 平均 12×），并自带最优性证书。任何在显式图上做 GFN–OT 的实验都必须把它列为基线；GFlowNet 只剩隐式图（无法枚举节点，ε-relaxation 根本跑不起来）。
- 与趋势报告的关系：`TRENDS_OT_2026.md` §2.1 把它列为「可借用 + 潜在竞争」，本文确认这一判断。

##### 7. 可复用的 insight 与开放问题

1. **课题④的正确形态**：不是模糊的「GFN proposal + 修正」，而是「GFN 状态流 → 对偶预测 → ε-relaxation」，误差到时间的映射由 Theorem 2 给出；实验只需报 \(\|\hat p-p^\star\|_\infty\) 与加速比。
2. **误差度量的启示**：运行时间只依赖 \(\|\hat p-p^\star\|_\infty\)。若要证明「balance 残差 → 对偶势误差」，目标应是无穷范数界而不是 \(\ell_2\)。
3. **隐式图上的对偶预测无用武之地**：ε-relaxation 每轮扫所有节点；\(20!\) 个状态下不可能。本篇因此反过来给 GFN–OT 划定了边界：显式图别做，隐式图才是空位。
4. 开放：能否把 Theorem 2 的分析迁到「用近似对偶做 warm-start 的 Sinkhorn / 熵正则求解器」？原文只做零温 LP。
5. 开放：本篇的 PAC 界要求实例分布固定；GFlowNet 的隐式图「一族实例」如何定义，仍无答案（与课题②同一障碍）。

### 8.4.2 N02 · Stop the Sampler!（ICML 2026 SPIGM Workshop）

> **一句话**：把 MCMC 装进连续状态空间的非无环 GFlowNet 框架：每一步用一个学到的分类器 \(d_F(s)\) 决定「停不停」，用 detailed balance 把最优分类器和目标密度钉在一起（Theorem 3.6），并证明总流最小当且仅当期望轨迹长度取到一个由 Markov 链定义的下界（Corollary 3.7）。它与 O07/O08 出自同一 HSE 团队，是「期望轨迹长度 = 总流」这条恒等式的第三个出口——前两个是最短路（O07）和最优传输（O08），这一个是 MCMC 的自适应终止。

| 字段 | 内容 |
|---|---|
| arXiv | [2606.16073](https://arxiv.org/abs/2606.16073)（v2，2026-07-17；本仓库 PDF 为 v2，23 页） |
| 发表 | **ICML 2026 SPIGM Workshop（非主会）**，arXiv comment |
| 作者 | Kirill Korolev、Nikita Morozov、Stepan Pavlenko、Esmeralda S. Whitammer、Sergey Samsonov（HSE University 等） |
| 代码 | [github.com/kkorolev1/stop-the-sampler](https://github.com/kkorolev1/stop-the-sampler)（原文第 2 页给出；本仓库未复现） |
| 本仓库 PDF | `papers/2606.16073.pdf` · 中译 `papers_zh/2606.16073.zh.pdf`（QA 2 issues） |
| 阅读优先级 | P1：非无环 GFlowNet 理论在连续空间的落地，与 O08 共享 Prop. 3.5 恒等式 |

##### 2. 核心贡献（按原文编号）

- **Proposition 3.3**：Def. 3.2 的流测度与 \(P_F\) 满足流匹配条件 (6)。
- **Theorem 3.4**：流匹配 + 奖励匹配 ⇒ 采样分布 \(P_T(A)=\pi(A)\)。
- **Proposition 3.5（Eq. (14)）**：\(\mathbb E_{\tau\sim P}[n_\tau]=F(\mathcal S)/F(\{s_0\})\)。原文明说这是把 Brunswic et al.（T19）的不等式 (12) 收紧为等式——与 T36 在离散情形做的事相同，这里是连续版。
- **Theorem 3.6**：固定转移核 \(Q_F\)（密度 \(q_F\)，全支撑、一致几何遍历，平稳分布 \(\pi_Q\)），则 \((f,q_F,d_F,q_B,d_B)\) 满足 detailed balance (8) **当且仅当**：(1) 流测度 \(F(A)=Z\,(U(A)+n_Q\pi_Q(A))\)（Eq. (18)），其中 \(U(A)=\sum_{n\ge0}(p_0Q_F^n-\pi Q_F^{n+1})(A)\)（Eq. (19)；第二项用**目标分布** \(\pi\)，不是核的平稳分布 \(\pi_Q\)）是「注入质量与移除质量的累积差」，\(n_Q\ge n_Q^\star:=\sup_s\frac{\max\{\pi(s),p_0(s)\}-u(s)}{\pi_Q(s)}\)；(2) 分类器与后向核由 Eq. (20) 给出：\(d_F(s)=r(s)/f(s)\)，\(d_B(s)=Zp_0(s)/f(s)\)，\(q_B(s\mid s')=\frac{f(s)-r(s)}{f(s')-Zp_0(s')}q_F(s'\mid s)\)。
- **Corollary 3.7**：常数 \(n_Q\) 就是期望轨迹长度 \(\mathbb E[n_\tau]\)；**总流 \(F(\mathcal S)\) 最小当且仅当 \(n_Q=n_Q^\star\)**。
- **Proposition 3.8** + 多层级方案：把状态扩成 \((s,\ell)\)，\(\ell\in\{1,\dots,L\}\)，中间奖励 \(r(s,\ell)\) 按 \(\beta_\ell\) 退火插值（Eq. (23)），相邻层的初始与目标分布由 Eq. (24) 给出；**逐层满足 trajectory balance**（不只是奖励匹配）则终止分布为 \(\pi\)。
- 训练目标：前缀 TB（Morozov et al. 2026 的 prefix trajectory balance，Eq. (21)）+ 流正则（由分类器经 Eq. (16) 直接算出）+ 按停止概率加权（stopgrad）。原文报告 DB 在连续环境表现差、SubTB 项数二次增长，故选前缀 TB。

##### 5. 前提假设与适用边界

一致几何遍历的固定核 \(Q_F\)（Theorem 3.6 前提）；\(\mathbb E[n_\tau]<\infty\)（Assumption 3.1）；连续状态空间上需要 \(\pi,p_0\) 相对 Lebesgue 绝对连续；多层级方案要求可定义退火中间奖励。学到的 \(q_F\) 作为 ULA 核的修正时，理论只覆盖固定核情形。

##### 6. 在 GFlowNet × OT 主线中的位置

- 与 O07/O08 是**同一个量的三种用法**：三篇都从 \(\mathbb E[n_\tau]\propto\) 总流出发。O07 最小化它得到最短路；O08 固定源分布后最小化它得到 OT；本篇把它当 MCMC 的「什么时候停」并给出 \(n_Q^\star\) 的闭式。
- 对主线的启示：Theorem 3.6 的「特解 + \(n_Q\pi_Q\)」分解，是连续空间上的「无环流 + 环空间」；若要把 O08 推到连续状态空间，这里已经给出了流空间的结构。
- 与 Berner 等（TRENDS_GFN §2.3，离散↔连续等价）互补：那篇是极限等价，这篇是直接在连续空间构造非无环流。

##### 7. 可复用的 insight 与开放问题

1. **停止概率 = 状态流的倒数**（Eq. (16)）：一个可迁移的参数化技巧，O08 类模型可以用同样方式省掉状态流网络。
2. **最小流有闭式最优值** \(n_Q^\star\)：在 O08 的离散设定里最小流是 LP 的解；这里因为核固定，最小值可以显式写出，是一个可作 sanity check 的解析基准。
3. 前缀 TB + 流正则 + 停止概率加权：非无环连续训练的一套可直接复用的配方，O08 若做连续版实验应从这里起步。
4. 开放：把 Theorem 3.6 中的固定核换成 O08 式的「学一个使总流最小的核」，是否得到连续空间上的 OT 表述？原文未讨论。
5. 开放：\(n_Q^\star\) 的定义含 \(\sup_s\)，高维时如何估计；原文用学习替代，没有给估计误差。

### 8.4.3 N03 · Stable GFlowNets with TV Monitoring and Probabilistic Guarantees（预印本）

> **一句话**：先证明「学到的分布与目标的 TV 距离很小」并不排除「训练损失无界」（Prop. 3.3–3.4：TV 由聚合对比度 \(1-\Lambda_{\mathcal X}\) 控制，损失上确界由最坏局部对比度 \((\log\min\Lambda_{\{x\}})^2\) 控制，两者可以相差任意多），再反过来给出**损失 → TV 的证书**：逐轨迹 \(\mathcal L_{TB}(\tau)\le c^2\) ⇒ \(\mathrm{TV}\le1-e^{-2c}\)（Theorem 3.5），并把「逐轨迹」这一不可验证的条件换成「抽 \(m+n\) 条轨迹取最大损失」的概率证书（Theorem 3.6）。最后用自适应参考流 \(\delta(\tau)\) 稳定训练，代价是一个可算的保真度折损因子 \((1+\Delta/Z^\star)\)（Theorem 3.10–3.11）。**这是 GFlowNet 侧与本仓库首选课题「balance 残差 → OT 误差界」最近的一篇**：它把「残差 → 边缘误差」这一半做完了。

| 字段 | 内容 |
|---|---|
| arXiv | [2605.01729](https://arxiv.org/abs/2605.01729)（v3，2026-08-09；本仓库 PDF 为 v3，32 页） |
| 发表 | arXiv 预印本（comment 为空；PDF 无会议页眉；截至 2026-09 未见接收信息） |
| 作者 | Zengxiang Lei、Ananth Shreekumar、Jonathan Rosenthal、Ruoyu Song、Alvaro A. Cardenas、Daniel J. Fremont、Dongyan Xu、Satish Ukkusuri（通讯）、Z. Berkay Celik（通讯）——Purdue / UC Santa Cruz |
| 代码 | 原文未给出链接 |
| 本仓库 PDF | `papers/2605.01729.pdf` · 中译 `papers_zh/2605.01729.zh.pdf`（QA 6 issues） |
| 阅读优先级 | **P0**（对课题①而言）：TB→TV 界与概率证书可直接迁移 |

##### 2. 核心贡献（按原文编号）

- **Prop. 3.3（TV 双侧界）**：\(\frac{Z^\star-Z^\star_{\mathcal X_{\text{sub}}}}{Z^\star}(1-\Lambda_{\mathcal X})\le\mathrm{TV}(P_T,\pi_{\text{target}})\le1-\Lambda_{\mathcal X}\)（Eq. (5)）。
- **Prop. 3.4（损失尺度）**：\(\sup|\mathcal L_{GFN}|=\big(\log\min_{\{x\}\subseteq\mathcal X_{\text{sub}}}\Lambda_{\{x\}}\big)^2\)（Eq. (6)），对 FM/DB/TB/SubTB 都成立。**结论**：新模式奖励质量小（TV 小）但相对增幅大（局部对比度小）时，「目标微变、优化信号巨大」——这就是 loss spike 的机理。
- **Theorem 3.5（损失 → TV）**：轨迹级：\(\mathcal L_{TB}(\tau)\le c^2\ \forall\tau\) ⇒ \(\mathrm{TV}\le1-e^{-2c}\)（Eq. (7)），与轨迹长度无关；转移级：\(\mathcal L_{DB}\) 或 \(\mathcal L_{FM}\le c^2\) ⇒ \(\mathrm{TV}\le1-e^{-2Lc}\)（Eq. (8)），\(L\) 为最大轨迹长度，误差在对数域随深度线性退化。
- **Theorem 3.6（概率证书）**：从 \(\hat\pi(\tau)=\pi_{\text{target}}(x_\tau)P_B(\tau\mid x_\tau)\) 抽 \(m\) 条、从 \(P_F\) 抽 \(n\) 条，令 \(c=\max_i\sqrt{\mathcal L_{TB}(\tau_i)}\)（最大损失的**平方根**；\(c\) 控制的是对数比的绝对值），则以置信 \(1-2\alpha\)：\(\mathrm{TV}\le e^{2c}+1-\alpha^{1/m}-\alpha^{1/n}\le e^{2c}-1+\frac{\log(1/\alpha)}{m}+\frac{\log(1/\alpha)}{n}\)（Eq. (9)）。**与状态空间大小无关**。
- **Cor. 3.7**：把目标限制到子集 \(\mathcal X_{\text{sub}}\) 得子图证书。
- **Def. 3.8 / Remark 3.9（参考流）**：\(F_{\text{aug}}(\tau)=ZP_F(\tau)+\delta(\tau)\)，\(R_{\text{aug}}(\tau)=R(\tau)+\delta(\tau)\)；增广损失 \(\mathcal L_{\text{aug}}=\gamma^{-2}\mathcal L_{TB}\)，\(\gamma>1\)（Eq. (12)）；使 \(\mathcal L_{\text{aug}}\le c^2\) 的最小参考流有闭式（Eq. (13)）。
- **Theorem 3.10（保真度折损）**：\(\Delta=\sum_\tau\delta(\tau)\)，若 \(\mathcal L_{\text{aug}}\le c^2\) 则 \(\mathrm{TV}\le\frac{(1-e^{-2c})(1+\Delta/Z^\star)}{1+(1-e^{-c})\Delta/Z^\star}\le(1-e^{-2c})(1+\Delta/Z^\star)\)（Eq. (14)）；\(\Delta/Z^\star\) 可写成 \(\hat\pi\) 下 \(\delta(\tau)/R(\tau)\) 的期望，Monte Carlo 估计 \(M_{TV}\)（Eq. (15)）。
- **Theorem 3.11**：对所有 \(c\in\mathcal C\) 同时成立的概率界（Eq. (16)），允许在训练中**优化阈值 \(c\)**。
- **Algorithm 1（Stable GFlowNets）**：按 Theorem 3.11 自适应注入 \(\delta(\tau)\)，配 top-\(K\) 高奖励缓冲区。

##### 3. 方法与理论推导要点

Theorem 3.5 的证明思路（Appendix B）：\(\mathcal L_{TB}(\tau)\le c^2\) 等价于 \(e^{-c}\le ZP_F(\tau)/R(\tau)\le e^{c}\)，逐轨迹的比值夹在 \([e^{-c},e^c]\)；对所有轨迹求和得 \(Z/Z^\star\in[e^{-c},e^c]\)，再把两个比值相乘得终止概率比 \(P_T(x)/\pi(x)\in[e^{-2c},e^{2c}]\)；TV 是 \(\frac12\sum|P_T-\pi|\)，由比值界得 \(1-e^{-2c}\)。DB/FM 的版本把逐边比值沿长度 \(L\) 的轨迹累乘，指数变成 \(2Lc\)。

Theorem 3.6 把「所有轨迹」换成「抽样的最大值」：从 \(\hat\pi\) 抽的 \(m\) 条覆盖目标质量、从 \(P_F\) 抽的 \(n\) 条覆盖模型质量，未被抽到的部分用 \(\alpha^{1/m}\)、\(\alpha^{1/n}\) 形式的尾概率控制——这就是「与状态空间大小无关」的来源。

**Algorithm 1（Stable GFlowNets）的机制。** 输入 TV 目标 \(d\)、置信 \(1-2\alpha\)、损失阈值 \(c\)、耐心 \(N\)。每轮：从 \(P_F\) 采样、再从 top-\(K\) 缓冲区 \(\mathcal X_{\text{sub}}\) 按 \(R\) 抽终点做后向采样（第 3 行）；用新轨迹经过的状态更新缓冲区（第 4 行）；缓冲区连续 \(N\) 轮不变时才计算证书 \(B_{TV}\)（第 5–8 行，Theorem 3.11 的子图版本）；若证书首项已低于 \(d\) 就**跳过训练只累积样本**（第 9–10 行），否则按 Eq. (13) 计算最小参考流并用 \(\mathcal L_{\text{aug}}\) 更新 \(P_F,P_B,\log Z\)（第 12–13 行）；结束时报 \(M_{TV}\)（Eq. (15)）。阈值 \(c\) 用指数移动平均更新，\(B_{TV},M_{TV}\) 用有界一维优化求（Appendix D）。两个设计决定值得记：证书只在缓冲区稳定后算（避免在分布漂移时报假证书）；训练可以被证书**叫停**（证书不只是事后诊断，也是训练控制器）。

**本报告判断：**这套证明只用到两件事——(i) 残差是 \(\log\) 比值的平方，(ii) TV 是比值偏离 1 的线性泛函。O08 的 OT cost gap \(\sum_e\mathcal F(e)-\mathrm{OT}^\star\) 也是流的线性泛函，边缘违反量 \(\|\hat L-L\|_1\) 同样是。把 Theorem 3.5 的右端换成这两个量，缺的只是「逐轨迹比值界 ⇒ 每条边流的比值界」这一步（在 O08 的 LP 里流 = 轨迹分布的边缘化，是线性的），以及零温 LP 下最优解在多面体顶点、比值界不直接给 cost gap（需要对偶势做桥）。

##### 5. 前提假设与适用边界

DAG（Theorem 3.5 的 DB/FM 版本用到最大长度 \(L\)，有环时失效——这一点对 O08 的非无环设定很关键：只有 TB 版本 \(1-e^{-2c}\) 与长度无关，可迁移）；能从 \(\pi_{\text{target}}\) 抽样（Theorem 3.6 的 \(m\) 条需要目标样本，实践中用 top-\(K\) 缓冲区近似）；参考流 \(\delta>0\) 要求缓冲区中有目标样本。

##### 6. 在 GFlowNet × OT 主线中的位置

- 与 Stable-GFN（Kwon 等，ICML 2026 Spotlight，成对比较消 \(Z\)）**不是同一篇**，名字相近但问题不同：这篇是证书，那篇是红队训练稳定性。
- 与 INSIGHTS §5 课题①的关系：本篇 = 「残差 → 边缘误差」；缺的是「残差 → cost gap」。O08 Thm. 3.3 的对偶势 \(\pi\) 与互补松弛 \(\mathcal F(s\to s')(\pi_{s'}-1-\pi_s)=0\) 提供了另一半：一个近似可行流 \(\hat{\mathcal F}\) 与一个对偶可行 \(\hat\pi\) 之间的 primal–dual gap 就是 cost gap 的上界。
- 与 Evaluation Balance（ICLR 2026，TRENDS_GFN §2.2）同向：两篇都把 balance 残差从「损失」升格为「度量」。

##### 7. 可复用的 insight 与开放问题

1. **可直接复用**：Theorem 3.5 的轨迹级界与 Theorem 3.6 的抽样证书（含 \(\alpha^{1/m}\) 尾项）。O08 的实验若报 primal–dual gap，可以同一格式报 TV 证书。
2. **必须改的**：O08 是非无环图，DB 版本的 \(L\) 依赖失效；O08 的训练目标 Eq. (20) 是带源分布项的正则化 TB，形式上能接 Theorem 3.5 的轨迹级分支（迁移仍需前提：双向完整轨迹律归一化、源边与终止边计入残差、O08 中状态可继续转移故终止概率对应最后终止边质量）；若用 T36 公开代码的 DB + 状态流正则实现，Theorem 3.5 的 DB 分支在 \(L\to\infty\) 时退化为 1 的**平凡界**（不是无穷）。以期望长度替代最大长度的界需要**重新证明**（明确用哪个轨迹分布的期望、归一化条件与尾部控制），不能把均值直接代入指数；T36 Prop. 3.12 只给出期望长度与总流的恒等式，是这条路的起点而非证明。
3. **参考流是「共同加性参考轨迹测度」的稳定化机制**：\(\delta(\tau)\) 同时加到模型轨迹测度与目标轨迹测度上。它与 OT 正则的关系**尚未建立**——同一轨迹测度在起点与终点上的投影一般不同，所以它不等于给两个边缘加相同质量；若要声称对应某种 OT 正则化，须写出增广原问题、两端边缘与对偶。
4. 开放：Theorem 3.6 要求从目标抽样，在 O08 设定里目标耦合 \(\Pi^\star\) 未知——但目标**边缘** \(L,R\) 已知，能否只用边缘样本给证书？
5. 开放：损失集中比值（Figure 3）在 O08 的排列环境里是否同样出现？若是，流正则 \(\lambda\) 的作用可能主要是压这个比值。

### 8.4.4 N04 · Generative Modeling on Metric Graphs via Neural OT（预印本）

> **一句话**：第一篇在**度量图**（连续支撑在边上的分布，如路网上的上车点）上做深度生成建模的工作：把图嵌进光滑环境空间（欧氏实现，或 tropical Abel–Jacobi 嵌入到 Jacobian 环面），在嵌入空间解熵正则 Kantorovich 问题的神经半对偶形式，再把生成样本最近投影回图上；证明神经表达力增大时生成器弱收敛到合法的图上传输耦合（Theorem 4.1）。它与 O08 在「图」这个词上重合，在对象与输出上不重合：这里的分布是边上的连续测度、训练代价是嵌入空间的二次代价 \(c_\Psi=\tfrac12d_\Psi^2\)（图测地距离只用于评估）、输出是采样器而不是逐边策略。

| 字段 | 内容 |
|---|---|
| arXiv | [2606.16273](https://arxiv.org/abs/2606.16273)（v1，2026-06-15，31 页） |
| 发表 | arXiv 预印本（comment 为空；PDF 无会议页眉；截至 2026-09 未见接收信息） |
| 作者 | Alessandro Micheli、Yueqi Cao（共同一作）、Anthea Monod、Samir Bhatt（共同通讯）——Imperial College London / KTH / Statens Serum Institut |
| 代码 | 原文未给出链接 |
| 本仓库 PDF | `papers/2606.16273.pdf` · 中译 `papers_zh/2606.16273.zh.pdf`（QA 6 issues） |
| 阅读优先级 | P1：图上 OT 的「连续边」分支，与 O08 的「离散顶点」分支互补 |

##### 2. 核心贡献（按原文编号）

- 三步方法（§3）：嵌入 → 在嵌入空间用神经势解熵 OT 半对偶 → 由 Gibbs 条件律采样并**最近投影**回 \(\Gamma\)（projection–pullback 生成器）。
- **Theorem 4.1（Graph-supported recovery）**：若神经函数族 \(\mathcal F\) 在 \(C(\mathbb R^n,\mathbb R)\) 中按 ucc 拓扑稠密，且目标支撑上的神经特征映射 \(\varphi_\Psi\)（与图到环境空间的映射 \(\Psi\) 是两个不同对象）连续单射，则存在神经势序列 \(g_m\)，其诱导的嵌入 Gibbs 律满足 \(\|\pi^\varepsilon_{m,\Psi}-\pi^\star_{\varepsilon,\Psi}\|_{TV}\to0\)；并存在联合子序列 \(m_k\to\infty,t_k\downarrow0\)（热核平滑参数）使投影–回拉生成器弱收敛到原图上的合法传输耦合。含桥的图须先做 Appendix B 的平行虚拟边增广，tropical 映射的单射性才成立。恢复的是**嵌入空间熵 OT** 最优耦合的回拉，不是对图测地代价最优的耦合。
- 实验（§5）：合成度量图（theta、wheel、grid、road），对比两个「源感知推前」启发式基线——节点插值（顶点级离散图 OT 再沿边插值）与环境推前；指标为图上 \(W_1^\Gamma,W_2^\Gamma\)、密度 \(L_1\)、边 CDF \(L_1\)。真实数据：曼哈顿路网上 \(10^6\) 个 Uber 上车点。

##### 5. 前提假设与适用边界

度量图有限、边长已知、可显式嵌入；\(\varepsilon>0\)（熵正则，零温不在框架内）；需要源与目标的**样本**（不是未归一化密度）；Theorem 4.1 是存在性 + 极限陈述，无速率。

##### 6. 在 GFlowNet × OT 主线中的位置

- **不是竞品，是互补分支**。O08：离散顶点上的概率质量、单位跳数代价、零温 LP、输出逐边策略、隐式图。N04：边上的连续测度、嵌入空间二次代价训练（图测地距离仅评估）、熵正则、输出采样器、显式且需要嵌入的图。两者唯一的公共部分是「图上 OT 的目标分布」。
- 对 O08 的有用参照：N04 的评估协议（图上 \(W_1/W_2\) + 密度 \(L_1\) + **评估噪声底**）比 O08 的 TV + \(\mathbb E|\tau|\) 更完整，尤其「噪声底」这一行是 O08 实验缺的。
- 在 `TRENDS_OT_2026.md` §2.1 记为「相邻竞争」；读完全文后本报告把它下调为「互补」。

##### 7. 可复用的 insight 与开放问题

1. **评估噪声参照**：用两份独立测试子样本在同一评估预算下互比，给出有限测试样本造成的经验误差参照（并报重复采样的不确定性）——O08 类实验应加这一行。
2. **内在 vs 外在嵌入**的对照设计可迁移：O08 的排列图 Cayley 结构也有「内在」（群距离）与「外在」（某个特征嵌入）两种表示。
3. 开放：把 N04 与 O08 接起来需要先统一代价——N04 训练用的是嵌入空间二次代价，O08 是图最短路代价；只让熵系数趋零不会完成这个转换。度量图上最短路代价的 \(W_1\) 有 Beckmann 形式（O01 §6.5），N04 没有利用这一点。
4. 开放：N04 需要样本访问；O08 的定理设定需要已归一化的两端边缘点值（Assumption 3.1），一般 GFlowNet 的未归一化奖励访问在这条线上尚未推导。三种输入访问模式之间的中间形态在本仓库核查范围内未见。

### 8.4.5 N05 · Orlicz-Sobolev Transport for Unbalanced Measures on a Graph（NeurIPS 2025 Spotlight）

> **一句话**：处理图度量空间上**总质量不等**的两个测度之间的传输。先用 Caffarelli–McCann 的思路把熵部分传输（EPT）改写成一个带非负地面代价 \(\hat c\) 的标准（平衡）OT（Prop. 3.1，加一个虚拟点 \(\hat s\) 吸收多余质量），再赋予 Orlicz 几何得到 Orlicz-EPT（Eq. (8)）；发现它仍是两层优化、超立方复杂度，于是借对偶 EPT 与图结构构造正则化版本 **Orlicz-Sobolev 传输（OST，Def. 4.1）**，并证明 OST 可归结为**一个一元优化问题**（Theorem 4.2），离散情形有显式表达（Cor. 4.3），实测比 Orlicz-EPT 快 250–13800 倍。对主线的意义：O08 要求 \(\sum L=\sum R=1\)；若放开成 unbalanced，这篇给出了一个不靠 KL 罚、有闭式计算、且天然定义在图上的候选目标。

| 字段 | 内容 |
|---|---|
| arXiv | [2502.00739](https://arxiv.org/abs/2502.00739)（v2，2025-10-24，42 页） |
| 发表 | **NeurIPS 2025 Spotlight**（arXiv comment "to appear in Neural Information Processing Systems (NeurIPS), 2025. [spotlight]"） |
| 作者 | Tam Le、Truyen Nguyen（共同一作）、Hideitsu Hino、Kenji Fukumizu——统计数理研究所（ISM）/ University of Akron |
| 代码 | [github.com/lttam/OST_OrliczEPT](https://github.com/lttam/OST_OrliczEPT)（原文第 2 页脚注 2） |
| 本仓库 PDF | `papers/2502.00739.pdf` · 中译 `papers_zh/2502.00739.zh.pdf`（QA 11 issues） |
| 阅读优先级 | P1：unbalanced 图上 OT 的非 KL 路线 |

##### 2. 核心贡献（按原文编号）

- **Prop. 3.1（EPT ⇔ 标准 OT）**：\(ET_\lambda(\mu,\nu)=(\mu(G)+\nu(G))\big(W_{\hat c}(\hat\mu,\hat\nu)-b\lambda\big)\)（Eq. (7)，仿射关系而非直接相等），其中 \(\hat\mu=(\mu+\nu(G)\delta_{\hat s})/(\mu(G)+\nu(G))\)、\(\hat\nu\) 对称定义——先加入交叉总质量的虚拟点 \(\hat s\)，再以共同总质量归一化；代价 \(\hat c\) 见 Eq. (5)；**Remark 3.2**：与既有做法不同，\(\hat c\ge0\) 有保证，这是后面能套 Orlicz-Wasserstein 框架的关键校准。
- **Orlicz-EPT（Eq. (8)）**：\(OE_\Phi(\mu,\nu)=(\mu(G)+\nu(G))(W_\Phi(\hat\mu,\hat\nu)-b\lambda)\)；**Prop. 3.3** 单调性；**Prop. 3.4/3.5** 熵正则版本 \(A_\varepsilon\) 的单调性与上下界，二分搜索求解（Eq. (11)，二次复杂度的内层替代超立方的 (9)），但两层结构仍重。
- **Def. 4.1（OST）**：\(OS_{\Phi,\alpha}(\mu,\nu)=\sup_{f\in U_{\Psi,\alpha}}\big(\int f\,d\mu-\int f\,d\nu\big)\)（Eq. (14)），\(U_{\Psi,\alpha}\) 是图上 Orlicz–Sobolev 空间 \(WL^\Psi(G,\omega)\) 中导数 Orlicz 范数 \(\le b\)、根值落在区间 \(I_\alpha\) 的函数——OST 是一个积分概率度量（IPM）。
- **Theorem 4.2（一元优化）**：\(OS_{\Phi,\alpha}(\mu,\nu)=\Theta\,|\mu(G)-\nu(G)|+\inf_{k>0}\frac1k\Big(1+\int_G\Phi\big(kb\,|\mu(\Lambda(x))-\nu(\Lambda(x))|\big)\omega(dx)\Big)\)（Eq. (16)），\(\Theta\) 由 Eq. (15) 按哪边质量大取值，\(\Lambda(x)\) 是 \(x\) 的「子树」（经 \(x\) 的路径所达的点集）。质量差被一个线性项 \(\Theta|\mu(G)-\nu(G)|\) 显式吸收。
- **Cor. 4.3（离散情形）**：测度支撑在顶点时积分变成对边的求和（Eq. (17)），配合 Dijkstra 预处理 \(O(|E|+|V|\log|V|)\) 与「只对 \(E_{\mu,\nu}\) 中的边求和」的稀疏性。
- **Prop. 5.5/5.6**：极限 \(N\)-函数 \(\Phi_0\) 下 OST 有闭式，Orlicz-EPT 退化为图上 EPT。

##### 5. 前提假设与适用边界

图为物理图且路径 \([z_0,x]\) 唯一（树状；非物理图见 Remark 4.4 的处理）；测度非负有限；\(\Phi\) 为 \(N\)-函数；OST 是 IPM 而非 EPT 本身——它是 EPT 的正则化替代，不是等价物。

##### 6. 在 GFlowNet × OT 主线中的位置

- O08 Assumption 3.1 第 4 条要求两边缘归一化。放开它有两条路：KL 罚（ULOT/C01、GSBoG 一系的做法）或本篇的「虚拟点 + 线性质量差项」。后者的好处：质量差由线性项显式吸收，计算是一元优化；坏处：一元优化不是线性规划（一般 \(\Phi\) 给出非线性 Orlicz 范数），且 OST 计算的是两测度间的标量差异，并不自动产出非负边流、可采样策略或不平衡质量的生成机制；此外需要根 \(z_0\) 与根最短路系统（Appendix B.2 给出有环图、权重扰动与多重最短路的择路约定）。
- 与 O02（Essid & Solomon 图上二次正则 OT）同属「图上 OT 的正则化」，但正则对象不同：O02 正则边流，本篇正则对偶势的导数。
- `TRENDS_OT_2026.md` §2.1 记为「互补」，本报告维持。

##### 7. 可复用的 insight 与开放问题

1. **质量差可以线性吸收**（Theorem 4.2 首项）：给 O08 加 unbalanced 时，不必上 KL 罚，先试「虚拟汇点 + 线性项」。
2. **树上的 Beckmann 净流是显式的**：O08 若在树状状态图、顶点支撑、\(\Phi(t)=t\)、\(b=1\) 的条件下做 sanity check，OST 与图 OT 精确相等（Prop. A.2）可作参考值；「近似树状」不能直接获得精确参考。
3. 开放：O08 的状态图有环且有向。有环不等于根最短路不唯一（可选定唯一根最短路系统），但选定后 OST 值也不等同于原有环图上的 Beckmann 最优值；Remark 4.4 只放宽物理欧氏实现要求，不处理有向性。一般有向有环图上的对应关系待建立。
4. 开放：把 OST 的对偶势 \(f\)（Orlicz–Sobolev 类）与 O08 固定双边缘问题的对偶势 \(\pi\)（Appendix A.4，有向内部边上的单侧约束 \(\pi_{s'}-\pi_s\le1\)；Thm. 3.3 只针对去掉首步约束的单源特例）并列——两者是同一对偶变量在不同函数类下的版本，本仓库核查范围内未见比较。

### 8.4.6 N06 · A Benchmark for Discrete Schrödinger Bridges and EOT（ICLR 2026）

> **一句话**：离散空间上的熵正则 OT / Schrödinger 桥（SB）第一次有了**有解析解的基准**。核心是 Theorem 3.1：给定初始分布 \(p_0\) 与一个使 \(q^\star\) 成为合法概率分布的标量函数 \(v^\star\)（充分条件：\(v^\star\ge0\)、非零，且 \(0<c^\star(x_0)<\infty\)），令 \(q^\star(x_1\mid x_0)\propto v^\star(x_1)q^{\text{ref}}(x_1\mid x_0)\)，则 \((p_0,\ p_1:=q^\star\text{ 的第二边缘})\) 之间以 \(q^{\text{ref}}\) 为参考的 EOT/SB 解就是 \(q^\star\)；用 CP 分解参数化让它在 \(|\mathcal X|=S^D\) 的高维空间可算（Prop. 3.1/3.2）。副产品是三个求解器 DLightSB、DLightSB-M、\(\alpha\)-CSBM，其中 DLightSB 总体最强（有例外，见第 4 节）。对主线的意义：任何「熵正则 GFN–OT」若声称在解 SB，就必须在这个基准上报数，而它同时是课题③窗口关闭的证据——离散 SB 已从「方法」进入「基准」阶段。

| 字段 | 内容 |
|---|---|
| arXiv | [2509.23348](https://arxiv.org/abs/2509.23348)（v2；本仓库 PDF 24 页） |
| 发表 | **ICLR 2026 主会**（依据：PDF 页眉 "Published as a conference paper at ICLR 2026"；arXiv comment 为空，本仓库 CSV 已据此修正） |
| 作者 | Xavier Aramayo Carrasco、Grigoriy Ksenofontov（共同一作）、Aleksei Leonov、Iaroslav Koshelev 等——Applied AI Institute / MIRAI（莫斯科） |
| 代码 | [github.com/gregkseno/catsbench](https://github.com/gregkseno/catsbench)（摘要末给出；CSBM 求解器见 github.com/gregkseno/csbm） |
| 本仓库 PDF | `papers/2509.23348.pdf` · 中译 `papers_zh/2509.23348.zh.pdf`（QA 5 issues） |
| 阅读优先级 | P1：熵正则路线的必报基准 |

##### 2. 核心贡献（按原文编号）

- **M3.1 / Theorem 3.1（基准对构造）**：任意使 \(q^\star\) 合法的 \((p_0,v^\star)\) 诱导基准对 \((p_0,p_1)\)，其 EOT/SB 解 \(q^\star(x_1\mid x_0)=\frac{1}{c^\star(x_0)}v^\star(x_1)q^{\text{ref}}(x_1\mid x_0)\)（Eq. (7)）闭式已知。原文指出这是 Gushchin et al. (2023b) 连续空间构造的离散版。
- **M3.2 / Prop. 3.1、3.2（可算参数化）**：高维下 \(c^\star(x_0)\) 与 \(q^\star\) 的求和有 \(S^D\) 项；把 \(v^\star\) 取成 CP 分解形式（混合分量 \(k\)、维度 \(d\) 各一非负向量 \(r_{kd}\in\mathbb R_+^S\)），归一化与条件分布都分解成一维求和的乘积，可精确计算（Prop. 3.1 针对条件分布，Prop. 3.2 针对 SB 转移分布）。
- **M3.3**：据此构造高维高斯混合基准，覆盖 \(q^{\text{gauss}}\)（\(\gamma=0.02\)）与 \(q^{\text{unif}}\)（\(\gamma=0.005\)）两种参考过程、不同 \(D\)。
- **M4 求解器**：CSBM（Ksenofontov & Korotin 2025，既有）；**\(\alpha\)-CSBM**（M4.2，把 \(\alpha\)-DSBM/O04 的在线更新并入 CSBM）；**DLightSB**（M4.3，LightSB 的离散版，直接来自基准构造）；**DLightSB-M**（M4.4，动态扩展）。
- **评估指标**：条件 Shape Score 与 Trend Score（Table 1a/b）、轨迹 KL 与反向 KL（Table 3/4）、C2ST（Table 2，作者说明其数值「不具信息量」——所有方法都接近）。

##### 5. 前提假设与适用边界

离散时间、因子化（逐维独立）参考过程；\(v^\star\) 为 CP 形式；\(\varepsilon>0\)（熵正则是 SB 的定义本身）。零温 LP（O08 的设定）不在基准范围内，但作为 \(\varepsilon\to0\) 的极限可以对照。

##### 6. 在 GFlowNet × OT 主线中的位置

- 对课题③（熵正则 GFN–OT / 图上 SB）：**必报基准**。GSBoG（C02）、DDSBM（C03）之后，N06 定了口径；任何新的离散 SB 求解器若不在此基准上报 Shape/Trend Score 与轨迹 KL，就无法说明自己解的是 SB。
- 对课题③窗口关闭的证据强度：**中等偏强，且有边界**。乘积结构离散空间上有了解析基准 + 多个求解器 + 失效模式分析，新进入者在这类设定下没有「先定义问题」的红利；但基准不覆盖一般图、非因子化动力学与其他输入访问模式，是否「解的是 SB」由目标、参考过程与边缘约束决定，不由是否使用某个基准决定。
- 对课题①的间接价值：N06 的 \(\alpha\)-CSBM 把 O04 的 \(\alpha\) 在线更新移植到离散空间；原文（第 7 页）明确称离散 \(\alpha\)-IMF 为启发式类比，**没有证明**它与 GFlowNet 的固定反向策略或边流归一化同构，O04 报告 §6 的对应关系在离散空间仍是猜想。

##### 7. 可复用的 insight 与开放问题

1. **构造有解析解的基准的配方**：固定参考动力学 + 选一个使耦合合法的势 + 定义边缘。移植到 O08 的零温设定**有条件可行但不是「任选势即可」**：互补松弛只给出允许承载最优流的饱和边集合，不自动生成满足非负性、质量守恒与指定终端支撑的流——常数势就是反例（单位代价下无边饱和，只能零流，得 \(R=L\)；源终端支撑不相交时不可行）。正确做法是联合构造可行势、紧边路径与非负运输流，证明每个正源质量都能到达允许终点，再定义 \(R\) 并检查非退化。这是 O08 实验缺的东西，但配方需要本文自己补证明（见 INSIGHTS §6c 第 3 天）。
2. C2ST 的教训需谨慎：Table 2 是 Appendix D.1 反向基准上对端点对 \((x_0,x_1)\) 的分类 ROC AUC，作者因数值接近而弃用；按通常定义 0.90–1.00 表示生成与真实**可区分**，与作者的解读方向相反，指标定义与方向需核实。不能由此推出高维二分类检验普遍失效，只能说它在该基准上不区分求解器优劣。
3. 开放：把 N06 的基准 \(\varepsilon\to0\) 后与 O08 的 LP 解对照，验证熵选择原理在离散图上选出的计划是否等于最小总流计划。
4. 开放：Theorem 3.1 的端点重加权构造对一般离散参考链成立（Eq. (11) 的 Doob 变换不要求乘积结构），但本文只为逐维因子化的参考链给出可扩展的解析归一化与采样参数化（Prop. 3.1/3.2）；图上非乘积参考链的**高效**构造仍缺。

### 8.4.7 六篇合看：对四个候选课题评级的影响

| 课题 | 六篇带来的变化 |
|---|---|
| ① 残差 → OT 误差界 + 对偶势证书 | 从「配件在别处」变成「配件在手」：N03 Thm. 3.5/3.6 是残差 → 边缘误差的现成证书，N01 Thm. 2 是对偶误差 → 运行时间的界，N06 Thm. 3.1 是构造解析基准的配方。评级维持**做**，且可开工的程度提高 |
| ② 条件 GFN 摊销图上 OT | 无新证据改变「不做」 |
| ③ 熵正则 GFN–OT / 图上 SB | N06 以 ICLR 2026 主会身份定下离散 SB 的评测口径，窗口关闭的证据更强；N02 说明连续非无环 GFN 的团队自己也在往采样器方向走，而不是 SB |
| ④ GFN proposal + 经典 OT 修正 | N01 把它具体化为「对偶预测 → ε-relaxation」，但只在显式图上成立；N04 说明显式连续图上另有神经 OT 路线。评级维持「降为①的应用」 |

# 第 9 章 Insight、候选课题评级与决定性实验

本章收录跨论文综合文档 `reports/INSIGHTS.md` 的全部内容：一页结论、主线逻辑链、OT 对照表、竞争格局、四个课题的评级、可立即执行的决定性实验、八个易误读点与开放问题。

### 1. 一页结论

1. **一般图上，终止奖励不足以唯一确定内部流；奖励只钉住边界。** 终止流由 \(R\) 决定，非终止边上的 \(P_B\) 自由（T02 Prop. 18 第 3 条；§2.6）；有环图上自由度再加一个环空间 \(H^1_+(G)\)（T19 Prop. 5）。这是整条主线的起点。
2. **最小总流是一个合法的选择原则，且有精确的行为学含义。** 有限离散非无环图上 \(\sum_{s}F(s)=Z\cdot\mathbb E[n_\tau]\)（T36 Prop. 3.12 / Eq. (10)，求和取内部状态，把 T19 Thm. 2 的「≤」收紧为「=」），所以「总流最小」= 「期望轨迹最短」。
3. **最短期望轨迹 ⇔ 只走最短路。** 单源情形下期望长度取最小值当且仅当策略把全部质量放在最短路上（O07 Thm. 3.4，充要）。
4. **固定源分布后，最小总流 = 图最短路代价下的 Kantorovich OT。** \(\mathrm{GFlow}^\star=\mathrm{OT}^\star\)，且最优策略采出的轨迹端点分布是最优耦合（O08 Thm. 3.2）。O08 Thm. 3.3 讨论的是**去掉第一步分布约束的单源原问题**：对偶势在终止状态上等于从 \(s_0\) 出发的最短路距离，互补松弛把最优流限制在最短路子图；固定多源分布 \(L\) 后适用的是 Appendix A.4 的扩展对偶（目标含源势项），终点势不再一般地等于单源距离。
5. **数学等价关系已有经典结果，新增之处在于 GFlowNet 的策略接口。** 图上 shortest-path OT ≡ min-cost flow ≡ 离散 Beckmann（O02 Eq. (1)⇔(3)；O01 Prop. 6.23、Prop. 14.9）。O08 的贡献是把它翻译成 GFlowNet 语言之后，输出的不是耦合矩阵而是能在 \(20!\) 规模隐式图上执行的局部路由策略（O08 §4.2）。
6. **在标准单源 DAG 设定下，「GFlowNet 学到 OT 计划」是空话。** 源边缘是 \(\delta_{s_0}\) 时耦合集是单点集（O01 Remark 3.2）。非平凡的 OT 结构必须同时放开初始流分布并加入外生准则，两者缺一不可。
7. **主线放弃了 GFlowNet 的招牌能力。** O08 Assumption 3.1 要求 \(\sum L=\sum R=1\)，\(Z\) 已知；「只需未归一化奖励」在这条线上不成立（O08 §7.6）。
8. **竞争格局：本方两篇是 Workshop，三篇竞品全是主会。** ULOT（NeurIPS 2025）、GSBoG（ICML 2026）、DDSBM（ICLR 2025）；GSBoG 与 O08 同象限、卖点句几乎相同、规模高四个数量级（COMPETITOR_MATRIX §0–§3）；本仓库核查的 GSBoG 版本中未见收敛率、边缘误差或目标次优性中任一种明确保证（C02 报告 §5），这是与 O08 的 LP 对偶证书最具体的差异。
9. **最不拥挤的后续课题是误差证书，且配件已经齐了。** GFN 侧 N03（`reports/N03_2605.01729.md`）已把「逐轨迹 TB 损失 \(\le c^2\) ⇒ \(\mathrm{TV}\le1-e^{-2c}\)」（Thm. 3.5）与「抽样概率证书、与状态空间大小无关」（Thm. 3.6）做完；OT 侧 N01（AAAI 2026）给出「对偶预测误差 \(\|\hat p-p^\star\|_\infty\) → 运行时间」的界（Thm. 2），N06（ICLR 2026）给出构造有解析解基准的配方（Thm. 3.1）。缺的只有一段：把 N03 的右端从 TV 换成 OT cost gap 与边缘违反量，桥梁是 O08 Thm. 3.3 的对偶势与互补松弛。
10. **两个课题已经拥挤：** 条件/摊销 GFN–OT（UNOT、ULOT、切片势摊销），熵正则 GFN–OT / 图上 SB（GSBoG、DDSBM、Sampling Decisions 已把 GFN 流函数写成单侧 Schrödinger 传输）。

### 2. 主线的逻辑链：每一环缺什么、下一环补什么

| 环 | 论文 | 这一环建立了什么 | 它留下的缺口 |
|---|---|---|---|
| ① | T00 / T02 | 流守恒 + 终止流 = 奖励 ⇒ \(P_T\propto R\)（T02 核心正确性定理）；Markovian flow 由终止流与 \(P_B\) 唯一确定（Prop. 18 第 3 条），可行集是线性的（Prop. 19 / Eq. (22)） | 只在 DAG 上；\(P_B\) 自由度被当作「工程选择」，没有任何原则说该选哪个。原文唯一一次提到 shortest 是 §2.6 的一句「可以偏好更短路径」 |
| ② | T03 / T05 / T10 | TB 把约束搬到整条轨迹，\(\log Z\) 成为可学参数；SubTB(\(\lambda\)) 在 DB 与 TB 之间插值；T10 诊断训练难点 | 全部在 DAG 上、全部关于「如何更快到达某个合法流」，不问「到达哪一个」 |
| ③ | T19 | 有环时流是测度、轨迹集合无限；0-flow 刻画环；比值型损失把流堆进环（Thm. 3），差值型损失稳定（Thm. 4）；R-流 = 一个无环流 + 环空间（Prop. 5）；\(\mathbb E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\)（Thm. 2） | 「极限流无环」不等于「解唯一」——DAG 上环空间为零时 R-流照样不唯一（T19 报告编者注）；总流与长度之间只有不等式 |
| ④ | T36 | 以 \(P_B\) 为第一性对象，流 = 期望访问次数 × 终止流，与 \((P_B,F(s_f))\) 一一对应（Prop. 3.7）；\(\sum_{s\ \text{内部}}F(s)=Z\,\mathbb E[n_\tau]\)（Prop. 3.12 / Eq. (10)，求和取内部状态，\(n_\tau\) 计内部状态访问次数）；固定 \(P_B\) 时解唯一（Cor. 3.11）；写出「最小总流」约束优化 Eq. (11) 与 DB + 状态流正则的近似解法 | 最小值落在多面体边界、\(P_B\) 部分为零处，超出 T36 自身的正性假设，只能用 \(\lambda\) 正则逼近；没有说最小流「是什么」 |
| ⑤ | O07 | 期望长度最小 ⇔ 只走最短路（Thm. 3.4，充要）；用 Assumption 3.1 \(\mathbb E[n_\tau]<\infty\) 替换 \(P_B>0\)；把寻路归约为训练带流正则的非无环 GFN | 单源；\(R\equiv1\) 或单目标；只回答「路径」不回答「质量怎么分配」 |
| ⑥ | O08 | 再钉一条约束：第一步边流 = 源分布 \(L\)。目标从双线性化为 LP（Appendix A.1），约简成 divergence 约束 \(\operatorname{div}\mathcal F=L-R\)（Eq. (11)），两侧夹逼得 \(\mathrm{GFlow}^\star=\mathrm{OT}^\star\)（Thm. 3.2）；对偶 + 互补松弛（Thm. 3.3） | 神经训练与 LP 最优之间无界；\(Z\) 必须已知；无 unbalanced；hypergrid \(H\le20\)、排列 \(n\le20\)，\(n=20\) 无 \(\mathrm{OT}^\star\) 参考 |

**本文判断：**这条链上真正的「转折点」是 ④ 的等式 \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\)。它建立了流空间上的线性泛函（总流）与策略行为量（期望步数）之间的精确关系，此后 ⑤⑥ 都是对这个线性泛函在不同约束下取极小。理解了 ④，⑤⑥ 就是线性规划课的两道习题；没理解 ④，会把 O08 读成「GFlowNet 神奇地学会了 OT」。

### 3. OT 侧对照：哪些是等价，哪些只是类比

从 O01 报告 §7 的 16 行对照表提炼（讲义章节号按 arXiv v3）：

| OT 概念 | GFlowNet 概念 | 关系 | 出处 |
|---|---|---|---|
| 有向边流 \(m_e\)，目标 \(\sum_e\ell_e\lvert m_e\rvert\) | 边流 \(F(s\to t)\) 与总流 | **严格等价** | O01 Def. 6.20、Prop. 6.23；O02 Eq. (1)⇔(3) |
| 守恒律 \(\operatorname{div}_G m=r\) | flow matching（内部顶点 \(r_i=0\)） | **严格等价** | O01 Prop. 6.23 |
| 代价 = 图最短路 \(d_G\) | 轨迹长度 / 期望步数 | **条件等价**（路径作用取边长之和） | O01 Prop. 14.9；O08 Eq. (7)–(8) |
| 边缘约束 \(\pi_1=\alpha,\pi_2=\beta\) | 奖励匹配 \(F(x\to s_f)=R(x)\) + 第一步边流 = \(L\) | **条件等价**（\(R\) 须归一化，\(Z=1\)） | O08 Assumption 3.1 |
| 互补松弛：支撑 ⊂ 接触集 | 最优流只在紧边上为正 | **严格**（LP 形式下） | O01 Prop. 5.3；O08 Thm. 3.3 |
| Kantorovich 对偶势 \(f_i\) | 状态流 \(\log F(s)\) / 价值函数 | **类比**（OT 是不等式约束，DB 是等式） | O01 Def. 5.1、Prop. 6.23 |
| 耦合 \(\mathbf P\in\mathcal U(\mathbf a,\mathbf b)\) | 轨迹分布的端点边缘 | **类比**，单源时退化为单点 | O01 Remark 3.2 |
| 熵正则 \(\varepsilon\mathrm{KL}(\pi\|\alpha\otimes\beta)\) | MaxEnt GFN 的路径熵 | **类比**，差一个条件项（端点熵 vs 路径熵） | O01 Def. 8.2、Eq. (8.8) |
| Sinkhorn = 交替 KL 投影 | 交替强制约束的训练 | **类比**（OT 半步有闭式，GFN 只能 SGD） | O01 Prop. 9.4/9.5 |
| Schrödinger 桥参考路径律 | 固定的 \(P_B\) | **条件等价** | O01 Prop. 14.11；O04 \(\mathrm{proj}_{\mathcal R}/\mathrm{proj}_{\mathcal M}\) 同构表 |
| Benamou–Brenier / JKO | 「流 × 速度」/ 训练动力学 | **仅氛围类比**，不要当定理用 | O01 Thm. 14.5、Def. 15.1 |

### 4. 竞争格局

三条分界线决定谁和谁真正在竞争（COMPETITOR_MATRIX §2）：

- **图是「被传输的对象」还是「传输发生的场所」。** ULOT（C01）、DDSBM（C03）属前者：传输在图与图之间；O08、O07、GSBoG（C02）属后者：传输在一张图内部。跨线比较数字没有意义。
- **显式枚举还是隐式展开。** GSBoG 跑到 \(10^6\) 节点，但要求 \(\mu,\nu\) 是显式向量、能枚举邻域、拓扑完全已知；O08 的排列环境 \(20!\) 个状态没有任何显式表示。这是 GFN 侧唯一无法被复制的结构性优势。
- **温度。** O08/O07 在 \(\varepsilon=0\)（纯 LP、顶点解、有对偶证书）；GSBoG/DDSBM 在 \(\varepsilon>0\)（指数族解、有 IPF/IMF 收敛机制、无误差界）。给 O08 加入 KL 项后，研究定位将与 GSBoG 重叠。

具体威胁（吸收 O04/O05 报告的结论）：

| 来源 | 威胁什么 | 强度 | 反制点 |
|---|---|---|---|
| **GSBoG**（C02，ICML 2026 主会） | O08 的整个定位叙事：「不是静态耦合而是可执行策略」几乎逐字相同 | 最高 | 隐式图；不定时域（GSBoG 必须固定 \(T\)）；O08 有 LP 对偶证书，而核查版本的 GSBoG 未见任一种明确误差保证；GSBoG 未开源 |
| **UNOT**（O05，ICML 2025） | 摊销轴：跨数据集/跨分辨率泛化，非训练集相对误差 1.3–2.8%（Table 4）；它摊销的对偶势正是 O08 Thm. 3.3 中 \(\pi_x\) 的连续版 | 高 | 换图即换代价（UNOT §6 自述做不到）；\(\lvert\mathcal S\rvert\) 大到存不下代价矩阵 |
| **ULOT**（C01，NeurIPS 2025） | 「条件 GFN 学一族图 OT」课题 | 高 | ULOT 硬上限 \(n\le10^4\)、输出 plan 不输出 policy、全文零定理、代价是二次 FUGW |
| **α-DSBM**（O04，NeurIPS 2024 Spotlight） | 不是威胁，是互补：\(\mathrm{proj}_{\mathcal R}/\mathrm{proj}_{\mathcal M}\) 与「固定 \(P_B\)」/「由边流反解 \(P_F\)」逐条同构，Eq. (26) + Lemma D.2 是「图上 SB = 熵正则 min-flow GFN」的现成证明模板 | — | 用它，不要和它比 |
| **DDSBM**（C03，ICLR 2025） | 「熵正则 GFN–OT 用于分子编辑」的应用位 | 中 | DDSBM 需解 QAP 图匹配，GFN 沿路径走不需要；DDSBM 不条件化 |

### 5. 四个候选课题的最终评级

| 课题 | 评级 | 理由（含 2026 趋势扫描后的更新） |
|---|---|---|
| **① Balance 残差 → OT 误差界，对偶势作证书** | **做。撞车风险低，配件齐全（见 N01/N03/N06）** | GFN 侧：Stable GFlowNets 已有 TB 残差 → TV 界（2605.01729），Evaluation Balance 把残差当评估器（2603.01047，ICLR 2026）。OT 侧：误差理论全在势上——对偶预测 → \(\varepsilon\)-relaxation 的运行时间界（2601.20203，AAAI 2026）、Sinkhorn 势统计率（2608.29152，O08 合作者 Belomestny）、QOT 的 PL 不等式（2605.27175）。**本仓库核查的文献集合内未见把近似可行流的残差映射到 cost gap 的工作**（经典 LP 的原始–对偶后验误差分析需另行补查）。 O08 Thm. 3.3 的互补松弛给出逐边证书 \(\mathcal F(s\to s')(\pi_{s'}-1-\pi_s)=0\)，直接可用 |
| ② 条件 GFN 摊销一族图上 OT | 不做（降为对照） | UNOT、ULOT 之后又来了切片势摊销（2604.15114）、min-sliced 计划（2511.19741）；唯一护城河是隐式图，但隐式图上「一族」怎么定义还没人说清 |
| ③ 熵正则 GFN–OT / 图上 Schrödinger 桥 | 不做（作①的 \(\varepsilon>0\) 推广） | GSBoG 占了同象限主会位；Sampling Decisions（2503.14549）已把 GFN 流函数写成单侧 Schrödinger 传输；离散 SB 有了解析解基准（2509.23348）与收敛率（2607.19176）。新进入者没有定义问题的红利 |
| ④ GFN proposal + 经典 OT 修正 | 改形态 | 模糊的「修正」已被 2601.20203 具体化为「对偶预测 → \(\varepsilon\)-relaxation，预测误差映射到运行时间」。可行形态：把 GFlowNet 学出的状态流当对偶预测喂给经典求解器，在显式图上报带保证的加速比——但这是①的一个应用，不是独立课题 |

**本文判断：**如果只做一件事，做①；②③④都应作为①的实验设计里的基线或推广出现，而不是独立的论文。

#### 5b. 每个课题的失败模式与最小可发表单元

评级讨论是否开展课题；本节进一步列出主要失败模式与最低成文目标。

| 课题 | 最可能的失败模式 | 早期预警信号 | 最小可发表单元（MPU） |
|---|---|---|---|
| ① 残差 → OT 误差界 + 对偶证书 | **界太松，没有指导意义**：命题 D 的 \(D_{\max}(1-e^{-2c})\) 项在大图上远大于实际 gap，审稿人问「这个界什么时候非平凡」 | 三日计划第 2 天的残差–gap 曲线若斜率远小于界的斜率（界比实际松两个数量级以上） | 命题 A + C + D 的证明 + hypergrid/\(S_8\) 上「界压住实际 gap」的图 + \(S_{20}\) 上只报证书。Workshop 长度。主会需要 A′ 或紧化 D |
| ② 条件 GFN 摊销图上 OT | **跨实例泛化被 UNOT/ULOT 压住**：在显式图上它们更快更准；隐式图上「一族实例」无定义 | 任何需要把代价矩阵实例化才能训练的设计 | 无诚实的 MPU；只能作①实验里的对照行：「同族 \((L,R)\) 上条件 GFN 的 gap vs 单实例 GFN 的 gap」 |
| ③ 熵正则 GFN–OT / 图上 SB | **被要求在 N06 基准上比数**，而 DLightSB 在那里全设定最强；GSBoG 的叙事已占位 | 论文里出现「我们首次把 SB 与 GFN 联系起来」——Sampling Decisions（2503.14549）已经写过 | 作①的 \(\varepsilon>0\) 推广一节：证明熵正则版本的证书是 Sinkhorn 势的 gap，并在 N06 基准上给一组数。不单独成文 |
| ④ GFN proposal + 经典修正 | **只在显式图上成立，而显式图上 GFN 本身多余**：N01 的 ε-relaxation 需要枚举节点 | 实验规模停在 \(n\le10^4\) 的显式图 | 作①的应用一节：把学到的状态流当对偶预测喂 ε-relaxation，报 \(\|\hat p-p^\star\|_\infty\) 与加速比（N01 Thm. 2 的格式）。不单独成文 |

**读法。** 在当前人员、时间与证据条件下，四个课题里只有①有独立的 MPU；②③④的最好归宿都是①的某一节。这是优先级建议，不是由证据推出的结论：竞品已发表不排除②③④仍有新贡献，而①能否成文取决于 §6b 修正版列出的三件未解决的事（原始修复记账、隐式图全局证书、绕路项）先通过「数学正确、超出经典原始–对偶证书、隐式图可计算、非平凡」四项检查。

### 6. 一个可立即执行的决定性实验

目标：验证「balance 残差可以认证 OT 误差」，同时给出 GFN–OT 相对经典与神经 OT 的诚实定位。全部环境与基线都有现成代码或精确解。

**环境**（三档，由可枚举到隐式）：

1. Hypergrid \(\{0,\dots,H-1\}^D\)，\(H\in\{10,15,20\}\)，\(D=2\)，允许 \(\pm1\) 转移（有环），源分布 \(L\) 取 O08 的 Ball/Moon，目标 \(R\) 取角落多模态并归一化——精确 LP 解可由 `scipy.linprog` 得到（O08 Figure 1 的做法），\(\mathrm{OT}^\star\) 可由 network simplex 得到。
2. 对称群 \(S_n\) 的 Cayley 图（相邻对换），\(n\in\{4,8\}\)（有精确解），\(n=20\)（无精确解，只能报证书）。
3. 一个带权变体：给 hypergrid 的边随机赋权 \(\{1,2\}\)，检验 O08 单位边长假设之外证书是否仍成立。

**方法组**：min-flow GFN（O08 的正则化 TB，Eq. (20)；或 T36 代码的 DB + 状态流正则，两者都跑，\(\lambda\in\{10^{-1},10^{-2},10^{-3}\}\)）；同一模型换 PPO 训练器（2606.15793）；同一模型换 \(f\)-TB 损失（2605.15417）。

**基线**：network simplex（精确）；Sinkhorn（\(\varepsilon\in\{0.1,0.01\}\)，含 2509.23348 的解析解校验）；UNOT 移植版（把最短路矩阵当代价，势预测器 + 一步 Sinkhorn，仅显式档）；GSBoG 复现（若代码仍不可得，用 α-DSBM 的离散化作替身并注明）。

**指标**（每档、每方法、5 种子）：

- 边缘误差：终止分布 TV（可枚举档用精确枚举；隐式档只能用 N03 Thm. 3.6 式的概率证书，或标为「未认证」）；T36 的不动点数分布 \(C(k)L^1\) 单独命名为粗粒化诊断——它是完整 TV 的下界方向，不能反过来上界完整 TV；源边缘违反量 \(\lVert \hat L-L\rVert_1\)（硬约束下总体为零，报经验偏差）。
- 传输代价：分三个量报——端点耦合代价 \(\sum_{u,x}d(u,x)\Pi^\theta_{u,x}\)、实际运输段代价 \(\mathbb E\lvert\tau\rvert\)、绕路项二者之差；各自与 \(\mathrm{OT}^\star\) 比。不报 \(\lVert\Pi_\theta-\Pi^\star\rVert_1\)：零温最优耦合可能不唯一，到某一个参考最优解的距离不是有效的次优性指标。
- **证书**：由学到的状态流构造对偶可行 \(\pi\)（对偶势），报 primal–dual gap；再报每条轨迹的 balance 残差 \(\sum_t\log\)-ratio 的均值与最大值。核心图：\(x\) 轴残差、\(y\) 轴 cost gap，看是否落在一条可证明的界之下。
- 成本：wall-clock、内存、能量函数调用次数。

**预期结果**：显式小图上用精确求解器校验目标值与约束满足（资源成本排名分开报，不作为正确性判据）；隐式档上的目标是构造可计算且非平凡的全局证书，并与常数势、解析可行势比较——这是待验证的研究目标，不是预设结论；证书应覆盖真实次优性并另报松紧度。

**失败判据与负对照**：必须先加入两个负对照——精确 balance 但非最优的耦合（把质量放在交叉配对上，见 §6b 反例 1）、相同耦合但带绕路（§6b 反例 2）——检查证书是否如预期把它们判为次优；若证书对负对照失效，构造有误。残差与 gap 的相关系数只作经验诊断，不是证书成立的必要条件。带权变体（边权 \(\{1,2\}\)）须同步使用加权目标、加权最短路代价与以边权为右端的对偶约束；失败只说明该扩展的证明或实现未通过，不能推出单位边长假设是本质的。零权环需另设分析。

### 6b. 课题①的定理草稿：从 balance 残差到 OT 证书

下面把「残差 → OT 误差界」拆成四个可独立证明的命题。每条标明依赖的已知结果、目前是否成立、以及还缺什么。记号沿用 O08：有限有向图 \(G\)，源集 \(U\)、终止集 \(\mathcal X\)，源分布 \(L\)、归一化目标 \(R\)，内部边集 \(E^\circ\)，最短路距离 \(d(u,x)\)。学到的模型给出前向策略 \(P_F^\theta\)、由它诱导的边流 \(\mathcal F^\theta\) 与端点耦合 \(\Pi^\theta_{u,x}=\sum_{\tau:u\rightsquigarrow x}\mathbb P^\theta(\tau)\)。

**命题 A（残差 → 端点耦合的边缘误差）。** 若对所有轨迹 \(\mathcal L_{TB}(\tau)\le c^2\)，则 \(\|\Pi^\theta_{\cdot,x}\text{ 的第二边缘}-R\|_{TV}\le1-e^{-2c}\)。
依赖：N03 Thm. 3.5（轨迹级分支）逐字成立——终止边缘就是 GFlowNet 的 \(P_T\)。状态：**已成立**。抽样版本用 N03 Thm. 3.6，代价是置信项 \(\log(1/\alpha)(1/m+1/n)\)。
注意：N03 的 DB/FM 分支依赖最大轨迹长度 \(L\)，O08 的图有环、\(L=\infty\)，**不能用**；O08 的训练目标 Eq. (20) 是**带源分布项的正则化 TB**（分子用 \(L(s_1)\) 顶替 \(Z\)，加流正则 \(\lambda R(s)/P_F(s_f\mid s)\)），所以 N03 的轨迹级分支在形式上对得上；T36 公开代码里的 DB + 状态流正则是另一套实现，用它时才需要命题 A′。此外认证对象应是 TB 的平方残差，不含流正则项。

**命题 A′（非无环 DB 残差 → 边缘误差，用期望长度替代最大长度）。** 若逐边 \(\mathcal L_{DB}(s\to s')\le c^2\)，且 \(\mathbb E_{\mathbb P^\theta}[n_\tau]\le\bar n\)，则 \(\mathrm{TV}\le1-e^{-2\bar n c}\) 在期望意义下成立（对随机长度的轨迹按 Jensen 处理）。
依赖：T36 恒等式 \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) 给出 \(\bar n\) 的可计算上界（总流除以 \(Z\)）。状态：**待证**；难点是 N03 的证明按最长轨迹取一致界，换成期望需要控制长尾，Assumption 3.1（\(\mathbb E[n_\tau]<\infty\)）恰好是所需前提。

**命题 B（源边缘的违反量）。** O08 把第一步边流固定为 \(L\)。若实现上是软约束（罚项系数 \(\lambda_L\)），则 \(\|\hat L-L\|_1\) 是训练日志里可直接读出的量，不需要证明；若是硬约束（第一步直接从 \(L\) 采样，O08 的实现），则 \(\hat L=L\) 恒成立。状态：**平凡**，但必须在论文里写明用的是哪一种。

**命题 C（近似可行流 → cost gap，对偶证书）。** 令 \(\hat\pi\) 为任意对偶可行势（\(\hat\pi_{s_0}=0\)，\(\hat\pi_{s'}-\hat\pi_s\le1\) 对所有内部边）。对任何精确可行的流 \(\mathcal F\)（\(\operatorname{div}\mathcal F=L-R\)），LP 弱对偶给出
\[\sum_{e\in E^\circ}\mathcal F(e)\;-\;\mathrm{OT}^\star\;\le\;\sum_{e\in E^\circ}\mathcal F(e)\;-\;\Big(\sum_xR(x)\hat\pi_x-\sum_uL(u)\hat\pi_u\Big)\;=:\;\mathrm{gap}(\mathcal F,\hat\pi).\]
右端只含可观测量。状态：**已成立**（就是 O08 Thm. 3.3 与 Prop. A.4 的对偶加弱对偶）。缺的是两件事：(i) 学到的 \(\mathcal F^\theta\) 只**近似**可行，需把 divergence 残差 \(\|\operatorname{div}\mathcal F^\theta-(L-R)\|_1\) 折进 gap——这一项由命题 A/A′ 控制；(ii) 怎么从 \(\theta\) 构造一个**对偶可行**的 \(\hat\pi\)：取 \(\hat\pi_s=\) 学到的状态流对数的某个仿射变换后，再做一次「1-Lipschitz 投影」（对每条边把 \(\hat\pi_{s'}\) 截到 \(\le\hat\pi_s+1\)，即一次 Bellman–Ford 式松弛），投影后可行性有保证，代价是要遍历边——显式图可做，隐式图只能在采样到的子图上做，得到的是子图证书（对应 N03 Cor. 3.7 的子图版本）。

**命题 D（把 A 与 C 合起来）——目前不成立，原因是两条被混用的量。** 草稿最初写成「A + C 的机械组合」，经外部审稿（Codex, gpt-6-astra）指出两个反例后撤回：

- **反例 1（零残差 ≠ OT 最优）。** 两个等质量源 \(u_1,u_2\)、两个目标 \(x_1,x_2\)，同编号配对距离 1、交叉配对距离 \(K\)。把质量全部放在交叉配对上并沿其最短路运输：边缘精确、DB/TB 残差为零、流守恒精确成立，但代价 \(\approx K\) 而最优为 1。**balance 残差控制的是「是不是一个合法的 GFlowNet」，从来不控制「是不是最小总流的那个」**；最优性只能靠对偶证书（命题 C 的 gap）单独认证。所以 §6 原来的「残差—gap 曲线单调」不是证书成立的必要条件，只是经验诊断。
- **反例 2（端点代价 ≠ 路径代价）。** 一个耦合可以用任意长的绕路或循环实现并仍精确满足 balance：\(\sum_e\mathcal F^\theta(e)=\mathbb E|\tau|\ge\sum_{u,x}d(u,x)\Pi^\theta_{u,x}\)，等号仅当所有轨迹走最短路。认证「端点耦合的 Kantorovich 代价」不等于认证「策略执行的运输代价」；O08 Table 1 报的 \(\mathbb E|\tau|\) 是后者。若论文主张策略执行效率，必须直接认证访问流的加权成本或单独控制绕路项 \(\mathbb E[|\tau|-d(u,x)]\)。

**修正后的命题 D。** 设源边缘精确为 \(L\)（硬约束实现），\(\delta:=\mathrm{TV}(P_T,R)\) 由命题 A 控制。(i) 存在 \(\Pi'\in\Gamma(L,R)\) 使 \(\|\Pi^\theta-\Pi'\|_1\le2\delta\)（把多出的终止质量沿任意路由挪到缺失的目标上；因子 2 来自 TV 与 \(L_1\) 的换算，原稿漏掉了）；(ii) 把 \(\Pi'\) 沿最短路路由成精确可行流 \(\tilde{\mathcal F}\)，则 \(\text{cost}(\tilde{\mathcal F})\le\text{cost}(\Pi^\theta)+2\delta\,D_{\max}\)，其中 \(D_{\max}=\max_{u,x}d(u,x)\) 是**最大源—终点最短路代价**（O08 全对可达假设下有限；它不是通常意义的图直径）；(iii) 对 \(\tilde{\mathcal F}\) 与任一对偶可行 \(\hat\pi\) 用命题 C，得
\[\text{cost}(\Pi^\theta)-\mathrm{OT}^\star\;\le\;\mathrm{gap}(\tilde{\mathcal F},\hat\pi)\;+\;2\delta\,D_{\max}.\]
这条**只认证端点耦合的代价**；执行代价还要加绕路项。用互补松弛把 \(D_{\max}\) 换成更紧的量目前**没有证明**——精确最优流只走紧边，不代表近似流的误差局限在紧边附近，修复还可能要到当前支撑之外的终点。

**对偶可行势的构造（命题 C 的第 (ii) 条，修正）。** 原稿写「一次 Bellman–Ford 式松弛」是错的：一次逐边扫描不保证所有差分约束同时成立（反例：\(a\to b\to c\)，初始势 \((0,100,100)\)，先查 \(b\to c\) 再修 \(a\to b\) 得 \((0,1,100)\)，\(b\to c\) 随即违反）。正确做法是解差分约束系统到闭包（Bellman–Ford 跑到收敛，\(O(|V||E|)\)），结束后检查全部边的最大违反量并报数值容差；这是**可行化**而不是任何范数意义下的投影。隐式图上只能在采样到的子图上做，但**子图上的可行势在原图上可能不可行**（漏掉的边可能是捷径），得到的不是全图对偶证书——这与 N03 Cor. 3.7 的「条件于终止子集的 TV 证书」是两种不同的限制操作，原稿把两者等同是错的。**隐式图上如何得到非平凡的全局对偶证书，目前没有答案**；常数势给出的平凡下界是任何方法都能拿到的基线。

**这份草稿说明了什么（修正版）。** A 在补齐前提后可迁移（要求 TB 形式、双向完整轨迹律归一化、源边与终止边都计入残差）；A′ 在固定源汇流为 1 的设定下有简短证明（审稿人给出的路线：\(\mathrm{TV}(P_T,R)\le\mathrm{TV}(P,Q)\le\mathbb E_P[1-e^{-cN}]\le1-e^{-c\,\mathbb E_PN}\)，\(N\) 为计入残差的边数），但若 \(Z\) 可学习则有反例；B 的总体分布版本成立，经验频率另报；C 对精确可行流成立；D 只认证端点代价且有 \(2\delta D_{\max}\) 的松项。真正有数学内容、且决定这条线能否成文的是三件事：**近似流的原始修复与残差记账**（把 \(\operatorname{div}\mathcal F^\theta-(L-R)\) 折进 gap）、**隐式图上的全局对偶证书**、以及**绕路项的控制**。本文判断：这比 GSBoG「本仓库核查版本中未见任一种误差保证」的现状是明确的差异化方向，但「A + C + D 全部成立、机械组合即成文」的原判断撤回；发表层级的判断也降为「当前人员、时间、证据条件下的优先级建议」，而非由证据推出的结论。

### 6c. 三日开工计划

不写代码之前先把三件事钉死，每件半天到一天：

| 日 | 做什么 | 产出 | 判据 |
|---|---|---|---|
| 1 | 在 hypergrid \(H=10\)、\(D=2\) 上用 `scipy.linprog` 解 O08 Eq. (11) 的精确 LP，同时用 network simplex 解 Kantorovich，验证两者代价相等（复现 O08 Figure 1 的 4.351）；再对精确解构造对偶 \(\pi^\star\)，验证互补松弛逐边成立 | 一个 200 行的 `reference_lp.py`，输出 \((\mathcal F^\star,\pi^\star,\mathrm{OT}^\star)\) | 三个数字一致到 \(10^{-6}\)；任何一条不成立说明对 O08 设定的理解有误 |
| 2 | 按 O08 Eq. (20) 的正则化 TB 训练 min-flow GFN（基于 T36 公开代码，\(\lambda\in\{10^{-1},10^{-2},10^{-3}\}\)），每 500 步导出 \(\mathcal F^\theta\)，按命题 C 构造 \(\hat\pi\)（Bellman–Ford 投影）并算 gap；同时算逐边 DB 残差的均值、最大值与 TB 残差（对采样轨迹） | 一张图：\(x\) = 残差，\(y\) = gap，三条 \(\lambda\) 曲线 | 曲线单调且 gap 随残差 → 0；若 gap 不随残差下降，说明流正则把流压到了非最优顶点，课题①的叙事要改 |
| 3 | 把 N06 的基准构造配方改成零温版——**不能只随机生成势**（常数势下无边饱和、只得 \(R=L\) 的退化实例）：须联合构造可行势 \(\pi\)、紧边路径与非负运输流，保证每个正源质量沿紧边到达允许终点，再定义 \(R\) 并检查非退化，得到一族**有解析最优流的 GFN–OT 实例**；在其上重跑第 2 天的流程 | `synthetic_bench.py` + 10 个实例 + 一个短证明（构造的流确为最优） | 分别报真实次优性与证书上界；gap 接近零时用绝对误差；覆盖率（证书 ≥ 真值）应为 100%，松紧度另报。零温构造是本文需要证明的结果，不能从 N06 的有限温度构造直接继承 |

三天之后的分岔：若第 2 天曲线单调，走 A + C + D 写 workshop 版并同时攻 A′；若不单调，先诊断流正则与最优性的冲突（T36 报告 §3.5 讨论过：最小值落在多面体边界，\(\lambda\) 正则只能逼近），再决定是否改训练目标。

### 7. 最容易被误读的八个点

1. **「\(P_B\) 可以自由选择」≠「DB 是空约束」。** 一旦要求 \(\hat P_B(s\mid s')=\hat P_F(s'\mid s)\hat F(s)/\hat F(s')\)，归一化 \(\sum_s\hat P_B=1\) 立刻反推出 flow matching（T02 Eq. (31)）。自由是「在每个状态对父集合归一化」这张流形上的自由，且只在确定性环境成立（Counterexample 50）。
2. **T19「极限流无环」≠「解唯一」。** T19 杀掉的是可无限放大的有向环方向；真正选出一个点的是 T36 Eq. (11) 的线性目标与 O02 Cor. 1 意义下的 \(\ell^2\) tie-breaking。
3. **O08 的 \(\mathrm{GFlow}^\star\) 是约简目标 \(\sum_{E^\circ}\mathcal F\)，不是 \(\sum_{s\in\mathcal I}\mathcal F(s)=\mathbb E[n_\tau]\)，两者差 1**（\(s_0\to u\) 那一步不属于运输段）。Table 1 里 \(\mathbb E\lvert\tau\rvert\) 与 \(\mathrm{OT}^\star\) 直接对齐，说明报告的是运输段长度。
4. **单源 DAG 上「学到 OT 计划」是空话**（O01 Remark 3.2）。第一个非平凡情形是 \(2\times2\)，自由度 \((n-1)(m-1)\)。
5. **O04 的 \(\alpha\) 不是学习率，是耦合空间的插值系数**（Eq. (22)），且论文没有证明 \(\alpha<1\) 更好——Appendix B 说单步代价恒定时 \(\alpha=1\) 最优。
6. **O06 的 "flow neural network" 是连续正规化流的速度场，不是 GFlowNet。**
7. **T02 本地 PDF 是 arXiv v5（76 页）扩写版，命题编号可能与 JMLR 2023 发表版（24(210):1–55）不一致**；T03 引用 Foundations 用的是早期编号（Prop. 3 / Cor. 1 / Prop. 6 / Prop. 10）。
8. **O07/O08 是 ICML 2026 SPIGM Workshop，不是主会；GSBoG 是 ICML 2026 主会。** 任何把两者写成同一档次的表述都是错的；本方在「叙事占位」上处于劣势，这是选题时必须正视的事实。

### 8. 开放问题

1. **残差 → cost gap 的界长什么样？** TB 残差 → TV 的界已有（N03 Thm. 3.5/3.6，注意其 DB/FM 分支依赖最大轨迹长度 \(L\)，在 O08 的非无环设定下失效，只有 TB 分支可迁移）；OT cost 是关于耦合的线性泛函，边缘违反量与 cost gap 的关系应比 TV 界更紧。来源：O08 §7.2。
2. **带权图。** O08 只覆盖单位边长；带权时零权边会造零代价环，最小流原理是否仍选出无环解？来源：O07 报告 §7、O08 §5 第 5 条。
3. **神经训练与 LP 最优之间的差距。** O08 用软罚 \(\lambda\) 逼近约束，Table 2 显示 \(\lambda\) 的定量权衡，但无理论。来源：O08 §5 第 7 条。
4. **unbalanced 与未知 \(Z\)。** 放开 \(\sum L=\sum R\) 后 GFlowNet「只需未归一化奖励」的能力能否回来？Orlicz–Sobolev 图上不平衡传输（TRENDS_OT §2.1）是候选目标。来源：O08 §7.6。
5. **连续状态空间的最小总流对应什么泛函？** Berner 等给出离散↔连续等价工具（TRENDS_GFN §2.3），Ergodic Generative Flows 给出连续非无环框架，但无人写出连续版 Thm. 3.2。
6. **隐式正则化与最小流是否一致？** 不加流正则时训练收敛到哪个内部流（Secrets of GFlowNets' Learning Behavior，2505.02035；Fisher 几何，2608.03967）。
7. **对称性诱导的路径重数与内部流自由度的关系。** Symmetry-Aware GFlowNets（2506.02685）与 T02 的 \(P_B\) 自由度是同一现象的两面，尚无 OT 角度的分析。
8. **短流是否有利于泛化？** O08 §7.7 指出尚未验证，且可能为负：最短路策略把质量集中在少数路径上，与 GFlowNet 多样性卖点相反。
9. **最小流与熵选择原理的关系。** 距离代价下 OT 计划不唯一时，小正则极限选出的计划（TRENDS_OT §2.2）与最小总流选出的计划是否相同？
10. **多目标组合。** Routing by Reaching（2602.21565）在推断时组合预训练 GFlowNet；OT 语言里对应多边缘传输，尚无对应定理。

---

*本文件由 awesome_Gflow_OT 项目组根据 18 篇解读报告与两份趋势扫描综合写成；每条判断的证据指针见括号内的报告编号与原文定理号。*


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

本报告的综合部分（INSIGHTS、摘要与第 1 章、六篇 2026 增补解读）经本地 Codex CLI（模型 `gpt-6-astra`，推理档位 max，只读沙箱）以「ICML 领域主席级读者」提示词逐份审稿（脚本 `scripts/codex_review.sh`，结构化 JSON 输出存于 `reviews/codex/`）。四份审稿共提出 78 条事实风险、35 条逻辑缺口。已按审稿修正的实质性问题包括：O08 Eq. (20) 的训练目标是带源分布项的正则化 TB（原误写为 DB + 流正则）；T36 恒等式的编号是 Prop. 3.12 / Eq. (10)（原误写 Prop. 3.6）；N03 Thm. 3.6 中 \(c\) 是最大损失的平方根；N02 Eq. (19) 第二项用目标分布 \(\pi\) 而非核的平稳分布；O08 Thm. 3.3 只针对去掉首步约束的单源问题，固定双边缘须用 Appendix A.4；N01 原文确与 network simplex 比过且组均值快 12 倍；N02/N05/N06 的代码链接原文均有给出；INSIGHTS §6b 命题 D「机械组合即成立」的判断撤回（零残差不蕴含 OT 最优、端点代价不等于路径代价两个反例）；O07 的 Workshop 状态降为待核实。凡审稿指出而本报告无法独立核验的条目，均在正文改为限定表述或标注「待核实」。

# 附录 A 论文清单与中译 QA

| ?? | ?? | ?? | ?? | ???? | ?? QA |
|---|---|---|---|---|---|
| O08 | Your GFlowNet Secretly Learns an Optimal Transport Plan | ICML 2026 SPIGM Workshop | workshop | `O08_2606.06272.md` | 3 issues |
| O07 | Learning Shortest Paths with Generative Flow Networks | ICML 2026 SPIGM Workshop (unverified: arXiv comment empty, no header in PDF; label inherited from prior survey) | workshop | `O07_2603.01786.md` | 0 issues |
| T19 | A Theory of Non-Acyclic Generative Flow Networks | AAAI 2024 (Proceedings of the AAAI Conference on Artificial Intelligence 38(10): 11124-11131) | main | `T19_2312.15246.md` | 35 issues |
| T36 | Revisiting Non-Acyclic GFlowNets in Discrete Environments | ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267) | main | `T36_2502.07735.md` | 8 issues |
| T00 | Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation | NeurIPS 2021 (Advances in Neural Information Processing Systems 34) | main | `T00_2106.04399.md` | 7 issues |
| T02 | GFlowNet Foundations | JMLR 24(210):1-55, 2023 | journal | `T02_2111.09266.md` | 13 issues |
| T03 | Trajectory Balance: Improved Credit Assignment in GFlowNets | NeurIPS 2022 (Advances in Neural Information Processing Systems 35) | main | `T03_2201.13259.md` | 1 issues |
| T05 | Learning GFlowNets from Partial Episodes for Improved Convergence and Stability | ICML 2023 | main | `T05_2209.12782.md` | 2 issues |
| T10 | Towards Understanding and Improving GFlowNet Training | ICML 2023 | main | `T10_2305.07170.md` | 3 issues |
| O01 | Optimal Transport for Machine Learners | lecture notes (arXiv preprint, v3 2026-08-08) | lecture-notes | `O01_2505.06589.md` | 69 issues |
| O02 | Quadratically Regularized Optimal Transport on Graphs | SIAM Journal on Scientific Computing 40(4): A1961-A1986, 2018 | journal | `O02_1704.08200.md` | 3 issues |
| O03 | GeONet: a neural operator for learning the Wasserstein geodesic | UAI 2024 | main | `O03_2209.14440.md` | 2 issues |
| O04 | Schrodinger Bridge Flow for Unpaired Data Translation | NeurIPS 2024 (Spotlight) | main | `O04_2409.09347.md` | 13 issues |
| O05 | Universal Neural Optimal Transport | ICML 2025 | main | `O05_2212.00133.md` | 7 issues |
| O06 | Computing high-dimensional optimal transport by flow neural networks | AISTATS 2025 | main | `O06_2305.11857.md` | 0 issues |
| C01 | Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs | NeurIPS 2025 (Main Conference Track) | main | `C01_2506.12025.md` | 0 issues |
| C02 | Generalized Schrodinger Bridge on Graphs | ICML 2026 (PMLR 306); dblp as of 2026-09 lists CoRR entry only | main | `C02_2602.04675.md` | 3 issues |
| C03 | Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation | ICLR 2025 (Poster) | main | `C03_2410.01500.md` | 10 issues |

# 附录 B 符号与术语

| 符号 / 术语 | 含义 | 首见 |
|---|---|---|
| \(G=(\mathcal S,E)\)、\(s_0\)、\(s_f\) | 状态图、源、汇（吸收态） | T02 |
| \(\mathcal X\)、\(U\) | 终止状态集、O08 中的源状态集 | T02 / O08 |
| \(F(s\to s')\)、\(F(s)\) | 边流、状态流（有环时 = 期望访问次数 × 终止流） | T02 / T36 |
| \(P_F\)、\(P_B\) | 前向 / 后向策略，\(F(s\to s')=F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\) | T02 |
| \(R(x)\)、\(Z\) | 奖励、配分函数 \(Z=F(s_0)=\sum_xR(x)\) | T00 |
| FM / DB / TB / SubTB(\(\lambda\)) | 流匹配 / 细致平衡 / 轨迹平衡 / 子轨迹平衡 训练目标 | T00 / T02 / T03 / T05 |
| 0-flow、\(H^1_+(G)\) | 终止流为零的守恒流；环空间 | T19 |
| flow explosion | 比值型损失把流无限堆进环 | T19 Thm. 3 |
| \(n_\tau\)、\(\mathbb E[n_\tau]\) | 轨迹步数及其期望；\(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) | T36 |
| 最小流（minimum flow）GFlowNet | 在奖励匹配约束下最小化总流的 GFlowNet | T36 Eq. (11) |
| \(L\)、\(d_G(u,x)\) | 源分布；图最短路（跳数）距离 | O08 |
| \(\Gamma(L,R)\)、\(\Pi\) | 耦合集、传输计划 | O01 |
| Kantorovich 问题 / 对偶势 | \(\min_\Pi\sum c\,\Pi\)；顶点标量 \(\pi\) 满足 \(\pi_{s'}-\pi_s\le1\) | O01 / O08 Thm. 3.3 |
| Beckmann 问题 / 最小费用流 | 以边流为变量、顶点守恒为约束的 OT 等价形式 | O02 / O01 Prop. 6.23 |
| 互补松弛 | 最优流只在紧边上为正：\(\mathcal F(s\to s')(\pi_{s'}-1-\pi_s)=0\) | O08 Thm. 3.3 |
| Schrödinger 桥（SB）、IPF / IMF | 熵正则动态传输；迭代比例拟合 / 迭代 Markov 拟合 | O04 / C02 / C03 |
| 熵正则 \(\varepsilon\)、Sinkhorn | \(\varepsilon\mathrm{KL}(\Pi\|\alpha\otimes\beta)\) 正则与其交替缩放算法 | O01 |
| unbalanced OT | 两端总质量不等的传输，边缘以罚项代替硬约束 | C01 |
| TV | 全变差距离，本仓库评估终止分布保真度的金标准 | T10 / O08 |

# 参考文献

- **[O08]** Ian Maksimov, Nikita Morozov, Denis Belomestny, Sergey Samsonov. *Your GFlowNet Secretly Learns an Optimal Transport Plan*. ICML 2026 SPIGM Workshop, 2026. arXiv:[2606.06272](https://arxiv.org/abs/2606.06272).
- **[O07]** Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov. *Learning Shortest Paths with Generative Flow Networks*. ICML 2026 SPIGM Workshop (unverified: arXiv comment empty, no header in PDF; label inherited from prior survey), 2026. arXiv:[2603.01786](https://arxiv.org/abs/2603.01786).
- **[T19]** Leo Maxime Brunswic, Yinchuan Li, Yushun Xu, Shangling Jui, Lizhuang Ma. *A Theory of Non-Acyclic Generative Flow Networks*. AAAI 2024 (Proceedings of the AAAI Conference on Artificial Intelligence 38(10): 11124-11131), 2024. arXiv:[2312.15246](https://arxiv.org/abs/2312.15246).
- **[T36]** Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov. *Revisiting Non-Acyclic GFlowNets in Discrete Environments*. ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267), 2025. arXiv:[2502.07735](https://arxiv.org/abs/2502.07735).
- **[T00]** Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio. *Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation*. NeurIPS 2021 (Advances in Neural Information Processing Systems 34), 2021. arXiv:[2106.04399](https://arxiv.org/abs/2106.04399).
- **[T02]** Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, Emmanuel Bengio. *GFlowNet Foundations*. JMLR 24(210):1-55, 2023, 2023. arXiv:[2111.09266](https://arxiv.org/abs/2111.09266).
- **[T03]** Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio. *Trajectory Balance: Improved Credit Assignment in GFlowNets*. NeurIPS 2022 (Advances in Neural Information Processing Systems 35), 2022. arXiv:[2201.13259](https://arxiv.org/abs/2201.13259).
- **[T05]** Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin. *Learning GFlowNets from Partial Episodes for Improved Convergence and Stability*. ICML 2023, 2023. arXiv:[2209.12782](https://arxiv.org/abs/2209.12782).
- **[T10]** Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani. *Towards Understanding and Improving GFlowNet Training*. ICML 2023, 2023. arXiv:[2305.07170](https://arxiv.org/abs/2305.07170).
- **[O01]** Gabriel Peyre. *Optimal Transport for Machine Learners*. lecture notes (arXiv preprint, v3 2026-08-08), 2025. arXiv:[2505.06589](https://arxiv.org/abs/2505.06589).
- **[O02]** Montacer Essid, Justin Solomon. *Quadratically Regularized Optimal Transport on Graphs*. SIAM Journal on Scientific Computing 40(4): A1961-A1986, 2018, 2018. arXiv:[1704.08200](https://arxiv.org/abs/1704.08200).
- **[O03]** Andrew Gracyk, Xiaohui Chen. *GeONet: a neural operator for learning the Wasserstein geodesic*. UAI 2024, 2024. arXiv:[2209.14440](https://arxiv.org/abs/2209.14440).
- **[O04]** Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, Arnaud Doucet. *Schrodinger Bridge Flow for Unpaired Data Translation*. NeurIPS 2024 (Spotlight), 2024. arXiv:[2409.09347](https://arxiv.org/abs/2409.09347).
- **[O05]** Jonathan Geuter, Gregor Kornhardt, Ingimar Tomasson, Vaios Laschos. *Universal Neural Optimal Transport*. ICML 2025, 2025. arXiv:[2212.00133](https://arxiv.org/abs/2212.00133).
- **[O06]** Chen Xu, Xiuyuan Cheng, Yao Xie. *Computing high-dimensional optimal transport by flow neural networks*. AISTATS 2025, 2025. arXiv:[2305.11857](https://arxiv.org/abs/2305.11857).
- **[C01]** Sonia Mazelet, Remi Flamary, Bertrand Thirion. *Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs*. NeurIPS 2025 (Main Conference Track), 2025. arXiv:[2506.12025](https://arxiv.org/abs/2506.12025).
- **[C02]** Panagiotis Theodoropoulos, Juno Nam, Evangelos Theodorou, Jaemoo Choi. *Generalized Schrodinger Bridge on Graphs*. ICML 2026 (PMLR 306); dblp as of 2026-09 lists CoRR entry only, 2026. arXiv:[2602.04675](https://arxiv.org/abs/2602.04675).
- **[C03]** Jun Hyeong Kim, Seonghwan Kim, Seokhyun Moon, Hyeongwoo Kim, Jeheon Woo, Woo Youn Kim. *Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation*. ICLR 2025 (Poster), 2025. arXiv:[2410.01500](https://arxiv.org/abs/2410.01500).


