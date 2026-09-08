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

## 目录

1. [GFlowNet x 最优传输（核心）](#core)
2. [非无环 GFlowNet 理论（通向 OT 的桥）](#nonacyclic)
3. [GFlowNet 基础与训练目标](#foundations)
4. [最优传输先修与图上 OT](#ot-prereq)
5. [神经·摊销 OT 与 Schrödinger 桥](#neural-ot)
6. [竞品与相邻工作：图上 OT 与 SB](#competitors)
7. [2026 增补（已配解读）](#new-2026)
8. [2026 增补与趋势](#trends-2026)
9. [深度解读与中译 PDF](#deep-dive-reports)
10. [Insight 与开放问题](#insights)
11. [汇总报告与幻灯](#deliverables)
12. [贡献与引用](#contributing)

<a id="core"></a>
### GFlowNet x 最优传输（核心）

1. **Your GFlowNet Secretly Learns an Optimal Transport Plan.** ICML 2026 SPIGM Workshop, 2026. [paper](https://arxiv.org/abs/2606.06272) [解读](reports/O08_2606.06272.md) [中译PDF](papers_zh/2606.06272.zh.pdf) [原文PDF](papers/2606.06272.pdf)

    *Ian Maksimov, Nikita Morozov, Denis Belomestny, Sergey Samsonov* · `P0`

    > 在非无环 GFlowNet 的最小总流问题上固定第一步边流 F(s0->u)=L(u)，目标函数就精确变成以图最短路为 ground cost 的 Kantorovich OT（定理 3.2），最优前向策略给出的不是静态耦合矩阵，而是可在隐式组合图上逐步执行的传输路由策略。

2. **Learning Shortest Paths with Generative Flow Networks.** ICML 2026 SPIGM Workshop (unverified: arXiv comment empty, no header in PDF; label inherited from prior survey), 2026. [paper](https://arxiv.org/abs/2603.01786) [code](https://github.com/GreatDrake/gfn-pathfinding) [解读](reports/O07_2603.01786.md) [中译PDF](papers_zh/2603.01786.zh.pdf) [原文PDF](papers/2603.01786.pdf)

    *Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov* · `P0`

    > 非无环 GFlowNet 的期望轨迹长度取最小值，当且仅当所有非最短路轨迹被赋零概率（定理 3.4，充要）；据此把任意无权图的最短路问题归约为训练一个带流正则的非无环 GFlowNet，训完的后向策略即最短路求解器，在魔方上以更小 beam 预算达到 SOTA 可比解长。


<a id="nonacyclic"></a>
### 非无环 GFlowNet 理论（通向 OT 的桥）

1. **A Theory of Non-Acyclic Generative Flow Networks.** AAAI 2024 (Proceedings of the AAAI Conference on Artificial Intelligence 38(10): 11124-11131), 2024. [paper](https://arxiv.org/abs/2312.15246) [解读](reports/T19_2312.15246.md) [中译PDF](papers_zh/2312.15246.zh.pdf) [原文PDF](papers/2312.15246.pdf)

    *Leo Maxime Brunswic, Yinchuan Li, Yushun Xu, Shangling Jui, Lizhuang Ma* · `P0`

    > 用 0-flow 把 GFlowNet 流理论推广到含环的可测空间，证明比值型 FM/DB/TB 损失不稳定（流堆进环里）并给出差值型稳定损失族；附录把 R-流集合刻画为「一个无环流 + 环空间」，并以总流上界期望轨迹长度。

2. **Revisiting Non-Acyclic GFlowNets in Discrete Environments.** ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267), 2025. [paper](https://arxiv.org/abs/2502.07735) [code](https://github.com/GreatDrake/non-acyclic-gfn) [解读](reports/T36_2502.07735.md) [中译PDF](papers_zh/2502.07735.zh.pdf) [原文PDF](papers/2502.07735.pdf)

    *Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov* · `P0`

    > 在有限图上以反向策略为第一性对象重建非无环 GFlowNet 理论：流 = 期望访问次数，正守恒流与 (P_B, F(s_f)) 一一对应，固定 P_B 时解唯一、稳定性无关，最小期望长度等价于最小总流（流多面体上的线性目标）；并把熵正则 RL 等价推广到有环情形。


<a id="foundations"></a>
### GFlowNet 基础与训练目标

1. **Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation.** NeurIPS 2021 (Advances in Neural Information Processing Systems 34), 2021. [paper](https://arxiv.org/abs/2106.04399) [code](https://github.com/bengioe/gflownet) [解读](reports/T00_2106.04399.md) [中译PDF](papers_zh/2106.04399.zh.pdf) [原文PDF](papers/2106.04399.pdf)

    *Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio* · `P0`

    > 提出 GFlowNet：把对象的逐步构造看成 DAG 上的流网络，证明满足局部流守恒且终端流为 R(x) 的流会诱导 pi(x)=R(x)/Z 的策略（Prop. 2），修掉树视角/MaxEnt RL 的 n(x)R(x) 路径数偏置（Prop. 1c），并用对数域的 TD 式目标（Eq. 12）训练，在全支撑下可离线优化（Prop. 3）。

2. **GFlowNet Foundations.** JMLR 24(210):1-55, 2023, 2023. [paper](https://arxiv.org/abs/2111.09266) [解读](reports/T02_2111.09266.md) [中译PDF](papers_zh/2111.09266.zh.pdf) [原文PDF](papers/2111.09266.pdf)

    *Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, Emmanuel Bengio* · `P0`

    > 把 GFlowNet 的流重铸为完整轨迹集合上的测度，于是状态流/边流是可测集的测度、流守恒成为推论（Prop. 8）而非公设；证明 Markovian flow 的刻画（Prop. 16）、三种等价参数化（Prop. 18）、detailed balance 判据（Prop. 21）以及每个边流等价类中 Markovian 代表元的唯一性（Prop. 23）；并引入条件流与状态条件流用于自由能、熵与互信息估计。

3. **Trajectory Balance: Improved Credit Assignment in GFlowNets.** NeurIPS 2022 (Advances in Neural Information Processing Systems 35), 2022. [paper](https://arxiv.org/abs/2201.13259) [code](https://github.com/GFNOrg/gflownet/tree/trajectory_balance) [解读](reports/T03_2201.13259.md) [中译PDF](papers_zh/2201.13259.zh.pdf) [原文PDF](papers/2201.13259.pdf)

    *Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio* · `P0`

    > 用一条完整轨迹上的约束 Z prod P_F = R(x) prod P_B（Eq. 13）取代 FM/DB 的局部约束，证明所有完整轨迹残差为零即可推出 reward matching 与正比采样（Prop. 1），指出任一固定的反向策略都唯一确定一个 reward-matching 流（Sec. 3.1），并报告在长序列与大动作空间上更快的信用传播以及相对 FM 最多 5 倍的运行加速。

4. **Learning GFlowNets from Partial Episodes for Improved Convergence and Stability.** ICML 2023, 2023. [paper](https://arxiv.org/abs/2209.12782) [解读](reports/T05_2209.12782.md) [中译PDF](papers_zh/2209.12782.zh.pdf) [原文PDF](papers/2209.12782.pdf)

    *Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin* · `P1`

    > SubTB(λ) 把 GFlowNet 的平衡约束推广到任意长度的子轨迹，用长度的几何权重 λ 在高偏差低方差的 DB 与低偏差高方差的 TB 之间连续插值，在稀疏奖励与长轨迹环境上显著加速收敛。

5. **Towards Understanding and Improving GFlowNet Training.** ICML 2023, 2023. [paper](https://arxiv.org/abs/2305.07170) [code](https://github.com/maxwshen/gflownet) [解读](reports/T10_2305.07170.md) [中译PDF](papers_zh/2305.07170.zh.pdf) [原文PDF](papers/2305.07170.pdf)

    *Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani* · `P1`

    > 在可枚举的生化基准上，GFlowNet 会系统性欠拟合目标分布、长期超采样低奖励对象；本文把「流的好坏」定义为泛化能力，并用优先回放、相对边流参数化与引导式轨迹平衡来显式控制信用分配。


<a id="ot-prereq"></a>
### 最优传输先修与图上 OT

1. **Optimal Transport for Machine Learners.** lecture notes (arXiv preprint, v3 2026-08-08), 2025. [paper](https://arxiv.org/abs/2505.06589) [code](https://github.com/gpeyre/ot4ml) [解读](reports/O01_2505.06589.md) [中译PDF](papers_zh/2505.06589.zh.pdf) [原文PDF](papers/2505.06589.pdf)

    *Gabriel Peyre* · `P0`

    > Peyré 的 16 章 OT 讲义；对 GFlowNet 读者真正必读的是 §6.5（W1 的图 Beckmann 形式：边流 + 顶点守恒）、第 5 章（对偶与 c-变换）、§14.3（路径空间到端点耦合的约简）、§3.1（耦合多面体）与 §9.1（Sinkhorn = 交替 Bregman 投影），合计约 40 页。

2. **Quadratically Regularized Optimal Transport on Graphs.** SIAM Journal on Scientific Computing 40(4): A1961-A1986, 2018, 2018. [paper](https://arxiv.org/abs/1704.08200) [解读](reports/O02_1704.08200.md) [中译PDF](papers_zh/1704.08200.zh.pdf) [原文PDF](papers/1704.08200.pdf)

    *Montacer Essid, Justin Solomon* · `P1`

    > 把图上 W1 写成边流上的 Beckmann 最小费用网络流（以最短路为费用的 Kantorovich 问题等价于最小费用流），加二次正则并证明小正则解即为 LP 解（稀疏性），并给出以活动子图 Laplacian 为 Hessian 的对偶 Newton 型求解器。


<a id="neural-ot"></a>
### 神经·摊销 OT 与 Schrödinger 桥

1. **GeONet: a neural operator for learning the Wasserstein geodesic.** UAI 2024, 2024. [paper](https://arxiv.org/abs/2209.14440) [code](https://github.com/agracyk2/GeONet) [解读](reports/O03_2209.14440.md) [中译PDF](papers_zh/2209.14440.zh.pdf) [原文PDF](papers/2209.14440.pdf)

    *Andrew Gracyk, Xiaohui Chen* · `P2`

    > 用双分支 DeepONet 直接拟合 Benamou-Brenier 问题的原始-对偶 KKT 方程组（连续性方程 + Hamilton-Jacobi），把「一对边缘分布 → 整条 Wasserstein 测地线」摊销成一次前向传播；训练只需边界分布对、不需要测地线真值，输出网格无关因而支持零样本超分辨率。

2. **Schrodinger Bridge Flow for Unpaired Data Translation.** NeurIPS 2024 (Spotlight), 2024. [paper](https://arxiv.org/abs/2409.09347) [解读](reports/O04_2409.09347.md) [中译PDF](papers_zh/2409.09347.zh.pdf) [原文PDF](papers/2409.09347.pdf)

    *Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, Arnaud Doucet* · `P1`

    > 把 DSBM 的「交替解两个投影子问题」改写成路径测度上的一条流，其唯一不动点即 Schrodinger 桥；用步长 alpha 离散得到 alpha-IMF（alpha=1 退化为原 IMF），并给出对任意 alpha in (0,1] 的收敛定理。参数化实现 alpha-DSBM 只用一个网络、一套损失、在线微调，不再需要缓存样本与交替优化。

3. **Universal Neural Optimal Transport.** ICML 2025, 2025. [paper](https://arxiv.org/abs/2212.00133) [code](https://github.com/GregorKornhardt/UNOT) [解读](reports/O05_2212.00133.md) [中译PDF](papers_zh/2212.00133.zh.pdf) [原文PDF](papers/2212.00133.pdf)

    *Jonathan Geuter, Gregor Kornhardt, Ingimar Tomasson, Vaios Laschos* · `P1`

    > 用 Fourier Neural Operator 预测熵 OT 的对偶势，输入是任意一对离散测度、任意分辨率；训练靠一个可证明覆盖全部非负分布的对抗生成器 + 一个自监督 bootstrapping 损失（目标由 5 步 Sinkhorn 给出，Prop. 5 保证它压住与真解的误差），因此不依赖任何真实数据集。是目前唯一做到跨数据集与跨分辨率泛化的摊销 OT 求解器，也是评估条件 GFlowNet-OT 泛化能力的强基线。

4. **Computing high-dimensional optimal transport by flow neural networks.** AISTATS 2025, 2025. [paper](https://arxiv.org/abs/2305.11857) [code](https://github.com/hamrel-cxu/FlowOT) [解读](reports/O06_2305.11857.md) [中译PDF](papers_zh/2305.11857.zh.pdf) [原文PDF](papers/2305.11857.pdf)

    *Chen Xu, Xiuyuan Cheng, Yao Xie* · `P2`

    > Q-flow 用神经 ODE 直接解两端只有样本的 Benamou-Brenier 动态 OT：两个终端约束都松弛成 KL（由 logistic 分类器现场估计密度比），再加一个有限差分形式的 W2 传输代价项，双向交替训练；学到的 OT 轨道被复用于图像翻译与高维密度比估计。注意：标题里的 flow neural network 指连续正规化流 / 神经 ODE 的速度场，与 GFlowNet 无关。


<a id="competitors"></a>
### 竞品与相邻工作：图上 OT 与 SB

1. **Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs.** NeurIPS 2025 (Main Conference Track), 2025. [paper](https://arxiv.org/abs/2506.12025) [code](https://github.com/smazelet/ULOT) [解读](reports/C01_2506.12025.md) [中译PDF](papers_zh/2506.12025.zh.pdf) [原文PDF](papers/2506.12025.pdf)

    *Sonia Mazelet, Remi Flamary, Bertrand Thirion* · `P1`

    > ULOT 用条件于 FUGW 超参 (alpha, rho) 的交叉注意力 GNN，无监督地摊销预测两张显式图之间的不平衡 OT plan，推理复杂度 O(n1*n2)，比经典求解器快至多 100 倍，并可作为求解器的热启动。

2. **Generalized Schrodinger Bridge on Graphs.** ICML 2026 (PMLR 306); dblp as of 2026-09 lists CoRR entry only, 2026. [paper](https://arxiv.org/abs/2602.04675) [解读](reports/C02_2602.04675.md) [中译PDF](papers_zh/2602.04675.zh.pdf) [原文PDF](papers/2602.04675.pdf)

    *Panagiotis Theodoropoulos, Juno Nam, Evangelos Theodorou, Jaemoo Choi* · `P1`

    > GSBoG 把广义 Schrodinger 桥搬到固定稀疏图上的受控 CTMC，导出离散 Hopf-Cole 系统并给出最优跳转率 u*(y,x) = r(y,x)exp(V(x)-V(y))，用离散 gIPF 损失加一个 TD 项训练，后者防止 running cost 在 IPF 求和中被抵消。

3. **Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation.** ICLR 2025 (Poster), 2025. [paper](https://arxiv.org/abs/2410.01500) [code](https://github.com/junhkim1226/DDSBM) [解读](reports/C03_2410.01500.md) [中译PDF](papers_zh/2410.01500.zh.pdf) [原文PDF](papers/2410.01500.pdf)

    *Jun Hyeong Kim, Seonghwan Kim, Seokhyun Moon, Hyeongwoo Kim, Jeheon Woo, Woo Youn Kim* · `P2`

    > DDSBM 把 Iterative Markovian Fitting 推广到有限状态空间的 CTMC 并证明其单调收敛到 Schrodinger 桥，指出当参考动力学取节点/边独立跳变时对应的熵正则 OT cost 正比于图编辑距离，应用于 ZINC250K 与 Polymer 上的分子优化。


<a id="new-2026"></a>
### 2026 增补（已配解读）

1. **Minimum-Cost Network Flow with Dual Predictions.** AAAI 2026, 2026. [paper](https://arxiv.org/abs/2601.20203) [解读](reports/N01_2601.20203.md) [中译PDF](papers_zh/2601.20203.zh.pdf) [原文PDF](papers/2601.20203.pdf)

    *Zhiyang Chen, Hailong Yao, Xia Yin* · `P1`

    > 第一个用学到的对偶预测热启动的最小费用流算法：ε-relaxation 的时间变为 O(min{n^3 log||p_hat-p*||_inf, n^3 log(nC)})（Theorem 2），预测越准越快、全错也不差于经典；O08 的对偶势可直接充当这里的预测。

2. **Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels.** ICML 2026 SPIGM Workshop, 2026. [paper](https://arxiv.org/abs/2606.16073) [解读](reports/N02_2606.16073.md) [中译PDF](papers_zh/2606.16073.zh.pdf) [原文PDF](papers/2606.16073.pdf)

    *Kirill Korolev, Nikita Morozov, Stepan Pavlenko, Esmeralda S. Whitammer, Sergey Samsonov* · `P1`

    > 把 MCMC 装进连续非无环 GFlowNet：学一个停止分类器 d_F(s) 决定何时终止，detailed balance 把最优分类器钉在目标密度上（Theorem 3.6），总流最小当且仅当期望长度取到闭式 n_Q*（Corollary 3.7）；与 O07/O08 是同一恒等式的第三个出口。

3. **Stable GFlowNets with TV Monitoring and Probabilistic Guarantees.** arXiv preprint, 2026. [paper](https://arxiv.org/abs/2605.01729) [解读](reports/N03_2605.01729.md) [中译PDF](papers_zh/2605.01729.zh.pdf) [原文PDF](papers/2605.01729.pdf)

    *Zengxiang Lei, Ananth Shreekumar, Jonathan Rosenthal, Ruoyu Song, Alvaro A. Cardenas, Daniel J. Fremont, Dongyan Xu, Satish Ukkusuri, Z. Berkay Celik* · `P0`

    > 先证明小 TV 不排除无界损失（Prop. 3.3-3.4），再给出反向证书：逐轨迹 TB 损失 <= c^2 推出 TV <= 1-e^{-2c}（Theorem 3.5），并有与状态空间大小无关的抽样概率版（Theorem 3.6）；这是「残差→OT 误差界」课题已完成的一半。

4. **Generative Modeling on Metric Graphs via Neural Optimal Transport.** arXiv preprint, 2026. [paper](https://arxiv.org/abs/2606.16273) [解读](reports/N04_2606.16273.md) [中译PDF](papers_zh/2606.16273.zh.pdf) [原文PDF](papers/2606.16273.pdf)

    *Alessandro Micheli, Yueqi Cao, Anthea Monod, Samir Bhatt* · `P2`

    > 度量图上连续分布的第一个深度生成模型：嵌入图（欧氏或 tropical Abel-Jacobi）、用神经半对偶解熵正则 OT、投影回图；生成器弱收敛到合法图上耦合（Theorem 4.1）；与 O08 是图上 OT 的连续边/离散顶点两个互补分支。

5. **An Efficient Orlicz-Sobolev Approach for Transporting Unbalanced Measures on a Graph.** NeurIPS 2025 Spotlight, 2025. [paper](https://arxiv.org/abs/2502.00739) [解读](reports/N05_2502.00739.md) [中译PDF](papers_zh/2502.00739.zh.pdf) [原文PDF](papers/2502.00739.pdf)

    *Tam Le, Truyen Nguyen, Hideitsu Hino, Kenji Fukumizu* · `P1`

    > 图上不平衡 OT 的非 KL 路线：把熵部分传输改写为带非负代价的平衡 OT（Prop. 3.1），赋予 Orlicz 几何，再正则化为只需一元优化的 Orlicz-Sobolev 传输（Theorem 4.2），比 Orlicz-EPT 快 250-13800 倍；质量差由一个线性项显式吸收。

6. **Entering the Era of Discrete Diffusion Models: A Benchmark for Schrodinger Bridges and Entropic Optimal Transport.** ICLR 2026, 2026. [paper](https://arxiv.org/abs/2509.23348) [解读](reports/N06_2509.23348.md) [中译PDF](papers_zh/2509.23348.zh.pdf) [原文PDF](papers/2509.23348.pdf)

    *Xavier Aramayo Carrasco, Grigoriy Ksenofontov, Aleksei Leonov, Iaroslav Sergeevich Koshelev* · `P1`

    > 离散空间 EOT/SB 第一个有解析解的基准：任意 (p_0, v*) 诱导出闭式最优耦合的基准对（Theorem 3.1），CP 参数化使它在 S^D 高维可算；副产品求解器 DLightSB 全设定最强；任何熵正则 GFN-OT 都必须在此报数。


<a id="trends-2026"></a>
### 2026 增补与趋势

2026 趋势扫描收录（relevance >= 4）。完整分析见 [reports/TRENDS_GFN_2026.md](reports/TRENDS_GFN_2026.md)、[reports/TRENDS_OT_2026.md](reports/TRENDS_OT_2026.md)。

1. **Learning fMRI activations dictionaries across individual geometries via optimal transport.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2605.20883) [code](unknown)

    *Sonia Mazelet, Remi Flamary, Bertrand Thirion*

    > Learns fMRI activation dictionaries across individual brain geometries by comparing graphs with the Fused Gromov-Wasserstein distance, using an amortized neural network to predict approximate transport plans and atoms that depend on the FGW tradeoff parameter.

2. **Controlling Exploration-Exploitation in GFlowNets via Markov Chain Perspectives.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2602.01749) [code](unknown)

    *Lin Chen, Samuel Drapeau, Fanghao Shao, Xuekai Zhu, Bo Xue, et al.*

    > Shows GFlowNet objectives are equivalent to reversibility of an induced Markov chain and introduces alpha-GFN, which tunes the forward/backward mixing ratio.

3. **Proximal Policy Optimization for Amortized Discrete Sampling.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2606.15793) [code](unknown)

    *Anna Zykova-Myzina, Timofei Gritsaev, Daniil Tiapkin, Nikita Morozov*

    > Derives policy-gradient equivalents for GFlowNet training and is the first to apply proximal policy optimization, reporting faster convergence than standard GFlowNet objectives.

4. **$f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data.** ICML 2026 (per arXiv comment), 2026. [paper](https://arxiv.org/abs/2605.15417) [code](unknown)

    *Jake Fawkes, Jason Hartford*

    > Extends the trajectory-balance surrogate to the whole family of f-divergences, so on-policy gradients match the chosen f-divergence while the off-policy global minimizer is unchanged.

5. **Evaluating GFlowNet from partial episodes for stable and flexible policy-based training.** ICLR 2026 (per arXiv comment), 2026. [paper](https://arxiv.org/abs/2603.01047) [code](unknown)

    *Puhua Niu, Shili Wu, Xiaoning Qian*

    > Shows flow balance also yields a policy evaluator and proposes an evaluation-balance objective over partial episodes that supports parameterized backward policies and offline data.

6. **Information-Geometric Forward Policy Training in GFlowNets.** arXiv preprint (comment has no venue, unverified), 2026. [paper](https://arxiv.org/abs/2608.03967) [code](unknown)

    *Yordan Raykov, Rodrigo Veiga*

    > Treats the forward policy as a trajectory sampler, identifies its intrinsic geometry as the Fisher-Rao metric, and decomposes the trajectory Fisher into per-step conditional second moments.

7. **GFlowRL: Scaling Distribution-Matching RL to Large Language Models.** arXiv preprint (comment has no venue, unverified), 2026. [paper](https://arxiv.org/abs/2607.13394) [code](https://github.com/microsoft/gflowrl)

    *Xiaodong Liu, Michael Xu, Jack W. Stokes, Paul Smolensky, Doug Burger, et al.*

    > Removes the learned partition network from GFlowNet-style LLM RL by using an in-batch Monte Carlo estimate, plus importance-sampling correction and asymmetric flow-gap clipping.

8. **Regularity of Solutions to Beckmann's Parametric Optimal Transport.** arXiv preprint (comment has no venue, unverified), 2026. [paper](https://arxiv.org/abs/2603.19755) [code](unknown)

    *Hanno Gottschalk, Tobias J. Riedlinger*

    > Develops Holder regularity theory for Beckmann's problem through an unconstrained Lagrangian, showing the multiplier enforcing the divergence constraint solves a Poisson equation and the flux is its gradient, with joint parameter regularity for conditional targets.

9. **Statistical Mechanics of the Sub-Optimal Transport.** arXiv preprint (comment has no venue, unverified), 2026. [paper](https://arxiv.org/abs/2602.04308) [code](unknown)

    *Riccardo Piombo, Lorenzo Buffa, Dario Mazzilli, Aurelio Patelli*

    > Mean-field theory for the Sub-Optimal Transport model, an ensemble of weighted bipartite graphs where a coupling parameter interpolates between entropy-dominated dense couplings and cost-dominated sparse ones; the crossover is smooth, not a phase transition.

10. **Exponential Convergence of the Sinkhorn Algorithm for the Schrodinger Bridge with Regime Switching.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2607.19176) [code](unknown)

    *Katharina Eichinger, Anna Kazeykina, Zhenjie Ren, Hecheng Wang*

    > Proves exponential convergence in relative entropy of the Sinkhorn algorithm for the Schrodinger bridge with regime switching on a hybrid state space R^d times a finite set, including a partially observed terminal setting.

11. **Amortized Optimal Transport from Sliced Potentials.** arXiv preprint (comment has no venue, unverified), 2026. [paper](https://arxiv.org/abs/2604.15114) [code](unknown)

    *Minh-Phuc Truong, Khai Nguyen*

    > Predicts OT plans across many measure pairs by amortizing Kantorovich potentials from sliced OT, with a regression-based and an objective-based variant, then recovering the plan from the estimated potentials.

12. **SinkSLOT: Sinkhorn via Sparse Lifted Optimal Transport.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2608.28262) [code](https://github.com/cai4cai/SinkSLOT)

    *Ian Hsieh, Soumya Snigdha Kundu, Tom Vercauteren, Reuben Dorent*

    > SinkSLOT sparsifies the Gibbs kernel using an expected sliced lifted transport plan as a non-independent reference coupling, giving O(LN) per-iteration cost with a convergence proof and no debiasing needed.

13. **Stability of Quadratically Regularized Optimal Transport.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2605.27883) [code](unknown)

    *Alberto Gonzalez-Sanz, Marcel Nutz*

    > Quantitative stability theory for quadratically regularized OT under perturbations of marginals, cost and regularization, centred on an L-infinity stability result for the dual potentials and yielding local Lipschitz stability of the optimal support.

14. **Polyak-Lojasiewicz Inequality for Quadratically Regularized Optimal Transport.** SIAM Journal on Optimization, to appear (per arXiv comment), 2026. [paper](https://arxiv.org/abs/2605.27175) [code](unknown)

    *Alberto Gonzalez-Sanz, Marcel Nutz, Andres Riveros Valdevenito*

    > Establishes a local error bound and a Polyak-Lojasiewicz inequality for the quadratically regularized OT dual with explicit constants, giving linear convergence rates for gradient and coordinate ascent.

15. **Statistical Estimation of Monge Transport Maps via Brenier Potentials.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2604.22366) [code](unknown)

    *Elsa Cazelles, Edouard Pauwels, Leo Portales*

    > Statistical estimator for Monge maps built from the dual solution of the discrete sampled problem, with convergence rates from a new error bound for quadratic OT and sharper rates in the semi-discrete case.

16. **Uniform Statistical Convergence of Empirical Sinkhorn Potentials with Exponential and Polynomial Dependence on the Regularization Parameter.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2608.29152) [code](unknown)

    *Denis Belomestny*

    > Non-asymptotic n^{-1/2} rates for empirical Sinkhorn potentials in the quotient supremum norm, with conditions under which the constant depends polynomially rather than exponentially on 1/epsilon, plus matching minimax lower bounds.

17. **Reinforcement Learning via Value Gradient Flow.** ICLR 2026 (per arXiv comment), 2026. [paper](https://arxiv.org/abs/2604.14265) [code](https://ryanxhr.github.io/vgf)

    *Haoran Xu, Kaiwen Hu, Somayeh Sojoudi, Amy Zhang*

    > Value Gradient Flow casts behavior-regularized RL as an optimal transport problem from the reference distribution to the value-induced optimal policy, solved by discrete gradient flow with regularization controlled by the transport budget.

18. **Global Convergence of Wasserstein Policy Gradient for Entropy-Regularized Reinforcement Learning.** arXiv preprint (comment empty, unverified), 2026. [paper](https://arxiv.org/abs/2605.26078) [code](unknown)

    *Zhaoyu Zhu, Rui Gao, Shuang Li*

    > Proves global convergence of Wasserstein policy gradient for entropy-regularized RL by replacing convexity with a Bellman-based argument that yields a distributional Polyak-Lojasiewicz condition under a uniform log-Sobolev inequality.

19. **Exploration through Generation: Applying GFlowNets to Structured Search.** arXiv preprint (comment has no venue, unverified), 2025. [paper](https://arxiv.org/abs/2510.21886) [code](unknown)

    *Mark Phillip Matovic*

    > Applies trajectory-balance GFlowNets to shortest path, minimum spanning tree and TSP on small benchmark graphs and reports agreement with Dijkstra, Kruskal and exact solvers.

20. **A Theory of Multi-Agent Generative Flow Networks.** Accepted at SPIGM Workshop NeurIPS 2025 (Workshop, per arXiv comment), 2025. [paper](https://arxiv.org/abs/2509.20408) [code](unknown)

    *Leo Maxime Brunswic, Haozhi Wang, Shuang Luo, Jianye Hao, Amir Rasouli, et al.*

    > Builds a theory of multi-agent GFlowNets with centralized, independent, joint and conditional flow networks, proving that local flows can be trained as one global flow.

21. **Ergodic Generative Flows.** ICML 2025 (per arXiv comment), 2025. [paper](https://arxiv.org/abs/2505.03561) [code](unknown)

    *Leo Maxime Brunswic, Mateo Clemente, Rui Heng Yang, Adam Sigal, Amir Rasouli, et al.*

    > Ergodic Generative Flows use finitely many globally defined diffeomorphisms to make flow-matching loss tractable in continuous settings and add a KL-weak-FM loss for imitation learning.

22. **Sampling Decisions: Exact Path-Space Control for Physics-Informed Generative Sampling.** arXiv preprint (comment has no venue, unverified), 2025. [paper](https://arxiv.org/abs/2503.14549) [code](unknown)

    *Michael Chertkov, Hamidreza Behjoo, Sungsoo Ahn*

    > Frames sequential structured generation as an exact path-space control problem, with the corrected law given by a Doob h-transform and identified as a one-sided Schrodinger transport and an ideal GFlowNet flow function.

23. **Relative Trajectory Balance is equivalent to Trust-PCL.** arXiv preprint (comment empty, unverified), 2025. [paper](https://arxiv.org/abs/2509.01632) [code](unknown)

    *Tristan Deleu, Padideh Nouri, Yoshua Bengio, Doina Precup*

    > Proves Relative Trajectory Balance is equivalent to Trust-PCL, an off-policy KL-regularized RL method, and shows KL-regularized RL matches RTB on the paper's illustrative example.

24. **From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training.** TMLR (per arXiv comment/journal_ref), 2025. [paper](https://arxiv.org/abs/2501.06148) [code](https://github.com/GFNOrg/gfn-diffusion/tree/stagger)

    *Julius Berner, Lorenz Richter, Marcin Sendera, Jarrid Rector-Brooks, Nikolay Malkin*

    > Proves equivalences between discrete-time entropic RL objectives (GFlowNets) and continuous-time objects (PDEs, path-space measures) in the small-step limit, and shows coarse time discretization speeds up training.

25. **Unrealized Expectations: Comparing AI Methods vs Classical Algorithms for Maximum Independent Set.** TMLR (per arXiv comment/journal_ref), 2025. [paper](https://arxiv.org/abs/2502.03669) [code](unknown)

    *Yikai Wu, Haoyu Zhao, Sanjeev Arora*

    > Compares AI methods against classical solvers on Maximum Independent Set and finds the classical KaMIS solver on one CPU beats leading GPU-based AI methods, with a serialization analysis showing the GFlowNet-based LTFT reasons like a degree-based greedy.

26. **gfnx: Fast and Scalable Library for Generative Flow Networks in JAX.** arXiv preprint (comment has no venue, unverified), 2025. [paper](https://arxiv.org/abs/2511.16592) [code](https://github.com/d-tiapkin/gfnx)

    *Daniil Tiapkin, Artem Agarkov, Nikita Morozov, Ian Maksimov, Askar Tsyganov, et al.*

    > gfnx is a JAX library with single-file implementations of core GFlowNet objectives plus hypergrid, sequence, molecular, phylogenetic and Ising environments, reporting up to 55x-80x speedups over PyTorch baselines.

27. **A convex approach for Markov chain estimation from aggregate data via inverse optimal transport.** arXiv preprint (comment has no venue, unverified), 2025. [paper](https://arxiv.org/abs/2511.16458) [code](unknown)

    *Michele Mascherpa, Axel Ringh, Amirhossein Taghvaei, Johan Karlsson*

    > Estimates the transition matrix of a discrete-state Markov chain from aggregate distributions at successive times by jointly optimizing over the matrix and entropic transport plans, yielding a convex problem with a proximal algorithm.

28. **Adjoint Schrodinger Bridge Sampler.** NeurIPS 2025 (per arXiv comment), 2025. [paper](https://arxiv.org/abs/2506.22565) [code](https://github.com/facebookresearch/adjoint_samplers)

    *Guan-Horng Liu, Jaemoo Choi, Yongxin Chen, Benjamin Kurt Miller, Ricky T. Q. Chen*

    > Adjoint Schrodinger Bridge Sampler learns to sample from unnormalized energies with a matching-based objective that needs no target samples, generalizing Adjoint Sampling to arbitrary source distributions by dropping the memoryless condition.

29. **Efficient Transferable Optimal Transport via Min-Sliced Transport Plans.** arXiv preprint (comment empty, unverified), 2025. [paper](https://arxiv.org/abs/2511.19741) [code](unknown)

    *Xinran Liu, Elaheh Akbari, Rocio Diaz Martin, Navid NaderiAlizadeh, Soheil Kolouri*

    > Studies whether an optimized slicer in the min-Sliced Transport Plan framework transfers to new distribution pairs, proving stability under perturbations and adding a minibatch formulation with statistical guarantees.

30. **An efficient algorithm for entropic optimal transport under martingale-type constraints.** arXiv preprint (comment empty, unverified), 2025. [paper](https://arxiv.org/abs/2508.17641) [code](unknown)

    *Xun Tang, Michael Shavlovsky, Holakou Rahmanian, Tesi Xiao, Lexing Ying*

    > Entropic OT under martingale-type conditions, noting that these are row-wise equality or inequality constraints on the coupling, solved by Sinkhorn-type algorithms with sparse Newton iterations.

31. **Weighted Conditional Flow Matching.** arXiv preprint (comment has no venue, unverified), 2025. [paper](https://arxiv.org/abs/2507.22270) [code](unknown)

    *Sergio Calvo-Ordonez, Matthieu Meunier, Alvaro Cartea, Christoph Reisinger, Yarin Gal, et al.*

    > Weighted Conditional Flow Matching reweights each training pair with a Gibbs kernel, recovering the entropic OT coupling up to a marginal bias, and is shown equivalent to minibatch OT in the large-batch limit.


<a id="deep-dive-reports"></a>
### 深度解读与中译 PDF

| 编号 | 论文 | 解读报告 | 原文 PDF | 中译 PDF（SuperTranslate） | QA |
|---|---|---|---|---|---|
| O08 | Your GFlowNet Secretly Learns an Optimal Transport Plan | [O08_2606.06272.md](reports/O08_2606.06272.md) | [2606.06272.pdf](papers/2606.06272.pdf) | [2606.06272.zh.pdf](papers_zh/2606.06272.zh.pdf) | 3 issues |
| O07 | Learning Shortest Paths with Generative Flow Networks | [O07_2603.01786.md](reports/O07_2603.01786.md) | [2603.01786.pdf](papers/2603.01786.pdf) | [2603.01786.zh.pdf](papers_zh/2603.01786.zh.pdf) | 0 issues |
| T19 | A Theory of Non-Acyclic Generative Flow Networks | [T19_2312.15246.md](reports/T19_2312.15246.md) | [2312.15246.pdf](papers/2312.15246.pdf) | [2312.15246.zh.pdf](papers_zh/2312.15246.zh.pdf) | 35 issues |
| T36 | Revisiting Non-Acyclic GFlowNets in Discrete Environments | [T36_2502.07735.md](reports/T36_2502.07735.md) | [2502.07735.pdf](papers/2502.07735.pdf) | [2502.07735.zh.pdf](papers_zh/2502.07735.zh.pdf) | 8 issues |
| T00 | Flow Network based Generative Models for Non-Iterative Diverse Candida | [T00_2106.04399.md](reports/T00_2106.04399.md) | [2106.04399.pdf](papers/2106.04399.pdf) | [2106.04399.zh.pdf](papers_zh/2106.04399.zh.pdf) | 7 issues |
| T02 | GFlowNet Foundations | [T02_2111.09266.md](reports/T02_2111.09266.md) | [2111.09266.pdf](papers/2111.09266.pdf) | [2111.09266.zh.pdf](papers_zh/2111.09266.zh.pdf) | 13 issues |
| T03 | Trajectory Balance: Improved Credit Assignment in GFlowNets | [T03_2201.13259.md](reports/T03_2201.13259.md) | [2201.13259.pdf](papers/2201.13259.pdf) | [2201.13259.zh.pdf](papers_zh/2201.13259.zh.pdf) | 1 issues |
| T05 | Learning GFlowNets from Partial Episodes for Improved Convergence and  | [T05_2209.12782.md](reports/T05_2209.12782.md) | [2209.12782.pdf](papers/2209.12782.pdf) | [2209.12782.zh.pdf](papers_zh/2209.12782.zh.pdf) | 2 issues |
| T10 | Towards Understanding and Improving GFlowNet Training | [T10_2305.07170.md](reports/T10_2305.07170.md) | [2305.07170.pdf](papers/2305.07170.pdf) | [2305.07170.zh.pdf](papers_zh/2305.07170.zh.pdf) | 3 issues |
| O01 | Optimal Transport for Machine Learners | [O01_2505.06589.md](reports/O01_2505.06589.md) | [2505.06589.pdf](papers/2505.06589.pdf) | [2505.06589.zh.pdf](papers_zh/2505.06589.zh.pdf) | 69 issues |
| O02 | Quadratically Regularized Optimal Transport on Graphs | [O02_1704.08200.md](reports/O02_1704.08200.md) | [1704.08200.pdf](papers/1704.08200.pdf) | [1704.08200.zh.pdf](papers_zh/1704.08200.zh.pdf) | 3 issues |
| O03 | GeONet: a neural operator for learning the Wasserstein geodesic | [O03_2209.14440.md](reports/O03_2209.14440.md) | [2209.14440.pdf](papers/2209.14440.pdf) | [2209.14440.zh.pdf](papers_zh/2209.14440.zh.pdf) | 2 issues |
| O04 | Schrodinger Bridge Flow for Unpaired Data Translation | [O04_2409.09347.md](reports/O04_2409.09347.md) | [2409.09347.pdf](papers/2409.09347.pdf) | [2409.09347.zh.pdf](papers_zh/2409.09347.zh.pdf) | 13 issues |
| O05 | Universal Neural Optimal Transport | [O05_2212.00133.md](reports/O05_2212.00133.md) | [2212.00133.pdf](papers/2212.00133.pdf) | [2212.00133.zh.pdf](papers_zh/2212.00133.zh.pdf) | 7 issues |
| O06 | Computing high-dimensional optimal transport by flow neural networks | [O06_2305.11857.md](reports/O06_2305.11857.md) | [2305.11857.pdf](papers/2305.11857.pdf) | [2305.11857.zh.pdf](papers_zh/2305.11857.zh.pdf) | 0 issues |
| C01 | Unsupervised Learning for Optimal Transport plan prediction between un | [C01_2506.12025.md](reports/C01_2506.12025.md) | [2506.12025.pdf](papers/2506.12025.pdf) | [2506.12025.zh.pdf](papers_zh/2506.12025.zh.pdf) | 0 issues |
| C02 | Generalized Schrodinger Bridge on Graphs | [C02_2602.04675.md](reports/C02_2602.04675.md) | [2602.04675.pdf](papers/2602.04675.pdf) | [2602.04675.zh.pdf](papers_zh/2602.04675.zh.pdf) | 3 issues |
| C03 | Discrete Diffusion Schrodinger Bridge Matching for Graph Transformatio | [C03_2410.01500.md](reports/C03_2410.01500.md) | [2410.01500.pdf](papers/2410.01500.pdf) | [2410.01500.zh.pdf](papers_zh/2410.01500.zh.pdf) | 10 issues |
| N01 | Minimum-Cost Network Flow with Dual Predictions | [N01_2601.20203.md](reports/N01_2601.20203.md) | [2601.20203.pdf](papers/2601.20203.pdf) | [2601.20203.zh.pdf](papers_zh/2601.20203.zh.pdf) | 7 issues |
| N02 | Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kern | [N02_2606.16073.md](reports/N02_2606.16073.md) | [2606.16073.pdf](papers/2606.16073.pdf) | [2606.16073.zh.pdf](papers_zh/2606.16073.zh.pdf) | 2 issues |
| N03 | Stable GFlowNets with TV Monitoring and Probabilistic Guarantees | [N03_2605.01729.md](reports/N03_2605.01729.md) | [2605.01729.pdf](papers/2605.01729.pdf) | [2605.01729.zh.pdf](papers_zh/2605.01729.zh.pdf) | 6 issues |
| N04 | Generative Modeling on Metric Graphs via Neural Optimal Transport | [N04_2606.16273.md](reports/N04_2606.16273.md) | [2606.16273.pdf](papers/2606.16273.pdf) | [2606.16273.zh.pdf](papers_zh/2606.16273.zh.pdf) | 6 issues |
| N05 | An Efficient Orlicz-Sobolev Approach for Transporting Unbalanced Measu | [N05_2502.00739.md](reports/N05_2502.00739.md) | [2502.00739.pdf](papers/2502.00739.pdf) | [2502.00739.zh.pdf](papers_zh/2502.00739.zh.pdf) | 11 issues |
| N06 | Entering the Era of Discrete Diffusion Models: A Benchmark for Schrodi | [N06_2509.23348.md](reports/N06_2509.23348.md) | [2509.23348.pdf](papers/2509.23348.pdf) | [2509.23348.zh.pdf](papers_zh/2509.23348.zh.pdf) | 5 issues |

<a id="insights"></a>
### Insight 与开放问题

跨论文综合见 [reports/INSIGHTS.md](reports/INSIGHTS.md)，竞品矩阵见 [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md)。核心结论：

1. **内部流非唯一性是入口。** 一般图上，匹配终止分布只固定终止边流；后向策略的自由度（T02 Prop. 18）与有环时的环流锥（T19 Prop. 5）留下一族合法内部流。最小化内部总流从中选出一个，固定源边缘后这一个实现一个最优传输计划（O08 Thm. 3.2）。
2. **等价本身是经典结果，贡献在组合。** 图上最短路代价的 Kantorovich OT 等于最小费用流（Beckmann）。「输出局部策略」也不新（最小费用边流归一化后就是；GSBoG 同样学策略）。新的是这一整套组合：非无环 GFlowNet 最小流目标 + 固定两端边缘 + 神经策略参数化 + 在无法构造代价矩阵的隐式组合图上训练与执行。\(20!\) 排列实验是可执行性示范，该规模无精确 OT 参照。
3. **最不拥挤的后续课题是最优性证书。** Balance 残差能认证终止分布的全变差误差（N03 Thm. 3.5/3.6），但零残差**从不**认证总流最小——一个流可以完全守恒却把质量全部绕远路运。证书因此必须同时处理原始可行性修复、对偶可行势与最优性缺口（O08 App. A.4 的弱对偶）；隐式图上可计算的**全局**证书仍是开放问题。两种失效模式已在 [experiments/](experiments/README.md) 数值复现：完全守恒但交叉配对的流代价高 60–78%，最优耦合经绕路执行后代价升至 3.2–3.5 倍而 balance 残差仍为 \(10^{-15}\)。在本仓库核查的文献内无人为 GFlowNet 做过这件事；声称空白前须补查经典 LP 的后验误差分析。见 `reports/INSIGHTS.md` §6b。
4. **两个后续课题已经拥挤。** 条件 GFlowNet 摊销一族图上的 OT 与 ULOT（NeurIPS 2025）、UNOT（ICML 2025）撞车；熵正则 GFN–OT 与离散/图上 Schrödinger 桥一系（DDSBM，ICLR 2025；GSBoG，ICML 2026 主会，与 O08 同象限但无误差界）撞车。
5. **发表状态的现实。** 两篇 GFN × OT 论文（O07、O08）都是 ICML 2026 SPIGM *Workshop*，三篇竞品全是主会。GFlowNet 剩下的结构性护城河只有一处：无法构造代价矩阵的隐式组合图。

<a id="deliverables"></a>
### 汇总报告与幻灯

| 交付物 | 路径 |
|---|---|
| 汇总报告 · 中文（105 页） | [PDF](reports/pdf/awesome_gflow_ot_report_zh.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_zh.md) |
| 汇总报告 · 英文（142 页） | [PDF](reports/pdf/awesome_gflow_ot_report_en.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_en.md) |
| 汇总幻灯（单文件 HTML，24 页；键盘 / 滚轮 / 触摸） | [slides/awesome_gflow_ot_slides.html](slides/awesome_gflow_ot_slides.html) |
| 汇总幻灯（Beamer PDF，29 页含备份页） | [slides/awesome_gflow_ot_slides.pdf](slides/awesome_gflow_ot_slides.pdf) |
| 跨论文综合 | [reports/INSIGHTS.md](reports/INSIGHTS.md) |
| 竞品矩阵 | [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md) |
| 2026 趋势扫描 | [reports/TRENDS_GFN_2026.md](reports/TRENDS_GFN_2026.md)、[reports/TRENDS_OT_2026.md](reports/TRENDS_OT_2026.md) |
| 第 1 天参考 LP 实验（GFlow\* = OT\*、对偶证书、两个反例） | [experiments/README.md](experiments/README.md) |
| 外部审稿归档（Codex CLI，gpt-6-astra，7 份结构化审稿） | [reviews/codex/](reviews/codex/) |

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
