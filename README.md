# Awesome GFlowNet x Optimal Transport

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Papers](https://img.shields.io/badge/papers-18%20core%20%2B%202026%20additions-orange.svg)](#content)
[![Reports](https://img.shields.io/badge/deep--dive%20reports-18-green.svg)](#deep-dive-reports)
[![Translated](https://img.shields.io/badge/zh%20PDFs-18%2F18-blueviolet.svg)](#deep-dive-reports)
[![Report](https://img.shields.io/badge/report-zh%2091p%20%7C%20en%20120p-informational.svg)](#deliverables)

[English](README.md) | [中文](README_zh.md)

A curated, evidence-first reading list on the intersection of **Generative Flow Networks (GFlowNets)** and **Optimal Transport (OT)**: the theory that a GFlowNet trained to match a reward-induced target distribution, when asked to minimise its total internal edge flow on a (possibly cyclic) state graph, recovers a **Kantorovich optimal transport plan** under the shortest-path cost of that graph.

Every paper here comes with:

- a **deep-dive report** in Chinese (`reports/`), written from the PDF with theorem/table-level citations (8-section template; editorial notes flag every judgement call);
- the **original PDF** (`papers/`) and a **layout-preserving Chinese translation** produced by [SuperTranslate](https://github.com/asimfish/super_translate) with object-level QA (`papers_zh/`);
- a machine-readable metadata card (`data/meta/`), from which this README is generated (`src/generator.py`).

Venue discipline: main conference / journal / workshop / preprint are always labelled separately. A workshop paper is never written as a main-conference paper.

*Maintained by [asimfish](https://github.com/asimfish). Structure follows [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co).*

## Content

1. [GFlowNet x Optimal Transport (core)](#core)
2. [Non-Acyclic GFlowNet Theory (the bridge to OT)](#nonacyclic)
3. [GFlowNet Foundations and Training Objectives](#foundations)
4. [Optimal Transport Prerequisites and Graph OT](#ot-prereq)
5. [Neural / Amortized OT and Schrodinger Bridges](#neural-ot)
6. [Competing and Adjacent Works: OT and SB on Graphs](#competitors)
7. [2026 Additions with Deep-Dive Reports](#new-2026)
8. [2026 Additions and Trends](#trends-2026)
9. [Deep-Dive Reports and Translated PDFs](#deep-dive-reports)
10. [Insights and Open Problems](#insights)
11. [Reports and Slides](#deliverables)
12. [Contributing and Citation](#contributing)

<a id="core"></a>
### GFlowNet x Optimal Transport (core)

1. **Your GFlowNet Secretly Learns an Optimal Transport Plan.** ICML 2026 SPIGM Workshop, 2026. [paper](https://arxiv.org/abs/2606.06272) [report](reports/O08_2606.06272.md) [zh-PDF](papers_zh/2606.06272.zh.pdf) [PDF](papers/2606.06272.pdf)

    *Ian Maksimov, Nikita Morozov, Denis Belomestny, Sergey Samsonov* · `P0`

    > Fixing the initial edge-flow distribution L in a minimum-flow non-acyclic GFlowNet turns its objective into a Kantorovich OT problem with graph shortest-path ground cost (Theorem 3.2), so the optimal forward policy encodes an executable optimal transport plan rather than a static coupling matrix.

2. **Learning Shortest Paths with Generative Flow Networks.** ICML 2026 SPIGM Workshop, 2026. [paper](https://arxiv.org/abs/2603.01786) [code](https://github.com/GreatDrake/gfn-pathfinding) [report](reports/O07_2603.01786.md) [zh-PDF](papers_zh/2603.01786.zh.pdf) [PDF](papers/2603.01786.pdf)

    *Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov* · `P0`

    > Minimizing the expected trajectory length of a non-acyclic GFlowNet is equivalent to assigning zero probability to every non-shortest trajectory (Theorem 3.4, iff), which reduces unweighted shortest-path search to training a flow-regularized non-acyclic GFlowNet whose backward policy solves the task.


<a id="nonacyclic"></a>
### Non-Acyclic GFlowNet Theory (the bridge to OT)

1. **A Theory of Non-Acyclic Generative Flow Networks.** AAAI 2024 (Proceedings of the AAAI Conference on Artificial Intelligence 38(10): 11124-11131), 2024. [paper](https://arxiv.org/abs/2312.15246) [report](reports/T19_2312.15246.md) [zh-PDF](papers_zh/2312.15246.zh.pdf) [PDF](papers/2312.15246.pdf)

    *Leo Maxime Brunswic, Yinchuan Li, Yushun Xu, Shangling Jui, Lizhuang Ma* · `P0`

    > Extends GFlowNet flow theory to measurable spaces with cycles via 0-flows, proves ratio-type FM/DB/TB losses are unstable (flows pile into cycles) and gives a family of difference-type stable losses; the appendix characterizes R-flows as one acyclic flow plus the cycle space and bounds expected trajectory length by total flow.

2. **Revisiting Non-Acyclic GFlowNets in Discrete Environments.** ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267), 2025. [paper](https://arxiv.org/abs/2502.07735) [code](https://github.com/GreatDrake/non-acyclic-gfn) [report](reports/T36_2502.07735.md) [zh-PDF](papers_zh/2502.07735.zh.pdf) [PDF](papers/2502.07735.pdf)

    *Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov* · `P0`

    > Rebuilds non-acyclic GFlowNet theory on finite graphs with the backward policy as primitive: flows are expected visit counts, positive conservative flows biject with (P_B, F(s_f)), fixed P_B gives a unique solution so loss stability is irrelevant, and minimal expected length equals minimal total flow (a linear objective over the flow polytope); also generalizes the entropy-regularized RL equivalence.


<a id="foundations"></a>
### GFlowNet Foundations and Training Objectives

1. **Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation.** NeurIPS 2021 (Advances in Neural Information Processing Systems 34), 2021. [paper](https://arxiv.org/abs/2106.04399) [code](https://github.com/bengioe/gflownet) [report](reports/T00_2106.04399.md) [zh-PDF](papers_zh/2106.04399.zh.pdf) [PDF](papers/2106.04399.pdf)

    *Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio* · `P0`

    > Introduces GFlowNet: casts sequential object construction as a DAG flow network, proves that any flow satisfying the local flow-matching conditions with terminal flow R(x) induces a policy with pi(x) = R(x)/Z (Prop. 2), fixes the n(x)R(x) path-count bias of tree/MaxEnt-RL views (Prop. 1c), and trains it with a log-domain TD-like objective (Eq. 12) that is provably off-policy under full support (Prop. 3).

2. **GFlowNet Foundations.** JMLR 24(210):1-55, 2023, 2023. [paper](https://arxiv.org/abs/2111.09266) [report](reports/T02_2111.09266.md) [zh-PDF](papers_zh/2111.09266.zh.pdf) [PDF](papers/2111.09266.pdf)

    *Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, Emmanuel Bengio* · `P0`

    > Recasts GFlowNet flows as measures on the set of complete trajectories, so state and edge flows are measures of trajectory sets and flow conservation becomes a corollary (Prop. 8) rather than an axiom; proves the Markovian-flow characterization (Prop. 16), three equivalent parametrizations (Prop. 18), the detailed balance criterion (Prop. 21), and uniqueness of the Markovian representative of each edge-flow equivalence class (Prop. 23); adds conditional/state-conditional flows for free-energy, entropy and mutual-information estimation.

3. **Trajectory Balance: Improved Credit Assignment in GFlowNets.** NeurIPS 2022 (Advances in Neural Information Processing Systems 35), 2022. [paper](https://arxiv.org/abs/2201.13259) [code](https://github.com/GFNOrg/gflownet/tree/trajectory_balance) [report](reports/T03_2201.13259.md) [zh-PDF](papers_zh/2201.13259.zh.pdf) [PDF](papers/2201.13259.pdf)

    *Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio* · `P0`

    > Replaces the local flow-matching and detailed-balance constraints by a single whole-trajectory constraint Z prod P_F = R(x) prod P_B (Eq. 13), proves that zero trajectory-balance residual on all complete trajectories implies reward matching and proportional sampling (Prop. 1), notes that any fixed backward policy pins down a unique reward-matching flow (Sec. 3.1), and reports faster credit propagation on long sequences and large action spaces plus up to 5x runtime speedup over flow matching.

4. **Learning GFlowNets from Partial Episodes for Improved Convergence and Stability.** ICML 2023, 2023. [paper](https://arxiv.org/abs/2209.12782) [report](reports/T05_2209.12782.md) [zh-PDF](papers_zh/2209.12782.zh.pdf) [PDF](papers/2209.12782.pdf)

    *Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin* · `P1`

    > SubTB(lambda) generalizes the GFlowNet balance constraint from single transitions and full trajectories to subtrajectories of any length, geometrically weighted by lambda, interpolating between the high-bias low-variance DB objective and the low-bias high-variance TB objective.

5. **Towards Understanding and Improving GFlowNet Training.** ICML 2023, 2023. [paper](https://arxiv.org/abs/2305.07170) [code](https://github.com/maxwshen/gflownet) [report](reports/T10_2305.07170.md) [zh-PDF](papers_zh/2305.07170.zh.pdf) [PDF](papers/2305.07170.pdf)

    *Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani* · `P1`

    > On enumerable biochemical benchmarks GFlowNets systematically underfit the target distribution by oversampling low-reward objects; the paper grounds flow quality in generalization and proposes prioritized replay, relative edge flow parametrization, and guided trajectory balance to control credit assignment.


<a id="ot-prereq"></a>
### Optimal Transport Prerequisites and Graph OT

1. **Optimal Transport for Machine Learners.** lecture notes (arXiv preprint, v3 2026-08-08), 2025. [paper](https://arxiv.org/abs/2505.06589) [code](https://github.com/gpeyre/ot4ml) [report](reports/O01_2505.06589.md) [zh-PDF](papers_zh/2505.06589.zh.pdf) [PDF](papers/2505.06589.pdf)

    *Gabriel Peyre* · `P0`

    > A 16-chapter graduate course on optimal transport for machine learning; for GFlowNet readers the essential parts are the graph Beckmann form of W1 (Sec. 6.5), Kantorovich duality and c-transforms (Ch. 5), the path-space Schrodinger reduction (Sec. 14.3), the coupling polytope (Sec. 3.1), and Sinkhorn as alternating Bregman projections (Sec. 9.1).

2. **Quadratically Regularized Optimal Transport on Graphs.** SIAM Journal on Scientific Computing 40(4): A1961-A1986, 2018, 2018. [paper](https://arxiv.org/abs/1704.08200) [report](reports/O02_1704.08200.md) [zh-PDF](papers_zh/1704.08200.zh.pdf) [PDF](papers/1704.08200.pdf)

    *Montacer Essid, Justin Solomon* · `P1`

    > Writes graph W1 as a Beckmann min-cost network flow on edges (Kantorovich with shortest-path cost is equivalent to min-cost flow), adds a quadratic regularizer whose small-alpha solution coincides with an LP solution (sparsity), and derives a dual Newton-type solver whose Hessian is the active-subgraph Laplacian.


<a id="neural-ot"></a>
### Neural / Amortized OT and Schrodinger Bridges

1. **GeONet: a neural operator for learning the Wasserstein geodesic.** UAI 2024, 2024. [paper](https://arxiv.org/abs/2209.14440) [code](https://github.com/agracyk2/GeONet) [report](reports/O03_2209.14440.md) [zh-PDF](papers_zh/2209.14440.zh.pdf) [PDF](papers/2209.14440.pdf)

    *Andrew Gracyk, Xiaohui Chen* · `P2`

    > A DeepONet-based neural operator that maps a pair of endpoint measures to the whole Wasserstein geodesic by fitting the primal-dual KKT system (continuity + Hamilton-Jacobi) of the Benamou-Brenier problem, trained without any geodesic ground truth.

2. **Schrodinger Bridge Flow for Unpaired Data Translation.** NeurIPS 2024 (Spotlight), 2024. [paper](https://arxiv.org/abs/2409.09347) [report](reports/O04_2409.09347.md) [zh-PDF](papers_zh/2409.09347.zh.pdf) [PDF](papers/2409.09347.pdf)

    *Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, Arnaud Doucet* · `P1`

    > Defines a flow of path measures whose unique fixed point is the Schrodinger Bridge, discretises it with a stepsize alpha to obtain alpha-IMF (alpha=1 recovers IMF/DSBM), and implements it as alpha-DSBM: an online, single-network, single-loss version of DSBM that needs no inner minimisation per iteration.

3. **Universal Neural Optimal Transport.** ICML 2025, 2025. [paper](https://arxiv.org/abs/2212.00133) [code](https://github.com/GregorKornhardt/UNOT) [report](reports/O05_2212.00133.md) [zh-PDF](papers_zh/2212.00133.zh.pdf) [PDF](papers/2212.00133.pdf)

    *Jonathan Geuter, Gregor Kornhardt, Ingimar Tomasson, Vaios Laschos* · `P1`

    > Trains a Fourier Neural Operator to predict the entropic-OT dual potential for arbitrary pairs of discrete measures at arbitrary resolution, using an adversarial measure generator plus a self-supervised bootstrapping loss whose minimisation provably bounds the loss against the true potential; yields state-of-the-art Sinkhorn initialisation with up to 7.4x speedup.

4. **Computing high-dimensional optimal transport by flow neural networks.** AISTATS 2025, 2025. [paper](https://arxiv.org/abs/2305.11857) [code](https://github.com/hamrel-cxu/FlowOT) [report](reports/O06_2305.11857.md) [zh-PDF](papers_zh/2305.11857.zh.pdf) [PDF](papers/2305.11857.pdf)

    *Chen Xu, Xiuyuan Cheng, Yao Xie* · `P2`

    > Q-flow solves the Benamou-Brenier dynamic OT between two sample-only distributions with a neural ODE: both terminal constraints are relaxed into KL divergences estimated on the fly by logistic classifiers, plus a finite-difference W2 transport cost, trained bi-directionally; the learned OT trajectory is then reused for image-to-image translation and high-dimensional density ratio estimation. Note: here flow means a continuous normalizing flow velocity field, NOT a GFlowNet.


<a id="competitors"></a>
### Competing and Adjacent Works: OT and SB on Graphs

1. **Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs.** NeurIPS 2025 (Main Conference Track), 2025. [paper](https://arxiv.org/abs/2506.12025) [code](https://github.com/smazelet/ULOT) [report](reports/C01_2506.12025.md) [zh-PDF](papers_zh/2506.12025.zh.pdf) [PDF](papers/2506.12025.pdf)

    *Sonia Mazelet, Remi Flamary, Bertrand Thirion* · `P1`

    > ULOT amortizes FUGW optimal transport between two explicit graphs with a cross-attention GNN conditioned on the (alpha, rho) tradeoff hyperparameters, predicting unbalanced plans in O(n1*n2) time, up to 100x faster than classical solvers and usable as a solver warm start.

2. **Generalized Schrodinger Bridge on Graphs.** ICML 2026 (PMLR 306); dblp as of 2026-09 lists CoRR entry only, 2026. [paper](https://arxiv.org/abs/2602.04675) [report](reports/C02_2602.04675.md) [zh-PDF](papers_zh/2602.04675.zh.pdf) [PDF](papers/2602.04675.pdf)

    *Panagiotis Theodoropoulos, Juno Nam, Evangelos Theodorou, Jaemoo Choi* · `P1`

    > GSBoG lifts the generalized Schrodinger bridge to controlled CTMCs on a fixed sparse graph, deriving a discrete Hopf-Cole system whose optimal jump rate is u*(y,x) = r(y,x)exp(V(x)-V(y)), and learns it with a discrete gIPF loss plus a temporal-difference term that prevents the state cost from cancelling out.

3. **Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation.** ICLR 2025 (Poster), 2025. [paper](https://arxiv.org/abs/2410.01500) [code](https://github.com/junhkim1226/DDSBM) [report](reports/C03_2410.01500.md) [zh-PDF](papers_zh/2410.01500.zh.pdf) [PDF](papers/2410.01500.pdf)

    *Jun Hyeong Kim, Seonghwan Kim, Seokhyun Moon, Hyeongwoo Kim, Jeheon Woo, Woo Youn Kim* · `P2`

    > DDSBM extends Iterative Markovian Fitting to CTMCs on finite state spaces with a proof of monotone convergence to the Schrodinger bridge, and shows that with node/edge-independent reference dynamics the induced entropic-OT cost is proportional to the graph edit distance, applied to molecular optimization on ZINC250K and Polymer.


<a id="new-2026"></a>
### 2026 Additions with Deep-Dive Reports

1. **Minimum-Cost Network Flow with Dual Predictions.** AAAI 2026, 2026. [paper](https://arxiv.org/abs/2601.20203) [report](reports/N01_2601.20203.md) [zh-PDF](papers_zh/2601.20203.zh.pdf) [PDF](papers/2601.20203.pdf)

    *Zhiyang Chen, Hailong Yao, Xia Yin* · `P1`

    > First minimum-cost flow algorithm warm-started by a learned dual prediction: epsilon-relaxation runs in O(min{n^3 log||p_hat - p*||_inf, n^3 log(nC)}) (Theorem 2), consistent and robust, with PAC bounds for learning the prediction and 12.7x / 1.6x speedups on traffic and PCB routing.

2. **Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels.** ICML 2026 SPIGM Workshop, 2026. [paper](https://arxiv.org/abs/2606.16073) [report](reports/N02_2606.16073.md) [zh-PDF](papers_zh/2606.16073.zh.pdf) [PDF](papers/2606.16073.pdf)

    *Kirill Korolev, Nikita Morozov, Stepan Pavlenko, Esmeralda S. Whitammer, Sergey Samsonov* · `P1`

    > Puts MCMC inside continuous non-acyclic GFlowNet theory: a learned stopping classifier d_F(s) decides termination, detailed balance ties the optimal classifier to the target (Theorem 3.6), total flow is minimal iff the expected length hits a closed-form n_Q* (Corollary 3.7); trajectory lengths drop 1-2 orders of magnitude vs ULA (Table 1).

3. **Stable GFlowNets with TV Monitoring and Probabilistic Guarantees.** arXiv preprint, 2026. [paper](https://arxiv.org/abs/2605.01729) [report](reports/N03_2605.01729.md) [zh-PDF](papers_zh/2605.01729.zh.pdf) [PDF](papers/2605.01729.pdf)

    *Zengxiang Lei, Ananth Shreekumar, Jonathan Rosenthal, Ruoyu Song, Alvaro A. Cardenas, Daniel J. Fremont, Dongyan Xu, Satish Ukkusuri, Z. Berkay Celik* · `P0`

    > Shows small TV does not bound GFlowNet loss (Prop. 3.3-3.4), then proves the converse certificate: per-trajectory TB loss <= c^2 implies TV <= 1 - e^{-2c} (Theorem 3.5) with a sampling-based probabilistic version independent of state-space size (Theorem 3.6); adaptive reference flows stabilize training at a computable fidelity cost (Theorem 3.10-3.11).

4. **Generative Modeling on Metric Graphs via Neural Optimal Transport.** arXiv preprint, 2026. [paper](https://arxiv.org/abs/2606.16273) [report](reports/N04_2606.16273.md) [zh-PDF](papers_zh/2606.16273.zh.pdf) [PDF](papers/2606.16273.pdf)

    *Alessandro Micheli, Yueqi Cao, Anthea Monod, Samir Bhatt* · `P2`

    > First generative model for distributions supported on metric graphs: embed the graph (Euclidean or tropical Abel-Jacobi), solve entropic OT via a neural semidual, project samples back; the generator converges weakly to a valid graph coupling (Theorem 4.1); scales to 10^6 Uber pickups on the Manhattan road graph.

5. **An Efficient Orlicz-Sobolev Approach for Transporting Unbalanced Measures on a Graph.** NeurIPS 2025 Spotlight, 2025. [paper](https://arxiv.org/abs/2502.00739) [report](reports/N05_2502.00739.md) [zh-PDF](papers_zh/2502.00739.zh.pdf) [PDF](papers/2502.00739.pdf)

    *Tam Le, Truyen Nguyen, Hideitsu Hino, Kenji Fukumizu* · `P1`

    > Unbalanced OT on graph metric spaces without a two-level problem: EPT is recast as a balanced OT with nonnegative cost (Prop. 3.1), endowed with Orlicz geometry, then regularized into Orlicz-Sobolev transport that reduces to a single univariate optimization (Theorem 4.2), 250-13800x faster than Orlicz-EPT.

6. **Entering the Era of Discrete Diffusion Models: A Benchmark for Schrodinger Bridges and Entropic Optimal Transport.** ICLR 2026, 2026. [paper](https://arxiv.org/abs/2509.23348) [report](reports/N06_2509.23348.md) [zh-PDF](papers_zh/2509.23348.zh.pdf) [PDF](papers/2509.23348.pdf)

    *Xavier Aramayo Carrasco, Grigoriy Ksenofontov, Aleksei Leonov, Iaroslav Sergeevich Koshelev* · `P1`

    > First benchmark with analytically known solutions for discrete-space EOT/SB: any (p_0, v*) yields a pair (p_0, p_1) with closed-form optimal coupling (Theorem 3.1), made tractable in S^D via CP parameterization; by-product solvers DLightSB / DLightSB-M / alpha-CSBM, with DLightSB strongest across all settings.


<a id="trends-2026"></a>
### 2026 Additions and Trends

Papers found by the 2026 trend scan (relevance >= 4). Full analysis: [reports/TRENDS_GFN_2026.md](reports/TRENDS_GFN_2026.md), [reports/TRENDS_OT_2026.md](reports/TRENDS_OT_2026.md).

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
### Deep-Dive Reports and Translated PDFs

| ID | Paper | Report | Original PDF | Chinese PDF (SuperTranslate) | QA |
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
### Insights and Open Problems

The cross-paper synthesis lives in [reports/INSIGHTS.md](reports/INSIGHTS.md) and the competitor matrix in [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md). Headline conclusions:

1. **Internal-flow non-uniqueness is the entry point.** Matching the terminal distribution fixes only the boundary; the cycle space of the state graph leaves a large family of valid internal flows. Minimum total flow selects one, and that one is an OT plan.
2. **The equivalence itself is classical; the contribution is the translation.** Shortest-path-cost Kantorovich OT on a graph equals min-cost flow. What GFlowNets add is an executable *local routing policy* on combinatorial graphs where the cost matrix is never materialised.
3. **The least-crowded follow-up is an error bound.** Balance residuals already certify total-variation error of the terminal distribution; extending them to OT cost gap and marginal violation (with dual potentials as certificates) has no direct competitor yet.
4. **Two follow-ups are crowded.** Conditional GFlowNets that amortise OT over a family of graphs collide with ULOT (NeurIPS 2025) and UNOT (ICML 2025); entropic GFN-OT collides with the discrete / graph Schrodinger-bridge line (DDSBM, ICLR 2025; GSBoG, ICML 2026 main track, same quadrant as O08 but with no error bound).
5. **Venue reality check.** Both GFN x OT papers (O07, O08) are ICML 2026 SPIGM *workshop* papers; all three competitors are main-track. The structural moat left to GFlowNets is the implicit combinatorial graph where no cost matrix can be materialised.

<a id="deliverables"></a>
### Reports and Slides

| Deliverable | Path |
|---|---|
| Consolidated report, Chinese (91 pages) | [PDF](reports/pdf/awesome_gflow_ot_report_zh.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_zh.md) |
| Consolidated report, English (120 pages) | [PDF](reports/pdf/awesome_gflow_ot_report_en.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_en.md) |
| Summary slides (single-file HTML, 23 slides; keyboard / wheel / touch) | [slides/awesome_gflow_ot_slides.html](slides/awesome_gflow_ot_slides.html) |
| Summary slides (Beamer PDF, 28 pages incl. backup) | [slides/awesome_gflow_ot_slides.pdf](slides/awesome_gflow_ot_slides.pdf) |
| Cross-paper synthesis | [reports/INSIGHTS.md](reports/INSIGHTS.md) |
| Competitor matrix | [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md) |
| 2026 trend scans | `reports/TRENDS_GFN_2026.md`, `reports/TRENDS_OT_2026.md` |

<a id="contributing"></a>
### Contributing and Citation

Add a paper by dropping a JSON card into `data/meta/` (see any existing card for the schema), optionally a report into `reports/`, then run `python3 src/generator.py`. Please keep the venue discipline (main / journal / workshop / preprint) and cite theorem or table numbers when summarising.

```bibtex
@misc{awesome_gflow_ot_2026,
  title  = {Awesome GFlowNet x Optimal Transport: a curated, evidence-first reading list},
  author = {asimfish},
  year   = {2026},
  url    = {https://github.com/asimfish/awesome_Gflow_OT}
}
```

Translation engine: [SuperTranslate](https://github.com/asimfish/super_translate). Slides tooling: [beamer-skill](https://github.com/Noi1r/beamer-skill). Writing discipline: [anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing), [shuorenhua](https://github.com/MrGeDiao/shuorenhua). List conventions: [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co).

Licensed under [MIT](LICENSE). Paper PDFs remain under their original licenses (arXiv).
