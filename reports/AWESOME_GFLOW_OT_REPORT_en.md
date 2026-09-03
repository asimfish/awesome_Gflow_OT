---
title: "GFlowNet x Optimal Transport: From Flow Conservation to Kantorovich Plans"
subtitle: "awesome_Gflow_OT Consolidated Report"
author: "awesome_Gflow_OT project"
date: "2026-09"
lang: en
---
# Abstract and Reading Guide

**Abstract.** Generative Flow Networks (GFlowNets) formulate reward matching as flow conservation on a graph: terminal edge flow equals reward, and inflow to an internal state equals outflow. Rewards only anchor the boundary, leaving a family of valid solutions for internal flows. This report organizes 18 papers around a central theme: on finite graphs that allow cycles, minimizing the total internal flow is equivalent to minimizing the expected trajectory length (T36); for a single source, this is equivalent to only traversing shortest paths (O07); further, fixing the first-step edge flow to a source distribution precisely transforms it into Kantorovich optimal transport with graph shortest paths as costs, where the endpoint distribution of trajectories sampled by the optimal forward policy is the optimal coupling (O08). Each link in this chain of equivalences has a counterpart in classical minimum cost flow and the Beckmann problem (O01, O02). The novelty lies in translating this into the GFlowNet language, where the output is no longer a coupling matrix but a local routing policy that can be executed step-by-step on implicit combinatorial graphs of size \(20!\). The report also presents the competitive landscape (three main conference competitors ULOT, GSBoG, DDSBM against two workshop papers from our side), a trend scan for 2025–2026 (199 arXiv hits), ratings for four candidate research topics, and a decisive experiment that can be executed immediately. Conclusion: The least crowded follow-up topic, with all components ready, is "balance residual → OT error bound, with dual potentials as certificates."

**Reading Guide.**

| Reader | Entry Point |
|---|---|
| Familiar with GFlowNet, unfamiliar with OT | Chapter 1 → Chapter 6 (O01 reading guide and correspondence table) → Chapter 5 |
| Familiar with OT, unfamiliar with GFlowNet | Chapter 2 → Chapter 3 → Chapter 5 |
| Only want to know whether to pursue this direction | Chapter 7 → Chapter 8 → Chapter 9 |
| Want to start working immediately | Decisive experiment in Chapter 9 + Repository tour in Chapter 10 |

**Evidence Discipline.** Every theorem, number, and experimental result is cited (repository ID + original theorem/table number); any inference made by this report, rather than a conclusion from the paper, begins with "Our assessment:". Publication status distinguishes between main conference / journal / workshop / preprint: O07, O08 are ICML 2026 SPIGM Workshop (not main conference); GSBoG is ICML 2026 main conference; ULOT is NeurIPS 2025 main conference; DDSBM is ICLR 2025 main conference.

**How this report was generated.** Paper-level content in Chapters 2–7 is taken from corresponding sections (core contributions, assumptions, main line position, insight) of 18 individual interpretation reports under `reports/`, assembled by `scripts/build_report.py`; Chapters 1, 8, 9, 10 and the appendix are comprehensively written. For notation, derivations, and experimental details for each paper, please refer to the original interpretation reports.

# Chapter 1: Problem: Why is the internal flow not unique, and why is this an optimal transport problem?

## 1.1 Rewards only anchor the boundary

GFlowNets define non-negative edge flows \(F(s\to s')\) on a directed graph \(G\), requiring two conditions: internal state flow conservation \(\sum_{u}F(u\to s)=\sum_{v}F(s\to v)\), and terminal edge flow equals reward \(F(x\to s_f)=R(x)\). As long as both hold, the termination distribution of the forward policy \(P_F(s'\mid s)=F(s\to s')/F(s)\) is \(R(x)/Z\), where \(Z=F(s_0)=\sum_xR(x)\) (T02 core correctness theorem; see `reports/T02_2111.09266.md` §2).

Both constraints are linear (T02 Prop. 19 / Eq. (22)), and the feasible set is a polytope. Rewards only appear on terminal edges, so they fix the "boundary" of the polytope; how much freedom internal edge flows have depends on the graph structure. T02 states this in Prop. 18, item 3: Markovian flow is uniquely determined by terminal flow and the backward policy \(P_B\) on non-terminal edges. In other words, **choosing a \(P_B\) is choosing an internal flow**, and the original paper only states in §2.6 regarding "which \(P_B\) to choose": one might prefer shorter paths.

## 1.2 More degrees of freedom, and more danger, with cycles

On a DAG, trajectory length has an upper bound, and flow freedom only comes from \(P_B\). Once cycles are allowed, T19 proves that the set of R-flows is "an acyclic flow + the cycle space \(H^1_+(G)\)" (Prop. 5). Flow on cycles does not change the termination distribution but can be arbitrarily amplified; ratio-based losses (FM/DB/TB) will infinitely accumulate mass into cycles (Thm. 3, flow explosion), while difference-based losses are stable (Thm. 4). There is only an inequality between total flow and expected trajectory length \(\mathbb E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\) (Thm. 2).

T36 reconstructs this theory into a computable form on finite discrete graphs: taking \(P_B\) as the primary object, flow is defined as expected visit count times terminal flow, corresponding one-to-one with \((P_B,F(s_f))\) (Prop. 3.7), and tightens the inequality to an equality:

\[\sum_{s\notin\{s_0,s_f\}}F(s)=Z\cdot\mathbb E[n_\tau].\]

This equality is the turning point of the entire report: a linear functional on the flow space (total flow) and a behavioral quantity (expected number of steps) are tied together. "Learning a GFlowNet with the shortest expected trajectory" henceforth equals "minimizing a linear function on a polytope" (T36 Eq. (11)).

## 1.3 Why this is optimal transport

Linear objective, linear constraints, non-negative variables—this is a linear program, and a special one: constraints are vertex conservation, variables are edge flows, and the objective is total flow. O01 (Peyré's lecture notes) Prop. 6.23 and O02 (Essid & Solomon) Eq. (1)⇔(3) present two ways of saying the same thing: the Kantorovich problem on a graph with shortest path costs is equivalent to minimum cost flow with edge flows as variables and vertex conservation as constraints (discrete Beckmann problem). What's missing is a source distribution—under a single source \(\delta_{s_0}\), the coupling set is a singleton, and the OT structure is empty (O01 Remark 3.2).

O08 precisely fills this gap: by fixing the first-step edge flow to a source distribution \(L\), the minimum total flow objective precisely becomes \(\min_{\Pi\in\Gamma(L,R)}\sum_{u,x}d_G(u,x)\Pi(u,x)\) (Thm. 3.2). Chapters 3–5 clarify each link in this chain in the order T19 → T36 → O07 → O08.

## 1.4 Our assessment: What is new and what is not

- **Not new**: OT with shortest path costs on a graph ≡ minimum cost flow (classical, Beckmann 1952; O02 2018 provides a regularized version).
- **New**: Translating this equivalence into the GFlowNet language, the solution is no longer an \(|U|\times|\mathcal X|\) coupling matrix, but a local policy \(P_F(s'\mid s)\) that depends only on the current state, executable on combinatorial graphs where states cannot be enumerated and cost matrices cannot be constructed (O08 §4.2's \(S_{20}\) permutation graph).
- **Cost**: To make the problem an LP, O08 requires \(\sum L=\sum R=1\) and \(Z\) to be known (Assumption 3.1)—the GFlowNet's signature ability of "only needing unnormalized rewards" is abandoned along this line.

# Chapter 2: GFlowNet Basics and Training Objectives

Only content necessary for the main theme is retained: definition of flow and correctness theorem, original source of internal flow degrees of freedom, and the family of training objectives and their meaning in flow space. For complete interpretations of each paper, see `reports/`.

## 2.1 Overview of Training Objective Families

| Objective | Constraint Granularity | Key Quantity | Relationship to Main Thread |
|---|---|---|---|
| FM (T00) | Single state inflow = outflow | Edge flow \(F(s\to s')\) | Directly constrains flow conservation; ratio-based unstable with cycles (T19 Thm. 3) |
| DB (T02) | Single edge \(F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\) | State flow \(F(s)\) | Requires explicit state flow; O08's training objective Eq. (20) is DB + state flow regularization |
| TB (T03) | Full trajectory \(Z\prod P_F=R\prod P_B\) | \(\log Z\) | \(\log Z\) is the learned baseline; \(Z\) is known in O08's LP setting |
| SubTB(\(\lambda\)) (T05) | Arbitrary sub-trajectory, \(\lambda^{n-m}\) weighted | Interpolation parameter \(\lambda\) | Continuous interpolation between DB and TB; O08's \(\lambda\) is a flow regularization coefficient, not this \(\lambda\) |

When residuals are zero, all four point to the same reward-matching solution family; the only differences are in the spatial scale of gradients and the credit assignment distance. **None of them answer "which internal flow to choose."**

## 2.2 T00 · Original GFlowNet

> **In a nutshell**: This paper reformulates the problem of "sampling composite objects proportional to their reward" as a flow conservation problem on a directed acyclic graph, and proves that a policy satisfying flow conservation necessarily yields \(\pi(x)=R(x)/Z\). It solves the fundamental flaw of autoregressive/MaxEnt RL being biased by the number of paths \(n(x)\) when "an object has multiple generation paths." On the map of GFlowNet × OT, it is the origin: the terminal distribution is uniquely fixed by the reward, but the internal flow leaves an entire family of solutions—the paper explicitly writes out this family of solutions in the appendix but provides no selection principle. The OT main thread aims to equip this degree of freedom with an objective function.

| Field | Content |
|---|---|
| arXiv | [2106.04399](https://arxiv.org/abs/2106.04399) |
| Publication | **NeurIPS 2021 Main Conference** (Paper homepage footnote: 35th Conference on Neural Information Processing Systems (NeurIPS 2021)). Local PDF is arXiv v2 (2021-11-19) |
| Authors | Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio |
| Code | <https://github.com/bengioe/gflownet> (Given at the end of §1 and beginning of Appendix A in the original paper) |
| Local Repository PDF | `papers/2106.04399.pdf` · Chinese translation `papers_zh/2106.04399.zh.pdf` (not yet generated) |
| Reading Priority | **P0** · All subsequent GFlowNet theories (including OT results) rewrite its flow conservation equations; without reading it, it's impossible to judge which constraints are essential and which were added later |

#### 2. Core Contributions (Numbered as in Original Text)

Section 1 of the original paper lists four contributions: proposing GFlowNets based on flow networks and local flow-matching conditions; proving the connection between flow-matching conditions and reward-matching generative policies, as well as off-policy properties and asymptotic convergence; demonstrating the value of "modeling the entire distribution rather than individual modes" on synthetic data; and conducting comparative experiments with PPO and MCMC in large-scale molecular synthesis domains. These are elaborated below by proposition number.

**Proposition 1 (§2.1, proof in A.1) — Path count bias from a tree perspective.** Let \(\tilde V(s)=\sum_{\vec b\in A^*(s)}R(s+\vec b)\) be the sum of rewards for terminal states reachable from \(s\), and the policy be \(\pi(a|s)=\tilde V(s+a)/\sum_b \tilde V(s+b)\). Then (a) \(\pi(s)=\sum_{\vec a_i:C(\vec a_i)=s}\pi(\vec a_i)\); (b) if \(C\) is a bijection, \(\pi(x)=R(x)/\sum_{x}R(x)\); (c) if \(C\) is not injective and there are \(n(x)\) distinct action sequences reaching \(x\), then
\[\pi(x)=\frac{n(x)R(x)}{\sum_{x'}n(x')R(x')}.\]
The original paper's assessment of (c) is that "in combinatorial spaces ... this can become exponentially bad as trajectory lengths increase," specifically leading to large molecules being oversampled exponentially just because they have many paths (§2.1). This proposition is the motivation for the entire paper and a litmus test for determining "whether a method is a true GFlowNet."

**Proposition 2 (§2.1) — Flow conservation implies proportional sampling.** When \(\pi(a|s)=F(s,a)/F(s)\) (Eq. (5)), \(F(s)=R(s)+\sum_{a\in A(s)}F(s,a)\) (for internal nodes \(R(s)=0\), for terminal states \(F(x)=R(x)>0\)), and flow conservation Eq. (4) holds: (a) \(\pi(s)=F(s)/F(s_0)\); (b) \(F(s_0)=\sum_{x\in\mathcal X}R(x)\); (c) \(\pi(x)=R(x)/\sum_{x'}R(x')\).

**Eq. (11) → Eq. (12) — Log-domain flow matching objective.** The naive squared residual (Eq. (11)) fails in high-dimensional spaces because flows near the root are exponentially larger than those near the leaves. It is replaced by a log-domain version with a smoothing constant \(\epsilon\):
\[\mathcal L_{\theta,\epsilon}(\tau)=\sum_{s'\in\tau\neq s_0}\Big(\log\big[\epsilon+\!\!\sum_{s,a:T(s,a)=s'}\!\!\exp F^{\log}_\theta(s,a)\big]-\log\big[\epsilon+R(s')+\!\!\sum_{a'\in A(s')}\!\!\exp F^{\log}_\theta(s',a')\big]\Big)^2 .\]
The original paper is very direct about the role of \(\epsilon\): it does not change the global minimum, only avoids taking the logarithm of extremely small flows, and "trades-off how much pressure we put on matching large versus small flows." In experiments, it is set close to the smallest possible value of \(R\) (§2.2).

**Proposition 3 (§2.2, proof in A.1) — Off-policy property.** If training trajectories come from an exploration policy \(P\) with the same support as the optimal \(\pi\), the model family is rich enough (\(\exists\theta:F_\theta=F^*\)), and the loss achieves its minimum when flows are matched, then the global optimum of the expected loss satisfies \(F_{\theta^*}=F^*\), for all \(\tau\sim P\) we have \(\mathcal L_{\theta^*}(\tau)=0\), and \(\pi_{\theta^*}(x)=R(x)/Z\). The original paper compares this to asynchronous dynamic programming: "converges provided every state is visited infinitely many times asymptotically" (§2.2).

**A paragraph after the proof of Proposition 3 in Appendix A.1 — Non-uniqueness of internal flows.** This is the sentence of most concern to this report: the original text states "Note that in the general case, an infinite number of solutions exist," and provides an explicit counterexample: two trajectories \(s_0\!\to\!s_A\!\to\!s_T\) and \(s_0\!\to\!s_B\!\to\!s_T\) lead to the same terminal state with reward \(r\). Then the solution set is a one-dimensional family
\[F(s_A)=u,\qquad F(s_B)=r-u,\qquad u\in[0,r].\]
The paper merely presents this as a technical note that "solutions are not unique, so the proof can only guarantee the terminal distribution," without providing any principle for choosing \(u\).

**Proposition 4 (A.2) — Correspondence with Q-functions (bijection case only).** Let \(\mu(a|s)=1/|A(s)|\) be a uniform policy, \(f(x)=\prod_{t=0}^{n}|A(s_t)|\), and \(\hat R(x)=R(x)f(s_{n-1})\). Then \(Q^\mu(s,a;\hat R)=F(s,a;R)f(s)\). For the non-injective case, the original paper only provides a conjecture (end of A.2, "Conjecture") and explicitly states "since an infinite number of valid flows exists, it's not clear that such a simple equivalence always exists."

**Distinction from Soft Q-Learning (§3).** Both have similar objective forms, but the difference is that GFlowNet's incoming flow includes **all parent nodes**, while soft Q-learning only uses the parent node on the trajectory. The consequence is given by Proposition 1: soft Q-learning yields \(P(\tau)\propto R(\tau)\), while GFlowNet yields \(P(x)\propto R(x)\).

**Algorithm 1 (A.5) — Multi-round active learning loop.** Inputs are initial dataset \(D_0=\{x_i,y_i\}\), number of rounds \(N\), and inverse temperature \(\beta\);
In each round, first fit a proxy \(M\) on \(D_{i-1}\), then train \(\pi_\theta\) using \(r(x)=M(x)^\beta\) as the reward.
Sample a batch \(B\) from \(\pi_\theta\), evaluate it with oracle \(O\), and merge the new labels into the dataset.
This outer loop explains why the paper places such importance on diversity: the coverage of the proxy is determined by the generator; if the generator collapses to a single mode, the proxy will never learn anything else.

#### 5. Underlying Assumptions and Applicable Boundaries

The scope within which the conclusions hold, stated positively:

1.  **Finite, acyclic, deterministic** state graph: Action sequence length is bounded, and environment transitions \(T(s,a)\) are deterministic (§2).
2.  **Rewards only at terminal states**, and strictly positive \(R(x)>0\); internal nodes have \(R(s)=0\) (§2.1).
3.  **All edge flows are strictly positive**; zero-flow edges are excluded by restricting the set of available actions for parent states (§2.1).
4.  **The training policy's support covers the optimal policy's support** (precondition for Prop. 3), approximated in practice by \(0.95\pi+0.05\text{uniform}\).
5.  **The model family can realize the true flow**, and optimization achieves the global optimum of the desired loss (two preconditions for Prop. 3).
6.  The conclusions only fix the **terminal distribution**. Internal flows, path distributions, and trajectory entropy are not constrained by Prop. 2 (see the one-dimensional solution family in A.1 cited in §2).

The paper's own listed limitations (§5 Discussion & Limitations) contain only one point: like TD methods, it relies on bootstrapping, which may lead to optimization difficulties, and suggests combining this generative method with local optimization in the future.

Where things specifically go wrong when assumptions fail, point by point:

-   **Cyclic graphs**: The inductive proof of Prop. 2 relies on topological ordering (induction on parent nodes starting from \(s_0\)). If there are cycles, induction has no starting point, and trajectories may not terminate. T02 §3.3.1 bypasses this with timestamp augmentation, while T19/T36 truly reconstruct the theory.
-   **Stochastic environments**: Prop. 2 assumes \(T(s,a)\) is deterministic; under stochastic transitions, the policy cannot freely allocate edge flows. Counterexample 50 in Appendix C of T02 provides an explicit counterexample.
-   **\(R(x)=0\)**: Eq. (12) requires taking \(\log\), and zero-reward terminal states would violate the positive flow assumption; in engineering, \(R_{\min}\) clipping is used (A.4 uses 0.01), at the cost of altering the target distribution.
-   **Incomplete training distribution support**: The conclusion of Prop. 3 does not hold for uncovered edges. The uniform random policy dataset in Fig. 12 is an instance of partial failure—the paper states that "many points are left out intentionally as a generalization test," and the fitting quality is significantly worse than in Fig. 13.

#### 6. Position in the GFlowNet × OT Main Thread

**Predecessors.** Buesing et al. (2019)'s MCTS+value method (the original paper §3 treats it as the main baseline, Prop. 1 was written to refute it), MaxEnt RL / soft Q-learning (Haarnoja et al. 2017), discrete MCMC (Grathwohl et al. 2021, Xie et al. 2021's MARS).

**Successors.** T02 (GFlowNet Foundations) recasts the flow network from §2 as a trajectory measure and introduces \(P_B\) and DB; T03 (Trajectory Balance) replaces local constraints with full trajectory constraints. Eq. (12) in this paper is what later became known as the FM loss.

**Contributions to the main thread of "internal flow selection = optimal transport" are threefold, all within the original paper.**

1.  **It explicitly states the degrees of freedom.** Appendix A.1's one-dimensional solution family \(u\in[0,r]\) is the first explicit construction in GFlowNet literature of "same terminal distribution, different internal flows." The task of the OT main thread is to add an objective function to this solution set.

2.  **It explicitly declares this degree of freedom as an open problem.** A.6 states: "Note that our loss does not enforce any kind of distribution on flows, and a uniform flow is not necessarily desirable (investigating this could be interesting future work, perhaps some distributions of flows have better generalization properties)." This sentence is the source of legitimacy for the entire OT main thread in this repository—it's not a problem fabricated by later researchers, but an opening left by the original authors.

3.  **It provides an empirical picture of how degrees of freedom are "arbitrarily chosen."** The middle plot of Fig. 10 shows the visit distribution on all paths leading to \((6,6)\), while the right plot shows a uniform distribution on the same set of paths; the two are clearly different. The original paper explains this as "some preference towards other corners, presumably due to early bias during learning as well as the position of the other modes." That is: **without regularization, internal flow is determined by the serendipity of optimization dynamics.** This is precisely what min-flow / OT principles aim to replace.

The differences between the three settings are clearest when viewed together (the OT column is based on the premises of O08, as organized in this report):

| Dimension | T00 (2021) | T02 (Foundations) | O08's OT Theorem Setting |
|---|---|---|---|
| Graph | Finite DAG, single source, each terminal is a sink | Finite pointed DAG, single source \(s_0\), single sink \(s_f\) | Finite directed graph, **cycles allowed** |
| Source constraint | \(F(s_0)=Z\), no source distribution | Same as left | Fixed source distribution \(L\) over initial state set \(U\) |
| Terminal constraint | \(F(x)=R(x)>0\) | \(F(s\to s_f)=R(s)\ge0\) | Fixed terminal distribution \(R\), \(\sum L=\sum R=1\) |
| Backward policy | No such object | \(P_B\) can be freely specified on non-terminal edges | Induced by optimal flow |
| Internal flow | A family of solutions, no selection principle | A family of solutions, parameterized by \(P_B\) | Uniquely determined by **minimum total flow** (in general cases) |
| Flow semantics | Trajectory quality (each trajectory traverses each edge at most once) | Trajectory measure | expected visit counts |

**A crucial counter-limitation that must be clarified.** T00's setting does not satisfy the premises of the OT theorem in 2606.06272 (O08): that work requires a fixed **source distribution** \(L\) over the set of initial states \(U\), allows cycles, and has a minimum total flow objective; T00 has a single source \(s_0\), is strictly acyclic, and has no flow regularization. Therefore, stating that "T00's GFlowNet is doing OT" is incorrect. T00 provides **the existence of the problem** (degrees of freedom exist and are unconstrained), not the answer.

#### 7. Reusable Insights and Open Questions

1. **The double-path counterexample in T00 is precisely a degenerate case of min-flow.** In the example in A.1, the two paths are of equal length, and the total internal flow \(2u+2(r-u)=2r\) is independent of \(u\) (derived in this report; the original text only provides the solution family without calculating the total flow). This indicates that: **the minimum total flow principle makes no selection at all between paths of equal length**, requiring additional tie-breaking (entropy regularization, \(P_B\) prior, or a second-order objective). This can be directly formulated as a minimal experiment: verifying the dimension of the solution set for min-flow LP on a diamond DAG.

2. **The path count bias \(n(x)\) will return in another form in the OT context.** Prop. 1(c) states that "uniform over trajectories \(\implies\) sampling according to \(n(x)R(x)\)". In minimum flow problems, optimal solutions typically concentrate mass on a few shortest paths, which is equivalent to artificially suppressing \(n(x)\) to a very small value. It's worth drafting a proposition: the relationship between the support set size of min-flow solutions and \(n(x)\).

3. **The terminal loss weighting with \(\lambda_T=10\) is a transferable engineering conclusion.** In the OT version, marginal constraints (source distribution + terminal distribution) are terminal constraints. Weighting them means prioritizing the correctness of coupling margins first, and then optimizing the transport cost. This can be directly used as a starting point for hyperparameter tuning in min-flow regularized TB.

4. **Data generated backward is more effective than data generated forward randomly (Fig. 12 vs Fig. 13).** This experience aligns with the intuition in OT of "inferring coupling backward from the target marginals," and also foreshadows the backward sampling training method introduced in T02 after introducing \(P_B\). A reproducible check: compare \(L_1\) and average trajectory length under three datasets on the same hypergrid: forward uniform sampling, backward sampling, and backward sampling along shortest paths.

5. **It is unclear whether log-domain + \(\epsilon\)-smoothing is beneficial or detrimental for min-flow regularization.** The role of \(\epsilon\) is to suppress the gradient weights of small flows; however, the minimum total flow objective precisely requires precise control over small flow edges (to push them to 0). The tension between these two is a concrete, experimentally verifiable open question.

6. **Prop. 4's Q-function correspondence only holds on trees; for non-injective cases, it's only a conjecture.** If the main OT line wants to borrow RL's duality theory (e.g., writing minimum flow as an optimization of some Q), this gap must first be filled, or one must detour via T02's \(P_B\) parameterization.

## 2.3 T02 · GFlowNet Foundations

> **One sentence summary**: This paper re-casts GFlowNets from "a set of flow conservation equations" into "a measure over the set of complete trajectories," thereby establishing that state flow/edge flow are measures of measurable sets rather than axioms, three equivalent parameterizations of Markovian flow, and the detailed balance condition. For the main OT line, it is the only essential prerequisite: **Proposition 18 precisely characterizes "how much freedom remains after fixing the terminal reward" — what remains is exactly the backward policy \(P_B\) on non-terminal edges**, and §2.6 of the original text states, "we may prefer shorter paths." What the OT paper does is to replace this degree of freedom with an optimizable objective.

| Field | Content |
|---|---|
| arXiv | [2111.09266](https://arxiv.org/abs/2111.09266) |
| Publication | **JMLR 24(210):1−55, 2023 (Journal)**, see <https://jmlr.org/papers/v24/22-0364.html>. **The local PDF is arXiv v5 (2026-01-24), 76 pages, with "24 (2023) 1-76" printed in the header, which is a longer, updated version than the published JMLR version** — the content of the two versions differs, see editor's note |
| Authors | Yoshua Bengio, Salem Lahlou, Tristan Deleu (three co-first authors), Edward J. Hu, Mo Tiwari, Emmanuel Bengio |
| Code | Not publicly available (purely theoretical paper, no code link provided in the original text) |
| Local repository PDF | `papers/2111.09266.pdf` · Chinese translation `papers_zh/2111.09266.zh.pdf` (not yet generated) |
| Reading Priority | **P0** · The precise formulation of internal flow degrees of freedom is here; the problem of the main OT line is defined by this paper |

**Numbering Warning (stated upfront, used repeatedly later)**: This paper has at least three sets of numbering in circulation — the early arXiv version (Prop. 3 / Corollary 1 / Prop. 6 / Prop. 10 cited by T03), the published JMLR 2023 version (55 pages), and the local arXiv v5 (76 pages). **This report exclusively cites according to the numbering of the local v5 PDF** (Prop. 8/10/14/16/18/19/21/23, etc.). When citing elsewhere, always include the proposition content, not just the number. See editor's note for correspondence.

#### 2. Core Contributions (Numbered as in Original Text)

**Proposition 16 (§2.4) — Three equivalent characterizations of Markovian flow.** The following three are equivalent: (1) \(F\) is a Markovian flow (Def. 15: \(P(s\to s'|\tau)=P(s\to s'|s)=P_F(s'|s)\)); (2) there exists a **unique** compatible \(\hat P_F\) such that all complete trajectories satisfy \(P(\tau)=\prod_{t=1}^{n+1}\hat P_F(s_t|s_{t-1})\), and \(\hat P_F=P_F\); (3) there exists a **unique** compatible \(\hat P_B\) such that \(P(\tau)=\prod_{t=1}^{n+1}\hat P_B(s_{t-1}|s_t)\), and \(\hat P_B=P_B\).

**Corollary 17 (§2.4) — Sampling theorem.** Starting from \(s_0\), iteratively sample according to \(P_F(\cdot|s)\) until \(s_f\). The probability of terminating at \(s\) is \(P_T(s)\). The first sentence of the proof is "the procedure terminates with probability 1, given that \(G\) is acyclic" — **acyclicity is used here**, which is where non-acyclic theories must be rebuilt later.

**Proposition 18 (§2.4) — Three parameterizations, also the main subject of this report.** Given a pointed DAG \(G\), a Markovian flow is **completely and uniquely** determined by any of the following combinations:

1. Total flow \(\hat Z\) + forward transitions \(\hat P_F(s'|s)\) on all edges;
2. Total flow \(\hat Z\) + backward transitions \(\hat P_B(s|s')\) on all edges;
3. **Terminal flows \(\hat F(s\to s_f)\) on all terminal edges + backward transitions \(\hat P_B(s|s')\) on all non-terminal edges**.

Item 3 is the entry point for the OT main line: the reward only fixes the first term, while the second term is completely free. The proof is constructive — define \(\hat Z:=\sum_{s\in Par(s_f)}\hat F(s\to s_f)\) from the terminal flows, extend \(\hat P_B\) to terminal edges \(\hat P_B(s|s_f):=\hat F(s\to s_f)/\hat Z\), which reduces to case 2.

**Proposition 19 (§2.5) — Sufficiency and necessity of flow matching condition.** A non-negative function \(\hat F\) (defined on states and edges) corresponds to a flow if and only if the flow matching condition Eq. (22) holds; in this case, it **uniquely** determines a Markovian flow
\[F(\tau)=\frac{\prod_{t=1}^{n+1}\hat F(s_{t-1}\to s_t)}{\prod_{t=1}^{n}\hat F(s_t)}\quad\text{(Eq. (23))}.\]
The original text specifically points out after the proof: Eq. (22) can be used to **recursively** define flows on all states — given \(Z\) and one of the forward (or backward) transitions, one can distribute along the DAG starting from \(s_0\) (or \(s_f\)). "A setting of particular interest, that will be central in Sec. 3, is when we are given all the terminal flows \(F(s\to s_f)\)".

**Definition 20 + Proposition 21 (§2.5) — Detailed balance.** \(\hat F\) (on states), \(\hat P_F\), \(\hat P_B\) jointly correspond to a flow if and only if
\[\forall s\to s'\in A,\quad \hat F(s)\hat P_F(s'|s)=\hat F(s')\hat P_B(s|s')\quad\text{(Eq. (26))},\]
and in this case, \(\hat P_F,\hat P_B\) are compatible (Def. 20). The selling point of DB relative to FM is clearly stated in the original text: **it does not involve summation over successors or predecessors**, thus applicable to cases with a very large number of successors or continuous state spaces (introduction to §2.5).

**Proposition 23 (§2.7) — Equivalence classes and unique Markovian representatives.** Define \(F_1\sim F_2\) if and only if they are identical on **all edge flows** (Def. 22). Then: two equivalent Markovian flows must be equal; and for any flow \(F'\), there exists a **unique** Markovian flow in its equivalence class. Fig. 4 provides explicit numerical examples: \(F_1,F_2\) are equivalent, \(F_3,F_4\) are equivalent, \(F_2,F_4\) are Markovian while \(F_1,F_3\) are not, and all four are identical in terms of terminal flows.

The purpose of this proposition is **dimensionality reduction**: learning a flow originally requires specifying \(|\mathcal T|\) numbers (exponential in the number of edges), but restricting to Markovian flows only requires specifying \(|A|\) edge flows (which must also satisfy Eq. (22)). It also tells us that "choosing internal flows" = "choosing edge flows", because edge flows uniquely determine a Markovian flow.

**Definition 24–26 + Examples 1–6 (§3.2) — Formal definition of GFlowNet and loss classification.** A flow parametrization is a triplet \((\mathcal O,\Pi,H)\): \(\Pi\) maps configurations to \(\Delta(\mathcal T)\), \(H\) is an **injective** map from \(\mathcal F_{Markov}(G,R)\) to \(\mathcal O\), and \(\Pi(H(F))\) is precisely the measure induced by \(F\). GFlowNet = \((G,R,\mathcal O,\Pi,H)\) (Def. 25). The flow-matching loss is defined as \(L(o)=0\iff o\in H(\mathcal F_{Markov}(G,R))\) (Def. 26), and classified by edge/state/trajectory decomposability. Three specific examples: Example 4 = FM loss (with smoothing constant \(\delta\), state-decomposable), Example 5 = DB loss (edge-decomposable), Example 6 = TB loss (trajectory-decomposable, attributed to Malkin et al. 2022).

**Eq. (38)–(39) — Formalization of off-policy correctness.** For edge-decomposable loss,
\[\min_{o\in\mathcal O}L(o)\iff\min_{o\in\mathcal O}\ \mathbb E_{(s\to s')\sim\pi_T}\big[L(o,s\to s')\big],\]
where \(\pi_T\) is **any full-support** distribution on \(A\). The same applies to state/trajectory decomposable cases. This is the precise form in the original text of the statement "GFlowNet can be trained arbitrarily off-policy": full support + decomposable + global minimum, all are indispensable.

**§4 Conditional Flows and Free Energy.** Def. 27 Free energy \(e^{-\mathcal F(s)}=\sum_{s'\ge s}e^{-E(s')}\); Def. 28 Conditional flow network; Def. 29 reward-conditional; Def. 30 state-conditional (subgraph \(G_s\) anchored at \(s\), requiring \(F_s(s'\to s_f)=F(s'\to s_f)\)); Prop. 31 Existence (constructive solution \(F_s(\tau):=F(C_\tau)+\frac1n F(U_{s'|s})\)); **Prop. 32**: \(F_s(s_0|s)=F_s(s)=\sum_{s'\ge s}F(s'\to s_f)=\exp(-\mathcal F(s))\); Cor. 33: \(P_T(s'|s)=\mathbb 1_{s'\ge s}e^{-E(s')+\mathcal F(s)}\).

**The counterexample in Fig. 5 is worth noting separately**: In the original flow, \(F(s_2)=4\), but the sum of terminal flows reachable from \(s_2\) is 6. The discrepancy comes from the part of the flow passing through \((s_0,s_1,s_5)\) – there is no order relation between \(s_1\) and \(s_2\). **Conclusion: The ordinary state flow \(F(s)\) is not the "sum of downstream rewards from \(s\)"**. To obtain marginalization, one must switch to state-conditional flow. This is one of the most common misuses in practice.

**Proposition 35 / 36 (§4.7) – Entropy and Mutual Information.** Train a second GFlowNet to match the entropy-transformed reward \(R'(s)=-R(s)\log R(s)\) (Def. 34, requiring \(R(s)<1\)), then \(H[S]=F'(s_0)/F(s_0)+\log F(s_0)\); the conditional version replaces \(s_0\) with \(s_0|x\); mutual information is obtained by subtracting the two (Eq. (57)).

**§5 GFlowNets on Sets and Graphs.** Def. 37 takes the state space as \(2^{\mathcal U}\cup\{s_f\}\), adding one element at each step, requiring \(Z=\sum_{s\in2^{\mathcal U}}R(s)<\infty\) (Eq. (58)); **Prop. 38**: The probability mass of all supersets of any set \(s\) is \(P_T(\mathcal S(s))=e^{-\mathcal F(s)}/Z=F(s|s)/F(s_0)\). Graphs are treated as "sets of two types of elements" (§5.2); §5.3 uses sets of "\((i,x_i)\) pairs" for marginalization of joint distributions; §5.4 discusses modular energy decomposition.

**§6 Continuous/Mixed Spaces.** Sums are replaced by integrals; the real difficulty lies in the output end being able to compute both density and sample simultaneously. Proposed approaches include: integrable normalization constants (Gaussian), cluster-ID mixing, autoregressive/normalizing flow, diffusion-style multi-step resampling, and direct parameterization of edge flows \(F((s^i,s^x)\to(s'^i,s'^x))\). The original text notes that Lahlou et al. (2023) completed a full continuous theory during the review period.

**Appendix A (Prop. 39–41) – Direct Credit Assignment.** \(\frac{d\log F(s')}{d\log F(s)}=P(s|s')\) (Prop. 39), \(\frac{d\log F(s')}{d\log F(s\to s')}=P_B(s|s')\) (Prop. 40); from these, two unbiased gradient estimators \(G_1,G_2\) and their convex combination \(G=\lambda G_1+(1-\lambda)G_2\) are constructed, which are unbiased in the limit where flows are matched (Prop. 41). The original text self-assesses: "something very close to policy gradient actually provides an asymptotically unbiased gradient," but this holds only when on-policy and flows are matched; otherwise, it is biased.

**Appendix C (Prop. 47–49 + Counterexample 50) – Stochastic Environments.** Transitions are split into "even state \(s\) → odd state \((s,a)\) → even state \(s'\)" (Def. 46), \(P_F(s_{t+1}|s_t)=\sum_{a_t}P(s_t\to s_{t+1}|s_t,a_t)\pi(a_t|s_t)\) (Eq. (79)). **Prop. 49**: In stochastic environments, any policy can yield a Markovian flow, but it **may not** achieve \(\hat F(s\to s_f)=R(s)\) (counterexample: the environment's transition probability to a positive reward state is 0). **Counterexample 50**: In stochastic environments, \(P_B\) cannot be freely chosen. That is, the "freedom of \(P_B\)" in §2.6 is a property **exclusive to deterministic environments**.

**Appendix D (Prop. 51–55).** \(V_{P_T}(s)=\frac{\sum_{s'\ge s}R(s')^2}{\sum_{s'\ge s}R(s')}\) (Prop. 52); policy improvement theorem (Prop. 53) and existence of optimal policy (Cor. 54); using a second flow matching \(R^2\), we can get \(V_{P_T}(s)=F'(s|s)/F(s|s)\) (Prop. 55). D.1 points out that by designing \(P_B\), GFlowNet can be made to prefer "constructing parts with high expected reward first."

**Appendices E–F.** Def. 56–59 introduce intermediate rewards and return-augmented states; Def. 60–61 outcome-conditioned / distributional GFlowNet; **Prop. 62**: A trained outcome-conditioned GFlowNet can **post-hoc** synthesize flows for any \(R=r\circ f\): \(F_{r\circ f}(A)=\sum_y r(y)F(A|y)\), at the cost of summing over the outcome space at runtime; Def. 63–64 Pareto GFlowNet.

#### 5. Assumptions and Applicability Boundaries

Positive statements of applicability:

1.  **Finite state space** (Def. 1 explicitly states \(S\) is finite) and **pointed DAG** (Def. 3). Single-source single-sink can be reduced from general DAGs by adding nodes, which is lossless.
2.  **Acyclic**. Places where acyclicity is used are verifiable: definition of partial order \(<\) (Def. 1/3), termination of Cor. 17, recursive assignment of Prop. 19, construction of Markovian representatives in Prop. 23. §3.3.1 provides the only relaxation: replacing states with time-stamped augmented states \((s_t,t)\), which automatically makes it acyclic.
3.  **Deterministic environment** (default in main text). Appendix C deals with stochastic environments and presents two negative results: Prop. 49 (may not achieve target terminal flow) and Counterexample 50 (\(P_B\) cannot be freely chosen).
4.  **Non-negative terminal rewards** \(R:S^f\to\mathbb R^+\) (Eq. (32)); set GFlowNet additionally requires \(Z<\infty\) (Eq. (58)); entropy estimation additionally requires \(R(s)<1\) (Def. 34 / Prop. 35).
5.  **Correctness only guarantees terminal distribution**: \(o\in H(\mathcal F_{Markov}(G,R))\Rightarrow P_T\propto R\) (after Def. 25). Internal flows are determined by the second free term in Prop. 18(3), and rewards impose no constraints on it.
6.  **Full support training distribution + global minimum + realizability** (combination of Eq. (39) and Def. 26). Remark 44 is the most direct statement in v5: in an amortized setting with shared parameters, \(L=0\) might be unattainable with finite capacity, and the actual optimum is a "compromise between conditions."
7.  **State flow is not equal to the sum of downstream rewards** (Fig. 5 counterexample). For marginalization, state-conditional flow must be used (Def. 30 + Prop. 32). This is the most common pitfall in implementation.

#### 6. Position in the GFlowNet × OT Main Line

##### 6.1 Predecessors and Successors

**Predecessor**: T00 (2106.04399). Section 1 of this paper states that it "provide an in-depth formal foundation and expansion" building upon Bengio et al. (2021). The FM loss becomes Example 4, and Prop. 1 of T00 (MaxEnt RL yields \(P_T\propto n(s)R(s)\)) is incorporated into §7.2 as a demarcation from regularized RL.

**Successors/Comparisons**: T03 (TB) is Example 6 within this paper's parameterization framework; SubTB (T05) is the hub version of Appendix A.2 of T03; the non-acyclic theory of T19/T36 aims to reconstruct the parts supported by acyclicity in §2 of this paper; the minimum flow and OT results of O07/O08 provide objective functions for the degrees of freedom left by Prop. 18(3) of this paper.

##### 6.2 Non-uniqueness of Internal Flow: Original Proposition List

This is the core retrieval task of this report, ordered by importance, with all sources indicated.

**(i) Proposition 18 (§2.4) Item 3 — Precise parameterization of degrees of freedom.**
> a Markovian flow on \(G\) is completely and uniquely specified by ... 3. the combination of the terminating flows \(\hat F(s\to s_f)\) for all terminating edges and the backwards transition probabilities \(\hat P_B(s|s')\) for all non-terminating edges \(s\to s'\in A^{-f}\)

The reward only specifies the first item; the second is completely free. **The question "how many degrees of freedom does the internal flow have" is answered here as: equal to the degrees of freedom of \(P_B\) on \(A^{-f}\).**

**(ii) Entire Section §2.6 "Backwards Transitions can be Chosen Freely" — The natural language statement in the original text.**
> What this means is that the terminating flows do not specify the flow completely, e.g., because many different paths can land in the same terminating state. The preference over such different ways to achieve the same final outcome is specified by the backwards transition probability \(P_B\) ... For example, we may want to give equal weight to all parents of a node \(s\), **or we may prefer shorter paths, which can be achieved if we keep track in the state \(s\) of the length of the shortest path to the node \(s\)**, or we may let a learner discover a \(P_B\) that makes learning \(P_F\) or \(F\) easier.

The bolded part is the only instance of "shortest" in Foundations. **The original authors already pointed out shortest paths as an internal flow selection principle**, but only as an example of \(P_B\) design, without formalization, objective function, or connection to OT. This is the direct starting point for O07/O08.

**(iii) Definition 22 + Proposition 23 (§2.7) — Equivalence classes at the edge flow level.** Two flows are equivalent if and only if their edge flows are identical everywhere; each equivalence class contains exactly one Markovian flow. Corollary: **"choosing an internal flow" is equivalent to "choosing an edge flow"**, and the feasible set of edge flows is given by the linear system of equations Eq. (22) from Prop. 19 plus terminal conditions—this is a polytope, precisely what linear programming can act upon. Fig. 4 shows \(F_1,\dots,F_4\) as four flows that are identical in their terminating flows, serving as a minimal numerical example of the same phenomenon.

**(iv) Proposition 19 (§2.5) — The feasible set is linear.** The flow matching condition Eq. (22) is a linear equality on edge flows; after adding the terminal constraint \(F(x\to s_f)=R(x)\), the feasible set is a polytope (this report's phrasing; the original only says "flow matching conditions"). This is the structural reason why OT papers can formulate internal flow selection as an LP.

**(v) Eq. (31) and its discussion (end of §2.5) — Degrees of freedom are not arbitrary.** \(P_B\) is only free on the manifold where "each state normalizes over its parent set"; once an explicit solution like Eq. (31) is required, it is equivalent to requiring flow matching itself. This prevents misinterpreting " \(P_B\) is free" as " \(P_B\) is an arbitrary function".

**(vi) Counterexample 50 (Appendix C.3) — Degrees of freedom disappear in stochastic environments.** "Whereas with a deterministic environment for the GFlowNet, one can freely choose \(P_B\) for non-terminal edges, it is not so for stochastic environments". Therefore, the entire OT main line implicitly assumes deterministic environments.

**(vii) Proposition 39 / 40 (Appendix A) — Differential characterization of degrees of freedom.**
\[\frac{d\log F(s')}{d\log F(s)}=P(s|s'),\qquad \frac{d\log F(s')}{d\log F(s\to s')}=P_B(s|s').\]
In the neighborhood where the flow is matched, these two equations indicate that "perturbing the flow somewhere propagates along \(P_B\) elsewhere". If one wants to compute gradients for minimum flow regularization, these are ready-made tools in the original text.

**(viii) §3.3.1 "Introducing Time Stamps to Allow Cycles" — The only place where cyclic cases and shortest path preference appear together.**
> Define the augmented state space \(S'=S\times\mathbb N\) ... With this augmented state space, we automatically avoid cycles. Furthermore, we may design or train the backwards transition probabilities \(P_B(s'_t\mid s'_{t+1}=(s_{t+1},t+1))\) to create a preference for shorter paths towards \(s_{t+1}\), as discussed in Sec. 2.6.

**(ix) Appendix D.1 "Preference for High-Reward Early Trajectory" — Another internal flow selection principle.** By weighting \(P_B\) with \(V((s_t,a_t))\), GFlowNet is biased towards "constructing parts with high expected reward early". This shows that Foundations already recognized that " \(P_B\) is a design interface that can carry preferences", but the examples given are reward-driven rather than geometrically driven.

##### 6.3 How Many Degrees of Freedom Are There? (Derived in This Report)

From Prop. 18(3), after fixing \(R\), the remaining free parameters are \(P_B(\cdot|s')\) for each \(s'\in S\setminus\{s_0,s_f\}\), each contributing \(|Par(s')|-1\) degrees of freedom. Note that non-terminal edges are precisely those whose head node \(s'\neq s_f\), and \(s_0\) has no parent nodes. Thus,
\[\dim=\sum_{s'\in S\setminus\{s_0,s_f\}}\big(|Par(s')|-1\big)=|A^{-f}|-\big(|S|-2\big).\]
Let's verify this with the diamond example from Appendix A.1 of T00: \(S=\{s_0,s_A,s_B,s_T,s_f\}\), \(|S|=5\), 4 non-terminal edges, so \(\dim=4-5+2=1\)—which is exactly the family \(u\in[0,r]\). **This formula provides a computable scale indicator for "internal flow degrees of freedom"**, which can be directly used to judge whether a certain environment value is worth applying OT regularization: when \(\dim=0\) (tree), OT has nothing to do.

##### 6.4 Translating the Minimum Flow Objective Back to the Language of Foundations (Derived in This Report)

On a DAG, each complete trajectory passes through any edge at most once. Therefore,
\[\sum_{e\in A}F(e)=\sum_{e\in A}\sum_{\tau\ni e}F(\tau)=\sum_{\tau}F(\tau)\,|\tau|=Z\cdot\mathbb E_{P}\big[|\tau|\big],\]
where \(|\tau|\) is the number of edges. Each complete trajectory has exactly one terminal edge, so the total flow on internal (non-terminal) edges is
\[\sum_{e\in A^{-f}}F(e)=Z\cdot\big(\mathbb E_P[|\tau|]-1\big)=\sum_{s\in S\setminus\{s_0,s_f\}}F(s),\]
The last equality holds because \((s_0,s_1,\dots,s_n,s_f)\) visits exactly \(n\) internal states and uses exactly \(n\) non-terminal edges.

Three corollaries:

1.  **"Minimum total internal flow" = "Minimum expected trajectory length"** (\(Z\) is fixed by \(R\) and does not participate in optimization). The objective used in O08 is this in the DAG special case.
2.  **Counting by internal edges and counting by internal state visits are exactly equal on a DAG** (not just differing by a constant). The two diverge only when cycles are present, because flow must be changed to expected visit counts.
3.  From Prop. 18(3), this objective can be directly written as a function of \(P_B\): first, derive all edge flows from \(P_B\) (recursively as in §3.2), then sum them up. **Thus, the minimum flow problem in OT within the Foundations framework is "minimizing expected trajectory length on the \(P_B\) manifold"**, without introducing any new objects. This is what this report considers most worth drafting as a proposition.

##### 6.5 What Is Still Missing Compared to O08

Foundations **cannot** directly derive O08's OT theorem; there are three gaps that must be clarified:

| Gap | Foundations' Setting | O08's Required Setting |
|---|---|---|
| Source side | Single \(s_0\), \(F(s_0)=Z\), no concept of "source distribution" | Fixed \(L(u)\) on the initial state set \(U\), \(\sum_u L(u)=\sum_x R(x)=1\) |
| Cycles | Strictly acyclic; only provides timestamp augmentation from §3.3.1 | Allows cycles, flow interpreted as expected visit counts |
| Objective | No preference regarding flow magnitude (degrees of freedom fully open) | Minimize total internal flow, achieving global optimum |

The second point requires particular caution: although timestamp augmentation \((s,t)\) transforms a cyclic graph into a DAG, it splits "different visit times to the same state" into different nodes. The shortest path cost \(d(u,x)\) on the original graph needs to be redefined in the augmented graph (Our assessment: the original text does not discuss this). That is to say, §3.3.1 is a **feasible but semantically altering** bridge, not a proof path for O08's conclusions.

##### 6.6 Why It Is Still a Necessary Prerequisite for the Main OT Line

In short: **The problem statement of the main OT line ("pick one among all reward-matching flows") is well-defined only after Prop. 18 + Prop. 23.** Without Prop. 23, the object "internal flow" is not unique at the trajectory level (\(F_1\) and \(F_2\) in Fig. 4 have the same edge flow but different trajectory flows); without Prop. 18(3), we don't know which parameters hold the degrees of freedom; without §2.6, we cannot determine whether "preference for short paths" is an imposition by later researchers or an interface already present in the original framework.

#### 7. Reusable Insights and Open Problems

1.  **Write min-flow as an optimization problem over \(P_B\)** (§6.4, point 3). Proposition draft: For a finite pointed DAG and fixed \(R\), \(\min_{P_B}\sum_{e\in A^{-f}}F_{P_B}(e)\) has the same optimal value as O08's edge flow LP; \(F_{P_B}\) is given by the reverse topological order recursion in §3.2. This can first be solved and compared on a \(4\times4\) hypergrid using `scipy.optimize.linprog` and direct projected gradient descent on \(P_B\).
2.  **Degrees of freedom dimension \(|A^{-f}|-|S|+2\) as an environment screening metric** (§6.3). Experimental draft: Calculate this number on several standard environments (hypergrid, bag, molecular fragment graphs) and check if it predicts the magnitude of "internal flow differences between different training objectives." It should always be 0 on trees, which can serve as a sanity check.
3.  **Design low-variance gradients for min-flow regularization using Prop. 39/40.** The \(G_2\) estimator in Prop. 41 already samples parent nodes using \(P_B(s|s_t)\); replacing \(L\) with a flow penalty term yields a ready-made estimator, but its unbiasedness is only valid in the neighborhood where "flow is matched"—after adding regularization, the optimum is no longer a reward-matching flow, so whether this premise still holds is a truly open question.
4.  **Use the entropy estimates from Prop. 35/36 as a diagnostic for internal flow.** Training a second flow matching \(R'=-R\log R\) yields \(H[S]\); applying the same technique at the trajectory level (calculating entropy for the trajectory distribution induced by \(P_B\)) can quantitatively characterize "how concentrated the minimum flow solution is." The original text did not make this generalization, but the free energy mechanism of Prop. 32 is readily available.
5.  **Remark 44 (capacity trade-off for amortized parameterization) is a direct warning for conditional OT.** If one wants to learn a family of OT plans that vary with the source distribution \(L\), when using a shared network, \(L=0\) may be unreachable under finite capacity, and the optimal solution is a compromise between different conditions. Any experiment on "conditional GFlowNet learning OT" must report the error distribution for each condition, not just the average.
6.  **The counterexample in Fig. 5 becomes a trap in the OT context.** If someone tries to use state flow \(F(s)\) to interpret "remaining transport cost from \(s\)," that is incorrect—\(F(s)\) is not a marginalization of downstream quantities; state-conditional flow (Prop. 32) must be used. This can be directly written as an implementation guideline: "do not do this."

## 2.4 T03 · Trajectory Balance

> **One sentence summary**: This paper elevates GFlowNet's training constraints from "each state" and "each edge" to "the entire complete trajectory," replacing layer-by-layer bootstrapping with a single equation \(Z_\theta\prod P_F=R(x)\prod P_B\), and proves that a zero-residual solution still yields \(P_T\propto R\). What it changes is not correctness but the **scale of credit assignment**: the terminal reward acts on the starting point in one step, at the cost of stochastic gradient variance and the necessity of running the entire trajectory. For the main OT line, it is the mother machine—the neural experiments in O08 add flow regularization on top of TB; at the same time, it provides the clearest empirical image (Fig. 1 right) of how internal flow degrees of freedom are "self-selected" by the optimizer.

| Field | Content |
|---|---|
| arXiv | [2201.13259](https://arxiv.org/abs/2201.13259) |
| Publication | **NeurIPS 2022 Main Conference** (Paper footnote: 36th Conference on Neural Information Processing Systems (NeurIPS 2022)). Local PDF is arXiv v3 (2023-10-04) |
| Authors | Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio |
| Code | Molecular domain <https://github.com/GFNOrg/gflownet/tree/trajectory_balance>; hypergrid and bit-sequence each have a gist (original footnotes 4, 5, 6) |
| This repository PDF | `papers/2201.13259.pdf` · Chinese translation `papers_zh/2201.13259.zh.pdf` (not yet generated) |
| Reading Priority | **P0** · It is the default training objective for GFlowNet and the baseline for all regularized GFlowNet-OT implementations |

#### 2. Core Contributions (Numbered as in Original Text)

First, we note the paper's restatement of two existing objectives (§2.2), which will serve as the basis for all subsequent comparisons:

- **Flow matching (Eq. (9)–(10), attributed to T00)**: The model \(F_\theta(s,s')\) approximates edge flows, \(P_F(s'|s;\theta)\propto F_\theta(s,s')\), with loss
  \(\mathcal L_{FM}(s)=\big(\log\frac{\sum_{(s''\to s)\in A}F_\theta(s'',s)}{\sum_{(s\to s')\in A}F_\theta(s,s')}\big)^2\). For terminal nodes, there is an additional term \(\mathcal L'_{FM}\) that pushes the inflow towards \(R(x)\). Stochastic gradients are computed by summing over states for \(t=1..n-1\) and the endpoint for trajectories sampled along \(\pi_\theta\) (Eq. (10)). The original text states that its correctness relies on "Proposition 10" from Foundations.
- **Detailed balance (Eq. (11)–(12), attributed to T02)**: The model outputs three types of quantities for each state: \(F_\theta(s)\), \(P_F(\cdot|s;\theta)\), and \(P_B(\cdot|s;\theta)\). The loss is \(\mathcal L_{DB}(s,s')=\big(\log\frac{F_\theta(s)P_F(s'|s;\theta)}{F_\theta(s')P_B(s|s';\theta)}\big)^2\). For terminal edges, there is an additional \(\mathcal L'_{DB}\). The correctness relies on "Proposition 6" from Foundations. The original text specifically points out that the Markovian flow uniquely determined by \(P_F\) and \(F_\theta(s_0)\) **may not be compatible with the \(P_B\) output by the model**—this is precisely the inconsistency that the DB loss aims to eliminate.

**Eq. (13)—trajectory balance constraint.** For a complete trajectory \(\tau=(s_0\to\cdots\to s_n=x)\), direct algebraic manipulation from Eq. (3)(4)(5) yields
\[Z\prod_{t=1}^{n}P_F(s_t|s_{t-1})=F(x)\prod_{t=1}^{n}P_B(s_{t-1}|s_t),\]
The only fact used is \(P(s_n=x)=F(x)/Z\).

**Eq. (14)—TB loss.** The model outputs \(P_F(\cdot|s;\theta)\), \(P_B(\cdot|s;\theta)\), and **a global scalar** \(Z_\theta\):
\[\mathcal L_{TB}(\tau)=\Big(\log\frac{Z_\theta\prod_{t=1}^{n}P_F(s_t|s_{t-1};\theta)}{R(x)\prod_{t=1}^{n}P_B(s_{t-1}|s_t;\theta)}\Big)^2 .\]
Stochastic gradients are computed for trajectories sampled along the training policy \(\pi_\theta\) (typically \(P_F\) or a tempered version thereof) (Eq. (15)). The complete algorithm is given in Algorithm 1.

**Proposition 1 (§3, proof in A.1)—Correctness.** Assume \(R\) is strictly positive on \(\mathcal X\).
(a) If \(P_F,P_B,Z_\theta\) come from a Markovian flow satisfying reward matching (Eq. (8)), then \(\mathcal L_{TB}(\tau)=0\) for all complete trajectories.
(b) Conversely, if \(\mathcal L_{TB}(\tau)=0\) for all complete trajectories, then the corresponding Markovian flow \(F_\theta\) satisfies Eq. (8), and \(P_F(\cdot|\cdot;\theta)\) samples proportionally to the reward.
The original text immediately adds: if \(\pi_\theta\) has full support and \(\mathbb E_{\tau\sim\pi_\theta}\mathcal L_{TB}(\tau)\) is globally minimized over all \((P_F,P_B)\) and \(Z\), then the conclusion also holds; \(R>0\) is only to avoid division by zero and can be relaxed with a smoothing constant (consistent with the approach in T00/T02).

**§3.1—Canonical choice for reward-matching flows.** Quoting the original text: "The constraint (8), in general, does not have a unique solution: if the underlying undirected graph of \(G\) has cycles, there may be multiple Markovian flows whose corresponding action policies sample proportionally to the reward. However, by the uniqueness properties, **for any choice of backward policy \(P_B\), there is a unique flow satisfying (8)**, and thus a unique corresponding forward policy \(P_F\) for states with nonzero flow. (See Fig. 1)" A natural default choice is for \(P_B(\cdot|s)\) to be uniform over all parent nodes of \(s\), i.e., \(1/\#\{s'\mid(s'\to s)\in A\}\). The motivation given in the original text for fixing \(P_B\) is engineering-related: it is difficult to construct a parent distribution model in the molecular domain that is invariant to molecular isomorphism.

**Secondary contribution: First empirical validation of DB.** §1 explicitly states: "As a secondary contribution, we perform the first empirical validation of the detailed balance training objective." This means that the DB loss proposed in T02 had not been experimentally validated before this work.

**Appendix A.2—Two generalizations.**
- **Subtrajectory balance (Eq. (19))**: For any partial trajectory \(\tau=(s_m\to\cdots\to s_n)\), \(F(s_m)\prod_{t=m}^{n-1}P_F(s_{t+1}|s_t)=F(s_n)\prod_{t=m}^{n-1}P_B(s_t|s_{t+1})\). Both sides are equal to "the sum of flows of complete trajectories passing through this subtrajectory" (Eq. (20)). **DB is a single-edge special case, and TB is a complete trajectory special case**; the model only needs to output state flows for a subset of states (referred to as "hubs" in the original text). DB corresponds to "all nodes are hubs," and TB corresponds to "only \(s_0\) is a hub."
- **Non-forward trajectories (Eq. (21))**: Starting from a terminal node \(s_n\), moving backward to a branching point \(s_1=s_1'\), and then moving forward to another terminal node \(s'_{n'}\), we have
\[R(s'_{n'})\!\!\prod_{t=1}^{n'-1}\!\!P_B(s'_t|s'_{t+1})\prod_{t=1}^{n-1}P_F(s_{t+1}|s_t)=R(s_n)\!\!\prod_{t=1}^{n-1}\!\!P_B(s_t|s_{t+1})\prod_{t=1}^{n'-1}P_F(s'_{t+1}|s'_t).\]
The original text emphasizes that **\(F(s_1)\) does not appear in this equation**, so this constraint can become a training objective that "does not require the model to output any state flows, not even \(Z\)". The derivation involves dividing the TB constraints of two trajectories that start from \(s_0\) and are identical up to \(s_1\).

**Appendix A.3 – Relationship to Variational Methods.** Fixing \(P_B\) and assuming \(\sum_x R(x)=1\), the on-policy TB gradient (Eq. (22)) and the Reinforce gradient of \(D_{KL}(P_F(\tau)\Vert R(x)P_B(\tau|x))\) (Eq. (24)) **differ by a constant in expectation**, because \(\mathbb E[\nabla_\theta\log P_F(\tau;\theta)]=0\) in Eq. (25). A further variance comparison shows that the difference in their variances is
\[-\mathbb E_{(\tau,x)\sim P_F}\Big[\big(\nabla_\theta\log P_F\,\nabla_\theta\log P_F^\top\big)\Big(1+2\log\frac{R(x)P_B(\tau|x)}{P_F(\tau;\theta)}\Big)\Big],\]
When the term in parentheses is always positive (especially near the solution where \(P_F(\tau)=R(x)P_B(\tau|x)\)), the TB estimator has lower variance. **Note that this is a local conclusion for the "neighborhood of the optimum", not a global one.**

#### 5. Assumptions and Applicability Boundaries

1.  **Finite DAG**, unique initial state \(s_0\), no outgoing edges from terminal states (footnote 1 provides conversion methods to align with T00 conventions).
2.  **Strictly positive rewards** (precondition for Prop. 1, used for division in Eq. (14)), can be relaxed with a smoothing constant.
3.  **Sufficient expressivity + global minimum + full support of training policy**: The sentence after Prop. 1 lists all three together. If any of these is missing, the conclusion only covers visited trajectories.
4.  **Must complete the entire trajectory** to compute a loss once (Algorithm 1, lines 3–4); cannot start training from an intermediate state. This is a concrete limitation compared to FM/DB.
5.  **\(Z_\theta\) is a global scalar** (unconditional case). Conditional GFlowNets require \(Z_\theta(x)\), which is not covered in this paper.
6.  **Variance argument is local**: The variance comparison in A.3 requires the parenthesized term \(1+2\log\frac{R(x)P_B(\tau|x)}{P_F(\tau;\theta)}\) to be positive, which the original paper states holds in the neighborhood of the solution. TB may not have lower variance far from the optimum.
7.  **The treatment of \(P_B\) is a design choice, not a theorem**: §3.1 only guarantees "each \(P_B\) corresponds to a unique flow", not that a uniform \(P_B\) is a good choice. The experiments in Fig. 2 precisely show that a fixed uniform \(P_B\) converges slower on large grids.

#### 6. Position in the GFlowNet × OT Main Thread

**Predecessors**: T00 (FM loss, hypergrid and molecular environments, PPO/MCMC baselines all adopted), T02 (notation, Markovian flow, uniqueness property, DB loss).

**Successors/Comparisons**: SubTB (T05, hub version of A.2 in this paper, known during review and cited as [18]); GFlowNets and Variational Inference (Malkin et al. 2022, [19]) and A Variational Perspective (Zimmermann et al., [31]) complete the connection from A.3; O08's neural experiments use **regularized TB** as the training objective.

**For the main thread of "internal flow selection = optimal transport", this paper contributes three things.**

**(1) It explicitly hands over the degrees of freedom to \(P_B\) (§3.1).** The sentence "for any choice of backward policy \(P_B\), there is a unique flow satisfying (8)", combined with the three plots in Fig. 1 (reward, \(P_F\) obtained with fixed uniform \(P_B\), \(P_F\) obtained by learning \(P_B\)), provides the most intuitive illustration that "the same terminal distribution can be realized by internal flows of completely different forms". What the OT main thread aims to do is to add another plot for "minimum flow \(P_B\)" to this figure.

**(2) It provides an empirical answer to "which internal flow the optimizer will pick when no regularization is added" (§5.1).** The original paper observes that when \(P_F,P_B\) are learned jointly, the model prefers L-shaped paths that "complete one coordinate before moving to another", and provides an explanation: "a constant distribution over two actions ('continue to the right' and 'terminate') can be modeled with higher precision over a large portion of the grid than the complex position-dependent distribution". **In other words, the choice of internal flow is determined by the model's representational capacity and numerical precision, not by any geometric principle.** This is the most direct motivation for OT regularization: without constraints, the chosen flow is one that is "easy for the network to fit", not one with "low transport cost". Incidentally, on a hypergrid, all monotonic paths have equal length, so the minimum flow principle degenerates here (Our assessment: this report's judgment, not discussed in the original paper), thus the L-shaped preference does not violate the shortest path.

**(3) It provides constraints that do not depend on state flow, or even on \(Z\) (A.2 Eq. (21)).** The "backward then forward" invariant for end-to-end paths only involves \(R\), \(P_F\), and \(P_B\). In the OT setting, where marginals on both source and target sides are fixed and total mass is normalized (\(Z=1\)), Eq. (21) is a family of trainable constraints **that do not introduce any additional network heads**. This point is worth recording as an implementation option in this repository.

**Interface and gap with O08.** TB is the base loss for O08's neural experiments, but TB itself has no preference for internal flows; O08 adds a flow regularization term and reports that "stronger flow regularization shortens paths but may increase terminal distribution bias". In TB's language, this trade-off is expected: the regularization term pushes the optimum away from the "TB zero-residual manifold", and Prop. 1 only holds at zero-residual points. **TB's theorems do not cover objectives with regularization**, which is a boundary that must be clearly stated.

#### 7. Reusable Insights and Open Problems

1.  **The extent to which Prop. 1 fails after adding min-flow regularization to TB can be quantified.** Proposition draft: Let the regularization objective be \(\mathbb E_\tau[\mathcal L_{TB}(\tau)]+\lambda\cdot\mathbb E_{P_F}[|\tau|]\) (using the identity from T02 Report §6.4, the second term is the total internal flow divided by \(Z\)). Provide an upper bound on the deviation of \(P_T^\lambda\) from \(R/Z\) as a function of \(\lambda\). Numerical scans can be performed on small hypergrids first, comparing with the qualitative consistency of "shorter paths, larger deviation" reported in O08.
2.  **\(\log Z_\theta\) is a ready-made conservation diagnostic in the OT setting.** When both source and target marginals are normalized, theoretically \(Z=1\). Thus, \(|\log Z_\theta|\) directly provides a reading of "whether total mass is conserved." In TB training, \(\log Z\) is already optimized separately with a higher learning rate, making it a nearly zero-cost monitoring metric.
3.  **Whether the variance advantage of A.3 is retained after adding regularization is a truly open problem.** The original paper's variance comparison relies on \(1+2\log\frac{R P_B}{P_F}>0\). At the minimum flow solution, \(P_F\neq R P_B/Z\), so the sign of the term in parentheses is no longer guaranteed. This can be directly formulated as a proposition to be verified.
4.  **Comparing the convergence speed of fixed uniform \(P_B\) with min-flow \(P_B\) is an inexpensive experiment.** Fig. 2 already shows that "fixed \(P_B\) converges slower than learned \(P_B\)". Min-flow regularization essentially adds constraints to \(P_B\), so it can be expected to also slow down convergence (a corollary of this report). Running the same set of metrics on a \(64\times64\) hypergrid with three types of \(P_B\) (uniform fixed, freely learned, flow regularized) can verify this corollary.
5.  **The design paradigm of the bit sequence experiment can be directly transferred to OT evaluation.** By changing \(k\), the trajectory length and action space are altered while **keeping the target distribution invariant**—this is precisely the experimental design needed to isolate the two factors: "transport cost" and "distribution fitting." The OT version can fix source/target marginals and only change the granularity of the graph.
6.  **The end-to-end constraint in Eq. (21) can be combined with MCMC-style local search.** The original text already points this out (end of A.2, citing [32]; and mentioning the special case of "one step back, two steps forward" used in Bayesian structure learning). In the OT context, this is equivalent to reallocating mass between two target points, which is a natural coupling correction operator.

## 2.5 T05 · SubTB(λ)

> **One sentence summary**: SubTB(λ) generalizes GFlowNet's consistency constraint from "single edges" and "entire trajectories" to "sub-trajectories of arbitrary length," using a geometric weight λ that only depends on length to continuously interpolate between DB and TB, achieving a better trade-off between gradient bias and variance. It is the culmination of the GFlowNet training objective lineage and the only training paper in this repository that articulates "what is the use of state flow \(F(s)\)" as a measurable proposition. On the GFlowNet × OT map, it stands on the side of "how the degrees of freedom of internal flow are implicitly selected by the training objective," serving as a necessary step towards "internal flow selection = optimal transport."

| Field | Content |
|---|---|
| arXiv | [2209.12782](https://arxiv.org/abs/2209.12782) (v3, 2023-06-03) |
| Publication | ICML 2023 **Main Conference** (Proceedings of the 40th ICML, PMLR 202, 2023) |
| Authors | Kanika Madan, Jarrid Rector-Brooks\*, Maksym Korablyov\*, Emmanuel Bengio, Moksh Jain, Andrei Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin (\* equal contribution, footnote on first page) |
| Code | No repository link provided in the original text; Appendix §B only states "based on publicly available code from Malkin et al. (2022)" |
| Priority for Reading | P1 — The conclusion itself is more engineering-oriented, but its interpretation of "state flow = learned expected estimate of the stochastic term in the TB loss" is the direct precursor to treating \(F(s)\) as a dual potential/value function later on. |

#### 2. Core Contributions (numbered as in the original text)

**(C1) SubTB constraint and its sufficiency (§2.3, Eq. (8)).**
Conclusion: The DB condition Eq. (6) holds for all actions **if and only if** the sub-trajectory balance condition Eq. (8) holds for all (not necessarily complete) trajectories.
The original text attributes this equivalence to §A.2 of Malkin et al. (2022); this paper's contribution is using it as a training objective.
Prerequisite: Requires a state flow function \(F(\cdot;\theta)\), with \(F(x;\theta)=R(x)\) enforced at terminal states.

**(C2) SubTB(λ) objective (§2.3, Eq. (9) + Eq. (11)).**
Conclusion: For all \(\binom{n+1}{2}=O(n^2)\) sub-trajectories (Eq. (10) defines \(\tau_{i:j}\)) of a sampled complete trajectory, a convex combination weighted by \(\lambda^{\,j-i}\) is taken as the loss.
Prerequisite: \(\lambda>0\) is a hyperparameter; \(\lambda=1\) corresponds to uniform weighting.
The original text explicitly states "other weighting schemes are possible and should be explored in future work"—this is an open question marked by the authors themselves.

**(C3) DB and TB as two limits (§2.3, paragraph after Eq. (11)).**
Conclusion: As \(\lambda\to 0^+\), Eq. (11) **exactly** degenerates to the average DB loss \(\mathcal L_{DB}(s_i\to s_{i+1})\) over all transitions in the trajectory;
as \(\lambda\to+\infty\), it degenerates to the TB loss \(\mathcal L_{TB}(\tau)\), where the corresponding relationship is \(Z_\theta=F(s_0;\theta)\).
This is the full strict meaning of "interpolation"—it is an interpolation at the level of **loss weights**, not at the level of estimators.

**(C4) Computational cost is not \(O(n^2)\) (§2.3 "Computational considerations").**
Conclusion: The gradients of Eq. (11) with respect to \(\log F(s_i;\theta)\), \(\log P_F(-\mid s_i;\theta)\), and \(\log P_B(-\mid s_i;\theta)\)
only require **one forward pass and one backward pass** through the neural network; the \(O(n^2)\) overhead only occurs in linear operations on the already computed log-flows and logits.
Therefore, SubTB has almost no additional computational cost compared to DB/TB—this is the practical reason it can be directly adopted.

**(C5) Bias-variance hypothesis and empirical evidence (§2.3 "Hypothesized benefits" + §4.1.1).**
Conclusion: The mini-batch gradient variance of SubTB(λ) is between that of DB and TB,
and its estimation accuracy for "large-batch TB gradients" is **superior to mini-batch TB itself** in the mid-training phase.
This is the most informative point in this paper, see Section 4.

**(C6) Acceleration due to generalization of state flows (§2.3 "Faster learning due to generalization of state flows").**
Conclusion: The scalar \(\log F(s;\theta)\) is easier to fit with high precision and generalize between states than high-dimensional policy logits,
especially important in environments where "the graph widens further from the starting point."
The original text adds a statistical fact: except for the hypergrid in §4.1 (and the largest hypergrids),
the number of terminal states in all experimental domains is **several orders of magnitude larger** than the total number of states seen during training. However, C6 uses "may come from" in the original text, which is a **hypothesis** with only indirect evidence.

#### 6. Position in the GFlowNet × OT Main Thread

- **Predecessors**: T00 (Flow Network based Generative Models, FM objective and hypergrid environment);
  T02 (GFlowNet Foundations, DB condition and "fixed \(P_B\) implies unique global optimum");
  T03 (Trajectory Balance, TB objective, and the fact that "the expectation of the TB gradient = the gradient of the trajectory distribution KL", cited in §4.1.1 of this paper from its §A.3).
- **Contemporaries/Competitors**: T10 (Shen et al., ICML 2023) tackles the same problem from another angle.
  T05 believes that "the degrees of freedom of the flow are a training dynamics problem, to be solved with better objectives";
  T10 believes that "the degrees of freedom of the flow are an inductive bias problem, to be solved with guiding distributions".
  T10 §4 explicitly comments on T05: it parameterizes state flow and assumes that generalizing state flow is beneficial, but **does not use the learned state flow for decision-making**.
  This comment marks the dividing line between the two papers.
  Footnote 4 of this paper also mentions contemporary Pan et al. (2023) (local credit / incomplete trajectories),
  and Hu et al. (2023) which applies incomplete trajectories to Bayesian posterior inference for combinatorial objects.
- **Contribution to the Main Thread**: SubTB transforms "what is the internal flow \(F(s)\)" from an accounting variable into an object with operational meaning—
  it is the **learned conditional expectation estimate** of the stochastic term in the TB loss (the paragraph around Eq. (12)(13)).
  In OT language, this is precisely the role of the Kantorovich dual potential:
  the potential function itself is not a transport plan; it is a **certificate** that decomposes the global balancing problem into locally verifiable conditions;
  Prop. 5.3 of O01 (discrete complementary slackness) describes the static version of the same thing.
  To understand O08's statement that "for a minimal flow GFlowNet with a fixed initial flow distribution, its optimal solution is a Kantorovich transport plan",
  one must first accept the interpretation of "\(F(s)\) as a potential/value function", and T05 is the first training paper to clearly articulate this interpretation.
- **Another Implicit Thread**: Eq. (8) states that "local (edge-wise) consistency ⟺ global (arbitrary sub-segment) consistency", which is isomorphic to the mechanism in O01 Prop. 6.23—
  under graph geodesic distance, the Lipschitz constraint for all pairs of points can be compressed to constraints **only on edges**, because summing along paths can recover the constraint for any pair of points.
  Both instances show how "local conditions are automatically globalized due to the path structure of the graph".

#### 7. Reusable Insights and Open Problems

1.  **The optimal value of λ systematically falls near 1, not at the endpoints** (Fig. A.2 + hyperparameter tuning results for six tasks).
    This suggests an optimal interpolation point determined by "trajectory length distribution × graph width".
    Possible experiment: On a hypergrid, fix \(H\) and sweep dimension \(d\), observe if optimal λ changes monotonically with average trajectory length.
    If so, a prior rule for λ could be fitted, saving an entire dimension of grid search.
2.  **"Mini-batch SubTB gradients are more similar to full-batch TB gradients than mini-batch TB gradients" is a formalizable proposition** (Fig. 4 right).
    Theorem draft: In a tabular setting, provide a bias-variance decomposition of \(\mathbb E\|g_{\mathrm{SubTB}}^{(k)}(\lambda)-g_{\mathrm{TB}}^{(\infty)}\|^2\) with respect to λ,
    and prove that there exists a finite \(\lambda^{*}>0\) such that it is strictly less than the value at \(\lambda\to\infty\). The original text only provides cosine similarity curves.
3.  **Truncating short sub-trajectories barely loses performance** (Fig. A.4, \(j-i\le 4\)).
    This extends the applicability of SubTB from "complete episodes" to "only observing fragments".
    The counterpart on the OT side is the mechanism in O01 Prop. 6.23 of "compressing Lipschitz constraints for all pairs of points to edge constraints".
    It is worth writing a strict correspondence: SubTB's truncation length ↔ the maximum geodesic range that local constraints on the graph can generate.
4.  **The generalization advantage of state flow has not been directly measured** (C6 is a hypothesis).
    Possible experiment: On a hypergrid, artificially freeze \(\log F(s;\theta)\) to the analytical ground truth \(F_F\) (algorithm given in §A.2),
    and compare the convergence curves with the learned flow version. If the gap is small, C6 is disproven, and all benefits of SubTB are attributed to variance reduction.
5.  **λ weighting is a geometric function of length, completely ignoring graph structure information**.
    Open problem: Replacing \(\lambda^{j-i}\) with a weight dependent on the difference in endpoint flow values \(|\log F(s_i)-\log F(s_j)|\),
    would this be equivalent to some adaptive Bregman projection step size?
    The original text lists "learnable sub-trajectory selection and weighting strategies" as the most interesting future direction in its conclusion.
6.  **The connection to variational methods has not been empirically validated**. The original conclusion explicitly states that "future work should empirically evaluate the connection between SubTB and
    the methods in Malkin et al. (2023) (GFlowNets and variational inference)".
    This is a clue marked by the authors themselves as incomplete, and it is also the shortest path to translating GFlowNet objectives into KL projection language, and then connecting to Sinkhorn alternating projections (O01 §9.1).

## 2.6 T10 · Training Diagnostics

> **One sentence summary**: The true contribution of this paper is not three training tricks, but a diagnostic—on enumerable biochemical benchmarks,
> GFlowNet **systematically underfits the target distribution**, consistently assigning excessively high probabilities to low-reward objects over time, a fact that common Spearman correlation fails to reveal.
> Building on this, it proposes "the quality of the flow distribution = generalization ability" as an evaluation criterion, and uses guided trajectory balance to allow users to **specify** where the flow should go.
> On the GFlowNet × OT map, it is the paper that elevates "the degrees of freedom of the internal flow" from a side effect to a **designable object**—
> and "using an exogenous criterion to select which of multiple solutions" is precisely the premise of OT.

| Field | Content |
|---|---|
| arXiv | [2305.07170](https://arxiv.org/abs/2305.07170) (v1, 2023-05-11) |
| Publication | ICML 2023 **Main Conference** (Proceedings of the 40th ICML, PMLR 202, 2023) |
| Authors | Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani |
| Code | <https://github.com/maxwshen/gflownet> (explicitly given in original Appendix §B) |
| Reading Priority | P1 — The methods section is optional reading, but §3 (evaluation metrics) and §5 (substructure credit assignment) are essential prerequisites for the main thread of this repository |

#### 2. Core Contributions (Numbered as in the Original Text)

**(C1) A set of enumerable evaluation schemes and "underfitting" diagnosis (§3, Remark 1).**
Remark 1 states: "A primary practical challenge during GFlowNet training is reducing the probability of sampling low-reward \(x\)." The approach is to design benchmarks where \(|\mathcal X|\) is enumerable, and use **reward samples** (instead of computing \(p_\theta(x)\)) to perform goodness-of-fit tests against the target reward distribution.
Prerequisite: \(\mathcal X\) must be enumerable, thus not applicable to real-scale tasks.

**(C2) Pointing out flaws in existing evaluation metrics (§3).** Several works use the Spearman correlation between \(\log p_\theta(x)\) and \(R(x)\) on a held-out set (specifically naming Madan et al., 2022; Nica et al., 2022); however, "when \(\log p_\theta(x)=cR(x)\), the correlation is 1.0 for **any** \(c>0\), but only \(c=1\) truly matches the target distribution." This is a purely logical argument, independent of experiments.

**(C3) Remark 2: Flow distribution determines generalization, generalization determines distribution matching (§4).** Original text: "The flow distribution is important for generalization, and generalization is important for matching the target distribution over \(\mathcal X\). Furthermore, given data \(\{x,R(x)\}\), the flow is generally underdetermined." This transforms "which flow is better" from an empty question into an assessable one.

**(C4) Remark 3 + Prop. 5.2/5.4: TB and MaxEnt underestimate important substructures (§5).** When each \(x\) has multiple trajectories, TB and MaxEnt GFlowNets **insufficiently** credit "substructures most responsible for \(R(x)\)" in how they allocate flow; such substructures exist when the reward function is compositional. Two quantifiable propositions are in Section 3.

**(C5) Guided trajectory balance (§6, Theorem 6.1).** Given any (potentially non-Markovian, potentially training-varying) guiding distribution \(p(\tau_{\to x})\), if constraint (3) holds for all \(x\) and all \(\tau_{\to x}\), then \(P_F\) samples \(x\) with \(R(x)/Z\)—having the **same asymptotic guarantee** as existing GFlowNet objectives.
Prerequisite: General guiding distributions are non-Markovian, which standard GFlowNets cannot satisfy everywhere, requiring two-stage optimization (see Section 3).

**(C6) Three composable modifications + an MDP design conclusion (§3, §4, §6, §7).** Prioritized replay training (PRT), relative edge flow parametrization (SSR), substructure guidance (Sub); and a counter-intuitive empirical conclusion: **autoregressive MDPs are not necessarily the best**—on SIX6, only "prepend/append MDP + Sub + PRT + SSR" can match the target mean.

#### 6. Position in the GFlowNet × OT Main Line

- **Predecessors**: T00 (treating MDPs as flow networks); T02 (DB condition, and Corollary 1 "TB's unique global optimum under fixed Markovian \(P_B\)" which is cited as a theorem in this paper—this is the pivot for two-stage GTB); T03 (TB objective, Eq. (1)); Zhang et al. (2022) (MaxEnt GFlowNet: when \(P_B\) is uniform, TB's unique global optimum is the Markovian flow with maximum flow entropy \(H[F]=\mathbb E_{\tau\sim P_F}\sum_{t=0}^{n-1}H[P_F(\cdot\mid s_t)]\)).
- **Contemporaries/Comparisons**: T05 (Madan et al., 2023). This paper §4 directly comments on it: "parameterizing state flows to learn from partial episodes, and assuming state flow generalization is beneficial, but their method **does not use learned state flows for decision-making**"—SSR precisely fills this gap. Both papers acknowledge that "flows are underdetermined," but diverge on whether underdetermination is a **harm** (T05: causes variance/bias issues) or a **resource** (T10: can be designed).
- **Contributions to the main line, three points**:
  1. **It turns "which flow to choose" into an explicit variational problem.** The solution set of TB is a polytope (all Markovian flows compatible with \(R\)). MaxEnt selects a point using entropy maximization, while Sub selects another using guided likelihood. Structurally, this is the premise of OT: the feasible set is a coupling polytope \(\mathcal U(a,b)\) (O01 Def. 3.1), and a point is chosen from it using an exogenous cost/regularization term. MaxEnt ↔ the \(\varepsilon\to\infty\) limit of entropy-regularized OT (O01 Prop. 8.10: \(P_\varepsilon\to a\otimes b\), the most diffuse coupling); Sub ↔ using a different reference measure for KL projection (O01 Prop. 8.9 states that the reference measure only differs by a constant when the marginals are fixed, but in **unfixed** cases, it genuinely changes the solution—which is precisely the effect T10 aims for).
  2. **Two-stage GTB is algorithmically alternating projection**: First, project \(P_B\) onto the guiding distribution (Eq. (4) is the squared log-ratio, an empirical KL proxy), then project \(P_F\) onto the "TB constraint with fixed \(P_B\)". This shares the same skeleton as Sinkhorn's two KL projections (O01 §9.1 Eq. (9.2), Prop. 9.5), with the difference that T10's two constraint sets are asymmetric and the first changes with \(X\).
  3. **Substructure credit assignment ↔ cost function design**. Prop. 5.2's \(\Theta(1/(n-k))\) states that under a uniform prior, flow is allocated by "path count" rather than "semantic importance." In OT, this corresponds to "using the number of paths in the graph as cost." O08 fixes the cost as the **graph-induced shortest path cost** (geometric, not counting), while T10's guiding distribution is a **learned** cost—two ways to fill the same slot.
- **Successors**: O08 (Your GFlowNet Secretly Learns an Optimal Transport Plan) and O07 (Learning Shortest Paths with GFlowNets) push "which flow to choose" to its limit: by adding the exogenous criterion of "minimizing total flow," the optimal solution is no longer arbitrary but precisely the Kantorovich transport plan. T10 is the step on this line that "first realized there was a choice."

#### 7. Reusable Insights and Open Problems

1.  **The criticism "Spearman is 1 for \(\log p_\theta=cR\) for any \(c\)" can be directly quantified**.
    Experiment: On enumerable benchmarks, simultaneously report Spearman, AD, mean error, and the true \(\mathrm{KL}(p_\theta\|p^*)\) or TV distance, to see when the four diverge.
    This paper only computed the correlation between AD and mean error (\(R^2=0.87\)), without calculating \(p_\theta\) itself—
    However, on SIX6 with \(|\mathcal X|=65{,}536\), this **is computable** (though it requires dynamic programming over \(2^{n-1}\) trajectories). This is the biggest and easiest gap to fill in this paper.
2.  **The Pólya urn model (Thm. C.7) provides a falsifiable prediction for TB training**: the proportion of flow absorbed by \(s^*\) follows a Beta-binomial distribution with known parameters.
    Experiment: In small-scale tabular environments, directly measure this distribution and check how much it deviates under neural parameterization.
    If the deviation is large, it suggests that "the rich get richer" effect is offset by generalization under function approximation—which would weaken the motivation for Sub.
3.  **\(\alpha=1\) being universally optimal (§B) is an overlooked finding**: the best approach is to **completely discard** \(P_B\) and only regress the guided likelihood,
    i.e., instead of learning a Markov backward policy, directly specify a non-Markovian prior.
    Compare with O01 §14.3: The endpoint reduction of Schrödinger bridges (Prop. 14.11) precisely shows that once the endpoint coupling is fixed,
    the optimal path law is a **mixture of reference bridges**—the reference dynamics can be any prior, not necessarily Markov-representable.
    It is worth writing a strict correspondence: GTB's guiding distribution ↔ Schrödinger problem's reference path law \(\mathcal R^\varepsilon\).
4.  **The fact that PRT becomes slower on Bag (13820 → 20575) is unexplained**. Bag's reward is discrete three-valued (0.01 / 10 / 30) and stochastic (75%/25%).
    Conjecture: Reward quantiles degenerate with discrete rewards, and top-\(\beta\) quantile sampling is equivalent to uniform sampling from some large set.
    Experiment: Replace PRT's quantile threshold with a reward-value-based threshold, and see if Bag recovers (the original takes top 10% to occupy 50% of the batch, §B).
5.  **SSR is effective despite being "not more expressive and less efficient," indicating it's purely an inductive bias gain**.
    Open question: Can SSR's inductive bias be written as an explicit regularization term added to the SA parameterization? The original text juxtaposes it with "optimal transport regularization" but doesn't provide a form.
    If it can be written as an explicit penalty for "similar states having similar action policies," it would directly fall within the framework of O01 §12.5 (metric learning and inverse OT).
6.  **The original open questions in this paper (§1 end, §8 end)**: "how to induce GFlowNets to learn better flows, thereby improving their ability to solve unnormalized density estimation problems,"
    and "how to best learn advantageous flow distributions remains an open question."
    In OT language, there is a nascent answer: add a cost functional to the flow space to make "optimal" well-defined.
    O08 chose shortest path cost, O07 chose expected trajectory length. T10 didn't go this far, but it asked the right question.

# Chapter 3: From DAGs to Cycles: Theory of Non-Acyclic GFlowNets

Two papers constitute the entire theoretical prerequisite for O07/O08. T19 establishes cyclic flow theory on general measurable spaces and diagnoses flow explosion; T36 reconstructs it into a computable form on finite discrete graphs, providing \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) and "minimum total flow" constrained optimization. This chapter contains the full text of two interpretation reports (notation, contributions, derivations, experiments, assumptions, positioning, insights).

## 3.1 T19 · A Theory of Non-Acyclic Generative Flow Networks (AAAI 2024)

> **One sentence summary**: This paper extends GFlowNet flow theory from DAGs to general measurable spaces (including continuous and cyclic), uses "0-flow" to uniformly characterize cycles and all "never-terminating mass," proves that ratio-based losses like FM/DB/TB will infinitely accumulate flow into cycles on cyclic graphs (Theorem 3), and proposes a family of difference-based stable losses (Theorem 4). On the GFlowNet × OT map, it is the first to state "R-flow set = an acyclic flow + cycle space" (Theorem 5 / Prop. 5) and "expected trajectory length ≤ total flow / total reward" (Theorem 2, Corollary 1) as theorems—these two are precisely the premises for O07/O08 to use "minimum total flow" as the objective function.

| Field | Content |
|---|---|
| arXiv | [2312.15246](https://arxiv.org/abs/2312.15246) (v1, 2023-12-23; this repository's PDF is this version, including Appendices A–C) |
| Publication | AAAI 2024 Main Conference (Proceedings of the AAAI Conference on Artificial Intelligence, 38(10): 11124–11131); 8-page conference version does not include appendices, appendix-level results only in arXiv version |
| Authors | Leo Maxime Brunswic, Yinchuan Li (corresponding), Yushun Xu, Shangling Jui, Lizhuang Ma (arXiv version attribution; AAAI version also lists Yiheng Feng, see editorial note) |
| Code | Not publicly available (original text provides no link; Appendix C.1 only states implementation using TensorFlow 2.4.4) |
| This repository's PDF | `papers/2312.15246.pdf` · Chinese translation `papers_zh/2312.15246.zh.pdf` |
| Reading Priority | P0: The first theoretical paper on non-acyclic GFlowNets; all corrections in T36 and the "minimum flow" objective in O07/O08 start from its notation and theorems |

#### 1. Problem Setup and Notation

**Graph Notation (Sec. 2).** A directed graph \(G\), state space \(S=S^*\cup\{s_0,s_f\}\), where \(s_0\) has no incoming edges, \(s_f\) has no outgoing edges, and it is a simple graph. Edgeflow is a non-negative assignment on edges \(F(s\to s')\ge 0\). Definition 1 has three levels:

- flow: satisfies the flow matching constraint Eq. (1): \(\forall s\in S^*,\ \sum_{s'\to s}F(s'\to s)=\sum_{s\to s'}F(s\to s')\);
- R-edgeflow: satisfies the reward constraint Eq. (2): \(\forall s\in S^*,\ F(s\to s_f)=R(s)\);
- R-flow: satisfies both.

The forward policy is induced by the flow, Eq. (3): \(P(s_{t+1}=s\mid s_t)=F(s_t\to s)/\sum_{s_t\to s'}F(s_t\to s')\). The sampling time is the last step before reaching \(s_f\), Eq. (8): \(\tau=\max\{t\mid s_t\ne s_f\}\), and \(s_\tau\) is the sampled object. Eqs. (4)–(6) are the squared log-ratio losses for FM, DB, and TB, where the expectation is taken over the path distribution sampled by the current policy.

**Differences from Standard DAG-GFlowNets.**

- Allows directed cycles: the set of trajectories is infinite, \(\tau\) is unbounded, and \(E(\tau)\) can be arbitrarily large or even infinite.
- The primary object is the flow (a measure): \(\pi_f,\pi_b\) are derived from the flow via Radon–Nikodym derivatives (Prop. 1), there is no step of "first fixing \(P_B\)", and the FM loss does not involve \(P_B\) at all.
- The only marginal constraint is the reward constraint Eq. (2): the source end \(F(s_0\to\cdot)\) is not fixed (only in the Cayley experiment, Sec. 5.2, it is artificially fixed to be uniform).

**Generalization to Measurable Spaces (Sec. 3.2, Table 1).**

- Edgeflow is a finite non-negative measure \(F\) on \(S\times S\); \(F_{\mathrm{out}}(A):=F(A\to S)\), \(F_{\mathrm{in}}(A):=F(S\to A)\); reward is a measure \(R\) on \(S\).
- The two constraints become measure equalities: \(F(\cdot\to s_f)=R\) and \(\mathbf 1_{S^*}F_{\mathrm{in}}=\mathbf 1_{S^*}F_{\mathrm{out}}\).
- "Edges" are generalized to a dominating measure \(\mu\): \(F\ll\mu\), \(\mu(s_f\to S^*)=\mu(S\to s_0)=0\), \(\mu(s_f\to s_f)=1\).
- Forward kernel \(\pi_f^F(\cdot\to A)=dF(\cdot\to A)/dF(\cdot\to S)\) (Eq. (23)), \(F=F_{\mathrm{out}}\otimes\pi_f=F_{\mathrm{in}}\otimes\pi_b\) (Prop. 1).
- Trajectory flow \(F^\otimes\) is \(F_{\mathrm{out}}(s_0)\) multiplied by the path distribution from \(s_0\) by repeatedly applying \(\pi_f\) until \(s_f\).
- Table 1 compares the notation systems of Bengio et al. (DAG), Lahlou et al. (topological space), and this paper (measurable space, row by row). The last row is path length: on DAGs \(\tau\le\#S\), in Lahlou's framework \(\tau\le\tau_{\max}\), and in this paper only \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\).

**0-flow, Subflow, Stability (Sec. 3.1).** For a cycle \(\gamma=(s_1\to\cdots\to s_\ell\to s_1)\), the indicator flow \(\mathbf 1_\gamma\) is conserved and has zero terminal flow. Definition 2: A 0-flow is an R-flow with \(R=0\). \(F_0\) is a subflow of \(F\) if \(F_0\le F\) (edge-wise). Definition 3: A loss \(L\) is stable if for any 0-flow \(F_0\), \(L(F_1+F_0,\dots,F_p+F_0)\ge L(F_1,\dots,F_p)\). The \(E(\tau)\) in this paper and \(\mathbb E[n_\tau]\) in T36 are the same quantity.

#### 2. Core Contributions (Numbered as in Original Text)

**C1. Flow Trapping Cycle Mechanism and Stability Definition (Sec. 3.1, Definition 3, Lemma 1).**
Conclusion: On cyclic graphs, gradient descent for FM/DB/TB converges to edgeflow that is infinite along cycles, \(E(\tau)\to+\infty\).
Premise: The loss only considers the ratio of inflow/outflow. Lemma 1 provides a sufficient condition for stability: for every 0-flow \(F_0\) that is also a subflow of each \(F_i\), the directional derivative \(\partial_{F_0}L\ge 0\).

**C2. Stable Regularization Limit is Acyclic (Theorem 1, Eq. (7)).**
Conclusion: For \(L_\alpha=L+\alpha\mathcal R\), if we take \(\alpha_n\to0^+\), and the sequence of minimal R-edgeflows for \(L_{\alpha_n}\) converges to a flow, then the limit is an acyclic R-flow.
Premise: \(L\) is stable, \(\alpha>0\), and \(\partial_{F_0}\mathcal R>0\) holds for all 0-subflows. The appendix version (Appendix B) rephrases the condition as "S is at most countable + \(L_{\mathrm{FM}}\) is a strong FM loss + \(\partial_{F_0}L_{\mathrm{FM}}\ge0\)" and removes the convergence assumption.

**C3. Measurable Space Framework and Sampling Theorem (Sec. 3.2, Theorem 2, Eq. (9)).**
Conclusion: For an R-flow with \(R\ne0\), \(\tau\) is almost surely finite, \(s_\tau\sim R/R(S^*)\), and \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\).
Premise: \(F\) is a finite measure. This is Bengio's sampling theorem in a measure-theoretic version without the acyclicity assumption, and it includes an upper bound on the total flow for the expected length.

**C4. Existence of Acyclic 0-Flows (Sec. 3.2 "Acyclic 0-Flows").**
The irrational angle rotation \(\pi_f(z)=e^{2i\pi\theta}z\) on the unit circle never closes but never terminates, representing a 0-flow that contains no cycles. Therefore, in continuous spaces, "controlling cycles" is insufficient; all 0-flows must be controlled. This is the reason why this paper uses 0-flows instead of cycles as the stability primitive.

**C5. Unified Representation of Three Loss Types and Instability Theorem (Sec. 4.1, Eq. (10)–(16), Theorem 3).**
Conclusion: \(\mathrm{div}_{g,\nu}(\alpha,\beta)=\int g(d\alpha/d\beta)\,d\nu\); FM = \(\mathrm{div}_{g,\nu_{\mathrm{state}}}(F_{\mathrm{in}},F_{\mathrm{out}})\), DB = \(\mathrm{div}_{g,\nu_{\mathrm{edge}}}(F_f,F_b)\), TB = \(\mathrm{div}_{g,\nu_{\mathrm{path}}}(F_f^\otimes,F_b^\otimes)\), with \(g=\log^2\) recovering Eq. (4)–(6). Theorem 3: If \(\mathrm{div}_f\) is a proper f-divergence, then \(L_{\mathrm{FM},f},L_{\mathrm{DB},f}\) are unstable, with the only exception being total variation; \(L_{\mathrm{FM},g,\nu},L_{\mathrm{DB},g,\nu},L_{\mathrm{TB},g,\nu}\) are all unstable.
Premise: \(f(x)\ge0\) and is zero only at \(x=1\) (which excludes KL), plus the regularity conditions from Appendix B.1.

**C6. Stable Loss Family (Sec. 4.2, Eq. (17)–(18), Theorem 4, Example 1–3).**
Conclusion: By replacing ratio-type losses with difference-type losses \(\Delta_{f,g,\nu}(\alpha,\beta)=\int f(\alpha-\beta)\,g(\alpha,\beta)\,d\nu\), under the five conditions of Theorem 4, \(L_{\mathrm{FM},\Delta},L_{\mathrm{DB},\Delta}\) are stable for R-edgeflows. Example 1 = stable FM (Eq. (19)), Example 2 = stable DB (Eq. (20)), Example 3 = stable CFlowNet (Eq. (21)).
TB has no stable version; the paper explicitly states the reason is that \(F\mapsto F^\otimes\) is nonlinear.

**C7. Sampler Flow and Total Flow Identity (Appendix A.3, Definition 5, Prop. 2, Corollary 1).**
The flow \(\bar F\) actually implemented by the sampler is the expected visit count flow; \(\bar F\le F\), and \(E(\tau)=\bar F(S^*\to S^*)/R(S^*)\le F(S^*\to S^*)/R(S^*)\).

**C8. Structure of Flow Spaces on Graphs (Appendix A.4–A.5, Prop. 3–6, Theorem 5, Corollary 2–3, Lemma 2).**
On a connected graph, \(\mathcal F_R=F+H^1(G)\) (the affine space of R-flows is spanned by the cycle space), \(\mathcal F_R^+=\mathcal F^+_{R,\mathrm{acyclic}}+H^1_+(G)\) (every non-negative R-flow = acyclic R-flow + non-negative combination of directed cycles), and all 0-flows are cyclic. Any edgeflow can be decomposed into \(F=F_0+F_{\min}\) (maximal 0-subflow + minimal flow), but the decomposition is not unique (Example 7.1).

**C9. Experiments (Sec. 5).** Hypergrid (±step transitions), Cayley graph \(S_{20}\), continuous Point-Robot-Sparse, verifying flow explosion under unstable losses and boundedness under stable losses.

#### 3. Key Points of Methods and Theoretical Derivations

##### 3.1 Expected Visit Flow: Sampler Flow \(\bar F\) and "Zombie Flow"

Definition 5 (Eq. (24)): For an edgeflow \(F\) with \(E(\tau)<+\infty\), let
\[
\bar F_{\mathrm{out}}(A):=F_{\mathrm{out}}(s_0)\times E\big(\mathrm{card}\{t\in\mathbb N\mid s_f\ne s_t\in A\}\big),\qquad \bar F:=\bar F_{\mathrm{out}}\otimes\pi_f .
\]
That is: run the forward chain from \(s_0\) until absorption, count the expected number of times it falls into \(A\), and then multiply by the source flow. When \(\bar F=F\), \(F\) is said to be exactly sampled. Prop. 2: \(\bar F\) is a flow if and only if \(E(\tau)<+\infty\); if \(F\) is an R-flow, then \(\bar F\) is also an R-flow and \(\bar F\le F\).

Why \(\bar F\le F\), and what is the difference? Lemma 3 provides an equivalent characterization of an R-edgeflow being a flow:
\[
F_{\mathrm{out}}\pi_f-F_{\mathrm{out}}=R(S^*)\,\delta_{s_f}-F_{\mathrm{out}}(s_0)\,\delta_{s_0}.
\]
Repeatedly applying this for telescoping sums (Appendix B, proof of Prop. 2):
\[
\mathbf 1_{S^*}\bar F_{\mathrm{out}}=\mathbf 1_{S^*}\sum_{k\ge0}\big(F_{\mathrm{out}}\pi_f^k-F_{\mathrm{out}}\pi_f^{k+1}\big)=\mathbf 1_{S^*}\Big(F_{\mathrm{out}}-\lim_{k}F_{\mathrm{out}}\pi_f^k\Big).
\]
The difference term \(\lim_kF_{\mathrm{out}}\pi_f^k\) is written as \(\nu+\beta\delta_{s_f}\) in the proof of Prop. 8, where \(\nu\pi_f=\nu\) and \(\nu\otimes\pi_f(S^*,s_f)=0\): this is a mass that is invariant under \(\pi_f\) and can never leak to \(s_f\). \(F-\bar F\) is this "conserved but unsamplable" 0-flow.

When is \(\bar F<F\) (Our assessment:, based on the above equation):

- On a finite graph, if \(F\) is strictly positive on all edges and every state can reach \(s_f\) ⇒ the forward chain is an absorbing chain ⇒ \(\nu=0\) ⇒ \(\bar F=F\).
- If there is flow on some cycle, but the edge flow entering it from the \(s_0\)-reachable set is zero ⇒ the forward chain never enters it ⇒ all flow on that cycle is zombie flow. This violates T36 Prop. 3.7's condition that "\(F>0\) on all edges".
- Irrational rotation in continuous space (C4): mass wanders forever, \(\nu\ne0\).

This is precisely why T36 uses "positivity + Assumption 3.1" to obtain the uniqueness of "flow = expected number of visits": in T36's world, zombie flow does not exist.

##### 3.2 Absorption Condition and Total Flow Upper Bound (Proof Chain for Theorem 2: Lemma 3 → Prop. 7 → Prop. 8 → Corollary 1)

- Prop. 7 (\(\tau\) is almost surely finite): From Lemma 3, \(P(\tau>k)=\delta_{s_0}\pi_f^k(S^*)=\big[F_{\mathrm{out}}\pi_f^k(S^*)-F_{\mathrm{out}}\pi_f^{k+1}(S^*)\big]/F(s_0)\). The sequence \(F_{\mathrm{out}}\pi_f^k(S^*)\) is non-increasing and non-negative, so it converges, and the difference between adjacent terms tends to zero, thus \(P(\tau>k)\to0\). Absorptivity is not assumed but derived from "F is a finite measure and \(R\ne0\)": finite total flow does not allow mass to loop indefinitely in \(S^*\) with positive probability.
- Prop. 8 (\(s_\tau\sim R/R(S^*)\)): Summing \(P(s_\tau\in A)=\big(\sum_k\delta_{s_0}\pi_f^k\big)\otimes\pi_f(A\times\{s_f\})\) telescopically yields \(\frac{1}{R(S^*)}\big[F(A\times\{s_f\})-\nu\otimes\pi_f(A\times\{s_f\})\big]=R(A)/R(S^*)\), where the second term is zero because \(\nu\) does not leak to \(s_f\).
- Corollary 1: Each step lands exactly in some state in \(S^*\), so \(E(\tau)=\bar F(S^*\to S^*)/R(S^*)\). Then, from \(\bar F\le F\), we get the inequality in Theorem 2.
- T36 Prop. 3.12 later points out that under its positivity assumption, \(\bar F=F\), so this inequality itself becomes an equality.

Comparing with T36's starting point: T19 assumes flow is a finite measure and derives absorption; T36 assumes \(P_B>0\) and connectivity and derives finite flow. The two paths converge on finite positive graphs.

##### 3.3 Flow Explosion: Why Ratio-Based Losses Are Inherently Unstable

The four-state example from Sec. 3.1: a chain \(s_0\xrightarrow{f_1}A\xrightarrow{f_2}B\xrightarrow{f_3+c}C\xrightarrow{1}s_f\) with an additional back-edge \(C\to B\) (flow \(c\)), and parameters \(\theta=(f_1,f_2,f_3,c)\). The flows satisfying the reward constraint are precisely \(c\ge0\), \(f_1=f_2=f_3=1\). However, \(L_{\mathrm{FM}}\) can be minimized by letting \(c\to+\infty\) and \(f_i\to0\), with \(\partial_cL_{\mathrm{FM}}(F^\theta)<0\). The original paper explicitly states that DB and TB behave similarly.

Editorial note: (Numerical values are not from the original text, but are used to clarify the mechanism.) At state \(C\), the FM residual is \(\log^2\frac{1+c}{f_3+c}\). If we fix an incorrect \(f_3=1/2\) and only decrease \(c\):

| \(c\) | 0 | 1 | 4 | \(\to\infty\) |
|---|---|---|---|---|
| \(\log^2\frac{1+c}{1/2+c}\) | 0.480 | 0.083 | 0.011 | 0 |

The loss monotonically decreases to 0. The mismatch \(f_3\ne1\) is never corrected, and the termination probability of the induced policy \(P(s_f\mid C)=\frac{1}{1+c}\to0\).

General mechanism (beginning of Sec. 4.2): \(t\mapsto\frac{d(F_1+tF_0)}{d(F_2+tF_0)}\) approaches 1 as \(t\to\infty\). The proof of Theorem 3 (Appendix B.1, Lemma 4) calculates this as a directional derivative: let \(F_2=\alpha F_1\), \(F_0=\beta F_1\),
\[
\varphi(t)=\int g\Big(\tfrac{\alpha+t\beta}{1+t\beta}\Big)(1+t\beta)\,dF_1,\qquad \varphi'(0)=\int\psi(\alpha-1)\,dF_0,\quad \psi(x)=g(1+x)-x\,g'(1+x).
\]
If \(g\) is convex and \(g(1)=0\), then \(\psi\le\psi(0)=0\), which implies \(\varphi'(0)\le0\): the loss does not increase along any 0-subflow direction. Equality holds for all \((F_0,F_1,F_2)\) if and only if \(g\) solves \(y+(x-1)y'=0\), i.e., \(g\propto|1-x|\) – this is the source of the total variation exception: TV measures differences, and adding common mass does not change the differences. For \(g=\log^2\) (the case in Lemma 5, where \(\psi(x)=-x\,g'(1+x)\)), there exists \(F_2\) such that the derivative is strictly negative, so the three classical losses in Eq. (4)–(6) are all unstable.

TB is slightly weaker (Lemma 6): under conditions such as \(\mathbf 1_{S^*\times S^*}\mu\ll F_0\), \(dF_f/dF_b\) being bounded, and \(\int g(e^{\xi\,\mathrm{len}(s)})d\nu<\infty\),
\[
\lim_{c\to\infty}L_{\mathrm{TB},g,\nu}\big((F_f+cF_0)^\otimes,(F_b+cF_0)^\otimes\big)=\int g\Big[\tfrac{dF_f}{dF_b}(s_0\to s)\,\tfrac{dF_f}{dF_b}(s'\to s_f)\Big]d\omega(s,s'),
\]
only the ratio of the first and last edges remains; if the implementation forces the first and last flows to be consistent (as in the hypergrid TB implementation in this paper), the limit is 0. Infinitely adding flow along a cycle can drive TB to 0, but this limit might be larger or smaller than when \(c=0\). Therefore, it can only be said that "instability occurs in some cases" (Sec. 6.2: TB "may be stable for small flows").

A second deterioration comes from on-policy learning: \(\nu_{\mathrm{state}},\nu_{\mathrm{edge}},\nu_{\mathrm{path}}\) are sampled by the current policy. The more flow is trapped in a cycle, the greater the weight of the "ratio approaching 1" pseudo-zero residual in the loss. As described in Sec. 5.1: \(F(s_t\to s_f)\ll F(s_t\to s')\), even if the reward is high, the termination probability is suppressed by the cyclic flow, and the chain continues to wander (Figure 1, 3).

##### 3.4 Family of Training Losses: Ratio-type vs. Difference-type (Theorem 3, Theorem 4, Theorem 1)

The stability proof for the difference-type \(\Delta_{f,g,\nu}\) is only two lines long: after adding \(t\gamma\), \(f(\alpha+t\gamma-\beta-t\gamma)=f(\alpha-\beta)\) is independent of \(t\), leaving only \(g(\alpha+t\gamma,\beta+t\gamma)\). The condition \(\partial_{(1,1)}g\ge0\) ensures it is non-decreasing (Lemma 8). \(f\ge0\), \(f(x)=0\Leftrightarrow x=0\), and \(g\ge1\) ensure that \(\Delta\) is distance-like (Lemma 7). The five conditions for Theorem 4 are: \(f\ge0,\ g\ge1\); \(f(x)=0\Leftrightarrow x=0\); \(f,g\) are continuous and piecewise \(C^1\); \(f\) is decreasing on \(\mathbb R^-\) and increasing on \(\mathbb R^+\); \(\partial_{(1,1)}g\ge0\). Appendix B.2 also requires \(R\ll\nu\) and \(dR/d\nu\) to be bounded.

| Loss | Form (Source) | Comparison Object | Scale | Stability |
|---|---|---|---|---|
| FM | \(\mathrm{div}_{g,\nu_{\mathrm{state}}}(F_{\mathrm{in}},F_{\mathrm{out}})\), Eq. (11); \(g=\log^2\) is Eq. (4) | State inflow vs. outflow | Ratio | Unstable (Theorem 3, Prop. 9) |
| DB | \(\mathrm{div}_{g,\nu_{\mathrm{edge}}}(F_f,F_b)\), Eq. (12); Eq. (5) | Edge forward flow vs. backward flow | Ratio | Unstable (Theorem 3, Prop. 10) |
| TB | \(\mathrm{div}_{g,\nu_{\mathrm{path}}}(F_f^\otimes,F_b^\otimes)\), Eq. (13); Eq. (6) | Trajectory forward flow vs. backward flow | Ratio | Unstable (Theorem 3; Lemma 6 only gives partial conclusions) |
| f-divergence FM/DB/TB | Eq. (14)–(16) | Same as above | Ratio | Unstable, only exception TV (Theorem 3) |
| Stable FM | \(\Delta_{f,g,\nu_{\mathrm{state}}}(F_{\mathrm{in}},F_{\mathrm{out}})\), Eq. (17); Example 1 = Eq. (19) | State inflow vs. outflow | Difference | Stable (Theorem 4) |
| Stable DB (SDB) | \(\Delta_{f,g,\nu_{\mathrm{edge}}}(F_f,F_b)\), Eq. (18); Example 2 = Eq. (20) | Edge flow difference | Difference | Stable (Theorem 4) |
| Stable CFlowNet | Eq. (21), using densities \(f_i,f_o\) instead of \(F_{\mathrm{in}},F_{\mathrm{out}}\) | Density difference | Difference | Stable (Example 3, per Theorem 4) |
| Stable TB | — | — | — | Not provided in the original text |

Example 1 (Eq. (19)): \(f(x)=\log(1+\epsilon|x|^\alpha)\), \(g(x,y)=(1+\eta(x+y))^\beta\),
\[
L=E\sum_{t=1}^{\tau}\log\!\big[1+\varepsilon\,|F_{\mathrm{in}}(s_t)-F_{\mathrm{out}}(s_t)|^\alpha\big]\cdot\big(1+\eta(F_{\mathrm{in}}(s_t)+F_{\mathrm{out}}(s_t))\big)^\beta ,
\]
The recommended values are \((\alpha,\beta,\epsilon,\eta)=(2,1,0.001,1)\), because convexity facilitates the convergence of this underdetermined linear problem. This continues the design intent of Bengio's original FM, where \(\log(1+\cdot)\) suppresses the dominance of nodes far from satisfying conservation in training. Example 2 (Eq. (20)) replaces the difference with \(|F^f(s_t\to s_{t+1})-F^b(s_t\to s_{t+1})|\) and the weight with \((1+\eta F_{\mathrm{out}}(s_t))^\beta\), where \(F^f=F_{\mathrm{out}}(s_t)\pi_f\) and \(F^b=F_{\mathrm{out}}(s_{t+1})\pi_b\)—this is the loss denoted as SDB in T36.

The proof of Theorem 1 (Appendix B): If a minimizer \(F\) of some \(L_\alpha\) is not a minimal flow, use Prop. 3 to decompose \(F=F_0+F_{\min}\), where \(F_0\ne0\). Minimality requires \(0=\partial_{F_0}L_\alpha=\partial_{F_0}L+\alpha\,\partial_{F_0}\mathcal R\), but the right side is \(\ge0+\alpha\cdot(\text{positive number})>0\), which is a contradiction. Thus, each \(F_n\) is a minimal flow. Lemma 2 states that the set of minimal flows on at most a countable space is closed under narrow convergence, so the limit is still minimal, and minimal \(\Rightarrow\) acyclic. The comments after Theorem 1 directly state that "regularization that controls the total edgeflow magnitude (e.g., the norm of the edge flow matrix) may help kill cycles and shorten expected sampling times"—this is the first appearance of the "minimal flow principle" in the literature.

##### 3.5 Correspondence with DAG Theory: Geometry of Flow Spaces on Graphs (Appendix A.5)

Viewing a graph as a countable discrete space dominated by the counting measure \(\mu_E\), A.5 yields:

- Prop. 6: Generalized R-flows form a closed affine subspace, with the directional space being generalized 0-flows; R-flows (non-negative) are closed convex sets.
- Theorem 5 (Prop. 4): For a connected graph and summable \(R\), \(\mathcal F_R=F+H^1(G)\) and \(\mathcal F_0=H^1(G)\), where \(H^1(G)\) is the closure in \(\ell^1(E)\) of the space spanned by indicator flows of topologically simple cycles (edges can be traversed forwards or backwards). The original text states this can be derived from Kalpazidou (2007) Theorem 3.3.1; the construction for non-emptiness is to choose a path \(\gamma_e\) for each terminal edge \(e=s\to s_f\) and set \(F=\sum_eR(e)\mathbf 1_{\gamma_e}\).
- Prop. 5: \(\mathcal F_R^+=\mathcal F^+_{R,\mathrm{acyclic}}+H^1_+(G)\), where \(H^1_+(G)\) is the closure of non-negative combinations of directed simple cycles. Corollary 3: All 0-flows are cyclic.
- Prop. 3: Any edgeflow has a maximal 0-subflow (Zorn's Lemma), \(F=F_0+F_{\min}\); maximal 0-subflows are not unique (Example 7.1 provides an example with two different maximal 0-subflows, red and blue).
- Lemma 2: On at most countable spaces, the set of acyclic edgeflows and minimal edgeflows is closed under narrow convergence; for edgeflows, "minimal ⇒ acyclic"; for flows, "minimal ⇒ exactly sampled"; on uncountable spaces, there exist acyclic, exactly sampled but not minimal flows.

Translated into DAG language: On a DAG, there are no directed cycles, so \(H^1_+(G)=\{0\}\), which implies \(\mathcal F_R^+=\mathcal F^+_{R,\mathrm{acyclic}}\). Every R-flow is automatically "acyclic / minimal / exactly sampled"—but \(\mathcal F_R^+\) is still not a single point (on a diamond graph \(s_0\to A\to C\), \(s_0\to B\to C\), \(C\to s_f\), \(F(s_0\to A)\) can be any value in \([0,R(C)]\)). This degree of freedom is the freedom to "choose \(P_B\)" in GFlowNet Foundations. The extra component in cyclic graphs is \(H^1_+(G)\), which represents directions that can be scaled infinitely; Theorem 1 only eliminates this component, it does not shrink \(\mathcal F_R^+\) to a single point. The decomposition in Prop. 5 is the same as the "flow decomposition theorem" in minimum cost flow literature (O02 Theorem 1 citing Ahuja–Magnanti–Orlin): path flow + cycle flow.

##### 3.6 Dictionary of Three Languages (Editor's compilation, sources noted item by item)

| Concept | DAG-GFlowNet (T00/T02) | T19 (This paper) | T36 |
|---|---|---|---|
| Primary object | Edge flow or \((P_F,Z)\) | Measure \(F\) (Sec. 3.2) | \(P_B\), Eq. (7) |
| Flow semantics | Unnormalized probability of trajectories passing through edges | Abstract measure; sampling realizes \(\bar F\) (Def. 5) | Expected number of visits × \(F(s_f)\) (Def. 3.5) |
| Absorbing | Automatically holds for finite DAGs | Finite measure + \(R\ne0\) ⇒ Theorem 2 | \(P_B>0\) + Assumption 3.1 ⇒ Lemma 3.4 |
| Expected length | \(\le\#S\) (Table 1) | \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\) (Eq. (9)); \(=\bar F(S^*\to S^*)/R(S^*)\) (Cor. 1) | \(=\sum_sF(s)/F(s_f)\) (Prop. 3.12) |
| Internal flow degrees of freedom | Choose \(P_B\) | \(H^1_+(G)\) plus convex combinations of acyclic parts (Prop. 5) | Choose \(P_B\) (bijection in Prop. 3.7) |
| Cycles | Do not exist | 0-flow (Def. 2); all are cyclic on graphs (Cor. 3) | Adding flow along a cycle = changing \(P_B\) |
| Loss failure modes | — | Ratio-type instability (Theorem 3) | \(\Delta\log F\) scaling + length divergence when learning \(P_B\) (Sec. 4) |
| Correction | — | Difference-type stable loss + flow regularization (Theorem 4, 1) | Fix \(P_B\) or state flow regularization \(\lambda F_\theta(s)\) (Eq. (12)) |

#### 4. Experiments and Evidence

| Task | Environment Setup (Source) | Baselines | Metrics | Results |
|---|---|---|---|---|
| Hypergrid, 2D, \(W=20\) | \(S^*=\{1,\dots,W\}^D\), transitions \(s\to s\pm e_i\) (bidirectional, thus cyclic), initial transition \(s_0\to a\), terminal transition \(\forall s:\ s\to s_f\); flow implemented with adjacency matrix + edge flow array (Sec. 5.1, Appendix C.1) | \(L_{\mathrm{FM}}\), \(L_{\mathrm{TB}}\) vs \(L_{\mathrm{FM},\Delta,f,g}\); Figure 3 additionally compares \(L_{\mathrm{FM},\chi^2,\nu}\) (\(f=(1-x)^2\)) and \(L_{\mathrm{FM},TV,\nu}\) (\(f=\lvert1-x\rvert\)), stable version uses \(f(x)=x^2\) | Average reward, average sampled path length | All losses converge, but \(L_{\mathrm{FM}}\) exhibits exploding expected path length and path wandering; stable FM paths go "relatively straight" towards the reward region. TB paths wander but length does not explode, stabilizing at values significantly higher than stable losses, in some cases catching up after long training (Sec. 5.1; Figures 1, 3. Numerical values for curves not provided in original text) |
| Cayley Graph, \(S_{20}\) | States are permutations, edges \(g\to g\sigma_i\); generators are one transposition, a 20-cycle and its inverse; \(R_1(\sigma)=c\,\mathbf 1_{\sigma\in S_1}\), \(S_1=\{\sigma\mid\sigma(i)=i,\,i\le k\}\), \(c=20,\ k=1\); \(\pi_f(s_0\to\cdot)\) fixed uniform; path truncation 80 (Sec. 5.2, Figure 4) | Stable-GFlowNets vs GFlowNets vs Metropolis–Hastings | Average reward, average length | Optimal policy expected reward 20, expected length 5 (Figure 4 caption); flow grows uncontrollably under unstable loss, flow is bounded under stable loss (Sec. 5.2). Numerical values for curves not provided in original text |
| Point-Robot-Sparse (continuous, CFlowNets framework) | Two targets \((10,10)\), \((0,0)\), starting point \((5,5)\), max episode length 12; action angle range changed from \((0,90^\circ)\) to \((0,360^\circ)\) to create cycles; other settings same as Li et al. 2023d (Sec. 5.3) | Stable-CFlowNets (loss from Example 3) vs CFlowNets, DDPG, TD3, SAC, PPO | Number of valid and distinct trajectories in 5000 explorations; average reward | Stable loss does not compromise exploration ability, reward "greatly improved" with smaller variance (Figure 5); trajectories are cyclic in early training, converging to targets in later stages (Figure 6). Specific numerical values not provided in original text |

**Implementation Details (Appendix C).**

- Hypergrid: TensorFlow 2.4.4, Adam, learning rate 0.01; "self-training" updates the training distribution \(\nu\) to \(\mu=F_{\mathrm{out}}+\delta\) every so often, \(\delta=0.001\), 200 gradient steps per epoch; various reward distributions "have little impact on the final results."
- Evaluation: \(\hat R=\max(F(S\to\cdot)-F(\cdot\to S^*),0)\) estimates reward; \(F_{\mathrm{out}}\) approximated by power iteration, truncated at \(\lambda W\), \(\lambda\in\{4,10\}\); metrics \(E_F\) (sampling distribution error), \(E(\tau)=F_{\mathrm{out}}(S^*)/R(S^*)\), \(E_R\) (reward error), \(E_I\) (initial flow error).
- Cayley: \(\rho(\sigma)=(\sigma(1),\dots,\sigma(p))\) embedding, MLP depth 3 width 32, LeakyReLU, learning rate 1e-2, path batch 64; initial flow \(F(s_0\to\cdot)=F_{\mathrm{init}}\,U(S^*)\), \(F_{\mathrm{init}}\) trainable; when sampling paths, transitions to \(s_f\) are ignored, paths run to truncation length, then weighted by conditional probability for "\(t\le\tau\)".
- Appendix C.2 observations on stable loss variants: hyperparameters are difficult to tune, "high stabilization regularization is useful"; when \(g\) is a power \(<1\), small flows "implode" to 0 but sampling properties remain good; non-imploding versions like \(g(x)=1+|x+y|^\beta\) with \(\beta=0\) or \(\beta>1\) do not explode but have suboptimal expected rewards.

**To what extent is the evidence supported?**

- The experiments directly support only qualitative conclusions: unstable loss \(\Rightarrow\) flow/path length growth; stable loss \(\Rightarrow\) bounded. None of the three environments provide reproducible numerical results.
- The main metric is average reward, which does not measure the accuracy of fitting \(R/Z\) (T36 Sec. 4.1 criticism: learning the highest mode can yield high rewards).
- Stable loss calculates error on the \(\Delta F\) scale, and T36 Table 1, Figure 2 later show that such losses significantly deviate in fitting the reward distribution on a \(20^4\) hypergrid and \(S_{20}\).
- The original text itself lists limitations in Sec. 6.2: assumptions of stable/unstable theorems are "not optimal"; TB is only partially studied, with no stable variant; experimental scale is relatively small compared to theoretical coverage, no comparison of stable DB in continuous settings; exploration ergodicity is completely open; initial flow on Cayley graph failed to converge quickly to theoretical values; only the simplest feedforward architecture was used.

#### 5. Premise Assumptions and Applicable Boundaries

- **Measure Finiteness**: All \(F,F_{\mathrm{in}},F_{\mathrm{out}},R\) are finite non-negative measures (Sec. 3.2). The absorptivity of Theorem 2 is derived precisely from finiteness; "loops with infinite flow" are not within the framework, appearing only as the limit of training dynamics.
- **\(R\ne0\)**: Required by Theorem 2, Prop. 7–8. A 0-flow with \(R=0\) can be non-terminating (C4).
- **Transition Constraints**: \(F\ll\mu\), \(\mu(s_f\to S^*)=\mu(S\to s_0)=0\), \(\mu(s_f\to s_f)=1\); for graphs, \(\mu_E\) is taken as the edge set counting measure.
- **Technical Conditions for Theorem 3 (Appendix B.1)**: \(f\) is continuously piecewise \(C^1\); \(dF_{\mathrm{in}}/dF_{\mathrm{out}}\) is bounded and \(F_{\mathrm{out}}\ll\nu\ll F_{\mathrm{out}}\) (Prop. 9, FM); \(F_b\ll F_f\) and \(dF_b/dF_f\) is bounded, \(\pi_f,\pi_b\) are independently parameterized (Prop. 10, DB); TB also requires \(\mathbf 1_{S^*\times S^*}\mu\ll F_0\) and \(\int g(e^{\xi\,\mathrm{len}})d\nu<\infty\) (Lemma 6). The original text calls these "non-minimal simple assumptions."
- **Applicability of Theorem 4**: Holds only for R-edgeflow, meaning reward constraints are enforced by realization; also requires \(R\ll\nu\), \(dR/d\nu\) bounded (Appendix B.2). Stability means "adding 0-flow does not decrease loss," not "gradient descent always converges," nor "unique solution."
- **Theorem 1**: The main text version requires the additional assumption that the minimizing sequence converges; the appendix version replaces this with \(S\) being at most countable + strong FM loss (the latter is not defined in the original text). The conclusion is "acyclic" not "unique" (see 3.5).
- **Flow Space Structure (A.5)**: Graph is connected (every state is on some \(s_0\to s_f\) path), \(R\) is summable, graph is at most countable; Corollary 2–3 "all 0-flows are cyclic" holds only for countable graphs, there are counterexamples in uncountable spaces (C4).
- **Sampler Flow**: Definition 5 requires \(E(\tau)<\infty\).
- **Positive Statement of Scope**: Any finite total flow, \(R\ne0\), finite or countable graphs allowing cycles, and Polish spaces with a dominating measure (Footnotes 4, 6); stable FM/DB can be directly used in these spaces, TB requires other approaches.

#### 6. Position in the GFlowNet × OT Main Line

- **Predecessors**: T00 (FM loss, sampling theorem, source of Eq. (1)–(4) in this paper); T02 (DB, trajectory flow, decomposition of \(F_{\mathrm{out}}\otimes\pi_f\), Table 1 first column); T03 (TB, Eq. (6)); Lahlou et al. 2023 continuous GFlowNet (Table 1 second column; this paper claims to be "less involved" and does not assume finite absorption at \(s_f\)); Li et al. 2023d CFlowNets (foundation for Example 3 and Sec. 5.3); Kalpazidou 2007 (cycle representation theory, mathematical source of Theorem 5).
- **Successors / Counterparts**: T36 reconstructs and corrects this paper on finite graphs—tightening Theorem 2's inequality to an equality (T36 Prop. 3.12), pointing out that stability is irrelevant to the result when \(P_B\) is fixed (T36 Corollary 3.11), reinterpreting stability as error scale \(\Delta F\) vs \(\Delta\log F\), and providing constrained optimization for "minimum expected length ⇔ minimum total flow" (T36 Eq. (11)). O07 extends "minimum total flow" to "shortest path," and O08 further extends it to Kantorovich coupling. A competing approach is Lahlou et al. 2023's "finite absorption ⇒ acyclic" framework: it directly excludes cycles, while this paper chooses to accommodate cycles and then control them with loss.
- **Contributions to the "Internal Flow Selection = Optimal Transport" Main Line**: Three building blocks.
  - (a) Theorem 5 / Prop. 5 clarifies what the space of selectable internal flows is: an affine family spanned by the cycle space, where the non-negative part = acyclic flow + non-negative combinations of directed cycles. "Selecting internal flow" becomes an optimization problem with a clearly defined domain for the first time here.
  - (b) Theorem 2 / Corollary 1 links "expected trajectory length" and "total flow," thus giving behavioral meaning to O07/O08's objective function \(\sum_eF(e)\).
  - (c) Theorem 1 is the prototype of the "minimum flow principle": adding a regularizer that strictly increases along 0-flows to a stable loss and letting its coefficient tend to zero results in an acyclic limit flow—the theoretical seed for O07's "shortest path is selected" is "cutting all detours" here.
- **What this paper does not do**: It does not have a multi-source setting, does not formulate total flow minimization as a linear program, does not discuss the uniqueness of the limit flow, and does not explicitly formalize "learning \(P_B\)". These are addressed by O02, T36, and O07/O08, respectively.

#### 7. Reusable Insights and Open Problems

1.  **Zombie Flow Detection**. \(F-\bar F=\nu\otimes\pi_f\) is a 0-flow preserved by \(\pi_f\) (3.1). Experimental draft: On T36's non-acyclic hypergrid, calculate the true expected visit count flow using the fundamental matrix \(N=(I-Q)^{-1}\) from T36 Appendix B.4, and compare it edge-by-edge with the network output \(F_\theta\); the positive part of \(F_\theta-\bar F\) concentrated on which cycles indicates where the training accumulates excess mass. Before O08 interprets the learned flow as a transport plan, this difference must be zero.
2.  **What is the limit of Theorem 1 when \(\mathcal R=\lVert F\rVert_{\ell^1}\)**. \(\partial_{F_0}\lVert F\rVert_1=\lVert F_0\rVert_1>0\) satisfies the condition, and the limit is acyclic. Theorem draft: On a finite graph, if \(L\) is exactly zero on \(\mathcal F_R^+\), then the minimizer of \(L_{\alpha}\) as \(\alpha\to0^+\) converges to some point in \(\arg\min_{\mathcal F_R^+}\lVert F\rVert_1\), which is the minimum cost flow in the sense of O02 (unit edge costs). The missing step is how the non-zero gradient of \(L\) competes with \(\alpha\mathcal R\)—this is precisely the bias observed experimentally by T36 Sec. 4 using \(\lambda\) scanning (Figure 5).
3.  **Stability ≠ Unbiasedness**. Difference-based losses have gradients that tend to zero in regions where \(\log F_F\ll\log F_B\) (T36 Figure 3), systematically underestimating the flow. Theorem draft: Under the \(\Delta\)-loss in Eq. (19), the unique solution for fixed \(P_B\) is still a global minimum (loss is zero), but the steady-state bias of finite-step gradient descent increases monotonically with \(\eta\). Experiment: Fix \(P_B\), compare the \(L_1\) error curves of SDB and DB as a function of \(\eta\).
4.  **Stabilization of TB remains open**. Lemma 6 shows that the limit of TB when adding flow infinitely along a 0-flow only depends on the ratio of the first and last edges. O07/O08's neural experiments use "TB + total flow regularization", and whether it is stable in the sense of Definition 3 is not given in the original text. A provable direction: Add \(\alpha\sum_sF(s)\) to TB; is the lower bound of the directional derivative along a 0-subflow, \(\alpha\lVert F_0\rVert_1-\lvert\partial_{F_0}L_{\mathrm{TB}}\rvert\), bounded under the conditions of Lemma 6?
5.  **0-flows in continuous spaces are not cycles** (C4). When extending O08's graph OT to continuous state spaces, "acyclic" is no longer the correct goal; it should be replaced by "no 0-subflows" (minimality in Definition 7). Beckmann's continuous model mentioned in O02 Sec. 7 is a point of connection.
6.  **Maximal 0-subflows are not unique** (Prop. 3, Example 7.1). There is no canonical way to "remove cycles"; \(F=F_0+F_{\min}\) where \(F_{\min}\) depends on which maximal 0-subflow is chosen. Minimum cost (O02's \(\alpha\to0\) sparse limit, its Prop. 4, Corollary 1) provides a canonical choice; clarifying the relationship between "T19's minimal flow" and "O02's minimum cost flow" is a feasible theorem.

#### 8. References

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

**Editorial note.**

-   Author list: The PDF in this repository (arXiv v1) lists five authors, excluding Yiheng Feng; T36's bibliography and the AAAI proceedings citation list "Brunswic, Li, Xu, Feng, Jui, Ma". The table follows the PDF, while the bibtex follows the AAAI version.
-   In the four-state example in Sec. 3.1, the endpoints of the back edge are unreadable in the PDF text extraction; inferring from the original conclusion " \(f_1=f_2=f_3=1\) and \(c\ge0\) arbitrary", the back edge can only be \(C\to B\) (if it were \(C\to A\), conservation would force \(f_2=1+c\)). The numerical table in 3.3 was calculated by the editor based on this example and is not from the original text.
-   Figure 3 caption denotes stable FM as \(L_{\mathrm{FM},\Delta,f,0,\nu}\); the third subscript "0" is understood as \(g\equiv1\) (\(\eta=0\)) from context.
-   The assumptions in the main text version and appendix version of Theorem 1 are inconsistent (main text requires convergence, appendix uses strong FM loss); both are reported here without choosing between them.
-   3.5 equates Prop. 5's decomposition with O02 Theorem 1's flow decomposition theorem, which is the editor's judgment: both are "non-negative conservative flow = path flow + cycle flow", the former using Zorn's Lemma and Kalpazidou's representation, the latter using Ahuja–Magnanti–Orlin's combinatorial construction. The dictionary table in 3.6 was compiled by the editor, with sources noted in each cell.

## 3.2 T36 · Revisiting Non-Acyclic GFlowNets in Discrete Environments (ICML 2025)

> **One Sentence Summary**: This paper reconstructs the theory of non-acyclic GFlowNets from scratch on finite discrete graphs: it takes \(P_B\) as the primary object, defines flow as "expected number of visits × terminal flow", proves its one-to-one correspondence with \((P_B,F(s_f))\) (Prop. 3.7), thereby definitively closing the issue in T19 where "flows can exist but not be sampled". Its two main conclusions—uniqueness of solution when \(P_B\) is fixed, and irrelevance of stability (Corollary 3.11); and "minimum expected length ⇔ minimum total flow" when learning \(P_B\) (Prop. 3.12, Eq. (11))—are direct precursors to O07/O08 writing internal flow selection as minimum cost flow / optimal transport.

| Field | Content |
|---|---|
| arXiv | [2502.07735](https://arxiv.org/abs/2502.07735) (PDF in this repository is v3, 2025-09-11) |
| Publication | ICML 2025 Main Conference (Proceedings of the 42nd ICML, Vancouver; PMLR 267, 44887–44910) |
| Authors | Nikita Morozov\*, Ian Maksimov\* (HSE University), Daniil Tiapkin (CMAP, École polytechnique / Université Paris-Saclay), Sergey Samsonov (HSE University); \* Equal contribution, corresponding author Morozov |
| Code | [github.com/GreatDrake/non-acyclic-gfn](https://github.com/GreatDrake/non-acyclic-gfn) (End of Sec. 1) |
| PDF in this repository | `papers/2502.07735.pdf` · Chinese translation `papers_zh/2502.07735.zh.pdf` |
| Reading Priority | P0: The "minimum total flow" objective, the identity \(\sum_sF(s)=Z\cdot\mathbb E[n_\tau]\), and "degrees of freedom of \(P_B\) = degrees of freedom of internal flow" in O07/O08 all originate from here |

#### 1. Problem Setup and Notation

**Environment (Sec. 3.1, Assumption 3.1).** \(G=(S,E)\) is a finite directed graph, allowing cycles. Three assumptions: 0) \(G\) is finite; 1) initial state \(s_0\) has no incoming edges, sink \(s_f\) has no outgoing edges; 2) any \(s\in S\) is reachable from \(s_0\) and can reach \(s_f\). \(\mathrm{in}(s)\), \(\mathrm{out}(s)\) are parent and child sets. Terminal states \(\mathcal X\) = states with edges pointing to \(s_f\), reward \(R(x)>0\), \(Z=\sum_{x\in\mathcal X}R(x)\). Definition 3.2: Trajectory \(\tau=(s_0\to s_1\to\cdots\to s_{n_\tau}\to s_{n_\tau+1}=s_f)\), length \(n_\tau\) counts only internal states; \(\mathcal T\) is the set of all finite-length trajectories, which can be infinite and contain arbitrarily long trajectories when cycles exist.

**Policies and Trajectory Distribution (Sec. 3.2).** Definition 3.3: \(P_F(s'\mid s)\) is a distribution over \(\mathrm{out}(s)\) (\(s\ne s_f\)), \(P_B(s\mid s')\) is a distribution over \(\mathrm{in}(s')\) (\(s'\ne s_0\)). This paper defines the trajectory distribution from \(P_B\) in Eq. (7):
\[
P(\tau)\triangleq\prod_{t=0}^{n_\tau}P_B(s_t\mid s_{t+1}).
\]
The graph is symmetric with respect to "swapping \(s_0,s_f\) and reversing all edges", so starting from \(P_F\) is equivalent; \(P_B\) is chosen because reward constraints naturally fall on \(P_B(x\mid s_f)\).

**Flow (Sec. 3.3, Definition 3.5, Eq. (8)).** Given full edge \(P_B>0\) and terminal flow \(F(s_f)>0\),
\[
F(s\to s')\triangleq F(s_f)\,\mathbb E_{\tau\sim P}\Big[\sum_{t=0}^{n_\tau}\mathbb I\{s_t=s,s_{t+1}=s'\}\Big],\qquad
F(s)\triangleq F(s_f)\,\mathbb E_{\tau\sim P}\Big[\sum_{t=0}^{n_\tau+1}\mathbb I\{s_t=s\}\Big].
\]
In acyclic cases, expected number of visits = visit probability, consistent with classical definitions; in cyclic cases, these two differ, and only the former is conserved (Section 3.2).

**Differences from Standard DAG-GFlowNets and T19.**

- With DAG theory (Sec. 2.1): DB condition Eq. (2), FM condition Eq. (3), TB condition Eq. (1) remain unchanged in form, but the trajectory set is infinite, and \(\mathbb E[n_\tau]\) is no longer bounded; the DAG fact "fixing \(P_B\) uniquely determines \(F,P_F\)" is generalized by Prop. 3.8.
- With T19 (Sec. 2.2): T19 uses measure theory and flow as primitives; this paper only deals with finite graphs and policies as primitives. This paper explicitly corrects a statement in T19—"all definitions for acyclic cases can be directly transferred to cyclic cases" is false, because visit probability flow is not conserved.
- Marginal constraints: Reward matching Eq. (9) \(F(x\to s_f)=R(x)\); source \(P_F(\cdot\mid s_0)\) is fixed to uniform in "trainable \(P_B\)" experiments (Appendix B.3).

#### 2. Core Contributions (Numbered as in Original Sec. 1)

**C1. From-scratch Construction of Finite Discrete Non-acyclic GFlowNets (Sec. 3, Lemma 3.4, Def. 3.5, Prop. 3.6–3.10).**
Conclusion: Starting from a fully positive \(P_B\), the trajectory distribution is well-defined and has finite expected length (Lemma 3.4); expected visit counts satisfy DB and FM (Prop. 3.6); positive conserved edge flows correspond one-to-one with \((P_B,F(s_f))\) (Prop. 3.7); \(P_F\) exists and is unique (Prop. 3.8), and the reverse also holds (Prop. 3.9); reward matching is equivalent to \(F(x\to s_f)=R(x)\) (Prop. 3.10).
Prerequisites: Assumption 3.1 and \(P_B>0\). Our assessment: Simpler than T19, and clarifies the "nature of flow" and "importance of \(P_B\)" which T19 did not address.

**C2. Stability Irrelevance When \(P_B\) is Fixed (Corollary 3.11).**
Conclusion: Once \(P_B>0\) is fixed, any loss from acyclic literature can be directly used to learn \(P_F\); the solution is unique, \(\mathbb E[n_\tau]\) is finite, and T19's loss stability "does not apply".
Prerequisites: Same as above. This is a direct correction to T19: stability is only an issue when \(P_B\) is also being trained.

**C3. When Learning \(P_B\): Minimum Expected Length ⇔ Minimum Total Flow (Prop. 3.12, Eq. (10)–(12)).**
Conclusion: \(\mathbb E_{\tau\sim P}[n_\tau]=\frac{1}{F(s_f)}\sum_{s\notin\{s_0,s_f\}}F(s)\), tightening T19 Theorem 2's "≤" to "="; thus, "learning a non-acyclic GFlowNet with minimum expected length" is equivalent to "learning a non-acyclic GFlowNet with minimum total flow", expressed as constrained optimization Eq. (11); an approximate solution is DB + state flow regularization \(\lambda F_\theta(s)\) (Eq. (12)).
The original paper lists "pointing out this equivalence" as one of its key contributions and states that leveraging it is a crucial direction for future research.

**C4. Scale Hypothesis and Experiments (Sec. 4, Appendix B.2).**
Conclusion: In practice, stability is determined by the scale at which errors are computed: \(\Delta\log F\) (standard DB) can lead to arbitrarily large lengths without regularization; \(\Delta F\) (SDB) biases towards small flows and thus does not diverge, but fits the reward distribution worse. Unstable losses + state flow regularization, surprisingly, yield the best sampling quality.
This is a hypothesis plus experiments, not a theorem.

**C5. Non-acyclic Generalization of Entropy-Regularized RL Equivalence (Theorem 3.13, Appendix A.7, Lemma A.1).**
Conclusion: On a graph-induced deterministic MDP, by setting \(r(s,s')=\log P_B(s\mid s')\), \(r(x,s_f)=\log R(x)\), \(\gamma=1\), \(\lambda=1\), the entropy-regularized optimal policy is \(P_F\), with \(V^\star=\log F(s)\) and \(Q^\star=\log F(s\to s')\).
Prerequisites: \(P_B>0\) is fixed and satisfies reward matching. The proof bypasses the topological order induction relied upon by Tiapkin et al. 2024, instead using strictly convex optimization on occupancy measures.

#### 3. Key Points of Methods and Theoretical Derivations

##### 3.1 Absorbing Chains and Finite Expected Length (Lemma 3.4, Appendix A.1)

Reverse all edges of the graph, start from \(s_f\) and perform a random walk according to \(P_B\), letting \(s_0\) be an absorbing state. For any intermediate state \(s\), the return probability \(p_s:=\mathbb P[\exists t>0:Y_t=s\mid Y_0=s]<1\): Assumption 3.1 guarantees an acyclic path from \(s_0\) to \(s\), and \(P_B>0\) guarantees a strictly positive probability of returning to \(s_0\) along it, and once at \(s_0\), it's impossible to return to \(s\). Thus \(\mathbb P[N'_s>k]=p_s^k\),
\[
\mathbb E[N'_s]=\sum_{k\ge0}p_s^k=\frac1{1-p_s}<\infty,\qquad
\mathbb E[n_\tau]=\sum_{s\notin\{s_0,s_f\}}\mathbb E[N_s]\le\sum_s\mathbb E[N'_s]<\infty .
\]
Finite expected length, in turn, implies \(\sum_\tau P(\tau)=1\): otherwise, infinite trajectories would occur with positive probability, contradicting finite expectation. Contrast with T19: T19 derives absorption from "flow is a finite measure" (its Theorem 2), while this paper derives finite flow from "\(P_B>0\) + connectivity". These two paths converge on finite positive graphs.

##### 3.2 Visit Probabilities Not Conserved, Expected Visit Counts Conserved (Counterexample in Sec. 3.3)

The original text uses an example from T19: \(s_0\to a\to b\to c\to s_f\) with a back edge \(c\to b\), \(P_B(a\mid b)=P_B(c\mid b)=\tfrac12\), and other \(P_B=1\). A trajectory that loops \(k\) times, \(s_0\to a\to b\to(c\to b)^k\to c\to s_f\), has probability \(2^{-(k+1)}\).

- Edge visit probabilities: \(a\to b\), \(b\to c\), \(c\to s_f\) are all 1, \(c\to b\) is \(\tfrac12\). At \(b\): in \(1+\tfrac12\ne\) out \(1\); at \(c\): in \(1\ne\) out \(1+\tfrac12\). This is the " \(1\ne1+0.5\) " in the original text.
- Expected visit counts: \(\mathbb E[k]=\sum_kk\,2^{-(k+1)}=1\), so \(F(b\to c)=2\), \(F(c\to b)=1\), \(F(a\to b)=F(c\to s_f)=1\) (taking \(F(s_f)=1\)). At \(b\): in \(1+1=\) out \(2\); at \(c\): in \(2=\) out \(1+1\). Conservation is restored.

The general proof of Prop. 3.6 writes this observation as an identity: for each trajectory, point-wise \(\mathbb I\{s_t=s\}=\sum_{s''\in\mathrm{in}(s)}\mathbb I\{s_{t-1}=s'',s_t=s\}=\sum_{s'\in\mathrm{out}(s)}\mathbb I\{s_t=s,s_{t+1}=s'\}\). Taking expectations yields FM; the DB condition \(F(s\to s')=F(s')P_B(s\mid s')\) uses the Markov property of the reverse chain plus Fubini (Lemma 3.4 guarantees commutativity). \(F(s_0)=F(s_f)\) because each trajectory visits \(s_0\) exactly once.

##### 3.3 Existence and Uniqueness: Positive Conservative Flow \(\Leftrightarrow (P_B,F(s_f))\) (Prop. 3.7, Appendix A.3)

Given an all-edge positive \(F\) that satisfies FM, define \(P_B(s\mid s')=F(s\to s')/\sum_{s''}F(s''\to s')\), and let \(\hat F\) be the expected visit count flow it induces. By Prop. 3.6, \(\hat F(s\to s')=\hat F(s')P_B(s\mid s')=C(s')F(s\to s')\), where \(C:=\hat F/F\). Both are conservative, so
\[
\forall s\ne s_f:\quad\sum_{s'\in\mathrm{out}(s)}C(s')F(s\to s')-C(s)F(s)=0,\qquad C(s_f)=1 .
\]
With \(|S|\) unknowns and \(|S|\) equations, \(C\equiv1\) is a solution. If there is a non-constant solution \(C'\), take \(S_{\max}=\arg\max C'\): any trajectory passing through \(S_{\max}\) must have a step \(s_t\in S_{\max}\) and \(s_{t+1}\notin S_{\max}\). At \(s_t\), \(1=\sum_{s'}C'(s')F(s_t\to s')/(C'(s_t)F(s_t))<1\), which is a contradiction (if \(s_f\in S_{\max}\), the same argument applies to \(S_{\min}\)). This is a maximum principle, which uses \(F>0\) (strict inequality) and Assumption 3.1 (trajectories can exit \(S_{\max}\)).

Meaning: T19's "zombie flow" \(F-\bar F\) (T19 Def. 5, Prop. 2) is always zero here—every positive conservative flow is precisely sampled by its own \(P_B\). Appendix B.4 expresses the same idea as the linear system Eq. (17) \(\hat F(s)=\sum_{s'\in\mathrm{out}(s)}P_B(s\mid s')\hat F(s')\), \(\hat F(s_f)=F(s_f)\), whose solution is the row corresponding to \(s_f\) of the fundamental matrix \(N=(I-Q)^{-1}\) of the absorbing chain (Kemeny–Snell Theorem 3.2.1), multiplied by \(F(s_f)\).

Prop. 3.8 (\(P_F\) uniqueness): Repeating the calculation of Prop. 3.6 for the forward chain yields \(F(s\to s')=F(s)P_F(s'\mid s)\). Combining this with DB gives \(P_F(s'\mid s)=F(s')P_B(s\mid s')/F(s)\); existence relies on \(F(s_0)=F(s_f)\) making the telescopic products of \(\prod P_B\) and \(\prod P_F\) equal. Prop. 3.9 is the converse: any positive triplet \((F,P_F,P_B)\) satisfying DB induces the same trajectory distribution, and \(F\) is the expected visit count flow of \(P_B\). Prop. 3.10: termination edges are visited at most once, \(F(x\to s_f)=F(s_f)\,\mathbb P[s_{n_\tau}=x]\), so reward matching \(\Leftrightarrow\) Eq. (9), and \(F(s_0)=F(s_f)=Z\), \(P_B(x\mid s_f)=R(x)/Z\).

##### 3.4 Why Internal Flows Are Not Unique When Termination Distributions Are the Same

Reading the conclusions from 3.3 together:

- Reward matching only constrains \(P_B(\cdot\mid s_f)=R/Z\) (Prop. 3.10). For every other \(s'\ne s_0,s_f\), \(P_B(\cdot\mid s')\) can be any positive distribution over \(\mathrm{in}(s')\), totaling \(\sum_{s'}(|\mathrm{in}(s')|-1)\) free parameters.
- Different \(P_B\) values yield different edge flows (Prop. 3.7 is a bijection), but the termination distribution is always \(R/Z\) (Prop. 3.10). Thus, "non-unique internal flows" is not pathological; it is simply the parameter space of \(P_B\). T19's "adding a 0-flow" translates to "changing \(P_B\)".
- The original text Sec. 3.4 points out that T19 overlooked a common knowledge in acyclic literature: manually choosing \(P_B\) (e.g., \(1/|\mathrm{in}(s')|\) for non-terminal states and \(R/Z\) for \(s_f\)) yields a valid solution. The problem of unknown \(Z\) is circumvented by learning unnormalized flows or setting \(Z\) as a learnable parameter.

Editorial note: (extending the calculation from the example in 3.2 by replacing \(P_B(c\mid b)\) with a free parameter \(q\); numerical values are not from the original text): The probability of circling \(k\) times is \((1-q)q^k\), \(\mathbb E[k]=q/(1-q)\). Let \(F(s_f)=Z\):
\[
F(a)=Z,\qquad F(b)=F(c)=\frac{Z}{1-q},\qquad
\mathbb E[n_\tau]=\frac{F(a)+F(b)+F(c)}{Z}=1+\frac{2}{1-q}.
\]
The unique terminal state is \(c\). Any \(q\in(0,1)\) gives the same termination distribution, but the internal flow increases from \((1,1,1)\) as \(q\to0\) to infinity as \(q\to1\). For \(q=\tfrac12\), it is \((1,2,2)\) from the original text, with \(\mathbb E[n_\tau]=5\). On this graph, "internal flows with the same termination distribution" form a curve parameterized by \(q\), not a single point. The only difference between cyclic and acyclic graphs is that on a DAG, this curve is compact (bounded length), while on a cyclic graph, it extends to infinity.

##### 3.5 How to Select Minimum Flow (Prop. 3.12, Eq. (10)–(12), Appendix B.1)

The proof of Prop. 3.12 is a single line: the trajectory length is the sum of visit counts for each state, taking the expectation and dividing by \(F(s_f)\),
\[
\mathbb E_{\tau\sim P}[n_\tau]=\sum_{s\notin\{s_0,s_f\}}\mathbb E_\tau\Big[\sum_t\mathbb I\{s_t=s\}\Big]=\frac{1}{F(s_f)}\sum_{s\notin\{s_0,s_f\}}F(s).
\]
Thus, the behavioral objective of "shortest expected trajectory" becomes a linear functional on the flow space. Eq. (11):
\[
\min_{F,P_F,P_B}\ \sum_{s\notin\{s_0,s_f\}}F(s)\quad\text{s.t.}\quad\log^2\frac{F(s)P_F(s'\mid s)}{F(s')P_B(s\mid s')}=0\ \ \forall s\to s',\qquad F(s_f)P_B(x\mid s_f)=R(x)\ \ \forall x\to s_f .
\]
The feasible set is "all legitimate \(P_B\) corresponding to positive conservative flows" as described in Section 3.4; the objective is linear. In the example from Section 3.4, it is \(1+2/(1-q)\), taking its infimum at \(q\to0\) as 3, which corresponds to the shortest path \(s_0\to a\to b\to c\to s_f\). This is how minimum flow is selected: the minimum of a linear objective on a closed convex polytope lies on its boundary, and the boundary precisely corresponds to "some \(P_B\) taking value 0"—meaning backward edges are no longer used, and detours are pruned. Two points need clarification:

- The minimum point is outside the positivity assumption of Definition 3.5 (\(P_B(c\mid b)=0\)). This paper does not directly solve Eq. (11), but instead uses the \(\lambda F_\theta(s)\) regularization in Eq. (12) to approximate it from within; \(\lambda\) balances expected length and reward distribution accuracy (Figure 5). The theory of closure is provided by T19 Prop. 5: a non-negative R-flow = an acyclic R-flow + a non-negative combination of directed cycles. Removing any cycle component strictly decreases \(\sum_sF(s)\), so the minimum point must be acyclic.
- Appendix B.1: During on-policy training, the expected gradient of the regularization term is \(\mathbb E_{\tau\sim P_F}\sum_t\nabla_\theta F_\theta(s_t)=\frac{1}{2F_\theta(s_f)}\nabla_\theta\sum_sF_\theta(s)^2\)—what is actually minimized is the sum of squares of state flows, because the sampling distribution weights each state precisely by its expected visit count. If \(P_F(\cdot\mid s_0)\) is fixed to be uniform and regularization is applied only to the first state of each trajectory, the weights return to uniform. The authors state that the difference is not significant in experiments. This sum of squares is precisely the quadratic regularization of O02 (its Eq. (4)).

The relationship between Corollary 3.11 and Eq. (11): fixing \(P_B\) pins a point in the feasible set; stability only affects the optimization path, not the endpoint. Learning \(P_B\) means choosing a point in the feasible set, and in this case, without an objective function (the standard loss is zero across the entire feasible set), it will be arbitrarily pushed to infinity by the optimization dynamics—this is the paper's re-diagnosis of "flow trapped in cycles": it's not that the loss is "unstable," but that the problem is underdetermined.

##### 3.6 Reinterpretation of Stability: Error Scale (Sec. 4, Appendix B.2, Figure 3)

Eq. (15) defines two types of residuals:
\[
\Delta_{\log F}(s,s',\theta)=\log F_\theta(s)+\log P_F(s'\mid s,\theta)-\log F_\theta(s')-\log P_B(s\mid s',\theta),\qquad
\Delta_F=e^{\log F_\theta(s)+\log P_F}-e^{\log F_\theta(s')+\log P_B}.
\]
Standard DB = \(\Delta_{\log F}^2\); SDB = \(\log(1+\varepsilon\Delta_F^2)(1+\eta F_\theta(s))\) (Eq. (5)). The two can be interchanged in scale, yielding four combinations, all tested in Sec. 4.

- Figure 3 fixes the backward \(\log\) flow to 1 and sweeps the forward \(\log\) flow \(x\): green \(y=(x-1)^2\), red \(y=(e^x-e)^2\), brown \(y=\log(1+(x-1)^2)(1+0.001e^x)\), blue \(y=\log(1+(e^x-e)^2)(1+0.001e^x)\).
- The two curves for the \(\Delta F\) scale quickly plateau and their derivatives approach zero as \(x\) decreases: there is almost no gradient when the flow needs to be increased, but a large gradient when it needs to be decreased.
- Combined with Prop. 3.12 (length \(\propto\) total flow), this explains why the \(\Delta F\) loss is "stable"—it systematically underestimates flow and favors short trajectories—and also explains why it fits worse (collapse in Table 1).
- The original text points out that the same reasoning applies to T19's stable FM (Eq. (19)), as it also calculates differences on the \(\Delta F\) scale.
- The formal statement of the scale hypothesis (beginning of Sec. 4): when \(P_B\) is trainable, the primary factor determining whether "average trajectory length is controlled" in practice is the error scale; \(\Delta\log F\) without regularization can lead to arbitrarily large lengths, while \(\Delta F\) favors small flows and does not diverge.

##### 3.7 Equivalence to Entropy-Regularized RL (Theorem 3.13, Lemma A.1)

MDP \(\mathcal M_G\): states = vertices, actions = \(\mathrm{out}(s)\), deterministic transitions, \(\gamma=1\), \(r(s,s')=\log P_B(s\mid s')\), \(r(x,s_f)=\log R(x)\). The entropy-regularized value in Eq. (13) simplifies at \(\lambda=1\) to
\[
V^\pi_{\lambda=1}(s_0)=\mathbb E_{\tau\sim P^\pi_T}\Big[\sum_t r(s_t,s_{t+1})-\log\pi(s_{t+1}\mid s_t)\Big]=\log Z-\mathrm{KL}\big(P^\pi_T\,\|\,P\big),
\]
so the optimal \(\pi^\star\) is the policy with zero KL, which is the unique \(P_F\) of Prop. 3.8. The soft Bellman equation Eq. (14) is satisfied by \(Q^\star(s,s')=\log F(s\to s')\) (using Prop. 3.6's \(\log F(s\to s')=\log F(s')+\log P_B(s\mid s')\)), and \(V^\star=\log\sum_{s'}e^{Q^\star}=\log F(s)\). Lemma A.1 addresses the well-posedness of the undiscounted case: it calls this the "regularized shortest path problem." Under the assumptions that \(r\le0\) (\(r=0\) only if \(|\mathrm{in}(s')|=1\)) and the optimal policy has finite expected length, the optimization is strictly concave on the occupancy measure polytope
\[
\mathcal K=\Big\{d\ge0:\ \sum_{s'}d(s,s')=\sum_{s''}d(s'',s),\ \sum_{s'}d(s_0,s')=1,\ \sum_{s''}d(s'',s_f)=1\Big\}
\]
with a unique solution; \(d^\pi(s,s')=\pi(s'\mid s)d^\pi(s)\) and the bijection with Prop. 3.7 on the reverse graph bind occupancy and policy. \(\mathcal K\) is the normalized version of the flow polytope from Section 3.5.

##### 3.8 Training Loop Details (Sec. 3.5, Appendix B.1, B.3, C.1)

The original text does not provide pseudocode. The training process, pieced together from Eq. (12), Eq. (16), and Appendix C, is as follows:

1. The network shares a backbone, with three linear heads outputting \(\log F_\theta(s)\), logits for \(P_F\), and logits for \(P_B\) (Appendix C.2). When training \(P_B\), there is an additional scalar \(\log Z_\theta\).
2. Sample a batch of trajectories using the current \(P_F\) (on-policy; batch size hypergrid 16, permutation 512). The first step \(s_0\to s\) is sampled according to a fixed uniform \(P_F(\cdot\mid s_0)\).
3. For each transition, calculate the DB residual: for intermediate transitions, use Eq. (4); for initial transitions, use Eq. (16) \(\big[\log Z_\theta-\log|S\setminus\{s_0,s_f\}|-\log P_B(s_0\mid s,\theta)-\log F_\theta(s)\big]^2\); for terminal transitions, replace \(F_\theta(s_f)P_B(x\mid s_f)\) with \(R(x)\).
4. For each visited state that is not \(s_0\) or \(s_f\), add \(\lambda F_\theta(s_t)\) (do not add to \(\log Z_\theta\), Appendix B.3).
5. One Adam step; \(\log Z_\theta\) uses a 10x learning rate (following Malkin et al. 2022).

Consequences of on-policy (Appendix B.1): Step 4 is, in expectation, \(\frac{\lambda}{2F_\theta(s_f)}\nabla\sum_sF_\theta(s)^2\). Choosing a loss other than FM is intentional: FM does not explicitly parameterize \(P_B\), nor can it be used for experiments with fixed \(P_B\) (Appendix C.1).

##### 3.9 Theorem-level Comparison with T19 (Editorial note)

| T19 (B24) | T36 (This paper) | Relationship |
|---|---|---|
| Definition 5 sampler flow \(\bar F\le F\) | Definition 3.5 expected visit count flow | T36 directly uses \(\bar F\) as a definition; under positivity, \(\bar F=F\) (Prop. 3.7) |
| Theorem 2: \(E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\) | Prop. 3.12: \(\mathbb E[n_\tau]=\sum_sF(s)/F(s_f)\) | Inequality tightened to equality (original text explicitly states "refinement") |
| Theorem 2 absorbency (finite measure \(\Rightarrow\)) | Lemma 3.4 absorbency (\(P_B>0\) + Assumption 3.1 \(\Rightarrow\)) | Assumptions are in opposite directions, conclusions are the same |
| Prop. 5: \(\mathcal F_R^+=\) acyclic flow \(+H^1_+(G)\) | Prop. 3.7: positive conservative flow \(\leftrightarrow(P_B,F(s_f))\) | Two parameterizations of the same set: cycle coordinates vs \(P_B\) coordinates |
| Definition 3 stability; Theorem 3 instability | Corollary 3.11: irrelevant when \(P_B\) is fixed; scaling hypothesis | Rewrites "loss properties" as "whether the problem is underdetermined + error scale" |
| Theorem 1: stable loss + flow regularization \(\Rightarrow\) acyclic limit | Eq. (11)–(12): minimum total flow + \(\lambda F_\theta(s)\) | Same idea; T36 removes the premise that "loss must be stable" |
| Example 2 stable DB | Eq. (5) SDB | Same loss; T36 experiments show it is biased under \(\Delta F\) scaling |
| (None) | Theorem 3.13 equivalence with entropy-regularized RL | New in T36 |

#### 4. Experiments and Evidence

**Common Setup (Sec. 4.1, Appendix C).** Loss notation: DB / SDB refers to the loss form, \(\Delta\log F\) / \(\Delta F\) refers to the error scale, and \(\lambda=C\) refers to the state flow regularization strength; (DB, \(\Delta\log F\)) = Eq. (4), (SDB, \(\Delta F\)) = Eq. (5), (DB, \(\Delta\log F\), \(\lambda\)) = Eq. (12). Two \(P_B\) settings: fixed (\(\mathrm{out}(s_0)=\{s_{\mathrm{init}}\}\), for other states \(P_B\) is uniform over parent nodes, \(P_B(s_0\mid s_{\mathrm{init}})=1-\varepsilon\), \(\varepsilon=10^{-8}\)) and trainable (\(\mathrm{out}(s_0)=S\setminus\{s_0,s_f\}\), \(P_F(\cdot\mid s_0)\) fixed uniform, \(\log Z_\theta\) learnable, Eq. (16)). All on-policy, Adam learning rate \(10^{-3}\), \(\log Z_\theta\) learning rate \(10^{-2}\); SDB uses \(\varepsilon=1.0\), \(\eta=10^{-3}\) (larger \(\eta\) shortens trajectories but "significantly interferes with sampling fidelity"). The metric from T19 (average reward) is not used because it does not measure the fit to \(R/Z\).

| Task | Setup (Source) | Metrics | Results |
|---|---|---|---|
| Hypergrid \(7\times7\) | States \(\{0,\dots,H-1\}^D\), transitions ±1 and within bounds, each state can be terminal; \(R=R_0+R_1\prod_i\mathbb I\{0.25<\lvert s_i/(H-1)-0.5\rvert\}+R_2\prod_i\mathbb I\{0.3<\lvert s_i/(H-1)-0.5\rvert<0.4\}\), \((R_0,R_1,R_2)=(10^{-3},0.5,2.0)\) (Appendix C.2); MLP 2×256, batch 16, \(2\cdot10^6\) trajectories total | TV distance \(=\tfrac12L_1\) between empirical distribution of last \(2\cdot10^5\) samples and \(R/Z\) (Sec. 4.2, Appendix C.2; y-axis labeled \(L_1\) in figures), average trajectory length | With fixed \(P_B\), both (DB, \(\Delta\log F\)) and (SDB, \(\Delta F\)) converge to the true \(\mathbb E[n_\tau]\) for that \(P_B\) (exact solution in Appendix B.4), verifying Corollary 3.11; trainable \(P_B\) yields shorter trajectories for all losses; \(\Delta F\) scale converges slower and with slight bias; unregularized \(\Delta\log F\) does not diverge in this environment (Figure 1) |
| Hypergrid \(20^4\) | Same as above, only trainable \(P_B\) (fixed \(P_B\)'s \(\mathbb E[n_\tau]\) is orders of magnitude larger than small grid, unusable) | \(L_1\), average length, \(\log Z_\theta\) | \(\Delta F\) scale (DB and SDB) both learn biased policies, with significantly larger bias than \(7\times7\); \(\Delta\log F\) + \(\lambda=0.001\) correctly fits and learns \(\log Z\); final lengths of all methods are similar, but \(\Delta F\) is shorter in the middle of training (Figure 2). Unregularized \(\Delta\log F\) average length "tends to infinity" (Appendix D, Figure 4); for \(\lambda\in\{0.1,0.01,0.001,0.0001\}\), larger \(\lambda\) leads to shorter lengths and more biased policies (Figure 5) |
| Permutation \(S_n\), \(n=4\) | Cayley graph: \(n-1\) adjacent transpositions + one right cyclic shift; \(R(s)=\exp(\tfrac12\sum_k\mathbb I\{s(k)=k\})\) (harder than T19's \(\mathbb I[s(1)=1]\), which has trivial policies); MLP 2×128, batch 512, \(10^5\) steps | \(L_1\) of \(C(k)\) (probability vector of \(k\) fixed points), average length | Conclusions for fixed vs. trainable \(P_B\) are the same as hypergrid, with the only difference being that (SDB, \(\Delta F\)) converges faster with trainable \(P_B\) (Figure 6) |
| Permutation \(S_n\), \(n=8,20\) | Same as above, all trainable \(P_B\); \(n=20\) has \(\approx2.4\cdot10^{18}\) states; true \(\log Z\approx11.2533\) (\(n=8\)), \(42.9843\) (\(n=20\)) calculated analytically from rencontres numbers (Appendix C.3.1) | \(C(k)\) \(L_1\), \(\Delta R\) (relative mean reward error from Shen et al. 2023), \(\Delta\log Z\), \(\mathbb E[n_\tau]\); 3 seeds | See table below (Table 1) |

**Table 1 (Original numbers, mean ± standard deviation, 3 seeds).**

| Loss | \(n=8\): \(C(k)\,L_1\) | \(\Delta R\) | \(\Delta\log Z\) | \(\mathbb E[n_\tau]\) | \(n=20\): \(C(k)\,L_1\) | \(\Delta R\) | \(\Delta\log Z\) | \(\mathbb E[n_\tau]\) |
|---|---|---|---|---|---|---|---|---|
| DB, \(\Delta F\) | 0.215 ±0.198 | 0.214 ±0.086 | 0.814 ±0.826 | 2.43 ±0.28 | 0.453 ±0.002 | 0.343 ±0.000 | 42.98 ±0.000 | 2.00 ±0.00 |
| SDB, \(\Delta F\) | 0.031 ±0.012 | 0.046 ±0.023 | 0.074 ±0.025 | 3.32 ±0.15 | 0.452 ±0.001 | 0.343 ±0.000 | 42.98 ±0.000 | 2.01 ±0.00 |
| DB, \(\Delta\log F\), \(\lambda=10^{-3}\) | 0.036 ±0.015 | 0.056 ±0.024 | 0.018 ±0.010 | 2.80 ±0.04 | 0.041 ±0.002 | 0.064 ±0.000 | 0.023 ±0.005 | 3.23 ±0.00 |
| SDB, \(\Delta\log F\), \(\lambda=10^{-3}\) | 0.037 ±0.013 | 0.056 ±0.019 | 0.020 ±0.015 | 2.79 ±0.04 | 0.041 ±0.002 | 0.064 ±0.000 | 0.026 ±0.003 | 3.22 ±0.00 |
| DB, \(\Delta\log F\), \(\lambda=10^{-5}\) | 0.005 ±0.001 | 0.001 ±0.000 | 0.005 ±0.004 | 4.31 ±0.05 | 0.017 ±0.002 | 0.035 ±0.002 | 0.003 ±0.003 | 7.55 ±0.50 |
| SDB, \(\Delta\log F\), \(\lambda=10^{-5}\) | 0.005 ±0.001 | 0.002 ±0.000 | 0.006 ±0.006 | 4.36 ±0.09 | 0.014 ±0.001 | 0.025 ±0.001 | 0.005 ±0.005 | 7.31 ±0.07 |

Reading: For \(n=20\), the \(\Delta\log Z=42.98\) for both \(\Delta F\) losses exactly equals the true value \(\log Z\approx42.98\), meaning the learned \(\log Z_\theta\approx0\), and \(\mathbb E[n_\tau]\approx2\)—they completely failed to learn the normalization constant, terminating after just two steps; "stability" here means collapse. \(\Delta\log F\) + regularization reduces all three errors to the \(10^{-2}\) order of magnitude for both \(n\) values. As \(\lambda\) decreases from \(10^{-3}\) to \(10^{-5}\), \(C(k)\,L_1\) for \(n=20\) drops from 0.041 to 0.017/0.014, at the cost of \(\mathbb E[n_\tau]\) increasing from 3.2 to 7.5—this is direct numerical evidence of the length-accuracy trade-off in Eq. (11).

**To what extent is the evidence supported?**

- Direct verification: Corollary 3.11 (the fixed \(P_B\) curves in Figures 1 and 6 fall on the analytical \(\mathbb E[n_\tau]\)); Prop. 3.12 is implicit in all length curves (length estimated by sampling, consistent with \(\sum F/Z\) but not plotted separately in the original text).
- Hypothesis level: The scale hypothesis is supported by Figures 2–4 and Table 1, but the original text itself calls it a hypothesis, and the explanation for Figure 3 is heuristic.
- Untested: Theorem 3.13 has no RL algorithm experiments (not provided in the original text); Appendix B.1 states that the difference between "sum of squares vs. sum" is "not significant" but provides no numbers.
- Three conclusions from Sec. 4.4: With fixed \(P_B\), stable losses and regularization may not be needed, but manually selecting \(P_B\) with low \(\mathbb E[n_\tau]\) is difficult; when learning \(P_B\), \(\Delta F\) is scale-stable but often fails to fit; \(\Delta\log F\) fits well to scale but must be accompanied by state flow regularization.

#### 5. Prerequisites and Applicability Boundaries

- **Finiteness and Connectivity**: Assumption 3.1 is used throughout. Lemma 3.4's \(p_s<1\) and Prop. 3.7's maximum principle both rely on "every state being on some \(s_0\to s_f\) trajectory."
- **Positivity**: \(P_B>0\) for all edges, \(F(s_f)>0\), \(R(x)>0\); Prop. 3.7 requires \(F:E\to\mathbb R_{>0}\), and Prop. 3.9 requires \(P_F,P_B>0\). Minimal points of minimal flow (where some \(P_B=0\)) fall outside these assumptions and can only be approximated.
- **Markov Flow**: Only Markovian trajectory distributions are considered (explicitly stated in Sec. 3.2); non-Markovian trajectory flows are not within this framework.
- **Form of Reward Matching**: Eq. (9) \(F(x\to s_f)=R(x)\), equivalent to \(P_B(x\mid s_f)=R(x)/Z\); the fact that terminal edges can only be visited once is the sole basis for this equivalence.
- **Theorem 3.13**: \(P_B>0\) is fixed and satisfies reward matching; \(\gamma=1\), \(\lambda=1\); Lemma A.1 additionally requires \(r\le0\) (\(r=0\) only if \(|\mathrm{in}(s')|=1\)) and finite expected length for the optimal policy—the proof normalizes \(R\) to \(Z=1\) and uses \(\log R<0\) to satisfy the former.
- **Actual Goal of Regularization**: Under on-policy, Eq. (12) minimizes \(\sum_sF(s)^2\) rather than \(\sum_sF(s)\) (Appendix B.1); \(\log F_\theta(s_0)\) should not be regularized because at the optimal solution it is always equal to \(\log Z\) (Appendix B.3).
- **Experimental Setup**: When \(P_B\) is trainable, \(P_F(\cdot\mid s_0)\) must be fixed (otherwise it's not learnable when \(\mathrm{out}(s_0)=S\setminus\{s_0,s_f\}\)), and \(P_B(s_0\mid s)\) must be trainable (Appendix B.3).
- **Positive Statement of Applicability**: Any GFlowNet on a finite graph with positive flows on all edges, with or without cycles, can learn \(P_F\) (with fixed \(P_B\)) using losses from acyclic literature, or learn \((P_F,P_B)\) using "standard loss + \(\lambda F_\theta(s)\)"; SubTB and the implicit flow parameterization of Deleu et al. 2022 can also add the same regularization (Appendix C.1).

#### 6. Position in the GFlowNet × OT Main Thread

- **Predecessors**: T19 (object of reconstruction and correction: its Theorem 2 → Prop. 3.12, Theorem 3 → Corollary 3.11 and the scale hypothesis, Theorem 1's flow regularization → Eq. (12), its environment and SDB are adopted); T02 (DB/FM/TB conditions on DAGs, "unique solution for fixed \(P_B\)", Definition 4's policy definition); T03 (TB, learnability of \(\log Z\), common sense of hand-picking \(P_B\)); T05 (SubTB, mentioned as an object for transferable regularization); T10 (Shen et al. 2023's \(\Delta R\) metric); Tiapkin et al. 2024 (object generalized by Theorem 3.13, not in repository); Deleu 2025 PhD thesis (precursor to constrained optimization ideas).
- **Successors / Comparisons**: O07 interprets Eq. (11)'s "minimum total flow" as shortest path; O08 extends it to multi-source settings and identifies it as Kantorovich coupling. The competitor is T19's stable loss approach—this paper uses Table 1 to show its collapse in large environments. C02 (Schrödinger bridge on graphs) is another path for selecting internal flow based on "entropy regularization" rather than "minimum flow"; the KL form of Theorem 3.13 is their connection point.
- **Contributions to the "internal flow selection = optimal transport" main thread**:
  - (a) Prop. 3.7 + 3.10 provide the first statement of the main thread: once the terminal distribution is fixed, all degrees of freedom for internal flow = choice of \(P_B\) on non-terminal states (3.4). Without this bijection, "selecting internal flow" is not parameterized.
  - (b) Prop. 3.12 provides the objective function for the main thread: \(\sum_sF(s)=Z\cdot\mathbb E[n_\tau]\), which is linear, additive, and has a behavioral interpretation. O07/O08's \(\sum_eF(e)=\mathbb E[|\tau|]\) differs from it only by a constant that does not affect the minimum.
  - (c) Eq. (11) is the first explicit optimization problem in the main thread; Lemma A.1's polyhedron \(\mathcal K\) is the feasible region for minimum cost flow (O02 Eq. (3)'s \(\{J\ge0:D^\top J=f\}\) in single-source single-sink, normalized form).
  - (d) Appendix B.1 unexpectedly provides the regularized version of the main thread: on-policy training actually minimizes \(\sum F^2\), which is O02's quadratic regularized minimum cost flow.
- **What this paper did not do**: It did not recognize Eq. (11) as a linear program, did not discuss the structure of its minimizers (acyclic, shortest path), did not handle multiple sources, and did not use OT language. These are the contributions of O02 (mathematics) and O07/O08 (identification).

#### 7. Reusable Insights and Open Problems

1.  **Eq. (11) is a unit-cost minimum cost flow**. Theorem draft: Under Assumption 3.1, let \(\overline{\mathcal F}_R\) be the set of edge flows that are "non-negative, conserved, \(F(x\to s_f)=R(x)\), \(F(s_0)=Z\)" (closure of the image of Prop. 3.7's bijection); then \(\min_{\overline{\mathcal F}_R}\sum_sF(s)\) is a linear program, and each vertex of the minimizer set is a family of path flows on shortest paths from \(\mathrm{out}(s_0)\) to \(\mathcal X\); \(Z\cdot\min\mathbb E[n_\tau]=\sum_xR(x)\,d_G(\text{source},x)\) (in the single-source case). Proof tools: T19 Prop. 5 (cycle removal reduces total flow) + O02 Theorem 1 (path decomposition). This is exactly what O07 aims to prove.
2.  **On-policy regularization = quadratic regularization** (Appendix B.1). O02 Prop. 4 / Corollary 1 states that quadratic regularization selects a unique solution of the LP for sufficiently small \(\alpha\), and the solution does not change with \(\alpha\). Experimental draft: On a \(7\times7\) grid, precisely calculate the steady-state flow of Eq. (12) for different \(\lambda\) using Appendix B.4, and check if there exists a \(\lambda\) threshold below which the flow no longer changes; if so, it indicates that GFlowNet training accurately reaches the minimum flow solution for small \(\lambda\), rather than merely approximating it.
3.  **State flow regularization = negative reward per step in RL**. From Prop. 3.12 and Theorem 3.13's \(V=\log Z-\mathrm{KL}\): adding a constant reward \(-c\) to each non-terminal transition changes the objective to \(\log Z-\mathrm{KL}-c\,\mathbb E[n_\tau]\), which differs from the expected form of Eq. (12) only by a weight. Theorem draft: The optimal policy for soft Q-learning with \(r(s,s')=\log P_B(s\mid s')-c\) corresponds to a reweighted version of \(F\), and the limit as \(c\to\infty\) gives the shortest path policy (the "regularized shortest path" in Lemma A.1's name). This provides an RL implementation path for O07.
4.  **Design problem for fixed \(P_B\)**. For \(20^4\), uniform \(P_B\) leads to \(\mathbb E[n_\tau]\) being several orders of magnitude larger, and the authors state that hand-picking is difficult. However, the fundamental matrix \(N=(I-Q)^{-1}\) in Appendix B.4 can be calculated exactly for small graphs, and for large graphs, local heuristics for \(P_B\) can be used (e.g., biasing towards decreasing Manhattan distance to \(s_{\mathrm{init}}\)). Experiment: Compare \(\mathbb E[n_\tau]\) and \(L_1\) for uniform, distance-biased, and learned \(P_B\).
5.  **Quantification of the scale hypothesis**. Figure 3 is only a 1D slice. A provable proposition: For the \(\Delta F\) loss, when \(\log F_F-\log F_B\to-\infty\), the gradient decays as \(e^{\log F_F}\), while the \(\Delta\log F\) gradient grows linearly; this allows for an explicit comparison of the steady-state biases of the two losses on a convex problem with fixed \(P_B\), explaining the collapse of \(\Delta F\) at \(n=20\) in Table 1.
6.  **Gap in closure theory**. Prop. 3.7 only covers \(F>0\). Minimum flow solutions are on the boundary, requiring an existence and uniqueness theorem for " \(P_B\) allowing zeros but maintaining reachability in the sense of Assumption 3.1"; T19 Prop. 5–6's closed convex domain provides the topology, but the bijection "boundary flow ⇔ partially degenerate \(P_B\)" has not yet been written.

#### 8. References

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

**Editorial note.**

- Pages 44887–44910 are taken from the PMLR entry verified against background documents; the PDF in this repository (arXiv v3) only states "PMLR 267, 2025" without printed page numbers.
- In the counterexample in Sec. 3.3, edge labels were misaligned during PDF text extraction. In Sec. 3.2, based on the original text's "\(1\ne1+0.5\)" and the "2" in the expected visit count graph, the graph structure \(s_0\to a\to b\to c\to s_f\) with a back edge \(c\to b\) was deduced and verified for consistency item by item. The generalization of \(P_B(c\mid b)\) to a calculation with parameter \(q\) in Sec. 3.4 is an extension by the editor and not from the original text.
- Sec. 3.5's "minimal points are acyclic" cites T19 Prop. 5, not this paper; this paper makes no assertions about the structure of minimal points for Eq. (11).
- In Sec. 6, equating the polyhedron \(\mathcal K\) from Lemma A.1 with the feasible region of O02 Eq. (3) is the editor's judgment: both are "non-negative + pointwise conservation + fixed endpoint margins," differing only in multi-source/single-source and normalization.
- Table 1 values are transcribed cell by cell from the PDF; the original uses blue/red to highlight optimal metrics and shortest lengths, which are not preserved here.

# Chapter 4: Minimum Flow ⇔ Shortest Path (O07)

O07 transforms T36's minimum total flow objective into a necessary and sufficient characterization: the expected trajectory length is minimized if and only if the policy places all its mass on the shortest path (Thm. 3.4). It also reduces pathfinding in unweighted graphs to training a non-acyclic GFlowNet with flow regularization. This is a special case of O08 where "\(R\equiv1\)" and "single-source." O08 Thm. 3.3 explicitly states that it "recovers the corresponding claim of Morozov et al. (2026)." Publication status: ICML 2026 SPIGM Workshop, not the main conference. This chapter provides an interpretation of the full report.

> **One sentence summary**: It proves that the expected trajectory length \(\mathbb E[n_\tau]\) of a non-acyclic GFlowNet is minimized if and only if the policy places all its probability mass on the shortest path from \(s_0\) to the terminal state (Theorem 3.4, necessary and sufficient). Based on this characterization, any unweighted graph pathfinding problem is reduced to "training a non-acyclic GFlowNet with flow regularization": reverse edges, treat the target point as \(s_0\), set every point as a terminal state, and take \(R\equiv1\). The trained backward policy then serves as a shortest path solver. It achieves comparable solution lengths to CayleyPy Cube on a 3×3×3 Rubik's Cube with a smaller beam budget (Table 1). This is a direct bridge to O08: O08 merely replaces the single source \(s_0\) here with a source distribution \(L\).

| Field | Content |
|---|---|
| arXiv | [2603.01786](https://arxiv.org/abs/2603.01786) (v1 [cs.LG] 2026-03-02) |
| Publication | **ICML 2026 SPIGM Workshop (Workshop paper, not main conference)** |
| Authors | Nikita Morozov¹, Ian Maksimov¹, Daniil Tiapkin²˒³, Sergey Samsonov¹ (¹HSE University, ²CMAP, CNRS, École polytechnique, ³LMO, Université Paris-Saclay; Corresponding author Nikita Morozov) |
| Code | [github.com/GreatDrake/gfn-pathfinding](https://github.com/GreatDrake/gfn-pathfinding) (given at the end of original Sec. 1) |
| PDF in this repository | `papers/2603.01786.pdf` (14 pages: 10 pages main text + Appendices A/B/C) |
| Reading Priority | **P0** — Essential prerequisite for O08. The necessary and sufficient proof for "why minimum flow favors shortest paths" is only here; O08 only provides its LP dual version. |

### 1. Problem Setting and Notation

Following Sec. 2.1 of the original paper (notation and theory from Morozov et al. 2025, T36 in this repository).

**Environment**: A finite directed graph \(\mathcal G=(S,E)\). **Assumption 2.1**: (1) \(s_0\) has no incoming edges, and \(s_f\) has no outgoing edges; (2) For any \(s\in S\), there exists a path from \(s_0\) to \(s\) and a path from \(s\) to \(s_f\). Cycles are allowed.

**Trajectories and Termination**: \(\tau=(s_0\to s_1\to\cdots\to s_{n_\tau}\to s_f)\), with the convention \(s_{n_\tau+1}=s_f\). The set of terminal states is \(\mathcal X\), and the target distribution is \(R(x)/Z\), where \(R(x)>0\) and \(Z=\sum_{x\in\mathcal X}R(x)\).

**Policy and Consistency** (original paper Eq. (1)(2)): \(\mathbb P(\tau)=\prod_tP_F(s_{t+1}\mid s_t)=\prod_tP_B(s_t\mid s_{t+1})\), with reward matching written as \(P_B(x\mid s_f)=R(x)/Z\).

**Flow and Expected Length** (original paper Eq. (3), i.e., T36 Definition 3.5 / Proposition 3.12):

\[
\mathcal F(s)=Z\cdot\mathbb E_{\tau\sim\mathbb P}\Big[\sum_{t=0}^{n_\tau+1}\mathbb I\{s_t=s\}\Big],
\qquad
\mathbb E[n_\tau]=\frac1Z\sum_{s\in S\setminus\{s_0,s_f\}}\mathcal F(s).
\]

Thus, "minimum expected length" = "minimum total flow," which is the starting point of the entire paper. Sec. 2.1 of the original paper also provides a practical approach: adding \(\lambda\mathcal F_\theta(s)\) to the loss, e.g., DB + state flow regularization:

\[
\mathcal L(s\to s')=\Big(\log\frac{\mathcal F_\theta(s)P_F(s'\mid s,\theta)}{\mathcal F_\theta(s')P_B(s\mid s',\theta)}\Big)^2+\lambda\mathcal F_\theta(s),
\]

where reward matching is enforced by substituting \(\mathcal F_\theta(s_f)P_B(x\mid s_f,\theta)=R(x)\).

**Three Differences from Standard DAG-GFlowNet Settings**:

1. The graph has cycles, \(n_\tau\) is unbounded, and flow must be understood as expected visit counts;
2. **The assumption \(P_B>0\) has been replaced**. T36 requires \(P_B>0\) on all edges for \(\mathbb P\) to be well-defined, but the policy that minimizes \(\mathbb E[n_\tau]\) must assign 0 to non-shortest path edges (this is precisely the content of Theorem 3.4), which directly conflicts. The original paper uses **Assumption 3.1: \(\mathbb E[n_\tau]<+\infty\)** as a replacement, where \(n_\tau\) is the number of steps for a backward random walk starting from \(s_f\) and following \(P_B\) to reach \(s_0\). This is weaker and more natural, while still preserving " \(P_B\) is the transition kernel of an absorbing Markov chain, and \(\mathbb P\) is a legitimate probability distribution on \(\mathcal T\) " (derivation in Appendix A, citing Kemeny & Snell 1969);
3. The goal is not sampling quality itself, but rather **using the backward policy for pathfinding**: \(\ell(s')\) denotes the shortest path length from \(s_0\) to \(s'\), which always exists by Assumption 2.1.

### 2. Core Contributions (as numbered in the original paper)

**Contribution 1 (Theorem 3.4, Necessary and Sufficient Characterization)**: Under Assumption 2.1 + Assumption 3.1, a backward policy \(P_B\) that satisfies reward matching \(P_B(s\mid s_f)=R(s)\) minimizes \(\mathbb E[n_\tau]\) **if and only if** for any trajectory \(\tau\) terminating at \(x\in\mathcal X\), \(n_\tau\ne\ell(x)\Rightarrow\mathbb P(\tau)=0\). That is: minimizing expected trajectory length ≡ assigning zero probability to all non-shortest path trajectories. Remark 3.5 of the original paper adds: By T36 Proposition 3.8 (for any \(P_B\) there exists a unique equivalent \(P_F\), and vice versa), this theorem also holds for forward policies, as the condition only concerns the trajectory distribution itself.

**Contribution 2 (Sec. 3.2, Constructive Reduction)**: The problem of "shortest path from all points to a target point \(v_g\)" on any unweighted finite graph can be reduced to training a non-acyclic GFlowNet that minimizes \(\mathbb E[n_\tau]\). Unlike approaches that "learn a value function and then feed it to search" (Agostinelli et al. 2019; Chervov et al. 2025b), this directly learns a policy whose optimal solution is the exact shortest path.

**Contribution 3 (Sec. 3.3, Training Algorithm)**: Regularized trajectory balance (original paper Eq. (7)(8), Algorithm 1), plus two engineering decisions: training trajectory length truncated to \(N_{\max}\), and TB calculated for each prefix.

**Contribution 4 (Sec. 4, Experiments)**: Swap Puzzle (\(n=15,20\)) + 2×2×2 / 3×3×3 Rubik's Cube, compared with CayleyPy Cube (Chervov et al. 2025b, NeurIPS 2025).

### 3. Key Points of Method and Theoretical Derivation

#### 3.1 Why the Minimum Flow Criterion Favors Shortest Paths (Lemma 3.2 + Lemma 3.3 → Theorem 3.4)

This is the core of the paper. Two lemmas sandwich an equality, with sufficiency derived from "lower bound attainability + strictness."

**Lower Bound (Lemma 3.2)**: Under Assumption 2.1, any \(P_B\) satisfying Assumption 3.1 satisfies (original Eq. (4))

\[
\mathbb E[n_\tau]\ \ge\ \sum_{x\in\mathcal X}P_B(x\mid s_f)\,\ell(x).
\]

**Why it holds**: By tower property for conditional expectation (original Eq. (5)), \(\mathbb E[n_\tau]=\sum_{x}P_B(x\mid s_f)\mathbb E[n_\tau\mid s_{n_\tau}=x]\). Any trajectory from \(s_0\) to \(x\) has a length of at least \(\ell(x)\) by definition, so term by term, \(\mathbb E[n_\tau\mid s_{n_\tau}=x]\ge\ell(x)\). Note that the first step here uses the structure where "the backward walk starts from \(s_f\) and first selects a terminal state according to \(P_B(\cdot\mid s_f)\)"—the probability of selecting a terminal state is fixed by reward matching, so the lower bound is constant and does not change with the policy.

**Lower Bound Attainability (Lemma 3.3, constructive)**: Under Assumption 2.1, there always exists a \(P_B\) satisfying reward matching whose expected length is exactly (original Eq. (6)) \(\mathbb E[n_\tau]=\sum_{x\in\mathcal X}\frac{R(x)}{Z}\ell(x)\).

**How the construction works**: Define \(\mathrm{par}:S\setminus\{s_0,s_f\}\to S\setminus\{s_f\}\). For each \(s\), arbitrarily choose a parent node \(\mathrm{par}(s)\) such that \((\mathrm{par}(s)\to s)\in E\) and \(\ell(s)=\ell(\mathrm{par}(s))+1\). Such a parent node must exist, as the predecessor of \(s\) on a shortest path from \(s_0\) to \(s\) satisfies this. Then set \(P_B(x\mid s_f)=R(x)/Z\) and \(P_B(s\mid s')=\mathbb I\{s=\mathrm{par}(s')\}\). Thus, backward sampling first jumps from \(s_f\) to some \(x\), and then at each step moves strictly one edge closer to \(s_0\), necessarily sampling a shortest path. Hence, \(\mathbb E[n_\tau\mid s_{n_\tau}=x]=\ell(x)\).

**Two halves of Theorem 3.4** (proof in Appendix B):

- **(⇐) Only shortest paths ⇒ Optimal**: If \(\mathbb P(\tau)>0\) implies \(n_\tau=\ell(s_{n_\tau})\), then from Eq. (5) we directly get \(\mathbb E[n_\tau]=\sum_x\frac{R(x)}{Z}\ell(x)\), which is exactly the lower bound from Lemma 3.2, hence optimal.
- **(⇒) Optimal ⇒ Only shortest paths**: By contradiction. If there exists a \(\tau\) terminating at \(x'\) such that \(n_\tau>\ell(x')\) and \(\mathbb P(\tau)>0\), then \(\mathbb E[n_\tau\mid s_{n_\tau}=x']>\ell(x')\). For other terminal states, we still have \(\ge\ell(x)\), so \(\mathbb E[n_\tau]>\sum_x\frac{R(x)}{Z}\ell(x)\). But Lemma 3.3 shows that this right-hand side is attainable, which is a contradiction.

**Key mechanism in one sentence**: The marginal distribution of terminal states is fixed by reward matching (\(R>0\) ensures that the weight of each \(x\) is strictly positive), so \(\mathbb E[n_\tau]\) is a weighted sum of conditional expected lengths under a fixed set of weights; the lower bound for each term is \(\ell(x)\), and these lower bounds can be attained simultaneously. Therefore, "minimum total flow" does not **tend towards** shortest paths—it is **precisely equivalent** to only traversing shortest paths. Remark 3.5 in the original text also points out that the optimal \(P_B\) is generally **not unique**: for any \(s\), it can arbitrarily distribute probabilities among "parent nodes on a shortest path," as long as it assigns 0 to parent nodes not on a shortest path.

**Byproduct of Remark 3.6**: If the construction in Lemma 3.3 uses a degenerate \(P_B(s\mid s')=\mathbb I\{s=\mathrm{par}(s')\}\), then \(\mathrm{par}\), after excluding \(s_f\), defines a directed tree rooted at \(s_0\). The corresponding forward policy has a closed form \(P_F(s'\mid s)=V(s')/V(s)\), where \(V(s)\) is the sum of GFlowNet rewards on the leaves reachable from \(s\) (citing Bengio et al. 2021).

#### 3.2 Relationship with T36: Removing an Assumption and Adding a Characterization

**What T36 Lacks**: T36 established a non-acyclic framework (expected visit count flow, \(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\), arbitrary acyclic loss usable when \(P_B\) is fixed, entropy-regularized RL equivalence), and also proposed \(\lambda\mathcal F_\theta(s)\) regularization to suppress expected path length. However, as stated in Sec. 1 of this paper, "the structural implications of minimizing this quantity have not been fully analyzed"—T36 knew to shorten paths but not what shape the shortest paths would take. Theorem 3.4 provides this shape, and it is a necessary and sufficient condition.

**Removed Assumption (Appendix A, this is the most substantial technical relationship)**: T36's theory is built upon the assumption that "\(P_B(s\mid s')>0\) for all edges" (T36 Lemma 3.4 uses this to prove backward chain absorption, \(\mathbb P\) normalization, and \(\mathbb E[n_\tau]<\infty\)). This paper points out that this assumption **does not hold for the optimal solution itself**: the \(P_B\) that minimizes \(\mathbb E[n_\tau]\) must assign zero probability to non-shortest path edges, so "T36's theory cannot be directly used to learn non-acyclic GFlowNets with minimum expected length." There are two ways to fix this, both provided in the original text:

1.  Replace \(P_B>0\) with Assumption 3.1 (\(\mathbb E[n_\tau]<\infty\)). The original text explains that the only place in T36 where \(P_B>0\) is truly used is in Lemma 3.4 itself; the proofs of other conclusions only require "\(\mathbb E[n_\tau]<\infty\)" + "\(\mathbb P\) is a valid probability measure on \(\mathcal T\)", and thus can be rewritten under weaker assumptions. Appendix A explicitly verifies the latter: \(\mathbb E[n_\tau]<\infty\) implies that the backward walk reaches \(s_0\) with probability 1 (otherwise \(\{\sum_t\mathbb I\{X_t\ne s_0\}=\infty\}\) would have positive probability), which in turn implies \(\sum_{\tau\in\mathcal T}\mathbb P(\tau)=\mathbb P[\exists t:X_t=s_0]=1\).
2.  Alternatively, perform subgraph reduction: take \(\mathcal T'=\{\tau:\mathbb P(\tau)>0\}\). The subgraph \(\mathcal G'\) induced by \(\mathcal T'\) still satisfies Assumption 2.1 (contains \(s_0,s_f\), and every state is on some positive probability trajectory). On \(\mathcal G'\), \(P_B>0\) holds, so T36 Proposition 3.8 can be used, and the resulting \(P_F\) is set to zero outside \(\mathcal G'\). The original text uses this to legitimize the uniqueness conclusion in Remark 3.5.

**Components from T36 Still Used**: \(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\) (Proposition 3.12), unique correspondence between \(P_B\leftrightarrow P_F\) (Proposition 3.8), terminal edge flow \(\mathcal F(s\to s_f)=R(s)\) (Proposition 3.10), \(\lambda\mathcal F_\theta(s)\) regularization, and the DB loss form. Therefore, this paper, relative to T36, is "adding a structural theorem + relaxing an assumption + changing an application domain within the same framework," not a replacement.

#### 3.3 Reduction Construction (Sec. 3.2)

Given an arbitrary finite directed graph \(G=(V,E)\) and a target node \(v_g\), we want to find the shortest path from each \(v\) to \(v_g\) (unweighted graph, assuming every node can reach \(v_g\)):

1.  First, remove all outgoing edges from \(v_g\) in \(G\)—this does not change the problem, as these edges are not on any shortest path to \(v_g\);
2.  States \(S\) correspond to vertices of \(V\), plus a sink \(s_f\); **\(s_0\) corresponds to \(v_g\)**;
3.  Transitions \(\mathcal E\) are the **reverse** of edges in \(G\), plus an edge from each state (except \(s_f\)) to \(s_f\).

**Why this is a valid environment**: \(s_0\) has no incoming edges (because outgoing edges from \(v_g\) were removed and then reversed); \(s_f\) has no outgoing edges; every state has an edge to \(s_f\), so **all non-\(s_f\) states are terminal states** (this is repeatedly utilized by the training algorithm later); \(s_0\) is reachable from every state, because in the original graph every node can reach \(v_g\), which means \(v_g\) can reach every node in the reversed graph.

**Why this solves the original problem**: Take any positive reward \(R\) (the original text recommends task-independent \(R\equiv1\)). By Theorem 3.4, the \(P_B\) that minimizes \(\mathbb E[n_\tau]\) assigns non-zero probability only to shortest paths from \(s_0\) (i.e., \(v_g\)) to terminal states (i.e., all vertices); since \(P_B\) traverses the reversed graph, it corresponds to traversing the original graph in the forward direction. Thus, sampling from \(P_B\) starting from an arbitrary state \(s\) yields the shortest path from \(v\) to \(v_g\) in the original graph. When \(R\equiv1\), the optimal \(\mathbb E[n_\tau]\) is exactly the average shortest path length over the entire graph \(\frac1{|V|}\sum_{v\in V}\ell(v)\), which provides a directly readable ground-truth reference for training.

**Division of Labor between the Two Policies** (original text Sec. 3.2 "Forward policy"): \(P_B\) is the product—solving from any configuration to the target; \(P_F\) is the auxiliary component needed for training—walking backward from the target, sampling states according to \(R\) (uniformly), and deciding when to stop (\(P_F(s_f\mid s)\)). In the Rubik's Cube context: \(P_B\) restores any configuration in the minimum number of steps, while \(P_F\) scrambles the restored state into a uniformly distributed configuration in the minimum number of steps.

#### 3.4 Training Objective (Sec. 3.3, Eq. (7)(8), Algorithm 1)

Two practical modifications are made, with reasons provided in the original text:

- **Truncating trajectory length to \(N_{\max}\)**: In the early stages of training, \(\mathbb E[n_\tau]\) is extremely large, and the cost of sampling full trajectories is prohibitive in a large environment (Brunswic et al. 2024 also does this). During sampling, the stop action \(P_F(s_f\mid s,\theta)\) is masked, resulting in fixed-length partial trajectories \(\tau'=(s_0\to\cdots\to s_{N_{\max}})\).
- **Abandoning DB for TB**: In early experiments, DB converged extremely slowly. The original text provides two explanations: TB offers more efficient credit assignment (Malkin et al. 2022); more critically, **the training signal in this construction primarily comes from the target state \(s_0\) itself**, whereas DB is defined on single transitions, and most transitions do not involve \(s_0\), while TB is defined on entire trajectories, and by construction, every trajectory involves \(s_0\). The authors leave a deeper analysis of this phenomenon for future work.

Since every state is a terminal state, every prefix \(\tau'_{0:i}\) plus a terminal transition forms a complete trajectory. Thus, we sum over all prefixes (original Eq. (7)). Also, because \(R\equiv1\), \(Z=|V|\) is known, and there is no need to learn \(\log Z_\theta\). The flow also does not need to be learned separately: in an environment where "every point has an edge to \(s_f\)", \(\mathcal F(s)=R(s)/P_F(s_f\mid s)\) — this step is obtained by combining the terminal edge flow \(\mathcal F(s\to s_f)=R(s)\) (T36 Proposition 3.10) and a part of DB \(\mathcal F(s\to s_f)=\mathcal F(s)P_F(s_f\mid s)\) (T36 Proposition 3.8). The final objective (original Eq. (8)):

\[
\mathcal L_{\mathrm{regTB}}(\theta,\tau)=\sum_{i=0}^{N_{\max}}\Big(\mathcal L_{\mathrm{TB}}(\theta,\tau_{0:i})+\frac{\lambda}{P_F(s_f\mid s_i,\theta)}\Big),
\qquad
\mathcal L_{\mathrm{TB}}(\theta,\tau_{0:i})=\Big(\log\frac{P_F(s_f\mid s_i,\theta)\prod_{t=1}^iP_F(s_t\mid s_{t-1},\theta)}{(1/|V|)\prod_{t=1}^iP_B(s_{t-1}\mid s_t,\theta)}\Big)^2.
\]

Algorithm 1 is as follows: Sample \(B\) trajectories of length at most \(N_{\max}\) → Calculate \(\frac1B\sum_i\nabla_\theta\mathcal L_{\mathrm{regTB}}\) → Update. The original text emphasizes that **simultaneously optimizing \(P_F\) and \(P_B\) is a critical component of the algorithm**, and suggests that specialized backward policy optimization methods (Jang et al. 2024; Gritsaev et al. 2025) might further improve training.

#### 3.5 Beam Search at Test Time (Sec. 3.4)

Theoretically optimal policies precisely yield the shortest path, but on large graphs, what is learned is only an approximation. Therefore, beam search is used at test time (following Chervov et al. 2025b): with width \(W\), at each step, all continuations are expanded, and the top \(W\) are retained based on the product of the logarithms of the backward transition probabilities of the trajectories. It stops upon reaching \(s_0\); an additional heuristic to remove restarts (discarding duplicate states at each step) is mentioned, which the original text states slightly improves performance. \(W=1\) degenerates to a greedy approach for \(P_B\), \(\arg\max_sP_B(s\mid s',\theta)\) — **if the policy is optimal, greedy search still produces the shortest path**, because the probability of non-shortest path transitions must be 0 (Theorem 3.4). The original text also mentions that entropy-regularized MCTS (Morozov et al. 2024; Xiao et al. 2019) is an optional alternative.

### 4. Experiments and Evidence

Network: An MLP with 6 `ReLU(Linear(LayerNorm(x)) + x)` residual blocks, one-hot state input, \(P_F,P_B\) sharing a backbone with different linear heads (forward logits count = number of outgoing edges in the original graph + 1 for stop). The difference from previous work is replacing BatchNorm with LayerNorm to stabilize training and evaluation. Implemented in JAX, with the entire training loop JIT-compiled (Appendix C, citing Tiapkin et al. 2025's gfnx).

#### 4.1 Swap Puzzle (Sec. 4.1, Figure 2, Figure 3 Top Left)

Task: Sort an arbitrary permutation of \(n\) elements using the minimum number of adjacent swaps, i.e., finding a path on the Cayley graph of \(S_n\) with adjacent transpositions as generators. **The ground truth can be calculated**: the length of the optimal solution equals the number of inversions in the permutation (two reasons given in the original text: only the identity permutation has 0 inversions; one adjacent swap changes the number of inversions by \(\pm1\)).

- Scale: \(n=15\) and \(n=20\), with Cayley graphs having approximately \(1.3\cdot10^{12}\) and \(2.4\cdot10^{18}\) states, respectively (original Sec. 4.1).
- Test set: 500 uniformly sampled permutations for each.
- Three evaluation protocols: faithful sampling from \(P_B\), greedy search on \(P_B\) (\(W=1\)), and beam search with \(W=4\).
- Results (Figure 2): After sufficient training, **both greedy and beam search protocols yield the exact shortest path for every permutation in the test set**; the faithfully sampled policy is on average close to optimal.
- Generalization: The \(n=20\) model only encountered about \(10^9\) states during training, which is a tiny fraction of \(2.4\cdot10^{18}\) (original Sec. 4.1).
- Cost: Training for 100k iterations for \(n=20\) takes 15 minutes on a single NVIDIA H200 (Figure 2 caption).
- Hyperparameters (Appendix C.1): 100,000 iterations, batch size 128, AdamW (weight decay \(10^{-5}\)), learning rate \(3\cdot10^{-4}\), hidden size 1024, gradient norm clipping threshold 100, \(N_{\max}=50\); \(\lambda=10^{-3}\) (\(n=15\)), \(\lambda=10^{-4}\) (\(n=20\)).

#### 4.2 Rubik's Cube (Sec. 4.2, Table 1)

Opponent: CayleyPy Cube (Chervov et al. 2025b, NeurIPS 2025), the current SOTA ML method for this task, which has been shown to outperform DeepCubeA (Agostinelli et al. 2019) and EfficientCube (Takano 2023) in solution length and runtime efficiency. Fairness treatment: For CayleyPy, two models were trained: one with "original hyperparameters" and one with "network size equivalent to this paper," and the better performing one was chosen. Both methods used beam search. Only 90° face turns were allowed, so solution length is measured in QTM. Test set: 100 instances for 2×2×2 from Chervov et al. 2025b, and 1000 instances for 3×3×3 from Agostinelli et al. 2019. Solve rate refers to the proportion of instances where a valid path was found within 100 steps.

| 2×2×2 Beam | Ours Sol. Len. | Ours Solve | CayleyPy Sol. Len. | CayleyPy Solve |
|---|---|---|---|---|
| \(W=2^0\) | 11.62 | 1.0 | x | 0.00 |
| \(W=2^2\) | 11.24 | 1.0 | x | 0.00 |
| \(W=2^4\) | 10.78 | 1.0 | x | 0.00 |
| \(W=2^6\) | **10.64** | 1.0 | x | 0.06 |
| \(W=2^8\) | **10.64** | 1.0 | x | 0.89 |
| \(W=2^{10}\) | **10.64** | 1.0 | 10.64 | 1.0 |
| \(W=2^{12}\) | **10.64** | 1.0 | 10.64 | 1.0 |

| 3×3×3 Beam | Ours Sol. Len. | Ours Solve | CayleyPy Sol. Len. | CayleyPy Solve |
|---|---|---|---|---|
| \(W=2^0\) | x | 0.471 | x | 0.000 |
| \(W=2^3\) | x | 0.984 | x | 0.001 |
| \(W=2^6\) | 25.33 | 1.0 | x | 0.687 |
| \(W=2^9\) | 23.49 | 1.0 | 24.34 | 1.0 |
| \(W=2^{12}\) | 22.42 | 1.0 | 22.44 | 1.0 |
| \(W=2^{15}\) | 21.70 | 1.0 | 21.61 | 1.0 |
| \(W=2^{18}\) | 21.24 | 1.0 | 21.15 | 1.0 |

(Numbers are copied directly from the original Table 1; "x" is the original notation, indicating that no valid solution was found for the entire test set at that beam width, so solution length is not reported. 10.64 is the optimal average length for this 2×2×2 test set, verified by BFS in the original paper.)

- 2×2×2: This paper achieves the optimal 10.64 at \(W=2^6\), while CayleyPy requires \(W=2^{10}\), meaning **16 times smaller beam width**; furthermore, this paper provides valid solutions for the entire test set even with greedy search (\(W=1\)), whereas CayleyPy fails to find any valid paths with small beam widths.
- 3×3×3: In the range \(W\in[2^0,2^9]\), this paper performs better (\(W=2^9\): 23.49 vs 24.34). For \(W\in\{2^{12},2^{15},2^{18}\}\), both are comparable (22.42/22.44, 21.70/21.61, 21.24/21.15, with CayleyPy slightly better in the latter two categories).
- Runtime (\(W=2^{18}\), single H200): This paper's 25M parameter model averages **1.74 seconds** to solve one configuration, while CayleyPy's 4M parameter model takes **6.19 seconds**. The mechanism explanation given in the original paper: CayleyPy and DeepCubeA-like methods need to run a forward pass for each neighbor to estimate value/distance (3×3×3 has 12 neighbors, thus 12 times more forward passes), whereas this paper outputs backward policy logits for all neighbors in a single forward pass.
- Hyperparameters (Appendix C.2): 2×2×2 uses 500,000 iterations / batch 128 / hidden 1024 / \(\lambda=10^{-2}\) / \(N_{\max}=12\); 3×3×3 uses 1,000,000 iterations / batch 2048 / hidden 2048 / \(\lambda=5\cdot10^{-7}\) / \(N_{\max}=24\). The original paper notes that these \(N_{\max}\) values are both smaller than the longest path in their respective environments, yet still train well, serving as further evidence of generalization ability. The same number of sampled configurations were used when comparing with CayleyPy.

#### 4.3 \(\lambda\) Ablation (Sec. 4.3, Figure 3)

Observation: **Larger \(\lambda\) values are generally better, but too large will lead to complete failure**—the largest \(\lambda\) in both subfigures of Figure 3 (Swap \(n=15\), 2×2×2) causes the model to fail to find any valid paths. The resulting tuning rule of thumb: evaluate after very few iterations and choose the largest \(\lambda\) for which "a valid path to the target can still be found."

#### 4.4 Assessment of Evidence Strength

- Directly supported by experiments: Greedy/beam search protocols on Swap Puzzle yield **exact** shortest paths (with ground truth inversion numbers); 2×2×2 achieves the BFS-verified optimal 10.64 with 16 times less beam budget; 3×3×3 outperforms SOTA in the small beam range and is comparable in the large beam range; wall-clock time for solving a single configuration is 1.74s vs 6.19s.
- Author's inference, not directly verified: "TB is better than DB because of credit assignment + every trajectory contains the target state"—the original paper states this as a hypothesize and leaves it for future work; generalization ability is supported by two indirect pieces of evidence: "number of seen states / total number of states" and "effectiveness even when \(N_{\max}\) is smaller than the longest path," with no out-of-distribution testing.
- Not provided in the original paper: Specific numerical table for Swap Puzzle (only Figure 2 curves, precise numbers cannot be cited in the report); comparison with classic BFS/bidirectional BFS on medium-sized graphs; explanation for why 3×3×3's \(\lambda=5\cdot10^{-7}\) is five orders of magnitude smaller than 2×2×2's \(10^{-2}\); weighted graphs (authors list this as future work in Sec. 5).

### 5. Assumptions and Scope of Applicability

1.  **Finite Directed Graph + Assumption 2.1**: \(s_0\) has no incoming edges, \(s_f\) has no outgoing edges, and every state is reachable from \(s_0\) and can reach \(s_f\). The reduction construction (Sec. 3.2) verifies that these three conditions automatically hold under "reverse graph + connect every point to \(s_f\)".
2.  **Assumption 3.1: \(\mathbb E[n_\tau]<+\infty\)** (finite expected number of steps for backward walks). This replaces \(P_B>0\) from T36 and is the premise for this paper to discuss "optimal policies assigning 0 to certain edges".
3.  **Unweighted Graph, Hop Distance**: \(\ell(\cdot)\) counts the number of edges. Weighted/cost-sensitive settings are explicitly listed as future work in Sec. 5 of the original paper.
4.  **\(R>0\) for all terminal states**: The rigorous argument for Theorem 3.4 requires each \(x\) to have a positive weight in \(\mathbb E[n_\tau]\). The reduction uses \(R\equiv1\), which automatically satisfies this.
5.  **Reachability**: The reduction assumes every vertex can reach \(v_g\). If the graph is not connected to the target, reachability pruning is needed first.
6.  **Theorem is about exact optimal solutions**: Theorem 3.4 characterizes the policy that minimizes \(\mathbb E[n_\tau]\). Actual training uses a soft version with \(\lambda\) penalty, and too large \(\lambda\) can cause training collapse (Sec. 4.3). Therefore, for large graphs, beam search is needed to mitigate approximation errors. This is not a theoretical flaw but a real cost when translating the theorem into an algorithm.

**Scope of applicability**: Unweighted pathfinding where the state space is too large to store, but transition rules are clear and reversible (Cayley graphs, permutation puzzles, combinatorial configuration spaces). It is particularly suitable for the amortized mode of "train once, solve from any starting point", and scenarios where all neighbors need to be scored at each step (single forward pass to output logits for all neighbors is its structural advantage over value-based methods). It is not suitable for weighted graphs, scenarios requiring optimality certificates, or graphs small enough for direct BFS.

### 6. Position in the GFlowNet × OT Main Line

**Predecessors**:

-   **T19** (Brunswic et al. 2024, AAAI): Non-acyclic GFlowNets and flow regularization; the practice of truncating trajectory length also comes from it.
-   **T36** (Morozov et al. 2025, ICML): All notation, expected visit count flow, \(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\), \(P_F\leftrightarrow P_B\) uniqueness, terminal edge flow condition, \(\lambda\mathcal F_\theta(s)\) regularization. This paper makes an assumption-level correction to it (\(P_B>0\to\mathbb E[n_\tau]<\infty\)), which is the most substantial technical relationship between the two papers.
-   **TB** (Malkin et al. 2022): The training objective itself.
-   **Pathfinding competitors/references**: DeepCubeA (Agostinelli et al. 2019), EfficientCube (Takano 2023), CayleyPy (Chervov et al. 2025a,b).

**Successors**:

-   **O08** (arXiv 2606.06272): Directly builds upon this paper. The bridging relationship is precise—this paper deals with **single-source** (one point \(s_0\)) to multiple endpoints, with cost \(\ell(x)=d(s_0,x)\); O08 adds a constraint \(\mathcal F(s_0\to u)=L(u)\) to replace the single-point source with a **source distribution**, and the objective changes from \(\sum_xR(x)\ell(x)\) to \(\sum_{u,x}d(u,x)\Pi(u,x)\), which is Kantorovich OT. The proofs in both directions also correspond one-to-one: Lemma 3.2 of this paper ("actual path length \(\ge\ell(x)\)") corresponds to the inequality in O08 Eq. (18); Lemma 3.3 of this paper ("construct along the shortest path") corresponds to "distribute mass along shortest paths" in O08 Part 1. Additionally, O08 Theorem 3.3 re-derives the conclusions of this paper using LP duality + complementary slackness (O08 states "recovers the corresponding claim of Morozov et al. (2026)") and inherits the TB objective, per-prefix computation trick, and \(\lambda\) trade-off observation from this paper.
-   From another perspective, the division of labor between the two papers: **This paper provides a probabilistic proof (necessary and sufficient) for "why minimum flow chooses shortest paths", while O08 provides an LP duality proof for the same fact and gives certificates (\(\pi^\star_x=d(x)\), complementary slackness)**. To implement a primal-dual GFN-OT, both papers must be read: the semantics of the objective function are here, and the identity of the dual variables is in O08.

**Contribution to the main line**: It is the first link in the chain "internal flow selection = optimal transport". The chain has three segments: T36 defines flow as expected visit counts, making the "minimum total flow" objective expressible → This paper proves that minimum total flow \(\iff\) only shortest paths are taken → O08 adds source distribution constraints, elevating it to Kantorovich OT. Without this link, O08's Theorem 3.2 could still be proven (its proof is self-contained), but the questions "why is this objective function worth minimizing" and "what does the path structure of the optimal solution look like" are only answered by this paper, and answered more strongly than O08's dual version—**necessary and sufficient**, rather than just providing support set inclusion.

### 7. Reusable Insights and Open Questions

1.  **"Reward matching pinning the terminal edges" is the hidden fulcrum for the minimum flow criterion to work.** The lower bound \(\sum_xP_B(x\mid s_f)\ell(x)\) in Lemma 3.2 is a constant and can be used for comparison as a lower bound only because \(P_B(x\mid s_f)=R(x)/Z\) is fixed. If the terminal distribution is relaxed, the solution minimizing \(\mathbb E[n_\tau]\) would degenerate into "only terminating at the state closest to \(s_0\)", and the shortest path structure would be lost. Any GFN-OT variant that intends to modify reward matching (especially unbalanced versions) must re-evaluate this argument.
2.  **The optimal \(P_B\) is not unique and can be treated as a free regularization degree of freedom** (Remark 3.5). The probability distribution among shortest path parent nodes is completely free. This means that within the constraint of "only traversing shortest paths", a second objective can be optimized: maximum entropy (diverse shortest paths), minimum variance, or robustness to unseen states. Theorem draft: Constructing a maximum entropy GFlowNet on the shortest path subgraph yields a "uniform distribution over all shortest paths", which is directly applicable to tasks requiring diverse solutions (multiple puzzle solutions, molecular synthesis pathways).
3.  **Training signal localization explains TB ≫ DB** (Sec. 3.3.1). This argument can be transferred to any GFlowNet task where "signals are concentrated in a few special states": DB's per-transition loss is diluted by a large number of signal-free transitions, while TB's per-trajectory loss, by construction, contains signals in every trajectory. Testable hypothesis: Scanning a partial trajectory length of SubTB as an interpolation parameter, the convergence speed should monotonically change with the "probability that the segment contains the target state".
4.  **Single-pass forward computation of all neighbor logits is a structural advantage of policy-based methods over value-based methods** (Sec. 4.2). On a 3×3×3 cube, this leads to a 12-fold difference in forward passes, with actual measurements showing 1.74s vs 6.19s. This advantage will be amplified in environments with large branching factors (molecular editing action spaces can reach hundreds), and it is worth reproducing as an independent efficiency argument in other tasks.
5.  **The cliff-like failure of \(\lambda\) requires a better mechanism.** The rule of thumb in Sec. 4.3 (choosing the largest \(\lambda\) that still finds a path) is essentially manual trial and error. Moreover, the optimal \(\lambda\) for 2×2×2 and 3×3×3 differs by five orders of magnitude (\(10^{-2}\) vs \(5\cdot10^{-7}\)), indicating a strong coupling with graph diameter/scale. Possible replacements: Replace the \(\lambda\) penalty with an explicit constraint on \(\mathbb E[n_\tau]\) (augmented Lagrangian or dual ascent for automatic \(\lambda\) tuning). In this case, O08's dual potential \(\pi_s=d(s)\) provides an analytical reference for the multiplier.
6.  **Weighted graph extension (listed as future work in original Sec. 5) is technically low-risk:** Replace \(\ell(\cdot)\) with weighted shortest paths, and \(\lambda\mathcal F_\theta(s)\) with \(\lambda c(s)\mathcal F_\theta(s)\). The arguments in Lemma 3.2/3.3 only rely on "path cost \(\ge\) shortest path cost" and "existence of a deterministic parent mapping along the shortest path", both of which hold under strictly positive weights. Zero-weight edges would create zero-cost cycles and must be excluded—this is the only place requiring an additional condition.
7.  **The lack of optimality certificates is the main gap in the current form.** The Swap Puzzle has inversion count, and 2×2×2 has BFS, allowing verification; 3×3×3 lacks ground truth, so it can only rely on horizontal comparison. O08's complementary slackness \(\mathcal F^\star(s\to s')(\pi^\star_{s'}-1-\pi^\star_s)=0\) provides a per-edge verifiable alternative: by learning an additional \(\pi_\phi(s)\approx d(s)\), a proxy for the gap can be reported on graphs without ground truth. This is the most direct actionable benefit of reading the two papers together.

8.  **\(N_{\max}\) truncation lacks theoretical cost analysis, but empirical results show it can be much smaller than the graph diameter.** For 2×2×2, \(N_{\max}=12\), and for 3×3×3, \(N_{\max}=24\). The original paper states that both are smaller than the longest path in their respective environments, yet training remains effective (Appendix C.2). Truncation changes the trajectory distribution during training: \(\mathcal L_{\mathrm{regTB}}\) is evaluated only on prefixes of length \(\le N_{\max}\), which means balancing constraints are applied only to the portion of the state space "within \(N_{\max}\) steps of the target". This leaves an open question: Let \(p\) be the proportion of states whose shortest path length is \(>N_{\max}\). Can the difference between the minimal solution of the training objective and the true minimum flow solution be bounded by \(p\)? If such a bound can be provided, \(N_{\max}\) would transform from an engineering knob into a well-justified hyperparameter. The original paper treats it purely as a cost control measure, without analysis.

9.  **The performance of greedy \(W=1\) should be treated as an optimality diagnostic, not just a cheap evaluation protocol.** Theorem 3.4 states that the optimal policy assigns zero probability to non-shortest path transitions, so greedy search can also find shortest paths. Conversely, if greedy search fails, it indicates that non-shortest path transitions still carry maximum probability, meaning the policy has a localizable deviation from optimality. The original paper implicitly used this signal in Figure 2 by not drawing points for greedy failure checkpoints (the caption explicitly states these points are omitted), but it did not quantify it. An actionable approach: Report the curve of "difference between greedy and \(W\)-large solution length" during training, stratified by state distance to the target, to localize errors in near-target or far-target regions. This provides much more information than a single average solution length.

10. **The reduction construction encodes the "target point" into \(s_0\), so one model serves only one target point.** In Rubik's Cube, the target is the solved state, which is not an issue. However, for multiple targets (shortest path between any pair of start and end points, or time-varying targets), the target needs to be conditioned into the network \(P_B(s\mid s',v_g)\). In this case, Theorem 3.4 holds only for each \(v_g\) individually, and consistency across targets is not theoretically guaranteed. This is another amortization axis orthogonal to O08's amortization problem: O08 amortizes over "source-target distribution pairs \((L,R)\)", while here it amortizes over "target vertex \(v_g\)". The latter has additional structure available in Cayley graphs—group translation invariance means \(d(u,x)=d(e,u^{-1}x)\), so conditioning can in principle be simplified to replacing the state with \(v_g^{-1}s\). This point is not mentioned at all in the original paper, but it does not hold for general graphs beyond permutation puzzles, and it is worth verifying separately.

### 8. References

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

### Editorial note (independent judgment, distinct from the original text)

1.  **Publication Status**: The PDF does not contain a venue line (the template appears to be a UAI/AISTATS-style two-column format + Supplementary Material on separate pages). According to the repository's `data/papers.yaml`, it is registered as `ICML 2026 SPIGM Workshop`. Both the report and the bibtex entry indicate it is a Workshop paper, not a main conference paper.
2.  **Theorem Numbering**: The main conclusion is **Theorem 3.4**, the two lemmas are **Lemma 3.2 / Lemma 3.3**, and the weakened backward policy assumption is **Assumption 3.1** (the environmental assumption is **Assumption 2.1**). The proof is in Appendix B, and the discussion of the backward policy assumption is in Appendix A.
3.  **Swap Puzzle No Numerical Table**: The original text only reports results using curves in Figure 2, without providing a table. The report therefore only restates qualitative conclusions (greedy and beam search yield exact shortest paths) and explicit numbers from the caption (500 test cases, \(\approx1.3\cdot10^{12}\) / \(2.4\cdot10^{18}\) states, \(10^9\) visited states, 15 minutes / H200), without reading values from the curves.
4.  **"x" in Table 1**: This is marked in the original text. Its meaning, inferred from the caption, is "no valid solution found for the entire test set at this beam width, so average path length is not reported"; the original text does not explain each instance, this is my interpretation.
5.  **Items 2, 5, 6, 8, 9, 10 in §7 are my inferences and experimental drafts**, not provided in the original text; the weighted extension direction in item 6 is supported by Sec. 5 of the original text (listed as future work), but the constraint "zero-weight edges can create zero-cost cycles" is my addition. In item 10, the translation invariance of Cayley graphs \(d(u,x)=d(e,u^{-1}x)\) is common knowledge in group theory, not a conclusion of the paper.
6.  **Correspondence with O08** (Lemma 3.2 ↔ O08 Eq. (18), Lemma 3.3 ↔ O08 Part 1) is my mapping; the original O08 text only broadly states "recovers the corresponding claim of Morozov et al. (2026)" without item-by-item alignment.

# Chapter 5: Core Theorem: GFlowNet Min Total Flow ⇔ Kantorovich Optimal Transport (O08)

This chapter is the core of the entire report, containing the full interpretation of O08: the four-step proof skeleton (bilinear → LP → divergence constraint → two-sided bound), duality and complementary slackness certificates, eight prerequisite assumptions, all numbers from hypergrid and permutation experiments, and the basis for the judgment "the equivalence is classic, the interface is new." Publication status: ICML 2026 SPIGM Workshop, not a main conference.

> **One sentence summary**: By adding another constraint to the minimum total flow problem in non-acyclic GFlowNets—fixing the first-step edge flow to the source distribution \(L\)—the minimum total flow objective precisely transforms into Kantorovich OT with graph shortest path as the ground cost. The endpoint distribution of trajectories sampled by the optimal forward policy is then the optimal coupling (Theorem 3.2). The equivalence between shortest-path OT on a graph and min-cost flow is a classic result; the value of this paper lies not in this equivalence itself, but in translating it into the GFlowNet language, where the output is no longer a coupling matrix, but a routing policy \(P_F(s'\mid s)\) that can progressively execute legal local actions on a huge implicit graph. This work defines the interface for the GFlowNet × OT main thread in this repository and is the only one that provides geometric principles for "which internal flow to choose."

| Field | Content |
|---|---|
| arXiv | [2606.06272](https://arxiv.org/abs/2606.06272) (v1 [cs.LG] 2026-06-04; PDF footer self-identifies as "Preprint. June 5, 2026") |
| Publication | **ICML 2026 SPIGM Workshop (Workshop paper, not main conference)** |
| Authors | Ian Maksimov¹, Nikita Morozov¹, Denis Belomestny¹˒², Sergey Samsonov¹ (¹HSE University, ²Duisburg-Essen University; Corresponding author Ian Maksimov) |
| Code | Not publicly available (neither the main text nor the appendix provides code links; Appendix B only states that the implementation is based on publicly available code from Morozov et al. 2025) |
| This repository PDF | `papers/2606.06272.pdf` (17 pages: 8 pages main text + Appendix A theory + Appendix B experimental details) |
| Reading Priority | **P0** — The only work in the entire repository that equates the GFlowNet objective function with Kantorovich OT; all subsequent GFN-OT projects must start from its set of assumptions |

### 1. Problem Setting and Notation

We adopt the notation from Sec. 2.1 of the original paper (paraphrased from Morozov et al. 2025, this repository T36).

**Environment**: A finite directed graph \(G=(S,E)\), where \(E\subseteq S\times S\). \(\mathrm{out}(s)\) is the set of child nodes, and \(\mathrm{in}(s)\) is the set of parent nodes. Cycles are allowed (non-acyclic), which is a prerequisite for all subsequent conclusions, not just background.

**Trajectories**: \(\tau=(s_0\to s_1\to\cdots\to s_{n_\tau}\to s_f)\), where \(n_\tau\) is the trajectory length, and by convention \(s_{n_\tau+1}=s_f\). An edge of the form \(s\to s_f\) is called a terminal transition. States with outgoing edges pointing to \(s_f\) form the terminal state set \(\mathcal X\). The target distribution \(R(x)/Z\) is defined on \(\mathcal X\), with \(R(x)>0\) and \(Z=\sum_{x\in\mathcal X}R(x)\).

**Policies and Balance Condition**: Forward policy \(P_F(s'\mid s)\) and backward policy \(P_B(s\mid s')\). The training objective is for the trajectory distributions induced by both to be identical (Eq. (1) in the original paper):

\[
\mathbb P(\tau)=\prod_{t=0}^{n_\tau}P_F(s_{t+1}\mid s_t)=\prod_{t=0}^{n_\tau}P_B(s_t\mid s_{t+1}),
\]

plus reward matching (Eq. (2) in the original paper) \(P_B(x\mid s_f)=R(x)/Z,\ \forall x\in\mathcal X\).

**Flow = Expected Visit Count** (Eq. (3) in the original paper, i.e., T36 Definition 3.5):

\[
\mathcal F(s)=Z\cdot\mathbb E_{\tau\sim\mathbb P}\Big[\sum_{t=0}^{n_\tau+1}\mathbb I\{s_t=s\}\Big],
\qquad
\mathcal F(s\to s')=Z\cdot\mathbb E_{\tau\sim\mathbb P}\Big[\sum_{t=0}^{n_\tau}\mathbb I\{s_t=s,s_{t+1}=s'\}\Big].
\]

When cycles are present, "visit probability" is not conserved, but "expected visit count" is. Therefore, this definition is necessary (Sec. 2.1 of the original paper cites T36 Proposition 3.6, which gives the flow matching condition Eq. (4): \(\sum_{s\to s'}\mathcal F(s\to s')=\sum_{s''\to s}\mathcal F(s''\to s)\), as well as \(\mathcal F(s_0)=\mathcal F(s_f)=Z\)).

**Minimum Total Flow Problem**: Let \(\mathcal I:=S\setminus\{s_0,s_f\}\) be the set of internal states. Sec. 2.1 of the original paper restates the key equation from T36, \(\mathbb E[n_\tau]=\frac1Z\sum_{s\in\mathcal I}\mathcal F(s)\) (T36 Proposition 3.12). Thus, "minimum expected trajectory length" = "minimum total flow", Eq. (5) in the original paper:

\[
\min_{\mathcal F,P_F,P_B}\sum_{s\in\mathcal I}\mathcal F(s)
\quad\text{s.t.}\quad
\mathcal F(s)P_F(s'\mid s)=\mathcal F(s')P_B(s\mid s'),\ \forall (s,s')\in E;\quad
\mathcal F(s_f)P_B(x\mid s_f)=R(x),\ x\in\mathcal X.
\]

A common practice is to add \(\lambda\mathcal F_\theta(s)\) to the loss (Brunswic et al. 2024; T36). Sec. 2.1 of the original paper explicitly states that it follows this approach.

**OT Side**: Definition 2.1 in the original paper defines coupling, and Eq. (6) gives the Kantorovich LP.

**Novel Structural Assumptions of This Paper** (Assumption 3.1 in the original paper, four points, authors state "similarly to Essid & Solomon (2018)"):

1. The special initial state \(s_0\) has no incoming edges, and the special sink \(s_f\) has no outgoing edges;
2. There exist state sets \(U\) and \(\mathcal X\) such that outgoing edges from \(s_0\) only enter \(U\), and incoming edges to \(s_f\) only come from \(\mathcal X\);
3. For any \(u\in U\) and \(x\in\mathcal X\), there exists a path of finite length;
4. Given distributions \(L(u)\) on \(U\) and \(R(x)\) on \(\mathcal X\), such that \(\sum_{u\in U}L(u)=\sum_{x\in\mathcal X}R(x)=1\) (\(L,R\) are zero outside their support, considered defined on all internal states).

**Ground Cost** (Eq. (7) in the original paper): \(d(u,x)=|\tau_{u,x}|\), where \(\tau_{u,x}\) is a shortest path from \(u\) to \(x\), and \(|\cdot|\) is its number of edges. Thus, Eq. (8) in the original paper is the Kantorovich problem between \(L\) and \(R\) with cost \(d\). The original paper explicitly states after Eq. (8) that this graph OT with shortest path as ground cost has a classical min-cost-flow formulation (Essid & Solomon 2018) and is also equivalent to the discrete graph version of Beckmann's problem (Beckmann 1952).

**Key New Constraint**: The authors call \(L\) the *leward* (left reward) and, analogous to reward matching, fix the first-step edge flow:

\[
\mathcal F(s_0\to u)=L(u).
\]

**Differences from Standard DAG-GFlowNet Setting**: Four points, all hard differences:

- The graph allows cycles, so flow must be interpreted as expected visit counts; otherwise, the objective function \(\sum_s\mathcal F(s)=\mathbb E[n_\tau]\) cannot be formulated.
- Not only is the terminal distribution fixed, but also the first-step distribution \(\mathcal F(s_0\to u)=L(u)\) is fixed—this is the fulcrum of all theorems in the paper.
- Both marginals are normalized, so \(Z=1\) is known (Sec. 3.1 of the original paper explicitly states that the normalized formulation is chosen to "eliminates the need to handle an unknown normalizing constant"). GFlowNet's signature unknown-\(Z\) capability is voluntarily abandoned in this setting.
- The goal is not to "find a solution that satisfies the terminal distribution," but rather to "find the solution with the minimum total internal flow among all solutions that satisfy both marginals," i.e., to fix the severely underdetermined internal flow degrees of freedom in standard GFlowNet using a geometric principle.

### 2. Core Contributions (Numbered as in Original Text)

**Contribution 1 (Original Sec. 1, Item 1 + Theorem 3.2 + Appendix A.1)**: The minimum total flow learning problem in T36 can be equivalently reformulated as a linear program (LP). When the initial edge flow distribution is fixed in this LP, the minimum total flow objective precisely becomes Kantorovich OT with graph distance as cost, which is also equivalent to the discrete graph form of Beckmann's problem (Beckmann 1952; Essid & Solomon 2018). The optimal forward policy under this setting samples optimal paths between the initial and terminal distributions, thereby inducing an optimal coupling. As a special case, this construction reproduces the shortest path theory of Morozov et al. (2026) (O07 in this repository).

**Contribution 2 (Original Sec. 1, Item 2 + Sec. 3.3 + Sec. 4)**: The GFlowNet learning framework provides a feasible means to approximate graph OT solutions. Based on neural parameterization, the GFlowNet objective can learn a policy that approximates the optimal coupling. Experiments reproduce exact OT solutions when they are computable and provide approximations at scales where exact solutions are intractable (permutations \(n=20\)).

The original text places theorem-level results in three locations. This report will refer to them by these numbers:

- **Theorem 3.2** (Original Sec. 3.2): Under Assumption 3.1, the reduction of the original problem Eq. (11) is equivalent to the Kantorovich problem Eq. (8) between \(L\) and \(R\), i.e., \(\mathrm{GFlow}^\star=\mathrm{OT}^\star\) (Eq. (12)); and if \(\mathbb P^\star\) is the trajectory distribution induced by the \(\mathrm{GFlow}^\star\) solution, then \(\Pi^\star_{u,x}:=\sum_{\tau:u\rightsquigarrow x}\mathbb P^\star(\tau)\) is an optimal coupling for Eq. (8).
- **Theorem 3.3** (End of Original Sec. 3.2; full version is Theorem A.5 in Appendix A.5): The dual of the original problem Eq. (10) **without** the first-step constraint is \(\max_\pi\sum_{x}R(x)\pi_x\) s.t. \(\pi_{s_0}=0,\ \pi_{s'}-\pi_s\le 1\) (\(s'\ne s_f\)). Three conclusions: any feasible \(\pi\) satisfies \(\pi_s\le d(s)\); \(d(\cdot)=|\tau_{s_0,\cdot}|\) is feasible and dual optimal; if \(R(x)>0\ \forall x\), then any optimal \(\pi^\star\) satisfies \(\pi^\star_x=d(x)\) at terminal states. Complementary slackness yields \(\mathcal F^\star(s\to s')\big(\pi^\star_{s'}-(1+\pi^\star_s)\big)=0\), meaning optimal flow can only occur on the tight subgraph—which is the shortest path subgraph. The original text explicitly states that this "recovers the corresponding claim of Morozov et al. (2026)".
- **Proposition A.4 and Appendix A.4**: Derivation of the dual for the lifted LP (Eq. (22)→Eq. (23)); the dual of the extended problem (including the first-step constraint) is \(\max_\pi\sum_x R(x)\pi_x+\sum_u L(u)(-\pi_u)\). By relabeling \(a_x=\pi_x,\ b_u=-\pi_u\), we get \(a_x+b_u\le d(u,x)\) (Eq. (24)→Eq. (25)), which is isomorphic to the Kantorovich dual.

### 3. Key Points of Method and Theoretical Derivation

The entire chain consists of four steps: **bilinear → linear (LP) → reduced to divergence constraint → squeezed from both sides to obtain equality**, plus one more step for the dual to provide certificates.

#### 3.1 Step 1: Eliminating Policies, Transforming Bilinear Constraints into Linear Ones (Appendix A.1)

The detailed balance constraint in Eq. (5) is bilinear in the decision variables because (Original Eq. (9))

\[
\mathcal F(s\to s')=\mathcal F(s)P_F(s'\mid s)=\mathcal F(s')P_B(s\mid s').
\]

By treating \(\mathcal F(s)\) and \(\mathcal F(s\to s')\) as independent variables and eliminating \(P_F,P_B\), the problem becomes linear. **Why this holds**: Proposition A.1 states that "there exists \(P_F(\cdot\mid s)\in\Delta(\mathrm{out}(s))\) such that \(\mathcal F(s\to v)=\mathcal F(s)P_F(v\mid s)\)" is equivalent to "\(\mathcal F(s\to v)\ge0\) and \(\sum_{v\in\mathrm{out}(s)}\mathcal F(s\to v)=\mathcal F(s)\)". The forward direction substitutes the probability sum to 1, while the reverse direction defines the policy by direct normalization when \(\mathcal F(s)>0\). When \(\mathcal F(s)=0\), all outgoing edge flows are forced to 0 by non-negativity and the sum-to-zero condition, and any policy satisfies the equality. Proposition A.2 applies similarly to incoming edges. Corollary A.3 summarizes this into the lifted LP (labeled (P) in the original text). This step is the technical starting point of the entire paper: detailed balance and flow matching are the same constraint in the sense that "policies can be freely reconstructed".

#### 3.2 Step 2: Reducing the State Flow to Divergence Form (Original Eq. (10)→(11))

Adding \(\mathcal F(s_0\to u)=L(u)\) and \(\mathcal F(x\to s_f)=R(x)\) yields the extended original problem Eq. (10). Define the set of internal edges

\[
E^\circ:=\{s\to s'\in E:\ s\ne s_0,\ s'\ne s_f\},
\]

and substitute all \(\mathcal F(s)\) using the inflow equations to obtain the reduced original problem (original Eq. (11)):

\[
\min_{\mathcal F(s\to s')\ge0}\sum_{s\to s'\in E^\circ}\mathcal F(s\to s')
\quad\text{s.t.}\quad
\sum_{v:\,s\to v\in E^\circ}\mathcal F(s\to v)-\sum_{u:\,u\to s\in E^\circ}\mathcal F(u\to s)=L(s)-R(s),\ \ s\in\mathcal I.
\]

**There is a constant that must be remembered in this step**. The original text states after Eq. (11):

\[
\sum_{s\in\mathcal I}\mathcal F(s)=\sum_{s_0\to u}\mathcal F(s_0\to u)+\sum_{s\to s'\in E^\circ}\mathcal F(s\to s')=1+\sum_{s\to s'\in E^\circ}\mathcal F(s\to s'),
\]

and explicitly states, "We will omit this constant, as it does not change minimum of the problem." Therefore, **\(\mathrm{GFlow}^\star\) in Theorem 3.2 refers to the reduced objective \(\sum_{E^\circ}\mathcal F\), not \(\sum_{s\in\mathcal I}\mathcal F(s)=\mathbb E[n_\tau]\) itself**. The difference between them is 1. This 1 corresponds to the \(s_0\to u\) step: it is the action of "entering the source distribution" and does not belong to the transport segment \(u\rightsquigarrow x\). The direct alignment of \(\mathbb E|\tau|\) with \(\mathrm{OT}^\star\) values in Table 1 (e.g., for \(H=10\), Moon: 4.352 vs 4.351) indicates that the reported \(\mathbb E|\tau|\) measures the length of the transport segment and is on the same scale as the reduced objective.

The right-hand side of the constraint \(L(s)-R(s)\) is the discrete divergence: this is the standard form for min-cost flow / discrete Beckmann. The original text also points out that summing all flow-matching constraints for internal states yields \(\sum_x R(x)=\sum_u L(u)\), meaning mass balance is a necessary consequence of the constraints, so Assumption 3.1, item 4, is not an optional embellishment.

#### 3.3 Step 3: \(\mathrm{GFlow}^\star\le\mathrm{OT}^\star\) (Original Theorem 3.2 Proof Part 1)

Take the optimal coupling \(\Pi^\star\) of Eq. (8) (existence cited from Villani 2008, Theorem 4.1, p. 43). For each pair \((u,x)\), choose an arbitrary shortest path \(\tau_{u,x}\), and let

\[
\mathcal F(s\to s')=\sum_{u\in U}\sum_{x\in\mathcal X}\Pi^\star_{u,x}\cdot\mathbb I[s\to s'\in\tau_{u,x}].
\]

**Why feasibility holds**: Define \(\Delta^{u,x}_s=\sum_{v:s\to v\in E}\mathbb I[s\to v\in\tau_{u,x}]-\sum_{v:v\to s\in E}\mathbb I[v\to s\in\tau_{u,x}]\). Since \(\tau_{u,x}\) is a directed path, it only has outgoing edges at the start, only incoming edges at the end, and equal counts of incoming and outgoing edges at intermediate nodes. Thus, \(\Delta^{u,x}_s=\mathbb I[s=u]-\mathbb I[s=x]\). Substituting this into the two marginal constraints of \(\Pi^\star\) yields (original Eq. (13))

\[
\Delta\mathcal F(s)=\sum_{u,x}\Pi^\star_{u,x}\big(\mathbb I[s=u]-\mathbb I[s=x]\big)=L(s)-R(s).
\]

**Why the objective value equals the OT cost**: Swapping the order of summation, \(\sum_{s\to s'\in E^\circ}\mathbb I[s\to s'\in\tau_{u,x}]=|\tau_{u,x}|\). Since \(\tau_{u,x}\) is a shortest path, \(|\tau_{u,x}|=d(u,x)\). Therefore, (original Eq. (14)(15)) the objective value \(=\sum_{u,x}d(u,x)\Pi^\star_{u,x}=\mathrm{OT}^\star\). This flow is feasible but not necessarily optimal, hence \(\mathrm{GFlow}^\star\le\mathrm{OT}^\star\) (original Eq. (16)).

The entirety of this half is: **distribute each unit of mass from the coupling along shortest paths as edge flow**. It also proves that this "GFlowNet problem with additional equality constraints" in Eq. (10)/(11) is always feasible—a point the original text specifically highlights at the end of Sec. 3.1, noting that it's not evident from GFlowNet theory itself.

#### 3.4 Step 4: \(\mathrm{GFlow}^\star\ge\mathrm{OT}^\star\) (Original Theorem 3.2 Proof Part 2)

Take the optimal solution \(\mathcal F^\star\) of Eq. (11) (existence from non-empty feasible set + bounded objective), and restore the forward policy using the equivalence in Appendix A.1:

\[
P^\star_F(s'\mid s)=\mathcal F^\star(s\to s')\Big/\sum_{s\to s''}\mathcal F^\star(s\to s''),
\qquad
\mathbb P^\star(\tau)=\prod_{i=0}^{n_\tau}P^\star_F(v_{i+1}\mid v_i),
\]

From Assumption 3.1, we have \(Z=1\). Define \(\Pi_{u,x}:=\sum_{\tau:u\rightsquigarrow x}\mathbb P^\star(\tau)\). The two marginals can be directly read from the constraints: \(\sum_x\Pi_{u,x}=P^\star_F(u\mid s_0)=L(u)\) (first step constraint), \(\sum_u\Pi_{u,x}=R(x)\) (termination constraint), so \(\Pi\) is a valid coupling. Then calculate the objective value (original Eq. (17)):

\[
\mathrm{GFlow}^\star=\sum_{s\to s'\in E^\circ}\mathcal F^\star(s\to s')=\sum_\tau|\tau|\,\mathbb P^\star(\tau),
\]

where \(|\tau|\) is the number of edges in trajectory \(\tau\) that fall within \(E^\circ\). Finally, use \(|\tau_{u,x}|\ge d(u,x)\) (\(d\) is the shortest path length) for term-by-term bounding (original Eq. (18)) to get \(\mathrm{GFlow}^\star\ge\sum_{u,x}d(u,x)\Pi_{u,x}\ge\mathrm{OT}^\star\) (Eq. (19)). Combining with Eq. (16) yields Eq. (12).

**The essence of this half**: The endpoints of trajectories sampled by the policy, when marginalized, must form a valid coupling, and any path actually traversed is no shorter than the shortest path. Therefore, minimizing the total flow = forcing every sampled path to degenerate into a shortest path. This also indicates that the optimal GFlowNet provides more information than just the coupling matrix: it not only tells "how much mass to transport from \(u\) to \(x\)" but also "which specific valid local actions are taken."

#### 3.5 Duality and Certificates (Theorem 3.3 / Appendix A.2–A.4)

The derivation of Proposition A.4 is clean: taking the infimum for \(\mathcal F(s)\) (unconstrained sign) requires its coefficient to be zero, yielding \(c=\bar\alpha+\bar\beta\); taking the infimum for \(\mathcal F(s\to s')\ge0\) requires the coefficient to be non-negative, yielding \(\alpha_s+\beta_{s'}\ge0\) and \(\eta_x\le\alpha_x+\beta_{s_f}\). From \(c=\bar\alpha+\bar\beta\), we get \(\alpha_{s_0}=0,\beta_{s_f}=0,\beta_s=1-\alpha_s\ (s\in\mathcal I)\). Substituting these back into the edge inequality simplifies it to \(\alpha_{s'}-\alpha_s\le1\), which, by renaming to \(\pi\), becomes Eq. (23).

The three steps of Theorem A.5 are also elementary but crucial:

1. Summing \(\pi_{v_{i+1}}-\pi_{v_i}\le1\) along any path \(s_0\rightsquigarrow s\), telescoping sum gives \(\pi_s-\pi_{s_0}\le k\), taking the minimum yields \(\pi_s\le d(s)\);
2. \(d(\cdot)\) is feasible because connecting a shortest path with an edge gives \(d(s')\le d(s)+1\);
3. \(R>0\) makes the objective strictly monotonic, so any optimal solution must attain the upper bound \(\pi^\star_x=d(x)\) at termination states.

**The identity of the dual variables is BFS distance**, which is worth stating directly: the optimal dual potentials in Eq. (23) are precisely the hop distances from \(s_0\). Complementary slackness \(\mathcal F^\star(s\to s')(\pi^\star_{s'}-1-\pi^\star_s)=0\) thus transforms "the support of the optimal flow \(\subseteq\) shortest path subgraph" into an equality that can be checked edge by edge. This is the LP dual proof of O07's main theorem. Appendix A.4 further maps the dual of the extended problem \(\max_\pi\sum_xR(x)\pi_x-\sum_uL(u)\pi_u\) to the Kantorovich dual Eq. (25) via \(a_x=\pi_x,b_u=-\pi_u\) and telescoping sum along paths \(a_x+b_u=\pi_x-\pi_u\le d(u,x)\).

#### 3.6 Training Objective (Original Sec. 3.3, Eq. (20))

Instead of detailed balance + flow regularization, the authors use regularized trajectory balance with a "leward" term (the authors state that Morozov et al. 2026 has proven TB to be more effective in non-acyclic pathfinding, and TB is most suitable for their setting):

\[
\mathcal L_{\mathrm{TB}}(\theta,\tau)=\left(\log\frac{L(s_1)\prod_{t=1}^{n_\tau}P_F(s_{t+1}\mid s_t,\theta)}{R(s_{n_\tau})\prod_{t=0}^{n_\tau-1}P_B(s_t\mid s_{t+1},\theta)}\right)^{2}+\lambda\,\frac{R(s_{n_\tau})}{P_F(s_f\mid s_{n_\tau},\theta)}.
\]

The three components have distinct origins: \(L(s_1)\) in the numerator replaces the usual \(Z\) in TB—because \(Z=1\) and the first-step distribution is fixed to \(L\), the original text writes it as leward matching \(P_F(s\mid s_0)=L(s)\); \(R(s_{n_\tau})\) in the denominator handles reward matching \(P_B(s\mid s_f)=R(s)\); the regularization term \(\lambda R(s)/P_F(s_f\mid s)\) is the state flow \(\mathcal F(s)\) at the termination state (derived from the reward matching condition in Eq. (5)), which serves as a penalty term for the minimum total flow objective. Training is on-policy: a batch of trajectories is sampled using \(P_F\) at each step to compute the loss. When every internal state is a termination state (\(\mathcal X=\mathcal I\)), Eq. (20) can be computed for each prefix of sampled trajectories, leading to higher sample efficiency (citing Morozov et al. 2026); this option was adopted in the experiments.

### 4. Experiments and Evidence

Two environments, all results averaged over 3 random seeds (original Sec. 4.1 and Sec. 4.2 both state "averaged over three seeds"). Model: 2 hidden layers, 128 width MLP, one-hot input state, \(\mathcal F_\theta(s)\), \(P_F\), \(P_B\) share backbone but have different linear heads; on-policy, batch 512, AdamW, \(\mathrm{lr}=10^{-3}\), weight decay \(10^{-4}\); **all experiments were performed on CPU** (Appendix B).

#### 4.1 Hypergrid (Original Table 1)

Environment: Grid points \(\{0,\dots,H-1\}^D\) plus \(s_0,s_f\); starting from \(U\) according to \(L\), transitions are single-coordinate \(\pm1\) and within bounds. Every state has a terminal transition (thus \(\mathcal X=\mathcal I\)). Reward \(R\) is corner-shaped multimodal (definition in Appendix B.1, following Bengio et al. 2021 / Madan et al. 2023 / Malkin et al. 2022). The prior \(L\) is moon-shaped and ball-shaped (formulas in Appendix B.1). TV is estimated using \(2\cdot10^5\) model samples. \(\mathrm{OT}^\star\) is computed by the POT solver (Flamary et al. 2021).

| \(L\) | \(H\) | \(\widehat{\mathrm{TV}}\downarrow\) | \(\mathrm{TV}^\star\downarrow\) (Perfect Sampler Reference) | \(\mathbb E|\tau|\) | \(\mathrm{OT}^\star\) |
|---|---|---|---|---|---|
| Ball | 10 | 0.024 ±0.0004 | 0.024 | 3.990 ±0.015 | 3.997 |
| Ball | 15 | 0.036 ±0.0009 | 0.033 | 6.325 ±0.011 | 6.303 |
| Ball | 20 | 0.037 ±0.0006 | 0.040 | 8.326 ±0.021 | 8.325 |
| Moon | 10 | 0.022 ±0.018 | 0.024 | 4.352 ±0.015 | 4.351 |
| Moon | 15 | 0.023 ±0.006 | 0.033 | 6.907 ±0.010 | 6.868 |
| Moon | 20 | 0.032 ±0.008 | 0.040 | 9.001 ±0.112 | 9.059 |

(All numbers are copied verbatim from the original Table 1. \(\mathrm{TV}^\star\) is non-zero because it is the TV distance between a finite-sample empirical distribution and the true distribution.)

Original Figure 1 also provides a visualization for \(H=10,\ L=\mathrm{Moon}\): The left figure shows the edge flow on the grid from the exact LP solution of Eq. (11) (`scipy.linprog`, Virtanen et al. 2020), and the right figure shows the direct source-target connections of the Kantorovich optimal coupling. Both have a **transport cost equal to 4.351** (Figure 1 caption). This figure is the most convincing piece of evidence in the entire paper: the same numerical value, one from edge flow, the other from coupling.

**Evidence Strength Assessment**: For the three settings of \(H\in\{10,15,20\}\), the relative difference between \(\mathbb E|\tau|\) and \(\mathrm{OT}^\star\) is in the order of \(10^{-3}\)–\(10^{-2}\). At the same time, \(\widehat{\mathrm{TV}}\) is not higher than the perfect sampler reference (for \(H=20\) Ball, 0.037 vs 0.040; for Moon, 0.032 vs 0.040). Therefore, the conclusion that "the learned policy simultaneously satisfies the target marginal + achieves optimal OT cost" is directly supported by experiments on a small scale. Note that for \(H=20\) Moon, \(\mathbb E|\tau|=9.001<\mathrm{OT}^\star=9.059\): this does not mean it beats the OT lower bound, but rather that the reading will be lower when the sampler does not precisely satisfy the marginal constraint (non-zero TV) – the original text does not discuss this, which is my assessment.

#### 4.2 Permutations (Original Table 2)

Environment: Cayley graph of the symmetric group \(S_n\) (following T36); internal states are permutations of length \(n\), transitions are adjacent transpositions \(s(k)\leftrightarrow s(k+1)\). \(L\) is a uniform distribution over all permutations, \(R(s)=\exp\!\big(\tfrac12\sum_{k=1}^n\mathbb I\{s(k)=k\}\big)/Z\) (T36 provides a closed-form for this normalization constant). Diagnostic metrics follow T36: \(C(k)\) is the probability of "exactly \(k\) fixed points" under the reward distribution, reporting the \(L^1\) error between the model's empirical estimate and the true value (Appendix B.2, estimated using the last \(10^5\) training samples). Reference values: for \(n=4\), \(\mathrm{OT}^\star=0.567\); for \(n=8\), \(\mathrm{OT}^\star=1.008\) (POT solver); for larger \(n\), exact \(\mathrm{OT}^\star\) is not computable.

| \(\lambda\) | \(n=4\) \(C(k)L^1\downarrow\) | \(n=4\) \(\mathbb E|\tau|\) | \(n=8\) \(C(k)L^1\downarrow\) | \(n=8\) \(\mathbb E|\tau|\) | \(n=20\) \(C(k)L^1\downarrow\) | \(n=20\) \(\mathbb E|\tau|\) |
|---|---|---|---|---|---|---|
| \(10^{-1}\) | 0.012 ±0.001 | 0.445 ±0.002 | 0.011 ±0.005 | 0.645 ±0.013 | 0.016 ±0.000 | 2.313 ±0.018 |
| \(10^{-2}\) | 0.002 ±0.000 | 0.557 ±0.001 | 0.001 ±0.002 | 1.010 ±0.011 | 0.002 ±0.000 | 4.436 ±0.014 |

(Numbers copied verbatim from original Table 2.)

**The trade-off for \(\lambda\) is quantitative in this table**: when \(\lambda=10^{-2}\), for \(n=4\), \(\mathbb E|\tau|=0.557\) vs \(\mathrm{OT}^\star=0.567\); for \(n=8\), \(1.010\) vs \(1.008\), while \(C(k)L^1\) drops to 0.002/0.001. When \(\lambda=10^{-1}\), paths become significantly shorter (0.445, 0.645) but \(C(k)L^1\) rises to 0.012/0.011, indicating that the shorter paths are achieved by sacrificing the terminal marginal. The description in original Sec. 4.2 is consistent with T36: larger \(\lambda\) leads to shorter trajectories and more biased sampling, while smaller \(\lambda\) leads to accurate sampling and longer trajectories.

**Evidence Strength Assessment**:

- Directly supported by experiments: On small scales (hypergrid \(H\le20\), permutations \(n\le8\)), the learning method can simultaneously approximate \(\mathrm{OT}^\star\) and the target marginal; the exact LP solution and the exact Kantorovich solution have the same numerical value (Figure 1); the directional trade-off of \(\lambda\).
- Inferred by authors, not directly verified by experiments: For \(n=20\), "produces a reasonable approximation" – this setting lacks an \(\mathrm{OT}^\star\) reference value, so we can only see that \(C(k)L^1=0.002\) indicates correct sampling marginals, but path optimality has no ground truth; "scalability" is supported by only one point, \(n=20\).
- Not provided in the original text: Wall-clock and memory comparison with network simplex / Sinkhorn / neural OT; error at the coupling matrix level (e.g., \(\|\Pi_\theta-\Pi^\star\|_1\)); measured primal-dual gap; hyperparameter search range beyond learning rate; specific values of \(H,D\) (Table 1 only gives \(H\), the main text states the state space is \(\{0,\dots,H-1\}^D\) but **does not give \(D\)**).

### 5. Premise Assumptions and Applicable Scope

Rephrased as applicable scope. **Theorem 3.2 holds under the following set of conditions**:

1.  **Finite directed graph**, cycles allowed. Cycles are necessary: on acyclic layered graphs (e.g., unidirectional hypergrids), all paths to a given \(x\) have the same length, the minimum flow principle degenerates to a constant, and there are no optimizable degrees of freedom in the OT objective (original Assumption 3.1 only requires finite + reachability, but flow theory itself requires a non-acyclic framework, see T36).
2.  **Endpoint structure**: \(s_0\) has no incoming edges, \(s_f\) has no outgoing edges; \(s_0\) only connects to \(U\), \(s_f\) only receives from \(\mathcal X\) (Assumption 3.1, first two points). This physically isolates the "source set" and "target set" on the graph, which is why the right-hand side of the divergence constraint can be written as \(L(s)-R(s)\).
3.  **All-to-all reachability**: there is a finite path between any \(u\in U\) and \(x\in\mathcal X\) (Assumption 3.1, third point). Otherwise, \(d(u,x)=\infty\), and mass in the coupling set has no path.
4.  **Both marginals are normalized probability distributions**: \(\sum_uL(u)=\sum_xR(x)=1\) (Assumption 3.1, fourth point), so \(Z=1\) is known. The original text states this comes from both a necessary corollary of the constraints (summing flow matching constraints) and implementation convenience.
5.  **Shortest path cost with unit edge length**: \(d(u,x)=|\tau_{u,x}|\) counts the number of edges (Eq. (7)), i.e., hop distance in an unweighted graph. Weighted graphs are not covered by the theorem. Since the graph is directed, \(d\) is not necessarily symmetric; it is a legitimate OT ground cost but not necessarily a metric in the usual sense.
6.  **Absorbing / Finite expected length**: The semantic meaning of expected flow visits requires the reverse chain to be an absorbing chain (T36 Lemma 3.4; O07 uses a weaker Assumption 3.1 \(\mathbb E[n_\tau]<\infty\) instead of \(P_B>0\)). The main text does not restate this, but it is a prerequisite for Eq. (3) to be well-defined.
7.  **Exact flow conservation + global optimality**: Theorem 3.2 is a statement about LP optimal solutions. Neural training uses the soft penalty of Eq. (20) (coefficient \(\lambda\)), which does not equal exact constraint satisfaction. Therefore, the step "the trained model is OT optimal" is supported by numerical proximity at the experimental level, not guaranteed by the theorem.
8.  **Theorem 3.3's dual optimality additionally requires \(R(x)>0\ \forall x\in\mathcal X\)** (third conclusion); this is the condition for the dual potential at the terminal state to be uniquely determined as \(d(x)\).

**Applicable scope in a nutshell**: Suitable for equilibrium OT on combinatorial graphs where "state space is too large to enumerate, cost matrix cannot be explicitly constructed, but local valid actions are clear and backtracking is allowed"—permutations/groups, molecule editing, program transformations, robot configuration spaces. Not suitable for small-scale explicit graphs (classic network simplex is faster, more accurate, and comes with certificates), weighted/continuous costs, or unbalanced OT with unequal mass.

### 6. Position in the GFlowNet × OT Main Line

**Predecessors (dependency chain, all indispensable)**:

-   **T19** (Brunswic et al. 2024, AAAI): Measure-theoretic framework for non-acyclic GFlowNets, 0-flow and loss stability, flow regularization idea. The hypergrid environment in this paper is directly taken from it (original Sec. 4.1).
-   **T36** (Morozov et al. 2025, ICML): The notation in Sec. 2.1 of this paper, flow definition, \(\mathbb E[n_\tau]=\frac1Z\sum_s\mathcal F(s)\) (Proposition 3.12), \(\lambda\mathcal F_\theta(s)\) regularization, permutation environment, and \(C(k)\) diagnostic protocol all come from it. Without T36's expected visit count flow, the objective function of this paper could not be written.
-   **O07** (Morozov et al. 2026, this repository O07, arXiv 2603.01786): Proves minimum total flow \(\Rightarrow\) only shortest paths are taken. Theorem 3.3 in this paper re-derives this conclusion using LP duality + complementary slackness (the original text explicitly states "recovers the corresponding claim"), and inherits the TB objective and the trick of per-prefix computation.
-   **OT side**: Beckmann (1952)'s continuous transport model, Essid & Solomon (2018) (this repository O02)'s quadratically regularized OT on graphs—this paper's Assumption 3.1 and min-cost-flow formulation both claim to be modeled after the latter; Villani (2008) Theorem 4.1 provides existence of optimal coupling.

**Successors and Comparisons**:

-   **C01 / ULOT** (arXiv 2506.12025, NeurIPS 2025): GNN + cross-attention amortized prediction of inter-graph FUGW (fused unbalanced Gromov-Wasserstein) plan, conditioned on tradeoff hyperparameters, two orders of magnitude faster than classic solvers, and can warm-start solvers. Its division of labor with this paper is clear: ULOT outputs **coupling matrices** and focuses on node alignment between two explicitly given graphs, and is natively unbalanced; this paper outputs **executable local routing policies** and focuses on an implicit, huge transition graph, requiring normalized marginals. ULOT has already occupied the amortized/conditional path; if this paper were to pursue "conditional GFN learning a family of \((L,R)\)", it would directly collide.
-   **C02 / GSBoG** (arXiv 2602.04675): Generalized Schrödinger bridge on graphs, learning CTMC control policies, fixing both marginals, and simultaneously optimizing state-dependent running cost, using IPF + TD objectives, claiming to be the first graph-based GSB formulation. It is most adjacent to this paper: both require "endpoint marginals + intermediate cost + executable policy". The differences are threefold—GSBoG is **continuous-time** CTMC and includes KL entropy regularization to a reference process (entropic OT family), while this paper is **discrete-step** and has no entropy term (pure LP, optimal solutions are at polyhedron vertices, can degenerate to deterministic routing); GSBoG's cost is a state cost functional, while this paper's cost is graph hop count; GSBoG relies on IPF/TD, while this paper relies on TB + flow regularization. To add entropy regularization on the GFN side to interface with Schrödinger bridge, one must directly answer "the relationship between the TB objective and path space KL", which is not automatically true.
-   **C03 / DDSBM** (arXiv 2410.01500, ICLR 2025): Discrete diffusion SB for graph transformation, also in the entropic camp.

**Contribution to the main line**: It provides the first geometric principle for "which internal flow to choose". Standard reward matching only fixes the terminal distribution, leaving the internal flow severely underdetermined (this underdetermination affects credit assignment and generalization to unseen states). This paper's answer is: **among all flows that correctly terminate the distribution, choose the one with the minimum transport cost**, and this choice is equivalent to an explicit OT problem, thus allowing the use of the entire OT toolkit (dual potentials, complementary slackness, primal-dual gap) to analyze and verify GFlowNet training. Conversely, it also brings GFlowNet into the domain of computational OT: on implicit combinatorial graphs where \(C_{ux}\) cannot be explicitly constructed, GFlowNet only needs to access local neighbors.

### 7. Reusable Insights and Open Problems

#### 7.1 What's Truly Novel: Not Equivalence Itself, But the Interface

"OT on graphs with shortest path as ground cost ≡ min-cost network flow" is a classic result, long stated this way in graph OT literature (the paper itself acknowledges this after Eq. (8) by citing Essid & Solomon 2018, and also attributes it to Beckmann 1952). Thus, the mathematical content of Theorem 3.2 is not news to OT researchers: Part 1 is "distributing mass along shortest paths," and Part 2 is "actual paths are no shorter than shortest paths." Both steps are undergraduate-level arguments.

The novelty lies in three aspects, all on the GFlowNet side:

1.  **Translation**: By replacing the primal variables of min-cost flow with GFlowNet edge flows, and constraints with detailed balance + reward/leward matching, the OT problem can for the first time be solved via "sampler training." Propositions A.1/A.2 in Appendix A.1 serve as the technical glue for this translation—they show that detailed balance and flow matching are the same constraint in the sense of policy reconstructability.
2.  **Output Upgrade**: Classic min-cost flow provides edge flows, requiring additional flow decomposition to obtain path-level policies; GFlowNet directly outputs \(P_F(s'\mid s)\), allowing for step-by-step transport execution without enumerating \((u,x)\) pairs or storing an \(|U|\times|\mathcal X|\) matrix. The paper's self-positioning in Sec. 5 is accurate: "addressing both where and how transport occurs."
3.  **Elevating Shortest Path Conclusions to OT Conclusions**: O07 deals with shortest paths from a single source (\(s_0\)) to multiple destinations; by adding the constraint \(\mathcal F(s_0\to u)=L(u)\), this paper transforms it into a multi-source to multi-destination problem, making OT the natural super-problem of shortest paths. This step incurs almost no technical cost but changes the entire problem class.

**Easily Misinterpreted Point** (and what I consider the most important one): Theorem 3.2 states **optimality value equality + optimal solution inducing optimal coupling**, not "any trained GFlowNet is doing OT." The "secretly" in the title holds only under two additional constraints: (i) the first-step edge flow is fixed to \(L\), and (ii) the total internal flow is minimized. Without (i), the first-step distribution is freely determined by \(P_F(\cdot\mid s_0)\), and the problem degenerates to O07's single-source shortest path; without (ii), any flow satisfying reward matching is valid, allowing mass to be infinitely accumulated in cycles (flow explosion in T19), which has nothing to do with OT. Ordinary GFlowNets do not "secretly" learn OT plans.

#### 7.2 Balance Residual → OT Error Bound (Our assessment: most promising direction, lowest collision risk)

**Current Status**: The paper only provides equations for the optimal solution, with no quantitative characterization of approximate solutions. During training, you have the TB residual \(\delta_\tau=\log\frac{L(s_1)\prod P_F}{R(s_{n_\tau})\prod P_B}\) (Eq. (20)); what you want are bounds for two things: source/target marginal errors, and the suboptimality of transport cost \(\mathrm{cost}(\Pi_\theta)-\mathrm{OT}^\star\).

**Why it's Feasible**: This setting is more favorable than general GFlowNets. First, \(Z=1\) is known, so there are no unknown constants in the TB residual (which is precisely the most difficult term to handle in standard TB→TV bounds). Second, both marginals are hard constraints, so the residual carries information about both "initial marginal error" and "terminal marginal error." Third, the cost function is \(\mathbb E_{\mathbb P_\theta}[|\tau|]\), which is the trajectory length, a directly estimable quantity, rather than an abstract functional requiring an additional model.

**Draft Proposition**: Let \(\mathbb P_\theta\) be the model trajectory distribution, \(\Pi_\theta\) be its endpoint marginalization coupling, \(\varepsilon_1=\mathrm{TV}(\text{first-step marginal},L)\), \(\varepsilon_2=\mathrm{TV}(\text{terminal marginal},R)\), and \(D=\max_{u,x}d(u,x)\) (a graph diameter-level quantity). Then

\[
\Big|\mathbb E_{\mathbb P_\theta}[|\tau|]-\mathrm{OT}^\star\Big|
\ \le\
\underbrace{\big(\mathbb E_{\mathbb P_\theta}[|\tau|]-\textstyle\sum_{u,x}d(u,x)\Pi_\theta(u,x)\big)}_{\text{detour relaxation,}\ \ge 0}
\;+\;
\underbrace{C\,D\,(\varepsilon_1+\varepsilon_2)}_{\text{cost of marginal violation}},
\]

The first term is directly obtained from the inequality in Eq. (18) (it is naturally non-negative and can be unbiasedly estimated by the sample mean of "actual path length − shortest path length between endpoints"); the second term requires the stability of OT value with respect to marginals (Lipschitz-in-marginals), which is a standard result for finite graphs with bounded costs. Then, by bounding \(\varepsilon_1,\varepsilon_2\) using the second moment of the TB residual, we obtain an end-to-end "residual → OT gap" bound. **This bound can directly explain the phenomenon of \(\mathbb E|\tau|<\mathrm{OT}^\star\) in Section 4.1**: the low reading is precisely traded for \(\varepsilon_2\ne0\), and the two terms in the bound have opposite signs. The collision scan results also support this direction: no direct competitors were found, and ULOT/GSBoG are not along this line.

#### 7.3 Primal-Dual: Dual Potentials as Critic, Complementary Slackness as Edge-wise Certificate

Theorem 3.3 / Theorem A.5 have already provided all the necessary materials, but the paper does not use them in the algorithm at all:

-   The optimal value of the dual potential \(\pi_s\) is the BFS distance \(d(s)\). This can be fitted by a network \(\pi_\phi(s)\), and the constraint \(\pi_{s'}-\pi_s\le1\) is edge-wise and can be penalized by sampling—this functionally completely overlaps with approaches like DeepCubeA / CayleyPy that "learn distance-to-goal," but here it has a clear LP dual identity, not a heuristic.
-   **Complementary slackness provides free edge-wise certificates**: \(\mathcal F^\star(s\to s')(\pi^\star_{s'}-1-\pi^\star_s)=0\). During training, one can monitor \(\sum_{s\to s'}\mathcal F_\theta(s\to s')\cdot|\pi_\phi(s')-1-\pi_\phi(s)|\) as a proxy for the primal-dual gap. This can be evaluated for each edge and does not require a global solver. This is a rare, computable optimality indicator in GFlowNet training (a zero TB residual only guarantees consistency, not optimality).
-   Specific experiment: actor = \((P_F,P_B)\), critic = \(\pi_\phi\), loss = regularized TB + dual feasibility penalty + gap penalty. A measurable hypothesis is that "adding the critic leads to lower \(\mathbb E|\tau|\) on permutations \(n=20\) while keeping \(C(k)L^1\) unchanged." Note that in the dual of the extended problem (Appendix A.4), \(a_x=\pi_x,\ b_u=-\pi_u\) are generated by the same potential function, so only one network is needed, not two.

#### 7.4 Differentiation from ULOT (arXiv 2506.12025): Avoid Amortized Prediction

ULOT's selling points are: GNN + cross-attention, conditioned on FUGW's \((\rho,\alpha)\) hyperparameters, \(O(n_1n_2)\) complexity directly outputs the plan, two orders of magnitude faster than classical solvers, plan differentiable with respect to input graphs, and can warm-start IBPP solvers. It is inherently unbalanced, inherently conditional, and inherently amortized.

Therefore, the seemingly most natural next step, "training a conditional GFlowNet to cover a family of \((L,R)\)", has a high overlap in value proposition with ULOT. Moreover, GFlowNet has no structural advantage in **node alignment between two explicitly given small-to-medium graphs**—it requires rollouts, has higher variance, and needs to handle cycles.

There is only one moat for differentiation that must be defended: **the graph is implicit, the cost matrix cannot be constructed, and transport can only be executed via legal local actions**. Permutations with \(n=20\) (\(\sim2.4\cdot10^{18}\) states, see O07 Sec. 4.1) are already in this range; molecular editing, program/proof transformations, and robot configuration spaces also fall here. Comparable experiments should acknowledge that it cannot beat (or is on par with) ULOT on SBM graphs where ULOT can run, but on Cayley graphs where \(|U||\mathcal X|\) cannot be enumerated, ULOT cannot even construct the input. Clearly defining this boundary is more convincing than merely improving scores.

#### 7.5 Differentiation from GSBoG (arXiv 2602.04675): Entropy Regularization is Not Automatic

GSBoG solves the generalized Schrödinger bridge on graphs: fixing marginals at both ends, it minimizes \(\mathbb E[\int f_t\,\mathrm dt]+\mathrm{KL}(p^u\|p^r)\) over path space, controlling the jump rates of a CTMC. Training uses IPF (path-version of Sinkhorn) + a TD objective (because the state cost term cancels out in IPF aggregation, IPF alone is insufficient to constrain it).

Three structural differences make them not directly interchangeable:

1.  **Presence/Absence of Entropy Term**. This paper uses unregularized LP, where the optimal solution can be a vertex solution and degenerate to deterministic routing (Theorem 3.3 states flow only falls on shortest path subgraphs). GSBoG's KL term makes the solution strictly stochastic and its support covers all feasible paths. To turn GFN-OT into an entropic version, one needs to explicitly add path KL to a reference policy \(P^0\) in the objective, and **standard TB is not automatically equivalent to path space KL minimization**—TB is a consistency constraint, not a divergence objective; this step must be re-derived (this is also a high-collision but unsolved technical point already noted in the background analysis of this repository).
2.  **Time Parameterization**. GSBoG has continuous time \(t\in[0,1]\) and time marginals; this paper only has discrete steps and no concept of "where the mass is at \(t=0.5\)". To compare intermediate behavior, one must first introduce time conditioning (or step conditioning) to the GFN side, which is a modeling choice in itself.
3.  **Expressiveness of Cost**. GSBoG supports state- and distribution-dependent running costs (including mean-field terms); this paper only has hop count. **The easiest extension for the GFN side to incorporate is replacing \(\lambda\mathcal F_\theta(s)\) with \(\lambda c(s)\mathcal F_\theta(s)\)**, i.e., state-dependent flow penalties. In this case, the LP objective becomes \(\sum_s c(s)\mathcal F(s)\), and the ground cost becomes weighted shortest path—for strictly positive \(c\), the two-part argument of Theorem 3.2 holds almost verbatim (Part 1 lays mass along weighted shortest paths, Part 2 uses \(\mathrm{cost}(\tau_{u,x})\ge d_c(u,x)\) for bounding). Zero-weight edges would introduce zero-cost cycles, breaking the acyclicity of optimal flow, and must be excluded. This is a low-risk, high-certainty theorem-level increment.

#### 7.6 Unbalanced and Unknown-\(Z\): GFlowNet Abandons Its Strongest Capability

Standard GFlowNet's hallmark is requiring only unnormalized \(R\). To ensure mass balance for min-cost flow, this paper requires \(\sum_uL(u)=\sum_xR(x)=1\) (Assumption 3.1) and explicitly states in Sec. 3.1 that this is to avoid dealing with unknown normalization constants. The cost is that \(L(s_1)\) replaces \(Z\) in Eq. (20), and the model no longer learns \(Z\).

The open problem is very specific: add a "discard/create" edge between sink \(s_f\) and source \(s_0\) and assign it a unit cost \(\kappa\). The LP then becomes unbalanced OT with a penalty (equivalent to \(\ell_1\)-type marginal relaxation). In this case, mass balance is absorbed by the virtual edge, and \(L,R\) need not have equal mass. The questions to answer are: (i) how the two-part argument of Theorem 3.2 is rewritten when virtual edges exist (Part 1 needs to explicitly route "untransported mass" to the virtual edge); (ii) whether TB remains stable after \(Z\) becomes unknown again in the training objective; (iii) whether GFN's advantage over ULOT's FUGW-unbalanced version still lies only in implicit graphs. This is a crucial step to reconnect GFlowNet's native capabilities to OT, and it is currently completely unexplored.

#### 7.7 Do Short Flows Benefit Generalization: Unverified, and Potentially Negative

This paper uses "shortest" as the internal flow selection principle but has not measured its impact on generalization. Minimum flow solutions concentrate mass on the shortest path subgraph (complementary slackness of Theorem 3.3), and the support set can be very narrow. This has two measurable consequences, in opposite directions: positive is shorter rollouts, more direct terminal credit, and cheaper inference; negative is exploration collapse, less sub-structure sharing, and worse policy extrapolation to unseen states.

Feasible experiment: scan \(\lambda\) in the permutation environment (using the two existing settings from Table 2 and adding more points), and in addition to \(C(k)L^1\) and \(\mathbb E|\tau|\), report: coverage of visited states, entropy of the terminal state support, and greedy success rate on states unseen during training (O07 Table 1 evaluation protocol can be directly adopted). The hypothesis is that "\(\lambda\) and generalization form an inverted U-shape"—if true, "minimum flow" cannot be unconditionally considered the correct internal flow principle, but rather needs to be formulated as a regularized version.

#### 7.8 Other Open Problems (Brief)

-   **Consequences of Directed Asymmetry**: Theorem 3.2 still holds when \(d(u,x)\ne d(x,u)\) (the proof does not use symmetry), but the resulting "distance" is not a metric, so it cannot be directly called Wasserstein. If downstream tasks use it for distance comparison, separate justification is needed.
-   **Bridge between Exact LP and Neural Solutions**: The paper uses `scipy.linprog` to verify small-scale equivalence (Sec. 4.1, Figure 1). For intermediate scales (e.g., \(H=30,D=3\)), a comparison of "LP solution vs. neural solution" at the edge flow level (not just scalar cost) should be performed to determine whether the neural solution finds another optimal solution with the same cost or is near the same solution.
-   **\(D\) Not Reported**: Table 1 only gives \(H\), and the main text states the state space is \(\{0,\dots,H-1\}^D\), but **the original text does not provide the value of \(D\)**; Figure 1's visualization is a 2D grid, suggesting that graph corresponds to \(D=2\), but it's unclear from the original text whether the three settings in the table use the same \(D\). This number must be determined for reproduction.

### 8. References

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

Related references (comparison works used in this report):

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

### Editorial note

1.  **Publication Status**: The PDF footer only states "Preprint. June 5, 2026", and there is no venue line in the main text. According to the `data/papers.yaml` repository entry, `ICML 2026 SPIGM Workshop`, it is recorded as a Workshop paper (not main conference). This distinction has been noted in both the report header and the `note` field of the bibtex entry.
2.  **Theorem Numbering**: The main theorem in the original text is **Theorem 3.2** (not 3.1, as 3.1 is an Assumption), and its dual conclusion is **Theorem 3.3**. Its full version is labeled **Theorem A.5** in Appendix A.3. At the end of Sec. 3.2, the original text states "Detailed proof of Theorem 3.3 can be found in Appendix A.5", while in the Appendix, this theorem is located in Section **A.3** and numbered A.5. This is a minor typo in the original text, and this report cites it according to its actual content. Another instance: Appendix A.4 begins with "see Appendix A.4 for derivation" as a self-reference, which is also a typo.
3.  **Emphasis on the constant 1 in §3.2** is my judgment (the original text only states "we will omit this constant"), because it determines the relationship between \(\mathrm{GFlow}^\star\) and \(\mathbb E[n_\tau]\), and is the most common source of error during reproduction.
4.  **Explanation of \(\mathbb E|\tau|<\mathrm{OT}^\star\) in §4.1** (\(H=20\) Moon: 9.001 vs 9.059) was not discussed in the original text. This is my inference based on the marginals not being precisely satisfied, and it has been noted in the main text.
5.  **The bounds in §7.2 are my draft**, not conclusions from the paper; the first term comes from the scaling direction of Eq. (18) in the original text, and the second term's Lipschitz-in-marginals is standard OT stability, requiring a separate citation.
6.  **The weighted extension in §7.5** (\(\sum_sc(s)\mathcal F(s)\)) is my inference. The original text only mentions in Sec. 5 that this item is not covered in future work; O07 Sec. 5 lists weighted graphs as future work, which can be seen as author endorsement in the same direction.
7.  **Code**: Neither the main text nor the appendix of the original paper provides a link to the code for this paper (it only states that the implementation is based on the public code of T36). Therefore, the metadata `code_url` is recorded as `null`.
8.  **Hypergrid dimension \(D\)**: Not provided in the original text. The report handles this as "not provided in the original text" without any imputation.

# Chapter 6: Optimal Transport Side: Prerequisites and Related Work

## 6.1 O01 · Optimal Transport for Machine Learners (Peyré's Lecture Notes) — A Guide for GFlowNet Readers

> **One sentence summary**: Peyré's 460-page lecture notes serve as the "dictionary" for the OT side of this repository.
> For GFlowNet readers, what makes it truly indispensable is not Wasserstein distance or Sinkhorn itself,
> but rather **§6.5**, which expresses \(\mathcal W_1\) in the form of "edge flow on a graph + vertex conservation" —
> that is precisely the object of GFlowNets, word for word.
> The task of this guide is to provide a "OT concept ↔ GFlowNet concept" correspondence table,
> and to **clearly mark which correspondences are strictly equivalent and which are merely analogous**, to avoid treating "GFlowNet does OT" as an unconditional slogan.

| Field | Content |
|---|---|
| arXiv | [2505.06589](https://arxiv.org/abs/2505.06589) (This repository's copy is v3, stat.ML, 2026-08-08; title page dated August 11, 2026) |
| Publication | **Lecture notes / book manuscript**, not a conference or journal paper. Author describes it as "an OT textbook organized with ML as the driving force" |
| Author | Gabriel Peyré (CNRS and ENS, PSL Université) |
| Code | All figures and code can be found at `gpeyre/ot4ml`; most computational graphs are generated using the Python Optimal Transport (POT) library (explicitly acknowledged on the abstract page) |
| This repository's PDF | `papers/2505.06589.pdf` (480 PDF pages / 16 chapters + conclusion + notation table + index, main text pages up to around 460) |
| Reading Priority | P0 — but **do not read cover-to-cover**. Use the chapter map in Section 2; only about 40 pages of the entire book are directly relevant to the main thread of this repository |

#### 2. Chapter Map: Which Chapters Should GFlowNet Readers Read, and What Does Each Chapter Address?

The book has 16 chapters. They are divided into four tiers based on their relevance to the main theme of this repository. **Tier A, totaling approximately 40 pages, contains all essential reading material.**

| Tier | Chapter | Start Page | What this chapter addresses | Why GFlowNet readers need it |
|---|---|---|---|---|
| **A** | **§6.5 Wasserstein-1** | p.108 | Rewrites \(\mathcal W_1\) from "point-to-point coupling" to "flow on edges + vertex conservation": Kantorovich–Rubinstein duality Eq. (6.23), Beckmann problem Def. 6.20, **graph version Def. 6.22 + Prop. 6.23** | This is the only place in the book where the OT variables are **edge flows on a graph**, and the constraints are **conservation at each vertex**. This is the same equation as GFlowNet's flow matching condition. |
| **A** | **§5.1–5.3 Dual Problem** | p.84 | Dual potential Def. 5.1, strong duality Prop. 5.2/5.5, **complementary slackness Prop. 5.3/5.7**, \(c\)-transform Def. 5.8 and its regularization effect Prop. 5.9/5.10 | The strict origin of the interpretation "state flow \(\log F(s)\) = dual potential/value function". Complementary slackness explains "optimal flow can only live on the contact set" = sparse support. |
| **A** | **§14.3 Path-Space Schrödinger** | p.314 | Path-space transport Def. 14.8, **endpoint reduction Prop. 14.9 / Prop. 14.11**, Schrödinger bridge Def. 14.10, KL chain decomposition Eq. (14.20) | The only place that rigorously handles the relationship between "trajectory distribution vs. endpoint coupling". Directly answers "what is the difference between GFlowNet's path entropy regularization and OT's endpoint entropy regularization". |
| **A** | **§3.1 Discrete Relaxation** | p.42 | Coupling polytope Def. 3.1, LP structure Def. 3.5, **sparse optimal solutions Prop. 3.8**, Birkhoff–von Neumann Thm. 3.17, integerization of rational weights Prop. 3.21 | This entire language: "flow is underdetermined, feasible set is a polytope, optimal solutions can be sparse vertices". Remark 3.2 provides a precise explanation of single-source degeneration. |
| **A** | **§9.1 Bregman Perspective** | p.157 | Sinkhorn = performing **alternating KL projections** on two affine marginal constraint sets: Eq. (9.1), Algorithm 9.1, Prop. 9.4, **Prop. 9.5 (projection is row/column scaling)** | Clearly explains the algorithmic skeleton of "alternatingly enforcing two constraints". The two-stage GTB in T10 is isomorphic to this. |
| **B** | §8.1–8.5 Entropic Regularization and Sinkhorn | p.123 | Entropic regularization Def. 8.2, scaled form Prop. 8.5, KL rewrite Eq. (8.8), two limits of \(\varepsilon\) Prop. 8.10, duality Prop. 8.18, **soft-\(c\)-transform Def. 8.19** | soft-min = log-sum-exp is isomorphic to GFlowNet's log-domain loss; Prop. 8.10 provides a quantitative version of "larger entropy tends towards independent coupling". |
| **B** | §3.2 LP Algorithms | p.53 | Transportation simplex, network simplex, Orlin's strongly polynomial minimum cost flow, interior point method Eq. (3.5) | Explains that "minimum cost flow on graphs" has mature exact solutions, serving as a baseline for evaluating the quality of GFlowNet's approximate solutions. |
| **B** | §14.1–14.2 Dynamic OT | p.308 | Continuity equation Def. 14.1, **Benamou–Brenier Thm. 14.5**, momentum convexification Def. 14.6 | The continuous version of "flow = conservation law + minimum kinetic energy"; helps understand why GFlowNet's discrete counterpart is Beckmann and not BB. |
| **B** | §2.3 Monge Form | p.16 | Monge problem Def. 2.14, Monge map on empirical measures is a permutation Prop. 2.15, **indivisibility of mass as an obstacle Example 2.18** | Explains why "plan" must be used instead of "map" – GFlowNet's stochastic policy is naturally on the plan side. |
| **B** | §4.1 Wasserstein Distance | p.64 | Def. 4.2/4.6, gluing lemma Lemma 4.4, triangle inequality Prop. 4.3/4.7 | Minimum knowledge required when using \(\mathcal W_p\) as an evaluation metric. |
| **C** | §1.1–1.2 Assignment Problem | p.1 | Def. 1.1, one-dimensional sorting optimality Prop. 1.2, Hungarian algorithm Prop. 1.8, dual certificate Prop. 1.7 | Builds intuition that "discrete OT is essentially combinatorial optimization", can be skipped. |
| **C** | §12.5 Metric Learning and Inverse OT | p.258 | Inferring cost from observed coupling | This is the OT version of T10's guided TB "learning a guiding distribution = learning a cost". |
| **C** | §11.1 Unbalanced OT | p.201 | Relaxing marginal constraints | Corresponds to GFlowNet variants where rewards are not normalized, or "leaky flow" is allowed. |
| **C** | §15.1 Wasserstein Gradient Flow | p.335 | JKO implicit Euler Def. 15.1, Wasserstein gradient Def. 15.5, Eq. (15.4) | **Only need to know this framework exists**. It describes gradient descent in measure space, which is not the same as GFlowNet's parameter space training; do not over-analogize. |
| **D (Skip)** | §7 Divergences and Dual Norms, §10 Statistical OT, §11.2–11.6, §12.1–12.4, §13, §16 | — | MMD/GAN, sample complexity, sliced/low-rank/multi-marginal/Gromov–Wasserstein, flow matching generative models | No direct relation to the main theme of "flow selection on discrete graphs". |

**Reading suggestions.** First, read §6.5 (approx. 5 pages) to understand "what OT looks like on graphs".
Then, go back to §3.1 to complete the polytope language, followed by §5.1–5.3 to obtain the dual potential.
Finally, §14.3 addresses the distinction between trajectories and endpoints. §9.1 should be consulted when designing training algorithms.

#### 5. Premise Assumptions and Applicable Boundaries

The lecture notes themselves are a mathematics textbook, so the premises are clearly stated. For GFlowNet readers, the following points determine how far the correspondence table can be extended.

1.  **Compactness and Continuity**. Continuous duality (Prop. 5.5) yields the maximum attainable when \(\mathcal X, \mathcal Y\) are compact and \(c\) is continuous;
    in general cases, it degrades to lower semi-continuity + integrability assumptions, and max becomes sup. On discrete finite graphs, all these conditions are automatically satisfied.
2.  **Balanced Mass**. Chapters 3–9 exclusively deal with **balanced** OT where \(\alpha, \beta\) are both probability measures. Relaxing marginals is covered in §11.1 (Unbalanced OT).
    If "flow leakage" or non-normalized rewards are allowed in GFlowNets, it corresponds to the unbalanced branch, and Prop. 6.23 cannot be directly applied.
3.  **Positivity/Finiteness of Cost**. Prop. 8.3 and Prop. 8.5 require \(\mathbf C\) to be finite and \(\mathbf K>0\);
    if \(c\) can take \(+\infty\) (i.e., some edges do not exist), existence and uniqueness require additional support conditions (explicitly stated in §8.2).
    **Unreachable state pairs in directed graphs correspond to the case where \(c=+\infty\)**, which is a practical limitation in GFlowNet applications.
4.  **Graph Beckmann requires connectivity, positive edge lengths, and \(\sum_i r_i=0\)** (used in the proof of Prop. 6.23).
    The lecture notes do not provide a version for directed graphs—\(d_G\) in Def. 6.22 is an undirected graph geodesic distance,
    and Remark 6.24 describes a transportation problem where each undirected edge is split into two directed arcs.
    **GFlowNet's DAGs are directed and irreversible, which is not covered by Prop. 6.23**, and needs to be supplemented.
5.  **Beckmann/flow form only exists for \(p=1\)**. The specialty of \(\mathcal W_1\) comes from Prop. 6.17:
    when \(c=d\), the image of the \(c\)-transform is precisely 1-Lipschitz functions, and \(f^c=-f\), so duality collapses to a single potential.
    For \(p=2\), it corresponds to Benamou–Brenier (Thm. 14.5), which is a **time-continuous** flow, not a static edge flow on a graph.
    Analogizing GFlowNets to \(\mathcal W_2\) requires additional justification.
6.  **Sinkhorn's convergence conclusion is for algorithm convergence with "fixed \(\varepsilon\), fixed marginals"** (explicitly distinguished at the beginning of Ch. 9);
    statistical convergence (marginals changing with samples) is a separate problem in Ch. 10.
7.  **Entropy regularization changes the problem itself, not just the algorithm** (contrast illustrated in Fig. 3.8):
    \(\varepsilon\) in interior point methods is a barrier parameter that tends to 0 along the central path;
    \(\varepsilon\) in Sinkhorn is usually fixed and is **part of the objective function**. Conflating these two will lead to incorrect limit conclusions.

#### 6. Position in the GFlowNet × OT Main Line: Conceptual Correspondence Table

**Criterion for Judgment**:
"**Strict**" = The two sides are the same mathematical object, can be translated symbol by symbol, and there are corresponding propositions in the lecture notes;
"**Conditionally Strict**" = Strictly equivalent under clearly stated additional conditions, which are listed;
"**Analogous**" = Structurally isomorphic but not the same object; theorems cannot be directly transferred.

| # | OT Concept | Lecture Notes Reference | GFlowNet Concept | Judgment | Explanation |
|---|---|---|---|---|---|
| 1 | Coupling \(\mathbf P\in\mathcal U(\mathbf a,\mathbf b)\) | Def. 3.1, Eq. (3.1); Def. 3.5, Eq. (3.2) | Trajectory distribution \(P_F(\tau)\) (endpoint perspective) | **Analogous**, degenerate under standard settings | For a single source, \(\mathcal U(\delta_{s_0},\beta)\) is a singleton set (Remark 3.2). Must relax initial flow distribution to be non-trivial |
| 2 | Directed edge flow \(m_e\), objective \(\sum_e\ell_e\lvert m_e\rvert\) | Def. 6.20; **Prop. 6.23** | Edge flow \(F(s\to t)\) | **Strict** (on undirected graphs) | Same variable, same objective form. Directed DAG version not given in lecture notes, see §5 Boundary 4 |
| 3 | Marginal constraints \(\pi_1=\alpha,\pi_2=\beta\) | Def. 3.1, Eq. (3.1); Def. 3.25 | Reward matching \(F(s\to x)=R(x)\) at terminal states | **Conditionally Strict** | Requires normalizing \(R\) to \(\beta=R/Z\). Source side corresponds to \(F(s_0)=Z\) |
| 4 | Conservation law \(\mathrm{div}_G m=r\) | **Prop. 6.23**; continuous version Def. 14.1, Eq. (14.2) | Flow matching: inflow = outflow for internal states | **Strict** | This is the strongest correspondence in the table. \(r=\mathbf a-\mathbf b\) is net supply; for internal vertices \(r_i=0\), i.e., \(\mathrm{div}\,m=0\) |
| 5 | Cost \(c(x,y)\) = graph shortest path \(d_G\) | Def. 6.22; path version Eq. (14.16) + **Prop. 14.9** | Trajectory length / expected number of trajectory steps | **Conditionally Strict** | Condition: path action is "sum of edge lengths". Prop. 14.9 guarantees "path space problem = Kantorovich problem with cost \(c_{\mathcal A}\)" |
| 6 | Kantorovich dual potentials \(f_i\) (vertex scalars, \(\lvert f_i-f_j\rvert\le\ell_e\)) | Def. 5.1, Eq. (5.1); **Prop. 6.23** left side | State flow \(\log F(s)\) / value function | **Analogous** (with a strict core) | Both are "vertex scalars + edge relations". Difference: OT has **inequality** constraints, DB has **equality**. Equality = already on the contact set of complementary slackness |
| 7 | Complementary slackness: support ⊂ contact set | **Prop. 5.3, Eq. (5.4)**; Prop. 5.7, Eq. (5.7) | Optimal flow is positive only on "tight" edges | **Strict** (in LP form) | Provides a mechanistic explanation for sparse support, combined with the \(n+m-1\) upper bound from Prop. 3.8 |
| 8 | Entropy regularization \(-\varepsilon H(\mathbf P)\) / \(\varepsilon\mathrm{KL}(\pi\|\alpha\otimes\beta)\) | Def. 8.2, Eq. (8.1); Eq. (8.8); Def. 8.13 | Path entropy of MaxEnt GFlowNet \(H[F]=\mathbb E_\tau\sum_t H[P_F(\cdot\mid s_t)]\) | **Analogous**, differs by a conditional term | OT entropy is on **endpoint couplings**, GFN entropy is on **paths**. The difference between them via Eq. (14.20) is \(\int\mathrm{KL}(M_{x,y}\|\mathcal R^{\varepsilon,x,y})\mathrm d\pi\) |
| 9 | Sinkhorn scaling \(\mathbf P=\mathrm{diag}(\mathbf u)\mathbf K\mathrm{diag}(\mathbf v)\) | Prop. 8.5, Eq. (8.2); Eq. (8.5) | \(F(s\to t)=F(s)P_F(t\mid s)=F(t)P_B(s\mid t)\) | **Analogous** | Both sides are "edge quantities = two vertex factors × a kernel". GFN's "kernel" is the graph's adjacency structure |
| 10 | Sinkhorn = alternating KL projection | Eq. (9.1)(9.2); **Prop. 9.4/9.5** | Training with alternating enforced constraints (e.g., T10 two-stage GTB) | **Analogous** (algorithmic isomorphism) | Essential difference: each half-step of OT has a **closed-form** projection (row/column scaling), GFN can only approximate with SGD |
| 11 | soft-\(c\)-transform \(-\varepsilon\log\sum_j e^{-h_j/\varepsilon}\mathbf b_j\) | **Def. 8.19, Eq. (8.21)**; Alg. 8.2 | DB/TB residuals in log domain and logsumexp normalization | **Analogous** | Both are "softening of hard min". GFN's \(\log Z\) and \(\log F\) act as an additive gauge, isomorphic to the gauge freedom of OT potentials |
| 12 | Schrödinger bridge reference path law \(\mathcal R^\varepsilon\) | Def. 14.10, Eq. (14.17); **Prop. 14.11** | Fixed backward policy \(P_B\) | **Conditionally Strict** | After fixing Markovian \(P_B\), TB is globally uniquely optimal (T02 Cor. 1), corresponding to "given a reference bridge, match endpoint marginals" |
| 13 | Birkhoff–von Neumann decomposition | Thm. 3.17; **Cor. 3.20** | Stochastic policy = convex combination of deterministic policies | **Analogous** | OT side is permutation decomposition under uniform marginals; GFN side's "deterministic policies" do not form permutation matrices |
| 14 | Benamou–Brenier kinetic action | Def. 14.4; **Thm. 14.5, Eq. (14.8)** | Minimum action interpretation of "flow × velocity" | **Analogous** (do not use as a theorem) | BB is continuous time and continuous state; the correct discrete counterpart on graphs is Beckmann (Prop. 6.21/6.23), not BB |
| 15 | Wasserstein gradient flow / JKO | Def. 15.1, Eq. (15.1); Def. 15.5 | GFlowNet training dynamics | **Atmospheric analogy only** | JKO is implicit Euler on measure spaces, GFN training is SGD on parameter spaces. Unless otherwise constructed, do not establish correspondence |
| 16 | Inverse OT: inferring cost from observed coupling | §12.5 (p.258); Centroid projection near Prop. 12.31 | T10's guiding distribution \(p(\tau_{\to x}\mid X)\) | **Analogous** | Both sides involve "cost/preference being learned". Can serve as an interface to formalize guided TB |

**The role of this table in the main narrative.**
Rows 2, 4, 5, and 7 together form the skeletal theorem of O08:
With a fixed initial flow distribution, the objective of a minimal flow GFlowNet degenerates into a **Kantorovich problem with graph shortest paths as costs**,
whose optimal solution is the transport plan.
Prop. 6.23 provides the static form of "edge flow + conservation," and Prop. 14.9 provides the reduction from "path → endpoint."
Together, these two form the complete translation path from GFlowNet to OT.
O07 (learning shortest paths with GFlowNet) takes the other end of the same path: first define the cost as the shortest path, then ask if the flow will automatically follow the shortest path.

#### 7. Reusable insights and open problems

1.  **Generalizing Prop. 6.23 to directed DAGs is a theorem draft that can be directly worked on.**
    The lecture notes' Beckmann graph is based on **undirected** graph geodesic distances (Def. 6.22).
    Remark 6.24 obtains a transportation problem by splitting undirected edges into two reverse arcs.
    GFlowNet graphs are directed and **irreversible**: \(s\to t\) existing does not imply \(t\to s\) existing.
    In this case, \(d_G\) is no longer symmetric, and the 1-Lipschitz characterization of Kantorovich–Rubinstein (Prop. 6.17) needs to be replaced with
    "\(f_j-f_i\le\ell_e\) single-edge constraint," and flows must also be non-negative \(m_e\ge0\).
    Proposition draft: On a directed connected DAG, \(\min\{\sum_e\ell_e m_e:m\ge0,\ \mathrm{div}_G m=r\}
    =\max\{\sum_i f_ir_i:f_j-f_i\le\ell_e\}\), and the support of the optimal solution is acyclic.
    Once this is clearly stated, rows 2 and 4 of the table in Section 6 will be upgraded from "undirected strict" to "DAG strict."
2.  **The sparsity upper bound is an empirically testable prediction.**
    Prop. 3.8 gives an upper bound of \(n+m-1\) non-zero elements, and Remark 3.19 states that the support is acyclic.
    In enumerable environments like hypergrid or SIX6, truncate the edge flows of a trained GFlowNet by a threshold,
    and measure the size of the effective support and whether it contains cycles. If it far exceeds the upper bound, it indicates that the training objective did not push the solution towards an LP vertex—
    this precisely quantifies what T10 Remark 2 calls "flow is underdetermined."
3.  **The difference between "path entropy vs endpoint entropy" \(\int\mathrm{KL}(M_{x,y}\|\mathcal R^{\varepsilon,x,y})\mathrm d\pi\) is computable.**
    The decomposition in Eq. (14.20) is a finite sum on finite DAGs.
    Experimental draft: Train MaxEnt GFlowNet and entropy-regularized OT solutions separately on small DAGs,
    explicitly calculate this conditional term, and see if it increases with "the number of trajectories for each \(x\)."
    If it increases, then the common statement "MaxEnt GFlowNet ≈ entropy-regularized OT" is incorrect in multi-trajectory environments—
    and T10 Prop. 5.2's \(\Theta(1/(n-k))\) precisely hints at this.
4.  **Closed-form projection is the sole source of Sinkhorn's advantage, and it is worth asking if GFlowNet has a counterpart.**
    Remark 8.4's judgment is straightforward: entropy is placed on the **elements** of \(\mathbf P\), and constraints are only row and column marginals.
    This separable structure causes Bregman projection to degenerate into diagonal scaling.
    GFlowNet's constraints are "one conservation law per vertex," which is also separable on trees.
    Open question: On directed trees (autoregressive MDPs), does DB's KL projection have a closed form?
    If so, we would obtain a GFlowNet training algorithm that does not require SGD, which could serve as a strong baseline.
5.  **Network simplex is an overlooked evaluation baseline.**
    §3.2 and Remark 6.24 both point out that min-cost flow on sparse graphs has mature exact solutions (network simplex, Orlin's strongly polynomial algorithm).
    O07/O08 in this repository both work on graphs and should report "the gap with exact min-cost-flow solutions."
    O08's abstract indeed mentions "consistent with exact OT solvers," and this baseline should be systematized.
6.  **Inverse OT (§12.5) is a ready-made interface for formalizing guided TB.**
    What T10's guiding distribution \(p(\tau_{\to x}\mid X)\) does, in OT language, is
    "infer a cost from observed high-reward samples that makes these samples optimal transport targets."
    §12.5 provides standard formulations for such problems and a framework for deriving differentiable OT losses.
    Rewriting guided TB as inverse OT can immediately lead to questions not previously asked, such as "is the guiding distribution unique, is it identifiable?"
7.  **Two areas not covered by the lecture notes but needed for the main narrative.**
    First: **OT on directed graphs/asymmetric costs** (point 1 above).
    Second: **Simultaneously optimizing coupling and internal routing**—the lecture notes separate "endpoint coupling" and "path filling" into two steps
    (Prop. 14.9 selects coupling first, then fills optimal paths),
    whereas GFlowNet learns both simultaneously within a single policy.
    These two areas are the remaining true gaps in the main narrative "internal flow selection = optimal transport."

## 6.2 O02 · Quadratic Regularized Optimal Transport on Graphs (SIAM J. Sci. Comput. 2018)

> **One sentence summary**: This paper formulates the 1-Wasserstein distance on graphs as a min-cost network flow on edge flows (Beckmann form, Eq. (3)), adds a quadratic regularization \(\frac\alpha2\sum_eJ_e^2\) to the cost to ensure solution uniqueness, and proves that for sufficiently small \(\alpha\), the regularized solution is one of the solutions to the unregularized LP (Prop. 4, Corollary 1); the Hessian of the dual problem is the graph Laplacian of the active subgraph (Prop. 2), leading to a Newton-type algorithm with closed-form line search and rank-one Cholesky updates. Its value to O08 is not in the algorithm, but in the mathematics on page Sec. 3.1: Kantorovich coupling (cost = graph shortest path) ⇔ min-cost flow on edge flows, and the flow decomposition theorem (Theorem 1) translates solutions between the two.

| Field | Content |
|---|---|
| arXiv | [1704.08200](https://arxiv.org/abs/1704.08200) (v4, 2018-03-23, math.OC; this repository's PDF is this version) |
| Publication | Journal: SIAM Journal on Scientific Computing 40(4): A1961–A1986, 2018, DOI 10.1137/17M1132665 (Submitted 2017-06-01, Accepted 2018-03-08, Online 2018-07-03) |
| Authors | Montacer Essid (NYU Courant), Justin Solomon (MIT) |
| Code | Link not provided in original text; Algorithm 1 caption mentions "accompanying Matlab implementation," Sec. 6 states use of SuiteSparse/CHOLMOD |
| This repository's PDF | `papers/1704.08200.pdf` · Chinese translation `papers_zh/1704.08200.zh.pdf` |
| Reading Priority | P1: The mathematics for "Kantorovich ⇔ min-cost flow on graphs," which is required for O08's "GFlowNet implicitly learns OT plans," is most cleanly stated here; the algorithm section is irrelevant to the main narrative |

#### 2. Core Contributions (Numbered as in Original Text)

**C1. Beckmann Formulation of \(W_1\) on Graphs (Sec. 3.1, Eq. (1) ⇔ Eq. (3)).** Conclusion: The transportation cost between two points can be decomposed into edge-wise costs along the shortest path. Thus, Eq. (1) and Eq. (3) yield the same \(W_1\). The original text treats this step as known ("Formalizing this argument provides an alternative to (1)"), provides no formal proof, and refers to Santambrogio 2015's Beckmann problem.

**C2. Motivation for Quadratic Regularization (Sec. 3.2).** Three drawbacks of entropy regularization in the graph setting: it is written on \(T\) rather than \(J\); alternating projections converge slowly and have numerical issues as \(\alpha\to0\); and \(T_{vw}>0\) strictly holds for any regularization strength (losing sparsity). Two reasons for quadratic regularization: it allows \(J_e=0\) to hold exactly; \(\alpha\) controls sparsity in a manageable way; the algorithm is suitable for low regularization and sparse graphs where \(|E|\ll|V|^2\).

**C3. Duality and Laplacian Structure (Sec. 3.3, Prop. 1–2, Eq. (5)–(9)).** Prop. 1: \(W_{1,\alpha}=\frac1\alpha\sup_p\big[\alpha f^\top p-\frac12|(Dp-c)_+|_2^2\big]\), and \(J_e=0\) on edges where \((Dp-c)_e\le0\). Definition 1: The active set \(S(p)=\{e:(Dp-c)_e>0\}\), and complementary slackness gives \(S(J^\alpha)=S(p^\alpha)=:S(\alpha)\). Eq. (8)–(9): \(\nabla g=\alpha f-D^\top M(p)(Dp-c)\), \(\mathrm{Hess}[g]=-D^\top M(p)D\); Prop. 2: The dual Hessian is the (negative of the) unweighted Laplacian of the active subgraph \((V,S(p))\), and its null space is spanned by indicator vectors of connected components.

**C4. Sparsity in the Small Regularization Limit (Sec. 4, Prop. 3–5, Corollary 1, Lemma 1, Conjecture 1).** Prop. 3: If \(0<\alpha<\alpha'\) and \(J^\alpha\ne J^{\alpha'}\) then \(c^\top J^\alpha<c^\top J^{\alpha'}\) and \(|J^\alpha|^2>|J^{\alpha'}|^2\). Prop. 4 (Sparsity): There exists \(\tilde\alpha>0\) depending only on \(G,f\) such that for all \(\alpha\in(0,\tilde\alpha)\), \(J^\alpha\) is also a solution to (LP). Corollary 1: Furthermore, there exists a unique \(J_0\) such that for all \(\alpha\in(0,\tilde\alpha)\), it is the unique solution to (QP). Two counterexamples: Figure 1 (\(J^\alpha\) can be sparser than some LP solutions), Figure 2 (active set does not vary monotonically with \(\alpha\)). Conjecture 1 proposes a monotonicity conjecture expressed using divergence-free flows \(R_i\).

**C5. Algorithm (Sec. 5, Eq. (20)–(28), Algorithm 1).** Dual ascent, alternating between gradient direction and pseudo-Newton direction \(L_k^+(\alpha f-D^\top M_kv_k)\); closed-form line search (parabolic minimum vs. "hit time" for active set flip); \(L_k^+=(L_k+N_kN_k^\top)^{-1}P_k\) handles the null space; adding/removing active edges corresponds to rank-one updates of the Laplacian, maintained using CHOLMOD for sparse Cholesky factors.

**C6. Experiments (Sec. 6, Figure 6–8).** Comparison on random graphs with 50–5000 nodes against gradient ascent, full-graph Laplacian preconditioning, and simplex method; comparison on \(\mathbb R^2\) grids against Li–Osher–Gangbo's Fast \(L_1\).

#### 3. Key Points of Method and Theoretical Derivation

##### 3.1 Kantorovich ⇔ Min-Cost Flow: Why Eq. (1) and Eq. (3) are Equal (Original Statement + Editorial Elaboration)

The original text only states that "the transportation cost between two points can be decomposed into edge-wise costs along the shortest path." To complete this, two directions are needed:

- Coupling → Flow: Given a feasible \(T\), push each unit of mass \(T_{vw}\) along a shortest path from \(v\) to \(w\), yielding an edge flow \(J=\sum_{v,w}T_{vw}\,\delta(r_{vw})\) (\(\delta(r)\) is the edge indicator vector for path \(r\), notation from Sec. 4.3.1). Summing edge-wise gives \(D^\top J=\rho_1-\rho_0\), and cost \(c^\top J=\sum T_{vw}C_{vw}\). Thus, \(\min\text{(3)}\le\min\text{(1)}\).
- Flow → Coupling: Given a feasible \(J\), Theorem 1 (flow decomposition, citing Ahuja–Magnanti–Orlin Theorem 3.5) decomposes it into path flows from sources to sinks plus cycle flows \(\hat J:\mathcal P\cup\mathcal C\to\mathbb R_+\), where each positive flow path starts from a source with \(f_s<0\) and ends at a sink with \(f_t>0\). Discarding cycles only decreases cost (original text in Sec. 4.3.1); let \(T_{vw}=\sum_{r:s(r)=v,t(r)=w}\hat J(r)\). Then \(T\) is a feasible coupling, and the cost of each path is \(\ge C_{vw}\), so \(c^\top J\ge\sum T_{vw}C_{vw}\), and \(\min\text{(3)}\ge\min\text{(1)}\).

Combining these two directions yields the statement needed by O08: **The Kantorovich problem on graphs with shortest path costs is equivalent to the min-cost flow problem on edge flows; every path decomposition of an optimal edge flow yields an optimal coupling, and conversely, pushing every optimal coupling along shortest paths yields an optimal edge flow.** The decomposition is not unique (Sec. 4.3.1 "a corresponding path flow"), which is precisely the analogue in GFlowNets where "the same edge flow corresponds to multiple trajectory distributions."

##### 3.2 Duality (Proof of Prop. 1, Eq. (6))

Lagrangianizing \(D^\top J=f\), and swapping max and min (convex quadratic program + affine constraints, strong duality guaranteed by affine Slater condition):
\[
W_{1,\alpha}=\max_p\Big[f^\top p+\min_{J\ge0}\big(J^\top(c-Dp)+\tfrac\alpha2J^\top J\big)\Big].
\]
The inner problem is independent for each edge: \(J_e=\max\{(Dp-c)_e,0\}/\alpha\), i.e., \(J=(Dp-c)_+/\alpha\). Substituting back yields Eq. (5). Editorial note: \(p\) is the Kantorovich potential on the graph, \((Dp)_e=p_w-p_v\) is the potential difference; the unregularized limit requires \((Dp-c)_+=0\), i.e., \(p_w-p_v\le c_e\) for every edge—this is the form of the 1-Lipschitz constraint on graphs, a discrete version of the Kantorovich–Rubinstein duality. Regularization replaces "tight edges" \(p_w-p_v=c_e\) with "active edges" \(p_w-p_v>c_e\), where flow is proportional to the excess. Eq. (7) uses \(M(p)=\mathrm{diag}(\mathbb I\{e\in S(p)\})\) to write the objective as \(g(p)=\alpha f^\top p-\frac12(Dp-c)^\top M(p)(Dp-c)\). Within regions where \(M\) is constant, it is quadratic, and its Hessian \(-D^\top MD\) is the Laplacian of the active subgraph (Prop. 2).

##### 3.3 Small Regularization Limit Selects a Unique LP Solution (Prop. 3, 4, Corollary 1)

- The proof of Prop. 3 involves subtracting two variational inequalities: from the minimality of \(J^\alpha\), we get Eq. (11) \(c^\top(J^{\alpha'}-J^\alpha)+\alpha(J^\alpha)^\top(J^{\alpha'}-J^\alpha)<0\) (strict, because strictly convex and \(J^\alpha\ne J^{\alpha'}\)); from the minimality of \(J^{\alpha'}\), we get the reverse \(\ge0\). Subtracting them yields Eq. (12) \((J^{\alpha'})^\top(J^{\alpha'}-J^\alpha)>0\), and substituting back gives the two conclusions.
- The proof of Prop. 4 proceeds in three steps (Sec. 4.4): (i) Using Lemma 1, any LP solution can be written as \(\hat J_0=\hat J^\alpha+\sum_k\epsilon_kR_k\), where \(R_k\) is a divergence-free perturbation in path space (\(D^\top R_k=0\)), formed by "more paths in set \(X_-^k\)" minus "fewer paths in set \(X_+^k\)", with each source/sink connected exactly once; (ii) Proving that \(\check c^\top\hat R_k\le0\) for all \(k\) (otherwise \(J_0\) could be improved), and at least one is strictly \(<0\) (otherwise \(J^\alpha\) is already an LP solution); (iii) Perturbing by \(\epsilon\) along this \(R_k\), Eq. (17) gives the objective change \(\epsilon[\check c^\top\hat R_k+\alpha(\hat J^\alpha)^\top S\hat R_k]+\frac{\epsilon^2}{2}\alpha\hat R_k^\top S\hat R_k\). The uniform boundedness from Prop. 5 (\(\hat J(r)\le-f_{s(r)}\), \(J_e\le-\sum_{f_v<0}f_v\)) and path finiteness provide constants \(K_1,K_3\). Eq. (18) defines \(K_{\min}<0\) as the value closest to zero among all "negative cost path combinations". Choosing \(\tilde\alpha=|K_{\min}|/(2K_1)\) leads to a contradiction.
- Corollary 1: If \(0<\alpha<\alpha'<\tilde\alpha\) yield different LP solutions, their \(c^\top J\) values are equal. Prop. 3 requires \(|J^\alpha|^2>|J^{\alpha'}|^2\), but \(J^\alpha\) being the unique minimizer of \(V_\alpha\) requires the opposite, leading to a contradiction.
- Editorial note (a direct consequence of Corollary 1, not explicitly stated in the original text): For \(\alpha<\tilde\alpha\), \(J_0\) is the unique minimizer of \(V_\alpha\) over all feasible flows, and specifically, it is the minimizer over the set of LP solutions. Since \(c^\top J\) is constant on the set of LP solutions, \(J_0=\arg\min\{|J|^2:J\text{ solves (LP)}\}\). Quadratic regularization is a clear tie-breaking rule: among all minimum cost flows, pick the one with the smallest \(\ell^2\) norm.
- Two counterexamples define the boundaries: In Figure 1 (\(c\equiv1\), node values \(-1,-99,10,90\)), \(J^\alpha\) uses only one sparse edge, while the two LP solutions are denser; in Figure 2 (one long path with low total \(L^1\) cost but composed of many consecutive edges vs. a few edges with high \(L^1\) cost), for small \(\alpha\), the long path is taken, but for large \(\alpha\), due to the accumulated \(L^2\) cost of many consecutive edges, the high-cost edges are taken instead, so the active set is not monotonic. Figure 4's 5-node, 7-edge example shows that for \(\alpha=10\), \(R_1\) is activated with coefficient \(\epsilon_1=0.07\), and for \(\alpha=10^3\), \(\epsilon_1=0.25\) and \(R_2\) is activated with \(\epsilon_2=0.25\).

##### 3.4 Algorithm Loop (Sec. 5, Algorithm 1)

1. Randomly initialize \(p\); \(v=Dp-c\), \(M=\mathrm{diag}(v>0)\), \(L=D^\top MD\); use flood fill to obtain a null space basis \(N\) (one column per connected component); perform sparse QR on \(W_0=[MD;\,N^\top]\) to get \(L+NN^\top=R^\top R\) (Eq. (28)).
2. Direction Eq. (25): Odd steps use gradient \(s=\alpha f-D^\top Mv\); even steps use pseudo-Newton \(s=R^{-1}R^{-\top}(s-NN^\top s)\) (Eq. (27)); translate \(s\) so its sum is zero.
3. Line search: Parabolic minimum Eq. (21) \(t_{\mathrm{quad}}=(\alpha f^\top s-v^\top MDs)/(s^\top Ls)\); hit time Eq. (22) \(h=-v\oslash(Ds)\), Eq. (23) \(t_{\mathrm{active}}=\min\{h_e>0\}\); Eq. (24) \(t=\min(t_{\mathrm{quad}},t_{\mathrm{active}})\).
4. \(p\leftarrow p+ts\); for each newly activated/deactivated edge, perform a rank-one update of \(\pm d_ed_e^\top\); if connected components merge/split, perform at most four rank-one updates on the columns of \(N\) (Figure 5), updating first and then reducing updates to maintain full rank.

Each gradient step is \(O(|V|)\), and the pseudo-Newton step avoids \(O(|V|^3)\) by using factor updates. The original text states that \(t_{\mathrm{active}}=0\) was never encountered before convergence, but no proof is provided.

##### 3.5 Quadratic Regularization vs. Entropy Regularization (Comparison of Sec. 2 and Sec. 3.2, compiled by editor)

| Dimension | Entropy Regularization (Cuturi 2013; Benamou et al. 2015) | Quadratic Regularization (Eq. (4) in this paper) |
|---|---|---|
| Variable regularized | Coupling matrix \(T\): \(-\sum_{vw}T_{vw}\ln T_{vw}\) | Edge flow \(J\): \(\frac\alpha2\sum_eJ_e^2\) |
| Number of variables | \(|V|^2\) | \(|E|\), much smaller than \(|V|^2\) on sparse graphs |
| Support of solution | \(T_{vw}>0\) strictly holds for any regularization strength, no sparsity | Allows \(J_e=0\) to hold exactly; solution is LP solution when \(\alpha<\tilde\alpha\) (Prop. 4) |
| Small regularization limit | Alternating projections converge slowly, near-zero values numerically unstable (original text cites Schmitzer 2016 for improvements) | Solution does not change after \(\alpha<\tilde\alpha\) (Corollary 1), algorithm suitable for low regularization |
| Large regularization limit | Independent coupling \(\rho_0\otimes\rho_1\) | Electrical flow-like, Sec. 2 points out connections to Christiano et al. 2011, Mądry 2013 |
| Algorithm | Sinkhorn–Knopp iterative scaling; no Sinkhorn-type method for general graphs (Sec. 2) | Dual Newton-type, Hessian is active subgraph Laplacian (Prop. 2) |
| Source of uniqueness | Strictly convex entropy | Strictly convex \(\ell^2\) term; as \(\alpha\to0\), tie-breaking selects the \(\ell^2\)-minimal LP solution (3.3 editor's deduction) |
| Counterpart in GFlowNets | KL-type loss on trajectory distribution (TB), Schrödinger bridge (C02) | On-policy state flow regularization \(\sum_sF(s)^2\) in T36 Appendix B.1 |

The original text also mentions (Sec. 2): for bipartite graphs, Benamou et al.'s alternating projection framework can replace entropy with a quadratic term to yield a "less efficient Sinkhorn-type" algorithm, where each step requires sorting floating-point numbers (Duchi et al. 2008's \(\ell^1\) ball projection).

#### 5. Prerequisites and Applicability Boundaries

- **Graph**: Connected, directed, no capacity constraints, edge costs \(c\ge0\). The cost matrix \(C\) must be the shortest path distance on the graph—this is the sole condition for Eq. (1) ⇔ Eq. (3) to hold; for general \(C\), the Beckmann form is no longer equivalent to Kantorovich.
- **Data**: \(\sum_vf_v=0\); feasibility is guaranteed by connectivity (experiments use bidirectional graphs to ensure this).
- **Regularization**: When \(\alpha>0\), (QP) is strictly convex and the solution is unique; \(\Gamma\)-convergence as \(\alpha\to0\) yields a specific solution of the LP. Prop. 4's \(\tilde\alpha=|K_{\min}|/(2K_1)\) depends on \(G\) and \(f\), and is non-constructive (\(K_{\min}\) requires enumerating path combinations).
- **Objects of Prop. 3–5**: Path decomposition of non-negative flows, cycle flows are zero in optimal solutions (Sec. 4.3.1); Lemma 1 requires both flows to be decomposable into source-sink path flows.
- **Algorithm**: The objective is piecewise quadratic, and the closed-form line search holds under the premise that the active set does not change within a step (guaranteed by Eq. (24)); \(t_{\mathrm{active}}>0\) is an observation, not a theorem; the dimension of the Laplacian null space equals the number of connected components of the active subgraph, which needs explicit handling.
- **Not covered**: Capacity constraints, nonlinear costs, continuous Beckmann (listed as future work in Sec. 7), graph version of entropy regularization (Sec. 2 states no Sinkhorn-type method for general graphs).
- **Positive statement of applicability**: For any finite directed connected graph, for \(W_1\) (EMD) with shortest path costs, when a unique and sparse solution is desired, regularization is very small, and the graph is sparse, the formulation and algorithm presented in this paper are applicable.

#### 6. Position in the GFlowNet × OT Main Line

- **Predecessors**: O01 (Kantorovich form, \(W_1\), textbook background of duality); outside the repository: Beckmann 1952 (continuous flow form), Santambrogio 2015 (Beckmann problem), Ahuja–Magnanti–Orlin 1993 (flow decomposition Theorem 3.5, min-cost flow algorithms), Cuturi 2013 / Benamou et al. 2015 (entropy regularization and Sinkhorn, used for comparison), Li–Osher–Gangbo 2016 (quadratic regularized \(L^1\) algorithm on grids, used for comparison), Yin 2010 (more general results for \(L^1+L^2\) sparsity, original text acknowledges Prop. 4 can be derived from it).
- **Successors / Comparisons**: O08 directly depends on Sec. 3.1; O07's "minimum total flow = shortest path" is a special case of Eq. (3) under single source and \(c\equiv1\); T36 Appendix B.1's "on-policy regularization actually minimizes \(\sum F^2\)" is precisely the regularization term of Eq. (4); C01/C02 (graph-based (unbalanced) OT and Schrödinger bridge) follow the entropy regularization route, which is the side criticized in Sec. 3.2 of this paper; O03/O06 are OT solvers in continuous domains, parallel to the graph discrete setting here.
- **What mathematical piece it provides for O08**: The proposition in O08 is "the internal flow learned by GFlowNet is a Kantorovich optimal transport plan." For this statement to hold, three things are needed, all from Sec. 3.1 and Sec. 4.3.1 of this paper:
  - (a) **Kantorovich ⇔ min-cost flow on graphs with shortest-path cost** (Eq. (1) ⇔ Eq. (3)). GFlowNet's edge flow \(F\) satisfies \(D^\top F=\) (terminal marginal) \(-\) (source marginal), and the total flow \(\sum_eF(e)\) is the Beckmann cost for \(c\equiv1\); T36 Prop. 3.12 states that total flow = \(Z\cdot\mathbb E[n_\tau]\). Thus, "minimum expected length" = min-cost flow of Eq. (3) = \(W_1\) of Eq. (1), with ground cost being graph shortest path (which is a true metric on non-acyclic graphs allowing bidirectional movement).
  - (b) **Flow decomposition theorem** (Theorem 1) translates edge flow into path flow. GFlowNet's trajectory distribution itself is a path decomposition, where \(T_{vw}=\) mass of trajectories starting at \(v\) and ending at \(w\); the non-uniqueness of decomposition explains why O08 can state "different \(P_F\) for the same edge flow yield the same coupling."
  - (c) **Optimal flow is acyclic** (Sec. 4.3.1 "removing a cycle can only decrease the total cost"). This is the combinatorial root of T19 Theorem 1 "regularization limit is acyclic" and T36 Eq. (11) where the minimum point lies on the boundary.
- **What this paper does not do but the main line needs**: GFlowNet semantics beyond single-source single-sink (this paper is naturally multi-source, thus closer to O08's setting than T36); stochastic policies and expected visit counts (this paper has no probabilities); relationship between dual variable \(p\) and GFlowNet's \(\log F\) (see 7.3).

#### 7. Reusable Insights and Open Problems

1.  **Theorem Draft for O08's Core Identity**. Let \(G\) satisfy T36 Assumption 3.1, with source marginal \(L\) on \(\mathrm{out}(s_0)\) and terminal marginal \(R\) on \(\mathcal X\), such that \(\sum L=\sum R=Z\). Let \(\mathcal F_{L,R}\) be the set of non-negative, conservative internal edge flows with fixed marginals. Then \(\min_{\mathcal F_{L,R}}\sum_eF(e)=Z\cdot W_1(L/Z,R/Z;d_G)\). The minimizer is acyclic, and any path decomposition of it yields an optimal coupling for \(W_1\). Proof: Eq. (1) ⇔ Eq. (3) with \(c\equiv1\), \(f=R-L\), plus Theorem 1. This is the precise reference O08 should cite.
2.  **Quadratic Regularization = T36's On-Policy Regularization, and it has a Canonical Limit**. T36 Appendix B.1 states that on-policy training minimizes \(\sum_sF(s)^2\); according to the editor's inference in 3.3, when \(\alpha<\tilde\alpha\), the solution is "the minimum \(\ell^2\) norm among all minimum cost flows." Experimental draft: On a hypergrid with multiple shortest paths of equal length, check if the learned flow for small \(\lambda\) distributes mass evenly across all shortest paths (minimum \(\ell^2\) ⇒ as uniform as possible), and compare edge-by-edge with the (QP) solution from O02; if they match, GFlowNet's "diversity" in OT language is the tie-breaking mechanism of quadratic regularization.
3.  **Dual Potential \(p\) and \(\log F\)**. Prop. 1's \(p\) is the Kantorovich potential, with \(J_e=(p_w-p_v-c_e)/\alpha\) on active edges; T36 Theorem 3.13's \(V^\star(s)=\log F(s)\) is the soft Bellman potential. Theorem draft: Push T36's per-step negative reward \(-c\) (see T36 report 7.3) to \(c\to\infty\), then \(\log F(s)/c\) converges to the shortest path potential on the graph, which is the dual solution \(p\) for \(\alpha\to0\); this would connect "GFlowNet's state flow" with "OT dual variables."
4.  **Meaning of Non-Monotonic Active Set (Figure 2) for \(\lambda\) Scan**. T36 Figure 5 scans \(\lambda\) only for length and error; within the framework of Conjecture 1, the set of edges used by GFlowNet for different \(\lambda\) might switch non-nestedly. Experiment: Record the set of edges where \(F_\theta>\)threshold for each \(\lambda\) in the T36 \(20^4\) experiment, and check if Figure 2-type route switching occurs.
5.  **Correspondence between Entropy Regularization vs. Quadratic Regularization in GFlowNets**. Entropy regularization is written on the coupling \(T\) (Sinkhorn, Schrödinger bridge in C01/C02), while quadratic regularization is written on the edge flow \(J\) (this paper). The TB loss is a KL-divergence-like divergence on trajectory distributions (T19 Eq. (13)), which is more like the former; state flow regularization is more like the latter. Open question: Which type of regularized OT solution does O08's "learned OT plan" correspond to for finite \(\lambda\)?
6.  **Can Newton Structure be Used for O08's Solver Baseline?** Prop. 2's Laplacian Hessian + rank-one update is a fast method for exactly solving OT on small graphs; using it to compute the exact solution of (QP) on T36's \(7\times7\) and \(S_4\) can serve as ground truth for the flow learned by GFlowNet.

## 6.3 O03 · GeONet: Learning Neural Operators for Wasserstein Geodesics (UAI 2024)

> **One-liner**: GeONet reformulates "given a pair of marginal distributions, find the Wasserstein geodesic" from "re-solve OT for each new pair" to a one-time trained operator learning: it uses two sets of DeepONets to fit the primal-dual KKT equations of the Benamou–Brenier problem (continuity equation + Hamilton–Jacobi), training only requires boundary distribution pairs and no geodesic ground truth, and inference is a single forward pass. On the GFlowNet × OT map, it stands on the "amortization" side: it proves that the optimality conditions for dynamic OT can be captured by an operator across a family of distributions, which is precisely the capability desired by "conditional GFlowNet–OT"—except that GeONet's implementation mechanism (PDE residuals) has no direct counterpart on discrete graphs.

| Field | Content |
|---|---|
| arXiv | [2209.14440](https://arxiv.org/abs/2209.14440) (v4, 2024-05-23) |
| Publication | **UAI 2024 Main Conference** (Original homepage footnote: Accepted for the 40th Conference on Uncertainty in Artificial Intelligence) |
| Authors | Andrew Gracyk (UIUC Statistics), Xiaohui Chen (USC Mathematics) |
| Code | https://github.com/agracyk2/GeONet (Given in original Sec. 4) |
| This Repo PDF | `papers/2209.14440.pdf` · Chinese translation Not generated |
| Reading Priority | **P2** — The method itself shares no structure with GFlowNet, but it is the earliest work in the "amortized OT solver" branch to output the *entire dynamic trajectory* rather than just a static map. It must be read as a control. |

#### 2. Core Contributions (Numbered as in Original Text)

**(C1) Rewriting geodesic solving as an operator learning problem (Sec. 3, Eq. (12)–(13)).** Conclusion: After one training, for new \((\mu_0,\mu_1)\), a single forward pass yields the entire path \(\{\mu_t\}\). Prerequisite: The training distribution family is isomorphic to the test distribution family (original OOD experiments show out-of-distribution error increases by 2–3 times, see Sec. 4).

**(C2) Training does not require ground truth geodesics (Sec. 1 "Surprisingly..." paragraph + Eq. (19)).** Conclusion: The only supervisory signals are boundary pairs \((\mu_0^{(i)},\mu_1^{(i)})\); intermediate times rely on PDE residuals to "weld" the KKT conditions. This is its core advantage over purely data-driven neural operators, and also why its error at \(t=0.25/0.5/0.75\) is significantly larger than at \(t=0,1\) (acknowledged by the original paper in Appendix H.1).

**(C3) Joint training of primal and dual networks, using zero duality gap as a correctness anchor (Sec. 2.1 + Appendix B, Eq. (32)).** Conclusion: As long as \((\mu^\ast,u^\ast)\) simultaneously satisfy CE and HJ and boundary conditions, the duality gap is zero, and the solution is a geodesic. This is the linchpin of GeONet's entire design.

**(C4) Output is grid-independent, supporting zero-shot super-resolution (Sec. 1, Fig. 1; "high-res." row in Table 2).** Conclusion: Training on \(24\times24\) and evaluating on \(75\times75\) yields errors roughly on par with random tests at the same resolution (2D high-res. \(6.29\%\text{–}7.88\%\) vs 2D random \(6.33\%\text{–}7.13\%\), Table 2).

**(C5) Entropy-regularized version ER-GeONet (Appendix D, Eq. (56)–(58)).** Conclusion: By changing the constraint to \(\partial_t\mu+\mathrm{div}(\mu v)+\varepsilon\Delta\mu=0\), KKT becomes \(\partial_t\mu+\mathrm{div}(\mu\nabla u)=-\varepsilon\Delta\mu\) and \(\partial_t u+\tfrac12\|\nabla u\|_2^2=\varepsilon\Delta u\). The latter is a parabolic equation with a unique smooth solution, leading to more stable training; as \(\varepsilon\downarrow0\), it converges to the Benamou–Brenier solution via the vanishing viscosity method (citing Mikami 2004, Evans 2010). **Note that this is in the Appendix, and the main text experiments do not report numbers for ER-GeONet.**

**(C6) Gradient enhancement (Appendix E, Eq. (61)–(62)).** This involves taking another derivative of the PDE with respect to spatial coordinates and adding the new residual to the loss. The original paper claims improved sample efficiency and accuracy but does not provide comparative numbers before and after enhancement.

#### 6. Position in the GFlowNet × OT Main Thread

- **Predecessors**: O01 (OT for Machine Learners, standard reference for Benamou–Brenier and Kantorovich duality); numerical reference implementation relies on POT (Flamary et al. 2021). Methodological predecessors are DeepONet (Lu et al. 2021), physics-informed DeepONet (Wang et al. 2021b), and enhanced DeepONet (Tan & Chen 2022).
- **Successors / Counterparts**:
  - **O05 (UNOT, ICML 2025)** is its direct competitor and provides comparisons unfavorable to GeONet (O05 Sec. 4.3, Fig. 8). The objects amortized differ: GeONet amortizes "the entire geodesic," while UNOT amortizes "entropy OT dual potentials," from which plans, barycenters, and geodesics are then derived.
  - **O06 (Q-flow, AISTATS 2025)** solves the same Benamou–Brenier problem but uses neural ODEs + bidirectional KL relaxation, solving per-pair without amortization.
  - **O04 (α-DSBM, NeurIPS 2024)** solves the entropy-regularized dynamic version (Schrödinger bridge), also per-pair. GeONet's ER-GeONet (Appendix D) and O04 overlap in objective, but GeONet uses PDE residuals, while O04 uses iterative projection + bridge matching.
- **Contribution to the main thread ("internal flow selection = optimal transport")**: GeONet provides a **continuous-side dual template** for this main thread. Theorem 3.2 of O08 states that a min-flow non-acyclic GFlowNet, after fixing the initial edge flow distribution, is equivalent to a Kantorovich problem with graph shortest path as cost (also equivalent to discrete Beckmann); Theorem 3.3 of O08 gives its dual LP and shows that the optimal flow can only be supported on the shortest path subgraph. Comparing these two with GeONet's KKT conditions reveals a one-to-one structural correspondence:
  - Continuity equation \(\partial_t\mu+\mathrm{div}(\mu\nabla u)=0\) ↔ flow conservation on graph (O08 Eq. (4));
  - HJ equation \(\partial_t u+\tfrac12\|\nabla u\|^2=0\) ↔ graph shortest path/Bellman equation, whose solution \(d(u,x)\) is precisely the ground cost in O08;
  - Zero duality gap ↔ \(\mathrm{GFlow}^\star \leqslant \mathrm{OT}^\star\) and the reverse inequality squeeze in O08's proof.
  In other words: **the \(H_\psi\) learned by GeONet is the continuous version of the "log of state flow/potential function" in GFlowNet**. This correspondence is the only reliable interface for porting GeONet's amortization technique to GFlowNet—not by porting PDE residuals, but by porting the idea of "using dual quantities for ground-truth-free supervision."

#### 7. Reusable Insights and Open Problems

1.  **"Replacing ground truth with optimality conditions" can be directly transplanted as an auxiliary loss for GFlowNets.** The graph counterpart of GeONet's \(\mathcal{L}_{\mathrm{HJ}}\) is "the potential function satisfies \(\pi_x - \pi_u \leqslant d(u,x)\) and equality holds on supporting edges" (O08 Theorem 3.3's dual feasibility + complementary slackness). This can be formulated as an experiment: add a dual complementary slackness residual term to min-flow GFlowNet training and see if it accelerates convergence to OT⋆ without requiring longer trajectory sampling.
2.  **The correct amortized cross-section is "dual potentials" rather than "trajectories."** The cost of GeONet outputting the entire \(\{\mu_t\}\) is that the error at intermediate times is significantly larger than at the two ends (each row in Table 2 shows high in the middle, low at the ends); UNOT only amortizes potentials, and the rest is compensated by one-step Sinkhorn, which is more accurate. Implications for conditional GFlowNet–OT: the conditional network should predict "potential-like" quantities such as \(\log F\) or \(P_B\), rather than directly predicting the entire trajectory distribution.
3.  **Grid-independent ≠ input-independent**, this pitfall is worth re-examining on graphs. GeONet's trunk is independent while its branch is dependent; if GFlowNet is to generalize across graph scales, the encoder must be invariant to the number of nodes (e.g., graph neural networks + positional encoding), otherwise it will repeat GeONet's "input side fixed" limitation.
4.  **Lack of supervision at intermediate times is a structural weakness of physics-informed amortization.** A draft theorem: under a training objective with only boundary constraints + PDE residuals, the lower bound of the error at intermediate times is related to the collocation sampling density and the condition number of the PDE residual operator; corresponding to GFlowNet, this is "the credit assignment error when only using TB loss (endpoints) and not DB/SubTB (intermediate)" (can be compared with the conclusions of T03/T05).
5.  **Entropy regularization is a shared stabilization knob.** ER-GeONet's \(\varepsilon\Delta u\) makes HJ parabolic, and the solution is uniquely smooth (Appendix D); O04's \(\varepsilon\) is the Brownian motion variance; in GFlowNet, this corresponds to "policy entropy/temperature." Open question: explicitly adding an entropy term in min-flow GFlowNet, does it yield entropy OT on graphs (i.e., Sinkhorn plan), and does its \(\varepsilon\downarrow0\) limit recover shortest path support (O08 Theorem 3.3).
6.  **The magnitude of OOD degradation can serve as a measure of conditional capability.** GeONet's OOD error increases from \(\sim5\%\) to \(\sim20\%\) (1D, Table 2). When evaluating "whether conditional GFlowNet–OT truly has a cross-task generalization advantage," this protocol should be copied: training distribution parameter range → expand range during testing → report error increase, rather than just reporting in-distribution metrics.

## 6.4 O04 · Schrödinger Bridge Flow (NeurIPS 2024 Spotlight)

> **One sentence summary**: This paper replaces DSBM's "alternatingly solve two projection subproblems" with "single-step gradient descent along a path measure flow": define a flow on path measures \(\partial_s\hat{\mathbb{P}}_s=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}_s))-\hat{\mathbb{P}}_s\), whose unique fixed point is the Schrödinger bridge; discretize with step size \(\alpha\in(0,1]\) to obtain \(\alpha\)-IMF (\(\alpha=1\) degenerates exactly into the original IMF), and the parameterized implementation is called \(\alpha\)-DSBM, with one network, one set of losses, and online fine-tuning. For the GFlowNet–OT main line, it is the **only neighboring work that explicitly formulates "how to allocate mass among multiple construction paths" as two explicit projection operators and provides an operator-level convergence proof**: \(\mathrm{proj}_{\mathcal{R}}\) is "fixing \(P_B\)", and \(\mathrm{proj}_{\mathcal{M}}\) is "solving \(P_F\) from edge flows in reverse", and their alternation is structurally homologous to GFlowNet training.

| Field | Content |
|---|---|
| arXiv | [2409.09347](https://arxiv.org/abs/2409.09347) (v1, 2024-09-14) |
| Publication | **NeurIPS 2024 Main Conference (Spotlight)**; the PDF in this repository is v1 preprint, footer still says "Submitted to 38th Conference on NeurIPS 2024" |
| Authors | Valentin De Bortoli\*, Iryna Korshunova\* (equal contribution), Andriy Mnih, Arnaud Doucet (all Google DeepMind) |
| Code | **Not publicly available** (NeurIPS Checklist item 5: code repository cannot be made public due to IP restrictions, plans to release a small-scale reproduction notebook) |
| PDF in this repository | `papers/2409.09347.pdf` (47 MB, mostly figures) · Chinese translation Not generated |
| Reading Priority | **P1** — The two-projection operator framework it provides is the best template for translating "GFlowNet's \(P_F/P_B\) consistency training" into measure-theoretic language, and its failure mode (not simulation-free, must self-sample) directly corresponds to GFlowNet's on-policy sampling cost |

#### 2. Core Contributions (Numbered as in Original Text)

**(C1) Schrödinger Bridge Flow (Sec. 3.1, Eq. (3)).** Conclusion: Define

\[
\hat{\mathbb{P}}_0=(\pi_0\otimes\pi_1)\mathbb{Q}_{|0,1},\qquad \partial_s\hat{\mathbb{P}}_s=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}_s))-\hat{\mathbb{P}}_s,\qquad \mathbb{P}_s=\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}_s),
\]

then the **unique fixed point of this flow is the SB**. The argument is only two lines (Sec. 3.1): a fixed point \(\bar{\mathbb{P}}\) satisfies \(\bar{\mathbb{P}}=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\bar{\mathbb{P}}))\), and repeated substitution shows that \(\bar{\mathbb{P}}\) is the limit point of the IMF sequence, hence it is the SB (citing Peluchetti 2023, Thm. 2). Prerequisite: the flow itself is well-defined (assumed in the original text).

**(C2) \(\alpha\)-IMF and Convergence Theorem (Eq. (4) + Theorem 3.1).** \(\hat{\mathbb{P}}^{n+1}=(1-\alpha)\hat{\mathbb{P}}^{n}+\alpha\,\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}^{n}))\), \(\mathbb{P}^n=\mathrm{proj}_{\mathcal{M}}(\hat{\mathbb{P}}^n)\). For **any** \(\alpha\in(0,1]\), we have \(\lim_n\mathbb{P}^n=\mathbb{P}^\star\); \(\alpha=1\) is exactly IMF/DSBM. Note that \(\hat{\mathbb{P}}^n\in\mathcal{R}(\mathbb{Q})\) holds for all \(n\)—convex combinations preserve the reciprocal class, which is key for the entire construction to be closed.

**(C3) Non-parametric Update = \(\alpha\)-IMF (Prop. 3.2 / Prop. D.1).** Let \(\delta_n=\alpha\) and \(\mu_n=(1-\alpha)\hat{\mathbb{P}}^n+\alpha\,\mathrm{proj}_{\mathcal{R}}(\mathbb{P}^n)\). Then the path measure \(\mathbb{P}_{v^n}\) generated by the functional gradient descent \(v_t^{n+1}=v_t^{n}-\delta_n\nabla_{\mu_n}L_t(v_t^n,\mathbb{P}_{v^n})\) is exactly equal to \(\mathbb{P}^n\). **This is the most informative technical conclusion of the entire paper** (see Section 3 for details).

**(C4) Bidirectional Process and Single Network Parameterization (Sec. 4, Eq. (10)–(13), Prop. 4.1).** Unidirectional online updates accumulate errors (Appendix I provides exact recurrence for Gaussian cases, Prop. I.1/I.2); this can be remedied by training both forward and backward processes simultaneously (because Markov projection is consistent for forward and backward, citing Shi et al. 2023, Prop. 9). Instead of two networks for parameterization, a single network is used with a direction input \(s\in\{0,1\}\), where \(v_\theta(1,\cdot)\approx\overrightarrow{v}\) and \(v_\theta(0,\cdot)\approx\overleftarrow{v}\).

**(C5) Parametric Update is a Preconditioned Version of Non-parametric Update (Prop. D.4 / D.5).** \(\theta\leftarrow\theta-\alpha\nabla_\theta L(\theta,\mathbb{P}_{\bar\theta})\) (where \(\bar\theta\) is stop-gradient) is, in an average sense, the descent direction of the non-parametric loss as \(\alpha\to0\). Prerequisite: Bounded Hessian (constant \(C\) in Eq. (28)).

**(C6) Three Parallel Relationships with Sinkhorn Flow / EM / RL (Appendix H).** The difference between \(\gamma\)-Sinkhorn (Karimi et al. 2024) and \(\gamma\)-IMF boils down to one sentence: **the former is an implicit update, the latter is an explicit update** (Table 2). According to Brekelmans–Neklyudov's classification, IPF is Expectation–Expectation, IMF is Maximisation–Maximisation, and the algorithm in this paper is an incremental version of MM (Appendix H.3). Appendix H.2 provides a replay-buffer version (Algorithm 6), which degenerates to Algorithm 1 when \(n_{\mathrm{refresh}}=1,\ N=B\).

#### 5. Assumptions and Applicability Boundaries

1.  **The reference process must have a closed-form bridge.** The premise for \(\mathrm{proj}_{\mathcal{R}}\) being "free" is that \(\mathbb{Q}_{|0,1}\) is a Brownian bridge (Eq. (12)/(30)); for general costs, reciprocal projection needs to be learned (Sec. 5 citing Neklyudov et al., Liu et al. 2022).
2.  **\(\varepsilon>0\) is fixed and known**, and \(\pi_0,\pi_1\) have finite second moments and finite differential entropy (explicit conditions in Lemma D.2: \(\int\|x\|^2\mathrm{d}\pi_i<\infty\), \(H(\pi_i)<\infty\)).
3.  **Must be able to self-sample from the model.** The algorithm is not simulation-free: each fine-tuning step requires solving a forward/backward SDE (listed as a major limitation in the Discussion).
4.  **Bidirectional training is a necessary condition for stability**, not an optional optimization (quantitative evidence in Appendix I).
5.  **Convex combinations preserving the reciprocal class** is a structural prerequisite for the legitimacy of \(\alpha\)-IMF (Sec. 3.1); if \(\mathcal{R}(\mathbb{Q})\) is not a convex set, the entire construction fails.
6.  **The theoretical \(\alpha\) and the implemented \(\alpha\) are not the same.** Implementation uses Adam, with \(\alpha\) implicitly adaptive (end of Sec. 4); Prop. D.4/D.5 only guarantees the descent direction when \(\alpha\to0\).

#### 6. Position in the GFlowNet × OT Main Thread

- **Predecessors**: DSBM/IMF (Shi et al. 2023; Peluchetti 2023), bridge matching / stochastic interpolants (Peluchetti 2021; Albergo–Vanden-Eijnden 2023; Lipman et al. 2023), DSB/IPF (De Bortoli et al. 2021). On the OT side, predecessors include Léonard's survey on the Schrödinger problem and Cuturi's Sinkhorn. Within this repository: O01 (OT basics), O03 (also dynamic OT, but using PDE residuals).
- **Successors / Competitors**: C02 (Generalized Schrödinger Bridge on Graphs, preprint 2026) and C03 (Discrete Diffusion SB Matching for Graph Transformation, ICLR 2025) are attempts to port this projection framework to discrete/graph state spaces—**these two are the direct competitors of GFN–OT**. O04 is their methodological parent.
- **Formula-level comparison with the main thread ("internal flow selection = optimal transport")**. Align GFlowNet objects with O04 objects one by one:

  | GFlowNet (T02 / O08 notation) | O04 (SB notation) | Isomorphic? |
  |---|---|---|
  | Trajectory measure \(\mathbb{P}(\tau)=\prod_t P_F(s_{t+1}|s_t)\) | Markov path measure \(\mathbb{P}\in\mathcal{M}\), \(\mathrm{d}X_t=v_t\mathrm{d}t+\sqrt{\varepsilon}\mathrm{d}B_t\) | Isomorphic: both are "global path laws generated by local transition rules" |
  | Fixed reference \(P_B(s|s')\) → trajectory conditional law determined after given endpoint | \(\mathbb{P}=\mathbb{P}_{0,1}\mathbb{Q}_{|0,1}\), i.e., \(\mathrm{proj}_{\mathcal{R}}\) | **Isomorphic**: \(\mathrm{proj}_{\mathcal{R}}\) is "projecting \(P_B\) back to the reference \(P_B\)", both only modify the middle, not the ends |
  | Forward policy \(P_F(s'|s)=F(s\to s')/F(s)\) solved from edge flow | \(v_t^\star(x_t)=(\mathbb{E}[X_1|X_t=x_t]-x_t)/(1-t)\), i.e., \(\mathrm{proj}_{\mathcal{M}}\) | **Isomorphic**: both are "finding a Markov policy with the same marginals as a given (non-Markov) trajectory measure", and both preserve marginals (\(\mathbb{M}_t=\mathbb{P}_t\)) |
  | Training objective: \(P_F\) and \(P_B\) induce the same trajectory distribution (O08 Eq. (1)) | Fixed-point condition: \(\bar{\mathbb{P}}=\mathrm{proj}_{\mathcal{R}}(\mathrm{proj}_{\mathcal{M}}(\bar{\mathbb{P}}))\) (Sec. 3.1) | Isomorphic: both are "common fixed points of two operators" |
  | Cost = expected trajectory length \(\mathbb{E}[n_\tau]\) (O07/O08) | Cost = \(\int\tfrac12\|x-y\|^2\mathrm{d}\Pi-\varepsilon H(\Pi)\) (Eq. (1)) | **Not isomorphic**: graph cost comes from "number of steps", SB cost comes from "squared displacement + entropy"; \(1-t\) denominator has no graph counterpart |
  | Trajectory length \(n_\tau\) unbounded (not acyclic) | Time interval fixed \([0,1]\) | **Not isomorphic**, and this is the second barrier for transplantation |
  | Entropy/temperature knob: policy entropy, reward exponent | \(\varepsilon\) (Brownian motion variance) | Structurally homologous, dimensionally different |

- **Similarities and differences in "multiple construction path assignments", in terms of formulas**:
  - **Similarities**: Both sides are solving "how to distribute mass among multiple paths between the same pair of endpoints". GFlowNet's answer is \(\mathbb{P}(\tau)=\prod P_F=\prod P_B\) (O08 Eq. (1)) combined with reward matching \(P_B(x|s_f)=R(x)/Z\) (O08 Eq. (2)); O04's answer is \(\mathbb{P}=\mathbb{P}_{0,1}\mathbb{Q}_{|0,1}\) combined with marginal constraints \(\mathbb{P}_0=\pi_0,\mathbb{P}_1=\pi_1\). **Both are instances of the pattern "endpoint constraints + path law after given endpoints fixed by a reference measure."**
  - **Differences (First, fundamental)**: GFlowNet's path assignment is **combinatorial and discrete**; "multiple paths" refers to different edge sequences on a graph. The analogue of \(\mathrm{proj}_{\mathcal{R}}\) is "resampling intermediate states by random walk according to a reference after given \((s_0\to u,\ x\to s_f)\)", which **has no closed form** on graphs (the closed-form interpolation of Brownian bridges in Eq. (12) is a gift of Euclidean structure). The reason C03 introduces discrete diffusion is to create a samplable discrete bridge.
  - **Differences (Second, exploitable)**: O04's \(\alpha\)-interpolation occurs in the **coupling space** (Eq. (22) collapsed form: new drift = conditional expectation of endpoint under \((1-\alpha)\hat{\mathbb{P}}^n+\alpha\,\mathrm{proj}_{\mathcal{R}}(\mathbb{P}^n)\)). The corresponding operation in GFlowNet training is "off-policy updates using a mixed distribution of old policy and new target", i.e., replay buffer—O04 itself has already written out this relationship (Appendix H.2, Algorithm 6). **This means GFlowNet's replay buffer is not just an engineering trick; in SB language, it is a discretization of a path measure flow.**
  - **Differences (Third, directionality)**: GFlowNet's \(P_B\) is usually **learnable** (or chosen according to T02's degrees of freedom), while O04's \(\mathbb{Q}_{|0,1}\) is **fixed** (precisely the definition of "reciprocal class of \(\mathbb{Q}\)"). Therefore, O04's solution is unique (Lemma D.2), while GFlowNet's solution is not unique when \(P_B\) is free—this non-uniqueness is precisely what O08 resolves by using "minimum total flow", leading to shortest path support (O08 Theorem 3.3). **In other words: O04 selects flows using "fixed reference bridge + entropy", while O08 selects flows using "minimum total flow"; these are two regularizations of the same problem.**

#### 7. Reusable Insights and Open Problems

1.  **Directly translate \(\alpha\)-IMF into "graph-based \(\alpha\)-IMF GFlowNet".** Experimental draft: On O08's hypergrid environment, fix the reference \(P_B\) as a uniform distribution over parent nodes (acting as \(\mathbb{Q}_{|0,1}\)). Alternately perform (i) sampling trajectories with the current \(P_F\), keeping only endpoint pairs \((u,x)\), and resampling intermediate paths according to the reference \(P_B\) (discrete \(\mathrm{proj}_{\mathcal{R}}\)); (ii) taking one step of TB/DB gradient on the resampled trajectories (discrete \(\mathrm{proj}_{\mathcal{M}}\)) with step size \(\alpha\). Metric: Gap to \(\mathrm{OT}^\star\) of scipy LP solution vs. sampling budget. Hypothesis: \(\alpha<1\) is more economical on large graphs because (ii) does not require training to convergence.
2.  **A draft theorem: Convexity and uniqueness of fixed points for reciprocal classes on graphs.** The legitimacy of O04 relies on two things: \(\mathcal{R}(\mathbb{Q})\) is convex, and "Markov + reciprocal + correct marginals \(\implies\) SB" (Lemma D.2). On finite graphs, the class defined by "conditional law of reference random walk given endpoints" is also convex (convex combination only changes \(\Pi_{0,1}\)); what's needed is a discrete version of Léonard Thm. 2.12. If true, then the unique solution to "minimum total flow GFlowNet + entropy regularization" is the Schrödinger bridge on graphs, and as \(\varepsilon\downarrow0\), it should converge to the shortest path support solution of O08 Theorem 3.3. **This is the most worthwhile theoretical problem to tackle first in this repository.**
3.  **The GFlowNet counterpart of bidirectional training needs explicit verification.** O04 uses Prop. I.1/I.2 to prove that unidirectional training accumulates errors, while bidirectional training does not. "Bidirectional" naturally exists in GFlowNets (\(P_F\) and \(P_B\) are learned simultaneously), but many implementations fix \(P_B\). An experiment could be conducted: fixed \(P_B\) vs. joint learning of \(P_B\), in long trajectory environments (e.g., O07's Rubik's Cube), to see if the bias in \(\mathbb{E}[n_\tau]\) accumulates with iterations.
4.  **"Not simulation-free" is a shared cost bottleneck, deserving shared solutions.** O04 requires solving an SDE at each step, and GFlowNet requires rolling out trajectories at each step. The two mitigation strategies proposed by O04 can be directly adopted: replay buffer (Appendix H.2) and model stitching (Appendix G: initializing with two existing generative models, setting \(\sigma_{t'}^2=\varepsilon/2\), equivalent to the online version of DSBM-IPF). The GFlowNet translations are "experience replay" and "initializing by stitching two pre-trained samplers," respectively.
5.  **The resolution scaling rule for \(\varepsilon\) is a transferable engineering conclusion.** \(\varepsilon_{256}=\varepsilon_{64}\cdot(256/64)^2\) (Appendix K.4). The analogous problem on graphs: "When the state space size increases from \(n\) to \(n'\), how should the entropy regularization coefficient be scaled to maintain the same exploration?"—This is a scaling law experiment that can be directly performed.
6.  **Evaluation protocols must guard against being misled by noisy metrics.** O04 explicitly points out that FID is unreliable on test sets with <500 images, and 2-Wasserstein still has high variance with 10K samples (Fig. 11). If GFN–OT evaluation uses "difference in transport cost from exact OT solution," it must report multiple seeds and confidence intervals, and use LP as an anchor on solvable small graphs (as O08 already does).
7.  **An untapped technical point: Radon–Nikodym weights as "state-dependent step sizes" (Eq. (23)).** This suggests that theoretically correct updates should have larger effective step sizes for infrequently visited states. An isomorphic problem exists in GFlowNets (credit assignment for rare states), and an explicit importance weighting term could be designed based on this, and compared with the variance analysis in T05 (SubTB)/T10.
8.  **The distinction between implicit vs. explicit updates is worth porting to the GFlowNet side for ablation.** Appendix H.1's Table 2 positions the difference between \(\gamma\)-Sinkhorn and \(\gamma\)-IMF as "implicit update vs. explicit update" (\(v^{n+1}_t=v^n_t-\delta\nabla_{\mu_n}L_t(v^{n+1},\bar{\mathbb{P}}^n)\) vs. \(v^{n+1}_t=v^n_t-\delta\nabla_{\mu_n}L_t(v^{n},\mathbb{P}^n)\), Eq. (51) vs. (52)). GFlowNet's target network / stop-gradient choices are precisely choices along the same axis, but to the best of this report's knowledge, no one has analyzed it as "implicit–explicit discretization."
9.  **A negative conclusion to remember: don't waste computational power trying to reproduce "online is always better."** On AFHQ-64, online fine-tuning FID is worse than iterative fine-tuning (28.75 vs. 25.41, Table 1), and on 2D toy problems, \(\alpha\)-DSBM is worse than two baselines on 3/4 tasks (Table 4). If GFN–OT also performs an "online vs. staged" ablation, the benefits should only be expected in large-scale discrete environments where "the inner subproblem is expensive" (consistent with Appendix B's cost model), not in small environments.

## 6.5 O05 · Universal Neural Optimal Transport (ICML 2025)

> **One sentence summary**: UNOT trains a Fourier Neural Operator \(S_\phi\), which takes a pair of discrete measures as input and outputs the dual potential \(g\) for entropic OT. The potential then derives the plan, distance, barycenter, and geodesic. Training relies on an adversarial generator \(G_\theta\) to create distribution pairs + a self-supervised bootstrapping loss (using the result of 5-step Sinkhorn as the target), thus **not relying on any real dataset**. It is the only amortized OT solver in this repository that truly achieves "cross-dataset, cross-resolution generalization"—and therefore is an unavoidable strong baseline when evaluating whether "conditional GFlowNet–OT truly has a cross-task generalization advantage": if a conditional GFN–OT cannot beat "feeding the graph shortest path matrix to a UNOT-style dual predictor + one Sinkhorn step," then its generalization narrative is invalid.

| Field | Content |
|---|---|
| arXiv | [2212.00133](https://arxiv.org/abs/2212.00133) (v6, 2025-06-12) |
| Publication | **ICML 2025 Main Conference** (Original homepage: Proceedings of the 42nd ICML, Vancouver, PMLR 267, 2025) |
| Authors | Jonathan Geuter\* (Harvard SEAS / Kempner Institute), Gregor Kornhardt\*, Ingimar Tomasson\* (TU Berlin Mathematics Department), Vaios Laschos (Weierstrass Institute); the first three are co-first authors |
| Code | https://github.com/GregorKornhardt/UNOT (includes model weights for experiments) |
| PDF in this repository | `papers/2212.00133.pdf` · Chinese translation Not generated |
| Reading Priority | **P1** — It defines the current SOTA and evaluation protocol for the "amortized OT solver" track; GFN–OT's generalization claim must be stated relative to it |

#### 2. Core Contributions (Numbered as in the Original Text)

**(C1) The first neural OT solver capable of generalizing across datasets and input dimensions (Sec. 1, 1st item in contributions list).** Conclusion: For a fixed cost, a single model handles arbitrary resolutions from \(10\times10\) to \(64\times64\) and arbitrary datasets, with a relative error of 1–3%. In contrast, Meta OT (Amos et al. 2023) can only handle fixed dimensions and collapses when applied to data outside its training set (Table 4).

**(C2) Discrete-to-continuous convergence of dual potentials (Proposition 2, formal version in Appendix B).** Conclusion: If \(c\) is Lipschitz with respect to both variables, \(\mathcal{X}\subset\mathbb{R}^N\) is compact, and \((\mu_n),(\nu_n)\) are sequences of discretizations of absolutely continuous \(\mu,\nu\), then the extended potentials \(f_n,g_n\) (after normalization \(f_n(x_0)=0\)) **converge uniformly over the entire \(\mathcal{X}\)** to the solution \((f,g)\) of the continuous problem. This justifies the use of neural operators (instead of ordinary networks): the learned object itself has a continuous limit.

**(C3) Universality of the generator (Theorem 3 + Corollary 4).** Let \(0<\lambda\leqslant1\), \(G_\theta(z)=\mathrm{ReLU}(NN_\theta(z)+\lambda z)\), \(\mathrm{Lip}(NN_\theta)=L<\lambda\). Then \(\tilde G_\theta(z)=NN_\theta(z)+\lambda z\) is invertible on \(\mathbb{R}^d\) (citing the invertible ResNet result from Behrmann et al. 2019), and

\[
\rho_{G_{\theta\#}\rho_z}(x)\ \geqslant\ \frac{1}{(L+\lambda)^d}\,\mathcal{N}\big(\tilde G_\theta^{-1}(x)\,\big|\,0,I\big)\qquad \forall x\in\mathbb{R}^d_{\geqslant0}.
\]

This means the generator has positive density at **any** non-negative \(x\): in principle, the training process can encounter any pair of fixed-dimensional discrete distributions. Corollary 4 generalizes this to compositions of \(\tilde G_\theta\) (covering a large class of ResNets). **Note that Appendix C itself admits three discrepancies between Theorem 3 and the actual architecture** (a constant \(\delta\) is added, input and output dimensions differ, and the Lipschitz constant is not controlled during training). Thus, this is a "possibility theorem for architecture classes," not a "guarantee for the implementation in this paper."

**(C4) Self-supervised bootstrapping loss (Proposition 5).** Let \(\tau_k\) denote \(\boldsymbol{g}_{\tau_k}\) obtained by running \(k\) steps of Sinkhorn, initialized with the current prediction (and shifted to zero-sum). Then

\[
L_2(\boldsymbol{g}_\phi,\boldsymbol{g})\ \leqslant\ c(K,k,n)\,L_2(\boldsymbol{g}_\phi,\boldsymbol{g}_{\tau_k}),\qquad c(K,k,n)>1 .
\]

This means minimizing "the difference from the \(k\)-step Sinkhorn result" necessarily bounds "the difference from the true solution." The proof relies on Hilbert projection metrics (Appendix B, Lemma B.5 + the contraction property of Peyré–Cuturi Thm. 4.1). **This is the most transferable technique in the entire paper** (see Section 7). They use \(k=5\) and \(\epsilon=0.01\).

**(C5) Adversarial training objective (Eq. (7) + Algorithm 2).** \(\max_\theta\min_\phi\mathbb{E}_{z\sim\rho_z}\big[L_2\big(\tau_k(G(z),S(G(z))),\,S_\phi(G_\theta(z))\big)\big]\), where \(S,G\) without subscripts denote non-gradient-tracking. The generator **maximizes** the solver's loss (opposite to the direction where the generator minimizes the discriminator's objective in GANs; this is explicitly clarified in Sec. 5).

**(C6) Downstream capabilities: not just distance.** They use \(\nabla_{\boldsymbol{a}}\mathrm{SD}_\epsilon(\mu,\nu)=\boldsymbol{f}-\boldsymbol{p}\) (Eq. (10), citing Feydy et al. 2018; \(\boldsymbol{p}\) is the potential for \((\mu,\mu)\)) for projected gradient descent to find Sinkhorn divergence barycenters (Eq. (9), Algorithm 3, 200 steps); use \(t\)-weighted barycenters or entropic plans to approximate McCann interpolation for geodesics (Sec. 4.3); and use Eq. (11) (citing Li et al. 2024b) for "distribution over distributions" Wasserstein-on-Wasserstein gradient flow (Sec. 4.4).

**(C7) Metric geometric characterization of discretization error (Appendix A.3, Prop. A.13 + Cor. A.14).** \((\mathcal{P}_2([0,1]^2),W_2)\) and \((\mathcal{P}([[n]]^2/n),W_2)\) are \((1,\tfrac{1}{\sqrt{2n}})\)-quasi-isometric:

\[
W_2(\mu,\nu)-\tfrac{1}{\sqrt{2n}}\ \leqslant\ W_2(f(\mu),f(\nu))\ \leqslant\ W_2(\mu,\nu)+\tfrac{1}{\sqrt{2n}} ,
\]

Therefore, continuous constant-speed geodesics are strong-\(\epsilon\) quasi-geodesics in the discrete space. **This is extremely useful for GFN–OT**: it provides an explicit scale \(O(n^{-1/2})\) for the error between "computing OT on a grid graph and computing OT in continuous space," serving as a bridge to translate hypergrid GFlowNet conclusions back into the language of continuous \(W_2\). The motivation given in the paper is that "there are no non-constant constant-speed geodesics in discrete space" (Dirac counterexample in Appendix A.3), so one must relax to quasi-geodesics.

#### 4. Experiments and Evidence

Test sets: MNIST (28×28), grayscale CIFAR10 (28×28), Quick, Draw! teddy bear class (64×64), LFW (64×64), and cross-dataset combinations CIFAR-MNIST, LFW-BEAR (\(\mu\) from one set, \(\nu\) from another). Appendix adds CARS and Facial Expressions (48×48). Spherical setting projects images onto \(S^2\). Unless otherwise noted, **one step** of Sinkhorn is run on the \(S_\phi\) output to obtain \(\boldsymbol{f}\); errors are averaged over 500 samples.

| Task | Baselines | Metric | Key Figures (Source) |
|---|---|---|---|
| Relative error of OT distance after one Sinkhorn step, \(c=\|x-y\|_2^2\), all scaled to 28×28 | Meta OT, Gauss (Thornton–Cuturi 2022), Ones (default \(\mathbf{1}_n\)) | Relative Error (%) | UNOT **2.7±2.4 / 1.3±1.1 / 2.8±2.6 / 1.5±1.3 / 2.0±1.6 / 1.8±1.3** (MNIST/CIFAR/MNIST-CIFAR/LFW/BEAR/LFW-BEAR); Meta OT 2.4±1.8 / 23.1±15.7 / 11.4±5.8 / 24.6±15.7 / 11.8±8.3 / 31.0±14.8; Gauss 18.1 / 19.7 / 32.2 / 21.1 / 20.4 / 19.3; Ones 39.5 / 47.4 / 74.5 / 56.9 / 54.2 / 66.4 (Table 4) |
| Sinkhorn iterations required to reach 0.01 relative error | Ones, Gauss | Average Iterations | UNOT 3±5 / 3±6 / 4±4 / 7±8 / 4±6 / 4±6; Ones 16±9 / 80±22 / 32±15 / 78±20 / 41±16 / 53±18; Gauss 10±7 / 52±19 / 13±9 / 35±14 / 25±13 / 29±13 (Table 1) |
| Wall-clock speedup to reach 0.01 relative error (JAX, batch 64, float32, RTX 4090) | Ones | Speedup Factor | MNIST 1.25, CIFAR **7.4**, CIFAR-MNIST 2.07, LFW 5, BEAR 3.8, LFW-BEAR 4.4; 28×28 average 3.57, 64×64 average 4.4; Meta OT reported speedup 1.96 (Table 2) |
| MNIST barycenter (after 100 gradient descent steps) | Gauss, Ones initialization | \(W_2\) to true barycenter | UNOT **0.021±0.011**, Gauss 0.033±0.018, Ones 0.057±0.034 (Table 6) |
| Geodesics (McCann interpolation) | True OT plan, **GeONet (this repo O03)** | Visual comparison | Original paper Sec. 4.3 + Fig. 8: Both UNOT geodesics (from OT plan, from barycenter) are closer to ground truth than GeONet, **even though UNOT was neither trained on geodesics nor seen MNIST**. Qualitative only, no numerical table |
| Cross-resolution (10×10 to 64×64 up/downsampling) | Ones, Gauss | Relative error curve | Fig. 12: Stable across the entire range; Appendix D.5 variant with varying \(\epsilon\) shows increased error when resolution \(<15\times15\) or near \(70\times70\) |
| Varying \(\epsilon\) (0.01–1, as third input channel) | — | Relative error heatmap | Fig. 13: Performance relatively stable across \(\epsilon\) (same model) |
| Generator difficulty tracking | — | OT distance error of generated samples | Error of newly generated samples: 53.2% at 0% training, 3.1% at 10%, then 1.6–2.1%; re-evaluating early samples at end of training drops them to 1.1–2.0% (Table 5) |

Evidence strength grading:

- **Direct experimental support**: Cross-dataset generalization (Meta OT in Table 4 errs 11–31% on unseen datasets, UNOT consistently 1.3–2.8%); cross-resolution generalization (Fig. 12); SOTA status as Sinkhorn initialization (Table 1/2); barycenters more accurate than other initializations (Table 6).
- **Honest exceptions**: Meta OT (2.4%) is slightly better than UNOT (2.7%) on MNIST—but Meta OT was **trained on MNIST**, UNOT has not seen it. The original paper interprets this as "the generator covers the MNIST class distribution," which is an inference, not a proof. Additionally, Appendix D.7 points out that **wall-clock time with default initialization is faster on MNIST** (Fig. 17), which is two sides of the same phenomenon as the 1.25x in Table 2: for small problems, the overhead of network forward pass eats into the gains from fewer iterations.
- **Author-stated implementation disadvantages**: FNO handles complex numbers, while PyTorch is better optimized for real numbers; it would be faster with complex kernel support (mentioned in Sec. 4.1 footnote 7 and Appendix D.7); Thornton–Cuturi's initialization was implemented by them to be slower than default, so it was not included in wall-clock figures.
- **Unverified cheap alternative**: Appendix D.3 reports that for fixed-size scenarios, replacing the neural operator with an MLP can achieve <5% relative error in **minutes**. This raises serious cost-effectiveness questions about whether an operator is truly needed; the original paper does not provide a full comparison table for the MLP version.
- **Limitations (Sec. 6, self-reported by authors)**: Cannot extrapolate to inputs significantly higher than training resolution; cannot generalize to cost functions other than those trained on.

#### 6. Position in the GFlowNet × OT Main Thread

- **Predecessors**: Sinkhorn (Cuturi 2013) and its initialization research (Thornton & Cuturi 2022); Meta OT (Amos et al. 2023, uses the dual objective itself as loss, fully unsupervised but cannot OOD); Neural Operator / FNO (Li et al. 2021; Kovachki et al. 2024); gradient formula for Sinkhorn divergence (Feydy et al. 2018). Predecessors within this repository are O01 (OT basics) and O02 (quadratically regularized OT on graphs, providing a precedent for "discrete/graph structure + regularization").
- **Comparison / Competition**: **O03 (GeONet) is the direct object it outperforms** (Sec. 4.3, Fig. 8); C01 (Unsupervised Learning for OT plan prediction between unbalanced graphs, NeurIPS 2025) is its counterpart on graph structures and a closer competitor for GFN–OT.
- **Significance for the main thread ("internal flow selection = optimal transport")**: UNOT itself **does not** contribute any flow selection mechanism; it contributes **the comparative baseline and evaluation protocol for this main thread**. Specifically, three points:
  1. **It correctly identifies the dual potentials as the object for "amortization"**. In the min-flow GFlowNet dual LP given by O08 Theorem 3.3, the dual variable \(\pi_x\) is the Kantorovich potential on the graph. **This implies a very natural implementation path for "conditional GFN–OT": instead of conditioning \(P_F\), condition the potentials/state flow \(\log F\), and then derive the policy from the potentials via complementary slackness as in O08.** UNOT's success provides empirical support for this path.
  2. **It firmly establishes the evaluation protocol**: cross-dataset (\(\mu,\nu\) from different families), cross-resolution (including beyond training range), cross-\(\epsilon\), and the separation of "distance accuracy vs. geometric accuracy" (Appendix A.2 counterexample + centroid/geodesic experiments in Sec. 4.2/4.3). If GFN–OT only reports in-distribution metrics, its persuasiveness will be insufficient.
  3. **Prop. A.13's \((1,\tfrac{1}{\sqrt{2n}})\) quasi-isometry is a ready-made tool to translate hypergrid conclusions back to continuous \(W_2\)**: the OT cost calculated on an \(n\times n\) grid graph deviates from continuous \(W_2\) by no more than \(1/\sqrt{2n}\). O08's hypergrid experiment (\(H=10\)) can thus be converted into continuous language, making it directly comparable with UNOT/GeONet.

##### Experimental Design for Comparison (Evaluating "Whether Conditional GFN-OT Truly Has Cross-Task Generalization Advantage")

**Task Definition.** A task \(=(G, L, R)\) consists of: an environment graph \(G=(\mathcal{S},E)\), an initial flow distribution \(L\) on \(U=\mathrm{out}(s_0)\), and a target (reward) distribution \(R\) on the terminal states \(\mathcal{X}\). According to O08 Theorem 3.2, the optimal solution is the Kantorovich plan \(\Pi^\star\) with cost \(d(u,x)\) as the graph shortest path. Conditional GFN-OT = a policy network \(P_F(\cdot|\cdot;\,\mathrm{enc}(G,L,R))\), trained once on a task distribution, and then used for zero-shot inference on new tasks.

**Four Baselines, All Indispensable.**

| ID | Baseline | Purpose |
|---|---|---|
| B1 | Exact LP (`scipy.linprog`, same as O08 Sec. 4.1) | Provides \(\mathrm{OT}^\star\) anchor; only feasible for small graphs |
| B2 | Train a GFlowNet from scratch for each task until convergence (O08 Sec. 3.3 algorithm) | "Non-amortized" quality upper bound + cost upper bound |
| B3 | **UNOT Ported Version**: Precompute shortest path matrix \(D\), let \(K=\exp(-D/\epsilon)\), use an operator/GNN to predict \(\boldsymbol{g}\), +1 step Sinkhorn to get \(\boldsymbol{f}\) (this paper Eq. (5) + Prop. 1) | **Core Competitor**. It enjoys the exact same conditional inputs \((L,R)\) as GFN-OT, differing only in "solving a static problem with Sinkhorn" vs "solving a dynamic problem with a policy" |
| B4 | Meta OT Ported Version: Fixed-size MLP predicts \(\boldsymbol{g}\) (Amos et al. 2023's approach, "MetaOT" column in Table 4 of this paper) | Checks whether "discrete-invariant operator parameterization" is truly necessary; addresses the MLP-UNOT question in Appendix D.3 of this paper |

**Metrics (Report all five, reasons in parentheses).**
1. Transport cost relative error \(|\sum_{u,x}\Pi_{u,x}d(u,x)-\mathrm{OT}^\star|/\mathrm{OT}^\star\) (O08's primary metric).
2. Coupling error itself, e.g., \(\|\Pi-\Pi^\star\|_1\) or \(W_2\) to \(\Pi^\star\) (**must be reported separately**, reason is the counterexample in Appendix A.2 of this paper: similar cost does not imply similar coupling).
3. Marginal constraint violation MCV \(=\tfrac12(\|\mathbf{1}_m^\top\Pi-\nu^\top\|_1+\|\Pi\mathbf{1}_n-\mu\|_1)\) (this paper Eq. (20); for GFN, corresponds to reward matching and violation of \(P_F(\cdot|s_0)=L\)).
4. Expected trajectory length \(\mathbb{E}[n_\tau]\) (O07/O08 metric; B3 has no trajectories, this item is only meaningful for GFN, used to explain "what extra does a dynamic solution provide").
5. **Wall-clock time to achieve 1% relative error, including amortization**: Plot total time ("training + \(N_{\mathrm{task}}\) inferences") against number of tasks \(N_{\mathrm{task}}\), and find intersection points with \(N_{\mathrm{task}}\times\) single-task time for B1/B2. **This is the true test of the amortization claim**, as demonstrated by the pitfalls in Table 2 and Appendix D.7 (default initialization being faster on MNIST) of this paper.

**Five Generalization Axes (Copied from UNOT protocol, plus one unique to GFN).**
- **(a) New tasks from the same family**: \((L,R)\) resampled from the training family. Corresponds to MNIST→MNIST in this paper.
- **(b) Cross-family**: \(L\) is moon-shaped, \(R\) is multi-modal in corners (two types used in O08); trained on a single family, tested on mixed families. Corresponds to CIFAR-MNIST / LFW-BEAR in this paper. **This is the column where Meta OT failed and UNOT did not**, GFN-OT must provide numbers here.
- **(c) Scale extrapolation**: Hypergrid side length \(H\) swept from 8 to 64 (including beyond the training range), or permutation environment \(n\) swept from 10 to 20 (O08's approach, LP is intractable for \(n=20\), so B2 is used as reference). Corresponds to Fig. 12 and Appendix D.5 of this paper, showing "degradation beyond training resolution".
- **(d) Change cost = Change graph**: Trained on grid graphs, tested on graphs with random shortcuts added/edges removed. **This axis is structurally impossible for UNOT** (Sec. 6 limitation: does not generalize beyond training cost) – because its \(C\) is burned into weights during training, while GFN-OT's policy takes \(G\) as input. **If conditional GFN-OT has a real advantage, it will only appear on this axis.** If it cannot beat "recomputing \(D\) and rerunning B3" even on this axis, then the generalization narrative should be abandoned in favor of claiming memory advantage ("no need to instantiate \(n\times n\) kernel matrix").
- **(e) Change \(\epsilon\) / Change entropy regularization strength**: Corresponds to the varying-\(\epsilon\) variant in Appendix D.5 of this paper (treating \(\epsilon\) as an additional input channel). For GFN, corresponds to "flow regularization coefficient \(\lambda\)" (O08 Sec. 4.1 ablation).

**Two Confounding Variables That Must Be Controlled.**
1. **Coverage of the training task distribution**. UNOT's generalization largely comes from the adversarial generator \(G_\theta\) (Theorem 3's density lower bound + Table 5 showing the generator indeed creates difficult samples first). If conditional GFN-OT is only trained on manually designed task families, the comparison in (b)(c) would be unfair. **Either equip GFN-OT with a similar adversarial task generator, or train both on the same fixed task set and report simultaneously.**
2. **Computational budget alignment**. B3's 35-hour training cost (Sec. 4 of this paper) and GFN-OT's sampling cost are of different magnitudes; they must be compared on the same x-axis (wall-clock time or number of function evaluations), not by "number of iterations".

**Falsifiable Predictions (Written down to be disproven).** On axes (a)(b)(c), B3 (UNOT ported + one-step Sinkhorn) will significantly outperform conditional GFN-OT on small to medium-sized graphs, because it solves a convex problem and has contraction guarantees like Prop. 5; conditional GFN-OT's advantage should only appear in (d) (changing graphs/costs) and scenarios where "\(|\mathcal{S}|\) is too large to store \(D\) or \(K\)."

#### 7. Reusable Insights and Open Problems

1.  **Port bootstrapping loss to GFlowNets: use results from \(k\)-step exact updates as targets.** Prop. 5 has the structure "target = own prediction + \(k\)-step contraction operator", with correctness relying on contractivity. The corresponding candidate contraction operator in GFlowNets is "\(k\)-step flow-matching/DB exact back-substitution". **This can be written as a draft theorem**: If the discrete flow-matching operator is a \(\kappa<1\) contraction under some metric (e.g., Hilbert projection metric / span seminorm of \(\log F\)), then \(\|\log F_\phi-\log F^\star\|\leqslant\frac{1}{1-\kappa^k}\|\log F_\phi-\log F_{\tau_k}\|\). This would provide a self-supervised loss for conditional GFN–OT that **does not require the true value \(Z\)**.
2.  **Adversarial task generators are an underestimated component in the GFlowNet community.** Theorem 3 shows that as long as the architecture is \(\mathrm{ReLU}(NN_\theta(z)+\lambda z)\) and \(\mathrm{Lip}(NN_\theta)<\lambda\), the generator can cover all non-negative vectors. Transplanting this to graph tasks: a network generating \((L,R)\) pairs using the same invertible ResNet structure + softmax normalization can achieve the same coverage argument. Experiment: compare the gap in generalization axes (b)(c) between "training with hand-crafted task families" and "training with adversarial task generation".
3.  **The \((1,\tfrac{1}{\sqrt{2n}})\) quasi-isometry (Prop. A.13) should become the standard conversion for GFN–OT reporting.** Currently, OT cost numbers on hypergrids are not comparable with continuous \(W_2\) literature; with this bound, one can directly state "our solution differs from the continuous \(W_2\) optimum by at most \(\mathrm{err}_{\mathrm{graph}}+1/\sqrt{2n}\)". **This also leads to an open problem**: For non-grid graphs (e.g., permutation graphs, Cayley graphs), do similar quasi-isometric bounds exist? This determines whether GFN–OT results in these environments can be translated into geometric language.
4.  **"Distance accurate \(\neq\) geometrically accurate" must be included in the GFN–OT evaluation checklist.** The two-point counterexample in Appendix A.2 can be literally ported to graphs: construct two reward distributions such that the transport cost is arbitrarily close while the optimal coupling remains fixed at a distance. **This is a 10-line construction problem**, which, if solved, can be directly placed in the evaluation motivation section of GFN–OT papers.
5.  **A negative cost-effectiveness signal worth verifying first: are MLPs sufficient?** Appendix D.3 states that MLPs of fixed size train to <5% in a few minutes. If GFN–OT claims "GNNs are necessary for cross-graph generalization", a conditional MLP version for fixed graph sizes should be run as a baseline first, otherwise "architectural complexity" might be mistaken for "generalization capability".
6.  **The amortization breakeven point should be explicitly plotted.** The 1.25× (MNIST) vs 7.4× (CIFAR) in Table 2 shows a 6-fold difference, and Appendix D.7 even provides a counterexample where default initialization is faster on MNIST. For GFN–OT: plot a "number of tasks vs. total wall-clock" intersection graph, and explicitly state "amortization is only worthwhile for more than \(N^\ast\) tasks". This number is more convincing to reviewers than any relative error table.
7.  **A combination not yet explored: UNOT-style potential predictor + O08-style shortest path cost + GFlowNet policy extraction.** A three-step pipeline: (i) use an operator network to predict Kantorovich potentials \(\pi_x\) on the graph (dual variables of O08 Theorem 3.3); (ii) determine the support edge set (shortest path subgraph) from complementary slackness conditions; (iii) use GFlowNet on this subgraph to only learn "how to distribute mass among shortest paths". This approach, by delegating the convex part to operators and the combinatorial part to policies, might simultaneously avoid B3's memory bottleneck and GFN's slow convergence. **This is the solution this report considers most worth prioritizing.**

## 6.6 O06 · Computing High-Dimensional OT with Flow Neural Networks (AISTATS 2025)

> **One sentence summary**: Q-flow directly solves Benamou–Brenier dynamic OT using a neural ODE: both marginal distributions \(P,Q\) have only finite samples and no analytical densities, so the two terminal constraints are relaxed to KL (using a logistic classifier to estimate density ratios on the fly), plus a finite-difference form of the \(W_2\) transport cost term, trained bidirectionally and alternately. The learned OT trajectories are directly used for two downstream tasks: image-to-image translation and high-dimensional density ratio estimation (DRE). **To preempt the most common misinterpretation: "flow neural network" in the title refers to the velocity field of continuous normalizing flows / neural ODEs, and has no relation to GFlowNet** (see end of Section 1 for details).

| Field | Content |
|---|---|
| arXiv | [2305.11857](https://arxiv.org/abs/2305.11857) (v5, 2025-03-10) |
| Publication | **AISTATS 2025 Main Conference** (registered in this repository's `data/papers.yaml`); the PDF in this repository is pure arXiv v5, **without conference footer on pages**, so publication status cannot be confirmed by the PDF itself |
| Authors | Chen Xu (Georgia Tech, ISyE), Xiuyuan Cheng (Duke Math), Yao Xie (Georgia Tech, ISyE) |
| Code | https://github.com/hamrel-cxu/FlowOT (Original Sec. 5) |
| This repository's PDF | `papers/2305.11857.pdf` · Chinese translation Not generated |
| Reading Priority | **P2** — No structural sharing with GFlowNet, but it serves as a clean reference for the "solving dynamic OT for each pair of distributions individually" end (opposite to the amortized end of O03/O05), and its KL-relaxation technique is directly transferable to GFN–OT's marginal constraint handling |

#### 2. Core Contributions (Numbered as in Original Text)

**(C1) Q-flow net: Directly learns a continuous, invertible transport map from two sets of samples, and pushes it towards dynamic OT using a transport cost (Sec. 3, Eq. (3)–(8)).** Conclusion: End-to-end training can "refine" any initial flow (e.g., two concatenated CNFs, or a flow from a distribution interpolation network) from "matching only endpoints" to "matching endpoints with low transport cost." Prerequisite: The initial flow already approximately matches the two endpoints (Sec. 3.3), otherwise the classifier is difficult to train.

**(C2) Bidirectional training (Eq. (3) + Remark 1).** Conclusion: The Benamou–Brenier pair \(P,Q\) is symmetric—whether \(P\) is placed at \(t=0\) or \(t=1\), the optimal solution is the same velocity field, differing only in time direction. Therefore, simultaneously optimizing \(L^{P\to Q}\) and \(L^{Q\to P}\) points to the same solution at optimality, which can improve accuracy with finite samples. **This point is supported by experiments** (Table A.3: BPD improved from unidirectional 1.08/1.19/1.31 to bidirectional 1.05/1.14/1.31).

**(C3) Flow-ratio net: Uses OT trajectories for infinitesimal DRE (Sec. 4, Eq. (10)–(12)).** Conclusion: From \(\log(q/p)=\int_0^1\partial_t\log p(x,t)\mathrm{d}t\) (Eq. (A.2)), parameterize the "time score" \(\partial_t\log p(x,t)\) as \(r(x,t;\theta_r)\), and train with logistic classification on adjacent time grid points; once trained, \(\log(p(x,t)/p(x,s))\) for any \(s<t\) can be obtained by integration. **The key argument is that "using OT trajectories as bridges is better than using linear interpolation bridges,"** because the latter (Eq. (A.8)'s \(X(t_k)=\sqrt{1-\alpha_k^2}X(0)+\alpha_kX(1)\), the scheme used by TRE/DRE-\(\infty\)) is significantly different from OT trajectories (Fig. A.9 vs Fig. 3(a)).

**(C4) Experimentally, it simultaneously benchmarks against static and dynamic OT baselines, as well as SB baselines (Sec. 5).** Provides numbers on Korotin's OT benchmark, CelebA64 alignment, image translation, high-dimensional MI, and MNIST energy models (see Section 4).

#### 6. Position in the GFlowNet × OT Main Line

- **Predecessors**: Benamou–Brenier; CNF / Neural ODE (Chen et al. 2018; Kobyzev et al. 2020); CNF with transport regularization (Finlay et al. 2020; Onken et al. 2021; Xu et al. 2022, 2023's JKO-iFlow); Distribution Interpolation / Flow Matching (Albergo–Vanden-Eijnden 2023; Lipman et al. 2023; Liu 2022); on the DRE side, TRE (Rhodes et al. 2020) and DRE-\(\infty\) (Choi et al. 2022).
- **Comparison / Competition**: Solves the same Benamou–Brenier problem as **O03 (GeONet)** but in the opposite direction (O03 uses amortization + PDE residuals, O06 uses single-pair solving + KL relaxation + sampling); competes with **O04 (α-DSBM)** on the same set of image translation tasks, and DSBM significantly lags in Table 2 of this paper; operates on a completely different level from **O05 (UNOT)**—O05 amortizes the dual potentials of static entropy OT, while O06 does not amortize and solves dynamic OT.
- **Contribution to the main line ("internal flow selection = optimal transport")**: It contributes **constraint handling techniques**, rather than flow selection mechanisms. Specifically, three points:
  1. **The trick of "replacing hard boundary constraints with differentiable KL readings" is directly applicable to GFN–OT**. The construction in O08 requires fixing the initial edge flow distribution \(P_F(\cdot|s_0)=L\) (what O08 Sec. 3.3 calls leward matching) and reward matching \(P_B(x|s_f)=R(x)/Z\). Both are hard constraints, softened in practice by loss terms. O06's approach provides a stronger version: using a binary classifier to estimate the density ratio between the "currently pushed distribution" and the "target distribution," thereby obtaining an unbiased reading of KL—on a graph, this means "using a discriminator to distinguish between terminal states sampled by GFlowNet and target reward samples," which can serve as a differentiable proxy for the violation of reward matching.
  2. **The finite difference form of the transport cost term (Eq. (8)) is the continuous counterpart of "expected trajectory length"**. O07/O08 use \(\mathbb{E}[n_\tau]\) or total flow as cost; O06 uses \(\sum_k\|X(t_k)-X(t_{k-1})\|^2/h_k\). Both are "accumulating local movements along trajectories," and both are **free** (the former is trajectory length, the latter uses already computed intermediate points). This correspondence indicates that GFlowNet's flow regularization term is the kinetic energy term in a continuous setting, and vice versa.
  3. **The symmetry argument for bidirectional training (Remark 1) may not hold on graphs**, and warrants separate investigation. Benamou–Brenier's time symmetry comes from the equal status of \(P,Q\); GFlowNet's graph is directed, and \(s_0\) and \(s_f\) do not have equal status (O07 Assumption 2.1), so the conclusion that "forward training and backward training point to the same solution" **cannot be directly applied**. This is a concrete question that can be verified.
- **Reason it does not belong to the main line (must be clearly stated)**: The "flow neural network" in this paper shares a name with GFlowNet but is a different entity (clarified in Section 1). O06 can only be categorized under GFN–OT related work as a "neural solution for dynamic OT," not based on the literal similarity of "flow network."

#### 7. Reusable Insights and Open Problems

1.  **Use a discriminator to read KL, replacing hard edge constraints in GFN–OT.** Experimental draft: On the hypergrid environment of O08, replace the squared loss of reward matching with "discriminator-estimated KL" (discrete version of Eq. (6)–(7)), and compare convergence speed and final \(\mathrm{OT}^\star\) gap. The risk has been marked in the original text: the inner discriminator must be retrained frequently, otherwise the readings become invalid.
2.  **A theorem draft: Consistency of KL-relaxation.** The original text Sec. 6 lists "consistency analysis of KL handling boundary conditions" as an open problem. This problem is actually easier on **finite graphs**: the state space is finite, and the density ratio is a finite-dimensional vector, so a propagation bound can be directly given for "discriminator estimation error → KL estimation error → final coupling transport cost error". **Completing this would fill both the theoretical gap in O06 and the constraint relaxation theory for GFN–OT.**
3.  **The ablation of "OT trajectory vs. linear interpolation bridge" must be added, and it is cheap on graphs.** O06 did not do this (as pointed out in Section 4). It can be done on graphs: fix the downstream task of GFlowNet (e.g., use intermediate state distributions for curriculum learning or DRE), and use (i) trajectories from minimum flow GFlowNet, (ii) random walk trajectories, and (iii) uniformly interpolated trajectories as bridges, respectively, to observe differences in downstream metrics. This directly tests whether "optimal trajectories are indeed more useful," which is an implicit selling point of GFN–OT.
4.  **Reversibility is a diagnostic quantity worth introducing.** The inverse mapping error (\(10^{-7}\sim10^{-5}\)) in Table A.2 is a cheap and informative health check. The corresponding quantity for GFlowNet is "the difference between trajectory distributions induced by \(P_F\) and \(P_B\)" (violation of O08 Eq. (1)), and it is recommended to print this as a first-class citizen in training logs.
5.  **Whether time-direction symmetry holds on directed graphs is a decidable small problem.** The argument in Remark 1 relies on \(P,Q\) being symmetric; on graphs, \(s_0\to s_f\) is directed. Counterexamples can be constructed, or conditions for its validity can be given (e.g., the graph is reversible/bidirectional, or there exists a reference measure on the state space that makes the transition kernel reversible).
6.  **Cost intersection of amortized vs. single-pair solving.** O06 single-pair training takes 1.25–2.5 hours (Table A.1), while O05 one-time training takes about 35 hours. Therefore, the threshold for "amortization being worthwhile" is roughly in the range of 15–30 tasks. **This is a rough estimate obtained by directly dividing the numbers from O05 and O06** (calculation in this report, not a conclusion from the original text), but it sets a threshold that GFN–OT's amortization claim must cross.
7.  **A combination not yet explored**: Use O06's discriminator KL readings to handle marginal constraints, O08's minimum total flow to select flows, and O05's operator network for cross-task amortization. The division of labor for constraints/costs/amortization among the three does not conflict, making it a clean combination path.
8.  **The paradigm of "refining existing flows" itself can be transferred to GFlowNet.** O06's positioning is not to learn OT from scratch, but to **refine** any initial flow (concatenation of two CNFs / interpolation network) into an OT flow ("refinement" in Sec. 3.2), with pre-training accounting for only 9–12% of the total time (Appendix B.1). The corresponding approach for GFN–OT: first train a GFlowNet that only satisfies reward matching using standard TB (regardless of flow magnitude), then add flow regularization for a second-stage refinement, instead of training with the regularization term from the beginning. This is consistent with O04's two-stage structure (bridge matching pre-training + \(\alpha\)-DSBM fine-tuning). **The convergence of two different approaches to the sequence of "match endpoints first, then optimize cost" is worth adopting and verifying as an empirical rule.**
9.  **Byproduct perspective: Optimal coupling can serve as a sampler corrector.** See Section 3 (6). Corresponding experiment in GFN–OT: use the learned optimal coupling to correct an existing (suboptimal) terminal state sampler to the target reward distribution, and report the TV distance before and after correction—O08 already uses TV as a metric (its Table 1), so the evaluation pipeline for this experiment is ready.

# Chapter 7: Competitive Landscape and Collision Analysis

All three competing papers are main conference papers (NeurIPS 2025, ICML 2026, ICLR 2025), while our two papers are Workshop papers. This chapter first outlines the core contributions and main positions of the three papers, then provides a complete competitive matrix.

## 7.1 C01 · ULOT: Unsupervised Prediction of Unbalanced Inter-Graph OT Plans (NeurIPS 2025)

> **One sentence summary**: ULOT uses a GNN + cross-attention network conditioned on FUGW hyperparameters \((\alpha,\rho)\) to directly use the FUGW loss as a training signal, amortizing the prediction of unbalanced OT plans between two **explicit graphs**. Its inference complexity is \(O(n_1n_2)\), two orders of magnitude faster than classical solvers. It is currently the strongest contender for the topic of "amortized solving of a family of graph OTs": O08's goal of "conditional GFN learning a family of graph OTs" is largely covered by ULOT in terms of **amortized, conditional, graph, and unsupervised**. The only aspect not covered is "implicit combinatorial state graphs + executable local actions".

| Field | Content |
|---|---|
| arXiv | [2506.12025](https://arxiv.org/abs/2506.12025) (arXiv API: published 2025-05-21, updated 2025-07-08; local PDF page margin shows v3, 8 Jul 2025) |
| Publication | **NeurIPS 2025 Main Conference** (Main Conference Track, verified: NeurIPS proceedings page states "Advances in Neural Information Processing Systems 38 Main Conference (NeurIPS 2025) Main Conference Track", DOI 10.52202/085713-3146; dblp entry venue=NeurIPS year=2025 type=Conference and Workshop Papers). Note that the PDF front page still says "Preprint. Under review.", which is an arXiv v3 snapshot, not the final status |
| Authors | Sonia Mazelet, Rémi Flamary, Bertrand Thirion (CMAP École Polytechnique / Inria-Saclay) |
| Code | <https://github.com/smazelet/ULOT> (`github.com/SoniaMaz8/ULOT` will 301 redirect to this address; original appendix A states "code is available in the supplementary materials and will be released on github upon publication", and promises to release pre-trained weights) |
| PDF in this repository | `papers/2506.12025.pdf` |
| Reading Priority | **P1** — Not a theoretical competitor, but a **topic-occupying competitor**: it determines how much ground is left in the "conditional/amortized graph OT" direction |

#### 2. Core Contributions (Numbered as in Original Text)

The original text **does not contain any Theorems / Propositions / Lemmas**; all contributions are at the architectural and empirical levels. This point must be clarified first: ULOT's persuasiveness comes from experimental curves, not theorems.

1.  **Amortized Training Objective** (Original Text §2.2 Eq. (4)):
    \(\min_\theta\ \mathbb E_{G_1,G_2\sim\mathcal D^2,\ \alpha,\rho\sim\mathcal P}\big[\mathcal L_{\alpha,\rho}(G_1,G_2,P_\theta^{\rho,\alpha}(G_1,G_2))\big]\).
    The meaning of unsupervised lies here: **it does not require pre-computing a ground-truth plan using a solver**; instead, the FUGW objective itself is directly used as the loss for backpropagation. This is a standard approach in amortized optimization (Amos 2023), but it has been applied to non-convex quadratic OT.

2.  **Parameter Conditioning** (Original Text §2.2 "Encoding the parameters" + Eq. (5)): \(\rho\) is a positive scalar directly concatenated into node features; \(\alpha\) uses Fourier bases for positional encoding
    \(\hat\alpha=\big[(\cos(k\pi\alpha))_{k=1..d}\ \big|\ (\sin k\pi(1-\alpha))_{k=1..d}\big]\), with \(d=10\) in experiments. Sampling distributions: \(\mathcal P_\rho\) is log-uniform on \([10^{-7},1]\), \(\mathcal P_\alpha=\mathrm{Beta}(0.5,0.5)\). The original text self-assesses this as "is particularly novel and has not been done before to the best of our knowledge" (§2.4).

3.  **Cross-Attention Architecture** (Original Text §2.3 Eq. (6)–(10), Figure 1): Each layer has two paths—the self path is its respective GCN \(F_k^{\text{self}}=\mathrm{GCN}(F_k)\); the cross path first computes \(F_k^{\text{cross}}=\mathrm{MLP}(F_k,\rho,\hat\alpha)\), then calculates the cosine similarity matrix \(S_{i,j}=s((F_1^{\text{cross}})_i,(F_2^{\text{cross}})_j)\) and row/column softmax \(S_1=\mathrm{softmax}_{\text{row}}(a^2S),\ S_2=\mathrm{softmax}_{\text{col}}(a^2S)\) (\(a\) is a temperature hyperparameter).

4.  **Unbalanced Plan Output Layer** (Original Text Eq. (11)(12)): First, predict node gating \(v_k=\mathrm{sigmoid}(\mathrm{Linear}(F_k^{\text{final}},\rho,\hat\alpha))\), then
    \(P_\theta^{\rho,\alpha}(G_1,G_2)=\tfrac12\big(\tfrac1{n_1}S_1\mathrm{diag}(v_1)+\tfrac1{n_2}\mathrm{diag}(v_2)S_2\big)\).
    The design intent is explicitly stated in the original text: to decouple "cross-graph node correspondence" (managed by cross-attention) and "how much mass each node should output" (managed by \(v_k\)), with the latter being the unique degree of freedom in unbalanced OT.

5.  **Three Downstream Usages** (Original Text §3.1, §3.2): (a) The plan is fully differentiable with respect to \((\rho,\alpha)\) → allows gradient descent on hyperparameters (bi-level); (b) The plan is differentiable with respect to graph inputs → allows optimizing graphs for the FUGW functional; (c) The total mass of the plan \(m(P)=\sum_{i,j}P_{i,j}\) can be used as an \(O(n^2)\) graph similarity metric.

#### 5. Assumptions and Applicability Boundaries

ULOT's forward applicability scope:

1.  **Both graphs can be explicitly instantiated**: Requires \(F_k\) and \(D_k\) (adjacency or shortest path distance matrices) that can be constructed and fit into GPU memory, as well as an \(n_1\times n_2\) plan matrix.
2.  **Uniform node weights**: \(\omega_k^i=1/n_k\), explicitly assumed in original text §2.1. Non-uniform weights would require modification.
3.  **Scale limit \(n\le 10^4\)**: Original text §4 explicitly states limitations due to GPU memory, requiring specialized techniques like lazy tensors for larger scales.
4.  **Availability of a training set of graph pairs from the same distribution**: SBM relies on a generator, fMRI relies on random parcellation data augmentation. A different graph family would require retraining.
5.  **Cost structure fixed to FUGW**: Conditioning only covers the two scalars \((\alpha,\rho)\), not the form of the cost function itself.
6.  **Output is a plan matrix, not a policy**: It tells you "how much mass goes from \(i\) to \(j\)", not "how to get from \(i\) to \(j\)". For node matching tasks without intermediate path concepts, this is not a defect; for tasks requiring execution of valid local actions, this is a significant drawback.

#### 6. Position in the GFlowNet × OT Main Line

##### 6.1 Overlap with O08 "Conditional GFN for a Family of Graph OT"

**Occupied Portion (approx. 60–70% overlap)**:

| Selling Point | Done by ULOT? | Evidence |
|---|---|---|
| Amortization: one model serves a family of OT problems | ✅ Fully achieved | Eq. (4), one model trained on 14400 graph pairs on IBC |
| Conditioning: policy/network takes OT problem parameters | ✅ Achieved for \((\alpha,\rho)\) | Eq. (5), Fourier encoding |
| Unsupervised: no ground-truth plan needed | ✅ Achieved | Eq. (4) directly uses OT objective as loss |
| Graph structure | ✅ GNN + cross-attention | §2.3 |
| Unbalanced (mass can be added/removed) | ✅ \(\rho\)-KL + \(v_k\) gating | Eq. (3)(11) |
| "Neural prediction → classical solver warm start" paradigm | ✅ Demonstrated | Figure 8(right) |
| Generalization to unseen graph pairs | ✅ 60/20/20 test set | §3.2 |

Our assessment: **If "Conditional GFN for a Family of Graph OT" is taken literally, its selling points are almost entirely covered by ULOT, which is also a NeurIPS 2025 main track paper.** In particular, the "GFN proposal + classical OT correction" idea has already been demonstrated in Figure 8(right), so it can only be relegated to a baseline comparison, not a primary selling point.

**Unoccupied Portion (differentiated moat)**:

1.  **Object space.** ULOT requires explicit \(D_1,D_2\) and an \(n_1\times n_2\) plan, with a hard upper limit of \(n\le10^4\). In O08's permutation environment, for \(n=20\), the number of states is \(20!\approx2.4\times10^{18}\), making physical instantiation of the plan matrix impossible. This is a **qualitative, not quantitative, difference**.
2.  **Output type.** ULOT outputs a plan, O08 outputs \(P_F(s'|s)\). When "transport" must be achieved through a sequence of valid local actions (molecular editing, adjacent swaps in permutations, program transformations), the plan itself is useless—you still need to solve "how to move."
3.  **Cost structure.** ULOT uses quadratic FUGW (GW term couples two plan elements), while O08 uses linear Kantorovich, with cost induced by graph shortest paths \(d_G(u,x)\). These are not the same mathematical problem and cannot be directly compared numerically.
4.  **Error guarantees.** ULOT has none. O08 has LP strong duality (Theorem 3.3) and complementary slackness providing a primal-dual gap. **This is the only theoretically clear advantage for the GFN side** and the foundation for the "Balance residual → OT error bound" topic.
5.  **Unbalanced direction reversal.** ULOT is inherently unbalanced, while O08 strictly requires normalization at both ends. Therefore, the "unbalanced GFN-OT" topic **collides with ULOT in its setup but not in its method**—ULOT's \(v_k\) gating can be directly adapted for GFN's learnable mass sinks.

##### 6.2 A Comparison Table: ULOT vs. O08 (Comparable and Incomparable Aspects)

| Dimension | ULOT (C01) | O08 |
|---|---|---|
| Role of graph | Object being transported (node sets of two explicit graphs) | State space where transport occurs (inside a single directed graph) |
| Scale | \(n\le10^4\) (original paper §4 GPU memory limit) | permutation \(n=20\), state count \(20!\approx2.4\times10^{18}\) (O08 §4.2) |
| Cost | Quadratic FUGW (W + GW + KL) | Linear Kantorovich, cost \(=d_G(u,x)\) shortest path (O08 Eq. (7)(8)) |
| Marginal constraints | Soft (KL penalty, \(\rho\)) | Hard (Assumption 3.1: \(\sum L=\sum R=1\)) |
| Output | Plan matrix \(P\in\mathbb R^{n_1\times n_2}\) | Forward policy \(P_F(s'|s)\), coupling is a byproduct of sampling |
| Amortization | Yes (one model for multiple graph pairs + multiple hyperparameters) | No (one model trained per pair of \(L,R\)) |
| Theoretical guarantees | None (no theorems in the entire paper) | Theorem 3.2 (GFlow\(^\star=\)OT\(^\star\)), Theorem 3.3 (duality + complementary slackness) |
| Speedup evidence | 100× vs solver (Figure 7 right) | Consistent with POT exact solution: \(\mathbb E|\tau|=3.990\pm0.015\) vs OT\(^\star=3.997\) for \(H=10\) (O08 Table 1) |
| Code | Open-sourced | Not publicly available |

**Incomparable aspects must be clearly stated**: The OT problems are different (quadratic vs. linear), costs are different (FUGW vs. graph shortest path), and marginal constraint strengths are different. Therefore, **any assertion like "ULOT is 100 times faster, so GFN is out of the game" is an incorrect comparison**. The real competition lies at the "research narrative" level, not the numerical level.

##### 6.3 Predecessors and Baselines

-   **Predecessors**: FUGW (Thual et al., NeurIPS 2022, cited as [30] in the original text); GW (Mémoli 2011 / Peyré et al. 2016); Meta OT (Amos et al. 2022, cited as [2] in the original text, ULOT claims to be its generalization to unbalanced graph OT); amortized optimization (Amos 2023).
-   **Baselines/Competitors**: O05 Universal Neural OT (ICML 2025); Neural GW OT (Nekrashevich et al. 2023); deep graph matching family (SuperGlue, etc., but these are supervised).
-   **Contribution to the main line of this repository**: ULOT serves as an **external competitive reference** for the "internal flow selection = optimal transport" line. It demonstrates that amortized graph OT is engineeringly feasible and valuable (100× speedup is real), while also showing what it cannot do (implicit graphs, executable policies, error certificates). Differentiation must be firmly established on these latter three points.

#### 7. Reusable Insights and Open Problems

1.  **Incorporate \(v_k\) gating into GFN for unbalanced OT**. ULOT Eq. (11)(12) uses sigmoid gating for the output mass of each node. The GFN counterpart is to parameterize the terminal edge flow \(F(x\to s_f)\) as \(g_\theta(x)\cdot R(x)\), where \(g_\theta\in(0,1)\), and add a \(\rho\,\mathrm{KL}\) penalty to the loss instead of hard reward matching. This directly addresses the open problem of "unknown \(Z\) / unequal mass" left by O08, and is an experiment that can be run within a week.
2.  **Fourier parameter encoding → regularization strength input for conditional GFN**. O08's \(\lambda\) (flow regularization coefficient) is currently an external hyperparameter. Experimental Table 2 shows a strong trade-off between trajectory length and sampling accuracy for \(\lambda=10^{-1}\) vs \(10^{-2}\). Encoding \(\lambda\) into \(P_F(\cdot|s,\lambda)\) according to ULOT Eq. (5) would yield the entire regularization path in a single training run. This is the easiest form of "conditionalization" to implement on the GFN side without directly clashing with ULOT.
3.  **Plan-level evaluation is a gap; seize the opportunity**. ULOT only reports loss correlation. Any GFN-OT work that reports cost gap, marginal TV, support overlap rate, and primal-dual gap (O08 Theorem 3.3 provides this readily) will surpass ULOT in terms of evaluation rigor.
4.  **Break-even analysis is missing**. The core question for amortized methods is "how many queries are needed to amortize the training cost?" ULOT Appendix A mentions 100 GPU hours but does not provide a break-even point. This is an experiment anyone can do and get credit for.
5.  **Inverse problem: Can ULOT handle implicit graphs?** No, because cross-attention requires enumerating \(n_1\times n_2\) pairs. But a sharper question can be asked: **If O08's permutation environment is compressed to \(n_1,n_2\le10^4\) "representative states", can ULOT outperform GFN?** This is a red-team experiment that must be run first, otherwise reviewers will ask.
6.  **Draft of propositions to be proven**: ULOT completely lacks "bounds on the distance between the predicted plan and the optimal plan." A corresponding GFN version is possible—given a TB loss residual of \(\varepsilon\), can we prove \(\big|\mathbb E[|\tau|]-\mathrm{OT}^\star\big|\le C(\varepsilon,\lambda,|\mathcal S|)\)? O08 Theorem 3.3's complementary slackness condition \(F^\star(s\to s')(\pi^\star_{s'}-(1+\pi^\star_s))=0\) is a natural starting point for this bound.

## 7.2 C02 · GSBoG: Generalized Schrödinger Bridge on Graphs (ICML 2026)

> **One sentence summary**: GSBoG extends Generalized Schrödinger Bridge (GSB) from \(\mathbb R^d\) to **controlled CTMCs on fixed sparse graphs**, expressing the optimal jump rate as \(u_t^\star(y,x)=r_t(y,x)e^{V_t(x)-V_t(y)}\) using the Hopf–Cole potential \(V_t\), and learning it with two losses: gIPF + TD. It represents **two extremes in the same quadrant** as O08: both select a path measure under the setting of "directed graph + marginals at both ends + executable local policy output," but GSBoG chooses "entropy regularization (KL to reference dynamics) + intermediate state cost," while O08 chooses "unregularized minimum total flow." This is the most direct threat to O08's positioning in this repository.

| Field | Content |
|---|---|
| arXiv | [2602.04675](https://arxiv.org/abs/2602.04675) (v1 2026-02-04, v2 2026-06-10) |
| Publication | **ICML 2026 Main Conference**. Verification chain: (a) PDF footer is in camera-ready format "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"; (b) Corresponding author Jaemoo Choi's personal webpage entry for 2026-04 states "7 papers are accepted to ICML", explicitly listing "GSBoG: Generalized Schrödinger Bridge Sampler on graphs"; (c) Georgia Tech Research Conference Spotlight 2026 includes this title. **However, dblp as of 2026-09 only has a CoRR entry**, and the official PMLR volume is not yet searchable, so `venue_type` is recorded as `main` with this discrepancy noted. |
| Authors | Panagiotis Theodoropoulos, Juno Nam, Evangelos Theodorou, Jaemoo Choi (Georgia Tech / MIT) |
| Code | **Not publicly available** (no repository link in main text or appendix; not mentioned in acknowledgments/Impact Statement) |
| This repository's PDF | `papers/2602.04675.pdf` (29 pages) |
| Reading Priority | **P1** (effectively P0 for the "Entropy-Regularized GFN–OT/SB" topic) |

#### 2. Core Contributions (Numbered as in Original Text)

1.  **Theorem 3.1 (Dual Representation)**. Under "mild regularity assumptions," there exists a time-varying function \(V:[0,1]\times\mathcal X\to\mathbb R\) such that the optimal probability path \(p_t^\star\) satisfies the coupled system (Eq. (15) in original text):
    \[
    \partial_tp_t^\star(x)=\sum_y r_t(x,y)e^{-V_t(x)+V_t(y)}p_t^\star(y),\qquad
    \partial_tV_t(x)=\sum_y r_t(y,x)e^{-V_t(y)+V_t(x)}-f_t(x,p_t^\star),
    \]
    with boundary conditions \(p_0^\star=\mu,\ p_1^\star=\nu\); and the optimal control rate is (Eq. (16)):
    \[
    \boxed{\,u_t^\star(y,x)=r_t(y,x)\exp\big(-V_t(y)+V_t(x)\big).\,}
    \]
    The proof is in Appendix A.1: introduce Lagrange multiplier \(V_t\) for CE, integrate by parts, invoke strong duality, and differentiate with respect to \(u\) to get \(\log(u/r)+V_t(y)-V_t(x)=0\).

2.  **Remark 3.3 (Topology Automatically Satisfied)**. In Eq. (16), \(\exp(\cdot)>0\), so \(r_t(y,x)=0\Rightarrow u_t^\star(y,x)=0\). The optimal control **automatically inherits the sparse support of the reference generator**, requiring no additional feasibility projection.

3.  **Proposition 3.4 (Hopf–Cole on Graphs)**. Define \(\varphi(t,x):=e^{-V(t,x)},\ \hat\varphi(t,x):=p_t^\star(x)/\varphi(t,x)\) (Eq. (17)), then (Eq. (18)):
    \[
    \partial_t\varphi_t(x)=-\sum_y r_t(y,x)\varphi_t(y)+f_t(x,p_t^\star)\varphi_t(x),\qquad
    \partial_t\hat\varphi_t(x)=\sum_y r_t(x,y)\hat\varphi_t(y)-f_t(x,p_t^\star)\hat\varphi_t(x),
    \]
    and \(p_t^\star=\varphi_t\hat\varphi_t\), \(\nu=\varphi_1\hat\varphi_1\). The optimal rate transforms into a **local ratio** \(u_t^\star(y,x)=r_t(y,x)\varphi_t(y)/\varphi_t(x)\) (Eq. (19)).

4.  **Proposition 3.5 (Generator Identities)**. Let \(Y_t=\log\varphi_t,\ \hat Y_t=\log\hat\varphi_t\), \(Z_t(y,x)=Y_t(y)-Y_t(x)\). Then the forward/backward generators acting on \((Y,\hat Y)\) have closed forms (Eq. (21)–(24)). The core is \(A_t^uY(x)=f_t+\sum_y r_te^{Z_t}(Z_t-1)\), \(A_t^u\hat Y(x)=\sum_y(r_t(x,y)e^{\hat Z_t}+\hat Z_tr_te^{Z_t})-f_t\).

5.  **Proposition 3.6 (Discrete gIPF Loss)**. Based on Dynkin's formula (Eq. (20)), expand the endpoint likelihood into an integral of generators (Eq. (25)), yielding (Eq. (26)(27)):
    \[
    \mathcal L_{\mathrm{IPF}}^{Z}(\hat Z)=\mathbb E_{p^Z}\Big[\int_0^1\sum_y\Big(r_t(X_t,y)e^{\hat Z_t}+r_t(y,X_t)e^{Z_t}\big(-1+Z_t+\hat Z_t\big)\Big)\,dt\Big],
    \]
    which is backward symmetric. Appendix A.4 explains that it is an upper bound for \(-\log p_0(x_0)\), achieving equality at the optimum.

6.  **TD Loss (Eq. (28)(29))** – the most critical technical point of this paper, see §3.

7.  **Algorithm 1**: Alternates between "forward rollout → update \(\phi\)" and "backward rollout → update \(\theta\)", with total loss
    \(L=\mathcal L_{\mathrm{IPF}}^{Z}(\hat Z^\phi)+\mathcal L_{\mathrm{IPF}}^{\hat Z}(Z^\theta)+\lambda_{\mathrm{TD}}\big(\mathcal L_{\mathrm{TD}}^{Z}(\hat Y^\phi)+\mathcal L_{\mathrm{TD}}^{\hat Z}(Y^\theta)\big)\).

8.  **Appendix B (Equivalence to Stochastic Optimal Control)**: Starting from the KL-regularized finite-horizon control problem (Eq. (77)) and performing dynamic programming yields the HJB equation (Eq. (83)):
    \(-\partial_tV(t,x)=\sum_{y\ne x}r_t(y,x)\big(1-e^{V(t,x)-V(t,y)}\big)+f(t,x,p_t)\), which is consistent with Theorem 3.1. The original text explicitly states that the only conceptual difference is the **terminal condition**: SB uses hard boundary constraints, while SOC uses a soft terminal cost \(g(X_1)\).

#### 5. Assumptions and Applicability Boundaries

Forward applicability scope:

1.  **Graph topology is fixed and fully known** – The first sentence of §4.5 Limitations in the original text states this. Both the reference generator and learned rates are defined on a given edge set, and rollouts and loss evaluations require enumerating \(N(x)\).
2.  **Reference generator \(r_t\) must be given and sparse**: It simultaneously plays the roles of "which actions are legal" and "nominal preference." The entire method's sparsity benefit comes from the sparse support of \(r_t\).
3.  **Both marginals \(\mu,\nu\) are explicitly given probability vectors** (not sample sets), defined on the same set of nodes.
4.  **Fixed finite time horizon** \([0,1]\), practically a fixed number of steps \(T\) (100 for supply chain, 200 for Chignolin). If the target is not reached, it is not reached – there is no "trajectory length adaptation."
5.  **Cost can be written as \(f_t(x,p_t)\)**: state cost or mean-field cost. Pairwise costs require transformation using the intermediate node trick in Eq. (85).
6.  **Hard constraints like capacity are not enforced**, only implicitly reduced by congestion costs (explicitly remarked in §4.1 of the original text).
7.  The conclusion is **relative optimality with respect to the chosen reference process**: changing \(r_t\) changes the problem. This is a different philosophy from O08, where "cost is uniquely determined by graph topology."

#### 6. Position in the GFlowNet × OT Main Line

##### 6.1 Adjacency with O08: Two Poles in the Same Quadrant

Shared structural settings (the essence of "adjacency"):

- A given directed graph;
- Edge constraints at both ends (O08: \(F(s_0\to u)=L(u)\) and \(F(x\to s_f)=R(x)\); GSBoG: \(p_0=\mu,\ p_1=\nu\));
- Selecting an "optimal" one among **all flows/path measures satisfying the edge constraints**;
- The output is not a static coupling, but rather a **step-by-step executable local policy** (O08: \(P_F(s'|s)\); GSBoG: \(u_t(y,x)\));
- The selling points of both are almost verbatim. O08 conclusion paragraph: "Unlike standard OT approaches that only learn a coupling, our framework learns a stochastic policy that transports samples through feasible local moves"; GSBoG introduction second paragraph: "it does not describe how mass should move over time, nor does it yield an executable control policy on the graph". **The same narrative niche is occupied by both papers.**
- Both have dual potentials, and both characterize optimal support using potential differences: O08 Theorem 3.3's \(\pi\) (Kantorovich potential, where \(\pi_x^\star=d(x)\) at optimality), complementary slackness \(F^\star(s\to s')(\pi_{s'}^\star-(1+\pi_s^\star))=0\) indicates that flow only falls on the tight subgraph (=shortest path subgraph); GSBoG Theorem 3.1's \(V_t\) (HJB value function), \(u^\star=r\,e^{\Delta V}\) indicates that rates are exponentially reweighted by potential differences.

##### 6.2 The Difference Lies in "Entropy Regularization vs. Minimum Total Flow"

| Dimension | O08 (Minimum Total Flow) | GSBoG (Entropy-Regularized GSB) |
|---|---|---|
| Objective | \(\min\sum_{e\in E^\circ}F(e)\), linear, no regularization (O08 Eq. (11)) | \(\min\mathbb E\!\int f_t+\mathrm{KL}(p^u\|p^r)\) (Eq. (13)) |
| Reference Dynamics | None (implicitly "all internal edges cost 1") | \(r_t\) explicitly given, part of the problem |
| Solution Properties | LP vertex solution, supported on shortest path subgraph, **can be sparse or even degenerate** | Exponential family form \(u^\star=r\,e^{\Delta V}\), **strictly positive and smooth everywhere** on edges where \(r>0\) |
| Time Structure | No time axis, absorbing state \(s_f\), \(\mathbb E[n_\tau]\) free | Fixed time horizon \([0,1]\)/\(T\) steps, marginals fixed at both ends |
| Cost | \(d_G(u,x)\), depends only on origin-destination pair | \(f_t(x,p_t)\), can depend on intermediate states, can depend on current marginal (mean-field) |
| Error Characterization Provided? | Yes: LP strong duality + complementary slackness = primal-dual gap (Theorem 3.3) | No (not provided in the original text) |
| Convergence Proof | LP has exact solution; neural training no guarantee | IPF/TD alternation has no discrete convergence proof |
| Implicit Graph | Supported (permutation \(20!\) states) | Not supported (requires enumerating \(N(x)\) and \(\mu,\nu\) are explicit vectors) |

A one-sentence summary of this comparison: **GSBoG sets the "temperature" to a finite value and introduces a reference process, gaining smooth solutions, mean-field costs, and IPF training; O08 sets the temperature to 0 and discards the reference process, gaining exact equivalence to Kantorovich LP and dual certificates.**

##### 6.3 Collision Assessment

The topic of "entropy-regularized GFN–OT / Schrödinger bridge," if framed as "adding an \(\varepsilon\,\mathrm{KL}(P\|P_0)\) to O08's objective," carries a **high risk of collision**: GSBoG has already occupied the entire combination of "graph + reference dynamics + entropy regularization + executable policy + intermediate state cost" at ICML 2026 main conference, and its experimental scale (\(10^6\) nodes) far exceeds O08's hypergrid/permutation.

Remaining niches, ordered by feasibility:

1.  **Unfixed time horizon**. GSBoG must fix \(T\); GFNs are absorbing, and trajectory length is determined by the policy itself. "Entropy-regularized GFN–SB with unfixed time horizon" (where KL is applied to the trajectory distribution rather than a fixed time grid) has not been occupied, and it falls squarely within the comfort zone of non-acyclic GFlowNet theory (T19/T36).
2.  **Implicit combinatorial graphs**. All experimental graphs in GSBoG are explicit, enumerable, and from physical networks (supply chains, MSMs). It has not touched, nor can it easily touch, spaces like Cayley graphs or molecule editing graphs that "can only be locally expanded" (\(\mu,\nu\) would have to be \(n\)-dimensional vectors).
3.  **Error bounds**. Neither side has them. Whoever first writes down the bound for "balance loss residual → OT cost gap" will secure the sole theoretical increment in this line of work.
4.  **Mean-field cost has no counterpart in GFNs**. The setting where \(f_t(x,p_t)\) depends on the current marginal is currently blank in GFlowNet literature ("congestion-aware GFlowNets"). This is a genuine niche, but caution is needed: multi-agent GFlowNets (Brunswic et al. 2025b, cited by O08) might already be touching upon it.

##### 6.4 Predecessors and Comparisons

-   **Predecessors**: Chow, Li, Mou, Zhou (2022) dynamical SB on graphs (direct comparison GrSB in this paper, ran out of memory on 9559 nodes); Liu et al. (2022) DeepGSB (source of TD loss); Liu et al. (2024) GSBM; De Bortoli et al. (2021) DSB / IPF; Essid & Solomon (2018) quadratic regularized OT on graphs (O02 in this repository, also the source of O08 Assumption 3.1).
-   **Concurrent/Comparison**: Guo et al. (2026) discrete adjoint SB sampler (original text §3.1 explicitly states "concurrently considered the same formulation"); Ksenofontov & Korotin (2025) categorical SB; **C03 DDSBM** (original text §3 and Appendix D repeatedly emphasize that DDSBM does "generative modeling on graph spaces," which is a different problem from GSBoG's "routing on fixed topology"); Yang (2025) topological SB matching.
-   **Contribution to the main line**: GSBoG provides a complete technical stack for "entropy-regularized path measure selection on graphs" (duality → Hopf–Cole → generator → IPF+TD). For this repository, it is both a competitor and a **ready-made derivation template**: by replacing CTMC with discrete-time GFN transition kernels, the Lagrangian derivation of Theorem 3.1 can be almost line-by-line transferred.

#### 7. Reusable Insights and Open Questions

1.  **Draft Proposition for Direct Derivation (Optimal Policy Form for Entropic Regularized GFN–OT)**. Given a reference forward policy \(P_F^0\), minimize \(\mathbb E[|\tau|]+\varepsilon\,\mathrm{KL}(P\|P^0)\) under the constraints of O08 (\(F(s_0\to u)=L(u),\ F(x\to s_f)=R(x)\), flow conservation). By introducing a multiplier \(V(s)\) and differentiating with respect to the policy, similar to Appendix A.1 of GSBoG, we should obtain
    \(P_F^\star(s'|s)\propto P_F^0(s'|s)\exp\big((V(s)-V(s')-1)/\varepsilon\big)\).
    As \(\varepsilon\to0\), this should degenerate to the tight-subgraph support of O08 (Theorem 3.3). **This derivation is the most direct output of this in-depth reading and should be completed on paper before deciding whether to initiate a project.**
2.  **Intermediate Node Trick (Eq. (85)) Can Extend the Expressiveness of O08's Cost**. The cost in O08 is fixed as the shortest path \(d_G\) with unit edge length. Using the GSBoG approach—inserting intermediate nodes for transitions requiring additional cost and expressing the cost as a stopping cost—allows for expressing arbitrary positive pairwise costs (by replacing an edge with weight \(w\) with \(w\) unit-length segments) **without modifying Theorem 3.2**. This is a remark that can be immediately written into follow-up work for O08, and it also clarifies the boundary of the concern "zero-weight edges break the acyclicity of optimal flow" mentioned in this repository's background document: as long as the weights are positive integers, they can be expanded into unit edges.
3.  **Cancellation Check Should Be Standard Practice**. GSBoG §3.3 found that \(f_t\) cancels out in the IPF summation. Corresponding to GFN: any cost term added to the TB/DB loss must first be verified not to disappear in \(\log\frac{\prod P_F}{\prod P_B}\). The \(\lambda R/P_F(s_f|\cdot)\) term in O08 is spared because it is an additive additional term, but if someone writes the cost as a multiplicative reweighting of \(P_F\), they will fall into a trap.
4.  **GrSB Memory Exhaustion is a Reusable Baseline**. To argue that "GFN's locality leads to scalability," the most convincing experiment is to reproduce the four-axis scaling plot from GSBoG §4.4, including GSBoG as a baseline. Note: GSBoG is not open source, so reproduction requires self-implementation.
5.  **Points of Doubt to Verify**: Theorem 3.1 only mentions "mild regularity assumptions." On a finite \(\mathcal X\), if \(r_t(y,x)=0\) for some edges, leading to a disconnected graph, or if the support of \(\nu\) is unreachable from \(\mu\) within \(T\) steps, do strong duality still hold? The original text does not discuss this. Assumption 3.1, item three, of O08 (any \(u\to x\) has a path of finite length) is precisely an explicit version of this—**the GFN side is more rigorous on this point than GSBoG and can be highlighted in related work**.
6.  **Numerical Errata Memo**: The entropy values in Table 3 and Table 5 for \(n=20\) are inconsistent (0.17 / 0.22). Any reference to this number should state "inconsistent in two places in the original text."

## 7.3 C03 · DDSBM: Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation (ICLR 2025)

> **One sentence summary**: DDSBM extends Iterative Markovian Fitting (IMF) from continuous diffusion to CTMC on finite state spaces, proves monotonic convergence in the discrete case, and points out that when the reference process takes "independent jumps of nodes and edges," the corresponding entropic regularized OT cost is **proportional to the Graph Edit Distance (GED)**. It is the furthest paper from O08 on the "entropic regularized OT on graphs" line—the objects of transport are entire graphs rather than mass on graphs—but precisely because GED is "the shortest path distance on molecular editing graphs," there is a precise correspondence between it and O08's theorem, which is both a source of inspiration and a risk of collision.

| Field | Content |
|---|---|
| arXiv | [2410.01500](https://arxiv.org/abs/2410.01500) (v1 2024-10-02, v2 2025-02-28, arXiv comment explicitly states "Accepted to ICLR 2025") |
| Publication | **ICLR 2025 Poster** (Verified: OpenReview `tQyh0gnfqW` labels "ICLR 2025 Poster"; iclr.cc virtual poster 28054; dblp venue=ICLR year=2025 type=Conference and Workshop Papers). **Main conference poster, not workshop** |
| Authors | Jun Hyeong Kim\*, Seonghwan Kim\*, Seokhyun Moon\*, Hyeongwoo Kim\*, Jeheon Woo\*, Woo Youn Kim† (KAIST, first five authors contributed equally) |
| Code | <https://github.com/junhkim1226/DDSBM> (Official implementation, Python 3.9 / Torch 2.0.1 / cu118 / torch_geometric 2.3.1) |
| This repository PDF | `papers/2410.01500.pdf` (47 pages, 18 MB, main volume from molecular structure diagrams) |
| Reading Priority | **P2** — The setting is the most different, but it provides two directly reusable components: "discrete IMF convergence theorem" and "GED = OT cost" |

#### 2. Core Contributions (Numbered as in Original)

1.  **Definition 3.1 (reciprocal projection)**: \(\Pi_{\mathcal R(\mathbb Q)}(\Lambda)(\cdot)=\iint_{(\cdot)}\Lambda(dx_0,dx_\tau)\,\mathbb Q(dx_t|x_0,x_\tau)\). Preserves the coupling, replacing intermediate paths with a mixture of reference bridges (each bridge given by Doob's \(h\)-transform). The mixture is generally **no longer Markovian** (citing Léonard et al. 2014: the set of Markov measures is not convex).

2.  **Definition 3.2 (Markov projection)**: \(\Pi_{\mathcal M}(\Lambda)=\arg\min_M\{D_{\mathrm{KL}}(\Lambda\|M):M\in\mathcal M\}\). Preserves marginals at all times, **does not preserve coupling**. The generator of the projected measure and the explicit form of \(D_{\mathrm{KL}}(\Lambda\|\Pi_{\mathcal M}(\Lambda))\) are given in Proposition B.2.

3.  **Theorem 3.3 (Convergence of discrete IMFs)** – The theoretical core of this paper. Let \(D_{\mathrm{KL}}(\Lambda^{(0)}_{0,\tau}\|\mathbb P^{\mathrm{SB}}_{0,\tau})<\infty\) and \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\) (\(\forall n\)). Iterating \(\Lambda^{(2n+1)}=\Pi_{\mathcal M}(\Lambda^{(2n)}),\ \Lambda^{(2n+2)}=\Pi_{\mathcal R(\mathbb Q)}(\Lambda^{(2n+1)})\) (Eq. (4)), then
    \[
    D_{\mathrm{KL}}(\Lambda^{(2n)}\|\mathbb P^{\mathrm{SB}})\ \ge\ D_{\mathrm{KL}}(\Lambda^{(2n+1)}\|\mathbb P^{\mathrm{SB}})\ \ge\ D_{\mathrm{KL}}(\Lambda^{(2n+2)}\|\mathbb P^{\mathrm{SB}}),
    \]
    equality holds if and only if \(\Lambda^{(2n)}=\Lambda^{(2n+1)}=\mathbb P^{\mathrm{SB}}\); and \(\Lambda^{(n)}\) converges in distribution to \(\mathbb P^{\mathrm{SB}}\).

4.  **Training Loss** (Eq. (5)(6)): For the forward Markov projection, use
    \(\mathcal L(\theta)=\int_0^\tau\mathbb E_{\Lambda^{(2n)}_{t,\tau}}\big[(A_t^{\mathbb Q_{\cdot|\tau}}-A_t^{M^\theta})(X_t,X_t)+\sum_{y\ne X_t}A_t^{\mathbb Q_{\cdot|\tau}}\log\frac{A_t^{\mathbb Q_{\cdot|\tau}}}{A_t^{M^\theta}}(X_t,y)\big]dt\).
    For the backward projection, it is symmetrically given by the time-reversed generator \(\tilde A_t^{\mathbb Q_{\cdot|0}}\).

5.  **§4.3 Graph Permutation Matching**: The transition probabilities of the reference process depend on node numbering, so graph matching must be performed before calculating likelihoods/constructing reciprocal bridges. This is formalized as **QAP (NP-hard)**, approximated by max-pooling continuous relaxation from Cho et al. (2014) + Hungarian algorithm (Kuhn 1955, via Pygmtools) for discretization.

6.  **Appendix D.6 (Relationship with GED)**: Citing Bougleux et al. (2015), GED computation itself is equivalent to a QAP. Given \(\alpha=\bar\alpha(\tau)/\bar\alpha(0)\), the node/edge substitution costs are
    \(c^V(v_i,v'_a)=-\log\frac{(d^V-1)\alpha+1}{d^V}\) (same) or \(-\log\frac{-\alpha+1}{d^V}\) (different), and similarly for edges, where \(d^V,d^E\) are the state cardinalities. Thus, the QAP objective \(f(x)=\sum-\log P^E_{0:\tau}+\sum-\log P^V_{0:\tau}=p(\sigma G'|G)\).
    **The original text explicitly clarifies that GED \(\ne\) NLL, but is proportional**: the difference is proportional to the number of identical nodes and edges under the optimal matching \(\sigma^\star\) (because identity operations are also assigned a small but non-zero cost). Concluding sentence: "solving the SB problem can be understood as finding the OT plan between graph distributions, where the transport cost is defined by the GED."

#### 6. Position within the GFlowNet × OT Main Thread

**Farthest from O08, but with the most precise interface.** Differences across four dimensions:

| Dimension | DDSBM | O08 |
|---|---|---|
| Transport Object | Entire graph (one molecule = one point) | Mass on a graph (one node = one point) |
| Cost | \(-\log q_{\tau|0}\), \(\propto\) GED (Appendix D.6) | Graph-induced shortest path \(d_G(u,x)\) (Eq. (7)) |
| Entropy Regularization | Yes (KL to reference CTMC, temperature implicitly given by \(\bar\alpha\)) | No (pure linear minimum total flow) |
| Output | Generative model: sample \(x_\tau\) given \(x_0\) | Routing policy \(P_F(s'|s)\): select next step for a state |
| Convergence Guarantee | Theorem 3.3 (monotonic + convergence in distribution, no rate) | Theorem 3.2 (LP equivalence, exact); no guarantee for neural training |

**Precise Correspondence (one of the main outputs of this report)**: If we take O08's state graph \(G=(\mathcal S,E)\) to be a **molecular editing graph**—where nodes are molecules and edges are single-step edits (changing an atom type or a bond)—then the shortest path distance \(d_G(u,x)\) (Eq. (7)) defined by O08 **is the graph edit distance between \(u\) and \(x\)**. Thus:

> **O08's Theorem 3.2 applied to a molecular editing graph = an entropy-unregularized DDSBM with GED as cost.**

Conversely, DDSBM is its \(\varepsilon>0\) version. This correspondence has two sides:

- **Positive (Inspiration)**: It provides a ready-made problem definition, existing datasets (ZINC250K / Polymer), and established metrics (NLL / NSPDK / FCD / MAD) for "GFN doing molecular optimization OT." Moreover, GFN has an advantage that DDSBM lacks—**GFN traverses the edit path, and the path itself provides node correspondence, completely eliminating the need to solve QAP graph matching**. This is a point that can be directly written into the motivation.
- **Negative (Collision)**: Any work on "using GFN for entropy-regularized OT on molecular editing graphs" will be asked to explain its relationship with DDSBM. Furthermore, DDSBM has already been presented as a poster at ICLR 2025 main conference, with open-source code and complete experiments.

**Collision Rating**:
- For "Entropy-regularized GFN–OT / SB": **Medium-high**—it occupies the niche of "discrete SB + graphs + GED cost + molecular applications," but not "routing on fixed topology" (which is C02 GSBoG's niche).
- For "Conditional GFN learning a family of graph OTs": **Low**—DDSBM is not conditional at all; one model corresponds to a fixed pair of distributions.
- For "Balance residual \(\to\) OT error bound": **Low**, but its Theorem 3.3 is a **mandatory precedent to cite**—reviewers will ask, "IMF already has a convergence theorem, what's new about your TB residual bound?" The answer should be: Theorem 3.3 provides monotonicity for "iterating to SB," not a quantitative bound from "residual to cost gap"; they address different questions.

**Predecessors**: Peluchetti (2023a) / Shi et al. (2024) on IMF; Léonard (2013) on SB–EOT equivalence; De Bortoli et al. (2021) on DSB; Vignac et al. (2022) on DiGress (network and noise scheduling); Bougleux et al. (2015) on GED–QAP equivalence.
**Comparisons**: C02 GSBoG (its §3 and Appendix D repeatedly emphasize "DDSBM does generative modeling on graph space, we do routing on fixed topology," with both papers delineating their scopes); Ksenofontov & Korotin (2025) on categorical SB.

#### 7. Reusable Insights and Open Problems

1.  **Write a remark: GED = \(d_G\) on molecular editing graphs**. This formalizes the correspondence from §6, can be written in a single paragraph, and serves both "related work" and "application motivation." Note the boundaries: O08 requires unit edge length, so the equality holds only when all editing operations have the same cost; weighted GED requires using C02's intermediate node trick.
2.  **GFN's freedom from graph matching is a real comparative advantage**. DDSBM solves approximate QAP every time it constructs a reciprocal bridge (§4.3). GFN traverses the edit sequence, and the correspondence between source and target is implicitly given by the path. This can be made into an empirical comparison: for the same ZINC task, report the proportion of time DDSBM spends on graph matching.
3.  **Can IMF's "iterative coupling improvement" be transferred to GFN's \(P_B\)?** DDSBM's core mechanism is alternating updates of coupling and Markov processes. The GFN counterpart is alternating updates of \(P_B\) (determining credit assignment) and \(P_F\). O08's conclusion section itself mentions backward policy optimization (citing Jang et al. 2024, Gritsaev et al. 2025). **Mapping IMF's reciprocal/Markov dual projection structure to \((P_F,P_B)\) alternating optimization is a technical direction with clear provenance.**
4.  **Evaluation protocol is directly usable**: The three-layer structure of NLL (joint, measuring transport cost) + NSPDK/FCD (marginal, measuring distribution matching) + MAD (measuring side effects) precisely corresponds to OT's "cost / marginals / additional constraints." The GFN-OT experimental table can directly adopt this framework.
5.  **Theorem 3.3 lacks a rate, which is a point of attack**. It only provides monotonic decrease and convergence in distribution, and assumes \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\) (unverified for neural approximation). If a conclusion **with a rate** or **with a residual bound** can be provided on the GFN side (even under weaker assumptions), that would be a clear theoretical increment.
6.  **Unanswered question**: DDSBM has never measured the distance between its learned coupling and the true EOT plan—because the true solution cannot be computed. GFN can compute the true solution using POT/LP on a small scale (O08 Table 1/2 already does this). **A two-tier evaluation with "true solution for small scale, proxy metrics for large scale"** is something GFN can offer that DDSBM cannot.

## 7.4 Competitor Matrix

#### 0. Identity Cards of the Five Papers

| ID | Abbr. | Title | Publication Status (Verified) | Code |
|---|---|---|---|---|
| **O08** | GFN-OT | Your GFlowNet Secretly Learns an Optimal Transport Plan | **ICML 2026 SPIGM Workshop** (arXiv comment explicitly states; PDF front page "Preprint. June 5, 2026") | Not publicly available |
| **O07** | GFN-SP | Learning Shortest Paths with Generative Flow Networks | **ICML 2026 SPIGM Workshop** (This repository `data/papers.yaml`; arXiv 2603.01786 v1 2026-03-02) | `github.com/GreatDrake/gfn-pathfinding` (Given in original §1) |
| **C01** | ULOT | Unsupervised Learning for OT plan prediction between unbalanced graphs | **NeurIPS 2025 Main Conference** (proceedings page "Main Conference Track", DOI 10.52202/085713-3146; dblp NeurIPS 2025) | `github.com/smazelet/ULOT` |
| **C02** | GSBoG | Generalized Schrödinger Bridge on Graphs | **ICML 2026 Main Conference** (PDF footer PMLR 306 camera-ready; author's homepage 2026-04 acceptance announcement; GaTech Spotlight 2026). **dblp only CoRR entry as of 2026-09** | Not publicly available |
| **C03** | DDSBM | Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation | **ICLR 2025 Poster** (OpenReview `tQyh0gnfqW`; dblp ICLR 2025; arXiv comment "Accepted to ICLR 2025") | `github.com/junhkim1226/DDSBM` |

**The first fact to recognize**: Our two papers are **workshop** papers, while the three competing papers are from NeurIPS Main Conference, ICML Main Conference, and ICLR Main Conference, respectively. In terms of "narrative positioning," we are currently at a disadvantage.

#### 1. Main Matrix

| Dimension | **O08** GFN-OT | **O07** GFN-SP | **C01** ULOT | **C02** GSBoG | **C03** DDSBM |
|---|---|---|---|---|---|
| **Object Space** | **Implicit combinatorial graph**: inside a directed state graph \(G=(\mathcal S,E)\); for permutation \(n=20\), number of states \(20!\approx2.4\times10^{18}\) (§4.2) | **Implicit combinatorial graph**: Cayley graph, Rubik's Cube (3×3×3 states far exceed enumerability) | **Explicit graph**: between node sets of two given graphs; requires instantiation of \(D_1,D_2\) and an \(n_1\times n_2\) plan; hard limit \(n\le10^4\) (§4) | **Explicit graph**: fixed known topology, requires enumeration of \(N(x)\); tested up to \(10^6\) nodes (§4.4); but \(\mu,\nu\) must be explicit \(n\)-dimensional vectors | **Graph space**: one sample = an entire molecular graph; transport is between graphs, not within a graph |
| **What is transported** | Probability mass on a graph | Path from single source to target (degenerate case of OT) | Correspondence mass between nodes of two graphs | Probability mass (particles) on a graph | Entire graph (molecule → molecule) |
| **Cost structure** | Linear Kantorovich, \(d(u,x)=|\tau_{u,x}|\) graph-induced shortest path (Eq. (7)(8)) | Shortest path length (special case where \(R\equiv1\)) | **Quadratic non-convex** FUGW: W term + GW term (coupling \(P_{i,j}P_{k,l}\)) + \(\rho\)-KL (Eq. (1)(2)(3)) | \(\mathrm{KL}(p^u\|p^r)+\mathbb E\!\int f_t(x,p_t)\); \(f_t\) can depend on **current marginals** (mean-field, Eq. (13)) | \(-\log q_{\tau|0}(y|x)\), **proportional to GED** (Appendix D.6, original text states "not equal to, proportional to") |
| **Entropy regularized?** | **No**. Pure linear LP, optimal solution is a vertex solution, supported on the shortest path subgraph (Thm 3.3 complementary slackness) | No | **No** (it's quadratic regularized, not entropy regularized); Sinkhorn entropy-regularized FUGW baseline in comparative experiments | **Yes**. KL to reference CTMC \(r_t\), solution is exponential family \(u^\star=r\,e^{\Delta V}\) (Eq. (16)) | **Yes**. KL to reference CTMC; temperature implicitly given by noise schedule \(\bar\alpha\) |
| **Reference dynamics** | None (equivalent to all internal edge costs = 1) | None | Not applicable | **Yes, and must be given** \(r_t\), encoding topology + nominal routing preference | **Yes**: node/edge independent jumps (Eq. (8)(9)) |
| **Conditionable / Amortized?** | **No**. Train one model for each pair \((L,R)\) | No | **Yes, and is a core selling point**: one model serves multiple graph pairs + multiple \((\alpha,\rho)\); \(\alpha\) encoded with Fourier (Eq. (5)) | No. Train once for each \((\mu,\nu,r,f)\) | No. Train once for each pair of distributions |
| **Output: plan or policy** | **Policy** \(P_F(s'|s)\); coupling \(\Pi_{u,x}=\sum_{\tau:u\leadsto x}P^\star(\tau)\) is a byproduct of sampling (Thm 3.2) | **Policy** (+ optional beam search) | **Plan** matrix \(P\in\mathbb R^{n_1\times n_2}\). Does not answer "how to move" | **Policy** \(u_t(y,x)\), executable edge by edge; soft plan can be decoded from flux (Eq. (87)) | **Generative model**: given \(x_0\) sample \(x_\tau\). Neither a static plan nor a routing policy |
| **Error / Convergence guarantees** | **Thm 3.2** GFlow\(^\star=\)OT\(^\star\) (exact); **Thm 3.3** LP duality + complementary slackness = primal-dual gap. **No bound between neural training and LP** | **Thm 3.4**: minimum total flow ⟹ policy only takes shortest paths | **None. Zero theorems in the entire paper** | **No error bounds**. Thm 3.1 relies on unspecified "mild regularity assumptions"; IPF/TD has no discrete convergence proof | **Thm 3.3**: IMF monotonically decreasing + converges in distribution to \(\mathbb P^{\mathrm{SB}}\). **No rate** |
| **Temporal structure** | No time axis, absorbing state \(s_f\), \(\mathbb E[n_\tau]\) free | Same as left | Not applicable (static) | **Fixed finite time horizon** \(T\) (supply chain 100, MD 200) | Fixed \([0,\tau]\), 100 diffusion steps |
| **Marginal constraints** | Hard: \(\sum L=\sum R=1\) (Assumption 3.1) | Single source + target set | **Soft**: \(\rho\)-KL penalty, unbalanced | Hard: \(p_0=\mu,p_1=\nu\) | Hard (dataset level): \(\mathbb P_0=\Gamma,\mathbb P_\tau=\Xi\) |
| **Code availability** | Not publicly available | Not extracted within the scope of this detailed reading | **Open-sourced** | **Not publicly available** | **Open-sourced** |
| **Max experimental scale** | Hypergrid \(H=20\); permutation \(n=20\) (Table 1/2) | 3×3×3 Rubik's Cube | Graph \(n=1000\) (fMRI), scaled to \(10^4\) | Supply chain \(n=9559\); scaled to \(10^6\) nodes (Fig 6) | ZINC250K / Polymer 7603 pairs |
| **Compute disclosure** | MLP 2 layers ×128, CPU training (Appendix B) | Original text not extracted within the scope of this detailed reading | Single V100 **100 hours** (Appendix A) | AdamW \(5\times10^{-5}\); \(5\times10^4\) nodes 1.4 GB / ~1 h (Fig 6) | **4× RTX A4000** (Table 4) |

#### 2. Three Undilutable Dividing Lines

**Dividing Line A: Is the graph "the object being transported" or "the venue where transport occurs"?**
ULOT and DDSBM fall into the former (graph as object), while O08 / O07 / GSBoG fall into the latter (graph as venue). Comparing numbers across this line is meaningless—ULOT's "100x acceleration" and O08's "\(\mathbb E|\tau|=3.990\) vs OT\(^\star=3.997\)" address different problems.

**Dividing Line B: Explicit enumeration vs. implicit unfolding.**
Although GSBoG scales to \(10^6\) nodes, it requires \(\mu,\nu\) to be explicit \(n\)-dimensional vectors, requires \(N(x)\) to be enumerable, and requires the topology to be fully known (§4.5 Limitations, first sentence). In O08's permutation environment, \(20!\) states **have no explicit representation** and can only be locally unfolded via "current permutation + valid adjacent swaps." This is the only structural advantage on the GFN side that cannot be replicated.

**Dividing Line C: Temperature.**
O08/O07 operate at \(\varepsilon=0\) (pure LP, vertex solutions, with dual certificates); GSBoG/DDSBM operate at \(\varepsilon>0\) (exponential family solutions, smooth, with IPF/IMF training mechanisms). ULOT is not on this line (its regularization is quadratic, and the penalty is on the edges). **The action of "adding a KL term to O08" is moving oneself from \(\varepsilon=0\) to \(\varepsilon>0\)—where GSBoG (ICML 2026 main conference) and DDSBM (ICLR 2025) already stand.**

#### 3. Specific Threats from Competitors

| Competitor | What is threatened | Intensity | Counterpoints |
|---|---|---|---|
| **C02 GSBoG** | O08's **entire positioning narrative**. The selling points of both papers are almost verbatim ("not static coupling, but executable policy"). GSBoG is at ICML 2026 main conference, \(10^6\) nodes, three real-world applications | **Highest** | (i) Implicit combinatorial graphs; (ii) Non-fixed time domain (GSBoG must fix \(T\)); (iii) O08 has LP dual certificates, GSBoG has no error bounds; (iv) GSBoG is not open-source |
| **C01 ULOT** | The **specific topic** of "conditional GFN learning a family of graph OT" | High | (i) ULOT hard upper bound \(n\le10^4\); (ii) Outputs plan, not policy; (iii) Zero theorems in the entire paper; (iv) Cost is quadratic FUGW, not the same problem as linear Kantorovich |
| **C03 DDSBM** | The **application niche** of "entropy-regularized GFN-OT for molecular editing" | Medium | (i) DDSBM needs to solve QAP graph matching, GFN naturally doesn't need to follow paths; (ii) DDSBM does not condition; (iii) Its Thm 3.3 is "iterative convergence," not "residual \(\to\) cost gap" |
| **O07 (Our side)** | Not a threat, but a predecessor to O08. O08 Thm 3.3 explicitly states "This recovers the corresponding claim of Morozov et al. (2026)" | — | The two papers should merge their narratives, not fight separately |

#### 4. Collision Risk Assessment for Four Candidate Topics

Rating basis: **Overlap in settings** (whether competitors are already working under the same settings) \(\times\) **Overlap in selling points** (whether the core claim has already been made) \(\times\) **Publication venue** (whether the competitor is at a main conference or workshop).

##### Topic 1 · Balance Residual \(\to\) OT Error Bounds
> Goal: Prove quantitative bounds of the form \(\big|\mathbb E[|\tau|]-\mathrm{OT}^\star\big|\le C(\varepsilon_{\mathrm{TB}},\lambda,|\mathcal S|)\), translating the TB loss residual into transport cost gap and marginal error.

**Collision Risk: Low (most worthwhile among current four topics)**

Basis:
- None of the five competing papers **provides error bounds**. ULOT has zero theorems; GSBoG has no bounds, and the regularity conditions in Thm 3.1 are not specified; DDSBM's Thm 3.3 only gives monotonicity and convergence in distribution, no rates, and assumes \(\Lambda^{(n)}\ll\mathbb P^{\mathrm{SB}}\) which is not verified for neural approximation.
- O08 has already laid the groundwork: Thm 3.3 provides the LP dual \(\max_\pi\sum_x R(x)\pi_x+\sum_u L(u)(-\pi_u)\) s.t. \(\pi_{s'}-\pi_s\le1\) (Appendix A.4 Eq. (24)(25)), and complementary slackness \(F^\star(s\to s')(\pi^\star_{s'}-(1+\pi^\star_s))=0\). **The primal-dual gap is a ready-made error certificate carrier.**
- The background document for this repository §5 has identified primal-dual as the most worthwhile direction, independently reaching the same conclusion as this collision scan.
- Query to prevent: Reviewers might use DDSBM Thm 3.3 to ask "IMF already has convergence theorems." The answer must be prepared: that refers to "iterative sequence converging to SB," while this topic is "residual of a single model \(\to\) cost gap." The former does not imply the latter, nor can it provide a computable certificate.

##### Topic 2 · Conditional GFN Learning a Family of Graph OT
> Goal: Train \(P_F(a|s,L,R,c)\) to generalize to unseen \((L,R)\) and cost.

**Collision Risk: High**

Basis:
- ULOT (NeurIPS 2025 main conference) has already accomplished amortization + conditioning + graphs + unsupervised + unbalanced + warm-start all at once, and provided generalization experiments on 14400 graph pairs for fMRI.
- O05 Universal Neural OT (ICML 2025) also occupies the amortization niche in general neural OT.
- **Still feasible form (only)**: Take the conditional variables to be **things that cannot be explicitly represented on implicitly combinatorial graphs**—for example, conditioning on cost weights \(\lambda\) (O08 Table 2 shows a strong trade-off between \(\lambda=10^{-1}\) vs \(10^{-2}\), and training once to give the entire regularization path is a clean increment), or conditioning on the set of generators for Cayley graphs. **Doing conditioning on explicit graphs = directly colliding with ULOT.**
- If we insist on doing it, we must run ULOT as a baseline in experiments (at scales it can handle), otherwise reviewers will ask "why not use ULOT."

##### Topic 3 · Entropy-Regularized GFN–OT / Schrödinger Bridge
> Goal: \(\min_P\mathbb E_P[c(\tau)]+\varepsilon\mathrm{KL}(P\|P_0)\), with fixed marginals.

**Collision Risk: High**

Rationale:
- GSBoG (ICML 2026 main conference) has fully covered "graphs + reference dynamics + entropy regularization + executable local policies + intermediate state costs". It even provides a complete technical stack: duality (Thm 3.1) → Hopf–Cole (Prop 3.4) → generator identity (Prop 3.5) → gIPF (Prop 3.6) → TD correction (Eq. (28)(29)).
- DDSBM (ICLR 2025) covers discrete SB + graphs + GED cost.
- Coupled with Ksenofontov & Korotin (2025) categorical SB, Guo et al. (2026) discrete adjoint SB, and Yang (2025) topological SBM—**this ecosystem is already crowded**.
- **Still viable forms**: (i) **Unfixed time horizon**—GSBoG/DDSBM both require fixed \(T\), while GFNs are absorbing and have adaptive trajectory lengths, applying KL to the trajectory distribution rather than a fixed time grid; this is a genuine gap. (ii) **Entropy-regularized SB on implicit combinatorial graphs**. (iii) **Mean-field / congestion-aware GFlowNets** (GSBoG's \(f_t(x,p_t)\) has no direct counterpart in GFN literature, but multi-agent GFlowNets should be checked first for overlap).
- Recommendation: **Do not treat this as an independent topic; treat it as an extension of Topic 1 for \(\varepsilon>0\)**. Error bounds are often easier to prove for \(\varepsilon>0\) (due to strong convexity), so merging these two lines offers theoretical novelty without directly clashing with GSBoG.

##### Topic 4 · GFN Proposal + Classical OT Refinement
> Goal: GFN outputs a coarse solution → feed to network simplex / Sinkhorn for refinement.

**Collision Risk: Medium (recommend demoting to a baseline, not a main line of research)**

Rationale:
- ULOT §3.2 + Figure 8(right) has already demonstrated the idea of "neural prediction → classical solver warm start" and explicitly stated it in the abstract. Novelty has been claimed.
- However, it remains valuable as an **evaluation protocol**: sample the coupling from the GFN-learned policy, feed it to an exact solver, and report the percentage reduction in iterations—this is a nearly zero-cost experiment that can be directly included in ablation studies, and it's the simplest answer to questions like "Is GFN actually useful?".
- Its value for independent publication is low, but it fits well as a section within Topic 1.

#### 4.5 Four Red-Team Experiments to Start Immediately

The purpose of these experiments is not to publish papers, but to **answer potential reviewer questions ourselves before submission**.

1.  **ULOT Reverse Experiment**: Compress O08's permutation environment to \(n_1,n_2\le10^4\) representative states, construct an explicit cost matrix, and directly run ULOT / POT. If ULOT outperforms GFN at this scale, then the "implicit graph" moat must be narrowed to scenarios where "even \(10^4\) representative states cannot be extracted."
2.  **Warm-start Protocol**: Sample the GFN policy into a coupling → feed it to a network simplex, and report the percentage reduction in iterations. Benchmark against ULOT Figure 8(right). Zero cost, directly into ablation tables.
3.  **Primal-Dual Gap Monitoring**: During training, simultaneously track O08 Thm 3.3's dual objective and the complementary slackness violation \(\sum_{s\to s'}F(s\to s')\big|\pi_{s'}-(1+\pi_s)\big|\). This is both a prototype for Topic 1 and the only online certificate that can prove "GFN output is trustworthy."
4.  **Four-Axis Scaling Comparison**: Replicate GSBoG §4.4's four-axis memory and time plots (number of nodes / edge density / time steps / rollout budget), and include GFN. Note that GSBoG is **not open-source**, requiring self-implementation—which also means that if our work is open-source, we gain an advantage in reproducibility.

#### 5. One-Sentence Conclusion

**The biggest threat to O08 is C02 GSBoG**: it operates in the same quadrant as O08 (graphs + fixed marginals + executable local policies), uses almost identical selling points, is published in ICML 2026 main conference, and its experimental scale is four orders of magnitude larger. However, it has three shortcomings that O08 does not—it requires fully known explicit topology, must have a fixed finite time horizon, and **lacks any error bounds**.

Therefore, the recommended main line of research is: **Combine "Balance Residual → OT Error Bounds" (low collision) with primal-dual/dual potentials (O08 Thm 3.3 ready-made) into a single main line, fix experimental scenarios to implicit combinatorial graphs (permutation / molecule editing / Rubik's Cube), and use ULOT and GSBoG as baselines "at scales where they can run."** The entropy-regularized version should be treated as an extension of this main line for \(\varepsilon>0\), rather than an independent topic.

---

**Editorial note** (self-decisions on ambiguities):
1.  The publication status of O07 is taken from this repository's `data/papers.yaml` (ICML 2026 SPIGM Workshop) and was not individually verified against arXiv/dblp this time—it is our own paper and not within the scope of "verification of three competing papers." The workshop status of O08 was directly confirmed from the arXiv API's comment field.
2.  C02 is marked as "ICML 2026 main conference" rather than "preprint," based on three independent pieces of evidence (PMLR 306 camera-ready footer, author's homepage acceptance announcement, GaTech Spotlight), but dblp only has a CoRR entry as of 2026-09. This discrepancy has been doubly noted in the C02 report and this table.
3.  The comparability of "maximum experimental scale" in the table is limited: the scale metrics for the five papers have different meanings (number of states / number of nodes / number of molecule pairs), so horizontal sorting is not advisable; they are only for judging "what order of magnitude each works at."
4.  The risk ratings (low/medium/high) in §4 are judgments made in this report, not statements from any original text. The rating method (overlap of settings × overlap of selling points × publication venue) has been described at the beginning of that section.
5.  It is recommended to recheck whether O07/O08 have been upgraded to main conference status after NeurIPS 2026 results are released (2026-09-24), and to recheck the official page numbers for C02 in PMLR 306.

# Chapter 8: 2025–2026 Trends

Two separate arXiv system scans were conducted (106 papers on the GFlowNet side, 93 papers on the OT side, from 2025-01 to 2026-09). The retrieval criteria and individual paper cards can be found in `reports/TRENDS_GFN_2026.md` and `reports/TRENDS_OT_2026.md`. The candidate inclusion lists are in `data/candidates_gfn.csv` (38 entries) and `data/candidates_ot.csv` (49 entries). This chapter only includes trend judgments and their implications for the main research lines. Publication status is based solely on the arXiv `comment` field; empty fields are uniformly treated as preprints.

## 8.1 GFlowNet Side

#### 3. Trend Assessment

Each trend is supported by arXiv papers; a trend is not considered if it's supported by only one paper.

1.  **Non-acyclic GFlowNets transition from "theoretical fixes" to "generative tools," but remain at Workshops.** Within a year after T36 (2502.07735, ICML 2025 main conference), the same HSE team extended non-acyclic theory to shortest paths (2603.01786), optimal transport (2606.06272), and MCMC termination (2606.16073). The Brunswic group extended it to continuous/ergodic settings (2505.03561, ICML 2025 main conference) and multi-agent systems (2509.20408). All three application outlets landed in the SPIGM Workshop, with only two theoretical papers at main conferences. Our assessment: The theory for this line has been accepted by main conferences, but the application narrative has not yet.

2.  **Error certificates become a new point of competition.** Stable GFlowNets (2605.01729) proves that low TV does not preclude unbounded loss and provides a reverse bound for loss→TV; Evaluation Balance (2603.01047, ICLR 2026) uses flow balance as a policy evaluator; Secrets (2505.02035) discusses sample complexity and implicit regularization. The old problem of "low loss does not equal low distribution error" is being systematically addressed. The most important implication for this repository: The gap for extending residual bounds from TV to OT cost gap still exists, but GFN-side tools are already mature.

3.  **Training objectives enter the era of "families."** \(f\)-TB (2605.15417, ICML 2026), \(\alpha\)-GFN (2602.01749), Divergent TB (2602.17827), Hybrid-Balance (2510.04792, NeurIPS 2025), RapTB (2603.00454), Evaluation Balance (2603.01047): Objectives are no longer a choice among FM/DB/TB, but rather adjustable gradient geometries under the same minimum point. O08's minimum flow objective can be superimposed on any of these families, which is an axis to sweep during experiments.

4.  **Policy gradients flow back into GFlowNets.** PPO for amortized discrete sampling (2606.15793), information-geometric natural gradients (2608.03967), RTB ≡ Trust-PCL (2509.01632), PowerFlow (2603.18363, ICML 2026), GFlowRL (2607.13394): The equivalence between GFN and KL-regularized RL is repeatedly used to borrow RL optimizers. Implication: The RL counterpart of entropy-regularized GFN–OT most likely already exists in the KL-regularized RL literature.

5.  **Two fates for \(Z\).** In LLM scenarios, \(Z\) is either eliminated (Stable-GFN pairwise comparison, 2605.00553, ICML 2026 Spotlight; GFlowRL points out that prompt-conditioned \(Z\) is a source of instability, 2607.13394) or reused (\(Z\) as a difficulty scheduler, 2602.12642). In O08's LP setting, \(Z\) does not appear—this main line abandons GFlowNet's signature ability of "only needing unnormalized rewards," see O08 report §7.6.

6.  **Path space control and Schrödinger language are unifying GFN and samplers.** Sampling Decisions (2503.14549) unifies GFlowNet flow functions, Doob \(h\)-transforms, and one-sided Schrödinger transport into the same object; Berner et al. (2501.06148, TMLR) provide asymptotic equivalence between discrete and continuous time. The theoretical foundation for entropy-regularized GFN–OT has been half-built by others, thus increasing the risk of collision.

7.  **Main conference proportion and types.** Approximately 30 main conference/journal acceptances can be determined from comments: ICML 2025 ×4, ICML 2026 ×5, ICLR 2025/2026 ×3, NeurIPS 2025 ×2, AAAI 2026 ×1, AISTATS 2025 ×1, TMLR ×2, EMNLP 2025, KDD 2026, SIGMOD 2027, ACM MM 2025. Application-oriented papers outnumber theoretical ones; there are zero main conference papers directly related to OT (O07/O08 are both Workshop papers).

8.  **A second set of infrastructure emerges at the tool layer.** gfnx (JAX, 2511.16592) stands alongside torchgfn, and GFlowState (2604.21830) adds visualization. There are now two independent paths for reproducing experiments.

#### 4. Implications for GFlowNet × OT Direction

Comparing with the four candidate topics in `reports/COMPETITOR_MATRIX.md`:

| Topic | Window Status | Basis |
|---|---|---|
| Balance residual → OT error bound (+ dual potential certificate) | **Open, and tools are ready** | GFN side has loss→TV bounds (2605.01729) and balance-as-evaluator (2603.01047) directly citable; OT side has LP duality and complementary slackness ready in O08 Thm 3.3; no one has yet connected the two ends |
| Conditional GFN learns a family of OT on graphs | Closing | ULOT (this repository C01, NeurIPS 2025) and UNOT (O05, ICML 2025) have occupied the amortization axis; no new conditioning tools have appeared on the GFN side |
| Entropy-regularized GFN–OT / Schrödinger bridge | Closing | Sampling Decisions (2503.14549) has unified GFN flow functions and one-sided Schrödinger transport; GSBoG (C02, ICML 2026) and DDSBM (C03, ICLR 2025) occupy SB on graphs; α-DSBM (O04) provides an isomorphic projection algorithm |
| GFN proposal + classical OT correction | Demoted to baseline | Unrealized Expectations (2502.03669, TMLR) warns: classical solvers on explicit graphs are brutal baselines; GFN can only establish itself on implicit graphs, and there are no classical OT solvers to "correct" on implicit graphs |

Two newly opened small windows: (a) **Minimum flow objective + policy gradient/PPO trainer** (2606.15793 is the next step from the same team; first to do it gets to occupy it); (b) **Symmetry and internal flow degrees of freedom** (symmetric correction in 2506.02685 and \(P_B\) degrees of freedom in T02 are two sides of the same phenomenon; no one has yet written about it from an OT perspective).

## 8.2 Optimal Transport Side

#### 3. Trend Assessment

1.  **The theoretical focus of OT on graphs is on "potentials" rather than "plans."** Dual prediction accelerated minimum cost flow (2601.20203, AAAI 2026), statistical rates of Sinkhorn potentials (2608.29152), PL inequality for QOT duality (2605.27175, SIAM J. Optim.), Brenier potential estimation (2604.22366) – the common object of these four papers is dual variables. Implication: The state flow \(F(s)\) learned by GFlowNet in O08 is the primal variable counterpart of dual potentials. Connecting to this "potential theory" is more natural than connecting to "plan prediction."

2.  **Discrete SB has completed a closed loop from method to benchmark.** After DDSBM (C03, ICLR 2025) and GSBoG (C02, ICML 2026), discrete SB benchmarks with analytical solutions (2509.23348) and convergence rate results (2607.19176) emerged. When a direction has benchmarks and convergence rates, it means new entrants must report their numbers according to this standard – entropy-regularized GFN–OT no longer has the "define the problem first" advantage.

3.  **Amortized OT continues to increase in crowding.** After UNOT (O05) and ULOT (C01), sliced potential amortization (2604.15114), min-sliced transferable plans (2511.19741), and HyperTransport (2605.08254) appeared consecutively from 2025-11 to 2026-05. If conditional GFN–OT is to establish itself, its only remaining moat is "implicit graphs, inability to instantiate cost matrices."

4.  **Quadratic regularization is the only regularization direction not yet saturated.** Entropy regularization has a whole ecosystem of SB, while quadratic regularization (the choice in O02) is only being explored for optimization theory by the González-Sanz–Nutz group (2605.27175 / 2605.27883), and all in continuous/semi-discrete settings. Quadratic regularized flow on graphs + GFlowNet parameterization is currently unexplored.

5.  **"Which one to choose when not unique" has independent mathematical literature.** When the OT plan is not unique under a distance cost, the entropy selection principle (2512.05282, 2502.16370) studies which plan is selected in the small regularization limit. O08 uses the minimum total flow as the selection principle, which is parallel to this literature but without cross-citation; connecting the two would be the scope of a short paper.

6.  **Neural OT on continuous graphs (metric graphs) has just appeared.** 2606.16273 is the first paper, using embedding + semi-dual + projection; it does not provide edge-by-edge policies, complementing rather than replacing O08's discrete, executable policies.

7.  **Main conference proportion.** Among 93 papers, 12 can be identified as main conference/journal papers based on comments: NeurIPS 2025 ×2 (including Spotlight, Oral), AAAI 2026 ×2, ICML 2026 ×1, ICLR 2026 ×1, ICDM 2026, WACV 2026, SIAM J. Optim., L-CSS, VLDBJ, Globecom 2026. Theoretical work on OT on graphs primarily exists in preprint form.

#### 4. Implications for GFlowNet × OT Direction

**Directly usable techniques**

| Technique | Source | Where to use |
|---|---|---|
| Dual prediction → \(\varepsilon\)-relaxation refinement, including bounds from prediction error to runtime | 2601.20203 (AAAI 2026) | Treat GFlowNet-learned state flow as dual prediction, perform post-processing with guarantees; replace the vague statement of "GFN proposal + classical correction" |
| Statistical rates of Sinkhorn potentials and residual stability conditions | 2608.29152 | The statistical side of "dual potentials as certificates"; combine with O08 Thm 3.3's deterministic duality for a complete error decomposition |
| Local error bounds and PL inequality for QOT duality | 2605.27175 | Convergence rate proof template for quadratic regularized flow on graphs (O02 route) |
| Discrete SB benchmark with analytical solutions | 2509.23348 | A mandatory benchmark for any entropy-regularized GFN–OT experiment |
| Proof of Sinkhorn exponential convergence for Regime-switching SB | 2607.19176 | IPF convergence technique on discrete components |
| Hölder continuity of Beckmann solution with respect to parameters | 2603.19755 | The type of proposition needed for conditional GFN–OT generalization guarantees (continuous version) |
| Entropy selection principle | 2512.05282, 2502.16370 | Compare "minimum total flow" with "small regularization limit" as selection principles |

**Already occupied selling points**

-   "One model serving a family of source-target distributions": UNOT, ULOT, sliced potential amortization, min-sliced plans, four or more papers.
-   "Schrödinger bridges / entropy-regularized transport on graphs, outputting executable policies": GSBoG (ICML 2026 main conference) has used almost identical phrasing.
-   "Benchmarks and algorithms for discrete space SB": 2509.23348 has set the standard.

**Conclusion**: The OT-side scan has further advanced the rating in `COMPETITOR_MATRIX.md` – Topic 1 (residuals → error bounds + dual certificates) not only has the lowest risk of collision, but the OT side also happens to provide all the necessary components; the windows for Topics 2 and 3 are narrower than judged in the old survey of 2026-08.

#### 5. Strong Baselines for GFN–OT Experiments

| Method | Type | arXiv / Source | Code (according to comment/abstract) |
|---|---|---|---|
| Network simplex / \(\varepsilon\)-relaxation exact min-cost flow | Exact solver | Classical; dual prediction accelerated version 2601.20203 | Unknown |
| Sinkhorn (entropy-regularized OT, including GPU version cuRegOT) | Regularized solver | 2605.08793 | Unknown |
| Quadratic regularized graph OT (Essid & Solomon) | Regularized solver | This repository O02 | Unknown |
| UNOT | Amortized neural OT | This repository O05 (ICML 2025) | Yes (see O05 report) |
| ULOT | Inter-graph amortized OT plan | This repository C01 (NeurIPS 2025) | Yes |
| GSBoG | Generalized SB on graphs, edge-by-edge policy | This repository C02 (ICML 2026) | Not public |
| DDSBM / DLightSB / \(\alpha\)-CSBM | Discrete SB | C03; 2509.23348 | DDSBM has; others unknown |
| ASBS | SB sampler (continuous) | 2506.22565 (NeurIPS 2025 Oral) | Unknown |
| min-flow GFlowNet (T36 code) | Our approach | This repository T36 / O07 | Yes |

Principles for selecting baselines: For explicit small graphs, an exact solver must be included (otherwise, "learning OT" cannot be verified); for implicit large graphs, GSBoG or its reproduction must be included (the only competitor in the same quadrant); if entropy regularization is added, the analytical solution benchmark from 2509.23348 must be reported.

## 8.3 Our assessment: Combining both sides

-   The GFN side's "residuals as metrics" (Stable GFlowNets, Evaluation Balance) and the OT side's "error theory based on dual potentials" (dual prediction accelerated minimum cost flow, Sinkhorn potential statistical rates, PL inequality for QOT) both matured in 2026, yet no paper has connected the two. This is the window for Topic ①.
-   All three application outlets for non-acyclic GFlowNets (shortest path, OT, MCMC termination) are stuck at workshops; GSBoG in the same quadrant is at a main conference. The gap in narrative positioning is larger than the technical gap.
-   Both amortized OT and discrete SB directions have entered the "with benchmarks, with convergence rates" stage, meaning new entrants no longer have the advantage of defining the problem.

## 8.4 In-depth Analysis of Six Supplemental 2026 Papers

The six papers with relevance=5 from the trend scan have been downloaded, translated, and provided with in-depth analyses. They are numbered N01–N06 (see `reports/N0*.md`, "2026 Supplement" section in README):

| ID | Paper | Publication | Role in Main Thread |
|---|---|---|---|
| N01 | Minimum-Cost Network Flow with Dual Predictions | AAAI 2026 | The correct form for Topic ④: GFN state flow feeds dual predictions into ε-relaxation, error → time-bounded (Thm. 2) |
| N02 | Stop the Sampler! | ICML 2026 SPIGM Workshop | The third exit for "expected length = total flow" (MCMC termination); provides closed-form optimal value for minimum flow in continuous space |
| N03 | Stable GFlowNets with TV Monitoring | Preprint | Half of Topic ① completed: TB residual → TV bound (Thm. 3.5) + sampling probability certificate (Thm. 3.6) |
| N04 | Generative Modeling on Metric Graphs | Preprint | Continuous edge branch of OT on graphs, complementary to O08; evaluation protocol including "noise floor" is worth adopting |
| N05 | Orlicz-Sobolev Unbalanced Graph OT | NeurIPS 2025 Spotlight | Relaxes O08's \(\sum L=\sum R=1\) for non-KL candidates; mass difference absorbed by linear term (Thm. 4.2) |
| N06 | Discrete SB / EOT Benchmark | ICLR 2026 | Essential benchmark for the entropy-regularized path; its construction recipe can be transplanted to an analytical benchmark for GFN–OT |


# Chapter 9: Insights, Candidate Topic Ratings, and Decisive Experiments

This chapter includes the full content of the cross-paper synthesis document `reports/INSIGHTS.md`: a one-page conclusion, the main logical chain, an OT comparison table, competitive landscape, ratings for the four topics, immediately executable decisive experiments, and eight easily misinterpreted points and open questions.


### 1. One-Page Conclusion

1. **The internal flow of a GFlowNet is never unique, while the reward only fixes the boundary.** The terminating flow is determined by \(R\), and \(P_B\) on non-terminating edges is free (T02 Prop. 18, item 3; §2.6); on cyclic graphs, an additional degree of freedom is added by the cycle space \(H^1_+(G)\) (T19 Prop. 5). This is the starting point of the entire main thread.
2. **Minimum total flow is a legitimate selection principle and has precise behavioral implications.** On finite discrete non-acyclic graphs, \(\sum_{s}F(s)=Z\cdot\mathbb E[n_\tau]\) (identity near T36 Prop. 3.6, tightening T19 Thm. 2's "≤" to "="), so "minimum total flow" = "minimum expected trajectory length".
3. **Shortest expected trajectory ⇔ only taking shortest paths.** In the single-source case, the expected length is minimized if and only if the policy places all mass on the shortest path (O07 Thm. 3.4, necessary and sufficient).
4. **After fixing the source distribution, minimum total flow = Kantorovich OT under graph shortest path costs.** \(\mathrm{GFlow}^\star=\mathrm{OT}^\star\), and the endpoint distribution of trajectories sampled by the optimal policy is the optimal coupling (O08 Thm. 3.2); the dual potential on terminating states equals the shortest path distance, and complementary slackness restricts the optimal flow to the shortest path subgraph (O08 Thm. 3.3).
5. **The equivalence is classic, the interface is new.** Shortest-path OT on graphs ≡ min-cost flow ≡ discrete Beckmann (O02 Eq. (1)⇔(3); O01 Prop. 6.23, Prop. 14.9). O08's contribution is to translate it into GFlowNet language, where the output is not a coupling matrix but a local routing policy executable on implicit graphs of size \(20!\) (O08 §4.2).
6. **In the standard single-source DAG setting, "GFlowNet learns an OT plan" is an empty statement.** When the source marginal is \(\delta_{s_0}\), the coupling set is a singleton (O01 Remark 3.2). Non-trivial OT structure requires simultaneously relaxing the initial flow distribution and adding an exogenous criterion, both of which are indispensable.
7. **The main thread abandons GFlowNet's signature capability.** O08 Assumption 3.1 requires \(\sum L=\sum R=1\), \(Z\) is known; "only unnormalized rewards needed" does not hold on this line (O08 §7.6).
8. **Competitive landscape: Our two papers are Workshop papers, while three competitors are Main Conference papers.** ULOT (NeurIPS 2025), GSBoG (ICML 2026), DDSBM (ICLR 2025); GSBoG is in the same quadrant as O08, with almost identical selling points, and is four orders of magnitude larger in scale (COMPETITOR_MATRIX §0–§3).
9. **The least crowded subsequent topic is error certificates, and the components are already in place.** On the GFN side, N03 (`reports/N03_2605.01729.md`) has completed "per-trajectory TB loss \(\le c^2\) ⇒ \(\mathrm{TV}\le1-e^{-2c}\)" (Thm. 3.5) and "sampling probability certificate, independent of state space size" (Thm. 3.6); on the OT side, N01 (AAAI 2026) provides a bound for "dual prediction error \(\|\hat p-p^\star\|_\infty\) → runtime" (Thm. 2), and N06 (ICLR 2026) provides a recipe for constructing benchmarks with analytical solutions (Thm. 3.1). The only missing piece is to change the right-hand side of N03 from TV to OT cost gap and marginal violation, with the bridge being O08 Thm. 3.3's dual potential and complementary slackness.
10. **Two topics are already crowded:** Conditional/amortized GFN–OT (UNOT, ULOT, sliced potential amortization), entropy-regularized GFN–OT / SB on graphs (GSBoG, DDSBM, Sampling Decisions has written GFN flow functions as one-sided Schrödinger transport).

### 2. Main Logical Chain: What's Missing in Each Link, What the Next Link Adds

| Link | Paper | What This Link Established | Gaps It Left |
|---|---|---|---|
| ① | T00 / T02 | Flow conservation + terminating flow = reward \(\Rightarrow P_T\propto R\) (T02 core correctness theorem); Markovian flow uniquely determined by terminating flow and \(P_B\) (Prop. 18, item 3); feasible set is linear (Prop. 19 / Eq. (22)) | Only on DAGs; \(P_B\)'s degrees of freedom treated as an "engineering choice," with no principle for selection. The only mention of "shortest" in the original text is a single sentence in §2.6: "can prefer shorter paths." |
| ② | T03 / T05 / T10 | TB moves constraints to the entire trajectory, \(\log Z\) becomes a learnable parameter; SubTB(\(\lambda\)) interpolates between DB and TB; T10 diagnoses training difficulties | All on DAGs, all about "how to reach a valid flow faster," not "which one to reach." |
| ③ | T19 | With cycles, flow is a measure, trajectory set is infinite; 0-flow characterizes cycles; ratio-type loss pushes flow into cycles (Thm. 3), difference-type loss is stable (Thm. 4); R-flow = an acyclic flow + cycle space (Prop. 5); \(\mathbb E(\tau)\le F_{\mathrm{out}}(S^*)/R(S^*)\) (Thm. 2) | "Limit flow is acyclic" does not mean "solution is unique"—even on DAGs where cycle space is zero, R-flow is still not unique (T19 report editor's note); only an inequality between total flow and length. |
| ④ | T36 | Takes \(P_B\) as the primary object; flow = expected visit count × terminating flow, one-to-one correspondence with \((P_B,F(s_f))\) (Prop. 3.7); \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) (equality); unique solution when \(P_B\) is fixed (Cor. 3.11); formulates "minimum total flow" constrained optimization Eq. (11) and approximate solution using DB + state flow regularization | Minimum falls on the boundary of the polyhedron, where parts of \(P_B\) are zero, exceeding T36's own positivity assumption, can only be approximated by \(\lambda\) regularization; does not state what the minimum flow "is." |
| ⑤ | O07 | Minimum expected length \(\Leftrightarrow\) only shortest paths are taken (Thm. 3.4, iff); replaces Assumption 3.1 \(\mathbb E[n_\tau]<\infty\) with \(P_B>0\); reduces pathfinding to training a non-acyclic GFN with flow regularization | Single source; \(R\equiv1\) or single objective; only answers "path" not "how to distribute quality." |
| ⑥ | O08 | Adds another constraint: first-step edge flow = source distribution \(L\). Objective bilinearized into LP (Appendix A.1), reduced to divergence constraint \(\operatorname{div}\mathcal F=L-R\) (Eq. (11)); two-sided squeeze yields \(\mathrm{GFlow}^\star=\mathrm{OT}^\star\) (Thm. 3.2); duality + complementary slackness (Thm. 3.3) | No bound between neural training and LP optimality; \(Z\) must be known; no unbalanced; hypergrid \(H\le20\), permutation \(n\le20\), \(n=20\) no \(\mathrm{OT}^\star\) reference. |

**Our assessment:** The true "turning point" in this chain is the equality \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) in link ④. It ties a linear functional in flow space (total flow) to a behavioral quantity (expected number of steps). Links ⑤ and ⑥ are then just two exercises in linear programming, minimizing this linear functional under different constraints. Understanding ④ makes ⑤ and ⑥ straightforward; without understanding ④, one might misinterpret O08 as "GFlowNet miraculously learned OT."

### 3. OT Side Comparison: What is Equivalent, What is Merely Analogous

Extracted from the 16-row comparison table in O01 report §7 (lecture chapter numbers according to arXiv v3):

| OT Concept | GFlowNet Concept | Relationship | Source |
|---|---|---|---|
| Directed edge flow \(m_e\), objective \(\sum_e\ell_e\lvert m_e\rvert\) | Edge flow \(F(s\to t)\) and total flow | **Strictly equivalent** | O01 Def. 6.20, Prop. 6.23; O02 Eq. (1)⇔(3) |
| Conservation law \(\operatorname{div}_G m=r\) | Flow matching (internal vertices \(r_i=0\)) | **Strictly equivalent** | O01 Prop. 6.23 |
| Cost = graph shortest path \(d_G\) | Trajectory length / expected number of steps | **Conditionally equivalent** (path action sums edge lengths) | O01 Prop. 14.9; O08 Eq. (7)–(8) |
| Marginal constraints \(\pi_1=\alpha,\pi_2=\beta\) | Reward matching \(F(x\to s_f)=R(x)\) + first-step edge flow = \(L\) | **Conditionally equivalent** (\(R\) must be normalized, \(Z=1\)) | O08 Assumption 3.1 |
| Complementary slackness: support ⊂ contact set | Optimal flow is positive only on tight edges | **Strict** (in LP form) | O01 Prop. 5.3; O08 Thm. 3.3 |
| Kantorovich dual potentials \(f_i\) | State flow \(\log F(s)\) / value function | **Analogous** (OT has inequality constraints, DB has equality) | O01 Def. 5.1, Prop. 6.23 |
| Coupling \(\mathbf P\in\mathcal U(\mathbf a,\mathbf b)\) | Endpoint marginals of trajectory distribution | **Analogous**, degenerates to a single point for single source | O01 Remark 3.2 |
| Entropy regularization \(\varepsilon\mathrm{KL}(\pi\|\alpha\otimes\beta)\) | Path entropy of MaxEnt GFN | **Analogous**, differs by a conditional term (endpoint entropy vs. path entropy) | O01 Def. 8.2, Eq. (8.8) |
| Sinkhorn = alternating KL projection | Alternating constraint enforcement training | **Analogous** (OT has closed-form half-steps, GFN only SGD) | O01 Prop. 9.4/9.5 |
| Schrödinger bridge reference path law | Fixed \(P_B\) | **Conditionally equivalent** | O01 Prop. 14.11; O04 \(\mathrm{proj}_{\mathcal R}/\mathrm{proj}_{\mathcal M}\) isomorphism table |
| Benamou–Brenier / JKO | "Flow × velocity" / training dynamics | **Only atmospherically analogous**, not to be used as a theorem | O01 Thm. 14.5, Def. 15.1 |

### 4. Competitive Landscape

Three dividing lines determine who is truly competing with whom (COMPETITOR_MATRIX §2):

- **Whether the graph is the "object being transported" or the "locus of transport."** ULOT (C01) and DDSBM (C03) belong to the former: transport occurs between graphs. O08, O07, and GSBoG (C02) belong to the latter: transport occurs within a single graph. Comparing numbers across these lines is meaningless.
- **Explicit enumeration versus implicit expansion.** GSBoG scales to \(10^6\) nodes but requires \(\mu,\nu\) to be explicit vectors, neighborhood enumeration, and fully known topology; O08's permutation environment of \(20!\) states has no explicit representation. This is the only structural advantage of GFNs that cannot be replicated.
- **Temperature.** O08/O07 operate at \(\varepsilon=0\) (pure LP, vertex solutions, with dual certificates); GSBoG/DDSBM operate at \(\varepsilon>0\) (exponential family solutions, with IPF/IMF convergence mechanisms, no error bounds). Adding a KL term to O08 would move it into the quadrant already occupied by GSBoG.

Specific threats (absorbing conclusions from O04/O05 reports):

| Source | What it threatens | Intensity | Counterpoints |
|---|---|---|---|
| **GSBoG** (C02, ICML 2026 Main) | O08's entire positioning narrative: "not static coupling but executable policy" is almost verbatim | Highest | Implicit graphs; indefinite time horizon (GSBoG requires fixed \(T\)); O08 has LP dual certificates while GSBoG has no error bounds; GSBoG is not open-source |
| **UNOT** (O05, ICML 2025) | Amortization axis: cross-dataset/cross-resolution generalization, non-training set relative error 1.3–2.8% (Table 4); its amortized dual potentials are the continuous version of \(\pi_x\) in O08 Thm. 3.3 | High | Changing graphs means changing costs (UNOT §6 states it cannot do this); \(\lvert\mathcal S\rvert\) is too large to store the cost matrix |
| **ULOT** (C01, NeurIPS 2025) | "Conditional GFNs learning a family of graph OT" topic | High | ULOT hard limit \(n\le10^4\), outputs plan not policy, zero theorems in the entire paper, cost is quadratic FUGW |
| **α-DSBM** (O04, NeurIPS 2024 Spotlight) | Not a threat, but complementary: \(\mathrm{proj}_{\mathcal R}/\mathrm{proj}_{\mathcal M}\) is isomorphic to "fixed \(P_B\)"/"inferring \(P_F\) from edge flow" respectively, Eq. (26) + Lemma D.2 is a ready-made proof template for "SB on graphs = entropy-regularized min-flow GFN" | — | Use it, don't compete with it |
| **DDSBM** (C03, ICLR 2025) | "Entropy-regularized GFN–OT for molecular editing" application niche | Medium | DDSBM needs to solve QAP graph matching, GFNs don't need to follow paths; DDSBM does not condition |

### 5. Final Rating of Four Candidate Topics

| Topic | Rating | Rationale (including updates after 2026 trend scan) |
|---|---|---|
| **① Balance residual → OT error bound, dual potentials as certificates** | **Do it. Low collision risk, complete components (see N01/N03/N06)** | GFN side: Stable GFlowNets already have TB residual → TV bound (2605.01729), Evaluation Balance uses residual as an evaluator (2603.01047, ICLR 2026). OT side: error theory is all about potentials—dual prediction → \(\varepsilon\)-relaxation runtime bounds (2601.20203, AAAI 2026), Sinkhorn potential statistical rates (2608.29152, O08 collaborator Belomestny), QOT's PL inequality (2605.27175). **No paper maps the residual of an approximately feasible flow to the cost gap.** O08 Thm. 3.3's complementary slackness provides an edge-wise certificate \(\mathcal F(s\to s')(\pi_{s'}-1-\pi_s)=0\), directly usable |
| ② Conditional GFN amortizes a family of graph OTs | Don't do it (demoted to baseline) | After UNOT and ULOT, sliced potential amortization (2604.15114) and min-sliced plans (2511.19741) have emerged; the only moat is implicit graphs, but how "a family" is defined on implicit graphs has not yet been clearly articulated |
| ③ Entropy-regularized GFN–OT / Schrödinger bridges on graphs | Don't do it (as an \(\varepsilon>0\) generalization of ①) | GSBoG occupies the main conference slot in the same quadrant; Sampling Decisions (2503.14549) has already written GFN flow functions as one-sided Schrödinger transport; discrete SB has analytical solution benchmarks (2509.23348) and convergence rates (2607.19176). New entrants lack the benefit of problem definition |
| ④ GFN proposal + classical OT correction | Change form | The vague "correction" has been concretized by 2601.20203 as "dual prediction → \(\varepsilon\)-relaxation, prediction error mapped to runtime." Feasible form: use the state flow learned by GFlowNet as a dual prediction to feed into a classical solver, reporting guaranteed speedups on explicit graphs—but this is an application of ①, not an independent topic |

**Our assessment:** If only one thing is to be done, do ①; ②, ③, and ④ should all appear as baselines or generalizations in the experimental design of ①, rather than as independent papers.

### 6. An Immediately Executable Decisive Experiment

Objective: To verify that "balance residuals can certify OT error" and to provide an honest positioning of GFN-OT relative to classical and neural OT. All environments and baselines have existing code or exact solutions.

**Environments** (three tiers, from enumerable to implicit):

1. Hypergrid \(\{0,\dots,H-1\}^D\), \(H\in\{10,15,20\}\), \(D=2\), allowing \(\pm1\) transitions (with cycles). Source distribution \(L\) is Ball/Moon from O08, target \(R\) is a multimodal distribution in the corners and normalized—exact LP solutions can be obtained from `scipy.linprog` (as in O08 Figure 1), and \(\mathrm{OT}^\star\) can be obtained from network simplex.
2. Cayley graph of the symmetric group \(S_n\) (adjacent transpositions), \(n\in\{4,8\}\) (with exact solutions), \(n=20\) (no exact solution, only certification).
3. A weighted variant: Randomly assign weights \(\{1,2\}\) to the edges of the hypergrid, testing whether certificates still hold beyond O08's unit edge length assumption.

**Method Groups**: min-flow GFN (T36 code, DB + state flow regularization, \(\lambda\in\{10^{-1},10^{-2},10^{-3}\}\)); the same model with PPO trainer (2606.15793); the same model with \(f\)-TB loss (2605.15417).

**Baselines**: network simplex (exact); Sinkhorn (\(\varepsilon\in\{0.1,0.01\}\), including analytical solution verification from 2509.23348); ported UNOT (using shortest path matrix as cost, potential predictor + one-step Sinkhorn, explicit tier only); GSBoG reproduction (if code is still unavailable, use discretization of \(\alpha\)-DSBM as a substitute and note it).

**Metrics** (per tier, per method, 5 seeds):

- Marginal error: Terminal distribution TV (exact enumeration for enumerable tiers, T36's \(C(k)L^1\) diagnostic for implicit tiers); source marginal violation \(\lVert \hat L-L\rVert_1\).
- Transport cost: Relative difference between \(\mathbb E\lvert\tau\rvert\) and \(\mathrm{OT}^\star\); coupling level \(\lVert\Pi_\theta-\Pi^\star\rVert_1\) (not reported in O08).
- **Certificate**: Construct dual feasible \(\pi\) (dual potentials) from learned state flows, report primal–dual gap; then report the mean and maximum of the balance residual \(\sum_t\log\)-ratio for each trajectory. Core plot: \(x\)-axis residual, \(y\)-axis cost gap, check if it falls below a provable bound.
- Cost: Wall-clock time, memory, number of energy function calls.

**Expected Results**: Exact solvers should outperform in all metrics and costs on small explicit graphs (if not, the experiment has a bug); GFN should be the only method capable of providing certificates in the implicit tier; the residual-gap curve should be monotonic and bounded.

**Failure Criteria**: If the primal–dual gap and balance residual are uncorrelated (Spearman \(<0.5\)) for \(n=20\), then "residual certifies cost gap" is invalid, and research question ① should shift to certifying only marginal error; if the certificate fails on the weighted variant, O08's unit edge length assumption is not technical but essential.

### 7. Eight Most Easily Misinterpreted Points

1. **"\(P_B\) can be freely chosen" \(\neq\) "DB is an empty constraint."** Once \(\hat P_B(s\mid s')=\hat P_F(s'\mid s)\hat F(s)/\hat F(s')\) is required, normalization \(\sum_s\hat P_B=1\) immediately implies flow matching (T02 Eq. (31)). Freedom exists on the manifold of "normalizing to the parent set at each state," and only holds in deterministic environments (Counterexample 50).
2. **T19's "limit flow is acyclic" \(\neq\) "solution is unique."** T19 eliminates infinitely scalable directed cycles; what truly selects a point is T36 Eq. (11)'s linear objective and O02 Cor. 1's \(\ell^2\) tie-breaking.
3. **O08's \(\mathrm{GFlow}^\star\) is the reduced objective \(\sum_{E^\circ}\mathcal F\), not \(\sum_{s\in\mathcal I}\mathcal F(s)=\mathbb E[n_\tau]\); the two differ by 1** (the \(s_0\to u\) step is not part of the transport segment). In Table 1, \(\mathbb E\lvert\tau\rvert\) and \(\mathrm{OT}^\star\) are directly aligned, indicating that the reported value is the length of the transport segment.
4. **"Learning an OT plan" on a single-source DAG is meaningless** (O01 Remark 3.2). The first non-trivial case is \(2\times2\), with \((n-1)(m-1)\) degrees of freedom.
5. **O04's \(\alpha\) is not a learning rate, but an interpolation coefficient in the coupling space** (Eq. (22)), and the paper does not prove that \(\alpha<1\) is better—Appendix B states that \(\alpha=1\) is optimal when the single-step cost is constant.
6. **O06's "flow neural network" is the velocity field of a continuous normalizing flow, not a GFlowNet.**
7. **T02's local PDF is an expanded version of arXiv v5 (76 pages), and proposition numbering may not match the JMLR 2023 published version (24(210):1–55)**; T03 cites Foundations using earlier numbering (Prop. 3 / Cor. 1 / Prop. 6 / Prop. 10).
8. **O07/O08 are from the ICML 2026 SPIGM Workshop, not the main conference; GSBoG is from the ICML 2026 main conference.** Any statement equating the two in terms of prestige is incorrect; our side is at a disadvantage in "narrative positioning," which is a fact that must be faced when choosing the topic.

### 8. Open Questions

1.  **What does the bound of the residual → cost gap look like?** Bounds for TB residual → TV already exist (N03 Thm. 3.5/3.6; note that its DB/FM branches depend on the maximum trajectory length \(L\), which is invalid in O08's non-acyclic setting, so only the TB branch is transferable); OT cost is a linear functional of the coupling, and the relationship between marginal violation and cost gap should be tighter than the TV bound. Source: O08 §7.2.
2.  **Weighted graphs.** O08 only covers unit edge lengths; when weighted, zero-weight edges can create zero-cost cycles. Does the minimum flow principle still select acyclic solutions? Source: O07 report §7, O08 §5 item 5.
3.  **Gap between neural training and LP optimum.** O08 approximates constraints with soft penalty \(\lambda\). Table 2 shows a quantitative trade-off for \(\lambda\), but there is no theory. Source: O08 §5 item 7.
4.  **Unbalanced and unknown \(Z\).** After relaxing \(\sum L=\sum R\), can GFlowNet's ability to "only need unnormalized rewards" return? Unbalanced transport on Orlicz–Sobolev graphs (TRENDS_OT §2.1) is a candidate target. Source: O08 §7.6.
5.  **What functional corresponds to the minimum total flow in continuous state spaces?** Berner et al. provide discrete ↔ continuous equivalence tools (TRENDS_GFN §2.3), and Ergodic Generative Flows provide a continuous non-acyclic framework, but no one has written a continuous version of Thm. 3.2.
6.  **Are implicit regularization and minimum flow consistent?** When no flow regularization is added, to which internal flow does training converge (Secrets of GFlowNets' Learning Behavior, 2505.02035; Fisher geometry, 2608.03967)?
7.  **Relationship between path multiplicity induced by symmetry and internal flow degrees of freedom.** Symmetry-Aware GFlowNets (2506.02685) and T02's \(P_B\) degrees of freedom are two sides of the same phenomenon, and there is no OT-angle analysis yet.
8.  **Does short flow benefit generalization?** O08 §7.7 points out that this has not been verified and may be negative: shortest path strategies concentrate mass on a few paths, which contradicts GFlowNet's selling point of diversity.
9.  **Relationship between minimum flow and the entropy selection principle.** When the OT plan is not unique under distance cost, is the plan selected by the small regularization limit (TRENDS_OT §2.2) the same as the plan selected by the minimum total flow?
10. **Multi-objective combination.** Routing by Reaching (2602.21565) combines pre-trained GFlowNets during inference; in OT language, this corresponds to multi-marginal transport, for which there is no corresponding theorem yet.

---

*This document was compiled by the awesome_Gflow_OT project team based on 18 interpretation reports and two trend scans; evidence pointers for each judgment can be found in the parentheses, indicating the report number and original theorem number.*

# Chapter 10: Repository Guide and Methodology

## 10.1 Directory Structure

| Path | Content |
|---|---|
| `papers/` | 18 original PDF papers (arXiv version), filenames are arXiv numbers |
| `papers_zh/` | SuperTranslate layout-preserved Chinese translated PDFs (`<arXiv>.zh.pdf`) and object-level QA results (`<arXiv>.inspect.json`) |
| `reports/` | 18 in-depth interpretations (`<ID>_<arXiv>.md`), competitive matrix, two trend scans, INSIGHTS, this summary report and its PDF |
| `data/meta/` | One JSON metadata card per paper (title, authors, venue, venue_type, code, one-sentence Chinese and English summary) |
| `data/papers.yaml`、`data/candidates_*.csv`、`data/scan_*.json` | Seed list, trend candidates, raw arXiv scan |
| `src/generator.py` | Generates `README.md` / `README_zh.md` from `data/meta` and candidate CSVs (data-driven mode of awesome-ml4co) |
| `scripts/` | Download and ID parsing, translation batch processing, trend scanning, UTF-8 validation, report assembly, PDF building |
| `slides/` | Summary slides (single-file HTML and Beamer PDF) |

## 10.2 Interpretation Report Template

Each interpretation strictly follows 8 sections: one-sentence positioning and metadata table → problem setting and notation → core contributions (numbered as in the original text) → methods and derivation highlights → experiments and evidence → assumptions and applicability boundaries → position in the main storyline → reusable insights and open questions → references; "Editorial note" is appended at the end to record self-determined ambiguities. All numbers and theorem numbers include original sources; if not provided in the original text, "Original text not provided" is written.

## 10.3 Translation Pipeline and QA

The translation engine is SuperTranslate (`pdf_zh_translator`): it does not re-layout pages; formulas, figures, and citations are frozen first, then the main text is translated and backfilled according to original coordinates; parameters `--preserve-graphics-text --skip-overflow`, DeepSeek backend. After translation, `inspect` is run for each paper to perform page-by-page object-level comparison, and the number of issues is recorded in Appendix A. Known limitations: `--skip-overflow` will keep untranslated text in English if it doesn't fit, leading to `untranslated_block` in math-intensive sections of the appendix proof pages; font size scaling can trigger `font_size_drift`. All 18 papers have been translated. O01 (Peyré's lecture notes, 480 pages) is the largest, translated separately using the Gemini 2.5 Flash backend on OpenRouter (`scripts/translate_o01_openrouter.sh`), taking about 4 hours, and QA reported 69 issues (mainly font size drift and English remnants in math-intensive sections of the appendix).

## 10.4 Publication Status Discipline

Main conference / journal / workshop / preprint are marked separately; arXiv comments are often delayed, so if OpenReview, dblp, or official proceedings can be verified, the latter takes precedence (e.g., three competitive papers were thus verified as GSBoG being a main conference paper at ICML 2026); the trend scanning section only relies on the comment field and marks each item as "unverified".

## 10.5 How to Contribute

Add a JSON card in `data/meta/`, optionally add an interpretation in `reports/`, then run `python3 src/generator.py` to regenerate README; translate new papers using `scripts/translate_batch.sh <arXiv>`; rebuild this report using `python3 scripts/build_report.py zh && bash scripts/build_pdf.sh zh`.

# Appendix A Paper List and Chinese-English QA

| ID | Title | Venue | Type | File Path | QA Issues |
|---|---|---|---|---|---|
| O08 | Your GFlowNet Secretly Learns an Optimal Transport Plan | ICML 2026 SPIGM Workshop | workshop | `O08_2606.06272.md` | 3 issues |
| O07 | Learning Shortest Paths with Generative Flow Networks | ICML 2026 SPIGM Workshop | workshop | `O07_2603.01786.md` | 0 issues |
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

# Appendix B Notation and Terminology

| Symbol / Term | Meaning | First Seen |
|---|---|---|
| \(G=(\mathcal S,E)\), \(s_0\), \(s_f\) | State graph, source, sink (absorbing state) | T02 |
| \(\mathcal X\), \(U\) | Set of terminal states, set of source states in O08 | T02 / O08 |
| \(F(s\to s')\), \(F(s)\) | Edge flow, state flow (when cyclic = expected visits × terminal flow) | T02 / T36 |
| \(P_F\), \(P_B\) | Forward / backward policy, \(F(s\to s')=F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\) | T02 |
| \(R(x)\), \(Z\) | Reward, partition function \(Z=F(s_0)=\sum_xR(x)\) | T00 |
| FM / DB / TB / SubTB(\(\lambda\)) | Flow Matching / Detailed Balance / Trajectory Balance / Sub-Trajectory Balance training objectives | T00 / T02 / T03 / T05 |
| 0-flow, \(H^1_+(G)\) | Conservative flow with zero terminal flow; cycle space | T19 |
| flow explosion | Ratio-based loss pushes infinite flow into cycles | T19 Thm. 3 |
| \(n_\tau\), \(\mathbb E[n_\tau]\) | Number of steps in a trajectory and its expectation; \(\sum_sF(s)=Z\,\mathbb E[n_\tau]\) | T36 |
| minimum flow GFlowNet | GFlowNet that minimizes total flow under reward matching constraints | T36 Eq. (11) |
| \(L\), \(d_G(u,x)\) | Source distribution; graph shortest path (hop count) distance | O08 |
| \(\Gamma(L,R)\), \(\Pi\) | Coupling set, transport plan | O01 |
| Kantorovich problem / dual potentials | \(\min_\Pi\sum c\,\Pi\); vertex scalars \(\pi\) satisfying \(\pi_{s'}-\pi_s\le1\) | O01 / O08 Thm. 3.3 |
| Beckmann problem / minimum cost flow | OT equivalent form with edge flows as variables and vertex conservation as constraints | O02 / O01 Prop. 6.23 |
| complementary slackness | Optimal flow is positive only on tight edges: \(\mathcal F(s\to s')(\pi_{s'}-1-\pi_s)=0\) | O08 Thm. 3.3 |
| Schrödinger Bridge (SB), IPF / IMF | Entropy-regularized dynamic transport; Iterative Proportional Fitting / Iterative Markov Fitting | O04 / C02 / C03 |
| entropy regularization \(\varepsilon\), Sinkhorn | \(\varepsilon\mathrm{KL}(\Pi\|\alpha\otimes\beta)\) regularization and its alternating scaling algorithm | O01 |
| unbalanced OT | Transport with unequal total mass at both ends, marginals replaced by penalty terms | C01 |
| TV | Total Variation distance, the gold standard for evaluating terminal distribution fidelity in this repository | T10 / O08 |

# References

- **[O08]** Ian Maksimov, Nikita Morozov, Denis Belomestny, Sergey Samsonov. *Your GFlowNet Secretly Learns an Optimal Transport Plan*. ICML 2026 SPIGM Workshop, 2026. arXiv:[2606.06272](https://arxiv.org/abs/2606.06272).
- **[O07]** Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov. *Learning Shortest Paths with Generative Flow Networks*. ICML 2026 SPIGM Workshop, 2026. arXiv:[2603.01786](https://arxiv.org/abs/2603.01786).
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



