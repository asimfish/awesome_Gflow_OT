#!/usr/bin/env python3
"""Build data/candidates_gfn.csv and data/candidates_ot.csv from the local scan JSONs.

Curation (relevance, tags, one-liners, reasons) is hand-written below; venue strings are
derived ONLY from the arXiv `comment` / `journal_ref` fields captured by scripts/scan_trends.py.
Output is pure ASCII (author names are transliterated).
"""
import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ["id", "arxiv", "title", "authors", "venue", "venue_type", "year",
          "code_url", "one_line_en", "tags", "relevance(1-5)", "reason"]

EXTRA_MAP = {"\u0142": "l", "\u0141": "L", "\u00f8": "o", "\u00d8": "O", "\u00e6": "ae",
             "\u00c6": "AE", "\u00df": "ss", "\u0111": "d", "\u0110": "D",
             "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
             "\u201c": '"', "\u201d": '"', "\u00a0": " "}


def ascii_of(s):
    for k, v in EXTRA_MAP.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- venue rules
WS = re.compile(r"workshop", re.I)
VENUE_PAT = [
    (re.compile(r"(NeurIPS'?\s?\d{2,4}|NeurIPS \d{4})", re.I), None),
    (re.compile(r"(ICML \d{4})", re.I), None),
    (re.compile(r"(ICLR \d{4})", re.I), None),
    (re.compile(r"(AISTATS,? \d{4})", re.I), None),
    (re.compile(r"(AAAI \d{4})", re.I), None),
    (re.compile(r"(CVPR \d{4})", re.I), None),
    (re.compile(r"(EMNLP \d{4})", re.I), None),
    (re.compile(r"(SIGIR \d{4})", re.I), None),
    (re.compile(r"(SIGMOD \d{4})", re.I), None),
    (re.compile(r"(KDD \d{4})", re.I), None),
    (re.compile(r"(ICDM \d{4})", re.I), None),
    (re.compile(r"(WACV \d{4})", re.I), None),
    (re.compile(r"(ICASSP \d{4})", re.I), None),
    (re.compile(r"(ICPR[_ ]\d{4})", re.I), None),
    (re.compile(r"(ISBI\)? \d{4})", re.I), None),
    (re.compile(r"(Globecom \d{4})", re.I), None),
    (re.compile(r"(SDM\s?\d{4})", re.I), None),
    (re.compile(r"(TMLR)", re.I), None),
    (re.compile(r"(ACM Multimedia \d{4})", re.I), None),
]


def venue_of(rec, override=None):
    """Return (venue, venue_type, year). Only uses comment / journal_ref."""
    if override:
        return override
    c = ascii_of(rec.get("comment", "") or "")
    j = ascii_of(rec.get("journal_ref", "") or "")
    pub_year = rec["published"][:4]
    blob = (c + " || " + j).strip(" |")
    if not blob:
        return ("arXiv preprint (comment empty, unverified)", "preprint", pub_year)
    hit = None
    for pat, _ in VENUE_PAT:
        m = pat.search(blob)
        if m:
            hit = m.group(1)
            break
    if hit is None:
        if j:
            y = re.search(r"(20\d\d)", j)
            return (j[:70] + " (per arXiv journal_ref)", "journal", y.group(1) if y else pub_year)
        return ("arXiv preprint (comment has no venue, unverified)", "preprint", pub_year)
    y = re.search(r"(20\d\d)", hit)
    year = y.group(1) if y else pub_year
    if len(year) == 2:
        year = "20" + year
    hit = hit.replace("NeurIPS'25", "NeurIPS 2025")
    if WS.search(blob):
        # keep the workshop wording verbatim-ish so it can never be read as main track
        wtag = re.search(r"([A-Za-z0-9 ,.'&/-]{0,60}Workshop[A-Za-z0-9 ,.'&/-]{0,40})", blob)
        w = ascii_of(wtag.group(1)).strip(" ,.") if wtag else hit + " Workshop"
        return (w + " (Workshop, per arXiv comment)", "workshop", year)
    if hit.upper() == "TMLR":
        return ("TMLR (per arXiv comment/journal_ref)", "journal", year)
    return (hit + " (per arXiv comment)", "main", year)


CODE_RE = re.compile(r"https?://(?:github\.com|gitlab\.com|[\w.-]*\.github\.io)/?[^\s,)\]\}]*")


def code_of(rec, override=None):
    if override is not None:
        return override
    txt = (rec.get("abstract", "") or "") + " " + (rec.get("comment", "") or "")
    m = CODE_RE.findall(txt)
    if m:
        return ascii_of(m[0]).rstrip(".").rstrip("/").replace("\\", "")
    if re.search(r"code (?:is |will be |)?(?:available|released|public)", txt, re.I):
        return "mentioned-no-url"
    return "unknown"


# ------------------------------------------------------------------ curation
# (arxiv, tags, relevance, one_line_en, reason_cn->ascii? no: reason must be ASCII => English)
GFN = [
    # (1) non-acyclic / shortest path / min flow / OT
    ("2606.16073", "non-acyclic;min-flow;termination;mcmc;sampler;downloaded", 5,
     "Learns a state-dependent stopping classifier inside the non-acyclic GFlowNet framework so MCMC-style trajectories terminate on their own, cutting average trajectory length.",
     "Same lab as O07/O08; it optimises the same object (expected trajectory length under a non-acyclic flow) but through a learned terminal transition. Any residual-to-OT-gap bound must cover this stopping rule, and its multilevel training is a ready-made trick for the permutation environment."),
    ("2510.21886", "shortest-path;tsp;mst;combinatorial;trajectory-balance", 4,
     "Applies trajectory-balance GFlowNets to shortest path, minimum spanning tree and TSP on small benchmark graphs and reports agreement with Dijkstra, Kruskal and exact solvers.",
     "Only other 2025-2026 paper that puts GFlowNets and shortest path in the same sentence. Weak evidence (small explicit graphs, no theory), but it is the paper a reviewer will cite to ask why O07/O08 are new, so the boundary must be argued explicitly."),
    ("2509.20408", "multi-agent;flow-matching;theory;mean-field-adjacent", 4,
     "Builds a theory of multi-agent GFlowNets with centralized, independent, joint and conditional flow networks, proving that local flows can be trained as one global flow.",
     "The local-global principle is the closest existing GFlowNet analogue of the mean-field / congestion cost that GSBoG (C02) has and O08 lacks. If we want a distribution-dependent flow penalty, this is the prior art to cite and to beat."),
    ("2505.03561", "non-acyclic;continuous;flow-matching;ergodicity;imitation", 4,
     "Ergodic Generative Flows use finitely many globally defined diffeomorphisms to make flow-matching loss tractable in continuous settings and add a KL-weak-FM loss for imitation learning.",
     "Brunswic et al. again, the T19 lineage. It is the main 2025 attempt at non-acyclic training outside discrete graphs and shows what breaks when the state space stops being finite - the exact regime our weighted-cost extension would enter."),
    ("2602.01749", "theory;markov-chain;reversibility;exploration;objective", 4,
     "Shows GFlowNet objectives are equivalent to reversibility of an induced Markov chain and introduces alpha-GFN, which tunes the forward/backward mixing ratio.",
     "Reversibility is exactly the structure O08 exploits when it turns detailed balance into a linear flow constraint. If the mixing ratio is a free parameter, the minimum-flow LP may not be unique any more; worth checking before claiming the LP has a canonical solution."),
    ("2503.14549", "path-space-control;schrodinger;doob-h-transform;gibbs;theory", 4,
     "Frames sequential structured generation as an exact path-space control problem, with the corrected law given by a Doob h-transform and identified as a one-sided Schrodinger transport and an ideal GFlowNet flow function.",
     "Gives the missing dictionary between GFlowNet flow functions and one-sided Schrodinger bridges on a growing state graph. This is the technical bridge needed to make the epsilon>0 version of the GFN-OT objective rigorous instead of hand-waved."),
    # (2) training objectives / stability / policy gradient
    ("2605.01729", "stability;tv-bound;certificate;trajectory-balance;theory", 5,
     "Proves that a small total-variation gap does not bound the trajectory-balance loss, then derives converse loss-to-TV bounds and a Stable-GFlowNet training scheme with an adaptive reference flow.",
     "Closest prior art to our topic 1 (balance residual to error bound). It supplies the loss-to-TV half; what is still missing, and what we would add, is TV-to-transport-cost via OT stability in the marginals. Not downloaded because it never touches OT or shortest paths."),
    ("2606.15793", "policy-gradient;ppo;entropy-regularized-rl;training", 4,
     "Derives policy-gradient equivalents for GFlowNet training and is the first to apply proximal policy optimization, reporting faster convergence than standard GFlowNet objectives.",
     "Written by the O07/O08 authors. If PPO beats regularized TB on molecular graphs it probably also beats it on the permutation OT environment, so it is the training baseline our experiments must include rather than defaulting to TB."),
    ("2605.15417", "objective;f-divergence;off-policy;trajectory-balance;theory", 4,
     "Extends the trajectory-balance surrogate to the whole family of f-divergences, so on-policy gradients match the chosen f-divergence while the off-policy global minimizer is unchanged.",
     "Tells us which divergence the TB residual is actually controlling. Any residual-to-OT-gap statement is divergence-specific, so we need this taxonomy before writing the bound."),
    ("2603.01047", "evaluation-balance;partial-episodes;policy-based;backward-policy", 4,
     "Shows flow balance also yields a policy evaluator and proposes an evaluation-balance objective over partial episodes that supports parameterized backward policies and offline data.",
     "Directly relevant to whether the backward policy can stay learnable in the min-flow setting. O08 parameterizes P_B; this paper is the 2026 reference on what that costs in stability."),
    ("2608.03967", "natural-gradient;fisher-rao;information-geometry;optimization", 4,
     "Treats the forward policy as a trajectory sampler, identifies its intrinsic geometry as the Fisher-Rao metric, and decomposes the trajectory Fisher into per-step conditional second moments.",
     "The per-step Fisher decomposition is a concrete way to see why credit assignment degrades on long trajectories, which is precisely the regime the minimum-flow objective pushes into."),
    ("2509.01632", "rtb;trust-pcl;kl-regularized-rl;equivalence;theory", 4,
     "Proves Relative Trajectory Balance is equivalent to Trust-PCL, an off-policy KL-regularized RL method, and shows KL-regularized RL matches RTB on the paper's illustrative example.",
     "Places TB-family objectives inside KL-regularized RL. That mapping is the cleanest route to answering whether TB can express a path-space KL penalty, the open question flagged in the O08 report section 7.5."),
    ("2603.00454", "prefix-credit;subtrajectory;replay;mode-collapse;llm", 3,
     "Anchors subtrajectory supervision at the root and back-propagates terminal rewards to intermediate prefixes, paired with a submodular replay refresh to fight prefix collapse and length bias.",
     "Length bias is the failure mode a minimum-flow objective deliberately induces. Their diagnosis of prefix collapse is a warning about what short-trajectory pressure does to coverage."),
    ("2505.02035", "theory;convergence;sample-complexity;robustness", 3,
     "A theoretical study of GFlowNet learning behavior along four axes: convergence, sample complexity, implicit regularization and robustness.",
     "Generic but it is the only 2025 attempt at sample complexity for GFlowNets, so it is the default citation when we claim our OT-gap bound is the first quantitative optimality certificate."),
    ("2602.17827", "exploration;auxiliary-gfn;divergent-tb;coverage", 3,
     "Adaptive Complementary Exploration trains a second GFlowNet to search regions the main model under-explores, improving distribution accuracy and high-reward discovery.",
     "Relevant to the open question in the O08 report section 7.7 on whether minimum-flow solutions collapse the visited support. An auxiliary explorer is the obvious mitigation to test."),
    ("2604.17140", "pdg;unifying-framework;loss-design;inference", 3,
     "Local Inconsistency Resolution unifies EM, belief propagation, GANs and GFlowNets as instances of iteratively resolving inconsistency in probabilistic dependency graphs, and suggests a more natural GFlowNet loss.",
     "A loss derived from a graph-inconsistency principle rather than from balance conditions is a genuinely different starting point; worth checking whether the minimum-flow constraint survives the reformulation."),
    ("2511.09677", "boosting;residual-reward;exploration;ensembles", 3,
     "Boosted GFlowNets sequentially trains an ensemble where each member optimizes a residual reward, with a monotone non-degradation guarantee.",
     "The residual-reward decomposition is structurally similar to splitting a transport plan into partial couplings; a plausible route to an unbalanced or staged GFN-OT solver."),
    ("2505.15251", "exploration;auxiliary-agent;mode-collapse;loss-guided", 3,
     "Loss-Guided GFlowNets use an auxiliary GFlowNet whose exploration is driven directly by the main model's training loss to focus sampling on poorly fitted regions.",
     "Loss-driven sampling is exactly what a primal-dual gap monitor would enable if the dual potential critic from O08 Theorem 3.3 were implemented; this is the reward-free version of that idea."),
    ("2602.21565", "composition;multi-objective;inference-time;conditional", 3,
     "Composes pre-trained GFlowNets at inference time for new reward combinations, exact for linear scalarization and with a distortion bound for nonlinear operators.",
     "The cheap version of topic 2 (conditional GFN over a family of transport problems): if composition works, we may not need to retrain per (L,R) pair. Also sets the bar a conditional GFN-OT must clear."),
    ("2506.02685", "symmetry;bias-correction;graph;reward-scaling", 3,
     "Symmetry-Aware GFlowNets correct systematic bias from graph symmetries through reward scaling instead of explicit state transition probability computation.",
     "Permutation and Cayley graph environments are full of symmetry; an uncorrected symmetry bias would corrupt the terminal marginal and hence the measured transport cost."),
    # (3) diffusion samplers / continuous GFN / SOC
    ("2501.06148", "continuous-time;diffusion-sampler;asymptotic-equivalence;pde", 4,
     "Proves equivalences between discrete-time entropic RL objectives (GFlowNets) and continuous-time objects (PDEs, path-space measures) in the small-step limit, and shows coarse time discretization speeds up training.",
     "The formal statement that GFlowNet objectives converge to path-space measure objectives. If we want to compare a discrete-step GFN-OT against a continuous-time Schrodinger bridge such as GSBoG, this is the only rigorous bridge available."),
    ("2606.16222", "continuous-gfn;subtrajectory-balance;llm;latent-reasoning", 3,
     "Models LLM reasoning as variable-length continuous trajectories sampled by a continuous GFlowNet with an entropy-weighted subtrajectory balance objective and a reference-prior regularizer.",
     "A rare instance of variable trajectory length being treated as a first-class cost, plus an explicit reference prior - the two ingredients of an entropic transport objective, in a non-graph setting."),
    ("2607.06432", "unlearning;residual-nabla-gfn;distributional-alignment;diffusion", 3,
     "Formulates concept unlearning as finding the minimum-deviation conditional distribution under a forgetting constraint and trains it with residual nabla-GFlowNet score correction.",
     "Minimum deviation subject to a constraint is an I-projection, i.e. the static Schrodinger problem. Same mathematical shape as GFN-OT with an entropy term, arrived at from a completely different application."),
    # (4) LLM reasoning / alignment / red teaming
    ("2607.13394", "llm;partition-function-free;stability;distribution-matching;scale", 4,
     "Removes the learned partition network from GFlowNet-style LLM RL by using an in-batch Monte Carlo estimate, plus importance-sampling correction and asymmetric flow-gap clipping.",
     "O08 sidesteps the partition function by assuming Z=1. This paper shows the alternative: estimate Z from the rollout batch. That is what an unbalanced GFN-OT with unknown mass would need, so it is the reference design for section 7.6 of the O08 report."),
    ("2605.00553", "llm;red-teaming;partition-free;contrastive-tb;stability", 3,
     "Stable-GFN eliminates partition function estimation through pairwise comparisons and adds masking against noisy rewards plus a fluency stabilizer.",
     "Second independent 2026 data point that removing Z improves stability. Two papers agreeing makes it a trend rather than an anecdote."),
    ("2603.18363", "llm;trajectory-balance;length-bias;alpha-power;rlif", 3,
     "PowerFlow casts unsupervised LLM fine-tuning as distribution matching with a length-aware trajectory-balance objective targeting alpha-power distributions.",
     "Length-aware TB is the LLM-side mirror of minimum-flow: both try to stop the objective from being dominated by trajectory length. The correction term is directly transferable."),
    ("2602.12642", "llm;partition-function;curriculum;replay;sample-efficiency", 3,
     "Reinterprets the GFlowNet partition function as a per-prompt expected-reward signal and uses it to prioritize informative prompts and error-prioritized replay.",
     "Shows the flow/partition quantity carries usable information beyond normalization. In the OT setting the analogous quantity is the state flow F(s), which we currently only penalize rather than read."),
    ("2602.10583", "llm;dag-state-space;span;exploration", 3,
     "Flow of Spans builds a dynamic span vocabulary so the language-model state space is a DAG rather than a tree, letting GFlowNets explore multiple compositional paths to the same string.",
     "A clean demonstration that merging paths (DAG instead of tree) changes what a GFlowNet can express. The next step along that axis - allowing cycles - is exactly the non-acyclic setting."),
    # (5) combinatorial optimization and structure learning
    ("2502.03669", "benchmark;combinatorial;negative-result;mis;classical-baseline", 4,
     "Compares AI methods against classical solvers on Maximum Independent Set and finds the classical KaMIS solver on one CPU beats leading GPU-based AI methods, with a serialization analysis showing the GFlowNet-based LTFT reasons like a degree-based greedy.",
     "The strongest published warning that a learned sampler on a combinatorial graph problem can lose to a classical solver. Our GFN-OT versus network-simplex comparison faces the same critique, so this defines the evidence bar."),
    ("2510.04792", "vrp;combinatorial;detailed-balance;trajectory-balance;hybrid", 3,
     "Hybrid-Balance GFlowNet adaptively combines trajectory balance and detailed balance for vehicle routing, with a depot-specific inference strategy.",
     "Routing problems are min-cost-flow relatives. The finding that TB alone is insufficient and DB must be blended in matters because O08 trains with regularized TB only."),
    ("2503.01931", "vrp;tsp;adversarial;diversity;combinatorial", 3,
     "Adversarial GFlowNet pairs a generative flow network with a discriminator for capacitated vehicle routing and TSP, plus a hybrid decoding step.",
     "Establishes the GFlowNet-for-routing baseline family that a shortest-path or transport GFlowNet will be benchmarked against."),
    ("2506.12033", "matching;assignment;mechanism-design;diversity", 3,
     "EMERGENT applies GFlowNets to one-sided matching, trading rank efficiency against manipulation resistance by sampling diverse high-reward matchings.",
     "One-sided matching is a degenerate assignment problem, i.e. OT with 0/1 marginals. The clearest existing example of a GFlowNet sampling from the polytope of feasible couplings."),
    ("2503.06985", "structure-learning;decision-trees;amortized-inference;posterior", 3,
     "DT-GFN formulates decision-tree construction as sequential planning and trains a GFlowNet policy that samples trees from the Bayesian posterior.",
     "The reference example of GFlowNet as amortized structure inference over a huge implicit space, which is the same selling point we use for implicit combinatorial transport graphs."),
    ("2501.05498", "thesis;theory;structure-learning;foundations", 3,
     "Doctoral thesis covering GFlowNet mathematical foundations, links to variational inference and RL, extensions beyond discrete problems, and Bayesian structure learning over DAGs.",
     "Useful as a single consistent source for the measure-theoretic foundations that the non-acyclic extension has to modify."),
    # (6) tooling and other applications
    ("2511.16592", "library;jax;benchmark;hypergrid;reproducibility", 4,
     "gfnx is a JAX library with single-file implementations of core GFlowNet objectives plus hypergrid, sequence, molecular, phylogenetic and Ising environments, reporting up to 55x-80x speedups over PyTorch baselines.",
     "By the same group as O07/O08 and already contains the hypergrid environment used in O08 Table 1. Fastest path to a reproducible baseline harness for the transport experiments."),
    ("2604.21830", "visual-analytics;diagnostics;training-dynamics;transition-heatmap", 3,
     "GFlowState is a visual analytics system exposing sampling trajectories, state projections, a trajectory network node-link diagram and a transition heatmap during GFlowNet training.",
     "A transition heatmap is a rendering of the edge flow. That is exactly the object we need to compare against an exact LP solution on medium-sized grids, which O08 never does."),
    ("2603.01655", "ray-tracing;path-sampling;action-masking;replay;ood", 3,
     "Replaces exhaustive ray-path search in radio propagation with GFlowNet sampling, using a replay buffer for rare valid paths, a uniform exploratory policy and physics-based action masking.",
     "A path-sampling application where valid paths are rare and the graph is implicit - structurally the same problem as sampling shortest paths on a Cayley graph, and its action masking is directly reusable."),
    ("2602.11498", "state-space-partitioning;scalability;convergence", 3,
     "Partial GFlowNet introduces a planner that partitions a large state space into overlapping partial spaces and switches between them heuristically to speed up convergence.",
     "Partitioning the state graph is one of the few generic tricks for large implicit graphs, but it breaks global flow conservation - a tension we would inherit if we try it on permutation environments."),
]

OT = [
    # (1) OT on graphs / min-cost flow / Beckmann
    ("2606.16273", "graph-ot;metric-graph;neural-ot;entropic;semidual;theory;downloaded", 5,
     "First deep generative framework for measures supported on compact metric graphs: embeds the graph, solves an entropic Kantorovich problem with a neural semidual parameterization, projects back, and proves weak convergence to a valid coupling.",
     "The most direct competitor to the O08 claim of applying neural learning to OT on large graphs. Differences we must state explicitly: metric graph with continuous edges versus discrete state graph, entropic versus unregularized, plan versus executable routing policy."),
    ("2601.20203", "min-cost-flow;dual-prediction;learning-augmented;epsilon-relaxation;theory;downloaded", 5,
     "First minimum-cost network flow algorithm augmented with a learned dual prediction, built on epsilon-relaxation, with running-time bounds in the infinity-norm prediction error and PAC sample complexity for the predictor.",
     "This is the learned-dual-potential idea from the O08 report section 7.3, already published for classical min-cost flow. It occupies the warm-start framing, so our contribution has to be the certificate (complementary slackness monitoring) rather than the speedup."),
    ("2502.00739", "graph-ot;unbalanced;orlicz;entropy-partial-transport;scalable;downloaded", 5,
     "Introduces Orlicz entropy partial transport and Orlicz-Sobolev transport for measures of different total mass on graph metric spaces, computable by solving a single univariate optimization problem.",
     "Answers exactly the gap flagged in the O08 report section 7.6: unbalanced OT on graphs. It is fast and has theory, so a GFlowNet unbalanced extension needs a reason to exist beyond being unbalanced."),
    ("2603.19755", "beckmann;regularity;flux;poisson;parametric;theory", 4,
     "Develops Holder regularity theory for Beckmann's problem through an unconstrained Lagrangian, showing the multiplier enforcing the divergence constraint solves a Poisson equation and the flux is its gradient, with joint parameter regularity for conditional targets.",
     "Beckmann's divergence-constrained flux is the continuous twin of the O08 reduced LP. Parametric regularity in the target is precisely the property a conditional GFN-OT would need in order to generalize across (L,R) pairs."),
    ("2602.04308", "graph-ot;statistical-mechanics;entropy-cost-tradeoff;bipartite;mean-field", 4,
     "Mean-field theory for the Sub-Optimal Transport model, an ensemble of weighted bipartite graphs where a coupling parameter interpolates between entropy-dominated dense couplings and cost-dominated sparse ones; the crossover is smooth, not a phase transition.",
     "Gives the physics of the temperature axis that separates O08 (epsilon=0, vertex solutions) from GSBoG and DDSBM (epsilon>0). Useful when we argue that the unregularized regime is a distinct problem rather than a limit case."),
    ("2512.03194", "min-cost-flow;mapf;gnn-policy;rebalancing;hierarchical", 3,
     "GRAND couples a reinforcement-learned GNN policy that predicts a desired agent distribution with a minimum-cost flow rebalancing step and local assignment for multi-agent pickup and delivery.",
     "A working example of the learned-guidance plus min-cost-flow hybrid on a routing graph with a hard latency budget. Sets the bar for what a learned transport policy has to beat operationally."),
    ("2608.27500", "graph-ot;survey;wasserstein;gromov-wasserstein;bures", 3,
     "Review of optimal transport for comparing undirected unweighted graphs via Wasserstein, Gromov-Wasserstein and Bures-Wasserstein distances, including how transport plans localize the effect of perturbations.",
     "Fastest way to map which graph OT distances already exist, so that we do not accidentally reinvent one and so we can pick the right competitor family for each experiment."),
    ("2606.03317", "graph-ot;ollivier-ricci;cycles;approximation;scalability", 3,
     "Approximates Ollivier-Ricci curvature by restricting the transport to 3-, 4- and 5-cycles containing the edge, with a greedy pruning algorithm that keeps accuracy while cutting time on large scale-free graphs.",
     "A rare graph-OT paper that treats cycles as the computational primitive rather than an obstacle. Relevant because our whole setting is defined by allowing cycles."),
    ("2608.10619", "graph-ot;rewiring;over-squashing;budget-allocation;gnn", 3,
     "PairAlign scores pairwise communication shortage on a graph and uses an optimal transport allocation to spend a limited edge-insertion budget on the pairs whose demand is least supported.",
     "Uses OT to allocate a budget over node pairs on a graph, which is a discrete transport problem where the cost comes from propagation distance - the same functional form as shortest-path ground cost."),
    ("2511.01443", "graph;effective-resistance;curvature;efficiency;substitute", 3,
     "Replaces the optimal transport distance in Ollivier-Ricci curvature with effective resistance between node pairs, proving lower complexity and comparable geometric expressiveness.",
     "A concrete claim that a spectral surrogate can substitute for graph OT distance. If it holds, some of the applications we would target with GFN-OT are better served by resistance, and we should say where it fails."),
    # (2) discrete / graph Schrodinger bridges and entropic regularization
    ("2509.23348", "discrete-sb;eot;benchmark;analytic-solution;solver;downloaded", 5,
     "Builds the first benchmark for Schrodinger bridges on discrete spaces with analytically known solutions, and as a byproduct introduces the DLightSB, DLightSB-M and alpha-CSBM solvers.",
     "The evaluation protocol any entropic GFN-OT would be judged by, with ground truth available and code released. Also the cleanest evidence that the discrete SB field has enough solvers to need a benchmark - a crowding signal for topic 3."),
    ("2607.19176", "schrodinger-bridge;sinkhorn;hybrid-state-space;discrete-regime;convergence;theory", 4,
     "Proves exponential convergence in relative entropy of the Sinkhorn algorithm for the Schrodinger bridge with regime switching on a hybrid state space R^d times a finite set, including a partially observed terminal setting.",
     "The rate result for a bridge whose state has a genuinely discrete component. It is the template for what a convergence statement over a discrete transport graph should look like, and it is stated under compactness rather than smoothness."),
    ("2511.16458", "inverse-ot;markov-chain;aggregate-data;entropic;convex;theory", 4,
     "Estimates the transition matrix of a discrete-state Markov chain from aggregate distributions at successive times by jointly optimizing over the matrix and entropic transport plans, yielding a convex problem with a proximal algorithm.",
     "This is the inverse of what a GFlowNet does: they recover the kernel from marginals, we impose marginals and learn the kernel. The convexity of the joint formulation is a strong hint about which parameterizations keep the GFN-OT problem well posed."),
    ("2506.22565", "schrodinger-bridge;diffusion-sampler;adjoint-matching;boltzmann;soc", 4,
     "Adjoint Schrodinger Bridge Sampler learns to sample from unnormalized energies with a matching-based objective that needs no target samples, generalizing Adjoint Sampling to arbitrary source distributions by dropping the memoryless condition.",
     "The continuous-space counterpart of what an entropic GFN-OT would do with two prescribed endpoints, from a NeurIPS 2025 Oral with released code. Any epsilon>0 extension of O08 will be asked how it compares."),
    ("2604.01144", "schrodinger-bridge;discrete-time;gaussian-mixture;density-steering;policy", 3,
     "Constructs feasible Markovian policies for discrete-time Schrodinger bridges and density steering between Gaussian mixtures as mixtures of component-to-component optimal policies, with cost no worse than existing approximations.",
     "Decomposing a bridge into elementary component-to-component policies is structurally the same move as decomposing a coupling into per-pair shortest paths, which is exactly step 1 of the O08 Theorem 3.2 proof."),
    ("2503.23705", "mean-field-sb;multi-agent;covariance-steering;closed-form;control", 3,
     "Approximates mean-field Schrodinger bridges between Gaussian mixture boundary distributions in closed form as a mixture of covariance-steering policies, with no learning step and support for state constraints.",
     "Mean-field means the cost depends on the current population, the ingredient GSBoG has and GFlowNets do not. This shows what the closed-form regime looks like before any learning is involved."),
    ("2605.06829", "survey;flow-matching;score-based;schrodinger-bridge;eot", 3,
     "Unified measure-theoretic survey of diffusion, score-based and flow-matching generative models as learning a time-dependent vector field, with a section connecting them to Schrodinger bridges and entropic OT.",
     "Useful map of which continuous-time transport objective corresponds to which training loss, which is the vocabulary we need when arguing that a discrete-step GFlowNet is not just a coarse diffusion sampler."),
    # (3) neural / amortized / conditional OT
    ("2605.20883", "amortized-ot;fgw;graph-ot;conditional;dictionary-learning;fmri;downloaded", 5,
     "Learns fMRI activation dictionaries across individual brain geometries by comparing graphs with the Fused Gromov-Wasserstein distance, using an amortized neural network to predict approximate transport plans and atoms that depend on the FGW tradeoff parameter.",
     "Same authors as ULOT (C01). It confirms the amortized-conditional-graph-OT niche is being actively expanded by the incumbent team, which is the concrete reason topic 2 in the competitor matrix stays high risk."),
    ("2604.15114", "amortized-ot;sliced;kantorovich-potential;plan-prediction;regression", 4,
     "Predicts OT plans across many measure pairs by amortizing Kantorovich potentials from sliced OT, with a regression-based and an objective-based variant, then recovering the plan from the estimated potentials.",
     "Amortizes the dual potential rather than the plan. That is the same object as the BFS-distance potential in O08 Theorem 3.3, so the parameterization and the training losses transfer almost directly."),
    ("2511.19741", "amortized-ot;sliced;transferability;minibatch;theory", 4,
     "Studies whether an optimized slicer in the min-Sliced Transport Plan framework transfers to new distribution pairs, proving stability under perturbations and adding a minibatch formulation with statistical guarantees.",
     "The transferability question stated formally: does a solver trained on one (mu,nu) still work on a nearby pair. That is the theoretical core of any conditional GFN-OT claim, and here it already has a proof technique."),
    ("2608.28262", "sinkhorn;sparse;lifted;eot;solver;convergence", 4,
     "SinkSLOT sparsifies the Gibbs kernel using an expected sliced lifted transport plan as a non-independent reference coupling, giving O(LN) per-iteration cost with a convergence proof and no debiasing needed.",
     "A current-generation entropic solver baseline. If our claim is that GFlowNets help when N is large, this is the number to beat on any instance that can still be written down explicitly."),
    ("2602.03566", "neural-ot;riemannian;curse-of-dimensionality;approximation;theory", 3,
     "Proves that any manifold OT method producing discrete approximations of transport maps suffers the curse of dimensionality, then introduces continuous neural parameterizations with sub-exponential complexity.",
     "The lower bound for discretization-based transport is the general form of our argument against materializing a cost matrix on a huge implicit graph, stated for manifolds instead of combinatorial graphs."),
    ("2605.04255", "neural-ot;entropic;riemannian;schrodinger-potential;amortized", 3,
     "Entropic Riemannian Neural OT learns a single target-side Schrodinger potential through a neural pullback parameterization and recovers the induced Gibbs coupling with convergence guarantees for fixed regularization.",
     "Shows that learning one potential is enough to reconstruct the entropic coupling. The same single-network trick applies to the extended dual in O08 Appendix A.4, where a_x and b_u come from one potential."),
    ("2509.25444", "neural-ot;amortized;dual-potential;conformal;icnn", 3,
     "Parameterizes the conditional vector quantile function as the gradient of an input-convex neural network and amortizes the dual potentials, then uses the induced multivariate ranks for conformal prediction.",
     "A worked example of amortizing dual potentials with finite-sample validity attached. If we want a certificate rather than a point estimate, this is the closest published pattern."),
    ("2503.12633", "amortized-ot;filtering;clustering;mixture-of-experts;online", 3,
     "The amortized optimal transport filter clusters pre-trained OT maps offline and takes a weighted average of the relevant subset at inference time, avoiding per-step retraining.",
     "Cheap amortization by interpolating between cached solutions. It is the baseline that a conditional GFN-OT must beat, and it is much easier to implement than a conditional policy."),
    ("2605.08254", "amortized;hypernetwork;ot-loss;conditioning;steering", 3,
     "HyperTransport trains a hypernetwork end-to-end with an optimal transport loss to map concept embeddings directly to intervention parameters, producing a new intervention in one forward pass.",
     "Hypernetwork conditioning is an alternative to conditioning the policy input. Worth considering if we ever train one model over a family of (L,R) marginals."),
    ("2605.08793", "solver;gpu;entropic;sinkhorn;quasi-newton;engineering", 3,
     "cuRegOT is a GPU solver for entropic regularized OT with amortized symbolic analysis, asynchronous Sinkhorn iterate generation and a fused gradient kernel, with convergence guarantees.",
     "The engineering ceiling for classical entropic OT on GPU. Any wall-clock claim about learned transport has to be measured against this rather than against a CPU Sinkhorn."),
    # (4) unbalanced OT
    ("2601.22856", "unbalanced;fgw;graph;kl-penalty;regularizer", 3,
     "OptiMAG regularizes multimodal attributed graphs with Fused Gromov-Wasserstein plus a KL divergence penalty to align implicit modality structure with the explicit graph structure.",
     "A drop-in unbalanced FGW regularizer on graphs. Shows how routinely the KL-relaxed marginal is now used, which weakens any claim that handling unequal mass is itself novel."),
    ("2602.09933", "unbalanced;matching;longitudinal;pruning;medical", 3,
     "Uses unbalanced optimal transport with a registration-aware cost to match lesions across longitudinal CT scans, recovering appearance, disappearance, merging and splitting without heuristic rules.",
     "A clean demonstration that mass creation and destruction are what unbalanced OT buys you. The same vocabulary maps onto the virtual source-sink edge proposed in the O08 report section 7.6."),
    ("2602.08417", "unbalanced;graph-matching;localization;occlusion;robotics", 3,
     "Graph-Loc performs scan-to-map association through unbalanced optimal transport with a local graph-context regularizer, using the relaxed mass conservation to survive occlusion and fragmented structure.",
     "Real-time unbalanced graph matching under partial observation. It is evidence that relaxing marginals is the standard fix for missing mass, and it runs at sensor rates."),
    ("2605.02497", "unbalanced;gaussian;kl-relaxation;closed-form;riccati;theory", 3,
     "Derives an explicit solution for static KL-unbalanced OT between Gaussians with quadratic cost and two independent relaxation parameters and no coupling entropy, with the covariance solving a Riccati equation and a quadratic KL dual certificate.",
     "The rare unbalanced setting where a dual certificate is explicit. Since we want certificates rather than only convergence, the structure of their dual is worth copying even though the Gaussian case is far from graphs."),
    ("2606.07239", "unbalanced;molecular;flexible-size;generative;geometric-graph", 3,
     "Morph uses unbalanced optimal transport to let a 3D molecular generative model change the number of atoms during generation, integrating scaffolds and improving property steering.",
     "Variable object size handled by unbalanced transport in a generative model. That is the closest existing analogue of a GFlowNet whose terminal states have different sizes, which is the normal case in molecule editing."),
    ("2503.15105", "unbalanced;neural-ode;sinkhorn;convergence;error-estimate;theory", 3,
     "Constructs vector fields for Neural ODEs that converge to true unbalanced OT dynamics by generalizing a discrete UOT problem with Pearson divergence, with a Sinkhorn-inspired scheme and explicit error estimates.",
     "One of the few unbalanced results with explicit error estimates rather than asymptotics. The discrete-to-continuum construction is a usable template for turning our discrete flow into a certified approximation."),
    ("2506.11969", "unbalanced;neural-ot;frechet-regression;single-cell;interpolation", 3,
     "Robust local Frechet regression using an unbalanced neural OT Wasserstein distance to interpolate high-dimensional cell distributions at unobserved times, with generative networks producing the distributions.",
     "Interpolating between prescribed marginals at intermediate times is what a bridge does; doing it with unbalanced OT and a generative net is the applied version of the setting we care about."),
    # (5) OT error bounds and dual certificates
    ("2605.27883", "quadratic-regularization;stability;dual-potential;support;error-bound;theory", 4,
     "Quantitative stability theory for quadratically regularized OT under perturbations of marginals, cost and regularization, centred on an L-infinity stability result for the dual potentials and yielding local Lipschitz stability of the optimal support.",
     "This is the stability-in-the-marginals ingredient the draft bound in the O08 report section 7.2 needs, and it is for the quadratic regularization that O02 (Essid and Solomon) uses on graphs. Directly reusable."),
    ("2605.27175", "quadratic-regularization;polyak-lojasiewicz;error-bound;linear-convergence;theory", 4,
     "Establishes a local error bound and a Polyak-Lojasiewicz inequality for the quadratically regularized OT dual with explicit constants, giving linear convergence rates for gradient and coordinate ascent.",
     "A PL inequality on the dual is the strongest form of the claim that dual suboptimality controls primal error. It is what would let us translate a measured primal-dual gap into a transport-cost gap with a constant rather than an order."),
    ("2604.22366", "monge;brenier-potential;estimator;convergence-rate;error-bound;theory", 4,
     "Statistical estimator for Monge maps built from the dual solution of the discrete sampled problem, with convergence rates from a new error bound for quadratic OT and sharper rates in the semi-discrete case.",
     "Turns a finite-sample dual solution into a certified map estimate with no smoothness assumption. The semi-discrete case is the closest continuous analogue of transporting onto a finite terminal set."),
    ("2608.29152", "sinkhorn;potential;uniform-convergence;minimax;regularization-dependence;theory", 4,
     "Non-asymptotic n^{-1/2} rates for empirical Sinkhorn potentials in the quotient supremum norm, with conditions under which the constant depends polynomially rather than exponentially on 1/epsilon, plus matching minimax lower bounds.",
     "Quantifies how badly entropic estimates degrade as the regularization vanishes. That is the exact cost of moving an entropic GFN-OT towards the unregularized LP that O08 actually solves."),
    ("2508.17641", "eot;structural-constraints;martingale;sinkhorn;sparse-newton;solver", 4,
     "Entropic OT under martingale-type conditions, noting that these are row-wise equality or inequality constraints on the coupling, solved by Sinkhorn-type algorithms with sparse Newton iterations.",
     "The generic recipe for entropic OT with extra structural constraints on the coupling. O08 adds exactly one such constraint (fixing the first-step edge flow), so this is the classical solver we would be replacing."),
    ("2512.05282", "entropic-limit;distance-cost;transport-rays;selection;theory", 3,
     "Studies the small-regularization limit of entropic OT on the line with distance cost, proposes a candidate limit object and proves convergence under mutual singularity, plus weak multiplicativity of every limit point.",
     "Distance cost is the continuous version of a shortest-path ground cost, and it is exactly the degenerate case where the optimal plan is not unique - the same non-uniqueness we should expect on unweighted graphs."),
    ("2502.16370", "entropic-limit;monge;transport-rays;selection;theory", 3,
     "Resolves the small-regularization limit of entropic OT with Euclidean distance cost in dimension greater than one, showing the limit plan is supported on transport rays and minimizes a relative entropy within each ray.",
     "The entropic limit selects a specific solution among many optimal ones. If we ever add an entropy term to break the tie between equal-cost shortest paths, this describes what tie-break we would be choosing."),
    # (6) OT for discrete / combinatorial generation and RL
    ("2604.14265", "rl;ot;gradient-flow;transport-budget;behavior-regularized", 4,
     "Value Gradient Flow casts behavior-regularized RL as an optimal transport problem from the reference distribution to the value-induced optimal policy, solved by discrete gradient flow with regularization controlled by the transport budget.",
     "Regularization as a transport budget rather than a KL coefficient is a different knob from the one GFlowNets use. It is the cleanest existing statement that policy improvement itself can be written as a transport problem."),
    ("2605.26078", "rl;wasserstein-policy-gradient;entropy-regularized;global-convergence;theory", 4,
     "Proves global convergence of Wasserstein policy gradient for entropy-regularized RL by replacing convexity with a Bellman-based argument that yields a distributional Polyak-Lojasiewicz condition under a uniform log-Sobolev inequality.",
     "A convergence proof for a transport-geometry policy update in exactly the entropy-regularized RL framework that GFlowNets are known to be equivalent to. The proof strategy is the most plausible route to a convergence result for GFN-OT training."),
    ("2507.22270", "flow-matching;entropic-ot;gibbs-weighting;minibatch;theory", 4,
     "Weighted Conditional Flow Matching reweights each training pair with a Gibbs kernel, recovering the entropic OT coupling up to a marginal bias, and is shown equivalent to minibatch OT in the large-batch limit.",
     "Obtains an entropic coupling by reweighting samples instead of running a solver. If it transfers, the same reweighting could turn on-policy GFlowNet rollouts into an entropic transport objective without an inner Sinkhorn loop."),
    ("2510.15388", "rl;jko;flow-policy;wasserstein-trust-region;fine-tuning", 3,
     "Stepwise Flow Policy shows that discretizing flow-matching inference with a fixed-step Euler scheme aligns it with the variational JKO principle, giving stable online fine-tuning with Wasserstein trust regions.",
     "Each discrete step is a JKO update, i.e. a small transport problem. That is the closest existing justification for treating a fixed number of discrete steps as a legitimate transport discretization."),
    ("2603.12366", "sinkhorn-divergence;gradient-flow;drift;identifiability;generative", 3,
     "Links drifting generative dynamics to Sinkhorn-divergence gradient flows via a cross-minus-self decomposition, using the definiteness of the Sinkhorn divergence to resolve an identifiability gap in prior drift formulations.",
     "Uses a divergence property to prove that zero drift implies distribution matching. That is precisely the kind of identifiability argument a GFN-OT residual-to-gap bound needs at its base."),
    ("2602.23566", "graph-generation;flow-matching;ot-coupling;structural-prior;motifs", 3,
     "Flowette is a continuous flow-matching framework for attributed graph generation with optimal-transport-based coupling and graphette structural priors generalizing graphons.",
     "Uses OT coupling to align source and target graph representations during training. It is the graph-generation counterpart to the transport view, and its ablation isolates how much the OT coupling itself contributes."),
    ("2608.29635", "gromov-wasserstein;hypergraph;alignment;multi-scale;unsupervised", 3,
     "FALCON aligns hypergraphs without features or seeds by building a filtration-induced sequence of clique-based dissimilarity matrices and solving one shared multi-scale Gromov-Wasserstein objective.",
     "One shared plan across multiple resolutions of the same object. If we ever want a GFlowNet transport policy that is consistent across coarsenings of the state graph, this is the objective to compare against."),
    ("2607.17082", "unbalanced;fused-gromov-wasserstein;trajectory;evaluation;pseudo-metric", 3,
     "OTAP evaluates agent trajectories as an unbalanced fused Gromov-Wasserstein distance between attributed dependency graphs, provably invariant to dependency-preserving reordering and bounded in sensitivity to redundant steps.",
     "Treats a trajectory as a graph and transports between trajectories. That is an unusual inversion of our setting (we transport along a trajectory) and it shows what invariances a trajectory-level cost should have."),
]


def build(side, curated, records):
    by_id = {r["arxiv"]: r for r in records}
    rows = []
    prefix = "N-GFN" if side == "gfn" else "N-OT"
    for i, (aid, tags, rel, one, reason) in enumerate(curated, 1):
        rec = by_id[aid]
        venue, vtype, year = venue_of(rec)
        authors = ", ".join(ascii_of(a) for a in rec["authors"][:5])
        if len(rec["authors"]) > 5:
            authors += ", et al."
        rows.append({
            "id": f"{prefix}-{i:02d}",
            "arxiv": aid,
            "title": ascii_of(rec["title"]),
            "authors": authors,
            "venue": venue,
            "venue_type": vtype,
            "year": year,
            "code_url": code_of(rec),
            "one_line_en": ascii_of(one),
            "tags": tags,
            "relevance(1-5)": str(rel),
            "reason": ascii_of(reason),
        })
    out = ROOT / "data" / f"candidates_{side}.csv"
    with out.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)
    print(f"[{side}] {len(rows)} rows -> {out}")
    return rows


if __name__ == "__main__":
    g = build("gfn", GFN, json.load(open(ROOT / "data" / "scan_gfn.json")))
    o = build("ot", OT, json.load(open(ROOT / "data" / "scan_ot.json")))
    for name, rows in (("gfn", g), ("ot", o)):
        print(f"  {name}: venue_type " + ", ".join(
            f"{t}={sum(1 for r in rows if r['venue_type'] == t)}"
            for t in ("main", "workshop", "journal", "preprint")))
        print(f"  {name}: rel5={sum(1 for r in rows if r['relevance(1-5)'] == '5')}, "
              f"rel4={sum(1 for r in rows if r['relevance(1-5)'] == '4')}, "
              f"rel3={sum(1 for r in rows if r['relevance(1-5)'] == '3')}")
