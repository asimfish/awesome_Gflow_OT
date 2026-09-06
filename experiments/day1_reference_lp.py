#!/usr/bin/env python3
"""Day-1 experiment of INSIGHTS section 6c: verify, on an enumerable hypergrid, the structural facts
behind O08 (arXiv 2606.06272) that the rest of the repository relies on.

Checks
  1. GFlow* == OT*: the minimum-total-internal-edge-flow LP (O08 Eq. (11), divergence constraint
     div F = L - R on internal states, unit edge costs) has the same optimal value as the Kantorovich
     problem between L and R with the graph shortest-path (hop) cost.
  2. Dual certificate: the LP dual potentials pi satisfy pi_{s'} - pi_s <= 1 on every internal edge, and
     complementary slackness F(e) * (1 - (pi_{s'} - pi_s)) = 0 holds on the optimal flow (O08 Thm. 3.3 /
     App. A.4). We report the primal-dual gap of the exact solution (should be ~0).
  3. Counterexample 1 (INSIGHTS 6b): a flow with ZERO divergence residual that routes mass the long way
     has a strictly larger cost -- balance residual does not certify optimality; the dual gap does.
  4. Counterexample 2: routing the optimal coupling along detours keeps the coupling (endpoint cost)
     but raises the executed path cost -- endpoint-coupling cost != executed transport cost.

Distributions follow O08 App. B.1 in form (ball / moon source, corner-shaped target); the paper does not
publish r_in, r_out, delta, eps, D, R0/R1/R2, so numbers here are NOT a reproduction of Table 1 -- only
the identities are being tested. Everything is exact (scipy.optimize.linprog, HiGHS); no learning.

Usage: python3 experiments/day1_reference_lp.py [--H 10] [--D 2] [--source moon|ball] [--json out.json]
"""
import argparse
import itertools
import json
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path


# ----------------------------------------------------------------------------- graph
def hypergrid(H, D):
    """Internal states = {0..H-1}^D; bidirectional +-1 moves (so the graph has cycles)."""
    states = list(itertools.product(range(H), repeat=D))
    idx = {s: i for i, s in enumerate(states)}
    edges = []
    for s in states:
        for d in range(D):
            for step in (-1, +1):
                t = list(s)
                t[d] += step
                if 0 <= t[d] < H:
                    edges.append((idx[s], idx[tuple(t)]))
    return states, idx, edges


def hop_distance(n, edges):
    A = lil_matrix((n, n))
    for a, b in edges:
        A[a, b] = 1.0
    return shortest_path(csr_matrix(A), directed=True, unweighted=True)


# ----------------------------------------------------------------------------- distributions (O08 App. B.1 form)
def source_dist(states, H, kind, r_out=0.3, r_in=0.2, delta=0.12, eps=1e-3):
    z = np.array(states, dtype=float) / (H - 1)
    D = z.shape[1]
    c = np.full(D, 0.5)
    e1 = np.zeros(D); e1[0] = 1.0
    b = c + (r_out / 2) * e1
    ball = (np.sum((z - c) ** 2, axis=1) <= r_out ** 2).astype(float)
    ramp = 0.5 + 2 * np.minimum(1.0, np.maximum(0.0, 1 - np.linalg.norm(z - b, axis=1) / r_out))
    if kind == "ball":
        L = ball * ramp + eps
    elif kind == "moon":
        hole = (np.sum((z - (c - delta * e1)) ** 2, axis=1) <= r_in ** 2).astype(float)
        L = ball * (1 - hole) * ramp + eps
    else:
        raise ValueError(kind)
    return L / L.sum()


def corner_reward(states, H, R0=1e-3, R1=0.5, R2=2.0):
    s = np.array(states, dtype=float)
    u = np.abs(s / (H - 1) - 0.5)
    R = R0 + R1 * np.all(u > 0.25, axis=1) + R2 * np.all((u > 0.3) & (u < 0.4), axis=1)
    return R / R.sum()


# ----------------------------------------------------------------------------- solvers
def min_flow_lp(n, edges, L, R):
    """min sum_e F(e)  s.t.  out(s) - in(s) = L(s) - R(s)  for all internal s,  F >= 0.
    Returns flow, optimal value, dual potentials (sign-normalised so that pi_{t} - pi_{s} <= 1)."""
    m = len(edges)
    A = lil_matrix((n, m))
    for j, (a, b) in enumerate(edges):
        A[a, j] += 1.0   # out of a
        A[b, j] -= 1.0   # into b
    rhs = L - R
    res = linprog(c=np.ones(m), A_eq=csr_matrix(A), b_eq=rhs, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    # HiGHS returns equality duals y with  c - A^T y >= 0  on the optimum -> 1 - (y_a - y_b) >= 0.
    y = res.eqlin.marginals
    pi = -y  # so that pi_b - pi_a <= 1 on every edge (a->b)
    return res.x, res.fun, pi


def kantorovich(L, R, dist):
    """Exact OT between L and R with cost = hop distance (LP over the coupling)."""
    src = np.flatnonzero(L > 0); tgt = np.flatnonzero(R > 0)
    C = dist[np.ix_(src, tgt)]
    ns, nt = len(src), len(tgt)
    A = lil_matrix((ns + nt, ns * nt))
    for i in range(ns):
        for j in range(nt):
            A[i, i * nt + j] = 1.0
            A[ns + j, i * nt + j] = 1.0
    b = np.concatenate([L[src], R[tgt]])
    res = linprog(c=C.ravel(), A_eq=csr_matrix(A), b_eq=b, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.fun, res.x.reshape(ns, nt), src, tgt


def dual_gap(flow, edges, pi, L, R):
    """Weak-duality gap for a (possibly approximately) feasible flow and a dual potential.
    Also returns max edge violation of pi and the divergence residual."""
    viol = max(0.0, max(pi[b] - pi[a] - 1.0 for a, b in edges))
    dual_obj = float(np.dot(R - L, pi))          # sum_x R pi_x - sum_u L pi_u
    primal = float(flow.sum())
    div = np.zeros(len(L))
    for j, (a, b) in enumerate(edges):
        div[a] += flow[j]; div[b] -= flow[j]
    resid = float(np.abs(div - (L - R)).sum())
    return dict(primal=primal, dual=dual_obj, gap=primal - dual_obj, max_dual_violation=viol, div_residual=resid)


def complementary_slackness(flow, edges, pi):
    slack = np.array([1.0 - (pi[b] - pi[a]) for a, b in edges])
    return float(np.max(np.abs(flow * slack)))


# ----------------------------------------------------------------------------- counterexamples
def detour_flow(n, edges, L, R, dist, idx_edges, coupling, src, tgt, pred):
    """Counterexample 2: realise the OPTIMAL coupling but route every unit of mass via a detour
    (source -> a far waypoint -> target). Endpoint coupling cost is unchanged; executed cost grows."""
    flow = np.zeros(len(edges))
    def add_path(a, b, mass):
        cur = b
        while cur != a:
            p = pred[a, cur]
            flow[idx_edges[(p, cur)]] += mass
            cur = p
    far = int(np.argmax(dist[src[0]]))  # a waypoint far from the first source
    for i, u in enumerate(src):
        for j, x in enumerate(tgt):
            mass = coupling[i, j]
            if mass > 1e-12:
                add_path(u, far, mass); add_path(far, x, mass)
    return flow


def crossed_flow(n, edges, L, R, dist, idx_edges, src, tgt, pred):
    """Counterexample 1: a perfectly balanced flow that pairs sources with FAR targets
    (anti-greedy assignment), routed along shortest paths. Zero divergence residual, larger cost."""
    flow = np.zeros(len(edges))
    Ls = L[src].copy(); Rt = R[tgt].copy()
    order = sorted(((-dist[u, x], i, j) for i, u in enumerate(src) for j, x in enumerate(tgt)))
    coupling = np.zeros((len(src), len(tgt)))
    for _, i, j in order:
        m = min(Ls[i], Rt[j])
        if m <= 1e-15:
            continue
        coupling[i, j] += m; Ls[i] -= m; Rt[j] -= m
    for i, u in enumerate(src):
        for j, x in enumerate(tgt):
            if coupling[i, j] > 1e-12:
                cur = x
                while cur != u:
                    p = pred[u, cur]; flow[idx_edges[(p, cur)]] += coupling[i, j]; cur = p
    endpoint_cost = float((coupling * dist[np.ix_(src, tgt)]).sum())
    return flow, endpoint_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=10); ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--source", default="moon", choices=["moon", "ball"]); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    states, idx, edges = hypergrid(a.H, a.D)
    n = len(states); idx_edges = {e: j for j, e in enumerate(edges)}
    dist, pred = shortest_path(csr_matrix(lil_matrix((n, n)) + _adj(n, edges)), directed=True, unweighted=True, return_predecessors=True)
    L = source_dist(states, a.H, a.source); R = corner_reward(states, a.H)
    out = dict(H=a.H, D=a.D, source=a.source, n_states=n, n_edges=len(edges), mass_check=float(L.sum() - R.sum()))

    # 1. GFlow* vs OT*
    flow, gflow_star, pi = min_flow_lp(n, edges, L, R)
    ot_star, coupling, src, tgt = kantorovich(L, R, dist)
    out["GFlow_star"] = gflow_star; out["OT_star"] = ot_star; out["abs_diff"] = abs(gflow_star - ot_star)

    # 2. dual certificate on the exact solution
    cert = dual_gap(flow, edges, pi, L, R)
    cert["complementary_slackness_max"] = complementary_slackness(flow, edges, pi)
    out["exact_solution_certificate"] = cert
    # terminal potentials vs shortest-path distances are NOT expected to match in the multi-source case
    # (O08 Thm 3.3 is the single-source special case); record the relation for the record.
    out["pi_range"] = [float(pi.min()), float(pi.max())]

    # 3. counterexample 1: balanced but crossed
    cflow, c_endpoint = crossed_flow(n, edges, L, R, dist, idx_edges, src, tgt, pred)
    c_cert = dual_gap(cflow, edges, pi, L, R)
    out["counterexample_crossed"] = dict(executed_cost=float(cflow.sum()), endpoint_cost=c_endpoint, **c_cert)

    # 4. counterexample 2: optimal coupling, detoured routing
    dflow = detour_flow(n, edges, L, R, dist, idx_edges, coupling, src, tgt, pred)
    d_cert = dual_gap(dflow, edges, pi, L, R)
    out["counterexample_detour"] = dict(executed_cost=float(dflow.sum()), endpoint_cost=ot_star, **d_cert)
    out["seconds"] = round(time.time() - t0, 2)

    # ---------------------------------------------------------------- report
    print(f"hypergrid H={a.H} D={a.D} source={a.source}: {n} states, {len(edges)} directed edges; mass(L)-mass(R)={out['mass_check']:.2e}")
    print(f"[1] GFlow* (min total internal flow) = {gflow_star:.6f}   OT* (hop-cost Kantorovich) = {ot_star:.6f}   |diff| = {out['abs_diff']:.2e}")
    print(f"[2] exact solution: primal={cert['primal']:.6f} dual={cert['dual']:.6f} gap={cert['gap']:.2e}  max dual violation={cert['max_dual_violation']:.2e}  CS max={cert['complementary_slackness_max']:.2e}  div residual={cert['div_residual']:.2e}")
    print(f"[3] crossed flow (zero residual, far pairing): executed cost={out['counterexample_crossed']['executed_cost']:.4f}  gap vs same dual={out['counterexample_crossed']['gap']:.4f}  div residual={out['counterexample_crossed']['div_residual']:.2e}")
    print(f"[4] detoured flow (optimal coupling, long routes): endpoint cost={ot_star:.4f}  executed cost={out['counterexample_detour']['executed_cost']:.4f}  gap={out['counterexample_detour']['gap']:.4f}  div residual={out['counterexample_detour']['div_residual']:.2e}")
    print(f"done in {out['seconds']}s")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print("wrote", a.json)


def _adj(n, edges):
    A = lil_matrix((n, n))
    for a, b in edges:
        A[a, b] = 1.0
    return A


if __name__ == "__main__":
    main()
