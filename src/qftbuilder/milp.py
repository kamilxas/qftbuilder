"""Exact / certified walk lengths via MILP (scipy + HiGHS).

Formulation (selective TSP in the shortest-path metric, single-commodity
flow against subtours, depot node for an *open* path over anchors):

- ``y_v ∈ {0,1}`` — vertex v is an anchor (visited stop of the walk);
- ``x_e ∈ {0,1}`` — tour leg between anchors in the complete graph over
  anchors + depot, priced ``d[i][j]`` = shortest-path distance (depot legs
  cost 0, so the depot cycle is an open anchor path);
- ``f`` — flows proving connectivity.

Walk length in vertices = objective + 1. Any walk can be represented with
every walk vertex promoted to an anchor at equal cost (legs subdivide along
shortest paths), so the anchor formulation is exact.

Two coverage modes:

- **Full domination** (``shortest_dominating_walk_milp``): every vertex must
  see an anchor — the classical formulation, validated against an exact DP.
- **Budgeted coverage** (``shortest_budgeted_walk_milp``, new): coverage
  indicators ``z_v <= Σ_{u∈N[v]} y_u`` with ``Σ z_v >= k`` — the MILP twin of
  :mod:`qftbuilder.kwalk`, giving LP-relaxation and branch-and-bound *dual
  lower bounds* (and hence optimality certificates) for sub-QFT regions.

``relax=True`` solves the LP relaxation: a valid polynomial-time lower bound.
``return_dual=True`` additionally returns the B&B dual bound, valid even on
timeout.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

__all__ = [
    "shortest_dominating_walk_milp",
    "shortest_budgeted_walk_milp",
]


def _metric(G, nodes, idx):
    n = len(nodes)
    d = np.full((n, n), 10**6, dtype=float)
    for s in nodes:
        for t, dist in nx.single_source_shortest_path_length(G, s).items():
            d[idx[s], idx[t]] = dist
    return d


def _solve(
    G: nx.Graph,
    coverage_k: Optional[int],
    time_limit: float,
    relax: bool,
    return_dual: bool,
):
    """Shared builder. ``coverage_k=None`` -> full domination; otherwise at
    least ``coverage_k`` vertices must be covered by anchors.

    Returns ``(length_vertices, proven, dual_lb)``; ``dual_lb`` is None unless
    ``return_dual``. In relax mode: ``(lower_bound, solved, None)``."""
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    if n <= 1:
        return n, True, n

    # Single-anchor pre-check (the depot formulation forces >= 2 anchors).
    need = n if coverage_k is None else coverage_k
    for v in nodes:
        if len({v} | set(G.neighbors(v))) >= need:
            return 1, True, 1

    d = _metric(G, nodes, idx)
    R = n
    M = n + 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, d[i, j]))
    for v in range(n):
        edges.append((v, R, 0.0))
    E = len(edges)

    NY, NX = n, E
    NZ = 0 if coverage_k is None else n
    Y = lambda v: v
    X = lambda e: NY + e
    Z = lambda v: NY + NX + v            # coverage indicators (budgeted mode)
    FAB = lambda e: NY + NX + NZ + 2 * e
    FBA = lambda e: NY + NX + NZ + 2 * e + 1
    NV = NY + NX + NZ + 2 * E
    cap = float(M)

    c = np.zeros(NV)
    for e, (_, _, cost) in enumerate(edges):
        c[X(e)] = cost

    inc = [[] for _ in range(M)]
    for e, (i, j, _) in enumerate(edges):
        inc[i].append(e)
        inc[j].append(e)

    eq_data, b_eq = [], []

    def eq(coeffs, rhs):
        r = len(b_eq)
        b_eq.append(rhs)
        for col, val in coeffs:
            eq_data.append((r, col, val))

    ub_data, b_ub = [], []

    def ub(coeffs, rhs):
        r = len(b_ub)
        b_ub.append(rhs)
        for col, val in coeffs:
            ub_data.append((r, col, val))

    # anchor degree = 2*y; depot degree = 2 (the two open-path ends)
    for i in range(n):
        eq([(X(e), 1.0) for e in inc[i]] + [(Y(i), -2.0)], 0.0)
    eq([(X(e), 1.0) for e in inc[R]], 2.0)

    if coverage_k is None:
        # full domination: every vertex sees an anchor
        for v in range(n):
            coeffs = [(Y(v), -1.0)]
            for u in G.neighbors(nodes[v]):
                coeffs.append((Y(idx[u]), -1.0))
            ub(coeffs, -1.0)
    else:
        # budgeted coverage: z_v <= sum of anchors in N[v]; sum z >= k.
        # z may stay continuous in [0,1]: for integral y its maximum equals
        # the covered-vertex count, so the integer optimum is unaffected.
        for v in range(n):
            coeffs = [(Z(v), 1.0), (Y(v), -1.0)]
            for u in G.neighbors(nodes[v]):
                coeffs.append((Y(idx[u]), -1.0))
            ub(coeffs, 0.0)
        ub([(Z(v), -1.0) for v in range(n)], -float(coverage_k))

    # connectivity flows: only over selected legs; each anchor consumes 1
    for e in range(E):
        ub([(FAB(e), 1.0), (FBA(e), 1.0), (X(e), -cap)], 0.0)
    out_arc = [[] for _ in range(M)]
    in_arc = [[] for _ in range(M)]
    for e, (i, j, _) in enumerate(edges):
        out_arc[i].append(FAB(e))
        in_arc[j].append(FAB(e))
        out_arc[j].append(FBA(e))
        in_arc[i].append(FBA(e))
    for v in range(n):
        coeffs = (
            [(a, 1.0) for a in in_arc[v]]
            + [(a, -1.0) for a in out_arc[v]]
            + [(Y(v), -1.0)]
        )
        eq(coeffs, 0.0)
    coeffs = [(a, 1.0) for a in out_arc[R]] + [(a, -1.0) for a in in_arc[R]]
    coeffs += [(Y(v), -1.0) for v in range(n)]
    eq(coeffs, 0.0)

    A_eq = coo_matrix(
        ([x[2] for x in eq_data], ([x[0] for x in eq_data], [x[1] for x in eq_data])),
        shape=(len(b_eq), NV),
    ).tocsr()
    A_ub = coo_matrix(
        ([x[2] for x in ub_data], ([x[0] for x in ub_data], [x[1] for x in ub_data])),
        shape=(len(b_ub), NV),
    ).tocsr()

    integrality = np.zeros(NV)
    if not relax:
        integrality[: NY + NX] = 1  # y, x integer; z and flows continuous
    lb = np.zeros(NV)
    ubb = np.concatenate(
        [np.ones(NY + NX + NZ), np.full(2 * E, cap)]
    )

    cons = [LinearConstraint(A_eq, lb=b_eq, ub=b_eq), LinearConstraint(A_ub, ub=b_ub)]
    res = milp(
        c,
        constraints=cons,
        integrality=integrality,
        bounds=Bounds(lb, ubb),
        options={"time_limit": time_limit},
    )

    if relax:
        if res.x is None:
            return None, False, None
        return math.ceil(res.fun - 1e-6) + 1, (res.status == 0), None

    proven = res.status == 0
    length = int(round(res.fun)) + 1 if res.x is not None else None
    dual_lb = None
    if return_dual:
        db = getattr(res, "mip_dual_bound", None)
        if db is not None and math.isfinite(db):
            dual_lb = math.ceil(db - 1e-6) + 1
    return length, proven, dual_lb


def shortest_dominating_walk_milp(
    G: nx.Graph,
    time_limit: float = 120.0,
    relax: bool = False,
    return_dual: bool = False,
) -> Tuple[Optional[int], bool, Optional[int]]:
    """Exact (or best-in-budget) shortest dominating walk length in vertices.

    Returns ``(length, proven, dual_lb)``. ``dual_lb`` is a valid lower bound
    from the B&B tree even on timeout (None unless ``return_dual``). With
    ``relax=True`` the first element is the LP-relaxation lower bound."""
    return _solve(G, None, time_limit, relax, return_dual)


def shortest_budgeted_walk_milp(
    G: nx.Graph,
    k: int,
    time_limit: float = 120.0,
    relax: bool = False,
    return_dual: bool = False,
) -> Tuple[Optional[int], bool, Optional[int]]:
    """Exact (or best-in-budget) length of the shortest walk covering at least
    ``k`` vertices — the certified twin of :func:`qftbuilder.kwalk.budgeted_walk`.

    Same return convention as :func:`shortest_dominating_walk_milp`."""
    n = G.number_of_nodes()
    if not (1 <= k <= n):
        raise ValueError(f"k out of range 1..{n}: {k}")
    return _solve(G, k, time_limit, relax, return_dual)
