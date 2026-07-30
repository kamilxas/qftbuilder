"""Exact walk optimization via OR-Tools CP-SAT (optional dependency).

CP-SAT's ``AddCircuit`` gives native lazy subtour elimination, warm-start
hints and parallel search — in the source project it extended the provable
frontier well past the MILP (improving 100-150-vertex walks by 2.5-4% and
proving optima the MILP could not). Used by the solver's ``max`` profile as a
polish rung: seeded with the cascade winner, accepted only on strict
improvement, so it is monotone-safe.

Install with ``pip install qft-builder[exact]``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from .walk import _essential_seq, covers, walk_valid

__all__ = ["solve_walk_cpsat", "solve_whp_cpsat", "HAVE_ORTOOLS"]

try:  # pragma: no cover - trivial import guard
    from ortools.sat.python import cp_model

    HAVE_ORTOOLS = True
except ImportError:  # pragma: no cover
    cp_model = None
    HAVE_ORTOOLS = False


def _require():
    if not HAVE_ORTOOLS:
        raise ImportError(
            "ortools is not installed; run `pip install qft-builder[exact]`"
        )


def _metric_int(G, nodes, idx):
    n = len(nodes)
    d = np.full((n, n), 10**6, dtype=np.int64)
    for s in nodes:
        for t, dist in nx.single_source_shortest_path_length(G, s).items():
            d[idx[s], idx[t]] = dist
    return d


def solve_walk_cpsat(
    G: nx.Graph,
    time_limit: float = 60.0,
    hint_walk: Optional[List[int]] = None,
    workers: int = 8,
    log: bool = False,
) -> Dict:
    """Shortest dominating walk (vertex count) via CP-SAT circuit over the
    shortest-path metric (selective-TSP semantics, validated against exact DP
    in the source project).

    Returns ``{length, proven, dual_lb, walk, status}``; ``dual_lb`` is a
    valid lower bound even on timeout. ``hint_walk`` warm-starts the search
    and adds the valid cut ``objective <= len(hint) - 1``."""
    _require()
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    if n <= 1:
        return dict(length=n, proven=True, dual_lb=n, walk=list(nodes), status="OPTIMAL")
    d = _metric_int(G, nodes, idx)

    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{i}") for i in range(n)]
    arcs = []
    alit = {}
    for i in range(n):
        arcs.append((i, i, y[i].Not()))  # self-loop = vertex skipped
        for j in range(n):
            if i != j:
                lit = m.NewBoolVar("")
                alit[(i, j)] = lit
                arcs.append((i, j, lit))
    for v in range(n):  # depot legs (cost 0) -> open path over anchors
        lo = m.NewBoolVar("")
        li = m.NewBoolVar("")
        alit[(n, v)] = lo
        alit[(v, n)] = li
        arcs.append((n, v, lo))
        arcs.append((v, n, li))
    m.AddCircuit(arcs)

    for w in range(n):  # domination
        m.AddBoolOr([y[idx[u]] for u in G.neighbors(nodes[w])] + [y[w]])

    obj = sum(int(d[i, j]) * alit[(i, j)] for i in range(n) for j in range(n) if i != j)
    m.Minimize(obj)

    if hint_walk:
        m.Add(obj <= len(hint_walk) - 1)  # valid: the hint is feasible
        ess = list(dict.fromkeys(idx[v] for v in _essential_seq(hint_walk, G)))
        ess_set = set(ess)
        for i in range(n):
            m.AddHint(y[i], 1 if i in ess_set else 0)
        if ess:
            m.AddHint(alit[(n, ess[0])], 1)
            m.AddHint(alit[(ess[-1], n)], 1)
            for a, b in zip(ess, ess[1:]):
                m.AddHint(alit[(a, b)], 1)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = int(workers)
    if log:
        s.parameters.log_search_progress = True
    status = s.Solve(m)

    dual_lb = int(math.ceil(s.BestObjectiveBound() - 1e-6)) + 1
    length = walk = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        length = int(round(s.ObjectiveValue())) + 1
        nxt = {}
        for (i, j), lit in alit.items():
            if s.Value(lit):
                nxt[i] = j
        order = []
        cur = nxt.get(n)
        while cur is not None and cur != n:
            order.append(cur)
            cur = nxt.get(cur)
        wk = [nodes[order[0]]]
        for a, b in zip(order, order[1:]):
            sp = nx.shortest_path(G, nodes[a], nodes[b])
            wk.extend(sp[1:])
        if walk_valid(G, wk) and covers(G, wk) and len(wk) == length:
            walk = wk
        else:  # should not happen; stay honest rather than return junk
            length = None
    return dict(
        length=length,
        proven=(status == cp_model.OPTIMAL),
        dual_lb=dual_lb,
        walk=walk,
        status=s.StatusName(status),
    )


def solve_whp_cpsat(
    G: nx.Graph,
    time_limit: float = 60.0,
    workers: int = 8,
    log: bool = False,
) -> Dict:
    """Shortest *simple* dominating path directly on the edges of G.
    ``INFEASIBLE`` proves no such path exists (exact existence test).

    Returns ``{exists, length, proven, dual_lb, walk, status}``."""
    _require()
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    if n <= 1:
        return dict(exists=True, length=n, proven=True, dual_lb=n,
                    walk=list(nodes), status="OPTIMAL")
    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{i}") for i in range(n)]
    arcs = []
    alit = {}
    for i in range(n):
        arcs.append((i, i, y[i].Not()))
    for u, v in G.edges():
        a, b = idx[u], idx[v]
        l1 = m.NewBoolVar("")
        l2 = m.NewBoolVar("")
        alit[(a, b)] = l1
        alit[(b, a)] = l2
        arcs.append((a, b, l1))
        arcs.append((b, a, l2))
    for v in range(n):
        lo = m.NewBoolVar("")
        li = m.NewBoolVar("")
        alit[(n, v)] = lo
        alit[(v, n)] = li
        arcs.append((n, v, lo))
        arcs.append((v, n, li))
    m.AddCircuit(arcs)
    for w in range(n):
        m.AddBoolOr([y[idx[u]] for u in G.neighbors(nodes[w])] + [y[w]])
    m.Minimize(sum(y))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = int(workers)
    if log:
        s.parameters.log_search_progress = True
    status = s.Solve(m)
    if status == cp_model.INFEASIBLE:
        return dict(exists=False, length=None, proven=True, dual_lb=None,
                    walk=None, status=s.StatusName(status))
    dual_lb = int(math.ceil(s.BestObjectiveBound() - 1e-6))
    length = walk = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        length = int(round(s.ObjectiveValue()))
        nxt = {}
        for (i, j), lit in alit.items():
            if s.Value(lit):
                nxt[i] = j
        order = []
        cur = nxt.get(n)
        while cur is not None and cur != n:
            order.append(cur)
            cur = nxt.get(cur)
        wk = [nodes[i] for i in order]
        if (
            len(wk) == length
            and len(set(wk)) == len(wk)
            and walk_valid(G, wk)
            and covers(G, wk)
        ):
            walk = wk
        else:
            length = None
    return dict(
        exists=(length is not None or status == cp_model.FEASIBLE),
        length=length,
        proven=(status == cp_model.OPTIMAL),
        dual_lb=dual_lb,
        walk=walk,
        status=s.StatusName(status),
    )
