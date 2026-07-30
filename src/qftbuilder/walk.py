"""Dominating walks: validity, model-free construction, local search.

A *dominating walk* of a connected graph G is a sequence of vertices where
consecutive vertices are adjacent (repeats allowed) and every vertex of G is
on the walk or adjacent to a walk vertex. Its length (vertex count, repeats
included) drives the CNOT cost of the walk-based QFT construction, so every
routine here either builds a short walk or shortens one without breaking
coverage.

Provenance: ported from the author's thesis project, where each piece was
validated against exact solvers. The
local search is monotone-safe: output is never longer than the input and
always retains full coverage, else the input is returned unchanged.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Set

import networkx as nx

__all__ = [
    "coverage",
    "covers",
    "walk_valid",
    "complete_walk",
    "improve_walk",
    "improve_walk_strong",
    "dominating_walk",
    "repair_walk",
]


# -- validity / coverage -----------------------------------------------------


def coverage(walk, G) -> set:
    """Closed neighbourhood of the walk: walk vertices plus their neighbours."""
    cov = set()
    for w in walk:
        cov.add(w)
        cov.update(G.neighbors(w))
    return cov


def covers(G: nx.Graph, walk: List[int]) -> bool:
    """True iff the walk dominates every vertex of G."""
    if not walk:
        return False
    return coverage(walk, G) >= set(G.nodes())


def walk_valid(G: nx.Graph, walk: List[int]) -> bool:
    """True iff consecutive walk vertices are adjacent in G."""
    if not walk:
        return False
    if len(walk) == 1:
        return True
    return all(G.has_edge(walk[i], walk[i + 1]) for i in range(len(walk) - 1))


# -- completion (guarantee coverage) ----------------------------------------


def _nearest_useful(G, src, covered):
    """BFS from ``src`` to the nearest vertex whose closed neighbourhood is not
    fully covered yet. Returns ``(vertex, path src->vertex)`` or ``(None, None)``."""
    prev = {src: None}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if not (({u} | set(G.neighbors(u))) <= covered):
            path = []
            x = u
            while x is not None:
                path.append(x)
                x = prev[x]
            return u, path[::-1]
        for v in G.neighbors(u):
            if v not in prev:
                prev[v] = u
                dq.append(v)
    return None, None


def complete_walk(walk: List[int], G) -> List[int]:
    """Extend a partial walk until it dominates G (always possible on a
    connected graph): repeatedly route from the walk's end to the nearest
    vertex that still contributes new coverage."""
    if not walk:
        return walk
    full = set(G.nodes())
    walk = list(walk)
    covered = coverage(walk, G)
    guard = 0
    while not (full <= covered) and guard < 3 * len(full) + 5:
        guard += 1
        tgt, path = _nearest_useful(G, walk[-1], covered)
        if tgt is None or len(path) < 2:
            break
        walk.extend(path[1:])
        for w in path[1:]:
            covered |= {w} | set(G.neighbors(w))
    return walk


# -- local search ------------------------------------------------------------


def _redundancy_removal(walk: List[int], G, full: set) -> List[int]:
    """Splice out interior vertices and trim ends while coverage survives."""
    walk = list(walk)
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(walk) - 1:
            a, x, b = walk[i - 1], walk[i], walk[i + 1]
            if a == b or G.has_edge(a, b):
                cand = walk[:i] + walk[i + 1:]
                if full <= coverage(cand, G):
                    walk = cand
                    changed = True
                    continue
            i += 1
        while len(walk) > 1 and full <= coverage(walk[1:], G):
            walk = walk[1:]
            changed = True
        while len(walk) > 1 and full <= coverage(walk[:-1], G):
            walk = walk[:-1]
            changed = True
    return walk


def _essential_seq(walk: List[int], G) -> List[int]:
    """Walk vertices that contributed *new* coverage, in traversal order,
    consecutive duplicates removed. Their closed neighbourhoods union to full
    coverage by construction, so they can be re-ordered freely."""
    covered: set = set()
    ess: List[int] = []
    for w in walk:
        clo = {w} | set(G.neighbors(w))
        if not ess or (clo - covered):
            ess.append(w)
            covered |= clo
    if not ess:
        return ess
    ded = [ess[0]]
    for w in ess[1:]:
        if w != ded[-1]:
            ded.append(w)
    return ded


def _expand_order(order: List[int], G) -> Optional[List[int]]:
    """Join vertices in the given order by shortest paths -> a walk."""
    if not order:
        return []
    out = [order[0]]
    for a, b in zip(order, order[1:]):
        if a == b:
            continue
        try:
            sp = nx.shortest_path(G, a, b)
        except nx.NetworkXNoPath:
            return None
        out.extend(sp[1:])
    return out


def _optimize_order(ess: List[int], G) -> List[int]:
    """2-opt + Or-opt (segment relocation, both orientations) over the order
    of essential vertices in the shortest-path metric. Coverage is invariant
    to order, so only the tour length is minimized."""
    if len(ess) < 4:
        return ess
    uniq = list(dict.fromkeys(ess))
    dist = {s: nx.single_source_shortest_path_length(G, s) for s in uniq}
    INF = float("inf")

    def seglen(seq):
        return sum(dist[a].get(b, INF) for a, b in zip(seq, seq[1:]))

    best = list(ess)
    cur_len = seglen(best)
    improved = True
    guard = 0
    while improved and guard < 80:
        improved = False
        guard += 1
        L = len(best)
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 2-opt reverses best[i:j+1]; only the two boundary edges
                # change (interior edges reverse but the metric is symmetric),
                # so the length delta is O(1). Distances are integers, so this
                # accepts exactly the same moves as a full recompute.
                old = new = 0
                if i >= 1:
                    a = best[i - 1]
                    old += dist[a].get(best[i], INF)
                    new += dist[a].get(best[j], INF)
                if j <= L - 2:
                    b = best[j + 1]
                    old += dist[best[j]].get(b, INF)
                    new += dist[best[i]].get(b, INF)
                if new - old < 0:
                    best = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    cur_len += new - old
                    improved = True
        moved = True
        while moved:
            moved = False
            L = len(best)
            for seg in (1, 2, 3, 4):
                if seg >= L:
                    break
                for i in range(0, L - seg + 1):
                    segment = best[i:i + seg]
                    rest = best[:i] + best[i + seg:]
                    orients = (segment,) if seg == 1 else (segment, segment[::-1])
                    for piece in orients:
                        for k in range(0, len(rest) + 1):
                            cand = rest[:k] + piece + rest[k:]
                            if cand == best:
                                continue
                            cl = seglen(cand)
                            if cl < cur_len - 1e-9:
                                best, cur_len = cand, cl
                                moved = improved = True
                                break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    break
    return best


def improve_walk(walk: List[int], G, rounds: int = 4) -> List[int]:
    """Monotone-safe shortening: essential extraction -> shortest-path
    rerouting -> redundancy removal, iterated. Output covers G and is never
    longer than the input; otherwise the input is returned."""
    if not walk:
        return walk
    full = set(G.nodes())
    if not (full <= coverage(walk, G)):
        return walk
    best = list(walk)
    for _ in range(rounds):
        cur = best
        covered = set()
        ess: List[int] = []
        for w in cur:
            clo = {w} | set(G.neighbors(w))
            if not ess or (clo - covered):
                ess.append(w)
                covered |= clo
        ded = [ess[0]]
        for w in ess[1:]:
            if w != ded[-1]:
                ded.append(w)
        ess = ded
        rer = [ess[0]]
        ok = True
        for a, b in zip(ess, ess[1:]):
            if a == b:
                continue
            try:
                sp = nx.shortest_path(G, a, b)
            except nx.NetworkXNoPath:
                ok = False
                break
            rer.extend(sp[1:])
        if not ok:
            break
        rer = _redundancy_removal(rer, G, full)
        if (full <= coverage(rer, G)) and len(rer) < len(best):
            best = rer
        else:
            break
    return best


def improve_walk_strong(walk: List[int], G, rounds: int = 4) -> List[int]:
    """``improve_walk`` plus 2-opt/Or-opt over the essential-vertex order.
    Monotone-safe like ``improve_walk``. This is the main quality lever of the
    pipeline (measured ~-7.6% walk cost in the source project)."""
    if not walk:
        return walk
    full = set(G.nodes())
    if not (full <= coverage(walk, G)):
        return walk
    base = improve_walk(walk, G, rounds=rounds)
    ess = _essential_seq(base, G)
    if len(ess) >= 4:
        new_ord = _optimize_order(ess, G)
        cand = _expand_order(new_ord, G)
        if cand is not None:
            cand = _redundancy_removal(cand, G, full)
            if (full <= coverage(cand, G)) and len(cand) < len(base):
                base = improve_walk(cand, G, rounds=rounds)
    return base


# -- model-free construction -------------------------------------------------


def _greedy_cost_dom(G, dist, nbr, full, lam: float, first=None) -> List[int]:
    """Cost-aware greedy dominating set: each step takes the vertex maximizing
    (new coverage) / (1 + lam * distance to the current set). ``lam > 0``
    penalizes far-away hubs, keeping the set cheap to stitch into a tour.
    ``first`` forces the initial pick (multi-restart diversification)."""
    INF = 1 << 30
    nodes = list(G.nodes())
    covered: set = set()
    dom: List[int] = []
    dmin = {v: INF for v in nodes}
    while covered < full:
        if not dom and first is not None:
            best_v = first
        else:
            best_v, best_score = None, -1.0
            for v in nodes:
                if dmin[v] == 0:
                    continue
                gain = len(nbr[v] - covered)
                if gain <= 0:
                    continue
                denom = 1.0 + lam * (dmin[v] if dom else 0.0)
                score = gain / denom
                if score > best_score:
                    best_score, best_v = score, v
            if best_v is None:
                break
        dom.append(best_v)
        covered |= nbr[best_v]
        dv = dist[best_v]
        for v in nodes:
            d = dv.get(v, INF)
            if d < dmin[v]:
                dmin[v] = d
    return dom


def dominating_walk(G, lam_grid=(0.0, 0.5, 1.0), restarts: int = 4) -> Optional[List[int]]:
    """Model-free dominating walk: cost-aware greedy dominating set (grid over
    the distance penalty ``lam``, multi-restart over the initial hub), ordered
    by the 2-opt/Or-opt tour optimizer, stitched with shortest paths, then
    redundancy-cleaned. Returns the shortest covering walk found or None."""
    full = set(G.nodes())
    if len(full) <= 1:
        return list(full)
    dist = dict(nx.all_pairs_shortest_path_length(G))
    nbr = {v: ({v} | set(G.neighbors(v))) for v in G.nodes()}
    deg_order = sorted(G.nodes(), key=lambda v: -len(nbr[v]))
    firsts = [None] + deg_order[: max(0, restarts - 1)]
    best_walk = None
    for lam in lam_grid:
        for first in firsts:
            dom = _greedy_cost_dom(G, dist, nbr, full, lam, first=first)
            if not dom:
                continue
            order = _optimize_order(dom, G) if len(dom) >= 4 else dom
            w = _expand_order(order, G)
            if w is None:
                continue
            w = _redundancy_removal(w, G, full)
            if (full <= coverage(w, G)) and (
                best_walk is None or len(w) < len(best_walk)
            ):
                best_walk = w
    return best_walk


# -- warm-start repair -------------------------------------------------------


def repair_walk(walk: List[int], G: nx.Graph) -> Optional[List[int]]:
    """Adapt a walk to a graph that lost some vertices (full-QFT cascades
    remove one qubit per cascade): drop missing vertices from the essential
    order, restitch with shortest paths in the new graph, complete coverage if
    needed, then run the strong local search. Returns a valid covering walk of
    G or None."""
    if not walk:
        return None
    present = [v for v in walk if G.has_node(v)]
    if not present:
        return None
    ess = _essential_seq(present, G)  # order survives; gaps restitched below
    cand = _expand_order(ess, G)
    if cand is None:
        return None
    cand = complete_walk(cand, G)
    full = set(G.nodes())
    if not (full <= coverage(cand, G)):
        return None
    return improve_walk_strong(cand, G)
