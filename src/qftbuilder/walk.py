"""Dominating walks: validity, model-free construction, local search.

A *dominating walk* of a connected graph G is a sequence of vertices where
consecutive vertices are adjacent (repeats allowed) and every vertex of G is
on the walk or adjacent to a walk vertex. Its length (vertex count, repeats
included) drives the CNOT cost of the walk-based QFT construction, so every
routine here either builds a short walk or shortens one without breaking
coverage.

Provenance: ported from the author's thesis project, where each piece was
validated against exact solvers. The local search is monotone-safe: output is
never longer than the input and always retains full coverage, else the input
is returned unchanged.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Set

import networkx as nx

from .cost import EDGE_WEIGHT, walk_cost, weighted_sweep_cost

__all__ = [
    "coverage",
    "covers",
    "walk_valid",
    "complete_walk",
    "improve_walk",
    "improve_walk_strong",
    "dominating_walk",
    "walk_ending_at",
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


def _redundancy_removal(walk: List[int], G, full: set,
                        keep_last: bool = False) -> List[int]:
    """Splice out interior vertices and trim ends while coverage survives.

    ``keep_last`` protects the final vertex: callers that need the walk to end
    somewhere specific (the QFT cascade must finish on the qubit it finalizes)
    would otherwise have the endpoint trimmed away here."""
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
        while (not keep_last) and len(walk) > 1 and full <= coverage(walk[:-1], G):
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


def _expand_order(order: List[int], G,
                  weight: Optional[str] = None) -> Optional[List[int]]:
    """Join vertices in the given order by minimum-cost paths -> a walk."""
    if not order:
        return []
    out = [order[0]]
    for a, b in zip(order, order[1:]):
        if a == b:
            continue
        try:
            sp = _one_path(G, a, b, weight)
        except nx.NetworkXNoPath:
            return None
        out.extend(sp[1:])
    return out


_TOL = 1e-9


def _edge_len(G, u, v, weight: Optional[str]) -> float:
    if weight is None:
        return 1.0
    return float(G[u][v].get(weight, 1.0))


def _dist_from(G, s, weight: Optional[str]):
    """Single-source distances in the chosen metric (BFS or Dijkstra)."""
    if weight is None:
        return nx.single_source_shortest_path_length(G, s)
    return nx.single_source_dijkstra_path_length(G, s, weight=weight)


def _one_path(G, a, b, weight: Optional[str]) -> List[int]:
    if weight is None:
        return nx.shortest_path(G, a, b)
    return nx.dijkstra_path(G, a, b, weight=weight)


def _shortest_path_max_new(G, a, b, used,
                           weight: Optional[str] = None) -> Optional[List[int]]:
    """Among all *minimum-cost* ``a -> b`` paths, one maximizing the count of
    vertices outside ``used`` (the endpoint ``a`` is excluded, the previous
    segment already contributed it).

    Exact per segment: the minimum-cost paths are exactly the paths of the DAG
    ``{(v,w) : D(w) + w(v,w) = D(v)}`` for ``D(v) = d(v,b)``, along which the
    vertex set is additive, so backward DP in order of increasing ``D`` is
    optimal (docs/proofs.md, Lemma 3.6). Choosing among them never changes the
    path cost (Lemma 3.4), so this buys distinct (black) vertices for free.
    With ``weight`` set the same argument runs in the weighted metric
    (section 5), where ``D`` comes from Dijkstra."""
    if a == b:
        return [a]
    D = _dist_from(G, b, weight)
    da = D.get(a)
    if da is None:
        return None
    best = {b: (0 if b in used else 1)}
    choice: dict = {}
    for v, d in sorted(D.items(), key=lambda kv: kv[1]):
        if d > da + _TOL:
            break
        if v == b:
            continue
        bw = None
        bv = None
        for w in G.neighbors(v):
            dw = D.get(w)
            if dw is None or dw >= d:
                continue
            if abs(dw + _edge_len(G, v, w, weight) - d) > _TOL:
                continue
            val = best.get(w)
            if val is not None and (bw is None or val > bw):
                bw, bv = val, w
        if bw is None:
            continue
        best[v] = bw + (0 if v in used else 1)
        choice[v] = bv
    if a not in choice:
        return None
    path = [a]
    cur = a
    while cur != b:
        cur = choice[cur]
        path.append(cur)
    return path


def _expand_order_rich(order: List[int], G,
                       weight: Optional[str] = None) -> Optional[List[int]]:
    """``_expand_order`` with distinct-maximizing segment choice. Produces a
    walk of exactly the same cost as any other minimum-cost stitching of the
    same order, with at least as many distinct vertices."""
    if not order:
        return []
    out = [order[0]]
    used = {order[0]}
    for a, b in zip(order, order[1:]):
        if a == b:
            continue
        sp = _shortest_path_max_new(G, a, b, used, weight)
        if sp is None:
            return None
        out.extend(sp[1:])
        used.update(sp)
    return out


def _walk_key(walk: Optional[List[int]], G, objective: str,
              region_size: Optional[int] = None,
              weight: Optional[str] = None):
    """Ranking key, smaller is better. ``objective="length"`` ranks by length
    with the true CNOT cost as tie-break (so equal-length walks with more
    black vertices win -- free by Corollary 3.2); ``objective="cnot"`` ranks by
    the exact single-sweep cost ``2*whites + 3*moves`` with length as
    tie-break; ``objective="fidelity"`` ranks by the weighted account of
    docs/proofs.md section 5, with the unweighted cost as tie-break."""
    if not walk:
        return (float("inf"), float("inf"))
    s = G.number_of_nodes() if region_size is None else region_size
    c = walk_cost(walk, s)
    if objective == "fidelity":
        wc = weighted_sweep_cost(walk, G, region=set(G.nodes()),
                                 weight=weight or EDGE_WEIGHT)
        return (wc, float(c))
    if objective == "cnot":
        return (float(c), float(len(walk)))
    return (float(len(walk)), float(c))


def _optimize_order(ess: List[int], G, weight: Optional[str] = None,
                    pin_last: bool = False) -> List[int]:
    """2-opt + Or-opt (segment relocation, both orientations) over the order
    of essential vertices in the shortest-path metric (Dijkstra when
    ``weight`` is given). Coverage is invariant to order, so only the tour
    cost is minimized.

    ``pin_last`` forbids every move that would displace the final anchor, so
    the order still ends where the caller demands while the rest is optimized
    against it (the transition into the pinned vertex stays part of the cost --
    optimizing the prefix alone would ignore it)."""
    if len(ess) < 4:
        return ess
    uniq = list(dict.fromkeys(ess))
    dist = {s: _dist_from(G, s, weight) for s in uniq}
    INF = float("inf")
    # The O(1) 2-opt delta is exact for integer distances (Theorem 1); under
    # real weights the same delta is used with a tolerance, since "strictly
    # cheaper" can no longer be read off as "< 0" on floats.
    eps = 0.0 if weight is None else _TOL

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
        last = L - 1 if pin_last else L      # 2-opt may not touch a pinned tail
        for i in range(0, L - 1):
            for j in range(i + 1, last):
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
                if new - old < -eps:
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
                # a pinned tail may neither be relocated nor overtaken
                span = L - seg + (0 if pin_last else 1)
                for i in range(0, span):
                    segment = best[i:i + seg]
                    rest = best[:i] + best[i + seg:]
                    orients = (segment,) if seg == 1 else (segment, segment[::-1])
                    slots = len(rest) + (0 if pin_last else 1)
                    for piece in orients:
                        for k in range(0, slots):
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


def improve_walk(walk: List[int], G, rounds: int = 4,
                 weight: Optional[str] = None) -> List[int]:
    """Monotone-safe shortening: essential extraction -> minimum-cost
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
                sp = _one_path(G, a, b, weight)
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


def improve_walk_strong(walk: List[int], G, rounds: int = 4,
                        objective: str = "length",
                        region_size: Optional[int] = None,
                        weight: Optional[str] = None) -> List[int]:
    """``improve_walk`` plus 2-opt/Or-opt over the essential-vertex order and a
    distinct-maximizing re-stitch. Monotone-safe like ``improve_walk``: the
    result is valid, covering, and never worse than the input under the chosen
    ranking (Theorem 3 in docs/proofs.md).

    ``objective="length"`` (default) minimizes the walk length and uses the
    exact CNOT cost only to break ties, so the output is never *longer* than
    the input. ``objective="cnot"`` minimizes ``2*whites + 3*moves`` directly:
    it may return a slightly longer walk that visits more distinct vertices and
    is therefore strictly cheaper (Lemma 3.1). ``objective="fidelity"``
    minimizes the weighted account of section 5 and routes in the weighted
    metric, steering the walk away from bad couplings."""
    if not walk:
        return walk
    full = set(G.nodes())
    if not (full <= coverage(walk, G)):
        return walk
    if objective == "fidelity" and weight is None:
        weight = EDGE_WEIGHT

    def key(w):
        return _walk_key(w, G, objective, region_size, weight)

    base = improve_walk(walk, G, rounds=rounds, weight=weight)
    if objective != "length" and key(walk) < key(base):
        base = list(walk)

    # Free win: re-stitch the current essentials so that segments pick up as
    # many new vertices as possible. Same cost of the stitch (Lemma 3.4),
    # never fewer distinct vertices, hence never a worse total.
    ess = _essential_seq(base, G)
    rich = _expand_order_rich(ess, G, weight) if ess else None
    if rich is not None:
        rich = _redundancy_removal(rich, G, full)
        if (full <= coverage(rich, G)) and key(rich) < key(base):
            base = rich

    if len(ess) >= 4:
        new_ord = _optimize_order(ess, G, weight)
        for stitch in (_expand_order_rich, _expand_order):
            cand = stitch(new_ord, G, weight)
            if cand is None:
                continue
            cand = _redundancy_removal(cand, G, full)
            if (full <= coverage(cand, G)) and key(cand) < key(base):
                polished = improve_walk(cand, G, rounds=rounds, weight=weight)
                base = polished if key(polished) < key(cand) else cand
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


def dominating_walk(G, lam_grid=(0.0, 0.5, 1.0), restarts: int = 4,
                    objective: str = "length",
                    weight: Optional[str] = None) -> Optional[List[int]]:
    """Model-free dominating walk: cost-aware greedy dominating set (grid over
    the distance penalty ``lam``, multi-restart over the initial hub), ordered
    by the 2-opt/Or-opt tour optimizer, stitched with shortest paths chosen to
    maximize distinct vertices, then redundancy-cleaned. Returns the best
    covering walk found under ``objective`` (see :func:`improve_walk_strong`),
    or None."""
    full = set(G.nodes())
    if len(full) <= 1:
        return list(full)
    if objective == "fidelity" and weight is None:
        weight = EDGE_WEIGHT
    dist = {s: _dist_from(G, s, weight) for s in G.nodes()}
    nbr = {v: ({v} | set(G.neighbors(v))) for v in G.nodes()}
    deg_order = sorted(G.nodes(), key=lambda v: -len(nbr[v]))
    firsts = [None] + deg_order[: max(0, restarts - 1)]
    best_walk = None
    best_key = None
    for lam in lam_grid:
        for first in firsts:
            dom = _greedy_cost_dom(G, dist, nbr, full, lam, first=first)
            if not dom:
                continue
            order = _optimize_order(dom, G, weight) if len(dom) >= 4 else dom
            for stitch in (_expand_order_rich, _expand_order):
                w = stitch(order, G, weight)
                if w is None:
                    continue
                w = _redundancy_removal(w, G, full)
                if not (full <= coverage(w, G)):
                    continue
                kw = _walk_key(w, G, objective, weight=weight)
                if best_key is None or kw < best_key:
                    best_walk, best_key = w, kw
    return best_walk


def walk_ending_at(G, end, base: Optional[List[int]] = None,
                   objective: str = "cnot", weight: Optional[str] = None,
                   rounds: int = 4) -> Optional[List[int]]:
    """Best dominating walk of ``G`` that **finishes at** ``end``.

    The QFT cascade finalizes the qubit its carrier lands on, so the endpoint
    is a real constraint rather than a by-product. Repairing a finished walk
    (reverse it, or route to the nearest legal vertex) costs one extra CNOT per
    added step; searching under the constraint instead lets the anchor order
    account for the endpoint from the start.

    Returns None if no covering walk ending at ``end`` was found."""
    full = set(G.nodes())
    if len(full) == 1:
        return [end]
    if base is None:
        base = dominating_walk(G, objective=objective, weight=weight)
    if not base:
        return None
    if objective == "fidelity" and weight is None:
        weight = EDGE_WEIGHT

    cands: List[List[int]] = []
    for w in (base, base[::-1]):            # free if an endpoint already fits
        if w and w[-1] == end and covers(G, w):
            cands.append(list(w))

    ess = _essential_seq(base, G)
    ess = [v for v in ess if v != end] + [end]
    order = _optimize_order(ess, G, weight, pin_last=True)
    for stitch in (_expand_order_rich, _expand_order):
        w = stitch(order, G, weight)
        if w is None or w[-1] != end:
            continue
        w = _redundancy_removal(w, G, full, keep_last=True)
        if w and w[-1] == end and walk_valid(G, w) and (full <= coverage(w, G)):
            cands.append(w)

    if not cands:
        return None
    return min(cands, key=lambda w: _walk_key(w, G, objective, weight=weight))


# -- warm-start repair -------------------------------------------------------


def repair_walk(walk: List[int], G: nx.Graph, objective: str = "length",
                weight: Optional[str] = None) -> Optional[List[int]]:
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
    if objective == "fidelity" and weight is None:
        weight = EDGE_WEIGHT
    ess = _essential_seq(present, G)  # order survives; gaps restitched below
    cand = _expand_order(ess, G, weight)
    if cand is None:
        return None
    cand = complete_walk(cand, G)
    full = set(G.nodes())
    if not (full <= coverage(cand, G)):
        return None
    return improve_walk_strong(cand, G, objective=objective, weight=weight)
