"""Brute-force reference implementations for property tests.

Written independently of the library under test: ESU enumeration of all
connected k-subsets and an exact BFS dominating walk inside a fixed region.
``best_region_walk_bruteforce`` is the ground truth the budgeted-coverage
search must match exactly.
"""
from __future__ import annotations

from collections import deque
from typing import FrozenSet, List, Optional, Set, Tuple

import networkx as nx


def connected_ksets(G: nx.Graph, k: int, cap: int = 500_000) -> Tuple[List[FrozenSet], bool]:
    """All connected k-subsets of V(G), each exactly once (ESU/FANMOD)."""
    order = {v: i for i, v in enumerate(sorted(G.nodes()))}
    out: List[FrozenSet] = []

    def extend(sub: Set, ext: Set, root) -> bool:
        if len(sub) == k:
            out.append(frozenset(sub))
            return len(out) <= cap
        ext = set(ext)
        while ext:
            w = ext.pop()
            new_ext = ext | {
                u
                for u in G.neighbors(w)
                if order[u] > order[root]
                and u not in sub
                and all(u not in G[x] for x in sub)
            }
            sub.add(w)
            if not extend(sub, new_ext, root):
                return False
            sub.remove(w)
        return True

    for v in sorted(G.nodes(), key=lambda x: order[x]):
        ext0 = {u for u in G.neighbors(v) if order[u] > order[v]}
        if not extend({v}, ext0, v):
            return out, True
    return out, False


def exact_region_walk_len(G: nx.Graph, S) -> Optional[int]:
    """Exact shortest dominating-walk length (vertices) inside G[S]: BFS over
    (vertex, coverage-mask) states, unit steps."""
    nodes = list(S)
    k = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    sset = set(nodes)
    adj: List[List[int]] = [[] for _ in range(k)]
    nb = [0] * k
    for v in nodes:
        i = idx[v]
        m = 1 << i
        for u in G.neighbors(v):
            if u in sset:
                j = idx[u]
                adj[i].append(j)
                m |= 1 << j
        nb[i] = m
    full = (1 << k) - 1
    seen = [dict() for _ in range(k)]
    dq: deque = deque()
    for i in range(k):
        if nb[i] == full:
            return 1
        seen[i][nb[i]] = 1
        dq.append((i, nb[i], 1))
    while dq:
        i, m, d = dq.popleft()
        for j in adj[i]:
            m2 = m | nb[j]
            if m2 == full:
                return d + 1
            if m2 not in seen[j]:
                seen[j][m2] = d + 1
                dq.append((j, m2, d + 1))
    return None  # G[S] disconnected


def best_region_walk_bruteforce(G: nx.Graph, k: int) -> Optional[int]:
    """Ground truth: min over ALL connected k-regions of the exact inner
    dominating-walk length."""
    sets, overflow = connected_ksets(G, k)
    assert not overflow, "reference enumeration overflowed; shrink the test"
    best = None
    for S in sets:
        L = exact_region_walk_len(G, S)
        if L is not None and (best is None or L < best):
            best = L
    return best
