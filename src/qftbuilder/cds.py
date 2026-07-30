"""Polynomial fallback: greedy Connected Dominating Set + DFS Euler tour.

The classic two-step approximation (Guha & Khuller greedy CDS, then an Euler
tour of the CDS spanning tree). Always produces a valid dominating walk on a
connected graph, with the textbook ``2(ln Delta + 3) * OPT`` length guarantee.
It is the cascade's guaranteed floor: every stronger rung must beat it or the
selector keeps this one.
"""
from __future__ import annotations

from typing import List, Set

import networkx as nx

__all__ = ["greedy_cds", "cds_dfs_path"]


def greedy_cds(G: nx.Graph) -> Set[int]:
    """Greedy connected dominating set (Guha-Khuller style): grow from the
    max-degree vertex, always adding the 'gray' vertex (adjacent to the
    current CDS) that covers the most still-white vertices."""
    nodes = list(G.nodes())
    if len(nodes) <= 1:
        return set(nodes)

    deg = dict(G.degree())
    start = max(deg, key=lambda v: deg[v])
    cds: Set[int] = {start}
    dominated: Set[int] = {start} | set(G[start])
    gray: Set[int] = set(G[start])

    white = set(nodes) - dominated
    while white:
        best = max(gray, key=lambda v: (len(set(G[v]) & white), deg[v]))
        cds.add(best)
        gray.discard(best)
        newly = set(G[best]) & white
        white -= newly
        dominated |= newly | {best}
        gray |= set(G[best]) - cds
    return cds


def _euler_tour(tree: nx.Graph, start: int) -> List[int]:
    """DFS Euler tour of a tree (each vertex on entry and on every return to
    the parent): length 2|V|-1 for |V| >= 2. Consecutive vertices are adjacent
    tree edges, so the tour is a valid walk."""
    walk: List[int] = []
    visited: Set[int] = set()
    stack = [(start, iter(sorted(tree[start])))]
    visited.add(start)
    walk.append(start)
    while stack:
        node, it = stack[-1]
        advanced = False
        for nxt in it:
            if nxt not in visited:
                visited.add(nxt)
                walk.append(nxt)
                stack.append((nxt, iter(sorted(tree[nxt]))))
                advanced = True
                break
        if not advanced:
            stack.pop()
            if stack:
                walk.append(stack[-1][0])
    return walk


def cds_dfs_path(G: nx.Graph) -> List[int]:
    """Dominating walk via greedy CDS + Euler tour of its spanning tree."""
    nodes = list(G.nodes())
    if len(nodes) == 1:
        return [nodes[0]]
    cds = greedy_cds(G)
    sub = G.subgraph(cds)
    tree = nx.minimum_spanning_tree(sub)
    start = max(cds, key=lambda v: G.degree(v))
    return _euler_tour(tree, start)
