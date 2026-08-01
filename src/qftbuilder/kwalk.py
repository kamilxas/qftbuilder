"""Budgeted-coverage walks: the sub-QFT region problem without region search.

Problem. A QFT on ``k < n`` qubits needs a connected region ``S`` of exactly
``k`` vertices and a walk ``W`` inside ``G[S]`` dominating ``S``; the walk
length ``|W|`` (vertices, repeats counted) drives the CNOT cost. Both ``S``
and ``W`` are free — the method chooses them.

Reduction (the key idea of this module). The pair ``(S, W)`` is equivalent to
a *single walk with a coverage budget*:

    minimize |W| over walks W in G   s.t.   |N_G[V(W)]| >= k,

where ``V(W)`` is the set of distinct (black) walk vertices and ``N[.]`` the
closed neighbourhood; covered off-walk vertices are the whites. Given such a
walk, ``S := V(W)`` plus any ``k - |V(W)|`` whites is a valid region
(connected, dominated by W, walk inside it); conversely any feasible
``(S, W)`` has ``S ⊆ N[V(W)]`` and ``V(W) ⊆ S``. The region is a free
by-product of the walk — no enumeration of connected k-subsets needed.

Search. Breadth-first over states ``(last vertex, V(W) bitmask)`` by walk
length with global dedup; the first state whose coverage popcount reaches
``k`` on a *fully explored* level is provably optimal. States with coverage
``< k`` automatically have ``|V(W)| < k``, so the budget constraint needs no
explicit enforcement.

Complexity contract, with derivation (the problem is NP-hard at ``k = n``,
so nothing here is an exact polynomial algorithm; the contract is about
*budgets*, and the yardstick is the whole-graph solve, whose local search is
Theta(n^3)-scale):

1. **Work accounting.** Let ``B`` be the total-state budget, ``Delta`` the
   maximum degree, ``w = ceil(n/64)`` the machine words per n-bit vertex
   mask. Every stored state is inserted into the parent map exactly once
   (global dedup), so insertions <= B in the exact phase plus ``B/4`` in the
   beam tail. Each stored state is expanded at most once, probing <= Delta
   neighbours; a probe hashes one ``(vertex, mask)`` key and, when new, ORs
   one coverage mask — ``O(w)`` word operations. Hence total work

       W  <=  (5/4) * B * Delta * O(w)  =  O(B * Delta * n / 64).

2. **Choosing B so the constant is a genuine constant.** We fix a *work
   target* ``W* = WORK_FACTOR * n^3`` word operations (``WORK_FACTOR = 27``
   — a user-approved constant, independent of ``k`` and of ``Delta``) and
   solve step 1 for B:

       B  =  W* * (4/5) * 64 / (Delta * n)  =  1382 * n^2 / Delta.

   ``Delta`` cancels out of the work bound by construction (denser graphs
   simply get a smaller state budget), and ``k`` never appears — one budget
   serves every ``k``, so the multiplier of ``n^3`` cannot silently grow
   into an extra factor of ``n``. A small floor (``BUDGET_FLOOR=200_000``)
   covers tiny graphs at trivial absolute cost, and an absolute clip
   (``BUDGET_CLIP=8_000_000`` states) guards memory — clipping only lowers
   work, so the ``27 n^3`` bound stands.

   **What actually consumes the budget is the optimality PROOF, not the
   method.** A good (empirically optimal-length) walk is produced by the
   beam tail / whole-graph fallback in ~1 MB; the millions of states exist
   only to *prove by exhaustion* that nothing shorter exists. Measured on
   grid n=144: at k=28/32/36 the heuristic answer and the exhaustively
   proven answer have the *same length* (9/10/12), yet the proof costs
   ~540 MB / ~1.5 GB / ~2.0 GB (each stored state is ~220 bytes in Python).
   So the memory is the price of ``region_proven=True`` at large k, and it
   buys a certificate, not a shorter walk. Callers who only need the walk
   can pass a small ``state_budget``; callers who need a certificate more
   cheaply should prefer the budgeted MILP
   (:func:`qftbuilder.milp.shortest_budgeted_walk_milp`), whose LP/B&B dual
   proves optimality without enumerating states.

3. **Exactness against the budget.** Answers absorbed from *fully explored* BFS
   levels are provably optimal (BFS by length). If the budget trips
   mid-level, the search downgrades honestly: a coverage-greedy beam tail
   (width ``max(4096, 64 n)``, insertion budget ``B/4``) continues for
   mid-range k, and finally every still-open k is answered by the shortest
   *covering subwalk* of one whole-graph dominating walk — a single
   ``O(L^2)`` scan (L <= 2n) answers all budgets at once. Therefore

       cost(sub_qft)  <=  27 * n^3  +  cost(whole-graph solve)  +  O(n^2)

   word operations unconditionally (constant 27 by design, never k or
   Delta), and results state their provenance:
   ``region_proven`` plus ``method in {"bfs", "subwalk", "solver"}``.

4. **FPT regime.** Before any budget bites, reachable states are bounded by
   ``n * b* * (e*Delta)**(b*-1)`` with ``b* ~ k/Delta`` the optimum length
   (connected-subset counting) — linear in ``n`` for fixed ``k``. Measured
   on degree<=4 hardware graphs: k <= 20 answered exactly in milliseconds
   even at n ~ 150; the default budget extends proven answers to
   ``k ~ 28-36`` at seconds (millions of states), a few times one full
   solve in the worst mid-k case.

Exactness beyond the default is available by passing a larger
``state_budget``, and independently via the budgeted MILP certificate
(:func:`qftbuilder.milp.shortest_budgeted_walk_milp`).

Replaces the prior three-stage region search (whose exact stage was
exponential in ``k`` itself); property-tested against brute-force region
enumeration.
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

from .cost import cnot_upper_bound, walk_cost, weighted_sweep_cost

__all__ = ["budgeted_walk", "budgeted_sweep", "max_coverage_walk"]

WORK_FACTOR = 27                # target work: WORK_FACTOR * n^3 WORD ops --
                                # a genuine constant (user-approved 27):
                                # independent of k AND of the degree
                                # (Delta cancels in _budgets below)
BUDGET_FLOOR = 200_000          # small-n floor; absolute work stays trivial
BUDGET_CLIP = 8_000_000         # absolute RAM guard (~1.2 GB of states); a
                                # clip can only LOWER work, so the 27*n^3
                                # bound is untouched
BEAM_WIDTH_FLOOR = 4096
BEAM_WIDTH_PER_N = 64           # beam-tail width, same width/n as the decoders
TAIL_FRACTION = 4               # beam tail may insert at most budget/4 states
LEGACY_BUDGET_FACTOR = 25       # state_cap alias: budget = 25*cap, width = cap
_MAX_DEPTH_FACTOR = 2           # a dominating walk needs < ~2n vertices

# Derivation of the state budget (see module docstring, step 2):
#   work  W <= (5/4) * B * Delta * ceil(n/64)   word operations,
# so to guarantee W <= WORK_FACTOR * n^3 we set
#   B  =  WORK_FACTOR * (4/5) * 64 * n^2 / Delta  =  1382 * n^2 / Delta.
# Delta cancels out of the work bound by construction: denser graphs get a
# smaller state budget, and k never enters (one budget serves every k).
_BUDGET_NUM = (WORK_FACTOR * 4 * 64) // 5   # = 1382

# Pareto pruning: closed states remembered per last vertex. The check is
# O(PARETO_KEEP) big-int ANDs per generated state; checking a subset of the
# stored states is sound (docs/proofs.md, Remark 4.4), so this trades pruning
# power against a bounded constant, never correctness. Measured: it pays only
# where coverage saturates (k near n, where many distinct visited-sets share a
# coverage) and costs ~2x elsewhere -- hence the probe below switches it off
# on instances where it is not earning its keep.
PARETO_KEEP = 8
PARETO_PROBE = 20_000       # generated states before judging the prune rate
PARETO_MIN_RATE = 0.02      # keep pruning only above this hit rate

# Automorphism-orbit reduction of the start states (Lemma 4.5). Both caps make
# the enumeration cut off early; a partial automorphism set only refines the
# orbits, so the reduction stays sound either way.
ORBIT_CAP_MAPS = 64
ORBIT_CAP_SECS = 1.0
ORBIT_MIN_N = 12            # below this the search is trivial anyway


def _budgets(n: int, max_deg: int, state_budget: Optional[int],
             state_cap: Optional[int]) -> Tuple[int, int]:
    """-> (total state budget, beam-tail width)."""
    if state_budget is not None:
        return state_budget, max(BEAM_WIDTH_FLOOR, BEAM_WIDTH_PER_N * n)
    if state_cap is not None:  # legacy alias (old per-level-cap semantics)
        return LEGACY_BUDGET_FACTOR * state_cap, state_cap
    return (
        min(BUDGET_CLIP,
            max(BUDGET_FLOOR, (_BUDGET_NUM * n * n) // max(max_deg, 1))),
        max(BEAM_WIDTH_FLOOR, BEAM_WIDTH_PER_N * n),
    )


def _prepare(G: nx.Graph):
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj: List[List[int]] = [[] for _ in range(n)]
    nbmask = [0] * n
    for v in nodes:
        i = idx[v]
        m = 1 << i
        for u in G.neighbors(v):
            j = idx[u]
            adj[i].append(j)
            m |= 1 << j
        nbmask[i] = m
    return nodes, idx, adj, nbmask


def _orbit_reps(G: nx.Graph, idx, cap_maps: int = ORBIT_CAP_MAPS,
                cap_secs: float = ORBIT_CAP_SECS) -> List[int]:
    """One start-vertex index per orbit of a *subset* of ``Aut(G)``.

    Only the starts need reducing: by Lemma 4.5 (docs/proofs.md) an optimal
    walk starting anywhere maps, under an automorphism, to an equally long
    optimal walk starting at the orbit representative. Enumerating only part
    of ``Aut(G)`` merely *refines* the orbits, so a cut-off search stays sound
    -- it only forfeits reduction. On failure this returns every vertex, i.e.
    the unreduced search.

    Measured cost: 0.03-0.16 s up to n=144, against 2.0x (heavy-hex) to 6.9x
    (square lattice) fewer start states."""
    n = len(idx)
    try:
        from networkx.algorithms.isomorphism import GraphMatcher
    except Exception:                      # pragma: no cover - networkx always has it
        return list(range(n))
    par = list(range(n))

    def find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    t0 = time.perf_counter()
    seen = 0
    try:
        for mapping in GraphMatcher(G, G).isomorphisms_iter():
            seen += 1
            for u, v in mapping.items():
                a, b = find(idx[u]), find(idx[v])
                if a != b:
                    par[a] = b
            if seen >= cap_maps or time.perf_counter() - t0 > cap_secs:
                break
    except Exception:                      # pragma: no cover - defensive
        return list(range(n))
    return sorted({find(i) for i in range(n)})


def _reconstruct(parent, key) -> List[int]:
    seq = []
    while key is not None:
        seq.append(key[0])
        key = parent[key]
    seq.reverse()
    return seq


def _fill_region(walk_idx: List[int], cov: int, k: int, n: int,
                 white_cost: Optional[List[float]] = None) -> List[int]:
    """Region = black (walk) vertices + white fillers.

    Any ``k - |blacks|`` covered vertices form a valid region, so the choice is
    free; ``white_cost[i]`` (cheapest edge from ``i`` to a black neighbour)
    makes it *cheapest-first*, which is optimal for the white term of the
    weighted account since the per-white costs are independent
    (docs/proofs.md, Proposition 5.2). Without weights it keeps the previous
    lowest-index order."""
    S = sorted(set(walk_idx))
    if len(S) >= k:
        return S[:k]  # cannot happen for optimal walks; defensive
    have = set(S)
    whites = [i for i in range(n) if (cov >> i) & 1 and i not in have]
    if white_cost is not None:
        whites.sort(key=lambda i: white_cost[i])
    for i in whites:
        if len(S) == k:
            break
        S.append(i)
    return sorted(S)


def _result(nodes, walk_idx, cov, k, n, proven, method,
            white_cost: Optional[List[float]] = None) -> Dict:
    walk = [nodes[i] for i in walk_idx]
    S_idx = _fill_region(walk_idx, cov, k, n, white_cost)
    subset = [nodes[i] for i in S_idx]
    return {
        "k": k,
        "subset": subset,
        "walk": walk,
        "k_path": len(walk),
        "cost": cnot_upper_bound(len(walk), k),
        "cost_sweep": walk_cost(walk, k),
        "region_proven": bool(proven),
        "method": method,
    }


def _white_costs(G, nodes, idx, walk_idx, weight: str) -> List[float]:
    """For every vertex, the cheapest edge to a black (walk) vertex; infinite
    when it has none. Used to pick the cheapest whites into the region."""
    blacks = {nodes[i] for i in walk_idx}
    out = [float("inf")] * len(nodes)
    for v in G.nodes():
        best = None
        for u in G.neighbors(v):
            if u in blacks:
                w = float(G[u][v].get(weight, 1.0))
                if best is None or w < best:
                    best = w
        if best is not None:
            out[idx[v]] = best
    return out


def _subwalk_answers(walk_idx: List[int], nbmask, ks: Sequence[int]):
    """For every k, the shortest *contiguous subwalk* of ``walk_idx`` whose
    coverage reaches k. One O(L^2) scan answers all budgets at once.
    Returns {k: (length, start, end, cov)} for the ks it can satisfy."""
    L = len(walk_idx)
    ks_sorted = sorted(ks)
    best: Dict[int, tuple] = {}
    for i in range(L):
        cov = 0
        ptr = 0
        for j in range(i, L):
            cov |= nbmask[walk_idx[j]]
            pc = cov.bit_count()
            length = j - i + 1
            # for a fixed start i, length grows with j, so each k is best at
            # the first j reaching it: relax only the newly reachable ks
            while ptr < len(ks_sorted) and ks_sorted[ptr] <= pc:
                k = ks_sorted[ptr]
                cur = best.get(k)
                if cur is None or length < cur[0]:
                    best[k] = (length, i, j, cov)
                ptr += 1
            if ptr == len(ks_sorted):
                break
    return best


def _move_costs(G, nodes, idx, adj, weight: str) -> Tuple[List[List[float]], float]:
    """Per-move costs parallel to ``adj``, plus the cheapest move."""
    wadj = [[float(G[nodes[i]][nodes[j]].get(weight, 1.0)) for j in adj[i]]
            for i in range(len(nodes))]
    wmin = min((w for row in wadj for w in row), default=1.0)
    return wadj, wmin


def _astar_budgeted(prep, k: int, budget: int, pareto_keep: Optional[int] = None,
                    starts: Optional[Sequence[int]] = None,
                    wadj: Optional[List[List[float]]] = None,
                    wmin: float = 1.0):
    """A* for the shortest walk with ``|N[V(W)]| >= k`` (single k). The
    admissible, consistent heuristic ``h = ceil((k - coverage)/Delta)`` (each
    step adds at most Delta new covered vertices) makes the first popped goal
    optimal; consistency is proved in the module notes. Returns
    ``(walk_idx, cov)`` (proven optimal) or ``None`` if the closed set would
    exceed ``budget`` (caller falls back to the level BFS).

    Two state-space reductions, both proved in docs/proofs.md section 4:

    * states are keyed by ``(last, cov)`` rather than ``(last, visited)`` --
      coverage is a sufficient statistic for both the successor function and
      the goal test (Lemma 4.1), so this merges states and never splits them;
    * a generated state dominated by an already-closed state at the same last
      vertex (``cov`` a subset, ``g`` no larger) is discarded (Lemma 4.3).
      Only ``pareto_keep`` closed states per vertex are checked, which is
      sound by Remark 4.4 -- partial checking costs pruning power, not
      correctness.
    """
    nodes, idx, adj, nbmask = prep
    n = len(nodes)
    Delta = max((len(a) for a in adj), default=1)
    if pareto_keep is None:
        pareto_keep = PARETO_KEEP

    def h(pc: int):
        r = k - pc
        steps = (r + Delta - 1) // Delta if r > 0 else 0  # ceil, integer
        # Under weights each of those steps still costs at least wmin, which
        # keeps h admissible and consistent (docs/proofs.md, Lemma 5.4).
        return steps * wmin if wadj is not None else steps

    g0 = 0.0 if wadj is not None else 1
    heap = []            # (f, g, last, cov)
    g_score: Dict = {}
    parent: Dict = {}
    dom: List[List] = [[] for _ in range(n)]   # last -> [(cov, g)] closed
    for i in (range(n) if starts is None else starts):
        st = (i, nbmask[i])
        g_score[st] = g0
        parent[st] = None
        heapq.heappush(heap, (g0 + h(nbmask[i].bit_count()), g0, i, nbmask[i]))

    closed = set()
    gen = pruned = 0
    while heap:
        f, g, last, cov = heapq.heappop(heap)
        st = (last, cov)
        if g > g_score.get(st, 1 << 30) or st in closed:
            continue  # stale duplicate / already finalized
        closed.add(st)
        if len(closed) > budget:
            return None
        if cov.bit_count() >= k:
            walk = []
            cur = st
            while cur is not None:
                walk.append(cur[0])
                cur = parent[cur]
            walk.reverse()
            return walk, cov
        if pareto_keep:
            bucket = dom[last]
            bucket.append((cov, g))
            if len(bucket) > pareto_keep:
                # evict the least useful dominator (smallest coverage)
                worst = min(range(len(bucket)),
                            key=lambda t: bucket[t][0].bit_count())
                bucket.pop(worst)
        row = wadj[last] if wadj is not None else None
        for t, u in enumerate(adj[last]):
            ng = g + (row[t] if row is not None else 1)
            ncov = cov | nbmask[u]
            nst = (u, ncov)
            if ng >= g_score.get(nst, float("inf")):
                continue
            if pareto_keep:
                gen += 1
                if any(gg <= ng and not (ncov & ~c) for c, gg in dom[u]):
                    pruned += 1
                    continue  # Pareto-dominated by a closed state at u
                if gen >= PARETO_PROBE and pruned < PARETO_MIN_RATE * gen:
                    # The check is not paying for itself on this instance.
                    # Switching it off only forfeits pruning power, never
                    # correctness (docs/proofs.md, Remark 4.4).
                    pareto_keep = 0
            g_score[nst] = ng
            parent[nst] = st
            heapq.heappush(heap, (ng + h(ncov.bit_count()), ng, u, ncov))
    return None


def budgeted_walk(
    G: nx.Graph,
    k: int,
    state_budget: Optional[int] = None,
    state_cap: Optional[int] = None,
    fallback: Union[str, List, None] = "auto",
    engine: str = "astar",
    weight: Optional[str] = None,
) -> Dict:
    """Shortest walk whose closed neighbourhood covers at least ``k`` vertices
    (equivalently: the optimal sub-QFT region of size ``k`` plus its optimal
    dominating walk). ``1 <= k <= n``.

    ``engine="astar"`` (default) runs A* with an admissible/consistent
    coverage heuristic, which returns the *same* optimum as the level BFS
    while exploring far fewer states (measured 2--80x, growing with k) --- so
    it reaches deeper certified k within the same state budget. On over-budget
    or ``engine="bfs"`` it falls back to the level-BFS sweep machinery (beam
    tail + subwalk fallback).

    ``weight`` names an edge attribute of per-CNOT costs (see
    :func:`qftbuilder.graphs.with_edge_errors`). With it the search minimizes
    the *weighted* move cost instead of the length -- the region then forms
    around good couplings and the white fillers are the cheapest to collect
    (docs/proofs.md, Lemma 5.4 and Proposition 5.2). Note ``k_path`` is then no
    longer the quantity being minimized, and the orbit reduction is switched
    off (an automorphism of the graph need not preserve the weights).

    Returns a dict with keys ``k, subset, walk, k_path, cost`` (Theorem-4
    bound), ``cost_sweep`` (exact one-sweep CNOTs, 2*whites + 3*moves),
    ``region_proven`` and ``method`` (``"astar"`` / ``"astar-weighted"`` /
    ``"bfs"`` / ``"subwalk"``).
    ``state_cap`` is a legacy alias (budget ``25*cap``)."""
    n = G.number_of_nodes()
    if engine == "astar" and 1 <= k <= n:
        prep = _prepare(G)
        nodes, idx, adj, _ = prep
        max_deg = max((d for _, d in G.degree()), default=1)
        budget, _ = _budgets(n, max_deg, state_budget, state_cap)
        if weight is None:
            starts = _orbit_reps(G, idx) if n >= ORBIT_MIN_N else None
            res = _astar_budgeted(prep, k, budget, starts=starts)
            if res is not None:
                walk_idx, cov = res
                return _result(nodes, walk_idx, cov, k, n, True, "astar")
        else:
            # The weighted A* minimizes the *move* term only, while the sweep
            # also pays 2*w to collect each white; minimizing moves alone was
            # measured to lose against the plain optimum on the total. So both
            # candidates are generated and scored on the real objective, which
            # makes the answer never worse than the unweighted one. Neither is
            # a proof for the weighted total, hence region_proven=False.
            # Orbit reduction is off here: an automorphism of G need not
            # preserve the weights (docs/proofs.md, caveat after Lemma 4.5).
            wadj, wmin = _move_costs(G, nodes, idx, adj, weight)
            best = None
            for method, res in (
                ("astar", _astar_budgeted(prep, k, budget)),
                ("astar-weighted",
                 _astar_budgeted(prep, k, budget, wadj=wadj, wmin=wmin)),
            ):
                if res is None:
                    continue
                walk_idx, cov = res
                wc = _white_costs(G, nodes, idx, walk_idx, weight)
                r = _result(nodes, walk_idx, cov, k, n, False,
                            "weighted-best/" + method, wc)
                tot = weighted_sweep_cost(r["walk"], G, region=set(r["subset"]),
                                          weight=weight)
                if best is None or tot < best[0]:
                    best = (tot, r)
            if best is not None:
                best[1]["cost_weighted"] = best[0]
                return best[1]
        # over budget: fall through to the level-BFS path (beam + fallback)
    return budgeted_sweep(G, ks=[k], state_budget=state_budget,
                          state_cap=state_cap, fallback=fallback)[k]


def budgeted_sweep(
    G: nx.Graph,
    ks: Optional[Sequence[int]] = None,
    state_budget: Optional[int] = None,
    state_cap: Optional[int] = None,
    fallback: Union[str, List, None] = "auto",
) -> Dict[int, Dict]:
    """One search, answers for every requested ``k`` (``ks=None`` = all
    ``1..n``). Returns ``{k: result-dict}`` (see :func:`budgeted_walk`).

    Phases: exact BFS (no per-level pruning) until the state budget trips;
    then a coverage-greedy beam tail (bounded by ``budget/4`` insertions);
    then, for any still-unproven ``k``, the ``fallback``: ``"auto"``
    (default) computes one whole-graph ``dominating_walk`` + local search
    and offers its shortest covering subwalk per ``k``; a list uses that
    walk (vertices of G) instead; ``None`` disables the fallback."""
    n = G.number_of_nodes()
    if n == 0:
        return {}
    targets = sorted(set(ks) if ks is not None else range(1, n + 1))
    if targets and (targets[0] < 1 or targets[-1] > n):
        raise ValueError(f"k out of range 1..{n}: {targets}")

    max_deg = max((d for _, d in G.degree()), default=1)
    budget, beam_width = _budgets(n, max_deg, state_budget, state_cap)
    nodes, idx, adj, nbmask = _prepare(G)
    answers: Dict[int, Dict] = {}
    pending = list(targets)

    parent: Dict = {}
    frontier: Dict = {}  # key (last, cov) -> cov  (Corollary 4.2)
    proven = True

    starts = _orbit_reps(G, idx) if n >= ORBIT_MIN_N else range(n)
    for i in starts:
        key = (i, nbmask[i])
        parent[key] = None
        frontier[key] = nbmask[i]

    def _absorb(level_states):
        nonlocal pending
        if not pending or not level_states:
            return
        # cheap pre-check: skip the sort unless some pending k is reachable
        best_pc = max(c.bit_count() for c in level_states.values())
        if best_pc < pending[0]:
            return
        by_cov = sorted(level_states.items(), key=lambda kv: -kv[1].bit_count())
        for key, cov in by_cov:
            pc = cov.bit_count()
            newly = [k for k in pending if k <= pc]
            if not newly:
                break  # sorted desc: nothing later can answer either
            walk_idx = _reconstruct(parent, key)
            for k in newly:
                answers[k] = _result(nodes, walk_idx, cov, k, n, proven, "bfs")
            pending = [k for k in pending if k > pc]

    def _expand(cur_frontier, insert_limit):
        """One BFS level. Returns (next_frontier, completed?)."""
        nxt: Dict = {}
        for key, cov in cur_frontier.items():
            last = key[0]
            for u in adj[last]:
                ncov = cov | nbmask[u]
                nkey = (u, ncov)
                if nkey in parent:
                    continue
                parent[nkey] = key
                nxt[nkey] = ncov
                if len(parent) > insert_limit:
                    return nxt, False
        return nxt, True

    _absorb(frontier)
    depth = 1
    max_depth = _MAX_DEPTH_FACTOR * n + 2

    # -- exact phase: full levels, no pruning, until the budget trips -------
    while pending and frontier and depth < max_depth:
        depth += 1
        nxt, complete = _expand(frontier, budget)
        frontier = nxt
        if not complete:
            proven = False  # mid-level abort: this level is not exhaustive
            _absorb(frontier)
            break
        _absorb(frontier)  # full level: absorbed answers are proven optimal

    # -- beam tail: bounded continuation for mid-range k --------------------
    if pending and frontier:
        proven = False
        tail_limit = len(parent) + budget // TAIL_FRACTION
        while pending and frontier and depth < max_depth:
            if len(frontier) > beam_width:
                keep = heapq.nlargest(
                    beam_width, frontier.items(), key=lambda kv: kv[1].bit_count()
                )
                frontier = dict(keep)
            depth += 1
            frontier, complete = _expand(frontier, tail_limit)
            _absorb(frontier)
            if not complete:
                break

    # -- fallback: guarantee an answer for every k at whole-solve cost ------
    if pending or any(not answers[k]["region_proven"] for k in answers):
        fb_walk_idx: Optional[List[int]] = None
        if isinstance(fallback, list):
            fb_walk_idx = [idx[v] for v in fallback]
        elif fallback == "auto":
            from .walk import dominating_walk, improve_walk_strong

            w = dominating_walk(G)
            if w is not None:
                w = improve_walk_strong(w, G)
                fb_walk_idx = [idx[v] for v in w]
        if fb_walk_idx:
            need = pending + [k for k in answers if not answers[k]["region_proven"]]
            subs = _subwalk_answers(fb_walk_idx, nbmask, sorted(set(need)))
            for k, (length, i, j, cov) in subs.items():
                cur = answers.get(k)
                if cur is None or cur["k_path"] is None or length < cur["k_path"]:
                    answers[k] = _result(
                        nodes, fb_walk_idx[i : j + 1], cov, k, n, False, "subwalk"
                    )
            pending = [k for k in pending if k not in answers]

    for k in pending:  # only reachable if even the fallback failed
        answers[k] = {
            "k": k, "subset": None, "walk": None, "k_path": None,
            "cost": None, "cost_sweep": None, "region_proven": False,
            "method": None,
        }
    return answers


def max_coverage_walk(
    G: nx.Graph,
    length: int,
    state_budget: Optional[int] = None,
    state_cap: Optional[int] = None,
) -> Dict:
    """Dual query: among walks of at most ``length`` vertices, one covering
    the most vertices. Returns ``{covered, walk, proven}``. Same state
    budget (and derivation) as :func:`budgeted_sweep`; no fallback."""
    n = G.number_of_nodes()
    if n == 0 or length <= 0:
        return {"covered": 0, "walk": None, "proven": True}
    max_deg = max((d for _, d in G.degree()), default=1)
    budget, beam_width = _budgets(n, max_deg, state_budget, state_cap)
    nodes, _, adj, nbmask = _prepare(G)
    parent: Dict = {}
    frontier: Dict = {}
    proven = True
    for i in range(n):
        key = (i, 1 << i)
        parent[key] = None
        frontier[key] = nbmask[i]

    best_key, best_cov = max(frontier.items(), key=lambda kv: kv[1].bit_count())
    for _ in range(length - 1):
        if best_cov.bit_count() == n:
            break
        if len(parent) > budget:
            proven = False
            break
        if not proven or len(frontier) > beam_width:
            # beam mode once anything was pruned
            if len(frontier) > beam_width:
                proven = False
                keep = heapq.nlargest(
                    beam_width, frontier.items(), key=lambda kv: kv[1].bit_count()
                )
                frontier = dict(keep)
        nxt: Dict = {}
        for (last, vmask), cov in frontier.items():
            for u in adj[last]:
                nkey = (u, vmask | (1 << u))
                if nkey in parent:
                    continue
                parent[nkey] = (last, vmask)
                nxt[nkey] = cov | nbmask[u]
        if not nxt:
            break
        frontier = nxt
        cand_key, cand_cov = max(frontier.items(), key=lambda kv: kv[1].bit_count())
        if cand_cov.bit_count() > best_cov.bit_count():
            best_key, best_cov = cand_key, cand_cov

    walk = [nodes[i] for i in _reconstruct(parent, best_key)]
    return {"covered": best_cov.bit_count(), "walk": walk, "proven": proven}
