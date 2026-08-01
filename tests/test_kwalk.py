"""The budgeted-coverage walk must equal brute-force region enumeration,
and stay within its runtime budget on large graphs."""
import random
import time

import networkx as nx
import pytest

from qftbuilder import budgeted_sweep, budgeted_walk, max_coverage_walk
from qftbuilder.graphs import (
    cycle,
    heavy_hex,
    square_lattice,
    standard_benchmark,
    sun_16q,
)
from qftbuilder.walk import covers, walk_valid

from reference import best_region_walk_bruteforce


def _random_connected(n, p, seed):
    rng = random.Random(seed)
    while True:
        G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
        if G.number_of_nodes() and nx.is_connected(G):
            return G


def _check_result(G, r):
    """Structural invariants of a budgeted-walk result."""
    assert r["walk"] is not None
    assert walk_valid(G, r["walk"])
    S = set(r["subset"])
    assert len(S) == r["k"]
    assert set(r["walk"]) <= S
    sub = G.subgraph(S)
    assert nx.is_connected(sub)
    assert covers(sub, r["walk"])  # walk dominates the region inside G[S]
    assert r["k_path"] == len(r["walk"])


@pytest.mark.parametrize("name", list(standard_benchmark(max_n=30)))
def test_matches_bruteforce_on_topologies(name):
    G = standard_benchmark(max_n=30)[name]
    for k in (2, 4, 6, 8):
        if k > G.number_of_nodes():
            continue
        r = budgeted_walk(G, k)
        assert r["region_proven"], f"{name} k={k} unexpectedly degraded"
        gt = best_region_walk_bruteforce(G, k)
        assert r["k_path"] == gt, f"{name} k={k}: got {r['k_path']}, GT {gt}"
        _check_result(G, r)


@pytest.mark.parametrize("seed", range(6))
def test_matches_bruteforce_on_random(seed):
    G = _random_connected(n=10 + seed, p=0.3, seed=seed)
    for k in (3, 5, 7):
        r = budgeted_walk(G, k)
        gt = best_region_walk_bruteforce(G, k)
        if r["region_proven"]:
            assert r["k_path"] == gt
        else:  # beam degradation may only overestimate
            assert r["k_path"] >= gt
        _check_result(G, r)


def test_sweep_consistent_with_single_queries():
    G = standard_benchmark(max_n=30)["sun_16q"]
    sweep = budgeted_sweep(G)
    lengths = [sweep[k]["k_path"] for k in sorted(sweep)]
    assert lengths == sorted(lengths), "k_path must be monotone in k"
    for k in (2, 5, 9, 13, G.number_of_nodes()):
        single = budgeted_walk(G, k)
        assert single["k_path"] == sweep[k]["k_path"]
        _check_result(G, sweep[k])


def test_full_k_equals_exact_dominating_walk():
    # k = n on a small graph: budgeted walk == exact shortest dominating walk
    from qftbuilder.milp import shortest_dominating_walk_milp

    G = standard_benchmark(max_n=20)["sun_16q"]
    r = budgeted_walk(G, G.number_of_nodes())
    assert r["region_proven"]
    milp_len, proven, _ = shortest_dominating_walk_milp(G, time_limit=60.0)
    assert proven and milp_len == r["k_path"]


def test_max_coverage_dual():
    G = standard_benchmark(max_n=30)["grid_4x4"]
    r3 = max_coverage_walk(G, 3)
    assert r3["proven"] and len(r3["walk"]) <= 3
    # consistency with the primal: budgeted_walk(k=covered) fits in 3 steps
    k = r3["covered"]
    assert budgeted_walk(G, k)["k_path"] <= 3


def test_beam_cap_degrades_but_stays_valid():
    G = _random_connected(n=18, p=0.5, seed=99)
    r = budgeted_walk(G, 12, state_cap=50)  # force degradation
    _check_result(G, r)
    gt = best_region_walk_bruteforce(G, 12)
    assert r["k_path"] >= gt


def test_budget_contract_large_graph():
    """n=144 with k far past the exact frontier: the search must respect its
    state budget, finish quickly, and still return a valid answer (beam or
    subwalk fallback) — sub-QFT may not cost more than a full solve."""
    G = square_lattice(12, 12)
    t0 = time.time()
    r = budgeted_walk(G, 36)
    dt = time.time() - t0
    _check_result(G, r)
    assert r["method"] in ("astar", "bfs", "subwalk")
    assert dt < 60  # generous CI bound; measured seconds on a laptop


def test_fallback_answers_every_k():
    """Full-range sweep on n=87 heavy-hex: every k gets a valid answer and
    lengths stay monotone after merging BFS and subwalk candidates."""
    G = heavy_hex(3, 4)
    out = budgeted_sweep(G, ks=[10, 30, 50, 70, 87])
    lens = [out[k]["k_path"] for k in sorted(out)]
    assert all(l is not None for l in lens)
    assert lens == sorted(lens)
    for r in out.values():
        _check_result(G, r)


# -- state-space reductions (docs/proofs.md section 4) -----------------------


@pytest.mark.parametrize("name,G", [
    ("grid_5x5", square_lattice(5, 5)),
    ("heavy_hex_2x2", heavy_hex(2, 2)),
    ("sun_16q", sun_16q()),
])
def test_pruned_search_matches_unpruned(name, G):
    """Corollary 4.2 + Lemmas 4.3/4.5: coarser states, Pareto pruning and
    orbit-reduced starts must return the same optimal length as the plain
    search for every k. The walk itself may differ (ties)."""
    from qftbuilder.kwalk import _astar_budgeted, _orbit_reps, _prepare

    prep = _prepare(G)
    n = G.number_of_nodes()
    reps = _orbit_reps(G, prep[1])
    assert 1 <= len(reps) <= n
    budget = 1 << 22
    for k in range(1, n + 1):
        plain = _astar_budgeted(prep, k, budget, pareto_keep=0, starts=None)
        pruned = _astar_budgeted(prep, k, budget, starts=reps)
        assert (plain is None) == (pruned is None), (name, k)
        if plain is None:
            continue
        assert len(plain[0]) == len(pruned[0]), (name, k)
        for walk, cov in (plain, pruned):
            assert cov.bit_count() >= k
            assert all(b in prep[2][a] for a, b in zip(walk, walk[1:]))


def test_orbit_reps_are_sound_on_a_symmetric_graph():
    """A cycle is vertex-transitive: every vertex is one orbit, so the search
    may start from a single representative."""
    from qftbuilder.kwalk import _orbit_reps, _prepare

    G = cycle(12)
    prep = _prepare(G)
    assert len(_orbit_reps(G, prep[1])) == 1
