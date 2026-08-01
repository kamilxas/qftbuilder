"""Walk construction, local search and repair invariants."""
import networkx as nx
import pytest

from qftbuilder.cds import cds_dfs_path, greedy_cds
from qftbuilder.cost import walk_cost
from qftbuilder.graphs import standard_benchmark
from qftbuilder.walk import (
    _expand_order,
    _expand_order_rich,
    _shortest_path_max_new,
    complete_walk,
    coverage,
    covers,
    dominating_walk,
    improve_walk_strong,
    repair_walk,
    walk_valid,
)

BENCH = standard_benchmark(max_n=60)


@pytest.mark.parametrize("name", list(BENCH))
def test_dominating_walk_valid_everywhere(name):
    G = BENCH[name]
    w = dominating_walk(G)
    assert w is not None
    assert walk_valid(G, w) and covers(G, w)


@pytest.mark.parametrize("name", list(BENCH))
def test_local_search_monotone_safe(name):
    G = BENCH[name]
    base = cds_dfs_path(G)
    assert walk_valid(G, base) and covers(G, base)
    improved = improve_walk_strong(base, G)
    assert walk_valid(G, improved) and covers(G, improved)
    assert len(improved) <= len(base)


def test_cds_dominates():
    G = BENCH["heavy_hex_2x2"]
    cds = greedy_cds(G)
    dominated = set()
    for v in cds:
        dominated.add(v)
        dominated.update(G.neighbors(v))
    assert dominated == set(G.nodes())
    assert nx.is_connected(G.subgraph(cds))


def test_complete_walk_completes():
    G = BENCH["grid_5x5"]
    partial = [0, 1, 2]
    full = complete_walk(partial, G)
    assert walk_valid(G, full) and covers(G, full)
    assert full[: len(partial)] == partial


# -- true-cost local search (docs/proofs.md section 3) -----------------------


@pytest.mark.parametrize("name", list(BENCH))
def test_segment_dp_matches_brute_force(name):
    """Lemma 3.6: the DP picks a shortest path maximizing new vertices. Checked
    against exhaustive enumeration of all shortest paths."""
    G = BENCH[name]
    nodes = sorted(G.nodes())[:8]
    used = set(nodes[:2])
    for a in nodes:
        for b in nodes:
            got = _shortest_path_max_new(G, a, b, used)
            assert got is not None
            assert walk_valid(G, got) and got[0] == a and got[-1] == b
            assert len(got) - 1 == nx.shortest_path_length(G, a, b)
            best = max(len(set(p) - used - {a})
                       for p in nx.all_shortest_paths(G, a, b))
            assert len(set(got) - used - {a}) == best


@pytest.mark.parametrize("name", list(BENCH))
def test_rich_stitch_same_length_more_distinct(name):
    """Lemma 3.4 + Corollary 3.2: choosing among shortest paths cannot change
    the length, and the DP never yields fewer distinct vertices."""
    G = BENCH[name]
    order = dominating_walk(G)
    assert order is not None
    plain, rich = _expand_order(order, G), _expand_order_rich(order, G)
    assert plain is not None and rich is not None
    assert len(rich) == len(plain)
    assert len(set(rich)) >= len(set(plain))
    assert walk_valid(G, rich) and covers(G, rich)


@pytest.mark.parametrize("name", list(BENCH))
def test_cnot_objective_is_cost_monotone(name):
    """Theorem 3: under objective="cnot" the result never costs more than the
    input, and stays a valid dominating walk."""
    G = BENCH[name]
    n = G.number_of_nodes()
    base = cds_dfs_path(G)
    out = improve_walk_strong(base, G, objective="cnot")
    assert walk_valid(G, out) and covers(G, out)
    assert walk_cost(out, n) <= walk_cost(base, n)


@pytest.mark.parametrize("name", list(BENCH))
def test_length_objective_stays_monotone_in_length(name):
    """The default objective keeps the old guarantee: never longer."""
    G = BENCH[name]
    base = cds_dfs_path(G)
    out = improve_walk_strong(base, G, objective="length")
    assert walk_valid(G, out) and covers(G, out)
    assert len(out) <= len(base)


def test_repair_walk_after_vertex_removal():
    G = BENCH["sun_16q"]
    w = dominating_walk(G)
    for drop in (w[0], w[-1], w[len(w) // 2]):
        H = G.copy()
        H.remove_node(drop)
        if not nx.is_connected(H):
            continue
        r = repair_walk(w, H)
        assert r is not None
        assert walk_valid(H, r) and covers(H, r)


# -- endpoint-constrained search --------------------------------------------


@pytest.mark.parametrize("name", list(BENCH))
def test_walk_ending_at_respects_the_endpoint(name):
    """The QFT cascade finalizes the qubit its carrier lands on, so a walk can
    be required to finish somewhere specific."""
    from qftbuilder.walk import walk_ending_at

    G = BENCH[name]
    for end in list(G.nodes())[:4]:
        w = walk_ending_at(G, end)
        assert w is not None, (name, end)
        assert w[-1] == end
        assert walk_valid(G, w) and covers(G, w)


def test_pinned_order_keeps_the_last_anchor():
    """_optimize_order(pin_last=True) may reorder everything except the tail."""
    from qftbuilder.walk import _optimize_order

    G = BENCH["grid_5x5"]
    ess = [0, 6, 12, 18, 24, 7]
    out = _optimize_order(ess, G, pin_last=True)
    assert out[-1] == ess[-1]
    assert sorted(out) == sorted(ess)


def test_redundancy_removal_can_protect_the_end():
    """keep_last stops the trimmer from eating the endpoint the caller needs."""
    from qftbuilder.walk import _redundancy_removal

    G = BENCH["lnn_16"]
    full = set(G.nodes())
    walk = list(range(16))
    trimmed = _redundancy_removal(walk, G, full)
    kept = _redundancy_removal(walk, G, full, keep_last=True)
    assert kept[-1] == 15
    assert covers(G, trimmed) and covers(G, kept)
