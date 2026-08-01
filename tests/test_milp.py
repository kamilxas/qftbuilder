"""MILP formulations agree with combinatorial ground truth."""
import networkx as nx
import pytest

from qftbuilder.graphs import lnn, standard_benchmark, sun_16q
from qftbuilder.kwalk import budgeted_walk
from qftbuilder.milp import (
    shortest_budgeted_walk_milp,
    shortest_dominating_walk_milp,
)

from reference import best_region_walk_bruteforce


def test_full_domination_on_chain():
    # lnn(8): dominating walk must be the middle 6 vertices (ends covered)
    G = lnn(8)
    length, proven, _ = shortest_dominating_walk_milp(G, time_limit=60.0)
    assert proven and length == 6


@pytest.mark.parametrize("k", [3, 5, 8])
def test_budgeted_matches_kwalk_and_bruteforce(k):
    G = sun_16q()
    kw = budgeted_walk(G, k)
    assert kw["region_proven"]
    ml, proven, _ = shortest_budgeted_walk_milp(G, k, time_limit=120.0)
    assert proven
    assert ml == kw["k_path"] == best_region_walk_bruteforce(G, k)


def test_budgeted_single_anchor_shortcut():
    # a star center covers everything -> length 1 for any k <= n
    G = nx.star_graph(6)
    for k in (1, 4, 7):
        ml, proven, _ = shortest_budgeted_walk_milp(G, k, time_limit=30.0)
        assert proven and ml == 1


def test_relax_and_dual_are_lower_bounds():
    G = standard_benchmark(max_n=30)["grid_4x4"]
    exact, proven, _ = shortest_dominating_walk_milp(G, time_limit=120.0)
    assert proven
    lp, _, _ = shortest_dominating_walk_milp(G, relax=True)
    assert lp <= exact
    _, _, dual = shortest_dominating_walk_milp(G, time_limit=120.0, return_dual=True)
    assert dual is not None and lp <= dual <= exact
