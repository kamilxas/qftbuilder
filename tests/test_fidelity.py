"""Fidelity-aware cost model and routing (docs/proofs.md, section 5)."""
import math
import random

import pytest

from qftbuilder.cost import (
    EDGE_WEIGHT,
    error_to_weight,
    sweep_cost,
    walk_cost,
    weighted_sweep_cost,
)
from qftbuilder.graphs import (
    as_graph,
    heavy_hex,
    square_lattice,
    standard_benchmark,
    with_edge_errors,
)
from qftbuilder.kwalk import budgeted_walk
from qftbuilder.solver import Solver
from qftbuilder.walk import covers, dominating_walk, improve_walk_strong, walk_valid

BENCH = standard_benchmark(max_n=40)


def _noisy(G, seed, sigma=1.0, mu=0.01):
    rng = random.Random(seed)
    return with_edge_errors(
        G, lambda u, v: min(0.5, math.exp(rng.gauss(math.log(mu), sigma)))
    )


def test_error_to_weight_is_monotone_and_floored():
    assert error_to_weight(0.0) > 0                      # floored, never zero
    assert error_to_weight(0.01) < error_to_weight(0.1)
    # -ln(1-e) is the additive form of a survival product
    assert error_to_weight(0.02) == pytest.approx(-math.log(0.98))
    assert math.isfinite(error_to_weight(1.0))           # no infinities


@pytest.mark.parametrize("name", list(BENCH))
def test_unit_weights_reproduce_the_plain_account(name):
    """Proposition 5.1: with every weight 1 the weighted account equals
    2*whites + 3*moves exactly."""
    G = BENCH[name]
    w = dominating_walk(G)
    got = weighted_sweep_cost(w, G, region=set(G.nodes()))
    assert got == pytest.approx(float(walk_cost(w, G.number_of_nodes())))
    assert got == pytest.approx(
        float(sweep_cost(len(w), len(set(w)), G.number_of_nodes())))


def test_as_graph_preserves_edge_weights():
    """Every public entry point normalizes through as_graph; fidelity weights
    must survive the relabelling."""
    H = _noisy(square_lattice(4, 4), seed=1)
    G, labels = as_graph(H)
    assert G.number_of_edges() == H.number_of_edges()
    assert all(EDGE_WEIGHT in d for _, _, d in G.edges(data=True))
    got = sorted(round(d[EDGE_WEIGHT], 12) for _, _, d in G.edges(data=True))
    want = sorted(round(d[EDGE_WEIGHT], 12) for _, _, d in H.edges(data=True))
    assert got == want


def test_with_edge_errors_accepts_mapping_and_scalar():
    G = square_lattice(3, 3)
    flat = with_edge_errors(G, 0.05)
    assert all(d[EDGE_WEIGHT] == pytest.approx(error_to_weight(0.05))
               for _, _, d in flat.edges(data=True))
    one = next(iter(G.edges()))
    table = with_edge_errors(G, {one: 0.3})
    assert table[one[0]][one[1]][EDGE_WEIGHT] == pytest.approx(error_to_weight(0.3))
    # unlisted edges stay unannotated and read as 1.0
    assert sum(EDGE_WEIGHT in d for _, _, d in table.edges(data=True)) == 1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fidelity_local_search_is_monotone(seed):
    """Theorem 3 under the weighted objective: never a worse weighted cost."""
    G = _noisy(heavy_hex(1, 2), seed)
    base = dominating_walk(G)
    out = improve_walk_strong(base, G, objective="fidelity")
    assert walk_valid(G, out) and covers(G, out)
    region = set(G.nodes())
    assert (weighted_sweep_cost(out, G, region=region)
            <= weighted_sweep_cost(base, G, region=region) + 1e-9)


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("k", [8, 12, 16])
def test_weighted_sub_qft_never_worse(seed, k):
    """budgeted_walk(weight=...) scores both the plain and the weighted optimum
    on the real objective and keeps the better, so it cannot lose to the
    unweighted answer."""
    G = _noisy(square_lattice(5, 5), seed)
    plain = budgeted_walk(G, k)
    fid = budgeted_walk(G, k, weight=EDGE_WEIGHT)
    cp = weighted_sweep_cost(plain["walk"], G, region=set(plain["subset"]))
    cf = weighted_sweep_cost(fid["walk"], G, region=set(fid["subset"]))
    assert cf <= cp + 1e-9
    assert len(fid["subset"]) == k
    assert walk_valid(G, fid["walk"])
    # no optimality is claimed for the weighted total
    assert fid["region_proven"] is False
    assert fid["cost_weighted"] == pytest.approx(cf)


def test_solver_accepts_fidelity_objective():
    G = _noisy(square_lattice(4, 4), seed=7)
    r = Solver(profile="fast", objective="fidelity").solve(G)
    assert walk_valid(G, r.walk) and covers(G, r.walk)
    assert r.cost > 0


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        Solver(objective="nonsense")


def test_readme_example_runs():
    """The public snippet in README must keep working: helper exported at the
    package root, sub_qft accepting the weight argument."""
    import qftbuilder as qb

    G = qb.with_edge_errors(qb.heavy_hex(2, 2), 0.02)
    res = qb.Solver(profile="fast", objective="fidelity").solve(G)
    reg = qb.sub_qft(G, k=12, weight=qb.EDGE_WEIGHT)
    assert res.cost > 0
    assert len(reg["subset"]) == 12
    assert walk_valid(G, reg["walk"])
