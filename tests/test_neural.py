"""Neural rungs: checkpoint load, decode validity, cascade integration."""
import pytest

import qftbuilder as qb
from qftbuilder import neural
from qftbuilder.graphs import standard_benchmark, sun_16q
from qftbuilder.walk import covers, walk_valid

needs_torch = pytest.mark.skipif(not neural.HAVE_TORCH, reason="torch not installed")


@needs_torch
def test_checkpoints_load():
    whp = neural.load_model("whp")
    wnshp = neural.load_model("wnshp")
    assert getattr(whp, "target_prefix", "whp") == "whp"
    assert getattr(wnshp, "target_prefix", "whp") == "wnshp"


@needs_torch
@pytest.mark.parametrize("name", ["sun_16q", "grid_4x4", "heavy_hex_1x2"])
def test_wnshp_decode_covers(name):
    G = standard_benchmark(max_n=30)[name]
    model = neural.load_model("wnshp")
    probs, _ = model.predict(neural.graph_to_data(G))
    walk = neural.decode_wnshp(G, probs.cpu().numpy())
    assert walk is not None
    assert walk_valid(G, walk) and covers(G, walk)


@needs_torch
def test_balanced_profile_not_worse_than_fast():
    fast = qb.Solver(profile="fast")
    balanced = qb.Solver(profile="balanced")
    for name, G in standard_benchmark(max_n=30).items():
        rf = fast.solve(G)
        rb = balanced.solve(G)
        assert rb.neural_used
        assert rb.cost <= rf.cost, name  # cascade takes the min over more rungs


@needs_torch
def test_whp_refine_invariants():
    G = sun_16q()
    model = neural.load_model("whp")
    probs, _exists = model.predict(neural.graph_to_data(G))
    path = neural.decode_whp_best(G, probs.cpu().numpy())
    if path is not None:  # existence-gated: None is a legal answer
        assert neural.is_simple_walk(G, path) and covers(G, path)
