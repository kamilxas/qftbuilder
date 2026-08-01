"""Input adapters, label round-tripping, solver contract."""
import json

import networkx as nx
import numpy as np
import pytest

import qftbuilder as qb
from qftbuilder.graphs import as_graph


def test_as_graph_from_edges_and_matrix_and_dict():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    G1, _ = as_graph(edges)
    A = nx.to_numpy_array(nx.Graph(edges))
    G2, _ = as_graph(A)
    G3, _ = as_graph({0: [1, 3], 1: [0, 2, 3], 2: [1, 3], 3: [2, 0, 1]})
    assert (
        nx.utils.graphs_equal(G1, G2)
        and G1.number_of_edges() == G3.number_of_edges() == 5
    )


def test_as_graph_from_file(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"edges": [[0, 1], [1, 2], [0, 2]]}))
    G, _ = as_graph(str(p))
    assert G.number_of_nodes() == 3 and G.number_of_edges() == 3
    p2 = tmp_path / "g.edgelist"
    p2.write_text("0 1\n1 2  # comment\n")
    G2, _ = as_graph(str(p2))
    assert G2.number_of_edges() == 2


def test_as_graph_rejects_disconnected():
    with pytest.raises(ValueError, match="connected"):
        as_graph([(0, 1), (2, 3)])


def test_labels_round_trip_with_string_nodes():
    G = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("b", "d")])
    res = qb.solve(G, profile="fast")
    assert set(res.walk) <= {"a", "b", "c", "d"}
    # walk is valid in the ORIGINAL graph
    assert all(
        u == v or G.has_edge(u, v) for u, v in zip(res.walk, res.walk[1:])
    )


def test_solver_contract_on_benchmark():
    solver = qb.Solver(profile="fast")
    for name, G in qb.standard_benchmark(max_n=30).items():
        res = solver.solve(G)
        assert res.walk, name
        assert res.length == len(res.walk)
        assert res.cost == qb.walk_cost(res.walk, res.n)  # label-independent
        assert "cds" in res.candidates  # floor always present
        # cascade is never worse than its floor
        assert res.cost <= res.candidates["cds"]["cost"]


def test_certify_small():
    G = qb.lnn(8)
    res = qb.solve(G, profile="fast")
    cert = qb.certify(G, res, time_limit=60.0)
    assert cert.lower_bound <= res.length
    assert cert.proven_optimal == (cert.lower_bound == res.length)
    assert res.lower_bound == cert.lower_bound


def test_certify_sub():
    G = qb.sun_16q()
    r = qb.sub_qft(G, 8)
    cert = qb.Solver(profile="fast").certify_sub(G, 8, upper_length=r["k_path"],
                                                 time_limit=120.0)
    assert cert.lower_bound <= r["k_path"]
    assert cert.proven_optimal  # small instance: MILP should close it


def test_sub_qft_sweep_labels():
    G = nx.relabel_nodes(qb.sun_16q(), lambda i: f"q{i}")
    out = qb.sub_qft_sweep(G, ks=[4, 8])
    assert set(out) == {4, 8}
    for r in out.values():
        assert all(isinstance(v, str) and v.startswith("q") for v in r["walk"])
