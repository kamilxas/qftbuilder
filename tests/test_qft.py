"""QFT construction: textbook identity, swap-freeness, cost accounting."""
import networkx as nx
import pytest

from qftbuilder import Solver, build_full_qft, single_sweep, sub_qft
from qftbuilder.graphs import standard_benchmark, sun_16q
from qftbuilder.qft import HAVE_QISKIT, transpiled_cx

FAST = Solver(profile="fast")


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 7, 8])
def test_complete_graph_identity(m):
    """Full QFT on K_m must cost exactly m(m-1) CNOTs (textbook count)."""
    r = build_full_qft(nx.complete_graph(m), solver=FAST)
    assert r["cnot"] == m * (m - 1)
    assert r["cascades"] == m


def test_full_qft_best_strategy_dominates_components():
    G = standard_benchmark(max_n=30)["sun_16q"]
    greedy = build_full_qft(G, solver=FAST, strategy="greedy")
    look = build_full_qft(G, solver=FAST, strategy="lookahead")
    best = build_full_qft(G, solver=FAST)  # strategy="best"
    assert best["cnot"] == min(greedy["cnot"], look["cnot"])
    assert best["cascades"] == G.number_of_nodes()


def test_single_sweep_costs_match_walk():
    G = sun_16q()
    r = single_sweep(G, solver=FAST, build_circuit=False)
    res = r["result"]
    assert r["cnot"] == 2 * r["whites"] + 3 * r["moves"]
    assert r["cnot"] == res.cost  # selector metric == emitted circuit count
    assert r["moves"] == res.length - 1
    assert r["whites"] == res.n - res.distinct


def test_sub_qft_counts():
    G = sun_16q()
    r = sub_qft(G, 8)
    assert r["region_proven"]
    assert r["cnot"] == 2 * r["whites"] + 3 * r["moves"]
    assert len(r["subset"]) == 8


@pytest.mark.skipif(not HAVE_QISKIT, reason="qiskit not installed")
def test_sweep_circuit_swap_free():
    """Raw emitted CX == transpiled CX: every gate sits on a physical edge."""
    G = sun_16q()
    r = single_sweep(G, solver=FAST, build_circuit=True)
    assert r["qc"] is not None
    cx, _depth = transpiled_cx(r["qc"], G, opt_level=1)
    assert cx == r["cnot"]


@pytest.mark.skipif(not HAVE_QISKIT, reason="qiskit not installed")
def test_full_qft_circuit_swap_free():
    G = standard_benchmark(max_n=20)["lnn_16"]
    r = build_full_qft(G, solver=FAST, build_circuit=True)
    cx, _ = transpiled_cx(r["qc"], G, opt_level=1)
    assert cx == r["cnot"]
