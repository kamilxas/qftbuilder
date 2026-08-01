"""The faithful builder must emit the true QFT unitary, not a skeleton.

Ground truth is qiskit's textbook QFT. The construction permutes qubits (the
collect-and-move contains a SWAP) and emits no reversal swaps, so the reference
is placed on the reported input wires and permuted onto the reported output
wires before comparing."""
import networkx as nx
import numpy as np
import pytest

from qftbuilder.graphs import cycle, lnn, square_lattice, sun
from qftbuilder.qft import HAVE_QISKIT, build_full_qft, build_qft_circuit

pytestmark = pytest.mark.skipif(not HAVE_QISKIT, reason="needs qiskit")

CASES = {
    "lnn_3": lnn(3),
    "lnn_5": lnn(5),
    "cycle_5": cycle(5),
    "grid_2x3": square_lattice(2, 3),
    "sun_5_2": sun(5, 2),
    "star_5": nx.convert_node_labels_to_integers(nx.star_graph(4)),
    "complete_4": nx.convert_node_labels_to_integers(nx.complete_graph(4)),
}


def _textbook_qft(n):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for m in range(j + 1, n):
            qc.cp(np.pi / 2 ** (m - j), m, j)
    return qc


def _reference(out, n):
    """Textbook QFT on the wires the builder actually used."""
    from qiskit import QuantumCircuit

    ref = QuantumCircuit(n)
    ref.compose(_textbook_qft(n),
                qubits=[out["input_order"][l] for l in range(n)], inplace=True)
    cur = {l: out["input_order"][l] for l in range(n)}
    where = {w: l for l, w in cur.items()}
    for l in range(n):
        want, have = out["qubit_order"][l], cur[l]
        if have == want:
            continue
        other = where[want]
        ref.swap(have, want)
        cur[l], cur[other] = want, have
        where[want], where[have] = l, other
    return ref


@pytest.mark.parametrize("name", list(CASES))
def test_circuit_is_the_exact_qft(name):
    from qiskit.quantum_info import Operator

    G = CASES[name]
    n = G.number_of_nodes()
    out = build_qft_circuit(G)
    U = Operator(out["qc"]).data
    V = Operator(_reference(out, n)).data
    i = np.unravel_index(np.argmax(np.abs(V)), V.shape)
    phase = V[i] / U[i]
    phase /= abs(phase)
    assert np.linalg.norm(U * phase - V) < 1e-8, name


@pytest.mark.parametrize("name", list(CASES))
def test_reported_count_matches_the_circuit(name):
    G = CASES[name]
    out = build_qft_circuit(G)
    assert out["qc"].count_ops().get("cx", 0) == out["cnot"]


@pytest.mark.parametrize("m", [3, 4, 5])
def test_complete_graph_hits_the_textbook_identity(m):
    """On K_m every qubit is adjacent, so the endpoint constraint never bites
    and the faithful circuit costs exactly m(m-1) -- the same as the skeleton."""
    G = nx.convert_node_labels_to_integers(nx.complete_graph(m))
    out = build_qft_circuit(G)
    assert out["cnot"] == m * (m - 1)
    assert out["cnot"] == build_full_qft(G)["cnot"]


@pytest.mark.parametrize("name", list(CASES))
def test_faithfulness_costs_something_but_not_much(name):
    """Correctness is not free: the cascade must end on the vertex it
    finalizes, so some walks get reversed or extended."""
    G = CASES[name]
    out = build_qft_circuit(G)
    skeleton = build_full_qft(G)["cnot"]
    assert out["cnot"] >= skeleton
    assert out["cnot"] <= 1.25 * skeleton


@pytest.mark.parametrize("name", list(CASES))
def test_every_two_qubit_gate_sits_on_an_edge(name):
    """The whole point of the construction: no SWAP insertion needed."""
    G = CASES[name]
    out = build_qft_circuit(G)
    edges = {frozenset(e) for e in G.edges()}
    qc = out["qc"]
    for inst in qc.data:
        if len(inst.qubits) == 2:
            a, b = (qc.find_bit(q).index for q in inst.qubits)
            assert frozenset((a, b)) in edges, (name, a, b)
