"""Walk -> quantum circuit: single sweeps, sub-QFT regions, full QFT.

Construction follows the cascade scheme of Khadiev et al. ("Quantum Circuit
for QFT for Arbitrary Qubits Connection Graph"). Vertex terminology matches
the thesis: **black** vertices lie on the walk, **white** vertices are
covered but off the walk. A *sweep* moves along the dominating walk,
collecting each white qubit with a 2-CNOT gate and stepping along the walk
with a 3-CNOT collect-and-move ("move"); a *full QFT* is n such cascades,
finalizing one qubit per cascade and shrinking the graph. Every 2-qubit
gate lies on a physical edge, so the transpiler needs no SWAPs (asserted in
tests).

Improvements over the source project's construction:

- every cascade's walk comes from the full :class:`~qftbuilder.solver.Solver`
  cascade (neural rungs included), not from one heuristic;
- cascade candidates are ranked by the **exact** cost ``2*whites + 3*moves``
  (a longer walk with more distinct vertices can be cheaper — length alone is
  the wrong objective);
- each cascade is warm-started by repairing the previous cascade's walk on
  the shrunken graph (cheap, and often already optimal);
- the finalized-qubit choice is made with a one-step lookahead over endpoint
  candidates instead of a fixed rule.

Angle caveat: emitted rotation angles are placeholders (as in the reference
construction) — circuits are **CX-count-accurate skeletons**, not the exact
QFT unitary. Faithful angle bookkeeping is planned; all CNOT comparisons are
unaffected.

Circuit emission requires qiskit (``pip install qft-builder[quantum]``);
count-only results work without it.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .graphs import as_graph
from .kwalk import budgeted_walk
from .solver import Solver
from .walk import coverage, covers, improve_walk_strong, repair_walk, walk_valid

__all__ = [
    "HAVE_QISKIT",
    "single_sweep",
    "sub_qft",
    "build_full_qft",
    "transpiled_cx",
    "naive_qft",
]

try:  # pragma: no cover
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover
    QuantumCircuit = None
    HAVE_QISKIT = False

BASIS = ["cx", "rz", "sx", "x", "h", "s", "sdg"]
_THETA = math.pi / 8  # placeholder rotation (see module docstring)


# -- gate primitives (following the reference construction) ------------------


def _qc_collect(qc, target: int, control: int) -> None:
    """2-CNOT collect of a covered off-walk neighbour (white vertex)."""
    qc.rz(_THETA, target)
    qc.cx(control, target)
    qc.rz(-_THETA, target)
    qc.cx(control, target)


def _qc_collect_and_move(qc, target: int, control: int) -> None:
    """3-CNOT collect-and-move along a walk edge."""
    qc.rz(_THETA, target)
    qc.cx(control, target)
    qc.rz(-_THETA, target)
    qc.cx(target, control)
    qc.cx(control, target)


def assign_whites(
    walk: Sequence[int], region: Sequence[int], G: nx.Graph
) -> Dict[int, List[int]]:
    """Assign each white vertex (covered, off the walk) to the first adjacent
    black vertex in walk order. Returns ``{black_vertex: [whites...]}``."""
    blacks = list(dict.fromkeys(walk))
    black_set = set(blacks)
    whites = [v for v in region if v not in black_set]
    out: Dict[int, List[int]] = {b: [] for b in blacks}
    for w in whites:
        host = next((b for b in blacks if G.has_edge(w, b)), None)
        if host is None:  # cannot happen when the walk dominates the region
            host = blacks[0]
        out[host].append(w)
    return out


def _emit_sweep(qc, walk, whites_map) -> Tuple[int, int]:
    """Emit one sweep; returns (whites, moves)."""
    collected: set = set()
    n_whites = n_moves = 0
    for idx, black in enumerate(walk):
        black = int(black)
        for w in whites_map.get(black, []):
            if w in collected:
                continue
            _qc_collect(qc, black, int(w))
            collected.add(w)
            n_whites += 1
        if idx < len(walk) - 1:
            _qc_collect_and_move(qc, black, int(walk[idx + 1]))
            n_moves += 1
    return n_whites, n_moves


# -- public builders ---------------------------------------------------------


def single_sweep(
    graph,
    solver: Optional[Solver] = None,
    build_circuit: bool = True,
) -> Dict:
    """One full-graph sweep: best dominating walk + its circuit skeleton.

    Returns ``{walk, whites, moves, cnot, qc, result}`` — ``cnot`` is the
    exact count ``2*whites + 3*moves``; ``qc`` is a qiskit circuit (or None
    without qiskit / with ``build_circuit=False``)."""
    G, labels = as_graph(graph)
    solver = solver or Solver()
    res = solver.solve(G)
    walk = list(res.walk)  # G is 0..n-1 already, so labels are the identity
    wm = assign_whites(walk, list(G.nodes()), G)
    whites = G.number_of_nodes() - len(set(walk))
    moves = len(walk) - 1
    qc = None
    if build_circuit and HAVE_QISKIT:
        qc = QuantumCircuit(G.number_of_nodes())
        start, end = int(walk[0]), int(walk[-1])
        qc.s(start)
        for i in range(G.number_of_nodes()):
            if i != start:
                qc.h(i)
        _emit_sweep(qc, walk, wm)
        for i in range(G.number_of_nodes()):
            if i != end:
                qc.h(i)
        qc.sdg(end)
    return {
        "walk": [labels[i] for i in walk],
        "whites": whites,
        "moves": moves,
        "cnot": 2 * whites + 3 * moves,
        "qc": qc,
        "result": res,
    }


def sub_qft(graph, k: int, state_budget: Optional[int] = None,
            state_cap: Optional[int] = None,
            solver: Optional[Solver] = None,
            weight: Optional[str] = None) -> Dict:
    """Optimal size-``k`` sub-QFT region + walk via the budgeted-coverage
    search (:mod:`qftbuilder.kwalk`), in the caller's labels.
    ``state_cap=None`` uses the budgeted default, which guarantees the
    search never exceeds the whole-graph solve asymptotically (see the
    ``kwalk`` module docstring). ``k >= n`` delegates to the whole-graph
    cascade (``solver``, default balanced).

    ``weight`` names an edge attribute of per-CNOT costs (see
    :func:`qftbuilder.graphs.with_edge_errors`): the region then forms around
    good couplings instead of merely being short. See the fidelity caveat in
    :func:`qftbuilder.kwalk.budgeted_walk`.

    Returns the ``kwalk`` result dict plus ``whites/moves/cnot`` for one
    sweep over the region."""
    G, labels = as_graph(graph)
    n = G.number_of_nodes()
    if k >= n:
        sw = single_sweep(graph, solver=solver, build_circuit=False)
        res = sw["result"]
        return {
            "k": n, "subset": sorted(labels.values(), key=str),
            "walk": sw["walk"], "k_path": res.length,
            "cost": res.cost_bound, "cost_sweep": res.cost,
            "region_proven": False, "method": "solver",
            "whites": sw["whites"], "moves": sw["moves"], "cnot": sw["cnot"],
        }
    r = budgeted_walk(G, k, state_budget=state_budget, state_cap=state_cap,
                      weight=weight)
    if r["walk"] is None:
        return r
    if not r["region_proven"]:
        # beam-degraded search (and always so under weights): polish the walk
        # inside its region, under the same objective it was chosen by -- a
        # length polish would undo the fidelity gain.
        sub = G.subgraph(r["subset"])
        w = improve_walk_strong(
            r["walk"], sub,
            objective="length" if weight is None else "fidelity",
            weight=weight)
        if covers(sub, w):
            r["walk"], r["k_path"] = w, len(w)
    whites = k - len(set(r["walk"]))
    moves = r["k_path"] - 1
    r.update(
        {
            "walk": [labels[i] for i in r["walk"]],
            "subset": [labels[i] for i in r["subset"]],
            "whites": whites,
            "moves": moves,
            "cnot": 2 * whites + 3 * moves,
        }
    )
    return r


def _cascade_candidates(
    sub: nx.Graph, prev_walk: Optional[List[int]], solver: Solver
) -> Optional[List[int]]:
    """Best walk for one cascade: warm-start repair of the previous walk vs a
    fresh cascade solve, ranked by exact cascade cost."""
    n_sub = sub.number_of_nodes()
    if n_sub == 1:
        return list(sub.nodes())
    cands: List[List[int]] = []
    if prev_walk is not None:
        w = repair_walk(prev_walk, sub)
        if w and walk_valid(sub, w) and covers(sub, w):
            cands.append(w)
    res = solver.solve(sub)
    cands.append(list(res.walk))  # SolveResult is already in sub's labels

    def cost(w):
        return 2 * (n_sub - len(set(w))) + 3 * (len(w) - 1)

    return min(cands, key=cost) if cands else None


def _removable(sub: nx.Graph, walk: List[int], lookahead: bool) -> int:
    """Choose the qubit to finalize after a cascade: an endpoint (or low-degree
    vertex) whose removal keeps the graph connected; with ``lookahead``, the
    candidate whose removal leaves the cheapest next-cascade heuristic walk."""
    cands: List[int] = []
    seen = set()
    for v in (walk[-1], walk[0]):
        if v not in seen:
            seen.add(v)
            cands.append(v)
    for v in sorted(sub.nodes(), key=lambda u: sub.degree(u)):
        if v not in seen:
            seen.add(v)
            cands.append(v)

    feasible = []
    for v in cands:
        if sub.number_of_nodes() == 1:
            return v
        H = sub.copy()
        H.remove_node(v)
        if nx.is_connected(H):
            feasible.append(v)
        if len(feasible) >= (3 if lookahead else 1):
            break
    if not feasible:
        return walk[-1]
    if not lookahead or len(feasible) == 1:
        return feasible[0]

    from .walk import dominating_walk

    def next_cost(v):
        H = sub.copy()
        H.remove_node(v)
        w = dominating_walk(H)
        if w is None:
            return float("inf")
        return 2 * (H.number_of_nodes() - len(set(w))) + 3 * (len(w) - 1)

    return min(feasible, key=next_cost)


def build_full_qft(
    graph,
    solver: Optional[Solver] = None,
    strategy: str = "best",
    build_circuit: bool = False,
) -> Dict:
    """Full n-cascade QFT over an arbitrary connected coupling graph.

    ``strategy``: ``"greedy"`` finalizes the first safe endpoint per cascade
    (the source project's rule), ``"lookahead"`` picks among endpoint
    candidates by a one-step cost estimate, ``"best"`` (default) runs both
    chains and keeps the cheaper one — the removal order is a greedy
    sequence, so neither rule dominates on every topology.

    Returns ``{cnot, cascades, per, qc, walks, strategy}``: ``cnot`` is the
    exact total CNOT count, ``per`` lists ``(n_remaining, whites, moves)``
    per cascade, ``walks`` the chosen walk per cascade (caller's labels).
    Validated in tests against the textbook identity
    ``QFT(K_m) = m(m-1)`` CNOTs."""
    if strategy not in ("best", "lookahead", "greedy"):
        raise ValueError(f"unknown strategy: {strategy!r}")
    solver = solver or Solver()
    if strategy == "best":
        a = _build_chain(graph, solver, lookahead=False, build_circuit=False)
        b = _build_chain(graph, solver, lookahead=True, build_circuit=False)
        win_look = b["cnot"] < a["cnot"]
        if build_circuit:  # deterministic: rebuild the winner with gates
            out = _build_chain(graph, solver, lookahead=win_look,
                               build_circuit=True)
        else:
            out = b if win_look else a
        out["strategy"] = "lookahead" if win_look else "greedy"
        return out
    out = _build_chain(graph, solver, lookahead=(strategy == "lookahead"),
                       build_circuit=build_circuit)
    out["strategy"] = strategy
    return out


def _build_chain(
    graph,
    solver: Solver,
    lookahead: bool,
    build_circuit: bool,
) -> Dict:
    G, labels = as_graph(graph)
    n = G.number_of_nodes()
    remaining = set(G.nodes())
    total = 0
    per: List[Tuple[int, int, int]] = []
    walks: List[List] = []
    qc = QuantumCircuit(n) if (build_circuit and HAVE_QISKIT) else None
    prev_walk: Optional[List[int]] = None

    while remaining:
        sub = G.subgraph(remaining).copy()
        if not nx.is_connected(sub):  # defensive; removal keeps connectivity
            comp = max(nx.connected_components(sub), key=len)
            sub = G.subgraph(comp).copy()
            remaining = set(comp) | (remaining - set(sub.nodes()))
        walk = _cascade_candidates(sub, prev_walk, solver)
        if walk is None:
            break
        n_sub = sub.number_of_nodes()
        whites = n_sub - len(set(walk))
        moves = len(walk) - 1
        total += 2 * whites + 3 * moves
        per.append((n_sub, whites, moves))
        walks.append([labels[i] for i in walk])
        if qc is not None:
            wm = assign_whites(walk, list(sub.nodes()), sub)
            _emit_sweep(qc, walk, wm)
        rem_v = _removable(sub, walk, lookahead)
        remaining.discard(rem_v)
        prev_walk = walk

    return {"cnot": total, "cascades": len(per), "per": per, "qc": qc,
            "walks": walks}


# -- qiskit comparison helpers ----------------------------------------------


def _coupling_from_nx(G: nx.Graph):
    cm = CouplingMap()
    for v in G.nodes():
        cm.add_physical_qubit(int(v))
    for u, v in G.edges():
        cm.add_edge(int(u), int(v))
        cm.add_edge(int(v), int(u))
    return cm


def transpiled_cx(qc, graph, opt_level: int = 1, seed: int = 0) -> Tuple[int, int]:
    """Transpile onto the coupling graph, return ``(cx_count, depth)``.
    Requires qiskit."""
    if not HAVE_QISKIT:
        raise ImportError("qiskit not installed; run `pip install qft-builder[quantum]`")
    G, _ = as_graph(graph)
    pm = generate_preset_pass_manager(
        optimization_level=opt_level,
        coupling_map=_coupling_from_nx(G),
        basis_gates=BASIS,
        seed_transpiler=seed,
    )
    out = pm.run(qc)
    ops = out.count_ops()
    return ops.get("cx", 0) + ops.get("cz", 0), out.depth()


def naive_qft(k: int, num_qubits: int):
    """Textbook QFT on the first ``k`` qubits (baseline for the transpiler
    comparison). Requires qiskit."""
    if not HAVE_QISKIT:
        raise ImportError("qiskit not installed")
    import numpy as np

    qc = QuantumCircuit(num_qubits)
    for j in range(k):
        qc.h(j)
        for m in range(j + 1, k):
            qc.cp(np.pi / 2 ** (m - j), m, j)
    for j in range(k // 2):
        qc.swap(j, k - 1 - j)
    return qc
