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

Two builders, and the difference matters:

- :func:`build_full_qft` is the **cost model**. Its circuit is a skeleton:
  every rotation carries one placeholder angle, so it reproduces the CNOT
  account exactly but is *not* the QFT unitary. Use it for counting.
- :func:`build_qft_circuit` is the **real circuit**. It carries the textbook
  angles ``pi/2**(m-r)`` and is verified against qiskit's QFT by operator
  equivalence. Correctness costs a few percent more CNOTs, because each
  cascade must end on the vertex it finalizes.

Both keep every 2-qubit gate on a physical edge, so neither needs SWAP
insertion, and all published CNOT comparisons (which are about the skeleton
account) are unaffected.

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


# -- faithful-angle primitives ----------------------------------------------
#
# The skeleton primitives above apply one fixed rotation. The three below take
# the real QFT angle. Note the CNOT counts are identical (2 / 3 / 3), so
# faithfulness is free per gate -- every correction is single-qubit.


def _qc_cphase(qc, lam: float, control: int, target: int) -> None:
    """``CP(lam)``: the 2-CNOT collect with the true angle.

    The bare sandwich implements ``CRZ(lam)``, which differs from ``CP(lam)``
    by a phase on the control; ``p(lam/2, control)`` repairs it."""
    qc.rz(lam / 2, target)
    qc.cx(control, target)
    qc.rz(-lam / 2, target)
    qc.cx(control, target)
    qc.p(lam / 2, control)


def _qc_cphase_move(qc, lam: float, target: int, control: int) -> None:
    """``CP(lam)`` followed by ``SWAP``: the 3-CNOT collect-and-move.

    The swap carries the control's state to the other wire, so the
    ``CRZ -> CP`` phase lands on ``target`` *after* the swap. Putting it on the
    control at the end is a different (wrong) unitary -- verified numerically."""
    qc.rz(lam / 2, target)
    qc.cx(control, target)
    qc.rz(-lam / 2, target)
    qc.cx(target, control)
    qc.cx(control, target)
    qc.p(lam / 2, target)


def _qc_swap(qc, a: int, b: int) -> None:
    """Bare 3-CNOT swap: used when the walk revisits a vertex, where the pair
    has already been phased and must not be phased twice. Same cost as a
    collect-and-move, so revisits cost nothing extra."""
    qc.cx(b, a)
    qc.cx(a, b)
    qc.cx(b, a)


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


def _endpoint_removable(sub: nx.Graph, walk: List[int]) -> List[int]:
    """Make the walk end on a vertex whose removal keeps the graph connected.

    The carrier rides the swaps to ``walk[-1]``, and that is the qubit the
    cascade finalizes, so the walk *must* end where we are allowed to remove.
    Try the other endpoint first (a reversed dominating walk is still one, at
    identical cost), else route the carrier to the nearest legal vertex."""
    def ok(v):
        H = sub.copy()
        H.remove_node(v)
        return H.number_of_nodes() == 0 or nx.is_connected(H)

    if ok(walk[-1]):
        return walk
    if ok(walk[0]):
        return walk[::-1]
    targets = [v for v in sub.nodes() if ok(v)]
    if not targets:
        return walk
    tgt = min(targets, key=lambda v: nx.shortest_path_length(sub, walk[-1], v))
    return walk + nx.shortest_path(sub, walk[-1], tgt)[1:]


def _plan_faithful(G: nx.Graph, solver: Solver, lookahead: bool):
    """Pass 1: cascade walks and the finalization order. No angles needed --
    the plan depends only on the graph, which is what lets pass 2 know every
    logical index in advance."""
    live = set(G.nodes())
    steps: List[Tuple[List[int], int]] = []
    prev: Optional[List[int]] = None
    while live:
        sub = G.subgraph(live).copy()
        if sub.number_of_nodes() == 1:
            v = next(iter(live))
            steps.append(([v], v))
            live.discard(v)
            continue
        walk = _cascade_candidates(sub, prev, solver)
        if walk is None:
            break
        walk = _endpoint_removable(sub, walk)
        steps.append((walk, walk[-1]))
        live.discard(walk[-1])
        prev = walk
    return steps


def _logical_index(G: nx.Graph, steps) -> Dict[int, int]:
    """``{initial position: logical index}``. Logical ``r`` is by definition the
    qubit finalized by cascade ``r``; following the swaps back tells us which
    starting wire that is. Because logical index equals finalization order, the
    qubits still live during cascade ``r`` are exactly the indices ``> r`` --
    precisely the set the textbook QFT phases qubit ``r`` against."""
    at = {p: p for p in G.nodes()}
    out: Dict[int, int] = {}
    for r, (walk, _) in enumerate(steps):
        out[at[walk[0]]] = r
        for a, b in zip(walk, walk[1:]):
            at[a], at[b] = at[b], at[a]
    return out


def build_qft_circuit(graph, solver: Optional[Solver] = None,
                      strategy: str = "greedy") -> Dict:
    """The **exact** QFT unitary on an arbitrary coupling graph.

    Unlike :func:`build_full_qft`, whose circuit is a CX-count-accurate
    skeleton with placeholder angles, this emits the true QFT: every rotation
    carries the textbook angle ``pi/2**(m-r)`` between logical qubits ``r``
    (the one the cascade finalizes) and ``m``. Verified against qiskit's QFT by
    operator equivalence in the test suite.

    Two structural consequences, both real costs of correctness:

    * each cascade must end on the vertex it finalizes, so a walk whose
      endpoint cannot be removed is reversed or extended -- this is why the
      count here can exceed :func:`build_full_qft`'s;
    * the moves permute the qubits, so the result is the QFT **in a permuted
      wire order**: ``qubit_order[l]`` is the physical wire carrying logical
      ``l`` on output, and ``input_order[l]`` the wire it entered on. No
      reversal swaps are emitted; relabel instead of paying for them.

    Returns ``{qc, cnot, cascades, walks, qubit_order, input_order}``.
    Requires qiskit."""
    if not HAVE_QISKIT:
        raise ImportError("qiskit not installed; run `pip install qft-builder[quantum]`")
    G, labels = as_graph(graph)
    n = G.number_of_nodes()
    solver = solver or Solver()
    steps = _plan_faithful(G, solver, lookahead=(strategy == "lookahead"))
    logical_of_init = _logical_index(G, steps)

    qc = QuantumCircuit(n)
    at = {p: p for p in G.nodes()}          # physical wire -> initial position
    live = set(G.nodes())
    cnot = 0
    final_pos: Dict[int, int] = {}
    for r, (walk, rem) in enumerate(steps):
        sub = G.subgraph(live).copy()
        qc.h(walk[0])                       # H on the qubit being finalized
        wm = assign_whites(walk, list(sub.nodes()), sub)
        done: set = set()
        seen: set = set()
        for idx, black in enumerate(walk):
            for w in wm.get(black, []):
                if w in seen:
                    continue
                seen.add(w)
                m = logical_of_init[at[w]]
                if m in done or m == r:
                    continue
                done.add(m)
                _qc_cphase(qc, math.pi / 2 ** (m - r), control=int(w),
                           target=int(black))
                cnot += 2
            if idx < len(walk) - 1:
                nxt = int(walk[idx + 1])
                m = logical_of_init[at[nxt]]
                if m in done or m == r:
                    _qc_swap(qc, int(black), nxt)   # revisit: no second phase
                else:
                    done.add(m)
                    _qc_cphase_move(qc, math.pi / 2 ** (m - r),
                                    target=int(black), control=nxt)
                cnot += 3
                at[black], at[nxt] = at[nxt], at[black]
        final_pos[r] = rem
        live.discard(rem)

    inv = {r: ip for ip, r in logical_of_init.items()}
    return {
        "qc": qc,
        "cnot": cnot,
        "cascades": len(steps),
        "walks": [[labels[i] for i in w] for w, _ in steps],
        "qubit_order": {l: labels[final_pos[l]] for l in sorted(final_pos)},
        "input_order": {l: labels[inv[l]] for l in sorted(inv)},
    }


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
