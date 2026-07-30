"""qft-builder: CNOT-efficient QFT circuits for arbitrary qubit couplings.

Five-line tour::

    import qftbuilder as qb

    G = qb.heavy_hex(2, 2)                  # or edges / matrix / CouplingMap / file
    res = qb.solve(G)                       # best dominating walk (neural cascade)
    reg = qb.sub_qft(G, k=12)               # optimal 12-qubit sub-QFT region
    full = qb.build_full_qft(G)             # full-QFT CNOT account (+ circuit)
    qb.draw_solution(G, res)                # picture

See README for the method and the API table.
"""
from .cost import cnot_hp_exact, cnot_upper_bound, sweep_cost, walk_cost
from .graphs import (
    as_graph,
    cycle,
    double_sun_27q,
    heavy_hex,
    lnn,
    square_lattice,
    standard_benchmark,
    sun,
    sun_16q,
)
from .kwalk import budgeted_sweep, budgeted_walk, max_coverage_walk
from .qft import build_full_qft, naive_qft, single_sweep, sub_qft, transpiled_cx
from .solver import Certificate, Solver, SolveResult
from .viz import draw_benchmark, draw_solution, grid_layout

__version__ = "0.1.0"

_DEFAULT_SOLVER = None


def _default_solver() -> Solver:
    global _DEFAULT_SOLVER
    if _DEFAULT_SOLVER is None:
        _DEFAULT_SOLVER = Solver()
    return _DEFAULT_SOLVER


def solve(graph, **kwargs) -> SolveResult:
    """Solve with a shared default :class:`Solver` (``profile="balanced"``).
    Keyword arguments construct a one-off solver instead."""
    if kwargs:
        return Solver(**kwargs).solve(graph)
    return _default_solver().solve(graph)


def certify(graph, result: SolveResult | None = None, **kwargs) -> Certificate:
    """Certify a solve (lower bound + optimality flag) with the default solver."""
    return _default_solver().certify(graph, result=result, **kwargs)


def sub_qft_sweep(graph, ks=None, state_budget: int | None = None,
                  state_cap: int | None = None):
    """All-k sub-QFT sweep in one search (see :func:`qftbuilder.kwalk.budgeted_sweep`),
    in the caller's labels."""
    G, labels = as_graph(graph)
    out = budgeted_sweep(G, ks=ks, state_budget=state_budget,
                         state_cap=state_cap)
    for r in out.values():
        if r.get("walk"):
            r["walk"] = [labels[i] for i in r["walk"]]
            r["subset"] = [labels[i] for i in r["subset"]]
    return out


__all__ = [
    "__version__",
    # solving
    "solve", "certify", "Solver", "SolveResult", "Certificate",
    # sub-QFT / budgeted walks
    "sub_qft", "sub_qft_sweep", "budgeted_walk", "budgeted_sweep",
    "max_coverage_walk",
    # circuits
    "single_sweep", "build_full_qft", "transpiled_cx", "naive_qft",
    # cost model
    "cnot_upper_bound", "cnot_hp_exact", "sweep_cost", "walk_cost",
    # graphs
    "as_graph", "lnn", "cycle", "sun", "sun_16q", "double_sun_27q",
    "heavy_hex", "square_lattice", "standard_benchmark",
    # viz
    "draw_solution", "draw_benchmark", "grid_layout",
]
