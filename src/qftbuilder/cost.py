"""CNOT cost model for walk-based QFT construction.

Two levels of accounting:

1. **Closed-form bounds** (Khadiev, Khadieva, Sagitov — "Quantum Circuit for
   QFT for Arbitrary Qubits Connection Graph"): a single cascade over a walk
   of length ``k`` costs at most ``3k - 2`` CNOTs, hence the full circuit is
   bounded by ``3*k*n - 2*n`` (Theorem 4); with a Hamiltonian path the exact
   full-QFT cost is ``1.5n^2 - 1.5n - 1`` (Theorem 5).

2. **Exact per-sweep account** used by this library's construction and by the
   candidate selector, in the thesis's black/white vertex terminology
   (black = on the walk, white = covered but off the walk): a sweep over
   walk ``W`` in a region of ``s`` vertices costs exactly
   ``2*whites + 3*moves`` CNOTs where ``moves = len(W) - 1`` and
   ``whites = s - distinct(W)`` (each white vertex is collected with a
   2-CNOT gate; each walk step costs a 3-CNOT collect-and-move). Note this
   is *not* a function of ``len(W)`` alone: at equal length, a walk with
   more distinct (black) vertices is strictly cheaper. The solver therefore
   ranks candidates by this exact cost, not by length.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

__all__ = [
    "cnot_upper_bound",
    "cnot_hp_exact",
    "sweep_cost",
    "walk_cost",
    "EDGE_WEIGHT",
    "error_to_weight",
    "weighted_sweep_cost",
]

#: Edge attribute holding the per-CNOT cost of a coupling. Absent attributes
#: read as 1.0, which reproduces the unweighted account exactly
#: (docs/proofs.md, Proposition 5.1).
EDGE_WEIGHT = "cnot_weight"

#: Smallest weight a coupling may carry. A zero weight is legal but blinds the
#: budgeted search's heuristic (docs/proofs.md, Lemma 5.4, degenerate case).
WEIGHT_FLOOR = 1e-9


def cnot_upper_bound(k: int, n: int) -> int:
    """Theorem 4 upper bound for the full n-cascade QFT driven by a walk of
    length ``k``: ``3*k*n - 2*n``. Monotone in ``k``."""
    if n <= 0:
        return 0
    return 3 * k * n - 2 * n


def cnot_hp_exact(n: int) -> int:
    """Theorem 5: exact full-QFT CNOT count when a Hamiltonian path exists
    (``k = n``): ``1.5n^2 - 1.5n - 1``."""
    if n < 2:
        return 0
    return (3 * n * (n - 1)) // 2 - 1


def sweep_cost(walk_len: int, distinct: int, region_size: int) -> int:
    """Exact CNOT count of one sweep: ``2*whites + 3*moves`` with
    ``whites = region_size - distinct`` (covered off-walk vertices) and
    ``moves = walk_len - 1``. ``distinct`` is the number of distinct (black)
    walk vertices, ``region_size`` the number of qubits the sweep covers
    (= n for a whole-graph sweep)."""
    if region_size <= 0 or walk_len <= 0:
        return 0
    whites = max(region_size - distinct, 0)
    moves = max(walk_len - 1, 0)
    return 2 * whites + 3 * moves


def walk_cost(walk: Optional[Sequence[int]], region_size: int) -> Optional[int]:
    """``sweep_cost`` of a concrete walk (None-propagating)."""
    if walk is None:
        return None
    return sweep_cost(len(walk), len(set(walk)), region_size)


# -- fidelity-aware account --------------------------------------------------


def error_to_weight(err: float, floor: float = WEIGHT_FLOOR) -> float:
    """Turn a CNOT error rate into an additive weight ``-ln(1 - err)``, so that
    summing weights along a circuit equals ``-ln`` of the product of survival
    probabilities. Clamped to ``[floor, ...)``; ``err >= 1`` maps to a large
    finite penalty rather than infinity so the search stays well defined."""
    e = min(max(float(err), 0.0), 1.0 - 1e-12)
    return max(-math.log1p(-e), floor)


def _edge_w(G, u, v, weight: str, default: float) -> float:
    data = G.get_edge_data(u, v) or {}
    return float(data.get(weight, default))


def weighted_sweep_cost(walk, G, region=None, weight: str = EDGE_WEIGHT,
                        default: float = 1.0) -> float:
    """Fidelity-aware cost of one sweep (docs/proofs.md, section 5)::

        3 * sum over moves of w(edge)  +  2 * sum over whites of the cheapest
        edge to a black neighbour.

    ``region`` defaults to everything the walk covers. With every weight equal
    to 1 this returns exactly ``sweep_cost`` (Proposition 5.1); the white
    assignment is optimal by Proposition 5.2."""
    if not walk:
        return 0.0
    blacks = set(walk)
    total = 0.0
    for a, b in zip(walk, walk[1:]):
        total += 3.0 * _edge_w(G, a, b, weight, default)
    if region is None:
        region = set()
        for v in walk:
            region.add(v)
            region.update(G.neighbors(v))
    for x in region:
        if x in blacks:
            continue
        best = None
        for b in G.neighbors(x):
            if b in blacks:
                cw = _edge_w(G, b, x, weight, default)
                if best is None or cw < best:
                    best = cw
        if best is not None:
            total += 2.0 * best
    return total
