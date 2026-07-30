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

from typing import Optional, Sequence

__all__ = [
    "cnot_upper_bound",
    "cnot_hp_exact",
    "sweep_cost",
    "walk_cost",
]


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
