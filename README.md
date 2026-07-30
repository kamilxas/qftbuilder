# qft-builder

**CNOT-efficient QFT circuits for arbitrary qubit connectivity graphs.**

Feed in a coupling graph in any convenient form — edge list, adjacency
matrix, `networkx` graph, qiskit `CouplingMap`, or a file — and get back:

- the best **dominating walk** of the device (the CNOT-cost driver of
  walk-based QFT construction), found by a cascade of solvers with a neural
  scorer inside, with an optional **optimality certificate**;
- the provably **optimal k-qubit sub-QFT region** for any `k`, in
  milliseconds;
- a full **n-cascade QFT** CNOT account and SWAP-free circuit skeleton;
- a publication-ready **picture** of the solution.

The construction follows Khadiev–Khadieva–Sagitov, *Quantum Circuit for QFT
for Arbitrary Qubits Connection Graph* ([arXiv:2510.09824]): every 2-qubit
gate sits on a physical edge, so the transpiler inserts **zero SWAPs**.

[arXiv:2510.09824]: https://arxiv.org/abs/2510.09824

## Install

```bash
pip install .              # core: networkx + numpy + scipy + matplotlib
pip install ".[all]"       # + neural cascade (torch), circuits (qiskit),
                           #   exact polish (ortools)  <- recommended
```

Python 3.10+. The neural checkpoints ship inside the package — nothing to
download or train.

## Five lines

```python
import qftbuilder as qb

G = qb.heavy_hex(1, 2)                  # or [(0,1), (1,2), ...], or a matrix
res = qb.solve(G)                       # -> walk, exact CNOT/sweep, source rung
print(qb.certify(G, res))               # -> lower bound, proven_optimal
print(qb.sub_qft(G, k=12))              # -> optimal 12-qubit region + walk
qb.draw_solution(G, res).figure.savefig("solution.png")
```

A longer tour — every entry point, with pictures — is in
[`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py
```

## API

| Call | Returns |
|---|---|
| `solve(graph, profile="balanced")` | best dominating walk (`SolveResult`: walk in *your* labels, exact sweep CNOTs, per-rung candidates) |
| `certify(graph, result)` | `Certificate` — MILP/CP-SAT lower bound, `proven_optimal` |
| `sub_qft(graph, k)` | optimal size-`k` region + walk (+ `region_proven`) |
| `sub_qft_sweep(graph)` | the same for **every** `k` from one search |
| `build_full_qft(graph)` | full QFT: exact CNOT total, per-cascade account, optional qiskit circuit |
| `single_sweep(graph)` | one sweep + SWAP-free circuit skeleton |
| `draw_solution / draw_benchmark` | matplotlib figures (walk order, region, coverage) |
| `lnn / cycle / sun / heavy_hex / square_lattice / standard_benchmark` | device topologies |

Solver profiles: `fast` (pure heuristics), `balanced` (default; + GNN cascade
when torch is installed), `max` (+ CP-SAT exact polish under a time budget).

## Method, briefly

A dominating walk visits or neighbours every qubit. **Black** vertices lie on
the walk and **white** vertices are covered but off it; a QFT cascade
sweeping the walk costs exactly `2·whites + 3·moves` CNOTs. The solver runs a
**cascade**: neural simple-path rung → neural walk rung → cost-aware greedy
dominating-set walk → CDS+DFS floor, all polished by a monotone-safe local
search, ranked by the exact CNOT account (not by length — at equal length,
more black vertices is strictly cheaper), optionally tightened by CP-SAT with
a warm start.

For a QFT on `k < n` qubits, the region choice collapses to a **budgeted
coverage walk**: minimize walk length subject to the walk's closed
neighbourhood reaching `k` vertices — the region is a free by-product of the
walk. A BFS over `(endpoint, visited-set)` states solves it *exactly* at the
walk scale `~k/Δ` instead of the region scale `k`. A budgeted MILP twin
provides independent lower bounds.

## Runtime notes

The underlying problems are NP-hard, so nothing here claims an exact
polynomial algorithm; the guarantees are about budgets.

- `solve` is polynomial, `O(n³)` worst case (dominated by the 2-opt/Or-opt
  local search). Measured: ~2.4 s at `n=87`, ~4.5 s at `n=144` (fast
  profile, laptop CPU).
- `sub_qft` is fixed-parameter tractable in `k` — linear in `n` for fixed
  `k`, with the exponential in walk length `k/Δ` rather than region size
  `k`. It runs under a hard state budget that is a genuine constant multiple
  of `n³` (independent of `k` and `Δ`), so `cost(sub_qft)` stays bounded by
  the whole-graph solve.
- **The budget buys a *proof*, not a shorter walk.** A good, empirically
  optimal-length region comes out of the beam in ~1 MB; the millions of
  states only prove by exhaustion that nothing shorter exists. Raise
  `state_budget` only when you need `region_proven=True` at large `k`; for a
  cheap certificate prefer the budgeted MILP (`certify_sub`), which proves
  via LP/branch-and-bound without enumerating states.

Whenever a budget forces a heuristic degradation the result says so
(`region_proven=False`, `method` names the source).

## Caveats

- Emitted circuits are **CX-count-accurate skeletons**: rotation angles are
  placeholders pending faithful Algorithm-9/10 angle bookkeeping.
- Dense graphs can push the sub-QFT BFS past its memory cap; it then
  degrades to a beam and honestly reports `region_proven=False`
  (`certify_sub` still bounds the answer).

## License

MIT — see [LICENSE](LICENSE). Derived from the author's thesis project at
Kazan Federal University; the neural checkpoints were trained there on
exact-DP-labelled synthetic graphs (n ≤ 22).
