# qft-builder

[![tests](https://github.com/kamilxas/qftbuilder/actions/workflows/ci.yml/badge.svg)](https://github.com/kamilxas/qftbuilder/actions/workflows/ci.yml)

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
gate sits on a physical edge, so the transpiler inserts **zero SWAPs** —
against Qiskit's `optimization_level=3` on the same topology this saves
roughly a third of CNOTs on the device benchmark.

[arXiv:2510.09824]: https://arxiv.org/abs/2510.09824

## Install

```bash
pip install -e .            # core: networkx + numpy + scipy + matplotlib
pip install -e .[all]       # + neural cascade (torch), circuits (qiskit),
                            #   exact polish (ortools)  <- recommended
```

## Five lines

```python
import qftbuilder as qb

G = qb.heavy_hex(1, 2)                  # or [(0,1), (1,2), ...], or a matrix
res = qb.solve(G)                       # -> walk, exact CNOT/sweep, source rung
print(qb.certify(G, res))               # -> lower bound, proven_optimal
print(qb.sub_qft(G, k=12))              # -> optimal 12-qubit region + walk
qb.draw_solution(G, res).figure.savefig("solution.png")
```

More in [`examples/quickstart.py`](examples/quickstart.py).

## API

| Call | Returns |
|---|---|
| `solve(graph, profile="balanced")` | best dominating walk (`SolveResult`: walk in *your* labels, exact sweep CNOTs, per-rung candidates) |
| `certify(graph, result)` | `Certificate` — MILP/CP-SAT lower bound, `proven_optimal` |
| `sub_qft(graph, k)` | optimal size-`k` region + walk (+ `region_proven`) |
| `sub_qft_sweep(graph)` | the same for **every** `k` from one search |
| `build_full_qft(graph)` | full QFT **cost**: exact CNOT total, per-cascade account, skeleton circuit |
| `build_qft_circuit(graph)` | full QFT **circuit**: the exact unitary, true angles, permuted wire order |
| `single_sweep(graph)` | one sweep + SWAP-free circuit skeleton |
| `draw_solution / draw_benchmark` | matplotlib figures (walk order, region, coverage) |
| `lnn / cycle / sun / heavy_hex / square_lattice / standard_benchmark` | device topologies |
| `with_edge_errors(graph, errors)` | attach per-coupling CNOT error rates as weights |

Solver profiles: `fast` (pure heuristics), `balanced` (default; + GNN cascade
when torch is installed — checkpoints ship with the package), `max`
(+ CP-SAT exact polish under a time budget).

### Objectives

`Solver(objective=...)` chooses what "best" means:

| Objective | Minimizes |
|---|---|
| `cnot` (default) | exact sweep cost `2·whites + 3·moves` |
| `length` | raw walk length (the academic shortest-walk question) |
| `fidelity` | error-weighted cost — routes around bad couplings |

Real devices have CNOT error rates that differ by an order of magnitude
between couplings, so the circuit with the fewest gates is not always the one
most likely to survive:

```python
G = qb.with_edge_errors(qb.heavy_hex(2, 2), backend_error_map)  # {(u,v): rate}
res = qb.Solver(objective="fidelity").solve(G)
reg = qb.sub_qft(G, k=12, weight="cnot_weight")   # region avoids bad qubits
```

Weights are stored as `-ln(1 - error)`, so they add along a circuit and
`exp(-cost)` is the probability that every CNOT of the sweep succeeds. On
lognormal error models (median ~1%, heavy tail) this buys **1.12x** survival
for ~4.6% more CNOTs on whole-graph sweeps, and **1.14x** on sub-QFT regions.
With uniform weights the account reduces exactly to `2·whites + 3·moves`, so
nothing changes for unannotated graphs.

## Method, briefly

A dominating walk visits or neighbours every qubit. Following the thesis
convention, **black** vertices lie on the walk and **white** vertices are
covered but off it; a QFT cascade sweeping the walk costs exactly
`2·whites + 3·moves` CNOTs. The solver runs a **cascade**: neural
simple-path rung → neural walk rung → cost-aware greedy dominating-set walk
→ CDS+DFS floor, all polished by a monotone-safe local search, ranked by
the exact CNOT account (not by length — at equal length, more black
vertices is strictly cheaper), optionally tightened by CP-SAT with a warm
start.

For a QFT on `k < n` qubits, the region choice collapses to a **budgeted
coverage walk** (this project's main new result): minimize walk length
subject to the walk's closed neighbourhood reaching `k` vertices — the
region is a free by-product of the walk. A BFS over `(endpoint, visited-set)`
states solves it *exactly* at the walk scale `~k/Δ` instead of the region
scale `k`: on the 48-cell device benchmark it certifies **48/48** optima
(the prior region search: 29/48), beats it in 3 cells at large `k`, and runs
100–1000× faster. A budgeted MILP twin provides independent lower bounds.
Details and proofs: [`paper/`](paper/).

## Numbers (10-topology device benchmark)

- sub-QFT `k ∈ {4..20}`: optimal & certified in all 48 cells, ms per query;
- whole-graph walks: neural cascade never worse than heuristics, certified
  optimal on all benchmark topologies;
- full QFT: mean **0.985×** the prior construction's CNOTs (best 0.933),
  never worse; ~**0.68×** Qiskit opt-3 (measured in the source project).

## Complexity contract

The underlying problems are NP-hard, so nothing here claims an exact
polynomial algorithm; the contract is about budgets, with the whole-graph
solve as the yardstick.

- **Whole-graph solve** (`solve`, the "full" pipeline): polynomial, `O(n³)`
  worst case, dominated by the 2-opt/Or-opt local search over the ~`n/Δ`
  essential vertices. Measured: ~2.4 s at `n=87`, ~4.5 s at `n=144`
  (fast profile, laptop CPU).
- **Sub-QFT search** (`sub_qft`): *fixed-parameter tractable in `k`* —
  state count `≤ n·b*·(eΔ)^(b*−1)` with `b* ≈ k/Δ`, i.e. linear in `n` for
  fixed `k`; the exponential sits in `k/Δ` (walk length), not `k` (region
  size) as in prior ESU enumeration — that gap is why certified optima now
  reach `k≈32` instead of `k≤8`. On top of that sits a **hard state budget
  whose `n³`-multiplier is a genuine constant** (never `k`, never `Δ`),
  derived rather than asserted:

  1. every stored state `(endpoint, vertex-mask)` enters the parent map
     exactly once (global dedup) → insertions ≤ `B` exact phase + `B/4`
     beam tail;
  2. each stored state is expanded once, probing ≤ `Δ` neighbours; a probe
     hashes one key and ORs one `n`-bit mask → `⌈n/64⌉` word ops;
  3. total work `W ≤ (5/4)·B·Δ·⌈n/64⌉`;
  4. we *fix the work target* `W* = 27·n³` word ops (27 is a chosen
     constant, independent of `k` and `Δ`) and solve for the budget:
     `B = W*·(4/5)·64/(Δ·n) = 1382·n²/Δ` — `Δ` cancels out of the bound by
     construction (denser graphs get a smaller budget), and `k` never
     appears (one budget serves every `k`, including the all-`k` sweep).
     Floor `200 000` states for tiny graphs; absolute clip `8·10⁶` states
     guards memory (~220 bytes/state in Python) — clipping only lowers work.

  Answers absorbed from *fully explored* BFS levels are proven optimal;
  after the budget trips, a width-`max(4096, 64n)` beam tail (≤ `B/4`
  insertions) serves mid-range `k`, and any still-open `k` falls back to
  the shortest covering subwalk of one whole-graph walk (one `O(L²)` scan
  answers **all** `k`). Hence, in word operations,

  `cost(sub_qft) ≤ 27·n³ + cost(solve) + O(n²)` — constant 27 by design.

  **The budget buys a *proof*, not a shorter walk.** A good, empirically
  optimal-length region comes from the beam/fallback in ~1 MB; the millions
  of states only *prove by exhaustion* that nothing shorter exists.
  Measured (grid n=144): at `k = 28/32/36` the heuristic and the proven
  answer are the **same length** (9/10/12), but proving it costs
  ~0.5/1.5/2.0 GB and 10/30/50 s. So raise `state_budget` only when you
  need `region_proven=True` at large `k`; for the walk alone keep it small,
  and for a cheap certificate prefer the budgeted MILP (`certify_sub`),
  which proves via LP/branch-and-bound without enumerating states.

Whenever a budget forces a heuristic degradation the result says so
(`region_proven=False`, `method` names the source), and the budgeted MILP
dual bound remains available as an independent certificate.

## Caveats

- Two circuit builders, on purpose. `build_full_qft` is the cost model: its
  circuit is a **CX-count-accurate skeleton** with placeholder angles, and the
  published CNOT numbers refer to it. `build_qft_circuit` emits the **exact
  QFT unitary** (verified against qiskit by operator equivalence) and costs a
  few percent more CNOTs — each cascade has to end on the qubit it finalizes.
  It returns the QFT in a permuted wire order (`qubit_order` / `input_order`);
  relabel rather than pay for reversal swaps.
- Dense graphs can push the sub-QFT BFS past its memory cap; it then
  degrades to a beam and honestly reports `region_proven=False`
  (`certify_sub` still bounds the answer).
- Under `objective="fidelity"` no optimality is certified for the weighted
  total: the search returns the better of the plain and the weighted optimum
  (never worse than the plain one), and `region_proven` is `False`. The MILP
  still routes unweighted.

## Project layout

`src/qftbuilder/` library · `tests/` (180 tests incl. brute-force equivalence,
pruned-vs-unpruned search and the runtime-budget contract) ·
[`docs/proofs.md`](docs/proofs.md) correctness proofs for every
optimality-critical shortcut · `examples/quickstart.py`.

The LaTeX paper, the Russian design notes, the roadmap and the head-to-head
benchmark harness stay in the development repository.

## License

MIT. Derived from the author's thesis project at Kazan Federal University;
the neural checkpoints were trained there on exact-DP-labelled synthetic
graphs (n ≤ 22).
